"""Digest-manifested handoff bundles for shipping host shards to a collector.

A host shard ships as its whole ``records/`` tree: canonical ledgers, the
intake-extension valve ledgers, requeue grants, the text-corpus index plus its
content-addressed payloads, and the frozen intake snapshots and registry the
records were minted under. The bundle manifest freezes that tree file-by-file
on the SOURCE host; the collector re-verifies every byte before the shard is
allowed anywhere near a merge. Flock sidecars (``*.lock``) are host-local
runtime artifacts and are excluded on both sides.

The manifest is itself content-addressed (``manifest_sha256`` over its own
canonical body), so an edited manifest refuses before any file comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.models import JsonObject

HANDOFF_MANIFEST_FORMAT = "menagerie.crawler.host-shard-handoff-manifest.v1"

_EXCLUDED_SUFFIXES = (".lock",)
_EXCLUDED_NAMES = frozenset({".DS_Store"})


class HandoffBundleError(ValueError):
    """Raised when a handoff manifest is malformed or self-inconsistent."""


def _shipped_files(records_root: Path) -> list[Path]:
    """Return every shipped regular file below one records root.

    Parameters
    ----------
    records_root:
        Host-local canonical records root.

    Returns
    -------
    list[pathlib.Path]
        Sorted shipped files (locks and OS litter excluded).
    """

    return sorted(
        path
        for path in records_root.rglob("*")
        if path.is_file()
        and path.suffix not in _EXCLUDED_SUFFIXES
        and path.name not in _EXCLUDED_NAMES
    )


def _sealed(body: Mapping[str, Any]) -> JsonObject:
    """Return a manifest body sealed with its own content digest.

    Parameters
    ----------
    body:
        Manifest fields excluding the seal.

    Returns
    -------
    dict[str, Any]
        Body plus ``manifest_sha256``.
    """

    sealed = dict(body)
    sealed["manifest_sha256"] = stable_hash(dict(body))
    return sealed


def build_handoff_manifest(
    records_root: Path,
    *,
    shard_id: str,
    campaign_id: str,
    manifest_path: Path,
    created_at: Optional[str] = None,
) -> JsonObject:
    """Freeze one shard records tree into a digest manifest on the source host.

    Parameters
    ----------
    records_root:
        Host-local canonical records root to ship.
    shard_id:
        Caller-stable host/shard label.
    campaign_id:
        Campaign the shard belongs to.
    manifest_path:
        Destination manifest path OUTSIDE the records tree.
    created_at:
        Optional RFC3339 build stamp; defaults to now (UTC).

    Returns
    -------
    dict[str, Any]
        Persisted manifest.

    Raises
    ------
    HandoffBundleError
        If the records root is missing, empty, or would contain the manifest.
    """

    resolved_records = records_root.resolve()
    if not resolved_records.is_dir():
        raise HandoffBundleError(f"records root is not a directory: {resolved_records}")
    resolved_manifest = manifest_path.resolve()
    if resolved_records in resolved_manifest.parents:
        raise HandoffBundleError("the handoff manifest must live outside the records tree")
    files: dict[str, JsonObject] = {}
    total_bytes = 0
    for path in _shipped_files(resolved_records):
        payload = path.read_bytes()
        total_bytes += len(payload)
        files[path.relative_to(resolved_records).as_posix()] = {
            "sha256": hash_bytes(payload),
            "bytes": len(payload),
        }
    if not files:
        raise HandoffBundleError(f"records root ships no files: {resolved_records}")
    manifest = _sealed(
        {
            "format": HANDOFF_MANIFEST_FORMAT,
            "shard_id": shard_id,
            "campaign_id": campaign_id,
            "created_at": created_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": dict(sorted(files.items())),
        }
    )
    atomic_replace_bytes(resolved_manifest, canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_handoff_manifest(manifest_path: Path) -> JsonObject:
    """Load and self-verify one handoff manifest.

    Parameters
    ----------
    manifest_path:
        Manifest file path.

    Returns
    -------
    dict[str, Any]
        Verified manifest.

    Raises
    ------
    HandoffBundleError
        If the manifest is unreadable, the wrong format, or fails its own
        content digest.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffBundleError(f"handoff manifest unreadable: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != HANDOFF_MANIFEST_FORMAT:
        raise HandoffBundleError(f"not a {HANDOFF_MANIFEST_FORMAT} manifest: {manifest_path}")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if stable_hash(body) != manifest.get("manifest_sha256"):
        raise HandoffBundleError(f"handoff manifest fails its own content digest: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise HandoffBundleError(f"handoff manifest ships no files: {manifest_path}")
    return manifest


def verify_handoff_manifest(records_root: Path, manifest_path: Path) -> JsonObject:
    """Re-hash a received shard tree against its source-host manifest.

    Parameters
    ----------
    records_root:
        Received records root on the collector.
    manifest_path:
        Source-host manifest path.

    Returns
    -------
    dict[str, Any]
        ``{"verified": bool, "missing": [...], "mismatched": [...],
        "extra": [...], "file_count": int, "shard_id": str}``.

    Raises
    ------
    HandoffBundleError
        If the manifest itself is malformed (a broken manifest is never a
        passing bundle, and never a silently failing one either).
    """

    manifest = load_handoff_manifest(manifest_path)
    resolved_records = records_root.resolve()
    expected: Mapping[str, Mapping[str, Any]] = manifest["files"]
    present = {
        path.relative_to(resolved_records).as_posix(): path
        for path in _shipped_files(resolved_records)
    }
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    mismatched = sorted(
        relative
        for relative, entry in expected.items()
        if relative in present and hash_bytes(present[relative].read_bytes()) != entry["sha256"]
    )
    return {
        "verified": not (missing or extra or mismatched),
        "shard_id": str(manifest.get("shard_id")),
        "campaign_id": str(manifest.get("campaign_id")),
        "file_count": len(expected),
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the handoff bundle CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``build`` and ``verify`` subcommands.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="freeze a shard records tree on the source host")
    build.add_argument("--records-root", type=Path, required=True)
    build.add_argument("--shard-id", required=True)
    build.add_argument("--campaign-id", required=True)
    build.add_argument("--manifest", type=Path, required=True)
    verify = commands.add_parser("verify", help="re-verify a received shard on the collector")
    verify.add_argument("--records-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the handoff bundle CLI.

    Parameters
    ----------
    argv:
        Optional arguments excluding executable name.

    Returns
    -------
    int
        Zero on success, otherwise one.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_handoff_manifest(
                args.records_root,
                shard_id=args.shard_id,
                campaign_id=args.campaign_id,
                manifest_path=args.manifest,
            )
            print(
                json.dumps(
                    {
                        "shard_id": manifest["shard_id"],
                        "file_count": manifest["file_count"],
                        "total_bytes": manifest["total_bytes"],
                        "manifest_sha256": manifest["manifest_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        report = verify_handoff_manifest(args.records_root, args.manifest)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["verified"] else 1
    except (HandoffBundleError, OSError) as exc:
        print(f"handoff bundle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
