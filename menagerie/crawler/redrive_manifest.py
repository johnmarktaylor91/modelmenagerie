"""Digested re-drive target manifest: the named-mode blast-radius authority.

D-54: ``--redrive-targets`` is a named admission authority -- every enumerated
member buys a fresh ``dependency-refresh`` work generation (author plus checker
sessions, certain by construction) -- yet the target list was a bare CSV with
no cap, no reason, and no binding to the intake it was composed against. A
hand-typed list above the inline cap must instead arrive as this manifest: a
committed, self-digested JSON artifact scoped to the exact intake snapshot,
so "which members, why, against which census" is one durable, recomputable
fact rather than an argv string (the B-03 composition-attestation shape;
nothing here is taken on trust).

The manifest is refused, never repaired: a wrong field set, a failed
self-digest, an empty or duplicated target list, or (at leg start) a snapshot
sha that is not the ACTIVE intake's all raise :class:`RedriveManifestError`
typed. A snapshot sha is never carried forward by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from menagerie.crawler.identity import stable_hash

#: Format tag of the one supported manifest schema; prior/foreign formats
#: refuse typed rather than parse (no downgrade ladder).
REDRIVE_MANIFEST_FORMAT = "menagerie.crawler.redrive-manifest.v1"

#: Exact top-level field set; a subset or superset refuses.
REDRIVE_MANIFEST_FIELDS = frozenset(
    {"format", "reason", "intake_snapshot_sha256", "targets", "manifest_sha256"}
)


class RedriveManifestError(ValueError):
    """A re-drive manifest failed its structural or binding contract."""


@dataclass(frozen=True)
class RedriveManifest:
    """One validated re-drive manifest (loaded, self-digest verified)."""

    reason: str
    intake_snapshot_sha256: str
    targets: tuple[str, ...]
    manifest_sha256: str

    def require_snapshot(self, snapshot_sha256: str) -> None:
        """Refuse unless this manifest was composed against the ACTIVE intake."""

        if self.intake_snapshot_sha256 != snapshot_sha256:
            raise RedriveManifestError(
                "re-drive manifest is bound to intake snapshot "
                f"{self.intake_snapshot_sha256!r} but the active intake is "
                f"{snapshot_sha256!r}. Re-compose the manifest against the "
                "current intake -- a snapshot sha is never carried forward by hand"
            )


def render_redrive_manifest(
    *, reason: str, intake_snapshot_sha256: str, targets: tuple[str, ...]
) -> dict[str, object]:
    """Return the sealed manifest body (the writer half of parse/render parity)."""

    body: dict[str, object] = {
        "format": REDRIVE_MANIFEST_FORMAT,
        "reason": reason,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "targets": list(targets),
    }
    return {**body, "manifest_sha256": stable_hash(body)}


def load_redrive_manifest(path: Path) -> RedriveManifest:
    """Load and validate one manifest file, verifying its own content digest."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RedriveManifestError(f"re-drive manifest unreadable: {path} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RedriveManifestError(f"re-drive manifest is not JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RedriveManifestError(f"re-drive manifest must be a JSON object: {path}")
    if payload.get("format") != REDRIVE_MANIFEST_FORMAT:
        raise RedriveManifestError(
            f"re-drive manifest format must be {REDRIVE_MANIFEST_FORMAT!r}, "
            f"not {payload.get('format')!r}: {path}"
        )
    if set(payload) != REDRIVE_MANIFEST_FIELDS:
        raise RedriveManifestError(
            "re-drive manifest has an invalid field contract "
            f"(expected exactly {sorted(REDRIVE_MANIFEST_FIELDS)}, "
            f"got {sorted(payload)}): {path}"
        )
    body = {key: payload[key] for key in payload if key != "manifest_sha256"}
    if payload["manifest_sha256"] != stable_hash(body):
        raise RedriveManifestError(
            f"re-drive manifest fails its own content digest: {path}"
        )
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise RedriveManifestError(f"re-drive manifest reason must be nonempty: {path}")
    snapshot_sha = payload["intake_snapshot_sha256"]
    if not isinstance(snapshot_sha, str) or not snapshot_sha:
        raise RedriveManifestError(
            f"re-drive manifest intake_snapshot_sha256 must be nonempty: {path}"
        )
    targets_raw = payload["targets"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise RedriveManifestError(
            f"re-drive manifest must name at least one target: {path}"
        )
    targets = tuple(targets_raw)
    if any(not isinstance(target, str) or not target for target in targets):
        raise RedriveManifestError(
            f"re-drive manifest targets must be nonempty stable IDs: {path}"
        )
    if len(set(targets)) != len(targets):
        raise RedriveManifestError(
            f"re-drive manifest targets must be unique stable IDs: {path}"
        )
    return RedriveManifest(
        reason=reason,
        intake_snapshot_sha256=snapshot_sha,
        targets=targets,
        manifest_sha256=str(payload["manifest_sha256"]),
    )
