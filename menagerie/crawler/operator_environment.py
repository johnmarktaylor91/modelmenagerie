"""Exact-lock conda environment operator for crawler lifecycle actions."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    EnvironmentExactnessError,
    parse_exact_lock,
    parse_resolved_export,
)
from menagerie.crawler.envs import (
    EnvironmentSpecError,
    load_environment_registry,
    overlay_package_rows,
    parse_pip_overlay_manifest,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    durable_fsync,
    fsync_directory,
    hash_bytes,
)
from menagerie.crawler.package_namespace import (
    dependency_spec_name,
    inventory_row_provides_distribution,
)

EXIT_SUCCESS = 0
EXIT_PERMANENT = 64
EXIT_RETRYABLE = 75
PROTOCOL_VERSION = "menagerie.crawler.environment-operator.v1"
READINESS_RECEIPT_VERSION = "menagerie.crawler.environment-readiness.v1"

_DEFAULT_CONDA_LOCK = Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda-lock")
_DEFAULT_CONDA = Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda")
# ``conda create --offline`` is not usable as the creation tool. Against conda 26.3.2 it
# nondeterministically re-downloads packages that are already staged and verified in the
# generation-local cache: the run leaves ``.partial`` files behind and fails with an
# OfflineError naming a different package each time. That was reproduced outside this
# operator, so it is conda's own offline-install behaviour and not a staging defect here.
# ``mamba create --offline`` on byte-identical staged inputs installs from the cache and
# produces a conda-meta inventory byte-identical to the resolved export, so mamba is the
# only accepted creation tool. There is deliberately no conda fallback: falling back would
# silently reintroduce corrupted offline installs behind a passing exit status.
_DEFAULT_MAMBA = Path("/opt/homebrew/Caskroom/miniforge/base/bin/mamba")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GLOB_CHARACTERS = frozenset("*?[]{}")
_TRANSIENT_MARKERS = (
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection error",
    "could not connect",
    "failed to establish a new connection",
    "http 408",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "name or service not known",
    "network is unreachable",
    "remote end closed connection",
    "resource temporarily unavailable",
    "temporary failure",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "tls handshake",
)
_DETAIL_LIMIT = 1_500
_PROBE_DETAIL_LIMIT = 1_000
_SOLVE_TIMEOUT_SECONDS = 30 * 60
_CREATE_TIMEOUT_SECONDS = 30 * 60
_PROBE_TIMEOUT_SECONDS = 5 * 60
_OVERLAY_TIMEOUT_SECONDS = 20 * 60
_DOWNLOAD_TIMEOUT_SECONDS = 5 * 60
_TRANSIENT_RETRIES = 2


class EnvironmentOperatorError(RuntimeError):
    """Base class for typed environment-operator failures."""


class PermanentEnvironmentOperatorError(EnvironmentOperatorError):
    """A contract, configuration, or exactness failure that must not be retried."""


class TransientEnvironmentOperatorError(EnvironmentOperatorError):
    """An exhausted spawn, transport, or temporary service failure."""


@dataclass(frozen=True)
class SolvedPackage:
    """One target-specific conda package parsed from the unified lock."""

    name: str
    version: str
    build: str
    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class CommandObservation:
    """One completed child process with bounded captured streams."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Mapping[str, str], float], CommandObservation]
Sleeper = Callable[[float], None]


class EnvironmentOperator:
    """Solve, create, probe, and remove exact conda environments.

    conda-lock solves the intent and mamba creates the prefix offline from the
    verified content-addressed cache; see :data:`_DEFAULT_MAMBA` for why the
    creation step must not run through conda.
    """

    def __init__(
        self,
        *,
        state_root: Path | None = None,
        conda_lock_command: Sequence[str] | None = None,
        conda_command: Sequence[str] | None = None,
        mamba_command: Sequence[str] | None = None,
        runner: CommandRunner | None = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        """Configure durable state, external commands, and retry injection points.

        Parameters
        ----------
        state_root:
            Optional absolute operator-state root. When omitted, the root is inferred
            from the action path and placed below the repository's ``.crawl-local``.
        conda_lock_command, conda_command:
            Optional argv prefixes used instead of the installed tools.
        mamba_command:
            Optional argv prefix for the offline creation tool. Its executable is
            required to exist before :meth:`create` touches any state, and conda is
            never substituted for it.
        runner:
            Injectable non-shell subprocess boundary.
        sleeper:
            Injectable retry backoff.
        """

        if state_root is not None and not state_root.is_absolute():
            raise PermanentEnvironmentOperatorError("operator state root must be absolute")
        self._state_root = state_root.resolve() if state_root is not None else None
        self._conda_lock_command = tuple(
            conda_lock_command or _default_tool_command("MENAGERIE_CONDA_LOCK", _DEFAULT_CONDA_LOCK)
        )
        self._conda_command = tuple(
            conda_command or _default_tool_command("MENAGERIE_CONDA", _DEFAULT_CONDA)
        )
        self._mamba_command = tuple(
            mamba_command or _default_tool_command("MENAGERIE_MAMBA", _DEFAULT_MAMBA)
        )
        self._runner = runner or _default_command_runner
        self._sleeper = sleeper

    def solve(
        self,
        environment_file: Path,
        target: str,
        *,
        force_resolve: bool = False,
    ) -> dict[str, Any]:
        """Solve one intent/target pair and return exact artifact receipts.

        Parameters
        ----------
        environment_file:
            Absolute conda environment specification.
        target:
            Exact conda-lock platform.
        force_resolve:
            Whether an operator explicitly requested replacement of a valid solve cache.

        Returns
        -------
        dict[str, Any]
            Driver-facing lock/export paths, measurements, cache state, and receipts.
        """

        started = time.monotonic()
        environment_file = _require_absolute_file(environment_file, "environment specification")
        _require_safe_component(target, "target")
        intent = environment_file.parent.name
        _require_safe_component(intent, "intent")
        specification = _load_mapping(environment_file, "environment specification")
        _refuse_pip_sections(specification)
        state_root = self._state_root_for(environment_file)
        solve_root = state_root / "solves" / intent / target
        solve_root.mkdir(parents=True, exist_ok=True)
        declared = _declared_dependency_specs(specification)
        committed_lock = _committed_lock_bytes(environment_file, target)
        with _exclusive_file_lock(solve_root / ".solve.lock"):
            if not force_resolve:
                cached = self._cached_solve_result(solve_root, state_root, declared, committed_lock)
                if cached is not None:
                    cached["elapsed_seconds"] = time.monotonic() - started
                    cached["cache_hit"] = True
                    return cached
            result = self._fresh_solve(
                environment_file=environment_file,
                target=target,
                solve_root=solve_root,
                state_root=state_root,
                started=started,
            )
            result["cache_hit"] = False
            return result

    def materialize(self, lock_file: Path) -> dict[str, Any]:
        """Hydrate the durable CAS directly from one committed exact lock.

        This action is intentionally independent of :meth:`solve`: it consumes
        only the exact URLs and SHA-256 digests already committed in the lock,
        and it never writes the lock family or computes a replacement setup
        identity. It is therefore the cold-host recovery door for the offline
        creation path.

        Parameters
        ----------
        lock_file:
            Absolute committed SHA-256 explicit lock to hydrate.

        Returns
        -------
        dict[str, Any]
            Lock identity plus every verified CAS path and downloaded-byte
            total. A repeated call reports zero downloaded bytes.
        """

        lock_file = _require_absolute_file(lock_file, "exact lock")
        lock_bytes = lock_file.read_bytes()
        try:
            receipts = parse_exact_lock(lock_bytes)
        except EnvironmentExactnessError as exc:
            raise PermanentEnvironmentOperatorError(str(exc)) from exc
        state_root = self._state_root_for(lock_file)
        artifacts: list[dict[str, str]] = []
        downloaded_bytes = 0
        for receipt in receipts:
            path, materialized_bytes = self._materialize_receipt(receipt, state_root)
            downloaded_bytes += materialized_bytes
            artifacts.append(
                {
                    "url": receipt.url,
                    "path": str(path),
                    "sha256": receipt.sha256,
                }
            )
        overlay_file = lock_file.with_name(
            f"{lock_file.name.removesuffix('.lock')}.pip-overlay.txt"
        )
        wheel_rows: tuple[Mapping[str, str], ...] = ()
        overlay_bytes: bytes | None = None
        downloaded_wheels = 0
        if overlay_file.is_file():
            overlay_bytes = overlay_file.read_bytes()
            try:
                wheel_rows = parse_pip_overlay_manifest(overlay_bytes)
            except EnvironmentSpecError as exc:
                raise PermanentEnvironmentOperatorError(str(exc)) from exc
            environment = _clean_environment(
                state_root=state_root,
                tool_commands=((sys.executable,),),
            )
            downloaded_wheels = self._materialize_overlay_rows(
                manifest_file=overlay_file,
                rows=wheel_rows,
                state_root=state_root,
                interpreter=Path(sys.executable),
                environment=environment,
            )
        if lock_file.read_bytes() != lock_bytes:
            raise PermanentEnvironmentOperatorError(
                "exact lock changed while its artifacts were being materialized"
            )
        if overlay_bytes is not None and overlay_file.read_bytes() != overlay_bytes:
            raise PermanentEnvironmentOperatorError(
                "pip-overlay manifest changed while its artifacts were being materialized"
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lock_path": str(lock_file),
            "lock_sha256": hash_bytes(lock_bytes),
            "artifacts_materialized": len(artifacts),
            "downloaded_bytes": downloaded_bytes,
            "artifacts": artifacts,
            "overlay_path": str(overlay_file) if overlay_bytes is not None else None,
            "wheels_materialized": len(wheel_rows),
            "downloaded_wheels": downloaded_wheels,
        }

    def create(self, lock_file: Path, prefix: Path) -> dict[str, Any]:
        """Create one prefix offline from a verified exact lock and local CAS.

        Parameters
        ----------
        lock_file:
            Absolute engine-facing SHA-256 explicit lock.
        prefix:
            Absolute exact environment prefix.

        Returns
        -------
        dict[str, Any]
            Created-prefix and generation-local package-cache paths.

        Raises
        ------
        PermanentEnvironmentOperatorError
            If the mamba creation tool is unavailable. See :data:`_DEFAULT_MAMBA`
            for why conda is not an accepted substitute.
        """

        self._require_creation_tool()
        lock_file = _require_absolute_file(lock_file, "exact lock")
        prefix = _require_removable_prefix(prefix)
        lock_bytes = lock_file.read_bytes()
        try:
            receipts = parse_exact_lock(lock_bytes)
        except EnvironmentExactnessError as exc:
            raise PermanentEnvironmentOperatorError(str(exc)) from exc
        state_root = self._state_root_for(lock_file)
        registration_path = _registration_path(state_root, prefix)
        with _exclusive_file_lock(registration_path.with_suffix(".lock")):
            if registration_path.is_file():
                self._remove_registered(prefix, state_root, registration_path)
            if prefix.exists() or prefix.is_symlink():
                raise PermanentEnvironmentOperatorError(
                    "create prefix already exists without an operator registration"
                )
            lock_sha256 = hash_bytes(lock_bytes)
            package_cache = (
                state_root / "package-caches" / f"{_prefix_identifier(prefix)}-{lock_sha256[7:19]}"
            )
            registration = {
                "protocol_version": PROTOCOL_VERSION,
                "prefix": str(prefix),
                "package_cache": str(package_cache),
                "lock_sha256": lock_sha256,
            }
            # The registration is durable BEFORE any cache bytes exist: a crash
            # at any later point leaves residue the registration names, so the
            # next create (or ``remove``) heals it through ``_remove_registered``.
            # The old order left an unnamed cache dir that bricked the
            # (prefix, lock) pair permanently (T1-19).
            atomic_replace_bytes(registration_path, canonical_json_bytes(registration) + b"\n")
            if package_cache.exists():
                # Registration-less residue from a pre-fix crash at this exact
                # deterministic address. Nothing named it until the write above,
                # so it is provably ours to replace.
                shutil.rmtree(package_cache)
            package_cache.mkdir(parents=True, exist_ok=False)
            try:
                self._stage_package_cache(receipts, package_cache, state_root)
                environment = _clean_environment(
                    prefix=prefix,
                    state_root=state_root,
                    package_cache=package_cache,
                    tool_commands=(self._mamba_command, self._conda_command),
                )
                command = (
                    *self._mamba_command,
                    "create",
                    "--yes",
                    "--offline",
                    "--prefix",
                    str(prefix),
                    "--file",
                    str(lock_file),
                )
                self._run_required_command(
                    command,
                    environment=environment,
                    timeout_seconds=_CREATE_TIMEOUT_SECONDS,
                    label="mamba create",
                )
            except Exception:
                # The registration intentionally survives so lifecycle ``finally`` can
                # remove a partial prefix and cache through the same exact target.
                raise
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "package_cache_path": str(package_cache),
        }

    def probe(self, prefix: Path, probes_value: str) -> dict[str, Any]:
        """Run declared canaries with the prefix interpreter and a clean environment.

        Parameters
        ----------
        prefix:
            Absolute created environment prefix.
        probes_value:
            Either the probe JSON itself or an absolute path to a JSON file.

        Returns
        -------
        dict[str, Any]
            Exactly one ordered boolean receipt per declared canary.
        """

        prefix = _require_absolute_directory(prefix, "environment prefix")
        interpreter = prefix / "bin" / "python"
        if not interpreter.is_file():
            raise PermanentEnvironmentOperatorError(
                f"environment interpreter is missing below {prefix}"
            )
        probes = _load_probe_request(probes_value)
        state_root = self._state_root_for(prefix)
        environment = _clean_environment(prefix=prefix, state_root=state_root)
        results: list[dict[str, Any]] = []
        for module in _string_list(probes.get("imports"), "probe imports"):
            observation = self._run_probe_command(
                (
                    str(interpreter),
                    "-I",
                    "-c",
                    "import importlib,sys; importlib.import_module(sys.argv[1])",
                    module,
                ),
                environment=environment,
                attempts=1,
            )
            results.append(
                _probe_result(
                    f"import:{module}",
                    observation.returncode == 0,
                    observation,
                )
            )
        for raw in _mapping_list(probes.get("export_checks"), "probe export checks"):
            module = _nonempty_string(raw.get("module"), "probe export module")
            attribute = _nonempty_string(raw.get("attribute"), "probe export attribute")
            observation = self._run_probe_command(
                (
                    str(interpreter),
                    "-I",
                    "-c",
                    (
                        "import importlib,sys; "
                        "getattr(importlib.import_module(sys.argv[1]),sys.argv[2])"
                    ),
                    module,
                    attribute,
                ),
                environment=environment,
                attempts=1,
            )
            results.append(
                _probe_result(
                    f"export:{module}:{attribute}",
                    observation.returncode == 0,
                    observation,
                )
            )
        for raw in _mapping_list(probes.get("source_build", []), "probe source-build checks"):
            name = _nonempty_string(raw.get("name"), "source-build probe name")
            command = _string_list(raw.get("command"), "source-build probe command")
            max_attempts = raw.get("max_attempts")
            if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
                raise PermanentEnvironmentOperatorError(
                    "source-build max_attempts must be a positive integer"
                )
            if max_attempts < 1:
                raise PermanentEnvironmentOperatorError(
                    "source-build max_attempts must be a positive integer"
                )
            resolved_command = _prefix_python_command(command, interpreter)
            observation = self._run_probe_command(
                resolved_command,
                environment=environment,
                attempts=max_attempts,
            )
            results.append(
                _probe_result(
                    f"source-build:{name}",
                    observation.returncode == 0,
                    observation,
                )
            )
        return {"protocol_version": PROTOCOL_VERSION, "results": results}

    def overlay(self, prefix: Path, manifest_file: Path) -> dict[str, Any]:
        """Install one committed pip-overlay manifest into a registered prefix.

        This is the controlled overlay the rung-12 censuses converged on, and
        the opposite of an ambient ``pip install``:

        * every requirement arrives from the COMMITTED pinned manifest (closed
          ``name==version --hash=sha256:...`` grammar; option lines, URLs,
          extras, markers, and unpinned or unhashed rows are refused at parse);
        * wheels are materialized into the shared wheel CAS once, verified
          against the manifest digests, and reused offline forever after
          (a warm CAS makes the whole action network-free);
        * source builds are refused outright (``--only-binary=:all:`` at both
          download and install);
        * installation runs strictly offline (``--no-index --no-deps
          --require-hashes --find-links`` against a generation-local wheelhouse
          built from the CAS), into the registered prefix only;
        * an overlay row that would shadow a distribution the conda solve
          already installs is refused -- the overlay adds what conda lacks and
          never rewrites what conda provides;
        * the generation-local wheelhouse joins the prefix registration, so
          teardown removes it and ``owned-paths`` residue accounting covers it.

        Parameters
        ----------
        prefix:
            Absolute created environment prefix previously registered by
            :meth:`create`.
        manifest_file:
            Absolute committed pinned manifest.

        Returns
        -------
        dict[str, Any]
            Installed rows, wheelhouse path, and CAS materialization counts.
        """

        prefix = _require_removable_prefix(prefix)
        manifest_file = _require_absolute_file(manifest_file, "pip-overlay manifest")
        interpreter = prefix / "bin" / "python"
        if not interpreter.is_file():
            raise PermanentEnvironmentOperatorError(
                f"environment interpreter is missing below {prefix}; overlay intents "
                "must declare 'pip' among their conda dependencies"
            )
        manifest_bytes = manifest_file.read_bytes()
        try:
            rows = parse_pip_overlay_manifest(manifest_bytes)
        except EnvironmentSpecError as exc:
            raise PermanentEnvironmentOperatorError(str(exc)) from exc
        state_root = self._state_root_for(prefix)
        registration_path = _registration_path(state_root, prefix)
        if not registration_path.is_file():
            raise PermanentEnvironmentOperatorError(
                "overlay target is not an exact registered environment prefix"
            )
        environment = _clean_environment(prefix=prefix, state_root=state_root)

        # Refuse shadowing before anything is fetched or installed: the conda
        # resolved export is the authority for everything it provides.
        names = [str(row["name"]) for row in rows]
        observation = self._run_probe_command(
            (
                str(interpreter),
                "-c",
                (
                    "import importlib.metadata as m, json, sys\n"
                    "found = {}\n"
                    "for name in sys.argv[1:]:\n"
                    "    try:\n"
                    "        found[name] = m.version(name)\n"
                    "    except m.PackageNotFoundError:\n"
                    "        pass\n"
                    "print(json.dumps(found))"
                ),
                *names,
            ),
            environment=environment,
            attempts=1,
        )
        if observation.returncode != 0:
            raise PermanentEnvironmentOperatorError(
                f"overlay pre-install inventory scan failed: "
                f"{_bounded_detail(_combined_detail(observation))}"
            )
        already_installed = json.loads(observation.stdout or "{}")
        if already_installed:
            listed = ", ".join(sorted(already_installed))
            raise PermanentEnvironmentOperatorError(
                f"pip-overlay rows shadow conda-provided distributions: {listed}; "
                "the overlay adds only what the conda solve lacks"
            )

        downloaded = self._materialize_overlay_rows(
            manifest_file=manifest_file,
            rows=rows,
            state_root=state_root,
            interpreter=interpreter,
            environment=environment,
        )
        wheel_cas = state_root / "wheel-cas"

        wheelhouse = (
            state_root
            / "wheelhouses"
            / f"{_prefix_identifier(prefix)}-{hash_bytes(manifest_bytes)[7:19]}"
        )
        with _exclusive_file_lock(registration_path.with_suffix(".lock")):
            registration = dict(_load_mapping(registration_path, "prefix registration"))
            if registration.get("prefix") != str(prefix):
                raise PermanentEnvironmentOperatorError("prefix registration target mismatch")
            if wheelhouse.exists():
                shutil.rmtree(wheelhouse)
            wheelhouse.mkdir(parents=True, exist_ok=False)
            for row in rows:
                source = _wheel_cas_file(wheel_cas, str(row["sha256"]))
                if source is None:
                    # Verified-covered above, so a miss here means the object was
                    # corrupted (and quarantined) or removed between the checks.
                    raise PermanentEnvironmentOperatorError(
                        f"wheel CAS entry for {row['name']!r} failed verification "
                        "between materialization and install staging"
                    )
                destination = wheelhouse / source.name
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            registration["wheelhouse"] = str(wheelhouse)
            atomic_replace_bytes(registration_path, canonical_json_bytes(registration) + b"\n")
        self._run_required_command(
            (
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--only-binary",
                ":all:",
                "--require-hashes",
                "--no-warn-script-location",
                "--find-links",
                str(wheelhouse),
                "-r",
                str(manifest_file),
            ),
            environment=environment,
            timeout_seconds=_OVERLAY_TIMEOUT_SECONDS,
            label="pip-overlay offline install",
        )
        observation = self._run_probe_command(
            (
                str(interpreter),
                "-c",
                (
                    "import importlib.metadata as m, json, sys\n"
                    "print(json.dumps({name: m.version(name) for name in sys.argv[1:]}))"
                ),
                *names,
            ),
            environment=environment,
            attempts=1,
        )
        if observation.returncode != 0:
            raise PermanentEnvironmentOperatorError(
                "overlay post-install verification failed: "
                f"{_bounded_detail(_combined_detail(observation))}"
            )
        installed_versions = json.loads(observation.stdout or "{}")
        for row in rows:
            if installed_versions.get(str(row["name"])) != str(row["version"]):
                raise PermanentEnvironmentOperatorError(
                    f"installed overlay distribution {row['name']!r} is "
                    f"{installed_versions.get(str(row['name']))!r}, not the pinned "
                    f"{row['version']!r}"
                )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "wheelhouse": str(wheelhouse),
            "manifest_sha256": hash_bytes(manifest_bytes),
            "packages": [dict(row) for row in rows],
            "downloaded_artifacts": downloaded,
            "cache_hits": len(rows) - downloaded,
        }

    def _materialize_overlay_rows(
        self,
        *,
        manifest_file: Path,
        rows: Sequence[Mapping[str, str]],
        state_root: Path,
        interpreter: Path,
        environment: Mapping[str, str],
    ) -> int:
        """Hydrate pinned binary overlay rows into the shared wheel CAS.

        Parameters
        ----------
        manifest_file:
            Exact hash-pinned overlay requirements file.
        rows:
            Parsed closed manifest rows.
        state_root:
            Environment operator state root.
        interpreter:
            Python interpreter whose pip performs the network materialization.
        environment:
            Bounded child environment.

        Returns
        -------
        int
            Number of newly promoted wheel objects.
        """

        wheel_cas = state_root / "wheel-cas"
        # Byte-verified presence, not bare `is_dir()`: a corrupt entry is quarantined
        # by the lookup itself, lands in `missing`, and is re-downloaded in THIS pass
        # instead of shadowing the digest until an operator intervenes.
        missing = [
            row for row in rows if _wheel_cas_file(wheel_cas, str(row["sha256"])) is None
        ]
        downloaded = 0
        if missing:
            manifest_hashes = {str(row["sha256"]) for row in rows}
            download_root = state_root / "tmp" / f"overlay-download-{uuid.uuid4().hex[:12]}"
            download_root.mkdir(parents=True, exist_ok=False)
            try:
                fetch_environment = dict(environment)
                fetch_environment.pop("PIP_NO_INDEX", None)
                self._run_required_command(
                    (
                        str(interpreter),
                        "-m",
                        "pip",
                        "download",
                        "--no-deps",
                        "--only-binary",
                        ":all:",
                        "--require-hashes",
                        "--dest",
                        str(download_root),
                        "-r",
                        str(manifest_file),
                    ),
                    environment=fetch_environment,
                    timeout_seconds=_OVERLAY_TIMEOUT_SECONDS,
                    label="pip-overlay download",
                )
                for artifact in sorted(download_root.iterdir()):
                    if not artifact.is_file():
                        continue
                    digest = _stream_sha256(artifact)
                    if digest not in manifest_hashes:
                        raise PermanentEnvironmentOperatorError(
                            "downloaded overlay artifact digest is not pinned by the "
                            f"manifest: {artifact.name}"
                        )
                    entry = _wheel_cas_entry(wheel_cas, digest)
                    if entry.is_dir():
                        continue
                    staging = entry.with_name(f"{entry.name}.staging-{uuid.uuid4().hex[:8]}")
                    staging.mkdir(parents=True, exist_ok=False)
                    shutil.copy2(artifact, staging / artifact.name)
                    os.rename(staging, entry)
                    downloaded += 1
            finally:
                shutil.rmtree(download_root, ignore_errors=True)
        uncovered = [
            str(row["name"])
            for row in rows
            if _wheel_cas_file(wheel_cas, str(row["sha256"])) is None
        ]
        if uncovered:
            raise PermanentEnvironmentOperatorError(
                "wheel CAS lacks pinned overlay artifacts after materialization: "
                + ", ".join(sorted(uncovered))
            )
        return downloaded

    def remove(self, prefix: Path) -> dict[str, Any]:
        """Remove only an exact registered prefix and its generation-local cache.

        Parameters
        ----------
        prefix:
            Absolute exact prefix previously registered by :meth:`create`.

        Returns
        -------
        dict[str, Any]
            Exact removed prefix; shared CAS objects are retained.
        """

        prefix = _require_removable_prefix(prefix)
        state_root = self._state_root_for(prefix)
        registration_path = _registration_path(state_root, prefix)
        with _exclusive_file_lock(registration_path.with_suffix(".lock")):
            self._remove_registered(prefix, state_root, registration_path)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "cas_retained": True,
        }

    def owned_paths(self, prefix: Path) -> dict[str, Any]:
        """Enumerate every generation-owned filesystem root for one prefix.

        This is the create/remove residue receipt the sequential lifecycle
        absence-checks after teardown: the prefix, its durable registration,
        and the registered generation-local package cache. Shared CAS objects
        are deliberately NOT owned (they outlive generations by design).

        Parameters
        ----------
        prefix:
            Absolute exact prefix registered (or registerable) by :meth:`create`.

        Returns
        -------
        dict[str, Any]
            Protocol version plus the owned path inventory. A prefix without a
            live registration owns only itself and the registration slot.
        """

        prefix = _require_removable_prefix(prefix)
        state_root = self._state_root_for(prefix)
        registration_path = (
            state_root / "prefix-registrations" / f"{_prefix_identifier(prefix)}.json"
        )
        paths = [prefix, registration_path]
        if registration_path.is_file():
            registration = _load_mapping(registration_path, "prefix registration")
            package_cache = registration.get("package_cache")
            if isinstance(package_cache, str) and package_cache:
                paths.append(Path(package_cache))
            wheelhouse = registration.get("wheelhouse")
            if isinstance(wheelhouse, str) and wheelhouse:
                paths.append(Path(wheelhouse))
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "paths": [str(path) for path in paths],
        }

    def state_root(self, anchor: Path) -> dict[str, Any]:
        """Report the sole operator-state root resolved for an action anchor.

        Parameters
        ----------
        anchor:
            Absolute lock or environment path a later action will consume.

        Returns
        -------
        dict[str, Any]
            Protocol receipt naming the exact root every operator write uses.
        """

        if not anchor.is_absolute():
            raise PermanentEnvironmentOperatorError("state-root anchor must be absolute")
        root = self._state_root_for(anchor)
        return {"protocol_version": PROTOCOL_VERSION, "state_root": str(root)}

    def _fresh_solve(
        self,
        *,
        environment_file: Path,
        target: str,
        solve_root: Path,
        state_root: Path,
        started: float,
    ) -> dict[str, Any]:
        """Run conda-lock, materialize CAS bytes, and publish a complete solve."""

        with tempfile.TemporaryDirectory(prefix=".solve-", dir=solve_root) as temporary_value:
            temporary_root = Path(temporary_value)
            unified_lock = temporary_root / "conda-lock.yml"
            command = (
                *self._conda_lock_command,
                "lock",
                "--file",
                str(environment_file),
                "-p",
                target,
                "--lockfile",
                str(unified_lock),
                "--mamba",
            )
            self._run_required_command(
                command,
                environment=_clean_environment(
                    state_root=state_root,
                    tool_commands=(self._conda_lock_command, self._conda_command),
                ),
                timeout_seconds=_SOLVE_TIMEOUT_SECONDS,
                label="conda-lock solve",
            )
            if not unified_lock.is_file():
                raise PermanentEnvironmentOperatorError(
                    "conda-lock exited successfully without its requested lockfile"
                )
            packages = _parse_solved_packages(unified_lock, target)
            lock_bytes = _explicit_sha256_lock_bytes(packages)
            export_bytes = _resolved_export_bytes(packages)
            _require_lock_export_match(lock_bytes, export_bytes)
            artifact_bytes = 0
            artifacts: list[dict[str, str]] = []
            for package in packages:
                path, downloaded_bytes = self._materialize_artifact(package, state_root)
                artifact_bytes += downloaded_bytes
                artifacts.append(
                    {
                        "url": package.url,
                        "path": str(path),
                        "sha256": f"sha256:{package.sha256}",
                    }
                )
            published_lock = solve_root / "explicit.lock"
            published_export = solve_root / "resolved.json"
            published_unified = solve_root / "conda-lock.yml"
            unified_bytes = unified_lock.read_bytes()
            atomic_replace_bytes(published_lock, lock_bytes)
            atomic_replace_bytes(published_export, export_bytes)
            atomic_replace_bytes(published_unified, unified_bytes)
            manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "intent": environment_file.parent.name,
                "target": target,
                "lock_sha256": hash_bytes(lock_bytes),
                "resolved_export_sha256": hash_bytes(export_bytes),
                # Binds the unified-lock convenience copy so a torn solve
                # commit (a crash between these member writes) is detectable:
                # the rotation republication refuses a unified lock that
                # contradicts this digest instead of shipping another solve's
                # bytes into the campaign cache.
                "conda_lock_sha256": hash_bytes(unified_bytes),
                "artifacts": artifacts,
            }
            atomic_replace_bytes(
                solve_root / "manifest.json", canonical_json_bytes(manifest) + b"\n"
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lock_path": str(published_lock),
            "resolved_export_path": str(published_export),
            "artifacts": artifacts,
            "artifact_bytes": artifact_bytes,
            "elapsed_seconds": time.monotonic() - started,
        }

    def _cached_solve_result(
        self,
        solve_root: Path,
        state_root: Path,
        declared_dependencies: Sequence[str],
        committed_lock_bytes: bytes | None = None,
    ) -> dict[str, Any] | None:
        """Return a fully reverified solve cache or ``None`` for any cache miss.

        A cached solve is keyed only by intent and target, while the declared
        specification keeps evolving underneath it. A solve satisfies every
        requested package by construction, so a cached resolved export that
        omits a declared dependency name is PROOF the cache predates the
        declaration: it is superseded and reads as no cache at all. Version
        drift alone never supersedes -- a version is a solver outcome, not part
        of the currency question -- so an unchanged dependency set keeps the
        cache (and the environment generation) stable. This mirrors the
        engine-side ``LockArtifacts`` supersession rule one layer down: without
        it, the 2026-08-04 pilot rung admitted models against the current
        declaration, then served them a pre-declaration cached solve, and every
        model needing the new package died at the environment probes.

        Currency additionally includes the COMMITTED lock family (T1-04,
        S2-confirmed): when a committed ``locks/<target>.lock`` exists next to
        the specification, a cached lock with different bytes either predates a
        completed rotation or diverged from it -- serving it would silently
        revert the rotation on the next leg. Names-only currency cannot see
        that (the rotation may change only versions/builds), so the committed
        family supersedes the cache outright.
        """

        lock_path = solve_root / "explicit.lock"
        export_path = solve_root / "resolved.json"
        if not lock_path.is_file() or not export_path.is_file():
            return None
        try:
            lock_bytes = lock_path.read_bytes()
            export_bytes = export_path.read_bytes()
            receipts = parse_exact_lock(lock_bytes)
            if parse_resolved_export(export_bytes) != export_bytes:
                return None
            _require_lock_export_match(lock_bytes, export_bytes)
        except (OSError, EnvironmentExactnessError, PermanentEnvironmentOperatorError):
            return None
        if committed_lock_bytes is not None and lock_bytes != committed_lock_bytes:
            return None
        if _missing_declared_dependency(declared_dependencies, export_bytes) is not None:
            return None
        artifacts: list[dict[str, str]] = []
        for receipt in receipts:
            path = _cas_path(state_root, receipt.sha256)
            if not path.is_file() or _stream_sha256(path) != receipt.sha256:
                return None
            artifacts.append({"url": receipt.url, "path": str(path), "sha256": receipt.sha256})
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lock_path": str(lock_path),
            "resolved_export_path": str(export_path),
            "artifacts": artifacts,
            "artifact_bytes": 0,
        }

    def _materialize_artifact(self, package: SolvedPackage, state_root: Path) -> tuple[Path, int]:
        """Download one immutable artifact into the content-addressed store."""

        receipt = ArtifactReceipt(
            url=package.url,
            sha256=f"sha256:{package.sha256}",
        )
        return self._materialize_receipt(receipt, state_root)

    def _materialize_receipt(self, receipt: ArtifactReceipt, state_root: Path) -> tuple[Path, int]:
        """Download one lock-authorized receipt into the content-addressed store."""

        destination = _cas_path(state_root, receipt.sha256)
        expected = receipt.sha256.removeprefix("sha256:")
        filename = _artifact_filename(receipt.url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if _stream_sha256(destination) == receipt.sha256:
                return destination, 0
            destination.unlink()
            fsync_directory(destination.parent)
        last_error: BaseException | None = None
        for attempt in range(_TRANSIENT_RETRIES + 1):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                request = urllib.request.Request(
                    receipt.url,
                    headers={"User-Agent": "torchlens-menagerie-environment-operator/1"},
                )
                digest = hashlib.sha256()
                size = 0
                with urllib.request.urlopen(  # noqa: S310 -- lock-authorized artifact URL
                    request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response:
                    with temporary.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            handle.write(chunk)
                            size += len(chunk)
                        handle.flush()
                        durable_fsync(handle.fileno())
                if digest.hexdigest() != expected:
                    raise PermanentEnvironmentOperatorError(
                        f"downloaded artifact digest mismatch for {filename}"
                    )
                os.replace(temporary, destination)
                fsync_directory(destination.parent)
                return destination, size
            except PermanentEnvironmentOperatorError:
                temporary.unlink(missing_ok=True)
                raise
            except urllib.error.HTTPError as exc:
                temporary.unlink(missing_ok=True)
                if exc.code not in {408, 425, 429} and exc.code < 500:
                    raise PermanentEnvironmentOperatorError(
                        f"artifact service permanently rejected {filename}: HTTP {exc.code}"
                    ) from exc
                last_error = exc
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                temporary.unlink(missing_ok=True)
                last_error = exc
            if attempt < _TRANSIENT_RETRIES:
                self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"artifact download remained unavailable for {filename}"
        ) from last_error

    def _require_creation_tool(self) -> None:
        """Refuse to create anything unless the mamba executable is really present.

        Raises
        ------
        PermanentEnvironmentOperatorError
            If the configured creation command names no executable file.

        The check runs before the prefix registration, the package cache, and the
        CAS staging, so a host without mamba leaves no partial state behind. It
        fails loudly on purpose: the alternative -- quietly creating the prefix
        with ``conda create --offline`` -- produces environments whose installed
        inventory silently disagrees with the verified lock (see
        :data:`_DEFAULT_MAMBA`), which is exactly the corruption this operator
        exists to prevent.
        """

        configured = self._mamba_command[0]
        # A bare program name is resolved the way the child would resolve it, so an
        # operator may still spell the tool by name rather than by absolute path.
        resolved = configured if os.sep in configured else (shutil.which(configured) or configured)
        executable = Path(resolved)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise PermanentEnvironmentOperatorError(
                "offline environment creation requires the mamba executable, and "
                f"{str(executable)!r} is not one; set MENAGERIE_MAMBA to an installed "
                "mamba. conda is not an accepted substitute because its offline "
                "create re-downloads staged packages and corrupts the prefix"
            )

    def _stage_package_cache(
        self,
        receipts: Sequence[ArtifactReceipt],
        package_cache: Path,
        state_root: Path,
    ) -> None:
        """Stage verified CAS objects under conda's artifact filenames."""

        filenames: dict[str, str] = {}
        for receipt in receipts:
            filename = _artifact_filename(receipt.url)
            previous = filenames.setdefault(filename, receipt.sha256)
            if previous != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"lock maps artifact filename {filename!r} to multiple digests"
                )
            source = _cas_path(state_root, receipt.sha256)
            if not source.is_file() or _stream_sha256(source) != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"verified CAS artifact is missing for {filename}"
                )
            destination = package_cache / filename
            shutil.copyfile(source, destination)
            if _stream_sha256(destination) != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"staged package-cache artifact changed for {filename}"
                )
        fsync_directory(package_cache)

    def _run_required_command(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout_seconds: float,
        label: str,
    ) -> CommandObservation:
        """Run a required subprocess with typed transient-only retries."""

        last_error: BaseException | None = None
        for attempt in range(_TRANSIENT_RETRIES + 1):
            try:
                observation = self._runner(command, environment, timeout_seconds)
            except FileNotFoundError as exc:
                raise PermanentEnvironmentOperatorError(
                    f"{label} executable is unavailable"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                last_error = exc
            else:
                if observation.returncode == 0:
                    return observation
                detail = _combined_detail(observation)
                if observation.returncode != EXIT_RETRYABLE and not _is_transient_detail(detail):
                    raise PermanentEnvironmentOperatorError(
                        f"{label} failed permanently: {_bounded_detail(detail)}"
                    )
                last_error = RuntimeError(_bounded_detail(detail))
            if attempt < _TRANSIENT_RETRIES:
                self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"{label} remained unavailable after {_TRANSIENT_RETRIES + 1} attempts"
        ) from last_error

    def _run_probe_command(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        attempts: int,
    ) -> CommandObservation:
        """Run a canary, retrying only spawn/timeout infrastructure failures."""

        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                return self._runner(command, environment, _PROBE_TIMEOUT_SECONDS)
            except FileNotFoundError as exc:
                raise PermanentEnvironmentOperatorError("probe executable is unavailable") from exc
            except (OSError, subprocess.SubprocessError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"probe subprocess remained unavailable after {attempts} attempts"
        ) from last_error

    def _remove_registered(self, prefix: Path, state_root: Path, registration_path: Path) -> None:
        """Validate and remove one prefix registration without touching shared CAS."""

        _remove_registered_paths(prefix, state_root, registration_path)

    def _state_root_for(self, anchor: Path) -> Path:
        """Return and create the explicit or repository-inferred state root."""

        root = resolve_environment_state_root(anchor, explicit=self._state_root)
        root.mkdir(parents=True, exist_ok=True)
        return root


def _default_tool_command(environment_key: str, fixed_path: Path) -> tuple[str, ...]:
    """Resolve one configured, PATH-provided, or machine-baseline tool command."""

    configured = os.environ.get(environment_key)
    if configured:
        return (configured,)
    discovered = shutil.which(fixed_path.name)
    if discovered is not None:
        return (discovered,)
    return (str(fixed_path),)


def _default_command_runner(
    command: Sequence[str], environment: Mapping[str, str], timeout_seconds: float
) -> CommandObservation:
    """Run one argv-only child with captured bounded text streams."""

    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        start_new_session=True,
        env=dict(environment),
    )
    return CommandObservation(
        returncode=completed.returncode,
        stdout=completed.stdout[-_DETAIL_LIMIT:],
        stderr=completed.stderr[-_DETAIL_LIMIT:],
    )


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one JSON/YAML file and require a string-keyed mapping.

    Release-run inputs (prefix registrations, resolved lock exports) are canonical
    JSON and parse without a YAML engine. Only genuine YAML specs (setup/solve time)
    fall through to PyYAML, which is imported lazily so the crawler's run path inside
    a materialized release prefix never requires ``yaml`` to be installed there.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PermanentEnvironmentOperatorError(f"{label} is invalid: {path}") from exc
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # lazy: only genuine YAML specs (setup time) reach here

        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PermanentEnvironmentOperatorError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PermanentEnvironmentOperatorError(f"{label} must be a string-keyed mapping")
    return value


def _refuse_pip_sections(value: Any) -> None:
    """Refuse any mapping-declared pip installation lane recursively."""

    if isinstance(value, Mapping):
        if "pip" in value:
            raise PermanentEnvironmentOperatorError(
                "pip sections are unsupported by exact conda-meta inventory"
            )
        for nested in value.values():
            _refuse_pip_sections(nested)
    elif isinstance(value, list):
        for nested in value:
            _refuse_pip_sections(nested)


def _committed_lock_bytes(environment_file: Path, target: str) -> bytes | None:
    """Read the committed lock family member next to one specification.

    Parameters
    ----------
    environment_file:
        Absolute intent specification (``envs/<intent>/environment.yml``).
    target:
        Exact conda-lock platform.

    Returns
    -------
    bytes | None
        Committed ``locks/<target>.lock`` bytes, or ``None`` when the intent
        has no committed family (or it is unreadable -- currency then falls
        back to names-only, and the launch lane's own family validation owns
        the refusal).
    """

    committed = environment_file.parent / "locks" / f"{target}.lock"
    try:
        return committed.read_bytes() if committed.is_file() else None
    except OSError:
        return None


def _declared_dependency_specs(specification: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the string dependency match specs one specification declares.

    Parameters
    ----------
    specification:
        Loaded environment specification mapping.

    Returns
    -------
    tuple[str, ...]
        Every string entry of the ``dependencies`` list, in declaration order.
        Non-list and non-string shapes contribute nothing: pip sections are
        refused before this runs, and any other malformed shape is conda-lock's
        refusal to own, not the cache-currency question's.
    """

    dependencies = specification.get("dependencies")
    if not isinstance(dependencies, list):
        return ()
    return tuple(entry for entry in dependencies if isinstance(entry, str))


def _missing_declared_dependency(
    declared_dependencies: Sequence[str], export_bytes: bytes
) -> str | None:
    """Name one declared dependency a resolved export fails to install.

    Same rule as the engine-side ``LockArtifacts.missing_declared_dependency``:
    names only, never versions or bounds, and an unreadable export is owned by
    the exactness checks rather than read as evidence of supersession.

    Parameters
    ----------
    declared_dependencies:
        Declared dependency match specs from the current specification.
    export_bytes:
        Candidate cached resolved-export bytes.

    Returns
    -------
    str | None
        The first declared dependency name no export row provides, or ``None``
        when every declared name is present or the export is unreadable.
    """

    if not declared_dependencies:
        return None
    try:
        value = json.loads(export_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        return None
    row_names = [
        str(row["name"])
        for row in packages
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    ]
    for spec in declared_dependencies:
        name = dependency_spec_name(spec)
        if not name:
            continue
        if not any(inventory_row_provides_distribution(row_name, name) for row_name in row_names):
            return name
    return None


def _parse_solved_packages(path: Path, target: str) -> tuple[SolvedPackage, ...]:
    """Parse target conda packages and derive build strings from artifact filenames."""

    value = _load_mapping(path, "conda-lock output")
    raw_packages = value.get("package")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise PermanentEnvironmentOperatorError("conda-lock output has no package rows")
    packages: list[SolvedPackage] = []
    for raw in raw_packages:
        if not isinstance(raw, Mapping):
            raise PermanentEnvironmentOperatorError(
                "conda-lock output contains a non-object package row"
            )
        platform = _nonempty_string(raw.get("platform"), "locked package platform")
        if platform != target:
            continue
        manager = _nonempty_string(raw.get("manager"), "locked package manager")
        if manager != "conda":
            raise PermanentEnvironmentOperatorError(
                f"locked manager {manager!r} is unsupported; pip is out of scope"
            )
        name = _nonempty_string(raw.get("name"), "locked package name")
        version = _nonempty_string(raw.get("version"), "locked package version")
        url = _nonempty_string(raw.get("url"), "locked package URL")
        raw_hash = raw.get("hash")
        if not isinstance(raw_hash, Mapping):
            raise PermanentEnvironmentOperatorError("locked package hash is missing")
        sha256 = _nonempty_string(raw_hash.get("sha256"), "locked package SHA-256")
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise PermanentEnvironmentOperatorError(
                "locked package SHA-256 must be 64 lowercase hex characters"
            )
        filename = _artifact_filename(url)
        build = _build_from_filename(filename, name, version)
        packages.append(
            SolvedPackage(
                name=name,
                version=version,
                build=build,
                url=url,
                sha256=sha256,
                filename=filename,
            )
        )
    if not packages:
        raise PermanentEnvironmentOperatorError(
            f"conda-lock output has no conda packages for target {target!r}"
        )
    packages.sort(key=lambda package: (package.url, package.sha256))
    urls = [package.url for package in packages]
    if len(urls) != len(set(urls)):
        raise PermanentEnvironmentOperatorError(
            "conda-lock output contains duplicate target artifact URLs"
        )
    return tuple(packages)


def _artifact_filename(url: str) -> str:
    """Return a safe conda artifact filename from one exact URL."""

    filename = unquote(Path(urlsplit(url).path).name)
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or not filename.endswith((".conda", ".tar.bz2"))
    ):
        raise PermanentEnvironmentOperatorError(
            "locked artifact URL has no supported conda filename"
        )
    return filename


def _build_from_filename(filename: str, name: str, version: str) -> str:
    """Derive the conda build string absent from unified conda-lock rows."""

    stem = (
        filename[: -len(".tar.bz2")]
        if filename.endswith(".tar.bz2")
        else filename[: -len(".conda")]
    )
    prefix = f"{name}-{version}-"
    if not stem.startswith(prefix) or not stem[len(prefix) :]:
        raise PermanentEnvironmentOperatorError(
            f"artifact filename does not encode name/version/build: {filename}"
        )
    return stem[len(prefix) :]


def _explicit_sha256_lock_bytes(packages: Sequence[SolvedPackage]) -> bytes:
    """Synthesize the engine-facing SHA-256 explicit lock without conda-lock render."""

    lines = [f"{package.url}#{package.sha256}" for package in packages]
    if len(lines) != len(set(lines)):
        raise PermanentEnvironmentOperatorError(
            "solved packages do not form a unique explicit lock"
        )
    return ("@EXPLICIT\n" + "\n".join(lines) + "\n").encode()


def _resolved_export_bytes(packages: Sequence[SolvedPackage]) -> bytes:
    """Serialize the five-field inventory from the same solved package rows."""

    rows = [
        {
            "name": package.name,
            "version": package.version,
            "build": package.build,
            "url": package.url,
            "sha256": f"sha256:{package.sha256}",
        }
        for package in packages
    ]
    rows.sort(
        key=lambda row: (
            row["name"],
            row["version"],
            row["build"],
            row["url"],
            row["sha256"],
        )
    )
    return canonical_json_bytes({"packages": rows}) + b"\n"


def _require_lock_export_match(lock_bytes: bytes, export_bytes: bytes) -> None:
    """Require exact one-to-one URL/digest identity across lock and export."""

    try:
        lock_receipts = parse_exact_lock(lock_bytes)
        canonical_export = parse_resolved_export(export_bytes)
        value = json.loads(canonical_export)
    except (EnvironmentExactnessError, json.JSONDecodeError) as exc:
        raise PermanentEnvironmentOperatorError(str(exc)) from exc
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        raise PermanentEnvironmentOperatorError("resolved export has no package rows")
    export_receipts = tuple(
        (row.get("url"), row.get("sha256")) for row in packages if isinstance(row, Mapping)
    )
    if len(export_receipts) != len(packages) or set(export_receipts) != {
        (receipt.url, receipt.sha256) for receipt in lock_receipts
    }:
        raise PermanentEnvironmentOperatorError(
            "target lock and resolved export artifact identities differ"
        )


def _cas_path(state_root: Path, sha256: str) -> Path:
    """Return the content-addressed path for one canonical SHA-256."""

    digest = sha256.removeprefix("sha256:")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise PermanentEnvironmentOperatorError("CAS digest is not canonical SHA-256")
    return state_root / "artifact-cas" / "sha256" / digest


def _wheel_cas_entry(wheel_cas: Path, sha256: str) -> Path:
    """Return the wheel-CAS directory for one canonical artifact digest.

    Wheel entries are DIRECTORIES keyed by digest, each holding the artifact
    under its original filename: pip resolves wheels by filename tags, so the
    name must survive content addressing.
    """

    digest = sha256.removeprefix("sha256:")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise PermanentEnvironmentOperatorError("wheel CAS digest is not canonical SHA-256")
    return wheel_cas / "sha256" / digest


def _quarantine_wheel_cas_entry(wheel_cas: Path, entry: Path, reason: str) -> None:
    """Move one verification-violating wheel-CAS entry aside, or fail closed.

    The aside-rename is what makes the store self-heal: a quarantined entry stops
    satisfying presence checks, so the next materialization re-downloads the pinned
    artifact instead of wedging on (or worse, trusting) the corrupt object. The
    quarantine directory is deliberately outside ``wheel-cas/sha256/`` so no lookup
    can ever resolve into it; entries are named for their digest and violation so an
    operator can audit what was evicted and why.
    """

    quarantine_root = wheel_cas / "quarantine"
    try:
        quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = quarantine_root / f"{entry.name}.{reason}.{uuid.uuid4().hex[:8]}"
        os.rename(entry, destination)
    except OSError as exc:
        raise PermanentEnvironmentOperatorError(
            f"wheel CAS entry {entry.name} failed verification ({reason}) and could "
            f"not be quarantined: {exc}"
        ) from exc


def _wheel_cas_file(wheel_cas: Path, sha256: str) -> Path | None:
    """Return the byte-verified stored artifact for one digest, or ``None``.

    EN-3 (reproduced 200a probe): this lookup used to trust the sole FILENAME in the
    digest directory, so wrong bytes planted under a right digest were admitted and
    the readiness receipt sealed green over a corrupt store. Every lookup now streams
    the candidate's bytes against the digest and accepts only an entry holding
    exactly one regular, non-symlinked file; anything else is quarantined aside and
    reported as a miss (never admitted).
    """

    entry = _wheel_cas_entry(wheel_cas, sha256)
    if entry.is_symlink():
        _quarantine_wheel_cas_entry(wheel_cas, entry, "entry-symlink")
        return None
    if not entry.is_dir():
        return None
    children = sorted(entry.iterdir())
    files = [path for path in children if path.is_file() and not path.is_symlink()]
    if len(children) != 1 or len(files) != 1:
        _quarantine_wheel_cas_entry(wheel_cas, entry, "not-exactly-one-regular-file")
        return None
    expected = "sha256:" + sha256.removeprefix("sha256:")
    if _stream_sha256(files[0]) != expected:
        _quarantine_wheel_cas_entry(wheel_cas, entry, "digest-mismatch")
        return None
    return files[0]


def _stream_sha256(path: Path) -> str:
    """Hash one artifact without loading the package into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _clean_environment(
    *,
    state_root: Path,
    prefix: Path | None = None,
    package_cache: Path | None = None,
    tool_commands: Sequence[Sequence[str]] = (),
) -> dict[str, str]:
    """Build a bounded child environment with no inherited proxy or Python paths."""

    operator_home = state_root / "home"
    temporary = state_root / "tmp"
    operator_home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    path_entries = [
        str(Path(command[0]).parent)
        for command in tool_commands
        if command and Path(command[0]).is_absolute()
    ]
    if prefix is not None:
        path_entries.insert(0, str(prefix / "bin"))
    path_entries.extend(("/usr/bin", "/bin"))
    environment = {
        "HOME": str(operator_home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PATH": os.pathsep.join(path_entries),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
    }
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    if package_cache is not None:
        environment["CONDA_PKGS_DIRS"] = str(package_cache)
        environment["CONDA_OFFLINE"] = "true"
    return environment


def _load_probe_request(value: str) -> Mapping[str, Any]:
    """Parse an inline probe object or an absolute JSON request path."""

    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        try:
            parsed = json.loads(candidate.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PermanentEnvironmentOperatorError("probe request file is invalid") from exc
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PermanentEnvironmentOperatorError(
                "probe request must be inline JSON or an absolute existing path"
            ) from exc
    if not isinstance(parsed, dict) or set(parsed) != {
        "imports",
        "export_checks",
        "source_build",
    }:
        raise PermanentEnvironmentOperatorError(
            "probe request must contain exactly imports, export_checks, and source_build"
        )
    return parsed


def _mapping_list(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    """Require one list containing only string-keyed mappings."""

    if not isinstance(value, list):
        raise PermanentEnvironmentOperatorError(f"{label} must be a list")
    rows: list[Mapping[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise PermanentEnvironmentOperatorError(f"{label} must contain string-keyed objects")
        rows.append(raw)
    return tuple(rows)


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    """Require one list containing only non-empty strings."""

    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise PermanentEnvironmentOperatorError(f"{label} must contain only non-empty strings")
    return tuple(value)


def _nonempty_string(value: Any, label: str) -> str:
    """Require and return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise PermanentEnvironmentOperatorError(f"{label} must be a non-empty string")
    return value


def _prefix_python_command(command: Sequence[str], interpreter: Path) -> tuple[str, ...]:
    """Bind a declared source-build Python canary to the prefix interpreter."""

    if not command or command[0] not in {"python", "python3"}:
        raise PermanentEnvironmentOperatorError(
            "source-build canaries must invoke the prefix Python interpreter"
        )
    return (str(interpreter), *command[1:])


def _probe_result(name: str, passed: bool, observation: CommandObservation) -> dict[str, Any]:
    """Build one bounded, single-line canary receipt.

    A PASSED canary records exactly ``"ok"``: its streams are diagnostic
    noise by definition (OpenMP tmp warnings, library deprecation banners),
    and the receipt is a committed setup-artifact identity member that the
    runtime compares byte-for-byte after every launch's probe pass. Stable
    detail prevents harmless host noise from demanding a rotation. A FAILED
    canary keeps its bounded combined detail; that text is the diagnosis.
    """

    if passed:
        return {"name": name, "passed": True, "detail": "ok"}
    detail = _bounded_detail(_combined_detail(observation), limit=_PROBE_DETAIL_LIMIT)
    if not detail:
        detail = f"exit {observation.returncode}"
    return {"name": name, "passed": False, "detail": detail}


def _combined_detail(observation: CommandObservation) -> str:
    """Merge both child streams so one non-empty stream cannot mask the other."""

    return "\n".join(value for value in (observation.stderr, observation.stdout) if value)


def _bounded_detail(value: str, *, limit: int = _DETAIL_LIMIT) -> str:
    """Collapse and bound externally controlled diagnostic text."""

    collapsed = " ".join(value.split())
    return collapsed[-limit:]


def _is_transient_detail(value: str) -> bool:
    """Return whether child diagnostics carry a known retryable transport marker."""

    lowered = value.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _require_safe_component(value: str, label: str) -> None:
    """Require a path-component-safe intent or target identifier."""

    if _SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise PermanentEnvironmentOperatorError(f"{label} is not a safe path component")


def _require_absolute_file(path: Path, label: str) -> Path:
    """Require one absolute regular file and return its normalized path."""

    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError(f"{label} path must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise PermanentEnvironmentOperatorError(f"{label} is missing: {resolved}")
    return resolved


def _require_absolute_directory(path: Path, label: str) -> Path:
    """Require one absolute directory and return its normalized path."""

    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError(f"{label} path must be absolute")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise PermanentEnvironmentOperatorError(f"{label} is missing: {resolved}")
    return resolved


def _require_removable_prefix(path: Path) -> Path:
    """Refuse globbed, relative, symlinked, root, or otherwise broad prefixes."""

    raw = str(path)
    if any(character in raw for character in _GLOB_CHARACTERS):
        raise PermanentEnvironmentOperatorError("environment prefix cannot contain glob syntax")
    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError("environment prefix must be absolute")
    normalized = path.resolve(strict=False)
    if normalized == Path(normalized.anchor) or normalized.parent == Path(normalized.anchor):
        raise PermanentEnvironmentOperatorError("environment prefix is too broad to remove")
    return normalized


def _find_repository_root(anchor: Path) -> Path:
    """Find the containing TorchLens repository from an action path."""

    resolved = anchor.resolve(strict=False)
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "menagerie" / "crawler"
        ).is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "menagerie" / "crawler").is_dir():
        return cwd
    raise PermanentEnvironmentOperatorError(
        "cannot infer operator state root; pass --state-root explicitly"
    )


def resolve_environment_state_root(anchor: Path, *, explicit: Path | None = None) -> Path:
    """Resolve the one environment-operator state-root authority.

    Parameters
    ----------
    anchor:
        Action input used to locate the owning repository when no root is
        explicitly configured.
    explicit:
        Constructor-supplied root, which takes precedence over the environment.

    Returns
    -------
    pathlib.Path
        Absolute state root used by every operator action.

    Raises
    ------
    PermanentEnvironmentOperatorError
        If an explicit or environment-provided root is relative.
    """

    if explicit is not None:
        candidate = explicit
    else:
        configured = os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
        if configured:
            candidate = Path(configured)
        else:
            return _find_repository_root(anchor) / ".crawl-local" / "environment-operator"
    if not candidate.is_absolute():
        raise PermanentEnvironmentOperatorError("MENAGERIE_ENVIRONMENT_STATE_ROOT must be absolute")
    return candidate.resolve()


def _prefix_identifier(prefix: Path) -> str:
    """Return a stable filesystem-safe identifier for one exact prefix."""

    return hashlib.sha256(str(prefix).encode()).hexdigest()[:24]


def _registration_path(state_root: Path, prefix: Path) -> Path:
    """Return the exact durable registration path for one prefix."""

    root = state_root / "prefix-registrations"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_prefix_identifier(prefix)}.json"


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold one process-exclusive solve or prefix mutation lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _try_exclusive_file_lock(path: Path) -> Iterator[bool]:
    """Try one non-blocking exclusive lock; yield whether it was acquired."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _remove_registered_paths(prefix: Path, state_root: Path, registration_path: Path) -> None:
    """Validate and remove one prefix registration without touching shared CAS."""

    if not registration_path.is_file():
        raise PermanentEnvironmentOperatorError(
            "remove target is not an exact registered environment prefix"
        )
    registration = _load_mapping(registration_path, "prefix registration")
    if registration.get("protocol_version") != PROTOCOL_VERSION:
        raise PermanentEnvironmentOperatorError("prefix registration version is invalid")
    if registration.get("prefix") != str(prefix):
        raise PermanentEnvironmentOperatorError("prefix registration target mismatch")
    package_cache_value = registration.get("package_cache")
    if not isinstance(package_cache_value, str):
        raise PermanentEnvironmentOperatorError("prefix registration package cache is invalid")
    package_cache = Path(package_cache_value)
    expected_cache_root = (state_root / "package-caches").resolve()
    if (
        not package_cache.is_absolute()
        or package_cache.resolve(strict=False).parent != expected_cache_root
        or package_cache.is_symlink()
    ):
        raise PermanentEnvironmentOperatorError(
            "registered package cache escapes its exact operator root"
        )
    wheelhouse_value = registration.get("wheelhouse")
    wheelhouse: Path | None = None
    if wheelhouse_value is not None:
        if not isinstance(wheelhouse_value, str):
            raise PermanentEnvironmentOperatorError("prefix registration wheelhouse is invalid")
        wheelhouse = Path(wheelhouse_value)
        expected_wheelhouse_root = (state_root / "wheelhouses").resolve()
        if (
            not wheelhouse.is_absolute()
            or wheelhouse.resolve(strict=False).parent != expected_wheelhouse_root
            or wheelhouse.is_symlink()
        ):
            raise PermanentEnvironmentOperatorError(
                "registered wheelhouse escapes its exact operator root"
            )
    if prefix.is_symlink():
        raise PermanentEnvironmentOperatorError("registered prefix became a symlink")
    if prefix.exists():
        shutil.rmtree(prefix)
    if package_cache.exists():
        shutil.rmtree(package_cache)
    if wheelhouse is not None and wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    registration_path.unlink()
    fsync_directory(registration_path.parent)


# -- orphan GC (T1-19) -----------------------------------------------------------
#
# Every filesystem object the operator owns is named by exactly one durable
# authority: the union of ``package_cache``/``wheelhouse`` values across
# ``prefix-registrations/*.json``. Anything below the state root that no
# registration names is crash residue -- historically unnameable and therefore
# unsweepable. The sweep enumerates it (read-only); removal is a separate,
# per-candidate action that re-derives the verdict under the pair lock.

#: Residue younger than this is presumed live (an in-flight download, solve, or
#: staging rename) and is never named as an orphan. Deliberately generous: no
#: operator action legitimately holds scratch across a day.
ORPHAN_STALE_AGE_SECONDS = 86_400.0

_ORPHAN_DIR_NAME_PATTERN = re.compile(r"^(?P<identifier>[0-9a-f]{24})-[0-9a-f]{12}$")


@dataclass(frozen=True)
class EnvironmentOrphan:
    """One nameable piece of operator crash residue."""

    path: Path
    kind: str
    bytes_len: int
    detail: str


@dataclass(frozen=True)
class EnvironmentCasReachability:
    """Digest roots that must survive environment CAS collection."""

    artifact_digests: frozenset[str]
    wheel_digests: frozenset[str]


def build_environment_readiness_receipt(
    envs_root: Path,
    state_root: Path,
    *,
    target: str,
    intake_sha256: str,
    routed_intents: Sequence[str],
) -> dict[str, Any]:
    """Build a sealed cold-start receipt for one routed intake closure.

    Parameters
    ----------
    envs_root:
        Committed environment registry root.
    state_root:
        Exact operator state root whose CAS will serve the launch.
    target:
        Exact target platform.
    intake_sha256:
        Frozen intake digest the routes were derived from.
    routed_intents:
        Environment intent names reached by that frozen intake.

    Returns
    -------
    dict[str, Any]
        Canonical receipt sealed by ``receipt_sha256``.

    Raises
    ------
    PermanentEnvironmentOperatorError
        If the routed closure is unlocked or any required CAS object is absent
        or corrupt.
    """

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", intake_sha256):
        raise PermanentEnvironmentOperatorError("readiness intake digest is not canonical")
    selected = tuple(sorted(set(routed_intents)))
    if not selected:
        raise PermanentEnvironmentOperatorError("readiness routed closure is empty")
    registry = load_environment_registry(envs_root, target=target)
    unknown = sorted(set(selected) - set(registry.intents))
    if unknown:
        raise PermanentEnvironmentOperatorError(f"readiness names unknown intents: {unknown}")
    artifact_receipts: dict[str, ArtifactReceipt] = {}
    wheel_rows: dict[str, Mapping[str, str]] = {}
    families: list[dict[str, str]] = []
    for name in selected:
        intent = registry.intents[name]
        if intent.lock.status != "locked" or intent.generation is None:
            raise PermanentEnvironmentOperatorError(
                f"routed intent {name!r} is not a proven locked generation"
            )
        families.append({"intent": name, "generation": intent.generation})
        for receipt in parse_exact_lock(intent.lock.lock_bytes or b""):
            prior = artifact_receipts.setdefault(receipt.sha256, receipt)
            if prior.url != receipt.url:
                raise PermanentEnvironmentOperatorError(
                    f"artifact digest {receipt.sha256} is bound to multiple URLs"
                )
        for row in overlay_package_rows(intent):
            wheel_rows.setdefault(str(row["sha256"]), row)
    root = state_root.resolve()
    missing: list[str] = []
    for digest, receipt in sorted(artifact_receipts.items()):
        path = _cas_path(root, digest)
        if not path.is_file() or _stream_sha256(path) != receipt.sha256:
            missing.append(digest)
    for digest in sorted(wheel_rows):
        if _wheel_cas_file(root / "wheel-cas", digest) is None:
            missing.append(digest)
    if missing:
        raise PermanentEnvironmentOperatorError(
            "readiness CAS closure is incomplete: " + ", ".join(missing)
        )
    payload: dict[str, Any] = {
        "schema_version": READINESS_RECEIPT_VERSION,
        "intake_sha256": intake_sha256,
        "target": target,
        "envs_root": str(envs_root.resolve()),
        "state_root": str(root),
        "routed_intents": list(selected),
        "families": families,
        "artifact_digests": sorted(artifact_receipts),
        "wheel_digests": sorted(wheel_rows),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


def environment_cas_reachability(
    envs_root: Path,
    *,
    target: str,
    intents: Sequence[str] | None = None,
) -> EnvironmentCasReachability:
    """Derive live CAS digests from committed lock families.

    Parameters
    ----------
    envs_root:
        Committed environment registry root.
    target:
        Exact target whose lock families are reachable.
    intents:
        Optional routed family closure. ``None`` conservatively roots every
        registry intent; supplied names must all exist and be locked.

    Returns
    -------
    EnvironmentCasReachability
        Deduplicated conda-artifact and overlay-wheel digests.

    Raises
    ------
    PermanentEnvironmentOperatorError
        If a selected intent is unknown or lacks a proven committed family.
    """

    registry = load_environment_registry(envs_root, target=target)
    selected = tuple(sorted(registry.intents)) if intents is None else tuple(sorted(set(intents)))
    unknown = sorted(set(selected) - set(registry.intents))
    if unknown:
        raise PermanentEnvironmentOperatorError(
            f"CAS reachability names unknown environment intents: {unknown}"
        )
    artifacts: set[str] = set()
    wheels: set[str] = set()
    for name in selected:
        intent = registry.intents[name]
        if intent.lock.status != "locked":
            raise PermanentEnvironmentOperatorError(
                f"CAS reachability cannot root intent {name!r} with lock status "
                f"{intent.lock.status!r}"
            )
        artifacts.update(
            receipt.sha256.removeprefix("sha256:")
            for receipt in parse_exact_lock(intent.lock.lock_bytes or b"")
        )
        wheels.update(
            str(row["sha256"]).removeprefix("sha256:") for row in overlay_package_rows(intent)
        )
    return EnvironmentCasReachability(
        artifact_digests=frozenset(artifacts),
        wheel_digests=frozenset(wheels),
    )


def _tree_bytes(path: Path) -> int:
    """Total file bytes below one path (the path itself when it is a file)."""

    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.lstat().st_size
        except OSError:
            continue
    return total


def _registration_inventory(
    state_root: Path,
) -> tuple[dict[Path, dict[str, Any]], set[Path], bool]:
    """Read every registration; return (by path, referenced dirs, fully readable)."""

    registrations_root = state_root / "prefix-registrations"
    by_path: dict[Path, dict[str, Any]] = {}
    referenced: set[Path] = set()
    readable = True
    if not registrations_root.is_dir():
        return by_path, referenced, readable
    for entry in sorted(registrations_root.glob("*.json")):
        try:
            registration = _load_mapping(entry, "prefix registration")
        except PermanentEnvironmentOperatorError:
            readable = False
            continue
        by_path[entry] = dict(registration)
        for key in ("package_cache", "wheelhouse"):
            value = registration.get(key)
            if isinstance(value, str) and value:
                referenced.add(Path(value).resolve())
    return by_path, referenced, readable


def _pair_lock_for_orphan_dir(state_root: Path, name: str) -> Path | None:
    """Map one cache/wheelhouse dir name back to its (prefix, lock) pair lock."""

    match = _ORPHAN_DIR_NAME_PATTERN.match(name)
    if match is None:
        return None
    return state_root / "prefix-registrations" / f"{match.group('identifier')}.lock"


def _is_stale(path: Path, now: float, age_seconds: float) -> bool:
    try:
        return (now - path.lstat().st_mtime) >= age_seconds
    except OSError:
        return False


def _pair_lock_is_free(lock_path: Path) -> bool:
    """Probe one pair lock without creating it (the sweep stays read-only).

    A missing lock file means no operator action ever locked the pair on this
    host (create() materializes it first thing), so the pair is free.
    """

    if not lock_path.is_file():
        return True
    try:
        with lock_path.open("r+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return True
    except OSError:
        return False


def sweep_environment_orphans(
    state_root: Path,
    *,
    stale_temp_age_seconds: float = ORPHAN_STALE_AGE_SECONDS,
    now: float | None = None,
    cas_reachability: EnvironmentCasReachability | None = None,
) -> tuple[EnvironmentOrphan, ...]:
    """Enumerate every nameable orphan below one operator state root.

    Read-only: the sweep never deletes, never rewrites, and skips anything a
    concurrent operator action holds a pair lock on. When ``cas_reachability``
    is supplied, complete immutable CAS objects absent from that committed
    digest closure are nameable too. Age-gated classes (temporary files,
    staging dirs) are presumed live until
    ``stale_temp_age_seconds`` old, so an in-flight download is never named.
    """

    root = state_root.resolve()
    current = time.time() if now is None else now
    orphans: list[EnvironmentOrphan] = []
    registrations, referenced, registrations_readable = _registration_inventory(root)

    def _add(path: Path, kind: str, detail: str) -> None:
        orphans.append(
            EnvironmentOrphan(path=path, kind=kind, bytes_len=_tree_bytes(path), detail=detail)
        )

    # Unregistered generation-local dirs: nameable orphans unless ANY
    # registration is unreadable (then referenced-ness is unprovable and the
    # class is skipped entirely -- fail closed).
    if registrations_readable:
        for tree, kind in (
            (root / "package-caches", "unregistered-package-cache"),
            (root / "wheelhouses", "unregistered-wheelhouse"),
        ):
            if not tree.is_dir():
                continue
            for entry in sorted(tree.iterdir()):
                if entry.resolve() in referenced:
                    continue
                pair_lock = _pair_lock_for_orphan_dir(root, entry.name)
                if pair_lock is not None:
                    if not _pair_lock_is_free(pair_lock):
                        continue
                    _add(entry, kind, "no registration names this directory")
                elif _is_stale(entry, current, stale_temp_age_seconds):
                    _add(entry, kind, "unparseable name; stale and unreferenced")

    # Registrations whose prefix vanished (e.g. `crawl teardown --active-env`
    # removed the env root) -- removable through the registration itself.
    for registration_path, registration in registrations.items():
        prefix_value = registration.get("prefix")
        if not isinstance(prefix_value, str) or not prefix_value:
            continue
        prefix = Path(prefix_value)
        if prefix.exists() or prefix.is_symlink():
            continue
        if not _pair_lock_is_free(registration_path.with_suffix(".lock")):
            continue
        _add(
            registration_path,
            "dangling-registration",
            f"registered prefix {prefix} no longer exists",
        )

    registrations_root = root / "prefix-registrations"
    if registrations_root.is_dir():
        for entry in sorted(registrations_root.iterdir()):
            name = entry.name
            if (
                name.startswith(".")
                and name.endswith(".tmp")
                and _is_stale(entry, current, stale_temp_age_seconds)
            ):
                _add(entry, "registration-temp", "interrupted atomic registration write")
            elif (
                entry.suffix == ".lock"
                and not entry.with_suffix(".json").is_file()
                and _is_stale(entry, current, stale_temp_age_seconds)
            ):
                _add(entry, "orphan-pair-lock", "pair lock with no registration")

    cas_root = root / "artifact-cas" / "sha256"
    if cas_root.is_dir():
        for entry in sorted(cas_root.glob(".*.tmp")):
            if _is_stale(entry, current, stale_temp_age_seconds):
                _add(entry, "cas-partial", "interrupted artifact download")
        if cas_reachability is not None:
            for entry in sorted(cas_root.iterdir()):
                if (
                    entry.is_file()
                    and _SHA256_PATTERN.fullmatch(entry.name) is not None
                    and entry.name not in cas_reachability.artifact_digests
                ):
                    _add(
                        entry,
                        "cas-unreachable",
                        "no committed reachable lock family names this digest",
                    )

    wheel_cas_root = root / "wheel-cas" / "sha256"
    if wheel_cas_root.is_dir():
        for entry in sorted(wheel_cas_root.glob("*.staging-*")):
            if _is_stale(entry, current, stale_temp_age_seconds):
                _add(entry, "wheel-cas-staging", "interrupted wheel-CAS promotion")
        if cas_reachability is not None:
            for entry in sorted(wheel_cas_root.iterdir()):
                if not entry.is_dir() or entry.name in cas_reachability.wheel_digests:
                    continue
                if _SHA256_PATTERN.fullmatch(entry.name) is not None:
                    _add(
                        entry,
                        "wheel-cas-unreachable",
                        "no committed reachable overlay names this digest",
                    )

    temp_root = root / "tmp"
    if temp_root.is_dir():
        for entry in sorted(temp_root.iterdir()):
            if _is_stale(entry, current, stale_temp_age_seconds):
                _add(entry, "stale-temp", "operator or child-tool temporary residue")

    solves_root = root / "solves"
    if solves_root.is_dir():
        for entry in sorted(solves_root.glob("*/*/.solve-*")):
            if not _is_stale(entry, current, stale_temp_age_seconds):
                continue
            if _pair_lock_is_free(entry.parent / ".solve.lock"):
                _add(entry, "solve-temp", "interrupted solve temporary")

    return tuple(orphans)


def remove_environment_orphan(
    state_root: Path,
    orphan: EnvironmentOrphan,
    *,
    cas_reachability: EnvironmentCasReachability | None = None,
) -> bool:
    """Remove one swept orphan, re-deriving the verdict under the pair lock.

    Returns ``False`` when the candidate is no longer removable: a concurrent
    operator action holds its pair lock, a registration adopted it between the
    sweep and this call, or it already vanished. Never touches shared CAS
    objects or anything a registration names.

    A pair-lock file is unlinked only after its exclusive lock is held and no
    sibling registration exists; a create racing that exact window would need a
    concurrent sibling create of the same pair, which the sequential
    environment lifecycle already forbids.
    """

    root = state_root.resolve()
    target = Path(orphan.path)
    resolved = target.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise PermanentEnvironmentOperatorError(
            f"orphan {target} escapes the operator state root {root}"
        )

    def _delete(path: Path) -> bool:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
            return True
        if path.is_dir():
            shutil.rmtree(path)
            return True
        return False

    if orphan.kind in {"cas-unreachable", "wheel-cas-unreachable"}:
        if cas_reachability is None:
            return False
        if orphan.kind == "cas-unreachable":
            expected_parent = root / "artifact-cas" / "sha256"
            reachable = cas_reachability.artifact_digests
        else:
            expected_parent = root / "wheel-cas" / "sha256"
            reachable = cas_reachability.wheel_digests
        if target.parent.resolve() != expected_parent.resolve() or target.name in reachable:
            return False
        if _SHA256_PATTERN.fullmatch(target.name) is None:
            return False
        return _delete(target)

    if orphan.kind in {"unregistered-package-cache", "unregistered-wheelhouse"}:
        pair_lock = _pair_lock_for_orphan_dir(root, target.name)
        if pair_lock is None:
            if not _is_stale(target, time.time(), ORPHAN_STALE_AGE_SECONDS):
                return False
            return _delete(target)
        with _try_exclusive_file_lock(pair_lock) as acquired:
            if not acquired:
                return False
            if resolved in _registration_inventory(root)[1]:
                return False
            return _delete(target)

    if orphan.kind == "dangling-registration":
        with _try_exclusive_file_lock(target.with_suffix(".lock")) as acquired:
            if not acquired or not target.is_file():
                return False
            registration = _load_mapping(target, "prefix registration")
            prefix_value = registration.get("prefix")
            if not isinstance(prefix_value, str) or not prefix_value:
                return False
            prefix = Path(prefix_value)
            if prefix.exists() or prefix.is_symlink():
                return False
            _remove_registered_paths(prefix, root, target)
            return True

    if orphan.kind == "orphan-pair-lock":
        with _try_exclusive_file_lock(target) as acquired:
            if not acquired:
                return False
            if target.with_suffix(".json").is_file():
                return False
            target.unlink(missing_ok=True)
            return True

    if orphan.kind == "solve-temp":
        with _try_exclusive_file_lock(target.parent / ".solve.lock") as acquired:
            if not acquired:
                return False
            return _delete(target)

    if orphan.kind in {"registration-temp", "cas-partial", "wheel-cas-staging", "stale-temp"}:
        if not _is_stale(target, time.time(), ORPHAN_STALE_AGE_SECONDS):
            return False
        return _delete(target)

    raise PermanentEnvironmentOperatorError(f"unknown orphan kind {orphan.kind!r}")


def _emit_telemetry(
    state_root: Path | None, action: str, status: str, error_kind: str | None = None
) -> None:
    """Append one bounded JSONL operator event without external diagnostic text."""

    if state_root is None:
        return
    path = state_root / "telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "protocol_version": PROTOCOL_VERSION,
        "action": action,
        "status": status,
        "error_kind": error_kind,
    }
    line = canonical_json_bytes(event) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        durable_fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    """Build the direct argv action interface consumed by the crawler driver."""

    parser = argparse.ArgumentParser(prog="operator_environment.py")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--version", action="version", version=PROTOCOL_VERSION)
    subparsers = parser.add_subparsers(dest="action", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("environment_file", type=Path)
    solve.add_argument("target")
    solve.add_argument("--force-resolve", action="store_true")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("lock_file", type=Path)
    create = subparsers.add_parser("create")
    create.add_argument("lock_file", type=Path)
    create.add_argument("prefix", type=Path)
    probe = subparsers.add_parser("probe")
    probe.add_argument("prefix", type=Path)
    probe.add_argument("probes")
    overlay = subparsers.add_parser("overlay")
    overlay.add_argument("prefix", type=Path)
    overlay.add_argument("manifest_file", type=Path)
    remove = subparsers.add_parser("remove")
    remove.add_argument("prefix", type=Path)
    owned = subparsers.add_parser("owned-paths")
    owned.add_argument("prefix", type=Path)
    state_root = subparsers.add_parser("state-root")
    state_root.add_argument("anchor", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one action and map typed outcomes to the common operator exits."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        operator = EnvironmentOperator(state_root=arguments.state_root)
        action = str(arguments.action)
        if action == "solve":
            result = operator.solve(
                arguments.environment_file,
                arguments.target,
                force_resolve=bool(arguments.force_resolve),
            )
        elif action == "materialize":
            result = operator.materialize(arguments.lock_file)
        elif action == "create":
            result = operator.create(arguments.lock_file, arguments.prefix)
        elif action == "probe":
            result = operator.probe(arguments.prefix, arguments.probes)
        elif action == "overlay":
            result = operator.overlay(arguments.prefix, arguments.manifest_file)
        elif action == "remove":
            result = operator.remove(arguments.prefix)
        elif action == "owned-paths":
            result = operator.owned_paths(arguments.prefix)
        elif action == "state-root":
            result = operator.state_root(arguments.anchor)
        else:
            raise AssertionError(f"unhandled environment action: {action}")
    except TransientEnvironmentOperatorError as exc:
        _emit_telemetry(
            arguments.state_root, str(arguments.action), "retryable", type(exc).__name__
        )
        print(
            json.dumps(
                {"status": "retryable", "error": _bounded_detail(str(exc))},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_RETRYABLE
    except (PermanentEnvironmentOperatorError, OSError) as exc:
        _emit_telemetry(
            arguments.state_root, str(arguments.action), "permanent", type(exc).__name__
        )
        print(
            json.dumps(
                {"status": "permanent", "error": _bounded_detail(str(exc))},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_PERMANENT
    _emit_telemetry(arguments.state_root, str(arguments.action), "success")
    print(json.dumps(result, sort_keys=True))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
