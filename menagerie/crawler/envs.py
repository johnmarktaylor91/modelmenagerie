"""Typed loading for crawler environment intents and target-solved artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.package_namespace import (
    dependency_spec_name,
    inventory_row_provides_distribution,
)

DEFAULT_ENVS_ROOT = Path(__file__).with_name("envs")

#: Single production floor shared by environment runtime and launch readiness.
#: Fixtures may pass zero explicitly; production consumers import this value
#: instead of spelling a second load-bearing number.
DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES = 30 * 1024**3


class EnvironmentSpecError(ValueError):
    """Raised when an environment registry or intent artifact is malformed."""


@dataclass(frozen=True)
class ExportCheck:
    """One expected top-level attribute exposed by an imported module."""

    module: str
    attribute: str


@dataclass(frozen=True)
class SourceBuildProbe:
    """A bounded compiled/source-build probe declaration."""

    name: str
    packages: tuple[str, ...]
    command: tuple[str, ...]
    max_attempts: int


@dataclass(frozen=True)
class IntentProbes:
    """Import, export, and optional source-build checks for one intent."""

    imports: tuple[str, ...]
    export_checks: tuple[ExportCheck, ...]
    source_build: tuple[SourceBuildProbe, ...]


@dataclass(frozen=True)
class LockArtifacts:
    """Committed exact target lock and resolved export for one intent.

    A ``locked`` family is the authoritative BUILD INPUT of the environment
    lane: launches create prefixes FROM these committed bytes and never solve.
    Solving is a rotation-only operation (``tools/rotate_setup_artifacts.py``),
    which proves a fresh solve end-to-end before publishing a new family.
    """

    target: str
    lock_path: Path
    export_path: Path
    export_hash_path: Path
    lock_bytes: Optional[bytes]
    export_bytes: Optional[bytes]
    declared_export_hash: Optional[str]
    declared_dependencies: tuple[str, ...] = ()
    #: Whether the family may be a mixed-identity torn state that must not be
    #: read as a proven family: a live rotation publication journal without a
    #: family manifest (a pre-manifest publisher died between member writes),
    #: or a family manifest that fails validation (``family_fault``).
    publication_pending: bool = False
    #: Which on-disk view supplied the family bytes: ``"flat"`` (the historical
    #: committed member files) or ``"manifest"`` (the immutable generation
    #: directory named by the atomically switched ``<target>.family.json``).
    family_authority: str = "flat"
    #: Typed reason the family manifest could not prove its generation; set
    #: exactly when a manifest exists but fails closed (``publishing``).
    family_fault: Optional[str] = None
    #: Whether the manifest-named generation is committed but the flat member
    #: materialization has not completed (a crash between flat member writes,
    #: or a leftover publication journal). The family itself is complete and
    #: proven -- the launch preflight refuses until the roll-forward runs so
    #: the git-visible flat members never drift silently.
    materialization_pending: bool = False

    @property
    def status(self) -> str:
        """Return the target artifact readiness state.

        Returns
        -------
        str
            ``publishing`` (a rotation publication journal is live),
            ``unlocked``, ``incomplete``, ``hash-mismatch``, ``superseded``, or
            ``locked``.
        """

        if self.publication_pending:
            return "publishing"
        present = (self.lock_bytes is not None, self.export_bytes is not None)
        if not any(present) and self.declared_export_hash is None:
            return "unlocked"
        if not all(present) or self.declared_export_hash is None:
            return "incomplete"
        if hash_bytes(self.export_bytes or b"") != self.declared_export_hash:
            return "hash-mismatch"
        if self.missing_declared_dependency() is not None:
            return "superseded"
        return "locked"

    def missing_declared_dependency(self) -> Optional[str]:
        """Name one declared dependency the resolved export fails to install.

        Committed lock artifacts realize the declaration as of their LAST
        rotation. When the declaration gains a dependency after that rotation
        (the 2026-08-04 pilot rung: ``segmentation-models-pytorch`` was
        declared in ``core`` hours after the family's solve wrote a lock
        without it), the family silently contradicts the spec it claims to
        realize. Reading it as the routed environment's truth then refuses
        every model needing the new dependency with "no package row carries
        that name" -- a false permanent wall, because a rotation would install
        it. A ``superseded`` family therefore demands a rotation; the launch
        lane refuses to build from it and never solves in place.

        A solve satisfies every requested package by construction, so a
        resolved export missing a declared dependency name is PROOF the lock
        predates the declaration. This never inspects versions or bounds: a
        version drift is a solver outcome, not evidence of a superseded spec.

        Returns
        -------
        str | None
            The first declared dependency name absent from the resolved
            export, or ``None`` when every declared name is present, the
            export is unreadable (downstream parsing owns that refusal), or no
            declaration was provided.
        """

        if not self.declared_dependencies or self.export_bytes is None:
            return None
        try:
            value = json.loads(self.export_bytes.decode("utf-8"))
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
        for spec in self.declared_dependencies:
            name = dependency_spec_name(str(spec))
            if not name:
                continue
            if not any(
                inventory_row_provides_distribution(row_name, name) for row_name in row_names
            ):
                return name
        return None


_OVERLAY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_OVERLAY_HASH_PATTERN = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_OVERLAY_ROW_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9._+!]*)"
    r"\s+--hash=sha256:(?P<sha256>[0-9a-f]{64})\s*"
)

PIP_OVERLAY_DECLARATION_NAME = "pip-overlay.in"

#: Typed conda-export binding line inside a pinned overlay manifest (T3-T15).
#: The pip closure is compiled against exactly one conda resolved export; the
#: manifest records that export's digest so a manifest can never silently
#: install into a different conda solve than the one it was compiled for.
#: Grammar-wise the line is a comment, so pip consumes the manifest verbatim.
PIP_OVERLAY_EXPORT_BINDING_MARKER = "# conda-resolved-export-sha256:"
_OVERLAY_BINDING_PATTERN = re.compile(
    re.escape(PIP_OVERLAY_EXPORT_BINDING_MARKER) + r"\s*(sha256:[0-9a-f]{64})\s*\Z"
)


LEGACY_UNBOUND_OVERLAY_MANIFESTS: frozenset[str] = frozenset(
    {
        # graph: menagerie/crawler/envs/graph/locks/osx-arm64.pip-overlay.txt
        "sha256:570ae5eea1eb6fe3c605ae3a783514b3e1aeb7bfca539a027435e2e78d8edc83",
        # speech: menagerie/crawler/envs/speech/locks/osx-arm64.pip-overlay.txt
        "sha256:e0d8fcd895a00a784b2cfa8db15926993de397f48a286e708a3d97591eddbc0d",
    }
)
"""EXACT byte digests of the two committed pre-binding overlay manifests (EN-8).

Absent conda-export bindings now REFUSE at every guard site: an unbound
manifest silently pairs with ANY conda solve, the exact residue hole T3-T15's
binding closes. The two committed legacy families (graph, speech -- opus D4
spot-check) predate the binding and stay serviceable ONLY as these exact
bytes; a fresh unbound manifest can never match an enumerated digest.

SELF-RETIRING: ``test_overlay_binding_required`` asserts each digest still
names a committed manifest, so the EN-8 rotation that republishes BOUND
graph/speech manifests forces this enumeration's deletion in the same change.
Never add a member without that test's evidence trail -- a widened set here is
a new carve-out and needs its own reviewed change.
"""


def pip_overlay_export_binding(content: bytes) -> Optional[str]:
    """Return the manifest's conda-export binding digest, or ``None`` if unbound.

    Parameters
    ----------
    content:
        Committed manifest bytes.

    Returns
    -------
    str | None
        The bound ``sha256:...`` conda resolved-export digest, or ``None`` for
        a pre-binding legacy manifest that carries no marker line.

    Raises
    ------
    EnvironmentSpecError
        If a marker line is malformed or appears more than once: the binding
        is typed grammar, never a comment that can rot unnoticed.
    """

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentSpecError("pip-overlay manifest is not UTF-8") from exc
    bindings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(PIP_OVERLAY_EXPORT_BINDING_MARKER):
            continue
        match = _OVERLAY_BINDING_PATTERN.fullmatch(line)
        if match is None:
            raise EnvironmentSpecError(
                f"pip-overlay manifest conda-export binding line is malformed: {line[:200]!r}"
            )
        bindings.append(match.group(1))
    if len(bindings) > 1:
        raise EnvironmentSpecError("pip-overlay manifest binds more than one conda resolved export")
    return bindings[0] if bindings else None


def render_pip_overlay_manifest(
    rows: Sequence[Mapping[str, str]], *, conda_export_sha256: str
) -> bytes:
    """Render one canonical pinned overlay manifest bound to its conda export.

    Parameters
    ----------
    rows:
        Exact ``name``/``version``/``sha256`` package rows (``sha256:`` digests).
    conda_export_sha256:
        Canonical digest of the conda resolved export the closure was compiled
        against; recorded as the typed binding line (T3-T15).

    Returns
    -------
    bytes
        Canonical manifest bytes in the closed grammar, sorted by canonical
        name, round-trip validated.

    Raises
    ------
    EnvironmentSpecError
        If the rows or the binding digest fall outside the closed grammar.
    """

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", conda_export_sha256):
        raise EnvironmentSpecError("overlay conda-export binding must be a canonical sha256 digest")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: re.sub(r"[-_.]+", "-", str(row.get("name", ""))).casefold(),
    )
    lines = [
        "# Pinned pip overlay compiled by tools/rotate_setup_artifacts.py.",
        "# Consumed verbatim by pip with --no-index --no-deps --require-hashes.",
        f"{PIP_OVERLAY_EXPORT_BINDING_MARKER} {conda_export_sha256}",
    ]
    for row in ordered:
        digest = str(row.get("sha256", "")).removeprefix("sha256:")
        lines.append(f"{row.get('name')}=={row.get('version')} \\")
        lines.append(f"    --hash=sha256:{digest}")
    content = ("\n".join(lines) + "\n").encode("utf-8")
    parse_pip_overlay_manifest(content)
    if pip_overlay_export_binding(content) != conda_export_sha256:
        raise EnvironmentSpecError("rendered overlay manifest failed its own binding round-trip")
    return content


def parse_pip_overlay_manifest(content: bytes) -> tuple[Mapping[str, str], ...]:
    """Parse one committed pip-overlay manifest into exact package rows.

    The manifest is BOTH machine-validated identity input and the literal file
    pip consumes (``--require-hashes -r``), so the accepted grammar is a strict,
    closed subset of the requirements format: comments, blank lines, and logical
    lines of exactly ``name==version --hash=sha256:<64hex>`` (backslash
    continuations joined). Everything else -- option lines, URL/VCS/editable
    requirements, extras, environment markers, non-``==`` specifiers, missing or
    multiple hashes -- is refused. An ambient, unpinned, or unhashed install can
    therefore never ride in through this file.

    Parameters
    ----------
    content:
        Committed manifest bytes.

    Returns
    -------
    tuple[Mapping[str, str], ...]
        One ``name``/``version``/``sha256`` row per requirement, in file order.

    Raises
    ------
    EnvironmentSpecError
        If any logical line falls outside the closed grammar or a package name
        repeats.
    """

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentSpecError("pip-overlay manifest is not UTF-8") from exc
    # The conda-export binding line is comment-shaped but typed: a malformed or
    # duplicated marker refuses here instead of rotting as an ignored comment.
    pip_overlay_export_binding(content)
    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical_lines.append((pending + line).strip())
        pending = ""
    if pending:
        raise EnvironmentSpecError("pip-overlay manifest ends inside a continuation")
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for line in logical_lines:
        match = _OVERLAY_ROW_PATTERN.fullmatch(line)
        if match is None:
            raise EnvironmentSpecError(
                "pip-overlay manifest line is outside the closed "
                f"'name==version --hash=sha256:...' grammar: {line[:200]!r}"
            )
        name = match.group("name")
        canonical = re.sub(r"[-_.]+", "-", name).casefold()
        if canonical in seen:
            raise EnvironmentSpecError(f"pip-overlay manifest pins package {name!r} more than once")
        seen.add(canonical)
        rows.append(
            {
                "name": name,
                "version": match.group("version"),
                "sha256": f"sha256:{match.group('sha256')}",
            }
        )
    if not rows:
        raise EnvironmentSpecError("pip-overlay manifest pins no packages")
    return tuple(rows)


@dataclass(frozen=True)
class PipOverlay:
    """One intent's declared pip overlay and its committed pinned manifest.

    The overlay is IDENTITY-BEARING: the pinned manifest is a fifth committed
    setup-artifact member for the intent, participating in the quarantine/park
    identity exactly like the conda lock family, so rotation and release
    semantics apply to it unchanged.
    """

    declared_path: Path
    declared: tuple[str, ...]
    manifest_path: Path
    manifest_bytes: Optional[bytes]

    @property
    def status(self) -> str:
        """Return the overlay readiness state.

        Returns
        -------
        str
            ``unpinned`` (no committed manifest), ``invalid`` (manifest outside
            the closed grammar), ``superseded`` (a declared requirement gained
            after the manifest was compiled), or ``pinned``.
        """

        if self.manifest_bytes is None:
            return "unpinned"
        try:
            self.rows()
        except EnvironmentSpecError:
            return "invalid"
        if self.missing_declared_requirement() is not None:
            return "superseded"
        return "pinned"

    def rows(self) -> tuple[Mapping[str, str], ...]:
        """Return the manifest's exact package rows.

        Returns
        -------
        tuple[Mapping[str, str], ...]
            ``name``/``version``/``sha256`` rows.

        Raises
        ------
        EnvironmentSpecError
            If no manifest is committed or it is outside the closed grammar.
        """

        if self.manifest_bytes is None:
            raise EnvironmentSpecError(
                f"pip overlay declared at {self.declared_path} has no committed manifest"
            )
        return parse_pip_overlay_manifest(self.manifest_bytes)

    @property
    def export_binding(self) -> Optional[str]:
        """Return the conda resolved-export digest this manifest binds (T3-T15).

        Returns
        -------
        str | None
            The bound ``sha256:...`` digest, or ``None`` when no manifest is
            committed, the manifest is outside the closed grammar (``status``
            owns that refusal), or it is a pre-binding legacy manifest.
        """

        if self.manifest_bytes is None:
            return None
        try:
            return pip_overlay_export_binding(self.manifest_bytes)
        except EnvironmentSpecError:
            return None

    def missing_declared_requirement(self) -> Optional[str]:
        """Name one declared requirement the committed manifest fails to pin.

        This is the overlay's supersession rule, mirroring
        :meth:`LockArtifacts.missing_declared_dependency`: a manifest compiled
        before a declaration gained a requirement silently contradicts the spec
        it claims to realize, so it reads as last rotation's residue rather
        than this campaign's truth.

        Returns
        -------
        str | None
            The first declared requirement name absent from the manifest, or
            ``None`` when every declared name is pinned or nothing is parseable
            (downstream validation owns that refusal).
        """

        if self.manifest_bytes is None:
            return None
        try:
            rows = self.rows()
        except EnvironmentSpecError:
            return None
        pinned = {re.sub(r"[-_.]+", "-", str(row["name"])).casefold() for row in rows}
        for spec in self.declared:
            match = _OVERLAY_NAME_PATTERN.match(spec.strip())
            if match is None:
                continue
            canonical = re.sub(r"[-_.]+", "-", match.group(0)).casefold()
            if canonical not in pinned:
                return match.group(0)
        return None


@dataclass(frozen=True)
class EnvironmentIntent:
    """Complete in-memory view of one environment intent and target state."""

    name: str
    phase: EnvironmentPhase
    framework: str
    description: str
    split_guidance: str
    channels: tuple[str, ...]
    dependencies: tuple[str, ...]
    probes: IntentProbes
    lock: LockArtifacts
    generation: Optional[str]
    pip_overlay: Optional[PipOverlay] = None

    @property
    def lock_status(self) -> str:
        """Return the exact target lock status.

        Returns
        -------
        str
            Delegated lock readiness state.
        """

        return self.lock.status


def overlay_package_rows(intent: EnvironmentIntent) -> tuple[Mapping[str, str], ...]:
    """Return one intent's pinned pip-overlay package rows, or nothing.

    The rows carry exactly the ``name``/``version``/``sha256`` fields the
    conda resolved-export rows carry, so inventory consumers (coverage, author
    disclosure, R1 artifact-digest resolution) treat overlay-provided
    distributions identically to conda-provided ones. An unpinned, invalid, or
    superseded overlay contributes NOTHING: unknown is honest, and a stale
    manifest must never read as this campaign's inventory.

    Parameters
    ----------
    intent:
        Loaded environment intent.

    Returns
    -------
    tuple[Mapping[str, str], ...]
        Exact overlay package rows, or an empty tuple.
    """

    overlay = getattr(intent, "pip_overlay", None)
    if overlay is None or overlay.status != "pinned":
        return ()
    return overlay.rows()


@dataclass(frozen=True)
class EnvironmentRegistry:
    """Ordered small-set environment registry without a hard intent cap."""

    intents: Mapping[str, EnvironmentIntent]
    phase_order: tuple[EnvironmentPhase, ...]
    small_set_target: bool
    hard_cap: Optional[int]
    global_split_guidance: str


def load_environment_registry(
    root: Path = DEFAULT_ENVS_ROOT, *, target: str = "osx-arm64"
) -> EnvironmentRegistry:
    """Load registry, direct dependencies, probes, and optional target locks.

    Parameters
    ----------
    root:
        Directory containing ``registry.yml`` and intent subdirectories.
    target:
        Exact target lock basename.

    Returns
    -------
    EnvironmentRegistry
        Validated typed environment view.
    """

    registry = _load_json_yaml(root / "registry.yml")
    probes = _load_json_yaml(root / "probes.yml")
    raw_intents = _mapping(registry.get("intents"), "registry intents")
    raw_probes = _mapping(probes.get("intents"), "probe intents")
    intents: dict[str, EnvironmentIntent] = {}
    for name, raw_value in raw_intents.items():
        raw = _mapping(raw_value, f"intent {name}")
        env = _load_json_yaml(root / name / "environment.yml")
        probe = _parse_probes(name, _mapping(raw_probes.get(name), f"probes for {name}"))
        dependencies = _strings(env.get("dependencies"), f"dependencies for {name}")
        lock = _load_lock_artifacts(
            root / name / "locks", target, declared_dependencies=dependencies
        )
        pip_overlay = _load_pip_overlay(root / name, target, member_dir=lock.lock_path.parent)
        generation = compute_environment_generation(name, lock, probe, pip_overlay=pip_overlay)
        intents[name] = EnvironmentIntent(
            name=name,
            phase=EnvironmentPhase(_string(raw.get("phase"), f"phase for {name}")),
            framework=_string(raw.get("framework"), f"framework for {name}"),
            description=_string(raw.get("description"), f"description for {name}"),
            split_guidance=_string(raw.get("split_guidance"), f"split guidance for {name}"),
            channels=_strings(env.get("channels"), f"channels for {name}"),
            dependencies=dependencies,
            probes=probe,
            lock=lock,
            generation=generation,
            pip_overlay=pip_overlay,
        )
    if set(raw_probes) != set(raw_intents):
        raise EnvironmentSpecError("registry and probes must name exactly the same intents")
    phase_order = tuple(
        EnvironmentPhase(value) for value in _strings(registry.get("phase_order"), "phase order")
    )
    hard_cap = registry.get("hard_cap")
    if hard_cap is not None and not isinstance(hard_cap, int):
        raise EnvironmentSpecError("hard_cap must be null or an integer")
    return EnvironmentRegistry(
        intents=intents,
        phase_order=phase_order,
        small_set_target=bool(registry.get("small_set_target")),
        hard_cap=hard_cap,
        global_split_guidance=_string(registry.get("split_guidance"), "split guidance"),
    )


def compute_environment_generation(
    intent: str,
    lock: LockArtifacts,
    probes: IntentProbes,
    *,
    pip_overlay: Optional[PipOverlay] = None,
) -> Optional[str]:
    """Hash exact lock, resolved export, declared hash, and probe contract.

    Parameters
    ----------
    intent, lock, probes:
        Environment intent name and its target-solved generation inputs.
    pip_overlay:
        Optional declared pip overlay. A declared overlay joins the generation
        through its pinned manifest digest; an intent WITHOUT an overlay hashes
        the exact same payload it always has, so every existing generation is
        unchanged.

    Returns
    -------
    str or None
        Stable generation hash, or ``None`` until artifacts are complete and valid.
    """

    if lock.status != "locked":
        return None
    payload: dict[str, Any] = {
        "intent": intent,
        "target": lock.target,
        "lock_sha256": hash_bytes(lock.lock_bytes or b""),
        "resolved_export_sha256": lock.declared_export_hash,
        "probes": {
            "imports": probes.imports,
            "export_checks": [vars(check) for check in probes.export_checks],
            "source_build": [vars(probe) for probe in probes.source_build],
        },
    }
    if pip_overlay is not None:
        if pip_overlay.status != "pinned":
            return None
        # T3-T15: a manifest bound to a conda export other than THIS family's
        # is another rotation's residue -- the pip closure was compiled against
        # a different conda solve, so the pairing forms no valid generation.
        # EN-8: an ABSENT binding refuses too (an unbound manifest pairs with
        # any solve), except the two exact committed legacy manifests in
        # LEGACY_UNBOUND_OVERLAY_MANIFESTS, which retire with their rotation.
        binding = pip_overlay.export_binding
        if binding is None:
            manifest_digest = hash_bytes(pip_overlay.manifest_bytes or b"")
            if manifest_digest not in LEGACY_UNBOUND_OVERLAY_MANIFESTS:
                return None
        elif binding != lock.declared_export_hash:
            return None
        payload["pip_overlay_manifest_sha256"] = hash_bytes(pip_overlay.manifest_bytes or b"")
    return stable_hash(payload)


def probe_declaration_digest(intent: str, probes: IntentProbes) -> str:
    """Hash one intent's OWN probe declaration for per-intent currency.

    The whole-file ``probes.yml`` hash is historical for every sibling the
    moment any one intent's probes change, so nothing may compare it at
    runtime (T3-T7). This digest covers exactly the declaring intent's
    contract: a post-rotation edit to THIS intent's probes reads stale, a
    sibling edit never does.

    Parameters
    ----------
    intent, probes:
        Intent name and its declared probe contract.

    Returns
    -------
    str
        Typed digest of the per-intent probe declaration.
    """

    return stable_hash(
        {
            "intent": intent,
            "imports": probes.imports,
            "export_checks": [vars(check) for check in probes.export_checks],
            "source_build": [vars(probe) for probe in probes.source_build],
        }
    )


def _load_pip_overlay(
    intent_root: Path, target: str, *, member_dir: Optional[Path] = None
) -> Optional[PipOverlay]:
    """Load one intent's optional pip-overlay declaration and pinned manifest.

    Parameters
    ----------
    intent_root, target:
        Intent directory and exact conda-lock platform.
    member_dir:
        Directory the family members are read from -- the immutable generation
        directory under a manifest authority, or the flat ``locks/`` directory
        otherwise. Defaults to the flat directory.
    """

    declared_path = intent_root / PIP_OVERLAY_DECLARATION_NAME
    if not declared_path.is_file():
        return None
    declared = tuple(
        line.strip()
        for line in declared_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    if not declared:
        raise EnvironmentSpecError(
            f"pip overlay declaration {declared_path} declares no requirements"
        )
    manifest_path = (member_dir or intent_root / "locks") / f"{target}.pip-overlay.txt"
    return PipOverlay(
        declared_path=declared_path,
        declared=declared,
        manifest_path=manifest_path,
        manifest_bytes=manifest_path.read_bytes() if manifest_path.is_file() else None,
    )


#: Schema of the atomically switched lock-family manifest: the single file
#: whose one ``os.replace`` commits a rotation publication. It names the
#: immutable generation directory holding the complete family and binds every
#: member by exact digest, so a reader can prove the whole family or refuse.
FAMILY_MANIFEST_SCHEMA_VERSION = "menagerie.crawler.lock-family-manifest.v1"

#: Directory below one intent's ``locks/`` holding immutable per-generation
#: family copies. Runtime publication state: git-ignored, never a committed
#: solve output (the anti-fabrication gate scans tracked flat members only).
GENERATIONS_DIR_NAME = "generations"


def family_manifest_path(lock_dir: Path, target: str) -> Path:
    """Return the lock-family manifest path for one lock family.

    Parameters
    ----------
    lock_dir, target:
        Lock directory of one intent and the exact conda-lock platform.

    Returns
    -------
    pathlib.Path
        Manifest path inside the lock directory.
    """

    return lock_dir / f"{target}.family.json"


def resolve_family_manifest(lock_dir: Path, target: str) -> "dict[str, Any]":
    """Validate one family manifest and its immutable generation directory.

    Every check fails CLOSED: a manifest that cannot prove its complete,
    digest-exact generation raises instead of degrading to the flat members,
    because the flat view may be a mixed-identity torn state.

    Parameters
    ----------
    lock_dir, target:
        Lock directory of one intent and the exact conda-lock platform.

    Returns
    -------
    dict[str, Any]
        ``{"generation": str, "generation_dir": Path, "members": {name: digest}}``
        with every member digest verified against the generation directory.

    Raises
    ------
    EnvironmentSpecError
        If the manifest is unreadable, malformed, names a foreign target, or
        any generation member is missing or fails its digest.
    """

    path = family_manifest_path(lock_dir, target)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentSpecError(f"family manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvironmentSpecError(f"family manifest must be an object: {path}")
    if value.get("schema_version") != FAMILY_MANIFEST_SCHEMA_VERSION:
        raise EnvironmentSpecError(f"family manifest carries an unknown schema_version: {path}")
    if value.get("target") != target:
        raise EnvironmentSpecError(f"family manifest names a foreign target: {path}")
    generation = value.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise EnvironmentSpecError(f"family manifest generation must be a 64-hex digest: {path}")
    members = value.get("members")
    if (
        not isinstance(members, dict)
        or not members
        or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in members.items()
        )
    ):
        raise EnvironmentSpecError(f"family manifest members are malformed: {path}")
    required = {
        f"{target}.lock",
        f"{target}.resolved.json",
        f"{target}.resolved.sha256",
        f"{target}.probes.json",
    }
    missing = sorted(required - set(members))
    if missing:
        raise EnvironmentSpecError(f"family manifest lacks required members {missing}: {path}")
    generation_dir = lock_dir / GENERATIONS_DIR_NAME / generation
    for name, digest in sorted(members.items()):
        member_path = generation_dir / name
        try:
            content = member_path.read_bytes()
        except OSError as exc:
            raise EnvironmentSpecError(
                f"family generation member is missing: {member_path}: {exc}"
            ) from exc
        if hash_bytes(content) != digest:
            raise EnvironmentSpecError(
                f"family generation member fails its manifest digest: {member_path}"
            )
    return {
        "generation": generation,
        "generation_dir": generation_dir,
        "members": dict(members),
    }


def publication_journal_path(lock_dir: Path, target: str) -> Path:
    """Return the rotation publication journal path for one lock family.

    The journal exists exactly while a rotation publication transaction is in
    flight; its presence marks the family as possibly torn (``publishing``).
    The dotfile name keeps it invisible to every committed-family glob and to
    the setup-artifact identity's member paths.

    Parameters
    ----------
    lock_dir, target:
        Lock directory of one intent and the exact conda-lock platform.

    Returns
    -------
    pathlib.Path
        Journal path inside the lock directory.
    """

    return lock_dir / f".{target}.publication-journal.json"


def _load_lock_artifacts(
    lock_dir: Path, target: str, *, declared_dependencies: tuple[str, ...] = ()
) -> LockArtifacts:
    """Load optional setup-time target artifacts from one lock directory.

    A valid manifest is the authority while its committed flat members match
    it or are incomplete. A complete flat family independently bound by its
    own rotation provenance outranks a disagreeing host-local manifest: this
    is the ordinary second-host/git-update case, not a torn publication. A
    manifest that fails validation still fails closed; mixed flat bytes never
    become authoritative. Without a manifest the historical flat/journal
    behavior holds.
    """

    flat_lock_path = lock_dir / f"{target}.lock"
    flat_export_path = lock_dir / f"{target}.resolved.json"
    flat_export_hash_path = lock_dir / f"{target}.resolved.sha256"
    journal_live = publication_journal_path(lock_dir, target).is_file()
    if family_manifest_path(lock_dir, target).is_file():
        try:
            resolved = resolve_family_manifest(lock_dir, target)
        except EnvironmentSpecError as exc:
            return LockArtifacts(
                target=target,
                lock_path=flat_lock_path,
                export_path=flat_export_path,
                export_hash_path=flat_export_hash_path,
                lock_bytes=None,
                export_bytes=None,
                declared_export_hash=None,
                declared_dependencies=declared_dependencies,
                publication_pending=True,
                family_authority="manifest",
                family_fault=str(exc),
            )
        generation_dir = resolved["generation_dir"]
        members: dict[str, str] = resolved["members"]
        flat_matches_manifest = all(
            (lock_dir / name).is_file() and hash_bytes((lock_dir / name).read_bytes()) == digest
            for name, digest in members.items()
        )
        if (
            not journal_live
            and not flat_matches_manifest
            and _flat_family_has_valid_provenance(lock_dir, target)
        ):
            return LockArtifacts(
                target=target,
                lock_path=flat_lock_path,
                export_path=flat_export_path,
                export_hash_path=flat_export_hash_path,
                lock_bytes=flat_lock_path.read_bytes(),
                export_bytes=flat_export_path.read_bytes(),
                declared_export_hash=flat_export_hash_path.read_text(encoding="utf-8").strip(),
                declared_dependencies=declared_dependencies,
                publication_pending=journal_live,
                family_authority="flat-newer-than-manifest",
            )
        lock_path = generation_dir / f"{target}.lock"
        export_path = generation_dir / f"{target}.resolved.json"
        export_hash_path = generation_dir / f"{target}.resolved.sha256"
        pending = journal_live or not flat_matches_manifest
        return LockArtifacts(
            target=target,
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=export_hash_path,
            lock_bytes=lock_path.read_bytes(),
            export_bytes=export_path.read_bytes(),
            declared_export_hash=export_hash_path.read_text(encoding="utf-8").strip(),
            declared_dependencies=declared_dependencies,
            publication_pending=False,
            family_authority="manifest",
            materialization_pending=pending,
        )
    return LockArtifacts(
        target=target,
        lock_path=flat_lock_path,
        export_path=flat_export_path,
        export_hash_path=flat_export_hash_path,
        lock_bytes=flat_lock_path.read_bytes() if flat_lock_path.is_file() else None,
        export_bytes=flat_export_path.read_bytes() if flat_export_path.is_file() else None,
        declared_export_hash=(
            flat_export_hash_path.read_text(encoding="utf-8").strip()
            if flat_export_hash_path.is_file()
            else None
        ),
        declared_dependencies=declared_dependencies,
        publication_pending=journal_live,
    )


def _flat_family_has_valid_provenance(lock_dir: Path, target: str) -> bool:
    """Return whether flat members form one independently proven rotation.

    Parameters
    ----------
    lock_dir:
        Lock-family directory containing git-visible members.
    target:
        Exact conda-lock platform prefix.

    Returns
    -------
    bool
        True only when the flat lock, export, export digest, probe receipt, and
        optional overlay are complete and digest-bound by the flat provenance.
    """

    lock_path = lock_dir / f"{target}.lock"
    export_path = lock_dir / f"{target}.resolved.json"
    export_hash_path = lock_dir / f"{target}.resolved.sha256"
    receipt_path = lock_dir / f"{target}.probes.json"
    provenance_path = lock_dir / f"{target}.provenance.json"
    required = (lock_path, export_path, export_hash_path, receipt_path, provenance_path)
    if not all(path.is_file() for path in required):
        return False
    try:
        provenance = json.loads(provenance_path.read_bytes())
        export_bytes = export_path.read_bytes()
        if not isinstance(provenance, dict):
            return False
        if provenance.get("intent") != lock_dir.parent.name or provenance.get("target") != target:
            return False
        if provenance.get("lock_sha256") != hash_bytes(lock_path.read_bytes()):
            return False
        if provenance.get("resolved_export_sha256") != hash_bytes(export_bytes):
            return False
        if provenance.get("probe_receipt_sha256") != hash_bytes(receipt_path.read_bytes()):
            return False
        if export_hash_path.read_text(encoding="utf-8").strip() != hash_bytes(export_bytes):
            return False
        overlay_path = lock_dir / f"{target}.pip-overlay.txt"
        overlay = provenance.get("pip_overlay")
        if overlay_path.is_file():
            return isinstance(overlay, dict) and overlay.get("manifest_sha256") == hash_bytes(
                overlay_path.read_bytes()
            )
        return overlay is None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _parse_probes(name: str, raw: Mapping[str, Any]) -> IntentProbes:
    """Parse one intent's probe declaration."""

    checks: list[ExportCheck] = []
    for value in _sequence(raw.get("export_checks"), f"export checks for {name}"):
        item = _mapping(value, f"export check for {name}")
        checks.append(
            ExportCheck(
                module=_string(item.get("module"), "export module"),
                attribute=_string(item.get("attribute"), "export attribute"),
            )
        )
    builds: list[SourceBuildProbe] = []
    for value in _sequence(raw.get("source_build", []), f"source builds for {name}"):
        item = _mapping(value, f"source build for {name}")
        max_attempts = item.get("max_attempts")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise EnvironmentSpecError("source-build max_attempts must be positive")
        builds.append(
            SourceBuildProbe(
                name=_string(item.get("name"), "source-build name"),
                packages=_strings(item.get("packages"), "source-build packages"),
                command=_strings(item.get("command"), "source-build command"),
                max_attempts=max_attempts,
            )
        )
    imports = _strings(raw.get("imports"), f"imports for {name}")
    if len(imports) < 3:
        raise EnvironmentSpecError(f"intent {name!r} must declare at least three import canaries")
    return IntentProbes(imports=imports, export_checks=tuple(checks), source_build=tuple(builds))


def _load_json_yaml(path: Path) -> Mapping[str, Any]:
    """Load a dependency-free JSON-form YAML document."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentSpecError(f"cannot load {path}: {exc}") from exc
    return _mapping(value, str(path))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EnvironmentSpecError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    """Require a list value."""

    if not isinstance(value, list):
        raise EnvironmentSpecError(f"{label} must be a list")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    """Require a non-empty list of non-empty strings."""

    sequence = _sequence(value, label)
    if not sequence or not all(isinstance(item, str) and item.strip() for item in sequence):
        raise EnvironmentSpecError(f"{label} must contain non-empty strings")
    return tuple(sequence)


def _string(value: Any, label: str) -> str:
    """Require one non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise EnvironmentSpecError(f"{label} must be a non-empty string")
    return value
