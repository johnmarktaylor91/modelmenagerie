"""Runner code identity (DX-45: TC-25 + UU-10).

WHICH code adjudicated a leg was never recorded: the pilot launched through a
``.venv-crawler`` symlink, the editable finder could resolve torchlens or
menagerie outside the pilot tree (mid-leg dev edits are routine during fix
waves), and the launch certificate proved only the git tree. This module is
the ONE derivation of the runner's code identity, consumed by the launch
certificate (bind + drift refusal), the leg-open host probe pack
(outside-tree refusal), and the driver-lock owner record.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

#: The identity facts a certificate drift check compares, in order.
DRIFT_FACTS = ("interpreter", "python_version", "torchlens_origin", "menagerie_origin")


def _module_origin(name: str) -> Optional[str]:
    """Return one top-level module's resolved origin file, without importing it."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or spec.origin == "namespace":
        return None
    return str(Path(spec.origin).resolve())


def _distribution_version(name: str) -> Optional[str]:
    """Return one installed distribution's version, when it is installed."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_describe(origin: Optional[str]) -> Optional[str]:
    """Describe (with dirty marker) the repository enclosing one origin file."""

    if origin is None:
        return None
    try:
        completed = subprocess.run(
            ("git", "-C", str(Path(origin).parent), "describe", "--always", "--dirty"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def observed_repo_tree(repo_root: Path) -> dict[str, Any]:
    """Observe the running repository tree for the leg-config attestation.

    D-63 (PM-1): every leg records the tree it ran under. Unlike the launch
    certificate's ``current_tree_sha`` this NEVER refuses -- a leg-open
    attestation that raised on a dirty dev tree would turn provenance into an
    outage -- it discloses instead: ``tree_sha`` is ``HEAD^{tree}`` when git
    answers, ``dirty`` is whether anything outside ``.crawl-local/`` is
    modified, and both are ``None`` when the observation itself failed. The
    ENFORCING tree binding stays the certificate's recompute-and-compare;
    this row is what lets a rung census prove its legs shared one tree.

    Parameters
    ----------
    repo_root:
        Repository root enclosing the running code.

    Returns
    -------
    dict[str, Any]
        ``{"tree_sha": Optional[str], "dirty": Optional[bool]}``.
    """

    def _git(*argv: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ("git", "-C", str(repo_root), *argv),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout

    tree = _git("rev-parse", "HEAD^{tree}")
    status = _git("status", "--porcelain")
    dirty: Optional[bool] = None
    if status is not None:
        dirty = any(
            line.strip() and not line[3:].strip().strip('"').startswith(".crawl-local/")
            for line in status.splitlines()
        )
    return {
        "tree_sha": tree.strip() if tree else None,
        "dirty": dirty,
    }


def runner_code_identity() -> dict[str, Any]:
    """Derive THIS process's code identity.

    Returns
    -------
    dict[str, Any]
        Interpreter path, python version, resolved torchlens/menagerie
        origins, the installed torchlens distribution version, and a
        ``git describe --dirty`` of the tree enclosing the menagerie origin.
    """

    torchlens_origin = _module_origin("torchlens")
    menagerie_origin = _module_origin("menagerie")
    return {
        "interpreter": str(Path(sys.executable).resolve()),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torchlens_origin": torchlens_origin,
        "menagerie_origin": menagerie_origin,
        "torchlens_dist_version": _distribution_version("torchlens"),
        "git_describe": _git_describe(menagerie_origin or torchlens_origin),
    }


def runner_identity_drift(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Name every drift fact on which two runner identities disagree.

    ``git_describe`` and the distribution version are deliberately NOT drift
    facts: the tree binding is the launch certificate's own recomputed
    ``tree_sha`` check, and re-deriving describe strings across detached
    worktrees produces false drift. Drift here means THE RUNNER moved.

    Parameters
    ----------
    recorded, current:
        Identities minted earlier and derived now.

    Returns
    -------
    list[str]
        Sorted names of the disagreeing facts; empty means no drift.
    """

    return sorted(
        fact for fact in DRIFT_FACTS if recorded.get(fact) != current.get(fact)
    )


def origins_outside_tree(
    identity: Mapping[str, Any], tree_root: Path
) -> list[str]:
    """Name every resolved package origin outside one declared code tree.

    Parameters
    ----------
    identity:
        Identity from :func:`runner_code_identity`.
    tree_root:
        The tree the launch declares it is running (the pilot tree).

    Returns
    -------
    list[str]
        Sorted origin fact names resolving outside ``tree_root``.
    """

    resolved_root = tree_root.resolve()
    outside: list[str] = []
    for fact in ("torchlens_origin", "menagerie_origin"):
        origin = identity.get(fact)
        if origin is None:
            outside.append(fact)
            continue
        origin_path = Path(str(origin))
        if resolved_root != origin_path and resolved_root not in origin_path.parents:
            outside.append(fact)
    return sorted(outside)
