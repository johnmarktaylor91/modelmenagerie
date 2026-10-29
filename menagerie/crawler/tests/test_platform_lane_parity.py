"""Linux-lane parity: closed inventory + the closable macOS counterparts (A5).

The campaign fleet is macOS; 14-17 Linux-gated sandbox/broker/semantic-read
tests never run on it. This module makes that gap a CLOSED, enforced inventory
instead of a silent one:

- Every Linux-gated test function in the corpus must appear here, either with
  a named macOS counterpart (which must exist) or with a typed
  ``requires-linux-host`` reason from a closed mechanism vocabulary. A new
  Linux-gated test that is not classified fails the inventory test.
- The counterpart that was genuinely missing -- network and outside-write
  denial under the real macOS Seatbelt boundary, mirroring
  ``test_linux_os_sandbox_wraps_command_and_denies_network_and_outside_write``
  -- is added at the bottom.

This inventory documents which assertions still need a Linux host (one Linux
runner or a scheduled CI job runs them; the release gate separately
reports the linux-64 registry rows as documented-expected on macOS). It never
weakens the Linux tests themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import signal
import subprocess
import sys

import pytest

from menagerie.crawler.policy import detect_os_sandbox, generate_macos_sandbox_profile
from menagerie.crawler.tests.conftest import _REPOSITORY_ROOT

SKIP_NO_SEATBELT = (
    "macos-seatbelt-unavailable: no working sandbox-exec boundary on this host"
    " (nested sandboxing is denied under Seatbelt confinement)"
)

#: Closed vocabulary of Linux-only mechanisms. Widening this set is a reviewed
#: decision, not a drive-by: each entry names a mechanism macOS genuinely does
#: not have, so "requires-linux-host" can never become a convenience label.
LINUX_ONLY_MECHANISMS = frozenset(
    {
        # Real worker supervision composes bwrap namespaces + strace audit.
        "linux-real-supervisor-mechanism",
        # Semantic read AUDIT (observing which paths were read) is strace-based;
        # macOS covers the DENIAL side one layer down via Seatbelt kills.
        "linux-strace-read-audit-mechanism",
        # Exact interpreter ELF transport-library capability shared between
        # mounts and parent audit; macOS grants reads declaratively instead.
        "linux-transport-capability-mechanism",
        # Syscall-broker mediation of native reads.
        "linux-syscall-broker-mechanism",
        # Minimal-namespace mount construction.
        "linux-minimal-namespace-mechanism",
    }
)

_TESTS = "menagerie/crawler/tests"

#: node -> {"macos_counterpart": node} XOR {"why": mechanism, "macos_related": [...]}.
#: ``macos_related`` points at macOS tests covering the same guarantee one
#: layer down (denial instead of audit); it is context, not a parity claim.
LINUX_GATED_PARITY: dict[str, dict[str, object]] = {
    f"{_TESTS}/test_compiler_os_authority_composition.py::"
    "test_manifest_v3_real_os_policy_denies_undeclared_root_member": {
        "macos_counterpart": (
            f"{_TESTS}/test_environment_authority_composition.py::"
            "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
        ),
    },
    f"{_TESTS}/test_environment_authority_composition.py::"
    "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package": {
        "macos_counterpart": (
            f"{_TESTS}/test_environment_authority_composition.py::"
            "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
        ),
    },
    f"{_TESTS}/test_modes.py::"
    "test_each_explicit_mode_runs_in_a_fresh_process_without_train_state_leak": {
        "why": "linux-real-supervisor-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_modes.py::"
    "test_stateless_modes_and_cold_confirmations_use_fresh_processes_and_fixed_input": {
        "why": "linux-real-supervisor-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_os_sandbox.py::"
    "test_linux_os_sandbox_wraps_command_and_denies_network_and_outside_write": {
        "macos_counterpart": (
            f"{_TESTS}/test_platform_lane_parity.py::"
            "test_macos_seatbelt_denies_network_and_outside_write"
        ),
    },
    f"{_TESTS}/test_os_sandbox.py::"
    "test_linux_mount_and_parent_audit_share_exact_transport_capability": {
        "why": "linux-transport-capability-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_os_sandbox.py::test_constructor_cannot_self_certify_forged_success_receipt": {
        "why": "linux-real-supervisor-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_os_sandbox.py::"
    "test_real_supervisor_retains_and_classifies_constructor_failure": {
        "why": "linux-real-supervisor-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_policy.py::test_python_open_hidden_env_and_repository_data_cannot_earn_run": {
        "why": "linux-strace-read-audit-mechanism",
        "macos_related": [
            f"{_TESTS}/test_os_sandbox.py::"
            "test_macos_seatbelt_allows_authorized_pyc_and_kills_unauthorized_pyc",
        ],
    },
    f"{_TESTS}/test_policy.py::"
    "test_native_reads_hidden_env_and_repository_data_poison_successful_receipt": {
        "why": "linux-strace-read-audit-mechanism",
        "macos_related": [
            f"{_TESTS}/test_release_ci_composition.py::"
            "test_macos_committed_lock_seatbelt_award_and_denial",
        ],
    },
    f"{_TESTS}/test_policy.py::test_legacy_author_declared_input_cannot_become_a_read_capability": {
        "why": "linux-strace-read-audit-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_policy.py::test_missing_linux_read_audit_fails_closed": {
        "why": "linux-strace-read-audit-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_release_ci_composition.py::test_linux_committed_lock_provenance_awards_in_ci": {
        "macos_counterpart": (
            f"{_TESTS}/test_release_ci_composition.py::"
            "test_macos_committed_lock_seatbelt_award_and_denial"
        ),
    },
    f"{_TESTS}/test_release_transport_composition.py::"
    "test_closed_transport_capability_awards_and_rejects_unlisted_library": {
        "why": "linux-transport-capability-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_slice_d_proposal_author.py::"
    "test_caught_os_sandbox_denial_in_flat_v1_cannot_satisfy_run_award": {
        "why": "linux-real-supervisor-mechanism",
        "macos_related": [
            f"{_TESTS}/test_release_ci_composition.py::"
            "test_macos_committed_lock_seatbelt_award_and_denial",
        ],
    },
    f"{_TESTS}/test_worker_supervisor.py::"
    "test_native_undeclared_weight_probe_is_denied_and_reported_as_an_attempt": {
        "why": "linux-syscall-broker-mechanism",
        "macos_related": [],
    },
    f"{_TESTS}/test_worker_supervisor.py::"
    "test_linux_author_supplied_absolute_input_grant_is_refused": {
        "why": "linux-minimal-namespace-mechanism",
        "macos_related": [],
    },
}


def _linux_gated_test_nodes() -> set[str]:
    """Scan the corpus for every Linux-gated top-level test function.

    A test is Linux-gated when its decorators or body compare
    ``sys.platform != "linux"`` (skipif decorator or inline runtime skip).

    Returns
    -------
    set[str]
        ``path::function`` roots relative to the repository root.
    """

    nodes: set[str] = set()
    for root in ("menagerie/crawler/tests", "tests/crawler"):
        for path in sorted((_REPOSITORY_ROOT / root).glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                    continue
                gated = any(
                    "sys.platform != 'linux'" in ast.unparse(decorator)
                    for decorator in node.decorator_list
                ) or "sys.platform != 'linux'" in ast.unparse(node)
                if gated:
                    nodes.add(f"{path.relative_to(_REPOSITORY_ROOT).as_posix()}::{node.name}")
    return nodes


def _function_exists(node_id: str) -> bool:
    """Return whether a ``path::function`` node names a real test function.

    Parameters
    ----------
    node_id:
        Repository-relative ``path::function`` root.

    Returns
    -------
    bool
        True when the module defines the named top-level function.
    """

    path_text, _, function_name = node_id.partition("::")
    path = _REPOSITORY_ROOT / path_text
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.FunctionDef) and node.name == function_name for node in tree.body
    )


@pytest.mark.smoke
def test_every_linux_gated_test_is_classified_in_the_parity_inventory() -> None:
    """The Linux-gated set and the committed inventory agree exactly.

    A new Linux-gated test must be classified (counterpart or typed
    requires-linux-host reason) in the same change; a removed one must leave
    the inventory. Silent growth of the never-runs-on-the-fleet set is exactly
    what this inventory exists to prevent.
    """

    scanned = _linux_gated_test_nodes()
    inventoried = set(LINUX_GATED_PARITY)
    assert scanned == inventoried, (
        f"unclassified Linux-gated tests: {sorted(scanned - inventoried)};"
        f" stale inventory rows: {sorted(inventoried - scanned)}"
    )


@pytest.mark.smoke
def test_parity_rows_are_well_formed_and_counterparts_exist() -> None:
    """Each row claims a real counterpart XOR a closed-vocabulary reason."""

    for node_id, row in LINUX_GATED_PARITY.items():
        counterpart = row.get("macos_counterpart")
        why = row.get("why")
        assert (counterpart is None) != (why is None), (
            f"{node_id}: exactly one of macos_counterpart/why is required"
        )
        if counterpart is not None:
            assert isinstance(counterpart, str)
            assert _function_exists(counterpart), (
                f"{node_id}: claimed macOS counterpart does not exist: {counterpart}"
            )
        else:
            assert why in LINUX_ONLY_MECHANISMS, (
                f"{node_id}: reason {why!r} is not in the closed mechanism vocabulary"
            )
            related = row.get("macos_related")
            assert isinstance(related, list)
            for related_node in related:
                assert _function_exists(str(related_node)), (
                    f"{node_id}: related macOS node does not exist: {related_node}"
                )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt integration test")
def test_macos_seatbelt_denies_network_and_outside_write(tmp_path: Path) -> None:
    """The real Seatbelt boundary allows scratch writes, denies the rest.

    macOS counterpart of the Linux sandbox integration test: inside a profile
    whose sole write root is the scratch directory, a write inside scratch
    succeeds, a write outside it is denied, and an outbound UDP send is
    denied. Mirrors the interpreter-launch recipe proven by
    ``test_macos_seatbelt_allows_authorized_pyc_and_kills_unauthorized_pyc``.

    Parameters
    ----------
    tmp_path:
        Isolated scratch and probe root.
    """

    sandbox = detect_os_sandbox("Darwin")
    if sandbox is None:
        pytest.skip(SKIP_NO_SEATBELT)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside.bin"
    venv_configuration = Path(sys.prefix) / "pyvenv.cfg"
    exact_reads = [Path(sys.executable)]
    if venv_configuration.is_file():
        exact_reads.append(venv_configuration)
    profile = generate_macos_sandbox_profile(
        (scratch,),
        allowed_read_paths=exact_reads,
        runtime_read_roots=(Path(sys.prefix), Path(sys.base_prefix)),
    )
    script = (
        "import io, socket, sys\n"
        f"scratch = {str(scratch)!r}\n"
        f"outside = {str(outside)!r}\n"
        "io.open(scratch + '/inside.bin', 'wb').write(b'allowed')\n"
        "write_denied = False\n"
        "network_denied = False\n"
        "try:\n"
        "    io.open(outside, 'wb').write(b'forbidden')\n"
        "except OSError:\n"
        "    write_denied = True\n"
        "client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "try:\n"
        "    client.sendto(b'probe', ('203.0.113.1', 9))\n"
        "except OSError:\n"
        "    network_denied = True\n"
        "finally:\n"
        "    client.close()\n"
        "print(f'write_denied={write_denied} network_denied={network_denied}')\n"
        "sys.exit(71 if write_denied and network_denied else 70)\n"
    )
    observation = subprocess.run(
        (sandbox.executable, "-p", profile, sys.executable, "-I", "-S", "-B", "-c", script),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # A SIGKILLed interpreter means the launch recipe (not the boundary
    # semantics) failed; surface it distinctly so parity rot is diagnosable.
    assert observation.returncode != -signal.SIGKILL, observation.stderr
    assert observation.returncode == 71, (observation.stdout, observation.stderr)
    assert "write_denied=True network_denied=True" in observation.stdout
    assert (scratch / "inside.bin").read_bytes() == b"allowed"
    assert not outside.exists()
    # The profile's own text pins the two denials this test adjudicates.
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert re.search(
        r'\(allow file-write\* \(subpath "' + re.escape(str(scratch.resolve())), profile
    )
