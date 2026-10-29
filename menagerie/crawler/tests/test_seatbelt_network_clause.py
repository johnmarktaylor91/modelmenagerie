"""UU-32: the macOS Seatbelt profile 200b runs under DENIES network egress.

The profile text asserts ``(deny network*)`` statically, but "offline" on
Darwin was never verified at the OS boundary — an env-var-deep offline (HF/pip
offline variables) would read identically until a worker actually reached the
network. This probe launches a real confined child through the production
wrapper (:func:`run_isolated_subprocess`, the exact path 200b workers run
under) and asserts an attempted UDP and TCP egress are both denied BY THE
SANDBOX (``EPERM``), not merely unrouted or refused.

On hosts without a working Seatbelt boundary (e.g. nested confinement) the
probe skips typed; it must execute on the release-gate host, where the
attestation machinery counts skips as red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.worker_supervisor import run_isolated_subprocess

_EGRESS_PROBE = (
    "import errno, socket, sys\n"
    "def probe(attempt):\n"
    "    try:\n"
    "        attempt()\n"
    "    except OSError as exc:\n"
    "        return exc.errno\n"
    "    return None\n"
    "def udp():\n"
    "    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
    "    try:\n"
    "        client.sendto(b'probe', ('203.0.113.1', 9))\n"
    "    finally:\n"
    "        client.close()\n"
    "def tcp():\n"
    "    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "    client.settimeout(5)\n"
    "    try:\n"
    "        client.connect(('127.0.0.1', 9))\n"
    "    finally:\n"
    "        client.close()\n"
    "udp_errno = probe(udp)\n"
    "tcp_errno = probe(tcp)\n"
    "udp_denied = udp_errno == errno.EPERM\n"
    "tcp_denied = tcp_errno == errno.EPERM\n"
    "print(f'udp_denied={udp_denied} tcp_denied={tcp_denied} '\n"
    "      f'udp_errno={udp_errno} tcp_errno={tcp_errno}')\n"
    "sys.exit(71 if udp_denied and tcp_denied else 70)\n"
)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt network probe")
def test_macos_seatbelt_denies_network_egress_from_confined_worker(
    tmp_path: Path,
) -> None:
    """A confined worker's attempted egress dies at the OS boundary, typed."""

    sandbox = detect_os_sandbox("Darwin")
    if sandbox is None:
        pytest.skip(
            "os-sandbox-unavailable: no working macOS Seatbelt boundary on this "
            "host; the network-clause probe MUST run on the release-gate host"
        )
    scratch = tmp_path / "scratch"

    observation = run_isolated_subprocess(
        (sys.executable, "-c", _EGRESS_PROBE),
        scratch,
        timeout_seconds=30,
        rss_limit_bytes=1024**3,
    )

    # The child ran under the production profile, not a bare interpreter.
    assert str(Path(sandbox.executable).resolve()) in {
        str(Path(part).resolve()) if part.startswith("/") else part
        for part in observation.argv
    }
    assert any(part.endswith("worker-sandbox.sb") for part in observation.argv)
    # Both egress attempts observed EPERM: denied by Seatbelt, not ECONNREFUSED
    # (which would mean the packet LEFT the sandbox) and not a resolver gap.
    assert observation.exit_code == 71, (
        observation.stdout_tail,
        observation.stderr_tail,
    )
    assert "udp_denied=True tcp_denied=True" in observation.stdout_tail
