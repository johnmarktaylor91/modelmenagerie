"""SIGKILL child driver for the rotation all-or-nothing crash matrix.

Runs one real ``rotate_intent`` transaction in THIS process against the
scripted deterministic backend, and delivers a real ``SIGKILL`` to itself at
exactly one named cut. The parent test asserts the on-disk family disposition
afterwards, so every assertion sees the true post-mortem filesystem state --
no in-process exception unwinding can tidy up behind the crash.

Cut vocabulary (see ``rotation_transaction_support``):

* ``after-solve-commit`` -- the operator's solve cache and CAS are written,
  the family publication has not begun;
* ``after-probe-proof`` -- probes have passed, publication has not begun;
* ``before-generation-stage`` / ``before-manifest-switch`` /
  ``before-flat:<member>`` / ``before-journal-clear`` -- inside the
  publication transaction, via the publisher's observer hooks;
* ``after-publish`` -- the family is fully published, the campaign solve
  cache republication has not run.
"""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

import menagerie.crawler.tools.rotate_setup_artifacts as rotate_module
from menagerie.crawler.tests.rotation_transaction_support import (
    ScriptedRotationBackend,
    solve_fixture,
)


def _die() -> None:
    """Deliver a genuine SIGKILL to this process; nothing after this runs."""

    os.kill(os.getpid(), signal.SIGKILL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--kill-at", default=None)
    parser.add_argument("--new-tag", default="new")
    parser.add_argument("--campaign-state-root", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    kill_at = args.kill_at

    fixture = solve_fixture(args.new_tag)
    backend = ScriptedRotationBackend(
        args.state_root,
        fixture,
        after_solve_commit=_die if kill_at == "after-solve-commit" else None,
        after_probe_proof=_die if kill_at == "after-probe-proof" else None,
    )

    original_publish = rotate_module.publish_rotation_family

    def killing_publish(lock_dir: Path, target: str, members: dict, **kwargs) -> None:
        def observer(name: str) -> None:
            if kill_at == f"before-flat:{name}":
                _die()

        def phase_observer(token: str) -> None:
            if kill_at == f"before-{token}":
                _die()

        original_publish(
            lock_dir,
            target,
            members,
            observer=observer,
            phase_observer=phase_observer,
        )
        if kill_at == "after-publish":
            _die()

    rotate_module.publish_rotation_family = killing_publish
    try:
        old_identity, new_identity = rotate_module.rotate_intent(
            "demo",
            envs_root=args.envs_root,
            target="osx-arm64",
            environment_command=("scripted-test-operator",),
            state_root=args.state_root,
            env_root=args.env_root,
            minimum_free_bytes=0,
            backend=backend,
            campaign_state_roots=tuple(args.campaign_state_root),
            # EN-7d: the production default is the account-global execution
            # flock; the test universe must never touch a real campaign's lock.
            execution_lock_path=args.state_root / ".test-execution.flock",
        )
    finally:
        rotate_module.publish_rotation_family = original_publish
    print(f"rotated {old_identity} -> {new_identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
