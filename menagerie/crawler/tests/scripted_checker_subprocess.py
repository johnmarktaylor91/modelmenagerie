"""Argv checker subprocess for the true ``CommandCheckerLane`` E2E.

Invoked exactly like the production checker wrapper: ``argv[1]`` is the frozen
``request.json`` envelope path, the verdict travels ONLY as an atomic
``result.json`` at the envelope's ``required_output_path``, and exit ``0``
means "a candidate result exists" -- validation authority stays with the
dispatching lane's ``validate_checker_result``.

The checker-owned content is authored by the same decision-scripting support
the in-process rig uses (``checker_contract_support``), and the machine-owned
scaffold is stamped by the production ``apply_machine_owned_gate_fields`` --
the same publication sequence as the live operator wrapper, minus the model.

``G1_CHECKER_SUBPROCESS_FABRICATE=<gate field>`` makes the subprocess ship a
fabricated machine-owned value under an otherwise coherent self-hash, which the
dispatching lane MUST refuse: it simulates the checker-templated-from-a-fixture
class the pilot hit on 2026-08-04.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from menagerie.crawler.author_dispatch import write_envelope_atomic
    from menagerie.crawler.checker_dispatch import apply_machine_owned_gate_fields
    from menagerie.crawler.identity import stable_hash
    from menagerie.crawler.tests.checker_contract_support import (
        CheckerContractScript,
        _utc_now,
        author_result_for_envelope,
    )

    envelope = json.loads(Path(sys.argv[1]).read_text())
    script_env = os.environ.get("G1_CHECKER_SUBPROCESS_SCRIPT")
    script = (
        CheckerContractScript(**json.loads(script_env)) if script_env else CheckerContractScript()
    )
    raw = author_result_for_envelope(envelope, script)
    stamped = apply_machine_owned_gate_fields(
        raw, envelope, started_at=_utc_now(), finished_at=_utc_now()
    )
    fabricate = os.environ.get("G1_CHECKER_SUBPROCESS_FABRICATE")
    if fabricate:
        # A checker that invents a machine identity and reseals its own hash:
        # internally coherent bytes whose ONLY defect is the fabricated value.
        stamped[fabricate] = "sha256:" + "f" * 64
        stamped["result_envelope_sha256"] = stable_hash(
            {
                key: value
                for key, value in stamped.items()
                if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
            }
        )
    write_envelope_atomic(stamped, envelope["required_output_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
