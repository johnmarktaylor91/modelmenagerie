"""Print the unique test-module paths of a round21 release-proof registry.

The release jobs run the registry tier with the MATERIALIZED LOCK PREFIX's
python, whose package set is exactly the frozen lock. Pointing pytest at the
whole corpus root made *collection* import every corpus module — including
modules whose dependencies (e.g. PyYAML) postdate the lock — so the job died
with ImportError before reaching a single registry node (found the first time
the gate actually ran, 2026-08-16). The registry file itself is the closed
statement of what must run; this tool derives the collection roots from it so
the release run imports exactly the modules the registry names, fail-closed:
an unimportable registry module is still a collection error, and the conftest
attestation still requires every registry node collected and passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def registry_module_paths(registry_path: Path) -> list[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{registry_path} carries no registry nodes")
    return sorted({str(node).split("::", 1)[0] for node in nodes})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args(argv)
    for module_path in registry_module_paths(args.registry):
        print(module_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
