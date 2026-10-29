"""Export the honest current rows of the verification ledger to JSONL.

The June validation pipeline's verification ledger (``menagerie.ledger``) is an
append-only SQLite database that has always been machine-local and gitignored
(``menagerie/data/verification.db``, see ``gotchas.md``). This module is a
read-only, additive export: it changes nothing about how
``validate_menagerie.py`` writes ledger rows, and does not replace
``menagerie.ledger`` or ``menagerie.status``. It exists so one run's results
can be committed or consumed (for example by
``site/scripts/build_catalog.py --ledger``) without shipping a SQLite file.

Each output line is one JSON object: the ``current_verification_real`` view's
columns for one ``stable_id`` (its most recent real-forward-status row, with
frozen pre-fix batch-cascade artifacts already excluded by that view -- see
``menagerie.ledger._create_current_verification_real_view``).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from menagerie.ledger import connect

DEFAULT_OUT = Path("verification.jsonl")


def dump_ledger_jsonl(db_path: Path, out_path: Path) -> int:
    """Write one JSON line per current honest ledger row.

    Parameters
    ----------
    db_path:
        Path to a ``menagerie.ledger`` SQLite verification database.
    out_path:
        Destination JSONL path; parent directories are created as needed.

    Returns
    -------
    int
        Number of rows written.

    Raises
    ------
    FileNotFoundError
        ``db_path`` does not exist. ``menagerie.ledger.connect`` otherwise
        initializes a fresh, empty database at any path it is given, which
        would silently turn a missing or mistyped ledger path into a
        legitimate-looking zero-row export instead of a clear error.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"verification ledger database not found: {db_path}")
    conn = connect(db_path)
    try:
        rows = _read_current_rows(conn)
    finally:
        conn.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _read_current_rows(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Return every ``current_verification_real`` row as a plain dict."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM current_verification_real ORDER BY stable_id")
    return [dict(row) for row in cursor.fetchall()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Verification ledger SQLite path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    """Run the ledger export."""
    args = parse_args()
    written = dump_ledger_jsonl(args.db, args.out)
    print(json.dumps({"written": written, "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
