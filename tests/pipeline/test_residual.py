"""Frozen residual planning, denominator preservation and CLI dispatch."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, residual
from menagerie.pipeline.records import RecordError


def _rows(path: Path, rows: list[dict]) -> Path:
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=residual.FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_plan_retains_every_row_and_its_work_order(tmp_path: Path) -> None:
    rows = [{**dict.fromkeys(residual.FIELDS, ''), 'id': sid, 'list': label}
            for sid, label in [('m1', 'needs_adapter'), ('', 'discovered-paddle'),
                               ('m2', 'repin'), ('m3', 'missing_dependency'),
                               ('m3', 'x_standin')]]
    snapshot = _rows(tmp_path / 'rows.tsv', rows)
    out = tmp_path / 'plan.json'
    result = residual.plan(data=DataLayout(tmp_path), rows=snapshot, out=out)
    assert result['rows'] == 5 and result['order'] == list(residual.ORDER)
    assert result['held_lists'] == ['repin']
    assert result['work_lists']['discovered-paddle'][0]['held'] == 'no_stable_id'
    assert json.loads(out.read_text()) == result
    assert len(result['snapshot_sha256']) == 64


def test_plan_refuses_duplicate_ids_and_wrong_header(tmp_path: Path) -> None:
    row = {**dict.fromkeys(residual.FIELDS, ''), 'id': 'm1', 'list': 'needs_adapter'}
    path = _rows(tmp_path / 'rows.tsv', [row, row])
    with pytest.raises(RecordError, match='duplicate'):
        residual.plan(data=DataLayout(tmp_path), rows=path)
    path.write_text('id\tclass\nm1\tneeds_adapter\n')
    with pytest.raises(RecordError, match='missing columns'):
        residual.plan(data=DataLayout(tmp_path), rows=path)


def test_plan_current_empty_data_is_torch_free(tmp_path: Path) -> None:
    script = ('import sys; from pathlib import Path; '
              'from menagerie.pipeline import DataLayout, residual; '
              f'assert residual.plan(data=DataLayout(Path({str(tmp_path)!r})))['
              "'rows'] == 0; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', script], check=True)
