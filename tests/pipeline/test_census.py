"""Offline census boundary, restart and sampling checks."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, read_jsonl
from menagerie.pipeline import census, cli
from menagerie.pipeline.records import CensusRow, RosterRow, dumps


def seed(tmp_path: Path, rows: list[RosterRow]) -> DataLayout:
    data = DataLayout(tmp_path)
    data.roster.write_text("\n".join(dumps(r) for r in rows) + "\n")
    return data


def hinted(sid: str = "m1", url: str = "https://github.com/owner/repo") -> RosterRow:
    return RosterRow(name=sid, zoo="discovered-pytorch", stable_id=sid,
                     found_by=("legacy:c2-disco",), hint_urls=(url,), domain="vision")


def row(sid: int, domain: str = "vision", stars: int = 1) -> CensusRow:
    return CensusRow(id=f"m{sid}", repo=f"owner/repo{sid}", family=f"f-test{sid}",
                     checked="2026-10-08T00:00:00Z", alive=True, archived=False,
                     pytorch="requirements", size_kb=100, stars=stars,
                     domain=domain, eligible=True)


def fixture(records: list[CensusRow], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.jsonl").write_text("".join(dumps(r) + "\n" for r in records))
    census._save(out / "population.meta.json", {"repository_rows": len(records),
                 "ids": [r.id for r in records]})


def assert_hash_control(records: list[CensusRow], top: list[CensusRow], control: list[CensusRow]) -> None:
    selected = {r.repo for r in top}
    remaining = [r for r in records if r.repo not in selected]
    expected = sorted(remaining, key=lambda r: hashlib.sha256(r.family.encode()).hexdigest())[:50]
    assert [r.id for r in control] == [r.id for r in expected]


def fake_api(monkeypatch: pytest.MonkeyPatch, *, fail_at: int | None = None) -> list[str]:
    calls = []

    def get(self: census.GitHub, endpoint: str) -> object:
        calls.append(endpoint)
        self.calls += 1
        if fail_at == len(calls):
            raise census.CensusError("interrupted")
        if endpoint.endswith("/languages"):
            return {"Python": 50}
        if "/git/trees/" in endpoint:
            return {"tree": [{"type": "blob", "path": "requirements.txt", "size": 50, "sha": "a" * 40}]}
        if "/git/blobs/" in endpoint:
            return {"encoding": "base64", "content": base64.b64encode(b"torch>=2\n").decode()}
        return {"archived": False, "size": 100, "stargazers_count": 12,
                "default_branch": "main", "license": {"spdx_id": "MIT"}}

    monkeypatch.setattr(census.GitHub, "get", get)
    return calls


def test_population_filters_and_resolves_existing_ids(tmp_path: Path) -> None:
    data = seed(tmp_path, [hinted(), replace(hinted("m2"), state="not_architecture"),
                           replace(hinted("m3"), found_by=("legacy:c1-mech",)),
                           replace(hinted("m4"), stable_id=None)])
    from menagerie.identity import ensure_stable_ids
    ensure_stable_ids({("m4", "discovered-pytorch", "")}, tmp_path / "stable_ids.jsonl")
    assert [r.name for r, _ in census.population(data)] == ["m1", "m4"]
    assert all(r.stable_id for r, _ in census.population(data))


def test_run_caches_repository_and_resumes_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted(), hinted("m2")])
    calls = fake_api(monkeypatch)
    out = tmp_path / "sweep"
    first = census.run(data=data, out=out, limit=1)
    second = census.run(data=data, out=out)
    assert first["calls"] == 4 and second["calls"] == 0
    assert len(calls) == 4
    assert census.run(data=data, out=out)["processed"] == 0
    records = read_jsonl(CensusRow, out / "census.jsonl")
    assert len(records) == 2 and all(r.eligible for r in records)
    assert records[0].family == records[1].family
    assert "torch>=2" not in "".join(p.read_text() for p in out.rglob("*.json"))


def test_interrupted_repository_resumes_completed_api_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted()])
    out = tmp_path / "sweep"
    calls = fake_api(monkeypatch, fail_at=3)
    with pytest.raises(census.CensusError, match="interrupted"):
        census.run(data=data, out=out)
    assert not (out / "census.jsonl").exists()
    calls = fake_api(monkeypatch)
    census.run(data=data, out=out)
    assert len(calls) == 2 and "/git/trees/" in calls[0]


@pytest.mark.parametrize("meta,why", [
    ({"alive": False, "pytorch": "none"}, "not_alive"),
    ({"alive": True, "archived": True, "size_kb": 10, "pytorch": "none"}, "archived"),
    ({"alive": True, "archived": False, "size_kb": census.SIZE_KB, "pytorch": "none"}, "over_200mb"),
    ({"alive": True, "archived": False, "size_kb": 10, "pytorch": "none"}, "no_pytorch_evidence"),
])
def test_ineligible_rows_keep_typed_reason(meta: dict, why: str) -> None:
    assert census._row(hinted(), "owner/repo", meta).why == why


def test_import_and_requirement_evidence() -> None:
    assert census._torch_import("from torch.nn import Module")
    assert census._torch_import("import os, torch as t")
    assert not census._torch_import("# import torch\ns = 'import torch'")
    assert census._torch_requirement('dependencies=["torch>=2"]')
    assert not census._torch_requirement("# torch\ntorchvision\npytorch-lightning\n")


def http(status: int, body: dict, headers: dict[str, str] | None = None) -> str:
    lines = [f"HTTP/2.0 {status}", *(f"{k}: {v}" for k, v in (headers or {}).items())]
    return "\r\n".join(lines) + "\r\n\r\n" + json.dumps(body)


def test_gh_primary_and_secondary_limits_and_no_secret_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [http(403, {"message": "API rate limit exceeded"},
                      {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "200"}),
                 http(429, {}, {"Retry-After": "7"}), http(200, {"ok": True})]
    sleeps, commands = [], []
    monkeypatch.setattr(census.time, "time", lambda: 100)
    monkeypatch.setattr(census.time, "sleep", sleeps.append)

    def execute(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, responses.pop(0), "credential must never escape")

    monkeypatch.setattr(census.subprocess, "run", execute)
    assert census.GitHub().get("repos/a/b") == {"ok": True}
    assert sleeps[1] == 101 and sleeps[2] >= 7
    assert all(cmd[:2] == ["gh", "api"] and "--include" in cmd for cmd in commands)


def test_auth_failure_is_not_a_dead_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(census.time, "sleep", lambda _: None)
    monkeypatch.setattr(census.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, http(401, {}), "secret"))
    with pytest.raises(census.CensusError, match="HTTP 401") as exc:
        census.GitHub().get("repos/a/b")
    assert "secret" not in str(exc.value)


def test_stratified_500_and_hash_50_control_with_duplicates_and_used(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    out = tmp_path / "sweep"
    records = [row(i, "vision" if i <= 600 else "audio", stars=i) for i in range(1, 1001)]
    records += [replace(row(1001), repo=records[0].repo, family=records[0].family)]
    fixture(records, out)
    result = census.draw(data=data, out=out)
    selected = read_jsonl(CensusRow, out / "draw.jsonl")
    top = [r for r in selected if r.draw == "top"]
    control = [r for r in selected if r.draw == "control"]
    assert result["denominator"] == 550 and result["domains"] == {"audio": 200, "vision": 300}
    assert len({r.repo for r in selected}) == 550
    assert {r.id for r in top} == {f"m{i}" for i in (*range(301, 601), *range(801, 1001))}
    assert_hash_control(records[:1000], top, control)
    frozen = (out / "draw.jsonl").read_bytes()
    with pytest.raises(census.CensusError, match="frozen"):
        census.draw(data=data, out=out)
    assert (out / "draw.jsonl").read_bytes() == frozen


def test_used_repository_excludes_all_members_and_input_order_does_not_change_draw(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "packet.json").write_text(json.dumps({
        "members": [{"id": "m1"}], "sources": [{"url": "https://github.com/owner/repo2/tree/main"}]}))
    records = [row(i, stars=1) for i in range(1, 15)]
    for folder, ordered in (("a", records), ("b", list(reversed(records)))):
        fixture(ordered, tmp_path / folder)
        census.draw(data=data, out=tmp_path / folder, n=5, control=3)
    a = read_jsonl(CensusRow, tmp_path / "a" / "draw.jsonl")
    assert {r.id for r in a}.isdisjoint({"m1", "m2"})
    assert a == read_jsonl(CensusRow, tmp_path / "b" / "draw.jsonl")


def test_cli_draw_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path))
    out = tmp_path / "sweep"
    fixture([row(i) for i in range(1, 1001)], out)
    assert cli.main(["census", "draw", "--out", str(out)]) == 0
    assert len(read_jsonl(CensusRow, out / "draw.jsonl")) == 550


def test_resume_recovers_torn_final_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted()])
    out = tmp_path / "sweep"
    out.mkdir()
    (out / "census.jsonl").write_bytes(b'{"schema":"census/1","id":')
    fake_api(monkeypatch)
    census.run(data=data, out=out)
    assert len(read_jsonl(CensusRow, out / "census.jsonl")) == 1
    assert len(list(out.glob("*.torn-*"))) == 1


def test_embedded_hint_and_repository_redirect() -> None:
    assert census._hint(hinted(url="https://github.com/owner/repo (code); another link")) == "owner/repo"
    meta = {"alive": True, "archived": False, "size_kb": 1, "pytorch": "imports",
            "canonical_repo": "NewOwner/NewRepo"}
    result = census._row(hinted(), "owner/repo", meta)
    assert result.repo == "newowner/newrepo" and result.family == census._family(result.repo)


def test_invalid_repository_hint_is_counted_without_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted(url="https://github.com/owner")])
    calls = fake_api(monkeypatch)
    result = census.run(data=data, out=tmp_path / "sweep")
    assert result["population"] == 0 and result["invalid_hints"] == 1 and not calls
    meta = json.loads((tmp_path / "sweep" / "population.meta.json").read_text())
    assert meta["active_hinted_rows"] == 1
    assert meta["skipped"] == [{"id": "m1", "why": "invalid_repository_hint"}]


def test_missing_ids_fail_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [replace(hinted(), stable_id=None)])
    calls = fake_api(monkeypatch)
    with pytest.raises(census.CensusError, match="missing stable IDs"):
        census.run(data=data)
    assert not calls


def test_duplicate_repository_is_excluded_when_any_member_was_used(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "2026-10.jsonl").write_text('{"id":"m2"}\n')
    records = [row(1), replace(row(2), repo=row(1).repo), row(3)]
    fixture(records, tmp_path / "sweep")
    result = census.draw(data=data, out=tmp_path / "sweep", n=3)
    assert result["eligible_repositories"] == 1
    assert [r.id for r in read_jsonl(CensusRow, tmp_path / "sweep" / "draw.jsonl")] == ["m3"]


def test_empty_draw_still_freezes_denominator(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    fixture([], out)
    assert census.draw(data=DataLayout(tmp_path), out=out)["denominator"] == 0
    assert (out / "draw.jsonl").exists()
    with pytest.raises(census.CensusError, match="frozen"):
        census.draw(data=DataLayout(tmp_path), out=out)


def test_priority_uses_recency_license_and_paper_hint() -> None:
    from datetime import datetime, timezone
    instant = datetime(2026, 10, 8, tzinfo=timezone.utc)
    base = row(1, stars=0)
    recent = replace(base, pushed="2026-10-08T00:00:00Z")
    old = replace(base, pushed="2020-10-08T00:00:00Z")
    assert census.priority(recent, instant) > census.priority(old, instant) > census.priority(base, instant)
    assert census.priority(replace(base, license="MIT", paper_hint="arxiv:2601.00001"), instant) == 2


def test_largest_remainder_fills_exact_quota() -> None:
    groups = {"a": [row(1)] * 5, "b": [row(2)] * 3, "c": [row(3)] * 2}
    quotas = census._quotas(groups, 7)
    assert quotas == {"a": 4, "b": 2, "c": 1}


@pytest.mark.parametrize("status", [403, 409])
def test_permanent_http_refusal_is_not_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setattr(census.time, "sleep", lambda _: None)
    monkeypatch.setattr(census.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, http(status, {}), "secret"))
    api = census.GitHub()
    with pytest.raises(census.Unavailable, match=f"http_{status}"):
        api.get("repos/a/b")
    assert api.calls == 1


def test_unavailable_repository_does_not_block_following_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted(), hinted("m2", "https://github.com/owner/other")])
    fake_api(monkeypatch)
    original = census.GitHub.get

    def get(self: census.GitHub, endpoint: str) -> object:
        if endpoint == "repos/owner/repo":
            raise census.Unavailable("http_409")
        return original(self, endpoint)

    monkeypatch.setattr(census.GitHub, "get", get)
    out = tmp_path / "sweep"
    assert census.run(data=data, out=out)["complete"] == 2
    records = read_jsonl(CensusRow, out / "census.jsonl")
    assert records[0].why == "http_409" and records[1].eligible


def test_tree_checkpoint_retains_only_bounded_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted()])
    fake_api(monkeypatch)
    original = census.GitHub.get

    def get(self: census.GitHub, endpoint: str) -> object:
        if "/git/trees/" in endpoint:
            return {"tree": [{"path": f"model{i}.py", "type": "blob", "size": 50,
                              "sha": "a" * 40, "url": "discard"} for i in range(1000)]}
        return original(self, endpoint)

    monkeypatch.setattr(census.GitHub, "get", get)
    out = tmp_path / "sweep"
    census.run(data=data, out=out)
    cache = json.loads(next((out / "repos").glob("*.json")).read_text())
    assert len(cache["tree"]["tree"]) == census.FILE_LIMIT
    assert "discard" not in json.dumps(cache)


def test_draw_requires_complete_exact_population(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    fixture([row(1)], out)
    data = DataLayout(tmp_path)
    census._save(out / "population.meta.json", {"repository_rows": 2, "ids": ["m1", "m2"]})
    with pytest.raises(census.CensusError, match="incomplete"):
        census.draw(data=data, out=out)
    census._save(out / "population.meta.json", {"repository_rows": 1, "ids": ["m2"]})
    with pytest.raises(census.CensusError, match="incomplete"):
        census.draw(data=data, out=out)
    (out / "population.meta.json").unlink()
    with pytest.raises(census.CensusError, match="manifest missing"):
        census.draw(data=data, out=out)
    assert not (out / "draw.jsonl").exists()


def test_missing_domains_use_seeded_uniform_500_and_hash_control(tmp_path: Path) -> None:
    records = [replace(row(i, stars=i), domain="vision" if i == 1 else None) for i in range(1, 1001)]
    out = tmp_path / "sweep"
    fixture(list(reversed(records)), out)
    result = census.draw(data=DataLayout(tmp_path), out=out)
    selected = read_jsonl(CensusRow, out / "draw.jsonl")
    expected = sorted(records, key=lambda r: hashlib.sha256(
        (census.DRAW_SEED + "\0" + r.family).encode()).hexdigest())[:500]
    assert [r.id for r in selected[:500]] == [r.id for r in expected]
    rest = [r for r in records if r.id not in {x.id for x in expected}]
    assert [r.id for r in selected[500:]] == [r.id for r in sorted(rest, key=census._hash)[:50]]
    assert result["strategy"] == "seeded_uniform" and result["domain_stratification_deferred"]
    assert result["denominator"] == 550 and result["seed"] == census.DRAW_SEED



def test_used_old_repository_alias_excludes_canonical_redirect(tmp_path: Path) -> None:
    data = seed(tmp_path, [hinted(url="https://github.com/old/repo")])
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "packet.json").write_text(json.dumps({
        "sources": [{"url": "https://github.com/old/repo"}]}))
    groups = census._eligible_groups([replace(row(1), repo="new/repo")], data)
    assert not groups


@pytest.mark.parametrize("reason", ["missing", "truncated"])
def test_unavailable_tree_keeps_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    data = seed(tmp_path, [hinted()])
    fake_api(monkeypatch)
    original = census.GitHub.get

    def get(self: census.GitHub, endpoint: str) -> object:
        if "/git/trees/" in endpoint:
            return None if reason == "missing" else {"tree": [], "truncated": True}
        return original(self, endpoint)

    monkeypatch.setattr(census.GitHub, "get", get)
    out = tmp_path / "sweep"
    census.run(data=data, out=out)
    result = read_jsonl(CensusRow, out / "census.jsonl")[0]
    assert result.why == ("tree_unavailable" if reason == "missing" else "tree_truncated")



def test_non_utf8_evidence_retains_typed_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = seed(tmp_path, [hinted()])
    fake_api(monkeypatch)
    original = census.GitHub.get

    def get(self: census.GitHub, endpoint: str) -> object:
        if "/git/blobs/" in endpoint:
            return {"encoding": "base64", "content": base64.b64encode(b"\xff\xfe").decode()}
        return original(self, endpoint)

    monkeypatch.setattr(census.GitHub, "get", get)
    out = tmp_path / "sweep"
    census.run(data=data, out=out)
    assert read_jsonl(CensusRow, out / "census.jsonl")[0].why == "evidence_unreadable"
