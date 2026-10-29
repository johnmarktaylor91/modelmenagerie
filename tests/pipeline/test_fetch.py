"""Fetch routing, license URLs pinned, and the 2a fetchers offline.

The network fetches themselves are the lanes' acceptance runs (real papers, repositories, Hub
files and bibliographic records into the store), not unit tests: here HTTP is faked, git reads a
local repository and a local server stands in for a short transfer.
"""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout, StoreLayout, drive, fetch, fetch_repos, read_jsonl, store,
)
from menagerie.pipeline.records import StoreIndexLine


@pytest.mark.parametrize(
    ("ref", "arxiv"),
    [
        ("1512.03385", "1512.03385"), ("arXiv:2106.08254", "2106.08254"),
        ("https://arxiv.org/abs/2411.14402v2", "2411.14402"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("cs/9905014", "cs/9905014"), ("10.1162/neco.1997.9.8.1735", None),
    ],
)
def test_arxiv_ids(ref: str, arxiv: str | None) -> None:
    assert fetch._arxiv_id(ref) == arxiv


@pytest.mark.parametrize(
    ("ref", "doi"),
    [
        ("10.1162/neco.1997.9.8.1735", "10.1162/neco.1997.9.8.1735"),
        ("doi:10.1109/5.726791", "10.1109/5.726791"),
        ("https://doi.org/10.1038/nature14539", "10.1038/nature14539"),
        ("1512.03385", None),
    ],
)
def test_dois(ref: str, doi: str | None) -> None:
    assert fetch._doi(ref) == doi


def test_license_urls_resolve_to_raw_and_refuse_moving_refs() -> None:
    blob = "https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/LICENSE"
    assert fetch._raw_github(blob) == (
        "https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.30/LICENSE")
    with pytest.raises(ValueError, match="moving ref"):
        fetch._raw_github("https://github.com/pytorch/vision/blob/main/LICENSE")


def test_an_unknown_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="not an arXiv id"):
        fetch.fetch_paper("Deep Residual Learning", store=None, data=None)  # type: ignore[arg-type]


# --- Rung 1: the 2a fetchers, offline (fake HTTP, a local git repository) ---------------------

MIT = b"MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining\n"


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StoreLayout, DataLayout, list]:
    events: list[dict] = []
    monkeypatch.setattr(drive, "event", lambda lane, family, verb, **f: events.append(
        {"lane": lane, "family": family, "verb": verb, **f}))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.delenv("MENAGERIE_STORE_REMOTE", raising=False)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))  # the bib cache
    return StoreLayout(tmp_path / "store"), DataLayout(tmp_path / "data"), events


def _index(data: DataLayout) -> list[StoreIndexLine]:
    return [line for shard in sorted((data.root / "store-index").glob("*.jsonl"))
            for line in read_jsonl(StoreIndexLine, shard)]


def _router(monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, bytes]]) -> list[str]:
    """Replace HTTP with ``routes`` (URL prefix -> status and body); return the URLs asked for."""

    asked: list[str] = []

    def fake(url: str, **kwargs: object) -> fetch.Response:
        asked.append(url)
        for prefix, (status, body) in routes.items():
            if url.startswith(prefix):
                return fetch.Response(status, url, "text/html" if body[:1] == b"<" else "", body)
        return fetch.Response(404, url, "", b"no route")

    monkeypatch.setattr(fetch, "_http", fake)
    return asked


def _git_repo(path: Path) -> str:
    path.mkdir()
    (path / "model.py").write_text("import torch\n\nclass Net(torch.nn.Module):\n    pass\n")
    (path / "requirements.txt").write_text("torch==1.2.0\ntorchvision==0.2.1\n")
    (path / "LICENSE").write_bytes(MIT)
    (path / "big.txt").write_text("x" * (2**20 + 10))
    (path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
    for args in (["init", "-q"], ["config", "uploadpack.allowfilter", "true"],
                 ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"]):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return f"file://{path}"


def test_fetch_code_saves_a_filtered_tree_and_its_license_from_another_host(
        roots: tuple, tmp_path: Path) -> None:
    st, data, events = roots
    url = _git_repo(tmp_path / "repo")
    first = fetch.fetch_code(url, family="f-x", store=st, data=data)
    again = fetch.fetch_code(url, family="f-x", store=st, data=data)
    assert first.blob == again.blob and len(first.rev or "") == 40
    assert (first.framework, first.spdx) == ("pytorch", "MIT")
    files = store.snapshot_files(st.blob(first.blob))
    assert sorted(files) == ["LICENSE", "model.py", "requirements.txt"]
    dropped = {d["path"]: d["why"] for d in store.snapshot_manifest(st.blob(first.blob))["dropped"]}
    assert dropped == {"big.txt": "over_1mb", "logo.png": "binary"}
    license_line = next(x for x in _index(data) if x.derived_from == first.blob)
    assert st.blob(license_line.sha256).read_bytes() == MIT
    assert [e["outcome"] for e in events] == ["saved", "saved"] and events[0]["verb"] == "fetch"


def test_fetch_code_types_a_missing_repository_as_inaccessible(roots: tuple, tmp_path: Path) -> None:
    st, data, events = roots
    source = fetch.fetch_code(f"file://{tmp_path}/nothing", family="f-x", store=st, data=data)
    assert (source.sid, source.blob) == ("code-inaccessible", None)
    assert events[-1]["outcome"] == "inaccessible"


def test_fetch_code_on_github_resolves_the_commit_and_keeps_a_subpath(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data, _ = roots
    commit = "c" * 40

    def gh(path: str) -> dict:
        if path.startswith("repos/gone/"):
            raise fetch.Inaccessible("HTTP 404")
        return {"sha": commit} if "/commits/" in path else {"size": 10}

    sizes = {"LICENSE": len(MIT), "README.md": 3, "beit3/model.py": 5, "other/x.py": 4}
    bodies = {"LICENSE": MIT, "README.md": b"hi\n", "beit3/model.py": b"x=1\n", "other/x.py": b"y\n"}
    monkeypatch.setattr(fetch_repos, "_gh_json", gh)
    monkeypatch.setattr(fetch_repos, "_github_tarball", lambda o, r, s: (sizes, bodies))
    source = fetch.fetch_code("https://github.com/microsoft/unilm/tree/master/beit3",
                              family="f-beit3", store=st, data=data)
    assert source.rev == commit and source.url.endswith(f"/tree/{commit}/beit3")
    assert sorted(store.snapshot_files(st.blob(source.blob))) == ["LICENSE", "beit3/model.py"]
    gone = fetch.fetch_code("https://github.com/gone/repo", family="f-x", store=st, data=data)
    assert gone.sid == "code-inaccessible"


def test_selection_keeps_readme_license_and_code_first_under_the_cap() -> None:
    sizes = {"docs/notes.md": 900_000, "README.md": 10, "src/a.py": 900_000, "LICENSE": 10,
             "data/x.csv": 900_000, "w.safetensors": 5}
    kept, dropped = fetch_repos._select({**sizes, **{f"src/m{i}.py": 900_000 for i in range(30)}}, "")
    assert kept[:3] == ["LICENSE", "README.md", "src/a.py"]
    assert sum(sizes.get(p, 900_000) for p in kept) <= fetch_repos.TREE_CAP
    assert {"path": "w.safetensors", "bytes": 5, "why": "binary"} in dropped
    assert any(d["why"] == "over_20mb_total" for d in dropped)


def test_fetch_hf_saves_card_config_and_auto_map_code_across_repositories_never_weights(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data, events = roots
    a, b = "a" * 40, "b" * 40
    config = {"architectures": ["X"], "auto_map": {"AutoModel": "org/code--modeling.X"}}
    api = "https://huggingface.co/api/models/"
    asked = _router(monkeypatch, {
        f"{api}org/model/revision/main": (200, json.dumps({"sha": a, "siblings": [
            {"rfilename": n, "size": 10} for n in
            ("README.md", "config.json", "pytorch_model.bin", "tokenizer.json", "logo.png")]}).encode()),
        f"{api}org/code/revision/main": (200, json.dumps({"sha": b, "siblings": [
            {"rfilename": n, "size": 10} for n in ("modeling.py", "config.json",
                                                    "model.safetensors")]}).encode()),
        f"https://huggingface.co/org/model/resolve/{a}/config.json": (200, json.dumps(config).encode()),
        f"https://huggingface.co/org/model/resolve/{a}/": (200, b"card or tokenizer\n"),
        f"https://huggingface.co/org/code/resolve/{b}/modeling.py": (200, b"import torch\n"),
        f"https://huggingface.co/org/code/resolve/{b}/config.json": (200, b"{}"),
    })
    card, snapshot, code = fetch.fetch_hf("org/model", family="f-m", store=st, data=data)
    assert (card.role, card.blob, card.text) == ("page", card.blob, card.blob)
    assert (snapshot.rev, snapshot.weights, snapshot.framework) == (a, "found", "pytorch")
    assert code.url == "https://huggingface.co/org/code" and code.rev == b
    assert "modeling.py" in store.snapshot_files(st.blob(code.blob))
    assert not [u for u in asked if u.endswith((".bin", ".safetensors", ".png"))]
    urls = {x.url for x in _index(data)}
    assert f"https://huggingface.co/org/model/resolve/{a}/config.json" in urls
    assert [e["kind"] for e in events] == ["code", "code"]


def test_a_gated_hub_repository_is_inaccessible(roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data, _ = roots
    _router(monkeypatch, {"https://huggingface.co/api/models/org/gated": (401, b"gated")})
    assert [s.sid for s in fetch.fetch_hf("org/gated", family="f-g", store=st, data=data)] == [
        "code-inaccessible"]


@pytest.mark.parametrize(("status", "body", "saved"), [
    (200, b"<html><body><p>Model page</p><p>We introduce X.</p></body></html>", True),
    (403, b"<html>Forbidden</html>", False),
    (200, b"<html><title>Just a moment...</title><div id=cf-chl-widget></div></html>", False),
    (200, b"<html><head><title>Verify</title></head><body><p>Please complete the captcha"
          b"</p></body></html>", False),
    # CQVIP: a real abstract page that loads a captcha script is saved (fix 13).
    (200, b"<html><head><title>Dual-branch FCN</title><script src='/js/captcha.js'></script>"
          b"</head><body><p>We introduce X.</p>" + b"<p>Abstract text.</p>" * 20 + b"</body></html>",
     True),
])
def test_fetch_page_saves_text_or_types_a_bot_wall(roots: tuple, monkeypatch: pytest.MonkeyPatch,
                                                   status: int, body: bytes, saved: bool) -> None:
    st, data, events = roots
    _router(monkeypatch, {"https://example.org": (status, body)})
    source = fetch.fetch_page("https://example.org/p", family="f-x", store=st, data=data)
    if saved:
        assert "We introduce X." in st.blob(source.text).read_text()
    else:
        assert (source.sid, source.blob) == ("page-inaccessible", None)
        assert events[-1]["outcome"] == "inaccessible"


def test_fetch_bib_saves_three_records_and_types_the_openalex_refusal(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data, events = roots
    monkeypatch.setenv("OPENALEX_API_KEY", "k-test")
    s2 = {"paperId": "p1", "externalIds": {"DOI": "10.1109/CVPR.2016.90"}, "citationCount": 9}
    asked = _router(monkeypatch, {
        "https://api.semanticscholar.org/": (200, json.dumps(s2).encode()),
        "https://api.openalex.org/": (429, b'{"error": "Insufficient budget for this request"}'),
        "https://api.crossref.org/works/10.1109/CVPR.2016.90": (200, b'{"message": {}}'),
        "https://arxiv.org/bibtex/1512.03385": (200, b"@misc{he2015, title={Deep}}\n"),
    })
    got = fetch.bib_records("1512.03385", family="f-resnet", store=st, data=data)
    assert [s.api for s in got.sources] == ["bibtex", "s2", "crossref"]
    assert got.notes == ("bib 1512.03385: openalex refused (insufficient budget)",)
    assert not any("api_key" in u or "k-test" in u for u in asked)
    assert not any("k-test" in (x.url or "") for x in _index(data))
    assert not any("@" in u for u in asked)  # no email address in any request
    assert {e["verb"] for e in events} == {"bib"}
    assert [s.api for s in fetch.fetch_bib("1512.03385", family="f-resnet", store=st,
                                           data=data)] == ["bibtex", "s2", "crossref"]
    with pytest.raises(ValueError, match="arXiv id or a DOI"):
        fetch.fetch_bib("Deep Residual Learning", family="f-resnet", store=st, data=data)


@pytest.fixture()
def server() -> Iterator[str]:
    """A local server: ``/short`` promises 1,000 bytes and sends 10; ``/budget`` refuses."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/short":
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b"%PDF-1.5 x")
                self.close_connection = True
            else:
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b"Insufficient budget")

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_a_short_transfer_is_retried_then_refused_never_stored(
        server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(fetch.FetchError, match="short|Incomplete|incomplete"):
        fetch._download(f"{server}/short", tmp_path / "doc")
    with pytest.raises(fetch.FetchError):
        fetch._http(f"{server}/short")
    assert fetch._short(10, None, b"%PDF-1.5", b"no trailer") is True
    assert fetch._short(10, "10", b"%PDF-1.5", b"%%EOF\n") is False


def test_an_insufficient_budget_refusal_is_not_retried(server: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    got = fetch._http(f"{server}/budget")
    assert (got.status, sleeps) == (429, [])


def test_bib_answers_are_kept_for_the_day_and_s2_cools_down(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    store_layout, data, events = roots
    asked = _router(monkeypatch, {"https://api.semanticscholar.org": (429, b"slow down"),
                                  "https://api.openalex.org": (200, b'{"id": "W1"}'),
                                  "https://arxiv.org/bibtex": (200, b"@article{x,}")})
    first = fetch.bib_records("1512.03385", family="f-x", store=store_layout, data=data)
    second = fetch.bib_records("1512.03385", family="f-x", store=store_layout, data=data)
    assert sum("semanticscholar" in u for u in asked) == 1  # then it cools down
    assert sum("openalex" in u for u in asked) == 1 and sum("bibtex" in u for u in asked) == 1
    assert "bib 1512.03385: s2 http_429 (cooling down)" in second.notes
    assert [s.blob for s in second.sources] == [s.blob for s in first.sources]
    outcomes = [e["outcome"] for e in events if e["verb"] == "bib"]
    assert outcomes.count("cached") == 2 and outcomes.count("cooldown") == 1
