"""A library's own documentation as a describe text source, with every fetch stood in."""

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, describe_text, library_docs
from menagerie.pipeline import read_json, store as store_mod, tick_full, write_json
from menagerie.pipeline.records import (Availability, Identity, Member, Packet, Search, Source)

AT = "2026-10-08T20:00:00Z"
SK_MODULE = '''"""Selective Kernel Networks (ResNet base)

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)
"""


class SelectiveKernelBasic:
    pass


def skresnet18(pretrained=False):
    """Constructs a Selective Kernel ResNet-18 model.

    This variation splits the input channels to the selective convolutions to keep the
    parameter count down, unlike the configurations of the paper.
    """
'''
SK_PAGE = ("<!--Copyright 2024 licence text that is no documentation-->\n# SK-ResNet\n\n"
           "**SK ResNet** is a variant of a ResNet that employs a Selective Kernel unit. "
           "All the large kernel convolutions in the bottleneck blocks are replaced by SK "
           "convolutions, so the network chooses its receptive field sizes adaptively.\n")
BERT_MODULE = '"""PyTorch BERT model."""\n\nclass BertModel:\n    """The bare Bert Model."""\n'
BERT_PAGE = ("# BERT\n\nBERT is a bidirectional transformer pretrained on unlabeled text to "
             "predict masked tokens in a sentence and whether one sentence follows another. "
             "The model learns contextual representations that transfer to many tasks with a "
             "single extra output layer.\n")


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataLayout, StoreLayout]:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    return DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")


def _tarball(top: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, text in files.items():
            body = text.encode()
            info = tarfile.TarInfo(f"{top}/{rel}")
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class Web:
    """Stands in for the network: URL -> bytes; any other URL fails."""

    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages, self.asked = pages, []

    def __call__(self, url: str, dest: Path) -> None:
        self.asked.append(url)
        if url not in self.pages:
            raise RuntimeError(f"{url}: HTTP 404")
        dest.write_bytes(self.pages[url])


TIMM_URL = ("https://codeload.github.com/huggingface/pytorch-image-models/tar.gz/refs/tags/"
            "v1.0.30")


def _timm_web(module: str = SK_MODULE, page: str = SK_PAGE) -> Web:
    return Web({TIMM_URL: _tarball("pytorch-image-models-1.0.30", {
        "timm/models/sknet.py": module, "timm/models/resnet.py": '"""ResNet."""\n',
        "hfdocs/source/models/skresnet.mdx": page,
        "hfdocs/source/models/resnet.mdx": "# ResNet\n" + "x" * 300,
        "tests/test_models.py": "x = 1\n"})})


def _packet(family: str = "f-skresnet", *sources: Source) -> Packet:
    sources = sources or (
        Source(sid="s1", role="library", dist="timm", version="1.0.30",
               module="timm.models.sknet", symbol="skresnet18"),)
    return Packet(family=family, written=AT, by="research", members=(Member(id="m1"),),
                  identity=Identity(canonical="Example"), sources=sources,
                  availability=Availability(paper="not_found", code="found", license="found",
                                            description="insufficient"),
                  search=Search(places=("library registry",)))


def test_a_timm_packet_gains_its_module_docstrings_and_model_page(roots) -> None:
    data, store = roots
    write_json(_packet(), data.packet("f-skresnet"))
    web = _timm_web()
    out = library_docs.docs(None, data=data, store=store, get=web)
    assert out[0]["status"] == "docs" and out[0]["sids"] == ["docs-timm"]
    packet = read_json(Packet, data.packet("f-skresnet"))
    src = next(s for s in packet.sources if s.sid == "docs-timm")
    assert (src.role, src.rev, src.official) == ("code", "v1.0.30", True)
    assert src.url == ("https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/"
                       "hfdocs/source/models/skresnet.mdx")
    assert packet.amendments[-1].by == "library-docs"
    assert packet.amendments[-1].evidence == ("docs-timm",)
    text = store_mod.get(src.text, store=store).read_text()
    assert text.startswith("Library documentation: timm 1.0.30 at v1.0.30\n")
    assert "Selective Kernel Networks (ResNet base)" in text and "Constructs a Selective" in text
    assert "SK ResNet** is a variant" in text and "Copyright" not in text
    assert "ResNet.\n" not in text  # another model's module is never read
    assert store_mod.index_line(src.text, data=data).url == src.url
    assert describe_text.candidates(json.loads(data.packet("f-skresnet").read_text())) == {
        "docs-timm": "docs"}
    assert library_docs.docs(None, data=data, store=store, get=web) == []  # no longer due


def test_only_due_packets_are_read_and_a_dry_run_fetches_and_writes_nothing(roots) -> None:
    data, store = roots
    write_json(_packet(), data.packet("f-skresnet"))
    texted = _packet("f-texted", *_packet().sources,
                     Source(sid="p1", role="page", url="https://example.org", text="a" * 64))
    write_json(texted, data.packet("f-texted"))
    write_json(replace(_packet("f-plain"), sources=(
        Source(sid="c1", role="code", url="https://github.com/x/y"),)), data.packet("f-plain"))
    before = {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    web = Web({})
    out = library_docs.docs(None, data=data, store=store, write=False, get=web)
    assert out == [{"family": "f-skresnet", "status": "due", "dists": ["timm"]}]
    assert web.asked == [] and before == {p: p.read_bytes() for p in data.root.rglob("*")
                                          if p.is_file()}


def test_too_little_text_adds_nothing_and_rests_the_family(roots, monkeypatch) -> None:
    data, store = roots
    write_json(_packet(), data.packet("f-skresnet"))
    web = _timm_web(module='"""SK."""\n', page="# SK\n")
    out = library_docs.docs(None, data=data, store=store, get=web)
    assert out[0]["status"] == "none" and "useful characters" in out[0]["why"]
    assert read_json(Packet, data.packet("f-skresnet")) == _packet()
    tried = json.loads((library_docs._tried_path()).read_text())
    assert tried["f-skresnet"]["why"].startswith("timm 1.0.30:")
    assert library_docs.docs(None, data=data, store=store, get=web) == []
    tried["f-skresnet"]["at"] = "2026-01-01T00:00:00Z"  # past the retry window
    library_docs._write_tried(tried)
    assert library_docs.docs(None, data=data, store=store, get=_timm_web())[0]["status"] == "docs"
    assert "f-skresnet" not in json.loads(library_docs._tried_path().read_text())


def test_a_missing_tag_falls_back_to_the_pypi_sdist_in_a_src_layout(roots) -> None:
    data, store = roots
    write_json(_packet("f-bert", Source(
        sid="s1", role="library", dist="transformers", version="5.18.0",
        module="transformers.models.bert.modeling_bert", symbol="BertModel")),
        data.packet("f-bert"))
    sdist = "https://files.example/transformers-5.18.0.tar.gz"
    web = Web({
        "https://pypi.org/pypi/transformers/5.18.0/json": json.dumps({"urls": [
            {"packagetype": "bdist_wheel", "filename": "transformers-5.18.0-py3-none-any.whl",
             "url": "https://files.example/wheel"},
            {"packagetype": "sdist", "filename": "transformers-5.18.0.tar.gz",
             "url": sdist}]}).encode(),
        sdist: _tarball("transformers-5.18.0", {
            "src/transformers/models/bert/modeling_bert.py": BERT_MODULE,
            "docs/source/en/model_doc/bert.md": BERT_PAGE})})
    out = library_docs.docs(None, data=data, store=store, get=web)
    assert out[0]["status"] == "docs"
    assert web.asked[:2] == [
        "https://codeload.github.com/huggingface/transformers/tar.gz/refs/tags/v5.18.0",
        "https://codeload.github.com/huggingface/transformers/tar.gz/refs/tags/5.18.0"]
    src = next(s for s in read_json(Packet, data.packet("f-bert")).sources
               if s.sid == "docs-transformers")
    assert src.rev == "5.18.0" and src.url.startswith("https://pypi.org/project/transformers/")
    text = store_mod.get(src.text, store=store).read_text()
    assert "PyTorch BERT model." in text and "bidirectional transformer" in text


def test_one_archive_serves_a_run_and_a_failed_release_is_asked_once(roots) -> None:
    data, store = roots
    for family in ("f-a", "f-b"):
        write_json(_packet(family), data.packet(family))
        write_json(_packet(family.replace("f-", "f-x"), Source(
            sid="s1", role="library", dist="nolib", version="1.0", module="nolib.m",
            symbol="M")), data.packet(family.replace("f-", "f-x")))
    web = _timm_web()
    out = library_docs.docs(None, data=data, store=store, get=web)
    assert [r["status"] for r in out] == ["none", "none", "docs", "docs"]
    assert web.asked.count(TIMM_URL) == 1
    assert web.asked.count("https://pypi.org/pypi/nolib/1.0/json") == 1


def test_limit_and_seconds_bound_a_run(roots) -> None:
    data, store = roots
    for family in ("f-a", "f-b", "f-c"):
        write_json(_packet(family), data.packet(family))
    assert len(library_docs.docs(None, data=data, store=store, limit=2, get=_timm_web())) == 2
    assert library_docs.docs(None, data=data, store=store, seconds=-1, get=_timm_web()) == []
    with pytest.raises(ValueError):
        library_docs.docs(None, data=data, store=store, limit=-1, get=_timm_web())


@pytest.mark.parametrize(("stem", "symbol", "hit"), [
    ("skresnet", "skresnet18", True), ("tf-efficientnet", "tf_efficientnet_b0", True),
    ("resnet", "resnet50d", True), ("resnet", "resnetv2_50", False),
    ("mobilenet-v3", "mobilenetv3_large_100", True), ("dla", "dlax", False)])
def test_a_page_names_the_model_a_symbol_is_a_size_of(stem, symbol, hit) -> None:
    assert library_docs._prefix(stem, symbol) is hit


def test_transformers_pages_match_the_model_type_whatever_its_separators() -> None:
    archive = library_docs.Archive("transformers", "5.18.0", "v5.18.0", "u/", {
        "docs/source/en/model_doc/audio-spectrogram-transformer.md": "AST",
        "docs/source/en/model_doc/bert.md": "BERT"})
    src = Source(sid="s1", role="library", dist="transformers", version="5.18.0",
                 symbol="ASTModel", module="transformers.models.audio_spectrogram_transformer."
                 "modeling_audio_spectrogram_transformer")
    assert library_docs._pages(archive, [src], "f-ast") == [
        "docs/source/en/model_doc/audio-spectrogram-transformer.md"]


def test_a_run_keeps_only_the_named_modules_and_the_model_pages() -> None:
    keep = library_docs.wanted("transformers", ["transformers.models.bert.modeling_bert"])
    assert keep("src/transformers/models/bert/modeling_bert.py")
    assert keep("docs/source/en/model_doc/gpt2.md")
    assert not keep("src/transformers/models/gpt2/modeling_gpt2.py")
    assert not keep("docs/source/ja/model_doc/bert.md")


def test_the_trickle_adds_library_docs_only_when_configured(monkeypatch, tmp_path) -> None:
    from menagerie.pipeline import find_paper

    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    calls = []
    monkeypatch.setattr(find_paper, "harvest", lambda *a, **k: [{"family": "f-p"}])
    monkeypatch.setattr(library_docs, "docs", lambda *a, **k: calls.append(k) or [
        {"family": "f-d", "status": "docs"}])
    data = DataLayout(tmp_path / "data")
    assert tick_full.trickle(data, {"trickle_limit": 5}, dry_run=False) == [{"family": "f-p"}]
    assert calls == []
    got = tick_full.trickle(data, {"trickle_limit": 5, "library_docs": 7,
                                   "library_docs_seconds": 30}, dry_run=True)
    assert [r["family"] for r in got] == ["f-p", "f-d"]
    assert (calls[0]["limit"], calls[0]["seconds"], calls[0]["write"]) == (7, 30.0, False)
