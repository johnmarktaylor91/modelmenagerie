"""Facts a library states about a model: its module, version, license and the paper it links.

Split from ``harvest`` (rung 2). A module's docstring speaks for the model only when the module is
named after the model's family.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from menagerie.pipeline.harvest_build import REPO, _hf_config, _resolve
from menagerie.pipeline.inputs import import_module

ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
RST_LINK = re.compile(r"`([^`<]+?)\s*<(https?://[^>]+)>`")
LICENSES = (("agpl", "AGPL-3.0"), ("apache", "Apache-2.0"), ("mit", "MIT"),
            ("bsd", "BSD-3-Clause"))


def _library_facts(call: str, args: list, config: dict,
                   family: str) -> tuple[list, dict[str, Any]]:
    """Return ``(docstring, whole, name)`` triples that describe the model, and its source.

    A module's docstring speaks for the model only when the module is named after its family:
    timm's ``vision_transformer`` also builds AIMv2 and ``convnext`` builds ConvNeXt V2.
    """

    if call == "timm.create_model":
        import timm.models

        entry = timm.models.model_entrypoint(args[0])
        module, symbol = entry.__module__, args[0]
        docs = [(entry.__doc__, True, None)]
    elif call == "torchvision.models.get_model":
        import torchvision.models

        builder = torchvision.models.get_model_builder(args[0])
        module, symbol = builder.__module__, args[0]
        docs = [(builder.__doc__, True, None)]
    else:
        fn = _resolve(call)
        cls = _hf_class(fn, config) if config else getattr(fn, "__self__", fn)
        module, symbol = cls.__module__, cls.__name__
        docs = [(cls.__doc__, False, symbol)]
    if _bare(module.rpartition(".")[2]) == _bare(family):
        docs.append((import_module(module).__doc__, True, None))
    dist = metadata.packages_distributions().get(call.split(".")[0], [call.split(".")[0]])[0]
    source = {"role": "library", "dist": dist, "version": _version(dist), "module": module,
              "symbol": symbol}
    return docs, source


def _hf_class(fn: Any, config: dict) -> type:
    mapping = getattr(getattr(fn, "__self__", None), "_model_mapping", None)
    return mapping[type(_hf_config(config))] if mapping is not None else fn


# Moved unchanged from harvest.py, where its cognitive complexity was already 19: one pass
# per docstring over three link forms in order of trust.
def _paper_sources(docs: Sequence[tuple[str | None, bool, str | None]]) -> list[dict[str, Any]]:
    """Return a paper source for the paper the library's docstrings link, if any.

    A class docstring counts only through its own reference target (``.. _Unet: url``), a
    titled link or its first paragraph, since its parameter notes link other papers.
    """

    for doc, whole, name in docs:
        doc = doc or ""
        target = name and re.search(rf"\.\. _{re.escape(name)}:\s*(https?://\S+)", doc)
        if target and ARXIV.search(target.group(1)):
            return [_paper_source(target.group(1), None)]
        titled = [(t, u) for t, u in RST_LINK.findall(doc) if ARXIV.search(u)]
        if titled:
            return [_paper_source(titled[0][1], titled[0][0].strip("\" "))]
        match = ARXIV.search(doc if whole else doc.split("\n\n")[0])
        if match:
            line = next(ln for ln in doc.splitlines() if match.group(0) in ln)
            return [_paper_source(match.group(0), _title(line[:line.index(match.group(0))]))]
    return []


def _title(text: str) -> str | None:
    """Return the paper title written before a link: quoted, backticked or a bare heading."""

    quoted = re.search(r"[`\"“]([^`\"”<]{8,}?)\s*[`\"”<]", text)
    if quoted:
        return quoted.group(1).strip()
    bare = re.sub(r"^\W*(papers?|references?)\s*:\s*|[\s(\-:]*(https?://)?$", "", text.strip(),
                  flags=re.IGNORECASE).strip(" -:*'\"")
    prose = re.search(r"\b(paper|from|described|based|model|see|@)\b", bare, re.IGNORECASE)
    return bare if len(bare) > 8 and not prose else None


def _bare(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _paper_source(url: str, title: str | None) -> dict[str, Any]:
    arxiv = ARXIV.search(url)
    return {"role": "paper", "url": f"https://arxiv.org/abs/{arxiv.group(1)}" if arxiv else url,
            "official": True, "_title": title}


def _code_source(path: Path) -> dict[str, Any]:
    """Our own code in this repository: a classic module or an adapter."""

    return {"role": "code", "url": path.resolve().relative_to(REPO).as_posix(),
            "framework": "pytorch", "official": False}


def _version(dist: str) -> str:
    return metadata.version(dist).split("+")[0]


def _license(dist: str) -> dict[str, Any]:
    """Return the SPDX id of a distribution's own license and its LICENSE file URL."""

    meta = metadata.metadata(dist)
    texts = [meta.get("License-Expression") or "", (meta.get("License") or "")[:80],
             *[c for c in meta.get_all("Classifier") or [] if c.startswith("License")]]
    spdx = next((s for key, s in LICENSES for t in texts if key in t.lower()), "unknown")
    urls = [u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL") or []]
    repo = next((u for u in [*urls, meta.get("Home-page") or ""] if "github.com/" in u), None)
    url = f"{repo.rstrip('/')}/blob/v{_version(dist)}/LICENSE" if repo else None
    return {"spdx": spdx, "url": url}


def _timm_cfg(name: str) -> Any:
    import timm

    return timm.get_pretrained_cfg(name) if timm.is_model(name) else None
