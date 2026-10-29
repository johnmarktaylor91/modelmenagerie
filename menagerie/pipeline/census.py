"""Resumable GitHub census and reproducible priority/control draws.

Only ``gh api`` handles authentication. Checkpoints contain projected metadata and
boolean evidence, never response bodies, headers, stderr or credentials. ``--out``
is a sweep directory; run writes census.jsonl, draw writes draw.jsonl there.
"""
from __future__ import annotations

import ast
import base64
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from menagerie.identity import load_stable_ids
from menagerie.pipeline import DataLayout, append_jsonl, now, read_jsonl, roster
from menagerie.pipeline.fetch_repos import FILE_CAP, REQUIREMENT_NAME
from menagerie.pipeline.records import CensusRow, RosterRow, dumps

SWEEP = "census-2026-10"
SIZE_KB = 200 * 1024
INTERVAL = 0.75  # below 5,000 primary requests/hour and 900 secondary requests/minute
ATTEMPTS = 4
FILE_LIMIT = 2
DRAW_SEED = "census-2026-10-uniform-v1"


class CensusError(RuntimeError):
    """A resumable transport or data failure; no external response text is included."""


class Unavailable(CensusError):
    """A permanent repository-level refusal which belongs in the census denominator."""


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _repo(url: str) -> str | None:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != "github.com":
        return None
    if len(parts) < 2:
        return None
    repo = "/".join(parts[:2]).removesuffix(".git")
    return repo.lower() if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) else None


def _hint(row: RosterRow) -> str | None:
    urls = (*row.hint_urls, row.source_url or "")
    return next((r for text in urls
                 for u in re.findall(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text)
                 if (r := _repo(u))), None)


def _family(repo: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", repo.lower()).strip("-")
    return f"f-gh-{slug}-{hashlib.sha256(repo.encode()).hexdigest()[:12]}"


def _population(data: DataLayout) -> tuple[list[tuple[RosterRow, str]], list[dict[str, str]]]:
    """Active c2 hints, in roster order; resolve existing IDs without changing the roster."""
    path = data.root / "stable_ids.jsonl"
    ids = load_stable_ids(path) if path.exists() else {}
    rows, skipped = [], []
    for row in roster.read(data=data):
        if row.state != "active" or "legacy:c2-disco" not in row.found_by:
            continue
        repo = _hint(row)
        if not repo and any("github.com" in u for u in row.hint_urls):
            skipped.append({"id": row.stable_id or "", "why": "invalid_repository_hint"})
        if repo:
            sid = row.stable_id or ids.get((row.name, row.zoo, row.variant))
            rows.append((replace(row, stable_id=sid), repo))
    return rows, skipped


def population(data: DataLayout) -> list[tuple[RosterRow, str]]:
    """Return active c2 rows whose hints identify a repository."""
    return _population(data)[0]


def _response(stdout: str) -> tuple[int, dict[str, str], Any]:
    # gh --include uses a CRLF or LF blank line between headers and JSON.
    head, separator, body = stdout.replace("\r\n", "\n").partition("\n\n")
    if not separator:
        raise CensusError("gh returned no HTTP response headers")
    status = re.match(r"HTTP/\S+\s+(\d+)", head)
    if not status:
        raise CensusError("gh returned an invalid HTTP status")
    headers = {k.lower(): v.strip() for line in head.splitlines()[1:]
               if ":" in line for k, v in [line.split(":", 1)]}
    try:
        value = json.loads(body)
    except ValueError:
        value = None
    return int(status[1]), headers, value


class GitHub:
    """Paced requests with primary/secondary quota waits and bounded transient retries."""

    def __init__(self) -> None:
        self.next_at = 0.0
        self.calls = 0

    def get(self, endpoint: str) -> Any:
        for attempt in range(ATTEMPTS):
            time.sleep(max(0, self.next_at - time.time()))
            self.next_at = time.time() + INTERVAL
            done = self._request(endpoint, attempt)
            if done is None:
                continue
            self.calls += 1
            status, headers, value = _response(done.stdout)
            self._quota(headers)
            if status in (404, 410, 451):
                return None
            if status == 200 and isinstance(value, dict):
                return value
            if self._retry(status, headers, value, attempt):
                continue
            if status in (403, 409) and not self._limited(status, headers, value):
                raise Unavailable(f"http_{status}")
            raise CensusError(f"GitHub HTTP {status}; rerun to resume")
        raise CensusError("GitHub retry budget exhausted; rerun to resume")

    def _request(self, endpoint: str, attempt: int) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["gh", "api", "--hostname", "github.com", "--method", "GET", "--include",
                 "-H", "Accept: application/vnd.github+json", endpoint],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "GH_DEBUG": "", "GH_PROMPT_DISABLED": "1"})
        except (subprocess.TimeoutExpired, OSError):
            if attempt == ATTEMPTS - 1:
                raise CensusError("gh transport unavailable; rerun to resume") from None
            self.next_at = time.time() + 2 ** attempt
            return None

    def _quota(self, headers: dict[str, str]) -> None:
        if headers.get("x-ratelimit-remaining") == "0":
            reset = headers.get("x-ratelimit-reset", "")
            if not reset.isdigit():
                raise CensusError("GitHub quota exhausted without a reset time")
            self.next_at = max(self.next_at, int(reset) + 1)

    @staticmethod
    def _limited(status: int, headers: dict[str, str], value: Any) -> bool:
        message = str(value.get("message", "")).lower() if isinstance(value, dict) else ""
        return status == 429 or (status == 403 and (
            headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
            or "rate limit" in message))

    def _retry(self, status: int, headers: dict[str, str], value: Any, attempt: int) -> bool:
        limited = self._limited(status, headers, value)
        if limited:
            retry = headers.get("retry-after", "60")
            self.next_at = max(self.next_at, time.time() + (int(retry) if retry.isdigit() else 60) * 2 ** attempt)
        elif status >= 500:
            self.next_at = max(self.next_at, time.time() + 2 ** attempt)
        return attempt < ATTEMPTS - 1 and (limited or status >= 500)


def _torch_import(text: str) -> bool:
    try:
        nodes = ast.walk(ast.parse(text))
        return any((isinstance(node, ast.Import) and any(
            n.name.split(".")[0] == "torch" for n in node.names)) or (
            isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
                   for node in nodes)
    except SyntaxError:
        return bool(re.search(r"^\s*(?:from\s+torch(?:\.|\s)|import\s+torch(?:\s|,|$))", text, re.M))


def _torch_requirement(text: str) -> bool:
    clean = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return bool(re.search(r"(?<![\w.-])(?:torch|pytorch)(?![\w.-])", clean, re.I))


def _evidence_candidates(tree: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    files = [f for f in tree.get("tree", []) if f.get("type") == "blob"
             and f.get("size", FILE_CAP + 1) <= FILE_CAP]
    reqs = sorted((f for f in files if REQUIREMENT_NAME.fullmatch(Path(f["path"]).name)),
                  key=lambda f: (f["path"].count("/"), f["path"]))[:FILE_LIMIT]
    sources = sorted((f for f in files if f["path"].endswith(".py")),
                     key=lambda f: (not bool(re.search(r"model|network|train", f["path"], re.I)),
                                    f["path"].count("/"), f["path"]))[:FILE_LIMIT]
    return [("requirements", reqs), ("imports", sources)]


def _blob_evidence(api: GitHub, repo: str, sha: str) -> dict[str, bool]:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise CensusError("GitHub returned an invalid blob identifier")
    blob = api.get(f"repos/{repo}/git/blobs/{sha}")
    if not blob or blob.get("encoding") != "base64":
        raise Unavailable("evidence_unavailable")
    try:
        text = base64.b64decode(blob["content"], validate=False).decode("utf-8")
    except (ValueError, UnicodeError, KeyError, TypeError):
        raise Unavailable("evidence_unreadable") from None
    return {"requirements": _torch_requirement(text), "imports": _torch_import(text)}


def _evidence(api: GitHub, repo: str, tree: dict[str, Any], cache: dict[str, Any], path: Path) -> str:
    evidence = cache.setdefault("files", {})
    for kind, candidates in _evidence_candidates(tree):
        for item in candidates:
            sha = item["sha"]
            if sha not in evidence:
                evidence[sha] = _blob_evidence(api, repo, sha)
                _save(path, cache)
            if evidence[sha][kind]:
                return kind
    return "none"


def _metadata(raw: dict[str, Any]) -> dict[str, Any]:
    license_id = (raw.get("license") or {}).get("spdx_id")
    return {"alive": True, "archived": raw["archived"], "size_kb": raw["size"],
            "stars": raw["stargazers_count"], "pushed": raw.get("pushed_at"),
            "license": license_id if license_id not in (None, "NOASSERTION", "OTHER") else None,
            "branch": raw["default_branch"], "canonical_repo": raw.get("full_name")}


def _tree_checkpoint(api: GitHub, repo: str, branch: str) -> dict[str, Any]:
    tree = api.get(f"repos/{repo}/git/trees/{quote(branch, safe='')}?recursive=1")
    if tree is None:
        raise Unavailable("tree_unavailable")
    from menagerie.pipeline.census_harvest import candidate_paths  # noqa: PLC0415

    harvest = {p["path"] for p in candidate_paths(tree)}
    candidates = [item for _, items in _evidence_candidates(tree) for item in items]
    candidates += [item for item in tree.get("tree", []) if item.get("path") in harvest]
    compact = {item["path"]: {k: item[k] for k in ("path", "sha", "size", "type")}
               for item in candidates}
    return {"tree": list(compact.values()), "truncated": bool(tree.get("truncated"))}


def _inspect_steps(api: GitHub, repo: str, root: Path) -> dict[str, Any]:
    path = root / "repos" / f"{hashlib.sha256(repo.encode()).hexdigest()}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if "metadata" not in cache:
        raw = api.get(f"repos/{repo}")
        cache["metadata"] = _metadata(raw) if raw else {"alive": False}
        _save(path, cache)
    meta = cache["metadata"]
    if not meta["alive"] or meta["archived"] or meta["size_kb"] >= SIZE_KB:
        return {**meta, "pytorch": "none"}
    if "languages" not in cache:
        languages = api.get(f"repos/{repo}/languages")
        if languages is None:
            raise Unavailable("languages_unavailable")
        cache["languages"] = list(languages)
        _save(path, cache)
    if "tree" not in cache:
        cache["tree"] = _tree_checkpoint(api, repo, meta["branch"])
        _save(path, cache)
    kind = _evidence(api, repo, cache["tree"], cache, path)
    if kind == "none" and cache["tree"].get("truncated"):
        raise Unavailable("tree_truncated")
    return {**meta, "pytorch": kind}



def _inspect(api: GitHub, repo: str, root: Path) -> dict[str, Any]:
    """Continue past permanent repository refusals; transient/auth failures still stop."""
    try:
        return _inspect_steps(api, repo, root)
    except Unavailable as exc:
        path = root / "repos" / f"{hashlib.sha256(repo.encode()).hexdigest()}.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        return {**cache.get("metadata", {"alive": False}), "pytorch": "none", "unavailable": str(exc)}

def _row(row: RosterRow, repo: str, meta: dict[str, Any]) -> CensusRow:
    why = meta.get("unavailable")
    for failed, reason in ((not meta["alive"], "not_alive"), (meta.get("archived"), "archived"),
                           (meta.get("size_kb", 0) >= SIZE_KB, "over_200mb"),
                           (meta["pytorch"] == "none", "no_pytorch_evidence")):
        if failed and why is None:
            why = reason
            break
    return CensusRow(id=row.stable_id, repo=(meta.get("canonical_repo") or repo).lower(),
                     checked=now(), family=_family((meta.get("canonical_repo") or repo).lower()),
                     domain=row.domain, paper_hint=row.hint_paper, eligible=why is None, why=why,
                     **{k: v for k, v in meta.items() if k not in ("branch", "canonical_repo", "unavailable")})




def _lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise CensusError("census sweep already has a writer") from None

def _completed(path: Path) -> list[CensusRow]:
    """Recover an interrupted final append, retaining the torn bytes for inspection."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        prefix, _, tail = raw.rpartition(b"\n")
        # A complete JSON record without its newline is kept; only incomplete JSON is removed.
        try:
            json.loads(tail)
        except ValueError:
            backup = path.with_name(f"{path.name}.torn-{hashlib.sha256(tail).hexdigest()[:12]}")
            backup.write_bytes(tail)
            with path.open("wb") as handle:
                handle.write(prefix + b"\n" if prefix else b"")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return read_jsonl(CensusRow, path)

def run(*, data: DataLayout, out: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Collect up to limit new roster rows; rerunning skips complete IDs and cached API steps."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    root = Path(out) if out is not None else data.root / "sweeps" / SWEEP
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        _lock(lock.fileno())
        path = root / "census.jsonl"
        existing = _completed(path)
        completed = {r.id for r in existing}
        candidates, skipped = _population(data)
        _save(root / "population.meta.json", {"repository_rows": len(candidates),
              "active_hinted_rows": len(candidates) + len(skipped), "skipped": skipped,
              "ids": sorted(row.stable_id for row, _ in candidates if row.stable_id)})
        todo = [(row, repo) for row, repo in candidates if row.stable_id not in completed]
        if limit is not None:
            todo = todo[:limit]
        if any(row.stable_id is None for row, _ in todo):
            raise CensusError("c2 roster has missing stable IDs; repair identity before census run")
        api = GitHub()
        for row, repo in todo:
            record = _row(row, repo, _inspect(api, repo, root))
            append_jsonl([record], path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        return {"population": len(candidates), "invalid_hints": len(skipped), "processed": len(todo), "complete": len(existing) + len(todo),
                "calls": api.calls, "out": str(path)}


def _record_usage(value: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids = {m["id"] for m in value.get("members", [])}
    if value.get("id"):
        ids.add(value["id"])
    repos = {repo for source in value.get("sources", [])
             if (repo := _repo(source.get("url") or ""))}
    return ids, repos


def _used(data: DataLayout) -> tuple[set[str], set[str]]:
    ids, repos = set(), set()
    paths = [path for folder in ("models", "research")
             for path in (data.root / folder).rglob("*.json")]
    for path in paths:
        member_ids, source_repos = _record_usage(json.loads(path.read_text()))
        ids.update(member_ids)
        repos.update(source_repos)
    for path in (data.root / "ledger").glob("*.jsonl"):
        ids.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
    return ids, repos


def priority(row: CensusRow, as_of: datetime) -> float:
    """log1p(stars) + 1/(1+age_days/365) + known-license + paper-hint.

    Scores use the latest census check time, making a saved draw reproducible.
    """
    pushed = datetime.fromisoformat(row.pushed.replace("Z", "+00:00")) if row.pushed else None
    days = max(0, (as_of - pushed).total_seconds() / 86400) if pushed else None
    recency = 1 / (1 + days / 365) if days is not None else 0
    return math.log1p(max(0, row.stars or 0)) + recency + bool(row.license) + bool(row.paper_hint)


def _hash(row: CensusRow) -> str:
    return hashlib.sha256((row.family or _family(row.repo)).encode()).hexdigest()


def _quotas(groups: dict[str, list[CensusRow]], n: int) -> dict[str, int]:
    total = sum(map(len, groups.values()))
    exact = {k: n * len(v) / total for k, v in groups.items()} if total else {}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    order = sorted(groups, key=lambda k: (-(exact[k] - quotas[k]), hashlib.sha256(k.encode()).hexdigest()))
    for key in order[:n - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _eligible_groups(rows: list[CensusRow], data: DataLayout) -> dict[str, list[CensusRow]]:
    used_ids, used_repos = _used(data)
    if data.roster.exists():
        used_ids |= {r.stable_id for r, hint in population(data) if hint in used_repos}
    excluded_repos = used_repos | {r.repo.lower() for r in rows if r.id in used_ids}
    by_repo = {}
    for row in sorted(rows, key=lambda r: (int(r.id[1:]), r.id)):
        if row.eligible and row.repo.lower() not in excluded_repos:
            by_repo.setdefault(row.repo.lower(), row)
    groups = defaultdict(list)
    for row in by_repo.values():
        groups[row.domain or "unknown"].append(row)
    return groups


def _top_sample(groups: dict[str, list[CensusRow]], n: int) -> list[CensusRow]:
    if "unknown" in groups:
        population = [r for group in groups.values() for r in group]
        return sorted(population, key=lambda r: hashlib.sha256(
            (DRAW_SEED + "\0" + (r.family or _family(r.repo))).encode()).hexdigest())[:n]
    quotas = _quotas(groups, min(n, sum(map(len, groups.values()))))
    return [r for domain, group in sorted(groups.items())
            for r in sorted(group, key=lambda r: (-r.score, _hash(r)))[:quotas[domain]]]


def _select(groups: dict[str, list[CensusRow]], n: int, control: int) -> list[CensusRow]:
    top = [replace(r, draw="top") for r in _top_sample(groups, n)]
    selected = {r.repo for r in top}
    rest = [r for group in groups.values() for r in group if r.repo not in selected]
    controls = [replace(r, draw="control") for r in sorted(rest, key=_hash)[:control]]
    return top + controls


def _require_complete(root: Path, rows: list[CensusRow]) -> None:
    path = root / "population.meta.json"
    if not path.exists():
        raise CensusError("census population manifest missing; run census before drawing")
    meta = json.loads(path.read_text())
    ids = {r.id for r in rows}
    if len(rows) != len(ids) or len(ids) != meta["repository_rows"] or ids != set(meta["ids"]):
        raise CensusError("census is incomplete; finish census run before drawing")


def _draw(*, data: DataLayout, n: int = 500, control: int = 50, out: Path | None = None) -> dict[str, Any]:
    """Select one family/repository; retain selected failures in the immutable denominator.

    Hamilton apportionment uses the eligible repository population by roster domain.
    Missing domains select a seeded uniform sample instead; domain filling is deferred.
    Control is global family hash order from the remainder, without score weighting.
    This verb never queues, traces or rewrites the roster or ledger.
    """
    if n < 0 or control < 0:
        raise ValueError("draw sizes must be nonnegative")
    root = Path(out) if out is not None else data.root / "sweeps" / SWEEP
    rows = read_jsonl(CensusRow, root / "census.jsonl")
    _require_complete(root, rows)
    if (root / "draw.jsonl").exists():
        raise CensusError("draw already frozen; use a new sweep directory to redraw")
    as_of = max((datetime.fromisoformat(r.checked.replace("Z", "+00:00")) for r in rows),
                default=datetime(1970, 1, 1, tzinfo=timezone.utc))
    groups = {domain: [replace(r, score=priority(r, as_of)) for r in group]
              for domain, group in _eligible_groups(rows, data).items()}
    selected = _select(groups, n, control)
    summary = _draw_summary(selected, groups, n, control, as_of, root / "census.jsonl")
    _save(root / "draw.meta.json", summary)
    _freeze(selected, root / "draw.jsonl")
    return summary



def _draw_summary(selected: list[CensusRow], groups: dict[str, list[CensusRow]],
                  n: int, control: int, as_of: datetime, source: Path) -> dict[str, Any]:
    """Save selection provenance separately from the atomic denominator transaction."""
    top = [r for r in selected if r.draw == "top"]
    controls = [r for r in selected if r.draw == "control"]
    return {"eligible_repositories": sum(map(len, groups.values())), "top": len(top), "control": len(controls),
               "requested_top": n, "requested_control": control,
               "strategy": "seeded_uniform" if "unknown" in groups else "domain_priority",
               "domain_stratification_deferred": "unknown" in groups,
               "seed": DRAW_SEED if "unknown" in groups else None, "domains": dict(Counter(r.domain or "unknown" for r in top)),
               "as_of": as_of.isoformat(), "denominator": len(top) + len(controls),
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def _freeze(rows: list[CensusRow], path: Path) -> None:
    """Publish a complete denominator atomically, including an empty draw."""
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def draw(*, data: DataLayout, n: int = 500, control: int = 50, out: Path | None = None) -> dict[str, Any]:
    """Freeze the priority and control denominators under the census writer lock."""
    root = Path(out) if out is not None else data.root / "sweeps" / SWEEP
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        _lock(lock.fileno())
        return _draw(data=data, n=n, control=control, out=root)
