"""Deterministic fetchers: rung 0 has the paper and license routes.

Owned by the data lane. Fetched bytes go into the store through ``store.put``; nothing
fetched is committed except its index line. Every fetched page and PDF also gets its
extracted text stored beside it, so a packet's quotes can be checked offline.

``fetch_paper`` takes an arXiv id (``1512.03385``, ``arXiv:1512.03385`` or an arxiv.org URL),
a DOI (``10.x/y``, ``doi:10.x/y`` or a doi.org URL) or any other URL. arXiv gives the abstract
page and the PDF; a DOI gives the landing page and, when OpenAlex knows an open-access copy,
its PDF; a URL gives that one document.

Rung 1 (the research lane) adds the 2a fetchers: ``fetch_code`` (a repository tree at a resolved
commit, filtered, plus its LICENSE), ``fetch_hf`` (a Hub repository's card, config, tokenizer and
code files, following ``auto_map`` into other repositories, never weights), ``fetch_page`` (one
page and its text, or a typed ``inaccessible``) and ``fetch_bib`` (the four bibliographic
records of an arXiv id or DOI).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, now
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import Source, StoreIndexLine

USER_AGENT = "menagerie-pipeline/0 (+https://modelmenagerie.ai)"
MAX_BYTES = 200 * 2**20
ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
DOI = r"10\.\d{4,9}/\S+"
MOVING_REFS = frozenset({"main", "master", "HEAD", "dev", "develop", "trunk"})


class FetchError(RuntimeError):
    """A document could not be fetched (after retries)."""


_LAST_REQUEST: dict[str, float] = {}
SPACING_S = {"arxiv.org": 3.0, "export.arxiv.org": 3.0}  # arXiv asks for one request per 3 s


def _space(url: str) -> None:
    """Wait out the per-host spacing (arXiv cuts transfers short under rapid requests)."""

    host = urllib.parse.urlsplit(url).hostname or ""
    gap = SPACING_S.get(host.removeprefix("www."))
    if gap:
        wait = _LAST_REQUEST.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST[host] = time.monotonic()


def _short(body_len: int, length: str | None, head: bytes, tail: bytes) -> bool:
    """A transfer that ended early: fewer bytes than Content-Length, or a PDF with no trailer."""

    if length and length.isdigit() and body_len < int(length):
        return True
    return head.startswith(b"%PDF") and b"%%EOF" not in tail


RETRY_DOWNLOAD = (406, 429, 500, 502, 503, 504)
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError)


def _read_to(response: Any, dest: Path) -> int:
    with dest.open("wb") as out:
        while chunk := response.read(2**20):
            out.write(chunk)
            if out.tell() > MAX_BYTES:
                raise FetchError(f"{response.geturl()}: over {MAX_BYTES} bytes")
        return out.tell()


def _ends(dest: Path, size: int) -> tuple[bytes, bytes]:
    with dest.open("rb") as written:
        head = written.read(8)
        written.seek(max(0, size - 2048))
        return head, written.read()


def _download(url: str, dest: Path, *, tries: int = 3) -> tuple[str, str]:
    """Fetch ``url`` into ``dest``; return the final URL and the Content-Type.

    A transfer that ends short (``read`` returns nothing before Content-Length is reached, which
    ``http.client`` does not raise) or a PDF without its ``%%EOF`` trailer is retried, then refused.
    """

    if urllib.parse.urlsplit(url).hostname == "api.openalex.org":
        got = _openalex(url)
        if got.status != 200:
            raise FetchError(f"OpenAlex shim: HTTP {got.status}")
        dest.write_bytes(got.body)
        return got.url, got.mime
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        _space(url)
        cause: BaseException | None = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                size = _read_to(response, dest)
                final, length = response.geturl(), response.headers.get("Content-Length")
                mime = response.headers.get("Content-Type", "")
            if not _short(size, length, *_ends(dest, size)):
                return final, mime
            problem = f"transfer ended short at {size} bytes"
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_DOWNLOAD:
                raise FetchError(f"{url}: HTTP {err.code}") from err
            problem, cause = f"HTTP {err.code}", err
        except NETWORK_ERRORS as err:
            problem, cause = str(err), err
        if attempt == tries - 1:
            raise FetchError(f"{url}: {problem}") from cause
        time.sleep(5 * (attempt + 1))
    raise FetchError(f"{url}: no response")


def _fetch_into_store(url: str, *, store: StoreLayout, data: DataLayout, text: bool = True,
                      paper: bool = False, pdf: bool = False) -> list[StoreIndexLine]:
    """Fetch one document into the store and, for PDF and HTML, its extracted text.

    A ``paper`` document is checked first (``paper_problem``): a bot wall, a challenge page or a
    PDF that does not parse raises ``Inaccessible`` (``page-inaccessible``) and saves nothing.
    """

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc"
        final, mime = _download(url, path)
        mime = mime or "application/octet-stream"
        if paper and (problem := paper_problem(path.read_bytes(), url=final, mime=mime, pdf=pdf)):
            raise Inaccessible(f"{url}: page-inaccessible ({problem})")
        if path.read_bytes()[:4] == b"%PDF":
            mime = "application/pdf"
        line = store_mod.put(path, store=store, data=data, mime=mime,
                             license_class="third-party", url=url)
    lines = [line]
    if text and mime.startswith(("application/pdf", "text/html", "application/xhtml")):
        lines.append(store_mod.extract_text(line.sha256, store=store, data=data))
    return lines


def paper_problem(body: bytes, *, url: str, mime: str, pdf: bool) -> str | None:
    """Why a downloaded paper document is not the paper, or ``None`` when it may be saved.

    A PDF must have its header, a ``startxref`` and the ``%%EOF`` trailer; anything else must
    not be a bot wall or challenge page (the page check, ``_bot_wall``); a document fetched as
    the PDF (``pdf``) must be one.
    """

    if body[:5] == b"%PDF-":
        if b"%%EOF" in body[-2048:] and b"startxref" in body[-65536:]:
            return None
        return "a PDF that does not parse: no startxref or %%EOF"
    if _bot_wall(Response(200, url, mime, body)):
        return "a bot wall or challenge page"
    return f"not a PDF ({mime or 'no type'})" if pdf else None


def _arxiv_id(ref: str) -> str | None:
    match = re.fullmatch(rf"(?:arxiv:|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?{ARXIV_ID}"
                         r"(?:\.pdf)?/?", ref.strip(), flags=re.IGNORECASE)
    return match.group(1) if match else None


def _doi(ref: str) -> str | None:
    match = re.fullmatch(rf"(?:doi:|https?://(?:dx\.)?doi\.org/)?({DOI})", ref.strip(),
                         flags=re.IGNORECASE)
    return urllib.parse.unquote(match.group(1)) if match else None


def _open_access_pdf(doi: str) -> str | None:
    """Ask OpenAlex for an open-access PDF of a DOI; ``None`` when it knows none."""

    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='/')}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _download(url, Path(tmp) / "work.json")
        except FetchError:
            return None
        work = json.loads((Path(tmp) / "work.json").read_text(encoding="utf-8"))
    for location in (work.get("best_oa_location"), *(work.get("locations") or ())):
        if location and location.get("pdf_url"):
            return location["pdf_url"]
    return None


def fetch_paper(ref: str, *, store: StoreLayout, data: DataLayout) -> list[StoreIndexLine]:
    """Fetch a paper by arXiv id, DOI or URL: abstract page, PDF and its extracted text.

    A bot wall, a challenge page or a broken PDF answered with status 200 raises ``Inaccessible``
    (``page-inaccessible``) before anything is saved; a URL fetched as the PDF must be one.
    """

    arxiv, doi = _arxiv_id(ref), _doi(ref)
    if arxiv:
        urls = [(f"https://arxiv.org/abs/{arxiv}", False), (f"https://arxiv.org/pdf/{arxiv}", True)]
    elif doi:
        pdf = _open_access_pdf(doi)
        urls = [(f"https://doi.org/{doi}", False)] + ([(pdf, True)] if pdf else [])
    elif re.match(r"https?://", ref):
        urls = [(ref, urllib.parse.urlsplit(ref).path.lower().endswith(".pdf"))]
    else:
        raise ValueError(f"{ref!r} is not an arXiv id, a DOI or a URL")
    return [line for url, pdf in urls
            for line in _fetch_into_store(url, store=store, data=data, paper=True, pdf=pdf)]


def _raw_github(url: str) -> str:
    """Turn a GitHub ``blob`` URL into its raw URL; refuse a moving branch name."""

    blob = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", url)
    if blob:
        url = "https://raw.githubusercontent.com/{}/{}/{}/{}".format(*blob.groups())
    raw = re.fullmatch(r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/([^/]+)/.+", url)
    if raw and raw.group(1) in MOVING_REFS:
        raise ValueError(f"{url} names the moving ref {raw.group(1)!r}; pin a commit or tag")
    return url


def fetch_license(url: str, *, store: StoreLayout, data: DataLayout) -> StoreIndexLine:
    """Fetch one LICENSE file (a raw URL at a pinned commit) into the store."""

    return _fetch_into_store(_raw_github(url), store=store, data=data, text=False)[0]


# --- Rung 1 research lane (2a) ------------------------------------------------------------
# Every fetcher saves what it reads into the store (one index line per blob), logs one
# ``drive.event(..., verb="fetch")`` per request against the family's budget, and returns packet
# sources whose ``sid`` is a provisional ``<role>-<sha8>`` the 2a agent may keep or rename. A
# source with no ``blob`` and a ``<role>-inaccessible`` sid is the typed "could not read it":
# a bot wall, a login or a missing repository, never an exception.

# A challenge page's words, matched in its title and visible text only: a real page may load a
# captcha script (CQVIP's abstract pages do) and is not a wall for it (fix 13).
BOT_WALL = re.compile(r"just a moment\.\.\.|captcha|attention required|access denied|"
                      r"enable javascript and cookies|are you a robot|unusual traffic|"
                      r"request unsuccessful\. incapsula|ddos-guard|perimeterx|"
                      r"checking your browser|verify you are human", re.IGNORECASE)
# A challenge widget in the markup: a wall only when the page shows little else.
CHALLENGE_MARKUP = re.compile(rb"cf-chl|challenge-form|g-recaptcha|h-captcha", re.IGNORECASE)
WALL_TEXT_CHARS = 2_000  # a challenge page's visible text is shorter than this
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
BIB_APIS = ("openalex", "s2", "crossref", "bibtex")


class Inaccessible(FetchError):
    """The server answered but would not hand the content over (login, bot wall, 403, 404)."""


@dataclass(frozen=True)
class Response:
    status: int
    url: str
    mime: str
    body: bytes


def _openalex(url: str) -> Response:
    """Use one local broker executable; never transport a key or evaluate a remote shell.

    MENAGERIE_OPENALEX_COMMAND names the executable (default ``openalex``). Network trickles
    run where the broker is installed; SSH prefixes are refused because they evaluate a shell.
    """

    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname != "api.openalex.org" or parsed.scheme != "https":
        raise FetchError("OpenAlex shim: invalid endpoint")
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if any(k.lower() in ("api_key", "key", "token") for k, _ in query):
        raise FetchError("OpenAlex shim: credential in request refused")
    command = shlex.split(os.environ.get("MENAGERIE_OPENALEX_COMMAND", "openalex"))
    if len(command) != 1:
        raise FetchError("OpenAlex shim: requires one local broker executable")
    argv = [*command, "get", parsed.path, "--query", *(f"{k}={v}" for k, v in query)]
    env = {k: v for k, v in os.environ.items() if k != "OPENALEX_API_KEY"}
    try:
        done = subprocess.run(argv, capture_output=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FetchError(f"OpenAlex shim: {type(exc).__name__}") from None
    return _openalex_response(done, url)


def _openalex_response(done: subprocess.CompletedProcess, url: str) -> Response:
    """Validate broker output while keeping its credential-boundary diagnostics private."""

    if done.returncode:
        # The shim's diagnostics are not persisted: they belong to the credential boundary.
        status = re.search(rb"openalex: HTTP (\d{3})", done.stderr)
        code = int(status.group(1)) if status else 503
        budget = b"insufficient budget" in done.stderr.lower()
        body = b'{"error":"insufficient budget"}' if budget else b'{}'
        return Response(code, url, "application/json", body)
    try:
        record = json.loads(done.stdout)
    except (ValueError, UnicodeError):
        raise FetchError("OpenAlex shim: invalid JSON") from None
    if not isinstance(record, dict) or len(done.stdout) > MAX_BYTES:
        raise FetchError("OpenAlex shim: invalid response")
    return Response(200, url, "application/json", done.stdout)


def _event(family: str, verb: str = "fetch", **fields: Any) -> None:
    """One drive event; ``fetch`` events count against the family's budget, ``bib`` ones do not."""

    from menagerie.pipeline import drive

    drive.event(os.environ.get("FLEET_RUN_ID") or "manual", family, verb, **fields)


def _attempt(request: urllib.request.Request, limit: int) -> Response:
    """One GET; any HTTP status is a response, a short body raises ``IncompleteRead``."""

    url = request.full_url
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read(limit + 1)
            length = response.headers.get("Content-Length")
            got = Response(response.status, response.geturl(),
                           response.headers.get("Content-Type", ""), body)
    except urllib.error.HTTPError as err:
        return Response(err.code, url, (err.headers or {}).get("Content-Type", ""),
                        err.read(2**16))
    if len(body) > limit:
        raise FetchError(f"{url}: over {limit} bytes")
    if _short(len(body), length, body[:8], body[-2048:]):
        raise http.client.IncompleteRead(body)
    return got


def _http(url: str, *, headers: Mapping[str, str] | None = None, tries: int = 3,
          limit: int = MAX_BYTES, retry_on: tuple[int, ...] = (429, 500, 502, 503, 504)
          ) -> Response:
    """GET ``url``; return the response for any status, retrying network faults and ``retry_on``.

    A refusal that names an exhausted budget (OpenAlex's "Insufficient budget") is not retried.
    """

    if urllib.parse.urlsplit(url).hostname == "api.openalex.org":
        return _openalex(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(tries):
        last = attempt == tries - 1
        _space(url)
        try:
            got = _attempt(request, limit)
        except NETWORK_ERRORS as err:
            if last:
                raise FetchError(f"{url}: {err!r}") from err
        else:
            if last or got.status not in retry_on or b"insufficient budget" in got.body.lower():
                return got
        time.sleep(5 * (attempt + 1))
    raise FetchError(f"{url}: no response")


def _put_bytes(body: bytes, *, store: StoreLayout, data: DataLayout, mime: str, url: str | None,
               derived_from: str | None = None) -> StoreIndexLine:
    store_mod.check_room(store, len(body))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blob"
        path.write_bytes(body)
        return store_mod.put(path, store=store, data=data, mime=mime, license_class="third-party",
                             url=url, derived_from=derived_from)


def _inaccessible(role: str, url: str) -> Source:
    return Source(sid=f"{role}-inaccessible", role=role, url=url, fetched=now())  # type: ignore[arg-type]


# --- Web pages ------------------------------------------------------------------------------


def _bot_wall(got: Response) -> bool:
    """A refusal status, or a short page whose title or visible text is a challenge."""

    if got.status in (401, 403, 407, 429, 451, 503):
        return True
    if got.status != 200 or len(got.body) >= 200_000:
        return False
    page = got.body[:65536].decode("utf-8", errors="replace")
    title = TITLE.search(page)
    shown = store_mod.visible_text(page)
    if len(shown) >= WALL_TEXT_CHARS:
        return False
    words = f"{title.group(1) if title else ''}\n{shown}"
    return bool(BOT_WALL.search(words) or CHALLENGE_MARKUP.search(got.body[:65536]))


def fetch_page(url: str, *, family: str, store: StoreLayout, data: DataLayout) -> Source:
    """Save one web page (a project page, README, blog post or PDF) and its extracted text.

    Returns: a ``page`` source with ``url``, ``blob`` and ``text``; for a bot wall, a login or a
    refusal, a ``page-inaccessible`` source with no blob (not an error). Side effects: two blobs
    and their index lines; one fetch event for ``family``.
    """

    store_mod.check_room(store)
    got = _http(url, headers={"Accept": "text/html,application/xhtml+xml,application/pdf,*/*"},
                retry_on=(500, 502, 504))
    if _bot_wall(got) or got.status != 200:
        _event(family, kind="page", url=url, outcome="inaccessible", status=got.status)
        return _inaccessible("page", url)
    mime = "application/pdf" if got.body[:4] == b"%PDF" else (got.mime or "text/html")
    line = _put_bytes(got.body, store=store, data=data, mime=mime, url=got.url)
    if mime.startswith(("application/pdf", "text/html", "application/xhtml")):
        text = store_mod.extract_text(line.sha256, store=store, data=data).sha256
    else:
        text = line.sha256 if mime.startswith("text/") else None
    _event(family, kind="page", url=url, outcome="saved", bytes=len(got.body))
    return Source(sid=f"page-{line.sha256[:8]}", role="page", url=got.url, blob=line.sha256,
                  text=text, fetched=now())


# --- Code repositories and the Hugging Face Hub (``fetch_repos``, split out at rung 2) ---------


def fetch_code(url: str, *, rev: str | None = None, family: str, store: StoreLayout,
               data: DataLayout) -> Source:
    """Save a repository tree at a resolved commit (``fetch_repos.fetch_code``)."""

    from menagerie.pipeline import fetch_repos

    return fetch_repos.fetch_code(url, rev=rev, family=family, store=store, data=data)


def fetch_hf(repo: str, *, rev: str | None = None, family: str, store: StoreLayout,
             data: DataLayout) -> list[Source]:
    """Save a Hub repository's card, config, tokenizer and code (``fetch_repos.fetch_hf``)."""

    from menagerie.pipeline import fetch_repos

    return fetch_repos.fetch_hf(repo, rev=rev, family=family, store=store, data=data)


# --- Bibliographic records ------------------------------------------------------------------


@dataclass(frozen=True)
class BibResult:
    """What ``fetch_bib`` saved, plus typed notes for each API that gave no record."""

    sources: tuple[Source, ...]
    notes: tuple[str, ...]
    records: Mapping[str, Any]


# Semantic Scholar's keyless pool answers 429 in bursts: after one call is refused through all its
# tries, the API is skipped for this long, so the next families do not each wait it out.
S2_COOLDOWN_S = 900


def _bib_cache(url: str) -> Path:
    """Today's cache file for a bibliographic answer (UTC day; a new day asks again)."""

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return work_dir("bib-cache") / day / hashlib.sha256(url.encode()).hexdigest()[:32]


def _cached(url: str) -> Response | None:
    path = _bib_cache(url)
    if not path.is_file():
        return None
    mime = path.with_suffix(".mime").read_text(encoding="utf-8") \
        if path.with_suffix(".mime").is_file() else ""
    return Response(200, url, mime, path.read_bytes())


def _keep(url: str, got: Response) -> None:
    path = _bib_cache(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".mime").write_text(got.mime, encoding="utf-8")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(got.body)
    os.replace(tmp, path)


def _cooling(api: str) -> bool:
    mark = work_dir("bib-cache") / f"{api}-cooldown"
    return api == "s2" and mark.is_file() and time.time() - mark.stat().st_mtime < S2_COOLDOWN_S


def _bib_get(api: str, url: str, *, family: str, headers: Mapping[str, str] | None = None,
             shown: str | None = None) -> tuple[Response | None, str | None]:
    """One API call; return the 200 response, or ``None`` and a typed note.

    A 200 answer is kept for the UTC day (``bib-cache/`` in the work root) and reused. OpenAlex
    missing-work and budget refusals are also kept for the day so the same query cannot loop; after
    Semantic Scholar refuses a call with 429 through all its tries, it is not asked again for
    ``S2_COOLDOWN_S``.
    """

    key = shown or url
    hit = _cached(key)
    if hit is not None:
        _event(family, "bib", api=api, url=key, outcome="cached")
        return hit, None
    refusal = _bib_cache(key).with_suffix(".note")
    if refusal.is_file():
        _event(family, "bib", api=api, url=key, outcome="cached_refusal")
        return None, refusal.read_text(encoding="utf-8")
    if _cooling(api):
        _event(family, "bib", api=api, url=key, outcome="cooldown")
        return None, f"{api}:http_429 (cooling down)"
    try:
        # Semantic Scholar's keyless refusal is optional and receives no retries.
        got = _http(url, headers=headers, tries=1 if api == "s2" else 3)
    except FetchError as err:
        _event(family, "bib", api=api, url=key, outcome="error")
        return None, f"{api}:error ({str(err)[:80]})"
    if got.status == 200:
        _keep(key, got)
        _event(family, "bib", api=api, url=key, outcome="saved")
        return got, None
    if got.status == 429 and api == "s2":
        mark = work_dir("bib-cache") / f"{api}-cooldown"
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    budget = b"insufficient budget" in got.body.lower()
    note = f"{api}:refused (insufficient budget)" if budget else (
        f"{api}:not_found" if got.status == 404 else f"{api}:http_{got.status}")
    _event(family, "bib", api=api, url=key, outcome=note.split(":", 1)[1],
           status=got.status)
    if api == "openalex" and (got.status == 404 or budget):
        refusal.parent.mkdir(parents=True, exist_ok=True)
        refusal.write_text(note, encoding="utf-8")
    return None, note


@dataclass
class _Bib:
    """What one ``bib_records`` call has saved so far."""

    family: str
    store: StoreLayout
    data: DataLayout
    sources: list[Source] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    records: dict[str, Any] = field(default_factory=dict)

    def call(self, api: str, url: str, *, mime: str = "application/json",
             headers: Mapping[str, str] | None = None, shown: str | None = None) -> None:
        """Ask one API; save its answer as a ``bib`` source, or note why there is none."""

        got, note = _bib_get(api, url, family=self.family, headers=headers, shown=shown)
        if got is not None and mime == "application/x-bibtex" and b"@" not in got.body[:200]:
            got, note = None, f"{api}:not_bibtex"
        if got is None:
            self.notes.append(note or f"{api}:no_record")
            return
        line = _put_bytes(got.body, store=self.store, data=self.data, mime=mime, url=shown or url)
        self.sources.append(Source(sid=f"bib-{api}-{line.sha256[:8]}", role="bib", url=shown or url,
                                   api=api, blob=line.sha256, fetched=now()))  # type: ignore[arg-type]
        if mime == "application/json":
            self.records[api] = json.loads(got.body)


def s2_headers() -> dict[str, str]:
    """Semantic Scholar's key header when ``S2_API_KEY`` is set; keyless (shared pool) until then."""

    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def bib_records(ref: str, *, family: str, store: StoreLayout, data: DataLayout) -> BibResult:
    """Fetch and save the four bibliographic records of an arXiv id or DOI (see ``fetch_bib``)."""

    store_mod.check_room(store)
    arxiv, doi = _arxiv_id(ref), _doi(ref)
    if not (arxiv or doi):
        raise ValueError(f"{ref!r} is not an arXiv id or a DOI")
    bib = _Bib(family, store, data)
    s2_key = urllib.parse.quote(f"ARXIV:{arxiv}" if arxiv else f"DOI:{doi}", safe=":/")
    s2 = f"https://api.semanticscholar.org/graph/v1/paper/{s2_key}"
    work = f"10.48550/arXiv.{arxiv}" if arxiv else doi or ""
    shown = f"https://api.openalex.org/works/doi:{urllib.parse.quote(work, safe='/')}"
    bib.call("openalex", shown)
    if doi:
        bib.call("crossref", f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}")
    bib.call("bibtex", f"https://arxiv.org/bibtex/{arxiv}" if arxiv else
             f"https://doi.org/{urllib.parse.quote(doi or '', safe='/')}",
             mime="application/x-bibtex", headers=None if arxiv else {"Accept": "application/x-bibtex"})
    bib.call("s2", f"{s2}?fields=paperId,externalIds,title,year,venue,publicationVenue,"
             "publicationDate,journal,authors,citationCount,referenceCount,abstract", shown=s2,
             headers=None)  # Keyless, trailing, one attempt; never needed for agreement.
    published_doi = ((bib.records.get("s2") or {}).get("externalIds") or {}).get("DOI")
    if not doi and published_doi:
        bib.call("crossref", f"https://api.crossref.org/works/{urllib.parse.quote(published_doi, safe='/')}")
    elif not doi:
        bib.notes.append("crossref:no_doi")
    notes = tuple(f"bib {ref}: {n.replace(':', ' ', 1)}" for n in bib.notes)
    return BibResult(tuple(bib.sources), notes, bib.records)


def fetch_bib(ref: str, *, family: str, store: StoreLayout, data: DataLayout) -> list[Source]:
    """Save the bibliographic records of a paper from every API that answers.

    Inputs: an arXiv id or DOI (``packet write`` runs this for every citation identifier).
    Saved: the keyless Semantic Scholar paper, the OpenAlex work through the broker command
    (no credential or email is written), the Crossref work (by the DOI, or the published DOI Semantic
    Scholar names for an arXiv paper) and the BibTeX (arXiv's export, or ``doi.org`` content
    negotiation). Returns: one ``bib`` source per API that answered, each with ``api``, ``url``,
    ``blob`` and ``fetched``. An API that refuses (OpenAlex's "Insufficient budget") or has no
    record gives no source; its typed note is in the fetch event and, through ``packet write``,
    in the packet's ``search.note``. Side effects: blobs, index lines, one event per API call.
    """

    return list(bib_records(ref, family=family, store=store, data=data).sources)
