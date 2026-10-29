"""``menagerie-pipeline`` (or ``python -m menagerie.pipeline``): the pipeline's verbs.

Each verb parses its arguments and calls one named function in the module that owns the
work, so lanes change their modules, never this file. The hub-side verbs (fetch, store, family,
packet, research, src, giveup, done, implement, bridge, review, prose, next, lanes, measures,
collect, and rung 2's stage3 queue and collect, catalog plan, harvest tier, axon batch, submit and
collect, jev drive and checkin; rung 3's jev freeze, tick, runner ensure, paper, describe, census,
residual plan and repin flip; the speed-up build's verbs) load neither torch nor psutil. Results
print as JSON lines. Roots come from the environment: ``MENAGERIE_DATA`` (committed data),
``MENAGERIE_STORE`` (blobs) and ``MENAGERIE_ENVS`` (built environments).
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, read_json
from menagerie.pipeline.records import ModelRecord, Record, RunRequest, RunResult, dumps
from menagerie.pipeline.stubs import NOT_BUILT, NotBuilt

Handler = Callable[[argparse.Namespace], Any]


def _mod(name: str) -> Any:
    # Owning modules load per verb, so `--help` and the torch-free verbs never import torch.
    return importlib.import_module(f"menagerie.pipeline.{name}")


def _worker(host: str | None) -> dict[str, Any]:
    """A 2b launch's placement: nothing for the hub, else the tick config's worker entry, whose
    trace-host checks then read the saved tick config and state (``mini_placement.launch``)."""
    if host is None:
        return {}
    workers = _mod("tick").config().get("mini_workers") or {}
    if host not in workers:
        raise SystemExit(f"{host} is not in the tick config's mini_workers")
    return {"host": host, "worker": workers[host]}


def _exclusion(value: str) -> tuple[str, str]:
    """Parse one ID=REASON option; a value without '=' is a usage error."""

    sid, sep, reason = value.partition("=")
    if not sep or not sid.strip():
        raise argparse.ArgumentTypeError(f"--exclude needs ID=REASON, got {value!r}")
    return sid.strip(), reason.strip()


def _paper_decision(value: str) -> dict[str, Any]:
    """Load a bounded decision object and report malformed files as command usage errors."""

    try:
        with Path(value).open(encoding="utf-8") as handle:
            text = handle.read(2**20 + 1)
        if len(text) > 2**20:
            raise ValueError("decision exceeds 1 MiB")
        decision = json.loads(text)
        if not isinstance(decision, dict):
            raise ValueError("decision must be an object")
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid paper decision: {exc}") from None
    return decision


def _envs_root(args: argparse.Namespace) -> Path:
    root = args.root or os.environ.get("MENAGERIE_ENVS")
    if not root:
        raise SystemExit("set MENAGERIE_ENVS or pass --root")
    return Path(root)


def _emit(value: Any) -> None:
    if isinstance(value, Record):
        print(dumps(value))
    elif isinstance(value, (list, tuple)) and value and all(isinstance(v, Record) for v in value):
        for item in value:
            print(dumps(item))
    elif value is not None:
        print(json.dumps(value, default=str, ensure_ascii=False))


def _print_lines(names: list[str]) -> None:
    """One name per line, for shell loops (``runner_job.sh`` reads ``envs missing`` this way)."""

    for name in names:
        print(name)


def _run_model(args: argparse.Namespace, mode: str) -> RunResult:
    runner = _mod("runner")
    data = DataLayout.from_env()
    record = read_json(ModelRecord, data.model(args.id))
    request = runner.request_for(
        record, mode,
        intervention_ready=not getattr(args, "no_intervention_ready", False),
        model_mode="train" if getattr(args, "train", False) else "eval",
        trace=getattr(args, "trace_sha", None), extras=tuple(getattr(args, "with_", ()) or ()),
    )
    result = runner.run(request, spool=args.spool)
    if args.append and result.rows:
        _mod("ledger").append(result.rows, data=data)
    return result


def _try(args: argparse.Namespace) -> RunResult:
    return _mod("implement").try_(
        args.id, recipe=args.recipe, member=args.member, extras=tuple(args.with_ or ()),
        spool=args.spool, append=args.append, data=DataLayout.from_env())


def _trace(args: argparse.Namespace) -> RunResult:
    return _run_model(args, "trace-structure-only" if args.structure_only else "trace")


def _run_request(args: argparse.Namespace) -> RunResult:
    result = _mod("runner").run(read_json(RunRequest, args.request), spool=args.spool)
    if args.append and result.rows:
        _mod("ledger").append(result.rows, data=DataLayout.from_env())
    return result


HANDLERS: dict[str, Handler] = {
    "roster repair": lambda a: _mod("roster").repair(
        data=DataLayout.from_env(), sc_dir=a.sc_dir, write=a.write),
    "ledger import-legacy": lambda a: _mod("ledger").import_legacy(
        a.manifest or DataLayout.from_env().routing_manifest, data=DataLayout.from_env()),
    "status": lambda a: _mod("ledger").status_counts(data=DataLayout.from_env()),
    "store put": lambda a: _mod("store").put(
        a.path, store=StoreLayout.from_env(), data=DataLayout.from_env(), mime=a.mime,
        license_class=a.license_class, url=a.url, derived_from=a.derived_from),
    "store get": lambda a: str(_mod("store").get(a.sha256, store=StoreLayout.from_env())),
    "store extract-text": lambda a: _mod("store").extract_text(
        a.sha256, store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "store mirror": lambda a: _mod("store").mirror(store=StoreLayout.from_env(), dest=a.dest),
    "store restore": lambda a: str(_mod("store").restore(
        a.sha256, store=StoreLayout.from_env(), source=a.source)),
    "fetch paper": lambda a: _mod("fetch").fetch_paper(
        a.ref, store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "fetch license": lambda a: _mod("fetch").fetch_license(
        a.url, store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "envs build": lambda a: str(_mod("envs").build(a.name, root=_envs_root(a))),
    "envs canary": lambda a: _mod("envs").canary(a.name, root=_envs_root(a)),
    "envs list": lambda a: _mod("envs").inventory(root=_envs_root(a)),
    "envs missing": lambda a: _print_lines(_mod("envs").missing(root=_envs_root(a))),
    "envs prune-overlays": lambda a: _mod("envs").prune_overlays(
        _envs_root(a), keep_days=a.keep_days, cap_gib=a.cap_gib,
        keep_for=[Path(p) for pattern in a.keep_for for p in sorted(glob.glob(pattern))]),
    "harvest zoo": lambda a: _mod("harvest").harvest(
        a.zoo, a.names or None, data=DataLayout.from_env(), write=not a.dry_run),
    "runner serve": lambda a: _mod("runner").serve(spool=a.spool, workers=a.workers, once=a.once),
    "run": _run_request,
    "try": _try,
    "trace": _trace,
    "render": lambda a: _run_model(a, "render"),
    "worker": lambda a: _mod("worker").main(a.attempt_dir),
    "publish export": lambda a: str(_mod("publish").export_site(
        data=DataLayout.from_env(), out_dir=a.out_dir, ids=a.ids or None, as_of=a.as_of)),
}


def _fetcher(name: str) -> Handler:
    def handler(a: argparse.Namespace) -> Any:
        kwargs = {"rev": a.rev} if hasattr(a, "rev") else {}
        return getattr(_mod("fetch"), name)(a.target, family=a.family, **kwargs,
                                            store=StoreLayout.from_env(),
                                            data=DataLayout.from_env())
    return handler


# Rung 1 verbs, each dispatching to its owning lane's module.
HANDLERS.update({
    "fetch code": _fetcher("fetch_code"),
    "fetch hf": _fetcher("fetch_hf"),
    "fetch page": _fetcher("fetch_page"),
    "fetch bib": _fetcher("fetch_bib"),
    "store drain": lambda a: _mod("store").drain(
        tuple(a.sha or ()), store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "store materialize": lambda a: str(_mod("store").materialize(
        a.sha256, a.dest, store=StoreLayout.from_env())),
    "family seed": lambda a: _mod("research").seed_family(a.ids, data=DataLayout.from_env()),
    "packet write": lambda a: _mod("research").packet_write(
        a.family, a.file, store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "packet bib-fill": lambda a: _mod("packet").bib_fill(
        a.family, store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "packet licenses": lambda a: _mod("packet").relicense(
        a.families or None, store=StoreLayout.from_env(), data=DataLayout.from_env(),
        write=not a.dry_run),
    "packet bib-harvest": lambda a: [o.to_json() for o in _mod("bib_harvest").fill_all(
        store=StoreLayout.from_env(), data=DataLayout.from_env(), families=a.families or None,
        write=not a.dry_run)],
    "packet june-notes": lambda a: _mod("bib_harvest").june_notes(
        data=DataLayout.from_env(), write=not a.dry_run),
    "packet domains": lambda a: _mod("bib_harvest").set_domains(
        dict(line.split("\t")[:2] for line in a.file.read_text().splitlines()
             if line.strip() and not line.startswith("#")), data=DataLayout.from_env()),
    "packet spot-check": lambda a: _mod("bib_harvest").confirm(
        list(csv.DictReader(a.file.open(encoding="utf-8"), delimiter="\t")),
        data=DataLayout.from_env()),
    "packet check": lambda a: _mod("citecheck").check_packets(data=DataLayout.from_env()),
    "store audit-documents": lambda a: _mod("citecheck").audit_documents(
        store=StoreLayout.from_env(), data=DataLayout.from_env()),
    "research launch": lambda a: _mod("research").launch(
        a.batch_file, data=DataLayout.from_env(), role=a.role),
    "research second-search": lambda a: _mod("research").second_search(
        a.family, data=DataLayout.from_env()),
    "src": lambda a: _mod("implement").src(
        a.family, data=DataLayout.from_env(), store=StoreLayout.from_env()),
    "giveup": lambda a: _mod("implement").giveup(
        a.id, a.cls, a.reason, data=DataLayout.from_env(), recipe=a.recipe),
    "done": lambda a: _mod("implement").done(a.recipe_file, data=DataLayout.from_env(),
                                             rewrite_family=a.rewrite_family),
    "implement launch": lambda a: _mod("implement").launch(
        a.family, role=a.role, data=DataLayout.from_env(), tier=a.tier, **_worker(a.host)),
    "implement library-adapter": lambda a: _mod("implement").library_adapter(
        a.zoo, a.ids, data=DataLayout.from_env(), write=a.write),
    "implement sdist-wheel": lambda a: str(_mod("sdist").build(
        a.archive, name=a.env, root=a.root, interpreter=a.python,
        distribution=a.distribution, version=a.version)),
    "bridge serve": lambda a: _mod("implement").bridge(spool=a.spool),
    "review pick": lambda a: _mod("review").pick_reviewer(a.run_id, data=DataLayout.from_env()),
    "review calibrate": lambda a: _mod("review").calibrate(
        role=a.role, data=DataLayout.from_env(), runs=a.runs),
    "review launch": lambda a: _mod("review").launch(a.ids, data=DataLayout.from_env()),
    "review record": lambda a: _mod("review").record(
        a.id, a.file, data=DataLayout.from_env(), store=StoreLayout.from_env()),
    "prose audit": lambda a: _mod("describe").audit_prose(a.ids, data=DataLayout.from_env()),
    "next": lambda a: _mod("drive").next_steps(a.ids, data=DataLayout.from_env()),
    "lanes": lambda a: _mod("drive").lanes(data=DataLayout.from_env()),
    "measures": lambda a: _mod("drive").measures(a.run_id, data=DataLayout.from_env()),
    "collect": lambda a: _mod("drive").collect(data=DataLayout.from_env()),
    "queue": lambda a: _mod("drive").queued(),
    "requeue": lambda a: _mod("drive").requeue(),
})

# Rung 2 verbs (BUILD-PLAN r2 section 2), each dispatching to its owning lane's module.
HANDLERS.update({
    "stage3 queue": lambda a: _mod("stage3").queue(
        None if a.committed else a.lane, data=DataLayout.from_env(),
        ids=(a.lane, *a.ids) if a.committed else tuple(a.ids), spool=a.spool,
        timeout_s=a.timeout_s, memory_gib=a.memory_gib, again=a.again),
    "stage3 serve": lambda a: _mod("stage3").serve(spool=a.spool, workers=a.workers, once=a.once),
    "stage3 collect": lambda a: _mod("stage3").collect(data=DataLayout.from_env(),
                                                        spool=a.spool),
    "catalog plan": lambda a: _mod("harvest_tiers").catalog_plan(
        data=DataLayout.from_env(), skip=a.skip, write=not a.dry_run),
    "harvest tier": lambda a: _mod("harvest_tiers").harvest_tier(
        a.zoo, a.names or None, data=DataLayout.from_env(), write=not a.dry_run),
    "jev drive": lambda a: _mod("describe").jev_drive(a.packets, data=DataLayout.from_env(),
                                                     retry=a.retry),
    "jev freeze": lambda a: _mod("describe").jev_freeze(a.packets, data=DataLayout.from_env()),
    "axon batch": lambda a: str(_mod("cluster").axon_batch(
        a.kind, a.ids, data=DataLayout.from_env(), out=a.out)),
    "axon submit": lambda a: _mod("cluster").submit(a.batch, dry_run=a.dry_run),
    "axon run": lambda a: _mod("cluster").run(a.batch),
    "axon collect": lambda a: _mod("cluster").collect(a.batch, data=DataLayout.from_env()),
    "hash harness": lambda a: _mod("hash_harness").harness(
        a.ids, data=DataLayout.from_env(), out=a.out),
    "checkin": lambda a: str(_mod("drive").checkin(a.run_id, data=DataLayout.from_env(),
                                                   out=a.out)),
    "standin sweep": lambda a: _mod("standin").sweep(data=DataLayout.from_env(), apply=a.apply),
})

# Rung 3 verbs (BUILD-PLAN r3 section 1, item 3), each dispatching to its owning lane's module.
HANDLERS.update({
    "tick": lambda a: _mod("tick").tick(a.mode, dry_run=a.dry_run, data=DataLayout.from_env()),
    "runner ensure": lambda a: _mod("runner_job").ensure(spool=a.spool, dry_run=a.dry_run),
    "paper find": lambda a: _mod("find_paper").find(
        a.families or None, data=DataLayout.from_env(), store=StoreLayout.from_env(),
        write=not a.dry_run, from_dossier=a.from_dossier),
    "paper harvest": lambda a: _mod("find_paper").harvest(
        a.families or None, data=DataLayout.from_env(), store=StoreLayout.from_env(),
        limit=a.limit, write=not a.dry_run),
    "paper apply": lambda a: _mod("find_paper").apply(
        a.family, a.decision,
        data=DataLayout.from_env(), store=StoreLayout.from_env()),
    "describe drive": lambda a: _mod("describe_jev").drive(a.families, data=DataLayout.from_env()),
    "describe write": lambda a: _mod("describe_write").write(
        a.families or None, data=DataLayout.from_env(), store=StoreLayout.from_env(),
        role=a.role, write=not a.dry_run),
    "describe check": lambda a: _mod("describe_write").check(
        a.families or None, data=DataLayout.from_env(), store=StoreLayout.from_env()),
    "census run": lambda a: _mod("census").run(data=DataLayout.from_env(), out=a.out,
                                               limit=a.limit),
    "census draw": lambda a: _mod("census").draw(data=DataLayout.from_env(), n=a.n,
                                                 control=a.control, out=a.out),
    "residual plan": lambda a: _mod("residual").plan(data=DataLayout.from_env(), rows=a.rows,
                                                     out=a.out),
    "repin check": lambda a: _mod("repin").check(
        a.version, data=DataLayout.from_env(), out=a.out, part=a.part, ids=a.ids,
        smoke=a.smoke, deadline=a.deadline, explanations=a.explanations),
    "repin flip": lambda a: _mod("repin").flip(a.version, data=DataLayout.from_env(),
                                               dry_run=a.dry_run, check_dir=a.check_dir, go=a.go, spool=a.spool, c1_status=a.c1_status),
    "release build": lambda a: _mod("dataset").build(data=DataLayout.from_env(), out=a.out,
                                                     dry_run=a.dry_run),
    "site count": lambda a: _mod("dataset").site_count(a.build_dir, out=a.out),
})

# Speed-up verbs (speed-up BUILD-PLAN section 1, item 1), each dispatching to its owning lane's
# module; until a lane fills its body the verb prints "not built yet" and exits ``NOT_BUILT``.
HANDLERS.update({
    "implement launch-batch": lambda a: _mod("implement").launch_batch(
        a.families, role=a.role, data=DataLayout.from_env(), **_worker(a.host)),
    "census harvest": lambda a: _mod("census_harvest").harvest(
        data=DataLayout.from_env(), store=StoreLayout.from_env(), out=a.out, limit=a.limit,
        write=not a.dry_run),
    "research dossier": lambda a: _mod("dossier").build(
        a.ids or None, data=DataLayout.from_env(), store=StoreLayout.from_env(),
        harvest=a.harvest, write=not a.dry_run),
    "tick tripwires": lambda a: _mod("tick_full").tripwires(
        DataLayout.from_env(), _mod("tick").config(), dry_run=a.dry_run, resume=a.resume),
    "worklist build": lambda a: _mod("worklist_build").build(
        data=DataLayout.from_env(), rows=a.rows, draw=a.draw, out=a.out),
    "describe seats": lambda a: _mod("describe_write").seats(data=DataLayout.from_env()),
    "counts build": lambda a: _mod("counts").build(data=DataLayout.from_env(), out=a.out,
                                                   commit=a.commit, worker=a.worker),
    "replay table": lambda a: _mod("replay_table").table(
        data=DataLayout.from_env(), store=StoreLayout.from_env(), pin=a.pin, out=a.out,
        exclude=dict(a.exclude)),
    "runner big": lambda a: _mod("runner_job").big(dry_run=a.dry_run),
    "resolve run": lambda a: _mod("resolve").run(
        a.ids or None, data=DataLayout.from_env(), out=a.out, limit=a.limit),
    "resolve confirm": lambda a: _mod("resolve_confirm").confirm(
        data=DataLayout.from_env(), candidates=a.candidates, role=a.role),
})


def _review_calibrate(a: argparse.Namespace) -> Any:
    if a.all:
        return _mod("review").calibrate_all(data=DataLayout.from_env(), runs=a.runs)
    return _mod("review").calibrate(role=a.role, data=DataLayout.from_env(), runs=a.runs)


HANDLERS["review calibrate"] = _review_calibrate


def _store_trim(a: argparse.Namespace) -> Any:
    return _mod("store_trim").from_args(a, store=StoreLayout.from_env(), data=DataLayout.from_env())


HANDLERS["store offload"] = _store_trim
HANDLERS["store pack"] = _store_trim
HANDLERS["store pack-views"] = _store_trim
HANDLERS["store ceiling"] = lambda a: _mod("store_ceiling").from_args(
    a, store=StoreLayout.from_env(), data=DataLayout.from_env())


def _store_fetch(a: argparse.Namespace) -> Any:
    remote = a.remote or os.environ.get("MENAGERIE_STORE_REMOTE")
    if not remote:
        raise SystemExit("store fetch needs --remote HOST:PATH (the worker store) or "
                         "MENAGERIE_STORE_REMOTE")
    report = _mod("store_box").restore_blobs(a.sha256, remote=remote,
                                             data=DataLayout.from_env().root,
                                             desk=Path.home() / "desk")
    if report["failed"]:
        raise SystemExit(f"not restored: {report['failed']}")
    return report


HANDLERS["store fetch"] = _store_fetch


def _verb(sub: Any, name: str, help_text: str) -> argparse.ArgumentParser:
    parser = sub.add_parser(name.split()[-1], help=help_text)
    parser.set_defaults(verb=name)
    return parser


def _group(sub: Any, name: str, help_text: str) -> Any:
    """A verb group's ACTION subparsers; a later builder reopens a group an earlier one added."""

    if name in sub.choices:
        reopened = getattr(sub.choices[name], "actions_group", None)
        if reopened is None:
            raise ValueError(f"{name!r} is a verb, not a verb group")
        return reopened
    parser = sub.add_parser(name, help=help_text)
    parser.actions_group = parser.add_subparsers(required=True, metavar="ACTION")
    return parser.actions_group


def _add_ceiling_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-gib", type=float,
                        help="offload until the store's allocated bytes are at most this (20)")
    parser.add_argument("--min-age-days", type=float,
                        help="never offload a blob placed more recently than this (3)")
    parser.add_argument("--working-set", action="append", default=[], metavar="GLOB",
                        help="files (queued and running requests) whose named shas stay")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spool", type=Path, help="submit through this runner spool and wait")
    parser.add_argument("--append", action="store_true", help="append result rows to the ledger")


def _add_data_verbs(sub: Any) -> None:
    roster = _group(sub, "roster", "roster provenance")
    p = _verb(roster, "roster repair", "stamp provenance and hints on every roster row")
    p.add_argument("--sc-dir", type=Path, help="directory of the S-C discovery TSVs")
    p.add_argument("--write", action="store_true", help="rewrite the roster (default: counts)")
    ledger = _group(sub, "ledger", "the append-only ledger")
    p = _verb(ledger, "ledger import-legacy", "load routing_manifest.tsv as the 2026-06 shard")
    p.add_argument("--manifest", type=Path)
    _verb(sub, "status", "count models by state and failure class")
    store = _group(sub, "store", "the content-addressed store")
    p = _verb(store, "store put", "add a file or .tlspec directory to the store")
    p.add_argument("path", type=Path)
    p.add_argument("--mime", required=True)
    p.add_argument("--license-class", required=True)
    p.add_argument("--url")
    p.add_argument("--derived-from")
    for name, text in (("store get", "print a blob's path"),
                       ("store extract-text", "store a PDF or HTML blob's text")):
        _verb(store, name, text).add_argument("sha256")
    p = _verb(store, "store mirror", "copy new blobs to the mirror")
    p.add_argument("--dest", required=True)
    p = _verb(store, "store restore", "fetch a blob back from the mirror and verify it")
    p.add_argument("sha256")
    p.add_argument("--source", required=True)
    _verb(store, "store audit-documents", "check every saved PDF and page for bot walls")
    p = _verb(store, "store drain", "copy blobs to the worker store")
    p.add_argument("--sha", nargs="+", metavar="SHA256", help="these blobs (default: all missing)")
    p = _verb(store, "store materialize", "unpack a stored code snapshot")
    p.add_argument("sha256")
    p.add_argument("dest", type=Path)
    p = _verb(store, "store fetch", "hub: bring blobs only the box holds back into the worker store")
    p.add_argument("sha256", nargs="+")
    p.add_argument("--remote", help="the worker store as host:path (default MENAGERIE_STORE_REMOTE)")
    for name, text in (("store offload", "drop local copies a verified box snapshot holds"),
                       ("store pack", "pack trace saves as tar.gz behind a verified snapshot"),
                       ("store pack-views", "gzip view JSON, SVG and HTML behind a verified snapshot"),
                       ("store ceiling", "offload the oldest box-verified blobs down to a size")):
        p = _verb(store, name, text)
        if name == "store offload":
            p.add_argument("policy", choices=("superseded", "tiles", "portable-rendered",
                                              "portable-twin"))
        if name == "store pack-views":
            p.set_defaults(policy="pack-views")
        if name == "store ceiling":
            _add_ceiling_options(p)
        p.add_argument("--catalog", type=Path, required=True,
                       help="folder of fleet-storage catalog manifests (JSON)")
        p.add_argument("--host", required=True, help="the catalog host name of this store")
        p.add_argument("--links", type=Path, action="append", default=[],
                       help="a box job root whose sealed batches hard-link store blobs")
        p.add_argument("--extra", type=Path, action="append", default=[],
                       help="more committed folders whose shas are kept (the site export)")
        p.add_argument("--apply", action="store_true", help="act (default: report only)")
        p.add_argument("--limit", type=int)
        p.add_argument("--lines", type=Path, help="index lines out (default: the placement shard)")
        p.add_argument("--report", type=Path, help="write every item here as JSON")
    fetch = _group(sub, "fetch", "deterministic fetchers")
    _verb(fetch, "fetch paper", "abstract page, PDF and text").add_argument("ref")
    _verb(fetch, "fetch license", "a LICENSE file at a pinned URL").add_argument("url")
    for name, metavar, text, pinned in (
            ("fetch code", "URL", "a repository tree at a resolved commit", True),
            ("fetch hf", "REPO", "a Hub repository's card, config and code", True),
            ("fetch page", "URL", "one web page and its text", False),
            ("fetch bib", "REF", "bibliographic records from every API", False)):
        p = _verb(fetch, name, text)
        p.add_argument("target", metavar=metavar)
        p.add_argument("--family", required=True, metavar="F")
        if pinned:
            p.add_argument("--rev", help="commit, tag or branch (default: the default branch)")


def _add_lane_verbs(sub: Any) -> None:
    """The rung 1 agent-lane verbs: research (2a), implement (2b), review and drive."""

    family = _group(sub, "family", "family membership")
    _verb(family, "family seed", "propose a family from seeds").add_argument("ids", nargs="+")
    packet = _group(sub, "packet", "family packets")
    p = _verb(packet, "packet write", "check a draft packet and write it")
    p.add_argument("family")
    p.add_argument("file", type=Path)
    _verb(packet, "packet bib-fill", "fetch the bibliographic records a packet left pending"
          ).add_argument("family")
    p = _verb(packet, "packet licenses", "re-resolve derived license sources by each tree's own copy")
    p.add_argument("families", nargs="*")
    p.add_argument("--dry-run", action="store_true", help="list what would change; write nothing")
    p = _verb(packet, "packet bib-harvest", "fill harvester citations from records that agree")
    p.add_argument("families", nargs="*")
    p.add_argument("--dry-run", action="store_true", help="fetch and report; write nothing")
    _verb(packet, "packet june-notes", "carry June catalog notes into harvester packets"
          ).add_argument("--dry-run", action="store_true")
    _verb(packet, "packet domains", "set interim roster domains from a TSV (id, domain)"
          ).add_argument("file", type=Path)
    _verb(packet, "packet spot-check", "release held fills a TSV of verdicts finds right"
          ).add_argument("file", type=Path)
    _verb(packet, "packet check", "citation, SPDX and duplicate checks over every packet")
    research = _group(sub, "research", "2a lanes")
    p = _verb(research, "research launch", "start 2a lanes for batches")
    p.add_argument("batch_file", type=Path)
    p.add_argument("--role", help="launch every batch on this role (default: each batch's)")
    _verb(research, "research second-search", "the second code search on the other lab"
          ).add_argument("family")
    _verb(sub, "src", "a family's saved sources with paths").add_argument("family")
    p = _verb(sub, "giveup", "record a typed give-up")
    p.add_argument("id")
    p.add_argument("cls", metavar="CLASS")
    p.add_argument("reason")
    p.add_argument("--recipe", type=Path,
                   help="the draft to keep: required for oversized, kept for needs_gpu")
    p = _verb(sub, "done", "accept a passing recipe")
    p.add_argument("recipe_file", type=Path)
    p.add_argument("--rewrite-family", action="store_true",
                   help="allow changing the adapter the family's prior members run; re-run them")
    implement = _group(sub, "implement", "2b lanes")
    p = _verb(implement, "implement library-adapter", "draft audited library residual recipes")
    p.add_argument("zoo")
    p.add_argument("ids", nargs="+")
    p.add_argument("--write", action="store_true", help="write records; never queue traces")
    p = _verb(implement, "implement sdist-wheel", "build a saved source archive offline")
    p.add_argument("archive", type=Path)
    p.add_argument("--distribution", required=True, help="independently verified package name")
    p.add_argument("--version", required=True, help="independently verified release version")
    p.add_argument("--env", default="torch-main")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True, help="preprovisioned build interpreter")
    p = _verb(implement, "implement launch", "start a 2b lane for a family")
    p.add_argument("family")
    p.add_argument("--role", required=True)
    p.add_argument("--tier", help="the plan's case for the family, for measures")
    p.add_argument("--host", help="a worker in the tick config's mini_workers (default: the hub)")
    bridge = _group(sub, "bridge", "the hub-to-runner bridge")
    _verb(bridge, "bridge serve", "forward spool files to the runner").add_argument(
        "--spool", type=Path)
    review = _group(sub, "review", "fidelity review")
    _verb(review, "review pick", "the reviewer role for a 2b lane").add_argument("run_id")
    p = _verb(review, "review calibrate", "score a role on the calibration cases")
    seats = p.add_mutually_exclusive_group(required=True)
    seats.add_argument("--role")
    seats.add_argument("--all", action="store_true", help="every reviewer seat")
    p.add_argument("--runs", type=int, default=3, help="runs at one prompt hash")
    _verb(review, "review launch", "start review lanes").add_argument("ids", nargs="+")
    p = _verb(review, "review record", "record a review verdict")
    p.add_argument("id")
    p.add_argument("file", type=Path)
    prose = _group(sub, "prose", "published prose")
    _verb(prose, "prose audit", "check prose against packets").add_argument("ids", nargs="*")
    _verb(sub, "next", "each model's next step").add_argument("ids", nargs="*")
    _verb(sub, "lanes", "live lanes and leases")
    _verb(sub, "measures", "minutes, tokens, fetches and tries of a lane").add_argument("run_id")
    _verb(sub, "collect", "commit finished lanes' data files")
    _verb(sub, "queue", "steps no lab admitted, waiting for requeue")
    _verb(sub, "requeue", "re-read admission and start the queued steps it now admits")
    jev = _group(sub, "jev", "batch metadata checks")
    p = _verb(jev, "jev drive", "the jev identity test drive")
    p.add_argument("--packets", type=int, default=50)
    p.add_argument("--retry", action="store_true",
                   help="the rung 3 retry: the packets and cutoff in data/review/jev-retry.json")
    _verb(jev, "jev freeze", "freeze the rung 3 retry's packets and quote cutoff").add_argument(
        "--packets", type=int, default=50)
    p = _verb(sub, "checkin", "write a rung's check-in")
    p.add_argument("run_id")
    p.add_argument("--out", type=Path, required=True)
    standins = _group(sub, "standin", "stand-in recipes")
    p = _verb(standins, "standin sweep", "list stand-in recipes not typed yet")
    p.add_argument("--apply", action="store_true", help="type each one (ledger row and record)")


def _add_rung2_run_verbs(sub: Any) -> None:
    """Rung 2's stage-3, catalog, cluster and hash verbs."""

    stage3 = _group(sub, "stage3", "stage 3 on the worker")
    p = _verb(stage3, "stage3 queue", "queue stage 3 for a finished 2b lane")
    p.add_argument("lane", metavar="RUN_ID", help="the lane; with --committed, the first id")
    p.add_argument("ids", nargs="*")
    p.add_argument("--committed", action="store_true",
                   help="queue committed records by id (no lane; refused while a lane holds one)")
    p.add_argument("--spool", help="runner spool (default MENAGERIE_RUNNER_SPOOL)")
    p.add_argument("--timeout-s", type=int, help="this request's trace timeout")
    p.add_argument("--memory-gib", type=float, help="this request's trace memory cap")
    p.add_argument("--again", action="store_true",
                   help="queue members already traced at their revision and the TorchLens pin")
    p = _verb(stage3, "stage3 serve", "serve the stage-3 spool in the runner job")
    p.add_argument("--spool", type=Path, required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--once", action="store_true", help="drain the spool, then exit")
    p = _verb(stage3, "stage3 collect", "append finished stage-3 rows on main")
    p.add_argument("--spool", help="runner spool (default MENAGERIE_RUNNER_SPOOL)")
    catalog = _group(sub, "catalog", "the catalog pass")
    p = _verb(catalog, "catalog plan", "the catalog pass's ordered plan by group and wave")
    p.add_argument("--skip", type=Path, help="TSV of rows the pass leaves out (id column)")
    p.add_argument("--dry-run", action="store_true")
    axon = _group(sub, "axon", "AXON batches")
    p = _verb(axon, "axon batch", "write an axon-submit batch")
    p.add_argument("--kind", choices=("cpu", "gpu", "reviewed"), required=True)
    p.add_argument("ids", nargs="+")
    p.add_argument("--out", type=Path)
    p = _verb(axon, "axon submit", "copy a batch to the cluster and submit it")
    p.add_argument("batch", type=Path)
    p.add_argument("--dry-run", action="store_true", help="copy, print the plan, submit nothing")
    _verb(axon, "axon run", "run a batch inside its SLURM job").add_argument("batch", type=Path)
    _verb(axon, "axon collect", "bring a finished batch back").add_argument("batch", type=Path)
    hashes = _group(sub, "hash", "the hash family")
    p = _verb(hashes, "hash harness", "check the hash family over the harness models")
    p.add_argument("ids", nargs="*")
    p.add_argument("--out", type=Path)


def _families_verb(group: Any, name: str, text: str, *, dry_run: bool = True
                   ) -> argparse.ArgumentParser:
    p = _verb(group, name, text)
    p.add_argument("families", nargs="*", help="these families (default: every one due)")
    if dry_run:
        p.add_argument("--dry-run", action="store_true", help="report; write nothing")
    return p


def _add_rung3_verbs(sub: Any, runner: Any) -> None:
    """Rung 3's driver, paper, describe, census, residual, re-pin and release verbs."""

    p = _verb(sub, "tick", "one driver pass (collect rows, keep the runner alive, top up lanes)")
    p.add_argument("--mode", choices=("collect", "full", "refill"), default=None,
                   help="full also tops up lanes (after the go); refill (the refill timer) only "
                        "tops up 2b; default: the tick folder's mode file, else collect")
    p.add_argument("--dry-run", action="store_true", help="print the pass; change nothing")
    p = _verb(runner, "runner ensure", "restart a dead runner job; never two on one spool")
    p.add_argument("--spool", help="runner spool (default MENAGERIE_RUNNER_SPOOL)")
    p.add_argument("--dry-run", action="store_true")
    paper = _group(sub, "paper", "paper ids and bibliographic records")
    _families_verb(paper, "paper find", "find paper ids from saved text, arXiv and OpenAlex").add_argument(
        "--from-dossier", action="store_true", help="only agent-confirmed dossier papers")
    _families_verb(paper, "paper harvest", "harvest records for new ids").add_argument(
        "--limit", type=int, help="at most this many papers (the tick trickle)")
    apply = _verb(paper, "paper apply", "apply a paper-only lane's decision as an amendment")
    apply.add_argument("family")
    apply.add_argument("decision", type=_paper_decision)
    describe = _group(sub, "describe", "describe records")
    _verb(describe, "describe drive", "the jev test drive per vocabulary").add_argument(
        "--families", type=int, default=50)
    _families_verb(describe, "describe write", "describe records from saved text").add_argument(
        "--role", help="the prose role (default: the plan's)")
    _families_verb(describe, "describe check", "grounding, vocabulary and lineage checks",
                   dry_run=False)
    census = _group(sub, "census", "the GitHub PyTorch census")
    p = _verb(census, "census run", "read each GitHub-hinted c2 repository (resumable)")
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p = _verb(census, "census draw", "draw the families and the control")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--control", type=int, default=50)
    p.add_argument("--out", type=Path)
    residual = _group(sub, "residual", "the library residual")
    p = _verb(residual, "residual plan", "work lists by class from the frozen rows")
    p.add_argument("--rows", type=Path, help="the frozen rows TSV")
    p.add_argument("--out", type=Path)
    repin = _group(sub, "repin", "moving the TorchLens pin")
    p = _verb(repin, "repin check", "check a TorchLens release on the worker")
    p.add_argument("version")
    p.add_argument("--out", type=Path)
    p.add_argument("--part", choices=("day", "slice-a", "all"), default="all")
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--deadline", type=float, help="stop before this epoch time")
    p.add_argument("--explanations", type=Path, help="operator-supplied difference evidence")
    p = _verb(repin, "repin flip", "merge a checked pin and queue the re-pin rows")
    p.add_argument("version")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-dir", type=Path)
    p.add_argument("--c1-status", type=Path, help="independent catalog collector status")
    p.add_argument("--go", type=Path, help="explicit drain and operator approval record")
    p.add_argument("--spool", type=Path)
    release = _group(sub, "release", "the local release candidate")
    p = _verb(release, "release build", "write dataset/v1 locally (no upload)")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    site = _group(sub, "site", "the site build")
    p = _verb(site, "site count", "count a full-size build against the file and asset caps")
    p.add_argument("build_dir", type=Path)
    p.add_argument("--out", type=Path)


def _add_speedup_verbs(sub: Any, runner: Any) -> None:
    """The speed-up build's verbs (speed-up BUILD-PLAN section 1, item 1), in their lanes' groups."""

    p = _verb(_group(sub, "implement", "2b lanes"), "implement launch-batch",
              "start one 2b lane for a batch of families")
    p.add_argument("families", nargs="+")
    p.add_argument("--role", required=True)
    p.add_argument("--host", help="a worker in the tick config's mini_workers (default: the hub)")
    p = _verb(_group(sub, "census", "the GitHub PyTorch census"), "census harvest",
              "README, citation and license files of the alive census repositories")
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p = _verb(_group(sub, "research", "2a lanes"), "research dossier",
              "scripted dossiers for harvested census rows")
    p.add_argument("ids", nargs="*")
    p.add_argument("--harvest", type=Path, help="the harvest file (default: the census's)")
    p.add_argument("--dry-run", action="store_true")
    # `tick` stays a verb on its own; `tick tripwires` is an optional action under it.
    tick = sub.choices["tick"].add_subparsers(required=False, metavar="ACTION")
    p = _verb(tick, "tick tripwires", "stop launching a cohort whose early yield trips")
    p.add_argument("--resume", metavar="COHORT", help="start a stopped cohort again")
    p.add_argument("--dry-run", action="store_true")
    p = _verb(_group(sub, "worklist", "the tick's worklist"), "worklist build",
              "write worklist.txt as <id> [cohort] lines")
    p.add_argument("--rows", type=Path, required=True, help="the rung 3 rows TSV")
    p.add_argument("--draw", type=Path, help="a census draw file (the next cohort)")
    p.add_argument("--out", type=Path)
    _verb(_group(sub, "describe", "describe records"), "describe seats",
          "live describe seats and their leases")
    p = _verb(_group(sub, "counts", "counts on the worker"), "counts build",
              "next steps and status counts on the science-compute worker")
    p.add_argument("--out", type=Path)
    p.add_argument("--commit", help="the hub commit the worker job shipped (recorded in the file)")
    p.add_argument("--worker", action="store_true",
                   help="from the hub: run on the worker through run-on and install the files")
    p = _verb(_group(sub, "replay", "replay-typed traces"), "replay table",
              "the failing op of each replay-typed trace")
    p.add_argument("--pin", help="the TorchLens pin (default: the current one)")
    p.add_argument("--out", type=Path)
    p.add_argument("--exclude", action="append", default=[], type=_exclusion,
                   metavar="ID=REASON",
                   help="report a cohort id with no re-trace as excluded, with this reason")
    _verb(runner, "runner big", "the night big-row runner's pass").add_argument(
        "--dry-run", action="store_true")
    resolve = _group(sub, "resolve", "the 2a router (hints only, never a type)")
    p = _verb(resolve, "resolve run", "search candidate sources for name-only rows (resumable)")
    p.add_argument("ids", nargs="*")
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p = _verb(resolve, "resolve confirm", "confirm lanes over the router's candidates")
    p.add_argument("--candidates", type=Path)
    p.add_argument("--role", help="the confirm role (default: the plan's)")


def _add_run_verbs(sub: Any) -> Any:
    envs = _group(sub, "envs", "built environments (root: MENAGERIE_ENVS or --root)")
    for name, text in (("envs build", "build from the committed lock"),
                       ("envs canary", "import-and-forward canary")):
        p = _verb(envs, name, text)
        p.add_argument("name")
        p.add_argument("--root")
    _verb(envs, "envs list", "built environments and lock ids").add_argument("--root")
    _verb(envs, "envs missing", "committed environments not built at their lock").add_argument(
        "--root")
    p = _verb(envs, "envs prune-overlays", "remove overlays unused for --keep-days, then the "
              "least recently used above --cap-gib")
    p.add_argument("--root")
    p.add_argument("--keep-days", type=float, default=3.0)
    p.add_argument("--cap-gib", type=float, default=8.0)
    p.add_argument("--keep-for", action="append", default=[], metavar="GLOB",
                   help="spool request files (queued or running) whose overlays stay")
    harvest = _group(sub, "harvest", "packets and model records for library rows")
    for name, text in (("harvest zoo", "a zoo's rows, or rung0"),
                       ("harvest tier", "a zoo's catalog rows from library defaults")):
        p = _verb(harvest, name, text)
        p.add_argument("zoo")
        p.add_argument("names", nargs="*")
        p.add_argument("--dry-run", action="store_true")
    runner = _group(sub, "runner", "the spool runner")
    p = _verb(runner, "runner serve", "serve a spool")
    p.add_argument("--spool", type=Path, required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--once", action="store_true", help="drain the spool, then exit")
    p = _verb(sub, "run", "run one request file")
    p.add_argument("request", type=Path)
    _add_run_options(p)
    p = _verb(sub, "try", "forward in eval, then train on a fresh model")
    p.add_argument("id")
    p.add_argument("--recipe", type=Path, metavar="FILE",
                   help="a draft recipe instead of the record's")
    p.add_argument("--member", metavar="ID", help="the family member whose variant to build")
    p.add_argument("--with", dest="with_", action="append", metavar="PKG")
    _add_run_options(p)
    p = _verb(sub, "trace", "capture, validate, summarize and save one model")
    p.add_argument("id")
    p.add_argument("--structure-only", action="store_true", help="meta-device structure capture")
    p.add_argument("--no-intervention-ready", action="store_true", help="the ablation capture")
    p.add_argument("--train", action="store_true", help="trace the train-mode variant")
    _add_run_options(p)
    p = _verb(sub, "render", "render the artifact set from a saved trace")
    p.add_argument("id")
    p.add_argument("--trace", dest="trace_sha", required=True, help="sha256 of the .tlspec")
    _add_run_options(p)
    _verb(sub, "worker", "in-sandbox entry used by the runner").add_argument(
        "attempt_dir", type=Path)
    publish = _group(sub, "publish", "the site export")
    p = _verb(publish, "publish export", "write the site export")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--as-of", help="measurement date for the published counts")
    return runner


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for every verb."""

    parser = argparse.ArgumentParser(prog="menagerie-pipeline", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True, metavar="VERB")
    _add_data_verbs(sub)
    runner = _add_run_verbs(sub)
    _add_lane_verbs(sub)
    _add_rung2_run_verbs(sub)
    _add_rung3_verbs(sub, runner)
    _add_speedup_verbs(sub, runner)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, call the verb's function and print what it returns."""

    args = build_parser().parse_args(argv)
    try:
        value = HANDLERS[args.verb](args)
    except NotBuilt as exc:
        print(f"menagerie-pipeline: {exc}", file=sys.stderr)
        return NOT_BUILT
    if args.verb == "worker":
        return int(value)
    _emit(value)
    if isinstance(value, RunResult):
        return 0 if value.outcome == "pass" else 1
    return 0 if value is not False else 1


if __name__ == "__main__":
    sys.exit(main())
