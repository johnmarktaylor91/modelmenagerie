"""C-22 -- every producer-written string reaching a dispatch or an identity is closed.

The catalog names this the highest-value missing tripwire (``FAILURE_PATTERN_CATALOG``
C-22, process rule P-01): ``device_policy`` was found and fixed as an INSTANCE, and the
identical class was still live one field over in ``run_framework`` -- free text, consumed
verbatim as a machine dispatch key AND baked into the execution identity, feeding six
drifting closed maps (AREA_BACKENDS_opus B1). Sweeping the instance again would have left
the next one.

This module is the CLASS tripwire, in the shape of
``test_anti_substitution_inventories.py``: a closed inventory that fails when a new member
appears. Four mechanical rules:

* **Rule A (closure).** Every inventoried field declared ``closed`` must be constrained to a
  closed vocabulary at its exact JSON path -- an ``enum``, a ``const``, or a ``pattern``. A
  ``nonempty_string`` fails. The path matters: ``implementation.run_framework`` is a
  dispatch key and must be closed, while ``external_metadata.original_framework`` is a
  world-fact claim about where an architecture came from (the live pilot carries
  ``"Brian"``) and closing THAT would make an honest fact unrecordable.
* **Rule A2 (code-only closure is disclosed and cannot grow).** Two fields cannot be closed
  by schema without making immutable history unloadable, and each says why in the
  inventory. That set is committed as data and set-compared here, so a third field cannot
  join it silently -- which is the only thing separating a disclosed residual from the
  tripwire-disarming exemption the validation-integrity principle forbids.
* **Rule B (completeness).** Every author-gated schema leaf whose name is read inside the
  arguments of a closed set of dispatch/identity sinks must be an inventory row. A new free
  text field wired into a sink fails here, naming file, line and sink. This rule found
  ``recipe_type``, ``recipe_revision`` and ``device_policy`` on its first run.
* **Rule C (executability).** Every row names a live validator, and that validator must
  actually REFUSE a free-text sample. A row may not claim a closure no function enforces.

The author-gated leaf set is DERIVED from the ownership-annotated schemas
(``schema.author_gated_schema_paths``), never hand-listed, so a newly authored field is in
scope the moment it is declared.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.backend import (
    CODE_ONLY_CLOSURE_FIELDS,
    DISPATCH_AND_IDENTITY_SINKS,
    PRODUCER_DISPATCH_INVENTORY,
    ProducerDispatchField,
    SchemaClosure,
    producer_dispatch_validator,
)
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION_V3,
)
from menagerie.crawler.schema import author_gated_schema_paths

_CRAWLER_ROOT = Path(driver_module.__file__).parent
_SCHEMA_ROOT = _CRAWLER_ROOT / "schemas"
_INVENTORY_JSON = Path(__file__).parent / "producer_dispatch_inventory.json"

_SCHEMA_FILES = {
    "model-common.schema.json",
    "model-v3.schema.json",
    "author-proposal-common.schema.json",
    "author-proposal-v3.schema.json",
}

_SCHEMA_ENTRYPOINTS = {
    MODEL_SCHEMA_VERSION_V3: "model-v3.schema.json",
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3: "author-proposal-v3.schema.json",
}


@dataclass(frozen=True)
class _SinkRead:
    """One producer-field read inside a dispatch/identity sink call."""

    field: str
    sink: str
    location: str


def _load_schema_file(filename: str) -> Mapping[str, Any]:
    """Load one raw schema document from the packaged schema directory."""

    payload = json.loads((_SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping)
    return payload


def _resolve_ref(current_file: str, ref: str) -> tuple[str, Mapping[str, Any]]:
    """Resolve a local or sibling-file ``$defs`` reference.

    The v3 schemas share ``model-common``/``author-proposal-common``, so a leaf's real
    constraint frequently lives one file over. A census that followed only local refs would
    read every shared def as unconstrained and pass vacuously.
    """

    filename, _, pointer = ref.partition("#/")
    target_file = filename or current_file
    assert target_file in _SCHEMA_FILES, f"unknown schema in ref: {ref}"
    node: Any = _load_schema_file(target_file)
    for part in pointer.split("/"):
        assert isinstance(node, Mapping), f"unresolvable ref {ref}"
        node = node[part]
    assert isinstance(node, Mapping), f"unresolvable ref {ref}"
    return target_file, node


def _follow(current_file: str, node: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Follow ``$ref`` chains until a concrete definition is reached."""

    seen = 0
    while isinstance(node.get("$ref"), str):
        current_file, node = _resolve_ref(current_file, str(node["$ref"]))
        seen += 1
        assert seen < 32, "cyclic $ref chain"
    return current_file, node


def _definition_at(schema_version: str, path: str) -> Optional[Mapping[str, Any]]:
    """Return the property definition at one normalized ``$.a.b[].c`` schema path."""

    current_file = _SCHEMA_ENTRYPOINTS[schema_version]
    current_file, node = _follow(current_file, _load_schema_file(current_file))
    for segment in path.removeprefix("$.").split("."):
        name = segment.removesuffix("[]")
        for candidate in (node, *(node.get("allOf") or ())):
            if not isinstance(candidate, Mapping):
                continue
            current_candidate_file, resolved = _follow(current_file, candidate)
            properties = resolved.get("properties")
            if isinstance(properties, Mapping) and name in properties:
                current_file = current_candidate_file
                node = properties[name]
                break
        else:
            return None
        current_file, node = _follow(current_file, node)
        if segment.endswith("[]"):
            items = node.get("items")
            if not isinstance(items, Mapping):
                return None
            current_file, node = _follow(current_file, items)
    return node


def _is_closed_vocabulary(definition: Mapping[str, Any]) -> bool:
    """Return whether one resolved definition admits only a closed vocabulary.

    ``enum`` and ``const`` are closed by listing. A ``pattern`` is closed too, and
    deliberately so: an identity input such as ``recipe_revision`` cannot be an enum, but a
    fixed grammar still means the accepted set only widens through a reviewed change, which
    is exactly what C-22 requires. A bare ``nonempty_string`` is not closed.
    """

    if isinstance(definition.get("enum"), list) or "const" in definition:
        return True
    if isinstance(definition.get("pattern"), str):
        return True
    for combinator in ("oneOf", "anyOf", "allOf"):
        options = definition.get(combinator)
        if isinstance(options, list) and options:
            branches = [
                option
                for option in options
                if isinstance(option, Mapping) and option.get("type") != "null"
            ]
            if branches and all(
                _is_closed_vocabulary(_follow("model-common.schema.json", branch)[1])
                for branch in branches
            ):
                return True
    return False


def _string_subscript_fields(node: ast.AST) -> Iterator[str]:
    """Yield every string-literal subscript key read anywhere under ``node``."""

    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        key = child.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            yield key.value


def _callee_name(node: ast.Call) -> str:
    """Return the bare callee name for a simple or attribute call."""

    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _sink_reads() -> tuple[_SinkRead, ...]:
    """Census every producer-field subscript read inside a sink call's arguments."""

    reads: list[_SinkRead] = []
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a parse failure is a real defect
            raise AssertionError(f"{path} does not parse: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            sink = _callee_name(node)
            if sink not in DISPATCH_AND_IDENTITY_SINKS:
                continue
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                for field in _string_subscript_fields(argument):
                    reads.append(
                        _SinkRead(
                            field=field,
                            sink=sink,
                            location=f"{path.relative_to(_CRAWLER_ROOT.parent)}:{node.lineno}",
                        )
                    )
    return tuple(reads)


def _author_gated_leaf_names() -> frozenset[str]:
    """Return the terminal names of every author-gated schema leaf."""

    names: set[str] = set()
    for version in (MODEL_SCHEMA_VERSION_V3, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3):
        for leaf in author_gated_schema_paths(version):
            names.add(leaf.removesuffix("[]").rsplit(".", 1)[-1])
    return frozenset(names)


@pytest.mark.smoke
def test_the_inventory_is_a_closed_typed_record() -> None:
    """Rule 0: the inventory is typed, non-empty, duplicate-free, and path-exact."""

    assert PRODUCER_DISPATCH_INVENTORY, "the producer-dispatch inventory cannot be empty"
    assert all(isinstance(row, ProducerDispatchField) for row in PRODUCER_DISPATCH_INVENTORY)
    names = [row.field for row in PRODUCER_DISPATCH_INVENTORY]
    assert len(names) == len(set(names)), f"duplicate inventory rows: {sorted(names)}"
    # run_framework is the finding that motivated the census; without it the file is
    # decorative.
    assert "run_framework" in set(names)
    for row in PRODUCER_DISPATCH_INVENTORY:
        assert row.schema_paths, f"{row.field} names no schema path"
        assert row.sinks, f"{row.field} names no sink"
        assert row.finding, f"{row.field} names no audit finding"
        for version, path in row.schema_paths:
            assert path in author_gated_schema_paths(version), (
                f"{path} is not an author-gated leaf of {version}; the inventory is "
                "describing a field producers do not write"
            )


@pytest.mark.smoke
def test_every_closed_field_is_closed_by_its_own_schema() -> None:
    """Rule A: a dispatch/identity path declared closed is closed BY SCHEMA."""

    open_paths: list[str] = []
    for row in PRODUCER_DISPATCH_INVENTORY:
        if row.schema_closure is not SchemaClosure.CLOSED:
            continue
        for version, path in row.schema_paths:
            definition = _definition_at(version, path)
            assert definition is not None, f"{version}:{path} does not resolve in the schema"
            if not _is_closed_vocabulary(definition):
                open_paths.append(f"{version}:{path}")
    assert not open_paths, (
        "producer fields reaching a dispatch or an identity are still free text: "
        f"{sorted(open_paths)}"
    )


@pytest.mark.smoke
def test_the_code_only_closure_set_is_disclosed_and_cannot_grow() -> None:
    """Rule A2: exactly two fields are closed in code only, and each says why."""

    code_only = {
        row.field
        for row in PRODUCER_DISPATCH_INVENTORY
        if row.schema_closure is SchemaClosure.CODE_ONLY
    }
    assert code_only == set(CODE_ONLY_CLOSURE_FIELDS), (
        "the code-only closure set changed; widening it is a reviewed change: "
        f"{sorted(code_only ^ set(CODE_ONLY_CLOSURE_FIELDS))}"
    )
    for row in PRODUCER_DISPATCH_INVENTORY:
        if row.schema_closure is SchemaClosure.CODE_ONLY:
            assert len(row.closure_note) > 80, (
                f"{row.field} claims code-only closure without explaining why the schema "
                "cannot close it"
            )
        else:
            assert row.closure_note == ""


@pytest.mark.smoke
def test_the_inventory_is_complete_over_every_sink_call() -> None:
    """Rule B: no author-gated field reaches a sink without an inventory row."""

    inventoried = {row.field for row in PRODUCER_DISPATCH_INVENTORY}
    authored = _author_gated_leaf_names()
    escapes = [
        f"{read.field} -> {read.sink} at {read.location}"
        for read in _sink_reads()
        if read.field in authored and read.field not in inventoried
    ]
    assert not escapes, (
        "author-gated fields reach a dispatch/identity sink with no closed-vocabulary "
        f"binding: {sorted(escapes)}"
    )


@pytest.mark.smoke
def test_every_inventory_row_names_a_validator_that_refuses_free_text() -> None:
    """Rule C: the claimed closure is executable, not documentary."""

    for row in PRODUCER_DISPATCH_INVENTORY:
        validator = producer_dispatch_validator(row.field)
        assert callable(validator), f"{row.field} names no live validator"
        for sample in (row.free_text_sample, "", "  "):
            with pytest.raises(row.refusal_type):
                validator(sample)
        assert validator(row.canonical_sample) is not None


@pytest.mark.smoke
def test_the_census_would_have_caught_the_b1_instance() -> None:
    """The tripwire is anchored to the finding it exists to prevent.

    ``run_framework`` reaches ``derive_execution_identity`` in
    ``driver_admission._execution_identity``. Rule B must SEE that read; if the sink set or
    the subscript scan ever stops covering it, this census silently stops working and every
    rule above passes vacuously.
    """

    reads = {(read.field, read.sink) for read in _sink_reads()}
    assert ("run_framework", "derive_execution_identity") in reads, (
        "the census no longer sees run_framework flowing into the execution identity; "
        "the sink inventory or the AST scan has drifted"
    )
    assert ("device_policy", "parse_device_policy") in reads


@pytest.mark.smoke
def test_the_world_fact_placements_are_deliberately_left_open() -> None:
    """The closure is placement-exact, and over-closing would destroy honest facts.

    ``external_metadata.original_framework`` records where an architecture CAME FROM. The
    live pilot carries ``"Brian"`` (a spiking-network simulator) there. It steers no
    dispatch and enters no identity, and an enum would make that fact unrecordable -- so
    the census must not demand closure for it, and must say so out loud.
    """

    inventoried_paths = {
        path for row in PRODUCER_DISPATCH_INVENTORY for _version, path in row.schema_paths
    }
    for path in (
        "$.external_metadata.original_framework",
        "$.external_metadata.run_framework",
    ):
        assert path not in inventoried_paths
    reads = {read.field for read in _sink_reads()}
    # The dispatch reads are all `implementation[...]`; if a world-fact placement ever
    # starts steering a dispatch, Rule B catches it by name and this test's premise is
    # gone.
    assert "run_framework" in reads


@pytest.mark.smoke
def test_the_dispatch_inventory_is_disclosed_as_data() -> None:
    """The inventory is reviewable as JSON, not only as Python."""

    payload = json.loads(_INVENTORY_JSON.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "menagerie.crawler.producer-dispatch-inventory.v1"
    assert {row["field"] for row in payload["fields"]} == {
        row.field for row in PRODUCER_DISPATCH_INVENTORY
    }
    assert set(payload["sinks"]) == set(DISPATCH_AND_IDENTITY_SINKS)
    assert set(payload["code_only_closure"]) == set(CODE_ONLY_CLOSURE_FIELDS)
    by_field = {row["field"]: row for row in payload["fields"]}
    for row in PRODUCER_DISPATCH_INVENTORY:
        disclosed = by_field[row.field]
        assert disclosed["validator"] == row.validator
        assert disclosed["schema_closure"] == row.schema_closure.value
        assert disclosed["schema_paths"] == [
            {"schema_version": version, "path": path} for version, path in row.schema_paths
        ]
