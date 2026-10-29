"""Every empirically ranked gate-repair class must keep its rule in the author prompt.

The 2026-08-12 mini-rung cycled eight award-candidate models through metadata
repair rounds 16-19 on a small set of RECURRING, KNOWN fact-hygiene classes --
each round costing an author session plus a checker session. Aggregating the
pilot gate ledger (188 gate rows: 517 ``required_repairs`` plus 852
non-``accurate`` ``field_checks``, 1369 findings in all) ranks those classes:

======================================  ========  ======
repair class                            findings  models
======================================  ========  ======
input-contract-completeness                  173      47
website-clause-grounding                     123      43
description-outruns-evidence                 119      42
country-institution-affiliation              111      31
original-framework-provenance                103      28
lineage-predecessors                          86      28
venue-year-era                                81      23
citation-grounding                            80      28
rung-blocked-discipline                       61      21
taxonomy-tags-grounding                       58      22
modes-train-eval                              46      22
novel-ops-executed-impl                       45      20
license-grounding                             28      16
excerpt-locator-hygiene                       23       4  (+40 integrity failures)
supports-tag-vocabulary                       17      13
======================================  ========  ======

The author prompt was hardened with one crisp rule per class so authors get
these right on the FIRST pass. This module is the tripwire that keeps them
there: each class is bound to distinctive anchor phrases of its rule, so a
future prompt edit that drops or rewords a rule away fails here instead of
silently reopening a repair class the ledger already paid for. Anchors are
matched against the whitespace-normalized prompt, exactly like the sibling
``test_author_contract_discoverability`` checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.metadata import AVAILABILITY_KEYS

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_AUTHOR_PROMPT = _CRAWLER_ROOT / "prompts" / "claude_crawler_author_v2.txt"
_STAGE2_PROMPT = _CRAWLER_ROOT / "prompts" / "executor" / "stage2_author.md"
_POOL_STAGE_PROMPT = _CRAWLER_ROOT / "prompts" / "pool" / "stage_author.md"

#: Ranked repair class -> the anchor phrases its prompt rule must keep carrying.
#: Every phrase is matched verbatim against the whitespace-normalized canonical
#: author prompt. Rewording a rule is allowed only by updating the anchor in the
#: same reviewed change -- never by letting the rule silently vanish.
RANKED_REPAIR_CLASS_ANCHORS: dict[str, tuple[str, ...]] = {
    "input-contract-completeness": (
        "source-unspecified nonmaterial probe choice",
        # Leg-4 2026-08-12 (m9430): the previous anchor phrase told authors to
        # "make input_contract's non_tensor_values represent every literal
        # constructor kwarg", and the worker faithfully assembled those into
        # forward(). The rule now states the opposite, machine-enforced at
        # check time by _validate_constructor_keyword_collisions.
        "Constructor keywords are construction facts",
        "terminal path field matches a declared_choices field",
        # Tier-0 accept-rate ruling (2026-09-01, clause 2): every contract leaf
        # is a checked claim, and a minimum-resolution claim is refused outright.
        'Never assert a minimum or "smallest valid" resolution',
    ),
    "website-clause-grounding": ("covered by a website-tagged excerpt",),
    "description-outruns-evidence": (
        "WHICH BOUND EXCERPT STATES THIS?",
        "outran its evidence",
        # Accept-rate ruling clause 3: one unbound variant detail fails the whole
        # field, and the source's wording is never strengthened.
        "cannot-verify for the WHOLE field",
        "Do not strengthen the source's wording",
    ),
    "country-institution-affiliation": (
        "search the frozen paper bytes for the affiliation lines",
        "a not-found over affiliation bytes you hold is itself a false claim",
        # Accept-rate replay negative example: a country read off a co-author's
        # second affiliation.
        "second, unrelated affiliation",
    ),
    "original-framework-provenance": (
        "never the encoder's or backbone's",
        "NO key in the availability register",
    ),
    "lineage-predecessors": (
        "needs its own bound excerpt",
        # Accept-rate ruling clause 1: only an explicit derivation relation in
        # the source's own words grounds lineage; family classification does not.
        "explicit derivation relation",
        "REFUSED as lineage",
    ),
    "venue-year-era": (
        "Year and era are FIRST-PUBLIC",
        "arXiv v1 stamp",
        "Venue is grounded on the paper page or conference front matter",
        # Accept-rate replay negative example: a venue read off a workshop page.
        "workshop or event page",
    ),
    "citation-grounding": (
        "EVERY name in authors[] must occur inside a bound paper-role excerpt",
        "populate citation.bibtex",
    ),
    "rung-blocked-discipline": (
        "An import statement in source code proves only that the code imports the name",
        "needs-higher-tier",
        "The recipe must CONSTRUCT the variant you claim",
    ),
    "taxonomy-tags-grounding": (
        "a class name or general knowledge of the family grounds nothing",
        # Accept-rate ruling clause 4: every tag ELEMENT is bound on its own,
        # tags have no relevance exception, one unbound element fails the array.
        "Tags do not get the keywords relevance exception",
        "An array with one unbound element fails whole",
    ),
    # Accept-rate ruling clause 5: paradigm and architecture_class are grounded
    # on literal statements, not on results or class names.
    "paradigm-literal-evidence": ("literally states supervised training or labeled data",),
    "architecture-class-defining-operator": ("showing the defining operator",),
    # Accept-rate ruling clause 6 (rides item 13's routed_environment envelope
    # field): installed-package claims follow inventory_disclosed, and license
    # excerpts bind at the recipe's own pinned revision.
    "routed-environment-claims": (
        "routed_environment.inventory_disclosed is true",
        "SAME pinned revision as the selected recipe",
    ),
    "modes-train-eval": ("STRUCTURAL train/eval divergence",),
    "novel-ops-executed-impl": ("freeze and cite the implementation module that actually runs",),
    "license-grounding": ("a registry sidebar or badge is not license evidence",),
    "excerpt-locator-hygiene": (
        "Locator drift",
        "re-verify every locator",
        "U+00A0",
        "table ROW is never contiguous",
        "re-extract every excerpt back OUT of your own facts file",
    ),
    "supports-tag-vocabulary": ("EXACT STRING EQUALITY",),
    # Step-100 triage ranks 3 and 4 (2026-08-12): both rules are
    # machine-enforced at check time (proposal.parse_device_policy /
    # proposal._validate_train_mode_probe_batch); the prompt states them so
    # authors stop paying a burned session to learn each one.
    "device-policy-prose": (
        "LITERAL torch device string",
        "put placement narrative in notes",
    ),
    "bn-train-batch1": (
        "the probe batch dim MUST be >= 2",
        "Expected more than 1 value per channel when training",
    ),
    # Step-100 fresh-terminal triage (2026-08-12), ranked rows 7/8/9: three
    # AUTHORING classes whose rules landed with their machine gates (worker
    # pre-forward signature check, pre-publication version gate, offline-pinning
    # proposal seam) in the same change.
    "forward-kwarg-omitted": (
        "Enumerate EVERY required forward() parameter",
        "ERModel.forward(h, r, t, *, mode)",
    ),
    "lib-version-drift": (
        "that row's installed version verbatim",
        "vendor and adapt that layer too",
    ),
    "offline-pinning": (
        "documented offline switches",
        "sandbox-denied DNS attempt",
    ),
}


def _normalized(path: Path) -> str:
    """Return a prompt's text with all whitespace runs collapsed to one space.

    Parameters
    ----------
    path:
        Prompt file to read.

    Returns
    -------
    str
        Whitespace-normalized prompt text, so anchors match across line wraps.
    """

    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.mark.smoke
@pytest.mark.parametrize("repair_class", sorted(RANKED_REPAIR_CLASS_ANCHORS))
def test_each_ranked_repair_class_keeps_its_rule_in_the_author_prompt(
    repair_class: str,
) -> None:
    """The canonical prompt states the rule for each ledger-ranked repair class."""

    prompt = _normalized(_AUTHOR_PROMPT)
    for anchor in RANKED_REPAIR_CLASS_ANCHORS[repair_class]:
        assert anchor in prompt, (
            f"repair class {repair_class!r} lost its prompt rule: the anchor "
            f"{anchor!r} is no longer in {_AUTHOR_PROMPT.name}. Update the rule "
            "and this anchor together in one reviewed change; never drop the rule."
        )


@pytest.mark.smoke
def test_the_prompt_names_every_enforced_availability_key() -> None:
    """The prompt's availability-key list stays in lockstep with the register.

    The prompt shipped for a week listing twelve keys while the enforcing
    register (``menagerie.crawler.metadata.AVAILABILITY_KEYS``) held thirteen:
    ``taxonomy.era`` was added to the schema and never surfaced where the author
    reads, so a truthful ``taxonomy.era`` absence had no discoverable spelling.
    """

    prompt = _normalized(_AUTHOR_PROMPT)
    for key in sorted(AVAILABILITY_KEYS):
        assert key in prompt, (
            f"availability key {key!r} is enforced but not named in the author "
            "prompt; a truthful absence for it has no discoverable spelling"
        )


@pytest.mark.smoke
def test_the_dispatch_briefs_mirror_the_clause_sweep() -> None:
    """Both author dispatch briefs point at the canonical clause sweep.

    The executor stage brief drives every production author session and the pool
    fragment drives the operator-mediated lane; each must carry the sweep
    reminder (the largest repair class) and the executor brief must also carry
    the locator-drift trap beside its excerpt-hash traps.
    """

    stage2 = _normalized(_STAGE2_PROMPT)
    pool = _normalized(_POOL_STAGE_PROMPT)
    for brief, text in (("stage2_author.md", stage2), ("stage_author.md", pool)):
        assert "clause sweep" in text, f"{brief} lost the clause-sweep mirror"
        assert "Narrow or delete what you cannot bind" in text, (
            f"{brief} lost the sweep's remedy sentence"
        )
    assert "The locator dies the same way the digest does" in stage2, (
        "stage2_author.md lost the locator-drift trap"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("brief", ("stage1_discovery.md", "stage2_author.md"))
def test_both_stage_briefs_state_the_non_disclosed_inventory_rule(brief: str) -> None:
    """Accept-rate ruling clause 6 rides the routed_environment block both briefs carry.

    Item 13 gave stage 1 the SAME ``routed_environment`` object stage 2 gets, so
    each brief must say what a non-disclosed inventory forbids (an installed-
    package claim), and stage 2 must bind license excerpts at the recipe's own
    pinned revision.
    """

    text = _normalized(_CRAWLER_ROOT / "prompts" / "executor" / brief)
    assert "never state that a package is installed" in text, (
        f"{brief} lost the non-disclosed-inventory rule"
    )
    if brief == "stage2_author.md":
        assert "SAME pinned revision as the selected recipe" in text, (
            "stage2_author.md lost the license-revision binding"
        )
