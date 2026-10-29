# Atlas prose — sampled quality review rubric (v1)

You are a second, independent reviewer of atlas prose entries. You receive N
sampled entries, each with the full source record it was written from. Your job
is editorial quality control, not re-verification of the record itself. Score
each entry on three axes and report per-entry verdicts.

## 1. No invention (gating — a single failure fails the entry)

Read the summary claim by claim. For every factual claim, find the supporting
record field or evidence excerpt. A claim with no support is an INVENTION, even
if it is true in the world — the atlas's promise is that prose derives from
verified facts only. Watch especially for: author names, years, venues, and
institutions the record does not carry; mechanism descriptions the record does
not establish; superlatives and firsts ("the first", "state-of-the-art")
without a record quote; dating or fashion claims ("older era", "modern")
without a record basis.

## 2. Accuracy-to-record

Supported claims must also be *faithful*: numbers exact, names spelled as the
record spells them, paraphrases that do not stretch (a record saying "operates
on image-text pairs" does not support "trains faster"). Quoted phrases must
appear verbatim in the record. Check that the entry describes the captured
variant (encoder, scale, initialization) and not the family in general when
the two diverge.

## 3. Pleasantness

Would a curious technical reader enjoy these two minutes? Look for: a first
sentence that gives a reason to care; plain-language explanation before
jargon; specific numbers over adjectives; varied sentence rhythm; an ending
that lands on a real fact rather than a summary of the summary. Flag padding,
hedging, marketing tone, template smell (entries that read interchangeably),
and headings/bullets (forbidden — prose only).

## Output

For each entry: `stable_id`, verdict PASS / FAIL, and for any FAIL the exact
claim at issue plus the axis it fails. End with a one-paragraph batch note:
overall voice quality and any systematic drift you noticed.
