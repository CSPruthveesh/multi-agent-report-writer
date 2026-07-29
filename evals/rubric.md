# Evaluation Rubric — Multi-Agent Report Writer

**Status: FROZEN.** Committed before any report was generated. Do not edit after the first
Phase 0 run. If a criterion turns out to be badly worded, note the problem in the README's
limitations section rather than changing the rubric — a rubric edited mid-project invalidates
every comparison made under the old version.

This one rubric is used in three places:

1. **The Critic node**, scoring drafts inside the multi-agent graph
2. **The LLM judge** (`evals/judge.py`), scoring both systems' outputs in Phase 9
3. **Human scoring**, on 3 of the 6 topics, as a check on the judge

---

## Scoring rules

- Score the **whole report**, not individual sections.
- Integers 1–5 only. No half points.
- Criteria are scored **independently**. A report can be beautifully structured and factually
  ungrounded; that is a 5 on criterion 2 and a 1 on criterion 1.
- **Length is not quality.** A 900-word report that says everything it needs to scores higher
  than a 1400-word report that says the same thing with padding. See criterion 5.
- When a report sits between two anchors, round **down**. Optimistic scoring compresses the
  range and hides the very differences this project exists to measure.

---

## Criterion 1 — Factual grounding

*Do the report's claims trace back to retrieved evidence?*

**What to look at:** substantive claims — figures, causal assertions, characterisations of what
someone did or found. Ignore connective tissue ("this raises a further question") and common
knowledge.

| Score | Anchor |
|---|---|
| **1** | Claims float free of any retrieved evidence. Reads like the model wrote from memory. |
| **2** | A minority of substantive claims are traceable; the rest are asserted. |
| **3** | Most claims are traceable, but several load-bearing ones — the claims the argument depends on — are not. |
| **4** | Nearly all substantive claims trace to evidence; the untraceable ones are peripheral. |
| **5** | Every substantive claim traces to a retrieved finding. Where evidence is absent, the report says so instead of asserting anyway. |

**Common scoring mistake:** rewarding confident tone. Fluent unsourced assertion is the failure
mode this criterion exists to catch, and it reads *better* than honest hedging. A report that
says "the data here is thin" is being graded up, not down.

---

## Criterion 2 — Structural coherence

*Does the report build an argument, or is it a pile of sections?*

**What to look at:** section ordering, transitions, whether later sections depend on earlier
ones, whether any two sections say the same thing.

| Score | Anchor |
|---|---|
| **1** | Sections repeat each other, contradict each other, or appear in an order that could be shuffled without loss. |
| **2** | Sections are distinct but unconnected. No transitions; no cumulative argument. |
| **3** | A recognisable arc, but with one or two visible seams — a repeated point, an abrupt jump, a section that belongs elsewhere. |
| **4** | Clean arc with functional transitions. Each section earns its place. |
| **5** | The argument builds. Later sections use what earlier ones established, and the conclusion follows from the body rather than restating the introduction. |

**This is the criterion multi-agent is predicted to lose.** Score it strictly and honestly. Look
specifically for the multi-agent signature: two sections making the same point in different
words, because no single call ever saw the whole document.

---

## Criterion 3 — Depth of analysis

*Does the report synthesise, or does it summarise?*

| Score | Anchor |
|---|---|
| **1** | Restates sources one after another. No synthesis of any kind. |
| **2** | Groups related sources but draws no conclusions from the grouping. |
| **3** | Some genuine synthesis, but the report stops short of the implications a reader would want. |
| **4** | Connects findings across sources and draws supported conclusions. |
| **5** | Connects findings, surfaces tensions and disagreements between sources rather than averaging them away, and draws conclusions the sources individually do not state. |

**Common scoring mistake:** rewarding a report that resolves a genuine disagreement into a
smooth consensus. On a contested topic, flattening the disagreement is a *failure* of analysis,
not a success of writing.

---

## Criterion 4 — Citation integrity

*Do the citations resolve, and do they support what they are attached to?*

Two parts. Run the mechanical check first (`evals/check_citations.py`), then read.

| Score | Anchor |
|---|---|
| **1** | Cited IDs are missing, invented, or point at findings unrelated to the sentence. |
| **2** | Most IDs resolve, but several attach to claims the finding does not support. |
| **3** | All IDs resolve. Some are attached loosely — the finding is topically related but does not establish the specific claim. |
| **4** | All IDs resolve and support their claims. Coverage is slightly uneven — a few substantive claims carry no citation. |
| **5** | Every citation resolves, supports the exact sentence it sits on, and every claim that needs one has one. |

**Hard rule:** any invented ID (one not present in `findings`) caps this criterion at **1**,
regardless of how good the rest of the citation work is. Fabricated citations are the failure
mode that makes a research tool unusable.

Record the raw broken-citation count separately in `results/comparison.json` — it is a cleaner
number than the 1–5 score and belongs in the README table.

---

## Criterion 5 — Absence of filler

*Does every paragraph carry information?*

| Score | Anchor |
|---|---|
| **1** | Heavy padding: throat-clearing openers, restated section headers, generic hedging, a conclusion that adds nothing. |
| **2** | Several paragraphs could be deleted with no loss. |
| **3** | Mostly tight, with a recognisable filler section — usually the introduction or the conclusion. |
| **4** | Tight throughout. One or two sentences of scaffolding. |
| **5** | Every paragraph carries information. Nothing could be cut without losing content. |

**Filler tells:** "It is important to note that…", "In today's rapidly evolving landscape…",
restating the question as a finding, conclusions that only summarise, and hedges that hedge
nothing in particular ("various factors may play a role"). A hedge tied to specific missing
evidence is not filler — that is criterion 1 behaving correctly.

---

## Output format

Both the Critic node and the judge emit the same shape:

```json
{
  "scores": {
    "factual_grounding": 4,
    "structural_coherence": 3,
    "depth_of_analysis": 4,
    "citation_integrity": 5,
    "absence_of_filler": 3
  },
  "verdict": "revise",
  "target": "writer",
  "issues": [
    {
      "span": "exact substring quoted from the draft",
      "criterion": "structural_coherence",
      "problem": "what is wrong with it",
      "fix": "what would fix it"
    }
  ]
}
```

### Critic node only

- `verdict` is `revise` if **any** criterion is ≤ 3, else `pass`.
- `target` routes the revision: `writer` for prose problems (criteria 2, 5), `analyst` for
  structural or synthesis problems (criteria 3), `researcher` for evidence problems (criterion 1).
  Criterion 4 targets `writer` unless IDs were invented, which targets `researcher`.
- **Every issue must include `span`, quoted verbatim from the draft.** Issues whose `span` is not
  a substring of the draft are discarded in code before the revision prompt is built. An
  ungrounded critique produces plausible-sounding revisions that make the draft worse.
- `issues` may be empty only when `verdict` is `pass`.

### Judge only (Phase 9)

- The judge sees both reports for a topic, **unlabelled and shuffled**, and scores each
  independently. It is never told which system produced which.
- Shuffle order per topic and record the mapping outside the prompt.
- `verdict` and `target` are ignored; only `scores` are used.
- **Length-bias warning:** LLM judges systematically prefer longer outputs. The multi-agent
  system is expected to produce longer reports. Include an explicit instruction in the judge
  prompt that length is not evidence of quality, and sanity-check the result: if the judge
  prefers the longer report on all five criteria across all six topics, distrust it and weight
  the human scores instead.

---

## Human scoring protocol

Score 3 of the 6 topics by hand — pick one `abundant-sources`, one `thin-evidence` or
`contested`, and one `cross-domain`.

1. Read both reports for a topic before scoring either.
2. Score blind if you can manage it (have the filenames hashed), but do not pretend to blindness
   you do not have — if you know which is which, say so in the writeup.
3. Record the agreement with the judge as mean absolute difference per criterion.
4. Where you and the judge disagree by ≥ 2 points, write a sentence explaining why. Those
   sentences are the most interesting paragraph in the whole README.

---

## Prediction, recorded before any run

Written in advance so the result can be checked against it rather than rationalised afterwards.

| Criterion | Predicted winner | Reasoning |
|---|---|---|
| 1 — Factual grounding | **Multi-agent** | The Analyst→Researcher gap loop closes evidence holes a single agent writes around. |
| 2 — Structural coherence | **Single agent** | One call holds the whole document; the multi-agent Writer works from an outline it did not build. |
| 3 — Depth of analysis | Toss-up | A dedicated Analyst node should help, but it never sees the prose it shaped. |
| 4 — Citation integrity | **Multi-agent** | Structured `Finding` IDs plus a mechanical check make fabrication hard. |
| 5 — Absence of filler | **Single agent** | Each multi-agent revision loop tends to add rather than cut. |

**Cost prediction:** 4–8× tokens, 3–5× latency.
