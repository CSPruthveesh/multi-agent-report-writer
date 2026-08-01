# Multi-Agent Report Writer

A controlled comparison between a single LLM agent and a multi-agent LangGraph system on
the same task: research a topic from the live web and write a cited report.

Both systems get the same six topics, the same model, the same search backend and the same
frozen rubric. The question is whether the coordination buys anything.

**The two instruments that measure it disagree, and the disagreement is the finding.**

---

## The result

Six topics, judged blind by `gemini-3.5-flash` in both presentation orders. Three of them
also scored blind by hand.

| Criterion | Judge: single / multi | Hand: single / multi |
|---|---:|---:|
| Factual grounding | 4.67 / 4.42 | 4.00 / 4.33 |
| Structural coherence | 5.00 / 4.83 | 4.00 / 4.67 |
| Depth of analysis | 5.00 / 4.50 | 4.67 / 4.33 |
| Citation integrity | 4.75 / 4.33 | 4.00 / 4.33 |
| Absence of filler | 5.00 / 4.75 | 4.00 / 4.00 |
| **Mean** | **4.88 / 4.57** | **4.13 / 4.33** |
| **Delta** | **−0.32** | **+0.20** |

| Cost | Single agent | Multi-agent | Multiple |
|---|---:|---:|---:|
| Total tokens | 194,618 | 249,968 | 1.28× |
| Model calls | 78 | 96 | 1.23× |
| Wall clock | 535s | 401s | 0.75× |
| Broken citations | 0 | 0 | — |

The multi-agent system costs 28% more tokens. It is *faster*, because the graph stops
searching when the Analyst judges the evidence sufficient while the single agent runs its
full ladder every time. Neither system has ever produced a broken citation.

The judge says it is slightly worse. A human reading the same reports blind says it is
slightly better. Section below explains why, and which one to believe on which criterion.

### Where it does pay

Judge scores, by topic shape:

| Shape | n | Grounding delta | Mean delta |
|---|---:|---:|---:|
| abundant-sources | 2 | +0.00 | −0.20 |
| contested | 1 | **+1.00** | **+0.30** |
| cross-domain | 2 | −1.00 | −0.65 |
| thin-evidence | 1 | −0.50 | −0.50 |

The one positive shape is contested — where holding two sides apart instead of averaging
them is the whole job. Contested and thin-evidence are **n=1 each**, so those are single
topics, not estimates.

### The defect that was found, and fixed

An earlier evaluation had the graph losing on **structural coherence and nothing else** —
drop that criterion and the other four favoured it. Two blind comments, written on opposite
labels, independently said the same thing: *"no argument being created, just facts bundled
up in paragraphs."*

The mechanism was visible in the table of contents. Six of six single-agent reports ended
on a synthesis — Conclusion, Policy Implications, Acknowledged Gaps. **Zero of six** graph
reports did. Three ended on `## Known limitations`, which the judge strips before scoring,
so as judged all six ended mid-list on another topic bucket.

The cause was two rules fighting in `writer.py`. *"Follow the outline's sections"* is
concrete; *"build an argument"* is abstract. The concrete one won. The Analyst's outline is
a partition of the evidence — the right job for an Analyst — and a Writer following it
faithfully renders a partition.

The outline is now what must be **covered**, not the table of contents, and the closing
section is the Writer's to add. After the fix:

| Structural coherence, by hand | Delta | Per topic |
|---|---:|---|
| Before the fix | **−1.00** | 0, −2, −1 |
| After the fix | **+0.67** | +1, 0, +1 |

Every topic improved or held, none went backwards, and the criterion went from the graph's
worst to its best — for **1.6% more tokens** and nine seconds less wall clock.

### Why the judge cannot see that

The judge's structural coherence delta is −0.17 before the fix and −0.17 after. Unchanged
to two decimals, while the proportion of reports closing on an argument went from none to
all.

It was already scoring the argument-less reports **4.83 out of 5** on that criterion. There
was no room left to show an improvement. The human scored the same reports 2.33.

> **A judge that scores a pile of disconnected topic buckets 4.83/5 on structural coherence
> is not measuring whether a report builds an argument.**

That is a finding about the instrument, not the system, and it is why the hand scores are
not a second opinion here — they are the only opinion.

### The gap loop gathers evidence the report has no room for

The Analyst→Researcher gap loop is **32.6% of the multi-agent run**. It was measured
directly rather than through a judge: run all six topics with the loop disabled, and count.
Coverage is a mechanical fact — findings gathered, findings cited, gaps left declared — so
there is no ceiling effect, no position bias, and none of the run-to-run drift that makes
the judge unreliable at this scale.

```bash
uv run python -m src.run --system multiagent --all --max-research-loops 0 --out-suffix noloop
```

| | Loop on | Loop off | Difference |
|---|---:|---:|---:|
| Findings gathered | 283 | 206 | **+77** |
| Findings cited | 200 | 179 | +21 |
| Coverage | 71% | **87%** | −16 points |
| Unclosed gaps declared | 5 | 5 | **none** |
| Tokens | 249,968 | 158,113 | **−36.7%** |
| Model calls | 96 | 60 | −37.5% |

**77 more findings bought 21 more citations.** But not because the Writer ignores what the
loop gathers — that was the first reading of this table and it is wrong. Splitting the
loop-on findings by where they came from:

| | Findings | Cited | Rate |
|---|---:|---:|---:|
| First pass | 197 | 142 | 72% |
| Gap loop | 86 | 58 | **67%** |

The loop's findings are cited at almost the same rate as the first pass. Five points apart.
The Writer is not discarding them.

**The report fills up instead.** Across all twelve runs, citations grow far slower than
evidence — 18 findings yields 17 citations, 62 yields 42, and every report is ~1,000 words:

| Findings available | Coverage |
|---|---|
| ≤ 33 | 90–100% |
| ≥ 41 | 62–80% |

So the loop causes **crowding out**. First-pass findings are cited at 72% when the loop
adds competition, against 87% in the loop-off arm where nothing competes. The loop's
findings take slots rather than adding them, and the total barely moves because the report
was already near full.

The unclosed-gaps row is the sharpest of the lot. The loop exists to close evidence gaps.
Both arms declared five. By its own stated purpose it netted nothing.

**The cap was lowered from 2 to 1 as a result.** Not to 0, and that distinction is a
judgement rather than a measurement: the experiment tested 0 against 2 and never tested 1.
What supports 1 specifically is weaker evidence — the two topics that ran a *second* loop
scored worst of the six with the judge — so this drops the round with no support and keeps
the one the architecture was built around. Going to 0 is defensible on the data and is one
flag away.

This matters for what to do about it. If the Writer *ignored* the loop's output you would
fix the Writer. Because the report is *already full*, gathering more cannot help at any
Writer quality — the only levers are capping the loop or lengthening the report, and the
report spec is frozen at 800–1,200 words.

**Read the two signals differently.** t2 raised no gaps in either arm, so it ran the same
configuration twice — an accidental control. It moved by 4 findings and 10 citations on
identical settings. So the +77 findings is well clear of the noise and is real; the +21
citations is not, and should not be quoted as an effect.

That splits the conclusion rather than softening it: **the evidence gain is solid, and the
part of it that reaches a reader is inside the noise.**

One more count worth having, because it shows the selection is not arbitrary. Pooling both
arms, the Writer cites **79%** of high-confidence findings, **82%** of medium, and **50%**
of low. It drops weak evidence at roughly twice the rate of strong. When it has to choose,
it chooses reasonably — which is why the uncited 29% is a capacity result rather than a
Writer defect.

This says nothing about prose quality. The loop-off reports run 994–1,142 words and cite
normally, but nobody has read them. Judging that arm would cost 190k tokens and land inside
a two-point noise floor — a mechanical question now has a mechanical answer, and a judge
would only add a weaker one on top.

The revision loop (Critic→Writer) sat at **0% of run cost** for three consecutive six-topic
runs at threshold 3, and did not terminate at threshold 4. After the Writer change it fired
once, on t6, and resolved — 2 calls, 3.0% of the run. The Critic did not change; the Writer
did. One data point, but the row had never carried a number before.

---

## What cannot be claimed

This section is load-bearing. The numbers above are weak evidence, and the honest reading
matters more than the headline.

- **Both instruments drift on identical input.** The single agent was not re-run between
  the two evaluations, so its six reports are byte-identical files that got scored twice.
  The judge moved one topic by **2.0 points out of 25**; the human moved one by **5**. Only
  gaps measured *within* a single sitting are comparable. `max spread = 1` measures
  position bias between the two orderings inside one run — it says nothing about
  repetition, and reading it as stability is a mistake this project made twice.
- **The +0.20 hand result is carried by one topic.** Per-topic gaps are t1 0, t3 −1, t5
  **+4**. Drop t5 and it goes slightly negative. The structural coherence result is the
  robust one: +1, 0, +1 — positive on two topics, negative on none.
- **The judge sits near its ceiling.** In the first evaluation, 27 of the single agent's 30
  criterion-scores were exactly 5.0. It discriminates mostly by *penalising* the graph, so
  a negative mean reads as "the graph lost points the control did not", not as a symmetric
  comparison — and on structural coherence it had no room to register the fix at all.
- **Per-shape numbers are n=1** for contested and thin-evidence.
- **One judge, six topics, three hand-scored, one human.**
- **No dollar figures anywhere.** The price constants were never verified, and the cost
  *multiple* moves with the price ratio because the two systems have different input:output
  mixes. Only token multiples are rate-free.
- **Declared limitations are stripped before scoring, by both instruments** — 4 of 6
  multi-agent reports declare unclosed evidence gaps and no single-agent report ever does,
  because it has no mechanism to. That asymmetry identifies the system on sight. But
  criterion 1 explicitly rewards admitting missing evidence, so stripping removes something
  the graph genuinely earns. It is counted and reported instead of scored — see
  `docs/comparison.md`.
- **Hand scoring was not blind until recently.** The judge stripped that section from the
  start; the hand-scoring harness copied reports verbatim. On one round the scorer
  recognised it and identified the system, which cost two of three topics. Those two were
  re-scored under a fixed preparation; the numbers above are from the clean pass.

---

## The two systems

**Baseline** (`src/baseline/agent.py`) — one agent, one loop: plan queries, search,
extract, judge sufficiency, write. No delegation.

**Multi-agent** (`src/graph/`) — a LangGraph state machine with a routing supervisor:

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	researcher(researcher)
	analyst(analyst)
	supervisor(supervisor)
	writer(writer)
	critic(critic)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> researcher;
	analyst --> supervisor;
	critic --> supervisor;
	researcher --> analyst;
	supervisor -.-> analyst;
	supervisor -.-> finalize;
	supervisor -.-> researcher;
	supervisor -.-> writer;
	writer --> critic;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

That block is [`docs/graph.mmd`](docs/graph.mmd) pasted verbatim, so the two can be
diffed. It is generated by `scripts.viz` from the real graph object, not drawn by hand —
regenerate and re-paste if the topology changes. The approval-gate shape is a second
file, [`docs/graph_hitl.mmd`](docs/graph_hitl.mmd), because `build(hitl=True)` inserts a
node rather than reading a flag at runtime. Both `.png` renders are git-ignored; the
`.mmd` files are the tracked artifact.

The two dotted back-edges are the loops under test. `supervisor -.-> researcher` is the gap
loop, which buys evidence coverage. `critic → supervisor → writer` is the revision loop,
which buys prose quality. They are never averaged into one "overhead" number — they are
different products with different value, and measuring them separately is what showed the
first one gathering evidence that never gets cited.

Both are capped, and both caps travel in graph state rather than being read from a module
constant — so a run made with the gap loop disabled records that condition in its own
`run.json`. An experimental condition that is not in the artifact is one nobody can verify
afterwards.

The graph also checkpoints to SQLite, contains node failures, and supports an outline
approval gate (`--hitl`).

---

## The evaluation

The instrument was built before the experiment, and it took three attempts to get one that
works.

**Position control.** Every topic is judged in both orderings and the scores averaged. The
ordering comes from a stable `sha256`, and round 2 is the exact mirror of round 1 — not a
second random draw. Max disagreement between orderings across all six topics was **1
point**.

**Calibration.** `scripts/judge.py --calibrate` damages a clean report four known ways and
checks the judge notices on the right criterion. `gemini-3.5-flash` scored 4/4. A judge that
fails this gate makes every downstream number worthless.

**No self-judging.** The judge model must differ from the pipeline model, or the run stops.
`judge.json` records `pipeline_model` and a `self_judged` boolean, so a reader can tell from
the file alone.

**Blinding, for both instruments.** System names and the "Known limitations" section are
stripped before *either* the judge or the human reads anything — `handscore.py` imports the
same `_body` the judge uses, so the two cannot drift on what blinding means. The
declaration is counted and reported separately, because criterion 1 rewards it.

**Staleness refusal.** A re-run overwrites the reports but not the copies sitting under
`results/handscore/`. Scoring those, or joining them to fresh judge numbers, would put a
human column and a judge column in one table describing different documents. Each copy is
digested against the report it came from, and the harness refuses rather than proceeding.

**Separate evidence namespaces.** Both systems number findings from `F001` independently
and they collide completely — 67 of 67 on t1. Each report is scored against *its own*
evidence block, never a merged one. Merging them silently penalises the control on citation
integrity, in a table that looks entirely normal.

**Hand scoring.** `evals/handscore.py` shuffles the two reports per topic, writes the
mapping to a file it tells you not to open, and takes five integers each. Each report also
gets its own evidence file, so citation integrity is checkable without breaking the blind.

**Self-checks.** `evals/compare.py` runs four integrity checks on its own output — length
bias, baseline strength, judge/human agreement and loop value — and prints warnings above
the numbers rather than below them. Two of the four currently report a pass they have not
earned; both are documented in the Phase 9 write-up.

---

## Running it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a Gemini API key and a Tavily API
key.

```bash
cp .env.example .env    # then fill in GEMINI_API_KEY and TAVILY_API_KEY
uv sync
```

Generate reports:

```bash
uv run python -m src.run --system baseline   --all
uv run python -m src.run --system multiagent --all
uv run python -m src.run --system multiagent --topic-id t1 --hitl   # approval gate
uv run python -m src.run --resume <thread_id> --approve
uv run python -m src.run --status <thread_id>
```

Run a matched arm — same topics and first-pass budget, gap loop off — without overwriting
the control it is compared against:

```bash
# writes to results/multiagent_noloop/, leaving results/multiagent/ intact
uv run python -m src.run --system multiagent --all --max-research-loops 0 --out-suffix noloop
```

Evaluate. Only the first of these spends tokens:

```bash
export GEMINI_JUDGE_MODEL=gemini-3.5-flash   # must differ from GEMINI_MODEL
uv run python -m evals.judge --repeats 2     # ~190k tokens, both orderings
uv run python -m evals.handscore             # blind, t1/t3/t5, interactive
uv run python -m evals.compare               # free — the comparison table
uv run python -m evals.handscore --reveal    # mappings, after scoring
uv run python -m scripts.cost_report         # free — cost by node and phase
```

Regenerate the committed documents. None of these spend tokens:

```bash
uv run python -m evals.compare --markdown       > docs/comparison.md
uv run python -m scripts.cost_report --markdown > docs/cost.md
uv run python -m scripts.viz                    # docs/graph.mmd
uv run python -m scripts.viz --hitl             # docs/graph_hitl.mmd
```

These are generated files. The first two go stale the moment results change, so
regenerate them in the same commit as any new run; the diagrams only change when the
graph does. The mermaid block above is a manual copy of `docs/graph.mmd` — re-paste it
when `viz` reports a change.

**Ordering matters.** `compare.py` prints hand scores mapped to their systems, so it breaks
the blind for any topic not yet scored. Score first, compare after.

`GEMINI_JUDGE_MODEL` is not in `.env.example` on purpose — setting it in the shell keeps the
choice of judge visible in the command rather than buried in a file.

---

## Layout

| Path | What it holds |
|---|---|
| `src/baseline/` | The single-agent control |
| `src/graph/` | The LangGraph system — nodes, state, checkpointing, retry |
| `src/common/` | Shared LLM, search, schemas and run-record I/O |
| `src/analysis/` | Cost attribution by node, phase and call type |
| `evals/` | The judge, the comparison, blind hand-scoring, the frozen rubric |
| `scripts/` | Calibration, pairwise judging, chaos and resume demos, backfills |
| `data/topics.json` | The six frozen topics, with a pre-registered trap for each |
| `results/` | Reports, run records, judge output, hand scores |
| `results/multiagent_noloop/` | The gap-loop-off arm, for the matched-pair comparison |
| `docs/comparison.md` | Generated — the comparison table, with its own integrity checks |
| `docs/cost.md` | Generated — cost breakdown, tokens and ratios only |
| `docs/graph.mmd` | Generated — graph topology, and `graph_hitl.mmd` with the approval gate |

Every run record carries the commit it was produced at, and `cost_report` refuses to read a
set spanning more than one code version as a single measurement.

The six topics are frozen, and each carries a pre-registered trap — a specific way a report
can look good and be wrong. t1's is announced pilot capacity versus deployed capacity;
t3's is drifting back to the three countries the question excludes. Naming them before
either system ran is what stops the evaluation from deciding after the fact what counts as
a flaw.

---

## Write-ups

Each phase has a walkthrough in `analysis/` (git-ignored local artifacts). Phase 9 covers
the judge, the six defects found in it before it produced a number, the run, the hand
scores, and the two integrity checks that reported a pass they had not earned.
