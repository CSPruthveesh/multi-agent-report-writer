# Cost breakdown

### Provenance

| system | runs | commit | state |
|---|---:|---|---|
| baseline | 6 | `9e6844c` | inferred, not observed |
| multiagent | 6 | `2008af8` | dirty tree |

> **Warning:** multiagent: produced from a modified working tree, so it is not reproducible from commit 2008af8.

### Baseline (6 reports)

| node | calls | in | out | total | % |
|---|---:|---:|---:|---:|---:|
| agent | 78 | 138,494 | 56,124 | 194,618 | 100.0% |
| **TOTAL** | **78** | | | **194,618** | **100%** |

### Multi-agent (6 reports)

| node | calls | in | out | total | % |
|---|---:|---:|---:|---:|---:|
| researcher | 69 | 105,748 | 42,172 | 147,920 | 59.2% |
| analyst | 13 | 41,125 | 4,442 | 45,567 | 18.2% |
| writer | 7 | 17,689 | 8,975 | 26,664 | 10.7% |
| critic | 7 | 27,992 | 1,825 | 29,817 | 11.9% |
| **TOTAL** | **96** | | | **249,968** | **100%** |

### Cost by phase

| phase | calls | tokens | % of run |
|---|---:|---:|---:|
| first pass | 60 | 160,978 | 64.4% |
| gap loop (analyst -> researcher) | 34 | 81,411 | 32.6% |
| revision loop (critic -> writer) | 2 | 7,579 | 3.0% |

### Cost by call type

| call type | calls | tokens | % |
|---|---:|---:|---:|
| search | 28 | 77,271 | 30.9% |
| extract | 28 | 64,937 | 26.0% |
| analyze | 13 | 45,567 | 18.2% |
| critique | 7 | 29,817 | 11.9% |
| write | 6 | 23,391 | 9.4% |
| gap_plan | 7 | 4,671 | 1.9% |
| revise | 1 | 3,273 | 1.3% |
| plan | 6 | 1,041 | 0.4% |

### Failure-class overhead

| failure class | count | tokens |
|---|---:|---:|
| transport retries (429/500/timeout) | 0 | — |
| parse retries (schema mismatch) | 0 | 0 (0.0% of run) |

> Semantic retries are the revision loop, counted in the phase table
> above — they are not failures. A run where the Critic never asked for
> a revision has no revision-loop row at all.

### Head to head

| metric | baseline | multi-agent | multiple |
|---|---:|---:|---:|
| reports | 6 | 6 | — |
| model calls | 78 | 96 | 1.23x |
| input tokens | 138,494 | 192,554 | 1.39x |
| output tokens | 56,124 | 57,414 | 1.02x |
| total tokens | 194,618 | 249,968 | 1.28x |
| wall clock (s) | 535 | 401 | 0.75x |
| words written | 5,492 | 6,481 | — |

> **Tokens and ratios only.** No pricing rates were supplied, so no
> dollar figures are reported. Check current rates for the model in
> `GEMINI_MODEL`, then set `PRICE_IN_PER_M` and `PRICE_OUT_PER_M`.
