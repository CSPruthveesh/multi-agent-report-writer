# Cost breakdown

### Provenance

| system | runs | commit | state |
|---|---:|---|---|
| baseline | 6 | `9e6844c` | inferred, not observed |
| multiagent | 6 | `626098d` | inferred, not observed |

### Baseline (6 reports)

| node | calls | in | out | total | % |
|---|---:|---:|---:|---:|---:|
| agent | 78 | 138,494 | 56,124 | 194,618 | 100.0% |
| **TOTAL** | **78** | | | **194,618** | **100%** |

### Multi-agent (6 reports)

| node | calls | in | out | total | % |
|---|---:|---:|---:|---:|---:|
| researcher | 69 | 105,287 | 45,192 | 150,479 | 61.2% |
| analyst | 13 | 43,054 | 4,534 | 47,588 | 19.3% |
| writer | 6 | 13,411 | 8,706 | 22,117 | 9.0% |
| critic | 6 | 24,157 | 1,682 | 25,839 | 10.5% |
| **TOTAL** | **94** | | | **246,023** | **100%** |

### Cost by phase

| phase | calls | tokens | % of run |
|---|---:|---:|---:|
| first pass | 60 | 159,409 | 64.8% |
| gap loop (analyst -> researcher) | 34 | 86,614 | 35.2% |

### Cost by call type

| call type | calls | tokens | % |
|---|---:|---:|---:|
| search | 28 | 76,746 | 31.2% |
| extract | 28 | 68,047 | 27.7% |
| analyze | 13 | 47,588 | 19.3% |
| critique | 6 | 25,839 | 10.5% |
| write | 6 | 22,117 | 9.0% |
| gap_plan | 7 | 4,645 | 1.9% |
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
| model calls | 78 | 94 | 1.21x |
| input tokens | 138,494 | 185,909 | 1.34x |
| output tokens | 56,124 | 60,114 | 1.07x |
| total tokens | 194,618 | 246,023 | 1.26x |
| wall clock (s) | 535 | 410 | 0.77x |
| words written | 5,492 | 6,084 | — |

> **Tokens and ratios only.** No pricing rates were supplied, so no
> dollar figures are reported. Check current rates for the model in
> `GEMINI_MODEL`, then set `PRICE_IN_PER_M` and `PRICE_OUT_PER_M`.
