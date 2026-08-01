# Single agent vs multi-agent


Judge: gemini-3.5-flash, 2 round(s), 6 topics, blind and order-randomised.

| Criterion | Single agent | Multi-agent | Delta |
|---|---:|---:|---:|
| Factual grounding | 4.67 | 4.42 | -0.25 |
| Structural coherence | 5.00 | 4.83 | -0.17 |
| Depth of analysis | 4.83 | 4.50 | -0.33 |
| Citation integrity | 4.67 | 4.42 | -0.25 |
| Absence of filler | 5.00 | 4.67 | -0.33 |
| **Mean** | **4.83** | **4.57** | **-0.27** |
| Broken citations (count) | 0 | 0 | — |

### Cost

| Metric | Single agent | Multi-agent | Multiple |
|---|---:|---:|---:|
| Total tokens | 194,618 | 246,023 | 1.26x |
| Model calls | 78 | 94 | 1.21x |
| Wall clock (s) | 535 | 410 | 0.77x |
| Cost per report | — | — | no rates supplied |

**Where the multiple goes:**

  first pass                         64.8% of run cost
  gap loop (analyst->researcher)     35.2% of run cost

### By topic shape

| Shape | n | Grounding delta | Coherence delta | Mean delta |
|---|---:|---:|---:|---:|
| abundant-sources | 2 | +0.00 | +0.00 | -0.05 |
| contested | 1 | +1.50 | +0.00 | +0.80 |
| cross-domain | 2 | -0.50 | -0.25 | -0.55 |
| thin-evidence | 1 | -2.00 | -0.50 | -1.20 |

### Hand scores

| Criterion | Single | Multi | Delta | Judge delta | Gap |
|---|---:|---:|---:|---:|---:|
| Factual grounding | 4.00 | 4.33 | +0.33 | -0.25 | 0.58 |
| Structural coherence | 3.33 | 2.33 | -1.00 | -0.17 | 0.83 |
| Depth of analysis | 3.33 | 3.33 | +0.00 | -0.33 | 0.33 |
| Citation integrity | 4.33 | 4.00 | -0.33 | -0.25 | 0.08 |
| Absence of filler | 3.67 | 4.00 | +0.33 | -0.33 | 0.67 |

### Integrity checks


No warnings raised.
