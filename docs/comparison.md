# Single agent vs multi-agent


Judge: gemini-3.5-flash, 2 round(s), 6 topics, blind and order-randomised.

| Criterion | Single agent | Multi-agent | Delta |
|---|---:|---:|---:|
| Factual grounding | 4.67 | 4.42 | -0.25 |
| Structural coherence | 5.00 | 4.83 | -0.17 |
| Depth of analysis | 5.00 | 4.50 | -0.50 |
| Citation integrity | 4.75 | 4.33 | -0.42 |
| Absence of filler | 5.00 | 4.75 | -0.25 |
| **Mean** | **4.88** | **4.57** | **-0.32** |
| Broken citations (count) | 0 | 0 | — |

> **Declared limitations.** declared unclosed evidence gaps: baseline 0/6, multi-agent 4/6. Stripped from both before scoring by judge and hand, because only one system can produce the section. Criterion 1 rewards saying where evidence is absent, so this is reported rather than scored.

### Cost

| Metric | Single agent | Multi-agent | Multiple |
|---|---:|---:|---:|
| Total tokens | 194,618 | 249,968 | 1.28x |
| Model calls | 78 | 96 | 1.23x |
| Wall clock (s) | 535 | 401 | 0.75x |
| Cost per report | — | — | no rates supplied |

**Where the multiple goes:**

  first pass                         64.4% of run cost
  gap loop (analyst->researcher)     32.6% of run cost
  revision loop (critic->writer)      3.0% of run cost

### By topic shape

| Shape | n | Grounding delta | Coherence delta | Mean delta |
|---|---:|---:|---:|---:|
| abundant-sources | 2 | +0.00 | -0.25 | -0.20 |
| contested | 1 | +1.00 | +0.00 | +0.30 |
| cross-domain | 2 | -1.00 | -0.25 | -0.65 |
| thin-evidence | 1 | -0.50 | +0.00 | -0.50 |

### Hand scores

| Criterion | Single | Multi | Delta | Judge delta | Gap |
|---|---:|---:|---:|---:|---:|
| Factual grounding | 4.00 | 4.33 | +0.33 | -0.25 | 0.58 |
| Structural coherence | 4.00 | 4.67 | +0.67 | -0.17 | 0.83 |
| Depth of analysis | 4.67 | 4.33 | -0.33 | -0.50 | 0.17 |
| Citation integrity | 4.00 | 4.33 | +0.33 | -0.42 | 0.75 |
| Absence of filler | 4.00 | 4.00 | +0.00 | -0.25 | 0.25 |

### Integrity checks


> **Warning.** They disagree about WHICH SYSTEM IS BETTER on 3 of 5 criteria: factual_grounding, structural_coherence, citation_integrity. A small gap between two deltas of opposite sign is still a contradiction, and it is the more interesting one. Trust your own read and explain the divergence in the write-up.
