"""Print and export the graph topology.

    uv run python -m scripts.viz            # the evaluation graph
    uv run python -m scripts.viz --hitl     # with the approval gate

Two shapes, because build(hitl=True) inserts a node rather than reading a flag at
runtime. Rendering only one of them left build.py's docstring claiming a capability
this script did not have.
"""

from __future__ import annotations

import argparse

from src.graph.build import GRAPH, build, export_mermaid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hitl", action="store_true", help="render the approval-gate shape")
    args = ap.parse_args()

    g = (build(hitl=True) if args.hitl else GRAPH).get_graph()
    print(f"shape: {'hitl' if args.hitl else 'default'}\n")

    print("nodes:")
    for n in g.nodes:
        print(f"  {n}")

    print("\nedges:")
    for e in g.edges:
        cond = " (conditional)" if getattr(e, "conditional", False) else ""
        label = f" [{e.data}]" if getattr(e, "data", None) else ""
        print(f"  {e.source:<12} -> {e.target}{label}{cond}")

    p = export_mermaid("docs/graph_hitl.mmd" if args.hitl else "docs/graph.mmd",
                       hitl=args.hitl)
    print(f"\nwrote {p}")

    try:
        png = p.with_suffix(".png")
        png.write_bytes(g.draw_mermaid_png())
        print(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(png render skipped: {type(e).__name__} — the .mmd is the artifact that matters)")


if __name__ == "__main__":
    main()
