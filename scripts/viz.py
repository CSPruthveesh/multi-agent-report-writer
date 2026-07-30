from __future__ import annotations

from src.graph.build import GRAPH, export_mermaid


def main() -> None:
    g = GRAPH.get_graph()

    print("nodes:")
    for n in g.nodes:
        print(f"  {n}")

    print("\nedges:")
    for e in g.edges:
        cond = " (conditional)" if getattr(e, "conditional", False) else ""
        label = f" [{e.data}]" if getattr(e, "data", None) else ""
        print(f"  {e.source:<12} -> {e.target}{label}{cond}")

    p = export_mermaid()
    print(f"\nwrote {p}")

    try:
        png = p.with_suffix(".png")
        png.write_bytes(g.draw_mermaid_png())
        print(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(png render skipped: {type(e).__name__} — the .mmd is the artifact that matters)")


if __name__ == "__main__":
    main()
