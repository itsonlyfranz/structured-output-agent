"""The README's agent-flow diagram must match the compiled LangGraph graph (offline)."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)

EDGE = re.compile(r"^\s*(\w+)\s+(-->|-\.->)\s+(\w+);?\s*$")


def readme_flow_edges() -> set[tuple[str, str, bool]]:
    block = render_graph.BLOCK.search((ROOT / "README.md").read_text())
    assert block, "flow-diagram markers not found in README.md"
    edges = set()
    for line in block.group(0).splitlines():
        if m := EDGE.match(line):
            edges.add((m.group(1), m.group(3), m.group(2) == "-.->"))
    return edges


def test_readme_flow_diagram_matches_compiled_graph():
    graph = render_graph.drawable_graph()
    expected = {(e.source, e.target, e.conditional) for e in graph.edges}
    assert readme_flow_edges() == expected, "stale diagram: run `python scripts/render_graph.py`"


def test_flow_diagram_has_every_node():
    nodes = set(render_graph.drawable_graph().nodes)
    assert nodes == {"__start__", "generate", "validate", "act", "fail", "__end__"}


def test_check_mode_detects_drift():
    stale = (ROOT / "README.md").read_text().replace("validate -.-> fail;", "validate -.-> gone;")
    assert render_graph.updated_readme(stale) != stale
