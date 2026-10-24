"""The README's agent-flow diagram must match the compiled LangGraph graph (offline)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text()
    assert render_graph.render_block() in readme, "stale diagram: run `python scripts/render_graph.py`"


def test_flow_diagram_mentions_every_node():
    mermaid = render_graph.flow_mermaid()
    for node in ("generate", "validate", "act", "fail"):
        assert node in mermaid


def test_check_mode_detects_drift():
    stale = (ROOT / "README.md").read_text().replace("validate -.-> fail;", "validate -.-> gone;")
    assert render_graph.updated_readme(stale) != stale
