# Vahan — FSAE suspension kinematics + vehicle dynamics (PyQt6)

Run: `python app.py` (PyQt6 — needs full app restart to pick up code changes, no hot-reload).
Regression net: `python test_one_model.py` (offscreen, ~45 s; exit code = number of unexpected failures).

## Working rules
- ONE MODEL invariant: the 3D view, every kinematic graph, and the dynamics all derive from the single
  solved model — never a second/hardcoded model. Kinematics changes usually touch the `vahan/` solver +
  `gui/panels.py` + `gui/view3d.py` together.
- Done means: regression net exits 0 AND `git diff --stat` reviewed. The net is a safety net, not proof —
  numeric/visual verification of the actual model output is still required.
- Solver bug fixes add a failing-then-passing check to `test_one_model.py` (KNOWN-FAIL with reason if unfixed).
- `gui/main_window.py` (~10k lines) and `gui/panels.py` (~7k lines): NEVER read in full. Grep for
  `class`/`def`, Read with offset/limit, consult memory `project_codemap.md` first; update it after big edits.
- Broad exploration goes to a subagent; keep only the summary in context.
- At phase boundaries, write distilled results (decisions, numbers with units, file:line) to the memory dir —
  anything not in a file/memory/git may be lost to compaction.
- This repo is PUBLIC and tracks the software only. Never commit: `tire_data/`, TTC files, `DESIGN_2027/`,
  `configs/`, `BINDER/`, `external/`, tire-derived plots/screenshots. If asked to, check the TTC publishing
  rules first (memory: project_ttc_publishing).
- Plots/UI colors: user is colorblind — use yellow/red/white/blue; never red/green or purple/blue contrasts.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
