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
- New hardpoints ALWAYS go in `configs/` as the next version (`2027_v<N>_(what_changed).vahan`).
  During a long job, save the live candidate there as soon as it changes so the current points can
  always be opened in the app — never leave them only in a scratch/temp file. Follow the existing
  workflow; do not invent new formats, files or tools for work that already has a workflow.
- At phase boundaries, write distilled results (decisions, numbers with units, file:line) to the memory dir —
  anything not in a file/memory/git may be lost to compaction.
- This repo is PUBLIC and tracks the software only. Never commit: `tire_data/`, TTC files, `DESIGN_2027/`,
  `configs/`, `BINDER/`, `external/`, tire-derived plots/screenshots. No TTC raw data, fitted data,
  derived plots, screenshots or results may be public. The binder is team-only: never link or
  publish it from Vahan. De-identification is not permission to share. Work only on main;
  never create or publish another branch.
- Plots/UI colors: user is colorblind — use yellow/red/white/blue; never red/green or purple/blue contrasts.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Suspension coplanarity — user clarification, 2026-09-15
At static, independently at EACH corner, pushrod endpoints, rocker pivot/actuation points, spring endpoints and both bellcrank drop-link endpoints must be coplanar. Left and right do not share a plane. Do not impose fixed-plane-through-travel coplanarity or repackage dampers to achieve it. Preserve intended packaging; check actual travel clearance, stroke and joint articulation separately. Read docs/suspension_rules/01_coplanarity_no_bending.md and Rule 04. The v115/v116 repackaging was rejected; v117 restores the exact v114 packaging, with its unresolved issues retained rather than falsely marked fixed.

## Performance-first optimization — user instruction, 2026-09-15
Minimize member loads subject to justified suspension behavior; never sacrifice suspension behavior to clear packaging. Evaluate parameters jointly using the actual rim-matched TTC evidence and sourced synthetic class A AND B roads (do not imply measured MIS data). Include Ackermann, steer camber/camber gain, anti geometry, ride rates/MR, roll gradient, LLTD, Fz and utilization per tire, and aero goals with explicit inputs/sensitivity. Targets: 1.7 g sustained lateral without aero; 2.1 g with aero. Targets require verification, not forced fits or claimed universal perfection. BOTH ARBs have 12.7 mm outer diameter; solid/tube definition is awaiting user clarification and must remain explicit in alternatives. Reference aero speed is also unspecified: report required downforce/balance versus speed/radius rather than invent one. Use Fable 5.1 through the existing Claude app session plus smaller-model workers; Astra coordinates/reviews. Read DESIGN_2027/binder_run/FABLE_ENGINEERING_RESET_20260915.md and FABLE_REVIEW_GATES_20260915.md for the current delegated work and unresolved gates.

## Task continuity and axis normality — user reinforcement, 2026-09-15
A steering/interruption message ADDS TO or corrects the active work; previous tasks remain active unless the user explicitly cancels or replaces them. Retain the full dynamics/load/TTC/road/ARB/grip-target work while fixing static packaging. The rollback image is NOT a completed repair: its drop link is off-plane. Each corner's rocker pivot axis MUST be normal to that corner's static actuation plane, and both drop-link endpoints must lie in the same plane at static. These are mandatory paired gates before any corrected-model claim, alongside all existing behavior/clearance requirements. No axis-normality waiver; no independent point projection that breaks the actual solved mechanism. Preserve compact damper packaging. Do not ask to waive arms-up merely because one restricted fixed-rocker-plane search failed; first explore local rocker geometry within packaging/behavior constraints and report only the tested domain.

## Project-default hardware model — user instruction, 2026-09-15
Dimensions declared in the project folders are authoritative inputs for the current modeled repair; a manufacturer drawing is not a prerequisite. Keep conclusions scoped to the project-default model and do not present derived proxy shapes as measured production hardware.
