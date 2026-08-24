# SUSDYN / Vahan — Handoff (state as of 2026-08-20)

This archive is the COMPLETE working directory `C:\Users\xenos\SUSDYN` — including
`.git` (full history AND the uncommitted working tree), plus every gitignored data
directory (`configs/`, `tire_data/`, `DESIGN_2027/`, `external/`, `BINDER/`). Nothing
was omitted. Unzip it anywhere and you have an exact clone of the machine state.

> PRIVACY: `tire_data/` contains FSAE TTC tyre data. It is licensed to the team and
> must NEVER be pushed to the public GitHub repo or shared outside the team. This
> archive is for the team's private Drive only.

---

## What this is

**Vahan** — suspension kinematics + vehicle dynamics tool (Python/PyQt6). One solved
model drives everything: 3D view, kinematic graphs, dynamics, loads, lap sim.
Public repo: https://github.com/the-vedantin/SUSDYN

**2027 car design** — lives in `configs/` (the `.vahan` files ARE the car) and
`DESIGN_2027/` (binder generator + design records).

---

## Getting it running on a new machine

1. Install Python 3.12+ (3.14 is what this machine runs, via the `py` launcher).
2. `pip install -r requirements.txt`
   (core: numpy scipy matplotlib PyQt6 vispy; optional: cascadio+trimesh for STEP
   import, cadquery-ocp for STEP export, pywin32 for the SolidWorks bridge).
3. Run the app: `python app.py`
4. Regression net: `python test_one_model.py` — exit code 0 = healthy (~4 min).
   It auto-selects the HIGHEST `configs/2027_v*.vahan`.

---

## Current design state (READ THIS FIRST)

- **Working baseline: `configs/2027_v70_(front_actuation_flipped_fwd).vahan`**
  v70 = v69 with the FRONT actuation (rocker/coilover/ARB) mirrored about the
  vertical plane through the front pushrod, so it sits FORWARD of the axle —
  because chassis puts the front roll hoop at the front suspension's rear inboard
  pickups (y=+123..127 mm). Motion ratio/ARB/wheel kinematics identical to v69.
- **Two chassis-packaging candidates in `configs/experiments/`** (kept out of
  `configs/` root on purpose — every tool auto-selects the highest `2027_v*` there):
  - `2027_v71_(actuation_on_hoop_line).vahan` — actuation back toward the driver;
    damper mount ON the LCA-aft/UCA-aft line (collinearity 0.0000 mm) at z=600 —
    hoop tube = suspension mount. Cost: wide ARB (pivots x=±319) + thrust into
    bar mounts.
  - `2027_v72_(v70_tilted_down_15deg).vahan` — v70 rotated about the pushrod line;
    coilover exactly 15° from ground, running forward-down into the nose. Cost:
    ARB blade doubles (51→107 mm) + nose brackets at (34,−215,439) & (155,−278,411).
  Both: net-green, zero dynamics change, ARB triad exact. Chassis team decides.
- Rear (since v63): raised 1" for the big sprocket, pushrod foot at the ball joint,
  0.8" driveshaft, rocker re-tuned each step — rear MR 0.6628 / ARB ~12.0 kN/m held.
- Version-by-version history: the binder's "Design record" chapter (see below) and
  `DESIGN_2027/binder_run/OPEN_INSTRUCTIONS.md` (running instruction tracker).

## Engine model (the lap sim's torque curve)

- `vahan/engine.py` serves 3 curves from the SDM26 1-D solver data
  (`vahan/data/engine_*.json`): **corrected = default (77.7 hp crank @ 11,000)**,
  anchored (80 hp published level), raw (41.8 hp — the solver is PROVEN ~2x low:
  breathing floor + doubled friction; even unrestricted it only makes 48 hp).
- Engine page in the app: Ctrl+6 — every calibration knob is a visible input;
  "Apply to lap sim" stores the choice in the project file.
- The solver itself is vendored at `external/1dFVEngineSolver` (Sun Devil
  Motorsports, MIT — used unmodified, credited in README).
- When a real dyno pull exists: `anchor_curve_to_dyno()` in vahan/engine.py.

## The binder (design justification document)

- Generator: `DESIGN_2027/BINDER_2027_v2/make_html_binder.py` + `dump_*.py` scripts
  (all auto-select the highest config) + 2 GUI capture scripts (need native GL,
  pass the config as argv[1]).
- Published at: https://the-vedantin.github.io/Cougar-2027-VD-Binder/
  (repo `Cougar-2027-VD-Binder`, push = copy binder.html → index.html, commit, push).
- Currently FULLY re-baselined on v70 (commit a78cb16): AX-26 lap 40.83 s with aero,
  roll 0.631°/g, LLTD 36.2% front, live bearing-load sliders, engine chapter.
- RULE: "update the binder" = regenerate EVERY number/graph/figure/capture on the
  current config, not just edit sections.

## Uncommitted work in this archive (tracked files, not yet committed/pushed)

- `vahan/packaging.py` + `gui/packaging_page.py` (NEW) + edits to
  `vahan/interference.py`, `gui/main_window.py`, `test_one_model.py`:
  the PACKAGING DESIGN SYSTEM — Ctrl+7 page; hold all parameters within tolerance
  while moving actuation; manual transforms (mirror/rotate/translate/lever) with a
  live validity oracle (parameters + coplanarity + ARB triad + full-member clash at
  droop/static/bump), plus a generator (demo: 1005 candidates → 100 valid on v70 in
  319 s, wheel-parameter deltas exactly 0). Net-green including 2 new gates.
- `tools/sw_link.py` (NEW): external SolidWorks bridge — one-time `setup` builds a
  3D sketch of all hardpoints DRIVEN by global variables; weekly `update` rewrites
  the globals so points MOVE and downstream features survive. Trial first:
  `py tools\sw_link.py setup --coords vahan_hardpoints.txt --limit 2`
  (needs `pip install pywin32` + running SolidWorks; not yet tested against a live
  SW — per-point error reporting is built in for the first run).

## Known open items

1. Chassis to choose v71 vs v72 (or reject both) — then promote the winner into
   `configs/` root as v71/v72 proper.
2. Pre-existing interference: coilover body sits −3..−4 mm into the rocker-pivot
   hardware envelope on BOTH front corners (v69/v70 inherited; v71 improves it).
   Build-hardware question — check in the app's Interference view (now includes
   rocker-hardware spheres).
3. SolidWorks bridge: first live `--limit 2` trial pending on a SW machine.
4. Hardware data still owed by the team: pinion tooth count (18 assumed), Drexler
   diff ramps/preload, tube specs, upright material, rod-end size, damper data.
5. Real engine dyno pull → anchor the curve, retire the calibration debate.

## Claude Code session continuity (optional)

The AI-assistant session that produced this state lives on the origin machine at
`C:\Users\xenos\.claude\projects\C--Users-xenos-SUSDYN\` — the big `.jsonl` is the
conversation, `memory\` is the project knowledge base. To continue the same session
on another machine: install Claude Code, place the repo, copy the `.jsonl` +
`memory\` into the matching `~\.claude\projects\<repo-path-encoded>\` folder, then
`claude --resume <jsonl filename without extension>`.

## Key numbers cheat-sheet (v70)

| thing | value |
|---|---|
| mass (with driver) | 286.7 kg, 46% front |
| track F/R | 1269.5 / 1242.4 mm |
| roll gradient | 0.631 °/g · LLTD 36.2% front |
| front MR / ARB | 0.522 / 6.56 kN per m |
| rear MR / ARB | 0.663 / 12.02 kN per m |
| grip (asphalt, ×0.70 derate) | 1.666 g mech · 1.874 g w/ 350 N aero |
| engine (calibrated) | 77.7 hp crank @ 11,000 · torque 55.2 N·m @ 9,500 |
| AX-26 lap | 40.83 s aero / 41.70 s no aero |
| worst member load | LCA-front 7,922 N, pure braking 1.5 g |
| bearings | spacing 50.8 mm, outer 39.4 mm inboard of wheel CL |
| Ackermann | −27.7% (slight reverse, endorsed) |
