# Vahan — FSAE suspension kinematics + vehicle dynamics

Vahan is a desktop tool (Python, PyQt6, VisPy) for designing and analysing a double-wishbone,
pushrod/pullrod/direct-actuated race-car suspension and the vehicle dynamics that follow from it.
It was built by Cougar Racing for the 2027 FSAE car. This repository tracks the **software only**;
team design files, the team-only binder and licensed tyre data are private and are not distributed
with the software (see [Data policy](#data-policy)).

**One project geometry.** The 3-D view, kinematic graphs, steady-state dynamics, member loads,
lap simulation, Ackermann and corner-speed pages use the project's hardpoints and shared
solver paths. Individual analyses still have their own approximations,
fallbacks and update controls; these are described below. ONE MODEL is the architecture rule
(`docs/suspension_rules/00_one_model.md`), not proof that every path is equivalent.

> **Reading this README.** Each tool section states what it needs, what it gives you, and what it
> assumes. Words are used precisely: **implemented** = in the code and exercised by the regression
> net or a test; **partial** = present but with a documented gap; **planned** = written down in a
> plan or backlog, no code. Nothing here is "validated" in the sense of correlation against a
> real car: the regression net is a safety net against regressions, not proof of correctness
> (`test_one_model.py` docstring, line 14).

---

## Table of contents

1. [Status at a glance](#status-at-a-glance)
2. [Install and run](#install-and-run)
3. [Quick start: a project end to end](#quick-start-a-project-end-to-end)
4. [Conventions: frame, units, signs](#conventions-frame-units-signs)
5. [Pages, menus and shortcuts](#pages-menus-and-shortcuts)
6. [The 3-D view](#the-3-d-view)
7. [Editing hardpoints](#editing-hardpoints)
8. [Topologies](#topologies)
9. [Kinematics and the metrics catalogue](#kinematics-and-the-metrics-catalogue)
10. [Inverse kinematics and packaging](#inverse-kinematics-and-packaging)
11. [Interference, chassis bays, keep-out, rim fit, joint articulation](#interference-chassis-bays-keep-out-rim-fit-joint-articulation)
12. [Springs, dampers, motion ratio and anti-roll bars](#springs-dampers-motion-ratio-and-anti-roll-bars)
13. [Steady-state dynamics](#steady-state-dynamics)
14. [Tyres and TTC data](#tyres-and-ttc-data)
15. [Component loads, brakes and bearings](#component-loads-brakes-and-bearings)
16. [Ackermann, yaw-moment diagrams and steering](#ackermann-yaw-moment-diagrams-and-steering)
17. [Ride, road and transient models](#ride-road-and-transient-models)
18. [Lap time, engine, corner speed, acceleration](#lap-time-engine-corner-speed-acceleration)
19. [Build tolerance and sensitivities](#build-tolerance-and-sensitivities)
20. [CAD and other exports](#cad-and-other-exports)
21. [Project files and versioning](#project-files-and-versioning)
22. [Screenshots](#screenshots)
23. [Testing](#testing)
24. [Development and architecture map](#development-and-architecture-map)
25. [Contributing](#contributing)
26. [Troubleshooting](#troubleshooting)
27. [Data policy](#data-policy)
28. [Credits and license](#credits-and-license)

---

## Status at a glance

| Area | Status | Where |
|---|---|---|
| Double-wishbone corner solver (rigid links, Newton–Raphson, analytic Jacobian) | implemented | `vahan/solver.py` |
| Pushrod / pullrod + bellcrank, direct damper, control-arm ARB, decoupled twin rockers, T-bar ARB | implemented, exercised per topology by the net | `vahan/topology.py`, `test_one_model.py` CASES |
| Heave T-bar third element (corner springs + heave coilover on the T-bar) | implemented; the net's CASES table still carries a KNOWN-FAIL note about graph wiring, but the case passed on the run recorded below | `vahan/heave_tbar.py`, `test_one_model.py` lines 46-47 |
| Kinematic metrics (38 catalogue entries), live sweeps, hover/copy on every graph | implemented | `vahan/metrics_catalog.py` |
| Inverse kinematics (target curves, staged/hybrid/local/global) | implemented | `vahan/optimizer.py` |
| Packaging moves with every parameter held, relocation search, Design City | implemented | `vahan/packaging.py`, `vahan/relocate.py`, `design_city.py` |
| Interference (capsules, rocker plate, ARB, driveshaft, FSAE chassis bays, keep-out, real rim profile) | implemented | `vahan/interference.py`, `chassis.py`, `keepout.py`, `wheel_profile.py` |
| Steady-state dynamics with kinematics in the loop, LLTD, jacking, anti geometry, aero | implemented | `vahan/dynamics.py` |
| Tyre surfaces from TTC `.mat` / CSV / XLSX, parametric fallback | implemented (data not shipped) | `vahan/tire_model.py` |
| Member loads, ball joints, wheel bearings, caliper bolts, brakes | implemented, quasi-static, rigid | `vahan/loads.py` |
| Inboard spherical-bearing inputs (SKF calculator inputs, 27° swivel check) | implemented | `vahan/spherical_bearings.py` |
| Yaw-moment engine, Ackermann demand/capability, MMD | implemented | `vahan/ymd.py`, `vahan/ackermann.py` |
| 7-DOF ride on ISO 8608 class A/B synthetic roads, ride-rate solve | implemented (standard topology only) | `vahan/ride.py`, `ride_solve.py`, `road.py` |
| Transient bicycle + roll model (skidpad, step, ramp, sine) | implemented | `vahan/transient.py` |
| Quasi-steady lap sim with gear-resolved powertrain | implemented | `vahan/laptime.py` |
| Engine torque curve | implemented, **simulated, not dyno-validated** | `vahan/engine.py` |
| CG build-tolerance sweep, aero heave sweep | implemented (deterministic one-axis sweeps, not Monte Carlo) | `vahan/cg_tolerance.py` |
| Onshape points export (FeatureScript), SolidWorks CSV/equations, STEP import | implemented | see [CAD and other exports](#cad-and-other-exports) |
| SolidWorks live bridge (`tools/sw_link.py`) | **untested against a live SolidWorks** | `HANDOFF.md` |
| MCP server (operate the running app from an agent) | implemented, optional `mcp` package | `gui/mcp_server.py` |
| Measure mode in the 3-D view | **planned**, design only | `docs/MEASURE_MODE_PLAN.md` |
| Compliance (bushings, member or chassis flex), active suspension, autonomy, CFD, FEA solvers, Git-like design branching | **not implemented** | — |

What Vahan does **not** contain: a compliance model (every link, the upright and the chassis are
rigid; the only compliance in the code is the anti-roll-bar arm's own bending/torsion in its rate
and a first-order steer-actuator lag in the transient model), any FEA/CFD solver or exporter, any
autonomous-driving design layer, and any design-version branching beyond saving numbered files.
The transient skidpad model does include a path follower and speed controller.

---

## Install and run

Verified runtime for this README (fresh container, 2026-10-03):

| Component | Version used | Notes |
|---|---|---|
| Python | **3.12 or newer required** | the regression net uses PEP 701 f-strings and does not parse on 3.11 (`test_one_model.py` line 4559); the team's machine runs 3.14 |
| numpy / scipy | 2.4.6 / 1.17.1 | `requirements.txt` floors: 1.24 / 1.10 |
| matplotlib | 3.11.2 | graphs and analysis plots |
| PyQt6 | 6.11.0 | GUI |
| vispy | 0.17.0 | 3-D view (`gui/view3d.py`) — **now listed in `requirements.txt`**; it was missing before this revision |
| python-docx, Pillow | 1.2.0, 12.3.0 | `.docx` report export |

Optional packages (features degrade gracefully without them):

`cascadio`, `trimesh` and `cadquery-ocp` are currently included in `requirements.txt`,
so that install command also installs the STEP dependencies even though those features are optional.

| Package | Enables |
|---|---|
| `cascadio`, `trimesh` | File → Import STEP (differential / engine solids as clearance bodies) |
| `cadquery-ocp` | "Export moved part…" (write a repositioned STEP back out) |
| `mcp` | the embedded MCP server on 127.0.0.1:8765 (`VAHAN_MCP=0` disables) |
| `pandas` (+ `openpyxl` for `.xlsx`) | tyre data from CSV / XLSX instead of a TTC `.mat` |
| `pywin32` (Windows only) | `tools/sw_link.py` SolidWorks bridge |

```bash
pip install -r requirements.txt
python app.py
```

Linux note: PyQt6 + VisPy need the system OpenGL/EGL libraries. On a minimal Debian/Ubuntu
image `apt-get install libegl1 libopengl0` was required before the app would import, even
headless (`QT_QPA_PLATFORM=offscreen`).

The app has no hot reload: restart it after a code change.

---

## Quick start: a project end to end

1. **Start.** `python app.py` opens the startup dialog (`gui/startup_dialog.py`): *Open an
   existing project* (`.vahan` / `.json`) or *Start a new design*. A new design asks for the
   vehicle dimensions (wheelbase, front/rear track, rack length and total travel, spring and
   damper OD) and the per-axle topology (actuation, mount, anti-roll device, spring
   configuration); invalid combinations block *Continue* with the reason.
   Default hardpoints for the chosen topology are loaded (`gui/main_window.py` `DEFAULT_*_HP`).

2. **Enter your geometry.** Type hardpoints in the Front / Rear Hardpoints tables (mm), or use the
   Direct Edit panel and the 3-D view (section [Editing hardpoints](#editing-hardpoints)). Only
   the left side is stored; FR/RR are X-mirrors of FL/RL.

3. **Sweep and read.** The Motion panel sets heave / roll / pitch / steer and the travel range;
   graphs re-sweep automatically (150 ms debounce after an edit) and every curve is
   hover-readable. Pick metrics in Graph Selection; the Values panel shows the current pose.

4. **Set the car.** Car Parameters (masses, CG, tyre dimensions, rack), Alignment (static camber
   and toe applied as post-solve offsets), Steering (rack travel per wheel turn, total travel,
   direction), Dynamics (springs, dampers, bars, drivetrain, aero), and optionally a tyre file.

5. **Check the build.** Interference view mode, FSAE chassis bays (View menu), keep-out solid,
   real rim profile, Bearings page (rod-end swivel), Loads page.

6. **Analyse.** Dynamics sweeps, Ackermann + MMD, Corner Speed, Ride, Lap Time, Build Tolerance.

7. **Save and export.** File → Save Project (`.vahan`, atomic), Export Sweep Data (CSV), Export
   Report (`.docx`), Engineering Review (Markdown), Copy for Onshape / SolidWorks.

8. **Guard it.** `python test_one_model.py` runs the regression net offscreen
   (see [Testing](#testing) for what it needs).

---

## Conventions: frame, units, signs

The authoritative definitions are the docstrings of `vahan/hardpoints.py` (lines 6-12) and
`vahan/kinematics.py` (lines 1-19); `docs/suspension_rules/README.md` restates them.

```
        Z (up)
        |
        +------ X  lateral, outboard positive for the corner being modelled (left corner: +X = left)
       /
      Y  longitudinal, REARWARD positive: front axle at Y = 0, rear axle at Y = +wheelbase
```

- Origin: vehicle centreline (X = 0), front axle line (Y = 0), ground (Z = 0).
- Hardpoints are authored at **design ride height** with the wheel centre at tyre radius above
  Z = 0 on both axles; `hardpoints.tire_ground_gap_mm` checks this to 3 mm (Rule 10).
- **Units:** metres and radians inside `vahan/`; millimetres and degrees in the GUI, the
  project file is in metres. Wheel travel: metres in the solver, mm on the slider; **+ = bump**,
  − = droop.
- **Corners:** FL is modelled; FR is the X-mirror; RL is stored in absolute Y; RR is the mirror.
  `hardpoints.mirror_x` negates X of every point.

| Quantity | Positive means | Source |
|---|---|---|
| Camber | top of wheel leans **outboard** (negative = inboard); front-view XZ projection of the spin axis | `kinematics.py` 93-102 |
| Toe | toe-in | 107-114 |
| Caster | top of kingpin tilts rearward (+Y) | 119-128 |
| KPI | top of kingpin tilts inboard | 133-139 |
| Scrub radius | kingpin ground point inboard of the contact patch | 153-160 |
| Mechanical trail | contact patch behind (+Y of) the kingpin ground point | 163-171 |
| Roll-centre height | above ground | 216-242 |
| Member axial force | tension (negative = compression) | `vahan/loads.py` |
| Ball-joint / bearing V, H | up, forward (toward the nose) | `vahan/loads.py` |
| Lateral g in dynamics | left turn (read FL as FR for a right turn) | `vahan/dynamics.py` |
| Longitudinal g | acceleration (negative = braking) | `vahan/dynamics.py` |

Two things to know about the frame:

- The **contact patch** used by scrub, trail and the roll-centre construction is the wheel-centre
  X/Y dropped to Z = 0, an approximation that ignores camber and the tyre's lateral stiffness.
- The front-view `camber` metric is defined at zero steer. `kinematics.road_plane_camber_deg`
  gives the inclination to the road plane that remains valid when the wheel is steered; the
  dynamics solver uses it for the ground-referenced camber it feeds the tyre
  (`vahan/dynamics.py` 1338). Whether the steered-camber effect is also applied inside the tyre
  force lookup is an open task (`docs/GUI_TASKS.md`, "Steer camber in the dynamics / tyre model").
- A few older code comments (and `examples/fsae_front.py`, which predates the current axis
  convention and uses X forward / Y outboard) describe +Y as forward. The solver, the metrics
  and the project files use **+Y rearward**; trust those.

---

## Pages, menus and shortcuts

An always-visible page bar sits above the workspace; the same entries are in the **Page** menu.

| Shortcut | Page | Module | What it is for |
|---|---|---|---|
| Ctrl+1 | Suspension | `gui/main_window.py`, `gui/panels.py` | 3-D view, graphs, every side panel (the main workspace) |
| Ctrl+2 | Lap Time | `gui/laptime_page.py` | quasi-steady lap on a digitised track |
| Ctrl+3 | Design City | `gui/city_page.py` | packaging alternatives that hold every parameter within 0.1 % |
| Ctrl+4 | Loads | `gui/loads_page.py` | member/joint loads with a hoverable 3-D force view |
| Ctrl+5 | Ackermann + MMD | `gui/ackermann_page.py` | Ackermann demand / capability, yaw-moment diagrams |
| Ctrl+6 | Engine | `gui/engine_page.py` | torque curve calibration the lap sim runs on |
| Ctrl+7 | Packaging | `gui/packaging_page.py` | move the inboard actuation with the parameters held |
| Ctrl+8 | Ride | `gui/ride_page.py` | ISO 8608 road response, ride-rate solve, contact-patch and launch studies |
| Ctrl+9 | Corner Speed | `gui/corner_speed_page.py` | trimmed maximum speed per radius, per-corner grip budget |
| Ctrl+0 | Bearings | `gui/bearings_page.py` | inboard spherical-bearing inputs and swivel check |
| Ctrl+Shift+1 | Build Tolerance | `gui/build_tolerance_page.py` | CG tolerance bands, aero heave vs speed |

Pages other than Suspension are built the first time they are opened.

**Menus** (`gui/main_window.py` `_build_menu`):

- **File:** Save Project…, Load Project…, Change Topology… (re-opens the per-axle pickers; keeps
  car/steer/dimensions, resets corner hardpoints to the new topology's defaults), Export
  Report… (`.docx`), Engineering Review… (Markdown), Export Sweep Data (CSV)…, Import STEP
  (diff / engine)….
- **View:** All Hardpoints… (every point of every corner, plus derived bearing points, with the
  CAD copy buttons), STEP parts (show / hide), Manage STEP parts…, FSAE chassis bays (checkable),
  FSAE chassis settings…
- **Page:** the eleven pages above.
- **Help:** Task list… opens `docs/GUI_TASKS.md` as a checklist; ticking an item writes back to
  the file.

**Keyboard:** Ctrl+Z / Ctrl+Y undo and redo hardpoint edits. Direct-edit keys are listed under
[The 3-D view](#the-3-d-view).

---

## The 3-D view

`gui/view3d.py` renders the solved model with VisPy: control arms, upright and ball joints,
tie/toe rod, pushrod, rocker plate (legacy, double-shear or local-clevis styles), spring/damper
cylinder at the declared OD, anti-roll bar (bar, blades, drop links), driveshaft/diff package,
tyres with the **real rim profile** when `car['wheel_profile']` is set (from the manufacturer's
STEP, `vahan/wheel_profile.py`), brake rotor and caliper, wheel-bearing spheres, FSAE chassis
tubes, the keep-out solid, imported STEP parts, roll centres, roll axis, pitch axis, CG and
unsprung-CG markers, and a grey baseline ghost.

- **Mouse:** left-click picks the nearest hardpoint (30 px); right-drag orbits; middle-drag or
  Ctrl+right-drag pans; wheel zooms.
- **NavCube** snaps the camera to the standard views.
- **View-controls box** (under the NavCube): mode *Normal / Load / Interference*, and
  *Perspective*, *Floor*, *Thickness* toggles. Load mode draws force vectors on a desaturated
  model and shows the load under the cursor; Interference mode highlights clashing members red.
- **Motion level of detail:** while the slider moves, detail bodies are hidden and restored on
  the 200 ms settle frame.
- **Ground follows the tyres** in heave, so a dropping chassis is visible against the road.
- **Direct-edit keys** (when the Direct Edit panel is enabled): W/S = ±Y, A/D = ±X, Q/E = ±Z;
  1-6 set the step to 0.1 / 0.5 / 1 / 2 / 5 / 10 mm; Tab / Shift+Tab (or N / B) cycle the
  corner's points; F / L / P set the nudge constraint to free / along-link / in-plane.
- **Dance** (Motion panel) animates the four corners as a travelling wave; purely visual.

A **Measure mode** (pick two points/lines/planes, read distance or angle live, toggle every
construction line and plane in a layer tree) is planned in `docs/MEASURE_MODE_PLAN.md`. No code
exists for it yet.

---

## Editing hardpoints

- **Hardpoint tables** (Front / Rear Hardpoints panels): editable Name / X / Y / Z in mm,
  colour-coded by category (corner, ARB, heave, decoupled).
- **Direct Edit panel** (`gui/panels.py` `DirectEditPanel`): enable keyboard nudging, pick the
  corner and point, type X/Y/Z (0.00001 mm resolution), *Mirror F↔R* to apply the same delta to
  the other axle, *Set baseline* + *Show ghost* for a Δ-to-baseline readout and a grey overlay,
  step buttons, and constraint modes *Free / Link / Plane*. *Apply Changes* commits (and clears
  the undo history), *Discard* reverts.
- **Plane tilt:** rotate a whole actuation set about a chosen axis (pushrod, spring axis, rocker
  axis, plane normal, drop link, X, Y, Z) and pivot by an angle; snap buttons force the pin axis
  perpendicular to the plane, the actuation chain into the plane, and the pushrod onto the arm
  plane + 1 in.
- **Group move:** shift the *Spring set*, *ARB* or *Inboard arms* in / out / fwd / aft / up /
  down by a step as one undoable step.
- **Dimensions:** front and rear shock length (mount-to-mount).
- **Track / wheelbase changes** in Car Parameters move the outboard points and wheel centre; an
  option also pushes the inboard pickups so arm lengths are kept.
- **Motion panel:** *Apply Sag to Hardpoints* re-zeroes the drawn model at static compression,
  *Undo applied sag* restores the drawn points, *Go to static sag (0)* returns the slider.
- **Undo / redo:** Ctrl+Z / Ctrl+Y on hardpoint edits.

### Hardpoints per corner

| Group | Points |
|---|---|
| Control arms | `uca_front`, `uca_rear`, `uca_outer`, `lca_front`, `lca_rear`, `lca_outer` |
| Steering | `tie_rod_inner` (rack end), `tie_rod_outer` (steer arm) |
| Wheel | `wheel_center` |
| Actuation | `pushrod_outer`, `pushrod_inner`, `rocker_pivot`, `rocker_axis_pt`, `rocker_spring_pt`, `spring_chassis_pt` (direct damper: `damper_chassis_pt`, `damper_outer_pt`) |
| Bellcrank / control-arm ARB | `arb_drop_top`, `arb_arm_end`, `arb_pivot` |
| Heave T-bar / decoupled | additional per-axle sets (`front_heave`, `front_decoupled`, …) listed by `topology.required_hardpoints()` |

The rear toe link is modelled with the same `tie_rod_*` keys. The team's standing rule for the
2027 car is that the rear toe-link inner point *is* the aft LCA pickup (Rule 20); the net gates
it on the design file, not on arbitrary projects.

---

## Topologies

`vahan/topology.py` describes each axle by four choices:

| Enum | Values |
|---|---|
| `DamperActuation` | DIRECT, PUSHROD, PULLROD |
| `DamperMount` | UCA, LCA, UPRIGHT |
| `ARBType` | BELLCRANK, CONTROL_ARM, TBAR, NONE |
| `SpringConfig` | CORNER, HEAVE_TBAR, DECOUPLED |

`validate()` rejects: HEAVE_TBAR without a TBAR anti-roll device; DECOUPLED with DIRECT
actuation; a BELLCRANK ARB with DIRECT actuation. The default (`standard()`) is pushrod on the
UCA at the front, pushrod on the LCA at the rear, bellcrank ARBs, corner springs.

Status by topology, as exercised by the regression net's per-topology loop
(`test_one_model.py` CASES, lines 34-48: coplanarity where applicable, and the motion-ratio
graph must respond to the active spring's hardpoint):

| Case | Front/rear topology | Solver path | Status |
|---|---|---|---|
| pushrod | PUSHROD / UCA / BELLCRANK / CORNER | corner rocker, 1-DOF rocker solve | implemented |
| pullrod | PULLROD / LCA / BELLCRANK / CORNER | same kinematics, lower rocker | implemented |
| direct | DIRECT / UCA / NONE / CORNER | damper chassis ↔ arm or upright | implemented |
| control_arm | PUSHROD / UCA / CONTROL_ARM / CORNER | ARB drop link on the LCA | implemented |
| tbar_corner | PUSHROD / UCA / TBAR / CORNER | corner is `cradle_link` (no corner rocker); ride spring on the central bellcrank | implemented |
| decoupled | PUSHROD / UCA / BELLCRANK / DECOUPLED | `vahan/monoshock.py` twin-rocker 2-D Newton solve, cross-car heave + roll coilovers | implemented |
| heave_tbar | PUSHROD / LCA / TBAR / HEAVE_TBAR | `vahan/heave_tbar.py` third-spring rocker solver | implemented; the CASES table still carries a KNOWN-FAIL note ("kinematic graph not yet wired to the 3rd-spring solver"), but the motion-ratio graph is now fed from that solver (`_inject_heave_tbar_mr`) and the case passed on 2026-10-03 (0 known-fail counted) |

Notes and limits:

- `vahan/tbar.py` is a 1-DOF heave model for the T-bar; roll twist is handled by the torsion-bar
  stiffness in the dynamics, not by the kinematic solver.
- The ride (7-DOF) model and the ARB graph metrics run for the standard topology only.
- The previous README called one configuration "stable, fully-validated" and the rest "beta".
  That distinction had no backing in the code or tests and has been dropped: every topology above
  is held to the same net checks, and none is correlated against a physical car.

---

## Kinematics and the metrics catalogue

**Solver** (`vahan/solver.py` `SuspensionConstraints`): unknowns are the 12 coordinates of
`uca_outer`, `lca_outer`, `tr_outer` and `wheel_center`; equations are 11 squared-distance
constraints (rigid links and rigid upright) plus the drive equation `wc_z = wc0_z + travel`.
Newton–Raphson with an analytic Jacobian, tolerance 1e-10, at most 60 iterations, raises if it
does not converge. The pushrod outer point rides on the upright (or arm) by a rigid-body frame.
The rocker angle is a separate 1-D Newton solve (Rodrigues rotation about the plate normal) with
the branch picked by spring-length continuity; the rocker axis is forced to the actuation-plane
normal and `rocker_axis_pt` only sets its sign (Rule 02). The spin axis is +X at the design pose,
so the design hardpoints read exactly 0° camber and toe; static alignment is applied afterwards
as an offset.

**Motion modes:** heave, roll (wheel travel from the roll angle and track), pitch, steer (rack
travel from steering-wheel angle through `vahan/steering.py`, bounded by the physical rack
stroke). Steer sweeps re-solve the front corners at each rack position; the live Ackermann
value is taken from the FL/FR pair.

**Metrics** (`vahan/metrics_catalog.py` CATALOG, 38 entries):

| Category | Keys |
|---|---|
| Angles | camber, toe, caster, kpi, rocker_angle |
| Lengths | scrub, trail, spring_len, kp_len |
| Geometry | rc_height (per axle), ic_y, ic_z, roll_axis_incl, rc_lateral |
| Anti | anti_dive (front), anti_squat (rear), anti_lift (rear) |
| Half-shaft | halfshaft_len, halfshaft_lateral, halfshaft_angle |
| Wheel centre | wc_x, wc_y, wc_z |
| Ratios | motion_ratio (damper / wheel) |
| Steering | steer_angle, ackermann, turn_radius |
| ARB (standard topology) | arb_angle, arb_drop_travel, arb_mr |
| Topology | heave_spring_mr, roll_spring_mr, heave_spring_len, roll_spring_len (decoupled); third_spring_len, third_spring_mr (heave T-bar); tbar_twist, coil_len (T-bar) |

- The roll centre is built from the **instant axis**: each arm plane is traced on the transverse
  plane through the wheel centre from its pivot axis, the two traces meet at the front-view IC,
  and the IC-to-contact-patch lines of both corners meet the centre plane (Rule 17). Sliding a
  pickup along its own axis does not move it. Parallel arms report 0.
- Half-shaft length through travel *is* the plunge; the joint angle is the CV articulation to
  the spin axis (`vahan/driveshaft.py`).
- Graphs re-sweep on a 150 ms debounce after an edit; a generation stamp discards stale sweeps.
- Every graph surface is hover-readable; right-click copies the graph as displayed or in a
  light theme and saves PNG / PDF / SVG (`gui/plot_dialog.py`).

Analysis Plots panel (`vahan/analysis_plots.py`): brake-torque capacity, wheel-rate linearity,
LLTD, pitch over a bump, roll-centre height vs roll, steering torque, the Ackermann set
(demand, Fz–Fy map, pair analysis, slip vs load vs force), YMD trim sweep and MMD.

---

## Inverse kinematics and packaging

**Inverse kinematics** (`vahan/optimizer.py` `InverseSolver`, IK panel): choose an axle and
motion (heave / roll / pitch / steer), a target metric with a curve shape (linear, progressive,
digressive, exponential), the hardpoints and axes allowed to move, and a method:

- `staged` (default): metrics solved in sequence on orthogonal variable groups
  (motion ratio → pushrod/rocker; toe → tie rod; anti geometry → inboard Y; camber / RC →
  front-view X/Z; Ackermann → rack position), then a final polish.
- `hybrid` (several Levenberg–Marquardt starts), `local`, `global` (differential evolution).
- Targets: anti-dive, anti-squat, anti-lift, camber, bump steer (toe), Ackermann %, roll-centre
  height, caster, caster trail, motion ratio, ARB motion ratio.
- A regularisation term and optional tube-collision residuals keep solutions buildable; the
  Rule 01/02/04 chain metrics are checked on the result. *Find Solutions* widens the bounds
  (2×, 4×, 7×, 10×) and lists alternatives.

**Packaging page** (Ctrl+7, `gui/packaging_page.py`, `vahan/packaging.py`):

- *Manual:* mirror about the pushrod plane, rotate about the pushrod line, translate, scale the
  rocker lever, then *Re-tune MR + ARB to baseline*, *Revert axle*, *Save experiment…*
  (writes `configs/experiments/`), re-capture the baseline, and edit the *Tolerances* every
  held parameter is judged against (defaults: camber/caster/KPI/toe 0.05°, bump steer 0.02°,
  scrub/trail 1 mm, RC 2 mm, motion ratio 1 %, ARB rate 2 %, coplanarity 3 mm, ARB in-plane
  3 mm, triad 1°, clash may worsen by at most 0.25 mm).
- *Generator:* random candidates on a grid with a keep-out plane; each survivor passed
  `packaging.validate`, the single validity oracle (wheel curves, Rules 01/02/03/04, MR and ARB
  rate within tolerance, clash sweep at droop/static/bump).
- *Relocate (IK):* curve-preserving relocation of one point (`vahan/relocate.py`): ray bisection
  in 26 directions, ellipsoid sampling, farthest-point diversity; the live model is always
  restored. A "hoop-line" search (damper mount on the front-hoop line) exists in
  `vahan/relocate.py` but is reachable only through the MCP server, not a button.
- `packaging.full_state_audit` sweeps 39 states (droop / static / bump × 13 rack positions)
  across all corners; `packaging_qd.py` runs a MAP-Elites search over the rocker/ARB genome;
  `force_opt.py` minimises pushrod off-tangency with the motion ratio held.

**Design City** (Ctrl+3, `design_city.py`): enumerates chassis-side alternatives per axle from
the current design, keeps only those that hold every one of ~106-112 parameters within 0.1 %
(`packaging.parameter_vector` / `compare_parameters`, relative tolerance 1e-3 with per-parameter
floors) plus the geometry laws, zero clash negatives and the full-state audit, renders each
through the real 3-D view and groups them by complete-linkage clustering (default cut 20 mm).
Output goes to `designs_city/<run>/` (gitignored). `py design_city.py --trials 300 --workers 4`.

The team's standing geometry rules, each with the function that enforces it, are in
`docs/suspension_rules/` (Rules 00-21). Rules 07 (ARB rate preserved) and the old form of
Rule 11 are checked by the packaging validator but had no net gate as of 2026-09-21
(`docs/GUI_TASKS.md`); a Rule 11 gate for "both pushrods on the upper arm" was added on
2026-09-23.

---

## Interference, chassis bays, keep-out, rim fit, joint articulation

`vahan/interference.py`:

- Every member is a **capsule** (segment + radius). A pair clashes when the centreline distance
  minus both radii is below the margin: 1.0 mm within a corner, 3.0 mm across corners. Pairs that
  share an endpoint (within 6 mm) or are declared connected are skipped; the rear LCA and toe
  link count as one welded part.
- Bodies: the four arm legs, tie/toe rod and its outer joint, pushrod and its outer joint, upper
  and lower ball joints, coilover at the declared spring OD, ARB drop link, rod ends, rocker
  pivot bearing, the UCA chassis cross member, the ARB torsion bar and blades, the driveshaft, and
  the **rocker plate** as a 6 mm prism (members passing through it light up; the net requires
  3 mm plate clearance for the ARB drop link).
- Assumed sizes are stated in the code (0.625 in arm tubes, 1 in ball-joint spheres, 1.5 in rocker
  bearing, 0.315 in rod ends, 1 in UCA cross member, 1.5 in rack housing). They are inputs to a
  clearance check, not measured parts.

**FSAE chassis bays** (`vahan/chassis.py`, View → FSAE chassis bays): frame nodes 38.1 mm
(1.5 in) beyond each arm pickup along its leg, 25.4 mm tubes, one diagonal per bay (auto or
explicit), rear transverse tubes, and the sprocket disc / differential envelope from an imported
STEP as fixed obstructions. What is drawn is exactly what the clash set checks.

**Keep-out** (`vahan/keepout.py`, Rule 18): a planar-faced STEP solid named by the project is a
hard volume drawn as a translucent red block; the audit covers droop / static / bump × lock /
centre / lock. Curved faces are rejected; the distance is a conservative under-estimate near
edges.

**Rim fit** (Rule 16): `kinematics.rim_fit` checks joint *centres* against a clear circle and is
explicitly not the answer; `interference.rim_barrel_gap` checks bodies against an open cylinder
with lip edges, and `wheel_profile.member_clearance` checks against the real inner profile from
the manufacturer's STEP (spokes treated as a solid disc). The net requires 3 mm.

**Joint articulation** (`vahan/spherical_bearings.py`, Bearings page): for each inboard
control-arm pickup the arm rotation is split exactly into swing and twist for two bolt
orientations (normal to the arm plane, or along the pivot line). The built-in rod-end tilt is
checked against a 27° swivel limit (editable); the worst tilt over travel is reported but not
flagged.

**Front hoop line** (Rule 19): `packaging.front_arb_hoop_line_gap_mm` keeps the whole front ARB
3 mm ahead of the line through the aft pickups.

---

## Springs, dampers, motion ratio and anti-roll bars

- **Motion ratio** is damper travel / wheel travel from the solved pose (`motion_ratio` metric);
  the Dynamics panel reads it from the kinematics.
- **Wheel rate** = k·MR² + F_spring·dMR/dx, i.e. the geometric term of RCVD §16.3 is included
  (`vahan/dynamics.py` 318-346); **ride rate** is the wheel rate in series with the tyre rate.
- **Spring stroke and sag** come from the shock length, stroke and preload (Motion panel): the
  travel range is limited by the stroke, *Apply Sag* re-zeroes the model at static compression,
  and Rule 14 ("set rate with spring/MR, set sag with preload") is the operating rule.
- **Anti-roll bar rate** (Dynamics panel, `gui/panels.py` 3380-3383): torsion of the bar
  (G·J/(A²·L) at the arm tip) in series with the arm's bending (3·E·I/A³ for a blade section),
  then through the ARB motion ratio from the kinematics to a wheel rate; hollow bars by OD/ID.
  Roll stiffness per axle = (K_wheel + K_arb)·t²/2; decoupled and heave-T-bar modes have their
  own expressions.
- **Dampers:** linear bump and rebound coefficients (N·s/m at the shock, Skidpad panel) build
  the transient model's roll damping (Σ(c_bump + c_rebound)·MR²·t²/4 per axle); the ride model
  takes four wheel-frame damping coefficients as its own inputs. There is no digressive or
  velocity-dependent damper curve.
- **Damper sign** (Rule 05): the net checks that the damper compresses monotonically in bump.

---

## Steady-state dynamics

`vahan/dynamics.py` `SteadyStateSolver` solves the car at a lateral + longitudinal g with the
kinematics in the loop:

1. Static loads, pitch transfer m·a_x·h/L and optional aero Fz.
2. Each pass: roll → per-corner travel → corner kinematic solve → roll-centre heights and
   camber → load transfer → new roll; converges when Δroll < 0.002° (max 15 passes).
3. Load transfer has three parts: geometric (RC height / track), elastic (roll moment ×
   axle share of roll stiffness / track) and unsprung (m_u·a_y·h_u / track). **LLTD** is the
   front share of the total.
4. Roll: φ = m_s·a_y·h / (K_total − W_s·h), including the gravity term and saturation.
5. Tyre forces: front/rear Fy from yaw equilibrium, left/right split by looking each wheel up in
   the nonlinear tyre surface (Milliken pair analysis); Fx to the driven axle or by brake bias;
   utilisation = combined demand / (μ_peak(Fz, IA)·μ_scale·Fz).
6. **Jacking** force per corner along its jacking line; feeding it back into the kinematics is
   off by default (`car['jacking_feedback']`).
7. **Anti-dive / anti-lift / anti-squat** fractions come from the metrics catalogue at static;
   pitch travel per axle = m_s·a_x·h_s/L·(1 − anti) through the heave curve, on by default.
   Load transfer under braking/acceleration is independent of the anti geometry; the anti
   geometry changes how much of it goes through the springs (and so the squat/dive travel).
8. **Aero:** downforce = C_L·A·½ρV² split by centre of pressure; aero sink through the heave
   curve is on by default. `AeroDownforceSolver` gives the per-corner Fz needed to bring
   utilisation to a target at a given g, by bisection; *Apply Aero* feeds it back V²-scaled;
   a *Custom (CFD validation)* input takes F_ref / V_ref / CoP and back-calculates C_L·A.
   Vahan has no aerodynamics model of its own.
9. Sweeps over lateral g, longitudinal g, combined, speed and acceleration, with a secondary
   speed axis from the turn radius. Outputs per corner: Fz, Fy, Fx, travel, camber, utilisation,
   brake torque; scalars: roll, pitch, understeer gradient, LLTD.
10. **Grip multiplier** (`_mu_scale`): one project-wide scale on every tyre limit, used by the
    team as a belt-to-asphalt derate. It is a user input, not a measured value.

`DynamicsSensitivity` gives central finite-difference sensitivities of understeer, roll, pitch,
LLTD, utilisation and ideal Ackermann to springs, bars, CG, brake bias and motion ratio, and a
recommendation list of which knob reaches a target delta.

Assumptions stated in the code: rigid chassis and links; unsprung CG at wheel-centre height;
wheel/engine inertias and shift timings are "ASSUMED, NOT MEASURED"; a `LinearTireModel` is
substituted when no tyre file is loaded.

---

## Tyres and TTC data

`vahan/tire_model.py`:

- **`TireModel`** builds Fy(SA, Fz, IA) and Mz surfaces from test data by binned medians and a
  cubic spline on a regular grid. A Magic-Formula (MF 5.2 pure lateral) fit across loads locates
  the peak-slip line so the force table is clamped non-increasing past it.
- **Inputs:** a TTC `.mat` (channels SA, FZ, FY, MZ, MX, IA, P, V, plus the test IDs) or a CSV /
  XLSX with those columns (SA, FZ, FY, IA required; `pandas` needed). The user must pick **one
  pressure**; blending is refused and the file's pressures are listed. An optional test-speed
  window (`car['tire_speed_window_kph']`) keeps one conditioning block after the warm-up discard
  (default 1500 samples). Camber levels with thin coverage are dropped; the load axis is the
  test's own set-points.
- **Outputs:** `Fy`, `Mz`, `peak_Fy`, `peak_mu`, `cornering_stiffness`, `slip_angle_for_Fy`,
  `peak_slip_angle`, a data-range report that records every out-of-range excursion.
- **`LinearTireModel`** is the parametric fallback (load-sensitive C_α and μ, optional camber
  thrust, saturation at μ·Fz). Every screenshot in this README was taken with it.
- A split front/rear tyre is supported. The Dynamics panel's *Tire / Grip Plots* button opens
  Fy(α) families, cornering stiffness vs load, Mz(α) and the per-corner friction circle.

Limitations: no transient (relaxation-length) tyre behaviour; no combined-slip surface, the
friction circle combines Fy and Fx demand against a single μ_peak; the TTC rig sweeps ±12° so
the decline past the peak is partly extrapolated; above the tested load the surface is
extrapolated and flagged. No tyre data or tyre identity ships with this repository; see
[Data policy](#data-policy).

---

## Component loads, brakes and bearings

`vahan/loads.py` (Loads page, Ctrl+4, and the Component Loads panel):

- **Free body:** wheel + tyre + hub + rotor + bearings + upright + caliper, with the contact-patch
  force, m_u·(g − a) at the wheel centre and the half-shaft torque where a shaft carries it.
  With the pushrod on the upright the six two-force members are solved as a 6×6 system; with the
  pushrod on an arm an arm-body sub-solve carries a 3-D ball-joint force and reports leg shear and
  bending. Direct-damper corners are supported.
- **Validity:** a condition-number limit of 1e3; above it the result is NaN and flagged invalid.
- **Also:** rocker moment balance and the bellcrank/ARB free body (drop link, bar torsion, pivot
  reactions), corner moments, brake pressures / torques / lockup (tyre model required), and a
  single adiabatic stop for rotor temperature rise (100 % of kinetic energy into the rotors, no
  cooling).
- **Load cases** (`gui/wheel_package.py`): 2.0 g cornering, 1.6 g braking, 1.0 g acceleration,
  1.4 g lateral + 1.0 g braking, plus the current dynamics state; cornering cases use the turn
  radius speed, straight-line cases the aero reference speed.
- **Output:** the table, the hoverable 3-D arrows, and a CSV / text export (`Parameter, Unit,
  FL, FR, RL, RR`: patch forces, axial force in every link, leg shear, ball-joint V/H, wheel
  bearing inner/outer V/H, caliper bolt V/H, brake torque / clamp / line pressure, validity
  flags and condition number).

What these loads are and are not: quasi-static (wheel I·α neglected), from rigid geometry, at the
operating point you choose. They are joint reactions suitable as hand-sizing inputs or as
boundary loads you apply yourself in an FEA tool. **There is no FEA export, no FEA solver, and
no claim that a given load set is complete for a particular part** — fatigue, impact, kerb
strikes and compliance are outside the model.

**Bearings page** (Ctrl+0): for each inboard pickup of both arms, front and rear, per load case:
radial and axial force (kN), oscillation half-angle from the exact swing/twist split,
oscillation time (defaults to 1 / ride frequency), load direction, temperature and the built-in
tilt against the 27° limit. *Copy for SKF* copies tab-separated rows for the SKF plain-bearing
calculator. No bearing-life (L10) calculation is performed.

---

## Ackermann, yaw-moment diagrams and steering

`vahan/ymd.py` is the one yaw-moment engine: a double-track model with per-wheel loads from the
steady-state solver tabulated over lateral g, yaw-rate slip terms, static toe and the Ackermann
split, forces resolved in the body frame (N = Σ x·Fy − y·Fx, with the induced-drag couple and the
aligning moment Fy × pneumatic trail), in constant-radius or constant-speed mode. `trim_point`
finds N = 0; `mmm_metrics` gives N_δ and N_β; `trim_sweep_ackermann` sweeps the Ackermann
setting. The stability and control derivatives are read at a stated sub-limit point: at maximum
trim N_δ is zero by construction.

**Ackermann page** (Ctrl+5, `vahan/ackermann.py`): the geometric demand (wheel heading = travel
direction + slip angle inverted from the tyre surface at each wheel's load share, answered in
degrees), the force-ceiling sweep of front-axle capability vs setting with tie bands, the full
MMD per setting, the YMD grid (point T trimmed g, stability index, control moment, limit
character) and per-station lap effect. Convention: 0 % = parallel steer, 100 % = turn centre on
the rear-axle line, negative = reverse Ackermann; an unqualified percentage is quoted at full
lock. `vahan/ackermann_report.py` writes a self-contained HTML + PNG report (to the gitignored
`figs/` by default) with a self-consistency "PREMISE FAILED" box.

**Steering** (`vahan/steering.py`): steering-wheel angle → rack travel (mm per turn) →
road-wheel angle, from a dense probe of the kinematic solver with a linear-ratio fallback;
saturation at the rack limit; steering effort by virtual work (kingpin moment × dδ/d-rack). The
steering system is rigid: no compliance, no backlash; the Ackermann results carry that caveat in
the UI.

*The Ackermann page screenshot is not in the repository because it shows tyre-derived curves.*

---

## Ride, road and transient models

**Ride page** (Ctrl+8, `vahan/ride.py`, `ride_solve.py`, `road.py`):

- A linear **7-DOF** model (heave, pitch, roll, four wheel verticals), standard topology only;
  inertias and the four wheel-frame damping coefficients are user inputs labelled ASSUMED.
  Bilateral FFT response; optional unilateral tyre contact in the time domain; bump stops are
  not modelled.
- Roads: ISO 8608 **classes A and B only** (class-centre G0 = 16e-6 and 64e-6 m³ at 0.1 cycles/m,
  waviness 2), synthesised with random phases, wheelbase delay and selectable left/right
  coherence. These are roughness scenarios, not measured track surfaces.
- Ride solve: dynamic load coefficient, contact-loss indicator, body-acceleration RMS, travel and
  damper-velocity peaks against the bump-stop margin, half-sine bump settling with a flat-ride
  ratio; a front × rear ride-rate grid is swept, converted to spring rates and snapped to the
  nearest standard 25 lbf/in spring with preload. Also contact-patch-load vs ride frequency and
  launch load-lag studies.

**Transient / skidpad panel** (`vahan/transient.py`):

- Bicycle + roll model integrated by RK4; integrated states are vx, vy, yaw rate, roll, roll
  rate, X, Y, heading and the actual steer through a first-order actuator lag (τ = 0.02 s
  default). Elastic load transfer lags through the roll dynamics; camber and RC migration come
  from a 25-point travel lookup built at start.
- Inputs: constant / step / ramp / sine steer, FSAE skidpad (single circle or figure-8) with a
  closed-loop Stanley path follower, speed hold (PI) or open-loop Fx (clamped at 1.5 g).
- Outputs: yaw-rate rise / overshoot / settling, peak and steady lateral g and roll, and the
  full time history of any signal.
- No tyre relaxation length; longitudinal control is a constant-Fx mode.

---

## Lap time, engine, corner speed, acceleration

**Lap Time** (Ctrl+2, `vahan/laptime.py`): quasi-steady, three passes (corner-speed ceiling,
forward acceleration, backward braking; v = the minimum). The lateral ceiling is a grip table
built by bisecting the **full** steady-state solver to peak utilisation = 1 at nine speeds,
including the aero split; the point-mass μ·(g + D/m) is only a bracket. Powertrain: gear-resolved
wheel force from the tabulated crank torque (or a torque-plateau / constant-power model),
rotating inertia as equivalent mass, a shift model (torque cut, minimum interval, hysteresis).
Braking is tyre-limited with no brake-torque limit. Optional per-station Ackermann cap and scrub
drag, and a detailed pass that runs the full solver at sampled stations (Fz, roll, LLTD, …).
Tracks are JSON centrelines (`tracks/autocross26.json`, `autocross_2025.json`,
`mi2018_endurance.json`); `tools/trace_track.py` traces one from an image and
`tools/convert_oltra.py` reads an OptimumLap `.OLTra` file (turn direction is not stored there).

**Engine** (Ctrl+6, `vahan/engine.py`): the torque curve comes from the external
[1dFVEngineSolver](https://github.com/NIXELFi/1dFVEngineSolver) (Sun Devil Motorsports, MIT),
read from `vahan/data/engine_curve_sdm26.json` / `engine_sweep_sdm26.json`. Calibrations: `raw`
(known ~2× low), `corrected` (default: volumetric-efficiency target + friction line), `anchored`
(scaled to a stated crank peak), plus `anchor_curve_to_dyno()` for when a dyno pull exists. The
module says it plainly: **simulated, not dyno-validated; absolute level approximate.** A generic
restricted-600 fallback curve is labelled "ASSUMED" on the plot when it runs.

**Corner Speed** (Ctrl+9, `vahan/corner_speed.py`): the trimmed (N = 0) maximum lateral g per
radius (50 m down to 4.5 m plus the steering-lock radius) from the same `trim_sweep_ackermann`,
with speed, body slip, front steer and dN/dβ, with and without aero; and the per-corner grip
budget (first g at which any single tyre exceeds its friction circle, against the axle-aggregate
limit). *Copy table* exports tab-separated text. The net checks the page's rows equal a direct
`vahan.ymd` call.

**Acceleration** (`vahan/acceleration.py`): gear-resolved tractive force capped by rear grip with
longitudinal transfer and aero, explicit Euler at 5 ms to 150 m, reporting the 75 m time and top
speed. It is a library + net feature; there is no page for it yet.

**Differential** (`vahan/differential.py`): open / spool / Salisbury with Drexler ramp-angle
lock tables (three power/coast options); steady torque bias, no clutch-slip transient. The
yaw moment is capped by the inner driven wheel's grip.

---

## Build tolerance and sensitivities

**Build Tolerance** (Ctrl+Shift+1, `vahan/cg_tolerance.py`): a deterministic one-axis CG sweep
(height, or fore-aft) over a user span and step, rebuilding the solver and lap sim at each
offset; the tolerance band is where each metric crosses its allowance (grip limit g, front LLTD
share, roll per g, understeer, traction g at 20 km/h, braking g at 60 km/h, lap time), with the
slope per 10 mm. The AERO HEAVE tab sweeps travel, ride-height drop and aero sink from 0 to
130 km/h. This is not a Monte Carlo build study; hardpoint-position tolerances are not swept.

The Dynamics Opt panel's finite-difference sensitivities are described under
[Steady-state dynamics](#steady-state-dynamics).

---

## CAD and other exports

**Onshape** (View → All Hardpoints → *Copy for Onshape*): a pipe-delimited block that the bundled
`VahanHardpoints.fs` FeatureScript turns into labelled construction points per corner (plus
diff, tripod and driveshaft points) in a Part Studio, with per-corner show/hide. The team's copy
of the feature is in the
[VahanHardpoints Feature Studio](https://cad.onshape.com/documents/0fd1ba4fa3000364cc5e975c/w/c68fedaa2bfec6c13cd02fce/e/19856db1245e96443584ccac);
the source is in this repo. `VahanLayout.fs` is a blank labelled sketch workspace for one corner.
`docs/ONSHAPE_TO_SOLIDWORKS.md` is the team's recipe for moving those points to SolidWorks via
STEP (Y-up export, "import free curves and points as sketch").

**SolidWorks:** *Copy for SW (X lateral)* and *Copy for SW (Front = front, head up)* copy the
same CSV in SolidWorks' Y-up frames; *Export SW equations…* writes `"name"= value mm` global
variables to `vahan_hardpoints.txt`. `tools/sw_link.py` (`setup` / `update` / `selftest`, needs
`pywin32` and a running SolidWorks) drives a 3-D sketch from those globals — **it has never been
run against a live SolidWorks** (`HANDOFF.md`); treat it as a draft.

**STEP import** (File → Import STEP, `vahan/step_import.py`, `gui/step_import_dialog.py`):
STEP → mesh through `cascadio` + `trimesh`, source frame SolidWorks (Y-up) or Onshape (Z-up),
offset and flip; parts are drawn, used as fixed obstructions in the chassis-bay clash checks,
and stored in the project as base64 meshes. View → Manage STEP parts: show/hide, move, flip,
colour, opacity, rename, remove. *Export moved part…* needs `cadquery-ocp`.

**Other exports:** Export Report… (`.docx`: 3-D screenshot, parameters, heave and roll
kinematics, cornering sweep, acceleration/braking trajectories, loads; runs in a background
thread), Engineering Review… (Markdown built from live values), Export Sweep Data (CSV: every
metric × four corners vs the sweep axis, plus the last dynamics sweep), graph images (PNG / PDF /
SVG) and clipboard copies, the Loads CSV, Bearings and Corner Speed tab-separated copies, the
Ackermann HTML report, `capture_views.py <config.vahan> [out_dir] [tag]` (iso / front / rear /
interference renders, needs native GL) and `tools/gen_screenshots.py`.

**MCP server** (`gui/mcp_server.py`): with the `mcp` package installed the running app serves
streamable-HTTP MCP on 127.0.0.1:8765 (`VAHAN_MCP_PORT` to move, `VAHAN_MCP=0` to disable). Every
tool is marshalled onto the GUI thread so it behaves like a user action. Tools: `status`,
`load_config`, `get_hardpoints`, `set_hardpoint`, `axle_metrics`, `relocate`, `hoopline`,
`apply_hoopline`, `save_experiment`, `screenshot`, `present_hoopline`,
`present_hoopline_ghosts`, `present_relocate_ghosts`, `set_note`. Register with
`claude mcp add --transport http vahan http://127.0.0.1:8765/mcp`.

Not present: any FEA, CFD or multibody export, any live Onshape API link, a single-file
executable (listed in `docs/ROADMAP.md` as future work).

---

## Project files and versioning

A project is one JSON file, usually `.vahan` (`_project_to_dict`, `gui/main_window.py`):

| Block | Contents |
|---|---|
| `version` | 3 |
| `front_hp`, `rear_hp`, `front_arb`, `rear_arb` | required; every point a 3-vector in **metres** |
| `front_heave`, `rear_heave`, `front_decoupled`, `rear_decoupled` | topology extras, `{}` when unused |
| `car`, `steer`, `alignment`, `topology` | vehicle parameters, rack, static alignment, per-axle topology |
| `motion` | type, min, max, stroke, preloads, fully-extended length |
| `panels` | dynamics, skidpad, loads, aero, brake_calc panel state |
| `imported_parts` | STEP meshes as base64 |
| `sag_backup` | drawn-hardpoint backup from the last Apply Sag |

- **Save is atomic:** serialise, validate, write and fsync a temp file, re-read and validate
  that, then replace. A failure leaves the previous file untouched and says "Project was NOT
  saved".
- **Load is transactional:** the file is schema-checked first; the live project is snapshotted
  and restored if the load throws. Older v1/v2 files load with defaults for missing blocks.
- **Versioning convention** (team workflow, `CLAUDE.md`): a new set of hardpoints is saved as
  the next `configs/2027_v<N>_(what_changed).vahan`; experiments go in `configs/experiments/`.
  The regression net, Design City and the capture tools pick the highest `2027_v<N>`
  automatically. `configs/` is gitignored — the design is not in this repository — and there
  is no in-app branching, diffing or merging of versions: version control of designs is the
  numbered files plus the binder's design record.

---

## Screenshots

Screenshots and plots are not distributed. Local captures may contain private design or licensed tyre results and must not be published.

---

## Testing

### Regression net: `python test_one_model.py`

Offscreen (`QT_QPA_PLATFORM=offscreen`), Python 3.12+. It is, in its own words, "a NET, not
verification": every block prints a labelled line ending in `pass`, `UNEXPECTED FAIL` or
`KNOWN-FAIL: <reason>`; the exit code is the number of unexpected failures. About 104 labelled
blocks plus a seven-topology loop (coplanarity where applicable; the motion-ratio graph must
respond to the active spring's hardpoint). Blocks cover, among others: packaging plate sweeps,
ARB closure and triad, rim-fit and real wheel profile, keep-out, front hoop line, Rules 04 / 20 /
21, steering effort and direction, wheel-rate dMR term, camber and caster signs, tyre pressure /
camber-row / low-load honesty, Ackermann pair / solver / trust / ceiling, YMD trim, brake torque
and bias, static sag, physical dampers, save round-trip and atomic save, load-transfer
invariants, loads free bodies and sign contracts, bearings page and rod-end swivel, jacking,
corner speed, build tolerance, pitch + aero sink, "rc sphere = graph", FSAE chassis bays, and
the dynamics-camber unit tests.

**What it needs.** Most design-level gates load the highest `configs/2027_v*.vahan`
(`VAHAN_DESIGN=<path>` pins another). Since `configs/` is private, **a public clone does not
exit 0**. Recorded run on this clone (Python 3.12.3, no `configs/`, no tyre data, 2026-10-03):

```
19 unexpected failures, 0 known-fail (documented).
```

All seven topology cases passed (coplanarity and motion-ratio connectedness). The 19 failures
are design gates that cannot open a design file (rule 20, rule 21, real wheel profile, 3D view
rim/ground, stale sweep guard, roll/ARB sanity, IK integrity setup, jacking, dyn camber pinned,
steer inputs sync, bearings page, build tolerance, pitch + aero sink, turn radius + MR, rc sphere
= graph, fsae chassis bays) plus three blocks (packaging, relocate IK, spring support) that error
on variables the skipped design blocks would have defined. Twenty-odd further blocks print
"no config — skipped" or "no TTC file — skipped". On the team's machine with the design file and
tyre data the net takes roughly four minutes and the design gates run for real.

**Rules of the net** (`CLAUDE.md`): done means the net exits 0 **and** the diff is reviewed;
every solver fix adds a failing-then-passing check; an unfixed issue is a KNOWN-FAIL with its
reason, never a deleted check.

### Other tests

| File | What it checks |
|---|---|
| `test_dynamics_camber.py`, `test_dynamics_camber_ui.py` | camber frames (chassis / road / tyre) through the dynamics sweep and the panel display; no private data |
| `test_ride.py` | analytic checks of the vertical road model (unittest) |
| `test_road_plane_camber.py` | road-plane inclination maths |
| `test_rule04_strict.py` | physical rocker-plate Rule 04 gate |
| `test_double_shear_rocker.py` | double-shear plate geometry and clearances |
| `test_pushrod_envelope.py` | the arm-side spherical joint must not vanish behind a thin tube |
| `test_steering_direction.py` | signed rack linkage and the saved direction |
| `test_redesign_review.py` | the engineering-review text builder |
| `test_tire_selection_provenance.py` | needs local TTC files; skips otherwise |
| `super_smoke.py` | redundancy / solver / physics smoke across topologies; exit code = failures |

---

## Development and architecture map

```
app.py                       entry: QApplication + wheel guard + gui.main_window.launch()
vahan/                       pure computation, no Qt imports; usable as a library
  hardpoints.py  solver.py  kinematics.py  metrics_catalog.py  analysis.py  steering.py
  topology.py  heave_tbar.py  tbar.py  monoshock.py  ik_decoupled.py  optimizer.py
  packaging.py  packaging_qd.py  relocate.py  force_opt.py  interference.py  chassis.py
  keepout.py  wheel_profile.py  spherical_bearings.py  heave_curve.py
  dynamics.py  tire_model.py  ymd.py  ackermann.py  ackermann_report.py  loads.py
  ride.py  ride_solve.py  road.py  transient.py  aero_ride.py
  laptime.py  engine.py  acceleration.py  corner_speed.py  differential.py  driveshaft.py
  cg_tolerance.py  redesign_review.py  report_gen.py  analysis_plots.py  step_import.py
  data/                      engine curves from 1dFVEngineSolver (tracked)
gui/                         PyQt6
  main_window.py (~13.6k lines: workspace, wiring, sweeps, project I/O)   panels.py (~8k: sidebar panels)
  view3d.py (VisPy view + NavCube)   plot_dialog.py   section_info.py   startup_dialog.py
  laptime_page.py  city_page.py  loads_page.py  ackermann_page.py  engine_page.py
  packaging_page.py  ride_page.py  corner_speed_page.py  bearings_page.py  build_tolerance_page.py
  wheel_package.py  wheel_guard.py  step_import_dialog.py  task_list_dialog.py  mcp_server.py
docs/                        DESIGN.md (UI tokens), GUI_TASKS.md (backlog, editable in-app),
                             MEASURE_MODE_PLAN.md, ROADMAP.md, ONSHAPE_TO_SOLIDWORKS.md,
                             suspension_rules/ (Rules 00-21 + index)
tools/                       sw_link.py  trace_track.py  convert_oltra.py  gen_screenshots.py
tracks/                      lap-sim centrelines (JSON)
examples/fsae_front.py       minimal library sweep (older axis convention, see Conventions)
test_*.py, super_smoke.py    see Testing
```

Working rules the code is written to (from `CLAUDE.md`, which is also checked in):

- **ONE MODEL:** the 3-D view, every graph and the dynamics derive from the single solved model;
  a kinematics change usually touches the solver, `gui/panels.py` and `gui/view3d.py` together.
- `gui/main_window.py` and `gui/panels.py` are too large to read in full; grep for `class` /
  `def` and read ranges.
- Solver fixes add a net check. Done = net exits 0 and `git diff --stat` reviewed; numeric and
  visual verification of the actual model output is still required.
- UI and plot colours are chosen for a colour-blind user: yellow / red / white / blue; never a
  red/green or purple/blue contrast; yellow is banned from text (`docs/DESIGN.md`).
- Calculations for the team's design run headlessly **through the app** (MainWindow offscreen,
  or the MCP server), never through a separate script with its own physics.
- An optional knowledge graph (`graphify-out/`, gitignored) can be regenerated with the
  `graphify` tool for navigation; it is not required.

Team handoff documents are private and not distributed with the software.

---

## Contributing

- Open an issue or PR on GitHub. Keep `vahan/` free of Qt imports and keep physics in one place:
  a page or panel loads, calls and plots; it does not re-derive a quantity another module owns.
- Run `python test_one_model.py` (expect the design-file gates to skip or fail on a public clone,
  see [Testing](#testing)) and the unit tests you touched. Add a failing-then-passing check with
  any solver fix.
- Never commit tyre data, tyre-derived plots, `configs/`, `DESIGN_2027/`, `BINDER/` or
  `external/`; `.gitignore` already blocks them.
- Follow the colour rules in `docs/DESIGN.md` for any new UI or plot.
- Document promised-later features in `docs/GUI_TASKS.md` (one line each) rather than in code
  comments.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: vispy` | install `vispy` (now in `requirements.txt`) |
| `ImportError: libEGL.so.1` on Linux | `apt-get install libegl1 libopengl0` (Qt/VisPy need the system GL stack even offscreen) |
| `SyntaxError: f-string: expecting '}'` in `test_one_model.py` | Python < 3.12; the net needs 3.12+ |
| The net reports many UNEXPECTED FAILs on a fresh clone | no `configs/` design file; see [Testing](#testing) |
| STEP import dialog says the feature is disabled | `pip install cascadio trimesh`; re-export needs `cadquery-ocp` |
| "Blending pressures refused" when loading a tyre file | set *Tyre pressure* (psi) in the Dynamics panel to one of the pressures the loader lists for the file |
| Tyre file not found on load | `car['tire_file']` points to a local path; set it in Car Parameters |
| Geometry errors in the status bar (e.g. ARB linkage impossible) | the corner cannot close at that pose; the 3-D view suppresses the impossible link rather than drawing a fake one |
| Code change not visible | the app has no hot reload; restart `python app.py` |
| Port 8765 in use | `VAHAN_MCP_PORT=<port>` or `VAHAN_MCP=0` |

---

## Data policy

- This repository tracks the software only. The 2027 design (`configs/`), the design binder
  generator (`DESIGN_2027/`, `BINDER/`), vendored third-party sources (`external/`), reference
  spreadsheets and PDFs are gitignored and absent from public clones.
- FSAE Tire Test Consortium data is licensed to the team and is **never** redistributed: no
  `.mat` files, no tyre identities, no tyre-derived plots or screenshots. Load your own TTC
  files under your own access rights. De-identification does not authorize redistribution.
- Default vehicle parameters in the panels are one team's editable example inputs.
- The team's design binder is **team-only**. Do not publish it, link to it from public software
  documentation, or redistribute its figures or design results.
- TTC raw data, tyre identities, fitted surfaces and tyre-derived figures/results are **not for
  public sharing**, including through documentation, screenshots, exports or repository history.

---

## Credits and license

- **Suspension and vehicle dynamics, software:** Yu — Cougar Racing.
- **Engine torque curves:** generated with [1dFVEngineSolver](https://github.com/NIXELFi/1dFVEngineSolver),
  a 1-D finite-volume engine simulator by Sun Devil Motorsports (Arizona State University FSAE),
  MIT licensed, used unmodified; its outputs are simulated, not dyno-validated.
- Vahan is MIT licensed — see [`LICENSE`](LICENSE), Copyright (c) 2026 Cougar Racing. It is
  provided "as is"; the user assumes all risk for component failures, incorrect calculations or
  faulty data.
