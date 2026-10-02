# Measure mode — design plan (2026-10-01, planning only, no code yet)

What the user asked for: a new 3-D view mode, **Measure**, that works like the measure tool in a
CAD package: hover a point / line / plane, click two of them, read the distance or angle live;
and every imaginary line and plane the solver uses (arm pivot axes, kingpin axis, rocker axis,
actuation planes, instant centres, roll centre, roll axis, pitch axis, ...) can be switched on,
seen, picked and measured. Everything is built from the ONE solved model the view already draws
(`gui/main_window.py:_assemble_corners_draw` L6736 -> `View3D.update_scene` L1316); nothing is a
second geometry.

Frame: X lateral (outboard +), Y longitudinal (rearward +), Z up; metres inside, mm/deg shown.

## 1. Entity catalogue

Every entity has a stable ID (`FL.point.uca_outer`, `front.line.roll_axis`, `RL.plane.uca`),
a family (point / line / plane), a corner or axle scope, and a builder that reads the solved
pose. "exists" = the function already computes it today; "new" = write it in `vahan/measure.py`.

### 1.1 Points (per corner unless noted)
| ID | What it is | Source today |
|---|---|---|
| hardpoints (14) | every key in `HP_NAMES` (`gui/view3d.py` L626) at the current pose | `pts` dict from `_state_to_pts` (`main_window.py` L567); chassis ones static |
| rocker_axis_pt, damper_chassis_pt, damper_outer_pt | the optional hardpoints | `vahan/hardpoints.py` L90-96, same `pts` dict |
| arb_drop_top, arb_arm_end_world, arb_pivot | ARB drop-link top (moves), blade tip, bar pivot | `_assemble_corners_draw` injects the first two into `pts`; `arb_pivot` from `self._front_arb/_rear_arb` |
| wheel_center | hub centre | `SolvedState.wheel_center` (`vahan/solver.py` L705) |
| contact_patch | wheel centre dropped to the ground plane along Z | new; same rule `roll_center_height` uses (`vahan/kinematics.py` L216: X = wc_x, Z = 0) and the view's ground (`view3d.py` L1335: wc_z - tyre radius) — see risk 7.3 |
| kingpin_ground | where the kingpin axis pierces the ground | exists `KinematicMetrics._kingpin_ground` L143 |
| ic_front_view | front-view instant centre of this corner (XZ, at Y = wheel centre) | exists `KinematicMetrics.ic_front_view` L195 (instant-axis trace, Rule 17) |
| ic_side_view | side-view instant centre (YZ) — the anti-dive/squat construction | exists `metrics_catalog._sv_ic_point` L249 (arm-line method) and `_kinematic_ic` L102 (finite-difference). Pick ONE, see risk 7.1 |
| chassis_node_* | FSAE bay nodes 38.1 mm along each leg | exists `vahan/chassis.chassis_nodes` L165, already in `pts` |
| halfshaft_inner / outer (rear) | tripod at the diff, hub end | exists `vahan/driveshaft.package` L58 |
| roll_centre (per axle) | IC-to-contact-patch lines of both corners meet the centre plane | exists `KinematicMetrics.roll_center_height` L216 gives Z; the 3-D point today comes from `_update_3d._axle_rc` L7613 which still uses the pickup MIDPOINT (contradicts Rule 17; GUI_TASKS 2026-09-21 open item). Measure mode uses the Rule-17 builder and the RC sphere is rebased on it (phase 1) |
| pitch_centre | lateral line through the CG at the roll-axis height (Milliken), drawn today | `_update_3d` L7683-7705; user must confirm which definition (risk 7.1) |
| cg, unsprung_cg_front/rear | car dict `cg_*_mm`; axle line at `unsprung_cg_height_m` | `_update_3d` L7657 / L7666 |

### 1.2 Lines / axes
| ID | What it is | Source today |
|---|---|---|
| uca_axis, lca_axis | arm pivot axis through the two inboard pickups | new (2 points) |
| uca_leg_f/r, lca_leg_f/r, tie_rod, pushrod, damper | the drawn members | `LINKS` L612 + the member segs `update_scene` records in `_last_member_segs` |
| kingpin | LCA BJ -> UCA BJ | exists `KinematicMetrics.kingpin_axis` L86 |
| wheel_spin_axis | through wheel centre along `spin_axis` | `SolvedState.spin_axis` |
| rocker_axis | rocker_pivot -> rocker_axis_pt (Rule 02: normal to the actuation plane) | `packaging.rocker_plane` L133 (origin + normal) |
| arb_bar, arb_blade, arb_drop_link | torsion bar L<->R pivot, pivot -> arm end, arm end -> drop top | `_assemble_arb_segs` L7053 (`_last_arb_segs` in the view) |
| halfshaft (rear) | inner -> outer | `driveshaft.package` |
| swing_arm_front_view | IC_front -> contact patch ("projected line for roll centre"), both corners | new (from 1.1) |
| arm_trace_front_view (x2) | each arm's trace on the transverse plane through the wheel centre | exists `KinematicMetrics._arm_trace_xz` L176 |
| swing_arm_side_view | IC_side -> contact patch (or wheel centre, inboard-brake case) | new; `_sv_ic_coeff` L186 doc explains the two references |
| roll_axis | front RC -> rear RC | `update_rc` L1142 draws it |
| pitch_axis | lateral line through the pitch centre | `update_pitch_axis` L1270 |
| car axes X/Y/Z at origin + at any picked point | for "angle to car axis" measurements | new |
| chassis bay tubes | FSAE tubes incl. diagonal + transverse half | exists `chassis.bay_members` L204, drawn by `set_chassis_tubes` L1232 |
| rack | tie_rod_inner L -> R | `update_rack` L1302 |

### 1.3 Planes
| ID | What it is | Source today |
|---|---|---|
| uca_plane, lca_plane | the arm plane (two pickups + ball joint) | new (3 points) |
| actuation_plane | the corner's static actuation chain plane (Rule 01) | exists `packaging._chain_plane` L208 (SVD of 5 chain points) — and `_physical_rocker_plane` L221 (3-attach plate) and `rocker_plane` L133 (normal = rocker axis). Show all three as separate toggles; their mutual angle is itself a measurement (Rules 01/02 gates) |
| wheel_plane | through wheel centre, normal = spin axis | new |
| ground | the view's ground plane (follows the tyres in heave) | `View3D.ground_height_m` L1084 |
| centre_plane | X = 0 | new |
| axle_plane_front/rear | transverse plane Y = mean wheel-centre Y of the axle (the plane the front-view IC lives on) | new |
| keepout faces | bulkhead red-zone faces | `vahan/keepout.KeepOut.faces` L132-136 (`n`, `p0`, `verts`), drawn by `set_keepout` L1201 |
| rocker plate outline | the drawn plate polygon | `_last_rocker_polys` (`update_scene` L1505-1535) |

Each builder returns a small record: `{'id','family','corner','label','p' (point) | 'a','b' (line)
| 'origin','normal','outline' (plane), 'source': 'file:function'}` so the panel can print where
a number comes from.

## 2. Measurements

Two selected entities -> one measurement. All maths is pure numpy in `vahan/measure.py`:

| Pair | Reported |
|---|---|
| point–point | distance (mm) + dX, dY, dZ components |
| point–line | perpendicular distance + foot of perpendicular |
| point–plane | signed distance (+ on the normal side) + projected point |
| line–line | angle (0-90 deg, and the full 0-180 with direction) + shortest distance + "skew / parallel / intersecting" + the two closest points |
| line–plane | angle between the line and the plane (0-90) + pierce point |
| plane–plane | dihedral angle + line of intersection (or "parallel, d = ... mm") |
| line–car axes (one pick + Axis button) | angle to X, Y, Z and the two view-plane projections (front-view / side-view angle) |
| point travel arc (one pick + Arc button) | radius + centre of the point's path over the motion range (circle fit through the pose at -range / 0 / +range travel: re-solves 3 poses via the existing solvers) |
| single point | X, Y, Z (mm) — what the Direct Edit box shows, for non-editable points too |

Live behaviour: a measurement stores entity IDs, never coordinates. On every `_update_3d`
(slider `_on_position` L7799 -> `_deferred_3d` L7810, steer `_on_steer` L7877, hardpoint edit,
roll/pitch/heave mode L7212-7223) the measure layer rebuilds its entity table from the fresh
`corners_draw` + ARB segs + overlay points and re-evaluates every row. On light frames (slider
dragging, `light=True`) only point/line entities are rebuilt; IC/RC/plane entities and the list
refresh on the 200 ms settle frame, like the RC spheres do today (L7645-7652). An entity that
disappears (NaN pushrod on a decoupled corner, IC at infinity) shows "n/a" in the row instead of
dropping the row.

## 3. Interaction (CAD style)

- **Mode toggle**: add `Measure` to the navcube view-controls combo (`View3D._mode_combo` L1005,
  `set_view_mode` L2121 whitelist, `_on_view_controls_changed` L11055, Car panel combo mirror).
  Measure keeps the normal colours (no desaturation) and disables edit-mode key moves.
- **Hover**: `_qt_move` L2461 already routes hover to `_hover_load` in Load mode; add
  `_hover_measure(qpos)` that finds the nearest pickable entity (section 4), draws a white halo
  marker / thicker ghost line / brighter plane outline, and shows the entity label in the existing
  `_load_tooltip` QLabel (L1040, reused, renamed `_hover_tip`).
- **Click**: `_qt_press` L2425 -> in measure mode `_try_pick_entity` instead of `_try_pick`.
  First click = selection 1 (held in a yellow ring, the existing `_C_SEL`), second click =
  selection 2 -> measurement created, both clear, leader stays. Shift+click adds to an
  "anchor" selection so one point can be measured against many. Esc clears selection and hover;
  Delete removes the highlighted row. Clicking empty space clears.
- **In the view**: a leader `scene.Line` between the two closest points of the pair + a
  `scene.visuals.Text` label (vispy 0.16.1 has Text; verified) with the value, e.g.
  `312.4 mm` / `87.3 deg`. Text colour ink `#ECECEE` on a small dark background box; never yellow
  text (docs/DESIGN.md). Dimension lines white, selection ring yellow, hover halo white — the
  in-app line palette (yellow/red/white/blue) with no red/green pair; derived reference lines
  separated from members by line WEIGHT and alpha (thin, 0.6) not by another hue.
- **Measurements panel** (`gui/measure_panel.py`, a `CollapsibleSection` in the left column next
  to `_values_panel`, `_build_ui` L4750-4764): table rows = name, value, unit, dX/dY/dZ, source;
  monospace tabular numbers; buttons Copy row / Copy all (CSV), Clear, Pin to view (keep the label
  when the mode is left). Row hover highlights the pair in the view.
- **Layer tree** (same panel, below the list): a `QTreeWidget` with checkboxes — families
  (Hardpoints, Derived points, Member lines, Pivot axes, Kingpin/spin axes, Rocker & ARB,
  Swing-arm lines, Roll/pitch axes, Arm planes, Actuation planes, Wheel/ground/centre/axle planes,
  Chassis bays, Keep-out) x corner filter (FL/FR/RL/RR/axle) — exactly how CAD reference
  geometry is toggled. The four existing overlay checkboxes (`_chk_rc/_ra/_pa/_cg` ~L4770)
  move into this tree so there is ONE visibility source; their state is saved in `car['measure_layers']`.
- Planes draw as translucent quads (alpha 0.12) plus an outline, sized to the bounding box of
  the points that define them + 60 mm margin, with a short normal "flag" at the origin to pick.

## 4. Picking approach (vispy)

Recommendation: **screen-space nearest-entity picking**, extending what `_try_pick` L2647 and
`_hover_load` L2434 already do (project each candidate with
`self._view.scene.node_transform(self._canvas.scene)`, divide by w, compare pixels):
- points: pixel distance to the projected point (radius 12 px hover, 30 px click as today);
- lines: pixel distance from the cursor to the projected 2-D segment (point-to-segment, clip
  segments behind the camera); members are picked on their centre line, ignoring tube radius;
- planes: pixel distance to the projected outline edges OR to the normal-flag handle (the
  translucent fill is never pickable, so planes do not swallow clicks);
- tie-break by family priority point > line > plane, then by depth (w) so the nearer wins.

Pros: no extra render pass, works with the existing camera-aware transform, trivially filtered
by the layer tree (invisible = not a candidate), identical maths on screen for a 2-point line
and a long axis. Cons: O(N) numpy per mouse move — N is ~4 x (17 points + 15 lines + 8 planes)
plus ~40 chassis tubes = under 400 entities, one vectorised `tr.map` on an (N,4) array is well
under 1 ms, so hover at mouse-move rate is fine; cache the projected arrays and recompute only
on camera change or pose change (hash of camera az/el/distance/centre + pose counter).

Alternative considered: vispy colour-ID picking (`SceneCanvas.visual_at` / `visuals_at`,
`vispy/scene/canvas.py` L403/L464, renders with `set_picking()` and reads the framebuffer).
Rejected for now: it identifies a VISUAL, not a marker inside a `Markers` visual, so every
entity would need its own visual (hundreds of GL objects), it costs a render pass per hover, and
the app's offscreen net runs would need the picking framebuffer too. Keep it as the fallback if
screen-space picking ever becomes ambiguous on dense views.

## 5. Architecture

- `vahan/measure.py` (pure numpy, no Qt):
  - `build_entities(corners_draw, arb_segs, overlays, car, states) -> dict[id, Entity]` — the
    builders of section 1; `states` = the per-corner `SolvedState` so `KinematicMetrics` and
    `metrics_catalog._sv_ic_point` are called, not re-implemented; `overlays` = the RC/CG/
    unsprung/pitch records already assembled in `_update_3d` (moved into one helper
    `overlay_points(...)` that both the markers and the measure layer use).
  - `measure(e1, e2) -> Measurement` and `measure_axis(e)`, `measure_arc(point_id, solver, range)`.
  - `point_point`, `point_line`, `point_plane`, `line_line`, `line_plane`, `plane_plane`,
    `circle_fit_3pt` — small documented functions with units in the name (`_mm`, `_deg`).
- `gui/measure_panel.py`: `MeasurePanel(CollapsibleSection)` = measurements table + layer tree;
  signals `layers_changed(dict)`, `row_selected(id)`, `clear_requested`.
- `gui/view3d.py` hooks: `set_measure_entities(entities, layers)` (draws derived points/lines/
  planes into three new visuals `_meas_pts`, `_meas_lines`, `_meas_planes` + outline),
  `set_measurements(rows)` (leaders + Text labels), `_hover_measure`, `_try_pick_entity`,
  `set_on_measure_pick(cb)`; mode `'measure'` accepted by `set_view_mode`.
- `gui/main_window.py` hooks: in `_update_3d` after `update_scene` (L7453) and the overlay block,
  call `measure.build_entities(...)`, push to the view and to the panel; `_on_measure_pick(id)`
  keeps the two-pick state; `car['view_mode']`, `car['measure_layers']`, `car['measurements']`
  (list of ID pairs) saved with the project so measurements survive a reload.
- MCP (`gui/mcp_server.py`): `measure(id_a, id_b)` and `list_entities()` tools so headless runs can
  read the same numbers (HEADLESS APP ONLY rule).
- Regression net (`test_one_model.py`, new block "measure mode", offscreen):
  1. `FL.point.uca_outer`–`FL.point.lca_outer` distance == `norm(st.uca_outer - st.lca_outer)` to 1e-9 m;
  2. `front.point.roll_centre` Z == the Kinematics graph roll centre (`_compute_sweep` axle
     construction) AND `KinematicMetrics.roll_center_height` mean of L/R, to 0.01 mm — this is
     also the fix of the open 0.26 mm RC discrepancy in GUI_TASKS;
  3. `FL.line.rocker_axis`–`FL.plane.actuation` angle == 90.000 deg on the current config
     (Rule 02); `FL.plane.uca`–`FL.plane.lca` dihedral vs a hand value from the config file;
  4. kingpin vs Z angle == `KinematicMetrics.kpi` / caster from the same state (side-view and
     front-view projections), scrub == point–line distance of the contact patch to the kingpin;
  5. a measurement re-evaluated at +25 mm heave changes, at 0 mm returns to the static value
     (live binding), and NaN entities give "n/a" without an exception.

## 6. Phasing (effort = focused sessions of this app)

1. **Points + distances + layers** (1.5 sessions): `vahan/measure.py` point builders + point maths,
   `MeasurePanel` list + tree, view hooks, mode toggle, net checks 1, 2, 5; move the four overlay
   checkboxes into the tree; rebase the RC sphere on the Rule-17 builder.
2. **Lines/axes + angles** (1.5 sessions): all line builders incl. ARB, chassis tubes, swing-arm
   lines, car-axis measurements; line picking; net checks 3 (axis part), 4.
3. **Planes** (1 session): plane builders, translucent quads + outline + normal flag, plane
   picking, plane–plane/line–plane maths, net check 3 (plane part).
4. **Live + export** (1 session): travel-arc measurement, project save/load of measurements,
   Copy CSV, MCP tools, pinned labels, a `figs/gui_measure_*.png` render in `capture_gui_views.py`
   so the result is LOOKED at, not described.

## 7. Risks / open questions for the user

1. **Which "pitch centre"?** Today the view draws a lateral line through the CG at the
   interpolated roll-axis height (`_update_3d` L7683). The side-view ICs per axle
   (`_sv_ic_point`) are a different thing (anti-dive/squat). Plan: expose BOTH as separate
   entities named plainly ("pitch axis at CG (roll-axis height)" and "front/rear side-view
   instant centre"); confirm naming. Also: arm-line side-view IC vs finite-difference
   `_kinematic_ic` — which one is "the" IC for display (they differ when the projections are near parallel)?
2. **Roll-centre ONE-MODEL gap**: the 3-D RC sphere uses the pickup midpoint (`_axle_rc` L7613)
   while the graph/catalog use the instant axis. Measure mode will use the instant axis and
   move the sphere to it; the open 0.26 mm graph-vs-catalog difference gets settled by net check 2.
3. **Contact patch definition**: `roll_center_height` puts the patch at Z = 0 under the wheel
   centre; the view's ground is at wheel-centre Z minus tyre radius (follows heave). For the
   measure entities use the view's ground so the drawn and measured patch agree; the RC Z is then
   reported relative to that ground. Confirm.
4. **Planes drawn as translucent quads or outlines only?** Plan: quads at alpha 0.12 + outline,
   with a per-family "fill" toggle; quads cost nothing but can hide tubes behind them.
5. **Which actuation plane is "the" plane** (5-point SVD, 3-attach plate, or rocker-axis normal)?
   All three are exposed; the panel names them. Their mutual angles are shown as measurements.
6. Direct-damper, decoupled and heave-T-bar topologies have NaN chain points; their entities are
   listed but "n/a" — no attempt to invent a plane.
7. Hover picking of long axes (roll axis, rocker axis) that cross the whole view: candidates are
   limited to the drawn segment length (axes drawn +-300 mm about their anchor), not infinite lines.
8. Colour-blind check: the only colour pairs used are white/yellow (selection) and the existing
   blue chassis / red moving markers; reference geometry is separated by weight and alpha.
   Any new hue must be checked against docs/DESIGN.md before it ships.
