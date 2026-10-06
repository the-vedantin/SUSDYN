# Vahan GUI task list

Shown in the app under Help -> Task list. One line per task: `- [ ]` open, `- [x]` done.
Claude appends here when a feature is promised for later; the user ticks or edits freely.

- [ ] Anti-squat target knob: a button/spinbox that moves the rear pickups to a chosen % anti-squat (vahan.packaging.retarget_side_view_ic exists; no GUI yet) - 2026-09-21
- [ ] ARB nudge tool: project rough ARB points onto the exact-law (coplanar + triad) surface and report the mm moved + motion ratio (user: "we will discuss this later") - 2026-09-18
- [ ] "Snap pushrod" button should pick the arm from the topology (rear pushrod on the UCA), not by axle - 2026-09-19
- [ ] Loads model: use the arm that actually carries the pushrod (v139 rear UCA forces understated ~1.2 kN) - 2026-09-19
- [ ] Steer camber in the dynamics / tyre model (front camber gain includes steer) - 2026-09-20
- [ ] Ground-camber graph (static + kinematic + roll) next to the chassis-relative one - 2026-09-20
- [ ] Confirm outboard vs inboard rear brakes (rear anti-lift assumes outboard) - 2026-09-20
- [x] v142 own-bracket toe link WITHDRAWN: hard rule 20, rear toe inner must be the aft LCA pickup - 2026-09-21
- [x] Reset sag button (Motion panel) - 2026-09-21
- [x] Half-shaft length / separation / joint-angle curves (Kinematics graphs, category Half-shaft) - 2026-09-21
- [x] STEP part manager (View -> Manage STEP parts: show/hide, move, flip, colour, opacity, rename, remove) - 2026-09-21
- [x] Contact patch vs ride Hz + launch load lag tabs (Ride page) - 2026-09-21
- [x] ONE MODEL check: rear roll centre reads 68.0 mm on the Kinematics graph (axle post-processing in _compute_sweep) but 75.4 mm from the per-corner sweep the binder dumps - find which construction each uses and make them one - 2026-09-21
- [x] Roll centre on the Kinematics graph now uses the solver's instant-axis construction (was the retired pickup-midpoint; rear read 68.0 instead of 75.7) - 2026-09-21
- [x] Roll centre: axle construction on the graph (75.67 rear) vs per-corner property in the catalog/binder (75.41) differ by 0.26 mm - cause: the sweep's static row was the grid point nearest 0, not 0; the grid now holds t = 0 exactly (net 'audit C4') - 2026-10-06
- [ ] Net: "ackermann sweep" reports NaN because both capability values are inf (never front-limited); compare with isfinite-aware logic - 2026-09-21
- [ ] Net gates missing for Rule 07 (ARB rate preserved) and Rule 11 (pushrod mount near the ball joint) - 2026-09-21
- [x] Rim model: real Keizer 10x7 / 10x8 (6 in BS) profiles from the STEP files, net gate "real wheel profile" - 2026-09-21
- [ ] 3D view: draw the real wheel profile (barrel + disc) instead of the straight cylinder - 2026-09-21
- [ ] Decide 7 vs 8 in wheel width (tyre behaviour, not packaging): rear now packaged for the 8 in (v143); FRONT still hits the 8 in flange: tie rod -2.9 mm at lock+bump, pushrod -5.3 at droop+lock - needs a rack/upright-level change - 2026-09-21
- [ ] Rear UCA INBOARD lift 30 mm with camber + anti geometry held (LCA inboard +36, tilts) is possible but the rear roll centre goes 75.7 -> 140 mm; alternatives: tube on drop tabs above the pickups (chassis), diff up 23 mm (powertrain) - USER DECISION - 2026-09-22
- [ ] Confirm the UCA chassis cross member tube OD (assumed 1 in in car[uca_cross_member_od_mm]) - 2026-09-21
- [x] "Go to static sag (0)" button on the Motion panel: puts the position slider back at 0 mm (the old button, now "Undo applied sag", only restores the drawn points) - 2026-09-22
- [x] Rule 21 + net gate: pushrod arm mount must move IN the actuation plane (v146 front was 25 deg / 16 mm out; v147 fixes it) - 2026-09-22
- [ ] Astra audit fixes in progress: IK Apply axle/geometry binding + real coplanarity residual + failed-solve Apply block; recommend() unit clamp; lap/engine/ride items 21-24/27; member-load free bodies (braking, pushrod-on-UCA, lateral sign) - 2026-09-22
- [x] Jacking computed and fed back into the steady-state solve; one grip scale (Dynamics 'Grip multiplier') with limits at a user list of scales; steering inputs synced on load - 2026-09-23
- [x] Rule 11 gate: both pushrods on the UPPER arm - 2026-09-23
- [ ] Diff-drop study for a lower rear roll centre: needs driven sprocket size + floor height with shocks bottomed (sprocket/chain must never touch the floor) - 2026-09-23
- [ ] Model the real chassis members (from the chassis lead) as obstructions; confirm frame nodes for the moved v148 arm pickups - 2026-09-23
- [ ] Ackermann study: MMD stability, lap-sim corner radii/lat g, Ackermann baked into the lap sim, pick min lap time - 2026-09-23
- [x] FSAE chassis bays (View menu): nodes 1.5 in along each arm leg, 1 in tubes, per-axle diagonal control, rear transverse tubes, sprocket/diff as fixed obstructions in every clash check - 2026-09-23
- [x] 3D view: real Keizer rim from the STEP profile; ground plane follows the tyres in heave - 2026-09-23
- [x] Jacking computed + reported; feedback into kinematics OFF by default (car['jacking_feedback']); net 'dyn camber pinned' guards the 2026 baseline camber-vs-g - 2026-09-23
- [ ] Sprocket/chain-to-ground check as an app feature (damper bottom-out + tyre squash + chain drop) in the net - 2026-09-23
- [ ] Chain runs vs rear cross tubes: needs the drive sprocket position/size (user) - 2026-09-23
- [ ] Front bar rate cliff (v147 layout, 276 % swing): re-hang within +-80 mm of its current spot - user decision on location - 2026-09-23
- [ ] Measure mode (3-D view, CAD-style): plan written at docs/MEASURE_MODE_PLAN.md - pick two points/lines/planes, read distance / angle live while the slider moves; every imaginary line and plane (arm axes, kingpin, rocker axis, actuation planes, ICs, roll centre, roll/pitch axes, bays, keep-out) toggleable in a layer tree - 2026-10-01
- [ ] Measure mode phase 1: points + distances + layer tree + mode toggle; also rebases the 3-D roll-centre sphere on the Rule-17 instant-axis builder (today `_update_3d._axle_rc` still uses the pickup midpoint) - 2026-10-01
- [ ] Measure mode phase 2: lines/axes + angles (line-line, line-plane, line-car axes); phase 3: planes (translucent quads + outlines); phase 4: travel-arc radius, save/load measurements, CSV copy, MCP measure tool, GUI render - 2026-10-01
- [ ] Measure mode user decisions needed: which "pitch centre" (lateral line at CG vs per-axle side-view IC), contact-patch reference (Z=0 vs tyre-following ground), planes as quads or outlines - 2026-10-01
- [ ] Measure mode net checks: hardpoint distance == numpy, roll-centre point == KinematicMetrics/graph (closes the open 0.26 mm RC item), rocker axis vs actuation plane == 90 deg, kingpin angles == KPI/caster, live re-evaluation at +25 mm heave - 2026-10-01
