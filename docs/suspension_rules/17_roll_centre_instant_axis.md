# Rule 17 — Roll centre from the instant axis, not the pickup midpoint

## The rule
The front-view instant centre of a corner is where the upright's instant axis
(the intersection of the upper and lower arm planes) pierces the transverse
plane through the wheel centre. Each arm is drawn in that plane from its pivot
AXIS (the line through its two inboard pickups) and its ball joint. The roll
centre follows from that instant centre in the usual way (line to the contact
patch, intersected with the car centreline).

## Why
An A-arm's kinematics are set by its pivot axis and its ball joint, nothing
else. Where along the axis the two pickups sit is a structural choice (bracket
spacing, which chassis member they land on) with zero effect on wheel motion.
A roll-centre construction that uses the pickup MIDPOINT changes its answer
when a pickup slides along its own axis, and it mis-reads any axle whose pivot
axes are swept in plan view.

## Evidence (2026-09-09, v99 → v101)
- Sliding the front `lca_rear` 38.7 mm along the LCA axis (to put both aft
  pickups on one vertical line for the front hoop) left camber, toe, caster,
  KPI, scrub, trail, bump steer and motion ratio identical to 1e-4, yet the
  midpoint construction moved the front roll centre 48.37 → 49.49 mm.
- With the instant-axis construction the same slide changes the roll centre by
  8e-14 mm. The re-read v99 values are front 50.44 mm (was 48.37), rear
  75.41 mm (was 68.64) — the rear upper arm's pivot axis is swept 13.7 deg in
  plan, which the midpoint method could not represent.

## How Vahan holds it
- `vahan/kinematics.py` `KinematicMetrics.ic_front_view` / `roll_center_height`
  use `_arm_trace_xz` (axis + ball joint traced on the wheel-centre plane).
- `test_one_model.py` slides a pickup 30 mm along its axis on the live model
  and fails if the roll centre moves more than 1e-6 mm.
- The dynamics (roll gradient, geometric load transfer, jacking) read this
  roll centre, so their numbers moved with it on 2026-09-09; earlier binder
  values used the midpoint construction.
