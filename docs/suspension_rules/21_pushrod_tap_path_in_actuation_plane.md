# Rule 21 — The pushrod's arm mount and the actuation plane through travel

User check (2026-09-22): "if you trace the pushrod point on the control arm as the wheel goes up, is that
planar to the actuation plane? if not, that's bad."

The pushrod's outboard joint rides on a control arm, so it swings on an arc about that arm's pivot axis.
The rocker swings in the static actuation plane (Rule 01/02). If the arm-mount arc leaves that plane, the
pushrod leans out of the rocker plane through travel.

DIRECT effect (what actually matters): pushrod lean out of the rocker plane =
- rod-end misalignment at both pushrod ends, and
- a side load on the rocker and its bearings = sin(lean) x pushrod force.

Measured (2026-09-22, app solver, full travel −47..+67 mm):
- v146 FRONT: arm-mount motion 24.8 deg out of the plane at static, mount −19..+21 mm off the plane at the
  travel ends; pushrod lean 0.2 deg near static, 1.2 deg at ±25 mm, 3.0 deg at full bump (52 N side load per
  kN of pushrod force), 2.6 deg at full droop.
- v146 REAR: lean ≤ 1.0 deg (18 N per kN).

Fix constraints (user, 2026-09-22): dampers and rocker are chassis-locked — only control-arm and pushrod
geometry may change. Rotating the actuation slice (the first v147) is REJECTED
(configs/_rejected/2027_v147_REJECTED_rotated_actuation_slice.vahan).
With the rocker locked, a mount path in the plane needs the front UCA pivot axis pitched ~43 deg in side
view (today 18 deg: rear pickup ~96 mm lower) or a pushrod mount ~170 mm from the ball joint; moving the
mount near the joint cannot do it.

Gate: regression net "rule 21 pushrod lean" — REPORTED (no user threshold yet).
