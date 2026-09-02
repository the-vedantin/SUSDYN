# Rule 16 — The suspension fits inside the REAL rim, at all four corners

## The rule
Every outboard joint BODY and every member TUBE that lies within the rim's axial band
must sit inside the rim's inner barrel — at full droop, static and full bump, and under
full steer at the front — at ALL FOUR corners. "Fits" means the physical part, not the
joint centre: ball-joint bodies (1", radius 12.7 mm), rod ends, and the 0.625" tubes
(radius 7.94 mm), with a design margin (3 mm inside the barrel radius).

## Why
The rim model was carried at 330 mm for months while the real rims are 230 mm. Shrinking
the number exposed that the front upper ball joint (+2.4 mm) and tie-rod tube (+6–8 mm)
poked past the barrel — and the REAR was ~30 mm out (upper ball joint +33.9, toe-link
outer +21, upper-arm tubes +29). None of it was caught because the tool checked joint
CENTRES only, and because the check was run on the front and never on the rear.

## How Vahan checks it
- `KinematicMetrics.rim_fit()` (vahan/kinematics.py) checks only the CENTRES of
  uca_outer / lca_outer / tr_outer against the clear radius — it will say "fits" while a
  ball-joint body or a tube is outside. Treat it as a pre-check, never as the answer.
- The real check: for each corner's solved state, take the wheel spin axis through
  wheel_center; for every outboard point add its body radius, and for every member from
  `vahan.interference.corner_members` sample the tube and add the tube radius; keep only
  samples within the axial band; require every edge <= r_max − margin. Sweep travel; at
  the front ALSO sweep steer (`w._rebuild_solvers(steer_deg)` — the argument IS the steer
  angle) because the tie-rod tube swings into the barrel when the wheel is the outer one.
- The regression net carries this as a gate on the design config, FL and RL.

## The axial band
The barrel's inner lip position is set by the REAL rim width and wheel offset
(`wheel_offset_f_mm`), not by the tire width. Using tire_width/2 (±100 mm) centred on
wheel_center is CONSERVATIVE; a 7–8" rim with positive offset puts the inner lip at
≥ −90 mm. Confirm on the real wheel before spending geometry to satisfy the conservative
band.

## Tolerance
Edge <= (rim_dia/2) − 3 mm at every station (112 mm on a 230 rim). Under front steer, the
tube may eat the margin at the worst-case band but must never reach the barrel.

## History
- 2026-09-01: front fixed by sliding uca_outer 5.9 mm DOWN THE KINGPIN AXIS (caster/KPI/
  scrub/trail unchanged), tie_rod_outer 15 mm in + 20 mm up, rack 40 mm rearward. Cost:
  front RC −6.6 mm (restored by lowering the UCA inboard pickups 4.2 mm). A 10 mm axial
  tr_outer "insurance" move was tried and REJECTED — it swung Ackermann from −30 to −49%.
- 2026-09-01: rear found ~30 mm out after the user pointed at it; repackaged separately.
- "Inline" for the rear toe link means the same X as the LCA pickups (Rule 15); the rear
  lower arm + toe link are one welded part.
