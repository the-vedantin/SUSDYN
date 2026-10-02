# Rule 16 — The suspension fits inside the REAL rim, at all four corners

## Current assumed envelope (2026-09-05)
The user specified a 9.5-inch **inner** rim diameter (241.3 mm), with material
around the spherical joint included in the packaging allowance. The saved car
reserves 25.4 mm radially beyond the 12.7 mm-radius joint body. Of that reserve,
6.35 mm is allocated to an upright housing envelope, leaving at least 19.05 mm
outside that envelope. The 6.35 mm allowance is not a strength-derived wall size.

The supplied Keizer 10x7 drawing specifies a 7-inch bead width. The current
barrel check assumes a concentric, constant-ID cylinder 177.8 mm wide. It checks
capsule-to-cylinder distance including the finite lips; a tube centreline just
outside the axial band can still hit the lip. The full taper, web and fillets
are not represented by this assumed cylinder.

`test_one_model.py` sweeps all four corners through intermediate steering and
travel positions. Saved `rim_barrel_width_mm` enables the finite-barrel gate;
the three upright joint bodies must separately satisfy their larger reserve.
The v86 candidate fails the barrel gate at full lock, despite passing joint-body
clearance and member-to-member checks. This remains an open design constraint;
do not describe it as an interference-free car.

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

## 2026-09-21 — joint reserve relaxed to 12.7 mm (user)
No Keizer barrel drawing is on disk; the barrel is still the assumed 241.3 mm straight cylinder.
The 2026 car's rear toe joint (real, clears the Keizer 10x7) sits 108.0 mm from the wheel axis at
25.4 mm depth — its 1-inch body edge touches this assumed barrel — so the assumed barrel is
conservative by at least a joint body there. User: "1 inch doesn't need to be; if it clears we're
fine." `rim_joint_clearance_mm` = 12.7 on v142.5 (rear toe joint 92.5 mm from the axis, 15.5 mm to
the assumed barrel). Get the Keizer drawing (6 in backspacing) to replace the assumed cylinder.

## 2026-09-21 — REAL WHEEL PROFILE (supersedes the assumed cylinder where set)
Keizer 10x7 and 10x8 STEP files (6 in backspacing — DECIDED; 7 vs 8 in width still open) are in
`external/keizer/` (not committed). `vahan/wheel_profile.py` reduces a wheel STEP to its inner
profile (barrel radius vs axial station from the tyre centre plane + the centre-disc face) —
saved as `configs/wheel_profiles/keizer_10x7_6in_bs.json` / `keizer_10x8_6in_bs.json`.
`car['wheel_profile']` names the wheel in use; the net gate "real wheel profile" requires every
tube and joint body at FL/RL, droop/static/bump and front lock to clear it by 3 mm; wheels in
`car['wheel_profile_alternatives']` are reported only. Assumption stated: Vahan's wheel_center
plane = the flange (tyre) centre plane; backspacing is read from the mesh (10x7: 6.47 in pad to
inboard flange incl. lip; 10x8: 6.56 in). The 10x8 inboard flange reaches +104 mm inboard of the
tyre centre (10x7: +88) and meets the REAR PUSHROD (0.3 mm at full droop on v142.5) and the toe rod
(1.9 mm at full bump) — the 8 in wheel needs a pushrod / arm change; the 7 in clears (min 4.8 mm,
toe rod at the lip, full bump).
Reference: the 2026 car's rear toe joint (real, clears) reads −2.4 mm on this model → the model is a
few mm conservative (flange-centre placement / joint-body radius), so a 3 mm gate is not slack.
