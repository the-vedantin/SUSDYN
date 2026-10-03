# Rule 18 — Chassis keep-out: the red zone is not packaging space

## The rule
A project may name a keep-out solid (`car['keepout_step']`, a STEP export of the
chassis volume that must stay clear — the bulkhead / footwell "red zone" from
Onshape, metres, car frame). No suspension member may be inside it at any
audited state: droop / static / bump × full left lock / centre / full right
lock. Members are the same capsules the interference view draws (arms,
tie rods, pushrods, coilovers, rockers, ARB drop links and blades), plus the
steering-rack housing (1.5 in OD between the tie-rod inner points), both
torsion bars and the rear driveshafts.

## Why
Anything that crosses the car — the rack housing, a torsion bar — can only
live under the floor or above the roof of the footwell. Placing it "between
the rockers" or "on the roll-hoop side" without the volume in the model put
the v99 front bar 29 mm into the driver's box, the v101 bar 176 mm into it and
the rack housing 42 mm into it in every version since v97. The volume is a
hard constraint from CAD and has to be in the solver, not in a note.

## How Vahan holds it
- `vahan/keepout.py`: `load_step_keepout()` reads a planar-faced STEP solid
  into convex pieces; `KeepOut.capsule_gap_m()` is the signed gap of a member
  (conservative near edges); `audit_window()` sweeps the live model;
  `keepout_for_window()` resolves `car['keepout_step']`.
- The 3D view draws the solid as a translucent red block (`View3D.set_keepout`)
  as soon as a project that names it is loaded.
- `test_one_model.py` fails on any member inside the zone.
- Packaging builders gate candidates against it (rack Z under the floor, bar
  above the roof) before any other score.

## Evidence (2026-09-10)
`configs/keepout/bulkhead_redzone.step`: |X| ≤ 175 mm, Y −567 … +264 mm, floor
Z 166.6 mm, roof Z 516.6 mm ahead of Y 47.6 rising along a 45° chamfer to
Z 705.6 behind Y 120.6. v101 audit: rack housing −42.2 mm, front torsion bar
−176.4 mm, everything else ≥ 9 mm clear.
