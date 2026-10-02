# Rule 11 — Pushrod-on-arm mount: near the ball joint, never mid-arm

## The rule
When the pushrod picks up on a control arm, it mounts NEAR the outer ball joint,
not in the middle of the arm. The user's current placement is one inch
(25.4 mm) inboard in the control-arm plane and 1.25 inches (31.75 mm) above
that plane along its upward normal, near the arm bisector. BOTH axles mount to
the UPPER arm (user, 2026-09-22 — the earlier "rear mounts to the lower arm" text was stale;
the car has carried the rear pushrod on the upper arm for many versions).

## Why
A pushrod load applied in the middle of a control arm bends the arm — the arm has
to carry a transverse point load between its pickups, which is exactly what a
control arm is not built for. Feeding the load in close to the ball joint keeps it
near the wheel-load path and out of the middle of the arm, so the arm stays in
tension and compression, not bending. Some arm bending is unavoidable with any
pushrod-on-arm layout, but it must be kept near the joint, never near mid-span.

## The topology part — mount to the right arm
Which arm the pushrod rides changes the kinematics, so the topology must actually
be set, not faked. If the pushrod point is geometrically on the lower arm but the
model still treats it as riding the upper arm, the motion is driven by the wrong
body and the kinematics are skewed. In Vahan this is the axle's damper-mount
setting (upper arm / lower arm / upright), which maps to the solver's pushrod
body. Set it to match where the pushrod actually mounts.

## How Vahan holds it
- The damper-mount topology (`AxleTopology.damper_mount`, values upper/lower/
  upright) drives the solver's pushrod body.
- The mount point is placed one inch inboard along the arm and 1.25 inches up along
  the control-arm-plane normal from the ball joint.
- The regression net's "design actuation" gate measures each pushrod mount against the
  UPPER arm plane (0..35 mm above). v148: front 31.9 mm, rear 19.5 mm above; 41 / 52 mm
  from the ball joint.

## Caution when changing topology in code
Setting the topology reloads the whole hardpoint set to defaults (both axles).
Snapshot the wheel geometry you want to keep, change the topology, then restore
it — otherwise the wheel geometry and the untouched axle silently revert.
