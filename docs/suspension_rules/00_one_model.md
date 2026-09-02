# Rule 00 — One Model

## The rule
Every number Vahan reports — the 3D view, every kinematic graph, every dynamics
result, every packaging check — is derived from the single solved suspension
model. Nothing is computed from a second model, a hardcoded value, or a
duplicated copy of the physics.

## Why
When two pieces of code each implement the same physics, they drift apart and
give contradictory answers. This has bitten this project before: six rival
Ackermann implementations produced three different answers in one session. The
only defence is that there is exactly one place each quantity is computed, and
everything else calls it.

## How Vahan holds it
- The corner solvers (`MainWindow._build_corner_solvers`) are the single source
  of solved geometry at any travel.
- Kinematic metrics come only from `vahan.kinematics.KinematicMetrics`.
- Dynamics rates come only from the dynamics build (`_build_dynamics_solver`).
- Interference comes only from `vahan.interference` (the same member set the GUI
  interference view draws).
- Analysis scripts may load, call, and plot the model — they may never re-compute
  physics. A duplicated computation is the mechanism of contradictory answers.

## In practice
If you need a new number, add it as a Vahan feature that reads the solved model,
not as a script that re-derives it. If two paths report the same quantity, one of
them is wrong — stop and reconcile before shipping a third.
