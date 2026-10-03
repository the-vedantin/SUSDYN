# Rule 13 — Pushrod alignment

## The rule
Keep the pushrod close to the vertical-lateral (X-Z) plane, with little
longitudinal (Y) lean. The pushrod should run mostly up-and-inboard, not
fore-aft.

## Why
A pushrod with a large fore-aft component reacts wheel loads through a
longitudinal path the chassis is not laid out to carry, and it drags the
actuation plane fore-aft. Keeping the pushrod near the X-Z plane keeps the load
path clean and keeps the actuation plane roughly upright (a side-mounted shock),
which packages better and keeps the coplanarity residual small.

## Note on "not parallel to the ground"
The actuation plane is meant to be tilted, not flat — the spring is not parallel
to the ground plate. That tilt is a feature (it is roughly where it sits today),
not a target angle to hit. A plane that must contain the pushrod cannot be flatter
than the pushrod's own angle from horizontal, so a very shallow plane is not
achievable while obeying the no-bending rule.

## How Vahan checks it
Reported, not hard-gated: the pushrod direction's Y component (its lean off the
X-Z plane) and the actuation-plane orientation are measured during a build. The
baseline sits within a few degrees of the X-Z plane; keep new designs near that.

## Tolerance
"Very little" Y deviation — within a handful of degrees of the X-Z plane. This is
a design preference, weighed against the other rules, not a pass/fail gate.
