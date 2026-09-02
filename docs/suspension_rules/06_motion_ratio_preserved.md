# Rule 06 — Motion ratio preserved

## The rule
When the actuation is relocated or repackaged, the wheel-to-damper motion ratio
is re-tuned back to the baseline value. The motion ratio is the change in damper
(spring) length per unit of wheel travel, measured at design position.

## Why
The motion ratio sets how much of the wheel's motion and load reaches the spring
and damper. Change it and you change the ride frequency and the damping the car
actually runs, even if the spring and damper themselves are untouched. Packaging
moves must not silently retune the car.

## How Vahan holds it
- Measured by `vahan.packaging.solver_mr`: the central-difference slope of spring
  length versus travel at plus/minus one millimetre — the same method the
  dynamics build uses.
- Re-tuned by scaling the rocker input lever (the pushrod-inner to rocker-pivot
  distance) or sliding the pivot along that lever, bisecting until the motion
  ratio matches the target (`retune_mr`, and the pivot-slide search in the build
  scripts).

## Tolerance
Held to within 0.5 percent of the baseline for packaging work; the regression
net uses 1 percent. In practice the re-tune lands it exactly.

## Related
The anti-roll-bar rate depends on the motion ratio too; see Rule 07.
