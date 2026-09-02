# Rule 14 — Same shocks, tune the wheel rate with preload

## The rule
The car keeps its existing shocks. Any design must respect the shock's real
installed length and stroke — the spring/damper chassis mount sits at the
baseline eye-to-eye length, and the geometry must not demand more stroke than the
shock has. Where the shop springs are unknown, present a selection of spring
rates and set the effective wheel rate with preload, rather than assuming a rate.

## Why
The shocks are fixed hardware. A layout that stretches or compresses the shock
past its installed length or asks for more stroke than it has is not buildable. And
the exact spring rate on the shelf is not always known, so the honest way to hit a
target wheel rate is to state the available rates and use preload to set the
effective rate, not to invent a spring.

## How Vahan holds it
- The damper installed length (eye-to-eye at static) is preserved when the
  actuation is rebuilt — the spring eye is placed at the baseline length along the
  damper axis.
- The motion panel's travel range gives the real droop and bump the shock allows;
  the clash and stroke checks run across that range.
- Wheel rate is set through the dynamics panel, where spring rate and preload are
  inputs.

## Tolerance
Installed length held to the baseline. Stroke stays within the shock's range.
