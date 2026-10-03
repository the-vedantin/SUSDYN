# Rule 14 — Same shocks: set rate with spring/MR, set sag with preload

## The rule
The car keeps its existing shocks. Any design must respect the shock's real
installed length and stroke — the spring/damper chassis mount sits at the
baseline eye-to-eye length, and the geometry must not demand more stroke than the
shock has. Where the shop springs are unknown, present a selection of spring
rates and the resulting wheel rates from the actual motion ratio, rather than
assuming a rate.  Preload sets installed spring force, collar position, static
sag, and the available bump/droop stroke; it does not set the incremental wheel
rate of a linear spring at a fixed motion ratio.

## Why
The shocks are fixed hardware. A layout that stretches or compresses the shock
past its installed length or asks for more stroke than it has is not buildable. And
the exact spring rate on the shelf is not always known, so the honest way to hit a
target wheel rate is to state the available rates and motion ratio, then select a
spring that produces the target.  Use preload to put that spring at the required
installed force and sag, not to invent stiffness.

## How Vahan holds it
- The damper installed length (eye-to-eye at static) is preserved when the
  actuation is rebuilt — the spring eye is placed at the baseline length along the
  damper axis.
- The motion panel's travel range gives the real droop and bump the shock allows;
  the clash and stroke checks run across that range.
- For a linear spring at a fixed motion ratio, wheel rate is set by spring
  stiffness and motion ratio.  Preload is a collar/static-force and sag setting;
  it must be checked against installed length and remaining bump/droop stroke.
- A progressive spring, bump stop, or changing motion ratio can alter tangent
  wheel rate with position.  That is a spring/geometry effect which must be
  modelled explicitly, not an effect attributed to preload alone.

## Tolerance
Installed length held to the baseline. Stroke stays within the shock's range.
