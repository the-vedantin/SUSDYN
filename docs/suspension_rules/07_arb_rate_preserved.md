# Rule 07 — Anti-roll-bar rate preserved

## The rule
When the anti-roll bar is repackaged, its wheel-rate contribution (the roll
stiffness it adds) is re-tuned back to the baseline value.

## Why
The bar rate sets the roll stiffness balance between the axles, which sets the
handling balance. Moving or flipping the bar changes its geometry and therefore
its rate; the rate must be restored so the car's balance is unchanged by a
packaging move.

## The physics that makes it tunable
The bar's wheel rate scales with the fourth power of the bar diameter (through the
tube's area moments), divided by the square of the bar-to-wheel motion ratio:

    wheel rate  proportional to  (OD^4 - ID^4) / (bar motion ratio)^2

So there are two independent knobs:
- the bar diameter (a dynamics input, not a hardpoint) — a clean multiplier;
- the drop-link radius on the rocker (geometry) — changes the bar motion ratio.

Diameter is the cleaner knob for a redesigned bar: pick a bearing-safe drop
radius for packaging, then choose the diameter to hit the rate. A skinnier bar
also shrinks its own clash envelope.

## How Vahan holds it
- Rate read through the one dynamics-panel formula (`panel_arb_rate`).
- Re-tuned by the drop-link radius (`retune_arb`), the blade length
  (`retune_arb_blade`), or by solving the bar diameter (bisection on the panel
  rate). A hollow bar needs bisection; a solid bar is a direct fourth-root.

## Tolerance
Held to within 1.5 percent of the baseline for packaging work; the regression net
uses 2 percent.

## Note
When the bar is re-hung, its mount may not move more than about 100 mm from the
baseline chassis mount — a bar only mounts where structure exists. See Rule 09
and the mountability handling in `vahan.relocate`.
