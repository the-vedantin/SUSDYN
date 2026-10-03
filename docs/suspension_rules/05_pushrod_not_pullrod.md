# Rule 05 — Pushrod, not pullrod

## The rule
The damper must act as a pushrod: as the wheel moves up into bump the damper
COMPRESSES (spring length falls), and it does so monotonically across the whole
travel range. It must never extend in bump, and it must never reverse direction
part-way through the stroke.

## Why
A rocker that is geometrically valid can still run the damper backwards — the
pushrod effectively pulls the damper open in bump instead of closing it. The
motion-ratio magnitude looks identical, so a magnitude-only check misses it. A
backwards damper inverts the ride and roll response of the whole car.

## The trap this rule closes
The motion-ratio helper takes an absolute value, so it cannot tell a pushrod from
a pullrod. Before this rule was a hard check, a config (v73, first attempt)
shipped with the front damper extending in bump — it was caught by eye, not by
the tool. A near-zero net stroke with a non-monotonic curve is the other failure
mode (the damper compresses then extends, so start and end match).

## How Vahan checks it
- `vahan.packaging.damper_motion_sign(win, axle)` returns the sign of
  (spring length at full bump minus spring length at full droop). A pushrod is
  negative (compresses in bump). The baseline sign is captured and the oracle
  requires the candidate to match it.
- When designing a new actuation, sample the spring length at nine travel
  stations and require the curve to fall monotonically with a real net stroke
  (at least 15 mm), not just match the sign.
- The regression net's "damper sign" gate checks the design config reads as a
  pushrod on both axles and that a known-inverted config is correctly rejected.

## Tolerance
The sign must match the baseline exactly. The stroke must be monotonic.
