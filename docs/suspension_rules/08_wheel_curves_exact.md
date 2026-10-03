# Rule 08 — Wheel curves held exact

## The rule
A packaging or relocation move that only touches the INBOARD actuation (pushrod
inner, rocker, spring, damper, anti-roll bar) must hold every wheel-side
kinematic curve exactly: camber, toe, caster, king-pin inclination, scrub
radius, mechanical trail, roll-centre height, and bump steer, across the whole
travel range.

## Why
The wheel curves are the car's kinematics — how the contact patch behaves. They
are set entirely by the wheel-locating hardpoints: the control-arm pickups, the
tie-rod ends, and the wheel centre. The inboard actuation moves the spring and
bar, not the wheel. So if the wheel-locating points do not move, the curves are
held by construction, not by tolerance — the deviation is zero, not "small".

## Why this makes packaging tractable
Because the curves are held exactly by not touching the wheel side, the only
things that flex during an inboard move are the motion ratio and the bar rate
(Rules 06 and 07), which are re-tuned. This is what lets the packaging search
explore large actuation changes while the kinematics stay pinned.

## How Vahan holds it
- The wheel-locating points are never written by an inboard transform. Vahan's
  packaging validator records them as byte-identical to the baseline and, when
  they are identical, reports every wheel metric as an exact zero-delta.
- If any wheel point does move, the validator re-measures the curves and compares
  them to the baseline instead of assuming they held.

## Tolerance
Exact (zero) for inboard-only moves. If a wheel point is deliberately moved, the
tolerance set applies (for example 10 percent of each curve's own range).
