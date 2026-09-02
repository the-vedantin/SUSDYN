# Rule 01 — Coplanarity / nothing in bending

## The rule
The full actuation chain lies in one plane, and stays in that plane across the
whole wheel-travel range. The chain is: pushrod outer, pushrod inner, rocker
pivot, rocker spring eye, and spring/damper chassis mount. Because every member
sits in the plane, no member is loaded in bending — each carries only tension,
compression, or torsion.

## Why
A pushrod, rocker, or drop link that leaves the plane of its motion gets loaded
sideways and bends. Bending members are heavy (they need section to resist it)
and they flex, which corrupts the motion ratio. Keeping the chain planar keeps
every link a clean two-force or torsion member.

## The subtle part — across travel, not just static
It is easy to build the five points coplanar at static. The hard requirement is
that they STAY coplanar as the wheel moves. The wheel-side pushrod foot travels
on an arc; the actuation plane must contain that arc, and the rocker must turn
about the plane normal (see Rule 02) so the inboard points stay in the plane.
A design that is coplanar at static but drifts out of plane across travel still
violates this rule.

## How Vahan checks it
`vahan.packaging._axle_geometry_laws` fits a best plane (SVD) through the five
chain points at three travel positions (full droop, static, full bump) and
reports the largest out-of-plane distance as `coplanar_mm`. The regression net's
"design actuation" gate does the same.

## Tolerance
- Design target: under 0.1 mm at static.
- Accepted in the packaging tolerance set: 3.0 mm across travel (the arc of the
  pushrod foot makes a small unavoidable residual; the baseline sits near 1.1 mm).

## History
- The pushrod being 22 mm out of plane (v33) was caught only after the gate was
  extended to include the pushrod itself, not just the rocker plate points.
- Raising the actuation while holding the pushrod foot fixed re-tilts the plane;
  the chain must be re-solved to restore coplanarity, not merely translated.
