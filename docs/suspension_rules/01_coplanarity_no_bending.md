# Rule 01 — Per-corner static actuation coplanarity

## Authoritative user clarification — 2026-09-15

At STATIC ride position, independently at EACH corner, the pushrod outer and inner joint centres, rocker pivot, rocker spring eye, spring/damper chassis eye, and both bellcrank drop-link joint centres must lie in that corner's actuation plane. Left and right corners do not share a plane. Front and rear do not share a plane.

Use the static plane through that corner's rocker pivot, pushrod-inner attachment and spring eye. Check the actual joint centres against it, without subtracting declared offsets. The rocker axis is normal to that corner's plane (Rule 02). Rule 04 also applies to the drop-link endpoints.

Static coplanarity is the acceptance requirement. Do not demand that the wheel-side pushrod pickup stay in that fixed plane throughout travel, and do not repackage the dampers to make that happen. Across travel, continue to check physical clearance, damper stroke, kinematic closure, rod-end articulation and loads. Out-of-plane travel displacement may be reported as a diagnostic; it is not a static-coplanarity failure.

Preserve the intended damper packaging and use local geometry adjustments. Repackaging is not authorized merely because a stronger, invented plane constraint is easier to satisfy elsewhere.

## Tolerance

- Static design target: less than 0.1 mm for the actuation chain.
- Existing numerical gate: 3.0 mm at static, with actual residuals reported. This does not change the zero-offset design intent.
- Bellcrank drop-link endpoints: Rule 04's static tolerance, with actual endpoint distances reported.

## Physical interpretation

A spherical-ended rod can articulate while acting as a two-force member; nonplanar travel does not alone prove bending of that rod. Rockers, mounts, pins and bearings still carry loads and require structural checks. Coplanarity alone is not a strength certification.

## Superseded interpretation

The previous fixed-plane-through-full-travel acceptance and the v115/v116 redesign based on it were rejected by the user on 2026-09-15. They must not be reinstated as design requirements.
