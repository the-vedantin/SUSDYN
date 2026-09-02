# Rule 09 — Clash-free across full travel

## The rule
No two members may interfere at any point in the wheel-travel range. The check
runs at full droop, static, and full bump, using the same member set the GUI
interference view draws.

## The member set (must be the full one)
Arms (upper and lower, front and rear legs), tie or toe rod, pushrod, the
ball-joint spheres, the coilover, the anti-roll-bar drop link and torsion bar,
the rocker bearing, the rod ends, and the rear driveshaft. A "clash-free" claim
made against a subset is not valid — a real build-stopper (a pushrod through the
driveshaft) once passed a subset check that omitted the driveshaft.

## Standing near-misses
The baseline may carry members that already sit at or below zero gap in the clash
model (for example the coilover body passing the rocker bearing hub, where the
model treats the coilover as a full solid cylinder). Those standing conditions
are inherited; a move is judged by whether it introduces a NEW negative-gap pair
or makes an existing near-miss WORSE, not by the absolute count.

## How Vahan checks it
- `vahan.packaging._clash_sweep` builds the full member list per corner at the
  three travel stations and returns the clashing pairs.
- The oracle passes a move when it adds no new negative-gap pair and worsens no
  standing pair by more than 0.25 mm.
- For the anti-roll bar, clearance must also hold across travel as the rocker
  swings the drop link — a static-only check is not enough (a long bottom-ARB
  drop link can graze the pushrod at full bump while clearing at static).

## Tolerance
No new contact; standing near-misses may not worsen by more than 0.25 mm.
