# Rule 19 — Front ARB stays ahead of the front-hoop line

**Statement.** Every member of the front anti-roll bar (torsion bar, blades, drop links, drop-top rod ends)
stays AHEAD of the front-hoop line by at least 3 mm at droop, static and bump. The front-hoop line is the
line through the LCA-aft and UCA-aft chassis pickups in the side view, extended upward:

    Y_line(Z) = y_lca_rear + (Z - z_lca_rear) * (y_uca_rear - y_lca_rear) / (z_uca_rear - z_lca_rear)

(a vertical line at Y = 58.42 mm on v101 and later, where the two aft pickups share Y — Rule "aft pickup
line vertical", 2026-09-09). Gap = Y_line(Z) − (Y_member + radius).

**Why.** The front hoop is welded on that line; anything behind it is inside the cockpit / on the driver.
The user's instruction (2026-09-14): "flip front ARB towards the driver side … make sure it doesn't pass the
imaginary line extended upwards from LCA aft and UCA aft."

**Where it is computed.** `vahan.packaging.front_arb_hoop_line_gap_mm(win)` — assembled corners at the three
travel stations, bar radius from the dynamics panel OD, rod ends 8 mm, blade/link bodies 6.35 mm sampled along
their length. Net gate in `test_one_model.py` (design actuation block, "front hoop line" line). Study that
introduced it: `DESIGN_2027/v74_redesign/study_v105_front_arb_driver_side_full.py`.

**Interaction with the other rules.** Rule 03 (exact triad) fixes the drop-link direction to ⊥ the bar; Rule 04
(drop link in the rocker plane) then fixes it to the rocker plane's ⊥-X direction — on the 2027 front that is
60° from vertical, aft-up. A bar aft of the rocker therefore needs a link that runs aft toward this line, which
is why the hoop line, not the red zone, bounds how far aft the front bar can go.
