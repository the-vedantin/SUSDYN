# Rule 03 — Anti-roll-bar triad

## The rule
The three anti-roll-bar members form a mutually perpendicular triad at static —
ALL THREE pairs square, not just two:
- the torsion bar runs along the lateral (X) axis,
- the blade (arm, from bar pivot to arm end) is perpendicular to the bar,
- the drop link (arm end to drop top) is perpendicular to the blade,
- AND the drop link is perpendicular to the bar (bar↔drop = 90°).

In coordinates this means the bar pivot, the arm end and the drop top all sit at
the SAME lateral station: `arb_pivot.x = arb_arm_end.x = arb_drop_top.x`. Then
blade.x = 0 (bar⊥blade) and drop.x = 0 (bar⊥drop) fall out automatically. The
clean construction: put the arm end at the foot of the perpendicular from the
pivot onto the line (bellcrank plane ∩ {x = half-span}), and slide the drop top
along that same line for the rate — both ends stay in-plane (Rule 04) by
construction.

## Why
The bar stores roll energy in pure torsion about its own axis. For the drop-link
load to become pure torque on the bar, the blade must stand square to the bar and
the drop link must pull square to the blade. Off-square angles put bending into
the bar and the blade and change the effective rate in ways the rate formula does
not capture.

## How Vahan checks it
`vahan.packaging._axle_geometry_laws` reports the three angles:
`triad_bar_blade_deg`, `triad_blade_drop_deg`, `triad_bar_drop_deg`. When a
solution is built or relocated, the ARB is constructed to hold the baseline triad
angles by construction, and the oracle compares them to the baseline.

## Tolerance
Within 1 degree of the baseline triad angles.

## History — a wrong reading of this rule (2026-09-01)
An earlier version of this file said the bar↔drop angle was "a consequence of
packaging, not a target." That is WRONG and it shipped a non-triad: v74 measured
bar⊥blade 90 / blade⊥drop 90 but bar↔drop 104°, because `arb_drop_top.x` (155)
did not equal the bar half-span (170) — a 15 mm along-bar component in the drop
link, i.e. bending, not pure torque. The user caught it by eye. A drop link that
is square to the blade but NOT square to the bar puts a lateral (along-bar) force
component into the bar — exactly the bending the rule exists to prevent. Fix was
to align all three x-coordinates (half-span 140): 90.00 / 90.00 / 90.00.

## Note for bottom / control-arm bars
A torsion bar mounted low on the chassis (a "bottom ARB", Rule 04) is exempt from
the drop-link-in-plane law (Rule 04) but the three-way-perpendicular triad still
applies to its bar/blade/drop link.
