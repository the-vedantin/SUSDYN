# Rule 04 — Drop link in the rocker plane at static (bellcrank ARB)

## The rule
For a bellcrank anti-roll bar — the bar rides on the rocker, and the drop link
connects it into the actuation — the drop link lies in the rocker plane at static
(both ends). Across travel the blade end arcs with the bar and cannot stay
exactly in the plane; that lean is unavoidable and is minimized by design, not
held to zero.

## Why
At static (the design load case for setting rates) the drop link should pull the
rocker in its own plane of motion, so the rocker does not twist the bar off-axis.
This is the static counterpart of Rule 01 for the ARB branch.

## The exemption — bottom / control-arm bars
This rule is a BELLCRANK rule. A bottom (control-arm) anti-roll bar mounts the
torsion bar low on the chassis, well below the rocker, with a long drop link
reaching down to it. There the bar is chassis-fixed and the drop link is a
two-force rod-end member — it is never in bending regardless of plane — so the
in-plane requirement does not apply.

Vahan detects this automatically: if the bar pivot sits more than 120 mm below
the rocker pivot in Z, the layout is treated as a bottom ARB and the in-plane
check is skipped (`_axle_geometry_laws` sets `arb_is_bottom = True`; the same
120 mm test guards the regression net's "design actuation" gate). The triad
(Rule 03), rate (Rule 07), and clash (Rule 09) laws still apply. The threshold
cleanly separates the bellcrank bars (bar about 17 mm below the rocker) from the
2027 v73 bottom bar (about 250 mm below).

## How Vahan checks it
`_axle_geometry_laws` reports `arb_drop_top_inplane_mm` and
`arb_arm_end_inplane_mm` (both set to zero when the bottom-ARB exemption applies).

## Tolerance
Under 3 mm at static for a bellcrank bar; exempt for a bottom bar.
