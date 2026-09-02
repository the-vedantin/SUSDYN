# Rule 02 — Rocker pivot axis

## The rule
The rocker (bellcrank) turns about an axis that is perpendicular to the
actuation plane. The pivot-axis point is not a free hardpoint — it is derived:

    rocker_axis_pt = rocker_pivot + (unit plane normal) x (axis length)

## Why
If the rocker's rotation axis is not the plane normal, the rocker sweeps its
attached points (pushrod inner, spring eye, drop top) out of the actuation plane
as it turns, which breaks coplanarity (Rule 01) and loads members in bending.
Making the axis exactly the plane normal is what keeps the inboard points in the
plane across travel.

## How Vahan holds it
Whenever a chain point is relocated, the resolver regenerates `rocker_axis_pt`
as the pivot plus the new plane normal times the original axis length
(`vahan.relocate.resolve_bundle`). The point is treated as derived, never edited
directly — trying to relocate `rocker_axis_pt` itself is refused with the note
"move rocker_pivot instead."

## Tolerance
Exact by construction. The axis length (the physical width of the rocker bearing
boss) is preserved from the baseline.
