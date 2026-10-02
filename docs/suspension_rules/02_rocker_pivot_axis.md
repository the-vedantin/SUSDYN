# Rule 02 — Rocker pivot axis

## The rule
The rocker (bellcrank) turns about an axis that is perpendicular to the
actuation plane. The pivot-axis point is not a free hardpoint — it is derived:

    rocker_axis_pt = rocker_pivot + (unit plane normal) x (axis length)

## Why
The axis is normal to this corner's static actuation plane so that the rocker attachments rotate in their intended plane. This does not require the wheel-side pushrod pickup to remain in the same plane throughout suspension travel. Each corner has its own plane; see the user's 2026-09-15 clarification in Rule 01.

## How Vahan holds it
Whenever a chain point is relocated, the resolver regenerates `rocker_axis_pt`
as the pivot plus the new plane normal times the original axis length
(`vahan.relocate.resolve_bundle`). The point is treated as derived, never edited
directly — trying to relocate `rocker_axis_pt` itself is refused with the note
"move rocker_pivot instead."

## Tolerance
Exact by construction. The axis length (the physical width of the rocker bearing
boss) is preserved from the baseline.
