# Rule 04 — Drop link in the rocker plane at static (bellcrank ARB)

## Current mandatory interpretation — user clarification, 2026-09-15

Each corner has its own independent actuation plane; no common left/right plane is required. Both drop-link joint centres must lie in that corner’s actual actuation/rocker plane at
static, with zero offset as the design target and the documented numerical
tolerance reported explicitly. A parallel offset plane is NOT coplanarity.
Declaring a standoff or a waiver in a saved file does not make that geometry
compliant. Report physical endpoint distances before any offset convention.
For a symmetric double-shear rocker, this is the central force plane between
its two plates, not either outer plate face. Both joint centers must still lie
in that same central plane.
The v105–v114 offset convention described below is historical, not authorization
to use it for this design. v114 is noncompliant despite its old regression pass.

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

## Historical declared rod-end standoff (v105; not current acceptance)
The drop-top rod end may sit on a spacer that stands it off the rocker plate along
the rocker-plane normal. Declared in the project as `car['front_arb_drop_standoff_mm']`
(or `rear_…`; sign: positive toward +X outboard along the normal). The whole link
then runs in a plane PARALLEL to the rocker plane. This can reduce the force component
normal to the plate, but an offset force still applies a moment r × F to the support
and rocker. Parallel planes do NOT prove absence of bending or twist. The law measures both link ends against the declared
offset plane (`arb_drop_top_inplane_mm`, `arb_arm_end_inplane_mm` are the residuals
from that plane; `arb_link_plane_offset_mm` reports the measured offset) and the
coplanarity law removes the declared standoff from the drop top before its fit.
An undeclared offset still fails. The standoff is a HARDWARE item (spacer length,
bolt in bending) and is flagged in the engineering note of the version that uses it
(v105: −18 mm, inboard face of the front rocker). Passing this geometric law does
not establish physical clearance or support strength. Check the full link and rod-end
envelopes against the actual plate, boss and mounting hardware through travel; size
the offset support for its force and moment. Double shear alone does not eliminate
the moment transferred to that support.

## Historical declared waiver (v106; metadata is not approval)
`car['front_arb_rule04_waiver'] = "<reason>"` (or `rear_…`) records a user decision to run the drop link
OUT of the rocker plane. The law keeps reporting the true residuals and sets `arb_inplane_waived`;
`packaging.validate`, Design City and the regression net then accept the in-plane residuals and print
"RULE 04 WAIVED" on every run. The cost is a side load on the rocker drop tab (v106: 53 % of the link
force, 418 N at 1.5 g) — the reason the rule exists. Never set the key on the user's behalf.

## Tolerance
Under 3 mm at static for a bellcrank bar; exempt for a bottom bar.
