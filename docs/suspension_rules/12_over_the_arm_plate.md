# Rule 12 — Over-the-arm plate

## Current dimensional instruction

The user specified 1.25 inches (31.75 mm) above the respective arm plane along
its upward normal, with a 1 inch (25.4 mm) in-plane inboard offset. This
supersedes the earlier approximate one-inch height used below. Front is UCA;
rear is LCA. Measure the actual saved geometry, not only the declared topology.

## The rule
When the pushrod picks up "over the arm," the pushrod rod end loads onto a plate
welded on top of the control arm. The one-inch spherical rod end (its centre is
the pushrod-outer point) sits about one inch above the arm plane — clear of the
arm's own tube thickness, never buried below the arm, and never flung far off it.

## Why
The rod end needs to sit proud of the arm so the spherical bearing and the plate
clear the arm tube and can be welded and bolted. Buried below the plane it fouls
the tube; slid far off the arm (for example toward the tie rod) it puts the load
where there is no structure and re-introduces bending.

## How Vahan checks it
The regression net's "design actuation" gate takes the signed perpendicular
distance from the pushrod-outer point to the plane through the arm's three
pickups, with the plane normal oriented up, and requires the rod end to sit above
the arm plane by roughly the rod-end offset (about one inch), not below it and not
far off it.

## Tolerance
Current target: 31.75 mm above the arm plane, always positive. Report deviation
from that target; do not substitute the obsolete approximately 25 mm target.
