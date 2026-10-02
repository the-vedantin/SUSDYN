# Suspension Design Rules (Cougar Racing 2027 / Vahan)

These are the standing rules that every front and rear suspension design in Vahan
must obey. Each rule has its own file: what the rule is, why it exists, how Vahan
checks it, and the tolerance it is held to. The regression net
(`python test_one_model.py`) enforces most of them on the highest-numbered
`configs/2027_v*.vahan` automatically.

Units: millimetres and degrees. Frame: X lateral (outboard +), Y longitudinal
(rearward +), Z vertical (up +).

## Index

| # | Rule | One line |
|---|------|----------|
| [00](00_one_model.md) | One Model | Every number comes from the single solved model; no duplicated physics. |
| [01](01_coplanarity_no_bending.md) | Coplanarity / nothing in bending | Each corner’s complete actuation chain is coplanar at static; each corner has its own plane. |
| [02](02_rocker_pivot_axis.md) | Rocker pivot axis | The rocker turns about the actuation-plane normal. |
| [03](03_arb_triad.md) | Anti-roll-bar triad | Torsion bar, blade, and drop link are mutually perpendicular. |
| [04](04_arb_drop_link_in_plane.md) | Drop link in plane at static | A bellcrank ARB's drop link lies in the rocker plane at static. |
| [05](05_pushrod_not_pullrod.md) | Pushrod, not pullrod | The damper compresses in bump, monotonically. |
| [06](06_motion_ratio_preserved.md) | Motion ratio preserved | Re-tune the rocker so the motion ratio matches the baseline. |
| [07](07_arb_rate_preserved.md) | ARB rate preserved | Re-tune the bar so the roll rate matches the baseline. |
| [08](08_wheel_curves_exact.md) | Wheel curves held exact | Inboard-only moves never touch the wheel-locating points. |
| [09](09_clash_free_full_travel.md) | Clash-free across full travel | The full member set clears at droop, static, and bump. |
| [10](10_ground_contact.md) | Ground contact | The tires sit on the ground plane at design ride height. |
| [11](11_pushrod_lca_mount.md) | Pushrod-on-arm mount | Mount near the ball joint, never mid-arm. |
| [12](12_over_the_arm_plate.md) | Over-the-arm plate | The rod end sits about an inch above the arm plane. |
| [13](13_pushrod_alignment.md) | Pushrod alignment | Keep the pushrod near the X-Z plane, little Y lean. |
| [14](14_same_shocks_preload.md) | Same shocks: rate with spring/MR, sag with preload | Respect shock stroke and length; spring stiffness and motion ratio set wheel rate; preload sets installed force and sag. |
| [15](15_tie_rod_inline.md) | Tie rods inline | Tie-rod ends share the same X, not the same Y. |
| [16](16_rim_fit.md) | Fits inside the REAL rim, all four corners | Joint BODIES + member TUBES stay ≥3 mm inside the barrel at droop/static/bump (and front steer), FL and RL — `rim_fit()` checks centres only and is not the answer. |
| [17](17_roll_centre_instant_axis.md) | Roll centre from the instant axis | Arms are traced from their pivot AXIS on the wheel-centre plane, never from the pickup midpoint — a pickup slid along its own axis must not move the roll centre. |
| [18](18_chassis_keepout.md) | Chassis keep-out (red zone) | A STEP keep-out named by the project is a hard volume: no member inside it at droop/static/bump × lock/centre/lock — rack housing under the floor, torsion bar above the roof. |
| [19](19_front_hoop_line.md) | Front ARB ahead of the front-hoop line | Bar, blades, links and rod ends of the front ARB stay ≥ 3 mm ahead of the line through the LCA-aft / UCA-aft pickups extended upward, at droop/static/bump (user rule 2026-09-14). |

Tyre evidence note (2026-09-15): the tyre model can be pinned to one test-speed block (`car['tire_speed_window_kph']`), used for the rim-matched 7-in run-6 surface; see the top-level README.
