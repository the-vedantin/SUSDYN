# Rule 20 — Rear toe-link inner point IS the aft LCA inboard point

HARD RULE (user, 2026-09-21): the rear toe link's inboard joint must be the SAME point as the rear
lower control arm's aft inboard pickup (`rear_hp['tie_rod_inner'] == rear_hp['lca_rear']`).
The lower A-arm and the toe link share that chassis pickup (one bracket / one welded lower assembly).

Consequences: the toe link's outboard joint must sit on (or within a few mm of) the LCA plane for
zero bump steer, so inside the rim its distance from the LCA outer ball joint is capped by the rim
clearance rule — the v142 "own chassis bracket 243 mm aft" layout (2026-09-21) is REJECTED by this rule.

Gate: regression net, "rear toe inner on LCA rear pickup" (test_one_model.py).
