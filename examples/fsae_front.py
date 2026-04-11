"""
FSAE-style front double wishbone + pushrod/rocker — kinematic sweep example.

Run from the SUSDYN directory:
    python examples/fsae_front.py

Coordinate system (chassis frame, metres):
    X  →  forward
    Y  →  outboard  (positive toward the left wheel on a left corner)
    Z  →  up
    Origin: vehicle centreline (Y=0), front axle (X=0), ground (Z=0)

Wheel radius assumed 203 mm (8-inch rim + 13" tyre OD ≈ 406 mm).
Track half-width to wheel centre ≈ 610 mm.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from vahan import DoubleWishboneHardpoints, SuspensionAnalysis

# ── hardpoints (metres) ───────────────────────────────────────────────────────
#
# These represent a typical FSAE front corner:
#   - Short UCA (~250 mm arm length inboard-to-outboard)
#   - Long LCA  (~310 mm arm length)
#   - UCA higher than LCA → negative camber gain in bump (desired)
#   - Tie rod at same height as LCA outer → near-zero bump steer
#   - Pushrod from upright to rocker mounted on chassis rail

hp = DoubleWishboneHardpoints(
    # Upper Control Arm ─────────────────────────────────────────
    uca_front   = np.array([ 0.055,  0.105,  0.270]),
    uca_rear    = np.array([-0.055,  0.105,  0.270]),
    uca_outer   = np.array([ 0.000,  0.575,  0.270]),

    # Lower Control Arm ─────────────────────────────────────────
    lca_front   = np.array([ 0.090,  0.075,  0.045]),
    lca_rear    = np.array([-0.090,  0.075,  0.045]),
    lca_outer   = np.array([ 0.000,  0.600,  0.045]),

    # Steering (tie rod at same height and Y-offset as LCA outer → low bump steer)
    tie_rod_inner = np.array([-0.025,  0.075,  0.045]),
    tie_rod_outer = np.array([-0.025,  0.580,  0.045]),

    # Wheel centre — 203 mm radius tyre, track half-width 610 mm
    wheel_center  = np.array([ 0.000,  0.610,  0.203]),

    # Pushrod / Rocker ──────────────────────────────────────────
    # Pushrod outboard: on the upright just above the LCA balljoint
    pushrod_outer    = np.array([ 0.015,  0.598,  0.090]),
    # Pushrod inboard:  lower arm of the rocker bell-crank
    pushrod_inner    = np.array([ 0.015,  0.130,  0.235]),
    # Rocker pivot:     chassis-fixed, slightly above pushrod_inner
    rocker_pivot     = np.array([ 0.015,  0.115,  0.255]),
    # Rocker spring arm: short arm above pivot, contacts spring/damper
    rocker_spring_pt = np.array([ 0.015,  0.130,  0.275]),
    # Spring top mount: chassis
    spring_chassis_pt= np.array([ 0.015,  0.145,  0.390]),
)

# ── run sweep ─────────────────────────────────────────────────────────────────

analysis = SuspensionAnalysis(hp, side='left')

print("Vahan -- kinematic sweep  (-50 mm droop -> +50 mm bump)")
print("=" * 65)

results = analysis.sweep((-50, 50), n_steps=101)

# ── print results table ───────────────────────────────────────────────────────

header = (f"{'Travel':>7}  {'Camber':>7}  {'Toe':>7}  "
          f"{'Caster':>7}  {'RC_h':>7}  {'MR':>6}")
units  = (f"{'(mm)':>7}  {'(deg)':>7}  {'(deg)':>7}  "
          f"{'(deg)':>7}  {'(mm)':>7}  {'':>6}")
print(header)
print(units)
print("-" * 65)

# Print every 10th step
for i in range(0, len(results['travel_mm']), 10):
    print(f"{results['travel_mm'][i]:7.1f}  "
          f"{results['camber_deg'][i]:7.3f}  "
          f"{results['toe_deg'][i]:7.4f}  "
          f"{results['caster_deg'][i]:7.3f}  "
          f"{results['roll_center_height_mm'][i]:7.2f}  "
          f"{results['motion_ratio'][i]:6.4f}")

print()
print("Spot-check at 0 mm travel:")
m = analysis.at(0.0)
for k, v in m.summary().items():
    print(f"  {k:<28s} {v:+.4f}")

print()
print("Done.  results dict keys:", list(results.keys()))
