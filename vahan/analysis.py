"""
SuspensionAnalysis — sweeps wheel travel and collects kinematic results.

Usage:
    analysis = SuspensionAnalysis(hardpoints, side='left')
    results  = analysis.sweep((-50, 50), n_steps=101)

The returned dict contains numpy arrays (one value per step) for every metric
plus 3-D coordinate arrays for every moving hardpoint.
"""

import numpy as np
from .hardpoints import DoubleWishboneHardpoints
from .solver     import SuspensionConstraints
from .kinematics import KinematicMetrics


class SuspensionAnalysis:
    """
    High-level interface for kinematic sweeps.

    Args:
        hardpoints: DoubleWishboneHardpoints at design / static position.
        side:       'left' or 'right' (affects sign convention of angles).
    """

    def __init__(self, hardpoints: DoubleWishboneHardpoints, side: str = 'left'):
        self.hp          = hardpoints
        self.side        = side
        self.constraints = SuspensionConstraints(hardpoints)

    # ── full sweep ────────────────────────────────────────────────────────────

    def sweep(self,
              travel_range_mm: tuple,
              n_steps: int = 101) -> dict:
        """
        Sweep wheel travel and compute all kinematic metrics at every step.

        Args:
            travel_range_mm: (min, max) in mm.  e.g. (-50, 50).
                             Negative = droop, positive = bump.
            n_steps:         number of equally-spaced travel points.

        Returns:
            dict with keys:
                Scalar arrays (n_steps,):
                    travel_mm, camber_deg, toe_deg, caster_deg, kpi_deg,
                    scrub_radius_mm, mechanical_trail_mm, roll_center_height_mm,
                    spring_length_mm, rocker_angle_deg, motion_ratio

                3-D point arrays (n_steps, 3):
                    uca_outer, lca_outer, tr_outer, wheel_center,
                    pushrod_outer, pushrod_inner, rocker_spring_pt
        """
        travel_m = np.linspace(travel_range_mm[0] / 1000.0,
                               travel_range_mm[1] / 1000.0,
                               n_steps)

        # ── initialise output containers ─────────────────────────────────────
        scalar_keys = [
            'travel_mm', 'camber_deg', 'toe_deg', 'caster_deg', 'kpi_deg',
            'scrub_radius_mm', 'mechanical_trail_mm', 'roll_center_height_mm',
            'spring_length_mm', 'rocker_angle_deg',
        ]
        point_keys = [
            'uca_outer', 'lca_outer', 'tr_outer', 'wheel_center',
            'pushrod_outer', 'pushrod_inner', 'rocker_spring_pt',
        ]

        scalars = {k: np.empty(n_steps) for k in scalar_keys}
        points  = {k: np.empty((n_steps, 3)) for k in point_keys}

        # ── march through travel positions ────────────────────────────────────
        x_warm       = None    # warm-start for main solver (12-vector)
        theta_warm   = 0.0     # warm-start for rocker solver

        for i, travel in enumerate(travel_m):
            state = self.constraints.solve(
                travel,
                x0=x_warm,
                rocker_theta0=theta_warm,
            )

            # Update warm starts for next step
            x_warm     = state.x_vec()
            theta_warm = state.rocker_angle

            # Collect scalar metrics
            m = KinematicMetrics(state, self.side)
            sm = m.summary()
            for k in scalar_keys:
                scalars[k][i] = sm[k]

            # Collect 3-D point positions
            mp = state.all_moving_points()
            for k in point_keys:
                points[k][i] = mp[k]

        # ── motion ratio via central differences on spring length ─────────────
        # MR = d(spring_length) / d(wheel_travel)   [dimensionless]
        spring_m = scalars['spring_length_mm'] / 1000.0
        travel_m_arr = scalars['travel_mm'] / 1000.0
        motion_ratio = np.gradient(spring_m, travel_m_arr)

        # ── merge and return ──────────────────────────────────────────────────
        results = {**scalars, **points, 'motion_ratio': motion_ratio}
        return results

    # ── single-point solve ────────────────────────────────────────────────────

    def at(self, travel_mm: float) -> KinematicMetrics:
        """
        Solve and return KinematicMetrics at a single wheel-travel position.

        Useful for quick spot-checks without running a full sweep.
        """
        state = self.constraints.solve(travel_mm / 1000.0)
        return KinematicMetrics(state, self.side)
