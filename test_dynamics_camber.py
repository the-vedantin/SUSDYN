"""Camber frame/serialization regressions, independent of private tire data."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vahan.dynamics import VehicleParams, SteadyStateSolver


class DynamicsCamberTests(unittest.TestCase):
    def solver(self):
        solver = SteadyStateSolver(VehicleParams(
            camber_front_deg=-2., camber_rear_deg=-1.), {})
        solver.jacking_feedback = False
        return solver

    def test_all_sweep_routes_preserve_ground_and_tire_angles(self):
        calls = (
            ('sweep_lateral_g', dict(g_range=(0., .3), n_points=3)),
            ('sweep_longitudinal_g', dict(g_range=(0., .3), n_points=3)),
            ('sweep_combined', dict(lat_range=(0., .3), n_points=3)),
            ('sweep_by_speed', dict(v_min_mph=0., v_max_mph=4.,
                                    turn_radius_m=20., n_points=3)),
            ('sweep_acceleration', dict(v_min_kph=0., v_max_kph=4., n_points=3)),
            ('sweep_acceleration_trajectory', dict(max_steps=3, target_lon_g=.2)),
        )
        for name, kwargs in calls:
            with self.subTest(route=name):
                ss = self.solver()
                data = getattr(ss, name)(**kwargs)
                for corner in ('FL', 'FR', 'RL', 'RR'):
                    self.assertIn('camber_ground_' + corner, data)
                    self.assertIn('inclination_' + corner, data)
                    raw = data['camber_' + corner]
                    roll = data['roll_angle_deg']
                    alignment = -2. if corner.startswith('F') else -1.
                    sign = 1. if corner.endswith('L') else -1.
                    # Reduced zero-gain fixture: road lean = alignment +/- roll.
                    expected = raw + alignment + sign * roll
                    np.testing.assert_allclose(data['camber_ground_' + corner], expected,
                                               atol=1e-10)
                    # Both sides have a separate tire-frame field, never |camber|.
                    self.assertEqual(len(data['inclination_' + corner]), len(raw))

    def test_steered_spin_uses_full_road_plane_not_front_view_projection(self):
        # 60-degree wheel heading, 10-degree true inclination. Front-view
        # atan(z/x) wrongly gives 19.425 degrees. Independent arcsin(dot) oracle.
        heading, inclination = np.radians([60., 10.])
        spin = np.array([np.cos(inclination)*np.cos(heading),
                         np.cos(inclination)*np.sin(heading),
                         -np.sin(inclination)])
        ss = self.solver()
        ss._solvers = dict.fromkeys(('FL', 'FR', 'RL', 'RR'), object())
        ss._solve_corner = lambda *a: SimpleNamespace(spin_axis=spin)
        ss._query_rc_height = lambda *a: .05
        class Metrics:
            def __init__(self, state, side):
                self.roll_center_height = .05
                sign = 1. if side == 'left' else -1.
                self.camber = -sign * np.degrees(np.arctan2(spin[2], abs(spin[0])))
        with patch('vahan.dynamics.KinematicMetrics', Metrics):
            result = ss.solve(.3)
        for corner in ('FL', 'FR', 'RL', 'RR'):
            side = 1. if corner.endswith('L') else -1.
            static = -2. if corner.startswith('F') else -1.
            # Existing renderer convention: alignment and body roll about +Y.
            angle = np.radians(side * static + result.roll_angle_deg)
            z_ground = -np.sin(angle)*spin[0] + np.cos(angle)*spin[2]
            expected = -side * np.degrees(np.arcsin(z_ground))
            self.assertAlmostEqual(result.camber_ground[corner], expected, places=10)

    def test_configured_corner_failure_is_not_zero_camber(self):
        ss = self.solver()
        ss._solvers['FL'] = SimpleNamespace(
            solve=lambda *a, **k: (_ for _ in ()).throw(ValueError('unreachable travel')))
        with self.assertRaisesRegex(RuntimeError, 'FL.*kinematic'):
            ss.solve(.3)


if __name__ == '__main__':
    unittest.main()
