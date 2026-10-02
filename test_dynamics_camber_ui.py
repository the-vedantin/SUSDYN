"""Dynamics camber display must use the road referenced result."""

import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6.QtWidgets import QApplication

from gui.main_window import CurvesCanvas, MainWindow, _DynamicsSweepWorker
from gui.panels import DynamicsPanel


class DynamicsCamberUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_plot_uses_road_camber_and_missing_values_are_gaps(self):
        sweep = {'lateral_g': np.array([0.0, 1.0]),
                 'camber_FL': np.array([0.0, -0.5]),
                 'camber_ground_FL': np.array([-0.8, 1.2])}
        canvas = CurvesCanvas()
        canvas.plot_dynamics(sweep, graphs=['camber'], corners=['FL', 'FR'])
        ax = canvas.fig.axes[0]
        self.assertIn('road', ax.get_ylabel().lower())
        lines = {line.get_label(): np.asarray(line.get_ydata(), float)
                 for line in ax.lines if line.get_label() in ('FL', 'FR')}
        np.testing.assert_allclose(lines['FL'], [-0.8, 1.2])
        self.assertTrue(np.isnan(lines['FR']).all())

    def test_table_uses_road_camber_and_marks_missing(self):
        panel = DynamicsPanel()
        result = SimpleNamespace(
            Fz={}, travel={}, camber={'FL': 0.0, 'FR': 0.0},
            camber_ground={'FL': -0.8}, utilization={},
            geometric_lt_front_N=0.0, geometric_lt_rear_N=0.0,
            elastic_lt_front_N=0.0, elastic_lt_rear_N=0.0,
            unsprung_lt_front_N=0.0, unsprung_lt_rear_N=0.0,
            jacking_corner_N={}, roll_angle_deg=0.0,
            pitch_angle_deg=0.0, rc_height_front_m=0.0,
            rc_height_rear_m=0.0, lateral_g=0.0,
            longitudinal_g=0.0, iterations=1,
            understeer_gradient_deg=0.0)
        panel.show_result(result)
        self.assertIn('road', panel._result_table.verticalHeaderItem(2).text().lower())
        self.assertEqual(panel._result_table.item(2, 0).text(), '-0.800')
        self.assertEqual(panel._result_table.item(2, 1).text(), '—')

    def test_roll_sweep_uses_each_corner_track(self):
        seen = {}
        class Probe:
            def _do_sweep(self, solver, travels, side, **kwargs):
                seen[kwargs['label']] = np.asarray(travels)
                return {}

        corners = {label: {'wheel_center': np.array([x, 0.0, 0.0])}
                   for label, x in (('FL', 0.611), ('FR', -0.611),
                                    ('RL', 0.600), ('RR', -0.600))}
        job = dict(motion='roll', lo=-3.0, hi=3.0, car={},
                   topology=SimpleNamespace(front=None, rear=None),
                   corners=corners, solvers={c: object() for c in corners},
                   arb_front={}, arb_rear={},
                   alignment=dict(front_camber_deg=0.0, rear_camber_deg=0.0,
                                  front_toe_deg=0.0, rear_toe_deg=0.0),
                   spring_limits={c: (0.0, 1.0) for c in corners})
        MainWindow._compute_sweep(Probe(), job)
        self.assertAlmostEqual(seen['FL'][-1], np.sin(np.radians(3)) * 0.611)
        self.assertAlmostEqual(seen['RL'][-1], np.sin(np.radians(3)) * 0.600)
        self.assertAlmostEqual(seen['RR'][-1], -seen['RL'][-1])

    def test_aero_worker_keeps_raw_and_road_camber_distinct(self):
        result = SimpleNamespace(
            roll_angle_deg=1.0, pitch_angle_deg=0.0,
            rc_height_front_m=0.04, rc_height_rear_m=0.05,
            elastic_lt_front_N=1.0, elastic_lt_rear_N=1.0,
            geometric_lt_front_N=1.0, geometric_lt_rear_N=1.0,
            understeer_gradient_deg=0.0,
            jacking_force_front_N=0.0, jacking_force_rear_N=0.0,
            jacking_heave_front_mm=0.0, jacking_heave_rear_mm=0.0,
            Fz={}, travel={}, camber={'FL': -0.2},
            camber_ground={'FL': 0.9}, inclination={'FL': -0.9},
            utilization={})
        class FakeSolver:
            def solve(self, *args, **kwargs):
                return result

        worker = _DynamicsSweepWorker(FakeSolver(), 0.0, 1.0, 2,
                                      aero_Fz_per_g={'FL': 1.0})
        sweep = worker._sweep_with_aero()
        np.testing.assert_allclose(sweep['camber_FL'], [-0.2, -0.2])
        np.testing.assert_allclose(sweep['camber_ground_FL'], [0.9, 0.9])
        np.testing.assert_allclose(sweep['inclination_FL'], [-0.9, -0.9])
        self.assertTrue(np.isnan(sweep['camber_ground_FR']).all())


if __name__ == '__main__':
    unittest.main()
