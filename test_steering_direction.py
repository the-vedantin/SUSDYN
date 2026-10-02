"""Signed rack linkage and saved input direction regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VAHAN_MCP', '0')
import json
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from vahan.steering import SteeringGeometry
from gui.main_window import _rack_travel_from_angle, _ackermann_from_pair
from gui.panels import SteeringPanel


class SteeringDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def geometry(linkage=1, direction=1):
        # Factory supplies toe-in radians, including unequal static alignment.
        def probe(rack, side):
            yaw = linkage * 12.0 * rack
            return (.013 - yaw) if side == 'FL' else (-.008 + yaw)
        return SteeringGeometry.from_probe(probe, {}, {}, 100., 100.,
                                           rack_direction=direction)

    def test_reversed_linkage_preserves_physical_sign(self):
        normal = self.geometry()
        reverse = self.geometry(-1)
        self.assertAlmostEqual(float(normal.road_wheel_from_rack(.02)), .24)
        self.assertAlmostEqual(float(reverse.road_wheel_from_rack(.02)), -.24)
        self.assertAlmostEqual(float(reverse.road_wheel_from_rack(0)), 0)
        expected_ratio = 1 / np.degrees(12 * .1 / 360)
        self.assertAlmostEqual(normal.overall_ratio_deg_per_deg, expected_ratio)

    def test_input_direction_and_inverse(self):
        reverse = self.geometry(-1, -1)
        for sw in [-100., -25., 0., 25., 100.]:
            yaw = reverse.road_wheel_from_steering_wheel(sw)
            self.assertAlmostEqual(float(yaw), np.radians(sw) / reverse.overall_ratio_deg_per_deg)
            self.assertAlmostEqual(float(reverse.steering_wheel_from_road_wheel(yaw)), sw)
            self.assertAlmostEqual(float(reverse.rack_mm_from_road_wheel(yaw)), -sw*100/360)

    def test_rack_helper_clamps_after_direction(self):
        p = dict(rack_travel_per_rev_mm=100., total_rack_travel_mm=80., rack_direction=-1)
        self.assertAlmostEqual(_rack_travel_from_angle(36., p), -.01)
        self.assertAlmostEqual(_rack_travel_from_angle(900., p), -.04)
        del p['rack_direction']
        self.assertAlmostEqual(_rack_travel_from_angle(36., p), .01)

    def test_panel_saved_direction_and_legacy_default(self):
        p = SteeringPanel()
        p.set_params(dict(rack_travel_per_rev_mm=102.71, total_rack_travel_mm=97., rack_direction=-1))
        saved = json.loads(json.dumps(p.get_params()))
        q = SteeringPanel(); q.set_params(saved)
        self.assertEqual(q.get_params()['rack_direction'], -1)
        q._rack_ratio.setValue(103.)
        self.assertEqual(q.get_params()['rack_direction'], -1)
        q.set_params(dict(rack_travel_per_rev_mm=102.71, total_rack_travel_mm=97.))
        self.assertEqual(q.get_params()['rack_direction'], 1)
        self.assertAlmostEqual(q.get_params()['rack_travel_per_rev_mm'], 102.71)
        p.close(); q.close()

    def test_signed_ackermann_infers_inner_from_yaw(self):
        left = _ackermann_from_pair(-52., 36., 1.537, 1.270, inner=None)
        right = _ackermann_from_pair(36., -52., 1.537, 1.270, inner=None)
        self.assertGreater(left, 0)
        self.assertAlmostEqual(left, right)
        self.assertLess(_ackermann_from_pair(52., -36., 1.537, 1.270, inner=None), 0)


if __name__ == '__main__':
    unittest.main()
