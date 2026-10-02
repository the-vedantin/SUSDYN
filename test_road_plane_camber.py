"""Analytic inclination checks independent of the suspension linkage solver."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from vahan.kinematics import road_plane_camber_deg


class RoadPlaneCamberTests(unittest.TestCase):
    def test_heading_does_not_change_inclination(self):
        for heading in (-80., -30., 0., 60., 89.):
            for camber in (-10., 0., 7.):
                h, c = np.radians([heading, camber])
                spin = [np.cos(c)*np.cos(h), np.cos(c)*np.sin(h), -np.sin(c)]
                self.assertAlmostEqual(road_plane_camber_deg(spin), camber, places=11)

    def test_side_and_alignment_rotation(self):
        for side, side_sign in [('left', 1.), ('right', -1.)]:
            spin = Rotation.from_euler('y', side_sign * -2., degrees=True).apply([1., 0., 0.])
            self.assertAlmostEqual(road_plane_camber_deg(spin, side=side), -2.)

    def test_common_rotation_and_input_scaling(self):
        spin = np.array([1., 0., .2])
        normal = np.array([0., 0., 1.])
        before = road_plane_camber_deg(spin, normal)
        rotation = Rotation.from_euler('xyz', [21., -15., 72.], degrees=True)
        self.assertAlmostEqual(road_plane_camber_deg(rotation.apply(spin), rotation.apply(normal)), before)
        self.assertAlmostEqual(road_plane_camber_deg(spin*1e300, normal*1e-300), before)

    def test_road_bank_and_vertical_wheel(self):
        normal = Rotation.from_euler('y', 5., degrees=True).apply([0., 0., 1.])
        self.assertAlmostEqual(road_plane_camber_deg([1., 0., 0.], normal), -5.)
        self.assertAlmostEqual(road_plane_camber_deg([0., 0., 1.]), -90.)

    def test_invalid_inputs(self):
        for bad in ([0., 0., 0.], [float('nan'), 0., 1.], [0., float('inf'), 1.], [1., 0.], [[1., 0., 0.]]):
            with self.assertRaises(ValueError):
                road_plane_camber_deg(bad)
            with self.assertRaises(ValueError):
                road_plane_camber_deg([1., 0., 0.], bad)
        with self.assertRaises(ValueError):
            road_plane_camber_deg([1., 0., 0.], side='outer')


if __name__ == '__main__':
    unittest.main()
