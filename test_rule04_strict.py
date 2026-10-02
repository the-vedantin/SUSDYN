"""Focused regressions for the physical rocker-plate Rule 04 gate."""
import json
import os
import unittest

import numpy as np

from vahan.packaging import (actuation_chain_plate_metrics,
                             arb_drop_link_plate_compliant,
                             arb_drop_link_plate_metrics)


class Rule04PhysicalPlateTest(unittest.TestCase):
    @staticmethod
    def _reference_plate():
        return {'rocker_pivot': np.array([0.0, 0.0, 0.0]),
                'pushrod_inner': np.array([0.0, 0.1, 0.0]),
                'rocker_spring_pt': np.array([0.0, 0.0, 0.1])}

    def test_packaging_sweep_reports_real_plate_penetration(self):
        from types import SimpleNamespace
        from vahan.packaging import _clash_sweep
        points = dict(self._reference_plate(),
                      pushrod_outer=np.array([0.0, 0.02, 0.02]))
        corners = [{'label': label, 'pts': {k: v + [i, 0., 0.]
                   for k, v in points.items()}}
                   for i, label in enumerate(('FL', 'FR', 'RL', 'RR'))]
        win = SimpleNamespace(_car={}, _front_arb={}, _rear_arb={},
            _assemble_corners_draw=lambda *args: (corners, None))
        hits = _clash_sweep(win, [0.0])['+0.000 mm']
        plate_hits = [hit for hit in hits if hit['b'] == 'rocker plate']
        self.assertEqual({hit['corner'] for hit in plate_hits},
                         {'FL', 'FR', 'RL', 'RR'})
        self.assertTrue(all(hit['gap_mm'] < 0 for hit in plate_hits))

    def test_packaging_sweep_requires_plate_clearance_margin(self):
        from types import SimpleNamespace
        from vahan.packaging import _clash_sweep
        points = dict(self._reference_plate(),
            tie_rod_inner=np.array([0.0129375, 0.01, 0.03]),
            tie_rod_outer=np.array([0.0129375, 0.04, 0.03]))
        corners = [{'label': label, 'pts': {k: v + [i, 0., 0.]
                   for k, v in points.items()}}
                   for i, label in enumerate(('FL', 'FR', 'RL', 'RR'))]
        win = SimpleNamespace(_car={}, _front_arb={}, _rear_arb={},
            _assemble_corners_draw=lambda *args: (corners, None))
        hits = _clash_sweep(win, [0.0])['+0.000 mm']
        close = [hit for hit in hits if hit['a'] == 'tie / toe rod'
                 and hit['b'] == 'rocker plate']
        self.assertEqual(len(close), 4)
        for hit in close:
            self.assertAlmostEqual(hit['surface_gap_mm'], 2.0)
            self.assertAlmostEqual(hit['gap_mm'], -1.0)

    def test_48mm_translated_orthogonal_triad_is_noncompliant(self):
        # Known reference construction: plate X=0; bar, blade and drop link
        # follow X, Z and Y respectively.  Translating every ARB point +48 mm X
        # preserves those three directions while moving both link endpoints off
        # the physical rocker plate.
        hp = self._reference_plate()
        arb = {'arb_drop_top': np.array([0.048, 0.1, 0.1]),
               'arb_arm_end': np.array([0.048, 0.0, 0.1]),
               'arb_pivot': np.array([0.048, 0.0, 0.0])}
        measured = arb_drop_link_plate_metrics(hp, arb)
        self.assertAlmostEqual(measured['drop_top_signed_mm'], 48.0)
        self.assertAlmostEqual(measured['arm_end_signed_mm'], 48.0)
        self.assertAlmostEqual(measured['direction_deg'], 0.0)
        self.assertFalse(arb_drop_link_plate_compliant(measured, 3.0))

    def test_inplane_reference_link_is_compliant(self):
        arb = {'arb_drop_top': np.array([0.0, 0.1, 0.1]),
               'arb_arm_end': np.array([0.0, 0.0, 0.1]),
               'arb_pivot': np.array([0.0, 0.0, 0.0])}
        measured = arb_drop_link_plate_metrics(self._reference_plate(), arb)
        self.assertAlmostEqual(measured['drop_top_signed_mm'], 0.0)
        self.assertAlmostEqual(measured['arm_end_signed_mm'], 0.0)
        self.assertAlmostEqual(measured['direction_deg'], 0.0)
        self.assertTrue(arb_drop_link_plate_compliant(measured, 3.0))

    def test_static_corner_check_ignores_nonstatic_travel_state(self):
        base = dict(self._reference_plate(),
                    spring_chassis_pt=np.array([0.0, -0.1, 0.1]),
                    rocker_axis_pt=np.array([0.1, 0.0, 0.0]))
        static = dict(base, pushrod_outer=np.array([0.0136, 0.1, -0.1]))
        bump = dict(base, pushrod_outer=np.array([0.03686, 0.1, -0.1]))
        measured = actuation_chain_plate_metrics([(0.0, static), (0.05, bump)])
        self.assertAlmostEqual(measured['coplanar_static_mm'], 13.6)
        self.assertAlmostEqual(measured['coplanar_mm'], 13.6)
        self.assertEqual(measured['worst_point'], 'pushrod_outer')
        self.assertAlmostEqual(measured['worst_travel_m'], 0.0)
        self.assertAlmostEqual(measured['rocker_axis_normal_error_deg'], 0.0)

    def test_static_corner_check_requires_explicit_zero_state(self):
        base = dict(self._reference_plate(),
                    spring_chassis_pt=np.array([0.0, -0.1, 0.1]),
                    rocker_axis_pt=np.array([0.1, 0.0, 0.0]),
                    pushrod_outer=np.array([0.0, 0.1, -0.1]))
        with self.assertRaisesRegex(ValueError, 'explicit zero-travel'):
            actuation_chain_plate_metrics([(-0.01, base), (0.01, base)])

    def test_axle_law_checks_left_and_right_static_planes_independently(self):
        from types import SimpleNamespace
        from vahan.packaging import _axle_geometry_laws

        def state(base_x, outer_delta=0.0):
            return SimpleNamespace(
                rocker_pivot=np.array([base_x, 0., 0.]),
                pushrod_inner=np.array([base_x, .1, 0.]),
                rocker_spring_pt=np.array([base_x, 0., .1]),
                pushrod_outer=np.array([base_x + outer_delta, .1, -.1]),
                spring_chassis_pt=np.array([base_x, -.1, .1]))

        class StaticSolver:
            def __init__(self, st, axis):
                self.st = st; self.calls = []
                self.hp = SimpleNamespace(rocker_pivot=st.rocker_pivot,
                                          rocker_axis_pt=st.rocker_pivot + axis)
            def solve(self, travel):
                self.calls.append(travel); return self.st

        fl = StaticSolver(state(.1), np.array([1., 0., 0.]))
        fr = StaticSolver(state(-.1), np.array([-1., 0., 0.]))
        arb = {'arb_pivot': np.array([.1, 0., 0.]),
               'arb_arm_end': np.array([.1, 0., .1]),
               'arb_drop_top': np.array([.1, .1, .1])}
        topology = SimpleNamespace(front=SimpleNamespace(
            arb_type=SimpleNamespace(value='bellcrank')))
        win = SimpleNamespace(_solvers={'FL': fl, 'FR': fr},
                              _front_arb=arb, _topology=topology, _car={})
        measured = _axle_geometry_laws(win, 'front')
        self.assertAlmostEqual(measured['corner_static_mm']['FL'], 0.0)
        self.assertAlmostEqual(measured['corner_static_mm']['FR'], 0.0)
        self.assertAlmostEqual(measured['coplanar_mm'], 0.0)
        self.assertEqual(fl.calls, [0.0])
        self.assertEqual(fr.calls, [0.0])

    def test_static_drop_link_endpoints_are_part_of_corner_plane_gate(self):
        state = dict(self._reference_plate(),
                     pushrod_outer=np.array([0.0, .1, -.1]),
                     spring_chassis_pt=np.array([0.0, -.1, .1]),
                     rocker_axis_pt=np.array([.1, 0., 0.]))
        arb = {'arb_drop_top': np.array([.004, .1, .1]),
               'arb_arm_end': np.array([0.0, 0.0, .1])}
        measured = actuation_chain_plate_metrics([(0.0, state)], arb=arb)
        self.assertAlmostEqual(measured['coplanar_mm'], 4.0)
        self.assertEqual(measured['worst_point'], 'arb_drop_top')

    def test_saved_v114_standoff_metadata_does_not_make_geometry_pass(self):
        path = os.path.join(
            'configs',
            '2027_v114_(front_arb_outboard_coilover_clearance_WIP).vahan')
        if not os.path.isfile(path):
            self.skipTest('private v114 design fixture is not present')
        with open(path, encoding='utf-8') as stream:
            project = json.load(stream)
        self.assertEqual(project['car']['front_arb_drop_standoff_mm'], 48.0)
        measured = arb_drop_link_plate_metrics(project['front_hp'],
                                                project['front_arb'])
        self.assertGreater(abs(measured['drop_top_signed_mm']), 40.0)
        self.assertGreater(abs(measured['arm_end_signed_mm']), 40.0)
        self.assertFalse(arb_drop_link_plate_compliant(measured, 3.0))
        chain = actuation_chain_plate_metrics([(0.0, project['front_hp'])])
        self.assertGreater(chain['coplanar_static_mm'], 10.0)

    def test_degenerate_plate_fails_closed(self):
        hp = {'rocker_pivot': np.array([0.0, 0.0, 0.0]),
              'pushrod_inner': np.array([0.0, 0.1, 0.0]),
              'rocker_spring_pt': np.array([0.0, 0.2, 0.0])}
        arb = {'arb_drop_top': np.array([0.0, 0.1, 0.1]),
               'arb_arm_end': np.array([0.0, 0.0, 0.1])}
        measured = arb_drop_link_plate_metrics(hp, arb)
        self.assertTrue(np.isnan(measured['drop_top_signed_mm']))
        self.assertTrue(np.isnan(measured['arm_end_signed_mm']))
        self.assertTrue(np.isnan(measured['direction_deg']))
        self.assertFalse(arb_drop_link_plate_compliant(measured, 3.0))
        chain_state = dict(hp,
                           pushrod_outer=np.array([0.0, 0.0, 0.0]),
                           spring_chassis_pt=np.array([0.0, 0.0, 0.0]),
                           rocker_axis_pt=np.array([0.1, 0.0, 0.0]))
        chain = actuation_chain_plate_metrics([(0.0, chain_state)])
        self.assertTrue(np.isnan(chain['coplanar_mm']))
        self.assertTrue(np.isnan(chain['rocker_axis_normal_error_deg']))

    def test_live_solved_states_use_configured_rocker_axis(self):
        path = os.path.join(
            'configs',
            '2027_v114_(front_arb_outboard_coilover_clearance_WIP).vahan')
        if not os.path.isfile(path):
            self.skipTest('private v114 design fixture is not present')
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ['VAHAN_MCP'] = '0'
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from vahan.packaging import _axle_geometry_laws
        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        try:
            win._load_project_from_path(path)
            front = _axle_geometry_laws(win, 'front')
            rear = _axle_geometry_laws(win, 'rear')
            self.assertGreater(front['coplanar_mm'], 30.0)
            self.assertGreater(front['arb_drop_top_inplane_mm'], 40.0)
            self.assertTrue(np.isfinite(front['rocker_axis_normal_error_deg']))
            self.assertTrue(np.isfinite(rear['rocker_axis_normal_error_deg']))
        finally:
            win.close()

    def test_only_control_arm_topology_gets_rule04_exemption(self):
        measured = {'drop_top_signed_mm': 48.0, 'arm_end_signed_mm': 48.0}
        self.assertFalse(arb_drop_link_plate_compliant(measured, 3.0,
                                                       control_arm_arb=False))
        self.assertTrue(arb_drop_link_plate_compliant(measured, 3.0,
                                                      control_arm_arb=True))


if __name__ == '__main__':
    unittest.main()
