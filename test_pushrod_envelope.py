"""The arm-side spherical joint must not disappear behind a thin tube test."""
import unittest
import numpy as np
from vahan.interference import full_members, rim_barrel_gap, clashes, cross_corner_clashes, connected_for


class PushrodEnvelopeTests(unittest.TestCase):
    def test_tie_joint_catches_clear_tube_at_barrel(self):
        pts = {'tie_rod_outer': np.array([0., 0., .109]),
               'tie_rod_inner': np.array([-.2, 0., .109])}
        members = {m['name']: m for m in full_members(pts, {})}
        kwargs = dict(wheel_center=np.zeros(3), spin_axis=np.array([1.,0.,0.]),
                      inner_radius_m=.12, half_width_m=.09)
        self.assertGreater(rim_barrel_gap(members['tie / toe rod'], **kwargs), .003)
        self.assertLess(rim_barrel_gap(members['tie outer joint'], **kwargs), 0.)
        self.assertEqual(clashes(list(members.values())), [])

    def test_joint_catches_clear_tube_at_barrel(self):
        pts = {'pushrod_outer': np.array([0., 0., .109]),
               'pushrod_inner': np.array([-.2, 0., .109])}
        members = {m['name']: m for m in full_members(pts, {})}
        kwargs = dict(wheel_center=np.zeros(3), spin_axis=np.array([1.,0.,0.]),
                      inner_radius_m=.12, half_width_m=.09)
        self.assertGreater(rim_barrel_gap(members['pushrod'], **kwargs), .003)
        self.assertLess(rim_barrel_gap(members['pushrod outer joint'], **kwargs), 0.)
        # A physical common endpoint is a designed connection, not self-clash.
        self.assertEqual(clashes(list(members.values())), [])

    def test_opposite_coilovers_and_shared_bar(self):
        def member(name, a, b, radius):
            return dict(name=name, a=np.array(a,dtype=float),b=np.array(b,dtype=float),r=radius)
        left=member('coilover',[-.1,0,.5],[.1,.1,.5],.0315)
        right=member('coilover',[.1,0,.5],[-.1,.1,.5],.0315)
        self.assertEqual(clashes([left]), [])
        hits=cross_corner_clashes({'RL':[left], 'RR':[right]})
        self.assertEqual(len(hits),1)
        self.assertAlmostEqual(hits[0]['gap_mm'],-63.)
        bar=member('ARB torsion bar',[-.3,0,0],[.3,0,0],.01)
        self.assertEqual(cross_corner_clashes({'RL':[bar], 'RR':[dict(bar)]}),[])

    def test_nearby_ends_are_not_cross_car_joints(self):
        a=dict(name='coilover', a=np.array([0.,0.,0.]), b=np.array([.2,0.,0.]),r=.03)
        b=dict(name='coilover', a=np.array([0.,.005,0.]), b=np.array([.2,.005,0.]),r=.03)
        hits=cross_corner_clashes({'RL':[a],'RR':[b]})
        self.assertEqual(hits[0]['gap_mm'],-55.)

    def test_larger_bar_and_front_lca_are_not_exempt(self):
        small=dict(name='ARB torsion bar',a=np.array([-.3,0.,0.]),b=np.array([.3,0.,0.]),r=.01)
        large=dict(small,r=.02)
        sphere=dict(name='test',a=np.array([0.,.02,0.]),b=np.array([0.,.02,0.]),r=.001)
        hits=cross_corner_clashes({'FL':[small],'FR':[large],'RL':[sphere]})
        self.assertTrue(any(h['a_corner']=='FR' and h['b']=='test' and h['gap_mm']<0 for h in hits))
        pushrod=dict(name='pushrod',a=np.array([-.1,0.,0.]),b=np.array([.1,0.,0.]),r=.008)
        lower=dict(name='lower arm front',a=np.array([0.,-.1,0.]),b=np.array([0.,.1,0.]),r=.008)
        self.assertEqual(len(clashes([pushrod,lower],connected=connected_for('FL'))),1)


if __name__ == '__main__':
    unittest.main()
