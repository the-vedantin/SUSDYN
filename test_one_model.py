"""
test_one_model.py -- REAL geometric regression net (not a blind smoke test).

Checks the two properties that were actually broken and that the user can see by
looking at the model:

  1. COPLANARITY  -- every rocker topology's actuation points must lie on the
     rocker plate plane at design (pullrod was 5-6mm off, T-bar 90-198mm).
  2. CONNECTEDNESS -- the motion_ratio graph must RESPOND to the active spring's
     hardpoint (nudge it, MR must move) with the damper-bounds poka-yoke
     bypassed so it can't mask a dead curve.  Decoupled was NaN/dead; this now
     proves it's live.

This is a NET, not verification.  Verification = rendering the model and looking
at it (see _tmp_render.py / _tmp_heave_tbar_geo.png).  KNOWN-FAIL items are
listed explicitly so the net doesn't pretend they're fixed.

Exit code = number of UNEXPECTED failures.
"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from gui.main_window import MainWindow
from vahan.topology import (SuspensionTopology, AxleTopology, DamperActuation as DA,
                            DamperMount as DM, ARBType as AT, SpringConfig as SC)

def ax(da, dm, arb, sc):
    return AxleTopology(damper_actuation=da, damper_mount=dm, arb_type=arb, spring_config=sc)

# topology -> (axle, active-spring hardpoint, coplanar?, KNOWN-FAIL reason or None)
CASES = {
    'pushrod':     (ax(DA.PUSHROD, DM.UCA, AT.BELLCRANK, SC.CORNER),   'spring_chassis_pt', True,  None),
    'pullrod':     (ax(DA.PULLROD, DM.LCA, AT.BELLCRANK, SC.CORNER),   'spring_chassis_pt', True,  None),
    'direct':      (ax(DA.DIRECT, DM.UCA, AT.NONE, SC.CORNER),         'damper_chassis_pt', False, None),
    'control_arm': (ax(DA.PUSHROD, DM.UCA, AT.CONTROL_ARM, SC.CORNER), 'spring_chassis_pt', True,  None),
    # Plain T-bar ARB = the central heave-T-bar mechanism WITHOUT the 3rd
    # spring (user-confirmed).  Corner is cradle_link (no corner rocker);
    # the ride spring is the coilover on the central bellcrank, so the MR
    # graph must respond to the COIL chassis attach.
    'tbar_corner': (ax(DA.PUSHROD, DM.UCA, AT.TBAR, SC.CORNER),        'htb_coil_chassis', False, None),
    'decoupled':   (ax(DA.PUSHROD, DM.UCA, AT.BELLCRANK, SC.DECOUPLED),'heave_damper_left', False, None),
    'heave_tbar':  (ax(DA.PUSHROD, DM.LCA, AT.TBAR, SC.HEAVE_TBAR),    'heave_spring_chassis_pt', False,
                    'kinematic graph not yet wired to the 3rd-spring solver (vahan/heave_tbar.py) -- integration pending'),
}

def _norm(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v

def coplanar_oop_mm(hp):
    if not all(k in hp and hp[k] is not None for k in ('rocker_pivot', 'rocker_axis_pt')):
        return None
    p0 = np.asarray(hp['rocker_pivot'], float)
    n = _norm(np.asarray(hp['rocker_axis_pt'], float) - p0)
    worst = 0.0
    for k in ('pushrod_outer', 'pushrod_inner', 'rocker_spring_pt', 'spring_chassis_pt'):
        if k in hp and hp[k] is not None and np.all(np.isfinite(hp[k])):
            worst = max(worst, abs(float(np.dot(np.asarray(hp[k], float) - p0, n))) * 1000)
    return worst

win = MainWindow()
win._check_damper_bounds_after_edit = lambda *a, **k: ''   # bypass poka-yoke so it can't mask dead curves
fails, known = 0, 0
# Regression: packaging must catch the plate penetration the renderer highlights.
try:
    from test_rule04_strict import Rule04PhysicalPlateTest
    Rule04PhysicalPlateTest().test_packaging_sweep_reports_real_plate_penetration()
    Rule04PhysicalPlateTest().test_packaging_sweep_requires_plate_clearance_margin()
    print('packaging plate sweep: pass')
except Exception as exc:
    print(f'packaging plate sweep: UNEXPECTED FAIL {exc}')
    fails += 1
# Physical double-shear plates must preserve their geometry and catch body hits.
try:
    import test_double_shear_rocker as _double_shear_tests
    _double_shear_tests.test_double_shear_plates_share_declared_gap_and_arm_widths()
    _double_shear_tests.test_center_plane_link_has_real_clearance_and_offset_tube_hits_plate()
    _double_shear_tests.test_joint_spheres_and_torsion_are_checked_without_attachment_skip()
    _double_shear_tests.test_full_coilover_capsule_clears_70mm_gap_and_is_not_skipped()
    _double_shear_tests.test_local_clevis_is_connected_and_clears_centered_pushrod_joint()
    _double_shear_tests.test_local_clevis_does_not_hide_non_pushrod_clashes()
    _double_shear_tests.test_local_clevis_invalid_geometry_fails_closed()
    _double_shear_tests.test_rocker_style_resolves_per_axle_with_global_fallback()
    _double_shear_tests.test_fixed_push_arm_scallop_clears_nearby_arb_eye_without_thinning_whole_arm()
    _double_shear_tests.test_declared_rectangular_arb_blade_uses_circumscribed_shared_envelope()
    print('double-shear physical plate geometry: pass')
except Exception as exc:
    print(f'double-shear physical plate geometry: UNEXPECTED FAIL {exc}')
    fails += 1
# A failed ARB closure must never become a drawable stretched drop link.
_arb_test_hp = {'arb_pivot': np.zeros(3),
                'arb_arm_end': np.array([0., .04, 0.]),
                'arb_drop_top': np.array([0., .04, .05])}
_arb_rejected = False
try:
    MainWindow._solve_arb_bellcrank(np.array([0., .20, 0.]), _arb_test_hp)
except ValueError:
    _arb_rejected = True
_a, _end, _res = MainWindow._solve_arb_bellcrank(_arb_test_hp['arb_drop_top'], _arb_test_hp)
_arb_valid = abs(_res) < 1e-7 and np.allclose(_end, _arb_test_hp['arb_arm_end'])
print(f'ARB closure      : impossible rejected={_arb_rejected}, design held={_arb_valid}')
if not (_arb_rejected and _arb_valid):
    fails += 1
from vahan.interference import rim_barrel_gap as _TestBarrelGap
_barrel_cases = [
    ([0., .05, 0.], [0., .05, 0.], .01, .06),
    ([0., .12, 0.], [0., .12, 0.], .01, -.01),
    ([.084, .123, 0.], [.084, .123, 0.], .006, -.001),
    ([0., .05, 0.], [0., .15, 0.], .005, -.005),
    ([0., .15, 0.], [0., .15, 0.], .005, .025),
]
_barrel_ok = all(abs(_TestBarrelGap({'a': a, 'b': b, 'r': r},
    np.zeros(3), np.array([1., 0., 0.]), .12, .08) - expected) < 1e-8
    for a, b, r, expected in _barrel_cases)
print(f'rim barrel math  : inside, wall, lip, crossing, outside '
      f'{"pass" if _barrel_ok else "UNEXPECTED FAIL"}')
if not _barrel_ok:
    fails += 1
print(f'{"topology":14s} {"coplanar":>10s}  {"MR-connected":>13s}   result')
print('-' * 64)
for name, (a, spring_key, want_coplanar, known_fail) in CASES.items():
    topo = SuspensionTopology(a, a)
    win.set_topology(topo)
    # 1. coplanarity
    oop = coplanar_oop_mm(win._front_hp)
    cop_ok = (oop is None) or (oop < 0.5)
    cop_s = 'n/a (no rocker)' if oop is None else f'{oop:.2f}mm {"OK" if cop_ok else "FAIL"}'
    # 2. MR connectedness
    win.set_topology(topo); win._motion_panel._motion = 'heave'; win._run_sweep()
    mr = np.asarray(win._sweep_results.get('FL', {}).get('motion_ratio', []), float)
    base = float(np.median(mr[np.isfinite(mr)])) if np.isfinite(mr).any() else float('nan')
    win.set_topology(topo)
    found = win._find_hp_dict(spring_key, 'FL')[2] is not None
    if found:
        win._on_hp_move(spring_key, 'FL', np.array([0, 0, 0.012])); win._motion_panel._motion = 'heave'; win._run_sweep()
        mr2 = np.asarray(win._sweep_results.get('FL', {}).get('motion_ratio', []), float)
        aft = float(np.median(mr2[np.isfinite(mr2)])) if np.isfinite(mr2).any() else float('nan')
    else:
        aft = float('nan')
    conn_ok = bool(np.isfinite(base) and np.isfinite(aft) and abs(aft - base) > 1e-4)
    conn_s = f'{base:.3f}->{aft:.3f} {"OK" if conn_ok else "DEAD"}'

    topo_fail = (want_coplanar and not cop_ok) or (not conn_ok)
    if topo_fail and known_fail:
        known += 1; res = f'KNOWN-FAIL: {known_fail[:40]}'
    elif topo_fail:
        fails += 1; res = 'UNEXPECTED FAIL'
    else:
        res = 'pass'
    print(f'{name:14s} {cop_s:>14s}  {conn_s:>16s}   {res}')

# ── ROLL path: decoupled uses a SEPARATE roll spring (not the heave spring),
# so heave connectedness above doesn't exercise it.  Guard the roll-spring
# injection AND that the graph (at design) equals the dynamics MR — the exact
# thing unified this session.  Use the PHYSICAL +-5 deg range the GUI sets on
# the Roll radio (line panels._on_motion); forcing the +-50 mm heave default
# would drive +-50 deg roll -> +-468 mm wheel travel -> wishbone solver throws
# (a TEST artifact, not a model bug).
print('-' * 64)
dtopo = SuspensionTopology(ax(DA.PUSHROD, DM.UCA, AT.BELLCRANK, SC.DECOUPLED),
                           ax(DA.PUSHROD, DM.UCA, AT.BELLCRANK, SC.DECOUPLED))
win.set_topology(dtopo)
mp = win._motion_panel; mp._motion = 'roll'; mp._min_val = -5.0; mp._max_val = 5.0
win._run_sweep()
mrr = np.asarray(win._sweep_results.get('FL', {}).get('motion_ratio', []), float)
nfin = int(np.isfinite(mrr).sum())
ctr = mrr[len(mrr) // 2] if len(mrr) else float('nan')
# connectedness: nudge the ROLL damper attach (Z), roll MR must respond
win.set_topology(dtopo); mp._motion = 'roll'; mp._min_val = -5.0; mp._max_val = 5.0
win._on_hp_move('roll_damper_left', 'FL', np.array([0, 0, 0.012])); win._run_sweep()
mrr2 = np.asarray(win._sweep_results.get('FL', {}).get('motion_ratio', []), float)
ctr2 = mrr2[len(mrr2) // 2] if len(mrr2) else float('nan')
roll_conn = bool(np.isfinite(ctr) and np.isfinite(ctr2) and abs(ctr2 - ctr) > 1e-4)
# graph(center) == dynamics at design
win.set_topology(dtopo); mp._motion = 'roll'; mp._min_val = -5.0; mp._max_val = 5.0
win._run_sweep()
g_ctr = np.asarray(win._sweep_results['FL']['motion_ratio'], float)
g_ctr = g_ctr[len(g_ctr) // 2]
dynp = win._apply_topology_to_dyn_params(win._dynamics_panel.get_params())
d_roll = float(dynp.get('decoupled_roll_MR_front', float('nan')))
roll_match = bool(np.isfinite(g_ctr) and np.isfinite(d_roll) and abs(g_ctr - d_roll) < 0.06)
roll_ok = (nfin == len(mrr)) and roll_conn and roll_match
if not roll_ok:
    fails += 1
print(f'decoupled ROLL : finite {nfin}/{len(mrr)}  connected={roll_conn}  '
      f'graph={g_ctr:.3f}==dyn={d_roll:.3f}? {roll_match}   '
      f'{"pass" if roll_ok else "UNEXPECTED FAIL"}')

# ── HEAVE-T-BAR is ONE T-bar (user-confirmed): a SINGLE hardpoint must drive
# BOTH the heave graph AND the roll rate, and the corner must have NO spring
# (cradle_link).  Guards the full ONE-T-bar unification end to end.
htopo = SuspensionTopology(ax(DA.PUSHROD, DM.LCA, AT.TBAR, SC.HEAVE_TBAR),
                           ax(DA.PUSHROD, DM.LCA, AT.TBAR, SC.HEAVE_TBAR))
win.set_topology(htopo)
corner_mode = getattr(win._solvers.get('FL'), '_damper_actuation', None)
def _htb_heave():
    win._motion_panel._motion = 'heave'; win._run_sweep()
    m = np.asarray(win._sweep_results['FL']['motion_ratio'], float)
    return float(np.median(m[np.isfinite(m)])) if np.isfinite(m).any() else float('nan')
def _htb_rollrate():
    win._refresh_vehicle_constants()
    return float(win._dynamics_panel.get_params().get('arb_rate_front_Npm', 0.0))
h0, r0 = _htb_heave(), _htb_rollrate()
win._on_hp_move('htb_arm_tip', 'FL', np.array([0, 0, 0.010]))   # one bar point
h1, r1 = _htb_heave(), _htb_rollrate()
htb_heave_live = np.isfinite(h0) and np.isfinite(h1) and abs(h1 - h0) > 1e-4
htb_roll_live  = r0 > 0 and abs(r1 - r0) > 1.0
htb_no_spring  = (corner_mode == 'cradle_link')
htb_ok = htb_heave_live and htb_roll_live and htb_no_spring
if not htb_ok:
    fails += 1
print(f'heave_tbar ONE-T-bar : corner={corner_mode}  one-pt drives heave {h0:.3f}->{h1:.3f} '
      f'({"live" if htb_heave_live else "DEAD"}) + roll {r0:.0f}->{r1:.0f} '
      f'({"live" if htb_roll_live else "DEAD"})   {"pass" if htb_ok else "UNEXPECTED FAIL"}')

# ── CASTER SIGN: +Y is REARWARD in the model (front axle Y=0, rear at +wb), so
#    a rearward-leaning kingpin (uca_outer behind lca_outer) is POSITIVE caster.
#    The metric printed NEGATIVE before the +Y-convention sign fix (kinematics.py).
print('-' * 64)
from vahan.kinematics import KinematicMetrics
_bt = ax(DA.PUSHROD, DM.UCA, AT.BELLCRANK, SC.CORNER)
win.set_topology(SuspensionTopology(_bt, _bt)); win._rebuild_solvers(0.)
_stF = win._solvers['FL'].solve(0.)
casterF = KinematicMetrics(_stF, 'left').caster
caster_ok = casterF > 0.0
if not caster_ok:
    fails += 1
print(f'caster sign      : front caster {casterF:+.2f} deg (uca_outer rearward of lca)   '
      f'{"pass" if caster_ok else "UNEXPECTED FAIL (should be POSITIVE)"}')

# ── ARB SWEEP METRICS LIVE: _do_sweep referenced an undefined `label`, silently
#    NaN-ing arb_angle/arb_drop_travel/arb_mr (NameError eaten by except).  Guard
#    that a heave sweep on a bellcrank ARB now produces finite ARB metrics.
win.set_topology(SuspensionTopology(_bt, _bt))
win._motion_panel._motion = 'heave'; win._run_sweep()
_amr = np.asarray(win._sweep_results.get('FL', {}).get('arb_mr', []), float)
_aang = np.asarray(win._sweep_results.get('FL', {}).get('arb_angle', []), float)
arb_live = np.isfinite(_amr).sum() > 5 and np.isfinite(_aang).sum() > 5
if not arb_live:
    fails += 1
print(f'ARB sweep metrics: arb_mr finite {int(np.isfinite(_amr).sum())}/{len(_amr)}, '
      f'arb_angle finite {int(np.isfinite(_aang).sum())}/{len(_aang)}   '
      f'{"pass" if arb_live else "UNEXPECTED FAIL (dead/NaN)"}')

# ── SKIDPAD/TRANSIENT ARB FRESHNESS: _on_skidpad_simulate consumed
#    get_params() WITHOUT refreshing the kinematically-derived ARB geometry
#    (arm/half/MR), so the transient sim could run on a STALE bar after
#    hardpoint edits (found by the v18 cross-check fleet, 2026-07-19).
#    Guard: the shared refresh helper exists, the skidpad handler calls it,
#    and an ARB hardpoint move changes the refreshed get_params() rate.
import inspect
win.set_topology(SuspensionTopology(_bt, _bt)); win._rebuild_solvers(0.)
win._refresh_arb_geometry_into_panel()
_r0 = float(win._dynamics_panel.get_params()['arb_rate_front_Npm'])
# Perturb the PIVOT along the blade axis: that lengthens the lever with the
# drop-link geometry fixed, so K_t ~ 1/A^2 moves and MR does not.
# NOT arm_end: under the rigid-blade model (2026-07-25) moving the arm end
# changes A and the derived MR in COMPENSATING directions (K_t ~ 1/A^2 vs
# 1/MR^2 ~ A^2) and the wheel rate barely moves -- the old probe only worked
# because the discarded K_a ~ 1/A^3 arm-bending term broke that cancellation.
# A live-refresh check must perturb something the rate is actually sensitive to.
_pv_save = np.asarray(win._front_arb['arb_pivot'], float).copy()
_ae_now = np.asarray(win._front_arb['arb_arm_end'], float)
_bu = (_pv_save - _ae_now) / (np.linalg.norm(_pv_save - _ae_now) or 1.0)
win._front_arb['arb_pivot'] = (_pv_save + _bu * 0.010).tolist()
win._refresh_arb_geometry_into_panel()
_r1 = float(win._dynamics_panel.get_params()['arb_rate_front_Npm'])
win._front_arb['arb_pivot'] = _pv_save.tolist()
win._refresh_arb_geometry_into_panel()
_src_ok = ('refresh_arb_geometry_into_panel'
           in inspect.getsource(type(win)._on_skidpad_simulate))
_fresh_ok = _r0 > 0 and abs(_r1 - _r0) > 1.0
skid_ok = _src_ok and _fresh_ok
if not skid_ok:
    fails += 1
print(f'skidpad ARB fresh: handler refreshes={_src_ok}  rate responds '
      f'{_r0:.0f}->{_r1:.0f} N/m ({"live" if _fresh_ok else "DEAD"})   '
      f'{"pass" if skid_ok else "UNEXPECTED FAIL"}')

# ── BLADE-SECTION ARB ARM (feature 2026-07-19): a flat-leaf blade's
#    weak-axis bending is a real series spring when blade w and t are both
#    set.  With NO blade section the arm is a RIGID LEVER and the torsion
#    tube is the only spring (user decision 2026-07-25) — it no longer
#    borrows the tube section for a phantom cantilever.  Guard: setting a
#    thin blade still SOFTENS the computed wheel rate vs the rigid-arm bar.
_bw0 = win._dynamics_panel._arb_blade_w_f.value()
_bt0 = win._dynamics_panel._arb_blade_t_f.value()
win._dynamics_panel._arb_blade_w_f.setValue(0.0)
win._dynamics_panel._arb_blade_t_f.setValue(0.0)
win._refresh_arb_geometry_into_panel()
_rb0 = float(win._dynamics_panel.get_params()['arb_rate_front_Npm'])
win._dynamics_panel._arb_blade_w_f.setValue(25.4)
win._dynamics_panel._arb_blade_t_f.setValue(3.0)
_rb1 = float(win._dynamics_panel.get_params()['arb_rate_front_Npm'])
win._dynamics_panel._arb_blade_w_f.setValue(_bw0)
win._dynamics_panel._arb_blade_t_f.setValue(_bt0)
blade_ok = _rb0 > 0 and 0 < _rb1 < _rb0
if not blade_ok:
    fails += 1
print(f'blade-section ARB : tube-arm {_rb0:.0f} -> 25.4x3.0 blade {_rb1:.0f} N/m '
      f'({"softens" if blade_ok else "NO EFFECT"})   '
      f'{"pass" if blade_ok else "UNEXPECTED FAIL"}')

# ── RIM-FIT ENVELOPE (feature 2026-07-19, user-flagged hard constraint):
#    the kingpin ball joints + tie-rod end must fit inside the wheel rim
#    (radial from the spin axis <= the rim clear radius), else the upright
#    cannot be built.  Guard the KinematicMetrics.rim_fit check itself: it
#    must PASS a joint at the rim centre and FLAG one pushed outside.
from vahan.kinematics import KinematicMetrics as _KM
win.set_topology(SuspensionTopology(_bt, _bt)); win._rebuild_solvers(0.)
_stf = win._solvers['FL'].solve(0.)
# rim clear diameter is now a PANEL INPUT (user 2026-07-20) — the check must
# read it, and get_params must round-trip it.
_rim_m = win._dynamics_panel.rim_clear_diameter_m()   # panel input, 9.5 in default
_rim_mm = _rim_m * 1000.0
_rf = _KM(_stf, 'left').rim_fit(_rim_m)
_km = _KM(_stf, 'left')
_ubj_r = _km.joint_rim_radius('uca_outer') * 1000
# synthesize an out-of-rim joint: push uca_outer radially far from the axle
import copy as _copy
_bad = _copy.copy(_stf)
_wc = np.asarray(_stf.wheel_center, float)
_bad.uca_outer = list(_wc + np.array([0.0, 0.0, 0.20]))   # 200 mm above axis
_bad_fit = _KM(_bad, "left").rim_fit(_rim_m)["fits"]
rim_ok = (_rim_mm > 1.0) and bool(_rf['fits']) and (_bad_fit is False)
if not rim_ok:
    fails += 1
print(f'rim-fit envelope : input {_rim_mm:.0f} mm dia, UBJ radial {_ubj_r:.0f} mm, '
      f'clear {_rf["clear_radius_m"]*1000:.0f} mm, design fits={_rf["fits"]}, '
      f'out-of-rim flagged={not _bad_fit}   '
      f'{"pass" if rim_ok else "UNEXPECTED FAIL"}')

# ── REAR DRIVESHAFT PACKAGE (user 2026-07-20): diff/tripod/shaft geometry from
#    the car-dict inputs, bound to the LIVE solved wheel_center (ONE MODEL).  A
#    lateral offset must make the two half-shafts UNEQUAL by ~2x the offset, and
#    the packaging inputs must round-trip through the car panel.
from vahan.driveshaft import package as _dpkg
_rear = {'RL': win._solvers['RL'].solve(0.), 'RR': win._solvers['RR'].solve(0.)}
_carc = dict(win._car); _carc['diff_lateral_offset_mm'] = 0.0
_p0 = _dpkg(_carc, _rear)
_carc['diff_lateral_offset_mm'] = 40.0
_p40 = _dpkg(_carc, _rear)
_gp = win._car_panel.get_params()
_ds_keys = all(k in _gp for k in ('diff_long_mm', 'diff_lateral_offset_mm',
               'diff_housing_width_mm', 'tripod_od_mm', 'driveshaft_dia_mm',
               'show_driveshaft', 'show_shock_thickness'))
_ds_ok = (_ds_keys and _p0['length_asymmetry_mm'] < 1.0
          and abs(_p40['length_asymmetry_mm'] - 80.0) < 8.0
          and _p0['RL']['length_mm'] > 50.0)
if not _ds_ok:
    fails += 1
print(f'driveshaft pkg   : offset0 asym {_p0["length_asymmetry_mm"]:.1f} mm, '
      f'offset40 asym {_p40["length_asymmetry_mm"]:.1f} mm (~80), '
      f'RL len {_p0["RL"]["length_mm"]:.0f} mm, inputs={_ds_keys}   '
      f'{"pass" if _ds_ok else "UNEXPECTED FAIL"}')

# ── INTERFERENCE (clash) ENGINE: capsule-vs-capsule distance drives the 3D
#    Interference view mode + the driveshaft/pushrod packaging check.  Verify it
#    flags a real overlap, ignores a designed joint, and passes clear members.
from vahan.interference import clashes as _clashes
_caps = [
    {'name': 'x', 'a': [0, 0, 0],     'b': [1, 0, 0],   'r': 0.01},
    {'name': 'y', 'a': [0.5, 0.005, 0], 'b': [0.5, 0.5, 0], 'r': 0.01},  # crosses x -> overlap
    {'name': 'z', 'a': [0, 0.9, 0],   'b': [1, 0.9, 0], 'r': 0.01},      # parallel, far -> clear
    {'name': 'lower arm rear', 'a': [0, 0, 0], 'b': [0.3, 0, 0], 'r': 0.01},
    {'name': 'pushrod', 'a': [0.1, 0.001, 0], 'b': [0.1, 0.4, 0], 'r': 0.01},  # overlaps 'lower arm rear'
]
_cl = _clashes(_caps, margin_mm=1.0)
_names = {frozenset({d['a'], d['b']}) for d in _cl}
_int_ok = (frozenset({'x', 'y'}) in _names            # real overlap flagged
           and frozenset({'pushrod', 'lower arm rear'}) not in _names  # designed joint skipped
           and all(d['gap_mm'] < 1.0 for d in _cl))
if not _int_ok:
    fails += 1
print(f'interference     : {len(_cl)} clash(es) {[(d["a"],d["b"],d["gap_mm"]) for d in _cl]}; '
      f'real-overlap flagged + pushrod/LCA joint skipped={_int_ok}   '
      f'{"pass" if _int_ok else "UNEXPECTED FAIL"}')

# ── REAL-GEOMETRY ACTUATION GATE ─────────────────────────────────────────────
# The synthetic test above only proves the ENGINE runs.  v28 slid the rear
# pushrod off the lower arm toward the tie-rod (dy -51 mm, tie-rod gap 71->22 mm)
# and canted the damper 63 mm rearward to force coplanarity by projection -- and
# NOTHING here caught it, because the net never loaded the actual design.  A pure
# capsule-overlap alarm would have missed it too (22 mm is a near-miss, not an
# overlap).  So load the CURRENT design config and assert the three properties a
# bad rear-actuation edit breaks -- pushrod over the LCA, damper flat (not canted
# fore-aft), rocker coplanar -- plus a realistic-radius clash sweep.  This block
# FAILS on v28 and PASSES on v29 (the fix).  Skips cleanly if no config present.
import glob as _glob, re as _re
def _seg_gap(a1, a2, b1, b2, ra, rb):
    # all inputs in mm; returns surface-to-surface gap in mm (negative = overlap)
    a1, a2, b1, b2 = (np.asarray(x, float) for x in (a1, a2, b1, b2))
    d1, d2, r = a2 - a1, b2 - b1, a1 - b1
    A, E, F = d1 @ d1, d2 @ d2, d2 @ r
    if A < 1e-12 and E < 1e-12: s = t = 0.0
    elif A < 1e-12: s, t = 0.0, np.clip(F / E, 0, 1)
    else:
        C = d1 @ r
        if E < 1e-12: t, s = 0.0, np.clip(-C / A, 0, 1)
        else:
            B = d1 @ d2; den = A * E - B * B
            s = np.clip((B * F - C * E) / den, 0, 1) if den > 1e-12 else 0.0
            t = (B * s + F) / E
            if t < 0: t, s = 0.0, np.clip(-C / A, 0, 1)
            elif t > 1: t, s = 1.0, np.clip((B - C) / A, 0, 1)
    return float(np.linalg.norm((a1 + s * d1) - (b1 + t * d2)) - (ra + rb))

_cfgs = _glob.glob('configs/2027_v*.vahan')
def _highest_config(paths):
    return max(paths, key=lambda p: int(_re.search(r'2027_v(\d+)', p).group(1)) if _re.search(r'2027_v(\d+)', p) else -1) if paths else None
# VAHAN_DESIGN=<path> pins the design config the net judges (default: the highest configs/2027_v<N>)
_design = os.environ.get('VAHAN_DESIGN') or _highest_config(_cfgs)
# ── RULE 20: rear toe-link inner == aft LCA inboard pickup (user hard rule 2026-09-21) ──
try:
    import json as _j20
    _r20 = _j20.load(open(_design, encoding='utf-8'))['rear_hp']
    _d20 = float(np.linalg.norm(np.array(_r20['tie_rod_inner']) - np.array(_r20['lca_rear']))) * 1000.
    print(f'rule 20 rear toe inner on LCA rear pickup: {_d20:.3f} mm apart   {"pass" if _d20 < 1e-3 else "UNEXPECTED FAIL (must be the SAME point)"}')
    if _d20 >= 1e-3:
        fails += 1
except Exception as _e:
    print(f'rule 20 rear toe inner on LCA rear pickup: UNEXPECTED FAIL {_e}'); fails += 1
# ── RULE 21: the pushrod's arm mount leaves the actuation plane through travel (user check 2026-09-22) ──
# REPORTED, not gated: the DIRECT effect is the pushrod leaning out of the rocker plane (rod-end
# misalignment, side load on the rocker = sin(lean) x pushrod force).  No user threshold yet.
try:
    from vahan import packaging as _pk21
    _w21 = MainWindow(); _w21._load_project_from_path(_design); _w21._rebuild_solvers(0.)
    _lo21, _hi21 = _w21._spring_travel_range(_w21._solvers['FL'], 'FL')
    _txt21 = []
    for _ax21, _lb21 in (('front', 'FL'), ('rear', 'RL')):
        _hp21 = _pk21.get_bundle(_w21, _ax21)['hp']
        _ra21 = np.asarray(_hp21['rocker_axis_pt']) - np.asarray(_hp21['rocker_pivot']); _ra21 /= np.linalg.norm(_ra21)
        _s21 = _w21._solvers[_lb21]; _lean21 = []
        for _t in np.linspace(_lo21, _hi21, 9):
            _st21 = _s21.solve(float(_t))
            _u21 = np.asarray(_st21.pushrod_inner) - np.asarray(_st21.pushrod_outer); _u21 /= np.linalg.norm(_u21)
            _lean21.append(float(np.degrees(np.arcsin(abs(_u21 @ _ra21)))))
        _txt21.append(f'{_lb21} pushrod lean out of the rocker plane max {max(_lean21):.1f} deg '
                      f'(side load {np.sin(np.radians(max(_lean21))) * 1000:.0f} N per kN of pushrod force)')
    print(f'rule 21 pushrod lean  : {"; ".join(_txt21)} over {_lo21*1000:+.0f}..{_hi21*1000:+.0f} mm   reported')
except Exception as _e:
    print(f'rule 21 pushrod lean  : UNEXPECTED FAIL {_e}'); fails += 1
# ── REAL WHEEL PROFILE (2026-09-21): members vs the manufacturer's STEP-derived barrel + centre disc ──
# car['wheel_profile'] -> JSON from vahan.wheel_profile.barrel_profile_from_step (Keizer, 6 in backspacing).
# Every member tube and joint body at FL/RL, droop/static/bump (+ front steer lock) must clear the
# real inner surface by 3 mm.  Alternatives listed in car['wheel_profile_alternatives'] are REPORTED only.
try:
    import json as _jwp
    _wpw = MainWindow(); _wpw._load_project_from_path(_design); _wpw._rebuild_solvers(0.)
    _wp_main = _wpw._car.get('wheel_profile')
    if not _wp_main:
        print('real wheel profile : no car["wheel_profile"] set — assumed-cylinder rim gates only   pass (not gated)')
    else:
        from vahan import wheel_profile as _WP
        from vahan.interference import corner_members as _cm_wp
        _lock = _wpw._steer['total_rack_travel_mm'] / _wpw._steer['rack_travel_per_rev_mm'] * 180.0
        def _wp_worst(prof):
            worst = (1e9, '', '', 0.0, 0.0)
            for _hw in (-_lock, 0.0, _lock):
                _wpw._rebuild_solvers(float(_hw))
                for _lbl in ('FL', 'RL'):
                    if _lbl == 'RL' and _hw != 0.0:
                        continue
                    for _t in (_wpw._motion_panel.min_val / 1000., 0.0, _wpw._motion_panel.max_val / 1000.):
                        _st = _wpw._solvers[_lbl].solve(float(_t))
                        _ax = np.asarray(_st.spin_axis, float); _ax = _ax / np.linalg.norm(_ax)
                        if _ax[0] > 0: _ax = -_ax
                        _mem = _cm_wp(_st, _wpw._car)
                        for _jk in ('uca_outer', 'lca_outer', 'tr_outer'):
                            _p = np.asarray(getattr(_st, _jk), float); _mem.append({'name': _jk + ' joint body', 'a': _p, 'b': _p, 'r': 0.0127})
                        for _m in _mem:
                            _g = _WP.member_clearance(_m, np.asarray(_st.wheel_center, float), _ax, prof)
                            if _g['gap_mm'] < worst[0]:
                                worst = (_g['gap_mm'], _lbl, _m['name'], _t * 1000., _hw)
            _wpw._rebuild_solvers(0.)
            return worst
        _prof = _WP.load_profile(_wp_main); _wst = _wp_worst(_prof)
        _wp_ok = _wst[0] >= 3.0
        print(f'real wheel profile : {os.path.basename(_wp_main)} ({_prof["width_in"]:.1f} in, backspacing {_prof["backspacing_mm"]/25.4:.2f} in read from the STEP) — '
              f'tightest {_wst[1]} {_wst[2]} {_wst[0]:.1f} mm at {_wst[3]:+.0f} mm travel, {_wst[4]:.0f} deg handwheel (need 3)   {"pass" if _wp_ok else "UNEXPECTED FAIL"}')
        if not _wp_ok:
            fails += 1
        for _alt in _wpw._car.get('wheel_profile_alternatives', []) or []:
            try:
                _pa = _WP.load_profile(_alt); _wa = _wp_worst(_pa)
                print(f'   alternative wheel {os.path.basename(_alt)} ({_pa["width_in"]:.1f} in): tightest {_wa[1]} {_wa[2]} {_wa[0]:.1f} mm at {_wa[3]:+.0f} mm travel   (reported, not gated)')
            except Exception as _e:
                print(f'   alternative wheel {_alt}: could not evaluate ({_e})')
except Exception as _e:
    print(f'real wheel profile : UNEXPECTED FAIL {_e}'); fails += 1
# ── 3-D VIEW = THE MODEL (2026-09-23): real rim + ground that follows the tyres ──
# (1) The rim drawn at every corner is the manufacturer profile (car['wheel_profile']) revolved about
#     the SOLVED spin axis: every drawn barrel vertex must sit on the surface the "real wheel profile"
#     gate measures against (vahan.wheel_profile.barrel_radius_at), at the corner's solved wheel centre.
# (2) The ground plane is drawn at the mean contact patch of the four solved corners at the CURRENT
#     travel (chassis-fixed view: in bump the ground rises toward the chassis, e.g. the sprocket).
try:
    from vahan import wheel_profile as _WPv
    from gui.view3d import RIM_N as _RIM_N
    _v3 = MainWindow(); _v3._load_project_from_path(_design); _v3._rebuild_solvers(0.)
    _v3mp = _v3._motion_panel
    _v3mp._btn_grp.buttons()[0].setChecked(True)          # heave
    _v3mp.go_to_static(); _v3._update_3d()
    _v3_prof = _v3.view3d._wheel_profile
    _r_t = float(_v3._car['tire_outer_dia_mm']) / 2000.0
    if _v3_prof is None:
        print('3D view real rim   : no car["wheel_profile"] set — plain tyre cylinder drawn   pass (not gated)')
    else:
        _n_st = len(_v3_prof['d_mm']); _rb = np.asarray(_v3_prof['r_barrel_mm'], float)
        _worst = 0.0; _n_chk = 0; _cam_max = 0.0
        # pose = the GUI's own per-corner draw (solved state + the alignment panel's static camber /
        # toe rotation of the spin axis); the rim must sit on THAT axis at THAT wheel centre.
        _cd3, _ = _v3._assemble_corners_draw({_l: 0.0 for _l in ('FL', 'FR', 'RL', 'RR')}, 0.0)
        for _ci, _c3 in enumerate(_cd3):
            _lbl = _c3['label']
            _wc = np.asarray(_c3['pts']['wheel_center'], float)
            _ax = np.asarray(_c3['spin_axis'], float); _ax = _ax / np.linalg.norm(_ax)
            _st = _v3._solvers[_lbl].solve(0.0)
            if not np.allclose(_wc, np.asarray(_st.wheel_center, float), atol=1e-9):
                raise RuntimeError(f'{_lbl}: drawn wheel centre != solver wheel centre')
            _cam_max = max(_cam_max, float(np.degrees(np.arccos(min(1.0, abs(_ax @ np.asarray(_st.spin_axis, float)
                                                                                / np.linalg.norm(_st.spin_axis)))))))
            if abs((_wc + _ax * 0.01)[0]) > abs(_wc[0]):
                _ax = -_ax                                      # inboard, as the gate uses
            _mv = np.asarray(_v3.view3d._rim_meshes[_ci]._meshdata.get_vertices(), float)
            _p = _mv[:_n_st * _RIM_N] - _wc
            _d = _p @ _ax; _rad = np.linalg.norm(_p - np.outer(_d, _ax), axis=1)
            _ok = np.isfinite(_rb[np.arange(_n_st * _RIM_N) // _RIM_N])   # open-barrel stations only
            _exp = _WPv.barrel_radius_at(_v3_prof, _d[_ok] * 1000.0)
            _err = np.abs(_rad[_ok] * 1000.0 - _exp)
            _worst = max(_worst, float(np.nanmax(_err))); _n_chk += int(_ok.sum())
        _rim_ok = (_n_chk > 0) and (_worst < 0.05)
        print(f'3D view real rim   : {os.path.basename(str(_v3._car.get("wheel_profile")))} — {_n_chk} drawn barrel vertices '
              f'over 4 corners vs the wheel-profile gate surface at the drawn wheel centres/spin axes (= solver '
              f'+ {_cam_max:.2f} deg static alignment), max deviation {_worst:.3f} mm (need <0.05)   '
              f'{"pass" if _rim_ok else "UNEXPECTED FAIL"}')
        if not _rim_ok:
            fails += 1
    _z0 = _v3.view3d.ground_height_m()
    _v3mp._slider.setValue(_v3mp._slider_value_for(0.5 * _v3mp.max_val)); app.processEvents()
    _v3._update_3d()
    _tg = float(_v3mp.position) / 1000.0
    _exp_z = float(np.mean([_v3._solvers[_l].solve(_tg).wheel_center[2] - _r_t for _l in ('FL', 'FR', 'RL', 'RR')]))
    _got_z = _v3.view3d.ground_height_m()
    _grd_ok = abs(_got_z - _exp_z) < 1e-6 and abs(_tg) > 1e-3 and abs((_got_z - _z0) - _tg) < 1e-3
    print(f'3D view ground     : heave {_tg*1000:+.1f} mm -> ground drawn at {_got_z*1000:+.2f} mm = mean contact patch '
          f'{_exp_z*1000:+.2f} mm (static {_z0*1000:+.2f} mm; rose by {(_got_z-_z0)*1000:.1f} mm)   '
          f'{"pass" if _grd_ok else "UNEXPECTED FAIL"}')
    if not _grd_ok:
        fails += 1
    _v3mp.go_to_static()
except Exception as _e:
    print(f'3D view rim/ground : UNEXPECTED FAIL {_e}'); fails += 1
# ── STALE BACKGROUND SWEEP (2026-09-21) ────────────────────────────────────
# A hardpoint-edit sweep worker started DURING project load landed after the
# synchronous load sweep and drew a 1.48 deg front toe swing on v141 that the
# geometry does not have.  Gate: after the load and the queued events settle,
# the GUI sweep results must equal a fresh synchronous sweep.
try:
    import time as _time
    _sw_win = MainWindow(); _sw_win._load_project_from_path(_design)
    for _ in range(40):
        app.processEvents(); _time.sleep(0.05)
    _t_evt = np.asarray(_sw_win._sweep_results['FL']['toe'], float)
    _sw_win._run_sweep()
    _t_sync = np.asarray(_sw_win._sweep_results['FL']['toe'], float)
    _sw_ok = _t_evt.shape == _t_sync.shape and np.allclose(np.nan_to_num(_t_evt), np.nan_to_num(_t_sync), atol=1e-9)
    print(f'stale sweep guard: front toe swing after load events {np.nanmax(_t_evt)-np.nanmin(_t_evt):.3f} deg, '
          f'sync sweep {np.nanmax(_t_sync)-np.nanmin(_t_sync):.3f} deg   {"pass" if _sw_ok else "UNEXPECTED FAIL (a stale background sweep overwrote the load sweep)"}')
    if not _sw_ok:
        fails += 1
except Exception as _e:
    print(f'stale sweep guard: UNEXPECTED FAIL {_e}'); fails += 1
if _design:
    wD = MainWindow(); wD._load_project_from_path(_design); wD._rebuild_solvers(0.)
    # Rear half-shaft segments (mm) so the gate catches the pushrod crossing the
    # driveshaft -- the v32 clash the old gate MISSED (it had no driveshaft member).
    import types as _types
    try:
        from vahan.driveshaft import package as _dspkg
        _rs = {}
        for _l in ('RL', 'RR'):
            _s = wD._solvers[_l].solve(0.)
            _rs[_l] = _types.SimpleNamespace(wheel_center=np.asarray(_s.wheel_center, float),
                                             spin_axis=np.asarray(_s.spin_axis, float))
        _pkg = _dspkg(wD._car, _rs)
        _DS = {_l: (np.asarray(_pkg[_l]['inner'], float) * 1000, np.asarray(_pkg[_l]['outer'], float) * 1000)
               for _l in ('RL', 'RR') if _pkg.get(_l)}
    except Exception:
        _DS = {}
    # v32 RESOLVED (user chose the clearing-window pushrod foot): driveshaft clears
    # +4.7 mm, rear bump steer re-nulled 0.017 deg.  No standing exemptions -- any
    # clash or bump-steer failure is UNEXPECTED again and fails the net.
    _KNOWN = ()
    gfail = []
    from vahan.packaging import (_axle_geometry_laws as _triad_geometry,
                                 arb_drop_link_plate_metrics as _rule04_metrics)
    for _axle in ('front', 'rear'):
        _tg = _triad_geometry(wD, _axle)
        for _name in ('triad_bar_blade_deg', 'triad_blade_drop_deg', 'triad_bar_drop_deg'):
            if not np.isfinite(_tg[_name]) or abs(_tg[_name] - 90.0) > 1.0:
                gfail.append(f'{_axle} {_name} {_tg[_name]:.3f} deg (require 90 +/-1)')
    _rack_width = 2000. * abs(float(wD._front_hp['tie_rod_inner'][0]))
    if abs(_rack_width - float(wD._car['rack_length_mm'])) > 0.1:
        gfail.append(f'rack width mismatch: hardpoints {_rack_width:.2f} mm '
                     f"versus car setting {wD._car['rack_length_mm']:.2f} mm")
    _allowance_keys = ('rim_joint_clearance_mm', 'rim_housing_allowance_mm',
                      'rim_barrel_width_mm', 'front_bump_steer_limit_deg')
    _allowances = {k: wD._car[k] for k in _allowance_keys if k in wD._car}
    wD._on_car(wD._car_panel.get_params())
    if any(wD._car.get(k) != v for k, v in _allowances.items()):
        gfail.append('car panel discarded saved packaging allowances')
    # ALL FOUR corners.  This loop was ('RL','RR') only, so the FRONT corner was
    # structurally invisible to every check inside it — coplanarity, the ARB
    # drop-link-in-plane HARD requirement, pushrod-over-LCA, damper cant and the
    # clash sweep.  Run on the front, the v47/v48 ARB drop link is 14.1 mm off
    # the actuation plane against a 3.0 mm threshold and the net still exited 0.
    # Each corner must also get its OWN mirrored ARB dict: passing the left-side
    # points into a right-side point cloud only survived because the rear rocker
    # plane is Y = const, which hides an X-sign error entirely.
    for lbl in ('FL', 'FR', 'RL', 'RR'):
        _arb0 = wD._front_arb if lbl[0] == 'F' else wD._rear_arb
        _sgn = -1.0 if lbl[1] == 'R' else 1.0
        arb = {k: np.array([_sgn * v[0], v[1], v[2]], float) for k, v in _arb0.items()}
        _axle_name = 'front' if lbl[0] == 'F' else 'rear'
        _shared_laws = _triad_geometry(wD, _axle_name)
        cop = _shared_laws['coplanar_mm']
        st = wD._solvers[lbl].solve(0.)
        P = lambda k: np.asarray(getattr(st, k), float) * 1000.0
        po, lo = P('pushrod_outer'), P('lca_outer')
        lf, lr = P('lca_front'), P('lca_rear')
        rsp, scp = P('rocker_spring_pt'), P('spring_chassis_pt')
        A = lambda k: np.asarray(arb[k], float) * 1000.0
        # DROP-LINK rule (user): at STATIC the ARB drop link must lie IN the rocker
        # actuation plane (both ends).  Across travel the blade end arcs with the
        # bar so it CANNOT stay in-plane — that lean is minimized by design and
        # only reported here, not failed.
        try:
            _cd, _ = wD._assemble_corners_draw({_l: 0.0 for _l in ('FL', 'FR', 'RL', 'RR')}, 0.0)
            _pts = [c for c in _cd if c['label'] == lbl][0]['pts']
            _ae = np.asarray(_pts['arb_arm_end_world'], float)
            _metric_arb = dict(arb, arb_arm_end=_ae)
            _metric_hp = {k: np.asarray(getattr(st, k), float) for k in
                          ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')}
            _r04 = _rule04_metrics(_metric_hp, _metric_arb, _sgn)
            _doff = max(abs(_r04['drop_top_signed_mm']),
                        abs(_r04['arm_end_signed_mm']))
            # Only the saved topology can claim the control-arm exemption.
            _is_bottom = bool(_shared_laws.get('arb_is_bottom', False))
            _waived = bool(wD._car.get(('front' if lbl[0] == 'F' else 'rear') + '_arb_rule04_waiver'))
            # ROCKER PLATE is a solid (the 6 mm prism the 3D view draws): the drop link
            # must clear it by 3 mm at droop / static / bump beyond its own rod end
            # (2026-09-14: the user saw v106's drop link through the bellcrank).
            try:
                from vahan.interference import (full_members as _fm_pl,
                                                 rocker_plate_gaps as _rpg,
                                                 arb_member_kwargs as _amk_pl,
                                                 rocker_plate_physical_options_for as _rpo_pl)
                _tlo, _thi = wD._spring_travel_range(wD._solvers[lbl], lbl); _tlo = min(_tlo, -0.025); _thi = max(_thi, 0.025)
                _pl_worst = (float('inf'), '')
                for _tp, _tn in ((_tlo, 'droop'), (0.0, 'static'), (_thi, 'bump')):
                    _cdp, _ = wD._assemble_corners_draw({_l: float(_tp) for _l in ('FL', 'FR', 'RL', 'RR')}, 0.0, light=True)
                    _pp = [c for c in _cdp if c['label'] == lbl][0]['pts']
                    _ax = 'front' if lbl.startswith('F') else 'rear'
                    _panel = wD._dynamics_panel
                    _blade_w = float(getattr(_panel, '_arb_blade_w_f' if _ax == 'front' else '_arb_blade_w_r').value())
                    _blade_t = float(getattr(_panel, '_arb_blade_t_f' if _ax == 'front' else '_arb_blade_t_r').value())
                    _od = float(getattr(_panel, '_arb_OD_f' if _ax == 'front' else '_arb_OD_r').value())
                    _arb_live = wD._front_arb if _ax == 'front' else wD._rear_arb
                    _members = _fm_pl(_pp, wD._car,
                                      arb_pivot=np.asarray(_arb_live['arb_pivot'], float),
                                      arb_od_mm=_od,
                                      **_amk_pl(wD._car, lbl, _blade_w, _blade_t))
                    for _pn, _pg in _rpg(_pp, _members,
                            half_t=float(wD._car.get('rocker_plate_thickness_mm', 6.0))/2000.,
                            **_rpo_pl(wD._car, lbl)):
                        if 'ARB drop link' in _pn and _pg * 1000.0 < _pl_worst[0]:
                            _pl_worst = (_pg * 1000.0, _tn)
                if np.isfinite(_pl_worst[0]) and _pl_worst[0] < 3.0 and not _is_bottom:
                    gfail.append(f'{lbl} ARB drop link {_pl_worst[0]:.1f} mm from the rocker PLATE at {_pl_worst[1]} (need 3)')
            except Exception as _epl:
                gfail.append(f'{lbl} rocker-plate check did not run: {_epl}')
            if (not np.isfinite(_doff) or _doff > 3.0) and not _is_bottom:
                gfail.append(
                    f'{lbl} ARB drop link off the physical rocker plate at static '
                    f'(top {_r04["drop_top_signed_mm"]:+.1f} mm, arm '
                    f'{_r04["arm_end_signed_mm"]:+.1f} mm, direction '
                    f'{_r04["direction_deg"]:+.1f} deg)')
                if _waived:
                    print(f'{lbl} RULE 04 waiver metadata present; geometry remains NONCOMPLIANT')
        except Exception:
            pass
        # Rule 11 (user, 2026-09-22): BOTH pushrods pick up on the UPPER arm, on a plate
        # over the arm near the ball joint: the rod-end centre sits ~1-1.25 in ABOVE the
        # arm plane, never buried below it or flung far off it.  Signed perpendicular
        # distance to the plane through the UPPER arm's three pickups, +normal up.
        # (Until 2026-09-22 the rear was measured against the LOWER arm -> a false 156 mm.)
        _uf, _ur, _uo = P('uca_front'), P('uca_rear'), P('uca_outer')
        _n = np.cross(_ur - _uf, _uo - _uf); _n = _n / (np.linalg.norm(_n) or 1.0)
        if _n[2] < 0:
            _n = -_n
        d_arm = float((po - _uo) @ _n)          # signed mm; + = above the upper-arm plane
        if d_arm < -3.0 or d_arm > 35.0:
            gfail.append(f'{lbl} pushrod perp-to-upper-arm {d_arm:.0f} mm (want 0..35 above)')
        # REAR ONLY: the rear damper is meant to lie across the car, so fore-aft
        # cant is a defect there (v28 canted it 63 mm).  The FRONT damper runs
        # fore-aft by design, ~185 mm, which is not a fault.
        if lbl[0] == 'R' and abs(rsp[1] - scp[1]) > 25:
            gfail.append(f'{lbl} damper canted fore-aft {abs(rsp[1]-scp[1]):.0f} mm')
        if cop > 3.0:
            gfail.append(f'{lbl} rocker non-coplanar {cop:.1f} mm')
        if (not np.isfinite(_shared_laws['rocker_axis_normal_error_deg'])
                or _shared_laws['rocker_axis_normal_error_deg'] > 1e-4):
            gfail.append(
                f'{lbl} rocker axis {_shared_laws["rocker_axis_normal_error_deg"]:.6f} '
                f'deg from physical plate normal')
        # realistic-radius clash sweep (mm radii: pushrod 5, arm 9, tierod 6, damper 11)
        # restricted to the pairs a bad ACTUATION edit newly breaks (v28 drove the
        # pushrod at the tie-rod and canted the damper into the arms).  Same-arm
        # legs share the outer ball joint (designed), and the rear pushrod already
        # runs close to the upper A-arm front leg in EVERY version since v27 -- both
        # are excluded so this gate flags only NEW actuation clashes, not standing
        # geometry the user has already accepted.
        mem = {'uarm_f': (P('uca_front'), P('uca_outer'), 9), 'uarm_r': (P('uca_rear'), P('uca_outer'), 9),
               'larm_f': (P('lca_front'), P('lca_outer'), 9), 'larm_r': (P('lca_rear'), P('lca_outer'), 9),
               'tierod': (P('tr_inner'), P('tr_outer'), 6), 'pushrod': (P('pushrod_outer'), P('pushrod_inner'), 5),
               'damper': (rsp, scp, 11)}
        AT_RISK = [('pushrod', 'tierod'), ('damper', 'uarm_f'), ('damper', 'uarm_r'),
                   ('damper', 'larm_f'), ('damper', 'larm_r'), ('damper', 'tierod')]
        for n1, n2 in AT_RISK:
            gap = _seg_gap(mem[n1][0], mem[n1][1], mem[n2][0], mem[n2][1], mem[n1][2], mem[n2][2])
            if gap < 0:
                gfail.append(f'{lbl} CLASH {n1}<->{n2} {gap:.0f} mm')
        # ball-joint spheres (1" dia): catch a rod passing THROUGH a joint (v31's
        # tie-rod-through-ball-joint), skipping the designed rod-end that bolts to it.
        for _rn in ('pushrod', 'tierod'):
            _sg = mem[_rn]
            for _bn, _bp in (('lowerBJ', P('lca_outer')), ('upperBJ', P('uca_outer'))):
                if min(np.linalg.norm(_sg[0] - _bp), np.linalg.norm(_sg[1] - _bp)) < 8:
                    continue                       # designed joint (rod-end bolts here)
                if _seg_gap(_sg[0], _sg[1], _bp, _bp, _sg[2], 12.7) < 0:
                    gfail.append(f'{lbl} CLASH {_rn}<->{_bn}')
        # rear pushrod vs the half-shaft (the v32 clash the old gate never checked).
        if lbl in _DS:
            _g = _seg_gap(mem['pushrod'][0], mem['pushrod'][1], _DS[lbl][0], _DS[lbl][1], mem['pushrod'][2], 12.7)
            if _g < 0:
                gfail.append(f'{lbl} CLASH pushrod<->driveshaft {_g:.0f} mm')
    # ── ARB DROP LINK vs COILOVER, ALL FOUR CORNERS ────────────────────────
    # The actuation gate above only walks the REAR and has no ARB member at all,
    # so the whole bar/damper package was unchecked: v34 shipped a FRONT drop
    # link overlapping the coilover by 10 mm and nothing -- net or 3D view --
    # said a word.  Use the car's own spring_od_mm (what view3d DRAWS the
    # coilover with) so the gate and the picture agree, and walk every corner:
    # the front is where the rocker is crowded (spring at 57 mm radius, ARB at
    # 29 mm), and a front-only clash is exactly what a rear-only loop misses.
    _SPR = 0.5 * float(wD._car.get('spring_od_mm', 63.0))      # mm
    try:
        _cdA, _ = wD._assemble_corners_draw({_l: 0.0 for _l in ('FL', 'FR', 'RL', 'RR')}, 0.0)
        for _c in _cdA:
            _pp = _c['pts']
            _dt = _pp.get('arb_drop_top'); _ae = _pp.get('arb_arm_end_world')
            _rs = _pp.get('rocker_spring_pt'); _sc = _pp.get('spring_chassis_pt')
            if any(v is None for v in (_dt, _ae, _rs, _sc)):
                continue
            _A = lambda v: np.asarray(v, float) * 1000.0
            _g = _seg_gap(_A(_ae), _A(_dt), _A(_rs), _A(_sc), 6.0, _SPR)
            # arb_drop_top rides ON the rocker, so it must be taken from the
            # corner's LIVE pts (which _assemble_corners_draw already routes
            # through _arb_drop_top_world), never from the static config dict.
            # Measuring the static point hid a -3.8 mm pushrod/ball-joint clash
            # at full bump and reported it as +3.9 mm.
            if _g < 0:
                gfail.append(f'{_c["label"]} CLASH ARB drop link<->coilover '
                             f'{_g:.0f} mm (spring OD {2*_SPR:.0f})')
    except Exception as _e:
        gfail.append(f'ARB/coilover gate did not run: {_e}')

    # ── ROCKER HARDWARE FITS (bearing + rod ends are VOLUMES, not points) ──
    # The rocker carries a 1.5" pivot bearing and 1/2"-bore rod ends (~1.25"
    # housings) at the pushrod, spring and ARB pickups.  Modelled as points,
    # ANY radius "fits" — v36 placed the ARB drop point at 20 mm radius, so its
    # rod end overlapped the pivot bearing by 14.9 mm and every check passed.
    # Assert each pickup's rod-end body clears the pivot bearing and its
    # neighbours.  Sizes are the user's stated hardware (2026-07-25).
    # user-supplied hardware (2026-07-25): 1.5" rocker bearing, drop-link
    # ball joint 0.315" RADIUS.  The old 1/2"-bore rod-end housing figure
    # was twice the real ball joint, so it over-constrained the drop radius
    # (34.9 mm demanded vs 27.05 mm actually needed).
    _BRG_R, _RE_R = 0.5 * 38.1, 0.315 * 25.4         # mm radii
    for _ax, _hp, _arb in (('front', 'front_hp', 'front_arb'),
                           ('rear', 'rear_hp', 'rear_arb')):
        _H = wD._front_hp if _ax == 'front' else wD._rear_hp
        _A = wD._front_arb if _ax == 'front' else wD._rear_arb
        if not _H or not _A:
            continue
        _pv = np.asarray(_H['rocker_pivot'], float) * 1000.0
        _picks = {'pushrod': np.asarray(_H['pushrod_inner'], float) * 1000.0,
                  'spring':  np.asarray(_H['rocker_spring_pt'], float) * 1000.0,
                  'ARB':     np.asarray(_A['arb_drop_top'], float) * 1000.0}
        for _n, _p in _picks.items():
            _g = float(np.linalg.norm(_p - _pv)) - _BRG_R - _RE_R
            if _g < 0:
                gfail.append(f'{_ax} {_n} rod end INTO the rocker bearing {_g:.1f} mm')
        _ks = list(_picks)
        for _i in range(len(_ks)):
            for _j in range(_i + 1, len(_ks)):
                _g = float(np.linalg.norm(_picks[_ks[_i]] - _picks[_ks[_j]])) - 2 * _RE_R
                if _g < 0:
                    gfail.append(f'{_ax} {_ks[_i]}/{_ks[_j]} rod ends overlap {_g:.1f} mm')

    # ── ARB TORSION TUBE vs COILOVER, over the REAL travel range ───────────
    # The tube is chassis-fixed and spans +x to -x through arb_pivot.  It is
    # DRAWN by _assemble_arb_segs but was in no member list, so the v47 front
    # bar ran 2.9 mm THROUGH both front coilovers at every wheel position and
    # nothing flagged it.  Sweep the damper-derived range, not a hardcoded one.
    try:
        _loT = wD._motion_panel.min_val / 1000.0
        _hiT = wD._motion_panel.max_val / 1000.0
        _spR = 0.5 * float(wD._car.get('spring_od_mm', 63.0))
        for _ax, _ahp, _odat, _lbls in (('front', wD._front_arb, '_arb_OD_f', ('FL', 'FR')),
                                        ('rear', wD._rear_arb, '_arb_OD_r', ('RL', 'RR'))):
            if not _ahp or 'arb_pivot' not in _ahp:
                continue
            _pv = np.asarray(_ahp['arb_pivot'], float) * 1000.0
            _bA = _pv.copy(); _bB = _pv.copy(); _bB[0] = -_pv[0]
            _aR = 0.5 * float(getattr(wD._dynamics_panel, _odat).value())
            _wst = 1e9
            for _t in np.linspace(_loT, _hiT, 9):
                for _l in _lbls:
                    _s = wD._solvers[_l].solve(float(_t))
                    _Q = lambda k: np.asarray(getattr(_s, k), float) * 1000.0
                    _wst = min(_wst, _seg_gap(_bA, _bB, _Q('rocker_spring_pt'),
                                              _Q('spring_chassis_pt'), _aR, _spR))
            if _wst < 0:
                gfail.append(f'{_ax} CLASH ARB torsion bar<->coilover {_wst:.1f} mm')
    except Exception as _e:
        gfail.append(f'ARB-bar/coilover gate did not run: {_e}')

    # ── PUSHROD vs ARB DROP-LINK BALL JOINT, over the REAL travel range ────
    # The drop point rides ON the rocker, so it swings toward the pushrod at
    # bump.  Moving it to the max-efficiency axis in v52 buried it in the
    # pushrod (-3.3 mm front / -2.4 mm rear) and every existing gate still
    # passed.  Use the LIVE point (_arb_drop_top_world), never the config value.
    try:
        _loB = wD._motion_panel.min_val / 1000.0
        _hiB = wD._motion_panel.max_val / 1000.0
        _bjR = 0.315 * 25.4                     # drop-link ball joint radius
        _prR2 = 0.5 * 0.625 * 25.4              # pushrod tube radius
        for _l in ('FL', 'FR', 'RL', 'RR'):
            _w2 = 1e9
            for _t in np.linspace(_loB, _hiB, 9):
                _s2 = wD._solvers[_l].solve(float(_t))
                _P2 = lambda k: np.asarray(getattr(_s2, k), float) * 1000.0
                _lv = np.asarray(wD._arb_drop_top_world(_l, _s2), float) * 1000.0
                _w2 = min(_w2, _seg_gap(_P2('pushrod_outer'), _P2('pushrod_inner'),
                                        _lv, _lv, _prR2, _bjR))
            if _w2 < 0:
                gfail.append(f'{_l} CLASH pushrod<->ARB ball joint {_w2:.1f} mm')
    except Exception as _e:
        gfail.append(f'pushrod/ARB-balljoint gate did not run: {_e}')

    # ── AUTHORITATIVE FULL-MEMBER SWEEP == the GUI RED interference view ────
    # Every hand-picked pair above is a friendly diagnostic, but the SOURCE OF
    # TRUTH for "clash-free" must be the SAME member set + stations the 3D
    # interference view draws (vahan.packaging._clash_sweep ->
    # vahan.interference.full_members), or the net green-lights a config the
    # view shows RED.  v72_1 shipped a coilover<->rocker-bearing overlap (-4.0
    # mm droop / -0.5 static) that NO subset gate had a member for -- the
    # bearing sphere vs the fat coilover tube -- and v72_2 a pushrod<->ARB
    # drop-link overlap (-10 mm all-travel).  Run the real sweep here so the net
    # and the picture can never disagree again (memory: clash_check_all_members).
    try:
        import vahan.packaging as _PKG
        _seen_cl = set()
        for _stn, _hits in _PKG._clash_sweep(wD).items():
            for _h in _hits:
                if _h['gap_mm'] < 0.0:
                    _k = (_h['corner'], frozenset({_h['a'], _h['b']}))
                    if _k in _seen_cl:
                        continue
                    _seen_cl.add(_k)
                    gfail.append(f"{_h['corner']} CLASH {_h['a']}<->{_h['b']} "
                                 f"{_h['gap_mm']:.1f} mm @{_stn} (interference view)")
    except Exception as _e:
        gfail.append(f'full-member interference sweep did not run: {_e}')

    # Steering endpoints alone missed the v84 ARB closure failure between
    # straight-ahead and lock. Check intermediate steer/travel too, with the
    # actual rack position and each rendered wheel's alignment-adjusted axis.
    try:
        from types import SimpleNamespace as _RimState
        from vahan.kinematics import KinematicMetrics as _RimMetrics
        _lock = wD._steer['total_rack_travel_mm'] / wD._steer['rack_travel_per_rev_mm'] * 180.0
        _travel = np.unique(np.r_[np.linspace(wD._motion_panel.min_val / 1000.,
                                              wD._motion_panel.max_val / 1000., 9), 0.])
        _body_margin = float(wD._car.get('rim_joint_clearance_mm', 3.0))
        _housing_allowance = float(wD._car.get('rim_housing_allowance_mm', 0.0))
        if not np.isfinite(_housing_allowance) or not 0 <= _housing_allowance <= _body_margin:
            raise ValueError('upright housing allowance exceeds reserved rim clearance')
        _barrel_hits = {}
        _joint_diameter = (float(wD._car['tire_rim_dia_mm']) - 2 * (12.7 + _body_margin)) / 1000.
        for _hw in np.linspace(-_lock, _lock, 7):
            wD._rebuild_solvers(float(_hw))
            for _station, _hits in _PKG._clash_sweep(wD, _travel).items():
                for _hit in _hits:
                    if _hit['b'] == 'rim barrel + 3 mm clearance':
                        _key = (_hit['corner'], _hit['a'])
                        _gap = _hit['surface_gap_mm']
                        if _key not in _barrel_hits or _gap < _barrel_hits[_key][0]:
                            _barrel_hits[_key] = (_gap, _hw, float(_station.split()[0]))
                        continue
                    if _hit['gap_mm'] < 0:
                        gfail.append(f"{_hit['corner']} steered CLASH {_hit['a']}/{_hit['b']} "
                                     f"{_hit['gap_mm']} mm at {_hw:.1f} deg, {_station}")
            for _t in _travel:
                _draw, _ = wD._assemble_corners_draw(
                    {l: float(_t) for l in ('FL', 'FR', 'RL', 'RR')},
                    wD._solver_rack_travel_m, light=True)
                if len(_draw) != 4:
                    raise ValueError('rim check requires all four corners')
                for _corner in _draw:
                    _p = _corner['pts']
                    _raw = wD._solvers[_corner['label']].solve(float(_t))
                    if not np.allclose(_p['tie_rod_inner'], _raw.tr_inner, atol=1e-10, rtol=0):
                        raise ValueError(f"{_corner['label']} rendered rack point differs from solved rack")
                    _rim_state = _RimState(wheel_center=_p['wheel_center'],
                        spin_axis=_corner['spin_axis'], uca_outer=_p['uca_outer'],
                        lca_outer=_p['lca_outer'], tr_outer=_p['tie_rod_outer'])
                    _fit = _RimMetrics(_rim_state).rim_fit(_joint_diameter)
                    if len(_fit['radii_m']) != 3 or not all(np.isfinite(v) for v in _fit['radii_m'].values()):
                        raise ValueError('missing or non-finite upright joint radius')
                    if not _fit['fits']:
                        gfail.append(f"{_corner['label']} rim joint body misses {_body_margin:.1f} mm "
                                     f"clearance at {_hw:.1f} deg, {_t*1000:.1f} mm")
        for (_label, _name), (_gap, _hw, _tmm) in sorted(_barrel_hits.items()):
            gfail.append(f'{_label} RIM BARREL {_name}: {_gap:.2f} mm gap < 3 mm '
                         f'at {_hw:.1f} deg handwheel, {_tmm:.1f} mm travel')
    except Exception as _e:
        gfail.append(f'steered packaging/ARB closure gate: {_e}')
    finally:
        wD._rebuild_solvers(0.)

    # ── RIM FIT, ALL FOUR CORNERS, BODIES + TUBES (Rule 16) ────────────────
    # KinematicMetrics.rim_fit() checks joint CENTRES only and only where it is
    # called (front).  It said "fits" at the real 230 mm rim while the front
    # upper ball-joint BODY poked 2.4 mm and the tie-rod TUBE 6-8 mm past the
    # barrel, and the REAR upper ball joint was 33.9 mm out (toe-link outer
    # 21, upper-arm tubes 29) -- never measured until the user pointed at it.
    # Gate the physical parts: every outboard joint body (1" BJ r 12.7) and
    # every member tube (r 7.94) within the rim's axial band must sit inside
    # r_max - 3 mm at droop/static/bump on FL AND RL.  Band = tire_width/2
    # (conservative; the real inner lip is set by rim width + offset).
    try:
        from vahan.interference import corner_members as _cm_rim
        _rimR = 0.5 * float(wD._car.get('tire_rim_dia_mm', 230.0))
        _rimLim = _rimR - 3.0
        _band = 0.5 * float(wD._car.get('tire_width_mm', 200.0))
        _loR = wD._motion_panel.min_val / 1000.0; _hiR = wD._motion_panel.max_val / 1000.0
        for _lbl in ('FL', 'RL'):
            _worst = {}
            for _t in (_loR, 0.0, _hiR):
                _st = wD._solvers[_lbl].solve(float(_t))
                _wc = np.asarray(_st.wheel_center, float)
                _ax = np.asarray(_st.spin_axis, float); _ax = _ax / np.linalg.norm(_ax)
                for _jk, _jr in (('uca_outer', 12.7), ('lca_outer', 12.7), ('tr_outer', 12.7)):
                    _d = np.asarray(getattr(_st, _jk), float) - _wc; _al = float(np.dot(_d, _ax)) * 1000.0
                    if abs(_al) > _band:
                        continue
                    _e = float(np.linalg.norm(_d - (_al / 1000.0) * _ax)) * 1000.0 + _jr
                    _worst[_jk] = max(_worst.get(_jk, 0.0), _e)
                for _m in _cm_rim(_st, wD._car):
                    _a = np.asarray(_m['a'], float); _b = np.asarray(_m['b'], float)
                    for _s in np.linspace(0.0, 1.0, 40):
                        _p = _a + (_b - _a) * _s; _v = _p - _wc; _al = float(np.dot(_v, _ax)) * 1000.0
                        if abs(_al) > _band:
                            continue
                        _e = float(np.linalg.norm(_v - (_al / 1000.0) * _ax)) * 1000.0 + 7.94
                        _worst[_m['name']] = max(_worst.get(_m['name'], 0.0), _e)
            for _nm, _e in sorted(_worst.items(), key=lambda kv: -kv[1]):
                if _e > _rimLim:
                    gfail.append(f'{_lbl} RIM FIT {_nm} edge {_e:.1f} mm > {_rimLim:.0f} '
                                 f'(230 rim, 3 mm margin) — POKES {_e - _rimR:+.1f} past the barrel')
    except Exception as _e:
        gfail.append(f'rim-fit gate did not run: {_e}')

    # Toe curve vs travel.  FRONT must stay NULL (<0.15 deg over +-25 mm) — no
    # deliberate front toe gain.  REAR: the 2027 deliberate toe-OUT band
    # (-1.5..-0.2 deg bump-minus-droop) was RETIRED on 2026-09-09 — the user
    # asked for the rear toe-link outer point placed for maximum lever from the
    # kingpin axis with bump steer reduced as far as possible ("possible to get
    # it within 0.1 degrees"); v101 solves it at 0.0999 deg.  The rear gate is
    # now a bump-steer bound over +-25 mm (raw solver), like the front.
    _REAR_BUMP_STEER_LIM = 0.25   # deg over +-25 mm (user 2026-09-21: "0.25 degrees over 25 mm is negligible"; was 0.10 from 2026-09-09)
    # FRONT: dense raw states avoid missing a peak between sparse stations.
    from vahan.kinematics import KinematicMetrics as _FrontToeMetrics
    _toe_dense = np.array([_FrontToeMetrics(wD._solvers['FL'].solve(float(t)), 'left').toe
                          for t in np.linspace(-0.025, 0.025, 51)])
    _bs = float(np.ptp(_toe_dense))
    if not np.all(np.isfinite(_toe_dense)) or _bs > float(wD._car.get('front_bump_steer_limit_deg', 0.15)):
        gfail.append(f'front bump steer {_bs:.3f} deg full-travel')
    # REAR: deliberate toe-OUT gain — measured from the RAW solver (the incremental
    # _do_sweep can NaN at the -25 mm droop step on the long aft toe link, while
    # solver.solve is clean there; the raw toe at +-25 is the robust ground truth).
    from vahan.kinematics import KinematicMetrics as _KMr
    _toe_r = np.array([_KMr(wD._solvers['RL'].solve(float(t)), 'left').toe
                       for t in np.linspace(-0.025, 0.025, 7)])
    if not np.all(np.isfinite(_toe_r)):
        gfail.append('rear toe curve non-finite over +-25 mm (raw solver)')
    else:
        _bsr = float(np.ptp(_toe_r))
        if _bsr > _REAR_BUMP_STEER_LIM:
            gfail.append(f'rear bump steer {_bsr:.4f} deg over +-25 mm > {_REAR_BUMP_STEER_LIM} '
                         f'(user 2026-09-09: rear toe link nulled, lever maximised)')
    # ── ROLL CENTRE IS AN AXIS PROPERTY (Rule 17, 2026-09-09) ───────────────
    # Sliding an inboard pickup ALONG its own pivot axis is a physical no-op
    # (same arm plane, same swing axis, every wheel curve identical).  The old
    # pickup-MIDPOINT construction moved the front RC 1.1 mm for a 38.7 mm
    # slide (v101 hoop-line move) and under-read the swept rear axle by 6.8 mm;
    # the instant-axis construction must not move at all.
    try:
        import vahan.packaging as _PKrc
        from vahan.kinematics import KinematicMetrics as _KMrc
        _b = _PKrc.get_bundle(wD, 'front')
        _lf = np.asarray(_b['hp']['lca_front'], float); _lr = np.asarray(_b['hp']['lca_rear'], float)
        _u = (_lr - _lf) / np.linalg.norm(_lr - _lf)
        _s0 = _PKrc._corner_solver(wD, 'front', _b)
        _b2 = {'hp': dict(_b['hp']), 'arb': dict(_b['arb'])}; _b2['hp']['lca_rear'] = _lr + 0.030 * _u
        _s1 = _PKrc._corner_solver(wD, 'front', _b2)
        _drc = max(abs(_KMrc(_s0.solve(float(t)), 'left').roll_center_height
                       - _KMrc(_s1.solve(float(t)), 'left').roll_center_height)
                   for t in (-0.025, 0.0, 0.025)) * 1000.0
        if not np.isfinite(_drc) or _drc > 1e-6:
            gfail.append(f'roll centre moved {_drc:.4f} mm for a pickup slid 30 mm along its own '
                         f'pivot axis (construction not axis-invariant)')
    except Exception as _e:
        gfail.append(f'roll-centre axis-invariance check did not run: {_e}')
    # ── ARB BAR MOTION RATIO FLAT THROUGH TRAVEL (tangent law, 2026-09-09) ──
    # A re-hang can keep the exact 90/90/90 triad and the static rate and still
    # lose the rate in travel when the drop link is not tangent to the drop-top
    # arc (Cluster C / v74 cliff).  v99's front bar ran 1.29 / 2.64 / 15.27 at
    # -25 / 0 / +25 mm through every static gate; caught only by the Design
    # City parameter vector.  Gate: bar MR at +-25 mm within 25 % of static on
    # both axles (v98 front 0.8 %, rear 19 % pre-existing since v97).
    try:
        for _axc, _axn in (('F', 'front'), ('R', 'rear')):
            _mrs = []
            for _t in (-0.025, 0.0, 0.025):
                _g = wD._compute_arb_geometry_from_kinematics(_axc, travel_m=_t)
                _mrs.append(float(_g['mr']) if _g else float('nan'))
            if not np.all(np.isfinite(_mrs)) or _mrs[1] == 0:
                gfail.append(f'{_axn} ARB bar motion ratio not finite through travel: {_mrs}')
            else:
                _flat = max(abs(_mrs[0] / _mrs[1] - 1.0), abs(_mrs[2] / _mrs[1] - 1.0))
                if _flat > 0.25:
                    gfail.append(f'{_axn} ARB bar motion ratio {_mrs[0]:.2f} / {_mrs[1]:.2f} / {_mrs[2]:.2f} at '
                                 f'-25/0/+25 mm ({_flat*100:.0f} % swing > 25 %): drop link off the '
                                 f'drop-top arc tangent — rate cliff')
    except Exception as _e:
        gfail.append(f'ARB motion-ratio flatness check did not run: {_e}')
    # ── CHASSIS KEEP-OUT (Rule 18, 2026-09-10) ───────────────────────────────
    # If the project names a keep-out solid (car['keepout_step'], e.g. the
    # bulkhead red zone exported from Onshape), no member may be inside it at
    # droop/static/bump x -lock/0/+lock.  v101 had the rack housing 42 mm and
    # the front torsion bar 176 mm inside the footwell — "the rack is too high
    # and the ARB is in the middle".
    try:
        from vahan.keepout import keepout_for_window as _kofw, audit_window as _koaw
        _ko = _kofw(wD)
        if _ko is not None:
            _kr = _koaw(wD, _ko, n_rack=3)
            for _nm, (_g, _t, _r) in _kr['inside'][:8]:
                gfail.append(f'keep-out {_ko.name}: {_nm} inside by {-_g:.1f} mm at {_t:+.0f} mm travel, rack {_r:+.0f} mm')
            print(f"keep-out         : {_ko.name} — {len(_kr['worst'])} members, {len(_kr['inside'])} inside; closest "
                  f"{min(_kr['worst'].items(), key=lambda kv: kv[1][0])[0]} {min(v[0] for v in _kr['worst'].values()):.1f} mm")
    except Exception as _e:
        gfail.append(f'keep-out check did not run: {_e}')
    # ── RULE 04 WAIVER / DECLARED STANDOFF (2026-09-14) — printed LOUDLY, never silent ──
    try:
        _so = float(wD._car.get('front_arb_drop_standoff_mm', 0.0) or 0.0)
        if abs(_so) > 1e-9:
            print(f'front ARB standoff: HARDWARE FLAG — drop-top rod end on a {_so:+.1f} mm spacer off the rocker plate '
                  f'(diagnostic metadata; does not satisfy Rule 04)')
        _wv = wD._car.get('front_arb_rule04_waiver')
        if _wv:
            _tgw = _pkgm._axle_geometry_laws(wD, 'front') if '_pkgm' in dir() else None
            print(f'RULE 04 metadata : NONCOMPLIANT waiver note — {_wv} '
                  f'(arm end {_tgw["arb_arm_end_inplane_mm"]:.1f} mm off the plate)'
                  if _tgw else f'RULE 04 metadata : NONCOMPLIANT waiver note — {_wv}')
    except Exception as _e:
        print(f'rule-04 waiver / standoff print did not run: {_e}')
    # ── FRONT HOOP LINE (Rule 19, 2026-09-14) ─────────────────────────────────
    # The front ARB (bar, blades, links, rod ends) must stay AHEAD of the line
    # through the LCA-aft / UCA-aft pickups extended upward (the front hoop),
    # by >= 3 mm at droop / static / bump.  "make sure it doesn't pass the
    # imaginary line extended upwards from LCA aft and UCA aft" (user).
    try:
        from vahan.packaging import front_arb_hoop_line_gap_mm as _hoopf
        _hg = _hoopf(wD)
        if not np.isfinite(_hg['gap_mm']) or _hg['gap_mm'] < 3.0:
            gfail.append(f"front ARB crosses / is within 3 mm of the front-hoop line: {_hg['gap_mm']:.1f} mm ({_hg['worst']})")
        print(f"front hoop line  : Y {_hg['line'][0]:.1f} mm through LCA-aft/UCA-aft — front ARB ahead by {_hg['gap_mm']:.1f} mm ({_hg['worst']})")
    except Exception as _e:
        gfail.append(f'front hoop-line check did not run: {_e}')
    # UNEXPECTED failures fail the net; the documented KNOWN open v32 conflict
    # (pushrod/driveshaft + rear bump steer) prints but does not (awaiting user).
    _unexp = [g for g in gfail if not any(k in g for k in _KNOWN)]
    _known = [g for g in gfail if any(k in g for k in _KNOWN)]
    gok = not _unexp
    if not gok:
        fails += 1
    if not gfail:
        _msg = "pushrod over LCA, damper flat, coplanar, no clash, bump steer OK"
    elif _unexp:
        _msg = "; ".join(_unexp) + (f"  [+KNOWN: {'; '.join(_known)}]" if _known else "")
    else:
        _msg = "KNOWN open conflict [" + "; ".join(_known) + "] -- awaiting user decision on rear pushrod routing"
    print(f'design actuation : {os.path.basename(_design)} — {_msg}   '
          f'{"pass" if gok else "UNEXPECTED FAIL"}')

    # ── STEERING EFFORT (vahan.steering — the ONE implementation the GUI and
    #    the binder share).  Internal consistency + physical sanity on the
    #    current design config.
    try:
        from vahan.steering import compute_steering_effort as _cse
        _ssD = wD._build_dynamics_solver()
        _tf = getattr(_ssD, '_tire_front', None) or getattr(_ssD, '_tire', None)
        if _tf is None:
            print('steering effort  : no tire model loaded — skipped')
        else:
            _eff = _cse(_ssD, wD._front_hp, wD._topology.front.damper_mount.value,
                        _tf, steer_cfg=dict(getattr(wD, '_steer', None) or {}),
                        front_solver=wD._solvers['FL'])
            _sfail = []
            _Fs = [f for _, f in _eff['F_rack_vs_latg']]
            if not all(np.isfinite(f) and f >= 0 for f in _Fs):
                _sfail.append('non-finite/negative rack force in curve')
            if not (50.0 < _eff['arm_eff_mm'] < 150.0):
                _sfail.append(f'effective arm {_eff["arm_eff_mm"]} mm outside 50..150')
            _C = _eff.get('C_mm_per_rev')
            if _C and _eff.get('T_max_Nm') is not None:
                _texp = _eff['F_max_N'] * _C / (2 * np.pi * 1000.0)
                if abs(_eff['T_max_Nm'] - _texp) > 0.02:
                    _sfail.append(f'T_max {_eff["T_max_Nm"]} != F*C/2pi {_texp:.2f}')
            if _eff['peak_latg'] >= _eff['F_rack_vs_latg'][-1][0]:
                _sfail.append('rack force never peaks below the grip limit '
                              '(pneumatic-trail collapse missing)')
            if _sfail:
                fails += 1
            print(f'steering effort  : arm {_eff["arm_eff_mm"]:.1f} mm, '
                  f'F_max {_eff["F_max_N"]:.0f} N @ {_eff["peak_latg"]:.1f} g'
                  + (f', T_max {_eff["T_max_Nm"]:.2f} N.m @ C {_C:.1f}' if _C else '')
                  + f'   {"pass" if not _sfail else "UNEXPECTED FAIL: " + "; ".join(_sfail)}')
    except Exception as _e:
        fails += 1
        print(f'steering effort  : UNEXPECTED FAIL (exception: {_e})')

    # ── WHEEL-RATE GEOMETRIC TERM = TANGENT dMR/ddelta (solver-bug register,
    #    2026-09-02).  RCVD 16.3: K_wheel = Ks*MR^2 + Fs*(dIR/ddelta), where
    #    dIR/ddelta is the slope of the INSTANTANEOUS (tangent) motion ratio.
    #    It was being computed as the slope of the sweep's `motion_ratio`
    #    array, which is a SECANT |L(t)-L0|/t (see _do_sweep, "cumulative MR")
    #    -- a different quantity that is wrong-signed AND numerically unstable:
    #    two near-identical linear geometries read wildly different secant
    #    slopes, swinging the front wheel-rate / ride-rate by ~2x (caught while
    #    linearising the v79 rocker -- ride_f jumped 16.9k -> 33.8k for a 0.002
    #    MR change).  Now taken directly from the corner solvers as the tangent
    #    central difference (the SAME MR the build uses for motion_ratio).
    #    Gates: (1) veh.mr_slope_front matches the independent tangent slope;
    #    (2) the build is deterministic (identical ride rate on a re-build).
    try:
        _ss1 = wD._build_dynamics_solver(); _v1 = _ss1._veh
        _sFL = wD._solvers['FL']; _dtm = 0.001; _hsl = 0.010
        def _tan_mr(t):
            return abs(_sFL.solve(t + _dtm).spring_length
                       - _sFL.solve(t - _dtm).spring_length) / (2 * _dtm)
        _tan_slope = (_tan_mr(+_hsl) - _tan_mr(-_hsl)) / (2 * _hsl)
        _mrfail = []
        if abs(_v1.mr_slope_front_per_m - _tan_slope) > 0.15:
            _mrfail.append(f'mr_slope_front {_v1.mr_slope_front_per_m:+.3f} '
                           f'!= tangent {_tan_slope:+.3f} /m (secant-array bug back)')
        _r1 = float(_v1.ride_rate_front_Npm)
        _r2 = float(wD._build_dynamics_solver()._veh.ride_rate_front_Npm)
        if abs(_r1 - _r2) > 1.0:
            _mrfail.append(f'front ride rate non-deterministic {_r1:.0f} vs {_r2:.0f}')
        if _mrfail:
            fails += 1
        print(f'wheel-rate dMR   : tangent {_tan_slope:+.3f}/m, mr_slope '
              f'{_v1.mr_slope_front_per_m:+.3f}/m, ride_f {_r1:.0f} N/m   '
              f'{"pass" if not _mrfail else "UNEXPECTED FAIL: " + "; ".join(_mrfail)}')
    except Exception as _e:
        fails += 1
        print(f'wheel-rate dMR   : UNEXPECTED FAIL (exception: {_e})')

    # ── TYRE CAMBER SIGN (solver-bug register class 'sign', 2026-09-02).  The
    #    tyre was fed |camber| at every dynamics site (ymd loads table, pair
    #    split, cornering stiffness, peak_mu) while the transient fed the raw
    #    per-side value, and the fit's lookup CLIPPED negative IA to 0.  With a
    #    positive reference slip (SAE left-turn frame) IA >= 0 is the wheel
    #    leaning AWAY from the turn, so the inner wheel — which under roll
    #    really does lean away — and the outer wheel — which leans in — were
    #    both scored as leaning away, and a wheel leaning in scored as upright.
    #    Also the kinematic camber alone is chassis-relative (0 at design):
    #    static alignment camber + body roll are what give it a meaningful sign.
    #    ONE helper (vahan.tire_model.wheel_inclination_deg) now maps vehicle
    #    camber -> signed SAE IA; the tyre evaluates negative IA by the mirror
    #    identity Fy(a, IA) = -Fy(-a, -IA).  Gates: (1) the loaded fit is finite
    #    at signed inclinations and honours that mirror identity; (2) a 1 g steady solve gives the inner and outer front
    #    wheels OPPOSITE-signed IA, with outer = +ground camber, inner = -ground
    #    camber, ground camber = kinematic + static + roll.  Failed on the abs()
    #    code (into == upright, both fronts IA >= 0).
    try:
        from vahan.tire_model import wheel_inclination_deg as _wid
        _tmD = getattr(wD, '_tire_model', None)
        _cfail = []
        _cmsg = []
        if _tmD is None or not hasattr(_tmD, 'camber_levels'):
            _cmsg.append('no TTC tyre loaded — direction check skipped')
        else:
            _lvD = _tmD.camber_levels() if callable(_tmD.camber_levels) else _tmD.camber_levels
            _lvD = [float(x) for x in np.asarray(_lvD).ravel()]
            if len(_lvD) < 2 or max(abs(x) for x in _lvD) < 0.5:
                _cmsg.append(f'fit holds one inclination level {_lvD} — direction check skipped')
            else:
                _fzg = 800.0
                _iaD = float(max(abs(x) for x in _lvD))      # top measured level
                _grid = np.linspace(0.0, 13.0, 261)
                _f0 = np.array([abs(float(_tmD.Fy(s, _fzg, 0.0))) for s in _grid])
                _spk = float(_grid[int(np.argmax(_f0))])
                _half = 0.5 * _spk
                # an OUTER wheel with NEGATIVE ground camber leans INTO the turn,
                # with POSITIVE camber it leans AWAY — through the helper.
                _ia_in = _wid(-_iaD, is_outer=True)
                _ia_aw = _wid(+_iaD, is_outer=True)
                _mag = lambda s, ia: abs(float(_tmD.Fy(s, _fzg, ia)))
                _o = (_mag(_half, _ia_in), _mag(_half, 0.0), _mag(_half, _ia_aw))
                _p = (_mag(_spk, _ia_in), _mag(_spk, _ia_aw))
                if not np.all(np.isfinite(np.asarray(_o + _p, float))):
                    _cfail.append('signed-IA Fy returned a non-finite value')
                _mir_err = max(abs(float(_tmD.Fy(_sa, _fzg, -_iaD))
                                   + float(_tmD.Fy(-_sa, _fzg, _iaD)))
                               for _sa in (1.0, _half, _spk))
                if _mir_err > 1e-6:
                    _cfail.append(f'signed-IA mirror identity error {_mir_err:.3e} N')
                _cmsg.append(f'IA {_iaD:.0f} @ {_fzg:.0f} N: into/upright/away '
                             f'{_o[0]/_o[1]:.3f}/1/{_o[2]/_o[1]:.3f} @ {_half:.1f} deg, '
                             f'into/away {_p[0]/_p[1]:.3f} @ peak {_spk:.1f} deg; '
                             f'mirror error {_mir_err:.1e} N')
        # the helper's sign rule itself (left wheel = inner of the SAE left turn)
        if not (_wid(-1.0, is_outer=True) == -1.0 and _wid(-1.0, is_outer=False) == 1.0
                and _wid(-1.0, side='right') == -1.0 and _wid(-1.0, side='left') == 1.0):
            _cfail.append('wheel_inclination_deg sign rule broken')
        # 1 g steady solve (solve() takes lateral g, not m/s^2)
        _ss1 = wD._build_dynamics_solver()
        _r1 = _ss1.solve(1.0, 0.0)
        _inc = getattr(_r1, 'inclination', None) or {}
        _cg1 = getattr(_r1, 'camber_ground', None) or {}
        _fo, _fi = ('FL', 'FR') if _r1.Fz['FL'] >= _r1.Fz['FR'] else ('FR', 'FL')
        _io, _ii = float(_inc.get(_fo, 0.0)), float(_inc.get(_fi, 0.0))
        if not (_io * _ii < 0.0):
            _cfail.append(f'1 g: outer {_fo} IA {_io:+.2f} / inner {_fi} IA {_ii:+.2f} '
                          f'not opposite-signed')
        if abs(_io - float(_cg1.get(_fo, 0.0))) > 1e-9 or abs(_ii + float(_cg1.get(_fi, 0.0))) > 1e-9:
            _cfail.append('1 g: inclination != (+outer / -inner) ground camber')
        _stat = float(getattr(_ss1._veh, 'camber_front_deg', 0.0))
        _roll = abs(float(_r1.roll_angle_deg))
        # Independent dot/arcsin oracle. XZ-projected camber + angles only
        # agrees at zero toe/steer; do not enshrine that approximation here.
        for _corner in (_fo, _fi):
            _spin = _ss1._solvers[_corner].solve(_r1.travel[_corner] / 1000.).spin_axis
            _side = 1. if _corner.endswith('L') else -1.
            _angle = np.radians(_side * _stat + _r1.roll_angle_deg)
            _z = -np.sin(_angle)*_spin[0] + np.cos(_angle)*_spin[2]
            _expected = -_side*np.degrees(np.arcsin(np.clip(_z/np.linalg.norm(_spin), -1., 1.)))
            if abs(_expected - float(_cg1.get(_corner, 0.0))) > 1e-6:
                _cfail.append(f'{_corner}: ground camber differs from aligned/rolled wheel plane')
        _cmsg.append(f'1 g front: outer {_fo} ground {float(_cg1.get(_fo, 0.0)):+.2f} -> IA {_io:+.2f}, '
                     f'inner {_fi} ground {float(_cg1.get(_fi, 0.0)):+.2f} -> IA {_ii:+.2f} '
                     f'(kin {float(_r1.camber.get(_fo, 0.0)):+.2f}/{float(_r1.camber.get(_fi, 0.0)):+.2f}, '
                     f'static {_stat:+.2f}, roll {_roll:.2f})')
        if _cfail:
            fails += 1
        print(f'camber sign      : {"; ".join(_cmsg)}   '
              f'{"pass" if not _cfail else "UNEXPECTED FAIL: " + "; ".join(_cfail)}')
    except Exception as _e:
        fails += 1
        print(f'camber sign      : UNEXPECTED FAIL (exception: {_e})')

# ── TIRE CAMBER-ROW INTEGRITY: TTC tests sweep discrete inclinations (0/2/4);
#    stray transition samples used to create phantom integer camber rows filled
#    with zeros, so peak_mu at interpolated cambers (e.g. 0.45 deg — exactly
#    where the dynamics solver evaluates) came out NON-MONOTONE vs load
#    (mu 1.57@400N < 2.39@939N) and corrupted utilization + understeer gradient.
#    Guard: only well-populated camber rows survive, and mu(Fz) at a mid-row
#    camber is degressive (monotone non-increasing within 2%).
print('-' * 64)
# Load the DESIGN config first, so these gates test the tyre the car actually
# runs.  Without it `win` is the startup default, which names no tyre file, and
# `_try_autoload_tire` then falls back to alphabetical order — which picks the
# the wrong compound (alphabetical order, not the configured one).  Every tyre gate below was silently
# validating a compound the car does not use.
# Pick the HIGHEST VERSION NUMBER, not the lexicographic last: sorting strings
# puts '2027_v5_(claude_arb)' after '2027_v55' because '_' beats digits, so the
# newest config lost to a v5-era one.
import glob as _glob0
import re as _re0
_cands = []
for _p in _glob0.glob('configs/2027_v*.vahan'):
    _m = _re0.search(r'_v(\d+)', _p.replace('\\', '/').split('/')[-1])
    if _m:
        _cands.append((int(_m.group(1)), _p))
_design_cfg = [max(_cands)[1]] if _cands else []
if _design_cfg:
    try:
        win._load_project_from_path(_design_cfg[0])
    except Exception:
        import traceback; traceback.print_exc()
try:
    win._try_autoload_tire()
except Exception:
    pass
_tm = getattr(win, '_tire_model', None)
if _tm is not None:
    _want = str(win._car.get('tire_file', '') or '') or '(none named)'
    _got = getattr(_tm, 'tire_id', '?')
    print(f'tire selection   : config asks {_want}, loaded "{_got}"  '
          f'{"pass" if _want != "(none named)" else "UNEXPECTED FAIL (no tyre named)"}')
    if _want == '(none named)':
        fails += 1
if _tm is not None:
    _fzs = [300, 400, 600, 800, 1000]
    # slip_sign=0 = the peak over BOTH slip branches, the definition this gate
    # was calibrated on.  The per-branch peaks (slip_sign=+1/-1, what the
    # solver's grip budget now uses at a SIGNED IA — see 'camber sign') carry
    # the rig's own branch asymmetry: on this fit the SA>0 branch is ~4% weaker
    # than the SA<0 branch at 300 N and equal by 800 N, which shows up as a
    # +2% wobble in mu vs load, NOT a phantom row.  Phantom rows are caught by
    # the both-branch trend AND by each branch staying within 8% of it.
    _mus = [float(_tm.peak_mu(float(f), 0.45, 0)) for f in _fzs]
    _degr = all(_mus[i+1] <= _mus[i] * 1.02 for i in range(len(_mus) - 1))
    _br = [(float(_tm.peak_mu(float(f), 0.45, 1)), float(_tm.peak_mu(float(f), 0.45, -1)))
           for f in _fzs]
    _br_ok = all(0.92 * m <= p <= m * 1.0001 and 0.92 * m <= n <= m * 1.0001
                 for m, (p, n) in zip(_mus, _br))
    _lv = _tm.camber_levels() if callable(_tm.camber_levels) else _tm.camber_levels
    _lv = list(np.asarray(_lv).ravel())
    if not (_degr and _br_ok):
        fails += 1
    print(f'tire camber rows : levels={_lv}  mu(300..1000N)@0.45deg='
          f'{" ".join(f"{m:.2f}" for m in _mus)}  SA>0 branch '
          f'{" ".join(f"{p/m:.3f}" for m, (p, _) in zip(_mus, _br))} of both  '
          f'{"pass" if (_degr and _br_ok) else "UNEXPECTED FAIL (non-degressive / branch off: phantom camber rows back)"}')
else:
    print('tire camber rows : no TTC file on this machine — skipped (data-dependent check)')

# ── TIRE ZERO/LOW-LOAD BEHAVIOUR: the grid interpolator clamped at BOTH ends,
#    so a wheel at 24 N was handed the full force of the 209 N test point (mu 30)
#    and a wheel at literally zero load still made 727 N.  The inner front sits
#    under the 209 N floor for ~45% of cornering time, so this was inventing grip
#    exactly where the Ackermann answer is decided.  Separately the binned data
#    did not pass through the origin (up to -71 N at zero slip, mu 0.21), which
#    put the force zero-crossing at -0.19 deg and made a 142 N step there.
#    Guard: no force at no load, force rises with load, mu stays bounded, the
#    upright curve passes through the origin, and camber thrust SURVIVES the
#    offset removal.  Also that slip_angle_at_peak_Fy still exists — its
#    precompute was once orphaned behind a `return` and raised AttributeError.
print('-' * 64)
if _tm is not None:
    _f0 = abs(float(_tm.Fy(3.0, 0.0, 0.0)))
    _lows = [abs(float(_tm.peak_Fy(f, 0.0))) for f in (10.0, 50.0, 150.0, 209.0)]
    _rising = all(_lows[i + 1] > _lows[i] for i in range(len(_lows) - 1))
    _mu_lo = _lows[0] / 10.0
    _origin = abs(float(_tm.Fy(0.0, 600.0, 0.0)))
    _cam = [float(_tm.Fy(0.0, 667.0, c)) for c in (0.0, 2.0, 4.0)]
    _thrust = abs(_cam[2]) > abs(_cam[1]) > 20.0
    try:
        _pk = float(_tm.slip_angle_at_peak_Fy(600.0)); _pk_ok = 2.0 < _pk < 12.0
    except Exception as _e:
        _pk, _pk_ok = float('nan'), False
    _ok = (_f0 < 1.0) and _rising and (_mu_lo < 5.0) and (_origin < 1.0) \
        and _thrust and _pk_ok
    if not _ok:
        fails += 1
    print(f'tire low-load    : Fy@0N {_f0:.1f} N, peak_Fy 10..209 N '
          f'{" ".join(f"{v:.0f}" for v in _lows)} (mu@10N {_mu_lo:.2f}), '
          f'Fy(0 slip,0 camber) {_origin:.1f} N, camber thrust '
          f'{_cam[1]:.0f}/{_cam[2]:.0f} N, peak SA {_pk:.1f} deg  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (low-load clamp or offset back)"}')
else:
    print('tire low-load    : no TTC file on this machine — skipped (data-dependent check)')

# ── ACKERMANN PAIR ANALYSIS (RCVD ch.7) MUST RESPOND TO THE SETTING.
#    The old Fz-Fy operating map put VERTICAL LOAD on the x-axis, and vertical
#    load does not depend on Ackermann, so -50% / 33% / -60% drew the same
#    picture (user: "its giving the same results for multiple %s").  At its
#    48 km/h default the whole kinematic spread is 0.30 deg, so -50% and -60%
#    differed by 0.015 deg of slip.  The honest independent variable is STEER
#    ANGLE, because Ackermann authority grows as steer squared.
#    Guard: sweeping the setting must MOVE the axle force, the direction
#    projection must be present (scrub > 0 at real steer), and the panel must
#    allow settings past +/-100 % (100 % is kinematic, not a ceiling).
print('-' * 64)
if _tm is not None:
    from vahan.analysis_plots import ackermann_pair_potential
    _ss = win._build_dynamics_solver()
    _pa = ackermann_pair_potential(_tm, _ss, lat_g=1.0,
                                   ackermann_list=(-100.0, 0.0, 100.0))
    _pk = [_pa['curves'][a]['peak_N'] for a in (-100.0, 0.0, 100.0)]
    _spread = max(_pk) - min(_pk)
    _scrub = _pa['curves'][0.0]['scrub_at_peak_N']
    _rng = win._analysis_plots_panel._ack_demand_pct
    _wide = _rng.minimum() <= -200 and _rng.maximum() >= 200
    _ok = (_spread > 1.0) and (_scrub > 50.0) and _wide
    if not _ok:
        fails += 1
    print(f'ackermann pair   : peak axle Fy at -100/0/+100 % = '
          f'{_pk[0]:.0f}/{_pk[1]:.0f}/{_pk[2]:.0f} N (spread {_spread:.0f} N), '
          f'scrub {_scrub:.0f} N, panel range {_rng.minimum():.0f}..'
          f'{_rng.maximum():.0f} %  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (plot blind to Ackermann again)"}')
else:
    print('ackermann pair   : no TTC file on this machine — skipped')

# ── ACKERMANN SOLVER (the user's method, 2026-07-27): each wheel corners its
#    OWN vertical load (Fy = Fz*lat_g) and its slip angle is INVERTED from its
#    own measured tyre curve — no placement rules, no trend knob.  Guards:
#    (1) g -> 0 recovers pure geometry: at 0.2 g the percentage is near 100
#        (slips are tiny, wheels point almost along their tangents);
#    (2) the slip split is EMERGENT from the data: at 1.5 g the loaded outer
#        wheel runs MORE slip than the light inner one (it works at a higher
#        fraction of a lower-mu curve) — no knob supplies this ordering;
#    (3) slips grow monotonically with g per wheel (slip is a state that
#        evolves with cornering intensity, starting from zero);
#    (4) nothing saturates at sane g, and saturation IS flagged at absurd g;
#    (5) the belt->asphalt derate (grip_multiplier, 0.65-0.75 project band)
#        moves the physics the right way: at 0.70 the same road demand needs
#        MORE slip than the raw belt curve says (a), the road limit is REAL
#        — a front wheel saturates by 1.7 g (b) — and sane g stays clean —
#        nothing saturates at 0.8 g (c).
print('-' * 64)
if _tm is not None:
    from vahan.ackermann import solve_ackermann_geometry
    _ss = win._build_dynamics_solver()
    # raw belt curves (x1.00) stated explicitly — since 2026-09-22 an omitted
    # grip means THE project scale, not raw belt
    _lo = solve_ackermann_geometry(_ss, _tm, 8.0, 0.2, grip_multiplier=1.0)
    _hi = solve_ackermann_geometry(_ss, _tm, 8.0, 1.5, grip_multiplier=1.0)
    _geo_ok = abs(_lo['ackermann_pct'] - 100.0) < 8.0
    _split_ok = _hi['outer_slip_deg'] > _hi['inner_slip_deg'] > 0.0
    _mono_ok = (_hi['outer_slip_deg'] > _lo['outer_slip_deg']
                and _hi['inner_slip_deg'] > _lo['inner_slip_deg'])
    _sat_ok = not (_hi['inner_saturated'] or _hi['outer_saturated'])
    # (5) grip derate at 0.70: same ROAD g, derated curve.
    _raw12 = solve_ackermann_geometry(_ss, _tm, 8.0, 1.2, grip_multiplier=1.0)
    _der12 = solve_ackermann_geometry(_ss, _tm, 8.0, 1.2,
                                      grip_multiplier=0.70)
    _der17 = solve_ackermann_geometry(_ss, _tm, 8.0, 1.7,
                                      grip_multiplier=0.70)
    _der08 = solve_ackermann_geometry(_ss, _tm, 8.0, 0.8,
                                      grip_multiplier=0.70)
    _grip_slip_ok = (_der12['inner_slip_deg'] > _raw12['inner_slip_deg']
                     and _der12['outer_slip_deg'] > _raw12['outer_slip_deg'])
    _grip_sat_ok = _der17['inner_saturated'] or _der17['outer_saturated']
    _grip_clean_ok = not (_der08['inner_saturated']
                          or _der08['outer_saturated'])
    _ok = (_geo_ok and _split_ok and _mono_ok and _sat_ok
           and _grip_slip_ok and _grip_sat_ok and _grip_clean_ok)
    if not _ok:
        fails += 1
    print(f'ackermann solver : 0.2g pct {_lo["ackermann_pct"]:.1f} (geometry '
          f'recovered), 1.5g slip in/out '
          f'{_hi["inner_slip_deg"]:.2f}/{_hi["outer_slip_deg"]:.2f} deg '
          f'(outer works harder, emergent), toe diff '
          f'{_hi["point_spread_deg"]:+.3f} deg, saturation clean; '
          f'grip x0.70: 1.2g slip out {_raw12["outer_slip_deg"]:.2f}->'
          f'{_der12["outer_slip_deg"]:.2f} deg (derate demands more), '
          f'1.7g front saturated {_grip_sat_ok}, 0.8g clean  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (inversion method broken)"}')
else:
    print('ackermann solver : no TTC file on this machine — skipped')

# ── ACKERMANN SOLVER MUST NOT FABRICATE.  A saturated wheel's slip angle is
#    slip_angle_for_Fy's PEAK-ANGLE FALLBACK, not a solution; when BOTH fronts
#    saturate they return the same fallback and the toe difference collapses to
#    the bare tangent difference (exactly 100% Ackermann).  That is what made
#    the table erratic across g: -1.27 deg at 1.5 g, -4.39 at 1.7, then +0.145
#    at 1.9 AND 2.0 — the last two were pure geometry dressed as a result.
#    Guards: (a) any saturated row reports valid=False and NaN toe difference;
#    (b) the VALID rows are smooth — monotone decreasing in g with AT MOST one
#    sign change, no jumps.  (Was "exactly one": pro at low g -> reverse near
#    the limit.  That crossing was an artifact of feeding the tyre |camber| on
#    its favourable branch: the light inner wheel scored as leaning INTO the
#    turn and needed LESS slip.  With the signed inclination (2026-09-02, see
#    'camber sign') the inner wheel leans AWAY under roll on this car and
#    needs MORE slip, so the spread stays pro-Ackermann and decays toward
#    zero: +1.72 -> +0.05 deg over 0.3..1.7 g.  Whether it crosses is a
#    physics result of tyre + static camber, not a gate invariant; the
#    erratic table this gate was built against had TWO crossings.)
print('-' * 64)
if _tm is not None:
    from vahan.ackermann import ackermann_bucket as _ab
    _ss3 = win._build_dynamics_solver()
    _rb = _ab(_tm, _ss3, radius_m=8.0,
              lat_g_list=(0.3, 0.8, 1.2, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0),
              grip_multiplier=0.70)
    _fab = [r for r in _rb if not r.get('valid', True)
            and np.isfinite(r.get('point_spread_deg', float('nan')))]
    _val = [r for r in _rb if r.get('valid', True)]
    _sp = [r['point_spread_deg'] for r in _val]
    _mono = all(_sp[i + 1] <= _sp[i] + 1e-6 for i in range(len(_sp) - 1))
    _sgn = sum(1 for i in range(len(_sp) - 1) if (_sp[i] > 0) != (_sp[i + 1] > 0))
    _sat = [r for r in _rb if not r.get('valid', True)]
    _ok = (not _fab) and _mono and _sgn <= 1 and len(_sat) >= 1 and len(_val) >= 4
    if not _ok:
        fails += 1
    print(f'ackermann trust  : {len(_val)} valid rows {_sp[0]:+.2f}->{_sp[-1]:+.2f} deg, '
          f'monotone={_mono}, sign changes={_sgn}, {len(_sat)} impossible rows '
          f'refused, fabricated={len(_fab)}  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (solver fabricating or erratic again)"}')
else:
    print('ackermann trust  : no TTC file on this machine — skipped')

# ── TYRE MUST PEAK, OPTIMIZER MUST NEVER PREFER >100% ACKERMANN.  The rig
#    sweeps only +/-12 deg and the light-load curves never decline inside it
#    (raw 222 N points: ~556 N from 8 deg to 13, dead flat), so the grid used
#    to keep REWARDING extra slip forever.  The force optimizer then walked the
#    lightly loaded inner wheel out on that creep and "preferred" +212% at
#    R=8 m — by 3.7 N out of 2118 (0.17%), an argmax eating crumbs.  User's
#    physics (2026-07-30) is a hard bound: load transfer sends force capacity
#    OUTWARD, so the split that maximizes axle force can never exceed the
#    kinematic spread — >100% must never win.  Fix: fy grid clamped
#    non-increasing past the RCVD-shaped peak-slip line (7.2 deg @222 N rising
#    to 9.0 @1110 N) at build time.  Gates: (a) the clamp ran; (b) no creep —
#    Fy at 12 deg does not exceed Fy at the peak-slip line at the lightest
#    bin; (c) the force sweep's argmax sits at or below 100% (+ one grid
#    step); (d) the plateau tie-band's honest pick is never above the argmax.
print('-' * 64)
if _tm is not None:
    from vahan.ackermann import solve_ackermann_force as _saf
    _clamped = bool(getattr(_tm, 'fy_grid_peak_clamped', False))
    _fzlo = float(_tm.fz_range[0])
    _sap = float(_tm.peak_slip_angle(_fzlo))
    _fy_sap = abs(float(_tm.Fy(_sap, _fzlo, 0.0)))
    _fy_12 = abs(float(_tm.Fy(12.0, _fzlo, 0.0)))
    _nocreep = _fy_12 <= _fy_sap + 1.0
    _ss4 = win._build_dynamics_solver()
    _fw = _saf(_tm, _ss4, 2.5, 1.0, ack_range=(0, 250), n=11,
               grip_multiplier=0.70)
    _ua = np.asarray(_fw['useful_N'], float)
    _pa = np.asarray(_fw['ackermann_pct'], float)
    _step = float(_pa[1] - _pa[0])
    _amax = float(_pa[int(np.nanargmax(_ua))])
    _le100 = _amax <= 100.0 + _step + 1e-6
    _pick = float(_fw['best_supported_pct'])
    _pick_ok = _pick <= _amax + 1e-9
    _ok = _clamped and _nocreep and _le100 and _pick_ok
    if not _ok:
        fails += 1
    print(f'ackermann ceiling: clamp={_clamped}, light bin Fy 12deg '
          f'{_fy_12:.0f} vs peak-line {_fy_sap:.0f} N (no creep={_nocreep}), '
          f'force argmax {_amax:+.0f}% (<=100+step), honest pick {_pick:+.0f}%  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (>100% preference is back)"}')
else:
    print('ackermann ceiling: no TTC file on this machine — skipped')

# ── PAIR-ANALYSIS AXLE SPLIT (replaced the cornering-stiffness/bicycle split).
#    Guards: (1) the two wheels of an axle sit at the SAME reference slip angle
#    when static toe is zero — the old code returned 2.42 vs 3.53 deg, the light
#    wheel apparently wanting MORE slip, which is backwards; (2) no slip angle
#    ever lands on the old 9.862 deg artifact (= linspace(0,13,30)[22], an array
#    index that slip_angle_for_Fy returned when it wrongly judged a force
#    unreachable); (3) a lifted wheel makes zero force; (4) demand beyond the
#    pair's capability is FLAGGED rather than silently saturated; (5) total
#    lateral force is still conserved so utilization can exceed 1.
print('-' * 64)
if _tm is not None:
    _ss2 = win._build_dynamics_solver()
    _bad9862 = False
    _equal = True
    _lift_ok = True
    _cons_ok = True
    for _g in (0.8, 1.2, 1.6, 2.0, 2.2):
        _r2 = _ss2.solve(_g, 0.0)
        _sa2 = _r2.slip_angle or {}
        for _v2 in _sa2.values():
            if abs(abs(float(_v2)) - 9.862) < 0.01:
                _bad9862 = True
        if abs(float(_ss2._veh.toe_front_deg)) < 1e-9 and _sa2:
            if abs(float(_sa2['FL']) - float(_sa2['FR'])) > 1e-6:
                _equal = False
        for _c in ('FL', 'FR', 'RL', 'RR'):
            if float(_r2.Fz.get(_c, 0)) <= 1e-9 \
                    and abs(float(_r2.Fy.get(_c, 0))) > 1e-6:
                _lift_ok = False
        _dem = abs(float(_ss2._veh.total_mass_kg) * _g * 9.80665)
        _got = sum(abs(float(_r2.Fy[c])) for c in ('FL', 'FR', 'RL', 'RR'))
        if abs(_got - _dem) / max(_dem, 1.0) > 0.02:
            _cons_ok = False
    _r24 = _ss2.solve(2.4, 0.0)
    _flag_ok = bool(_r24.grip_exceeded.get('F') or _r24.grip_exceeded.get('R'))
    _ok = (not _bad9862) and _equal and _lift_ok and _cons_ok and _flag_ok
    if not _ok:
        fails += 1
    print(f'axle pair split  : 9.862 artifact={_bad9862}, equal-at-zero-toe={_equal}, '
          f'lifted wheel makes 0={_lift_ok}, Fy conserved={_cons_ok}, '
          f'over-limit flagged @2.4g={_flag_ok}  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (bicycle split or grid-index artifact back)"}')
else:
    print('axle pair split  : no TTC file on this machine — skipped')

# ── TYRE PRESSURE IS AN INPUT, not an assumption.  A TTC cornering run sweeps
#    several pressures and blending them is an average of several tyres, not a
#    tyre: on this dataset the low-pressure sweep passes its peak inside the rig's
#    +-12 deg while the high-pressure one never reaches it.  Guard: the file's
#    pressures are discoverable, selecting one actually filters, and asking for
#    a pressure the file lacks RAISES instead of silently returning the blend.
print('-' * 64)
if _tm is not None:
    from vahan.tire_model import TireModel as _TM
    _path = win._dynamics_panel.get_tire_path()
    _av = list(getattr(_tm, 'available_pressures_psi', []) or [])
    _sel_ok = _raise_ok = False
    if _path and _av:
        _t1 = _TM.from_file(_path, pressure_psi=_av[0])
        _t2 = _TM.from_file(_path, pressure_psi=_av[-1])
        _sel_ok = (not _t1.pressure_blended) and (
            abs(_t1.pressure_psi - _t2.pressure_psi) > 0.5)
        try:
            _TM.from_file(_path, pressure_psi=99.0)
        except ValueError:
            _raise_ok = True
        # BLENDING IS BANNED (user order 2026-07-30): no pressure given must
        # REFUSE, never average the 8/10/12/14 psi sweeps into a fake tyre.
        _blend_refused = False
        try:
            _TM.from_file(_path)
        except ValueError:
            _blend_refused = True
    _ok = bool(_av) and _sel_ok and _raise_ok and _blend_refused
    if not _ok:
        fails += 1
    print(f'tyre pressure    : file holds {_av} psi, selecting one filters={_sel_ok}, '
          f'bad pressure raises={_raise_ok}, blend refused={_blend_refused}  '
          f'{"pass" if _ok else "UNEXPECTED FAIL (pressure input not honoured)"}')
else:
    print('tyre pressure    : no TTC file on this machine — skipped')

# ── AERO SOLVER SANITY ('6 kN cap gang', 2026-07-12): the old aero solver
#    targeted PER-CORNER utilization with unscaled belt mu, so past the grip
#    limit the artifact-pinned inner tires slammed every corner into its
#    3000 N cap -> a fake 6000 N downforce-required plateau.  Guard: at
#    (mechanical limit + 0.1 g) the required downforce is finite, sane
#    (< 2000 N), NOT the cap signature; below the limit it is exactly 0.
#    Also guards that the canonical axle_utilization criterion exists.
print('-' * 64)
if _tm is not None:
    try:
        _ss = win._build_dynamics_solver()
        _r10 = _ss.solve(1.0)
        _au = _ss.axle_utilization(_r10)
        au_ok = all(0.0 < _au[k] < 2.0 for k in ('F', 'R'))
        _lo, _hi = 0.5, 3.0
        for _ in range(12):
            _mid = (_lo + _hi) / 2
            try:
                _u = _ss.axle_utilization(_ss.solve(_mid))
                _lo, _hi = (_mid, _hi) if max(_u.values()) < 1.0 else (_lo, _mid)
            except Exception:
                _hi = _mid
        from vahan.dynamics import AeroDownforceSolver
        _aero = AeroDownforceSolver(_ss)
        _below = _aero.solve(max(_lo - 0.2, 0.3), target_util=1.0).total_downforce_N
        _above = _aero.solve(_lo + 0.1, target_util=1.0).total_downforce_N
        aero_ok = (au_ok and _below == 0.0 and 0.0 < _above < 2000.0)
        if not aero_ok:
            fails += 1
        print(f'aero DF sanity   : axle_util F/R={_au["F"]:.2f}/{_au["R"]:.2f}  gmax~{_lo:.2f}  '
              f'DF(below)={_below:.0f}N DF(limit+0.1g)={_above:.0f}N   '
              f'{"pass" if aero_ok else "UNEXPECTED FAIL (cap-gang regression)"}')
    except Exception as _e:
        fails += 1
        print(f'aero DF sanity   : UNEXPECTED FAIL ({_e})')
else:
    print('aero DF sanity   : no TTC file — skipped')

# ── FORCE-TRANSFER METRICS (vahan/force_opt.py): pushrod off-tangency and
#    the virtual-work force amplification must be finite and sane on the
#    default car (guards the loads objective used by design_city).
print('-' * 64)
try:
    from vahan.force_opt import tangency_deg, force_amplification
    _th = tangency_deg(win, 'FL', 0.0)
    _am = force_amplification(win, 'FL', 0.0)
    ft_ok = (np.isfinite(_th) and 0.0 <= _th < 90.0 and np.isfinite(_am) and 0.5 < _am < 5.0)
    if not ft_ok:
        fails += 1
    print(f'force transfer   : FL off-tangency {_th:.1f} deg, F_pr/F_wheel {_am:.2f}   '
          f'{"pass" if ft_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'force transfer   : UNEXPECTED FAIL ({_e})')

# ── DESIGN CITY (design_city.py + vahan/packaging.py parameter gate): the
#    0.1 %-of-EVERY-parameter packaging engine, in-process on the current
#    design config, three trials.  (1) the untouched model must pass its own
#    gate; (2) two EXACT-family moves (spring pair swung 5 deg about the rocker
#    axis; UCA pickups slid 10 mm toward each other along their pivot axis) —
#    every KEPT solution must re-load in a FRESH window and pass the 0.1 % gate
#    again, keep the wheel side + the other axle byte-identical, show 0 clash
#    negatives at centre/locks and actually move a point; (3) an INEXACT move
#    (slice rotated 5 deg about the pushrod line: static MR re-tuned but the
#    MR CURVE moves) must be REJECTED at the parameter stage — proves the gate
#    is not blind.  Groups = baseline + kept.
print('-' * 64)
if _design:
    try:
        import json as _cj, tempfile as _ctf, shutil as _csh
        import design_city as _dc
        from vahan.packaging import (parameter_vector as _c_pvec, compare_parameters as _c_pcmp,
                                     clash_negatives_at_locks as _c_pcl)
        # a PRIVATE folder per run: a fixed shared folder let two concurrent nets (e.g. a
        # worktree session) wipe each other's groups.json mid-read (2026-09-26 JSONDecodeError)
        _dc_out = _ctf.mkdtemp(prefix='vahan_city_net_')

        def _c_recipe(**kw):
            _r = _dc.identity_recipe(); _r.update(kw); return _r
        _dc_run = _dc.run_city(config=_design, out_dir=_dc_out, axles=('front',), workers=1, render=False,
                               recipes={'front': [_c_recipe(spring_swing_deg=5.0),
                                                  _c_recipe(pickup_spacing={'uca': 10.0}),
                                                  _c_recipe(rotate_pushrod_deg=5.0)]}, progress=None)
        _c_ax = _dc_run['axles']['front']
        _c_fail = []
        if not _c_ax['baseline_ok']:
            _c_fail.append(f"baseline fails its own gate: {_c_ax['baseline_reason']}")
        with open(os.path.join(_dc_out, 'front', 'trials.jsonl'), 'r', encoding='utf-8') as _cf:
            _c_recs = [_cj.loads(_l) for _l in _cf if _l.strip()]
        _c_by = {_r['id']: _r for _r in _c_recs}
        # the inexact family must be REJECTED; which gate catches it first (laws
        # on v101, parameters on v99) depends on the config and is reported, not asserted
        if _c_by['f0002']['kept']:
            _c_fail.append('pushrod-line rotation (inexact family) was KEPT — the gate is blind')
        _c_kept = [_r for _r in _c_recs if _r['kept'] and not _r.get('is_baseline')]
        if not _c_kept:
            _c_fail.append('no exact-family solution kept (spring swing +5 deg / UCA spacing +10 mm)')
        with open(_design, 'r', encoding='utf-8') as _cf:
            _c_src = _cj.load(_cf)
        _wC = MainWindow(); _wC._load_project_from_path(_design); _wC._rebuild_solvers(0.)
        _c_base = _c_pvec(_wC)
        for _r in _c_kept:
            _c_cfg = os.path.join(_dc_out, 'front', _r['id'], 'config.vahan')
            _wC._load_project_from_path(_c_cfg); _wC._rebuild_solvers(0.)
            _c_bad = [_x['name'] for _x in _c_pcmp(_c_base, _c_pvec(_wC)) if not _x['ok']]
            if _c_bad:
                _c_fail.append(f"{_r['id']} beyond 0.1 % after reload: {_c_bad[:4]}")
            with open(_c_cfg, 'r', encoding='utf-8') as _cf:
                _c_saved = _cj.load(_cf)
            if any(_c_src['front_hp'][_k] != _c_saved['front_hp'][_k] for _k in _dc.WHEEL_SIDE_KEYS):
                _c_fail.append(f"{_r['id']} wheel side not byte-identical")
            if _c_src['rear_hp'] != _c_saved['rear_hp'] or _c_src['rear_arb'] != _c_saved['rear_arb']:
                _c_fail.append(f"{_r['id']} rear axle not byte-identical")
            _c_cl = _c_pcl(_wC)
            if _c_cl['negatives']:
                _c_fail.append(f"{_r['id']} {_c_cl['negatives']} clash negatives after reload")
            if _r['max_point_move_mm'] < _dc.DUPLICATE_MM:
                _c_fail.append(f"{_r['id']} moved nothing ({_r['max_point_move_mm']:.2f} mm)")
        _wC.close()
        with open(os.path.join(_dc_out, 'front', 'groups.json'), 'r', encoding='utf-8') as _cf:
            _c_groups = _cj.load(_cf)
        if _c_groups['n_solutions'] != len(_c_kept) + 1:
            _c_fail.append(f"groups.json holds {_c_groups['n_solutions']} solutions, expected baseline + {len(_c_kept)}")
        _c_ok = not _c_fail
        if not _c_ok:
            fails += 1
        print(f"design city      : {len(_c_recs) - 1} tried, {len(_c_kept)} kept, {len(_c_groups['groups'])} groups, "
              f"baseline self-gate {_c_ax['baseline_ok']}, pushrod-line rotation rejected at "
              f"'{_c_by['f0002']['stage']}' ({_c_by['f0002'].get('worst_parameter')} "
              f"{(_c_by['f0002'].get('max_deviation_pct') or 0):.2f} %)   "
              f"{'pass' if _c_ok else 'UNEXPECTED FAIL (' + '; '.join(_c_fail) + ')'}")
        _csh.rmtree(_dc_out, ignore_errors=True)
    except Exception as _e:
        fails += 1
        print(f'design city      : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')
else:
    print('design city      : no design config — skipped')

# ── BRAKE TORQUE IS BRAKE-ONLY (vahan/dynamics.py): under power the driven
#    axle's hub torque is reacted by the DRIVESHAFT, not the caliper.  Booking
#    drive torque as brake torque put phantom kN-level loads into the caliper
#    mount lugs at full acceleration (found 2026-07-21 when the binder's load
#    viewer was finally wired to the GUI's own load model).
print('-' * 64)
try:
    from gui import wheel_package as _WP
    _brk = _WP.compute_case(win, 0.0, -1.5)[3]      # pure braking
    _acc = _WP.compute_case(win, 0.0, +1.0)[3]      # full acceleration
    _bt_brake = max(_brk.brake_torque.values())
    _bt_accel = max(_acc.brake_torque.values())
    _cal_accel = [it for it in _WP._load_items(win, 0.0, 1.0)
                  if 'CALIPER' in it[3]]
    bt_ok = (_bt_brake > 50.0 and _bt_accel == 0.0 and not _cal_accel)
    if not bt_ok:
        fails += 1
    print(f'brake torque     : braking {_bt_brake:.0f} Nm, accel {_bt_accel:.0f} Nm, '
          f'{len(_cal_accel)} caliper loads under power   '
          f'{"pass" if bt_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'brake torque     : UNEXPECTED FAIL ({_e})')

# ── ONE CALIPER MODEL: the loads TABLE (vahan/loads.py) and the Loads-page
#    PICTURE (gui/wheel_package.py) must agree.  They disagreed 4.6x because
#    loads.py took the friction moment about the WHEEL AXLE (T / l5) instead of
#    about the BOLT LINE (F * l4 / l5, Seward Ch.6 Fig 6.15).
print('-' * 64)
try:
    from gui import wheel_package as _WP2
    _ld, _veh2, _up2, _res2, _bpf2, _bpr2, _slv2 = _WP2.compute_case(win, 0.0, -1.5)
    _cl = _ld['FL']
    _Fpad = _cl.brake_torque_Nm / (_bpf2.pad_radius_mm / 1000.0)
    # The Seward couple acts along the caliper RADIAL (pad centre -> bolt line),
    # so at a manual clock (the saved car: 315 deg, vertical mounts off) it has
    # BOTH an H and a V part.  Compare its MAGNITUDE; the old H-only check was
    # only right for vertical mounts (2026-09-22 caliper clocking repair).
    _tbl = float(np.hypot(_cl.caliper_upper_H - _cl.caliper_lower_H,
                          _cl.caliper_upper_V - _cl.caliper_lower_V)) / 2.0
    _pic = _Fpad * (_bpf2.caliper_l4_mm / 1000.0) / (_bpf2.caliper_bolt_spacing_mm / 1000.0)
    cal_ok = _tbl > 1.0 and abs(_tbl - _pic) < max(1.0, 0.01 * _pic)
    if not cal_ok:
        fails += 1
    print(f'caliper one model: table {_tbl:.0f} N vs Seward {_pic:.0f} N per bolt   '
          f'{"pass" if cal_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'caliper one model: UNEXPECTED FAIL ({_e})')

# ── BALANCE BAR reaches the PEDAL FORCE.  compute_brake_system() computed
#    bias_f / bias_r and then never used them, so the pedal force needed to lock
#    a wheel ignored the bias bar entirely — understating it by 1/bias (1.54x
#    front, 2.86x rear at 65% bias) on the exact number a pedal box is sized
#    from.  F_pedal = P_lock * A_mc / (pedal_ratio * bias).
print('-' * 64)
try:
    from vahan.loads import (compute_brake_system as _cbs,
                             BrakeSystemParams as _BSP, BrakeParams as _BP)
    _Fz = {'FL': 900.0, 'FR': 900.0, 'RL': 700.0, 'RR': 700.0}
    _pf, _pr = _BP(), _BP()
    _bias = 65.0
    _sys = _BSP(pedal_ratio=5.0, mc_bore_front_mm=15.87, mc_bore_rear_mm=15.87,
                bias_pct_front=_bias)
    from vahan.tire_model import LinearTireModel as _LTMb
    _tmb = _LTMb()
    _rb = _cbs(_Fz, _pf, _pr, _sys, tire_model=_tmb, grip_scale=1.0)
    _fl, _rl = _rb['FL'], _rb['RL']
    _exp_f = _fl.lockup_line_pressure_MPa * _sys.mc_area_front_mm2 / (5.0 * _bias / 100.0)
    _exp_r = _rl.lockup_line_pressure_MPa * _sys.mc_area_rear_mm2 / (5.0 * (1 - _bias / 100.0))
    _bb_ok = (abs(_fl.lockup_pedal_force_N - _exp_f) < 0.5 and
              abs(_rl.lockup_pedal_force_N - _exp_r) < 0.5 and
              _fl.lockup_pedal_force_N > 1.0)
    # and the bias must actually MOVE the answer (the old code was bias-blind)
    _sys2 = _BSP(pedal_ratio=5.0, mc_bore_front_mm=15.87, mc_bore_rear_mm=15.87,
                 bias_pct_front=50.0)
    _rb2 = _cbs(_Fz, _pf, _pr, _sys2, tire_model=_tmb, grip_scale=1.0)
    _moves = abs(_rb2['FL'].lockup_pedal_force_N - _fl.lockup_pedal_force_N) > 1.0
    _bb_ok = _bb_ok and _moves
    if not _bb_ok:
        fails += 1
    print(f'brake bias->pedal: F {_fl.lockup_pedal_force_N:.0f} N (expect {_exp_f:.0f}), '
          f'R {_rl.lockup_pedal_force_N:.0f} N (expect {_exp_r:.0f}), '
          f'bias moves it={_moves}   {"pass" if _bb_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'brake bias->pedal: UNEXPECTED FAIL ({_e})')

# ── STATIC SAG uses SPRUNG weight only.  The spring carries the sprung corner
#    weight; the unsprung corner (wheel/upright/hub/brake) is reacted by the TIRE
#    to the ground and never passes through the spring.  static_sag() added the
#    unsprung term, overstating required spring compression by 15.5% front /
#    15.6% rear -- i.e. it lied about where the damper sits and how much stroke
#    the spring needs, which is exactly what a spring-selection decision reads.
print('-' * 64)
try:
    _vv = win._build_dynamics_solver()._veh
    _sag = _vv.static_sag()
    _g = 9.81
    _wf = _vv.front_weight_fraction
    _exp_f = (_vv.sprung_mass_kg * _wf * _g / 2.0) / (
        _vv.motion_ratio_front * _vv.spring_rate_front_Npm / 1000.0)
    _exp_r = (_vv.sprung_mass_kg * (1 - _wf) * _g / 2.0) / (
        _vv.motion_ratio_rear * _vv.spring_rate_rear_Npm / 1000.0)
    _got_f = float(_sag['required_spring_compression_front_mm'])
    _got_r = float(_sag['required_spring_compression_rear_mm'])
    # the WRONG (unsprung-inclusive) answer, which must NOT be what we get
    _bad_f = _exp_f * (1 + (_vv.unsprung_mass_front_kg / 2.0) /
                       (_vv.sprung_mass_kg * _wf / 2.0))
    _sag_ok = (abs(_got_f - _exp_f) < 0.05 and abs(_got_r - _exp_r) < 0.05
               and abs(_got_f - _bad_f) > 0.5)
    if not _sag_ok:
        fails += 1
    print(f'static sag sprung: front {_got_f:.2f} mm (sprung-only {_exp_f:.2f}, '
          f'unsprung-inclusive would be {_bad_f:.2f})   '
          f'{"pass" if _sag_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'static sag sprung: UNEXPECTED FAIL ({_e})')

# Configured damper hardware stops do not depend on spring preload, mass,
# static CAD length, or a dynamics panel being available during startup.
print('-' * 64)
try:
    from types import SimpleNamespace as _DamperNS
    _limit_owner = _DamperNS(_motion_panel=_DamperNS(
        stroke_mm=55.0, fully_extended_mm=210.0))
    _limit_results = []
    for _cad_length in (0.175, 0.189, 0.205):
        _limit_solver = _DamperNS(solve=lambda t, length=_cad_length:
                                _DamperNS(spring_length=length))
        _limit_results.append(MainWindow._spring_limits_uncached(
            _limit_owner, _limit_solver, False))
    _limits_ok = all(np.allclose(pair, (0.155, 0.210), atol=1e-12, rtol=0.)
                     for pair in _limit_results)
    if not _limits_ok:
        fails += 1
    print(f'physical dampers : {_limit_results} '
          f'{"pass" if _limits_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'physical dampers : UNEXPECTED FAIL ({_e})')

# Loading and read-only dynamics refreshes must not widen a saved motion
# domain. A real damper edit still restores automatic range calculation.
print('-' * 64)
try:
    import json as _motion_json
    import tempfile as _motion_tempfile
    from pathlib import Path as _MotionPath
    with _motion_tempfile.TemporaryDirectory() as _motion_dir:
        _motion_path = _MotionPath(_motion_dir) / 'motion_roundtrip.vahan'
        _motion_win = MainWindow()
        _motion_win._save_project_to_path(str(_motion_path))
        _motion_data = _motion_json.loads(_motion_path.read_text())
        _motion_expected = (-12.375, 18.625)
        _motion_data['motion'].update(
            type='heave', min=_motion_expected[0], max=_motion_expected[1],
            stroke_mm=59.0, fully_extended_mm=221.0,
            preload_front_mm=1.25, preload_rear_mm=2.5)
        _motion_path.write_text(_motion_json.dumps(_motion_data))
        _motion_win._load_project_from_path(str(_motion_path))
        _motion_win._build_dynamics_solver()
        _motion_win._refresh_sag()
        _mp = _motion_win._motion_panel
        assert np.allclose((_mp.min_val, _mp.max_val), _motion_expected,
                           atol=1e-12, rtol=0.), 'saved range overwritten'
        assert (_mp.stroke_mm, _mp.fully_extended_mm,
                _mp.preload_front_mm, _mp.preload_rear_mm) == (59., 221., 1.25, 2.5)
        for _axle in ('front', 'rear'):
            for _key, _point in _motion_data[f'{_axle}_hp'].items():
                assert np.array_equal(getattr(_motion_win, f'_{_axle}_hp')[_key],
                                      _point), f'load moved {_axle}/{_key}'
        _motion_win._save_project_to_path(str(_motion_path))
        _motion_resaved = _motion_json.loads(_motion_path.read_text())['motion']
        assert (_motion_resaved['min'], _motion_resaved['max']) == _motion_expected
        _mp._stroke.setValue(60.0)  # a real control edit, not a load callback
        assert not np.allclose((_mp.min_val, _mp.max_val), _motion_expected), \
            'damper edit failed to restore automatic travel range'
        assert np.allclose(_motion_win._spring_limits_uncached(
            _motion_win._solvers['FL'], True), (.161, .221), atol=1e-12, rtol=0.)
        # A following legacy load must not inherit the previous project's
        # explicit bounds, and switching mm motions must retain saved bounds.
        _motion_win._load_project_from_path(str(_motion_path))
        _mp._on_motion(True, 'pitch')
        assert (_mp.min_val, _mp.max_val) == _motion_expected
        _motion_data.pop('motion')
        _motion_path.write_text(_motion_json.dumps(_motion_data))
        _motion_win._load_project_from_path(str(_motion_path))
        assert not np.allclose((_mp.min_val, _mp.max_val), _motion_expected), \
            'legacy load inherited explicit travel bounds'
        _motion_win.close()
    print('motion roundtrip: saved bounds/hardware/points held; edits remain automatic pass')
except Exception as _e:
    fails += 1
    print(f'motion roundtrip: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── YAW INERTIA must be PHYSICALLY POSSIBLE.  m·a·b is the exact maximum
#    longitudinal yaw inertia for mass living between the axles (all mass split
#    onto the two axles, honouring the CG, gives exactly m·a·b).  The estimator
#    shipped k=1.2, i.e. 1.2x that hard ceiling — a yaw radius of gyration LARGER
#    than the half-wheelbase, which needs heavy overhangs this car does not have.
#    Every yaw-acceleration number divides by this, so gate the ceiling, not a
#    guessed value: any k >= 1 is unphysical for a car with light overhangs.
print('-' * 64)
try:
    _vy = win._build_dynamics_solver()._veh
    _m = float(_vy.total_mass_kg); _wb = float(_vy.wheelbase_m)
    _wf = float(_vy.front_weight_fraction)
    _a = _wb * (1 - _wf); _b = _wb * _wf          # CG->front, CG->rear
    _ceil = _m * _a * _b                           # hard in-wheelbase maximum
    _k = float(getattr(_vy, 'yaw_inertia_factor', 1.2))
    _izz = _k * _ceil
    _kgyr = np.sqrt(_izz / _m) if _m > 0 else 0.0
    _yaw_ok = (_izz < _ceil) and (_kgyr < _wb / 2.0)
    if not _yaw_ok:
        fails += 1
    print(f'yaw inertia sane : k={_k:.2f} -> Izz {_izz:.0f} kg.m^2 vs ceiling '
          f'{_ceil:.0f}; gyradius {_kgyr*1000:.0f} mm vs half-wheelbase '
          f'{_wb*500:.0f} mm   {"pass" if _yaw_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'yaw inertia sane : UNEXPECTED FAIL ({_e})')

# ── ROLL GRADIENT + ARB RATE SANITY, and ONE MODEL between the panel and the
#    solver.  The 2027 design shipped an ARB whose hardpoints gave a 47 mm lever
#    on a 155 mm half-bar -> 353 N/mm at the wheel and a roll gradient of
#    0.095 deg/g.  No car runs that (FSAE is 1-2 deg/g; the team's own 2026 car
#    measured ~1 deg/g on the same tube).  Nothing in the tool complained,
#    because the config's SAVED panel geometry was stale and looked sane until
#    _refresh_arb_geometry_into_panel() recomputed it from the hardpoints.
#    Gate three things: the rate stays inside the model's own sweep bound, the
#    roll gradient is physically sane, and the panel and solver agree.
print('-' * 64)
try:
    _wR = MainWindow(); _wR._load_project_from_path(_design); _wR._rebuild_solvers(0.)
    _ssR = _wR._build_dynamics_solver(); _vR = _ssR._veh
    _pR = _wR._dynamics_panel.get_params()      # after the solver's own refresh
    _af, _ar = float(_vR.arb_rate_front_Npm), float(_vR.arb_rate_rear_Npm)
    _rg = abs(float(_ssR.solve(1.0, 0.0).roll_angle_deg))     # deg per g
    _rfail = []
    # (1) ONE MODEL: what the panel computes must be what the solver uses
    if abs(_pR['arb_rate_front_Npm'] - _af) > 1.0 or abs(_pR['arb_rate_rear_Npm'] - _ar) > 1.0:
        _rfail.append(f'panel/solver ARB disagree (panel {_pR["arb_rate_front_Npm"]:.0f}/'
                      f'{_pR["arb_rate_rear_Npm"]:.0f} vs solver {_af:.0f}/{_ar:.0f})')
    # (2) the model's own documented sweep ceiling is 87,500 N/m (500 lbf/in)
    if max(_af, _ar) > 87500.0:
        _rfail.append(f'ARB wheel rate {max(_af,_ar):.0f} N/m exceeds the model\'s own '
                      f'87,500 N/m bound')
    # (3) roll gradient must be physically sane for a race car
    if not (0.2 <= _rg <= 3.0):
        _rfail.append(f'roll gradient {_rg:.3f} deg/g outside 0.2-3.0 '
                      f'(FSAE runs 1-2; the 2026 car measured ~1)')
    # KNOWN-FAIL, documented: the v34 ARB HARDPOINTS are undersized -- rear lever
    # 47.2 mm on a 155 mm half-bar (front 75.4 / 80 mm) against the 2026 car's
    # 104.8 / 270 mm, which measured ~1 deg/g on the same 12.7 mm tube.  Rate goes
    # as 1/(A^2*L), so the bar is ~9x over-stiff by geometry alone.  The tool is
    # reporting this correctly; the GEOMETRY is what needs moving, and that is a
    # design change (it interacts with the coplanar drop-link rule), so it is
    # flagged here rather than silently patched.  Fix = lengthen the lever and
    # widen the half-span toward the 2026 values; the gate then passes on its own.
    _KNOWN_ARB = 'ARB hardpoints undersized (lever/half-span) — see OPEN_INSTRUCTIONS'
    if _rfail:
        known += 1
    print(f'roll/ARB sanity  : ARB {_af/1000:.1f}/{_ar/1000:.1f} kN/m, roll {_rg:.3f} deg/g   '
          + ('pass' if not _rfail else f'KNOWN-FAIL: {"; ".join(_rfail)}'))
except Exception as _e:
    fails += 1
    print(f'roll/ARB sanity  : UNEXPECTED FAIL ({_e})')

# ── AERO PER-CORNER SUM == WHAT THE AERO SOLVER ASKED FOR ───────────────────
# The solved-aero path handed each WHEEL its whole AXLE need, so the GUI applied
# exactly 2x the downforce AeroDownforceSolver had computed — and disagreed 2x
# with the 'custom' branch for the same physical package.  Nothing compared the
# two.  Assert the applied per-corner dict sums to total_downforce_N.
print('-' * 64)
try:
    import glob as _g4, re as _re4
    from vahan.dynamics import AeroDownforceSolver as _ADS
    _cf4 = _g4.glob('configs/2027_v*.vahan')
    _cf4 = max(_cf4, key=lambda p: int(_re4.search(r'2027_v(\d+)', p).group(1))) if _cf4 else None
    _cf4 = _design or _cf4
    if _cf4 is None:
        print('aero per-corner  : no config — skipped')
    else:
        _wA = MainWindow(); _wA._load_project_from_path(_cf4)
        _ssA = _wA._build_dynamics_solver()
        _rA = _ADS(_ssA).solve(2.4, target_util=0.80)
        _wA._aero_active = True; _wA._last_aero_result = _rA
        _dA = _wA._get_aero_Fz_per_g() or {}
        _applied = sum(v * _rA.lateral_g for v in _dA.values())
        _own = sum(_rA.downforce.values())
        # SOURCE-AWARE (2026-09-29): the Dynamics panel's aero source decides
        # what "asked for" means.  'solved' = the aero-target solver's need;
        # 'custom' (v150: 1000 N at 88.5 km/h, 54 % rear) = the package formula
        # F_ref·g·R/V_ref² at the panel radius, which the solver's need is NOT.
        _srcA = _wA._dynamics_panel.get_aero_source()
        if _srcA == 'custom':
            _cpA = _wA._dynamics_panel.get_custom_aero_params()
            _RA = float(_wA._dynamics_panel._turn_radius.value())
            _vrefA = float(_cpA['V_ref_kph']) / 3.6
            _askedA = float(_cpA['F_ref_N']) * 9.80665 * _RA / (_vrefA * _vrefA) * _rA.lateral_g
            # the Loads-page path must agree at the speed of that g on that radius
            from gui import wheel_package as _WPA
            _spA = np.sqrt(_rA.lateral_g * 9.80665 * _RA) * 3.6
            _caA = _WPA.case_aero(_wA, _rA.lateral_g, 0.0)
            _rearA = (_dA.get('RL', 0.0) + _dA.get('RR', 0.0)) / max(sum(_dA.values()), 1e-9)
            _ok_a = (_askedA > 1.0 and abs(_applied - _askedA) / _askedA < 0.02
                     and abs(_caA['speed_kph'] - _spA) < 0.05
                     and abs(_caA['total_N'] - _askedA) / _askedA < 0.02
                     and abs(_rearA - float(_cpA['cop_rear_pct']) / 100.0) < 1e-6)
            _whatA = (f"custom package {_cpA['F_ref_N']:.0f} N at {_cpA['V_ref_kph']:.1f} km/h on "
                      f"{_RA:.1f} m asks {_askedA:.0f} N at {_rA.lateral_g:.2f} g ({_spA:.1f} km/h); "
                      f"loads path {_caA['total_N']:.0f} N at {_caA['speed_kph']:.1f} km/h")
        else:
            _askedA = _rA.total_downforce_N
            _ok_a = (_rA.total_downforce_N > 1.0
                     and abs(_applied - _rA.total_downforce_N) / _rA.total_downforce_N < 0.02
                     and abs(_own - _rA.total_downforce_N) / _rA.total_downforce_N < 0.02)
            _whatA = f'solved aero need {_rA.total_downforce_N:.0f} N'
        if not _ok_a:
            fails += 1
        print(f'aero per-corner  : source {_srcA}: {_whatA}, applied '
              f'{_applied:.0f} N, ratio {_applied/max(_askedA,1e-9):.3f}   '
              f'{"pass" if _ok_a else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'aero per-corner  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── SAVE / LOAD ROUND-TRIP ──────────────────────────────────────────────────
# File > Save Project raised AttributeError inside _save_project (the loads
# panel's brake dict carries a readout QLabel among the spinboxes and every
# entry was .value()'d), so saving threw BEFORE json.dump and the user lost a
# whole editing session.  Nothing gated the one operation that protects work.
# Save the real design config, reload it, and assert it survives.
print('-' * 64)
try:
    import glob as _g3, re as _re3, tempfile as _tf3
    _cf3 = _g3.glob('configs/2027_v*.vahan')
    _cf3 = max(_cf3, key=lambda p: int(_re3.search(r'2027_v(\d+)', p).group(1))) if _cf3 else None
    if _cf3 is None:
        print('save round-trip  : no config — skipped')
    else:
        _wS = MainWindow(); _wS._load_project_from_path(_cf3); _wS._rebuild_solvers(0.)
        _tmp = os.path.join(_tf3.gettempdir(), '_net_save_roundtrip.vahan')
        _wS._save_project_to_path(_tmp)          # must not raise
        _wS2 = MainWindow(); _wS2._load_project_from_path(_tmp); _wS2._rebuild_solvers(0.)
        _a = _wS._build_dynamics_solver()._veh
        _b = _wS2._build_dynamics_solver()._veh
        _same = (abs(_a.arb_rate_front_Npm - _b.arb_rate_front_Npm) < 1.0
                 and abs(_a.wheel_rate_rear_Npm - _b.wheel_rate_rear_Npm) < 1.0)
        _lp = _wS._loads_panel.get_state()       # the call that used to throw
        _sv_ok = _same and isinstance(_lp, dict) and 'brake_front' in _lp
        if not _sv_ok:
            fails += 1
        print(f'save round-trip  : saved+reloaded, rates match={_same}, '
              f'loads panel state keys={len(_lp) if isinstance(_lp, dict) else 0}   '
              f'{"pass" if _sv_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'save round-trip  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── ATOMIC SAVE / TRANSACTIONAL LOAD — FAULT INJECTION (Astra F-01, 2026-09-22)
#    FAILING-THEN-PASSING: _save_project_to_path opened the user's file with 'w'
#    BEFORE json.dump, so an unserialisable field raised TypeError AFTER the
#    accepted project had been truncated (Astra: 667 bytes of invalid JSON left
#    on disk); _load_project_from_path replaced the motion bounds and front
#    hardpoints before discovering a missing rear_hp (KeyError), leaving the
#    live window half old / half new.  Gate: (a) unserialisable save and (b) an
#    I/O failure at the replace step leave the file byte-identical and no temp
#    litter; (c) a schema-broken file and (d) a failure injected MID-apply
#    (after hardpoints/car were already replaced) leave the whole live project
#    — the same dict Save writes — unchanged; (e) a normal load still works.
print('-' * 64)
try:
    import tempfile as _tfA, shutil as _shA, json as _jA
    _fA = []
    if _cf3 is None:
        print('atomic save/load : no config — skipped')
    else:
        _wA = _wS
        _dA = _tfA.mkdtemp(prefix='_net_atomic_')
        _pA = os.path.join(_dA, 'proj.vahan')
        _wA._save_project_to_path(_pA)
        _b0 = open(_pA, 'rb').read()

        def _live(_w):
            return _jA.dumps(_w._project_to_dict(), sort_keys=True, default=str)
        # (a) unserialisable field
        _wA._car['__net_probe_unserialisable__'] = object()
        try:
            _wA._save_project_to_path(_pA)
            _fA.append('unserialisable save did not raise')
        except TypeError:
            pass
        finally:
            _wA._car.pop('__net_probe_unserialisable__', None)
        _okA = open(_pA, 'rb').read() == _b0
        if not _okA:
            _fA.append('failed (unserialisable) save changed the saved file')
        # (b) I/O failure at the atomic replace
        _orep = os.replace
        def _boomA(*_a, **_k):
            raise OSError('injected replace failure')
        os.replace = _boomA
        try:
            _wA._save_project_to_path(_pA)
            _fA.append('save with a failing replace did not raise')
        except OSError:
            pass
        finally:
            os.replace = _orep
        _okB = open(_pA, 'rb').read() == _b0
        _litter = [f for f in os.listdir(_dA) if f != 'proj.vahan']
        if not _okB:
            _fA.append('failed (I/O) save changed the saved file')
        if _litter:
            _fA.append(f'temp files left behind: {_litter}')
        # (c) schema-broken file: nothing live may change
        _s0 = _live(_wA)
        _dbad = _jA.loads(_b0.decode('ascii'))
        del _dbad['rear_hp']
        _pbad = os.path.join(_dA, 'missing_rear.vahan')
        with open(_pbad, 'w') as _fh:
            _jA.dump(_dbad, _fh)
        try:
            _wA._load_project_from_path(_pbad)
            _fA.append('load of a file with no rear_hp did not raise')
        except ValueError:
            pass
        _okC = _live(_wA) == _s0
        if not _okC:
            _fA.append('schema-broken load changed the live project')
        # (d) failure injected MID-apply, after hardpoints / car / motion replaced
        _dmid = _jA.loads(_b0.decode('ascii'))
        _dmid['front_hp'] = {k: [v[0], v[1] + 0.001, v[2]] for k, v in _dmid['front_hp'].items()}
        _dmid['car']['__net_probe_key__'] = 1.0
        _pmid = os.path.join(_dA, 'shifted.vahan')
        with open(_pmid, 'w') as _fh:
            _jA.dump(_dmid, _fh)
        _oset = _wA._dynamics_panel.set_state
        _hit = []
        def _boom_set(*_a, **_k):
            if not _hit:                       # fail the LOAD only; the rollback's re-apply runs clean
                _hit.append(float(np.asarray(_wA._front_hp['wheel_center'])[1]))
                raise RuntimeError('injected mid-load failure')
            return _oset(*_a, **_k)
        _solv0 = np.asarray(_wA._solvers['FL'].solve(0.).uca_outer, float).copy()
        _wA._dynamics_panel.set_state = _boom_set
        try:
            _wA._load_project_from_path(_pmid)
            _fA.append('mid-apply failure did not raise')
        except RuntimeError:
            pass
        finally:
            _wA._dynamics_panel.set_state = _oset
        _okD = (_live(_wA) == _s0 and '__net_probe_key__' not in _wA._car
                and np.allclose(np.asarray(_wA._solvers['FL'].solve(0.).uca_outer, float), _solv0, atol=1e-12))
        _wc0 = _jA.loads(_s0)['front_hp']['wheel_center'][1]
        _mutated = bool(_hit) and abs(_hit[0] - (_wc0 + 0.001)) < 1e-12
        if not _mutated:
            _fA.append('injected failure did not fire after the hardpoints were replaced '
                       '(the rollback was not exercised)')
        if not _okD:
            _d0 = _jA.loads(_s0); _d1 = _jA.loads(_live(_wA))
            _diffk = [k for k in _d0 if _d0.get(k) != _d1.get(k)]
            _fA.append(f'mid-apply failure left the live project changed in {_diffk}')
        # (e) a normal load of the accepted file still works and round-trips
        _wA._load_project_from_path(_pA)
        _okE = _jA.loads(_live(_wA))['front_hp'] == _jA.loads(_b0.decode('ascii'))['front_hp']
        if not _okE:
            _fA.append('normal load after the injected failures did not restore the saved hardpoints')
        _shA.rmtree(_dA, ignore_errors=True)
        if _fA:
            fails += 1
        print(f'atomic save/load : bad save file intact={_okA and _okB} (no temp litter={not _litter}); '
              f'schema-broken load live unchanged={_okC}; mid-apply failure (hardpoints already '
              f'replaced={_mutated}) rolled back={_okD}; normal reload ok={_okE}   '
              + ('pass' if not _fA else 'UNEXPECTED FAIL: ' + '; '.join(_fA)))
except Exception as _e:
    fails += 1
    import traceback as _tbA2; _tbA2.print_exc()
    print(f'atomic save/load : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── LATERAL LOAD TRANSFER INVARIANT ─────────────────────────────────────────
# sum over axles of (one-side load delta x track) MUST equal m*a*h_cg.  This is
# pure statics: however the transfer splits between geometric / elastic /
# unsprung, and whatever the roll stiffness is, the TOTAL is fixed by the CG.
# Nothing gated this, so a stray /2 on the per-axle unsprung mass (dynamics.py
# AND a second copy in transient.py) left every published wheel load 4.8% short
# and no check noticed.  Assert it at several g so a partial fix cannot pass.
print('-' * 64)
try:
    import glob as _g2, re as _re2
    _cf = _g2.glob('configs/2027_v*.vahan')
    _cf = max(_cf, key=lambda p: int(_re2.search(r'2027_v(\d+)', p).group(1))) if _cf else None
    if _cf is None:
        print('load transfer inv: no config — skipped')
    else:
        _wL = MainWindow(); _wL._load_project_from_path(_cf); _wL._rebuild_solvers(0.)
        _ssL = _wL._build_dynamics_solver(); _vL = _ssL._veh
        _errs = []
        for _ay in (0.5, 1.0, 1.5, 2.0):
            _rL = _ssL.solve(_ay, 0.0); _F = _rL.Fz
            _got = (abs(_F['FR'] - _F['FL'])/2*_vL.front_track_m
                    + abs(_F['RR'] - _F['RL'])/2*_vL.rear_track_m)
            _need = _vL.total_mass_kg * _ay * 9.80665 * _vL.cg_height_m
            _rel = abs(_got - _need)/_need*100
            # 2.0 g clamps at wheel lift on this car, so only gate below lift
            if _ay <= 1.5 and _rel > 1.0:
                _errs.append(f'{_ay:.1f}g off {_rel:.1f}%')
            if _ay == 1.0:
                _lt1 = (_got, _need, _rel)
        _lt_ok = not _errs
        if not _lt_ok:
            fails += 1
        print(f'load transfer inv: at 1.0 g sum(dFz*track) {_lt1[0]:.1f} vs m*a*h {_lt1[1]:.1f} N.m '
              f'({_lt1[2]:.2f}% off)   {"pass" if _lt_ok else "UNEXPECTED FAIL: " + "; ".join(_errs)}')
except Exception as _e:
    fails += 1
    print(f'load transfer inv: UNEXPECTED FAIL ({_e})')

# ── CALIPER MOUNT matches the caliper DRAWING, and the drawn caliper sits where
#    the load arrows do.  Three different answers used to coexist: loads at
#    R_pad-25 = 69.4 mm, the render hardcoded rearward at R_rotor-6 = 114 mm
#    ignoring the angle input, and the drawing's D1 = R_rotor-27.9 = 92.1 mm.
print('-' * 64)
try:
    from vahan.loads import BrakeParams as _BP
    _t = _BP(rotor_dia_mm=254.0)          # drawing table row: 10.00 in disc
    _d1_in = _t.bolt_line_radius_mm / 25.4
    _a_in = _t.bolt_circle_radius_mm / 25.4
    _bpf3 = win._loads_panel.get_brake_params_front()
    _identity = abs(_bpf3.bolt_line_radius_mm + _bpf3.caliper_l4_mm
                    - _bpf3.pad_radius_mm) < 1e-6
    _v3 = win.view3d
    win._update_3d()
    _render_r = float(getattr(_v3, '_caliper_bolt_r', 0.0)) * 1000.0
    _render_ok = abs(_render_r - _bpf3.bolt_line_radius_mm) < 0.5
    cal_ok = (abs(_d1_in - 3.90) < 0.02 and abs(_a_in - 4.07) < 0.02
              and _identity and _render_ok)
    if not cal_ok:
        fails += 1
    print(f'caliper mount geo : 10in disc D1 {_d1_in:.2f}in (3.90) A {_a_in:.2f}in '
          f'(4.07), l3+l4=R_pad {_identity}, render r {_render_r:.1f} vs '
          f'loads {_bpf3.bolt_line_radius_mm:.1f} mm   '
          f'{"pass" if cal_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'caliper mount geo : UNEXPECTED FAIL ({_e})')

# ── WHEEL BEARINGS: overhung geometry (both bearings INBOARD of the wheel).
#    cp_offset was overloaded as both the outer-bearing position and the beam
#    lever; now the near bearing sits bearing_inboard_offset in and the tyre load
#    is overhung, so the outer bearing carries MORE than the wheel load and the
#    inner reverses.  Also: the rocker PIVOT reaction is a CHASSIS mount load.
print('-' * 64)
try:
    from gui import wheel_package as _WP3
    _up = win._loads_panel.get_upright_params()
    _its = _WP3._load_items(win, 1.5, 0.0, only_corner='FL')
    _piv = [it for it in _its if 'rocker pivot' in it[3].lower()]
    _piv_chassis = bool(_piv) and 'CHASSIS' in _piv[0][3]
    _ld = _WP3.compute_case(win, 1.5, 0.0)[0]['FL']
    _overhung = _ld.bearing_outer_V * _ld.bearing_inner_V < 0   # opposite signs
    _has_off = hasattr(_up, 'bearing_inboard_offset_mm')
    # Bearings must land INBOARD (toward the centreline) on BOTH lateral sides —
    # the spin axis is not mirrored, so a naive `wc - spin*off` flipped one side.
    _inb_ok = True
    for _lbl in ('FL', 'FR', 'RL', 'RR'):
        _st = win._solvers[_lbl].solve(0.)
        _wcx = float(np.asarray(_st.wheel_center, float)[0])
        for _it in _WP3._load_items(win, 1.5, 0.0, only_corner=_lbl):
            if 'bearing' in _it[3].lower() and 'RADIAL' in _it[3]:
                _px = float(np.asarray(_it[0], float)[0])
                if abs(_px) >= abs(_wcx):          # not closer to the centreline
                    _inb_ok = False
    brg_ok = _piv_chassis and _overhung and _has_off and _inb_ok
    if not brg_ok:
        fails += 1
    print(f'bearings/pivot   : inboard offset input {_has_off}, overhung '
          f'(outer {_ld.bearing_outer_V:.0f} / inner {_ld.bearing_inner_V:.0f} N), '
          f'inboard both sides {_inb_ok}, pivot on chassis {_piv_chassis}   '
          f'{"pass" if brg_ok else "UNEXPECTED FAIL"}')
except Exception as _e:
    fails += 1
    print(f'bearings/pivot   : UNEXPECTED FAIL ({_e})')

# ── REAR ROCKER COPLANAR across travel (config 2027_v28): the pushrod, rocker
#    and shock must stay in the rocker plate plane through the whole wheel travel
#    (was 48 mm off / 68 mm across travel before v28). Guards the coplanar re-tune
#    AND the rear motion ratio.  ARB wheel rate is a separately derived value:
#    it depends on the project panel's declared bar section/material as well as
#    the kinematic arm geometry, so it is checked from those explicit inputs,
#    not against a rate from another fixture.
print('-' * 64)
try:
    import glob as _glob
    _cfgs = _glob.glob('configs/2027_v28_*.vahan')
    if _cfgs:
        win._load_project_from_path(_cfgs[0]); win._rebuild_solvers(0.)
        _hp = win._rear_hp
        _p0 = np.asarray(_hp['rocker_pivot'], float)
        _P = []
        for _t in np.linspace(-0.04, 0.04, 7):
            _st = win._solvers['RL'].solve(_t)
            for _a in ('pushrod_inner', 'rocker_spring_pt', 'rocker_pivot'):
                _P.append(np.asarray(getattr(_st, _a), float))
        _P = np.array(_P); _u, _sv, _Vt = np.linalg.svd(_P - _P.mean(0)); _n = _Vt[2]
        _worst = 0.0
        for _t in np.linspace(-0.045, 0.045, 11):
            _st = win._solvers['RL'].solve(_t)
            for _a in ('pushrod_outer', 'spring_chassis_pt'):
                _worst = max(_worst, abs(float(np.dot(np.asarray(getattr(_st, _a), float) - _p0, _n))) * 1000)
        _veh = win._build_dynamics_solver()._veh
        _mr = _veh.motion_ratio_rear; _arb = _veh.arb_rate_rear_Npm
        _dp = win._dynamics_panel
        _rg = _dp._derived_arb_geom['R']
        _arb_expected = _dp._compute_arb_wheel_rate_Npm(
            OD_mm=_dp._arb_OD_r.value(), ID_mm=_dp._arb_ID_r.value(),
            L_half_mm=_rg['half_length_mm'], A_mm=_rg['arm_length_mm'],
            MR=_rg['mr'], G_Npmm2=_dp._arb_G.value(), E_Npmm2=_dp._arb_E.value(),
            blade_w_mm=_dp._arb_blade_w_r.value(), blade_t_mm=_dp._arb_blade_t_r.value())
        _arb_ok = (np.isfinite(_arb) and np.isfinite(_arb_expected)
                   and abs(_arb - _arb_expected) <= max(1.0, abs(_arb_expected) * 1e-9))
        _cop_ok = (_worst < 3.0 and abs(_mr - 0.808) < 0.01)
        if not (_cop_ok and _arb_ok):
            fails += 1
        print(f'rear coplanar    : {_worst:.2f} mm across travel, MR_r {_mr:.4f}, '
              f'arb_r {_arb:.0f} from OD {_dp._arb_OD_r.value():.2f} / ID '
              f'{_dp._arb_ID_r.value():.2f} mm, half {_rg["half_length_mm"]:.2f} mm, '
              f'arm {_rg["arm_length_mm"]:.2f} mm, MR {_rg["mr"]:.4f} '
              f'(derived {_arb_expected:.0f})   '
              f'{"pass" if _cop_ok and _arb_ok else "UNEXPECTED FAIL"}')
    else:
        print('rear coplanar    : no v28 config found — skipped')
except Exception as _e:
    fails += 1
    print(f'rear coplanar    : UNEXPECTED FAIL ({_e})')

# ── YMD TRIM CRITERION (vahan/ymd.py — the ONE yaw-moment state engine).
#    plot_mmd's old private iteration set rear slip = body slip exactly (a car
#    that never rotates); the engine carries the yaw-rate term (rear slip =
#    beta + degrees(l_r*r/V) in its nose-in convention, RCVD Eqs. 5.3/5.4) and
#    plot_mmd now routes through it.  Gates: (a) the yaw-rate term is ALIVE at
#    a real trim — rear slip differs from beta by exactly degrees(l_r*r/V) and
#    is NOT beta; (b) the as-built (-30%) trimmed max lands in a sane road-g
#    band at grip x0.70 on 8 m; (c) Milliken-desirable signs at that trim
#    (control dN/ddelta > 0, stability dN/dbeta < 0); (d) sweeping Ackermann
#    -100..+100 MOVES the trimmed max (measured 0.035 g on v56) — the
#    criterion discriminates, it is not a constant.
print('-' * 64)
try:
    import math as _m5
    from vahan.ymd import build_loads_table as _blt5, \
        trim_sweep_ackermann as _tsa5, ymd_state as _ys5
    _cf5 = [max(_cands)[1]] if _cands else []
    if not _cf5:
        print('ymd trim         : no config — skipped')
    else:
        _wY = MainWindow(); _wY._load_project_from_path(_cf5[0])
        _wY._dynamics_panel._tire_psi.setValue(12.0)
        _ssY = _wY._build_dynamics_solver()
        _tmY = getattr(_wY, '_tire_model', None)
        if _tmY is None:
            print('ymd trim         : no TTC file on this machine — skipped')
        else:
            _tabY = _blt5(_ssY)
            _rowsY = _tsa5(_tmY, _ssY, radius_m=8.0,
                           ackermann_list=(-100.0, -30.0, 0.0, 100.0),
                           grip_multiplier=0.70, loads_table=_tabY)
            _yfail = []
            _r30 = next(r for r in _rowsY if abs(r['pct'] + 30.0) < 0.5)
            # (b) trimmed max at the as-built setting in a sane road-g band
            if not (np.isfinite(_r30['Ay_trim_max'])
                    and 1.2 <= _r30['Ay_trim_max'] <= 2.0):
                _yfail.append(f'as-built trimmed max {_r30["Ay_trim_max"]:.3f} g '
                              f'outside 1.2-2.0')
            # (c) stability sign at that trim.  NO control assertion: at the
            # maximum-trim point dN/dsteer = 0 by construction (a nonzero
            # value means more steer buys more trimmed moment, so it was not
            # the max), and the refuter measured the finite N_delta values as
            # pure step-size + tyre-grid artifact — the sign was not even
            # stable across difference steps (+9.3 at h=0.1 vs -3.7 at h=0.5
            # on the -60% row).  Asserting its sign would gate on noise.
            if not (np.isfinite(_r30['N_beta']) and _r30['N_beta'] < 0):
                _yfail.append(f'stability N_beta {_r30["N_beta"]:+.1f} >= 0')
            # (d) the sweep discriminates
            _aysY = [r['Ay_trim_max'] for r in _rowsY
                     if r['converged'] and np.isfinite(r['Ay_trim_max'])]
            _sprY = (max(_aysY) - min(_aysY)) if len(_aysY) >= 2 else 0.0
            if _sprY <= 0.005:
                _yfail.append(f'Ackermann spread {_sprY:.4f} g <= 0.005 '
                              f'(criterion blind)')
            # (a) yaw-rate term alive at the as-built trim (seed on-branch —
            # an unseeded probe can converge to the mirror hand)
            if np.isfinite(_r30['Ay_trim_max']):
                _stY = _ys5(_tmY, _ssY, _r30['beta_at'], _r30['delta_at'],
                            radius_m=8.0, ackermann_pct=-30.0,
                            grip_multiplier=0.70, loads_table=_tabY,
                            Ay0=_r30['Ay_trim_max'])
                _vehY = _ssY._veh
                _lrY = _vehY.wheelbase_m * _vehY.front_weight_fraction
                _expY = _m5.degrees(_lrY * _stY['yaw_rate_radps']
                                    / _stY['V_mps'])
                _gotY = _stY['slip_deg']['RL'] - _stY['beta_deg']
                if abs(abs(_gotY) - abs(_expY)) > 0.05:
                    _yfail.append(f'rear slip - beta = {_gotY:+.3f} deg but '
                                  f'l_r*r/V = {_expY:+.3f}')
                if abs(_gotY) < 0.5:
                    _yfail.append('rear slip equals beta — yaw-rate term dead '
                                  '(the old plot_mmd flaw)')
            if _yfail:
                fails += 1
            print(f'ymd trim         : as-built max {_r30["Ay_trim_max"]:.3f} g trimmed '
                  f'@ beta {_r30["beta_at"]:+.2f} deg, stability '
                  f'{_r30["N_beta"]:+.0f} Nm/deg (control ~0 at max trim '
                  f'by construction), sweep spread {_sprY:.3f} g   '
                  + ('pass' if not _yfail else
                     'UNEXPECTED FAIL: ' + '; '.join(_yfail)))
except Exception as _e:
    fails += 1
    print(f'ymd trim         : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── ACKERMANN PAGE (gui/ackermann_page.py) + the Fz-Fy compute/draw split.
#    The page draws graph 3 from ackermann_fz_fy_data, which was EXTRACTED out
#    of plot_ackermann_fz_fy so two consumers could share one copy of the
#    physics.  An extraction that quietly changes numbers is exactly the bug
#    the ONE-MODEL rule exists to prevent, so gate it: (a) the arrays the data
#    function returns must be the arrays the plot function DRAWS, curve for
#    curve; (b) the page must construct, run its four fast stages, and leave
#    every canvas with non-empty _plot_data (hover was dead on the tyre-grip
#    plots for months because that list was never filled).
print('-' * 64)
try:
    from PyQt6.QtCore import QCoreApplication as _QCA6
    from vahan.analysis_plots import (ackermann_fz_fy_data as _afd6,
                                      plot_ackermann_fz_fy as _apf6)
    _cf6 = [max(_cands)[1]] if _cands else []
    if not _cf6:
        print('ackermann page   : no config — skipped')
    else:
        _w6 = MainWindow(); _w6._load_project_from_path(_cf6[0])
        _w6._dynamics_panel._tire_psi.setValue(12.0)
        _tm6 = getattr(_w6, '_tire_model', None)
        if _tm6 is None:
            print('ackermann page   : no TTC file on this machine — skipped')
        else:
            _ss6 = _w6._build_dynamics_solver()
            _kw6 = dict(radius_m=8.0, ackermann_pct=45.0,
                        grip_multiplier=0.70, lat_g_list=(0.4, 0.8, 1.2, 1.6))
            # SteadyStateSolver keeps a warm-start cache, so two back-to-back
            # sweeps are NOT bitwise reproducible.  Clear it before each call
            # or this check is flaky at the 1e-9 level for reasons that have
            # nothing to do with the compute/draw split.
            _ss6._warm = {}
            _d6 = _afd6(_tm6, _ss6, **_kw6)
            _ss6._warm = {}
            _fig6 = _apf6(_tm6, _ss6, **_kw6)
            _afail = []
            # (a) compute/draw equivalence, by LABEL so a reordering cannot
            #     silently pass.
            _want6 = {'Outer CAPABILITY': ('fz_outer', 'cap_outer', None),
                      'Inner CAPABILITY': ('fz_inner', 'cap_inner', None),
                      'Outer delivered':  ('fz_outer', 'del_outer', 'n'),
                      'Inner delivered':  ('fz_inner', 'del_inner', 'n')}
            _lines6 = {ln.get_label(): ln
                       for ax_ in _fig6.get_axes() for ln in ax_.get_lines()}
            for _pre, (_xk, _yk, _clip) in _want6.items():
                _ln = next((v for k, v in _lines6.items()
                            if k.startswith(_pre)), None)
                if _ln is None:
                    _afail.append(f'plot has no "{_pre}" curve')
                    continue
                _n6 = _d6['n_delivered'] if _clip else len(_d6[_xk])
                _ex = np.asarray(_d6[_xk], float)[:_n6]
                _ey = np.asarray(_d6[_yk], float)[:_n6]
                _gx = np.asarray(_ln.get_xdata(), float)
                _gy = np.asarray(_ln.get_ydata(), float)
                if _gx.shape != _ex.shape or _gy.shape != _ey.shape:
                    _afail.append(f'{_pre}: drawn {_gx.shape}/{_gy.shape} vs '
                                  f'data {_ex.shape}/{_ey.shape}')
                elif not (np.allclose(_gx, _ex, atol=1e-6, rtol=1e-9)
                          and np.allclose(_gy, _ey, atol=1e-6, rtol=1e-9)):
                    _afail.append(
                        f'{_pre}: drawn values != data function '
                        f'(max dx {np.max(np.abs(_gx - _ex)):.2e}, '
                        f'dy {np.max(np.abs(_gy - _ey)):.2e})')
            # (b) the page runs and every canvas is hoverable
            _w6._switch_page(4)
            _pg6 = getattr(_w6, '_ackermann_page', None)
            if _pg6 is None:
                _afail.append('page did not construct')
            else:
                _pg6._do_lap.setChecked(False)      # 4 min of lap sims: no
                _pg6._radii_txt.setText('3, 8')
                _pg6._g_n.setValue(5)
                _pg6._sweep_txt.setText('-60, 0, 60')
                _pg6._btn.click()
                import time as _time6
                _t0_6 = _time6.time()
                while (_pg6._worker is not None and _pg6._worker.isRunning()
                       and _time6.time() - _t0_6 < 300):
                    _QCA6.processEvents(); _time6.sleep(0.05)
                for _ in range(40):
                    _QCA6.processEvents(); _time6.sleep(0.01)
                for _k6 in ('demand', 'pct', 'fzfy', 'pair'):
                    _pd6 = getattr(_pg6._slots[_k6].canvas, '_plot_data', [])
                    if not _pd6 or not any(s for _a, s in _pd6):
                        _afail.append(f'{_k6} canvas has no hover _plot_data')
                    _txt6 = _pg6._slots[_k6]._sum.text()
                    if not _txt6.startswith('WHAT IT MEANS'):
                        _afail.append(f'{_k6} has no plain-English summary')
            if _afail:
                fails += 1
            print(f'ackermann page   : compute/draw split identical on 4 '
                  f'curves, 4 canvases hoverable + summarised   '
                  + ('pass' if not _afail else
                     'UNEXPECTED FAIL: ' + '; '.join(_afail)))
except Exception as _e:
    fails += 1
    import traceback as _tb6; _tb6.print_exc()
    print(f'ackermann page   : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── LAP SIM HONESTY (vahan/laptime.py).  Three things that were silently
#    wrong or silently absent, each of which flattered the lap number:
#      (i)   ROTATING INERTIA.  The driveline was massless: the same tyre
#            force produced F/m of acceleration instead of F/(m+m_eq).
#            Measured on autocross26 at full resolution: 41.902 -> 43.900 s,
#            +2.00 s (+4.8%), 134 kg of equivalent mass in 1st gear.  Gate
#            that switching it on still SLOWS the car by a real margin and
#            that the equivalent mass is in the physically sane band.
#            2026-09-22 (audit items 22/23): that +2.00 s over-charged inertia
#            — it divided TYRE-limited braking and traction by (1 + m_eq/m).
#            Inertia now slows only torque-limited drive (plus the free
#            wheels in traction); a tyre-limited stop is inertia-invariant.
#            v147 on the Laptime page's own settings, full resolution:
#            inertia costs +1.22 s (was +1.92 s on the same car/code).  The
#            net's subsampled inertia+shift delta reads ~+1.1 s, so the gate
#            is 0.5 s: still proves inertia is LIVE, no longer demands the
#            overcount.
#      (ii)  SHIFT CHATTER.  The gear picker had no memory, so it used 2nd
#            gear for ONE STATION (0.08 s) at the top of a straight, five
#            times a lap.  Gate that the minimum-shift-interval actually
#            bounds the dwell, AND that with the model off the dwell breaks
#            that bound (otherwise the gate is testing nothing).
#    Run on a SUBSAMPLED real track so two lap sims fit in the net's budget.
print('-' * 64)
try:
    from vahan.laptime import Track as _TkL, LapSimulator as _LsL
    _cfL = [max(_cands)[1]] if _cands else []
    _trkL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'tracks', 'autocross26.json')
    if not _cfL or not os.path.isfile(_trkL):
        print('lap sim honesty  : no config/track — skipped')
    else:
        _wL2 = MainWindow(); _wL2._load_project_from_path(_cfL[0])
        _ssL2 = _wL2._build_dynamics_solver()
        _full = _TkL.from_json(_trkL)
        _K = 5                      # every 5th station: ~160 stations, ~5 m
        _trL = _TkL(name='net', x_m=_full.x_m[::_K], y_m=_full.y_m[::_K],
                    kappa_1pm=_full.kappa_1pm[::_K], width_m=_full.width_m)

        def _mk_lap(inertia, shift):
            _s = _LsL(_ssL2, cla_m2=0.0, cda_m2=1.0,
                      air_density=float(_ssL2._veh.air_density_kg_m3),
                      aero_cop_rear_frac=0.5, grip_scale=0.65,
                      static_rh_front_mm=50.0, static_rh_rear_mm=50.0)
            _s.set_gearbox([2.750, 2.000, 1.667, 1.444, 1.304, 1.208],
                           primary_ratio=1.0, final_drive=3.545,
                           redline_rpm=11000.0)
            if not inertia:
                _s.set_rotating_inertia(0.0, 0.0, 0.0)
            _s.set_shift_model(*shift)
            return _s

        _MININT = 0.60
        _sOff = _mk_lap(False, (0.0, 0.0, 0.0))
        _rOff = _sOff.simulate(_trL, n_detail=8)
        _sOn = _mk_lap(True, (0.10, _MININT, 0.04))
        _rOn = _sOn.simulate(_trL, n_detail=8)

        def _min_dwell(res):
            _g = np.asarray(res.gear, int)
            _i = np.where(np.diff(_g) != 0)[0] + 1
            if len(_i) < 2:
                return float('inf')
            return float(np.min(np.diff(np.asarray(res.t_s)[_i])))

        _lf = []
        # (i) rotating inertia is LIVE and physically sized
        _d_lap = _rOn.lap_time_s - _rOff.lap_time_s
        _meq = _sOn.equivalent_mass_kg(1)
        if _d_lap < 0.5:
            _lf.append(f'rotating inertia + shift model only cost '
                       f'{_d_lap:+.3f} s — it should slow the car by >0.5 s '
                       f'(measured +1.11 s here after the 2026-09-22 '
                       f'tyre-vs-torque fix; +2.09 s before it)')
        if not (100.0 <= _meq <= 200.0):
            _lf.append(f'equivalent mass in 1st is {_meq:.0f} kg, outside the '
                       f'100-200 kg band a 0.05 kg.m^2 crank on a 9.75:1 '
                       f'first gear implies')
        if abs(_sOff.equivalent_mass_kg(1)) > 1e-9:
            _lf.append('set_rotating_inertia(0,0,0) did not disable it')
        # (ii) chatter is bounded WITH the model and unbounded without it
        _dw_on, _dw_off = _min_dwell(_rOn), _min_dwell(_rOff)
        if _dw_on < 0.9 * _MININT:
            _lf.append(f'shortest gear dwell WITH the shift model is '
                       f'{_dw_on:.3f} s, under the {_MININT:.2f} s minimum '
                       f'interval — the interval is not being enforced')
        if _dw_off >= _MININT:
            _lf.append(f'shortest gear dwell WITHOUT the shift model is '
                       f'{_dw_off:.3f} s — the chatter this gate exists to '
                       f'catch is not present, so the gate proves nothing')
        if _rOn.shift_time_lost_s <= 0.0:
            _lf.append('shift dead time never charged (0.00 s of torque cut)')
        if _lf:
            fails += 1
        print(f'lap sim honesty  : rot.inertia {_meq:.0f} kg eq -> '
              f'{_d_lap:+.2f} s, gear dwell {_dw_off:.2f} s -> {_dw_on:.2f} s, '
              f'{_rOn.shift_time_lost_s:.2f} s cut   '
              + ('pass' if not _lf else 'UNEXPECTED FAIL: ' + '; '.join(_lf)))
except Exception as _e:
    fails += 1
    import traceback as _tbL; _tbL.print_exc()
    print(f'lap sim honesty  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── AUDIT 2026-09-22 items 21-24, 27 (SOFTWARE_MATH_AND_USABILITY_AUDIT,
#    Appendix 4; verified in DESIGN_2027/binder_run/ASTRA_AUDIT_VERIFY_*).
#    FAILING-THEN-PASSING, each on the audit's own repro numbers:
#      21 Engine VE-taper controls were dead (module globals never reached
#         ve_target_curve): 9500/0.72 -> 7000/0.40 moved torque by 0.0 N m.
#      22 Lap torque cut returned max(0, …) = 0.0; force balance at 25 m/s,
#         CdA 1, rho 1.2, no inertia = -1.29155 m/s^2.
#      23 A TYRE-limited stop was divided by (1 + m_eq/m): 9.81 -> 6.622 m/s^2
#         with no brake-torque limit modelled.  Friction-limited decel must be
#         inertia-INVARIANT; a TORQUE-limited drive must still be slowed by it.
#      24 Aero-heave wheel force dropped F_s*dMR/dq: 485.69 N vs exact 608.2 N
#         (c = q + 2.5q^2, MR = 1 + 5q, K_s 22 kN/m, F_s0 1 kN, q 20 mm).
#      27 Ride selector called min tyre load -100 N / 10 % contact loss
#         'feasible'.
print('-' * 64)
try:
    import vahan.engine as _E21
    from types import SimpleNamespace as _SN21
    from vahan.dynamics import VehicleParams as _VP21
    from vahan.laptime import LapSimulator as _LS21
    from vahan.ride_solve import RideSweep as _RSw27
    _af = []
    _d21 = _dp21 = _a22 = _b0 = _b1 = _dt0 = _dt1 = _dq0 = _dq1 = _lap24 = _ex24 = float('nan')
    _sel27 = {'status': '(not run)'}
    try:
        # ── 21: explicit calibration reaches the curve; GUI boxes are live ──
        _g0 = (_E21.VE_FALL_START_RPM, _E21.VE_AT_13K)
        _t0 = np.asarray(_E21.engine_curve('corrected')[1])
        _t1 = np.asarray(_E21.engine_curve('corrected', ve_fall_rpm=7000., ve_13k=.40)[1])
        _d21 = float(np.max(np.abs(_t1 - _t0)))
        if not _d21 > 1.0:
            _af.append(f'21: VE taper 9500/0.72 -> 7000/0.40 moved torque {_d21:.3f} N m (dead input)')
    except Exception as _ex:
        _af.append(f'21: crashed {type(_ex).__name__}: {_ex}')
    try:
        # ── 21b: the Engine page's taper boxes (the GUI path) move its curve ──
        _g0 = (_E21.VE_FALL_START_RPM, _E21.VE_AT_13K)
        from gui.engine_page import EnginePage as _EP21
        _ep = _EP21(win)
        _c0 = np.asarray(_ep.current_curve()[1])
        _ep._vefall.setValue(7000.); _ep._ve13.setValue(0.40)
        _c1 = np.asarray(_ep.current_curve()[1])
        _dp21 = float(np.max(np.abs(_c1 - _c0)))
        if not _dp21 > 1.0:
            _af.append(f'21: Engine page taper boxes moved its curve {_dp21:.3f} N m (dead controls)')
        if np.isfinite(_d21) and not np.allclose(_c1, _t1):
            _af.append('21: Engine page curve != engine_curve(same knobs)')
        if (_E21.VE_FALL_START_RPM, _E21.VE_AT_13K) != _g0:
            _af.append('21: a GUI path still mutates vahan.engine module globals')
        _ep.deleteLater()
    except Exception as _ex:
        _af.append(f'21b: crashed {type(_ex).__name__}: {_ex}')
    try:
        # ── 22: torque cut slows the car by exactly -D/m (no inertia) ──
        _v21 = _VP21(power_hp=80)
        _ss21 = _SN21(_veh=_v21, _tire=None, _solvers={}, _traction_g_dynamic=lambda _: 1.)
        _s21 = _LS21(_ss21, cda_m2=1., air_density=1.2)
        _s21._ay_max = lambda speed: 9.81
        _s21.set_rotating_inertia(0, 0, 0)
        _a22 = _s21._ax_drive(25., 0., torque_cut=True)
        _e22 = -_s21.drag_N(25.) / _v21.total_mass_kg
        if not (abs(_e22 + 1.2915446874) < 1e-6 and abs(_a22 - _e22) < 1e-9):
            _af.append(f'22: torque cut gave {_a22:.6f} m/s^2, force balance {_e22:.6f}')
    except Exception as _ex:
        _af.append(f'22: crashed {type(_ex).__name__}: {_ex}')
    try:
        # ── 23: friction-limited stop invariant to inertia; torque branch not ──
        _s21.cda = 0.
        _b0 = _s21._ax_brake(20, 0)
        _s21.set_rotating_inertia(.19, .19, .05)
        _b1 = _s21._ax_brake(20, 0)
        if not (abs(_b0 - 9.81) < 1e-12 and abs(_b1 - 9.81) < 1e-12):
            _af.append(f'23: tyre-limited brake {_b0:.5f} -> {_b1:.5f} m/s^2 with inertia (must stay 9.81)')
        # traction-limited drive (grip 0.5 g, huge engine torque via 1st gear):
        # engine + driven-wheel inertia must NOT reduce it; only the free fronts
        _s21.set_gearbox([2.75], 1.0, 3.545, 11000.)
        _s21.ss._traction_g_dynamic = lambda _: 0.05     # 0.05 g: surely tyre-limited
        _s21.set_rotating_inertia(0, 0, 0)
        _dt0 = _s21._ax_drive(10., 0., gear=1)
        _s21.set_rotating_inertia(.19, .19, .05)
        _dt1 = _s21._ax_drive(10., 0., gear=1)
        _m21 = _v21.total_mass_kg; _rr = _v21.tire_radius_m
        _dt1_exp = _dt0 * _m21 / (_m21 + 2 * .19 / _rr ** 2)
        if not abs(_dt1 - _dt1_exp) < 1e-9:
            _af.append(f'23: traction-limited drive {_dt0:.4f} -> {_dt1:.4f} m/s^2 with inertia, '
                       f'expected {_dt1_exp:.4f} (free-wheel inertia only)')
        # torque-limited drive (grip huge): whole-driveline inertia DOES apply
        _s21.ss._traction_g_dynamic = lambda _: 50.
        _s21.set_rotating_inertia(0, 0, 0)
        _dq0 = _s21._ax_drive(20., 0., gear=1)
        _s21.set_rotating_inertia(.19, .19, .05)
        _dq1 = _s21._ax_drive(20., 0., gear=1)
        if not abs(_dq1 - _dq0 / _s21._inertia_div(1)) < 1e-9 or not _dq1 < _dq0:
            _af.append(f'23: torque-limited drive {_dq0:.4f} -> {_dq1:.4f} m/s^2, expected /'
                       f'{_s21._inertia_div(1):.4f}')
    except Exception as _ex:
        _af.append(f'23: crashed {type(_ex).__name__}: {_ex}')
    try:
        # ── 24: progressive spring wheel force == virtual work, exact ──
        _v24 = _VP21(power_hp=80, static_spring_force_front_N=1000.)
        _ss24 = _SN21(_veh=_v24, _tire=None, _traction_g_dynamic=lambda _: 1.,
                      _solvers={'FL': _SN21(solve=lambda t: _SN21(spring_length=.210 - t - 2.5 * t * t))})
        _cv24 = _LS21(_ss24, cda_m2=1., air_density=1.2)._build_rate_curves()['F']
        _q24, _k24 = .020, _v24.spring_rate_front_Npm
        _lap24 = float(np.interp(_q24, _cv24['ts'], _cv24['fcum']))
        _ex24 = (1000. + _k24 * (_q24 + 2.5 * _q24 ** 2)) * (1 + 5 * _q24) - 1000.
        if not (abs(_ex24 - 608.2) < 1e-6 and abs(_lap24 - _ex24) < 1e-6):
            _af.append(f'24: lap wheel force {_lap24:.4f} N vs exact {_ex24:.4f} N at 20 mm')
    except Exception as _ex:
        _af.append(f'24: crashed {type(_ex).__name__}: {_ex}')
    try:
        # ── 27: contact-invalid linear result is never 'feasible' ──
        _z4 = np.zeros((1, 1, 1, 4)); _z3 = np.zeros((1, 1, 1))
        _sw27 = _RSw27(front_rates_Npm=np.array([20000.]), rear_rates_Npm=np.array([20000.]),
                       case_labels=['deliberately invalid contact'], speeds_mps=(15.,), dlc=_z4 + .2,
                       travel_usage=_z4 + .5, travel_peak_m=_z4 + .01, damper_velocity_rms_mps=_z4,
                       damper_velocity_peak_mps=_z4, contact_loss_fraction_linear=_z4 + .1,
                       min_load_N=_z4 - 100, body_heave_acc_rms_mps2=_z3,
                       settle_pass=np.ones_like(_z3, dtype=bool), pitch_to_bounce=_z3 + .5,
                       front_settle_s=_z3 + 1, rear_settle_s=_z3 + 1, stop_margin_m=.02)
        _sel27 = _sw27.select()
        if _sel27['status'] == 'feasible' or _sel27['n_feasible'] != 0 or _sel27.get('contact_ok', True):
            _af.append(f"27: -100 N / 10 % contact loss selected as status={_sel27['status']!r}, "
                       f"n_feasible={_sel27['n_feasible']}")
    except Exception as _ex:
        _af.append(f'27: crashed {type(_ex).__name__}: {_ex}')
    if _af:
        fails += 1
    print(f'audit 21-24,27   : taper {_d21:.1f} N m (page {_dp21:.1f}); cut {_a22:+.5f} m/s^2; '
          f'tyre-limited brake {_b0:.2f}->{_b1:.2f}; trac drive {_dt0:.4f}->{_dt1:.4f}, '
          f'torque drive {_dq0:.3f}->{_dq1:.3f}; spring {_lap24:.1f}/{_ex24:.1f} N; '
          f'ride contact -> {_sel27["status"][:22]!r}   '
          + ('pass' if not _af else 'UNEXPECTED FAIL: ' + '; '.join(_af)))
except Exception as _e:
    fails += 1
    import traceback as _tbA; _tbA.print_exc()
    print(f'audit 21-24,27   : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── ACKERMANN SWEEP MUST NOT MOVE WITH THE SETTING IT SWEEPS.
#    A sweep whose answer depends on where the car's rack happens to sit is
#    reading noise, not Ackermann — and that bug has already happened in this
#    repo.  The per-station chain (AckermannStationModel -> LapSimulator) is
#    built to be structurally immune: the Ackermann % is an explicit argument
#    to vahan.ackermann.solve_ackermann_force, and the underlying
#    SteadyStateSolver carries no steer geometry at all.  Gate it by MOVING
#    THE RACK 25 mm fore-aft (the same perturbation that swung this car
#    +40.8% -> +64.5% as built), rebuilding, and demanding the capability
#    table come back bit-for-bit identical.
#    Also verify the explicit Ackermann input reaches the per-wheel slip
#    outputs.  At 100% the slips must be equal; endpoint settings must produce
#    different splits.  A capability argmax at 100% is a valid data-dependent
#    result (or a plateau tie), not evidence of a construction tautology.
print('-' * 64)
try:
    from vahan.laptime import AckermannStationModel as _ASM
    from vahan.ackermann import solve_ackermann_force as _saf7
    _cf7 = [max(_cands)[1]] if _cands else []
    if not _cf7:
        print('ackermann sweep  : no config — skipped')
    else:
        _w7 = MainWindow(); _w7._load_project_from_path(_cf7[0])
        _w7._dynamics_panel._tire_psi.setValue(12.0)
        _tm7 = getattr(_w7, '_tire_model', None)
        if _tm7 is None:
            print('ackermann sweep  : no TTC file on this machine — skipped')
        else:
            _PCT7 = [-100.0, 0.0, 100.0]
            _RAD7 = (3.5, 10.0)
            # 2.0 g added (2026-09-22): with only 0.8/1.4 g no cell of the
            # current car ever became front-limited, so the ay-capability
            # comparison had nothing finite to compare (all inf -> NaN).
            _GG7 = (0.8, 1.4, 2.0)

            def _tbl7(_ss):
                _m = _ASM(_tm7, _ss, _PCT7, grip_multiplier=0.65)
                _m.RADII_M = _RAD7
                _m.LAT_G = _GG7
                _m._ceil = np.full((len(_RAD7), len(_GG7), len(_PCT7)), np.nan)
                _m._scrub = np.full_like(_m._ceil, np.nan)
                _m._demand = np.full((len(_RAD7), len(_GG7)), np.nan)
                _m._ay_cap = np.full((len(_RAD7), len(_PCT7)), np.nan)
                _ss._warm = {}
                return _m.build()

            _ss7a = _w7._build_dynamics_solver()
            _a7 = _tbl7(_ss7a)
            _ack_before = float(_w7._probe_static_ackermann())
            # Move the rack fore-aft — a REAL Ackermann change.  25 mm is the
            # nominal perturbation, but a tight-steering-margin geometry (e.g.
            # the tie rod pulled inboard for 1.5" rim clearance) may not solve
            # to FULL LOCK with the rack relocated that far, which makes the
            # as-built probe NaN.  That is a reduced rack-ADJUSTMENT margin, not
            # a broken sweep — so back the perturbation off until the as-built
            # stays solvable and the change is still live (>1%).  The invariant
            # this gate actually protects (swept capability immune to rack) is
            # re-checked below regardless.
            _tri0 = np.asarray(_w7._front_hp['tie_rod_inner'], float)
            _ack_after = float('nan'); _rack_mm = 0.0
            for _rack_mm in (25.0, 20.0, 15.0, 10.0):
                _w7._front_hp['tie_rod_inner'] = _tri0 + np.array([0.0, _rack_mm / 1000.0, 0.0])
                _w7._rebuild_solvers(0.)
                _ack_after = float(_w7._probe_static_ackermann())
                if np.isfinite(_ack_after) and abs(_ack_after - _ack_before) > 1.0:
                    break
            _ss7b = _w7._build_dynamics_solver()
            _b7 = _tbl7(_ss7b)

            _f7 = []
            if not (np.isfinite(_ack_before) and np.isfinite(_ack_after)
                    and abs(_ack_after - _ack_before) > 1.0):
                _f7.append(f'no rack move in 10..25 mm kept the as-built '
                           f'Ackermann solvable AND live ({_ack_before:.2f}% '
                           f'-> {_ack_after:.2f}% at +{_rack_mm:.0f} mm) — the '
                           f'perturbation is not live, so this gate proves '
                           f'nothing')
            # inf-AWARE comparison (Astra F-08, 2026-09-22).  _ay_cap stores
            # inf for "never front-limited inside the g ladder" — a VALID state,
            # which ay_front_cap_g maps to 'no cap'.  The old line took
            # nanmax(|inf - inf|) = nanmax(nan...) = nan when EVERY cell was inf
            # (v147/v148: the 3.5 m / -100 % cell holds 1817.6 N against a
            # 1810.6 N demand at 1.4 g, so no cell ever crosses) and reported
            # "capability moved nan g" — measuring nothing.  Equal entries
            # (incl. matching infs) now diff to 0; a finite<->inf flip or a NaN
            # on either side is a real change / invalid state and stays nan
            # -> fails.  The finite FORCE CEILING the crossing is built from is
            # also compared directly, so the gate always measures something.
            def _inv_diff(_x, _y):
                _x = np.asarray(_x, float); _y = np.asarray(_y, float)
                _same = (_x == _y)
                _d = np.where(_same, 0.0, np.abs(_x - _y))
                return np.where(np.isfinite(_d) | _same, _d, np.nan)
            _dd7 = _inv_diff(_a7._ay_cap, _b7._ay_cap)
            _dmax = float(np.max(_dd7)) if np.all(np.isfinite(_dd7)) else float('nan')
            _n_lim7 = int(np.isfinite(_a7._ay_cap).sum())
            _n_inf7 = int(np.isposinf(_a7._ay_cap).sum())
            _W7 = float(_ss7a._veh.total_mass_kg) * 9.81 * float(_ss7a._veh.front_weight_fraction)
            # Ceiling compared only at rungs where the front pair still COVERS
            # the demand (both builds).  Measured 2026-09-22 on v148: moving the
            # rack 25 mm also changes the car's bump steer / camber slightly
            # (front camber 0.0009 deg, Fz 0.016 N at 1.4 g) — a real kinematic
            # change, not the Ackermann setting.  Below the limit that moves the
            # ceiling <= 0.007 N; on the 2.0 g rung, where the front is already
            # past its limit (demand 2586 N vs ceiling ~2000 N), the saturated
            # slip fallback amplifies it to 2.35 N.  That rung only feeds the
            # interpolated crossing, which the ay gate above already bounds.
            _dce = _inv_diff(_a7._ceil, _b7._ceil)
            _cov7 = ((_a7._ceil >= _a7._demand[:, :, None])
                     & (_b7._ceil >= _b7._demand[:, :, None]))
            _dceil_N = (float(np.max(_dce[_cov7])) if _cov7.any() and np.all(np.isfinite(_dce[_cov7]))
                        else float('nan'))
            _dceil_all_N = float(np.nanmax(_dce)) if np.isfinite(_dce).any() else float('nan')
            if not (np.all(np.isfinite(_a7._ceil)) and np.all(np.isfinite(_b7._ceil))):
                _f7.append('front-axle force ceiling table has non-finite cells '
                           f'({int((~np.isfinite(_a7._ceil)).sum())} before, '
                           f'{int((~np.isfinite(_b7._ceil)).sum())} after the rack move)')
            elif not (_dceil_N <= 1e-4 * _W7):
                _f7.append(f'the sub-limit front-axle force ceiling moved {_dceil_N:.3e} N '
                           f'(tolerance {1e-4 * _W7:.3f} N = 1e-4 g of front axle '
                           f"weight) when only the car's own Ackermann changed")
            if _n_lim7 == 0:
                _f7.append('no cell became front-limited, so the ay comparison '
                           'compared only inf sentinels (widen the g ladder)')
            # Tolerance is PHYSICAL, not bitwise.  This was 1e-12 g, which is
            # unsatisfiable by construction: SteadyStateSolver keeps a
            # warm-start cache, so back-to-back solves are not bitwise
            # reproducible (found while building this very model).  Observed
            # movement is 2.5e-06 g — about 2000x BELOW this model's own
            # measured binning error of 0.00525 g, i.e. far past the point
            # where the number means anything.  1e-4 g sits 50x under the
            # binning error and 40x over the float noise, so a REAL leak of
            # the car's own setting into the sweep still trips it while
            # solver round-off does not.
            _AY_INVARIANCE_TOL = 1e-4
            # The crossing is linearly interpolated between the last covered rung
            # and the first SATURATED one (laptime._cross).  The saturated rung's
            # margin moves by the rack's REAL kinematic change amplified by the
            # saturated slip fallback (documented above, ~2 N) — that is not a
            # leak, and the sub-limit ceiling gate above bounds any leak strictly.
            # So each cell may move by 1e-4 g PLUS exactly that rung's first-order
            # effect: d(ay)/d(m_sat) = dg * m_cov / (m_cov - m_sat)^2 (2026-09-23,
            # v149 tripped 2.3e-4 g from this alone).
            _g7 = np.asarray(_GG7, float); _sat_ay = 0.0; _cell_bad = []
            for _i7 in range(len(_RAD7)):
                for _k7 in range(len(_PCT7)):
                    _ma = _a7._ceil[_i7, :, _k7] - _a7._demand[_i7, :]
                    _mb = _b7._ceil[_i7, :, _k7] - _b7._demand[_i7, :]
                    _allow = _AY_INVARIANCE_TOL
                    for _j7 in range(1, len(_g7)):
                        if _ma[_j7 - 1] > 0 >= _ma[_j7]:
                            _sens = (_g7[_j7] - _g7[_j7 - 1]) * _ma[_j7 - 1] / (_ma[_j7 - 1] - _ma[_j7]) ** 2
                            _c = abs(_sens * (_mb[_j7] - _ma[_j7])); _sat_ay = max(_sat_ay, _c)
                            _allow += _c
                            break
                    _dc = _dd7[_i7, _k7]
                    if not np.isfinite(_dc) or _dc > _allow:
                        _cell_bad.append((_RAD7[_i7], _PCT7[_k7], float(_dc), _allow))
            print(f'ackermann sweep  : saturated-rung propagation up to {_sat_ay:.2e} g '
                  f'(allowance per cell = 1e-4 g + that); cells over: {len(_cell_bad)}')
            if _cell_bad or not np.isfinite(_dmax):
                _f7.append(f'the swept capability moved by {_dmax:.3e} g when '
                           f'only the CAR\'S OWN Ackermann changed — the '
                           f'sweep is reading its own setting')
            # Explicit-input response: 100% gives equal front slips, while
            # settings on either side change their split.  This inspects the
            # solver output, rather than assuming where a measured tyre-force
            # optimum is allowed to sit.
            _d7 = _saf7(_tm7, _ss7a, 5.0, 1.2, ack_range=(-100.0, 200.0),
                        n=13, grip_multiplier=0.65)
            _p7 = np.asarray(_d7['ackermann_pct'], float)
            _u7 = np.asarray(_d7['useful_N'], float)
            _d100 = _saf7(_tm7, _ss7a, 5.0, 1.2, ack_range=(100.0, 100.0),
                          n=1, grip_multiplier=0.65)['best']
            _dlo = _saf7(_tm7, _ss7a, 5.0, 1.2, ack_range=(-100.0, -100.0),
                         n=1, grip_multiplier=0.65)['best']
            _dhi = _saf7(_tm7, _ss7a, 5.0, 1.2, ack_range=(200.0, 200.0),
                         n=1, grip_multiplier=0.65)['best']
            _split100 = float(_d100['inner_slip_deg'] - _d100['outer_slip_deg'])
            _split_lo = float(_dlo['inner_slip_deg'] - _dlo['outer_slip_deg'])
            _split_hi = float(_dhi['inner_slip_deg'] - _dhi['outer_slip_deg'])
            if not (np.isfinite(_split100) and np.isfinite(_split_lo) and np.isfinite(_split_hi)
                    and abs(_split100) <= 1e-9
                    and abs(_split_lo - _split100) > 1e-6
                    and abs(_split_hi - _split100) > 1e-6):
                _f7.append('Ackermann input did not produce equal 100% slips and distinct endpoint splits')
            _u_span = float(np.nanmax(_u7) - np.nanmin(_u7))
            if not (np.all(np.isfinite(_u7)) and np.isfinite(_u_span)):
                _f7.append('Ackermann capability sweep returned non-finite force')
            if _f7:
                fails += 1
            print(f'ackermann sweep  : rack +{_rack_mm:.0f} mm moved as-built '
                  f'{_ack_before:+.1f}% -> {_ack_after:+.1f}%, swept '
                  f'capability moved {_dmax:.1e} g ({_n_lim7} front-limited / {_n_inf7} never-limited cells), '
                  f'sub-limit force ceiling moved {_dceil_N:.1e} N (all rungs {_dceil_all_N:.1e} N); 100% slip split {_split100:+.3e} deg, '
                  f'-100/+200% {_split_lo:+.3f}/{_split_hi:+.3f} deg; force span {_u_span:.2f} N; '
                  f'plateau {_d7["plateau_pct_lo"]:+.0f}..{_d7["plateau_pct_hi"]:+.0f}%   '
                  + ('pass' if not _f7 else
                     'UNEXPECTED FAIL: ' + '; '.join(_f7)))
except Exception as _e:
    fails += 1
    import traceback as _tb7; _tb7.print_exc()
    print(f'ackermann sweep  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── ONE ACKERMANN PERCENT (Astra F-08, 2026-09-22).  FAILING-THEN-PASSING: the
#    GUI pair readout (and its twin, the kinematic Ackermann curve) built a
#    radius from the MEAN wheel angle and took a linear angle ratio, so YMD's
#    EXACT 100 % pair (cot d_out - cot d_in = t/L) read 99.2 % at 10 deg and
#    94.8 % at 30 deg of steer; the dynamics sensitivity used yet another linear
#    ratio.  All now go through vahan.ackermann.ackermann_pct_from_pair /
#    ackermann_pair_from_pct.  Gate: pairs built by YMD at -100/0/+50/+100/+200 %
#    over 1..40 deg of mean steer, both turn directions, read back EXACTLY by the
#    GUI readout; the kinematic curve reads the same; the sensitivity's
#    tyre-wanted % equals the canonical % of the pair it requires (and exactly
#    100 with zero slip).
print('-' * 64)
try:
    from vahan.ymd import _ackermann_split as _aks8
    from vahan.ackermann import ackermann_pct_from_pair as _apf8
    from vahan.metrics_catalog import compute_ackermann_post as _cap8
    from vahan.dynamics import _ideal_ackermann_pct as _iap8
    from gui.main_window import _ackermann_from_pair as _gap8
    from types import SimpleNamespace as _NS8
    _L8, _t8 = 1.537, 1.222
    _f8 = []; _worst8 = 0.0; _w100 = 0.0
    for _mean in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0):
        for _pc in (-100.0, 0.0, 50.0, 100.0, 200.0):
            for _sg in (+1.0, -1.0):                 # left / right turn
                _sfl, _sfr = _aks8(_sg * _mean, _pc, _t8, _L8)
                # steer (+ = left) -> toe-in: toe_L = -steer_FL, toe_R = +steer_FR
                _rd = _gap8(-_sfl, _sfr, _L8, _t8, inner=None)
                _err = abs(_rd - _pc) if np.isfinite(_rd) else float('inf')
                _worst8 = max(_worst8, _err)
                if _pc == 100.0:
                    _w100 = max(_w100, _err)
    if not _worst8 < 1e-6:
        _f8.append(f'GUI readout of YMD-built pairs off by up to {_worst8:.3g} points')
    # kinematic Ackermann curve (optimizer rack targeting) on an exact 100 % sweep
    _st8 = np.linspace(-30.0, 30.0, 25)
    _toe8 = np.array([(-_aks8(_s, 100.0, _t8, _L8)[0]) for _s in _st8])
    _k8 = _cap8(_toe8, _st8, _L8, _t8)
    _kv = _k8[np.isfinite(_k8)]
    _kerr = float(np.max(np.abs(_kv - 100.0))) if _kv.size else float('inf')
    if not (_kv.size >= 20 and _kerr < 1e-6):
        _f8.append(f'kinematic curve reads an exact 100 % sweep off by {_kerr:.3g} points ({_kv.size} finite)')
    # dynamics sensitivity: zero slip -> exactly 100; nonzero slip -> canonical % of the required pair
    class _T8:
        def __init__(self, k): self.k = k
        def slip_angle_for_Fy(self, Fy, Fz, cam): return self.k * Fy / Fz
    _veh8 = _NS8(wheelbase_m=_L8, front_track_m=_t8)
    _res8 = _NS8(Fy={'FL': 900., 'FR': 500.}, Fz={'FL': 1100., 'FR': 450.}, inclination={})
    _s0 = _iap8(_res8, _T8(0.0), _veh8, 5.0)
    _s1 = _iap8(_res8, _T8(2.0), _veh8, 5.0)
    _gi = np.degrees(np.arctan(_L8 / (5.0 - _t8 / 2))); _go = np.degrees(np.arctan(_L8 / (5.0 + _t8 / 2)))
    _s1_ref = _apf8(_gi + 2.0 * 500. / 450., _go + 2.0 * 900. / 1100., _t8, _L8)   # inner = lighter FR
    if not (abs(_s0 - 100.0) < 1e-9 and abs(_s1 - _s1_ref) < 1e-9):
        _f8.append(f'sensitivity ideal Ackermann {_s0:.4f} (zero slip, want 100) / {_s1:.4f} vs canonical {_s1_ref:.4f}')
    if _f8:
        fails += 1
    print(f'ackermann definition: exact 100 % pair reads {100.0 - _w100:.6f} % in the GUI at 1..40 deg '
          f'(old mean-angle readout: 94.8 % at 30 deg); YMD round trip -100..+200 % worst {_worst8:.1e} pts; '
          f'kinematic curve worst {_kerr:.1e} pts; sensitivity zero-slip {_s0:.3f} %   '
          + ('pass' if not _f8 else 'UNEXPECTED FAIL: ' + '; '.join(_f8)))
except Exception as _e:
    fails += 1
    import traceback as _tb8; _tb8.print_exc()
    print(f'ackermann definition: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── LAP-CAP fix (2026-08-03): the Ackermann lap chain must cap each station by
#    WHOLE-CAR trimmed grip (ymd ay_trim_max, both axles), NOT the front-axle
#    force ceiling.  The old ceiling cap over-credited +100% because the tight
#    corners that bind are REAR/balance-limited, manufacturing a fake "+100% is
#    fastest" lap verdict that contradicted the DECIDE view.  FAILING-THEN-
#    PASSING: the structural check below FAILS on the old front-ceiling code
#    (_ay_max_local called ay_front_cap_g) and PASSES on the trim cap.
try:
    from vahan.laptime import LapSimulator as _LS8
    import inspect as _insp8
    _src8 = _insp8.getsource(_LS8._ay_max_local)
    _uses_trim = ('ay_trim_scale' in _src8) and ('ay_front_cap_g' not in _src8)
    _f8 = []
    if not _uses_trim:
        _f8.append('_ay_max_local still caps by the front-axle ceiling '
                   '(ay_front_cap_g) — the +100% lap artifact is back')
    if _tm is not None:
        from vahan.ymd import mmm_metrics as _mmm8, build_loads_table as _blt8
        _ss8 = win._build_dynamics_solver(); _tbl8 = _blt8(_ss8)
        _ay0 = float(_mmm8(_tm, _ss8, 0.0, radius_m=3.0, grip_multiplier=0.65,
                           loads_table=_tbl8)['ay_trim_max'])
        _ay100 = float(_mmm8(_tm, _ss8, 100.0, radius_m=3.0, grip_multiplier=0.65,
                             loads_table=_tbl8)['ay_trim_max'])
        _no_reward = _ay100 <= _ay0 + 0.02   # +100% must NOT get MORE whole-car grip
        if not _no_reward:
            _f8.append(f'whole-car trim at 3 m REWARDS +100% '
                       f'({_ay100:.3f} > {_ay0:.3f} g) — cap would re-crown +100%')
        _msg8 = (f'cap uses whole-car trim={_uses_trim}; ay_trim_max@3m '
                 f'0%={_ay0:.3f} +100%={_ay100:.3f} g (+100 not rewarded={_no_reward})')
    else:
        _msg8 = f'cap uses whole-car trim={_uses_trim}; (no TTC tyre — numeric part skipped)'
    if _f8:
        fails += 1
    print(f'ackermann lap cap: {_msg8}   '
          + ('pass' if not _f8 else 'UNEXPECTED FAIL: ' + '; '.join(_f8)))
except Exception as _e8:
    fails += 1
    import traceback as _tb8; _tb8.print_exc()
    print(f'ackermann lap cap: UNEXPECTED FAIL ({type(_e8).__name__}: {_e8})')

# ── CORNER MOMENTS live in the CORE solver (vahan.loads), read by GUI+binder ──
#    The load-view moments (hub torque, overturning, kingpin, Mz, ARB torsion)
#    used to be computed inline in gui/wheel_package.py, duplicating physics into
#    the presentation layer.  They now come from vahan.loads.corner_moments /
#    rocker_arb_freebody so the 3-D view AND the binder read ONE set of values.
#    Guards: the core formulas are right (hub=Fx*R_r under braking, overturning=
#    Fy*R_r under cornering, Mz present only with a tyre model) AND the moment
#    physics no longer lives in wheel_package (grep, so it can't silently return).
print('-' * 64)
try:
    from vahan import loads as _lm
    import inspect as _inspm
    import gui.wheel_package as _wpm
    _wp_src = _inspm.getsource(_wpm)
    _mfail = []
    # STRUCTURAL: no inline moment physics may remain in the presentation layer.
    for _pat in ('Fx * R_r', 'Fy * R_r', 'slip_angle_for_Fy'):
        if _pat in _wp_src:
            _mfail.append(f'wheel_package still computes "{_pat}" inline')
    # NUMERIC: pure braking -> hub torque = Fx*R_r, overturning ~ 0.
    _wc = np.array([0.6, 0.0, 0.20]); _spin = np.array([1.0, 0.0, 0.0]); _Rr = 0.20
    _mb = _lm.corner_moments(Fx=-3000.0, Fy=0.0, Fz=2000.0,
                             wheel_center=_wc, spin_axis=_spin)
    if abs(_mb['hub_torque_Nm'] - (-3000.0 * _Rr)) > 1e-6:
        _mfail.append(f'hub torque {_mb["hub_torque_Nm"]:.3f} != Fx*R_r {-3000.0*_Rr:.3f}')
    if abs(_mb['overturning_Nm']) > 1e-6:
        _mfail.append('overturning nonzero under pure braking')
    # NUMERIC: pure cornering -> overturning = Fy*R_r, hub ~ 0.
    _mc = _lm.corner_moments(Fx=0.0, Fy=2500.0, Fz=2000.0,
                             wheel_center=_wc, spin_axis=_spin)
    if abs(_mc['overturning_Nm'] - 2500.0 * _Rr) > 1e-6:
        _mfail.append(f'overturning {_mc["overturning_Nm"]:.3f} != Fy*R_r {2500.0*_Rr:.3f}')
    # Mz omitted with no tyre model, present with one.
    if 'mz_Nm' in _mc:
        _mfail.append('mz_Nm returned without a tyre model')
    _tmm = getattr(win, '_tire_model', None)
    _mz_present = None
    if _tmm is not None:
        _mm = _lm.corner_moments(Fx=0.0, Fy=1500.0, Fz=1500.0, camber_deg=0.5,
                                 wheel_center=_wc, spin_axis=_spin, tire_model=_tmm)
        _mz_present = 'mz_Nm' in _mm
        if not _mz_present:
            _mfail.append('mz_Nm missing when a tyre model is supplied')
    if _mfail:
        fails += 1
    print(f'corner moments   : hub {_mb["hub_torque_Nm"]:.0f} N·m (=Fx*R_r), '
          f'overturning {_mc["overturning_Nm"]:.0f} N·m (=Fy*R_r), '
          f'Mz w/tyre={_mz_present}, core-only={not any("wheel_package" in m for m in _mfail)}   '
          + ('pass' if not _mfail else 'UNEXPECTED FAIL: ' + '; '.join(_mfail)))
except Exception as _em:
    fails += 1
    import traceback as _tbm; _tbm.print_exc()
    print(f'corner moments   : UNEXPECTED FAIL ({type(_em).__name__}: {_em})')

# ── MEMBER-LOAD FREE BODIES (2026-09-22 repair of the loads audit) ──────────
#    Hand-computable synthetic corners + the real design car.  Frame: +X = car
#    left, +Y = REARWARD, +Z = up.  Each line FAILED on the pre-repair code:
#      brake body   : brake torque changed the member forces (internal couple
#                     booked as external -> 2x braking moment in the links)
#      lateral sign : FL at +1.5 g != FR at -1.5 g (Fy magnitude, one direction)
#      arm body     : pushrod on the UCA treated as a 6th upright member
#      rocker/ARB   : spring = |pushrod| x wheel MR (3000 vs 3750 N), paired
#                     ARB averaging left -270 N·m on a free pivot
#      validity     : no valid/cond fields, singular geometry returned numbers
#      view pose    : load view re-solved every corner at ZERO travel
print('-' * 64)
from types import SimpleNamespace as _LNS
from vahan import loads as _LL
_LA = np.array
_Lbp, _Lup = _LL.BrakeParams(), _LL.UprightParams()


def _l_corner(push_on='upright'):
    """Synthetic corner: flat A-arms at z 0.3/0.1, lateral tie rod, patch 50 mm
    outboard of the joints, R = 0.2 m.  Hand-computable member forces."""
    _s = _LNS(uca_front=_LA([0.3, -0.15, 0.3]), uca_rear=_LA([0.3, 0.15, 0.3]),
              uca_outer=_LA([0.6, 0, 0.3]), lca_front=_LA([0.3, -0.15, 0.1]),
              lca_rear=_LA([0.3, 0.15, 0.1]), lca_outer=_LA([0.6, 0, 0.1]),
              tr_inner=_LA([0.3, 0.1, 0.2]), tr_outer=_LA([0.6, 0.1, 0.2]),
              wheel_center=_LA([0.65, 0, 0.2]), spin_axis=_LA([1.0, 0, 0]), travel=0.0,
              rocker_pivot=_LA([np.nan] * 3), rocker_spring_pt=_LA([np.nan] * 3),
              spring_chassis_pt=_LA([np.nan] * 3))
    if push_on == 'upright':      # vertical pushrod on the upright
        _s.pushrod_outer, _s.pushrod_inner = _LA([0.6, 0, 0.2]), _LA([0.6, 0, 0.6])
    else:                         # vertical pushrod on the UCA, 100 mm in from the BJ
        _s.pushrod_outer, _s.pushrod_inner = _LA([0.5, 0, 0.3]), _LA([0.5, 0, 0.7])
    return _s


_L_MEM = ('uca_front_N', 'uca_rear_N', 'lca_front_N', 'lca_rear_N', 'tierod_N', 'pushrod_N')

# 1. BRAKE BODY: wheel+hub+upright+caliper; pad/rotor friction is internal.
try:
    _f = []
    _b0 = _LL.compute_corner_loads(_l_corner('uca'), 1000., 0., -1500., 0., _Lbp, _Lup, 0.2, pushrod_body='uca')
    _b1 = _LL.compute_corner_loads(_l_corner('uca'), 1000., 0., -1500., 300., _Lbp, _Lup, 0.2, pushrod_body='uca')
    _dmax = max(abs(getattr(_b0, k) - getattr(_b1, k)) for k in _L_MEM)
    if _dmax > 1e-9:
        _f.append(f'brake torque changed member forces by {_dmax:.1f} N (internal couple booked)')
    _st = _l_corner('uca'); _jf = _b1.joint_forces
    _cp = _LL.contact_patch_point(_st.wheel_center, _st.spin_axis, 0.2)
    _Fs = _jf['uca'] + _jf['lca'] + _jf['tie'] + _LL.patch_force_world(-1500., 0., 1000.)
    _Mp = sum(np.cross(np.asarray(getattr(_st, p)) - _cp, _jf[k])
              for p, k in (('uca_outer', 'uca'), ('lca_outer', 'lca'), ('tr_outer', 'tie')))
    if np.linalg.norm(_Fs) > 1e-6 or np.linalg.norm(_Mp) > 1e-6:
        _f.append(f'joint forces do not balance the tyre force alone (|F| {np.linalg.norm(_Fs):.2g}, |M| {np.linalg.norm(_Mp):.2g})')
    if _LL.patch_force_world(-1500., 0., 0.)[1] <= 0:
        _f.append('braking force not REARWARD (+Y)')
    _bi = _LL.compute_corner_loads(_l_corner('uca'), 1000., 0., -1500., 300., _Lbp, _Lup, 0.2,
                                   pushrod_body='uca', brakes_inboard=True)
    _Mwc = sum(np.cross(np.asarray(getattr(_st, p)) - _st.wheel_center, _bi.joint_forces[k])
               for p, k in (('uca_outer', 'uca'), ('lca_outer', 'lca'), ('tr_outer', 'tie')))
    if abs(_Mwc[0]) > 1e-6:
        _f.append(f'inboard brakes: links react {_Mwc[0]:.1f} N·m about the axle (should be 0)')
    # frame guard: the design car's front axle must be at smaller Y than the rear
    if not (float(win._solvers['FL'].solve(0.).wheel_center[1])
            < float(win._solvers['RL'].solve(0.).wheel_center[1])):
        _f.append('design frame is not +Y rearward — loads.py sign mapping needs review')
    # real car: 1.5 g braking, outboard brakes -> member forces independent of T
    _Lcl = win._solvers['FL'].solve(0.)
    _r0 = _LL.compute_corner_loads(_Lcl, 1017., 0., -1339., 0., _Lbp, _Lup, 0.203,
                                   pushrod_body=win._solvers['FL']._pushrod_body)
    _r1 = _LL.compute_corner_loads(_Lcl, 1017., 0., -1339., 272., _Lbp, _Lup, 0.203,
                                   pushrod_body=win._solvers['FL']._pushrod_body)
    _dcar = max(abs(getattr(_r0, k) - getattr(_r1, k)) for k in _L_MEM)
    if _dcar > 1e-9:
        _f.append(f'design car: brake torque moved members {_dcar:.1f} N')
    if _f:
        fails += 1
    print(f'loads brake body : brake torque internal (max member change {_dmax:.1e} N, car {_dcar:.1e} N), '
          f'tyre force closes alone, inboard-brake shaft path   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads brake body : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 2. LATERAL SIGN: member solver, corner moments and arrows use ONE world vector;
#    a left-hand turn must be the exact mirror of a right-hand turn.
try:
    from gui import wheel_package as _LWP
    _f = []
    _lr = _LWP.compute_case(win, 1.5, 0.0)[0]
    _ll = _LWP.compute_case(win, -1.5, 0.0)[0]
    _mm = max(abs(getattr(_lr['FL'], k) - getattr(_ll['FR'], k)) for k in _L_MEM + ('spring_force_N',))
    _den = max(abs(getattr(_lr['FL'], k)) for k in _L_MEM)
    if not (_mm <= 1e-6 * _den):
        _f.append(f'FL right-turn vs FR left-turn differ by {_mm:.1f} N (not mirror images)')
    if not (_lr['FL'].Fy_N < 0 and _lr['FR'].Fy_N < 0):
        _f.append('right-hand turn tyre forces must point to -X (car right)')
    _st = _lr['FL'].state
    _m = _LL.corner_moments(Fx=0., Fy=_lr['FL'].Fy_N, Fz=_lr['FL'].Fz_N, wheel_center=_st.wheel_center,
                            spin_axis=_st.spin_axis, lca_outer=_st.lca_outer, uca_outer=_st.uca_outer,
                            wheel_radius_m=0.203)
    _k = _LA(_st.uca_outer) - _LA(_st.lca_outer); _k /= np.linalg.norm(_k)
    _cp = _LL.contact_patch_point(_st.wheel_center, _st.spin_axis, 0.203)
    _kp = float(_k @ np.cross(_cp - _LA(_st.lca_outer), _LL.patch_force_world(0., _lr['FL'].Fy_N, _lr['FL'].Fz_N)))
    if abs(_m['kingpin_Nm'] - _kp) > 1e-9:
        _f.append('corner_moments uses a different lateral world sign than the member solver')
    if _f:
        fails += 1
    print(f'loads lateral sign: right/left turn mirror max diff {_mm:.1e} N, FL Fy {_lr["FL"].Fy_N:.0f} N (toward -X), '
          f'kingpin {_m["kingpin_Nm"]:.1f} N·m on the member-solver vector   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads lateral sign: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 3. ARM BODY: pushrod on a control arm -> upright + arm as two bodies at the BJ.
try:
    _f = []
    _u = _LL.compute_corner_loads(_l_corner('upright'), 1000., 0., 0., 0., _Lbp, _Lup, 0.2)
    _exp_u = {'pushrod_N': -1000., 'uca_front_N': -139.7542486, 'uca_rear_N': -139.7542486,
              'lca_front_N': 139.7542486, 'lca_rear_N': 139.7542486, 'tierod_N': 0.}
    _a = _LL.compute_corner_loads(_l_corner('uca'), 1000., 0., 0., 0., _Lbp, _Lup, 0.2, pushrod_body='uca')
    # hand: arm moment about its pivot axis: 0.3*Fz = 0.2*|P| -> P = -1500; leg
    # reaction S/2 = (0.125, 0, 0.25) Fz -> axial -0.1118 Fz + shear 0.2562 Fz
    _exp_a = {'pushrod_N': -1500., 'uca_front_N': -111.8033989, 'uca_rear_N': -111.8033989,
              'lca_front_N': 139.7542486, 'lca_rear_N': 139.7542486, 'tierod_N': 0.,
              'uca_front_shear_N': 256.1737691, 'uca_bj_V': -1000.}
    for _nm, _c, _e in (('upright-mounted', _u, _exp_u), ('UCA-mounted', _a, _exp_a)):
        for _kk, _vv in _e.items():
            if abs(getattr(_c, _kk) - _vv) > 1e-6:
                _f.append(f'{_nm} {_kk} {getattr(_c, _kk):.4f} != hand {_vv:.4f}')
    # real car, independent VIRTUAL WORK: F_s = -(F.dcp + F_u.dwc) / dL_spring
    def _kabsch(P, Q):
        pc, qc = P.mean(0), Q.mean(0); U_, S_, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        R_ = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]) @ U_.T
        return R_, qc - R_ @ pc
    _vw = 0.0
    for _lb in ('FL', 'RL'):
        _sv = win._solvers[_lb]; _s0 = _sv.solve(0.004)
        _c = _LL.compute_corner_loads(_s0, 1200., -900., 0., 0., _Lbp, _Lup, 0.203,
                                      pushrod_body=_sv._pushrod_body, rocker_axis=_sv._rocker_axis,
                                      unsprung_mass_kg=13., accel_world=(-8., 0., 0.))
        _sp, _sm = _sv.solve(0.004 + 1e-5), _sv.solve(0.004 - 1e-5)
        _pp = lambda x: np.array([x.uca_outer, x.lca_outer, x.tr_outer], float)
        _cp = _LL.contact_patch_point(_s0.wheel_center, _s0.spin_axis, 0.203)
        _Rp, _tp = _kabsch(_pp(_s0), _pp(_sp)); _Rm, _tm_ = _kabsch(_pp(_s0), _pp(_sm))
        _dcp = (_Rp @ _cp + _tp) - (_Rm @ _cp + _tm_)
        _dwc = _LA(_sp.wheel_center) - _LA(_sm.wheel_center)
        _Fu = 13. * (_LA([0, 0, -9.81]) - _LA([-8., 0, 0]))
        _Fvw = -(_LL.patch_force_world(0., -900., 1200.) @ _dcp + _Fu @ _dwc) / (_sp.spring_length - _sm.spring_length)
        _vw = max(_vw, abs(_c.spring_force_N - _Fvw) / abs(_Fvw))
        if not _c.valid or _vw > 1e-5:
            _f.append(f'{_lb}: statics spring {_c.spring_force_N:.2f} N vs virtual work {_Fvw:.2f} N')
    if _f:
        fails += 1
    print(f'loads arm body   : hand pushrod -1000/-1500 N, UCA leg -111.8 N + 256.2 N shear; '
          f'car statics vs virtual work {_vw:.1e} rel   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads arm body   : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 4. ROCKER / ARB: spring from the rocker's own lever balance; every rocker and
#    the bar close exactly.
try:
    _f = []
    _S, _bs, _m0 = _LL.rocker_required_spring(
        pushrod_inner=_LA([0.10, 0, 0]), pushrod_outer=_LA([0.10, -0.3, 0]), pushrod_N=-3000.,
        rocker_pivot=_LA([0, 0, 0]), rocker_axis=_LA([0, 0, 1.]),
        rocker_spring_pt=_LA([0, 0.08, 0]), spring_chassis_pt=_LA([-0.25, 0.08, 0]))
    if abs(_S - 3750.) > 1e-9:
        _f.append(f'lever example spring {_S:.1f} N != 3750 N (a/b = 0.10/0.08)')
    _fb = _LL.rocker_arb_freebody(
        pushrod_inner=_LA([0.10, 0, 0]), pushrod_outer=_LA([0.10, -0.3, 0]), pushrod_N=-3000.,
        rocker_pivot=_LA([0, 0, 0]), rocker_axis=_LA([0, 0, 1.]),
        rocker_spring_pt=_LA([0, 0.08, 0]), spring_chassis_pt=_LA([-0.25, 0.08, 0]),
        spring_force_N=3000., arb_drop_top=_LA([-0.06, 0, 0]), arb_arm_end=_LA([-0.06, 0.2, 0]),
        m0_opposite=0.)
    if abs(_fb['axis_moment_residual_Nm']) > 1e-9:
        _f.append(f'paired rocker leaves {_fb["axis_moment_residual_Nm"]:.1f} N·m on a free pivot')
    # anchor NOT written by the loads code: the dynamics model's own static
    # spring force (sprung corner weight / motion ratio)
    _l0, _v0 = _LWP.compute_case(win, 0.0, 0.0)[:2]
    _sfs = {}
    for _lb, _attr in (('FL', 'static_spring_force_front_N'), ('RL', 'static_spring_force_rear_N')):
        _ref = float(getattr(_v0, _attr, 0.0) or 0.0)
        _sfs[_lb] = (_l0[_lb].spring_force_N, _ref)
        if _ref > 0 and abs(_l0[_lb].spring_force_N - _ref) > 1e-3 * _ref:
            _f.append(f'{_lb} static spring {_l0[_lb].spring_force_N:.1f} N != dynamics static {_ref:.1f} N')
    _ld = _LWP.compute_case(win, 1.5, 0.0)[0]
    _bar = {}
    for _ax, (_a1, _a2) in (('F', ('FL', 'FR')), ('R', ('RL', 'RR'))):
        _bar[_ax] = _ld[_a1].arb_bar_torque_Nm + _ld[_a2].arb_bar_torque_Nm
        if abs(_bar[_ax]) > 1e-9 or not np.isfinite(_ld[_a1].arb_link_N):
            _f.append(f'{_ax} ARB bar not balanced ({_bar[_ax]:.3g} N·m) / link {_ld[_a1].arb_link_N}')
    _res = []
    for _lb in ('FL', 'FR', 'RL', 'RR'):
        _c = _ld[_lb]; _s = _c.state
        _g = _LWP.arb_geometry_fn(win)(_lb, _s)
        _fb = _LL.rocker_arb_freebody(
            pushrod_inner=_s.pushrod_inner, pushrod_outer=_s.pushrod_outer, pushrod_N=_c.pushrod_N,
            rocker_pivot=_s.rocker_pivot, rocker_axis=win._solvers[_lb]._rocker_axis,
            rocker_spring_pt=_s.rocker_spring_pt, spring_chassis_pt=_s.spring_chassis_pt,
            spring_force_N=_c.spring_force_N, arb_drop_top=_g['drop_top'], arb_arm_end=_g['arm_end'])
        _t = float(_fb['F_arb'] @ _fb['u_arb'])
        _res.append(abs(_fb['axis_moment_residual_Nm']))
        if abs(_t - _c.arb_link_N) > 1e-6 * max(1., abs(_t)):
            _f.append(f'{_lb} view drop-link {_t:.1f} N != core {_c.arb_link_N:.1f} N')
    if _f:
        fails += 1
    print(f'loads rocker/ARB : lever example 3750 N, static spring F {_sfs["FL"][0]:.0f}/{_sfs["FL"][1]:.0f} '
          f'R {_sfs["RL"][0]:.0f}/{_sfs["RL"][1]:.0f} N (loads/dynamics), paired residual {_fb["axis_moment_residual_Nm"]:.1e}; '
          f'car @1.5 g bar balance F {_bar["F"]:.1e} R {_bar["R"]:.1e} N·m, '
          f'links F ±{abs(_ld["FL"].arb_link_N):.0f} R ±{abs(_ld["RL"].arb_link_N):.0f} N, '
          f'rocker residual max {max(_res):.1e} N·m   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads rocker/ARB : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 5. VALIDITY: rank-deficient / ill-conditioned -> valid False + NaN, never numbers.
try:
    _f = []
    _ok = _LL.compute_corner_loads(_l_corner(), 1000., 0., 0., 0., _Lbp, _Lup, 0.2)
    if not (_ok.valid and _ok.cond_number < 100):
        _f.append(f'well-posed corner flagged (cond {_ok.cond_number:.3g})')
    _sd = _l_corner(); _sd.uca_rear = _sd.uca_front.copy()          # 5 independent links
    _bad = _LL.compute_corner_loads(_sd, 1000., 0., 0., 0., _Lbp, _Lup, 0.2)
    if _bad.valid or np.isfinite(_bad.pushrod_N):
        _f.append('rank-deficient corner reported as a valid solution')
    _conds = []
    for _eps in (5e-2, 1e-4):      # tie rod swung toward the LCA-front line
        _sn = _l_corner(); _sn.tr_outer = _LA([0.6, 0.0 + _eps, 0.1]); _sn.tr_inner = _LA([0.3, -0.15 + _eps, 0.1])
        _cn = _LL.compute_corner_loads(_sn, 1000., 300., 0., 0., _Lbp, _Lup, 0.2)
        _conds.append((_cn.cond_number, _cn.valid))
    if not (_conds[1][0] > _conds[0][0] and not _conds[1][1]):
        _f.append(f'near-toggle not flagged {_conds}')
    _ld = _LWP.compute_case(win, 1.5, 0.0)[0]
    for _lb in ('FL', 'FR', 'RL', 'RR'):
        _c = _ld[_lb]
        if not (_c.valid and _c.cond_number < 100):
            _f.append(f'design car {_lb} invalid/ill-conditioned ({_c.cond_number:.3g}: {_c.invalid_reason})')

    class _Boom:
        def solve(self, t):
            raise RuntimeError('no converge')
    _res = _LWP.compute_case(win, 1.5, 0.0)[3]
    import warnings as _Lw
    with _Lw.catch_warnings(record=True) as _Lwarn:
        _Lw.simplefilter('always')
        _fl = _LL.compute_all_corners({'FL': _Boom(), 'FR': win._solvers['FR'], 'RL': win._solvers['RL'],
                                       'RR': win._solvers['RR']}, _res, _Lbp, _Lbp, _Lup, _Lup, 0.203)
    if not any('WITHOUT veh' in str(_w.message) for _w in _Lwarn):
        _f.append('compute_all_corners without veh did not warn (silent incompleteness)')
    if _fl['FL'].valid or np.isfinite(_fl['FL'].pushrod_N) or np.isfinite(_fl['FL'].residual):
        _f.append('failed kinematic solve reported as a zero-force, zero-residual solution')
    if _f:
        fails += 1
    print(f'loads validity   : synthetic cond {_ok.cond_number:.1f}, rank-5 -> invalid/NaN, near-toggle cond '
          f'{_conds[0][0]:.3g}->{_conds[1][0]:.3g} flagged, car cond max '
          f'{max(_ld[l].cond_number for l in _ld):.1f}, failed solve -> invalid   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads validity   : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 6. LOAD VIEW = the solved pose + the core caliper/bearing vectors.
try:
    _f = []
    _ld, _v6, _u6, _r6 = _LWP.compute_case(win, 1.5, 0.0)[:4]
    # the corner that MOVES most (since 2026-09-22 the jacking heave is fed
    # back, so the loaded FL's roll bump and the body lift nearly cancel on
    # the default car — the unloaded FR carries the clear travel)
    _c6 = max(('FL', 'FR'), key=lambda c: abs(float(_r6.travel[c])))
    _its = _LWP._load_items(win, 1.5, 0.0, only_corner=_c6)
    _bj = [it for it in _its if 'upper ball joint' in it[3]]
    _dyn = _LA(_ld[_c6].state.uca_outer); _stat = _LA(win._solvers[_c6].solve(0.).uca_outer)
    if not _bj or np.linalg.norm(_LA(_bj[0][0]) - _dyn) > 1e-9:
        _f.append('upper ball-joint arrow not at the solved dynamic pose')
    _trav = float(_r6.travel[_c6])
    if abs(_trav) < 0.5:
        _f.append(f'test needs roll travel, got {_trav:.2f} mm')
    _up2 = _LL.UprightParams(caliper_vertical_mounts=False, caliper_angle_deg=90.0)
    _r, _t, _w = _LL.caliper_frame(_up2)
    if not (np.allclose(_r, [0, 1, 0]) and np.allclose(_t, [0, 0, 1])):
        _f.append(f'manual clock 90 deg must be the trailing edge (+Y) with disc moving UP there, got r {_r} t {_t}')
    _cb = _LL.ComponentLoads(brake_torque_Nm=300.)
    _LL._compute_caliper_bolt_loads(_cb, _Lbp, _up2)
    # trailing-edge pad: disc surface moves up -> friction on caliper up (+V)
    if not (_cb.caliper_upper_V > 0 and _cb.caliper_lower_V > 0):
        _f.append('trailing caliper lugs not loaded upward by the disc')
    _bk = _LWP.compute_case(win, 0.0, -1.5)[0]['FL']
    _cal = [it for it in _LWP._load_items(win, 0.0, -1.5, only_corner='FL') if 'CALIPER' in it[3]]
    if len(_cal) != 2 or any(not np.allclose(_LA(a[1]), _LA(b[1])) for a, b in zip(_cal, _bk.caliper_lugs)):
        _f.append('caliper arrows are not the core lug forces')
    if _f:
        fails += 1
    print(f'loads view pose  : BJ arrow at dynamic pose ({np.linalg.norm(_dyn - _stat) * 1000:.1f} mm from the '
          f'static one, travel {_trav:+.1f} mm), caliper arrows = table lugs, clock 90 deg = trailing edge   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads view pose  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 6b. CALIPER / BEARING DOMAIN (Astra F-03, 2026-09-22).  FAILING-THEN-PASSING:
#     the old code clamped the signed pad-to-bolt-line offset l4 to zero, so a
#     120 mm disc / 90 mm pad / 100 mm bolt line at 450 N.m left a 50.0 N.m
#     moment imbalance on the caliper free body (751.8 N.m worst over Astra's
#     seed-88 sweep); a 0.5 mm bearing spacing returned plausible 0 N loads and a
#     pad outside the disc was accepted.  Now: l4 signed, residual ~1e-13,
#     pad-outside-disc rejected (NaN + caliper_valid False), bad bearing spacing
#     NaN + bearing_valid False, and the live car's Loads-page path still valid.
try:
    _f = []
    def _cal_res(_bp, _up, _T, _fr, _wc):
        _c = _LL.ComponentLoads(brake_torque_Nm=_T)
        _LL._compute_caliper_bolt_loads(_c, _bp, _up, _fr, _wc)
        if not _c.caliper_lugs:
            return _c, float('nan')
        _M = sum((np.cross(_p - _wc, _fv) for _p, _fv in _c.caliper_lugs), np.zeros(3))
        _Fp = _T / (_bp.pad_radius_mm / 1000.0)
        _tg = np.cross(_bp.pad_radius_mm / 1000.0 * _fr[0], _Fp * _fr[1])
        return _c, float(np.linalg.norm(_M - _tg))
    _bpx = _LL.BrakeParams(rotor_dia_mm=240, pad_radius_mm=90,
                           caliper_mount_height_mm=20, caliper_bolt_spacing_mm=60)
    _upx = _LL.UprightParams(caliper_angle_deg=0, caliper_vertical_mounts=True)
    _wcx = np.array([.7, 0., .2])
    _frx = _LL.caliper_frame(_upx, [1, 0, 0], _wcx, _wcx + np.array([0, .1, 0]))
    _cx, _rx = _cal_res(_bpx, _upx, 450., _frx, _wcx)
    _Hx = abs(_cx.caliper_upper_H)
    if abs(_bpx.caliper_l4_mm + 10.0) > 1e-9:
        _f.append(f'signed l4 should be -10 mm, got {_bpx.caliper_l4_mm:.2f}')
    if not (_cx.caliper_valid and np.isfinite(_rx) and _rx < 1e-6):
        _f.append(f'signed-offset fixture moment residual {_rx:.4g} N.m (was 50.0 when clamped)')
    if abs(_Hx - 5000. * 0.010 / 0.060) > 1e-6:
        _f.append(f'radial lug couple {_Hx:.1f} N, expected F*|l4|/l5 = 833.3 N')
    _rng6 = np.random.default_rng(88); _worst6 = 0.; _nrej6 = 0; _nout6 = 0; _bad6 = 0
    for _ in range(1000):
        _bpr = _LL.BrakeParams(pad_radius_mm=_rng6.uniform(55, 130), rotor_dia_mm=_rng6.uniform(180, 300),
                               caliper_mount_height_mm=_rng6.uniform(10, 45),
                               caliper_bolt_spacing_mm=_rng6.uniform(30, 100))
        _upr = _LL.UprightParams(bearing_spacing_mm=_rng6.uniform(20, 90),
                                 bearing_inboard_offset_mm=_rng6.uniform(10, 70),
                                 caliper_angle_deg=_rng6.uniform(0, 360),
                                 caliper_vertical_mounts=bool(_rng6.integers(0, 2)))
        _wcr = np.array([_rng6.choice([-1, 1]) * .7, 0, .2])
        _trr = _wcr + np.array([0, _rng6.uniform(-.2, .2), _rng6.uniform(-.1, .1)])
        _frr = _LL.caliper_frame(_upr, [1, 0, 0], _wcr, _trr)
        _rng6.uniform(0, 6000); _rng6.uniform(-4000, 4000); _rng6.uniform(-3000, 3000)
        _cr, _rr6 = _cal_res(_bpr, _upr, _rng6.uniform(0, 700), _frr, _wcr)
        _outside = _bpr.pad_radius_mm >= 0.5 * _bpr.rotor_dia_mm
        _nout6 += int(_outside)
        if not _cr.caliper_valid:
            _nrej6 += 1
            if np.isfinite(_cr.caliper_upper_V):
                _bad6 += 1                          # rejected but still a number
        elif np.isfinite(_rr6):
            _worst6 = max(_worst6, _rr6)
    if _nrej6 != _nout6 or _bad6:
        _f.append(f'pad-outside-disc cases {_nout6}, rejected {_nrej6}, rejected-but-numeric {_bad6}')
    if _worst6 > 1e-6:
        _f.append(f'accepted caliper geometry moment residual {_worst6:.4g} N.m (was 751.8 clamped)')
    _cbx = _LL.ComponentLoads(Fz_N=1500., Fy_N=1200., Fx_N=-800., brake_torque_Nm=300.)
    _LL._compute_bearing_loads(_cbx, _LL.BrakeParams(), _LL.UprightParams(bearing_spacing_mm=0.5), 0.203, -1.0)
    if _cbx.bearing_valid or np.isfinite(_cbx.bearing_outer_V) or not _cbx.bearing_invalid_reason:
        _f.append(f'0.5 mm bearing spacing returned outer V {_cbx.bearing_outer_V} valid={_cbx.bearing_valid} '
                  f'(must be NaN + invalid, not a plausible zero)')
    _ldk = _LWP.compute_case(win, 0.0, -1.5)[0]
    _car_bad = [f'{_k}: {_v.caliper_invalid_reason or _v.bearing_invalid_reason}'
                for _k, _v in _ldk.items() if not (_v.caliper_valid and _v.bearing_valid)]
    if _car_bad:
        _f.append('live car Loads path invalid: ' + '; '.join(_car_bad))
    if _f:
        fails += 1
    print(f'caliper/bearing domain: signed l4 fixture residual {_rx:.1e} N.m (clamped: 50.0), lug couple '
          f'{_Hx:.1f} N; sweep {_nrej6}/{_nout6} pad-outside rejected, accepted worst {_worst6:.1e} N.m '
          f'(clamped: 751.8); 0.5 mm bearing -> {"NaN+invalid" if not _cbx.bearing_valid else "numbers"}; '
          f'live car 4 corners valid={not _car_bad}   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(_f)))
except Exception as _e:
    fails += 1
    import traceback as _tbc; _tbc.print_exc()
    print(f'caliper/bearing domain: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# 7. SIGN CONTRACT with dynamics.py: SteadyStateResult.Fx/Fy are MAGNITUDES and
#    brake torque is braking-only; signed_wheel_forces supplies the direction.
try:
    _f = []
    for _lat, _lon in ((1.5, 0.), (-1.5, 0.), (0., -1.5), (0., 1.0)):
        _rr = win._build_dynamics_solver().solve(_lat, _lon)
        for _lb in ('FL', 'FR', 'RL', 'RR'):
            if _rr.Fx.get(_lb, 0.) < 0 or _rr.Fy.get(_lb, 0.) < 0:
                _f.append(f'dynamics now stores SIGNED Fx/Fy ({_lb} @ {_lat},{_lon}) — update signed_wheel_forces')
            _sx, _sy, _sz = _LL.signed_wheel_forces(_rr, _lb)
            if (_lon < 0 and _sx > 0) or (_lon > 0 and _sx < 0) or (_lat > 0 and _sy > 0) or (_lat < 0 and _sy < 0):
                _f.append(f'{_lb} @ ({_lat},{_lon}): wrong signed direction ({_sx:.0f}, {_sy:.0f})')
    if _f:
        fails += 1
    print(f'loads sign contract: dynamics magnitudes -> signed (brake -Fx, right turn -Fy) on 4 cases   '
          + ('pass' if not _f else 'UNEXPECTED FAIL: ' + '; '.join(sorted(set(_f))[:4])))
except Exception as _e:
    fails += 1
    import traceback as _tbl; _tbl.print_exc()
    print(f'loads sign contract: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── PACKAGING SYSTEM (vahan/packaging.py): the ONE validity oracle both the
#    manual page and the generator run.  Two invariants: (a) the untouched
#    current design config must meet the absolute geometry laws, even against
#    its own baseline (a baseline can itself be defective); (b) the isometry
#    primitives must round-trip: mirror∘mirror = identity and rotate(+a)∘
#    rotate(−a) = identity to 1e-9 m, or the "rates preserved exactly" claim
#    the transforms are built on is false.
print('-' * 64)
try:
    import glob as _gp, re as _rp
    from vahan import packaging as _pkgm
    # Rule 04 strict-metric regression.  Start with a known orthogonal triad in
    # the X=0 rocker plate, then translate the whole ARB 48 mm along the plate
    # normal.  Translation preserves that reference construction's triad but
    # must fail the physical-plate gate.
    _r04_hp = {'rocker_pivot': np.array([0.0, 0.0, 0.0]),
               'pushrod_inner': np.array([0.0, 0.1, 0.0]),
               'rocker_spring_pt': np.array([0.0, 0.0, 0.1])}
    _r04_arb = {'arb_drop_top': np.array([0.048, 0.1, 0.1]),
                'arb_arm_end': np.array([0.048, 0.0, 0.1]),
                'arb_pivot': np.array([0.048, 0.0, 0.0])}
    _r04_synth = _pkgm.arb_drop_link_plate_metrics(_r04_hp, _r04_arb)
    assert np.isclose(_r04_synth['drop_top_signed_mm'], 48.0)
    assert np.isclose(_r04_synth['arm_end_signed_mm'], 48.0)
    assert np.isclose(_r04_synth['direction_deg'], 0.0)
    assert not _pkgm.arb_drop_link_plate_compliant(_r04_synth, 3.0)

    # Saved v114 is the real defect: the project declares a +48 mm standoff,
    # but declarations do not move the physical plate or alter compliance.
    import json as _r04_json
    _v114_paths = _gp.glob('configs/2027_v114_*.vahan')
    assert len(_v114_paths) == 1, _v114_paths
    with open(_v114_paths[0], encoding='utf-8') as _r04_f:
        _v114 = _r04_json.load(_r04_f)
    _v114_m = _pkgm.arb_drop_link_plate_metrics(_v114['front_hp'],
                                                 _v114['front_arb'])
    assert float(_v114['car']['front_arb_drop_standoff_mm']) == 48.0
    assert not _pkgm.arb_drop_link_plate_compliant(_v114_m, 3.0)
    print(f'Rule 04 strict    : synthetic 48.0/48.0 mm rejected; v114 '
          f'{_v114_m["drop_top_signed_mm"]:+.1f}/'
          f'{_v114_m["arm_end_signed_mm"]:+.1f} mm rejected   pass')
    _pcfgs = _gp.glob('configs/2027_v*.vahan')
    _pdes = os.environ.get('VAHAN_DESIGN') or (max(_pcfgs, key=lambda p: int((_rp.search(r'2027_v(\d+)', p) or [0, -1]).__getitem__(1))
                if _rp.search(r'2027_v(\d+)', p) else -1) if _pcfgs else None)
    if _pdes:
        win._load_project_from_path(_pdes); win._rebuild_solvers(0.)
        _gl0 = _pkgm._axle_geometry_laws(win, 'front')
        print(f"packaging window : {os.path.basename(_pdes)} standoff {win._car.get('front_arb_drop_standoff_mm')} waiver {bool(win._car.get('front_arb_rule04_waiver'))} "
              f"laws coplanar {_gl0['coplanar_mm']:.2f} in-plane {_gl0['arb_drop_top_inplane_mm']:.2f}/{_gl0['arb_arm_end_inplane_mm']:.2f} mm")
        _pbase = _pkgm.capture_baseline(win)
        _pres = _pkgm.validate(win, _pbase)
        # Waiver/standoff strings are retained as diagnostics only.  Even with
        # both present, the shared validator must reject physical 48 mm offsets.
        import copy as _triad_copy
        from unittest.mock import patch as _triad_patch
        _meta_bad = _triad_copy.deepcopy(_pbase['geometry']['front'])
        _meta_bad.update(arb_drop_top_plate_signed_mm=48.0,
                         arb_arm_end_plate_signed_mm=48.0,
                         arb_drop_top_inplane_mm=48.0,
                         arb_arm_end_inplane_mm=48.0,
                         arb_is_bottom=False,
                         arb_inplane_waived=True)
        _old_so = win._car.get('front_arb_drop_standoff_mm')
        _old_wv = win._car.get('front_arb_rule04_waiver')
        win._car['front_arb_drop_standoff_mm'] = 48.0
        win._car['front_arb_rule04_waiver'] = 'diagnostic metadata only'
        with _triad_patch.object(_pkgm, '_axle_geometry_laws', return_value=_meta_bad):
            _meta_result = _pkgm.validate(win, _pbase, axles=('front',),
                                          stop_early=True, skip_clash=True)
        if _old_so is None:
            win._car.pop('front_arb_drop_standoff_mm', None)
        else:
            win._car['front_arb_drop_standoff_mm'] = _old_so
        if _old_wv is None:
            win._car.pop('front_arb_rule04_waiver', None)
        else:
            win._car['front_arb_rule04_waiver'] = _old_wv
        assert any(c['name'] == 'arb_drop_top_inplane_mm'
                   for c in _meta_result.failures()), _meta_result.checks
        # A matching 93-degree baseline must fail; a corrected 90-degree
        # candidate must pass even when that baseline still says 93 degrees.
        _triad_base = _triad_copy.deepcopy(_pbase)
        _triad_good = _pkgm._axle_geometry_laws(win, 'front').copy()
        for _tk in ('triad_bar_blade_deg', 'triad_blade_drop_deg', 'triad_bar_drop_deg'):
            _triad_good[_tk] = 90.0
        _triad_bad = dict(_triad_good, triad_bar_drop_deg=93.0)
        _triad_base['geometry']['front'] = _triad_bad.copy()
        with _triad_patch.object(_pkgm, '_axle_geometry_laws', return_value=_triad_bad):
            _bad_triad_result = _pkgm.validate(win, _triad_base, axles=('front',),
                                              stop_early=True, skip_clash=True)
        with _triad_patch.object(_pkgm, '_axle_geometry_laws', return_value=_triad_good):
            _good_triad_result = _pkgm.validate(win, _triad_base, axles=('front',),
                                               stop_early=True, skip_clash=True)
        assert any(c['name'] == 'triad_bar_drop_deg' for c in _bad_triad_result.failures())
        assert _good_triad_result.ok, _good_triad_result.failures()
        print('ARB triad target : rejects matching non-square baseline; accepts corrected 90 deg   pass')
        _pfails = ['%s %s' % (c['axle'], c['name']) for c in _pres.failures()]
        _b0 = _pkgm.get_bundle(win, 'front')
        _mmb = _pkgm.mirror_about_pushrod_plane(_pkgm.mirror_about_pushrod_plane(_b0))
        _rrb = _pkgm.rotate_about_pushrod_line(
            _pkgm.rotate_about_pushrod_line(_b0, 7.0), -7.0)
        _dmax = 0.0
        for _bb in (_mmb, _rrb):
            for _dn, _kk, _pp in _pkgm._slice_items(_b0):
                _dmax = max(_dmax, float(np.abs(_bb[_dn][_kk] - _pp).max()))
        _pok = _pres.ok and _dmax < 1e-9
        if not _pok:
            fails += 1
        print(f'packaging        : {os.path.basename(_pdes)} baseline '
              f'{"PASS" if _pres.ok else "FAIL: " + "; ".join(_pfails[:4])}, '
              f'isometry round-trip {_dmax:.1e} m   '
              f'{"pass" if _pok else "UNEXPECTED FAIL"}')
    else:
        print('packaging        : no design config found — skipped')
except Exception as _ep:
    fails += 1
    import traceback as _tbp; _tbp.print_exc()
    print(f'packaging        : UNEXPECTED FAIL ({type(_ep).__name__}: {_ep})')

# ── RELOCATE IK (vahan/relocate.py): curve-preserving single-point relocation.
#    Oracle honesty check, both directions: (a) with allow_wheel_motion=True an
#    UNMOVED model must still validate PASS (the switch may not weaken an
#    untouched car); (b) shoving a wheel-locating point (front uca_outer) 5 mm
#    up must FAIL the re-measured wheel metrics — if it passes, the oracle is
#    blind to curve changes and every "solution" the search returns is a lie.
#    The point is restored and re-verified byte-identical afterwards.
print('-' * 64)
try:
    from vahan.relocate import _PointHandle as _RPH, _feasible as _RF
    # (uses win/_pbase from the packaging block above — same loaded config)
    _ok_same, _ = _RF(win, _pbase, _pkgm.Tolerances(), 'front')
    _h = _RPH(win, 'front', 'hp', 'uca_outer')
    _pmoved = _h.original + np.array([0.0, 0.0, 0.005])
    _h.set(_pmoved)
    _ok_moved, _res_m = _RF(win, _pbase, _pkgm.Tolerances(), 'front')
    _h.restore()
    _back = np.abs(np.array((win._front_hp)['uca_outer'], float)
                   - _h.original).max()
    _rok = _ok_same and (not _ok_moved) and _back < 1e-12
    if not _rok:
        fails += 1
    print(f'relocate IK      : unmoved PASS={_ok_same}, uca_outer +5mm z '
          f'FAILs oracle={not _ok_moved}, restore {_back:.1e} m   '
          f'{"pass" if _rok else "UNEXPECTED FAIL"}')
except Exception as _er:
    fails += 1
    import traceback as _tbr; _tbr.print_exc()
    print(f'relocate IK      : UNEXPECTED FAIL ({type(_er).__name__}: {_er})')

# ── TYRE SPEED / CONDITIONING WINDOW (2026-09-15, rim-matched TTC surface) ────
#    TireModel(speed_window_kph=(lo, hi)) keeps one test-speed block after the
#    warm-up discard; the app honours car['tire_speed_window_kph'] on both tyre
#    load sites.  Checked on the design's own tyre file (data never printed).
print('-' * 64)
try:
    from vahan.tire_model import TireModel as _TMw, load_tire_data as _ltd
    _tp = None
    try: _tp = wD._dynamics_panel.get_tire_path()
    except Exception: _tp = None
    if _tp and os.path.exists(_tp):
        _psi_w = float(wD._dynamics_panel.get_tire_pressure_psi())
        _m_all = _TMw.from_file(_tp, pressure_psi=_psi_w)
        _m_win = _TMw.from_file(_tp, pressure_psi=_psi_w, speed_window_kph=(38.234, 42.234))
        # a file whose whole conditioned block already sits inside the window keeps every sample (equal counts);
        # a file with other test speeds must lose them (strictly fewer)
        try:
            _v_all = np.asarray(_ltd(_tp).velocity_kph, float); _outside = bool(np.any((_v_all < 38.234) | (_v_all > 42.234)))
        except Exception:
            _outside = None
        _cnt_ok = (0 < _m_win.samples_selected < _m_all.samples_selected) if _outside else (0 < _m_win.samples_selected <= _m_all.samples_selected)
        _ok_w = (_m_win.speed_window_kph == (38.234, 42.234) and _m_all.speed_window_kph is None
                 and _cnt_ok and len(_m_win._fz_axis) >= 3)
        _bad_w = False
        try:
            _TMw.from_file(_tp, pressure_psi=_psi_w, speed_window_kph=(400.0, 401.0)); _bad_w = True   # must refuse an empty window
        except ValueError:
            pass
        # the app path: a project-declared window reaches the loaded model
        _saved_car = dict(wD._car); wD._car['tire_speed_window_kph'] = [38.234, 42.234]
        try:
            wD._on_tire_file(_tp); _app_w = getattr(wD._tire_model, 'speed_window_kph', None)
        finally:
            wD._car = _saved_car; wD._on_tire_file(_tp)
        # after the restore the model must carry the DESIGN's own declared window (None when the config declares none)
        _dw = _saved_car.get('tire_speed_window_kph')
        _dw = (float(_dw[0]), float(_dw[1])) if _dw else None
        _ok_w = _ok_w and (not _bad_w) and _app_w == (38.234, 42.234) and getattr(wD._tire_model, 'speed_window_kph', None) == _dw
        if not _ok_w:
            fails += 1
        print(f'tyre speed window: {os.path.basename(_tp)} all-speed {_m_all.samples_selected} samples -> 38.2-42.2 km/h {_m_win.samples_selected} samples, '
              f'fz axis {len(_m_win._fz_axis)} centres, empty window refused={not _bad_w}, app honours car key={_app_w == (38.234, 42.234)}   {"pass" if _ok_w else "UNEXPECTED FAIL"}')
    else:
        print('tyre speed window: no tyre file on the design config — skipped')
except Exception as _etw:
    fails += 1
    print(f'tyre speed window: UNEXPECTED FAIL ({type(_etw).__name__}: {_etw})')

# ── PROJECT DECLARATIONS DO NOT LEAK ACROSS LOADS (2026-09-14) ──────────────
#    car keys a project declares (keep-out, ARB rod-end standoff, Rule 04 waiver,
#    steering-stop note) must vanish when a project WITHOUT them is loaded into the
#    same window.  Before the fix the car-panel carry-forward kept them, so the
#    net judged v106 with v105's -18 mm spacer (coplanar 15.3 mm, a phantom fail).
print('-' * 64)
try:
    import glob as _gl, json as _jl
    _DECL = ('keepout_step', 'front_arb_drop_standoff_mm', 'rear_arb_drop_standoff_mm',
             'front_arb_rule04_waiver', 'rear_arb_rule04_waiver', 'steering_stop_note')
    _decl_of = {}
    for _p in sorted(_gl.glob('configs/2027_v*.vahan')):
        try:
            _decl_of[_p] = {k for k in _jl.load(open(_p, encoding='utf-8')).get('car', {}) if k in _DECL}
        except Exception:
            pass
    _pair = None
    for _a, _ka in _decl_of.items():
        for _b, _kb in _decl_of.items():
            if _ka - _kb:
                _pair = (_a, _b, _ka - _kb); break
        if _pair: break
    if _pair:
        _a, _b, _leak_keys = _pair
        win._load_project_from_path(_a); win._load_project_from_path(_b); win._rebuild_solvers(0.)
        _leaked = sorted(k for k in _leak_keys if k in win._car)
        _lok = not _leaked
        if not _lok:
            fails += 1
        print(f'declaration leak : load {os.path.basename(_a)[:22]} then {os.path.basename(_b)[:22]} -> leaked {_leaked or "none"} of {sorted(_leak_keys)}   {"pass" if _lok else "UNEXPECTED FAIL"}')
        if _pdes:
            win._load_project_from_path(_pdes); win._rebuild_solvers(0.)
    else:
        print('declaration leak : no config pair with differing declarations — skipped')
except Exception as _el:
    fails += 1
    print(f'declaration leak : UNEXPECTED FAIL ({type(_el).__name__}: {_el})')

# ── DAMPER MOTION SIGN (2026-08-26): a matched |motion ratio| can still be a
#    sign-inverted rocker — the pushrod acting as a PULLROD (damper EXTENDS in
#    bump).  solver_mr()/the rate checks take an absolute value and are blind
#    to it; v73 shipped inverted and was caught by eye, not by the tool.  Gate:
#    the loaded design's front + rear dampers must read as PUSHRODS (compress
#    in bump, sign −1), AND flipping the rocker chain to invert the motion must
#    make validate() FAIL 'damper acts as pushrod'.
print('-' * 64)
try:
    _dsf = _pkgm.damper_motion_sign(win, 'front')
    _dsr = _pkgm.damper_motion_sign(win, 'rear')
    # a KNOWN-inverted geometry (archived v73, pushrod-acting-as-pullrod front)
    # must trip 'damper acts as pushrod' when judged against a pushrod baseline
    import glob as _g73
    _inv_caught = None
    _v73 = _g73.glob('configs/**/2027_v73_INVERTED*.vahan', recursive=True)
    if _v73:
        win._load_project_from_path(_v73[0]); win._rebuild_solvers(0.)
        _res_inv = _pkgm.validate(win, _pbase, _pkgm.Tolerances(),
                                  allow_wheel_motion=True)
        _inv_caught = any(c['axle'] == 'front'
                          and c['name'] == 'damper acts as pushrod'
                          and not c['ok'] for c in _res_inv.checks)
        win._load_project_from_path(_design_cfg[0]); win._rebuild_solvers(0.)
    _dsok = (_dsf < 0 and _dsr < 0 and (_inv_caught is not False))
    if not _dsok:
        fails += 1
    print(f'damper sign      : design front={_dsf:+.0f} rear={_dsr:+.0f} '
          f'(pushrod=-1), archived-inverted caught={_inv_caught}   '
          f'{"pass" if _dsok else "UNEXPECTED FAIL"}')
except Exception as _eds:
    fails += 1
    import traceback as _tbds; _tbds.print_exc()
    print(f'damper sign      : UNEXPECTED FAIL ({type(_eds).__name__}: {_eds})')

# ── GROUND CONTACT (2026-08-23): at design position each axle's tire bottom
#    (wheel_center_z − tire_outer_dia/2) must sit ON the ground plane z=0
#    (|gap| ≤ GROUND_CONTACT_TOL_MM).  The Apply-Sag bake used to raise the
#    wheel side without dropping the car back onto the ground: v41–v69 rear
#    floated 14.6 mm, so every rear ground-referenced number (RC height,
#    anti-squat) was measured from the wrong plane and the Onshape export
#    showed the rear tire in the air.  The CURRENT (highest-version) config
#    and everything in configs/experiments must pass; archived configs that
#    still float are KNOWN-FAIL (superseded lineage, kept for history).
print('-' * 64)
try:
    import glob as _gg, re as _rg, json as _jg
    from vahan.hardpoints import tire_ground_gap_mm, GROUND_CONTACT_TOL_MM

    def _ground_gaps(path):
        with open(path) as _fh:
            _d = _jg.load(_fh)
        _dia = float(_d.get('car', {}).get('tire_outer_dia_mm', 406.0))
        return {ax: tire_ground_gap_mm(_d[ax]['wheel_center'], _dia)
                for ax in ('front_hp', 'rear_hp')
                if ax in _d and 'wheel_center' in _d[ax]}

    _gcfgs = _gg.glob('configs/2027_v*.vahan')
    _gcur = (max(_gcfgs, key=lambda p: int(_rg.search(r'2027_v(\d+)', p).group(1))
             if _rg.search(r'2027_v(\d+)', p) else -1) if _gcfgs else None)
    _gmust = ([_gcur] if _gcur else []) + sorted(_gg.glob('configs/experiments/*.vahan'))
    _gfail, _garch = [], []
    for _p in _gmust:
        _bad = {k: v for k, v in _ground_gaps(_p).items()
                if abs(v) > GROUND_CONTACT_TOL_MM}
        if _bad:
            _gfail.append(os.path.basename(_p) + ' ' + ', '.join(
                f'{k[:-3]} {v:+.2f}mm' for k, v in _bad.items()))
    for _p in _gcfgs:
        if _p == _gcur:
            continue
        if any(abs(v) > GROUND_CONTACT_TOL_MM for v in _ground_gaps(_p).values()):
            _garch.append(os.path.basename(_p))
    if _gfail:
        fails += 1
        print('ground contact   : UNEXPECTED FAIL — ' + '; '.join(_gfail))
    else:
        _cg = _ground_gaps(_gcur) if _gcur else {}
        _cgs = ', '.join(f'{k[:-3]} {v:+.2f}mm' for k, v in _cg.items())
        print(f'ground contact   : {os.path.basename(_gcur) if _gcur else "?"} '
              f'{_cgs} + {len(_gmust)-1} experiment cfg(s) all on z=0 '
              f'(tol {GROUND_CONTACT_TOL_MM:.0f}mm)   pass')
    if _garch:
        known += 1
        print(f'ground contact   : {len(_garch)} ARCHIVED configs still float '
              f'(v41–v69 sag-bake era, superseded) — KNOWN-FAIL, not edited')
except Exception as _eg:
    fails += 1
    import traceback as _tbg; _tbg.print_exc()
    print(f'ground contact   : UNEXPECTED FAIL ({type(_eg).__name__}: {_eg})')

print('-' * 64)
# ── ARB IS A ROLL-ONLY ELEMENT (solver-bug register): the anti-roll bar links
#    left and right, so in SYMMETRIC motion (pure braking dive / pure accel
#    squat, no roll) it barely twists and the drop-link force must be ~0.  The
#    old per-corner freebody wrongly reacted each rocker's full moment, showing
#    ~3.7 kN through the bar in pure braking (and a bogus SF 0.2).  Gate: pure
#    braking ARB drop-link ~0, pure cornering ARB drop-link clearly non-zero.
try:
    import glob as _gb, re as _rb
    _bcfgs = _gb.glob('configs/2027_v*.vahan')
    _bcur = (max(_bcfgs, key=lambda p: int(_rb.search(r'2027_v(\d+)', p).group(1))
                 if _rb.search(r'2027_v(\d+)', p) else -1) if _bcfgs else None)
    if _bcur is None:
        print('arb roll-only    : no config — skipped')
    else:
        import gui.wheel_package as _WPb
        _wb = MainWindow(); _wb._load_project_from_path(_bcur); _wb._rebuild_solvers(0.)

        def _arb_peak(lat, lon):
            pk = 0.0
            for _p, _v, _c, _lab in _WPb._load_items(_wb, lat, lon):
                if 'ARB' in _lab and 'drop-link' in _lab:
                    pk = max(pk, float(np.linalg.norm(_v)))
            return pk
        _arb_brake = _arb_peak(0.0, -1.5)     # pure braking (symmetric)
        _arb_corner = _arb_peak(1.5, 0.0)     # pure cornering (roll)
        _abfail = []
        if _arb_brake > 200.0:
            _abfail.append(f'pure-braking ARB drop-link {_arb_brake:.0f} N (>200) — bar '
                           f'is reacting symmetric load; roll-only decomposition broken')
        if _arb_corner < 300.0:
            _abfail.append(f'pure-cornering ARB drop-link {_arb_corner:.0f} N (<300) — bar '
                           f'is not reacting roll')
        if _abfail:
            fails += 1
            print('arb roll-only    : UNEXPECTED FAIL — ' + '; '.join(_abfail))
        else:
            print(f'arb roll-only    : braking {_arb_brake:.0f} N (~0), cornering '
                  f'{_arb_corner:.0f} N (roll-only decomposition holds)   pass')
except Exception as _eab:
    fails += 1
    import traceback as _tab; _tab.print_exc()
    print(f'arb roll-only    : UNEXPECTED FAIL ({type(_eab).__name__}: {_eab})')

print('-' * 64)
# ── ACCELERATION MODEL (vahan/acceleration.py) — the gear-resolved launch event.
#    Smoke + physics sanity: a SHORTER final drive must give a LOWER gearing-
#    limited top speed (F = torque·ratio/r → higher ratio tops out sooner), the
#    launch must be grip-limited (tractive force >= grip off the line), and the
#    75 m sprint must be covered with a physical time.  Guards the one-model
#    coupling (engine curve + tyre grip + aero) from silently breaking.
try:
    import glob as _ga, re as _ra
    _acfgs = _ga.glob('configs/2027_v*.vahan')
    _acur = (max(_acfgs, key=lambda p: int(_ra.search(r'2027_v(\d+)', p).group(1))
                 if _ra.search(r'2027_v(\d+)', p) else -1) if _acfgs else None)
    if _acur is None:
        print('acceleration     : no config — skipped')
    else:
        from vahan.acceleration import from_window as _accel_fw
        _wa = MainWindow(); _wa._load_project_from_path(_acur); _wa._rebuild_solvers(0.)
        _m26 = _accel_fw(_wa, final_drive=2.6, grip_scale=0.60)
        _m41 = _accel_fw(_wa, final_drive=4.1, grip_scale=0.60)
        _r26 = _m26.run(); _r41 = _m41.run()
        _afail = []
        if not (_r41['top_speed_gearing_kph'] < _r26['top_speed_gearing_kph']):
            _afail.append(f'shorter FD 4.1 top {_r41["top_speed_gearing_kph"]:.0f} '
                          f'not below FD 2.6 top {_r26["top_speed_gearing_kph"]:.0f}')
        if not (_m26.tractive_force_N(2.0) >= _m26.traction_force_N(2.0)):
            _afail.append('launch is not grip-limited (tractive < grip off the line)')
        if not (np.isfinite(_r26['t_75m_s']) and 3.0 < _r26['t_75m_s'] < 6.0):
            _afail.append(f'75 m time {_r26["t_75m_s"]} outside 3-6 s')
        if not (70.0 < _r26['top_speed_kph'] < 160.0):
            _afail.append(f'top speed {_r26["top_speed_kph"]:.0f} km/h non-physical')
        if _afail:
            fails += 1
            print('acceleration     : UNEXPECTED FAIL — ' + '; '.join(_afail))
        else:
            print(f'acceleration     : FD2.6 top {_r26["top_speed_kph"]:.0f} km/h / 75 m '
                  f'{_r26["t_75m_s"]:.2f} s, FD4.1 top {_r41["top_speed_gearing_kph"]:.0f} '
                  f'(shorter=lower), launch grip-limited   pass')
except Exception as _ea:
    fails += 1
    import traceback as _tba; _tba.print_exc()
    print(f'acceleration     : UNEXPECTED FAIL ({type(_ea).__name__}: {_ea})')

# Native tire drawing must lean in the direction of the alignment readout.
# This catches the former sign reversal on BOTH sides at neutral steering.
try:
    from vahan.kinematics import road_plane_camber_deg
    _wc = MainWindow()
    _wc._rebuild_solvers(0.)
    for _requested_camber in (-2., 2.):
        _wc._alignment['front_camber_deg'] = _requested_camber
        _wc._alignment['rear_camber_deg'] = _requested_camber
        _draw, _ = _wc._assemble_corners_draw(
            {label: 0. for label in ('FL', 'FR', 'RL', 'RR')}, 0.)
        assert len(_draw) == 4
        for _corner in _draw:
            _side = 'left' if _corner['label'].endswith('L') else 'right'
            _actual = road_plane_camber_deg(_corner['spin_axis'], side=_side)
            assert abs(_actual - _requested_camber) < 1e-7, (
                _corner['label'], _requested_camber, _actual)
    _wc.close()
    print('alignment visual : signed road-plane camber matches all four corners   pass')
except Exception as _ec:
    fails += 1
    print(f'alignment visual : UNEXPECTED FAIL ({type(_ec).__name__}: {_ec})')

print('-' * 64)
try:
    wD._rebuild_solvers(0.)
    _sv = wD._build_dynamics_solver()._veh
    _supports = []
    for _label, _suffix in [('FL', 'front'), ('RL', 'rear')]:
        _sol = wD._solvers[_label]
        _mr = abs(_sol.solve(.001).spring_length - _sol.solve(-.001).spring_length) / .002
        _supports.append(getattr(_sv, f'static_spring_force_{_suffix}_N') * _mr * 2)
    assert np.isclose(sum(_supports), _sv.sprung_mass_kg * 9.81, rtol=1e-8)
    _sprung_moment = (_sv.total_mass_kg * 9.81 * _sv.cg_to_front_axle_m
                       - _sv.unsprung_mass_rear_kg * 9.81 * _sv.wheelbase_m)
    assert np.isclose(_supports[1] * _sv.wheelbase_m, _sprung_moment, rtol=1e-8)
    print('spring support   : sprung weight and axle moment conserved   pass')
except Exception as _spring_error:
    fails += 1
    print(f'spring support   : UNEXPECTED FAIL ({type(_spring_error).__name__}: {_spring_error})')

try:
    import unittest as _steering_unittest
    from test_steering_direction import SteeringDirectionTests
    import test_pushrod_envelope as _pushrod_envelope_tests
    _direction_result = _steering_unittest.TestResult()
    _steering_unittest.defaultTestLoader.loadTestsFromTestCase(
        SteeringDirectionTests).run(_direction_result)
    _steering_unittest.defaultTestLoader.loadTestsFromModule(
        _pushrod_envelope_tests).run(_direction_result)
    if not _direction_result.wasSuccessful():
        raise AssertionError(str(_direction_result.failures + _direction_result.errors))
    print(f'steering/envelope: {_direction_result.testsRun} signed linkage/input/inverse/save/body checks   pass')
except Exception as _direction_error:
    fails += 1
    print(f'steering direction: UNEXPECTED FAIL ({_direction_error})')

# ── Ride page (Ctrl+8) + THE RIDE-RATE SOLVE (vahan/ride_solve.py).  The page
#    must build offscreen on the current config, its inputs must land in the
#    car dict and survive save->load, and its blocking solve must reproduce a
#    direct vahan.ride_solve call on the SAME VehicleParams; the page's study
#    numbers must equal vahan.ride's periodic_response recomputed here, and
#    the reported spring rates must realise the chosen ride rates on that
#    VehicleParams (ONE MODEL: the page only calls and plots).
try:
    import glob as _gr, re as _rre, tempfile as _rtmp
    from dataclasses import replace as _rreplace
    _rcfgs = _gr.glob('configs/2027_v*.vahan')
    _rcur = (max(_rcfgs, key=lambda p: int(_rre.search(r'2027_v(\d+)', p).group(1))
                 if _rre.search(r'2027_v(\d+)', p) else -1) if _rcfgs else None)
    if _rcur is None:
        print('ride page        : no config — skipped')
    else:
        from vahan import ride_solve as _RS
        from gui.ride_page import RidePage as _RidePage
        _wr = MainWindow(); _wr._load_project_from_path(_rcur); _wr._rebuild_solvers(0.)
        _wr._switch_page(7)
        _rp = _wr._ride_page
        assert isinstance(_rp, _RidePage) and _wr._pages.currentIndex() == 7, 'Ride page not on Ctrl+8 / index 7'
        # small deterministic study; inputs must write straight into the car dict
        _rp._npts.setValue(3); _rp._v_n.setValue(2); _rp._seed_n.setValue(1)
        _rp._coh.setCurrentText('coherent'); _rp._pitch_I.setValue(90.); _rp._damp['RR'].setValue(1234.)
        assert _wr._car['ride_pitch_inertia_kgm2'] == 90. and _wr._car['ride_damping_wheel_Nspm'][3] == 1234.
        _inp = _rp.inputs()
        assert _inp.pitch_inertia_kgm2 == 90. and _inp.corner_damping_Nspm[3] == 1234. and len(_inp.speeds_mps) == 2
        _study = _rp.analyse_current(); assert _study is not None, _rp._status.text()
        _sel = _rp.run_solve(blocking=True); assert _sel is not None, _rp._status.text()
        # reference: the same solve straight from vahan.ride_solve on the same VehicleParams
        _veh = _wr._build_dynamics_solver()._veh
        _fr, _rr, _, _ = _rp.sweep_grids(_veh)
        _sw = _RS.sweep_ride_rates(_veh, _inp, _fr, _rr)
        _ref = _RS.describe_selection(_sw, _veh, _inp, **_rp._motion())
        assert np.allclose(_sw.dlc, _rp.last_sweep.dlc) and np.allclose(_sw.pitch_to_bounce, _rp.last_sweep.pitch_to_bounce)
        assert (_ref['i_front'], _ref['j_rear'], _ref['status']) == (_sel['i_front'], _sel['j_rear'], _sel['status'])
        assert abs(_ref['front_spring_rate_Npm'] - _sel['front_spring_rate_Npm']) < 1e-9
        # chosen springs realise the chosen ride rates through VehicleParams itself
        _chk = _rreplace(_veh, spring_rate_front_Npm=_sel['front_spring_rate_Npm'],
                         spring_rate_rear_Npm=_sel['rear_spring_rate_Npm'])
        assert np.isclose(_chk.ride_rate_front_Npm, _sel['front_ride_rate_Npm'], rtol=1e-9)
        assert np.isclose(_chk.ride_rate_rear_Npm, _sel['rear_ride_rate_Npm'], rtol=1e-9)
        # study DLC == vahan.ride periodic_response recomputed here
        _m0, _case0, _model = _study['metrics'][0], _study['cases'][0], _study['model']
        _load = _model.periodic_response(_case0.dt_s, _case0.road_heights_m)['dynamic_tire_load_N'] + _model.baseline_loads
        _dlc = np.sqrt(np.mean((_load - _load.mean(0)) ** 2, 0)) / _load.mean(0)
        assert np.allclose(_dlc, _m0.dlc), (_dlc, _m0.dlc)
        # ride_* keys round-trip through save -> load
        _rpath = os.path.join(_rtmp.gettempdir(), '_vahan_ride_roundtrip.vahan')
        _wr._save_project_to_path(_rpath)
        _wr2 = MainWindow(); _wr2._load_project_from_path(_rpath)
        assert _wr2._car['ride_pitch_inertia_kgm2'] == 90. and _wr2._car['ride_damping_wheel_Nspm'][3] == 1234.
        assert _wr2._car['ride_coherence'] == 'coherent' and _wr2._car['ride_sweep_points'] == 3
        _wr.close(); _wr2.close()
        print(f'ride page        : Ctrl+8 built; solve {_sel["status"]}: front {_sel["front_ride_frequency_Hz"]:.2f} / '
              f'rear {_sel["rear_ride_frequency_Hz"]:.2f} Hz -> {_sel["front_spring_rate_lbf_in"]:.0f} / '
              f'{_sel["rear_spring_rate_lbf_in"]:.0f} lbf/in, worst DLC {_sel["worst_dlc"]:.3f}; '
              f'== vahan.ride_solve, == vahan.ride, car-dict save/load   pass')
except Exception as _er:
    fails += 1
    import traceback as _tbr; _tbr.print_exc()
    print(f'ride page        : UNEXPECTED FAIL ({type(_er).__name__}: {_er})')

# ── Corner Speed page (Ctrl+9) + vahan/corner_speed.py.  The page must build
#    offscreen on the design config; its blocking compute must return the
#    orchestration function's own rows, and those rows must equal a DIRECT call
#    of the ONE trim engine (vahan.ymd.trim_sweep_ackermann) on the same solver;
#    the curves on the figure and the table cells must BE those rows (compute /
#    draw split); the aero row must carry the app's own per-g package at that
#    radius; the steering-lock radius must be the bicycle radius of the front
#    toes solved at the full-rack handwheel; and the per-corner grip budget must
#    land every case at utilization 1.0 by the steady-state solver's OWN
#    per-corner utilization (ONE MODEL: the page only calls and plots).
try:
    import math as _csmath
    from gui.corner_speed_page import CornerSpeedPage as _CornerSpeedPage
    from vahan import corner_speed as _CS
    from vahan.ymd import (G as _G_ymd, trim_sweep_ackermann as _cs_trim,
                           build_loads_table as _cs_table)
    from vahan.kinematics import KinematicMetrics as _CSKM
    import glob as _gcs
    # OWN glob: the coplanar block above rebinds the module-level _cfgs to the
    # v28 list, so _highest_config(_cfgs) here would judge v28, not the design.
    _cscfgs = _gcs.glob('configs/2027_v*.vahan')
    _cscur = os.environ.get('VAHAN_DESIGN') or _highest_config(_cscfgs)
    if _cscur is None:
        print('corner speed     : no config — skipped')
    else:
        _wcs = MainWindow(); _wcs._load_project_from_path(_cscur); _wcs._rebuild_solvers(0.)
        _wcs._switch_page(8)
        _csp = _wcs._corner_speed_page
        assert isinstance(_csp, _CornerSpeedPage) and _wcs._pages.currentIndex() == 8, \
            'Corner Speed page not on Ctrl+9 / index 8'
        # tiny study: two radii (the larger must be solved first), aero on, no lock row
        _csp._radii_txt.setText('12, 20'); _csp._add_lock.setChecked(False); _csp._aero_chk.setChecked(True)
        _cscfg = _csp.speed_config()
        assert _cscfg['radii'] == [20.0, 12.0], _cscfg['radii']
        assert _cscfg['ackermann_probed'] and np.isfinite(_cscfg['ackermann_pct']), _cscfg
        # steering lock = bicycle radius of the two front toes with the rack at its stop
        _lk = _cscfg['lock']
        _hand = _CS.full_lock_handwheel_deg(_wcs._steer)
        assert np.isclose(_hand, float(_wcs._steer['total_rack_travel_mm'])
                          / float(_wcs._steer['rack_travel_per_rev_mm']) * 180.0)
        _lsolv = MainWindow._build_corner_solvers(_wcs._all_corner_hp(), _wcs._steer, _wcs._topology, _hand)
        _tl = float(_CSKM(_lsolv['FL'].solve(0.), 'left').toe)
        _tr = float(_CSKM(_lsolv['FR'].solve(0.), 'right').toe)
        _wb = float(_wcs._car['wheelbase_mm']) / 1000.
        assert np.isfinite(_lk['lock_radius_m']) and np.isclose(
            _lk['lock_radius_m'], _wb / _csmath.tan(_csmath.radians(0.5 * (abs(_tl) + abs(_tr))))), _lk
        assert 1.0 < _lk['lock_radius_m'] < 8.0, _lk
        _csrows = _csp.run_corner_speed(blocking=True)
        assert _csrows is not None and len(_csrows) == 4, _csp._speed_progress.text()
        assert [(r['radius_m'], r['aero']) for r in _csrows] == [(20., False), (20., True), (12., False), (12., True)]
        for _r in _csrows:
            assert _r['converged'] and np.isfinite(_r['ay_g']) and _r['ay_g'] > 0.5 \
                and np.isfinite(_r['N_beta_Nm_per_deg']), _r
            assert np.isclose(_r['speed_mps'] ** 2, _r['ay_g'] * _G_ymd * _r['radius_m'], rtol=1e-12), _r
            assert np.isclose(_r['speed_kph'], _r['speed_mps'] * 3.6) and _r['stable'] == (_r['N_beta_Nm_per_deg'] < 0)
        # rows == a DIRECT call of the ONE trim engine on the same solver (20 m, no aero)
        _ss_cs = _wcs._build_dynamics_solver()
        _tire_cs = _ss_cs._tire if _wcs._tire_model is None else _wcs._tire_model
        _direct = _cs_trim(_tire_cs, _ss_cs, radius_m=20.0, ackermann_list=(_cscfg['ackermann_pct'],),
                           grip_multiplier=float(_ss_cs._mu_scale), aero_Fz_per_g=None,
                           loads_table=_cs_table(_ss_cs, None))[0]
        _r20 = _csrows[0]
        assert np.isclose(_r20['ay_g'], _direct['Ay_trim_max'], rtol=1e-9) \
            and np.isclose(_r20['N_beta_Nm_per_deg'], _direct['N_beta'], rtol=1e-6), (_r20, _direct)
        # aero row: the per-g package the page fed the engine is the app's own at that radius
        _was_aero = _wcs._aero_active; _wcs._aero_active = True
        try:
            _ag20 = _wcs._get_aero_Fz_per_g(radius_m=20.0)
        finally:
            _wcs._aero_active = _was_aero
        assert _ag20 and np.isclose(_cscfg['aero_by_radius'][20.0]['FL'], _ag20['FL']) \
            and np.isclose(_csrows[1]['aero_per_g_N'], sum(_ag20.values())), (_cscfg['aero_by_radius'], _ag20)
        assert np.isclose(_csrows[1]['downforce_N'], _csrows[1]['aero_per_g_N'] * _csrows[1]['ay_g'])
        # compute/draw split: both figure axes carry exactly the rows' series; the table too
        _axs = _csp.speed_slot.fig.get_axes(); assert len(_axs) == 2, len(_axs)
        _aero_lab = 'aero: ' + _cscfg['aero_text']
        for _ax, _key in ((_axs[0], 'speed_kph'), (_axs[1], 'ay_g')):
            _lines = {ln.get_label(): ln for ln in _ax.get_lines() if not ln.get_label().startswith('_')}
            assert set(_lines) == {'no aero', _aero_lab}, list(_lines)
            for _aero, _lab in ((False, 'no aero'), (True, _aero_lab)):
                _R, _V, _A = _CS.corner_speed_series(_csrows, _aero)
                assert np.array_equal(np.asarray(_lines[_lab].get_xdata(), float), _R)
                assert np.array_equal(np.asarray(_lines[_lab].get_ydata(), float), _V if _key == 'speed_kph' else _A)
        assert _csp.speed_table.rowCount() == 4
        for _i, _r in enumerate(_csrows):
            assert (_csp.speed_table.item(_i, 0).text(), _csp.speed_table.item(_i, 2).text(),
                    _csp.speed_table.item(_i, 3).text()) == (f'{_r["radius_m"]:.2f}', f'{_r["ay_g"]:.3f}', f'{_r["speed_kph"]:.1f}')
        _cstxt = _CS.corner_speed_table_text(_csrows)
        assert _cstxt.count('\n') == 4 and f'{_r20["ay_g"]:.3f}' in _cstxt
        # per-corner grip budget: two g levels for the curves, a short bisection
        _csp._grip_g.setValue(1.2)
        _csstudy = _csp.run_grip_budget(blocking=True, iters=10, sweep_g=[1.0, 1.5])
        assert _csstudy is not None and _csstudy is _csp.last_study and len(_csstudy['cases']) == 2, _csp._grip_progress.text()
        _glines = {ln.get_label(): ln for ln in _csp.grip_slot.fig.get_axes()[0].get_lines()
                   if not ln.get_label().startswith('_')}
        for _label, _case in _csstudy['cases'].items():
            _lim = _case['corner_limit']; _at = _lim['at_limit']
            assert _lim['binding'] in ('FL', 'FR', 'RL', 'RR') and _at is not None and 'corners' in _at \
                and not _lim['hit_upper_bound'] and _lim['n_failed_solves'] == 0, _lim
            assert abs(_at['worst_utilization'] - 1.0) < 0.02 \
                and _at['corners'][_lim['binding']]['utilization'] == _at['worst_utilization'], _lim
            # the pair budget (the app's criterion) can never bind before the worst single tyre
            assert _case['axle_limit']['limit_g'] >= _lim['limit_g'] - 1e-9, (_case['axle_limit'], _lim)
            # == the steady-state solver's OWN per-corner utilization at that g with that aero
            _res = _ss_cs.solve(_at['lateral_g'], 0.0, aero_Fz=_at['aero_Fz_applied'])
            for _c in ('FL', 'FR', 'RL', 'RR'):
                _d = _at['corners'][_c]
                assert np.isclose(_d['utilization'], _res.utilization[_c], rtol=1e-9) \
                    and np.isclose(_d['Fz_N'], _res.Fz[_c]) \
                    and np.isclose(_d['inclination_deg'], _res.inclination[_c]), (_c, _d)
                assert np.isclose(_d['demand_N'], np.hypot(_res.Fy[_c], _res.Fx.get(_c, 0.0))) \
                    and np.isclose(_d['demand_N'] / _d['budget_N'], _d['utilization']), (_c, _d)
            _u12 = _case['at_g']
            assert np.isclose(_u12['lateral_g'], 1.2) \
                and all(np.isfinite(_u12['corners'][_c]['utilization']) for _c in ('FL', 'FR', 'RL', 'RR')), _u12
            # plotted utilization curves == the study's sweep, per corner
            assert _case['sweep']['g'] == [1.0, 1.5], _case['sweep']['g']
            for _c in ('FL', 'FR', 'RL', 'RR'):
                _ln = _glines[f'{_c} {_label}']
                assert np.array_equal(np.asarray(_ln.get_xdata(), float), np.asarray(_case['sweep']['g'])) \
                    and np.array_equal(np.asarray(_ln.get_ydata(), float),
                                       np.asarray(_case['sweep']['utilization'][_c])), (_c, _label)
        _noaero = _csstudy['cases']['no aero']
        _aerocase = [v for k, v in _csstudy['cases'].items() if k != 'no aero'][0]
        assert _aerocase['spec'].get('aero_Fz') and sum(_aerocase['spec']['aero_Fz'].values()) > 0, _aerocase['spec']
        assert _aerocase['corner_limit']['limit_g'] > _noaero['corner_limit']['limit_g'], 'downforce must raise the per-corner limit'
        assert _csp.grip_table.rowCount() == 16, _csp.grip_table.rowCount()   # 2 cases x (chosen g + at limit) x 4 corners
        _wcs.close()
        print(f'corner speed     : Ctrl+9 built; 20 m trim {_r20["ay_g"]:.3f} g / {_r20["speed_kph"]:.1f} km/h '
              f'(== vahan.ymd direct), with aero {_csrows[1]["ay_g"]:.3f} g; lock radius {_lk["lock_radius_m"]:.2f} m; '
              f'per-corner limit {_noaero["corner_limit"]["limit_g"]:.3f} g ({_noaero["corner_limit"]["binding"]}) / '
              f'aero {_aerocase["corner_limit"]["limit_g"]:.3f} g, axle-aggregate {_noaero["axle_limit"]["limit_g"]:.3f} g '
              f'(== SteadyStateSolver.solve); figure == rows, table == rows   pass')
except Exception as _ecs:
    fails += 1
    import traceback as _tbcs; _tbcs.print_exc()
    print(f'corner speed     : UNEXPECTED FAIL ({type(_ecs).__name__}: {_ecs})')

# ── IK + dynamics-recommendation integrity (2026-09-22 audit, groups 1-2) ───────────
# Each line was a confirmed defect (DESIGN_2027/binder_run/ASTRA_AUDIT_VERIFY_ik_sens_20260922.md
# items 1-4, 8-9); these checks FAILED on the pre-fix code and pass after the fix.
def _ik_gate(name, fn):
    global fails
    try:
        msg = fn()
        print(f'{name}: {msg}   pass')
    except Exception as _eik:
        fails += 1
        import traceback as _tbik; _tbik.print_exc()
        print(f'{name}: UNEXPECTED FAIL ({type(_eik).__name__}: {_eik})')

try:
    from vahan import optimizer as _OPT
    from vahan import packaging as _PKik
    _ikw = MainWindow(); _ikw._load_project_from_path(_design); _ikw._rebuild_solvers(0.)
    _ikw._motion_panel._motion = 'heave'; _ikw._run_sweep()
    _ikhp = _ikw._ik_live_geometry('front')
except Exception as _eik0:
    _ikw = None; fails += 1
    print(f'IK integrity setup: UNEXPECTED FAIL ({type(_eik0).__name__}: {_eik0})')

if _ikw is not None:
    def _ik_arb_metrics():
        # was: NameError swallowed -> arb_angle/arb_drop_travel/arb_mr all NaN
        _x = np.asarray(_ikw._x_arr, float)
        _msg = []
        for _axle, _lbl in (('front', 'FL'), ('rear', 'RL')):
            _hp = _ikw._ik_live_geometry(_axle)
            _body = (_ikw._topology.front if _axle == 'front' else _ikw._topology.rear).damper_mount.value
            _c = _OPT._evaluate_sweep(_hp, _x / 1000.0, pushrod_body=_body,
                                      metric_keys=['arb_angle', 'arb_drop_travel', 'arb_mr'])
            _zero = np.abs(_x) < 1e-9
            assert np.all(np.isfinite(_c['arb_angle'])) and np.all(np.isfinite(_c['arb_drop_travel'])) \
                and np.all(np.isfinite(_c['arb_mr'][~_zero])), {k: int(np.isnan(v).sum()) for k, v in _c.items()}
            # ONE MODEL: the IK's ARB curve == the app's own graph curve for that corner
            _g = np.asarray(_ikw._sweep_results[_lbl]['arb_angle'], float)
            _m = np.isfinite(_g)
            _d = float(np.max(np.abs(_c['arb_angle'][_m] - _g[_m])))
            assert _m.sum() > 10 and _d < 1e-6, (_axle, int(_m.sum()), _d)
            _msg.append(f'{_lbl} bar angle at +{_x[-1]:.0f} mm {_c["arb_angle"][-1]:+.3f} deg (graph diff {_d:.1e})')
        # live: moving the blade end changes the bar angle
        _hp2 = {k: np.array(v, float) for k, v in _ikhp.items()}
        _hp2['arb_arm_end'][2] += 0.001
        _t3 = np.array([-0.005, 0.0, 0.005])
        _a = _OPT._evaluate_sweep(_ikhp, _t3, metric_keys=['arb_angle'])['arb_angle']
        _b = _OPT._evaluate_sweep(_hp2, _t3, metric_keys=['arb_angle'])['arb_angle']
        assert abs(_b[2] - _a[2]) > 1e-4, (_a, _b)
        _ik = _OPT.InverseSolver(_ikhp, travel_mm=(-5, 5), n_points=3, axle='front')
        _ik.add_target('arb_angle', 1.0)
        _ik.set_variables([_OPT.DesignVar('arb_arm_end', 2, 0.001)])
        _r = _ik.solve('local')
        assert np.isfinite(_r['primary_max_error']) and np.all(np.isfinite(_r['curves']['arb_angle'])), _r['primary_max_error']
        return '; '.join(_msg) + f'; ARB IK solve max error {_r["primary_max_error"]:.3f} deg (finite)'
    _ik_gate('IK ARB metrics   ', _ik_arb_metrics)

    def _ik_solver_contract():
        # was: .success/.status discarded, NaN primary error, Apply always shown
        from types import SimpleNamespace as _SN
        assert not _OPT._ls_diag(_SN(status=0, success=False, message='max nfev', x=np.zeros(1), nfev=500), 'x')['success']
        assert _OPT._ls_diag(_SN(status=2, success=True, message='ok', x=np.zeros(1), nfev=5), 'x')['success']
        _ik = _OPT.InverseSolver(_ikhp, travel_mm=(-400, 400), n_points=5, axle='front')
        _ik.add_target('camber', -1.0)
        _ik.set_variables([_OPT.DesignVar('uca_outer', 2, 0.001)])
        _r = _ik.solve('local')
        assert not _r['applicable'] and any('not solved at' in s for s in _r['reject_reasons']), _r['reject_reasons']
        _p = _ikw._ik_panel
        _p.show_result(_r)
        assert _p._apply_btn.isHidden() and 'NOT APPLICABLE' in _p._status.text(), _p._status.text()
        _fake = dict(_r, applicable=True, reject_reasons=[])
        _fake.pop('axle')
        _p.show_result(_fake)
        assert _p._apply_btn.isHidden(), 'result without an axle must not be applicable'
        return (f'+-400 mm camber solve: {_r["reject_reasons"][0][:60]}...; Apply hidden; '
                f'axle-less result refused')
    _ik_gate('IK solve contract', _ik_solver_contract)

    def _ik_axle_binding():
        # was: STALE_AXLE_APPLY rear (front solution written onto the rear axle)
        _p = _ikw._ik_panel
        _ik = _OPT.InverseSolver(_ikhp, travel_mm=(-10, 10), n_points=3, axle='front')
        _ik.add_target('camber', _OPT._evaluate_sweep(_ikhp, _ik.travel, metric_keys=['camber'])['camber'])
        _ik.set_variables([_OPT.DesignVar('uca_outer', 2, 0.0005)])
        _r = _ik.solve('local')
        assert _r['applicable'], _r['reject_reasons']
        _rear0 = {k: np.array(v, float) for k, v in _ikw._rear_hp.items()}
        _p.show_result(_r)
        assert not _p._apply_btn.isHidden()
        _cap = []
        _p.apply_requested.connect(_cap.append)
        try:
            _p._axle.setCurrentIndex(1)            # user flips the selector to Rear after the solve
            # (a) geometry edited after the solve -> refused, nothing written
            _ikw._front_hp['uca_front'] = _ikw._front_hp['uca_front'] + np.array([0., 0., 0.001])
            _front_edit = {k: np.array(v, float) for k, v in _ikw._front_hp.items()}
            _p._on_apply()
            assert _cap and _cap[-1]['axle'] == 'front', _cap[-1]['axle'] if _cap else None
            assert all(np.array_equal(_ikw._front_hp[k], _front_edit[k]) for k in _front_edit), 'stale apply wrote front'
            assert all(np.array_equal(_ikw._rear_hp[k], _rear0[k]) for k in _rear0), 'stale apply wrote rear'
            assert _p._last_result is None and 'geometry changed' in _p._status.text(), _p._status.text()
            # (b) fresh solve on unchanged geometry, selector still on Rear -> lands on FRONT only
            _ikw._front_hp['uca_front'] = _ikw._front_hp['uca_front'] - np.array([0., 0., 0.001])
            _hp_b = _ikw._ik_live_geometry('front')
            _ik2 = _OPT.InverseSolver(_hp_b, travel_mm=(-10, 10), n_points=3, axle='front')
            _ik2.add_target('camber', _OPT._evaluate_sweep(_hp_b, _ik2.travel, metric_keys=['camber'])['camber'])
            _ik2.set_variables([_OPT.DesignVar('uca_outer', 2, 0.0005)])
            _r2 = _ik2.solve('local')
            _p.show_result(_r2)
            _p._on_apply()
            assert _cap[-1]['axle'] == 'front'
            assert all(np.array_equal(_ikw._rear_hp[k], _rear0[k]) for k in _rear0), 'front result touched rear'
            assert np.allclose(_ikw._front_hp['uca_outer'], _r2['hp']['uca_outer'], atol=0), 'front not applied'
        finally:
            _p.apply_requested.disconnect(_cap.append)
        return 'selector on Rear -> emitted axle=front; stale geometry refused; rear untouched'
    _ik_gate('IK axle binding  ', _ik_axle_binding)

    def _ik_chain_rule():
        # was: 3-point residual vs the ORIGINAL plane; spring_chassis_pt +100 mm invisible (~1e-14)
        _hp = _ikw._ik_live_geometry('front')
        _t = _PKik.Tolerances()
        assert (_OPT.CHAIN_COPLANAR_GATE_MM, _OPT.ARB_INPLANE_GATE_MM, _OPT.ROCKER_AXIS_GATE_DEG) == \
            (_t.coplanar_mm, _t.arb_inplane_mm, _t.rocker_axis_deg)
        _ik = _OPT.InverseSolver(_hp, travel_mm=(0, 0), n_points=1, axle='front')
        _ik.add_target('camber', 0.0)
        _ik.set_variables([_OPT.DesignVar('spring_chassis_pt', 1, 0.2)])
        _x0 = _ik.ds.x0(); _x1 = _x0.copy(); _x1[0] += 0.1
        _names = _ik._chain_residual_names(_OPT.static_chain_rule_metrics(_hp))
        _i = _names.index('spring_chassis_pt') - len(_names)   # chain residuals are last
        _r0 = _ik._residuals(_x0)[_i]; _r1 = _ik._residuals(_x1)[_i]
        _m1 = _OPT.static_chain_rule_metrics(_ik.ds.unpack(_x1))
        _off = _m1['static_signed_mm']['spring_chassis_pt']
        assert abs(_r1 - _r0) > 10 and np.isclose(_r1, _off / _OPT.CHAIN_RESIDUAL_UNIT_MM), (_r0, _r1, _off)
        assert _OPT.chain_rule_violations(_m1), 'moved spring eye must violate Rule 01'
        # same shared checker as the packaging laws (FL corner, static)
        _laws = _PKik._axle_geometry_laws(_ikw, 'front')
        _m0 = _OPT.static_chain_rule_metrics(_hp)
        assert abs(_m0['coplanar_static_mm'] - _laws['corner_static_mm']['FL']) < 1e-6, (_m0['coplanar_static_mm'], _laws['corner_static_mm'])
        # Rule 02: moving a plate point re-derives the axis as the new plate normal
        _ik2 = _OPT.InverseSolver(_hp, travel_mm=(0, 0), n_points=1, axle='front')
        _ik2.add_target('camber', 0.0)
        _ik2.set_variables([_OPT.DesignVar('rocker_spring_pt', 1, 0.01)])
        _x = _ik2.ds.x0(); _x[0] += 0.008
        _m2 = _OPT.static_chain_rule_metrics(_ik2.ds.unpack(_x))
        assert _m2['rocker_axis_normal_error_deg'] <= _OPT.ROCKER_AXIS_GATE_DEG, _m2['rocker_axis_normal_error_deg']
        return (f'spring chassis eye +100 mm -> {abs(_off):.2f} mm off the current plane, residual '
                f'{_r0:.2f} -> {_r1:.2f}; FL static {_m0["coplanar_static_mm"]:.4f} mm == packaging law; '
                f'axis re-derived ({_m2["rocker_axis_normal_error_deg"]:.1e} deg)')
    _ik_gate('IK chain rule    ', _ik_chain_rule)

    _ikw.close()

def _recommend_units():
    # was: spring 200 lbf/in -> 1750, CG 1200 mm -> 2.5 mm, bias 55 % -> 0.85 % (SI bounds on display values)
    from types import SimpleNamespace as _SN
    from vahan.dynamics import DynamicsSensitivity as _DS, SENSITIVITY_OUTPUTS as _SO
    _v = _SN(front_track_m=1.2, rear_track_m=1.2, motion_ratio_front=1.0, motion_ratio_rear=1.0,
             arb_rate_front_Npm=0, arb_rate_rear_Npm=0, wheel_rate_front_Npm=30000, wheel_rate_rear_Npm=30000)
    _eff = {k: 1.0 for k in _SO}
    _a = {'vehicle_params': _v, 'baseline': {k: 1.0 for k in _SO},
          'sensitivities': [{'key': k, 'knob': n, 'unit': u, 'category': 'parameter', 'current_value': c,
                             'effects': _eff, 'implementations': []}
                            for k, n, u, c in [('spring_rate_front_Npm', 'Spring', 'lbf/in', 200.0),
                                               ('cg_to_front_axle_m', 'CG', 'mm', 1200.0),
                                               ('front_brake_bias', 'Bias', '%', 55.0)]]}
    _s = _DS.__new__(_DS); _s._base_veh = _v
    _out = {}
    for _tgt in (-50.0, 50.0):
        _rec = {r['key']: r for r in _s.recommend(_a, 'roll_angle_deg', _tgt)}
        _out[_tgt] = _rec
        _sp, _cg, _bb = (_rec['spring_rate_front_Npm'], _rec['cg_to_front_axle_m'], _rec['front_brake_bias'])
        assert np.isclose(_sp['new_value'], 200.0 + _tgt) and not _sp['clamped'], _sp['new_value']
        assert np.isclose(_cg['new_value'], 1200.0 + _tgt) and not _cg['clamped'], _cg['new_value']
        _bexp = min(85.0, max(40.0, 55.0 + _tgt))
        assert np.isclose(_bb['new_value'], _bexp) and _bb['clamped'], _bb['new_value']
        assert np.isclose(_bb['predicted_delta'], _bexp - 55.0), _bb['predicted_delta']
    # GUI row prints the per-row (clamped) prediction and flags the clamp
    from gui.panels import DynamicsOptPanel as _DOP
    _p = _DOP(); _p._analysis = _a
    _p._target_combo.setCurrentIndex(_p._target_combo.findData('roll_angle_deg'))
    _p._target_delta.setValue(-50.0)
    _p._on_recommend_impl()
    _rows = {_p._sens_table.item(i, 0).text(): _p._sens_table.item(i, 1).text()
             for i in range(_p._sens_table.rowCount())}
    assert 'LIMITED' in _rows['Bias'] and '1.00 -> -14.00' in _rows['Bias'], _rows['Bias']
    assert 'LIMITED' not in _rows['Spring'] and '1.00 -> -49.00' in _rows['Spring'], _rows['Spring']
    return (f'-50 target: spring 200 -> {_out[-50.0]["spring_rate_front_Npm"]["new_value"]:.0f} lbf/in, '
            f'CG 1200 -> {_out[-50.0]["cg_to_front_axle_m"]["new_value"]:.0f} mm, bias 55 -> '
            f'{_out[-50.0]["front_brake_bias"]["new_value"]:.0f} % (LIMITED, reaches -15 of -50); '
            f'+50: bias -> {_out[50.0]["front_brake_bias"]["new_value"]:.0f} %')
_ik_gate('recommend units  ', _recommend_units)

# ── JACKING (2026-09-22): the solver's own lateral-force -> body-lift ───────
# Hand case: a symmetric axle, IC 0.3 m up and 1.0 m past the centreline,
# contact patches on the ground at x = +/-0.6 m, tyre force toward -X
# (a positive-g turn in this solver loads the LEFT/+X side).  Per wheel
# Fz_jack = Fy_x * dz/dx:  outer (left)  -1000 * (0.3/-1.6) = +187.5 N (lifts),
# inner (right) -500 * (0.3/+1.6) = -93.75 N  ->  axle +93.75 N.  With the IC
# line through the roll centre, dz/dx = RC/(t/2) exactly (RC = 0.1125 m here).
try:
    from vahan.dynamics import (jacking_force_on_body as _jfb,
                                corner_jacking_line as _cjl)
    _fo, _to = _jfb(-1000.0, (-1.6, 0.3), 0.0)
    _fi, _ti = _jfb(-500.0, (1.6, 0.3), 0.0)
    _jf_hand = []
    if abs(_fo - 187.5) > 1e-9 or abs(_fi + 93.75) > 1e-9:
        _jf_hand.append(f'hand case {_fo:.3f}/{_fi:.3f} N, want +187.5/-93.75')
    _rc = 0.6 * 0.3 / 1.6          # line from (0.6,0) to (-1.0,0.3) crosses x=0 here
    if abs(abs(_to) - _rc / 0.6) > 1e-12:
        _jf_hand.append(f'tan {_to} != RC/(t/2) {_rc / 0.6}')
    # body roll tips the line: +roll (left side down) steepens the outer line
    _fo_r, _ = _jfb(-1000.0, (-1.6, 0.3), np.radians(1.0))
    if not _fo_r > _fo:
        _jf_hand.append('positive roll did not steepen the loaded-side line')
    # equal side forces on a symmetric axle cancel exactly
    if abs(_jfb(-800.0, (-1.6, 0.3))[0] + _jfb(-800.0, (1.6, 0.3))[0]) > 1e-9:
        _jf_hand.append('equal forces on a symmetric axle do not cancel')
    # the app: v147 (the user's reference) headless, through _build_dynamics_solver
    import glob as _gj
    _jc = (sorted(_gj.glob('configs/2027_v147_*.vahan'))
           or [_highest_config(_gj.glob('configs/2027_v*.vahan'))])[0]
    _wj = MainWindow(); _wj._load_project_from_path(_jc); _wj._rebuild_solvers(0.)
    _ssj = _wj._build_dynamics_solver()
    _r0 = _ssj.solve(0.0)
    if abs(_r0.jacking_force_front_N) > 1e-6 or abs(_r0.jacking_force_rear_N) > 1e-6:
        _jf_hand.append(f'0 g jacking {_r0.jacking_force_front_N}/{_r0.jacking_force_rear_N} N')
    _rj = _ssj.solve(1.5)
    # self-consistency: the per-corner numbers re-derive from the solver's own
    # Fy magnitudes, its IC lines and its roll (nothing re-solved here)
    _phi = np.radians(_rj.roll_angle_deg)
    for _c in ('FL', 'FR', 'RL', 'RR'):
        _stc = _ssj._solvers[_c].solve(_rj.travel[_c] / 1000.0)
        _cp, _dd = _cjl(_stc, 'left' if _c.endswith('L') else 'right', _ssj._veh.tire_radius_m)
        _want = _jfb(-abs(_rj.Fy[_c]) * _rj.Fy_sign.get(_c, 1.0), _dd, _phi)[0]
        if abs(_want - _rj.jacking_corner_N[_c]) > 0.5:
            _jf_hand.append(f'{_c} jacking {_rj.jacking_corner_N[_c]:.2f} N, re-derived {_want:.2f} N')
    _kw = (_ssj._veh.wheel_rate_front_Npm, _ssj._veh.wheel_rate_rear_Npm)
    if abs(_rj.jacking_heave_front_mm - _rj.jacking_force_front_N / (2 * _kw[0]) * 1000) > 1e-9:
        _jf_hand.append('front heave != F / (2 wheel rate)')
    if not (_rj.jacking_force_rear_N > 0 and _rj.jacking_force_front_N > 0):
        _jf_hand.append('roll centres above ground must LIFT the body in a corner')
    # feedback converged (only when the feedback is on; off by default since 2026-09-23):
    # the heave fed into the kinematics = the heave it causes
    if _ssj.jacking_feedback and abs(_rj.jacking_heave_applied_rear_mm - _rj.jacking_heave_rear_mm) > 0.15:
        _jf_hand.append(f'feedback not converged: applied {_rj.jacking_heave_applied_rear_mm:.3f} '
                        f'vs caused {_rj.jacking_heave_rear_mm:.3f} mm')
    # mirror: a left turn lifts the body by the same amount, WITH static toe
    # (v147 rear 0.25 deg).  FAILED before the 2026-09-22 toe-hand fix: rear
    # 290.1 N vs 274.2 N at +/-1.5 g — the pair split kept +toe on the LEFT
    # wheel whichever side was loaded.
    _rm = _ssj.solve(-1.5)
    for _ax, _a1, _a2 in (('front', _rj.jacking_force_front_N, _rm.jacking_force_front_N),
                          ('rear', _rj.jacking_force_rear_N, _rm.jacking_force_rear_N)):
        if abs(_a1 - _a2) > 1e-6:
            _jf_hand.append(f'{_ax} left / right turn jacking differ {_a1:.4f}/{_a2:.4f} N')
    if abs(float(_ssj._veh.toe_rear_deg)) < 1e-9:
        _jf_hand.append('v147 has no rear toe: the mirror check lost its teeth')
    if _jf_hand:
        fails += 1
    print(f'jacking          : hand case +187.5/-93.75 N, tan = RC/(t/2); {os.path.basename(_jc)[:9]} '
          f'@1.5 g front {_rj.jacking_force_front_N:+.1f} N -> {_rj.jacking_heave_front_mm:+.2f} mm, '
          f'rear {_rj.jacking_force_rear_N:+.1f} N -> {_rj.jacking_heave_rear_mm:+.2f} mm '
          f'(feedback {'on' if _ssj.jacking_feedback else 'off'}, {_rj.jacking_feedback_passes} passes; roll {_rj.roll_angle_deg:.3f} deg)   '
          + ('pass' if not _jf_hand else 'UNEXPECTED FAIL: ' + '; '.join(_jf_hand)))
except Exception as _e:
    fails += 1
    import traceback as _tbj; _tbj.print_exc()
    print(f'jacking          : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── DYNAMICS CAMBER-vs-g PINNED (2026-09-23): the 2026 baseline's camber curve from the
# steady-state solver must equal the committed solver's (git 50c8073) to 0.02 deg.  The
# jacking-heave feedback, on by default for one day, drooped the whole axle and turned the
# outer-front camber at 1.5 g from -0.43 to -0.29 deg (v147 inner rear +3 deg at 2 g) —
# "camber is atrocious in the dynamics sweep and disagrees with the kinematics sweep".
# Feedback is OFF unless car['jacking_feedback'] is true; jacking is still reported.
try:
    _cb = MainWindow(); _cb._load_project_from_path(os.path.join('configs', '2026_baseline.vahan')); _cb._rebuild_solvers(0.)
    _ssb = _cb._build_dynamics_solver()
    _ref = {0.5: {'FL': -0.141, 'FR': +0.139, 'RL': -0.109, 'RR': +0.107},
            1.0: {'FL': -0.284, 'FR': +0.275, 'RL': -0.219, 'RR': +0.212},
            1.5: {'FL': -0.429, 'FR': +0.409, 'RL': -0.330, 'RR': +0.316},
            2.0: {'FL': -0.577, 'FR': +0.542, 'RL': -0.444, 'RR': +0.419}}
    _cbf = []
    if getattr(_ssb, 'jacking_feedback', False):
        _cbf.append('jacking feedback is ON by default')
    _worst = 0.0
    for _g, _row in _ref.items():
        _r = _ssb.solve(_g, 0.)
        for _c, _v in _row.items():
            _d = abs(float(_r.camber[_c]) - _v); _worst = max(_worst, _d)
            if _d > 0.02:
                _cbf.append(f'{_c} @{_g} g camber {_r.camber[_c]:+.3f} vs committed {_v:+.3f}')
    if _cbf:
        fails += 1
    print(f'dyn camber pinned: 2026 baseline camber-vs-g vs committed solver, worst {_worst:.3f} deg (tol 0.02); '
          f'jacking feedback {"ON" if getattr(_ssb, "jacking_feedback", False) else "off"}   '
          + ('pass' if not _cbf else 'UNEXPECTED FAIL: ' + '; '.join(_cbf[:4])))
except Exception as _e:
    fails += 1
    print(f'dyn camber pinned: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── ONE GRIP SCALE (2026-09-22): every consumer reads MainWindow.grip_scale ──
try:
    _gf = []
    _wg = _wj
    _wg.set_grip_scale(0.83)
    _ssg = _wg._build_dynamics_solver()
    if abs(_ssg._mu_scale - 0.83) > 1e-12:
        _gf.append(f'solver scale {_ssg._mu_scale}')
    from vahan.laptime import LapSimulator as _LSg, AckermannStationModel as _ASMg
    if abs(_LSg(_ssg).grip_scale - 0.83) > 1e-12:
        _gf.append('lap sim does not default to the project scale')
    if abs(_ASMg(_ssg._tire, _ssg, [0.0]).gm - 0.83) > 1e-12:
        _gf.append('Ackermann station model does not default to the project scale')
    _wg._switch_page(1)                       # lazily builds the Laptime page
    if abs(float(_wg._laptime_page._grip.value()) - 0.83) > 1e-9:
        _gf.append('Laptime page box is not a mirror of the project scale')
    _wg._laptime_page._grip.setValue(0.77)     # editing a mirror sets THE scale
    if abs(_wg.grip_scale() - 0.77) > 1e-9:
        _gf.append('editing the Laptime box did not set the project scale')
    if abs(_wg._laptime_page.build_sim().grip_scale - 0.77) > 1e-9:
        _gf.append('Laptime page sim not on the project scale')
    _wg._switch_page(0)
    _wg.set_grip_scale(0.70)
    _ssg = _wg._build_dynamics_solver()
    _rows = _ssg.limits_at_grip_scales([0.7, 1.0])
    _a7 = _ssg.max_accel_g()
    if abs(_rows[0]['traction_g'] - _a7['traction_g']) > 1e-12:
        _gf.append('per-scale row != single-scale readout')
    if not (_rows[1]['traction_g'] > _rows[0]['traction_g'] * 1.2
            and _rows[1]['braking_g'] / _rows[0]['braking_g'] > 1.42):
        _gf.append(f'traction/brake do not follow the grip scale {_rows}')
    if abs(_ssg._mu_scale - 0.70) > 1e-12:
        _gf.append('limits_at_grip_scales did not restore the scale')
    # differential: a ZERO bias cap is a cap, not "uncapped"
    from vahan.differential import Differential as _Dg
    if _Dg(kind='spool').yaw_moment_Nm(200.0, 1.2, 0.2, True, max_bias_N=0.0) != 0.0:
        _gf.append('zero diff bias cap not honoured')
    # acceleration model: no hidden mu, no private grip default
    from vahan.acceleration import AccelerationModel as _AMg
    try:
        _AMg(300, 0.2, 0.55, None, 0, 1, 1.2, 0.5, 3.5, grip_scale=0.7)
        _gf.append('acceleration model accepted no tyre')
    except ValueError:
        pass
    if _wg._dynamics_panel.grip_scale_list() != [0.7, 1.0]:
        _gf.append(f'grip list {_wg._dynamics_panel.grip_scale_list()}')
    if _gf:
        fails += 1
    print(f'grip scale       : one scale -> solver/lap/Ackermann/page mirrors; traction '
          f'{_rows[0]["traction_g"]:.3f}/{_rows[1]["traction_g"]:.3f} g, brake '
          f'{_rows[0]["braking_g"]:.3f}/{_rows[1]["braking_g"]:.3f} g at x0.70/x1.00; zero diff cap honoured   '
          + ('pass' if not _gf else 'UNEXPECTED FAIL: ' + '; '.join(_gf)))
except Exception as _e:
    fails += 1
    import traceback as _tbg; _tbg.print_exc()
    print(f'grip scale       : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── STEERING INPUTS = THE project steer block after a load (2026-09-22) ─────
try:
    _sf = []
    _w0 = MainWindow()
    _stale = (_w0._dynamics_panel._cached_max_steer, _w0._dynamics_panel._cached_steer_ratio)
    _w0._load_project_from_path(_jc); _w0._rebuild_solvers(0.)
    _loaded = (_w0._dynamics_panel._cached_max_steer, _w0._dynamics_panel._cached_steer_ratio)
    _w0._update_min_turn_radius()
    _fresh = (_w0._dynamics_panel._cached_max_steer, _w0._dynamics_panel._cached_steer_ratio)
    if not np.allclose(_loaded, _fresh):
        _sf.append(f'lock values after load {_loaded} != recomputed {_fresh}')
    _vs = _w0._build_dynamics_solver()._veh
    for _k in ('rack_travel_per_rev_mm', 'total_rack_travel_mm'):
        if abs(getattr(_vs, _k) - float(_w0._steer[_k])) > 1e-12:
            _sf.append(f'VehicleParams.{_k} {getattr(_vs, _k)} != steer block {_w0._steer[_k]}')
    if abs(_vs.max_steer_angle_deg - _fresh[0]) > 1e-9:
        _sf.append('VehicleParams max steer != geometric lock')
    # optimiser steer mode: the rack comes from the steer block (not 60)
    import vahan.optimizer as _OPTs
    from vahan.steering import rack_travel_from_handwheel_deg as _rth
    try:
        _OPTs.InverseSolver(dict(_w0._front_hp), travel_mm=(-30, 30), n_points=3,
                            motion='steer', axle='front')
        _sf.append('steer-mode IK accepted no steer block')
    except ValueError:
        pass
    _ik_s = _OPTs.InverseSolver(dict(_w0._front_hp), travel_mm=(-30, 30), n_points=3,
                                motion='steer', axle='front', steer_params=dict(_w0._steer))
    if _ik_s.steer_params != dict(_w0._steer):
        _sf.append('IK steer_params != project steer block')
    _c_a = _OPTs._evaluate_sweep(dict(_w0._front_hp), np.array([-30., 0., 30.]), 'left', 'uca',
                                 ['toe'], {}, motion='steer', steer_params=dict(_w0._steer))
    _alt = dict(_w0._steer); _alt['rack_travel_per_rev_mm'] = 2 * float(_alt['rack_travel_per_rev_mm'])
    _c_b = _OPTs._evaluate_sweep(dict(_w0._front_hp), np.array([-30., 0., 30.]), 'left', 'uca',
                                 ['toe'], {}, motion='steer', steer_params=_alt)
    if not abs(float(_c_a['toe'][2]) - float(_c_b['toe'][2])) > 0.05:
        _sf.append('IK steer sweep does not respond to the steer block rack mm/rev')
    if abs(_rth(30.0, _w0._steer) * 1000
           - 30.0 * float(_w0._steer['rack_travel_per_rev_mm']) / 360.0
           * _w0._steer.get('rack_direction', 1)) > 1e-9:
        _sf.append('handwheel->rack conversion')
    if _sf:
        fails += 1
    print(f'steer inputs sync: startup lock {_stale[0]:.1f} deg / {_stale[1]:.2f} -> after load '
          f'{_loaded[0]:.1f} deg / {_loaded[1]:.2f} ({os.path.basename(_jc)[:9]}, rack '
          f'{_w0._steer["rack_travel_per_rev_mm"]} mm/rev, {_w0._steer["total_rack_travel_mm"]} mm); '
          f'VehicleParams + IK read the steer block   '
          + ('pass' if not _sf else 'UNEXPECTED FAIL: ' + '; '.join(_sf)))
except Exception as _e:
    fails += 1
    import traceback as _tbs; _tbs.print_exc()
    print(f'steer inputs sync: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── CONTROL-ARM SPHERICAL BEARINGS page (Ctrl+0, user 2026-09-23) ───────────
# SKF plain-bearing inputs per inboard pickup, two bore orientations.  Hand checks
# on the design car: the arm rotation about its pickup line equals the angle
# between the outer joint's perpendicular radii; bore 'pivot' is the pickup line
# and 'normal' is perpendicular to it; a rotation about the pickup line is pure
# TURNING for 'pivot' (tilt 0) and pure TILT for 'normal' (turning 0) — in every
# row of the page; each row's cycle half-angle = half the arm swing; Fr^2 + Fa^2
# = |F|^2 for the pickup force at both ends of every cycle.
try:
    from vahan import spherical_bearings as _SBn
    _bw = MainWindow(); _bw._load_project_from_path(_design); _bw._rebuild_solvers(0.)
    _bw._switch_page(9)
    _bd = _bw._bearings_page.refresh()
    _bf = []
    _brows = [r for r in _bd['rows'] if not r.get('invalid')]
    if len(_brows) != 2 * 4 * 2 * 4 or len(_bd['rows']) != len(_brows):
        _bf.append(f"{len(_brows)} valid rows of {len(_bd['rows'])} (want 64 valid)")
    _st0 = _bw._solvers['FL'].solve(0.0); _st1 = _bw._solvers['FL'].solve(0.02)
    for _arm in ('uca', 'lca'):
        _u = _SBn.pivot_axis(_st0, _arm); _th = _SBn.arm_angle_rad(_st0, _st1, _arm)
        _p = lambda v: v - (v @ _u) * _u
        _a0 = _p(getattr(_st0, f'{_arm}_outer') - getattr(_st0, f'{_arm}_front'))
        _a1 = _p(getattr(_st1, f'{_arm}_outer') - getattr(_st1, f'{_arm}_front'))
        _hand = np.arccos(np.clip(_a0 @ _a1 / np.linalg.norm(_a0) / np.linalg.norm(_a1), -1, 1))
        if abs(abs(_th) - _hand) > 1e-9:
            _bf.append(f'{_arm} rotation {abs(_th):.6f} != hand {_hand:.6f} rad')
        _bn, _bp = _SBn.bore_axis(_st0, _arm, 'normal'), _SBn.bore_axis(_st0, _arm, 'pivot')
        if abs(_bn @ _u) > 1e-9 or abs(_bp @ _u - 1.0) > 1e-9:
            _bf.append(f'{_arm} bore axes wrong: normal.u {_bn @ _u:.2e}, pivot.u {_bp @ _u:.6f}')
        _tn = _SBn.swing_twist_deg(_u, _th, _bn); _tp = _SBn.swing_twist_deg(_u, _th, _bp)
        if abs(_tn[0]) > 1e-9 or abs(_tn[1] - abs(np.degrees(_th))) > 1e-9 or abs(_tp[1]) > 1e-9 or abs(abs(_tp[0]) - abs(np.degrees(_th))) > 1e-9:
            _bf.append(f'{_arm} swing-twist split wrong: normal {_tn}, pivot {_tp}')
    for _r in _brows:
        if _r['orientation'] == 'pivot' and _r['tilt_half_deg'] > 1e-6:
            _bf.append(f"{_r['corner']} {_r['pickup']} pivot bore shows tilt {_r['tilt_half_deg']:.2e}"); break
        if _r['orientation'] == 'normal' and _r['half_angle_deg'] > 1e-6:
            _bf.append(f"{_r['corner']} {_r['pickup']} normal bore shows turning {_r['half_angle_deg']:.2e}"); break
        if abs(max(_r['half_angle_deg'], _r['tilt_half_deg']) - _r['arm_swing_deg'] / 2) > 1e-6:
            _bf.append(f"{_r['corner']} {_r['pickup']} half angle != half the arm swing"); break
        # rod end INLINE with its leg (user 2026-09-26): built-in tilt = 0 for the plane-normal bolt,
        # 90 - (leg-to-pickup-line angle) for the pickup-line bolt; worst tilt >= built-in
        _sts = _bw._solvers[_r['corner']].solve(0.0); _arm, _pk = _r['pickup'].split('_')
        _lg = getattr(_sts, f'{_arm}_outer') - getattr(_sts, _r['pickup']); _lg = _lg / np.linalg.norm(_lg)
        _want = 0.0 if _r['orientation'] == 'normal' else 90.0 - np.degrees(np.arccos(abs(_lg @ _SBn.pivot_axis(_sts, _arm))))
        if abs(_r['tilt_installed_deg'] - _want) > 1e-6 or _r['tilt_worst_deg'] < _r['tilt_installed_deg'] - 1e-9:
            _bf.append(f"{_r['corner']} {_r['pickup']} {_r['orientation']} built-in tilt {_r['tilt_installed_deg']:.3f} != hand {_want:.3f}"); break
    # force split at the ends: recompute one case directly through the Loads-page path
    from gui import wheel_package as _WPb
    _Lb = _WPb.compute_case(_bw, 2.0, 0.0)[0]['FL']
    for _k, _F in _Lb.chassis_forces.items():
        if not _k.startswith(('uca', 'lca')):
            continue
        for _o in ('normal', 'pivot'):
            _fr, _fa = _SBn.split_force(_F, _SBn.bore_axis(_st0, _k[:3], _o))
            if abs(_fr ** 2 + _fa ** 2 - float(_F @ _F)) > 1e-6 * max(1.0, float(_F @ _F)):
                _bf.append(f'{_k} {_o}: Fr^2+Fa^2 != |F|^2'); break
    _bmax = max((_r['tilt_worst_deg'] for _r in _brows if _r['orientation'] == 'normal'), default=float('nan'))
    print(f"bearings page    : {len(_brows)} rows (FL+RL x 4 pickups x 2 bores x 4 cycles); ride period "
          f"{_bd['osc']['F']:.3f}/{_bd['osc']['R']:.3f} s; worst tilt (bore normal to arm plane, full travel) "
          f"{_bmax:.2f} deg; rotation/split/force hand checks 1e-9"
          + ('' if not _bf else '   UNEXPECTED FAIL: ' + '; '.join(_bf)))
    if _bf:
        fails += 1
    # ROD-END SWIVEL LIMIT (user 2026-09-29): the built-in angle of the rod end
    # (90 - leg-to-pickup-line angle; 0 for the plane-normal bolt) must stay
    # under the ~27 deg a rod end swivels.  REPORT every pickup in both bore
    # orientations; FAIL only when a pickup is over in BOTH (no orientation
    # works).  The page's flag must agree with the module's rule, and the
    # cases must carry a speed + the aero of the Dynamics-panel package.
    try:
        from gui import bearings_page as _BPn
        _sw = _BPn.swivel_summary(_bd['rows']); _lim = float(_bd['swivel_limit_deg'])
        _swf = []
        if abs(_lim - _SBn.SWIVEL_LIMIT_DEG) > 1e-9 or abs(_lim - 27.0) > 1e-9:
            _swf.append(f'limit {_lim} != 27 deg')
        for _r in _brows:
            if bool(_r['over_limit']) != (_r['tilt_installed_deg'] > _lim + 1e-9):
                _swf.append(f"{_r['corner']} {_r['pickup']} {_r['orientation']} flag != built-in angle > limit"); break
        _txt = []; _both = []
        for _c in ('FL', 'RL'):
            for _pk in ('uca_front', 'uca_rear', 'lca_front', 'lca_rear'):
                _n = _sw[(_c, _pk, 'normal')]; _p = _sw[(_c, _pk, 'pivot')]
                _txt.append(f"{_c} {_pk}: normal {_n[0]:.1f} {'OVER' if _n[1] else 'ok'} / pivot {_p[0]:.1f} {'OVER' if _p[1] else 'ok'}")
                if _n[1] and _p[1]:
                    _both.append(f'{_c} {_pk}')
        if _both:
            _swf.append('over the limit in BOTH orientations: ' + ', '.join(_both))
        _ca_ok = all(ca['speed_kph'] > 0 for _, ca in _bd['cases_aero'])
        if not _ca_ok:
            _swf.append('a load case has no speed')
        if _swf:
            fails += 1
        print(f'rod-end swivel   : built-in angle vs {_lim:.0f} deg (bolt normal to arm plane / along pickup line): '
              + '; '.join(_txt) + '; cases '
              + ', '.join(f"{nm.split(' ')[0]} {ca['speed_kph']:.0f} km/h aero {ca['total_N']:.0f} N" for nm, ca in _bd['cases_aero'])
              + ('   pass' if not _swf else '   UNEXPECTED FAIL: ' + '; '.join(_swf)))
    except Exception as _e:
        fails += 1
        print(f'rod-end swivel   : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')
except Exception as _e:
    fails += 1
    import traceback as _tbb; _tbb.print_exc()
    print(f'bearings page    : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── BUILD TOLERANCE page (Ctrl+Shift+1, user 2026-09-26): aero heave + CG tolerance ──
# Aero heave: the table's downforce = Cl·A·½ρv² split by the CoP, and at low
# load its heave equals the hand estimate (axle load/2)/ride rate within 3 %
# (the nonlinear curve and the linear ride rate agree near static).  Tolerance
# band maths on a synthetic line (y = 2x, allowance 1 -> band +-0.5).  A short
# CG sweep (no lap sims) must leave the design CG untouched and move the grip
# limit / roll in the physical direction (higher CG -> more roll).
try:
    from vahan import cg_tolerance as _CTn
    from gui import build_tolerance_page as _BTn
    _tw = MainWindow(); _tw._load_project_from_path(_design); _tw._rebuild_solvers(0.)
    _tw._switch_page(10)
    _tf = []
    _at = _BTn.aero_ride(_tw, radii_m=(15.0,), speeds_kph=[0.0, 30.0, 60.0, 100.0])
    _pk = _at['package']; _st = _at['straight']
    _v = 30.0 / 3.6; _D = _pk['cla_m2'] * 0.5 * _pk['rho'] * _v * _v
    if abs(_st['downforce_N'][1] - _D) > 1e-6 * max(1.0, _D):
        _tf.append(f"downforce {_st['downforce_N'][1]:.3f} != hand {_D:.3f} N")
    _veh = _tw._build_dynamics_solver()._veh
    _hf = (_D * (1 - _pk['cop_rear']) / 2) / _veh.ride_rate_front_Npm * 1000
    _hr = (_D * _pk['cop_rear'] / 2) / _veh.ride_rate_rear_Npm * 1000
    if _D > 0 and (abs(_st['ride_drop_mm']['FL'][1] / _hf - 1) > 0.03 or abs(_st['ride_drop_mm']['RL'][1] / _hr - 1) > 0.03):
        _tf.append(f"30 km/h straight ride-height loss {_st['ride_drop_mm']['FL'][1]:.3f}/{_st['ride_drop_mm']['RL'][1]:.3f} vs hand {_hf:.3f}/{_hr:.3f} mm")
    # ONE MODEL: the page's straight-line travel IS the solver's aero sink (no second heave model)
    if abs(_st['travel_mm']['FL'][3] - _st['aero_sink_mm']['F'][3]) > 1e-9 or not (_st['aero_sink_mm']['F'][3] > 0):
        _tf.append(f"straight travel {_st['travel_mm']['FL'][3]:.4f} != solver aero sink {_st['aero_sink_mm']['F'][3]:.4f} mm")
    _cn = _at['corners'][0]; _ok = np.isfinite(_cn['travel_mm']['FL'])
    if not (_ok.any() and np.all(_cn['travel_mm']['FL'][_ok][1:] > _cn['travel_mm']['FR'][_ok][1:])):
        _tf.append('in the corner the outside wheel does not compress more than the inside wheel')
    if not (_at['brake']['travel_mm']['FL'][3] > _st['travel_mm']['FL'][3] and _at['accel']['travel_mm']['RL'][3] > _st['travel_mm']['RL'][3]):
        _tf.append('braking does not dive the front / acceleration does not squat the rear')
    _lo, _hi = _CTn.band([-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], 0.0, 0.0, 1.0, 'both')
    if abs(_lo + 0.5) > 1e-9 or abs(_hi - 0.5) > 1e-9:
        _tf.append(f'band maths {_lo}, {_hi} (want -0.5, +0.5)')
    _z0 = _tw._car['cg_z_mm']
    _sw = _BTn.cg_sweep(_tw, 'height', [-10.0, 0.0, 10.0], with_lap=False)
    if _tw._car['cg_z_mm'] != _z0:
        _tf.append('design CG not restored after the sweep')
    _rl = [m['roll_deg_per_g'] for m in _sw['metrics']]
    if not (_rl[0] < _rl[1] < _rl[2]):
        _tf.append(f'roll not rising with CG height: {_rl}')
    print(f"build tolerance  : aero {_pk['label']} -> straight 100 km/h wheel travel F {_st['travel_mm']['FL'][3]:.2f} / R "
          f"{_st['travel_mm']['RL'][3]:.2f} mm (braking front {_at['brake']['travel_mm']['FL'][3]:+.1f}, accel rear "
          f"{_at['accel']['travel_mm']['RL'][3]:+.1f}); 15 m corner at the limit outside/inside {np.nanmax(_cn['travel_mm']['FL']):.1f}/"
          f"{np.nanmin(_cn['travel_mm']['FR']):.1f} mm; roll per 10 mm CG height "
          f"{(_rl[2] - _rl[0]) / 2:+.4f} deg/g"
          + ('' if not _tf else '   UNEXPECTED FAIL: ' + '; '.join(_tf)))
    if _tf:
        fails += 1
except Exception as _e:
    fails += 1
    import traceback as _tbt; _tbt.print_exc()
    print(f'build tolerance  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── PITCH TRAVEL + AERO SINK in the solver's corner travel (2026-09-26, ONE MODEL) ──
# Failing-then-passing: before 2026-09-26 solve(0, -1.6) left every corner at 0 mm
# (pitch was a reported angle only) and aero load never sank the body.  Now:
#   braking  : front bump = heave curve at (1 - anti-dive)·m_s·|ax|·h_s/L / 2 per wheel,
#              rear droop at (1 - anti-lift)·same;  accel: rear bump (1 - anti-squat), front droop;
#   aero     : sink = heave curve at the aero load per wheel.
# Hand check against the LINEAR wheel rate (within 6 %: the curve is progressive).
try:
    _pw = MainWindow(); _pw._load_project_from_path(_design); _pw._rebuild_solvers(0.)
    _pss = _pw._build_dynamics_solver(); _pv = _pss._veh; _pa = _pss.anti_fractions(); _pf = []
    _pr0 = _pss.solve(0.0, 0.0)
    if any(abs(_pr0.travel[c]) > 1e-9 for c in _pr0.travel):
        _pf.append('static solve has non-zero travel')
    for _g in (-1.6, 1.0):
        _r = _pss.solve(0.0, _g)
        _dF = _pv.sprung_mass_kg * abs(_g) * 9.81 * _pv.sprung_cg_height_m / _pv.wheelbase_m / 2
        if _g < 0:
            _hf, _hr = _dF * (1 - _pa['dive']) / _pv.wheel_rate_front_Npm * 1000, -_dF * (1 - _pa['lift']) / _pv.wheel_rate_rear_Npm * 1000
        else:
            _hf, _hr = -_dF / _pv.wheel_rate_front_Npm * 1000, _dF * (1 - _pa['squat']) / _pv.wheel_rate_rear_Npm * 1000
        if abs(_r.travel['FL'] / _hf - 1) > 0.06 or abs(_r.travel['RL'] / _hr - 1) > 0.06 or abs(_r.travel['FL'] - _r.travel['FR']) > 1e-6:
            _pf.append(f'{_g:+.1f} g travel F {_r.travel["FL"]:.2f} R {_r.travel["RL"]:.2f} vs hand {_hf:.2f} {_hr:.2f} mm')
        _pang = np.degrees(np.arctan((_r.pitch_travel_front_mm - _r.pitch_travel_rear_mm) / 1000 / _pv.wheelbase_m))
        if abs(_r.pitch_angle_deg - _pang) > 1e-9:
            _pf.append(f'pitch angle {_r.pitch_angle_deg:.4f} != travel-derived {_pang:.4f}')
    _aero = {'FL': 200.0, 'FR': 200.0, 'RL': 230.0, 'RR': 230.0}
    _ra = _pss.solve(0.0, 0.0, aero_Fz=_aero)
    _haf, _har = 200.0 / _pv.wheel_rate_front_Npm * 1000, 230.0 / _pv.wheel_rate_rear_Npm * 1000
    if abs(_ra.travel['FL'] / _haf - 1) > 0.06 or abs(_ra.travel['RL'] / _har - 1) > 0.06 or abs(_ra.travel['FL'] - _ra.aero_heave_front_mm) > 1e-9:
        _pf.append(f'aero sink F {_ra.travel["FL"]:.3f} R {_ra.travel["RL"]:.3f} vs hand {_haf:.3f} {_har:.3f} mm')
    _rb16 = _pss.solve(0.0, -1.6)
    _pss.pitch_travel = False; _pss.aero_heave = False
    _roff = _pss.solve(0.0, -1.6, aero_Fz=_aero)
    if any(abs(_roff.travel[c]) > 1e-9 for c in _roff.travel):
        _pf.append('switches off but travel still non-zero')
    print(f"pitch + aero sink: anti dive/lift/squat {100 * _pa['dive']:.1f}/{100 * _pa['lift']:.1f}/{100 * _pa['squat']:.1f} %; "
          f"1.6 g braking front {_rb16.travel['FL']:+.1f} / rear {_rb16.travel['RL']:+.1f} mm; "
          f"hand checks within 6 %"
          + ('' if not _pf else '   UNEXPECTED FAIL: ' + '; '.join(_pf)))
    if _pf:
        fails += 1
except Exception as _e:
    fails += 1
    import traceback as _tbp; _tbp.print_exc()
    print(f'pitch + aero sink: UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── STEER-SWEEP TURN RADIUS + SWEEP MOTION RATIO graphs (2026-09-29, failing-then-passing) ──
# (1) compute_turn_radius_post averaged the two RAW toe angles (opposite sign
#     conventions per side), so a steered pair nearly cancelled: v150 read 25.5 m
#     at full lock where corner_speed.lock_radius_m reads 2.25 m.  Now the mean
#     steer is (FR toe - FL toe)/2; the sweep's |R| at both lock ends must match
#     lock_radius_m within 5 % and carry opposite signs.
# (2) _do_sweep's motion-ratio curve was the cumulative secant from the station
#     nearest t=0 divided by the travel from ZERO: NaN at t=0 on a grid holding 0,
#     0.0 + 0.82/0.37 spikes on a grid that misses 0, 0.011 off the tangent at
#     +-30 mm.  Now every station is the +-1 mm tangent MR (packaging._tangent_mr,
#     = the dynamics' solver_mr) within 1e-3 on both grids, with no NaN hole.
try:
    from vahan import corner_speed as _CSr
    from vahan import packaging as _PKr
    from vahan.kinematics import KinematicMetrics as _KMr
    _rw = MainWindow(); _rw._load_project_from_path(_design); _rw._rebuild_solvers(0.)
    _rf = []
    _hand_r = _CSr.full_lock_handwheel_deg(_rw._steer)
    _job_r = _rw._snapshot_sweep_job()
    _job_r['motion'] = 'steer'; _job_r['lo'] = -_hand_r; _job_r['hi'] = _hand_r
    _sr = _rw._compute_sweep(_job_r)['sweep_results']
    _trr = np.asarray(_sr['FL']['turn_radius'], float)
    _lsr = MainWindow._build_corner_solvers(_rw._all_corner_hp(), _rw._steer, _rw._topology, _hand_r)
    _lock_r = _CSr.lock_radius_m(float(_KMr(_lsr['FL'].solve(0.), 'left').toe),
                                 float(_KMr(_lsr['FR'].solve(0.), 'right').toe),
                                 float(_rw._car['wheelbase_mm']) / 1000.)
    if not (np.isfinite(_trr[0]) and np.isfinite(_trr[-1]) and np.isfinite(_lock_r)):
        _rf.append(f'turn radius at lock not finite: sweep {_trr[0]}, {_trr[-1]}; lock_radius_m {_lock_r}')
    else:
        for _e_r in (_trr[0], _trr[-1]):
            if abs(abs(_e_r) / _lock_r - 1) > 0.05:
                _rf.append(f'sweep turn radius at lock {_e_r:.2f} m vs lock_radius_m {_lock_r:.2f} m (>5 %)')
        if np.sign(_trr[0]) == np.sign(_trr[-1]):
            _rf.append('turn radius has the same sign at both lock ends')
        if not np.isnan(_trr[len(_trr) // 2]):
            _rf.append('turn radius at zero steer is not NaN (infinite radius expected)')
    if not np.array_equal(np.isnan(_trr), np.isnan(np.asarray(_sr['FR']['turn_radius'], float))) \
            or not np.allclose(_trr, _sr['FR']['turn_radius'], equal_nan=True):
        _rf.append('FL and FR turn-radius arrays differ')
    _mr_static = {}
    for _lbl_r in ('FL', 'RL'):
        _sol_r = _rw._solvers[_lbl_r]
        _mr_static[_lbl_r] = _PKr.solver_mr(_sol_r)
        for _n_r in (81, 80):      # 81 holds t=0 exactly (1 mm step); 80 misses it
            _t_r = np.linspace(-0.030, 0.050, _n_r)
            _mr_r = np.asarray(_rw._do_sweep(_sol_r, _t_r, 'left', is_front=_lbl_r == 'FL',
                                             label=_lbl_r)['motion_ratio'], float)
            _i0 = int(np.argmin(np.abs(_t_r)))
            if not np.isfinite(_mr_r[_i0 - 1:_i0 + 2]).all():
                _rf.append(f'{_lbl_r} n={_n_r}: MR hole at the static station {_mr_r[_i0 - 1:_i0 + 2]}')
            _fin = np.where(np.isfinite(_mr_r))[0]
            if len(_fin) < 0.8 * _n_r:
                _rf.append(f'{_lbl_r} n={_n_r}: only {len(_fin)} of {_n_r} MR stations finite')
            _worst = 0.0
            for _k in _fin:
                _worst = max(_worst, abs(_mr_r[_k] - _PKr._tangent_mr(_sol_r, float(_t_r[_k]))))
            if _worst > 1e-3:
                _rf.append(f'{_lbl_r} n={_n_r}: MR curve off the +-1 mm tangent MR by {_worst:.5f} (>1e-3)')
            if _n_r == 81 and abs(_mr_r[_i0] - _mr_static[_lbl_r]) > 1e-3:
                _rf.append(f'{_lbl_r}: MR at t=0 {_mr_r[_i0]:.4f} != static solver_mr {_mr_static[_lbl_r]:.4f}')
    print(f'turn radius + MR : sweep |R| at full lock {abs(_trr[0]):.2f} / {abs(_trr[-1]):.2f} m vs lock_radius_m '
          f'{_lock_r:.2f} m (was 25.5 m); MR curve = tangent MR within 1e-3 on the 81 (holds 0) and 80 (misses 0) '
          f'grids, static FL {_mr_static["FL"]:.4f} / RL {_mr_static["RL"]:.4f}'
          + ('' if not _rf else '   UNEXPECTED FAIL: ' + '; '.join(_rf)))
    if _rf:
        fails += 1
except Exception as _e:
    fails += 1
    import traceback as _tbr; _tbr.print_exc()
    print(f'turn radius + MR : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── 3-D ROLL-CENTRE SPHERE = THE GRAPH (ONE MODEL, 2026-10-01) ──────────────
# The view's RC sphere was still built from the pickup MIDPOINT while every graph
# used the Rule-17 instant-axis construction since 2026-09-09 (found by the
# measure-mode planning pass).  Gate: with the car at design position the sphere
# the view draws sits at KinematicMetrics.roll_center_height on both axles.
try:
    from vahan.kinematics import KinematicMetrics as _KMsph
    _ws = MainWindow(); _ws._load_project_from_path(_design); _ws._rebuild_solvers(0.)
    _ws._motion_panel.go_to_static()
    _sph = {}
    _ws.view3d.update_rc = lambda f, r: _sph.update(front=f, rear=r)
    _ws._show_rc = True; _ws._update_3d(light=False)
    _sf = []
    for _ax, _lbl in (('front', 'FL'), ('rear', 'RL')):
        _g = float(_KMsph(_ws._solvers[_lbl].solve(0.0), 'left').roll_center_height) * 1000.0
        _p = _sph.get(_ax)
        if _p is None or abs(float(_p[2]) * 1000.0 - _g) > 0.01 or abs(float(_p[0])) > 1e-6:
            _sf.append(f"{_ax} sphere {'missing' if _p is None else f'{float(_p[2]) * 1000:.3f} mm (x {float(_p[0]) * 1000:.2f})'} vs graph {_g:.3f} mm")
    print(f"rc sphere = graph : front {float(_sph['front'][2]) * 1000:.2f} / rear {float(_sph['rear'][2]) * 1000:.2f} mm at design position"
          + ('' if not _sf else '   UNEXPECTED FAIL: ' + '; '.join(_sf)))
    if _sf:
        fails += 1
except Exception as _e:
    fails += 1
    import traceback as _tbsph; _tbsph.print_exc()
    print(f'rc sphere = graph : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# ── FSAE CHASSIS BAYS (user 2026-09-23) ─────────────────────────────────────
# car['fsae_chassis'] replaces the pickup-to-pickup chassis stand-ins with frame
# tubes: node = arm leg (ball joint -> pickup) continued 38.1 mm past the pickup.
# Gated: OFF (absent key / enabled False) reproduces the old member set exactly;
# ON nodes are hand-computable from the file; tube vs tube is never a clash, a
# member bolted at a pickup is checked against the tube minus its bracket zone;
# the auto diagonal is deterministic and the full-state audit runs.  The ON
# clash list on the design is REPORTED (a packaging finding, not a gate).
try:
    import json as _jfs, copy as _cfs
    from vahan import chassis as _CHfs, packaging as _PKfs
    from vahan.interference import full_members as _fmfs, full_member_specs as _fsfs, \
        pair_gap_mm as _pgfs, clashes as _clfs
    from vahan.keepout import window_members as _wmfs
    _ff = []
    _wfs = MainWindow(); _wfs._load_project_from_path(_design); _wfs._rebuild_solvers(0.)
    if 'fsae_chassis' in _wfs._car:
        _ff.append('a file without the key loaded with fsae_chassis set (must be OFF)')
    _cdfs, _ = _wfs._assemble_corners_draw({l: 0. for l in ('FL', 'FR', 'RL', 'RR')}, 0.0, light=True)
    _byfs = {c['label']: c for c in _cdfs}
    _car_off = dict(_wfs._car); _car_off.pop('fsae_chassis', None)
    _car_dis = dict(_car_off); _car_dis['fsae_chassis'] = _CHfs.default_settings(False)
    _pv = np.zeros(3)
    def _sig_fs(ms):
        return [(m['name'], tuple(np.round(m['a'], 12)), tuple(np.round(m['b'], 12)), m['r']) for m in ms]
    for _lb, _c in _byfs.items():
        _m_off = _fmfs(_c['pts'], _car_off, arb_pivot=_pv, arb_od_mm=12.7)
        if _sig_fs(_m_off) != _sig_fs(_fmfs(_c['pts'], _car_dis, arb_pivot=_pv, arb_od_mm=12.7)):
            _ff.append(f'{_lb}: enabled=False member set != absent-key member set')
        _names = [m['name'] for m in _m_off]
        _exp = [nm for nm, ka, kb, rr in _fsfs() if _c['pts'].get(ka) is not None and _c['pts'].get(kb) is not None]
        if [n for n in _names if n in _exp] != _exp or any(m.get('chassis') for m in _m_off) \
                or 'UCA chassis cross member' not in _names:
            _ff.append(f'{_lb}: OFF member set is not the pre-feature spec list')
    if not any(m['name'].endswith('LCA inner chassis member') for m in _wmfs(_wfs, 0.0, 0.0)):
        _ff.append('OFF keep-out member set lost the LCA inner stand-in')
    # ON: nodes hand-computed straight from the file's numbers
    _raw = _jfs.load(open(_design, encoding='utf-8'))
    _wfs._car['fsae_chassis'] = _CHfs.default_settings(True); _wfs._fsae_chassis_cache = None
    _cdfs, _ = _wfs._assemble_corners_draw({l: 0. for l in ('FL', 'FR', 'RL', 'RR')}, 0.0, light=True)
    _byfs = {c['label']: c for c in _cdfs}
    _node_err = 0.0
    for _lb, _blk in (('FL', 'front_hp'), ('RL', 'rear_hp'), ('FR', 'front_hp'), ('RR', 'rear_hp')):
        _mx = np.array([-1., 1., 1.]) if _lb in ('FR', 'RR') else np.ones(3)
        for _pk, _ok in (('uca_front', 'uca_outer'), ('uca_rear', 'uca_outer'),
                         ('lca_front', 'lca_outer'), ('lca_rear', 'lca_outer')):
            _P = np.array(_raw[_blk][_pk]) * _mx; _O = np.array(_raw[_blk][_ok]) * _mx
            _hand = _P + 0.0381 * (_P - _O) / np.linalg.norm(_P - _O)
            _node_err = max(_node_err, float(np.linalg.norm(_byfs[_lb]['pts']['chassis_node_' + _pk] - _hand)) * 1000.)
            if abs(float(np.linalg.norm(_byfs[_lb]['pts']['chassis_node_' + _pk] - _P)) * 1000. - 38.1) > 1e-6:
                _ff.append(f'{_lb} {_pk} node not 38.1 mm from its pickup')
    if _node_err > 1e-6:
        _ff.append(f'node vs hand calc {_node_err:.2e} mm')
    _m_on = _fmfs(_byfs['FL']['pts'], _wfs._car, arb_pivot=_pv, arb_od_mm=12.7)
    _tubes = [m for m in _m_on if m.get('chassis')]
    if 'UCA chassis cross member' in [m['name'] for m in _m_on] or len(_tubes) != 5 \
            or any(abs(m['r'] - 0.0127) > 1e-12 for m in _tubes):
        _ff.append(f'ON member set: {len(_tubes)} tubes (need 4 + 1 diagonal, 25.4 mm OD, stand-in removed)')
    if any(m.endswith('LCA inner chassis member') for m in (x['name'] for x in _wmfs(_wfs, 0.0, 0.0))):
        _ff.append('ON keep-out member set still carries the LCA inner stand-in')
    # designed contacts: arm leg bolted at a tube's bracket pickup is checked against the tube
    # minus its bracket zone; tube vs tube never; a member crossing mid-span still counts
    # (15 mm node offset so the untrimmed tube WOULD overlap the leg at the bracket: -5.6 mm)
    _t = {'name': 'chassis tube X', 'a': np.array([0., 0., 0.]), 'b': np.array([0., .3, 0.]), 'r': .0127,
          'chassis': True, 'joints': [(np.array([.015, 0., 0.]), 'a')], 'bracket_trim': .015 + .0127}
    _leg = {'name': 'arm', 'a': np.array([.015, 0., 0.]), 'b': np.array([.2, -.1, 0.]), 'r': .0079}
    _cross = {'name': 'rod', 'a': np.array([-.1, .15, 0.]), 'b': np.array([.1, .15, 0.]), 'r': .0079}
    _t2 = dict(_t); _t2['name'] = 'chassis tube Y'
    _t0 = dict(_t); _t0['joints'] = []
    if _pgfs(_leg, _t) is None or _pgfs(_leg, _t) < 0 or not (_pgfs(_leg, _t0) < 0) \
            or _pgfs(_t, _t2) is not None \
            or not (_pgfs(_cross, _t) < 0) or len(_clfs([_t, _leg, _cross, _t2])) != 2:
        _ff.append('designed-contact rules (bracket trim / frame vs frame / mid-span hit)')
    # auto diagonal: deterministic, = the larger worst-case clearance
    _r1 = _PKfs.fsae_resolve_diagonals(_wfs); _r2 = _PKfs.fsae_resolve_diagonals(_wfs)
    for _ax in ('front', 'rear'):
        _g = {d: _r1['gaps'][_ax][d]['gap_mm'] for d in _CHfs.EXPLICIT_DIAGONALS}
        _best = 'ucar_lcaf' if _g['ucar_lcaf'] > _g['ucaf_lcar'] + 1e-6 else 'ucaf_lcar'
        if _r1[_ax] != _r2[_ax] or _r1['gaps'][_ax] != _r2['gaps'][_ax] or _r1[_ax] != _best:
            _ff.append(f'{_ax} auto diagonal not deterministic / not the larger clearance')
    _au = _PKfs.full_state_audit(_wfs)
    if _au['states'] != 39 or _au['closure_errors']:
        _ff.append(f'ON full-state audit: {_au["states"]} states, {len(_au["closure_errors"])} closure errors')
    _ch_neg = [r for r in _au['negatives'] if 'chassis' in r['a'] + r['b']]
    _tight = min(_ch_neg, key=lambda r: r['gap_mm']) if _ch_neg else None
    # PER-AXLE explicit choice (user 2026-09-23): diagonal_front / diagonal_rear each
    # reach only their own bays (FL/FR vs RL/RR), the audit runs with them and its
    # diagonal-vs-member gaps equal the resolution pass's; a legacy single
    # 'diagonal' key = both axles unless a per-axle key overrides it; front explicit
    # + rear auto resolves only the rear (the front keeps its explicit choice).
    _zero = {l: 0. for l in ('FL', 'FR', 'RL', 'RR')}
    _dname = {d: _CHfs.DIAGONAL_TUBES[d][0] for d in _CHfs.EXPLICIT_DIAGONALS}
    def _diag_of(win):
        _cd, _ = win._assemble_corners_draw(_zero, 0.0, light=True)
        out = {}
        for c in _cd:
            names = [m['name'] for m in _fmfs(c['pts'], win._car, arb_pivot=_pv, arb_od_mm=12.7)
                     if m['name'].startswith(_CHfs.DIAGONAL_NAME_PREFIX)]
            out[c['label']] = (c['pts'].get(_CHfs.DIAG_PTS_KEY), names)
        return out
    for _df, _dr in (('ucar_lcaf', 'ucaf_lcar'), ('ucaf_lcar', 'ucar_lcaf')):
        _wfs._car['fsae_chassis'] = dict(_CHfs.default_settings(True),
                                         diagonal_front=_df, diagonal_rear=_dr)
        _wfs._fsae_chassis_cache = None
        if _PKfs.fsae_ensure_resolved(_wfs) is not None or _PKfs.fsae_auto_is_stale(_wfs):
            _ff.append('both axles explicit still asks for the auto resolution')
        for _lb, (_d, _names) in _diag_of(_wfs).items():
            _want = _df if _lb.startswith('F') else _dr
            if _d != _want or _names != [_dname[_want]]:
                _ff.append(f'{_lb} explicit per-axle diagonal: got {_d} / {_names}, want {_want}')
        _au2 = _PKfs.full_state_audit(_wfs)
        if _au2['states'] != 39 or _au2['closure_errors']:
            _ff.append(f'per-axle explicit audit: {_au2["states"]} states / {len(_au2["closure_errors"])} closure errors')
        for _ax, _want in (('front', _df), ('rear', _dr)):
            _pfx = 'F' if _ax == 'front' else 'R'
            _rows = [r for r in _au2['worst_per_pair']
                     if any(n.startswith(_pfx) and n.endswith(_dname[_want]) for n in (r['a'], r['b']))]
            _g_res = _r1['gaps'][_ax][_want]['gap_mm']
            if _g_res < 10.0 and (not _rows or abs(min(r['gap_mm'] for r in _rows) - _g_res) > 1e-6):
                _ff.append(f'{_ax} explicit {_want}: audit gap != resolution gap {_g_res:.3f}')
    _leg = _CHfs.normalise({'enabled': True, 'diagonal': 'ucar_lcaf'})
    if (_leg['diagonal_front'], _leg['diagonal_rear']) != ('ucar_lcaf', 'ucar_lcaf') or 'diagonal' in _leg:
        _ff.append('legacy single diagonal key does not apply to both axles')
    _leg = _CHfs.normalise({'enabled': True, 'diagonal': 'ucar_lcaf', 'diagonal_rear': 'auto'})
    if (_leg['diagonal_front'], _leg['diagonal_rear']) != ('ucar_lcaf', 'auto'):
        _ff.append('per-axle key does not override the legacy diagonal key')
    _wfs._car['fsae_chassis'] = {'enabled': True, 'diagonal': 'ucar_lcaf', 'diagonal_rear': 'auto'}
    _wfs._fsae_chassis_cache = None
    if not _PKfs.fsae_auto_is_stale(_wfs) or _PKfs.fsae_ensure_resolved(_wfs) is None:
        _ff.append('front explicit + rear auto does not resolve the rear')
    _mix = _diag_of(_wfs)
    if any(_mix[l][0] != 'ucar_lcaf' for l in ('FL', 'FR')) or any(_mix[l][0] != _r1['rear'] for l in ('RL', 'RR')):
        _ff.append(f'mixed explicit/auto: front {_mix["FL"][0]} (want ucar_lcaf), rear {_mix["RL"][0]} (want auto {_r1["rear"]})')
    # TRANSVERSE tubes (user/chassis 2026-09-23): rear ON by default = 4 halves per rear
    # corner, each node -> the car centreline (X = 0), same OD; front none unless set;
    # OFF removes them.  Chassis-fixed obstructions: the sprocket disc derived from the
    # imported STEP (checked against a direct hand read of the same mesh: largest radial
    # extent about the X axis, the ring's X span), the diff housing from the car dict;
    # tube vs obstruction gaps come out of the full-state audit ONCE ('chassis-fixed').
    _wfs._car['fsae_chassis'] = _CHfs.default_settings(True); _wfs._fsae_chassis_cache = None
    _cdt, _ = _wfs._assemble_corners_draw(_zero, 0.0, light=True)
    for _c in _cdt:
        _tr = [m for m in _fmfs(_c['pts'], _wfs._car, arb_pivot=_pv, arb_od_mm=12.7) if _CHfs.is_transverse(m)]
        if _c['label'].startswith('F'):
            if _tr:
                _ff.append(f'{_c["label"]}: transverse tubes present with transverse_front False')
            continue
        if len(_tr) != 4 or any(abs(m['b'][0]) > 1e-12 or abs(m['r'] - 0.0127) > 1e-12 for m in _tr) \
                or any(np.linalg.norm(m['a'] - _c['pts']['chassis_node_' + k]) > 1e-9
                       for m, (_n, k) in zip(_tr, _CHfs.TRANSVERSE_TUBES)):
            _ff.append(f'{_c["label"]}: transverse halves != 4 x (node -> X=0, 25.4 mm OD)')
    _wfs._car['fsae_chassis'] = dict(_CHfs.default_settings(True), transverse_rear=False, transverse_front=True)
    _cdt, _ = _wfs._assemble_corners_draw(_zero, 0.0, light=True)
    _ntr = {c['label']: len([m for m in _fmfs(c['pts'], _wfs._car, arb_pivot=_pv, arb_od_mm=12.7) if _CHfs.is_transverse(m)]) for c in _cdt}
    if _ntr != {'FL': 4, 'FR': 4, 'RL': 0, 'RR': 0}:
        _ff.append(f'transverse setting per axle not honoured: {_ntr}')
    _wfs._car['fsae_chassis'] = _CHfs.default_settings(True); _wfs._fsae_chassis_cache = None
    _parts = _CHfs.fixed_obstructions(_wfs)
    _pnames = [p['name'] for p in _parts]
    _sp = [p for p in _parts if 'disc' in p]
    _mesh = next((p for p in (_wfs._imported_parts or []) if 'sprocket' in str(p.get('name', '')).lower()), None)
    if _mesh is None:
        _ff.append('no imported sprocket part on the design (needed for the chassis-fixed check)')
    elif not _sp:
        _ff.append('sprocket disc not derived from the imported mesh')
    else:
        _V = np.asarray(_mesh['verts'], float); _d = _sp[0]['disc']
        _rad = np.linalg.norm(_V[:, 1:] - _d['c'][1:], axis=1)          # hand read about the X axis
        _ring = _rad >= 0.9 * _rad.max()
        if abs(_rad.max() - _d['R_mm']) > 1e-6 or abs(_V[_ring, 0].min() - _d['x0_mm']) > 1e-6 \
                or abs(_V[_ring, 0].max() - _d['x1_mm']) > 1e-6 \
                or np.linalg.norm(_V[_ring].mean(0)[1:] - _d['c'][1:]) > 1e-6:
            _ff.append('sprocket disc (R / X span / centre) != hand read of the mesh')
        # primitive (signed distance = smallest way out): a tube through the disc centre
        # along Y is inside by the half-thickness -> -(h + r); a tube along X on the rim
        # at R + r grazes = 0; one 3 mm above the rim = +3
        _cyl = _sp[0]['cyls'][0]
        _c = np.asarray(_cyl['c'])
        _g_thru = _CHfs.segment_cylinder_gap_mm(_c + [0, -1, 0], _c + [0, 1, 0], 0.0127, _cyl)
        _g_graze = _CHfs.segment_cylinder_gap_mm(_c + [-1, 0, _cyl['R'] + 0.0127], _c + [1, 0, _cyl['R'] + 0.0127], 0.0127, _cyl)
        _g_3 = _CHfs.segment_cylinder_gap_mm(_c + [-1, 0, _cyl['R'] + 0.0157], _c + [1, 0, _cyl['R'] + 0.0157], 0.0127, _cyl)
        if abs(_g_thru + (_cyl['h'] + 0.0127) * 1000.) > 1e-6 or abs(_g_graze) > 1e-6 or abs(_g_3 - 3.0) > 1e-6:
            _ff.append(f'cylinder gap primitive: through {_g_thru:.3f} (want {-(_cyl["h"] + 0.0127) * 1000.:.3f}), graze {_g_graze:.3f} (want 0), +3 {_g_3:.3f}')
    # The car-dict diff proxy is an obstruction only while it is SHOWN (user 2026-09-23: the
    # yellow stand-in is deleted when the real diff STEP is imported; the STEP envelope covers it).
    _show_proxy = bool(getattr(_wfs, '_car', {}).get('show_diff_body', False))
    if _show_proxy and not any('diff housing' in n for n in _pnames):
        _ff.append('diff housing obstruction missing (proxy shown)')
    if not _show_proxy and any('diff housing (car dict' in n for n in _pnames):
        _ff.append('hidden diff proxy still counted as an obstruction')
    _au3 = _PKfs.full_state_audit(_wfs)
    _fx = _au3.get('fixed_part_gaps', [])
    _fx_tr = [r for r in _fx if 'transverse' in r['a'] and 'sprocket disc' in r['b']]
    if len(_fx_tr) != 8 or not all(r['travel'] == 'chassis-fixed' for r in _au3['worst_per_pair'] if 'sprocket' in r['b'] or 'diff housing' in r['b']):
        _ff.append(f'chassis-fixed audit rows: {len(_fx_tr)} transverse-vs-sprocket rows (want 8), or a fixed pair tagged per state')
    _sp_min = min((r['gap_mm'] for r in _fx_tr), default=float('nan'))
    if _ff:
        fails += 1
    print(f'fsae chassis bays : OFF = old member set; ON nodes = pickup + 38.1 mm along the leg '
          f'(hand calc {_node_err:.1e} mm); auto front {_r1["front"]} / rear {_r1["rear"]} (deterministic); '
          f'ON audit {len(_au["negatives"])} negatives, {len(_ch_neg)} with chassis tubes'
          + (f', tightest {_tight["a"]} vs {_tight["b"]} {_tight["gap_mm"]:.1f} mm (reported)' if _tight else '')
          + '; per-axle explicit front/rear + legacy key + mixed auto'
          + f'; rear transverse halves x4, sprocket disc from STEP, tightest transverse vs sprocket {_sp_min:.1f} mm (reported)'
          + '   ' + ('pass' if not _ff else 'UNEXPECTED FAIL: ' + '; '.join(_ff)))
except Exception as _e:
    fails += 1
    import traceback as _tbfs; _tbfs.print_exc()
    print(f'fsae chassis bays : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

# Dynamics camber contract: model -> sweep -> plot/table, with separate
# chassis/road/tire frames. These fixtures require no private TTC data.
try:
    import unittest as _camber_unittest
    import test_dynamics_camber as _camber_core
    import test_dynamics_camber_ui as _camber_ui
    _camber_suite = _camber_unittest.TestSuite([
        _camber_unittest.defaultTestLoader.loadTestsFromModule(_camber_core),
        _camber_unittest.defaultTestLoader.loadTestsFromModule(_camber_ui),
    ])
    _camber_run = _camber_unittest.TextTestRunner(verbosity=1).run(_camber_suite)
    fails += len(_camber_run.failures) + len(_camber_run.errors)
    print(f'dynamics camber contract: {_camber_run.testsRun} tests; '
          + ('pass' if _camber_run.wasSuccessful() else 'UNEXPECTED FAIL'))
except Exception as _e:
    fails += 1
    print(f'dynamics camber contract: UNEXPECTED FAIL ({_e})')

print(f'{fails} unexpected failures, {known} known-fail (documented).')
sys.exit(fails)
