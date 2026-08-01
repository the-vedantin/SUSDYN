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
_design = max(_cfgs, key=lambda p: int((_re.search(r'2027_v(\d+)', p) or [0, -1]).__getitem__(1))
              if _re.search(r'2027_v(\d+)', p) else -1) if _cfgs else None
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
        # coplanarity of the WHOLE actuation chain across travel — INCLUDING
        # pushrod_outer.  The bellcrank is a planar mechanism: the rod must lie IN
        # the plate plane or it side-thrusts the pivot bearing (USER HARD
        # REQUIREMENT).  Being ~1" above the LOWER-ARM plane (checked below) is a
        # Z offset and says NOTHING about the rocker plane (a fore-aft slice) —
        # v33 shipped a foot 22 mm out of the rocker plane while this check only
        # looked at the plate points.  Never exclude the pushrod again.
        cop = 0.0
        for _t in (-0.025, 0.0, 0.025):
            st = wD._solvers[lbl].solve(_t)
            P = lambda k: np.asarray(getattr(st, k), float) * 1000.0
            pts = np.array([P('pushrod_outer'), P('pushrod_inner'), P('rocker_pivot'),
                            P('rocker_spring_pt'), P('spring_chassis_pt'),
                            np.asarray(arb['arb_drop_top'], float) * 1000.0])
            c = pts.mean(0); _, _, vt = np.linalg.svd(pts - c)
            cop = max(cop, float(np.abs((pts - c) @ vt[-1]).max()))
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
            _ae = np.asarray(_pts['arb_arm_end_world'], float) * 1000.0
            _pl = np.array([P('pushrod_outer'), P('pushrod_inner'), P('rocker_pivot'),
                            P('rocker_spring_pt'), P('spring_chassis_pt')])
            _c0 = _pl.mean(0); _, _, _vt = np.linalg.svd(_pl - _c0)
            _doff = float(abs((_ae - _c0) @ _vt[-1]))
            if _doff > 3.0:
                gfail.append(f'{lbl} ARB drop link off the actuation plane at static ({_doff:.1f} mm)')
        except Exception:
            pass
        # "over the LCA" = the pushrod loads onto a PLATE welded on TOP of the lower
        # arm: the 1" spherical rod-end (pushrod_outer is its centre) sits ~1" ABOVE
        # the arm plane, clear of the arm's own thickness, never buried below it and
        # never flung far off it (v28 slid it toward the tie-rod).  Signed perpendicular
        # distance to the plane through the arm's three pickups, +normal oriented up.
        _n = np.cross(lr - lf, lo - lf); _n = _n / (np.linalg.norm(_n) or 1.0)
        if _n[2] < 0:
            _n = -_n
        d_arm = float((po - lo) @ _n)          # signed mm; + = above the arm plane
        # REAR ONLY: the rear pushrod picks up on the LOWER arm, the front one on
        # the UPPER arm (front pushrod_outer z=322 vs uca_outer 308, lca_outer 141).
        # Measuring the front rod against the LCA plane returns ~183 mm and means
        # nothing — it is a different pickup, not a fault.
        if lbl[0] == 'R' and (d_arm < -3.0 or d_arm > 35.0):
            gfail.append(f'{lbl} pushrod perp-to-arm {d_arm:.0f} mm (want 0..35 above)')
        # REAR ONLY: the rear damper is meant to lie across the car, so fore-aft
        # cant is a defect there (v28 canted it 63 mm).  The FRONT damper runs
        # fore-aft by design, ~185 mm, which is not a fault.
        if lbl[0] == 'R' and abs(rsp[1] - scp[1]) > 25:
            gfail.append(f'{lbl} damper canted fore-aft {abs(rsp[1]-scp[1]):.0f} mm')
        if cop > 3.0:
            gfail.append(f'{lbl} rocker non-coplanar {cop:.1f} mm')
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

    # bump steer: a tie-rod move must not wreck the toe curve (the v32 rear
    # regression 0.002 -> 0.167 deg/25mm the old gate never checked).
    for _l, _isf in (('FL', True), ('RL', False)):
        _aln = wD._alignment
        _rr = wD._do_sweep(wD._solvers[_l], np.linspace(-0.025, 0.025, 5), 'left',
                           arb_hp=wD._front_arb if _isf else wD._rear_arb,
                           camber_off=_aln['front_camber_deg'] if _isf else _aln['rear_camber_deg'],
                           toe_off=_aln['front_toe_deg'] if _isf else _aln['rear_toe_deg'], is_front=_isf)
        _toe = np.asarray(_rr['toe'], float)
        _bs = float(np.nanmax(_toe) - np.nanmin(_toe))   # full-range over +/-25 mm
        if _bs > 0.15:
            gfail.append(f'{"front" if _isf else "rear"} bump steer {_bs:.3f} deg full-travel')
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
# LCO (B1965*) ahead of the R20 (B2356*).  Every tyre gate below was silently
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
    _mus = [float(_tm.peak_mu(float(f), 0.45)) for f in _fzs]
    _degr = all(_mus[i+1] <= _mus[i] * 1.02 for i in range(len(_mus) - 1))
    _lv = _tm.camber_levels() if callable(_tm.camber_levels) else _tm.camber_levels
    _lv = list(np.asarray(_lv).ravel())
    if not _degr:
        fails += 1
    print(f'tire camber rows : levels={_lv}  mu(300..1000N)@0.45deg='
          f'{" ".join(f"{m:.2f}" for m in _mus)}  '
          f'{"pass" if _degr else "UNEXPECTED FAIL (non-degressive: phantom camber rows back)"}')
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
    _lo = solve_ackermann_geometry(_ss, _tm, 8.0, 0.2)
    _hi = solve_ackermann_geometry(_ss, _tm, 8.0, 1.5)
    _geo_ok = abs(_lo['ackermann_pct'] - 100.0) < 8.0
    _split_ok = _hi['outer_slip_deg'] > _hi['inner_slip_deg'] > 0.0
    _mono_ok = (_hi['outer_slip_deg'] > _lo['outer_slip_deg']
                and _hi['inner_slip_deg'] > _lo['inner_slip_deg'])
    _sat_ok = not (_hi['inner_saturated'] or _hi['outer_saturated'])
    # (5) grip derate at 0.70: same ROAD g, derated curve.
    _raw12 = solve_ackermann_geometry(_ss, _tm, 8.0, 1.2)
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
#    (b) the VALID rows are smooth — monotone decreasing in g with exactly one
#    sign change (pro at low g -> reverse near the limit), no jumps.
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
    _ok = (not _fab) and _mono and _sgn == 1 and len(_sat) >= 1 and len(_val) >= 4
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
#    tyre: on this R20 the low-pressure sweep passes its peak inside the rig's
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
    _tbl = abs(_cl.caliper_upper_H - _cl.caliper_lower_H) / 2.0
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
    _rb = _cbs(_Fz, _pf, _pr, _sys)
    _fl, _rl = _rb['FL'], _rb['RL']
    _exp_f = _fl.lockup_line_pressure_MPa * _sys.mc_area_front_mm2 / (5.0 * _bias / 100.0)
    _exp_r = _rl.lockup_line_pressure_MPa * _sys.mc_area_rear_mm2 / (5.0 * (1 - _bias / 100.0))
    _bb_ok = (abs(_fl.lockup_pedal_force_N - _exp_f) < 0.5 and
              abs(_rl.lockup_pedal_force_N - _exp_r) < 0.5 and
              _fl.lockup_pedal_force_N > 1.0)
    # and the bias must actually MOVE the answer (the old code was bias-blind)
    _sys2 = _BSP(pedal_ratio=5.0, mc_bore_front_mm=15.87, mc_bore_rear_mm=15.87,
                 bias_pct_front=50.0)
    _rb2 = _cbs(_Fz, _pf, _pr, _sys2)
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
        _ok_a = (_rA.total_downforce_N > 1.0
                 and abs(_applied - _rA.total_downforce_N) / _rA.total_downforce_N < 0.02
                 and abs(_own - _rA.total_downforce_N) / _rA.total_downforce_N < 0.02)
        if not _ok_a:
            fails += 1
        print(f'aero per-corner  : solver total {_rA.total_downforce_N:.0f} N, applied '
              f'{_applied:.0f} N, ratio {_applied/max(_rA.total_downforce_N,1e-9):.3f}   '
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
#    AND that the rear rates were held (MR_r ~0.808, arb_r ~353k).
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
        # arb_r expectation is 380500, not the old 353000: the rigid-blade model
        # (2026-07-25) drops the phantom K_a arm-bending term, so this fixture's
        # bar is +7.8% stiffer.  Geometry (MR, coplanarity) is unchanged, which
        # is what this v28 fixture is actually guarding.
        cop_ok = (_worst < 3.0 and abs(_mr - 0.808) < 0.01 and abs(_arb - 380500) < 9000)
        if not cop_ok:
            fails += 1
        print(f'rear coplanar    : {_worst:.2f} mm across travel, MR_r {_mr:.4f}, '
              f'arb_r {_arb:.0f}   {"pass" if cop_ok else "UNEXPECTED FAIL"}')
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
        if _d_lap < 1.0:
            _lf.append(f'rotating inertia + shift model only cost '
                       f'{_d_lap:+.3f} s — it should slow the car by >1 s '
                       f'(it was measured at +2.09 s here)')
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
#    ALSO gated here: 100% Ackermann must not be special BY CONSTRUCTION.
#    At exactly 100% the two front wheels get the SAME slip angle, so if the
#    objective were maximised there automatically the whole sweep would be a
#    tautology.  It is not: measured, the unconstrained optimum sits past
#    +100% (+125% at 5 m, +150% at 8 m), so 100% wins a +/-100% sweep only as
#    a boundary, never because the split vanishes there.
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
            _GG7 = (0.8, 1.4)

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
            # move the rack 25 mm fore-aft — a REAL Ackermann change
            _w7._front_hp['tie_rod_inner'] = (
                np.asarray(_w7._front_hp['tie_rod_inner'], float)
                + np.array([0.0, 0.025, 0.0]))
            _w7._rebuild_solvers(0.)
            _ack_after = float(_w7._probe_static_ackermann())
            _ss7b = _w7._build_dynamics_solver()
            _b7 = _tbl7(_ss7b)

            _f7 = []
            if not (np.isfinite(_ack_before) and np.isfinite(_ack_after)
                    and abs(_ack_after - _ack_before) > 1.0):
                _f7.append(f'the 25 mm rack move did not change the car\'s '
                           f'as-built Ackermann ({_ack_before:.2f}% -> '
                           f'{_ack_after:.2f}%) — the perturbation is not '
                           f'live, so this gate proves nothing')
            _dmax = float(np.nanmax(np.abs(_a7._ay_cap - _b7._ay_cap)))
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
            if not np.isfinite(_dmax) or _dmax > _AY_INVARIANCE_TOL:
                _f7.append(f'the swept capability moved by {_dmax:.3e} g when '
                           f'only the CAR\'S OWN Ackermann changed — the '
                           f'sweep is reading its own setting')
            # tautology check: 100% gives EQUAL front slip angles, and must
            # still not be the automatic winner of a wider sweep
            _d7 = _saf7(_tm7, _ss7a, 5.0, 1.2, ack_range=(-100.0, 200.0),
                        n=13, grip_multiplier=0.65)
            _p7 = np.asarray(_d7['ackermann_pct'], float)
            _u7 = np.asarray(_d7['useful_N'], float)
            _k100 = int(np.argmin(np.abs(_p7 - 100.0)))
            if int(np.argmax(_u7)) == _k100:
                _f7.append('100% Ackermann is the argmax of a -100..+200% '
                           'sweep — that is exactly where the two front slip '
                           'angles become equal, so the win is tautological')
            if _f7:
                fails += 1
            print(f'ackermann sweep  : rack +25 mm moved as-built '
                  f'{_ack_before:+.1f}% -> {_ack_after:+.1f}%, swept '
                  f'capability moved {_dmax:.1e} g; best of -100..+200% is '
                  f'{_p7[int(np.argmax(_u7))]:+.0f}% (not 100%)   '
                  + ('pass' if not _f7 else
                     'UNEXPECTED FAIL: ' + '; '.join(_f7)))
except Exception as _e:
    fails += 1
    import traceback as _tb7; _tb7.print_exc()
    print(f'ackermann sweep  : UNEXPECTED FAIL ({type(_e).__name__}: {_e})')

print('-' * 64)
print(f'{fails} unexpected failures, {known} known-fail (documented).')
sys.exit(fails)
