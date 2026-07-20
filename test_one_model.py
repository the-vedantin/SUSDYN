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
# perturb the ARM END (arm length feeds the rate formula directly; a tab
# move can make the bellcrank solve fail -> refresh guard skips -> false DEAD)
_ae_save = np.asarray(win._front_arb['arb_arm_end'], float).copy()
win._front_arb['arb_arm_end'] = _ae_save + np.array([0.0, 0.0, 0.010])
win._refresh_arb_geometry_into_panel()
_r1 = float(win._dynamics_panel.get_params()['arb_rate_front_Npm'])
win._front_arb['arb_arm_end'] = _ae_save
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
#    weak-axis bending replaces the tube-section arm stiffness when
#    blade w and t are both set.  Guard: setting a thin blade SOFTENS the
#    computed wheel rate vs the legacy tube-arm model, and w or t = 0
#    restores the legacy value exactly.
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

# ── TIRE CAMBER-ROW INTEGRITY: TTC tests sweep discrete inclinations (0/2/4);
#    stray transition samples used to create phantom integer camber rows filled
#    with zeros, so peak_mu at interpolated cambers (e.g. 0.45 deg — exactly
#    where the dynamics solver evaluates) came out NON-MONOTONE vs load
#    (mu 1.57@400N < 2.39@939N) and corrupted utilization + understeer gradient.
#    Guard: only well-populated camber rows survive, and mu(Fz) at a mid-row
#    camber is degressive (monotone non-increasing within 2%).
print('-' * 64)
try:
    win._try_autoload_tire()
except Exception:
    pass
_tm = getattr(win, '_tire_model', None)
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

print('-' * 64)
print(f'{fails} unexpected failures, {known} known-fail (documented).')
sys.exit(fails)
