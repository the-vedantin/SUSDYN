# -*- coding: utf-8 -*-
"""design_city.py — Design City: FRONT and REAR packaging solutions that hold
EVERY parameter of the current design within 0.1 %, rendered, and grouped by
similarity.

What it does (the user's spec, 2026-09-09):
  1. From the auto-selected highest configs/2027_v<N> config, enumerate
     CHASSIS-SIDE packaging alternatives per axle: actuation-slice rotations
     about the pushrod line / mirror about the pushrod plane / in-plane
     translations, rocker spring-lever scaling, ARB re-hang branches and
     drop-top pose swings, and inboard arm pickups slid ALONG their own pivot
     axis.  Every transform is a vahan.packaging primitive; the motion ratio
     and ARB rate are re-tuned back with the packaging bisection knobs.
     Wheel-side hardpoints (outer ball joints, tie rod, wheel centre, pushrod
     foot) are never touched — checked byte-identical in the saved file.
  2. THE GATE: a candidate is kept ONLY if every one of the ~106 parameters
     vahan.packaging.parameter_vector measures (kinematics, curves, motion
     ratios, damper lengths, rates, ride frequencies, ARB rates, roll
     gradient, LLTD, Ackermann, travel range, ...) is within 0.1 % of the
     baseline (absolute floor = 0.1 % of the parameter's physical scale for
     near-zero baselines, documented per parameter in metrics.json), AND the
     axle geometry laws hold (coplanar / drop link in plane / 90-degree triad /
     rear damper cant), AND the rocker hardware separations of the net's
     design-actuation gate are non-negative, AND _clash_sweep has 0 negatives
     at rack centre and both full locks, AND the 39-state full audit
     (droop/static/bump x 13 rack positions, cross-corner, rack housing,
     torsion bars, driveshafts) closes everywhere with 0 negatives, AND the
     saved .vahan re-loads and passes the same 0.1 % gate again.
  3. Every kept solution is a loadable .vahan (saved through the main
     window's own save path) under designs_city/<run>/<axle>/<id>/ with
     metrics.json (every parameter: baseline, value, % deviation, floor).
  4. --render: native-GL captures (axle view + iso + top) of every kept
     solution through the real 3D view (QT_QPA_PLATFORM must be UNSET).
  5. Grouping: scipy complete-linkage clustering on the chassis-side
     hardpoint deltas; two solutions are in one group only if NO chassis-side
     point differs by more than --cluster-mm between them.

Run layout (read by gui/city_page.py):
    designs_city/<run>/run.json                 config, sha, parameter list, per-axle tallies
    designs_city/<run>/<axle>/groups.json       clusters
    designs_city/<run>/<axle>/trials.jsonl      one line per trial (kept or why not)
    designs_city/<run>/<axle>/<id>/config.vahan | metrics.json | recipe.json | gui_*.png

CLI:
    py design_city.py                          search + cluster + render, both axles
    py design_city.py --trials 300 --workers 4 --axles front rear --cluster-mm 20
    py design_city.py --render designs_city/<run>       (re)capture images
    py design_city.py --cluster designs_city/<run> [--cluster-mm 20]

ONE MODEL: this file never computes physics.  It calls vahan.packaging
(transforms, retunes, parameter_vector, audits) and the MainWindow load /
save / render paths.
"""
from __future__ import annotations

import os
import sys
import re
import json
import glob
import time
import hashlib
import argparse
import subprocess
import datetime as _dt

# The search runs offscreen; the --render subprocess needs NATIVE GL (vispy
# canvas.render() through the real 3D view), so it must NOT inherit offscreen.
if '--render' not in sys.argv:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VAHAN_MCP', '0')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
REL_TOL = 1e-3                    # the 0.1 % gate (vahan.packaging.REL_TOL)
CLUSTER_MM_DEFAULT = 20.0         # complete-linkage cut on max point delta
DUPLICATE_MM = 0.5                # a kept solution must move some point at least this far
NEAR_MM = 3.0                     # audit: no pair may come under this (the builders' acceptance)...
WORSEN_MM = 0.25                  # ...unless already there at baseline and not worse by more than this
WHEEL_SIDE_KEYS = ('uca_outer', 'lca_outer', 'tie_rod_outer', 'tie_rod_inner',
                   'wheel_center', 'pushrod_outer')
PICKUP_KEYS = ('uca_front', 'uca_rear', 'lca_front', 'lca_rear')
AXLES = ('front', 'rear')
FAMILIES = ('spring_swing', 'arb_swing', 'arb_branch', 'drop_link', 'pickup_slide',
            'rotate_pushrod', 'mirror', 'translate', 'lever')
LAW_COPLANAR_MM = 3.0
LAW_ROCKER_AXIS_DEG = 1e-4
LAW_INPLANE_MM = 3.0
LAW_TRIAD_DEG = 1.0
LAW_REAR_DAMPER_CANT_MM = 25.0
MR_RETUNE_TOL = 1e-5
ARB_RETUNE_TOL = 2e-4
GROUP_METRIC = ('max over chassis-side points of the Euclidean distance (mm) between '
                'the two solutions\' positions of that point; complete linkage, so every '
                'pair inside a group is within the threshold on every point')


def current_config() -> str:
    """Highest configs/2027_v<N>_*.vahan — the same rule test_one_model.py uses."""
    cfgs = glob.glob(os.path.join(REPO, 'configs', '2027_v*.vahan'))
    if not cfgs:
        raise SystemExit('no configs/2027_v*.vahan found')
    return max(cfgs, key=lambda p: int((re.search(r'2027_v(\d+)', p) or [0, -1]).__getitem__(1))
               if re.search(r'2027_v(\d+)', p) else -1)


def file_sha(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
#  RECIPES — what to try.  A recipe is a plain dict; apply_recipe() turns it
#  into a bundle with vahan.packaging primitives.
# ═════════════════════════════════════════════════════════════════════════════
def identity_recipe() -> dict:
    return {'spring_swing_deg': 0.0, 'arb_swing_deg': 0.0, 'arb_branch': 0, 'drop_link_dlen_mm': 0.0,
            'pickup_slides': {}, 'pickup_spacing': {}, 'mirror': False, 'rotate_pushrod_deg': 0.0,
            'translate_mm': [0.0, 0.0, 0.0], 'lever_s': 1.0}


# Families.  EXACT ones hold the motion-ratio curve by construction:
#   spring_swing  rotate the spring pair about the rocker axis (rocker shape +
#                 coilover chassis mount move on a circle, same spring law)
#   arb_swing     swing the drop top about the pivot in the chain plane, then
#                 re-hang the bar and re-match the rate (blade / drop radius)
#   arb_branch    one of the up-to-8 discrete re-hang poses of the bar
#   drop_link     lengthen / shorten the drop link along its own line
#   pickup_slide  slide an inboard arm pickup ALONG its own pivot axis
# INEXACT ones change the rocker plane or the pushrod direction, so the MR
# CURVE moves even when static MR is re-tuned; they are sampled at a small
# share so the run reports honestly what blocks them:
#   rotate_pushrod / mirror / translate / lever
EXACT_FAMILIES = ('spring_swing', 'arb_swing', 'arb_branch', 'drop_link', 'pickup_slide')
INEXACT_FAMILIES = ('rotate_pushrod', 'mirror', 'translate', 'lever')
INEXACT_SHARE = 0.15


def structured_recipes() -> list:
    """Deterministic grid over the exact families, one knob at a time,
    smallest moves first."""
    out = []
    for a in (15, 30, 45, 60, 90, 120, 150, 180):
        for sgn in ((+1, -1) if a != 180 else (+1,)):
            r = identity_recipe(); r['spring_swing_deg'] = float(sgn * a); out.append(r)
    for a in (10, 20, 30, 45, 60, 90):
        for sgn in (+1, -1):
            r = identity_recipe(); r['arb_swing_deg'] = float(sgn * a); out.append(r)
    for k in range(1, 8):
        r = identity_recipe(); r['arb_branch'] = k; out.append(r)
    for d in (-40.0, -20.0, 20.0, 40.0):
        r = identity_recipe(); r['drop_link_dlen_mm'] = d; out.append(r)
    for arm in ('uca', 'lca'):
        for d in (-20.0, -10.0, 10.0, 20.0):
            r = identity_recipe(); r['pickup_spacing'] = {arm: d}; out.append(r)
    for key in PICKUP_KEYS:
        for d in (-5.0, 5.0):
            r = identity_recipe(); r['pickup_slides'] = {key: d}; out.append(r)
    r = identity_recipe(); r['mirror'] = True; out.append(r)
    for a in (-3.0, -1.0, 1.0, 3.0):
        r = identity_recipe(); r['rotate_pushrod_deg'] = a; out.append(r)
    return out


def random_recipe(rng: np.random.Generator) -> dict:
    """A random composition: 1-3 exact families, plus (at INEXACT_SHARE) one
    inexact family."""
    r = identity_recipe()
    n_fam = int(rng.integers(1, 4))
    fams = list(rng.choice(EXACT_FAMILIES, size=n_fam, replace=False))
    if rng.uniform() < INEXACT_SHARE:
        fams.append(str(rng.choice(INEXACT_FAMILIES)))
    if 'spring_swing' in fams:
        r['spring_swing_deg'] = float(rng.uniform(-180.0, 180.0))
    if 'arb_swing' in fams:
        r['arb_swing_deg'] = float(rng.uniform(-90.0, 90.0))
    if 'arb_branch' in fams:
        r['arb_branch'] = int(rng.integers(1, 8))
    if 'drop_link' in fams:
        r['drop_link_dlen_mm'] = float(rng.uniform(-50.0, 50.0))
    if 'pickup_slide' in fams:
        if rng.uniform() < 0.7:
            for arm in rng.choice(('uca', 'lca'), size=int(rng.integers(1, 3)), replace=False):
                r['pickup_spacing'][str(arm)] = float(rng.uniform(-30.0, 30.0))
        else:
            r['pickup_slides'][str(rng.choice(PICKUP_KEYS))] = float(rng.uniform(-8.0, 8.0))
    if 'rotate_pushrod' in fams:
        r['rotate_pushrod_deg'] = float(rng.uniform(-5.0, 5.0))
    if 'mirror' in fams:
        r['mirror'] = True
    if 'translate' in fams:
        # [along pushrod, in-plane perpendicular, plane normal] mm; the normal
        # component is what breaks coplanarity (law < 3 mm) — keep it small
        r['translate_mm'] = [float(rng.uniform(-10.0, 10.0)), float(rng.uniform(-10.0, 10.0)),
                             float(rng.uniform(-1.0, 1.0))]
    if 'lever' in fams:
        r['lever_s'] = float(rng.uniform(0.97, 1.03))
    return r


def recipe_families(r: dict) -> list:
    fams = []
    if abs(r.get('spring_swing_deg', 0.0)) > 1e-9:
        fams.append('spring_swing')
    if abs(r.get('arb_swing_deg', 0.0)) > 1e-9:
        fams.append('arb_swing')
    if int(r.get('arb_branch', 0)) != 0:
        fams.append('arb_branch')
    if abs(r.get('drop_link_dlen_mm', 0.0)) > 1e-9:
        fams.append('drop_link')
    if r.get('pickup_slides') or r.get('pickup_spacing'):
        fams.append('pickup_slide')
    if abs(r.get('rotate_pushrod_deg', 0.0)) > 1e-9:
        fams.append('rotate_pushrod')
    if r.get('mirror'):
        fams.append('mirror')
    if np.linalg.norm(r.get('translate_mm', [0, 0, 0])) > 1e-9:
        fams.append('translate')
    if abs(r.get('lever_s', 1.0) - 1.0) > 1e-9:
        fams.append('lever')
    return fams


def make_recipes(n_trials: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = structured_recipes()[:max(0, n_trials)]
    while len(out) < n_trials:
        out.append(random_recipe(rng))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  WORKER — one MainWindow per process, baseline captured once
# ═════════════════════════════════════════════════════════════════════════════
_W = {}


def _worker_init(config: str, out_dir: str):
    """Load the pinned config into a fresh offscreen MainWindow and capture
    the baseline parameter vector + bundles (both axles)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('VAHAN_MCP', '0')
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)
    from PyQt6.QtWidgets import QApplication
    _W['app'] = QApplication.instance() or QApplication([])
    from gui.main_window import MainWindow
    import vahan.packaging as PK
    w = MainWindow()
    w._load_project_from_path(config)
    w._rebuild_solvers(0.0)
    _W['PK'] = PK
    _W['win'] = w
    _W['config'] = config
    _W['out_dir'] = out_dir
    with open(config, 'r', encoding='utf-8') as f:
        _W['src_json'] = json.load(f)
    _W['base_pv'] = PK.parameter_vector(w)
    _W['base'] = {ax: PK.get_bundle(w, ax) for ax in AXLES}
    _W['target_mr'] = {ax: _W['base_pv'][f'{ax}.motion_ratio']['value'] for ax in AXLES}
    _W['target_arb'] = {ax: _W['base_pv'][f'{ax}.arb_rate_Npm']['value'] for ax in AXLES}
    # baseline near-miss table of the 39-state audit (pairs within 10 mm):
    # a candidate may not bring any pair under NEAR_MM unless the baseline
    # already had it there and it is not worse by more than WORSEN_MM
    _W['base_audit'] = {frozenset((r['a'], r['b'])): float(r['gap_mm'])
                        for r in PK.full_state_audit(w)['worst_per_pair']}


def _restore():
    PK, w = _W['PK'], _W['win']
    for ax in AXLES:
        PK.set_bundle(w, ax, _W['base'][ax], rebuild=False)
    w._rebuild_solvers(0.0)


def _translate_xyz_mm(bundle, t3) -> np.ndarray:
    """[along pushrod, in-plane perpendicular, plane normal] mm -> XYZ mm."""
    hp = bundle['hp']
    u = np.asarray(hp['pushrod_inner'], float) - np.asarray(hp['pushrod_outer'], float)
    u /= np.linalg.norm(u)
    _, n = _W['PK'].rocker_plane(bundle)
    n = n - u * float(np.dot(n, u)); n /= np.linalg.norm(n)
    wv = np.cross(n, u)
    return u * t3[0] + wv * t3[1] + n * t3[2]


def apply_recipe(axle: str, recipe: dict) -> tuple:
    """base bundle -> candidate bundle via packaging primitives, then MR and
    ARB-rate re-tune.  Leaves the window ON the candidate.  Returns
    (bundle, info)."""
    PK, w = _W['PK'], _W['win']
    base = _W['base'][axle]
    b = PK._copy_bundle(base)
    info = {'lever_k': 1.0, 'arb_knob': 'none', 'arb_rate': None, 'mr': None}
    if recipe.get('mirror'):
        b = PK.mirror_about_pushrod_plane(b)
    if abs(recipe.get('rotate_pushrod_deg', 0.0)) > 1e-9:
        b = PK.rotate_about_pushrod_line(b, float(recipe['rotate_pushrod_deg']))
    t3 = np.asarray(recipe.get('translate_mm', [0, 0, 0]), float)
    if np.linalg.norm(t3) > 1e-9:
        dxyz = _translate_xyz_mm(b, t3)
        recipe['translate_xyz_mm'] = [float(x) for x in dxyz]
        b = PK.translate_actuation(b, dxyz / 1000.0)
    if abs(recipe.get('lever_s', 1.0) - 1.0) > 1e-9:
        b = PK.scale_spring_lever(b, float(recipe['lever_s']), hold_damper_length=True)
    if abs(recipe.get('spring_swing_deg', 0.0)) > 1e-9:
        b = PK.swing_spring_about_rocker_axis(b, float(recipe['spring_swing_deg']))
    for k, d in (recipe.get('pickup_slides') or {}).items():
        b = PK.slide_pickup_along_axis(b, k, float(d))
    # pickup SPACING: both pickups of one arm slid toward (+) / away from (-)
    # each other by d along the pivot axis.  The tool's roll-centre uses the
    # XZ-projected MIDPOINT of the two pickups (vahan.kinematics
    # .roll_center_height), so a single-pickup slide on a non-Y-parallel
    # axis shifts the REPORTED RC although the true swing axis is unchanged;
    # the symmetric spacing move holds the midpoint and the axis exactly.
    for arm, d in (recipe.get('pickup_spacing') or {}).items():
        for k in (arm + '_front', arm + '_rear'):
            b = PK.slide_pickup_along_axis(b, k, float(d))
    if abs(recipe.get('arb_swing_deg', 0.0)) > 1e-9:
        b = PK.rotate_arb_drop_top(b, float(recipe['arb_swing_deg']))
    # ARB re-hang ONLY when something ARB-related moved (the drop top rides
    # the rocker slice; a slice move or an ARB knob).  A spring swing or a
    # pickup slide leaves the bar physically untouched, and the v99 front bar
    # sits on a rate cliff at bump (MR 1.3 / 2.6 / 15.3 at -25 / 0 / +25 mm),
    # so an unneeded rigid re-hang alone would move the +25 mm bar MR by
    # percent-level.  The torsion bar is chassis-fixed along X, so the
    # baseline triangle is re-hung into the new chain plane; branch k picks
    # which of the up-to-8 discrete poses (bar side / chirality / root).
    slice_moved = (recipe.get('mirror') or abs(recipe.get('rotate_pushrod_deg', 0.0)) > 1e-9
                   or np.linalg.norm(t3) > 1e-9)
    arb_knob = (abs(recipe.get('arb_swing_deg', 0.0)) > 1e-9 or int(recipe.get('arb_branch', 0)) != 0
                or abs(recipe.get('drop_link_dlen_mm', 0.0)) > 1e-9)
    if b['arb'] and (slice_moved or arb_knob):
        variants = PK.refit_arb_variants(b, base)
        k = int(recipe.get('arb_branch', 0))
        if k >= len(variants):
            raise ValueError(f'ARB branch {k} does not exist ({len(variants)} branches)')
        b = variants[k]
        if abs(recipe.get('drop_link_dlen_mm', 0.0)) > 1e-9:
            L0 = float(np.linalg.norm(b['arb']['arb_arm_end'] - b['arb']['arb_drop_top']))
            L1 = L0 + float(recipe['drop_link_dlen_mm']) / 1000.0
            if L1 < 0.04:
                raise ValueError(f'drop link {L1 * 1000:.0f} mm too short')
            b = PK.set_drop_link_length(b, L1)
    # motion ratio back to baseline via the pushrod-lever bisection
    mr = PK.solver_mr(PK._corner_solver(w, axle, b))
    if abs(mr - _W['target_mr'][axle]) / _W['target_mr'][axle] > MR_RETUNE_TOL:
        b, info['lever_k'], mr = PK.retune_mr(w, b, axle, _W['target_mr'][axle], iters=26)
    info['mr'] = float(mr)
    PK.set_bundle(w, axle, b)
    if b['arb']:
        b, rate, knob = PK.rate_match_arb(w, b, axle, _W['target_arb'][axle], base, tol=ARB_RETUNE_TOL)
        info['arb_rate'] = float(rate); info['arb_knob'] = knob
    return b, info


def _chassis_deltas_mm(axle: str, bundle: dict) -> dict:
    PK = _W['PK']
    base = _W['base'][axle]
    out = {}
    for k in PK.ACTUATION_HP_KEYS + PICKUP_KEYS:
        if k in bundle['hp'] and bundle['hp'][k] is not None and k in base['hp']:
            out['hp.' + k] = [float(x) for x in (np.asarray(bundle['hp'][k]) - np.asarray(base['hp'][k])) * 1000.0]
    for k in PK.ARB_KEYS:
        if k in bundle['arb'] and k in base['arb']:
            out['arb.' + k] = [float(x) for x in (np.asarray(bundle['arb'][k]) - np.asarray(base['arb'][k])) * 1000.0]
    return out


def _wheel_side_identical(axle: str, saved_json: dict) -> bool:
    src = _W['src_json']
    return all(src[axle + '_hp'][k] == saved_json[axle + '_hp'][k] for k in WHEEL_SIDE_KEYS
               if k in src[axle + '_hp'])


def evaluate(job: tuple) -> dict:
    """(axle, trial_id, recipe) -> trial record.  Kept solutions are written to
    <out_dir>/<axle>/<trial_id>/ before returning."""
    axle, tid, recipe = job
    PK, w = _W['PK'], _W['win']
    rec = {'id': tid, 'axle': axle, 'recipe': recipe, 'families': recipe_families(recipe),
           'kept': False, 'stage': None, 'reason': None, 'max_deviation_pct': None,
           'worst_parameter': None, 'failed_parameters': [], 'is_baseline': tid == 'baseline'}
    t0 = time.time()
    try:
        # 1. transform + retune
        try:
            b, info = apply_recipe(axle, recipe)
        except Exception as e:
            rec.update(stage='transform', reason=str(e)[:200]); return _finish(rec, t0)
        rec['retune'] = info
        PK.set_bundle(w, axle, b)
        # 2. laws (net thresholds)
        g = PK._axle_geometry_laws(w, axle)
        rec['laws'] = {k: (float(v) if isinstance(v, (float, int, np.floating)) else v) for k, v in g.items()}
        bad = []
        if g['coplanar_mm'] > LAW_COPLANAR_MM:
            bad.append(f"coplanar {g['coplanar_mm']:.2f} mm")
        if (not np.isfinite(g['rocker_axis_normal_error_deg'])
                or g['rocker_axis_normal_error_deg'] > LAW_ROCKER_AXIS_DEG):
            bad.append(f"rocker axis {g['rocker_axis_normal_error_deg']:.6f} deg from plate normal")
        if not PK.arb_drop_link_plate_compliant(
                {'drop_top_signed_mm': g['arb_drop_top_plate_signed_mm'],
                 'arm_end_signed_mm': g['arb_arm_end_plate_signed_mm']},
                LAW_INPLANE_MM, g.get('arb_is_bottom', False)):
            for k in ('arb_drop_top_inplane_mm', 'arb_arm_end_inplane_mm'):
                bad.append(f'{k} {g[k]:.2f} mm')
        for k in ('triad_bar_blade_deg', 'triad_blade_drop_deg', 'triad_bar_drop_deg'):
            if not np.isfinite(g[k]) or abs(g[k] - 90.0) > LAW_TRIAD_DEG:
                bad.append(f'{k} {g[k]:.2f} deg')
        if axle == 'rear' and g['damper_cant_fore_aft_mm'] > LAW_REAR_DAMPER_CANT_MM:
            bad.append(f"rear damper cant {g['damper_cant_fore_aft_mm']:.1f} mm")
        if bad:
            rec.update(stage='laws', reason='; '.join(bad)); return _finish(rec, t0)
        # 3. rocker hardware separations (the net's rod-end / bearing formulas)
        hw = PK.rocker_hw_gaps(b)
        rec['rocker_hw_gaps_mm'] = {k: float(v) for k, v in hw.items()}
        neg = {k: v for k, v in hw.items() if v < 0.0}
        if neg:
            rec.update(stage='rocker_hw', reason=', '.join(f'{k} {v:.1f} mm' for k, v in neg.items()))
            return _finish(rec, t0)
        # 4. THE 0.1 % PARAMETER GATE
        pv = PK.parameter_vector(w)
        rows = PK.compare_parameters(_W['base_pv'], pv, rel=REL_TOL)
        fails = [r for r in rows if not r['ok']]
        finite = [r for r in rows if np.isfinite(r['deviation_pct'])]
        worst = max(finite, key=lambda r: r['deviation_pct']) if finite else None
        rec['max_deviation_pct'] = float(worst['deviation_pct']) if worst else float('nan')
        rec['worst_parameter'] = worst['name'] if worst else None
        rec['failed_parameters'] = [r['name'] for r in fails]
        if fails:
            rec.update(stage='parameters',
                       reason=f"{len(fails)} parameter(s) beyond 0.1 %: " +
                              ', '.join(f"{r['name']} {r['deviation_pct']:.3f} %" for r in fails[:6]))
            return _finish(rec, t0)
        # 5. clash sweep at rack centre and both full locks
        cl = PK.clash_negatives_at_locks(w)
        rec['clash_locks'] = {'negatives': int(cl['negatives']), 'worst_mm': cl['worst_mm'],
                              'pairs': cl['pairs'][:10]}
        if cl['negatives']:
            rec.update(stage='clash_locks', reason=f"{cl['negatives']} negative gap(s), worst {cl['worst_mm']:.2f} mm")
            return _finish(rec, t0)
        # 6. the 39-state audit (closure at full travel x rack, cross-corner)
        au = PK.full_state_audit(w)
        rec['audit'] = {'states': au['states'], 'negatives': len(au['negatives']),
                        'warnings': len(au['warnings']), 'closure_errors': au['closure_errors'],
                        'min_gap_mm': au['min_gap_mm'], 'worst_pairs': au['worst_per_pair'][:10]}
        if au['closure_errors'] or au['negatives']:
            rec.update(stage='audit_39', reason=(f"{len(au['negatives'])} negative pair(s), "
                                                 f"{len(au['closure_errors'])} closure error state(s)"))
            return _finish(rec, t0)
        near = []
        for r in au['worst_per_pair']:
            if r['gap_mm'] < NEAR_MM:
                b0 = _W['base_audit'].get(frozenset((r['a'], r['b'])), 10.0)
                if r['gap_mm'] < b0 - WORSEN_MM:
                    near.append(f"{r['a']} / {r['b']} {r['gap_mm']:.2f} mm (baseline {b0:.2f})")
        rec['audit']['new_near_misses'] = near
        if near:
            rec.update(stage='audit_39', reason=f'near-miss under {NEAR_MM:.0f} mm: ' + '; '.join(near[:4]))
            return _finish(rec, t0)
        # 7. save through the main window, reload, gate again, wheel side identical
        sdir = os.path.join(_W['out_dir'], axle, tid)
        os.makedirs(sdir, exist_ok=True)
        cfg = os.path.join(sdir, 'config.vahan')
        w._save_project_to_path(cfg)
        w._load_project_from_path(cfg)
        w._rebuild_solvers(0.0)
        pv2 = PK.parameter_vector(w)
        rows2 = PK.compare_parameters(_W['base_pv'], pv2, rel=REL_TOL)
        fails2 = [r for r in rows2 if not r['ok']]
        fin2 = [r['deviation_pct'] for r in rows2 if np.isfinite(r['deviation_pct'])]
        rec['reload'] = {'ok': not fails2, 'max_deviation_pct': float(max(fin2)) if fin2 else float('nan'),
                         'failed_parameters': [r['name'] for r in fails2]}
        with open(cfg, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        rec['wheel_side_byte_identical'] = bool(_wheel_side_identical(axle, saved))
        other = 'rear' if axle == 'front' else 'front'
        rec['other_axle_byte_identical'] = bool(
            all(_W['src_json'][other + '_hp'][k] == saved[other + '_hp'][k] for k in _W['src_json'][other + '_hp'])
            and _W['src_json'][other + '_arb'] == saved[other + '_arb'])
        if fails2 or not rec['wheel_side_byte_identical'] or not rec['other_axle_byte_identical']:
            rec.update(stage='reload', reason=('reload gate: ' + ', '.join(rec['reload']['failed_parameters'][:6]))
                       if fails2 else 'wheel side / other axle not byte-identical after save')
            return _finish(rec, t0)
        # kept — unless it is the baseline geometry again (an ARB branch can be
        # pose-degenerate; a knob can round-trip): a solution must MOVE
        deltas = _chassis_deltas_mm(axle, b)
        rec['chassis_deltas_mm'] = deltas
        rec['max_point_move_mm'] = float(max(np.linalg.norm(v) for v in deltas.values())) if deltas else 0.0
        if tid != 'baseline' and rec['max_point_move_mm'] < DUPLICATE_MM:
            import shutil
            shutil.rmtree(sdir, ignore_errors=True)
            rec.update(stage='duplicate', reason=f"identical to the baseline (max point move {rec['max_point_move_mm']:.2f} mm)")
            return _finish(rec, t0)
        rec.update(kept=True, stage='kept', reason='all gates pass')
        rec['config'] = 'config.vahan'
        rec['config_sha256'] = file_sha(cfg)
        rec['ok'] = True
        meta = dict(rec); meta['rows'] = rows; meta['rel_tol'] = REL_TOL
        with open(os.path.join(sdir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1, default=float)
        with open(os.path.join(sdir, 'recipe.json'), 'w', encoding='utf-8') as f:
            json.dump(recipe, f, indent=1, default=float)
        return _finish(rec, t0)
    except Exception as e:
        rec.update(stage='error', reason=f'{type(e).__name__}: {str(e)[:200]}')
        return _finish(rec, t0)
    finally:
        try:
            _restore()
        except Exception:
            pass


def _finish(rec, t0):
    rec['elapsed_s'] = round(time.time() - t0, 2)
    return rec


# ═════════════════════════════════════════════════════════════════════════════
#  DRIVER — search
# ═════════════════════════════════════════════════════════════════════════════
def parameter_table(base_pv: dict) -> list:
    """Documented parameter list: name, unit, baseline, physical scale, the
    absolute tolerance actually applied and whether the floor is active."""
    out = []
    for name, p in base_pv.items():
        bv = float(p['value']); sc = float(p['scale'])
        out.append({'name': name, 'unit': p['unit'], 'axle': p['axle'], 'baseline': bv,
                    'scale': sc, 'tolerance_abs': max(REL_TOL * abs(bv), REL_TOL * sc),
                    'floor_abs': REL_TOL * sc, 'floor_active': REL_TOL * sc > REL_TOL * abs(bv)})
    return out


def _axle_summary(records: list) -> dict:
    trials = [r for r in records if not r.get('is_baseline')]
    kept = [r for r in trials if r['kept']]
    stage = {}
    for r in trials:
        stage[r['stage']] = stage.get(r['stage'], 0) + 1
    out = {'trials': len(trials), 'kept': len(kept), 'fail_stage': stage,
           'kept_ids': [r['id'] for r in kept],
           'families_kept': sorted({f for r in kept for f in r['families']})}
    # blocking parameter: over every trial that reached the parameter gate
    reached = [r for r in trials if r['stage'] in ('parameters', 'kept', 'clash_locks', 'audit_39', 'reload')
               and r.get('max_deviation_pct') is not None]
    pf = {}
    for r in trials:
        for n in r.get('failed_parameters', []):
            pf[n] = pf.get(n, 0) + 1
    if reached:
        best = min(reached, key=lambda r: r['max_deviation_pct'])
        out['smallest_max_deviation'] = {'trial': best['id'], 'max_deviation_pct': best['max_deviation_pct'],
                                         'worst_parameter': best['worst_parameter'], 'stage': best['stage'],
                                         'families': best['families']}
    if pf:
        top = max(pf.items(), key=lambda kv: kv[1])
        out['blocking_parameter'] = {'name': top[0], 'fail_count': top[1], 'top': sorted(pf.items(), key=lambda kv: -kv[1])[:8]}
    return out


def run_city(config: str = None, out_dir: str = None, axles=AXLES, trials: int = 300,
             workers: int = 4, seed: int = 0, cluster_mm: float = CLUSTER_MM_DEFAULT,
             render: bool = True, recipes: dict = None, progress=print) -> dict:
    """Search both axles, cluster, render.  Returns the run.json dict.
    workers<=1 runs in THIS process (the regression net uses that)."""
    config = os.path.abspath(config or current_config())
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.abspath(out_dir or os.path.join(REPO, 'designs_city', f'city_{stamp}'))
    os.makedirs(out_dir, exist_ok=True)
    run = {'config': config, 'config_name': os.path.basename(config), 'config_sha256': file_sha(config),
           'started': _dt.datetime.now().isoformat(timespec='seconds'), 'rel_tol': REL_TOL,
           'cluster_mm': cluster_mm, 'group_metric': GROUP_METRIC, 'trials_per_axle': trials,
           'seed': seed, 'axles': {}, 'gates': [
               'every parameter within 0.1 % of baseline (floor 0.1 % of physical scale)',
               f'coplanar <= {LAW_COPLANAR_MM} mm, ARB drop link in plane <= {LAW_INPLANE_MM} mm, triad 90 +/- {LAW_TRIAD_DEG} deg, rear damper cant <= {LAW_REAR_DAMPER_CANT_MM} mm',
               'rocker rod ends (r 8.0 mm) clear each other and the 19.05 mm pivot bearing',
               '_clash_sweep 0 negatives at rack centre and both full locks (rim guard +3 mm)',
               f'39-state audit (droop/static/bump x 13 rack, cross-corner, rack housing, torsion bars, driveshafts): closes, 0 negatives, '
               f'no pair under {NEAR_MM:.0f} mm unless already so at baseline and not worse by {WORSEN_MM} mm',
               'saved .vahan reloads and passes the same 0.1 % gate; wheel side + other axle byte-identical']}
    with open(os.path.join(out_dir, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=1, default=float)
    jobs = []
    for ax in axles:
        os.makedirs(os.path.join(out_dir, ax), exist_ok=True)
        jobs.append((ax, 'baseline', identity_recipe()))
        rl = (recipes or {}).get(ax) or make_recipes(trials, seed + (0 if ax == 'front' else 1000))
        for i, r in enumerate(rl):
            jobs.append((ax, f'{ax[0]}{i:04d}', r))
    t0 = time.time()
    records = {ax: [] for ax in axles}
    logs = {ax: open(os.path.join(out_dir, ax, 'trials.jsonl'), 'w', encoding='utf-8') for ax in axles}

    def take(rec):
        records[rec['axle']].append(rec)
        logs[rec['axle']].write(json.dumps(rec, default=float) + '\n'); logs[rec['axle']].flush()
        n = sum(len(v) for v in records.values())
        if progress:
            progress(f"[{n}/{len(jobs)} {time.time() - t0:6.0f}s] {rec['axle']:5s} {rec['id']:9s} "
                     f"{rec['stage']:10s} {'KEPT' if rec['kept'] else ''} "
                     f"{'' if rec['max_deviation_pct'] is None else 'maxdev %.4f%%' % rec['max_deviation_pct']} "
                     f"{'' if rec['kept'] else (rec['reason'] or '')[:110]}")
    if workers <= 1:
        if 'win' not in _W or _W.get('config') != config:
            _worker_init(config, out_dir)
        _W['out_dir'] = out_dir
        for j in jobs:
            take(evaluate(j))
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=_worker_init, initargs=(config, out_dir)) as pool:
            for rec in pool.imap_unordered(evaluate, jobs):
                take(rec)
    for f in logs.values():
        f.close()
    base_pv = None
    for ax in axles:
        base = next((r for r in records[ax] if r.get('is_baseline')), None)
        summ = _axle_summary(records[ax])
        summ['baseline_ok'] = bool(base and base['kept'])
        summ['baseline_stage'] = base['stage'] if base else None
        summ['baseline_reason'] = base['reason'] if base else None
        run['axles'][ax] = summ
        if base and base['kept']:
            mj = os.path.join(out_dir, ax, 'baseline', 'metrics.json')
            if base_pv is None and os.path.exists(mj):
                with open(mj, 'r', encoding='utf-8') as f:
                    base_pv = {r['name']: {'value': r['baseline'], 'unit': r['unit'], 'axle': r['axle'],
                                           'scale': r['floor_abs'] / REL_TOL} for r in json.load(f)['rows']}
    if base_pv is not None:
        run['parameters'] = parameter_table(base_pv)
        run['n_parameters'] = len(run['parameters'])
    run['elapsed_search_s'] = round(time.time() - t0, 1)
    for ax in axles:
        g = cluster_axle(os.path.join(out_dir, ax), cluster_mm)
        run['axles'][ax]['groups'] = len(g['groups'])
    run['finished'] = _dt.datetime.now().isoformat(timespec='seconds')
    with open(os.path.join(out_dir, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=1, default=float)
    if render:
        render_run(out_dir)
    return run


# ═════════════════════════════════════════════════════════════════════════════
#  GROUPING — scipy hierarchical clustering on chassis-side point deltas
# ═════════════════════════════════════════════════════════════════════════════
def _load_solutions(axle_dir: str) -> list:
    sols = []
    for mj in sorted(glob.glob(os.path.join(axle_dir, '*', 'metrics.json'))):
        try:
            with open(mj, 'r', encoding='utf-8') as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get('ok') and m.get('chassis_deltas_mm'):
            m['_dir'] = os.path.dirname(mj)
            sols.append(m)
    return sols


def cluster_axle(axle_dir: str, threshold_mm: float = CLUSTER_MM_DEFAULT) -> dict:
    """Complete-linkage clustering: distance(a, b) = max over chassis-side
    points of |p_a - p_b| (mm).  Cut at threshold_mm, so within a group every
    point of every pair is within threshold_mm.  Writes groups.json."""
    sols = _load_solutions(axle_dir)
    axle = os.path.basename(axle_dir.rstrip('/\\'))
    out = {'axle': axle, 'metric': GROUP_METRIC, 'threshold_mm': threshold_mm,
           'n_solutions': len(sols), 'groups': []}
    if not sols:
        with open(os.path.join(axle_dir, 'groups.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
        return out
    keys = sorted({k for s in sols for k in s['chassis_deltas_mm']})
    X = np.array([[s['chassis_deltas_mm'].get(k, [0, 0, 0]) for k in keys] for s in sols], float)  # n x P x 3
    n = len(sols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(X[i] - X[j], axis=1).max())
    if n == 1:
        labels = np.array([1])
    else:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(D, checks=False), method='complete')
        labels = fcluster(Z, t=threshold_mm, criterion='distance')
    groups = []
    for lab in sorted(set(labels.tolist())):
        idx = [i for i in range(n) if labels[i] == lab]
        sub = D[np.ix_(idx, idx)]
        rep = idx[int(np.argmin(sub.sum(axis=1)))]
        mean_shift = {k: float(np.mean(np.linalg.norm(X[idx][:, ki, :], axis=1))) for ki, k in enumerate(keys)}
        moved = {k: v for k, v in mean_shift.items() if v > 0.5}
        groups.append({'members': [sols[i]['id'] for i in idx],
                       'representative': sols[rep]['id'],
                       'spread_mm': float(sub.max()) if len(idx) > 1 else 0.0,
                       'contains_baseline': any(sols[i].get('is_baseline') for i in idx),
                       'families': sorted({f for i in idx for f in sols[i].get('families', [])}),
                       'mean_point_shift_mm': dict(sorted(moved.items(), key=lambda kv: -kv[1])),
                       'max_point_move_mm': float(max(sols[i].get('max_point_move_mm', 0.0) for i in idx)),
                       'max_deviation_pct': float(max(sols[i].get('max_deviation_pct', 0.0) for i in idx))})
    # baseline group first, then by size
    groups.sort(key=lambda g: (not g['contains_baseline'], -len(g['members'])))
    for i, g in enumerate(groups):
        g['group'] = i + 1
    out['groups'] = groups
    out['pairwise_max_mm'] = float(D.max())
    with open(os.path.join(axle_dir, 'groups.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=float)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  RENDER — native GL captures through the real 3D view
# ═════════════════════════════════════════════════════════════════════════════
VIEWS = {  # name, azimuth, elevation, distance scale, centre ('front'/'rear'/None)
    'front': [('gui_axle', -140, 12, 0.55, 'front'), ('gui_iso', -60, 20, 1.0, None), ('gui_top', 0, 90, 0.6, 'front')],
    'rear':  [('gui_axle', -40, 12, 0.55, 'rear'), ('gui_iso', -60, 20, 1.0, None), ('gui_top', 0, 90, 0.6, 'rear')],
}


def render_run(run_dir: str, force: bool = False) -> int:
    """Spawn the native-GL capture in a subprocess with QT_QPA_PLATFORM
    UNSET (offscreen has no GL).  Returns the subprocess exit code."""
    env = dict(os.environ); env.pop('QT_QPA_PLATFORM', None); env['VAHAN_MCP'] = '0'
    env['PYTHONIOENCODING'] = 'utf-8'
    cmd = [sys.executable, os.path.abspath(__file__), '--render', run_dir] + (['--force'] if force else [])
    r = subprocess.run(cmd, cwd=REPO, env=env)
    return r.returncode


def _render_main(run_dir: str, force: bool = False):
    """Runs INSIDE the GL subprocess: one windowed MainWindow, every
    solution loaded in turn, three views captured per solution."""
    if os.environ.get('QT_QPA_PLATFORM', '') == 'offscreen':
        raise SystemExit('render needs QT_QPA_PLATFORM unset (native GL)')
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from PIL import Image
    app = QApplication.instance() or QApplication([])
    from gui.main_window import MainWindow
    jobs = []
    for ax in AXLES:
        for cfg in sorted(glob.glob(os.path.join(run_dir, ax, '*', 'config.vahan'))):
            sdir = os.path.dirname(cfg)
            if not force and all(os.path.exists(os.path.join(sdir, v[0] + '.png')) for v in VIEWS[ax]):
                continue
            jobs.append((ax, cfg, sdir))
    if not jobs:
        print('render: nothing to do'); return
    w = MainWindow()
    w.show()
    state = {'i': 0}

    def axle_center(which):
        labels = ('FL', 'FR') if which == 'front' else ('RL', 'RR')
        pts = [np.asarray(w._solvers[l].solve(0.).wheel_center, float) for l in labels]
        c = np.mean(pts, axis=0); c[2] += 0.15
        return c

    def load_next():
        if state['i'] >= len(jobs):
            print(f'render: done, {len(jobs)} solutions'); app.quit(); return
        ax, cfg, sdir = jobs[state['i']]
        w._load_project_from_path(cfg)
        w._rebuild_solvers(0.)
        w._update_3d()
        cam = w.view3d._cam
        if 'base_dist' not in state:
            state['base_dist'] = float(getattr(cam, 'distance', None) or 2.5)
            state['base_center'] = tuple(cam.center)
        QTimer.singleShot(400, lambda: capture(ax, sdir, 0))

    def capture(ax, sdir, k):
        if k >= len(VIEWS[ax]):
            state['i'] += 1
            QTimer.singleShot(50, load_next); return
        name, az, el, ds, center = VIEWS[ax][k]
        cam = w.view3d._cam
        cam.azimuth = az; cam.elevation = el; cam.distance = state['base_dist'] * ds
        cam.center = tuple(axle_center(center)) if center else state['base_center']
        w.view3d._canvas.update()

        def snap():
            img = w.view3d._canvas.render()
            Image.fromarray(img[..., :3]).save(os.path.join(sdir, name + '.png'))
            capture(ax, sdir, k + 1)
        QTimer.singleShot(350, snap)

    QTimer.singleShot(1200, load_next)
    app.exec()


# ═════════════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='pin a config (default: highest configs/2027_v<N>)')
    ap.add_argument('--out', default=None, help='run dir (default designs_city/city_<stamp>)')
    ap.add_argument('--axles', nargs='+', default=list(AXLES), choices=list(AXLES))
    ap.add_argument('--trials', type=int, default=300, help='trials per axle')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cluster-mm', type=float, default=CLUSTER_MM_DEFAULT)
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--render', metavar='RUN_DIR', help='only (re)capture the images of a run')
    ap.add_argument('--cluster', metavar='RUN_DIR', help='only regroup a run')
    ap.add_argument('--force', action='store_true', help='with --render: overwrite existing images')
    a = ap.parse_args(argv)
    if a.render:
        _render_main(os.path.abspath(a.render), force=a.force); return 0
    if a.cluster:
        for ax in a.axles:
            g = cluster_axle(os.path.join(os.path.abspath(a.cluster), ax), a.cluster_mm)
            print(f"{ax}: {g['n_solutions']} solutions -> {len(g['groups'])} groups at {a.cluster_mm} mm")
        return 0
    run = run_city(a.config, a.out, tuple(a.axles), a.trials, a.workers, a.seed, a.cluster_mm,
                   render=not a.no_render)
    print(json.dumps({ax: {k: v for k, v in s.items() if k != 'kept_ids'} for ax, s in run['axles'].items()},
                     indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
