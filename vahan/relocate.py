"""Curve-preserving single-point relocation ("inverse IK").

The in-house IK targets a NEW curve and finds hardpoints.  This module answers
the opposite question: pick ONE hardpoint — where ELSE in space can it live so
that EVERY kinematic curve stays within tolerance of the original?

Method
------
1. The oracle is packaging.validate(allow_wheel_motion=True): when the chosen
   point is wheel-locating, every wheel metric AND the camber/toe curves across
   the real travel are re-measured and compared to the captured baseline within
   packaging.Tolerances; geometric laws (coplanarity, ARB-in-plane, triad) and
   MR/ARB rates are always enforced; the full-member clash sweep at droop/
   static/bump runs on the final presented set (skipped during sampling for
   throughput).
2. The feasible region around a hardpoint is typically a thin, elongated body
   (large freedom along physical invariance directions, mm-thin across), so
   uniform sampling would starve.  Search runs in two stages:
     A. RAY PROBING — bisect the feasibility boundary along 26 directions
        (cube faces/edges/corners) to measure the region's extent per
        direction.
     B. ELLIPSOID SAMPLING — random samples inside the stretched envelope the
        rays revealed (with margin), each judged by the oracle.
3. DIVERSITY — solutions are presented so that no two are more similar than
   the user's similarity setting (default 20%): similarity is PACKAGING
   distance only (position space); min separation = min_sep_frac × the
   largest feasible extent found.  Greedy farthest-point selection.
4. DRILL-DOWN — calling the search again with center=<a chosen solution> and
   radius=<that solution's neighbourhood (the parent min-separation)> reveals
   the finer solutions the diversity filter hid, recursively.

The model loaded in the MainWindow is temporarily modified during the search
and ALWAYS restored (try/finally) — the live model is the ONE MODEL; nothing
here computes physics of its own.
"""
from __future__ import annotations

import numpy as np

from vahan import packaging as PK


# ── directions: 26 cube directions, normalised ──────────────────────────────
def _ray_directions():
    dirs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == y == z == 0:
                    continue
                v = np.array([x, y, z], float)
                dirs.append(v / np.linalg.norm(v))
    return dirs


class _PointHandle:
    """Read/write one point ('hp'|'arb', key) on one axle of the live window,
    with guaranteed restore."""

    def __init__(self, win, axle: str, dict_name: str, key: str):
        self.win, self.axle, self.dict_name, self.key = win, axle, dict_name, key
        d = self._dict()
        if key not in d or d[key] is None:
            raise KeyError(f"{axle} {dict_name}.{key} not present")
        self.original = np.array(d[key], float)

    def _dict(self):
        if self.dict_name == 'hp':
            return self.win._front_hp if self.axle == 'front' else self.win._rear_hp
        return self.win._front_arb if self.axle == 'front' else self.win._rear_arb

    def set(self, pos_m):
        self._dict()[self.key] = np.array(pos_m, float)
        self.win._rebuild_solvers()

    def restore(self):
        self.set(self.original)


def _feasible(win, baseline, tol, axle, with_clash=False):
    """Oracle call for the CURRENT model state. Returns (ok, ValidationResult)."""
    res = PK.validate(win, baseline, tol, axles=(axle,), stop_early=True,
                      allow_wheel_motion=True, skip_clash=not with_clash)
    return res.ok, res


def relocate_search(win, axle: str, dict_name: str, key: str,
                    baseline: dict = None, tol: PK.Tolerances = None,
                    radius_mm: float = 120.0, n_samples: int = 250,
                    min_sep_frac: float = 0.20, n_show: int = 12,
                    center_mm=None, target_mm=None, seed: int = 0,
                    progress=None) -> dict:
    """Find diverse feasible relocations for one hardpoint.

    min_sep_frac — the similarity setting: presented solutions are at least
    this fraction of the largest feasible extent apart (0.20 = "no two
    solutions within 20% similarity").  Drill-down: pass center_mm=<a chosen
    solution> and radius_mm=<result['min_sep_mm']> to explore inside one
    neighbourhood.
    """
    tol = tol or PK.Tolerances()
    handle = _PointHandle(win, axle, dict_name, key)
    p0 = handle.original.copy()
    b0_snap = PK.get_bundle(win, axle)
    resolvable = ((dict_name == 'hp' and key in CHAIN_KEYS)
                  or (dict_name == 'arb' and key == 'arb_drop_top'))
    center = (np.asarray(center_mm, float) / 1000.0
              if center_mm is not None else p0.copy())
    R = float(radius_mm) / 1000.0
    rng = np.random.default_rng(seed)
    tried = feas = 0
    n_resolver_reject = 0
    feasible_pts = []      # (pos_m, worst_frac, res.summary())

    def apply_pos(pos_m, retune=False):
        """Place the point WITH chain resolution when the key supports it.
        Returns (ok, reason, (bundle, info)|None)."""
        nonlocal n_resolver_reject
        if resolvable:
            try:
                b, info = resolve_bundle(win, axle, dict_name, key, pos_m,
                                         b0_snap, baseline['rates'], tol,
                                         retune=retune)
            except Infeasible as e:
                n_resolver_reject += 1
                PK.set_bundle(win, axle, b0_snap)
                return False, str(e), None
            if b is not None:
                PK.set_bundle(win, axle, b)
                return True, '', (b, info)
        handle.set(pos_m)
        return True, '', None

    def report(msg):
        if progress:
            progress(msg)

    try:
        if baseline is None:
            report('capturing baseline…')
            baseline = PK.capture_baseline(win)

        # target mode: search is CENTRED on the user's target; if the target
        # itself is infeasible, walk back along target->original to the
        # closest feasible approach and centre there instead (rescue).
        target = (np.asarray(target_mm, float) / 1000.0
                  if target_mm is not None else None)
        if target is not None:
            center = target.copy()
        target_note = ''

        # sanity: the point at its CENTER position must be feasible
        okr, _why, _ = apply_pos(center)
        ok0, res0 = (_feasible(win, baseline, tol, axle) if okr
                     else (False, None))
        if not ok0 and target is not None:
            lo_t, hi_t = 0.0, 1.0          # p0 (feasible) -> target (not)
            for _ in range(8):
                mid = 0.5 * (lo_t + hi_t)
                okr, _w, _ = apply_pos(p0 + (target - p0) * mid)
                okm = okr and _feasible(win, baseline, tol, axle)[0]
                lo_t, hi_t = (mid, hi_t) if okm else (lo_t, mid)
            center = p0 + (target - p0) * lo_t
            miss = float(np.linalg.norm(center - target)) * 1000
            target_note = ('target itself is NOT feasible — searching from the '
                           'closest valid approach, %.1f mm short of it' % miss)
            okr, _w, _ = apply_pos(center)
            ok0, res0 = (_feasible(win, baseline, tol, axle) if okr
                         else (False, None))
        if not ok0:
            fails = ('; '.join(f"{c['axle']} {c['name']}"
                     for c in res0.failures()[:4]) if res0 is not None
                     else 'chain resolution impossible here')
            return {'error': 'center position is not feasible: ' + fails,
                    'solutions': [], 'n_tried': 1, 'n_feasible': 0}

        # ── stage A: ray probing ────────────────────────────────────────────
        dirs = _ray_directions()
        extents = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            lo_r, hi_r = 0.0, R
            # quick reject: full radius feasible? then extent = R
            okr, _w, _ = apply_pos(center + d * R)
            ok = okr and _feasible(win, baseline, tol, axle)[0]
            if ok:
                extents[i] = R
            else:
                for _ in range(6):                      # bisect boundary
                    mid = 0.5 * (lo_r + hi_r)
                    okr, _w, _ = apply_pos(center + d * mid)
                    ok = okr and _feasible(win, baseline, tol, axle)[0]
                    lo_r, hi_r = (mid, hi_r) if ok else (lo_r, mid)
                extents[i] = lo_r
            tried += 7
            report(f'probing direction {i + 1}/{len(dirs)} '
                   f'(extent {extents[i] * 1000:.1f} mm)')
        r_max = float(extents.max())
        if r_max < 1e-6:
            return {'error': 'no feasible motion found in any direction at '
                             'this tolerance', 'solutions': [],
                    'n_tried': tried, 'n_feasible': 0,
                    'extents_mm': (extents * 1000).round(2).tolist()}

        # ── stage B: sampling inside the revealed envelope ──────────────────
        # direction drawn uniformly, radius scaled by that direction's extent
        # (interpolated from the nearest probed ray), with 15% overshoot so
        # the boundary is not undersampled.
        D = np.array(dirs)
        for j in range(int(n_samples)):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            near = int(np.argmax(D @ v))
            r = rng.uniform(0.05, 1.15) * max(extents[near], 0.002)
            cand = center + v * min(r, R)
            okr, _w, _ = apply_pos(cand)
            ok, res = (_feasible(win, baseline, tol, axle) if okr
                       else (False, None))
            tried += 1
            if ok:
                feas += 1
                worst = max((_check_frac(c)
                             for c in res.checks
                             if c['tol'] > 1e-12
                             and c['name'] != 'wheel points moved'
                             and not c['name'].startswith('clash')),
                            default=0.0)
                feasible_pts.append((cand.copy(), worst))
            if progress and (j + 1) % 20 == 0:
                report(f'sampling {j + 1}/{n_samples} — {feas} feasible')

        # ray endpoints are feasible by construction — include them
        for i, d in enumerate(dirs):
            if extents[i] > 1e-6:
                feasible_pts.append((center + d * extents[i] * 0.98, 1.0))

        # ── diversity + clash: greedy farthest-point selection where every
        # pick must ALSO survive the full clash sweep; a clashing pick is
        # discarded (extreme positions are exactly where hardware collides)
        # and selection continues from the remaining pool. ────────────────────
        min_sep = float(min_sep_frac) * r_max
        chosen, sols = [], []
        pool = [p for p, _w in feasible_pts]
        n_clash_reject = 0

        clash_tally = {}

        def try_take(idx):
            nonlocal n_clash_reject
            p = pool.pop(idx)
            okr, _w, rb = apply_pos(p, retune=True)
            ok, res = (_feasible(win, baseline, tol, axle, with_clash=True)
                       if okr else (False, None))
            if res is None:
                n_clash_reject += 1
                return False
            if not ok:
                n_clash_reject += 1
                # tally which pair actually binds, so the user sees the wall
                for lst in res.clashes.values():
                    for cl in lst:
                        if cl['gap_mm'] < 0.0:
                            nm = f"{cl['corner']} {cl['a']} vs {cl['b']}"
                            clash_tally[nm] = clash_tally.get(nm, 0) + 1
                return False
            worst = max((_check_frac(c)
                         for c in res.checks
                         if c['tol'] > 1e-12
                         and c['name'] != 'wheel points moved'
                         and not c['name'].startswith('clash')),
                        default=0.0)
            chosen.append(p)
            sol = {
                'pos_mm': (p * 1000).round(3).tolist(),
                'delta_mm': ((p - p0) * 1000).round(3).tolist(),
                'dist_from_og_mm': round(float(np.linalg.norm(p - p0)) * 1000, 2),
                'worst_tol_frac': round(float(worst), 3),
                'clash_checked': True,
            }
            if rb is not None:
                bnd, rinfo = rb
                sol['resolved'] = rinfo
                sol['bundle'] = {
                    'hp': {k: np.asarray(v, float).tolist()
                           for k, v in bnd['hp'].items()},
                    'arb': {k: np.asarray(v, float).tolist()
                            for k, v in bnd['arb'].items()}}
            sols.append(sol)
            return True

        # seed: in target mode the point CLOSEST to the target; otherwise the
        # farthest from the original (max exploration) — that survives clash
        while pool and not chosen:
            if target is not None:
                d0 = [-np.linalg.norm(p - target) for p in pool]
            else:
                d0 = [np.linalg.norm(p - p0) for p in pool]
            if not try_take(int(np.argmax(d0))):
                report(f'clash rejected an extreme ({n_clash_reject} so far)')
        while pool and len(chosen) < int(n_show):
            dmin = [min(np.linalg.norm(p - c) for c in chosen) for p in pool]
            k = int(np.argmax(dmin))
            if dmin[k] < min_sep:
                break                                   # rest are all "similar"
            if progress:
                report(f'clash-checking pick {len(chosen) + 1} '
                       f'({n_clash_reject} rejected)')
            try_take(k)
        if target is not None:
            for so in sols:
                so['dist_from_target_mm'] = round(float(np.linalg.norm(
                    np.asarray(so['pos_mm']) - target * 1000)), 2)
            sols.sort(key=lambda s: s['dist_from_target_mm'])
        else:
            sols.sort(key=lambda s: -s['dist_from_og_mm'])
        return {
            'point': f'{axle} {dict_name}.{key}',
            'original_mm': (p0 * 1000).round(3).tolist(),
            'center_mm': (center * 1000).round(3).tolist(),
            'radius_mm': radius_mm,
            'extents_mm': (extents * 1000).round(2).tolist(),
            'max_extent_mm': round(r_max * 1000, 2),
            'min_sep_mm': round(min_sep * 1000, 2),
            'similarity_frac': min_sep_frac,
            'n_tried': tried, 'n_feasible': feas + 1,
            'target_mm': (None if target is None
                          else (target * 1000).round(2).tolist()),
            'target_note': target_note,
            'n_clash_rejected': n_clash_reject,
            'n_resolver_rejected': n_resolver_reject,
            'resolution': ('chain re-solved around every candidate '
                           '(coplanarity/axis/triad/rates restored)'
                           if resolvable else
                           'plain point move (curves judged directly)'),
            'clash_binding': dict(sorted(clash_tally.items(),
                                         key=lambda kv: -kv[1])[:5]),
            'solutions': sols,
        }
    finally:
        PK.set_bundle(win, axle, b0_snap)


# ═════════════════════════════════════════════════════════════════════════════
#  Hoop-line search: damper chassis mount constrained ONTO the front-hoop line
# ═════════════════════════════════════════════════════════════════════════════
def hoopline_search(win, n_show: int = 8, sim_frac: float = 0.20,
                    z_lo_mm: float = 380.0, z_hi_mm: float = 700.0,
                    n_steps: int = 22, tol=None, progress=None) -> dict:
    """Generate solutions with the FRONT damper chassis mount ON the hoop line
    (the line through the front suspension's aft inboard pickups, lca_rear ->
    uca_rear, extended upward) — so hoop tube = damper mount, one member.

    For each candidate height z on the line: rotate the actuation chain about
    the pushrod line so its plane CONTAINS the new mount (coplanarity law),
    move spring_chassis_pt onto the line, re-tune the rocker lever to the
    baseline motion ratio, re-hang + re-tune the ARB to the baseline rate,
    then judge with the full oracle (curves, laws, rates, clash sweep).
    Wheel-locating points never move.  Presented solutions are >= sim_frac x
    (feasible z-span) apart; the model is restored afterwards."""
    from vahan import packaging as PK
    tol = tol or PK.Tolerances()
    axle = 'front'
    b0 = PK.get_bundle(win, axle)
    hp0 = b0['hp']

    def report(msg):
        if progress:
            progress(msg)

    A = np.asarray(hp0['lca_rear'], float)
    B = np.asarray(hp0['uca_rear'], float)
    d = B - A
    if abs(d[2]) < 1e-9:
        return {'error': 'hoop line is horizontal — cannot parametrise by z'}
    d = d / np.linalg.norm(d)
    P0 = np.asarray(hp0['pushrod_outer'], float)
    P1 = np.asarray(hp0['pushrod_inner'], float)
    u = (P1 - P0) / np.linalg.norm(P1 - P0)

    try:
        report('capturing baseline…')
        baseline = PK.capture_baseline(win)
        rates0 = baseline['rates']
        target_mr = float(rates0['motion_ratio_front'])
        target_arb = float(rates0['arb_rate_front_Npm'])
        _, n0 = PK._chain_plane(b0)

        cands = []
        zs = np.linspace(z_lo_mm, z_hi_mm, int(n_steps)) / 1000.0
        for i, z in enumerate(zs):
            t = (z - A[2]) / d[2]
            T = A + d * t                       # ON the line at height z
            report(f'height {z * 1000:.0f} mm ({i + 1}/{len(zs)})…')
            try:
                # plane through the pushrod line and the new mount
                n1 = np.cross(u, T - P0)
                ln = np.linalg.norm(n1)
                if ln < 1e-9:
                    continue
                n1 /= ln
                if np.dot(n1, n0) < 0:
                    n1 = -n1
                ang = float(np.degrees(np.arctan2(
                    np.dot(np.cross(n0, n1), u), np.dot(n0, n1))))
                b = PK.rotate_about_pushrod_line(b0, ang)
                b['hp']['spring_chassis_pt'] = T.copy()
                b, _k, mr = PK.retune_mr(win, b, axle, target_mr)
                b['hp']['spring_chassis_pt'] = T.copy()   # lever scale safety
                b, _m, rate = PK.retune_arb(win, b, axle, target_arb, b0)
                PK.set_bundle(win, axle, b)
                res = PK.validate(win, baseline, tol, axles=(axle,),
                                  allow_wheel_motion=True)
                dlen = float(np.linalg.norm(
                    np.asarray(b['hp']['rocker_spring_pt']) - T)) * 1000
                cands.append({
                    'z_mm': round(z * 1000, 1),
                    'mount_mm': (T * 1000).round(2).tolist(),
                    'rotation_deg': round(ang, 2),
                    'mr': round(mr, 5), 'arb_Npm': round(rate, 1),
                    'damper_static_mm': round(dlen, 1),
                    'damper_flag': '' if 160 <= dlen <= 205 else
                                   'CHECK damper length vs 210 mm extended spec',
                    'ok': bool(res.ok),
                    'fails': ['%s %s' % (c['axle'], c['name'])
                              for c in res.failures()[:3]],
                })
            except Exception as e:
                cands.append({'z_mm': round(z * 1000, 1), 'ok': False,
                              'fails': [f'{type(e).__name__}: {e}'][:1]})

        passers = [c for c in cands if c.get('ok')]
        if not passers:
            return {'error': 'no feasible mount height on the hoop line at '
                             'this tolerance', 'candidates': cands}
        z_span = max(p['z_mm'] for p in passers) - min(p['z_mm'] for p in passers)
        min_sep = max(sim_frac * max(z_span, 1.0), 1.0)
        # 1-D farthest-point pick along z
        chosen = [max(passers, key=lambda p: p['z_mm'])]
        pool = [p for p in passers if p is not chosen[0]]
        while pool and len(chosen) < int(n_show):
            dmin = [min(abs(p['z_mm'] - c['z_mm']) for c in chosen)
                    for p in pool]
            k = int(np.argmax(dmin))
            if dmin[k] < min_sep:
                break
            chosen.append(pool.pop(k))
        chosen.sort(key=lambda p: p['z_mm'])
        return {'line': {'A_mm': (A * 1000).round(2).tolist(),
                         'B_mm': (B * 1000).round(2).tolist()},
                'targets': {'mr': target_mr, 'arb_Npm': target_arb},
                'z_span_mm': round(z_span, 1), 'min_sep_mm': round(min_sep, 1),
                'n_heights_tried': len(cands), 'n_feasible': len(passers),
                'solutions': chosen, 'all_candidates': cands}
    finally:
        PK.set_bundle(win, axle, b0)


def hoopline_apply(win, z_mm: float, tol=None) -> dict:
    """APPLY the hoop-line construction at one mount height to the live model
    (same construction as hoopline_search, but the result STAYS applied).
    Returns the solution record incl. the oracle verdict."""
    from vahan import packaging as PK
    tol = tol or PK.Tolerances()
    axle = 'front'
    b0 = PK.get_bundle(win, axle)
    hp0 = b0['hp']
    A = np.asarray(hp0['lca_rear'], float)
    B = np.asarray(hp0['uca_rear'], float)
    d = (B - A) / np.linalg.norm(B - A)
    P0 = np.asarray(hp0['pushrod_outer'], float)
    P1 = np.asarray(hp0['pushrod_inner'], float)
    u = (P1 - P0) / np.linalg.norm(P1 - P0)
    baseline = PK.capture_baseline(win)
    rates0 = baseline['rates']
    _, n0 = PK._chain_plane(b0)
    z = float(z_mm) / 1000.0
    T = A + d * ((z - A[2]) / d[2])
    n1 = np.cross(u, T - P0)
    n1 /= np.linalg.norm(n1)
    if np.dot(n1, n0) < 0:
        n1 = -n1
    ang = float(np.degrees(np.arctan2(np.dot(np.cross(n0, n1), u),
                                      np.dot(n0, n1))))
    b = PK.rotate_about_pushrod_line(b0, ang)
    b['hp']['spring_chassis_pt'] = T.copy()
    b, _k, mr = PK.retune_mr(win, b, axle, float(rates0['motion_ratio_front']))
    b['hp']['spring_chassis_pt'] = T.copy()
    b, _m, rate = PK.retune_arb(win, b, axle,
                                float(rates0['arb_rate_front_Npm']), b0)
    PK.set_bundle(win, axle, b)
    res = PK.validate(win, baseline, tol, axles=(axle,),
                      allow_wheel_motion=True)
    try:
        win._update_3d()
    except Exception:
        pass
    dlen = float(np.linalg.norm(
        np.asarray(b['hp']['rocker_spring_pt']) - T)) * 1000
    return {'applied': True, 'z_mm': round(z * 1000, 1),
            'mount_mm': (T * 1000).round(2).tolist(),
            'rotation_deg': round(ang, 2), 'mr': round(mr, 5),
            'arb_Npm': round(rate, 1), 'damper_static_mm': round(dlen, 1),
            'ok': bool(res.ok),
            'fails': ['%s %s' % (c['axle'], c['name'])
                      for c in res.failures()[:4]]}


def hoopline_bundle(win, z_mm: float) -> dict:
    """Compute the FULL solution geometry (hp+arb bundle) for one hoop-line
    mount height WITHOUT leaving it applied — for ghost visualisation.
    Returns {'bundle', 'z_mm', 'mount_mm', 'ok', 'fails'}; the live model is
    restored before returning."""
    from vahan import packaging as PK
    b0 = PK.get_bundle(win, 'front')
    try:
        r = hoopline_apply(win, z_mm)
        b = PK.get_bundle(win, 'front')
        r['bundle'] = b
        return r
    finally:
        PK.set_bundle(win, 'front', b0)
        try:
            win._update_3d()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  RESOLUTION: place one point at a target and re-solve the REST of the chain
#  so every law re-holds — coplanarity, rocker pivot axis, ARB triad, rates.
#  (Members stay axially loaded because the in-plane laws ARE the no-bending
#  laws: pushrod/damper/drop-link lines live in the rocker plane.)
# ═════════════════════════════════════════════════════════════════════════════
CHAIN_KEYS = ('pushrod_inner', 'rocker_pivot', 'rocker_spring_pt',
              'spring_chassis_pt')


class Infeasible(Exception):
    pass


def _plane_rotation(b0, dict_name, key, target_m):
    """New chain plane + the rotation (axis point, axis dir, angle) that maps
    the old plane onto it.  For non-pushrod chain points the plane pencil is
    THROUGH THE PUSHROD LINE (rod stays put -> its axial-load law holds); for
    pushrod_inner the new plane contains the NEW rod and stays as close to the
    old plane as possible."""
    hp0 = b0['hp']
    P0 = np.asarray(hp0['pushrod_outer'], float)
    P1 = np.asarray(hp0['pushrod_inner'], float)
    _, n0 = PK._chain_plane(b0)
    t = np.asarray(target_m, float)
    if key == 'pushrod_inner':
        v = t - P0
        if np.linalg.norm(v) < 1e-6:
            raise Infeasible('target coincides with the pushrod foot')
        v = v / np.linalg.norm(v)
        n1 = n0 - np.dot(n0, v) * v          # normal closest to old, ⊥ new rod
        if np.linalg.norm(n1) < 1e-9:
            raise Infeasible('degenerate plane for this pushrod direction')
        n1 = n1 / np.linalg.norm(n1)
    else:
        u = (P1 - P0) / np.linalg.norm(P1 - P0)
        n1 = np.cross(u, t - P0)
        if np.linalg.norm(n1) < 1e-9:
            raise Infeasible('target lies on the pushrod line — plane undefined')
        n1 = n1 / np.linalg.norm(n1)
    if np.dot(n1, n0) < 0:
        n1 = -n1
    axis = np.cross(n0, n1)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return n1, (P0, np.array([1.0, 0, 0]), 0.0)     # planes already equal
    axis = axis / s
    ang = float(np.arctan2(s, np.dot(n0, n1)))
    return n1, (P0, axis, ang)


def _rotate_about(p, origin, axis, ang):
    p = np.asarray(p, float) - origin
    c, s = np.cos(ang), np.sin(ang)
    return (origin + p * c + np.cross(axis, p) * s
            + axis * np.dot(axis, p) * (1 - c))



def _check_frac(c) -> float:
    """Fraction of tolerance a check consumes.  Rate/MR checks carry a
    PERCENT tolerance with N/m or ratio values — normalize by the reference
    so 0.34%% of a 1.5%% band reads 0.23, not 14.8."""
    if c['tol'] <= 1e-12:
        return 0.0
    if c['name'].endswith('_Npm') or c['name'].startswith('motion_ratio'):
        ref = abs(c['ref'])
        return (abs(c['value'] - c['ref']) / ref * 100 / c['tol']
                if ref > 1e-12 else 0.0)
    return abs(c['value'] - c['ref']) / c['tol']

def resolve_bundle(win, axle: str, dict_name: str, key: str, target_m,
                   b0: dict, rates0: dict, tol=None, retune: bool = True,
                   collect_all: bool = False, ref_arb: dict = None):
    """RESOLVED bundle with (dict_name,key) at target and the rest of the
    actuation re-solved: chain rotated onto the new plane (pushrod-line pencil
    -> rod undisturbed for non-rod points), rocker_axis_pt regenerated as the
    plane normal, ARB re-hung (and rates re-tuned to baseline when retune=True).
    Returns (bundle, info).  Raises Infeasible with the reason.
    Returns (None, {}) for wheel-side / unsupported keys — caller does a plain
    set and lets the oracle judge the curves."""
    tol = tol or PK.Tolerances()
    if dict_name == 'hp' and key == 'rocker_axis_pt':
        raise Infeasible('rocker_axis_pt is DERIVED (pivot + plane normal) — '
                         'move rocker_pivot instead')
    is_chain = dict_name == 'hp' and key in CHAIN_KEYS
    is_droptop = dict_name == 'arb' and key == 'arb_drop_top'
    if not (is_chain or is_droptop):
        return None, {}
    t = np.asarray(target_m, float)
    hp0 = b0['hp']
    info = {}
    if is_chain:
        n1, (origin, axis, ang) = _plane_rotation(b0, dict_name, key, t)
        b = {'hp': {k: np.array(v, float) for k, v in b0['hp'].items()},
             'arb': {k: np.array(v, float) for k, v in b0['arb'].items()}}
        rot_keys_hp = [k for k in CHAIN_KEYS if k != key]
        for k in rot_keys_hp:
            b['hp'][k] = _rotate_about(b['hp'][k], origin, axis, ang)
        for k in list(b['arb'].keys()):
            b['arb'][k] = _rotate_about(b['arb'][k], origin, axis, ang)
        b['hp'][key] = t.copy()
        # rocker pivot AXIS law: regenerate as the new plane normal
        ax_len = float(np.linalg.norm(np.asarray(hp0['rocker_axis_pt'])
                                      - np.asarray(hp0['rocker_pivot'])))
        b['hp']['rocker_axis_pt'] = b['hp']['rocker_pivot'] + n1 * ax_len
        info['chain_rotation_deg'] = round(float(np.degrees(ang)), 2)
    else:                                   # arb_drop_top: keep it in-plane
        _, n0 = PK._chain_plane(b0)
        c0, _ = PK._chain_plane(b0)
        t_in = t - np.dot(t - c0, n0) * n0
        b = {'hp': {k: np.array(v, float) for k, v in b0['hp'].items()},
             'arb': {k: np.array(v, float) for k, v in b0['arb'].items()}}
        b['arb'][key] = t_in
        info['projected_into_plane_mm'] = round(
            float(np.linalg.norm(t - t_in)) * 1000, 2)
    # ARB re-hang — CLASH-AWARE against the WHOLE damper problem: the naive
    # re-hang can put the drop link OR the chassis-fixed torsion bar straight
    # through the coilover (v69 target diag: bar -7.5 mm at 7.6 deg of chain
    # rotation, drop link -37 mm at full droop).  The pose angle (where the
    # drop-top sits on the rocker circle) is the one real packaging knob — it
    # swings the whole refit triangle, bar included.  A pose must clear the
    # coilover with the drop link, its rod ends AND the bar; with retune=True
    # the clearance is additionally swept across full droop/static/bump
    # (drop-top rides the rocker, arm end re-solved about the bar axis by
    # link-length preservation).
    from vahan.interference import seg_seg_distance
    _, n_pl = PK._chain_plane(b)
    piv = np.asarray(b['hp']['rocker_pivot'], float)
    # The ARB REFERENCE (triangle shape, triad angles, bar mount, standing
    # near-miss allowances) is the true baseline design.  Callers that hand a
    # constructed/rotated bundle as b0 pass the original design as ref_arb.
    ref = ref_arb if ref_arb is not None else b0

    sr = 0.5 * float(win._car.get('spring_od_mm', 63.0)) / 1000.0
    try:
        aod = float(getattr(win._dynamics_panel,
                            '_arb_OD_f' if axle == 'front'
                            else '_arb_OD_r').value()) / 1000.0
    except Exception:
        aod = 0.016
    LINK_R, RE_R, BRG_R, CLR_MM = 0.006, 0.008, 0.5 * 0.0381, 2.0

    mr_key = 'motion_ratio_front' if axle == 'front' else 'motion_ratio_rear'
    arb_key = 'arb_rate_front_Npm' if axle == 'front' else 'arb_rate_rear_Npm'

    # MR first (pushrod-lever knob; independent of the ARB pose) — the chain
    # must be FINAL before the rocker swing angles are measured.
    if retune:
        try:
            mr_now = PK.solver_mr(PK._corner_solver(win, axle, b))
            if (abs(mr_now - rates0[mr_key]) / rates0[mr_key] * 100
                    > tol.motion_ratio_pct / 2):
                b, _k, _mr = PK.retune_mr(win, b, axle, float(rates0[mr_key]))
                if is_chain:
                    b['hp'][key] = t.copy()
                info['mr_retuned'] = True
        except ValueError as e:
            raise Infeasible('motion ratio not recoverable here: %s' % e)

    def _proj_u(v):
        v = v - np.dot(v, n_pl) * n_pl
        nv = float(np.linalg.norm(v))
        return v / nv if nv > 1e-12 else v

    # Rocker swing angles at full droop / full bump for THIS chain — pose-
    # independent, so solve once.  Only needed for the travel sweep
    # (retune=True); growth (retune=False) judges statics and lets the
    # presentation sweep decide.
    swing = [(0.0, np.asarray(b['hp']['rocker_spring_pt'], float))]
    ref_swing = [(0.0, np.asarray(ref['hp']['rocker_spring_pt'], float))]
    if retune:
        # reference swing FIRST — win still holds the entry state
        try:
            lo_t, hi_t = PK.travel_range_m(win)
            solver = win._solvers[PK._LABEL[axle]]
            _, n_ref = PK._chain_plane(ref)
            piv_ref = np.asarray(ref['hp']['rocker_pivot'], float)
            rsr = np.asarray(ref['hp']['rocker_spring_pt'], float)
            vr = rsr - piv_ref
            vr = vr - np.dot(vr, n_ref) * n_ref
            u0r = vr / max(np.linalg.norm(vr), 1e-12)
            for tt in (lo_t, hi_t):
                st = solver.solve(float(tt))
                rs_t = np.asarray(st.rocker_spring_pt, float)
                v1 = rs_t - piv_ref
                v1 = v1 - np.dot(v1, n_ref) * n_ref
                u1r = v1 / max(np.linalg.norm(v1), 1e-12)
                th = float(np.arctan2(np.dot(np.cross(u0r, u1r), n_ref),
                                      np.dot(u0r, u1r)))
                ref_swing.append((th, rs_t))
        except Exception:
            pass
        PK.set_bundle(win, axle, b)
        try:
            lo_t, hi_t = PK.travel_range_m(win)
            solver = win._solvers[PK._LABEL[axle]]
            rs0 = np.asarray(b['hp']['rocker_spring_pt'], float)
            u0 = _proj_u(rs0 - piv)
            for tt in (lo_t, hi_t):
                st = solver.solve(float(tt))
                rs_t = np.asarray(st.rocker_spring_pt, float)
                u1 = _proj_u(rs_t - piv)
                th = float(np.arctan2(np.dot(np.cross(u0, u1), n_pl),
                                      np.dot(u0, u1)))
                swing.append((th, rs_t))
        except Exception:
            pass

    def _arm_end_at(dt_t, ae_s, pv, link_len):
        """Arm end after bar twist: rotate ae_s about the bar (X) axis through
        pv so the drop-link length is preserved.  Coarse scan + refine."""
        rel = ae_s - pv
        best_phi, best_err = 0.0, abs(float(np.linalg.norm(ae_s - dt_t))
                                      - link_len)
        for phi in np.linspace(-1.2, 1.2, 25):
            c, sn = np.cos(phi), np.sin(phi)
            e = pv + np.array([rel[0], c * rel[1] - sn * rel[2],
                               sn * rel[1] + c * rel[2]])
            err = abs(float(np.linalg.norm(e - dt_t)) - link_len)
            if err < best_err:
                best_phi, best_err = float(phi), err
        for phi in np.linspace(best_phi - 0.1, best_phi + 0.1, 21):
            c, sn = np.cos(phi), np.sin(phi)
            e = pv + np.array([rel[0], c * rel[1] - sn * rel[2],
                               sn * rel[1] + c * rel[2]])
            err = abs(float(np.linalg.norm(e - dt_t)) - link_len)
            if err < best_err:
                best_phi, best_err = float(phi), err
        c, sn = np.cos(best_phi), np.sin(best_phi)
        return pv + np.array([rel[0], c * rel[1] - sn * rel[2],
                              sn * rel[1] + c * rel[2]])

    def _pair_margins_mm(bb, sweep, own_swing=None):
        """Worst clearance (mm) of the coilover vs (drop link, rod end, bar),
        each as its own number.  sweep=True also evaluates full droop + full
        bump via the rocker swing angles."""
        dt_s = np.asarray(bb['arb']['arb_drop_top'], float)
        ae_s = np.asarray(bb['arb']['arb_arm_end'], float)
        pv = np.asarray(bb['arb']['arb_pivot'], float)
        bar_b = pv.copy(); bar_b[0] = -pv[0]
        dmp_b = np.asarray(bb['hp']['spring_chassis_pt'], float)
        link_len = float(np.linalg.norm(ae_s - dt_s))
        worst = [1e9, 1e9, 1e9, 1e9]
        sw = own_swing if own_swing is not None else swing
        states = sw if sweep else sw[:1]
        for th, rs_t in states:
            dt_t = (_rotate_about(dt_s, piv, n_pl, th) if abs(th) > 1e-9
                    else dt_s)
            ae_t = (_arm_end_at(dt_t, ae_s, pv, link_len) if abs(th) > 1e-9
                    else ae_s)
            ms = (seg_seg_distance(rs_t, dmp_b, dt_t, ae_t) - (sr + LINK_R),
                  seg_seg_distance(rs_t, dmp_b, dt_t, dt_t) - (sr + RE_R),
                  seg_seg_distance(rs_t, dmp_b, pv, bar_b) - (sr + aod / 2),
                  np.linalg.norm(dt_t - piv) - (BRG_R + RE_R))
            for i in range(4):
                worst[i] = min(worst[i], float(ms[i]) * 1000.0)
        return worst

    # Baseline full-member clash gaps (win holds b0 on entry) — the FINAL
    # arbiter for a pose is the SAME sweep the oracle runs, judged with the
    # oracle's own rule (no new contact; standing near-miss not worse).
    base_gaps = None
    if retune:
        base_gaps = {}
        for _st, _lst in PK._clash_sweep(win).items():
            for _cl in _lst:
                base_gaps[(_st, _cl['a'], _cl['b'])] = _cl['gap_mm']

    def _sweep_ok(margin=0.25):
        for _st, _lst in PK._clash_sweep(win).items():
            for _cl in _lst:
                _k = (_st, _cl['a'], _cl['b'])
                _g0 = base_gaps.get(_k)
                if _g0 is None:
                    if _cl['gap_mm'] < 0.0:
                        return False
                elif _cl['gap_mm'] < _g0 - margin:
                    return False
        return True

    # The oracle allows a STANDING baseline near-miss as long as it does not
    # get worse — mirror that: each pair's requirement is the lesser of the
    # absolute clearance and (its own baseline margin - the worsen allowance).
    base_m = _pair_margins_mm(ref, sweep=retune, own_swing=ref_swing)
    req = [min(CLR_MM, base_m[i] - 0.25) for i in range(4)]

    def _deficit_mm(bb, sweep):
        m = _pair_margins_mm(bb, sweep)
        return min(m[i] - req[i] for i in range(4))

    n_clear_but_untunable = 0
    last_retune_err = ''
    best_clear = -1e9
    chosen_bb = None
    collected = []
    poses = [0.0]
    for a in range(15, 181, 15):
        poses += [float(a), float(-a)] if a < 180 else [180.0]
    # MOUNTABILITY: the torsion bar only mounts where chassis structure
    # exists, so rank every (pose, branch) candidate by how far its bar
    # lands from the BASELINE mount (y,z) and refuse anything beyond
    # tol.arb_mount_shift_mm — closest fitting geometry first, and no
    # more structurally-silly floating bars.
    pv0_yz = np.asarray(ref['arb']['arb_pivot'], float)[1:]
    cap_m = float(getattr(tol, 'arb_mount_shift_mm', 60.0)) / 1000.0
    tries = []
    n_unmountable = 0
    for ang_deg in poses:
        bp = {'hp': {k: np.array(v, float) for k, v in b['hp'].items()},
              'arb': {k: np.array(v, float) for k, v in b['arb'].items()}}
        if abs(ang_deg) > 1e-9:
            bp['arb']['arb_drop_top'] = _rotate_about(
                bp['arb']['arb_drop_top'], piv, n_pl, np.radians(ang_deg))
        for br, bb in enumerate(PK.refit_arb_variants(bp, ref)):
            disp = float(np.linalg.norm(
                np.asarray(bb['arb']['arb_pivot'], float)[1:] - pv0_yz))
            if disp > cap_m:
                n_unmountable += 1
                continue
            tries.append((disp, ang_deg, br, bb))
    tries.sort(key=lambda t4: t4[0])

    if not retune:
        for disp, ang_deg, br, bb in tries:
            m = _deficit_mm(bb, sweep=False)
            best_clear = max(best_clear, m)
            if m < 0.0:
                continue
            chosen_bb = bb
            info['arb_pose_deg'] = ang_deg
            info['arb_refit_branch'] = br
            info['arb_bar_shift_mm'] = round(disp * 1000, 1)
            info['arb_drop_clearance_mm'] = round(float(m), 1)
            break
    else:
        # ── the POSE ANGLE is the primary rate knob ──────────────────────────
        # Measured on v69 at a 35 deg chain rotation: the ARB motion ratio
        # swings 1.2..14.9 (rate 1.8k..183k N/m) across drop-top poses while
        # arm length stays constant — so the search is (a) poses already at
        # the target rate, (b) poses a bearing-safe drop-radius trim can
        # reach, (c) continuous pose-angle bisection between rate-bracketing
        # neighbours.  Every acceptance still passes clearance, mountability
        # and the FULL member sweep.
        arb_target = float(rates0[arb_key])

        rated = []
        for disp, ang_deg, br, bb in tries:
            try:
                PK.set_bundle(win, axle, bb)
                rr = float(PK.panel_arb_rate(win, axle))
            except Exception:
                continue
            if np.isfinite(rr) and rr > 0:
                rated.append((disp, ang_deg, br, bb, rr))

        X_AX = np.array([1.0, 0.0, 0.0])

        def _tri_angles(bb):
            bl = (np.asarray(bb['arb']['arb_arm_end'], float)
                  - np.asarray(bb['arb']['arb_pivot'], float))
            dr = (np.asarray(bb['arb']['arb_drop_top'], float)
                  - np.asarray(bb['arb']['arb_arm_end'], float))
            # SAME math as the oracle's _axle_geometry_laws triad: signed
            # direction, no abs — a chirality-mirrored blade reads 180-theta
            # and must be rejected, not aliased.
            a = np.degrees(np.arccos(np.clip(np.dot(
                bl / np.linalg.norm(bl), X_AX), -1, 1)))
            d = np.degrees(np.arccos(np.clip(np.dot(
                dr / np.linalg.norm(dr), X_AX), -1, 1)))
            return a, d

        tri0 = _tri_angles(ref)

        def _finalize(bt, ang_deg, br, knob):
            nonlocal chosen_bb, n_clear_but_untunable, last_retune_err, \
                best_clear
            # triad LAW: bar/blade and bar/drop angles must stay at their
            # baseline values (the oracle gates this; checking here lets the
            # search move on to the next pose instead of dying downstream)
            ta, td = _tri_angles(bt)
            if (abs(ta - tri0[0]) > tol.triad_deg
                    or abs(td - tri0[1]) > tol.triad_deg):
                n_clear_but_untunable += 1
                last_retune_err = ('pose %.1f breaks the triad (bar-blade '
                                   '%.1f vs %.1f deg, bar-drop %.1f vs %.1f)'
                                   % (ang_deg, ta, tri0[0], td, tri0[1]))
                return False
            m2 = _deficit_mm(bt, sweep=True)
            if m2 < 0.0:
                best_clear = max(best_clear, m2)
                return False
            dnow = float(np.linalg.norm(np.asarray(
                bt['arb']['arb_pivot'], float)[1:] - pv0_yz))
            if dnow > cap_m:
                return False
            PK.set_bundle(win, axle, bt)
            rr = float(PK.panel_arb_rate(win, axle))
            if abs(rr - arb_target) / arb_target * 100 > tol.arb_rate_pct:
                n_clear_but_untunable += 1
                last_retune_err = ('pose %.1f rate %.0f vs target %.0f N/m'
                                   % (ang_deg, rr, arb_target))
                return False
            if not _sweep_ok():
                n_clear_but_untunable += 1
                last_retune_err = ('pose %.1f/branch %s clears the coilover '
                                   'but hits another member in the full '
                                   'sweep' % (ang_deg, br))
                return False
            extra = {'arb_pose_deg': round(float(ang_deg), 1),
                     'arb_refit_branch': br,
                     'arb_rate_knob': knob,
                     'arb_rate_Npm': round(rr, 0),
                     'arb_bar_shift_mm': round(dnow * 1000, 1),
                     'arb_drop_clearance_mm': round(float(m2), 1)}
            if collect_all:
                collected.append((bt, {**info, **extra}))
                return False        # keep enumerating
            chosen_bb = bt
            info.update(extra)
            return True

        def _pose_variant(bb, ddeg):
            """bb's drop-top nudged ddeg around the rocker, re-hung from the
            REFERENCE triangle, nearest-pivot branch, mountable-only."""
            bp3 = {'hp': {k: np.array(v, float) for k, v in b['hp'].items()},
                   'arb': {k: np.array(v, float) for k, v in b['arb'].items()}}
            bp3['arb']['arb_drop_top'] = _rotate_about(
                np.asarray(bb['arb']['arb_drop_top'], float), piv, n_pl,
                np.radians(ddeg))
            pv_ref = np.asarray(bb['arb']['arb_pivot'], float)
            best_v, best_d = None, 1e9
            for v in PK.refit_arb_variants(bp3, ref):
                pvn = np.asarray(v['arb']['arb_pivot'], float)
                if float(np.linalg.norm(pvn[1:] - pv0_yz)) > cap_m:
                    continue
                dd = float(np.linalg.norm(pvn - pv_ref))
                if dd < best_d:
                    best_v, best_d = v, dd
            return best_v

        def _triad_polish(bt):
            """Alternate a Newton pose-nudge (restores the bar-drop angle)
            with a blade re-tune (restores the rate) — the two fight, the
            loop settles them together or gives up."""
            cand = bt
            for _ in range(6):
                _ta, td = _tri_angles(cand)
                err = td - tri0[1]
                if abs(err) <= tol.triad_deg * 0.6:
                    return cand
                probe = _pose_variant(cand, 0.4)
                if probe is None:
                    return None
                _tp, tdp = _tri_angles(probe)
                slope = (tdp - td) / 0.4
                if abs(slope) < 1e-4:
                    return None
                step = float(np.clip(-err / slope, -10.0, 10.0))
                nxt = _pose_variant(cand, step)
                if nxt is None:
                    return None
                try:
                    nxt, _sc2, _r2 = PK.retune_arb_blade(
                        win, nxt, axle, arb_target, ref, s_lo=0.4, s_hi=4.0)
                except ValueError:
                    return None
                cand = nxt
            _ta, td = _tri_angles(cand)
            return cand if abs(td - tri0[1]) <= tol.triad_deg else None

        done = False
        # (a)+(b): nearest-rate first — direct accepts, then safe radius trim
        for disp, ang_deg, br, bb, rr in sorted(
                rated, key=lambda t5: abs(np.log(t5[4] / arb_target))):
            if _deficit_mm(bb, sweep=False) < 0.0:
                continue
            err_pct = abs(rr - arb_target) / arb_target * 100
            if err_pct <= tol.arb_rate_pct / 2:
                if _finalize(bb, ang_deg, br, 'pose'):
                    done = True
                    break
                continue
            r_now = float(np.linalg.norm(np.asarray(
                bb['arb']['arb_drop_top'], float) - piv))
            m_safe = max(0.15, (BRG_R + RE_R + 0.0015) / r_now)
            mm_need = float(np.sqrt(arb_target / rr))
            trims = []
            if m_safe <= mm_need <= 3.5:
                trims.append('radius')
            trims += ['blade', 'radius+blade']
            hit = False
            for knob in trims:
                try:
                    if knob == 'radius':
                        bt, _k2, _rt = PK.retune_arb(
                            win, bb, axle, arb_target, ref,
                            m_lo=m_safe, m_hi=3.5)
                    elif knob == 'blade':
                        bt, _k2, _rt = PK.retune_arb_blade(
                            win, bb, axle, arb_target, ref)
                    else:
                        # radius to its bearing-safe floor + blade for the rest
                        bt = PK.refit_arb(
                            PK.scale_arb_drop_radius(bb, m_safe), ref)
                        bt, _k2, _rt = PK.retune_arb_blade(
                            win, bt, axle, arb_target, ref, s_hi=4.0)
                except ValueError as e:
                    n_clear_but_untunable += 1
                    last_retune_err = str(e)
                    continue
                ok_now = _finalize(bt, ang_deg, br, knob)
                if (not ok_now and 'triad' in last_retune_err):
                    bt2 = _triad_polish(bt)
                    if bt2 is not None:
                        ok_now = _finalize(bt2, ang_deg, br,
                                           knob + '+pose-polish')
                if ok_now:
                    hit = True
                    break
            if hit:
                done = True
                break

        # (c): continuous pose bisection between rate-bracketing neighbours
        if not done:
            def _variant_at(angle_deg, pivot_ref):
                bp2 = {'hp': {k: np.array(v, float)
                              for k, v in b['hp'].items()},
                       'arb': {k: np.array(v, float)
                               for k, v in b['arb'].items()}}
                bp2['arb']['arb_drop_top'] = _rotate_about(
                    bp2['arb']['arb_drop_top'], piv, n_pl,
                    np.radians(angle_deg))
                vs = PK.refit_arb_variants(bp2, ref)
                best_v, best_d = None, 1e9
                for v in vs:
                    pvn = np.asarray(v['arb']['arb_pivot'], float)
                    if float(np.linalg.norm(pvn[1:] - pv0_yz)) > cap_m:
                        continue
                    d = float(np.linalg.norm(pvn - pivot_ref))
                    if d < best_d:
                        best_v, best_d = v, d
                return best_v

            by_angle = sorted(rated, key=lambda t5: t5[1])
            for j in range(len(by_angle) - 1):
                d1, a1d, br1, bb1, r1 = by_angle[j]
                d2, a2d, br2, bb2, r2 = by_angle[j + 1]
                if (r1 - arb_target) * (r2 - arb_target) > 0:
                    continue
                if abs(a2d - a1d) > 45.0:
                    continue
                lo_a, hi_a = a1d, a2d
                r_lo = r1
                pivot_ref = 0.5 * (
                    np.asarray(bb1['arb']['arb_pivot'], float)
                    + np.asarray(bb2['arb']['arb_pivot'], float))
                sol = None
                for _ in range(14):
                    mid = 0.5 * (lo_a + hi_a)
                    bm = _variant_at(mid, pivot_ref)
                    if bm is None:
                        break
                    try:
                        PK.set_bundle(win, axle, bm)
                        rm = float(PK.panel_arb_rate(win, axle))
                    except Exception:
                        break
                    pivot_ref = np.asarray(bm['arb']['arb_pivot'], float)
                    if abs(rm - arb_target) / arb_target * 100 \
                            <= tol.arb_rate_pct / 2:
                        sol = (mid, bm)
                        break
                    if (r_lo - arb_target) * (rm - arb_target) <= 0:
                        hi_a = mid
                    else:
                        lo_a, r_lo = mid, rm
                if sol is None:
                    continue
                mid, bm = sol
                if _deficit_mm(bm, sweep=False) < 0.0:
                    continue
                ok_now = _finalize(bm, mid, 'bisect', 'pose-bisect')
                if (not ok_now and 'triad' in last_retune_err):
                    bm2 = _triad_polish(bm)
                    if bm2 is not None:
                        ok_now = _finalize(bm2, mid, 'bisect',
                                           'pose-bisect+polish')
                if ok_now:
                    done = True
                    break
            # collect_all: also harvest every direct/trim candidate the
            # (a)/(b) loop queued into `collected` — nothing more to do.

    def _why():
        if not tries:
            return ('every re-hang would move the torsion bar more than '
                    '%.0f mm from its chassis mount (%d candidates refused '
                    'as unmountable)' % (cap_m * 1000, n_unmountable))
        if best_clear < 0.0 and n_clear_but_untunable == 0:
            return ('coilover clearance to the ARB (bar, drop link or rod '
                    'end) falls %.1f mm short across droop/static/bump'
                    % -best_clear)
        return ('%d candidate(s) clear the coilover but fail rate re-tune '
                'or the full member sweep (%s)'
                % (n_clear_but_untunable, last_retune_err))

    if collect_all and retune:
        if not collected:
            raise Infeasible('no MOUNTABLE ARB re-hang works at this '
                             'position: %s (tried %d, refused %d beyond the '
                             '%.0f mm bar-mount shift cap)'
                             % (_why(), len(tries), n_unmountable,
                                cap_m * 1000))
        return collected
    if chosen_bb is None:
        raise Infeasible('no MOUNTABLE ARB re-hang works at this position: '
                         '%s (tried %d, refused %d beyond the %.0f mm '
                         'bar-mount shift cap)'
                         % (_why(), len(tries), n_unmountable, cap_m * 1000))
    b = chosen_bb
    return b, info


def test_position(win, axle: str, dict_name: str, key: str, target_mm,
                  baseline: dict = None, tol=None) -> dict:
    """Try ONE exact position with full chain resolution + the full oracle.
    Returns {'ok', 'fails', 'info', 'bundle'(when resolvable+ok)}; the live
    model is restored regardless."""
    tol = tol or PK.Tolerances()
    b0 = PK.get_bundle(win, axle)
    t = np.asarray(target_mm, float) / 1000.0
    resolvable = ((dict_name == 'hp' and key in CHAIN_KEYS)
                  or (dict_name == 'arb' and key == 'arb_drop_top'))
    try:
        if baseline is None:
            baseline = PK.capture_baseline(win)
        info = {}
        if resolvable:
            try:
                b, info = resolve_bundle(win, axle, dict_name, key, t, b0,
                                         baseline['rates'], tol, retune=True)
            except Infeasible as e:
                return {'ok': False, 'fails': [str(e)], 'info': {},
                        'resolvable': True}
            PK.set_bundle(win, axle, b)
        else:
            h = _PointHandle(win, axle, dict_name, key)
            h.set(t)
            b = None
        ok, res = _feasible(win, baseline, tol, axle, with_clash=True)
        out = {'ok': bool(ok), 'resolvable': resolvable, 'info': info,
               'fails': ['%s %s' % (c['axle'], c['name'])
                         for c in res.failures()[:5]]}
        if ok and b is not None:
            out['bundle'] = {'hp': {k: np.asarray(v, float).tolist()
                                    for k, v in b['hp'].items()},
                             'arb': {k: np.asarray(v, float).tolist()
                                     for k, v in b['arb'].items()}}
        return out
    finally:
        PK.set_bundle(win, axle, b0)
        try:
            win._update_3d()
        except Exception:
            pass


def generative_search(win, axle: str, dict_name: str, key: str, target_mm,
                      baseline: dict = None, tol=None, n_iter: int = 300,
                      step_mm: float = 8.0, goal_bias: float = 0.45,
                      n_show: int = 10, min_sep_frac: float = 0.20,
                      seed: int = 0, progress=None) -> dict:
    """GENERATIVE relocation (topology-optimization-style growth): grow a tree
    of feasible positions from the current point TOWARD the target, one
    resolver-validated step at a time.  Unlike the local ball search, the tree
    traverses long curved corridors — it reaches wherever a continuous feasible
    path exists.  Every extension re-solves the chain (plane/axis/ARB/rates);
    presented solutions additionally pass the full clash sweep, nearest-to-
    target first, then diverse alternates (min separation = min_sep_frac x
    tree span)."""
    tol = tol or PK.Tolerances()
    handle = _PointHandle(win, axle, dict_name, key)
    p0 = handle.original.copy()
    b0_snap = PK.get_bundle(win, axle)
    resolvable = ((dict_name == 'hp' and key in CHAIN_KEYS)
                  or (dict_name == 'arb' and key == 'arb_drop_top'))
    target = np.asarray(target_mm, float) / 1000.0
    step = float(step_mm) / 1000.0
    rng = np.random.default_rng(seed)
    tried = 0

    def report(msg):
        if progress:
            progress(msg)

    def apply_pos(pos_m, retune=False):
        if resolvable:
            try:
                b, info = resolve_bundle(win, axle, dict_name, key, pos_m,
                                         b0_snap, baseline['rates'], tol,
                                         retune=retune)
            except Infeasible as e:
                PK.set_bundle(win, axle, b0_snap)
                return False, str(e), None
            if b is not None:
                PK.set_bundle(win, axle, b)
                return True, '', (b, info)
        handle.set(pos_m)
        return True, '', None

    try:
        if baseline is None:
            report('capturing baseline…')
            baseline = PK.capture_baseline(win)

        nodes = [p0.copy()]
        lo = np.minimum(p0, target) - (0.5 * np.linalg.norm(target - p0) + 0.03)
        hi = np.maximum(p0, target) + (0.5 * np.linalg.norm(target - p0) + 0.03)
        best_d = float(np.linalg.norm(p0 - target))
        hit_target = best_d < 1.5e-3
        for it in range(int(n_iter)):
            if hit_target and it > int(n_iter) * 0.6:
                break                        # target reached; finish exploring
            if rng.random() < goal_bias:
                samp = target + rng.normal(scale=step * 1.5, size=3)
            else:
                samp = rng.uniform(lo, hi)
            arr = np.asarray(nodes)
            near = int(np.argmin(np.linalg.norm(arr - samp, axis=1)))
            d = samp - nodes[near]
            nd = float(np.linalg.norm(d))
            if nd < 1e-9:
                continue
            newp = nodes[near] + d / nd * min(step, nd)
            if resolvable:
                # chain keys: the resolver CONSTRUCTS coplanarity/axis/triad
                # and never touches the wheel side — growth validity is
                # resolver success alone.  Rates + clash are enforced on the
                # presented picks (full retune + full oracle) — this is what
                # lets the tree actually travel.
                try:
                    b, _i = resolve_bundle(win, axle, dict_name, key, newp,
                                           b0_snap, baseline['rates'], tol,
                                           retune=False)
                    ok = b is not None
                except Infeasible:
                    ok = False
            else:
                okr, _w, _ = apply_pos(newp)
                ok = okr and _feasible(win, baseline, tol, axle)[0]
            tried += 1
            if ok:
                nodes.append(newp)
                dt = float(np.linalg.norm(newp - target))
                if dt < best_d:
                    best_d = dt
                    if dt < 1.5e-3:
                        hit_target = True
            if progress and (it + 1) % 25 == 0:
                report('growing: %d/%d iterations, tree %d nodes, best '
                       'approach %.1f mm from target'
                       % (it + 1, n_iter, len(nodes), best_d * 1000))

        if len(nodes) < 2:
            return {'error': 'could not grow anywhere from the current '
                             'position at this tolerance',
                    'solutions': [], 'n_tried': tried, 'tree_size': 1}

        # ── presentation: nearest-to-target first, then diverse; every pick
        # gets the FULL resolve (retunes) + full oracle incl. clash ──────────
        span = float(max(np.linalg.norm(np.asarray(nodes) - p0, axis=1).max(),
                         1e-3))
        min_sep = float(min_sep_frac) * span
        cand = sorted(nodes[1:], key=lambda p: np.linalg.norm(p - target))
        sols, chosen = [], []
        n_reject = 0
        reject_reasons = {}
        n_evals = 0
        max_evals = max(6 * int(n_show), 48)
        for p in cand:
            if len(sols) >= int(n_show) or n_evals >= max_evals:
                break
            if chosen and min(np.linalg.norm(p - c) for c in chosen) < min_sep \
                    and len(sols) >= 1:
                continue                     # too similar to a shown solution
            n_evals += 1
            okr, _w, rb = apply_pos(p, retune=True)
            ok, res = (_feasible(win, baseline, tol, axle, with_clash=True)
                       if okr else (False, None))
            if not ok:
                n_reject += 1
                if not okr:
                    why = _w.split('(')[0].strip()[:70]
                else:
                    why = '; '.join(sorted({'%s %s' % (c['axle'], c['name'])
                                            for c in res.failures()}))[:70]
                reject_reasons[why] = reject_reasons.get(why, 0) + 1
                continue
            worst = max((_check_frac(c)
                         for c in res.checks
                         if c['tol'] > 1e-12
                         and c['name'] != 'wheel points moved'
                         and not c['name'].startswith('clash')),
                        default=0.0)
            sol = {'pos_mm': (p * 1000).round(3).tolist(),
                   'delta_mm': ((p - p0) * 1000).round(3).tolist(),
                   'dist_from_og_mm': round(float(np.linalg.norm(p - p0)) * 1000, 2),
                   'dist_from_target_mm': round(float(np.linalg.norm(p - target)) * 1000, 2),
                   'worst_tol_frac': round(float(worst), 3),
                   'clash_checked': True}
            if rb is not None:
                bnd, rinfo = rb
                sol['resolved'] = rinfo
                sol['bundle'] = {
                    'hp': {k: np.asarray(v, float).tolist()
                           for k, v in bnd['hp'].items()},
                    'arb': {k: np.asarray(v, float).tolist()
                            for k, v in bnd['arb'].items()}}
            sols.append(sol)
            chosen.append(p)
        return {'point': f'{axle} {dict_name}.{key}',
                'method': 'generative tree growth (goal-biased RRT), '
                          'chain resolved at every step',
                'original_mm': (p0 * 1000).round(3).tolist(),
                'target_mm': (target * 1000).round(2).tolist(),
                'tree_size': len(nodes),
                'best_approach_mm': round(best_d * 1000, 1),
                'target_reached': bool(hit_target),
                'span_mm': round(span * 1000, 1),
                'min_sep_mm': round(min_sep * 1000, 1),
                'n_tried': tried, 'n_feasible': len(nodes) - 1,
                'n_final_rejects': n_reject,
                'reject_reasons': reject_reasons,
                'solutions': sols}
    finally:
        PK.set_bundle(win, axle, b0_snap)
        try:
            win._update_3d()
        except Exception:
            pass


def pin_point_options(win, axle: str, dict_name: str, key: str, target_mm,
                      baseline: dict = None, tol=None, n_show: int = 6,
                      dedupe_mm: float = 15.0) -> dict:
    """The point is PINNED exactly at target_mm and HELD.  Returns every
    distinct way the rest of the hardware can be re-solved around it (chain
    rotation is unique; the variety is the ARB re-hang: pose around the rocker
    x refit branch), each judged by the FULL oracle including the travel clash
    sweep.  The live model is restored."""
    tol = tol or PK.Tolerances()
    b0 = PK.get_bundle(win, axle)
    t = np.asarray(target_mm, float) / 1000.0
    p0 = np.asarray((b0['hp'] if dict_name == 'hp' else b0['arb'])[key], float)
    try:
        if baseline is None:
            baseline = PK.capture_baseline(win)
        try:
            cands = resolve_bundle(win, axle, dict_name, key, t, b0,
                                   baseline['rates'], tol, retune=True,
                                   collect_all=True)
        except Infeasible as e:
            return {'point': f'{axle} {dict_name}.{key}', 'pinned': True,
                    'target_mm': np.asarray(target_mm, float).round(3).tolist(),
                    'error': str(e), 'solutions': []}
        out, seen, rejected = [], [], []
        for bnd, info in cands:
            ae = np.asarray(bnd['arb']['arb_arm_end'], float)
            if any(float(np.linalg.norm(ae - sv)) < dedupe_mm / 1000.0
                   for sv in seen):
                continue
            PK.set_bundle(win, axle, bnd)
            ok, res = _feasible(win, baseline, tol, axle, with_clash=True)
            if not ok:
                rejected.append({'pose_deg': info.get('arb_pose_deg'),
                                 'branch': info.get('arb_refit_branch'),
                                 'fails': sorted({'%s %s' % (c['axle'],
                                                             c['name'])
                                                  for c in res.failures()})})
                continue
            worst = max((_check_frac(c)
                         for c in res.checks
                         if c['tol'] > 1e-12
                         and c['name'] != 'wheel points moved'
                         and not c['name'].startswith('clash')),
                        default=0.0)
            seen.append(ae)
            out.append({'pos_mm': (t * 1000).round(3).tolist(),
                        'delta_mm': ((t - p0) * 1000).round(3).tolist(),
                        'dist_from_og_mm': round(
                            float(np.linalg.norm(t - p0)) * 1000, 2),
                        'dist_from_target_mm': 0.0,
                        'worst_tol_frac': round(float(worst), 3),
                        'clash_checked': True,
                        'resolved': info,
                        'bundle': {
                            'hp': {k: np.asarray(v, float).tolist()
                                   for k, v in bnd['hp'].items()},
                            'arb': {k: np.asarray(v, float).tolist()
                                    for k, v in bnd['arb'].items()}}})
            if len(out) >= int(n_show):
                break
        return {'point': f'{axle} {dict_name}.{key}', 'pinned': True,
                'method': 'point pinned at target; ARB re-hang options '
                          'enumerated (pose x refit branch), full oracle each',
                'original_mm': (p0 * 1000).round(3).tolist(),
                'target_mm': np.asarray(target_mm, float).round(3).tolist(),
                'n_candidates': len(cands),
                'rejected': rejected,
                'solutions': out}
    finally:
        PK.set_bundle(win, axle, b0)
        try:
            win._update_3d()
        except Exception:
            pass


def tolerances_pct(baseline: dict, axle: str = 'front', pct: float = 10.0,
                   **overrides):
    """Tolerances derived from the BASELINE's own kinematics: each wheel-side
    band = pct percent of that metric's real size on this car (curve range
    across travel where a curve exists, magnitude otherwise), with sane
    floors.  Dynamics stay tight (MR 0.5, ARB 1.5 percent) — "10 percent on
    kinematic curves, smaller on dynamics"."""
    m = baseline['wheel_metrics'][axle]
    f = pct / 100.0

    def rng(key):
        v = [x for x in m.get(key, []) if np.isfinite(x)]
        return (max(v) - min(v)) if len(v) >= 2 else 0.0

    kw = dict(
        camber_deg=max(f * rng('camber_curve_deg'), 0.05),
        toe_deg=max(f * rng('toe_curve_deg'), 0.04),
        bump_steer_deg=max(f * abs(m.get('bump_steer_deg', 0.0)), 0.02),
        caster_deg=max(f * abs(m.get('caster_deg', 0.0)), 0.10),
        kpi_deg=max(f * abs(m.get('kpi_deg', 0.0)), 0.10),
        scrub_mm=max(f * abs(m.get('scrub_radius_mm', 0.0)), 1.0),
        trail_mm=max(f * abs(m.get('mechanical_trail_mm', 0.0)), 1.0),
        rc_height_mm=max(f * abs(m.get('roll_center_height_mm', 0.0)), 2.0),
        motion_ratio_pct=0.5, arb_rate_pct=1.5,
        coplanar_mm=1.5, arb_inplane_mm=3.0, triad_deg=1.0,
        clash_worsen_mm=0.25)
    kw.update(overrides)
    return PK.Tolerances(**kw)
