"""vahan/corner_speed.py — TIGHTEST CORNER vs SPEED and the PER-CORNER GRIP BUDGET.

Orchestration only.  Two questions, both answered by engines that already exist
in this package; nothing here looks up a tyre, splits a load or sums a moment.

1.  corner_speed_rows()
    For each corner radius R: the highest TRIMMED (N = 0) lateral acceleration
    the car can hold on that radius (vahan.ymd.trim_sweep_ackermann — the ONE
    yaw-moment engine — at the car's as-built Ackermann), the speed that goes
    with it (a fixed radius ties speed to lateral g: V = sqrt(Ay * g * R)), and
    the stability derivative dN/dbeta at that trim (negative = the car
    self-corrects).  With and without aero.  At constant radius the downforce
    scales with Ay (F = 0.5*rho*ClA*V^2 = 0.5*rho*ClA*g*R * Ay), so aero enters
    as the per-corner N-per-g dict the app's aero path already produces
    (MainWindow._get_aero_Fz_per_g(radius_m=R)); build_loads_table scales it
    by Ay exactly the way vahan.ackermann does.  The steering-lock radius is
    NOT computed here: the caller solves the front corners at full rack travel
    (the kinematic solver) and hands the two toe angles to lock_radius_m().

2.  per_corner_utilization() / per_corner_limit_g() / grip_budget_study()
    The user's standard: "ALL four tyres inside their budget".  The steady-
    state solver (vahan.dynamics.SteadyStateSolver.solve) already reports every
    corner's utilization = demand / (mu_peak(Fz, IA) * mu_scale * Fz); this
    module reads it and bisects the first lateral g at which any corner
    exceeds 1.0.  The axle-aggregate criterion (solver.axle_utilization, the
    app's canonical limit) is bisected alongside so the two definitions sit
    next to each other instead of being quoted from different sessions.

Consumers: gui/corner_speed_page.py (Ctrl+9) and test_one_model.py.
"""
from __future__ import annotations

import math

import numpy as np

from .ymd import G, build_loads_table, trim_sweep_ackermann

CORNERS = ('FL', 'FR', 'RL', 'RR')

# Default radius list for the speed-vs-radius study (m): a fast sweeper down
# to an FSAE hairpin.  Large radii first — they trim fast, the hairpin last —
# so a live page fills in from the easy end.  The caller appends the
# steering-lock radius.
DEFAULT_RADII_M = (50.0, 30.0, 20.0, 12.0, 8.0, 6.0, 4.5)


# ═══════════════════════════════════════════════════════════════════════════
#  1. tightest corner vs speed
# ═══════════════════════════════════════════════════════════════════════════
def lock_radius_m(toe_left_deg: float, toe_right_deg: float,
                  wheelbase_m: float) -> float:
    """Kinematic minimum corner radius at full steering lock.

    Bicycle-model radius from the MEAN of the two front road-wheel angles (with
    positive Ackermann the inner wheel turns more and the outer less; the
    axle's mean angle is the bicycle steer): R = L / tan(mean steer).  The two
    toe angles come from the kinematic solver with the rack at full travel —
    the caller owns that solve.  NaN when the mean steer is below 0.1 deg.
    """
    d = 0.5 * (abs(float(toe_left_deg)) + abs(float(toe_right_deg)))
    if not np.isfinite(d) or d <= 0.1:
        return float('nan')
    return float(wheelbase_m) / math.tan(math.radians(d))


def full_lock_handwheel_deg(steer_cfg: dict) -> float:
    """Handwheel angle (deg, one side) that puts the rack at its physical stop:
    total travel / travel per rev = revs lock-to-lock, x 360 / 2."""
    per_rev = float(steer_cfg.get('rack_travel_per_rev_mm', 0.0) or 0.0)
    total = float(steer_cfg.get('total_rack_travel_mm', 0.0) or 0.0)
    if per_rev <= 0.0 or total <= 0.0:
        return float('nan')
    return total / per_rev * 180.0


def speed_from_lateral_g(ay_g: float, radius_m: float) -> float:
    """V (m/s) that holds radius R at lateral acceleration Ay: V^2 = Ay*g*R."""
    if not np.isfinite(ay_g) or ay_g <= 0.0:
        return float('nan')
    return math.sqrt(float(ay_g) * G * float(radius_m))


def _nan_row(radius_m: float, aero: bool, aero_per_g: dict | None,
             error: str | None) -> dict:
    return {'radius_m': float(radius_m), 'aero': bool(aero),
            'ay_g': float('nan'), 'speed_mps': float('nan'),
            'speed_kph': float('nan'),
            'N_beta_Nm_per_deg': float('nan'), 'N_delta_Nm_per_deg': float('nan'),
            'beta_deg': float('nan'), 'delta_deg': float('nan'),
            'converged': False, 'stable': None,
            'aero_per_g_N': (float(sum(aero_per_g.values())) if aero_per_g else 0.0),
            'downforce_N': float('nan') if aero else 0.0,
            'error': error}


def corner_speed_row(tire_model, solver, radius_m: float, *,
                     ackermann_pct: float, grip_multiplier: float,
                     aero_per_g: dict | None = None,
                     loads_table=None) -> dict:
    """ONE (radius, aero) row of the speed-vs-radius study.

    aero_per_g: None = no aero; else the per-corner N at 1 lateral g ON THIS
    RADIUS (MainWindow._get_aero_Fz_per_g(radius_m=R)).  loads_table: an
    already-built build_loads_table(solver, aero_per_g) to reuse (the no-aero
    table is radius-independent).
    """
    aero = bool(aero_per_g)
    try:
        table = (loads_table if loads_table is not None
                 else build_loads_table(solver, aero_per_g or None))
        r = trim_sweep_ackermann(tire_model, solver, radius_m=float(radius_m),
                                 ackermann_list=(float(ackermann_pct),),
                                 grip_multiplier=float(grip_multiplier),
                                 aero_Fz_per_g=aero_per_g or None,
                                 loads_table=table)[0]
    except Exception as e:                       # noqa: BLE001 — recorded per row
        return _nan_row(radius_m, aero, aero_per_g, f'{type(e).__name__}: {e}'[:120])
    ay = float(r.get('Ay_trim_max', float('nan')))
    V = speed_from_lateral_g(ay, radius_m)
    nb = float(r.get('N_beta', float('nan')))
    per_g_total = float(sum(aero_per_g.values())) if aero else 0.0
    return {'radius_m': float(radius_m), 'aero': aero,
            'ay_g': ay, 'speed_mps': V,
            'speed_kph': V * 3.6 if np.isfinite(V) else float('nan'),
            'N_beta_Nm_per_deg': nb,
            'N_delta_Nm_per_deg': float(r.get('N_delta', float('nan'))),
            'beta_deg': float(r.get('beta_at', float('nan'))),
            'delta_deg': float(r.get('delta_at', float('nan'))),
            'converged': bool(r.get('converged', False)),
            'stable': (bool(nb < 0.0) if np.isfinite(nb) else None),
            'aero_per_g_N': per_g_total,
            # downforce the trim engine had on the car at its limit: the same
            # per-g dict it was fed, scaled by the trimmed Ay (V^2 scaling on a
            # fixed radius) — no second aero formula.
            'downforce_N': (per_g_total * ay if aero and np.isfinite(ay) else
                            (float('nan') if aero else 0.0)),
            'error': None}


def corner_speed_rows(tire_model, solver, radii_m, *, ackermann_pct: float,
                      grip_multiplier: float, aero_per_g_by_radius=None,
                      progress=None, cancelled=None) -> list:
    """The speed-vs-radius study: one row per (radius, aero flag), in the
    order the radii are given (no-aero row first, then the aero row).

    aero_per_g_by_radius: None = no aero rows at all; else {radius_m: per-
    corner N-per-g dict or None}.  A radius whose entry is None gets no aero
    row.  progress(i_done, n_total, row) is called after every row;
    cancelled() -> True stops early (the rows so far are returned).
    """
    radii = [float(R) for R in radii_m]
    aero_map = dict(aero_per_g_by_radius or {})
    plan = []
    for R in radii:
        plan.append((R, None))
        af = aero_map.get(R)
        if af:
            plan.append((R, dict(af)))
    rows, table_noaero = [], None
    for i, (R, af) in enumerate(plan):
        if cancelled is not None and cancelled():
            break
        if af is None:
            if table_noaero is None:
                try:
                    table_noaero = build_loads_table(solver, None)
                except Exception as e:           # noqa: BLE001
                    rows.append(_nan_row(R, False, None,
                                         f'{type(e).__name__}: {e}'[:120]))
                    if progress is not None:
                        progress(i + 1, len(plan), rows[-1])
                    continue
            row = corner_speed_row(tire_model, solver, R,
                                   ackermann_pct=ackermann_pct,
                                   grip_multiplier=grip_multiplier,
                                   aero_per_g=None, loads_table=table_noaero)
        else:
            row = corner_speed_row(tire_model, solver, R,
                                   ackermann_pct=ackermann_pct,
                                   grip_multiplier=grip_multiplier,
                                   aero_per_g=af)
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(plan), row)
    return rows


def corner_speed_series(rows: list, aero: bool):
    """(radius, speed_kph, ay_g) arrays for one aero flag, in row order —
    exactly what the page plots, so a test can compare plot data to rows."""
    sel = [r for r in rows if bool(r['aero']) == bool(aero)]
    R = np.asarray([r['radius_m'] for r in sel], float)
    V = np.asarray([r['speed_kph'] for r in sel], float)
    A = np.asarray([r['ay_g'] for r in sel], float)
    return R, V, A


def corner_speed_table_text(rows: list, lock_radius: float = float('nan'),
                            sep: str = '\t') -> str:
    """The rows as a copy-pasteable table (tab-separated by default)."""
    hdr = ['radius_m', 'aero', 'lateral_g', 'speed_kph', 'speed_mps',
           'stability_N_beta_Nm_per_deg', 'stable', 'downforce_at_limit_N',
           'body_slip_deg', 'front_steer_deg', 'converged', 'note']
    out = [sep.join(hdr)]
    for r in rows:
        note = r.get('error') or ''
        if np.isfinite(lock_radius) and abs(r['radius_m'] - lock_radius) < 1e-9:
            note = ('steering lock' + (' — ' + note if note else ''))
        out.append(sep.join([
            f'{r["radius_m"]:.2f}', 'aero' if r['aero'] else 'no aero',
            f'{r["ay_g"]:.3f}', f'{r["speed_kph"]:.1f}', f'{r["speed_mps"]:.2f}',
            f'{r["N_beta_Nm_per_deg"]:+.1f}',
            {True: 'yes', False: 'NO', None: '?'}[r['stable']],
            f'{r["downforce_N"]:.0f}', f'{r["beta_deg"]:+.2f}',
            f'{r["delta_deg"]:.2f}', 'yes' if r['converged'] else 'no', note]))
    return '\n'.join(out)


# ═══════════════════════════════════════════════════════════════════════════
#  2. per-corner grip budget — "all four tyres inside their budget"
# ═══════════════════════════════════════════════════════════════════════════
def _aero_row(lateral_g: float, aero_Fz=None, aero_Fz_per_g=None):
    """The solver's aero_Fz input for one solve: a FIXED per-corner dict
    (downforce at one speed) or a per-g dict scaled by |g| (constant radius,
    V^2 scaling — the build_loads_table / vahan.ackermann convention)."""
    if aero_Fz:
        return {k: float(aero_Fz.get(k, 0.0)) for k in CORNERS}
    if aero_Fz_per_g:
        return {k: float(aero_Fz_per_g.get(k, 0.0)) * abs(float(lateral_g))
                for k in CORNERS}
    return None


def per_corner_utilization(solver, lateral_g: float, *, aero_Fz=None,
                           aero_Fz_per_g=None) -> dict:
    """Every corner's grip budget at one lateral g, read off the steady-state
    solver: Fz, signed inclination (IA < 0 = leaning into the turn), the
    demanded planar force (|Fy, Fx| — the solver's own combined demand), the
    budget mu_peak(Fz, IA) * mu_scale * Fz, and utilization = demand/budget.
    The budget is backed out of the solver's utilization (budget = demand /
    utilization) so there is ONE mu lookup in the tool, inside solve().
    below_data_floor flags a corner whose Fz sits under the tyre file's
    lowest tested load (mu is clamped at the floor there)."""
    af = _aero_row(lateral_g, aero_Fz, aero_Fz_per_g)
    res = solver.solve(float(lateral_g), 0.0, aero_Fz=af)
    corners = {}
    for c in CORNERS:
        fz = float(res.Fz.get(c, float('nan')))
        fy = float(res.Fy.get(c, 0.0)) if res.Fy else 0.0
        fx = float(res.Fx.get(c, 0.0)) if res.Fx else 0.0
        dem = float(np.hypot(fy, fx))
        util = float(res.utilization.get(c, float('nan')))
        bud = dem / util if (np.isfinite(util) and util > 0.0) else float('nan')
        tire = solver._tire_for(c)
        try:
            floor = float(np.asarray(tire.fz_range).ravel()[0])
        except Exception:                        # noqa: BLE001 — linear tyre etc.
            floor = 0.0
        corners[c] = {'Fz_N': fz,
                      'inclination_deg': float((res.inclination or {}).get(c, 0.0)),
                      'demand_N': dem, 'budget_N': bud, 'utilization': util,
                      'below_data_floor': bool(fz < floor), 'data_floor_N': floor}
    worst = max(CORNERS, key=lambda c: np.nan_to_num(corners[c]['utilization'], nan=-1.0))
    axle = solver.axle_utilization(res)
    return {'lateral_g': float(lateral_g), 'corners': corners,
            'worst_corner': worst,
            'worst_utilization': corners[worst]['utilization'],
            'axle_utilization': {k: float(v) for k, v in axle.items()},
            'grip_exceeded': dict(res.grip_exceeded or {}),
            'aero_Fz_applied': af, 'mu_scale': float(solver._mu_scale),
            'roll_angle_deg': float(res.roll_angle_deg)}


def per_corner_limit_g(solver, *, aero_Fz=None, aero_Fz_per_g=None,
                       criterion: str = 'corner', lo: float = 0.3,
                       hi: float = 4.0, iters: int = 24) -> dict:
    """The first lateral g at which the chosen criterion exceeds 1.0, by
    bisection on [lo, hi] (monotone in g: load transfer only ever moves grip
    from the inner to the outer tyre).

    criterion 'corner': max over the four corners of the solver's per-corner
    utilization — the user's "all four tyres inside their budget".
    criterion 'axle':  max over the two axles of solver.axle_utilization —
    the app's canonical pair-budget limit, for reference.
    A solve that raises (wheel-lift refusal) counts as OVER the limit and is
    tallied in n_failed_solves.  Returns the limit, the binding corner/axle
    and the full per-corner table AT the limit."""
    if criterion not in ('corner', 'axle'):
        raise ValueError("criterion must be 'corner' or 'axle'")

    def _metric(g):
        u = per_corner_utilization(solver, g, aero_Fz=aero_Fz,
                                   aero_Fz_per_g=aero_Fz_per_g)
        if criterion == 'corner':
            return u['worst_utilization'], u
        return max(u['axle_utilization'].values()), u

    lo, hi = float(lo), float(hi)
    hi0 = hi
    n_fail, last_ok = 0, None
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        try:
            m, u = _metric(mid)
        except Exception:                        # noqa: BLE001
            m, u, n_fail = float('inf'), None, n_fail + 1
        if m < 1.0:
            lo, last_ok = mid, u
        else:
            hi = mid
    g_lim = 0.5 * (lo + hi)
    if n_fail >= int(iters):
        # Every probe raised: nothing was measured.  Returning the lower
        # bracket here printed a fake "0.300 g" limit on the Corner Speed and
        # Build Tolerance pages when the tyre chain could not solve at all
        # (audit C2, 2026-10-05).  NaN + the failure count instead.
        g_lim = float('nan')
    try:
        _, at = _metric(g_lim)
    except Exception:                            # noqa: BLE001
        at = last_ok
    if at is not None:
        if criterion == 'corner':
            binding = at['worst_corner']
        else:
            binding = max(at['axle_utilization'], key=at['axle_utilization'].get)
    else:
        binding = None
    return {'criterion': criterion, 'limit_g': float(g_lim), 'binding': binding,
            'at_limit': at, 'lo': lo, 'hi': hi, 'iterations': int(iters),
            'n_failed_solves': n_fail,
            # hi never moved: no g in [lo, hi] ever exceeded the criterion,
            # so the "limit" is the search ceiling, not a measurement.
            'hit_upper_bound': bool(hi >= hi0 - 1e-12)}


def limits_at_grip_scales(solver, scales, *, aero_Fz=None, aero_Fz_per_g=None,
                          iters: int = 24, lo: float = 0.3, hi: float = 4.0,
                          reuse: dict = None, progress=None, cancelled=None) -> list:
    """The car's limits at EACH grip scale in ``scales`` (the user's list,
    e.g. [0.7, 1.0]): per-corner lateral limit (all four tyres inside their
    budget), axle-aggregate lateral limit, grip-limited traction and braking.
    THE grip scale on the solver is set for each row and restored after — the
    same code path as the single-scale numbers, so they cannot disagree.
    reuse = {scale: (corner_limit, axle_limit)} already bisected (skipped)."""
    prev = solver._mu_scale
    rows = []
    try:
        for sc in scales:
            if cancelled is not None and cancelled():
                break
            sc = float(sc)
            solver._mu_scale = sc
            solver._warm = {}
            hit = None
            for k, v in (reuse or {}).items():
                if abs(float(k) - sc) < 1e-9:
                    hit = v
            if hit is not None:
                lc, la = hit
            else:
                if progress is not None:
                    progress(f'grip x{sc:.2f}: bisecting per-corner and axle limits')
                lc = per_corner_limit_g(solver, criterion='corner', lo=lo, hi=hi,
                                        iters=iters, aero_Fz=aero_Fz,
                                        aero_Fz_per_g=aero_Fz_per_g)
                la = per_corner_limit_g(solver, criterion='axle', lo=lo, hi=hi,
                                        iters=iters, aero_Fz=aero_Fz,
                                        aero_Fz_per_g=aero_Fz_per_g)
            acc = solver.max_accel_g(speed_kph=0.0)
            rows.append({'grip_scale': sc,
                         'corner_limit_g': float(lc['limit_g']),
                         'corner_binding': lc['binding'],
                         'corner_hit_upper_bound': bool(lc.get('hit_upper_bound', False)),
                         'axle_limit_g': float(la['limit_g']),
                         'axle_binding': la['binding'],
                         'axle_hit_upper_bound': bool(la.get('hit_upper_bound', False)),
                         'traction_g': float(acc['traction_g']),
                         'braking_g': float(acc['braking_g'])})
    finally:
        solver._mu_scale = prev
        solver._warm = {}
    return rows


def grip_budget_study(solver, *, lateral_g: float, aero_cases: dict,
                      sweep_g=None, iters: int = 24, lo: float = 0.3,
                      hi: float = 4.0, progress=None, cancelled=None,
                      grip_scales=None) -> dict:
    """The per-corner grip readout for the page: for every aero case the
    four-corner table at the chosen g, the per-corner limit (all four tyres
    inside their budget), the axle-aggregate limit for reference, and a
    utilization-vs-g sweep for the plot.

    aero_cases: ordered {label: None | {'aero_Fz': dict} | {'aero_Fz_per_g':
    dict}} — 'no aero' is whatever the caller labels it.  sweep_g: g values
    for the utilization curves (default 12 points from lo to the larger limit
    + 0.3 g).  progress(text) reports stages; cancelled() -> True stops.
    """
    out = {'lateral_g': float(lateral_g), 'cases': {}, 'mu_scale': float(solver._mu_scale)}
    labels = list(aero_cases.keys())
    for k, label in enumerate(labels):
        if cancelled is not None and cancelled():
            break
        spec = aero_cases[label] or {}
        kw = dict(aero_Fz=spec.get('aero_Fz'), aero_Fz_per_g=spec.get('aero_Fz_per_g'))
        if progress is not None:
            progress(f'{label}: four-corner table at {lateral_g:.2f} g ({k + 1}/{len(labels)})')
        try:
            at_g = per_corner_utilization(solver, lateral_g, **kw)
        except Exception as e:                   # noqa: BLE001
            at_g = {'error': f'{type(e).__name__}: {e}'[:120]}
        if progress is not None:
            progress(f'{label}: bisecting the per-corner limit ({iters} solves)')
        lim_c = per_corner_limit_g(solver, criterion='corner', lo=lo, hi=hi,
                                   iters=iters, **kw)
        if progress is not None:
            progress(f'{label}: bisecting the axle-aggregate limit ({iters} solves)')
        lim_a = per_corner_limit_g(solver, criterion='axle', lo=lo, hi=hi,
                                   iters=iters, **kw)
        gs = (list(sweep_g) if sweep_g is not None else
              list(np.linspace(lo, max([x for x in (lim_c['limit_g'], lim_a['limit_g'])
                                        if np.isfinite(x)] or [hi]) + 0.3, 12)))
        if progress is not None:
            progress(f'{label}: utilization vs g ({len(gs)} solves)')
        sweep = {'g': [], 'utilization': {c: [] for c in CORNERS},
                 'axle': {'F': [], 'R': []}}
        for g in gs:
            try:
                u = per_corner_utilization(solver, float(g), **kw)
            except Exception:                    # noqa: BLE001
                continue
            sweep['g'].append(float(g))
            for c in CORNERS:
                sweep['utilization'][c].append(u['corners'][c]['utilization'])
            for a in ('F', 'R'):
                sweep['axle'][a].append(u['axle_utilization'][a])
        by_scale = []
        if grip_scales:
            by_scale = limits_at_grip_scales(
                solver, grip_scales, iters=iters, lo=lo, hi=hi,
                reuse={float(solver._mu_scale): (lim_c, lim_a)},
                progress=(None if progress is None else
                          (lambda t, _l=label: progress(f'{_l}: {t}'))),
                cancelled=cancelled, **kw)
        out['cases'][label] = {'spec': spec, 'at_g': at_g, 'corner_limit': lim_c,
                               'axle_limit': lim_a, 'sweep': sweep,
                               'limits_by_grip_scale': by_scale}
    return out


def grip_budget_table_text(study: dict, sep: str = '\t') -> str:
    """The study's tables as copy-pasteable text: for every aero case the four
    corners at the chosen g and at the per-corner limit."""
    hdr = ['case', 'lateral_g', 'point', 'corner', 'Fz_N', 'inclination_deg',
           'budget_N', 'demand_N', 'utilization', 'below_tyre_data_floor']
    out = [sep.join(hdr)]
    for label, case in study['cases'].items():
        for point, tab in (('chosen g', case['at_g']),
                           ('per-corner limit', (case['corner_limit'] or {}).get('at_limit'))):
            if not tab or 'corners' not in tab:
                continue
            for c in CORNERS:
                d = tab['corners'][c]
                out.append(sep.join([
                    label, f'{tab["lateral_g"]:.3f}', point, c, f'{d["Fz_N"]:.0f}',
                    f'{d["inclination_deg"]:+.2f}', f'{d["budget_N"]:.0f}',
                    f'{d["demand_N"]:.0f}', f'{d["utilization"]:.3f}',
                    'yes' if d['below_data_floor'] else 'no']))
        lc, la = case['corner_limit'], case['axle_limit']
        out.append(sep.join([label, f'{lc["limit_g"]:.3f}', 'per-corner limit (all four inside budget)',
                             str(lc['binding']), '', '', '', '', '', '']))
        out.append(sep.join([label, f'{la["limit_g"]:.3f}', 'axle-aggregate limit (reference)',
                             str(la['binding']), '', '', '', '', '', '']))
    return '\n'.join(out)
