"""CG build tolerance (user 2026-09-26: "irl [the CG] won't be [what it is] in
sim when we build the car. I want to see what our CG tolerance is, height and
front to rear").

The CG of the built car is measured on scales / by tilting, not known from CAD,
so the question is: how far can the real CG sit from the design value before
the car stops behaving as designed?  This module answers it from the ONE model:
the caller rebuilds the steady-state solver (and optionally the lap simulator)
with the CG moved, and this module measures the car's DIRECT metrics and finds,
for each one, the CG band inside which it stays within an allowance.

Metrics (all from the solver the app itself builds):
  grip_limit_g        first lateral g at which any tyre is out of grip
                      (corner_speed.per_corner_limit_g)
  front_share_pct     front share of the lateral load transfer at 1 g
  roll_deg_per_g      body roll at 1 g
  understeer_deg      front minus rear slip angle at 1 g (+ = understeer)
  traction_g          grip-limited forward acceleration at 20 km/h
  braking_g           grip-limited braking at 60 km/h
  lap_time_s          lap simulator on the Lap Time page's track (optional)

Allowances are ENGINEERING CHOICES, exposed as inputs with these defaults
(changing them changes the tolerance, not the physics)."""
from __future__ import annotations

import numpy as np

METRICS = (
    # key, label, unit, default allowance, direction ('both' = +-, 'down' = only a loss counts,
    # 'up' = only a rise counts), sets the headline tolerance?
    ('grip_limit_g', 'Grip limit (first tyre out of grip)', 'g', 0.01, 'down', True),
    ('front_share_pct', 'Front share of lateral load transfer', '%', 1.0, 'both', True),
    ('roll_deg_per_g', 'Roll at 1 g', 'deg', 0.05, 'both', True),
    ('understeer_deg', 'Understeer at 1 g (front - rear slip)', 'deg', 0.10, 'both', True),
    # traction / braking trade against each other with CG (a LOWER CG loses rear traction
    # but gains cornering); the lap time already carries both, so they are shown, not binding
    ('traction_g', 'Traction limit, 20 km/h', 'g', 0.01, 'down', False),
    ('braking_g', 'Braking limit, 60 km/h', 'g', 0.01, 'down', False),
    ('lap_time_s', 'Lap time', 's', 0.10, 'up', True),
)
METRIC_KEYS = tuple(m[0] for m in METRICS)
DEFAULT_ALLOWANCE = {m[0]: m[3] for m in METRICS}
DIRECTION = {m[0]: m[4] for m in METRICS}
HEADLINE = {m[0]: m[5] for m in METRICS}


def car_metrics(ss, sim=None, track=None, n_detail: int = 400) -> dict:
    """Direct metrics of the car represented by steady-state solver ss (and the
    lap simulator sim on track, if given)."""
    from .corner_speed import per_corner_limit_g
    out = {}
    r1 = ss.solve(1.0, 0.0)
    tf = r1.geometric_lt_front_N + r1.elastic_lt_front_N + r1.unsprung_lt_front_N
    tr = r1.geometric_lt_rear_N + r1.elastic_lt_rear_N + r1.unsprung_lt_rear_N
    out['front_share_pct'] = 100.0 * tf / (tf + tr) if (tf + tr) else float('nan')
    out['roll_deg_per_g'] = float(r1.roll_angle_deg)
    out['understeer_deg'] = float(getattr(r1, 'understeer_gradient_deg', float('nan')))
    out['grip_limit_g'] = float(per_corner_limit_g(ss)['limit_g'])
    try:
        out['traction_g'] = float(ss.max_accel_g(20.0, 0.0)['traction_g'])
        out['braking_g'] = float(ss.max_accel_g(60.0, 0.0)['braking_g'])
    except Exception:
        out['traction_g'] = out['braking_g'] = float('nan')
    if sim is not None and track is not None:
        try:
            out['lap_time_s'] = float(sim.simulate(track, n_detail=n_detail).lap_time_s)
        except Exception:
            out['lap_time_s'] = float('nan')
    else:
        out['lap_time_s'] = float('nan')
    return out


def band(xs, ys, x0: float, y0: float, allowance: float, direction: str = 'both') -> tuple:
    """(lo, hi) of x around x0 inside which y stays within `allowance` of y0,
    linearly interpolated between sweep points.  direction 'down': only drops
    below y0 count; 'up': only rises count.  -inf / +inf = still inside at the
    end of the sweep (tolerance wider than the swept range)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    o = np.argsort(xs); xs, ys = xs[o], ys[o]

    def excess(y):
        d = y - y0
        if direction == 'down':
            d = -d
        elif direction == 'both':
            d = abs(d)
        return d - allowance          # > 0 = outside the allowance

    e = np.array([excess(y) if np.isfinite(y) else np.inf for y in ys])
    i0 = int(np.argmin(abs(xs - x0)))
    hi = np.inf
    for i in range(i0, len(xs) - 1):
        if e[i + 1] > 0:
            hi = xs[i] + (xs[i + 1] - xs[i]) * (0 - e[i]) / (e[i + 1] - e[i]) if np.isfinite(e[i + 1]) and e[i + 1] != e[i] else xs[i]
            break
    lo = -np.inf
    for i in range(i0, 0, -1):
        if e[i - 1] > 0:
            lo = xs[i] - (xs[i] - xs[i - 1]) * (0 - e[i]) / (e[i - 1] - e[i]) if np.isfinite(e[i - 1]) and e[i - 1] != e[i] else xs[i]
            break
    return float(lo), float(hi)


def tolerance_table(sweep: dict, allowance: dict | None = None) -> list:
    """sweep = {'axis': 'height'|'fore_aft', 'x_mm': [...], 'x0_mm': design,
    'metrics': [dict per x]}.  Returns one row per metric: design value,
    slope per 10 mm (central, from the two points nearest the design), band
    (lo, hi) in mm of CG, and the binding side."""
    allowance = {**DEFAULT_ALLOWANCE, **(allowance or {})}
    xs = np.asarray(sweep['x_mm'], float); x0 = float(sweep['x0_mm'])
    i0 = int(np.argmin(abs(xs - x0)))
    rows = []
    for key, label, unit, _a, _d, _h in METRICS:
        ys = np.array([m.get(key, np.nan) for m in sweep['metrics']], float)
        if not np.isfinite(ys[i0]):
            rows.append(dict(key=key, label=label, unit=unit, design=float('nan'), per_10mm=float('nan'),
                             lo=float('nan'), hi=float('nan'), allowance=allowance[key], direction=DIRECTION[key],
                             headline=HEADLINE[key]))
            continue
        j = [k for k in (i0 - 1, i0 + 1) if 0 <= k < len(xs) and np.isfinite(ys[k])]
        slope = (ys[j[-1]] - ys[j[0]]) / (xs[j[-1]] - xs[j[0]]) * 10.0 if len(j) == 2 else float('nan')
        lo, hi = band(xs, ys, x0, ys[i0], allowance[key], DIRECTION[key])
        rows.append(dict(key=key, label=label, unit=unit, design=float(ys[i0]), per_10mm=float(slope),
                         lo=lo - x0, hi=hi - x0, allowance=allowance[key], direction=DIRECTION[key],
                         headline=HEADLINE[key]))
    return rows


def overall_band(rows) -> tuple:
    """Tightest (lo, hi) over the HEADLINE metric rows, and which metric sets each side."""
    lo, hi, klo, khi = -np.inf, np.inf, '', ''
    for r in rows:
        if not r.get('headline', True):
            continue
        if np.isfinite(r['lo']) and r['lo'] > lo:
            lo, klo = r['lo'], r['label']
        if np.isfinite(r['hi']) and r['hi'] < hi:
            hi, khi = r['hi'], r['label']
    return lo, hi, klo, khi
