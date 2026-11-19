"""vahan/tire_compare.py — compare candidate tyres on the SAME car.

WHY.  A tyre choice has to be justified with data, not folklore: the tyre's
own measured curves (peak grip vs load, cornering stiffness, peak slip angle,
camber response) AND what the car does on it (grip limit, lap times).  Every
number here comes from vahan.tire_model.TireModel (the same fit the dynamics
use) and from the steady-state solver / lap sim bound to the loaded car, so
the comparison is the ONE model with one tyre swapped at a time.

DE-IDENTIFIED BY CONSTRUCTION.  Nothing in this module reads or emits a tyre
make, compound or part number: a candidate is a TireModel plus the LABEL the
caller gives it.  The repo is public; the TTC agreement allows derived numbers
with attribution, not identification.

Usage (headless):
    cands = [TireCandidate('16x7.5-10 (chosen)', TireModel.from_file(p, pressure_psi=12,
                           speed_window_kph=(38.2, 42.2))), ...]
    study = compare(win, cands)            # win = MainWindow with the car loaded
    print(comparison_table_text(study))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ymd import G

FZ_PTS_N = (300.0, 500.0, 700.0, 900.0, 1100.0)
FZ_REF_N = 700.0            # a loaded outer tyre on this car at ~1 g
LOAD_SENS_LO_N, LOAD_SENS_HI_N = 400.0, 1100.0


@dataclass
class TireCandidate:
    label: str                      # de-identified: size + a letter, never a make
    model: object                   # vahan.tire_model.TireModel
    chosen: bool = False
    rim_in: float | None = None     # test rim width, if the caller wants it shown
    note: str = ''
    extra: dict = field(default_factory=dict)


def _f(x):
    try:
        v = float(np.asarray(x, float).ravel()[0])
        return v if np.isfinite(v) else float('nan')
    except Exception:                            # noqa: BLE001
        return float('nan')


def characterize(model, fz_pts=FZ_PTS_N, ia_deg: float = 0.0,
                 sa_sweep_deg=None) -> dict:
    """The tyre's own measured behaviour at signed inclination ia_deg on the
    SA > 0 branch (SAE left-turn frame: a NEGATIVE IA is the wheel leaning
    into the turn).  Per load: peak mu, peak Fy, cornering stiffness, peak
    slip angle.  Plus the load-sensitivity ratio mu(1100)/mu(400), the camber
    gain at FZ_REF (mu at -2 deg IA over mu at 0), the aligning moment peak at
    FZ_REF and the tested ranges."""
    per_load = []
    for fz in fz_pts:
        fz = float(fz)
        pk = abs(_f(model.peak_Fy(fz, ia_deg, slip_sign=1)))
        ca = abs(_f(model.cornering_stiffness(fz, ia_deg)))
        sa = _f(model.peak_slip_angle(fz, ia_deg)) if hasattr(model, 'peak_slip_angle') else float('nan')
        per_load.append({'fz_N': fz, 'peak_fy_N': pk, 'mu': pk / fz if fz > 0 else float('nan'),
                         'c_alpha_N_per_deg': ca, 'c_alpha_per_fz_1_per_deg': ca / fz if fz > 0 else float('nan'),
                         'peak_slip_deg': sa})
    mu_lo = abs(_f(model.peak_Fy(LOAD_SENS_LO_N, ia_deg, slip_sign=1))) / LOAD_SENS_LO_N
    mu_hi = abs(_f(model.peak_Fy(LOAD_SENS_HI_N, ia_deg, slip_sign=1))) / LOAD_SENS_HI_N
    mu_ref0 = abs(_f(model.peak_Fy(FZ_REF_N, 0.0, slip_sign=1))) / FZ_REF_N
    mu_ref_m2 = abs(_f(model.peak_Fy(FZ_REF_N, -2.0, slip_sign=1))) / FZ_REF_N
    mu_ref_p2 = abs(_f(model.peak_Fy(FZ_REF_N, 2.0, slip_sign=1))) / FZ_REF_N
    # Fy and Mz curves at FZ_REF for the overlay figure
    sa_axis = np.asarray(sa_sweep_deg if sa_sweep_deg is not None else np.linspace(0.0, 12.0, 49), float)
    fy_curve = [abs(_f(model.Fy(float(a), FZ_REF_N, ia_deg))) for a in sa_axis]
    mz_curve = [_f(model.Mz(float(a), FZ_REF_N, ia_deg)) for a in sa_axis] if hasattr(model, 'Mz') else []
    mz_peak = float(np.nanmax(np.abs(mz_curve))) if mz_curve else float('nan')
    try:
        fz_lo, fz_hi = (float(v) for v in model.fz_range)
    except Exception:                            # noqa: BLE001
        fz_lo = fz_hi = float('nan')
    try:
        sa_lo, sa_hi = (float(v) for v in model.sa_range)
    except Exception:                            # noqa: BLE001
        sa_lo = sa_hi = float('nan')
    fg = forgiveness_from_curve(sa_axis, fy_curve)
    wear = wear_proxies_from_curve(sa_axis, fy_curve, FZ_REF_N)
    return {'ia_deg': float(ia_deg), 'per_load': per_load, 'forgiveness': fg, 'wear': wear,
            'mu_ref_N': FZ_REF_N, 'mu_ref': mu_ref0,
            'load_sensitivity_ratio': mu_hi / mu_lo if mu_lo > 0 else float('nan'),
            'load_sens_lo_N': LOAD_SENS_LO_N, 'load_sens_hi_N': LOAD_SENS_HI_N,
            'camber_gain_m2': mu_ref_m2 / mu_ref0 if mu_ref0 > 0 else float('nan'),
            'camber_loss_p2': mu_ref_p2 / mu_ref0 if mu_ref0 > 0 else float('nan'),
            'mz_peak_Nm_ref': mz_peak,
            'sa_axis_deg': [float(a) for a in sa_axis], 'fy_curve_N': fy_curve, 'mz_curve_Nm': mz_curve,
            'fz_tested_N': (fz_lo, fz_hi), 'sa_tested_deg': (sa_lo, sa_hi),
            'pressure_psi': _f(getattr(model, 'pressure_psi', float('nan'))),
            'mf_fit_r2': _f(getattr(model, 'mf_r2', float('nan')))}


def car_metrics(win, model, *, tracks=None, cla_m2: float = 0.0, cop_rear_frac: float = 0.5,
                aero_per_g=None, turn_radius_m: float = 10.0, gears=None, final_drive: float = 3.545,
                redline_rpm: float = 11000.0, n_detail: int = 20, progress=None) -> dict:
    """What the LOADED car does on this tyre, with the project grip derate:
    mechanical axle limit, the G-G lateral limit with the aero package scaled
    on turn_radius_m (Vahan's G-G convention) and lap times on `tracks`
    ([(name, Track), ...]) with the package.  The car's geometry, rates, mass,
    tyre radius and gearing are held — only the grip surface changes — so the
    delta is the tyre, not a repackaged car.  win._tire_model is restored."""
    from .corner_speed import per_corner_limit_g
    from .laptime import LapSimulator
    keep_f, keep_r = getattr(win, '_tire_model', None), getattr(win, '_tire_model_rear', None)
    out = {}
    try:
        win._tire_model = model
        win._tire_model_rear = None
        ss = win._build_dynamics_solver()
        out['mu_scale'] = float(ss._mu_scale)
        lim = per_corner_limit_g(ss, criterion='axle')
        out['mech_limit_g'] = float(lim['limit_g'])
        out['mech_limit_binding'] = str(lim.get('binding', lim.get('binding_corner', '')))
        # how hard the tyre works ON THIS CAR: mean |slip angle| per axle at 1 g and at the limit
        slips = {}
        for tag, g in (('1g', 1.0), ('limit', max(0.98 * out['mech_limit_g'], 0.2))):
            try:
                r = ss.solve(float(g), 0.0)
                sa = {k: abs(float(v)) for k, v in (r.slip_angle or {}).items()}
                slips[tag] = {'g': float(g), 'front_deg': float(np.mean([sa[k] for k in sa if k[0] == 'F'])) if sa else float('nan'),
                              'rear_deg': float(np.mean([sa[k] for k in sa if k[0] == 'R'])) if sa else float('nan'),
                              'outer_front_deg': float(sa.get('FL', sa.get('FR', float('nan')))) if sa else float('nan')}
            except Exception:                    # noqa: BLE001
                slips[tag] = {'g': float(g), 'front_deg': float('nan'), 'rear_deg': float('nan'), 'outer_front_deg': float('nan')}
        out['operating_slip'] = slips
        if aero_per_g:
            gg = ss.gg_envelope(aero_Fz_per_g=aero_per_g, turn_radius_m=float(turn_radius_m), n_ay=5, iters=7)
            out['gg_limit_aero_g'] = float(gg['ay_limit_g'])
            out['gg_radius_m'] = float(turn_radius_m)
        laps = {}
        for name, trk in (tracks or []):
            if progress:
                progress(f'lap {name}')
            sim = LapSimulator(ss, cla_m2=float(cla_m2), grip_scale=float(ss._mu_scale),
                               aero_cop_rear_frac=float(cop_rear_frac))
            if gears:
                sim.set_gearbox(list(gears), 1.0, float(final_drive), float(redline_rpm))
            r = sim.simulate(trk, n_detail=int(n_detail))
            laps[name] = {'lap_s': float(r.lap_time_s), 'peak_lat_g': float(r.peak_lat_g),
                          'avg_speed_kph': float(r.avg_speed_kph)}
        out['laps'] = laps
    finally:
        win._tire_model, win._tire_model_rear = keep_f, keep_r
    return out


def compare(win, candidates, *, ia_deg: float = 0.0, tracks=None, cla_m2: float = 0.0,
            cop_rear_frac: float = 0.5, aero_per_g=None, turn_radius_m: float = 10.0,
            gears=None, final_drive: float = 3.545, redline_rpm: float = 11000.0,
            n_detail: int = 20, with_car: bool = True, progress=None) -> dict:
    """Characterise every candidate and (with_car) run the loaded car on each.
    Returns {'candidates': [{label, chosen, rim_in, note, tyre: characterize(),
    car: car_metrics()}], 'ranking': {...}, 'chosen_label': ...}."""
    rows = []
    for i, c in enumerate(candidates):
        if progress:
            progress(f'{i + 1} / {len(candidates)}: {c.label}')
        row = {'label': c.label, 'chosen': bool(c.chosen), 'rim_in': c.rim_in, 'note': c.note,
               'tyre': characterize(c.model, ia_deg=ia_deg)}
        if with_car:
            row['car'] = car_metrics(win, c.model, tracks=tracks, cla_m2=cla_m2, cop_rear_frac=cop_rear_frac,
                                     aero_per_g=aero_per_g, turn_radius_m=turn_radius_m, gears=gears,
                                     final_drive=final_drive, redline_rpm=redline_rpm, n_detail=n_detail,
                                     progress=progress)
        rows.append(row)
    chosen = next((r for r in rows if r['chosen']), None)
    rank = {}
    if rows:
        rank['mu_ref'] = sorted(rows, key=lambda r: -r['tyre']['mu_ref'])
        rank['c_alpha_ref'] = sorted(rows, key=lambda r: -_c_alpha_at(r['tyre'], FZ_REF_N))
        rank['load_sensitivity'] = sorted(rows, key=lambda r: -r['tyre']['load_sensitivity_ratio'])
        if with_car and all('car' in r for r in rows):
            rank['mech_limit_g'] = sorted(rows, key=lambda r: -r['car']['mech_limit_g'])
            for tn in (tracks or []):
                rank[f'lap:{tn[0]}'] = sorted(rows, key=lambda r: r['car']['laps'].get(tn[0], {}).get('lap_s', float('inf')))
    return {'candidates': rows, 'ranking': {k: [r['label'] for r in v] for k, v in rank.items()},
            'chosen_label': chosen['label'] if chosen else None, 'ia_deg': float(ia_deg),
            'fz_ref_N': FZ_REF_N, 'with_car': bool(with_car)}


def _c_alpha_at(t: dict, fz: float) -> float:
    for p in t['per_load']:
        if abs(p['fz_N'] - fz) < 1e-6:
            return p['c_alpha_N_per_deg']
    return float('nan')


def measured_peak_curve(ttc, *, pressure_psi: float, speed_window_kph=None, warmup_pts: int = 1500,
                        ia_level: float = 0.0, fz_bins=((200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1250)),
                        sa_bin_deg: float = 0.5, min_samples: int = 50, branch: int = +1) -> list:
    """Peak grip per load bin read DIRECTLY off the TTC samples: within each
    load bin, bin the chosen slip branch by sa_bin_deg, take mean|Fy| / mean Fz
    per slip bin, and report the highest.  No grid, no fit, no clamp — so a
    tyre whose peak sits past the rig's sweep end is reported at the sweep end
    and flagged censored (the true peak is higher).  branch=+1 is SA > 0.
    Returns [{fz_N, mu_peak, sa_peak_deg, censored, n, mu_at_end, sa_end_deg}]."""
    p = np.asarray(ttc.pressure_kPa, float) * 0.145038
    v = np.asarray(ttc.velocity_kph, float)
    fz = np.asarray(ttc.normal_load_N, float); sa = np.asarray(ttc.slip_angle_deg, float)
    fy = np.asarray(ttc.lateral_force_N, float); ia = np.asarray(ttc.camber_deg, float)
    keep = np.arange(len(sa)) >= int(warmup_pts)
    keep &= np.abs(p - float(pressure_psi)) < 0.6
    if speed_window_kph is not None:
        keep &= (v >= float(speed_window_kph[0])) & (v <= float(speed_window_kph[1]))
    keep &= np.abs(ia - float(ia_level)) < 0.5
    keep &= (sa > 0.0) if branch > 0 else (sa < 0.0)
    out = []
    for lo, hi in fz_bins:
        m = keep & (fz >= lo) & (fz < hi)
        if m.sum() < min_samples:
            out.append({'fz_N': 0.5 * (lo + hi), 'mu_peak': float('nan'), 'sa_peak_deg': float('nan'),
                        'censored': True, 'n': int(m.sum()), 'mu_at_end': float('nan'), 'sa_end_deg': float('nan')})
            continue
        s = np.abs(sa[m]); y = np.abs(fy[m]); z = fz[m]
        s_max = float(np.nanmax(s))
        edges = np.arange(0.0, s_max + sa_bin_deg, sa_bin_deg)
        mus, cs = [], []
        for e in edges[:-1]:
            k = (s >= e) & (s < e + sa_bin_deg)
            if k.sum() >= 5:
                mus.append(float(np.mean(y[k]) / np.mean(z[k]))); cs.append(e + 0.5 * sa_bin_deg)
        if not mus:
            out.append({'fz_N': float(np.mean(z)), 'mu_peak': float('nan'), 'sa_peak_deg': float('nan'),
                        'censored': True, 'n': int(m.sum()), 'mu_at_end': float('nan'), 'sa_end_deg': s_max})
            continue
        i = int(np.argmax(mus))
        out.append({'fz_N': float(np.mean(z)), 'mu_peak': mus[i], 'sa_peak_deg': cs[i],
                    'censored': bool(cs[i] >= cs[-1] - sa_bin_deg - 1e-9), 'n': int(m.sum()),
                    'mu_at_end': mus[-1], 'sa_end_deg': cs[-1]})
    return out


def measured_camber_response(ttc, *, pressure_psi: float, speed_window_kph=None, warmup_pts: int = 1500,
                             fz_bin=(600, 800), ia_level: float = 2.0) -> dict:
    """Peak grip at |IA| = ia_level on BOTH slip branches over the peak at IA 0,
    in the load bin: 'into' = the branch the camber helps (wheel leaning into
    the turn), 'out' = the branch it hurts.  Straight off the samples."""
    base = measured_peak_curve(ttc, pressure_psi=pressure_psi, speed_window_kph=speed_window_kph,
                               warmup_pts=warmup_pts, ia_level=0.0, fz_bins=(fz_bin,))[0]['mu_peak']
    pos = measured_peak_curve(ttc, pressure_psi=pressure_psi, speed_window_kph=speed_window_kph,
                              warmup_pts=warmup_pts, ia_level=ia_level, fz_bins=(fz_bin,), branch=+1)[0]['mu_peak']
    neg = measured_peak_curve(ttc, pressure_psi=pressure_psi, speed_window_kph=speed_window_kph,
                              warmup_pts=warmup_pts, ia_level=ia_level, fz_bins=(fz_bin,), branch=-1)[0]['mu_peak']
    hi, lo = (max(pos, neg), min(pos, neg)) if np.isfinite(pos) and np.isfinite(neg) else (float('nan'), float('nan'))
    return {'ia_deg': float(ia_level), 'fz_bin': list(fz_bin), 'mu_ia0': base,
            'into_ratio': hi / base if base and np.isfinite(hi) else float('nan'),
            'out_ratio': lo / base if base and np.isfinite(lo) else float('nan')}


def forgiveness_from_curve(sa_axis_deg, fy_curve_N, past_deg: float = 3.0) -> dict:
    """How much force the tyre keeps PAST its peak: Fy at (peak slip + past_deg)
    over the peak, on the measured sweep (clamped to its end).  1.0 = flat
    plateau (forgiving), 0.8 = drops 20 % when the driver overshoots."""
    sa = np.asarray(sa_axis_deg, float); fy = np.asarray(fy_curve_N, float)
    if sa.size < 3 or not np.all(np.isfinite(fy)):
        return {'peak_slip_deg': float('nan'), 'fy_peak_N': float('nan'), 'ratio_past_peak': float('nan'),
                'past_deg': past_deg, 'reached_sweep_end': True}
    i = int(np.nanargmax(fy)); sa_pk = float(sa[i]); fy_pk = float(fy[i])
    # the rig sweeps to ~12 deg; "past the peak" is judged at the END of the
    # measured sweep so every tyre is scored on measured points.  A tyre whose
    # peak sits at the sweep end scores 1.0 by construction — flagged.
    end = float(sa[-1]); fy_end = float(fy[-1])
    return {'peak_slip_deg': sa_pk, 'fy_peak_N': fy_pk, 'ratio_past_peak': fy_end / fy_pk if fy_pk > 0 else float('nan'),
            'past_deg': end - sa_pk, 'sweep_end_deg': end, 'peak_at_sweep_end': bool(sa_pk >= end - 0.26)}


def wear_proxies_from_curve(sa_axis_deg, fy_curve_N, fz_N: float, speed_mps: float = 11.1) -> dict:
    """The rig measures NO wear.  These are the proxies that scale with rubber
    abuse at a given grip demand, all from the measured curve at fz_N:
      peak_slip_deg       - slip the tyre needs to make its peak (more slip = more sliding)
      slide_speed_mps     - sliding speed at the peak, V·sin(SA_peak), at speed_mps (40 km/h)
      friction_power_W    - Fy_peak × sliding speed: heat into the tread at the peak
      slip_for_0p8_peak   - slip to reach 80 % of the peak (the everyday operating point)
      mu_per_deg          - peak mu per degree of slip to get there (higher = less sliding for the same grip)
    Lower slip / lower power for the same grip = the tyre works less hard."""
    sa = np.asarray(sa_axis_deg, float); fy = np.asarray(fy_curve_N, float)
    out = {'speed_mps': float(speed_mps), 'fz_N': float(fz_N)}
    if sa.size < 3 or not np.all(np.isfinite(fy)) or fz_N <= 0:
        return {**out, 'peak_slip_deg': float('nan'), 'slide_speed_mps': float('nan'), 'friction_power_W': float('nan'),
                'slip_for_0p8_peak_deg': float('nan'), 'mu_per_deg': float('nan')}
    i = int(np.nanargmax(fy)); sa_pk = float(sa[i]); fy_pk = float(fy[i])
    vs = float(speed_mps) * math.sin(math.radians(sa_pk))
    j = int(np.argmax(fy >= 0.8 * fy_pk))
    sa80 = float(np.interp(0.8 * fy_pk, fy[:j + 1], sa[:j + 1])) if j > 0 else float(sa[0])
    return {**out, 'peak_slip_deg': sa_pk, 'slide_speed_mps': vs, 'friction_power_W': fy_pk * vs,
            'slip_for_0p8_peak_deg': sa80, 'mu_per_deg': (fy_pk / fz_N) / sa_pk if sa_pk > 0 else float('nan')}


def condition_sensitivity(points, fz_N: float = FZ_REF_N, ia_deg: float = 0.0) -> dict:
    """mu at fz_N across test conditions: points = [(value, model), ...] (e.g.
    pressure in psi or test speed in km/h).  Returns the values, mu at each,
    the linear slope per unit and the spread as a fraction of the mid value."""
    vals, mus = [], []
    for v, m in points:
        mu = abs(_f(m.peak_Fy(fz_N, ia_deg, slip_sign=1))) / fz_N
        if np.isfinite(mu):
            vals.append(float(v)); mus.append(float(mu))
    # a condition block with too little data fits to nonsense (mu ~ 0.03 at a
    # pressure ramp): drop any point under half the best one, and say so
    dropped = []
    if mus:
        top = max(mus)
        keep = [i for i, m in enumerate(mus) if m >= 0.5 * top]
        dropped = [vals[i] for i in range(len(vals)) if i not in keep]
        vals = [vals[i] for i in keep]; mus = [mus[i] for i in keep]
    if len(vals) < 2:
        return {'values': vals, 'mu': mus, 'slope_per_unit': float('nan'), 'spread_frac': float('nan'), 'dropped': dropped}
    slope = float(np.polyfit(vals, mus, 1)[0])
    mid = float(np.mean(mus))
    return {'values': vals, 'mu': mus, 'slope_per_unit': slope,
            'spread_frac': (max(mus) - min(mus)) / mid if mid > 0 else float('nan'), 'dropped': dropped}


def comparison_rows(study: dict) -> list:
    """Flat rows for a table: one per candidate."""
    out = []
    for r in study['candidates']:
        t = r['tyre']; c = r.get('car', {})
        laps = c.get('laps', {})
        out.append({'label': r['label'], 'chosen': r['chosen'],
                    'mu_ref': t['mu_ref'], 'c_alpha_ref': _c_alpha_at(t, study['fz_ref_N']),
                    'peak_slip_ref': next((p['peak_slip_deg'] for p in t['per_load'] if abs(p['fz_N'] - study['fz_ref_N']) < 1e-6), float('nan')),
                    'load_sensitivity_ratio': t['load_sensitivity_ratio'],
                    'camber_gain_m2': t['camber_gain_m2'], 'mz_peak_Nm_ref': t['mz_peak_Nm_ref'],
                    'fz_tested_N': t['fz_tested_N'], 'pressure_psi': t['pressure_psi'],
                    'forgiveness': t['forgiveness']['ratio_past_peak'], 'wear': t['wear'],
                    'operating_slip': c.get('operating_slip', {}),
                    'pressure_sens': r.get('pressure_sens'), 'speed_sens': r.get('speed_sens'),
                    'mech_limit_g': c.get('mech_limit_g', float('nan')),
                    'gg_limit_aero_g': c.get('gg_limit_aero_g', float('nan')),
                    'laps': {k: v['lap_s'] for k, v in laps.items()}})
    return out


def comparison_table_text(study: dict, sep: str = '\t') -> str:
    rows = comparison_rows(study)
    track_names = sorted({k for r in rows for k in r['laps']})
    head = ['tyre', 'chosen', f'mu at {study["fz_ref_N"]:.0f} N', 'C_alpha N/deg', 'peak slip deg',
            'mu(1100)/mu(400)', 'mu gain at -2 deg IA', 'Mz peak N.m', 'tested Fz N', 'psi',
            'mech limit g', 'G-G limit with aero g'] + [f'lap {t} s' for t in track_names]
    lines = [sep.join(head)]
    for r in rows:
        lines.append(sep.join([r['label'], 'yes' if r['chosen'] else '', f"{r['mu_ref']:.3f}", f"{r['c_alpha_ref']:.0f}",
                               f"{r['peak_slip_ref']:.1f}", f"{r['load_sensitivity_ratio']:.3f}", f"{r['camber_gain_m2']:.3f}",
                               f"{r['mz_peak_Nm_ref']:.1f}", f"{r['fz_tested_N'][0]:.0f}-{r['fz_tested_N'][1]:.0f}",
                               f"{r['pressure_psi']:.0f}", f"{r['mech_limit_g']:.3f}", f"{r['gg_limit_aero_g']:.3f}"]
                              + [f"{r['laps'].get(t, float('nan')):.2f}" for t in track_names]))
    return '\n'.join(lines)
