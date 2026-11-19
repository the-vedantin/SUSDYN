"""vahan/ackermann_target.py — Ackermann from the lap sims (user method, 2026-10-10).

    1. run the QSS lap sim on the chosen tracks WITH the aero package,
    2. isolate the corner stations (lateral g > 0.30, the sim's own rule),
    3. time-weighted average lateral g and path radius over every corner of
       every lap,
    4. the Ackermann % that puts both front tyres at the same fraction of
       their peak at THAT radius and g (vahan.analysis_plots demand, aero
       per g at that radius),
    5. rotate the steering arm about the kingpin until the linkage gives
       that % at the steer the condition needs, re-level the arm so bump
       steer stays nulled.

Everything reads the ONE solved model: SteadyStateSolver loads, the tyre
model, the probed SteeringGeometry.  No GUI imports here — the GUI hands in
the two probes it owns (steering geometry from a hardpoint dict, bump steer
from a hardpoint dict) as callables.
"""
from __future__ import annotations
import math
import numpy as np

CORNER_LAT_G = 0.30           # the lap sim's own corner threshold (laptime.simulate)


def corner_condition_from_laps(ss_solver, track_files, *, cla_m2: float, cop_rear: float,
                               grip_scale: float, gears, final_drive: float = 3.545,
                               primary: float = 1.0, redline_rpm: float = 11000.0,
                               n_detail: int = 40) -> dict:
    """Run every track, pool the corner stations, return the representative
    cornering condition (time-weighted) plus per-track stats."""
    from .laptime import Track, LapSimulator
    tracks, g_num, r_num, t_sum, lap_total = [], 0.0, 0.0, 0.0, 0.0
    all_g, all_r, all_dt = [], [], []
    for tfile in track_files:
        trk = Track.from_json(tfile)
        sim = LapSimulator(ss_solver, cla_m2=float(cla_m2), grip_scale=float(grip_scale),
                           aero_cop_rear_frac=float(cop_rear))
        if gears:
            sim.set_gearbox(list(gears), float(primary), float(final_drive), float(redline_rpm))
        res = sim.simulate(trk, n_detail=int(n_detail))
        v, lat, s = np.asarray(res.v_ms, float), np.asarray(res.lat_g, float), np.asarray(res.s_m, float)
        kap = np.asarray(trk.kappa_1pm, float)
        if len(kap) != len(v):
            kap = np.interp(s, np.linspace(s[0], s[-1], len(kap)), kap)
        ds = np.diff(s); vm = 0.5 * (v[1:] + v[:-1]); dt = ds / np.maximum(vm, 0.1)
        latm = 0.5 * (lat[1:] + lat[:-1]); kapm = 0.5 * (kap[1:] + kap[:-1])
        radm = 1.0 / np.maximum(np.abs(kapm), 1e-6)
        c = latm > CORNER_LAT_G
        tc = float(np.sum(dt[c]))
        tracks.append(dict(track=str(tfile), lap_time_s=float(res.lap_time_s),
                           corner_time_s=tc, time_cornering_pct=float(100.0 * tc / res.lap_time_s),
                           g_avg=(float(np.sum(latm[c] * dt[c]) / tc) if tc > 0 else float('nan')),
                           R_avg_m=(float(np.sum(radm[c] * dt[c]) / tc) if tc > 0 else float('nan')),
                           peak_lat_g=float(np.max(lat))))
        g_num += float(np.sum(latm[c] * dt[c])); r_num += float(np.sum(radm[c] * dt[c])); t_sum += tc
        lap_total += float(res.lap_time_s)
        all_g.append(latm[c]); all_r.append(radm[c]); all_dt.append(dt[c])
        tracks[-1]['stations'] = dict(g=latm[c].tolist(), R_m=radm[c].tolist(), dt_s=dt[c].tolist(), s_m=(0.5 * (s[1:] + s[:-1]))[c].tolist())
    if t_sum <= 0:
        raise RuntimeError('no corner stations above 0.30 g on the chosen tracks')
    g_all = np.concatenate(all_g); r_all = np.concatenate(all_r); dt_all = np.concatenate(all_dt)
    order = np.argsort(r_all); cum = np.cumsum(dt_all[order]); r_med = float(r_all[order][np.searchsorted(cum, 0.5 * cum[-1])])
    return dict(tracks=tracks, g_avg=g_num / t_sum, R_avg_m=r_num / t_sum, R_median_m=r_med,
                R_harmonic_m=float(1.0 / (np.sum(dt_all / r_all) / np.sum(dt_all))),
                corner_time_s=t_sum, lap_time_total_s=lap_total, time_cornering_pct=100.0 * t_sum / lap_total,
                stations=dict(g=g_all.tolist(), R_m=r_all.tolist(), dt_s=dt_all.tolist()))


def pooled_demand_from_laps(tire_model, ss_solver, steer_geo, cond_stations: dict, *, grip_multiplier=None,
                            aero_per_g_fn=None, decimate: int = 3, radius_bin_m: float = 2.0, g_bin: float = 0.1) -> dict:
    """The lap-pooled Ackermann demand.  Evaluating the demand at the AVERAGE
    corner (mean g, mean radius) is meaningless on these tracks: the mean radius
    is 35-50 m where the kinematic spread is tiny and the % blows up.  So every
    corner station gets its own demand % (equal use of both front tyres at its
    own radius and g, aero per g at that radius), and the pool is averaged with
    weight dt x kinematic toe spread — the % that minimises the time-weighted
    TOE-ANGLE error over the laps (the toe difference is what the arm is built
    to).  Also returns the plain time-weighted mean and the demand by radius
    band, so the answer can be argued with.
    cond_stations: {'g': [...], 'R_m': [...], 'dt_s': [...]} corner stations.
    aero_per_g_fn(R) -> per-corner N-per-g dict (or None)."""
    from .analysis_plots import ackermann_pct_vs_g_data
    veh = ss_solver._veh; L = float(veh.wheelbase_m); t = float(veh.front_track_m)
    g = np.asarray(cond_stations['g'], float)[::decimate]; R = np.asarray(cond_stations['R_m'], float)[::decimate]
    dt = np.asarray(cond_stations['dt_s'], float)[::decimate] * decimate
    cache, pct, ms = {}, np.full(len(g), np.nan), np.full(len(g), np.nan)
    for i in range(len(g)):
        Rb = float(np.clip(round(R[i] / radius_bin_m) * radius_bin_m, 4.0, 200.0)); gb = round(float(g[i]) / g_bin) * g_bin
        key = (Rb, gb)
        if key not in cache:
            pg = aero_per_g_fn(Rb) if aero_per_g_fn else None
            d = ackermann_pct_vs_g_data(tire_model, ss_solver, steer_geo, radii_m=(Rb,), lat_g_list=(gb,),
                                        grip_multiplier=grip_multiplier, aero_Fz_per_g_by_radius=({Rb: pg} if pg else None))
            b = d['by_radius'][Rb]
            cache[key] = ((float(b['demand_pct'][0]), float(b['mean_steer_deg'][0])) if len(b['lat_g']) else (float('nan'), float('nan')))
        pct[i], ms[i] = cache[key]
    spread = np.degrees(np.arctan(L / np.maximum(R - t / 2.0, 0.05)) - np.arctan(L / (R + t / 2.0)))
    ok = np.isfinite(pct)
    if ok.sum() < 3:
        raise RuntimeError('demand could not be solved on enough corner stations')
    w_toe = dt[ok] * spread[ok]
    out = dict(n_stations=int(len(g)), n_solved=int(ok.sum()),
               pct_time_weighted=float(np.sum(pct[ok] * dt[ok]) / np.sum(dt[ok])),
               pct_toe_weighted=float(np.sum(pct[ok] * w_toe) / np.sum(w_toe)),
               mean_steer_toe_weighted_deg=float(np.sum(ms[ok] * w_toe) / np.sum(w_toe)),
               toe_difference_wanted_deg=float(np.sum(pct[ok] / 100.0 * spread[ok] * dt[ok]) / np.sum(dt[ok])),
               bands=[])
    for lo, hi in ((0, 10), (10, 15), (15, 25), (25, 50), (50, 1e9)):
        m = ok & (R >= lo) & (R < hi)
        if m.any():
            out['bands'].append(dict(R_lo=lo, R_hi=(hi if hi < 1e8 else None), time_pct=float(100 * dt[m].sum() / dt[ok].sum()),
                                     g_mean=float(np.sum(g[m] * dt[m]) / dt[m].sum()), demand_pct=float(np.sum(pct[m] * dt[m]) / dt[m].sum()),
                                     mean_steer_deg=float(np.sum(ms[m] * dt[m]) / dt[m].sum())))
    if steer_geo is not None and getattr(steer_geo, 'has_pair_curve', False):
        lk = np.array([steer_geo.ackermann_pct_from_mean(math.radians(m_), t, L) if np.isfinite(m_) else np.nan for m_ in ms])
        okl = ok & np.isfinite(lk)
        out['linkage_pct_toe_weighted'] = float(np.sum(lk[okl] * dt[okl] * spread[okl]) / np.sum(dt[okl] * spread[okl]))
    return out


def ackermann_target_at(tire_model, ss_solver, steer_geo, R_m: float, lat_g: float, *,
                        grip_multiplier=None, aero_Fz_per_g=None) -> dict:
    """The demand % (equal use of both front tyres) at ONE (radius, g), the mean
    steer that condition needs, and what the present linkage gives there."""
    from .analysis_plots import ackermann_pct_vs_g_data
    d = ackermann_pct_vs_g_data(tire_model, ss_solver, steer_geo, radii_m=(float(R_m),),
                                lat_g_list=(float(lat_g),), grip_multiplier=grip_multiplier,
                                aero_Fz_per_g_by_radius=({float(R_m): aero_Fz_per_g} if aero_Fz_per_g else None))
    b = d['by_radius'][float(R_m)]
    if not len(b['lat_g']):
        raise RuntimeError(f'no equal-use solution at R {R_m:.1f} m, {lat_g:.2f} g (past the grip limit?)')
    return dict(R_m=float(R_m), lat_g=float(lat_g), target_pct=float(b['demand_pct'][0]),
                mean_steer_deg=float(b['mean_steer_deg'][0]), linkage_pct_now=float(b['linkage_pct'][0]))


def fit_arm_length_for_torque(front_hp: dict, target_Nm: float, *, torque_fn, bump_steer_from_hp,
                              scale_range=(0.5, 1.8), relevel: bool = True) -> dict:
    """Scale the steering-arm length (tie_rod_outer moved along the arm from the
    kingpin, same angle) until the PEAK hand torque over the lateral-g sweep
    equals target_Nm (the team's steering-effort KPI: ~15 N·m — lower = numb,
    higher = heavy).  Longer arm = slower steering = less torque.  torque_fn(hp)
    -> peak hand torque (N·m) from vahan.steering.compute_steering_effort with
    the aero package; re-levels the arm height for bump steer afterwards."""
    hp0 = {k: np.asarray(v, float).copy() for k, v in front_hp.items()}
    tro0 = hp0['tie_rod_outer']; kp = _kingpin_point_at(hp0, float(tro0[2])); arm = tro0 - kp

    def hp_at(scale, z=None):
        hp = {k: v.copy() for k, v in hp0.items()}
        zz = float(tro0[2] if z is None else z); kpz = _kingpin_point_at(hp0, zz)
        hp['tie_rod_outer'] = np.array([kpz[0] + arm[0] * scale, kpz[1] + arm[1] * scale, zz])
        return hp

    def T_at(scale):
        try:
            return float(torque_fn(hp_at(scale)))
        except Exception:
            return float('nan')
    lo, hi = float(scale_range[0]), float(scale_range[1])
    T_lo, T_hi = T_at(lo), T_at(hi)          # torque falls with scale
    if not (np.isfinite(T_lo) and np.isfinite(T_hi)):
        raise RuntimeError('hand-torque probe failed at the scale range ends')
    if not (T_hi <= target_Nm <= T_lo):
        # out of range: take the nearest end
        scale = lo if abs(T_lo - target_Nm) < abs(T_hi - target_Nm) else hi
    else:
        for _ in range(22):
            m = 0.5 * (lo + hi); Tm = T_at(m)
            if not np.isfinite(Tm):
                break
            if Tm > target_Nm:
                lo = m
            else:
                hi = m
        scale = 0.5 * (lo + hi)
    hp = hp_at(scale); z_used = float(tro0[2])
    if relevel:
        best = None
        for dz in np.arange(-40.0, 40.1, 4.0):
            try:
                up, dn = bump_steer_from_hp(hp_at(scale, tro0[2] + dz / 1000.0))
            except Exception:
                continue
            slope = 0.5 * (up - dn)
            if best is None or abs(slope) < abs(best[1]):
                best = (dz, slope)
        if best is not None:
            z_used = float(tro0[2] + best[0] / 1000.0); hp = hp_at(scale, z_used)
    up, dn = bump_steer_from_hp(hp)
    return dict(front_hp=hp, scale=float(scale), arm_length_mm=float(np.linalg.norm(arm[:2]) * scale * 1000.0),
                peak_torque_Nm=T_at(scale), target_Nm=float(target_Nm), z_mm=z_used * 1000.0,
                tie_rod_outer_new_mm=(hp['tie_rod_outer'] * 1000).tolist(), bump_steer_deg=(float(up), float(dn)))


def _kingpin_point_at(hp: dict, z: float) -> np.ndarray:
    lo, uo = np.asarray(hp['lca_outer'], float), np.asarray(hp['uca_outer'], float)
    f = (z - lo[2]) / (uo[2] - lo[2])
    return lo + f * (uo - lo)


def fit_arm_for_pct(front_hp: dict, target_pct: float, mean_steer_deg: float, *,
                    steer_geo_from_hp, bump_steer_from_hp, track_m: float, wheelbase_m: float,
                    rot_range_deg=(-60.0, 60.0), relevel: bool = True) -> dict:
    """Rotate tie_rod_outer about the kingpin (plan view, same arm length) until
    the probed linkage gives target_pct at mean_steer_deg; then (optionally)
    move it in Z to keep the bump-steer slope at zero.  Returns the new
    hardpoint dict plus what was done.  steer_geo_from_hp(hp) -> SteeringGeometry,
    bump_steer_from_hp(hp) -> (toe change at +25 mm, at -25 mm) in degrees."""
    hp0 = {k: np.asarray(v, float).copy() for k, v in front_hp.items()}
    tro0 = hp0['tie_rod_outer']; kp = _kingpin_point_at(hp0, float(tro0[2]))
    arm = tro0 - kp
    ms = math.radians(float(mean_steer_deg))

    def hp_at(rot_deg, z=None):
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        hp = {k: v.copy() for k, v in hp0.items()}
        zz = float(tro0[2] if z is None else z)
        kpz = _kingpin_point_at(hp0, zz)
        hp['tie_rod_outer'] = np.array([kpz[0] + arm[0] * c - arm[1] * s, kpz[1] + arm[0] * s + arm[1] * c, zz])
        return hp

    def pct_at(rot_deg):
        sg = steer_geo_from_hp(hp_at(rot_deg))
        if sg is None or not getattr(sg, 'has_pair_curve', False):
            return float('nan')
        return float(sg.ackermann_pct_from_mean(ms, track_m, wheelbase_m))

    lo, hi = float(rot_range_deg[0]), float(rot_range_deg[1])
    grid = np.linspace(lo, hi, 25); vals = np.array([pct_at(g) for g in grid])
    ok = np.isfinite(vals)
    if ok.sum() < 3:
        raise RuntimeError('steering probe failed across the rotation range')
    k = int(np.nanargmin(np.abs(vals - target_pct)))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    fa, fb = pct_at(a) - target_pct, pct_at(b) - target_pct
    if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
        for _ in range(25):
            m = 0.5 * (a + b); fm = pct_at(m) - target_pct
            if not np.isfinite(fm):
                break
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        rot = 0.5 * (a + b)
    else:
        rot = float(grid[k])
    hp = hp_at(rot); z_used = float(tro0[2])
    if relevel:
        # arm height that zeroes the bump-steer slope (symmetric part), scanned +-40 mm
        best = None
        for dz in np.arange(-40.0, 40.1, 4.0):
            try:
                up, dn = bump_steer_from_hp(hp_at(rot, tro0[2] + dz / 1000.0))
            except Exception:
                continue
            slope = 0.5 * (up - dn)
            if best is None or abs(slope) < abs(best[1]):
                best = (dz, slope)
        if best is not None:
            z_used = float(tro0[2] + best[0] / 1000.0)
            hp = hp_at(rot, z_used)
    sg = steer_geo_from_hp(hp)
    got = float(sg.ackermann_pct_from_mean(ms, track_m, wheelbase_m)) if sg is not None else float('nan')
    up, dn = bump_steer_from_hp(hp)
    return dict(front_hp=hp, rotation_deg=float(rot), tie_rod_outer_old_mm=(tro0 * 1000).tolist(),
                tie_rod_outer_new_mm=(hp['tie_rod_outer'] * 1000).tolist(), z_mm=z_used * 1000,
                pct_got=got, target_pct=float(target_pct), mean_steer_deg=float(mean_steer_deg),
                bump_steer_deg=(float(up), float(dn)))
