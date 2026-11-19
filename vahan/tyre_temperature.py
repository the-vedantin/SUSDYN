"""Tyre temperature for the lap sim: grip vs tyre surface temperature, a
two-node tyre heat model, and the lap-to-lap loop that ties them together.

NOTHING IS BAKED IN.  Every number comes from:
  * a DATA FILE (tyre_temperature_*.json, kept with the tyre data, which is
    gitignored — measured tyre data is never in this public source): the
    grip-vs-temperature curve over the MEASURED temperature range, the
    reference temperature the tyre model's grip belongs to, the maker's
    temperature window, and the fitted heat-model constants;
  * the USER'S INPUTS on the Laptime page (TyreTempSettings): ambient and
    track temperature, mode, fixed or start temperature, number of laps,
    the ASSUMED grip loss per 10 °C outside the measured range, and every
    heat-model knob (all seeded from the data file, all editable).

Model
-----
Grip: factor(T) = rel_mu(T) / rel_mu(ref_T), linear interpolation inside the
measured range.  Outside it the curve is extended with the user's assumed
loss per 10 °C (cold side and hot side separately) and the lap is flagged.
The factor multiplies each axle's tyre forces (ScaledTyre), so it acts on
every limit the lap sim builds (corner, brake, drive, g-g-V map) the same
way the project grip scale does, but per axle.

Heat (per tyre, lumped per axle, the TWO-NODE model the data file was fitted
with: surface Ts and tyre body Tb):
    dTs/dt = a*P - k1*(Ts - Tb) - (h0 + h1*v)*(Ts - Te)
    dTb/dt = k2*(Ts - Tb) - hb*(Tb - Te)
    Te     = (1 - r)*T_ambient + r*T_track
P is the sliding power into one tyre:
    P = v*( |Fy|*tan(alpha_pk*u) + |Fx|*kappa_pk*u )
with the axle forces from the lap's lateral/forward g (front share of
cornering force = front weight fraction; braking split by brake bias; drive
on the rear), u = the axle's grip use from the lap's suspension solves, and
alpha_pk / kappa_pk the slip at peak grip.

Why lap-to-lap and not station-by-station: the g-g-V limit map is built for
one car (one grip level per axle) and costs a full map solve.  Rebuilding it
at every station would multiply the run time by the number of stations.
Tyre temperature changes over seconds-to-minutes, so each lap runs on the
mean temperature of the lap before; 2-3 laps converge (reported per lap)."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

G = 9.81


# ─────────────────────────────────────────────────────────────────────────────
#  data file
# ─────────────────────────────────────────────────────────────────────────────
def find_data_files(repo_root: str) -> list:
    """Every tyre_temperature_*.json in tire_data/ (gitignored)."""
    return sorted(glob.glob(os.path.join(repo_root, 'tire_data',
                                         'tyre_temperature_*.json')))


def load_data(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    for k in ('grip_curve', 'thermal'):
        if k not in d:
            raise ValueError(f'{os.path.basename(path)}: missing "{k}"')
    return d


class GripTempCurve:
    """factor(T) relative to the reference temperature of the tyre model."""

    def __init__(self, grip_curve: dict, cold_loss_pct_per_10C: float,
                 hot_loss_pct_per_10C: float):
        self.T = np.asarray(grip_curve['T_C'], float)
        self.rel = np.asarray(grip_curve['rel_mu'], float)
        self.ref_T = float(grip_curve['ref_T_C'])
        self.lo, self.hi = (float(x) for x in grip_curve['measured_C'])
        self.cold = float(cold_loss_pct_per_10C) / 100.0 / 10.0   # per °C
        self.hot = float(hot_loss_pct_per_10C) / 100.0 / 10.0
        self.ref_rel = float(np.interp(self.ref_T, self.T, self.rel))

    def factor(self, T_C: float) -> float:
        T = float(T_C)
        if T < self.lo:
            r = float(np.interp(self.lo, self.T, self.rel)) * (1.0 - self.cold * (self.lo - T))
        elif T > self.hi:
            r = float(np.interp(self.hi, self.T, self.rel)) * (1.0 - self.hot * (T - self.hi))
        else:
            r = float(np.interp(T, self.T, self.rel))
        return float(np.clip(r / self.ref_rel, 0.3, 1.5))

    def where(self, T_C: float) -> str:
        """'measured' inside the tested range, else 'assumed (cold)'/'(hot)'."""
        T = float(T_C)
        if T < self.lo:
            return 'assumed (colder than tested)'
        if T > self.hi:
            return 'assumed (hotter than tested)'
        return 'measured'


# ─────────────────────────────────────────────────────────────────────────────
#  grip-scaled tyre (one axle)
# ─────────────────────────────────────────────────────────────────────────────
class ScaledTyre:
    """Wraps a tyre model and multiplies its grip by k (forces, peak grip and
    cornering stiffness; the slip for a given force is found on the scaled
    curve).  Everything else is passed through untouched."""

    _SCALED = ('Fy', 'Mz', 'peak_Fy', 'mf_Fy', 'cornering_stiffness', 'peak_mu')

    def __init__(self, base, k: float):
        self._base = base
        self._k = float(k)

    def __getattr__(self, name):
        attr = getattr(self._base, name)
        if name in ScaledTyre._SCALED and callable(attr):
            k = self._k

            def scaled(*a, **kw):
                r = attr(*a, **kw)
                return np.asarray(r, float) * k if np.ndim(r) else float(r) * k
            return scaled
        return attr

    def slip_angle_for_Fy(self, Fy_target_N, *a, **kw):
        return self._base.slip_angle_for_Fy(Fy_target_N / self._k, *a, **kw)


def unwrap(t):
    while isinstance(t, ScaledTyre):
        t = t._base
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  user settings
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TyreTempSettings:
    mode: str = 'off'                  # 'off' | 'fixed' | 'estimated'
    data: dict = field(default_factory=dict)   # the loaded data file
    data_name: str = ''
    ambient_C: float = 20.0
    track_C: float = 25.0
    fixed_C: float = 50.0              # 'fixed': tyre surface temperature
    start_C: float | None = None       # 'estimated': lap-1 tyre temp (None = ambient)
    laps: int = 3                      # 'estimated': laps run (last reported)
    cold_loss_pct_per_10C: float = 5.0  # ASSUMED, below the measured range
    hot_loss_pct_per_10C: float = 5.0   # ASSUMED, above the measured range
    # heat model (seeded from the data file's fit; all user-editable)
    a_K_per_J: float = 0.0
    k1_per_s: float = 0.0
    h0_per_s: float = 0.0
    h1_per_m: float = 0.0
    k2_per_s: float = 0.0
    hb_per_s: float = 0.0
    road_share: float = 0.3
    slip_peak_deg: float = 8.0
    slip_ratio_peak: float = 0.10

    @classmethod
    def from_data(cls, data: dict, name: str = '', **over):
        th = data['thermal']; gc = data['grip_curve']
        s = cls(data=data, data_name=name,
                a_K_per_J=th['a_K_per_J'], k1_per_s=th['k1_per_s'],
                h0_per_s=th['h0_per_s'], h1_per_m=th['h1_per_m'],
                k2_per_s=th['k2_per_s'], hb_per_s=th['hb_per_s'],
                road_share=th.get('road_share', 0.3),
                slip_peak_deg=th.get('slip_peak_deg', 8.0),
                cold_loss_pct_per_10C=gc.get('default_cold_loss_pct_per_10C', 5.0),
                hot_loss_pct_per_10C=gc.get('default_hot_loss_pct_per_10C', 5.0))
        for k, v in over.items():
            setattr(s, k, v)
        return s

    def curve(self) -> GripTempCurve:
        return GripTempCurve(self.data['grip_curve'], self.cold_loss_pct_per_10C,
                             self.hot_loss_pct_per_10C)

    @property
    def env_C(self) -> float:
        r = float(np.clip(self.road_share, 0.0, 1.0))
        return (1.0 - r) * float(self.ambient_C) + r * float(self.track_C)


# ─────────────────────────────────────────────────────────────────────────────
#  heat model along a lap
# ─────────────────────────────────────────────────────────────────────────────
def axle_slip_power(sim, res, st: TyreTempSettings):
    """Sliding power (W) into ONE tyre of each axle at every station."""
    veh = sim.veh
    m = float(veh.total_mass_kg)
    wf = float(veh.front_weight_fraction)
    bias = float(getattr(veh, 'front_brake_bias', 0.6))
    v = np.asarray(res.v_ms, float)
    ay = np.abs(np.asarray(res.lat_g, float)) * G
    ax = np.asarray(res.lon_g, float) * G
    Fy_f, Fy_r = m * ay * wf, m * ay * (1.0 - wf)
    brk = ax < 0
    Fx_f = np.where(brk, m * -ax * bias, 0.0)
    Fx_r = np.where(brk, m * -ax * (1.0 - bias), m * np.maximum(ax, 0.0))
    s = np.asarray(res.s_m, float)
    ch = getattr(res, 'det_channels', {}) or {}
    note = ''
    if 'util_axle_F' in ch and res.det_s_m is not None and len(res.det_s_m) >= 2:
        uf = np.interp(s, res.det_s_m, np.nan_to_num(ch['util_axle_F'], nan=0.0))
        ur = np.interp(s, res.det_s_m, np.nan_to_num(ch['util_axle_R'], nan=0.0))
    else:   # no suspension solves: grip use from total g vs the lap's peak
        a = np.hypot(ay, ax); uf = ur = a / max(float(a.max()), 1e-6)
        note = 'grip use for heating estimated from total g (no suspension solves)'
    uf = np.clip(uf, 0.0, 1.2); ur = np.clip(ur, 0.0, 1.2)
    ap = np.radians(float(st.slip_peak_deg)); kp = float(st.slip_ratio_peak)
    Pf = v * (Fy_f * np.tan(ap * uf) + Fx_f * kp * uf) / 2.0
    Pr = v * (Fy_r * np.tan(ap * ur) + Fx_r * kp * ur) / 2.0
    return Pf, Pr, note


def integrate_heat(t, P, v, st: TyreTempSettings, Ts0: float, Tb0: float):
    """Surface temperature along the lap (two-node model) + end state."""
    Te = st.env_C
    a, k1, h0, h1, k2, hb = (st.a_K_per_J, st.k1_per_s, st.h0_per_s,
                             st.h1_per_m, st.k2_per_s, st.hb_per_s)
    Ts, Tb = float(Ts0), float(Tb0)
    out = np.empty(len(t)); out[0] = Ts
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        n = max(1, int(np.ceil(dt / 0.05)))
        h = dt / n
        Pi, vi = 0.5 * (P[i] + P[i - 1]), 0.5 * (v[i] + v[i - 1])
        for _ in range(n):
            dTs = a * Pi - k1 * (Ts - Tb) - (h0 + h1 * vi) * (Ts - Te)
            dTb = k2 * (Ts - Tb) - hb * (Tb - Te)
            Ts += h * dTs; Tb += h * dTb
        out[i] = Ts
    return out, Ts, Tb


def time_mean(t, x):
    t = np.asarray(t, float); x = np.asarray(x, float)
    if len(t) < 2 or t[-1] <= t[0]:
        return float(np.mean(x))
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(x, t) / (t[-1] - t[0]))


# ─────────────────────────────────────────────────────────────────────────────
#  the lap loop
# ─────────────────────────────────────────────────────────────────────────────
def simulate_with_temperature(sim, run_once, st: TyreTempSettings, progress_cb=None):
    """run_once(progress_cb) runs ONE lap on the sim's current tyres.
    Returns the last lap's LapResult with tyre temperature channels added."""
    curve = st.curve()
    ss = sim.ss
    base = dict(ss_f=unwrap(ss._tire), ss_r=unwrap(getattr(ss, '_tire_rear', ss._tire)),
                sf=unwrap(sim.tire), sr=unwrap(sim.tire_rear))

    def install(kf, kr):
        ss._tire = ScaledTyre(base['ss_f'], kf)
        ss._tire_rear = ScaledTyre(base['ss_r'], kr)
        sim.tire = ScaledTyre(base['sf'], kf)
        sim.tire_rear = ScaledTyre(base['sr'], kr)

    def restore():
        ss._tire, ss._tire_rear = base['ss_f'], base['ss_r']
        sim.tire, sim.tire_rear = base['sf'], base['sr']

    q = lambda k: round(float(k) / 0.002) * 0.002   # reuse cached maps for ~equal grip
    log = []
    try:
        if st.mode == 'fixed':
            k = q(curve.factor(st.fixed_C))
            install(k, k)
            res = run_once(progress_cb)
            n = len(res.s_m)
            res.tyre_temp_front_C = np.full(n, float(st.fixed_C))
            res.tyre_temp_rear_C = np.full(n, float(st.fixed_C))
            res.tyre_grip_factor = (k, k)
            where = curve.where(st.fixed_C)
            log.append(f'tyre {st.fixed_C:.0f} °C (fixed) -> grip x{k:.3f} ({where})')
            flags = [where] if where != 'measured' else []
        else:   # estimated
            T0 = float(st.ambient_C if st.start_C is None else st.start_C)
            Ts = {'F': T0, 'R': T0}; Tb = {'F': T0, 'R': T0}
            Tmean = {'F': T0, 'R': T0}
            laps = max(1, int(st.laps))
            flags = []
            res = None
            for lap in range(1, laps + 1):
                kf, kr = q(curve.factor(Tmean['F'])), q(curve.factor(Tmean['R']))
                install(kf, kr)
                if progress_cb:
                    progress_cb(f'Tyre temperature: lap {lap}/{laps} '
                                f'(F {Tmean["F"]:.0f} °C, R {Tmean["R"]:.0f} °C)…', 1)
                res = run_once(progress_cb)
                Pf, Pr, pnote = axle_slip_power(sim, res, st)
                t = np.asarray(res.t_s, float); v = np.asarray(res.v_ms, float)
                trace = {}
                for ax, P in (('F', Pf), ('R', Pr)):
                    trace[ax], Ts[ax], Tb[ax] = integrate_heat(t, P, v, st, Ts[ax], Tb[ax])
                used = dict(Tmean)
                Tmean = {ax: time_mean(t, trace[ax]) for ax in trace}
                log.append(f'lap {lap}: {res.lap_time_s:.2f} s on F {used["F"]:.0f} °C '
                           f'(x{kf:.3f}) / R {used["R"]:.0f} °C (x{kr:.3f}); lap mean '
                           f'F {Tmean["F"]:.0f} / R {Tmean["R"]:.0f} °C')
                for ax in ('F', 'R'):
                    w = curve.where(used[ax])
                    if w != 'measured' and lap == laps:
                        flags.append(f'{"front" if ax == "F" else "rear"} {w}')
            res.tyre_temp_front_C = trace['F']
            res.tyre_temp_rear_C = trace['R']
            res.tyre_grip_factor = (kf, kr)
            if pnote:
                log.append(pnote)
    finally:
        restore()
    res.tyre_temp_log = log
    lo, hi = curve.lo, curve.hi
    res.notes.append('tyre temperature (' + st.mode + ', ' + (st.data_name or 'data file')
                     + f'; ambient {st.ambient_C:.0f} °C, track {st.track_C:.0f} °C): '
                     + ' | '.join(log))
    if flags:
        res.notes.append(f'tyre grip vs temperature is MEASURED only {lo:.0f}-{hi:.0f} °C; '
                         + ', '.join(sorted(set(flags)))
                         + f' -> grip uses the ASSUMED loss of {st.cold_loss_pct_per_10C:g} %/10 °C '
                         f'(cold) / {st.hot_loss_pct_per_10C:g} %/10 °C (hot)')
    win = st.data.get('maker_window_C')
    if win:
        Ts_all = np.r_[res.tyre_temp_front_C, res.tyre_temp_rear_C]
        if Ts_all.min() < win[0] or Ts_all.max() > win[1]:
            res.notes.append(f'tyre temperature leaves the maker window {win[0]:.0f}-{win[1]:.0f} °C')
    return res
