"""
vahan/laptime.py — quasi-steady-state (QSS) lap-time simulator.

Three-pass point-mass-plus solver on a traced track centerline, driven by the
SAME models the rest of Vahan uses (single-model rule):

  * tire grip      — the loaded TTC TireModel / LinearTireModel peak_mu(Fz)
                     (load-sensitive, so aero downforce raises grip degressively)
  * traction       — SteadyStateSolver._traction_g_dynamic-style weight-transfer
                     fixed point for the driven axle, friction-circle combined
  * power          — a per-gear wheel force from either a TABULATED crank
                     torque curve or the torque-plateau/constant-power model,
                     minus drag; VehicleParams.drive_force_at_v_N is the
                     no-gearbox fallback
  * driveline      — ROTATING INERTIA as equivalent mass: every longitudinal
                     acceleration is divided by (1 + m_eq/m), and the engine
                     term is referred through the total ratio SQUARED
  * driver         — a real gearshift: torque cut for a dead time, a minimum
                     interval between shifts, and hysteresis on the decision
  * steering       — an optional AckermannStationModel caps each station's
                     lateral limit by what the FRONT AXLE can hold at THAT
                     station's radius, and charges the front tyres' scrub as
                     longitudinal drag there
  * suspension     — a detailed pass runs the full SteadyStateSolver at sampled
                     stations to extract Fz / utilization / travel / roll /
                     LLTD / shock actuation from the actual car

Passes:
  1. corner-speed ceiling   v_max(s) from lateral grip (+ aero, iterated)
  2. forward (accel)        from standing start, traction & power limited,
                            carrying GEAR STATE (shift interval + dead time)
  3. backward (braking)     friction-circle braking + drag assist
  v(s) = min of the three;  t = ∫ ds / v.

MEASURED COST OF THE THREE HONESTY FIXES, autocross26.json, 2027_v58, grip
x0.65, no aero (baseline 41.902 s):
    rotating inertia        +1.997 s  (+4.8%)   — always flattered us
    gearshift realism       +0.091 s            — 1.50 s of torque cut, but
                                                  most of it lands where the
                                                  car is not power-limited
    tabulated torque curve  +0.198 s            — with the ASSUMED generic
                                                  restricted-600 shape
    all three together      +2.404 s  ->  44.306 s
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

G = 9.80665


# ─────────────────────────────────────────────────────────────────────────────
#  Track
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Aero ride-height sizing — cap ride height for a given downforce at a speed
# ─────────────────────────────────────────────────────────────────────────────

def downforce_at_speed_N(cla_m2: float, speed_ms: float,
                         rho: float = 1.225) -> float:
    """Total aero downforce (N) = Cl·A · ½ρv²."""
    return float(cla_m2) * 0.5 * float(rho) * float(speed_ms) ** 2


def aero_heave_mm(downforce_axle_N: float, ride_rate_Npm: float) -> float:
    """Symmetric heave (mm) of one axle under its aero load.  Each of the two
    wheels carries half the axle load against its ride rate (wheel rate in
    series with the tire), so heave = (D_axle/2) / ride_rate."""
    return (float(downforce_axle_N) / 2.0) / max(float(ride_rate_Npm), 1.0) * 1000.0


def required_ride_rate_Npm(downforce_axle_N: float, max_heave_mm: float) -> float:
    """Minimum per-wheel ride rate (N/m) so the axle's aero heave does not
    exceed ``max_heave_mm`` — i.e. ride height stays at/above its cap."""
    return (float(downforce_axle_N) / 2.0) / (max(float(max_heave_mm), 1e-3) / 1000.0)


def wheel_rate_from_ride_rate_Npm(ride_rate_Npm: float,
                                  tire_rate_Npm: float) -> float:
    """Invert the spring-in-series-with-tire relation:
        ride_rate = (wheel·tire)/(wheel+tire)  ->  wheel = ride·tire/(tire-ride)
    Returns inf if the demanded ride rate meets/exceeds the tire rate (then the
    TIRE is the limit — no spring can get you there)."""
    rr, tr = float(ride_rate_Npm), float(tire_rate_Npm)
    if tr <= rr + 1e-9:
        return float('inf')
    return rr * tr / (tr - rr)


# ─────────────────────────────────────────────────────────────────────────────
#  Engine torque curve — the ASSUMED generic shape, clearly labelled
# ─────────────────────────────────────────────────────────────────────────────
# A restrictor-limited 600 cc four does NOT make flat torque: it climbs to a
# broad peak a couple of thousand rpm below peak power and falls away after it.
# This is that SHAPE only, normalised to its own peak — no manufacturer data,
# no dyno sheet, nothing measured on this car.  It exists so the cost of the
# torque-plateau assumption can be QUANTIFIED before the team has a dyno pull;
# `assumed_torque_curve()` scales it so it reproduces the car's own peak power
# at the car's own peak-power rpm, which is the one number we do have.
#
#  *** REPLACE IT WITH TWO COLUMNS FROM A DYNO AS SOON AS THEY EXIST. ***
_GENERIC_600_SHAPE = (
    (2000, 0.55), (3000, 0.62), (4000, 0.72), (5000, 0.80), (6000, 0.88),
    (7000, 0.95), (8000, 1.00), (9000, 0.99), (10000, 0.92), (11000, 0.80),
    (12000, 0.68),
)


def assumed_torque_curve(peak_power_W: float, peak_power_rpm: float):
    """(rpm[], crank torque N·m[]) — the generic restricted-600 SHAPE above,
    scaled so it makes exactly `peak_power_W` at `peak_power_rpm`.

    ASSUMED, not measured.  Every consumer must say so."""
    rpm = np.array([r for r, _f in _GENERIC_600_SHAPE], float)
    frac = np.array([f for _r, f in _GENERIC_600_SHAPE], float)
    rp = max(float(peak_power_rpm), 1000.0)
    f_at_peak = float(np.interp(rp, rpm, frac))
    # T_peak * f(rp) * omega(rp) = P  ->  T_peak
    T_peak = float(peak_power_W) / max(f_at_peak * rp * 2 * np.pi / 60.0, 1e-6)
    return rpm, frac * T_peak


@dataclass
class Track:
    name: str
    x_m: np.ndarray
    y_m: np.ndarray
    kappa_1pm: np.ndarray     # signed curvature at each station
    width_m: float = 3.5

    @property
    def n(self) -> int:
        return len(self.x_m)

    @property
    def s_m(self) -> np.ndarray:
        seg = np.linalg.norm(
            np.diff(np.column_stack([self.x_m, self.y_m]), axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    @property
    def length_m(self) -> float:
        return float(self.s_m[-1])

    @classmethod
    def from_json(cls, path: str) -> 'Track':
        d = json.load(open(path))
        k = np.asarray(d['kappa_1pm'], float)
        # DIGITIZATION KINKS ARE NOT CORNERS.  Hand-digitized centerlines
        # carry single-point curvature spikes (autocross_2025 held 15 points
        # implying radii down to 0.2 m — tighter than the car can even
        # steer), and any radius-sensitive consumer (the per-station
        # Ackermann capability above all) reads them as extreme hairpins
        # and manufactures large fake effects (caught 2026-07-31: the whole
        # 54 ms lap-vs-Ackermann spread lived in those points; the clean
        # 9 m-minimum 2026 track showed none).  A 5-point median filter
        # removes isolated spikes; real corners span many consecutive
        # points and pass through unchanged.
        if len(k) >= 5:
            pad = np.concatenate((k[:2][::-1], k, k[-2:][::-1]))
            k = np.array([np.median(pad[i:i + 5]) for i in range(len(k))])
        return cls(name=d.get('name', path),
                   x_m=np.asarray(d['x_m'], float),
                   y_m=np.asarray(d['y_m'], float),
                   kappa_1pm=k,
                   width_m=float(d.get('width_m', 3.5)))


# ─────────────────────────────────────────────────────────────────────────────
#  Ackermann, PER STATION
# ─────────────────────────────────────────────────────────────────────────────
class AckermannStationModel:
    """How much front-axle grip an Ackermann setting realises, and what it
    costs in scrub, AS A FUNCTION OF CORNER RADIUS.

    WHY PER STATION, AND WHY THE OLD WAY WAS WRONG.  The previous chain
    (gui/ackermann_page graph 5) turned one Ackermann setting into ONE GLOBAL
    grip scale and re-ran the lap.  That destroys the entire effect, because
    the Ackermann benefit is RADIUS-DEPENDENT.  Measured on this car at grip
    ×0.65, front-axle force ceiling swept from -100% to +100% Ackermann:

        3 m corner   15.7% spread     scrub drag spread 280 N
        5 m corner    5.4%                              67 N
        8 m corner    2.3%                              12 N
       12 m corner    0.9%                             2.0 N
       20 m corner    0.2%                             0.2 N
       35 m corner    0.03%                            0.0 N

    One global number cannot represent that.  A tight hairpin and a fast
    sweeper get opposite treatment, and the lap's own radius distribution —
    which is the thing the user actually wants weighted — decides the answer.

    THIS CLASS COMPUTES NO TYRE PHYSICS.  Every number comes out of
    vahan.ackermann.solve_ackermann_force (Milliken pair analysis on the
    SOLVED car's wheel loads, radius-aware travel headings, signed tyre
    forces resolved onto the car axes).  All this does is call it on a grid
    and interpolate — because calling it 800 times a lap, per setting, per
    track, is not affordable.

    TWO CHANNELS COME OUT, and both go into the lap:
      * ay_front_cap_g(R, pct) — the lateral g at which the front axle's
        force CEILING stops covering the corner demand (m*g*wf*ay).  This is
        a per-station lateral grip cap: tight corners get a low one, fast
        sweepers a high one.
      * scrub_drag_N(R, ay, pct) — the longitudinal component of the front
        tyre forces at the TRIM steer (the steer that just holds the circle).
        Pure loss: it slows the car and it buys nothing.

    BINNING (stated, as asked).  Radius bins 2.5 / 3.0 / 3.5 / 4.2 / 5.0 /
    6.0 / 7.0 / 8.5 / 10 / 14 / 20 / 30 / 50 m — dense under 10 m where the
    whole effect lives, coarse above it where it has already died.  The first
    version of this table used 2.5/3.5/5/7/... and its own binning error was
    0.0122 g at R = 4.2 m, i.e. right in the middle of the interesting band;
    the dense grid drops that by about an order of magnitude, and the
    remaining error is MEASURED (binning_error_g) rather than assumed.
    Lateral-g ladder 0.4 / 0.8 / 1.2 / 1.6 / 2.0 g.  Beyond 50 m the model
    returns NO modifier at all, because the measured spread there (0.03% of
    force, 0.0 N of scrub) is far inside any numerical noise this chain has.
    Linear interpolation in log-radius and in g between bins; held flat
    outside.  The table is built ONCE and is independent of the track, so
    every track and every setting shares it.
    """

    RADII_M = (2.5, 3.0, 3.5, 4.2, 5.0, 6.0, 7.0, 8.5, 10.0,
               14.0, 20.0, 30.0, 50.0)
    LAT_G = (0.4, 0.8, 1.2, 1.6, 2.0)
    R_NO_EFFECT_M = 50.0          # beyond this: no modifier, measured dead

    def __init__(self, tire_model, solver, pct_list,
                 grip_multiplier: float = 0.65, aero=False):
        self.tire = tire_model
        self.solver = solver
        self.pct = np.asarray(sorted(float(p) for p in pct_list), float)
        self.gm = float(grip_multiplier)
        self.aero = aero
        self.built = False
        self.n_failed_cells = 0
        self.notes = []
        nR, nG, nP = len(self.RADII_M), len(self.LAT_G), len(self.pct)
        self._ceil = np.full((nR, nG, nP), np.nan)   # axle force ceiling, N
        self._scrub = np.full((nR, nG, nP), np.nan)  # trim scrub drag, N
        self._demand = np.full((nR, nG), np.nan)     # m*g*wf*ay, N
        self._ay_cap = np.full((nR, nP), np.nan)     # front-limited ay, g

    # ── build ────────────────────────────────────────────────────────────
    def build(self, progress_cb=None):
        from vahan.ackermann import solve_ackermann_force
        lo, hi = float(self.pct[0]), float(self.pct[-1])
        n = len(self.pct)
        even = (n > 1 and np.allclose(np.diff(self.pct),
                                      (hi - lo) / (n - 1), atol=1e-6))
        total = len(self.RADII_M) * len(self.LAT_G)
        done = 0
        for i, R in enumerate(self.RADII_M):
            for j, ag in enumerate(self.LAT_G):
                done += 1
                if progress_cb:
                    progress_cb(f'Ackermann capability {R:.1f} m @ {ag:.1f} g '
                                f'({done}/{total})', done, total)
                try:
                    if even:
                        d = solve_ackermann_force(
                            self.tire, self.solver, float(R), float(ag),
                            ack_range=(lo, hi), n=n,
                            grip_multiplier=self.gm, aero=self.aero)
                        u = np.asarray(d['useful_N'], float)
                        sc = np.asarray(d['trim_drag_N'], float)
                    else:
                        # uneven pct list -> sweep a fine even grid and
                        # interpolate onto it (still ONE call)
                        d = solve_ackermann_force(
                            self.tire, self.solver, float(R), float(ag),
                            ack_range=(lo, hi), n=41,
                            grip_multiplier=self.gm, aero=self.aero)
                        px = np.asarray(d['ackermann_pct'], float)
                        u = np.interp(self.pct, px,
                                      np.asarray(d['useful_N'], float))
                        sc = np.interp(self.pct, px,
                                       np.asarray(d['trim_drag_N'], float))
                    self._ceil[i, j, :] = u
                    self._scrub[i, j, :] = np.abs(sc)
                    self._demand[i, j] = float(d['demand_N'])
                except Exception as e:
                    self.n_failed_cells += 1
                    self.notes.append(f'R={R:.1f} m @ {ag:.2f} g: {e}')
        # front-limited lateral g per (radius, pct): where the axle's force
        # ceiling stops covering the demand.  demand is linear in ay, the
        # ceiling falls slowly with it, so one crossing.
        for i in range(len(self.RADII_M)):
            for k in range(len(self.pct)):
                self._ay_cap[i, k] = self._cross(i, k)
        self.built = True
        if self.n_failed_cells:
            self.notes.insert(0, f'{self.n_failed_cells}/{total} capability '
                                 f'cells failed to solve — those radii fall '
                                 f'back to NO Ackermann modifier')
        return self

    def _cross(self, i, k):
        """ay (g) where ceiling(ay) - demand(ay) changes sign, at radius i."""
        g = np.asarray(self.LAT_G, float)
        f = self._ceil[i, :, k] - self._demand[i, :]
        ok = np.isfinite(f)
        if ok.sum() < 2:
            return np.nan
        g, f = g[ok], f[ok]
        if f[0] <= 0:
            return float(g[0])              # front-limited below the ladder
        for a in range(1, len(g)):
            if f[a] <= 0:
                t = f[a - 1] / (f[a - 1] - f[a])
                return float(g[a - 1] + t * (g[a] - g[a - 1]))
        return float('inf')                 # never front-limited in range

    # ── lookups ──────────────────────────────────────────────────────────
    def _r_weights(self, radius_m):
        """(index_lo, index_hi, blend) in LOG radius; None beyond the dead
        radius (no modifier at all out there)."""
        R = float(radius_m)
        if not np.isfinite(R) or R >= self.R_NO_EFFECT_M:
            return None
        lr = np.log(np.asarray(self.RADII_M, float))
        x = np.log(max(R, self.RADII_M[0]))
        if x <= lr[0]:
            return 0, 0, 0.0
        a = int(np.searchsorted(lr, x))
        a = min(max(a, 1), len(lr) - 1)
        return a - 1, a, float((x - lr[a - 1]) / (lr[a] - lr[a - 1]))

    def _pct_interp(self, arr_lo, arr_hi, w, pct):
        v = arr_lo * (1.0 - w) + arr_hi * w
        return float(np.interp(float(pct), self.pct, v))

    def ay_front_cap_g(self, radius_m, pct) -> float:
        """Lateral g the FRONT AXLE can still hold on this radius at this
        Ackermann setting.  inf = never the limit here."""
        if not self.built:
            return float('inf')
        w = self._r_weights(radius_m)
        if w is None:
            return float('inf')
        a, b, t = w
        lo, hi = self._ay_cap[a, :], self._ay_cap[b, :]
        if not (np.isfinite(lo).any() and np.isfinite(hi).any()):
            return float('inf')
        # "never front-limited in the ladder" is stored as inf.  Interpolating
        # between a real 1.4 g and an inf would produce a wild number, so it
        # is replaced by a value comfortably above any tyre's lateral limit —
        # the caller takes min(base, this), so it simply means "no cap here".
        NO_CAP = 5.0
        lo = np.where(np.isfinite(lo), lo, NO_CAP)
        hi = np.where(np.isfinite(hi), hi, NO_CAP)
        return self._pct_interp(lo, hi, t, pct)

    def scrub_drag_N(self, radius_m, lat_g, pct) -> float:
        """Longitudinal loss (N) from the front tyres pointing across the
        car's path, at the steer that just holds this corner."""
        if not self.built:
            return 0.0
        w = self._r_weights(radius_m)
        if w is None:
            return 0.0
        a, b, t = w
        gg = np.asarray(self.LAT_G, float)
        x = float(np.clip(lat_g, gg[0], gg[-1]))
        j = int(np.clip(np.searchsorted(gg, x), 1, len(gg) - 1))
        s = (x - gg[j - 1]) / (gg[j] - gg[j - 1])
        row_lo = self._scrub[a, j - 1, :] * (1 - s) + self._scrub[a, j, :] * s
        row_hi = self._scrub[b, j - 1, :] * (1 - s) + self._scrub[b, j, :] * s
        if not (np.isfinite(row_lo).all() and np.isfinite(row_hi).all()):
            return 0.0
        return max(self._pct_interp(row_lo, row_hi, t, pct), 0.0)

    # ── the method's OWN noise, measured rather than assumed ─────────────
    def perturbed(self, delta_ay_g: float = 0.0, scrub_scale: float = 1.0):
        """A shallow copy of this table with every front-axle cap shifted by
        `delta_ay_g` and every scrub value scaled by `scrub_scale`.  Run a lap
        against it and the lap-time difference IS the noise band — the table's
        own interpolation error, priced by the sim itself instead of by a
        conversion factor.  (Moving every radius the SAME way is the worst
        case: correlated errors, not cancelling ones.)"""
        import copy
        m = copy.copy(self)
        m._ay_cap = self._ay_cap + float(delta_ay_g)
        m._scrub = self._scrub * float(scrub_scale)
        return m

    def binning_error(self, off_bin_radii=(2.7, 3.9, 4.6, 5.5, 9.2, 16.0),
                      progress_cb=None) -> dict:
        """How wrong this table's INTERPOLATION is, measured by re-solving at
        radii deliberately chosen to fall between its bins.

        Returns BOTH channels, because the two tracks are limited by different
        ones: a track with hairpins is moved by the front-axle CAP, a track
        whose tightest corner is 10 m is moved only by SCRUB.
            'ay_cap_g'    worst absolute error in the front-limited lateral g
            'scrub_frac'  worst RELATIVE error in the trim scrub drag

        This is the honest noise floor of the whole per-station chain: if the
        lap-time spread across Ackermann settings is smaller than what these
        errors are worth, the settings TIE and the plot must say so."""
        from vahan.ackermann import solve_ackermann_force
        worst_ay, worst_sc = 0.0, 0.0
        for R in off_bin_radii:
            for k, p in enumerate(self.pct):
                f, g = [], []
                for ag in self.LAT_G:
                    try:
                        d = solve_ackermann_force(
                            self.tire, self.solver, float(R), float(ag),
                            ack_range=(float(p), float(p)), n=1,
                            grip_multiplier=self.gm, aero=self.aero)
                    except Exception:
                        continue
                    f.append(float(d['useful_N'][0]) - float(d['demand_N']))
                    g.append(float(ag))
                    # scrub, same call, same operating point
                    sc_direct = abs(float(d['trim_drag_N'][0]))
                    sc_interp = self.scrub_drag_N(R, ag, p)
                    if sc_direct > 20.0:      # relative error is meaningless
                        worst_sc = max(worst_sc,               # on ~0 N
                                       abs(sc_interp - sc_direct) / sc_direct)
                if len(g) < 2:
                    continue
                direct = float('inf')
                if f[0] <= 0:
                    direct = g[0]
                else:
                    for a in range(1, len(g)):
                        if f[a] <= 0:
                            t = f[a - 1] / (f[a - 1] - f[a])
                            direct = g[a - 1] + t * (g[a] - g[a - 1])
                            break
                got = self.ay_front_cap_g(R, p)
                if np.isfinite(direct) and np.isfinite(got):
                    worst_ay = max(worst_ay, abs(direct - got))
            if progress_cb:
                progress_cb(f'noise floor at R={R:.1f} m: {worst_ay:.4f} g '
                            f'of cap, {100 * worst_sc:.1f}% of scrub so far')
        return {'ay_cap_g': float(worst_ay), 'scrub_frac': float(worst_sc)}

    def binning_error_g(self, off_bin_radii=(2.7, 3.9, 4.6, 5.5, 9.2, 16.0),
                        progress_cb=None) -> float:
        """Back-compat scalar: just the front-axle cap error."""
        return self.binning_error(off_bin_radii, progress_cb)['ay_cap_g']

    # ── reporting ────────────────────────────────────────────────────────
    def spread_report(self):
        """Per-radius spread of BOTH channels across the swept settings —
        the number that says whether this chain can carry Ackermann at all."""
        out = []
        for i, R in enumerate(self.RADII_M):
            c = self._ceil[i, :, :]
            s = self._scrub[i, :, :]
            with np.errstate(invalid='ignore'):
                cs = np.nanmax(np.nanmax(c, axis=1) - np.nanmin(c, axis=1))
                cm = np.nanmax(c)
                ss = np.nanmax(np.nanmax(s, axis=1) - np.nanmin(s, axis=1))
            out.append(dict(radius_m=R,
                            ceil_spread_N=float(cs),
                            ceil_spread_pct=float(100.0 * cs / max(cm, 1e-9)),
                            scrub_spread_N=float(ss),
                            ay_cap_g=self._ay_cap[i, :].copy()))
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LapResult:
    # per-station (track resolution)
    s_m: np.ndarray = None
    v_ms: np.ndarray = None
    t_s: np.ndarray = None
    lat_g: np.ndarray = None
    lon_g: np.ndarray = None
    v_corner_cap_ms: np.ndarray = None
    aero_down_N: np.ndarray = None
    aero_drag_N: np.ndarray = None
    engine_rpm: np.ndarray = None
    gear: np.ndarray = None              # selected gear per station (0 = n/a)
    power_used_W: np.ndarray = None
    diff_yaw_Nm: np.ndarray = None       # diff yaw moment (+ understeer/stabilise)
    rh_front_mm: np.ndarray = None       # front ride height vs distance
    rh_rear_mm: np.ndarray = None        # rear ride height vs distance
    limit: np.ndarray = None      # 0=grip(corner) 1=power 2=braking per station
    shifting: np.ndarray = None   # 1 where drive torque is CUT for a shift
    scrub_N: np.ndarray = None    # front-tyre scrub drag per station (N)
    ay_cap_local_ms2: np.ndarray = None   # per-station lateral ceiling used
    # detailed (sampled stations, full steady-state solve)
    det_s_m: np.ndarray = None
    det_channels: dict = field(default_factory=dict)
    # scalars
    n_upshifts: int = 0
    n_downshifts: int = 0
    shift_time_lost_s: float = 0.0   # total torque-cut time on the lap
    equiv_mass_kg: dict = field(default_factory=dict)   # per gear index
    engine_model: str = ''           # which torque model actually ran
    lap_time_s: float = 0.0
    sector_times_s: list = field(default_factory=list)
    avg_speed_kph: float = 0.0
    max_speed_kph: float = 0.0
    # Corner / usage summary (binder-facing).  Corner stations are where the
    # car pulls more than 0.30 g laterally; averages are TIME-weighted.
    min_speed_kph: float = 0.0
    peak_lat_g: float = 0.0
    avg_corner_lat_g: float = 0.0
    time_cornering_pct: float = 0.0
    peak_accel_g: float = 0.0
    peak_brake_g: float = 0.0
    notes: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator
# ─────────────────────────────────────────────────────────────────────────────

class LapSimulator:
    """QSS lap sim bound to a SteadyStateSolver (= the current car)."""

    def __init__(self, ss_solver, cla_m2: float = 0.0,
                 cda_m2: float | None = None,
                 air_density: float | None = None,
                 aero_cop_rear_frac: float = 0.5,
                 grip_scale: float = 1.0,
                 static_rh_front_mm: float = 50.0,
                 static_rh_rear_mm: float = 50.0):
        self.ss = ss_solver
        self.veh = ss_solver._veh
        self.tire = ss_solver._tire                       # front (and default)
        self.tire_rear = getattr(ss_solver, '_tire_rear', ss_solver._tire)
        self.cla = float(cla_m2)
        # Static ride heights (mm) measured at static sag — the baseline the
        # aero-induced heave drops from as downforce builds with speed.
        self.rh_front0 = float(static_rh_front_mm)
        self.rh_rear0 = float(static_rh_rear_mm)
        self.cda = float(cda_m2 if cda_m2 is not None else self.veh.cda_m2)
        self.rho = float(air_density if air_density is not None
                         else self.veh.air_density_kg_m3)
        self.cop_rear = float(np.clip(aero_cop_rear_frac, 0.0, 1.0))
        # Track/test grip derate: TTC belt data over-reads real asphalt grip;
        # teams typically run the sim at ~0.6-0.7 of belt μ.  Scales lateral
        # AND longitudinal grip (not power).
        self.grip_scale = float(np.clip(grip_scale, 0.1, 1.5))
        # Optional GEARBOX (set via set_gearbox).  None -> fall back to the
        # VehicleParams single-ratio drive_force model.
        self._gears = None          # list of gearbox ratios (1st..Nth)
        self._primary = 1.0
        self._final = 1.0
        self._redline_rpm = 0.0
        self._grip_v = None         # load-transfer grip table (built in simulate)
        self._grip_ay = None
        # ── rotating inertia (equivalent mass) — seeded from the ONE car ──
        v_ = self.veh
        self._I_wheel_f = max(float(getattr(v_, 'wheel_inertia_front_kgm2',
                                            0.0) or 0.0), 0.0)
        self._I_wheel_r = max(float(getattr(v_, 'wheel_inertia_rear_kgm2',
                                            0.0) or 0.0), 0.0)
        self._I_engine = max(float(getattr(v_, 'engine_inertia_kgm2',
                                           0.0) or 0.0), 0.0)
        # ── gearshift model — seeded from the ONE car ────────────────────
        self._shift_time = max(float(getattr(v_, 'shift_time_s', 0.0) or 0.0),
                               0.0)
        self._shift_min_int = max(
            float(getattr(v_, 'min_shift_interval_s', 0.0) or 0.0), 0.0)
        self._shift_hyst = max(
            float(getattr(v_, 'shift_hysteresis_frac', 0.0) or 0.0), 0.0)
        # ── tabulated torque curve (optional; plateau is the fallback) ────
        self._tq_rpm = None
        self._tq_Nm = None
        self._tq_label = ''
        # ── per-station Ackermann (optional) ─────────────────────────────
        self._ack_model = None
        self._ack_pct = 0.0

    # ── gearbox ──────────────────────────────────────────────────────────
    def set_gearbox(self, gear_ratios, primary_ratio: float,
                    final_drive: float, redline_rpm: float):
        """Per-gear drive model.  total_ratio(gear) = primary × gear × final.
        Engine: constant TORQUE below the peak-power rpm, constant POWER
        above it, nothing beyond redline (the standard 2-segment QSS model
        when only peak power + its rpm are known) — UNLESS a measured torque
        curve has been supplied with set_torque_curve()."""
        gears = [float(g) for g in gear_ratios if float(g) > 0]
        if gears:
            self._gears = gears
            self._primary = max(float(primary_ratio), 1e-3)
            self._final = max(float(final_drive), 1e-3)
            self._redline_rpm = max(float(redline_rpm), 1000.0)

    def set_rotating_inertia(self, wheel_front_kgm2: float,
                             wheel_rear_kgm2: float, engine_kgm2: float):
        """Override the car's rotating inertias (kg·m²).  wheel_* are PER
        WHEEL; engine is crank + clutch + primary, referred through the total
        ratio SQUARED.  Set all three to 0 to recover the old massless-
        driveline behaviour."""
        self._I_wheel_f = max(float(wheel_front_kgm2), 0.0)
        self._I_wheel_r = max(float(wheel_rear_kgm2), 0.0)
        self._I_engine = max(float(engine_kgm2), 0.0)
        self._eqm_cache = {}

    def set_shift_model(self, shift_time_s: float, min_interval_s: float,
                        hysteresis_frac: float):
        """Driver/gearbox shift realism.  shift_time_s = torque CUT duration,
        min_interval_s = shortest gap between two shifts, hysteresis_frac =
        how much better the new gear must be before the driver takes it.
        All zero = the old free, instantaneous, chattering picker."""
        self._shift_time = max(float(shift_time_s), 0.0)
        self._shift_min_int = max(float(min_interval_s), 0.0)
        self._shift_hyst = max(float(hysteresis_frac), 0.0)

    def set_torque_curve(self, rpm, torque_Nm, label: str = 'measured curve'):
        """Tabulated CRANK torque (N·m) against engine speed (rpm).  Replaces
        the torque-plateau / constant-power model.  Drivetrain efficiency is
        applied on top (the plateau model gets it through wheel_power_W).
        Pass None/empty to go back to the plateau."""
        if rpm is None or torque_Nm is None or len(rpm) < 2:
            self._tq_rpm = self._tq_Nm = None
            self._tq_label = ''
            return
        r = np.asarray(rpm, float)
        t = np.asarray(torque_Nm, float)
        k = np.argsort(r)
        self._tq_rpm, self._tq_Nm = r[k], t[k]
        self._tq_label = str(label)

    # ── engine torque: table if supplied, plateau/constant-power if not ──
    def engine_model_name(self) -> str:
        # The label must describe the physics that RUNS, not the data that was
        # handed over.  Without a gearbox the force path falls back to
        # veh.drive_force_at_v_N, which cannot consume a torque table — and
        # this label still said 'TABULATED', so two identical laps carried two
        # different engine claims (caught 2026-07-30: plateau vs CBR600RR
        # curve, both 43.487 s, because set_gearbox was never called).
        if not self._gears:
            return ('POWER-CAP fallback (NO GEARBOX SET — torque curve, if '
                    'supplied, is NOT in use; call set_gearbox)')
        if self._tq_rpm is not None:
            return (f'TABULATED torque curve ({self._tq_label}, '
                    f'{len(self._tq_rpm)} points '
                    f'{self._tq_rpm[0]:.0f}-{self._tq_rpm[-1]:.0f} rpm)')
        return ('ASSUMED torque plateau then constant power '
                '(no measured curve supplied)')

    def _engine_torque_Nm(self, rpm: float) -> float:
        """CRANK torque at this engine speed, BEFORE drivetrain efficiency."""
        if self._tq_rpm is not None:
            # flat-held outside the measured range — an interpolation, never
            # an invented extrapolation slope
            return float(np.interp(float(rpm), self._tq_rpm, self._tq_Nm))
        veh = self.veh
        P = float(veh.wheel_power_W) / max(float(veh.drivetrain_efficiency),
                                           1e-3)
        rpm_peak = max(float(veh.engine_rpm), 1000.0)
        T_peak = P / (rpm_peak * 2 * np.pi / 60.0)
        if rpm <= rpm_peak:
            return T_peak                       # torque plateau
        return P / (max(float(rpm), 1.0) * 2 * np.pi / 60.0)   # const power

    def _total_ratio(self, gi: int) -> float:
        """Engine rev per wheel rev for 1-based gear index gi."""
        if not self._gears:
            return float(getattr(self.veh, 'total_drive_ratio', 10.0) or 10.0)
        g = self._gears[int(np.clip(gi, 1, len(self._gears))) - 1]
        return self._primary * g * self._final

    def _rpm_in_gear(self, v: float, gi: int) -> float:
        r = max(self.veh.tire_radius_m, 1e-3)
        return max(v, 0.05) / r * self._total_ratio(gi) * 60.0 / (2 * np.pi)

    def _force_in_gear(self, v: float, gi: int):
        """(wheel force N, rpm) in gear gi at speed v.  0 N over the redline."""
        veh = self.veh
        r = max(veh.tire_radius_m, 1e-3)
        rpm = self._rpm_in_gear(v, gi)
        if self._redline_rpm > 0 and rpm > self._redline_rpm:
            return 0.0, float(rpm)
        T = self._engine_torque_Nm(rpm)
        F = T * float(veh.drivetrain_efficiency) * self._total_ratio(gi) / r
        return float(max(F, 0.0)), float(rpm)

    def _gear_force_rpm(self, v: float):
        """(best_force_N, rpm_in_that_gear, gear_index 1-based) at speed v.
        Picks the highest-force gear whose rpm is under redline; if even top
        gear over-revs, returns top gear pinned at redline with zero force.
        This is the INSTANTANEOUS-optimum picker — it has no memory, so it is
        what chatters.  The shift state machine (_shift_decide) is what the
        lap actually runs when a shift model is set."""
        veh = self.veh
        if not self._gears or veh.wheel_power_W <= 0:
            return float(veh.drive_force_at_v_N(max(v, 0.5))), \
                v / (2 * np.pi * max(veh.tire_radius_m, 1e-3)) * 60.0 \
                * float(getattr(veh, 'total_drive_ratio', 10.0) or 10.0), 0
        best = None
        for gi in range(1, len(self._gears) + 1):
            F, rpm = self._force_in_gear(v, gi)
            if self._redline_rpm > 0 and rpm > self._redline_rpm:
                if gi < len(self._gears):
                    continue                      # over-rev -> shift up
                return (0.0, float(self._redline_rpm), gi)
            if best is None or F > best[0]:
                best = (F, rpm, gi)
        return best if best else (0.0, float(self._redline_rpm),
                                  len(self._gears))

    def _drive_force(self, v: float) -> float:
        return self._gear_force_rpm(v)[0]

    # ── shift state machine ──────────────────────────────────────────────
    def _shift_decide(self, v: float, gear: int, since_shift_s: float) -> int:
        """The gear a driver with a real gearbox would be in.  Sequential:
        one ratio at a time.  A redline up-shift is forced (the limiter makes
        it for you) and ignores the interval; every other change needs BOTH
        the hysteresis margin and the minimum interval."""
        if not self._gears:
            return 0
        n = len(self._gears)
        gear = int(np.clip(gear, 1, n))
        F_cur, rpm_cur = self._force_in_gear(v, gear)
        # forced up-shift: on the limiter, this gear makes no force at all
        if self._redline_rpm > 0 and rpm_cur > self._redline_rpm and gear < n:
            return gear + 1
        if since_shift_s < self._shift_min_int:
            return gear
        # would one ratio either side do better by more than the margin?
        best_gi, best_F = gear, F_cur * (1.0 + self._shift_hyst)
        for cand in (gear - 1, gear + 1):
            if not 1 <= cand <= n:
                continue
            F_c, rpm_c = self._force_in_gear(v, cand)
            if self._redline_rpm > 0 and rpm_c > self._redline_rpm:
                continue
            if F_c > best_F:
                best_gi, best_F = cand, F_c
        return best_gi

    def _gear_trace(self, v_arr, ds):
        """Walk the shift state machine over a finished speed profile.
        Returns (gear[], rpm[], shifting[], n_up, n_down, time_lost_s)."""
        n = len(v_arr)
        gear = np.zeros(n, int)
        rpm = np.zeros(n)
        cut = np.zeros(n, int)
        if not self._gears:
            for i in range(n):
                _F, r_, gi = self._gear_force_rpm(float(v_arr[i]))
                rpm[i], gear[i] = r_, gi
            return gear, rpm, cut, 0, 0, 0.0
        g = max(self._gear_force_rpm(float(v_arr[0]))[2], 1)
        since, cut_left = 1e9, 0.0
        n_up = n_dn = 0
        lost = 0.0
        for i in range(n):
            dt = (float(ds[i - 1]) / max(float(v_arr[i]), 0.1)) if i else 0.0
            since += dt
            if cut_left > 0.0:
                step = min(cut_left, dt)
                lost += step
                cut_left -= step
                cut[i] = 1
            else:
                g_new = self._shift_decide(float(v_arr[i]), g, since)
                if g_new != g:
                    n_up += int(g_new > g)
                    n_dn += int(g_new < g)
                    g = g_new
                    since = 0.0
                    cut_left = self._shift_time
                    cut[i] = 1
            gear[i] = g
            rpm[i] = self._rpm_in_gear(float(v_arr[i]), g)
        return gear, rpm, cut, n_up, n_dn, lost

    # ── rotating inertia as equivalent mass ──────────────────────────────
    def equivalent_mass_kg(self, gear: int = 0) -> float:
        """Extra mass (kg) the driveline's spinning parts LOOK like at the
        contact patch:  [2*(I_f + I_r) + I_engine*ratio^2] / r^2.
        The engine term is amplified by the ratio SQUARED, which is why it
        dominates in the low gears."""
        r = max(float(self.veh.tire_radius_m), 1e-3)
        I = 2.0 * (self._I_wheel_f + self._I_wheel_r)
        if self._I_engine > 0.0:
            I += self._I_engine * self._total_ratio(gear if gear else 1) ** 2
        return I / (r * r)

    def _inertia_div(self, gear: int = 0) -> float:
        """1 + m_eq/m — divide every longitudinal acceleration by this."""
        if self._I_wheel_f <= 0 and self._I_wheel_r <= 0 and self._I_engine <= 0:
            return 1.0
        return 1.0 + self.equivalent_mass_kg(gear) \
            / max(float(self.veh.total_mass_kg), 1.0)

    # ── aero helpers ────────────────────────────────────────────────────
    def _q(self, v):                       # dynamic pressure
        return 0.5 * self.rho * v * v

    def downforce_N(self, v):
        return self.cla * self._q(v)

    def drag_N(self, v):
        return self.cda * self._q(v)

    # ── ride-height heave through the REAL (possibly nonlinear) wheel rate ──
    def _build_rate_curves(self):
        """Precompute, per axle, the cumulative wheel force vs compression
        using MR(travel) from the corner solver, so a progressive/exponential
        MR genuinely reduces aero heave.  Cached on the instance."""
        if getattr(self, '_rate_curves', None) is not None:
            return self._rate_curves
        out = {}
        for ax, label, mr0, spr in (
            ('F', 'FL', self.veh.motion_ratio_front, self.veh.spring_rate_front_Npm),
            ('R', 'RL', self.veh.motion_ratio_rear, self.veh.spring_rate_rear_Npm)):
            ts = np.linspace(0.0, 0.08, 41)              # 0..80 mm bump
            sv = self.ss._solvers.get(label)
            mr = np.full_like(ts, float(mr0))
            if sv is not None:
                try:
                    sl = np.array([sv.solve(float(t)).spring_length for t in ts])
                    mr = np.abs(np.gradient(sl, ts))
                    mr = np.clip(mr, 0.05, 5.0)
                except Exception:
                    pass
            k_wheel = float(spr) * mr ** 2               # N/m at each travel
            fcum = np.concatenate([[0.0], np.cumsum(
                0.5 * (k_wheel[1:] + k_wheel[:-1]) * np.diff(ts))])
            out[ax] = dict(ts=ts, fcum=fcum,
                           kt=max(float(self.veh.tire_rate_Npm), 1.0))
        self._rate_curves = out
        return out

    def _aero_heave_mm(self, curve, aero_load_per_wheel: float) -> float:
        """Heave (mm) of one wheel under an additional aero load, through the
        nonlinear wheel rate + tire in series."""
        load = max(float(aero_load_per_wheel), 0.0)
        dt = float(np.interp(load, curve['fcum'], curve['ts']))   # suspension
        tire = load / curve['kt']                                  # tire defl
        return (dt + tire) * 1000.0

    # ── grip helpers (load-sensitive μ from the ONE tire model) ────────
    def _mu_at_speed(self, v: float) -> float:
        """Vehicle-level lateral μ at speed v: per-tire Fz includes the aero
        share; degressive μ(Fz) means downforce adds grip sub-linearly.
        Negative Cl·A (net LIFT) UNLOADS the tires — guarded so per-tire Fz
        never goes non-physical."""
        m = self.veh.total_mass_kg
        Fz_tot = max(m * G + self.downforce_N(v), 0.05 * m * G)
        wf = self.veh.front_weight_fraction
        Fz_f = Fz_tot * wf / 2.0
        Fz_r = Fz_tot * (1.0 - wf) / 2.0
        mu_f = float(self.tire.peak_mu(Fz_f, 0.0))         # front tire
        mu_r = float(self.tire_rear.peak_mu(Fz_r, 0.0))    # rear tire (split)
        return min(mu_f, mu_r) * self.grip_scale   # conservative + derated

    def _ay_max_pointmass(self, v: float) -> float:
        """Point-mass lateral ceiling (m/s²) — μ·(g + downforce/m).  Ignores
        lateral load transfer, so on a real car it OVERSTATES grip (the
        outer tire overloads and degressive μ loses grip).  Used only as the
        bisection bracket + fallback for the load-transfer-aware table."""
        m = self.veh.total_mass_kg
        mu = self._mu_at_speed(v)
        return max(mu * (G + self.downforce_N(v) / m), 0.2)

    def _ay_at_util1(self, v: float) -> float:
        """Lateral accel (m/s²) at which the FULL per-corner solve's peak tire
        utilization reaches 1.0 at speed v — i.e. the real grip limit
        INCLUDING lateral load transfer + degressive μ, from the same
        SteadyStateSolver the detailed pass uses (single-model).  Bisection;
        aero downforce applied at this speed."""
        m = self.veh.total_mass_kg
        # aero load at this speed, split front/rear by CoP
        aero = None
        if abs(self.cla) > 1e-6:
            Fd = float(self.downforce_N(v))
            fa = Fd * (1.0 - self.cop_rear) / 2.0
            ra = Fd * self.cop_rear / 2.0
            aero = {'FL': fa, 'FR': fa, 'RL': ra, 'RR': ra}

        def peak_util(ay_g):
            try:
                r = self.ss.solve(abs(ay_g), 0.0, aero_Fz=aero)
                return max(r.utilization.values())
            except Exception:
                return float('nan')

        lo, hi = 0.05, self._ay_max_pointmass(v) / G * 1.3
        u_hi = peak_util(hi)
        if not np.isfinite(u_hi):
            return self._ay_max_pointmass(v)        # solver failed → fallback
        if u_hi < 1.0:
            return hi * G                            # never saturates in range
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            u = peak_util(mid)
            if not np.isfinite(u):
                break
            if u > 1.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi) * G

    def _build_grip_table(self, v_hint_ms: float):
        """Precompute the real lateral ceiling ay_max(v) on a speed grid (one
        bisection per grid speed), then interpolate.  ay_max depends on speed
        only through aero load, so a small grid + interpolation is exact
        enough and keeps the corner-cap pass cheap."""
        v_top = max(v_hint_ms * 1.1, 15.0)
        self._grip_v = np.linspace(2.0, v_top, 9)
        prev = getattr(self.ss, '_mu_scale', 1.0)
        self.ss._mu_scale = self.grip_scale
        self.ss._warm = {}
        self._grip_ay = np.array([self._ay_at_util1(float(v))
                                  for v in self._grip_v])
        self.ss._mu_scale = prev
        # monotone-ish guard: clamp to >0
        self._grip_ay = np.maximum(self._grip_ay, 0.2)

    def _ay_max(self, v: float) -> float:
        """Real lateral ceiling at speed v (m/s²) — interpolated from the
        load-transfer-aware grip table when built, else the point mass."""
        if getattr(self, '_grip_v', None) is not None:
            return float(np.interp(v, self._grip_v, self._grip_ay))
        return self._ay_max_pointmass(v)

    # ── per-station Ackermann ────────────────────────────────────────────
    def set_ackermann(self, model, pct: float):
        """Bind an AckermannStationModel and the setting to run it at.
        With this set, EVERY station's lateral ceiling is additionally capped
        by what the front axle can hold on THAT station's radius, and the
        front tyres' scrub is charged as longitudinal drag there.  Without
        it the sim behaves exactly as before (measured byte-identical)."""
        self._ack_model = model
        self._ack_pct = float(pct)

    def _ay_max_local(self, v: float, kappa: float) -> float:
        """Lateral ceiling at this station: the car's own grip limit, capped
        by the FRONT AXLE's capability at this station's RADIUS and this
        Ackermann setting.  A 3 m hairpin and a 30 m sweeper get different
        caps from the same setting — that is the whole point."""
        base = self._ay_max(v)
        if self._ack_model is None:
            return base
        ak = abs(float(kappa))
        if ak < 1e-5:
            return base                    # straight: no steer, no Ackermann
        cap = self._ack_model.ay_front_cap_g(1.0 / ak, self._ack_pct)
        if not np.isfinite(cap):
            return base
        return min(base, float(cap) * G)

    def scrub_drag_N(self, v: float, kappa: float) -> float:
        """Front-tyre scrub as a retarding force (N) at this station.  It is
        the part of the front tyre force that points along the car's path
        instead of at the corner: pure loss, and Ackermann changes it."""
        if self._ack_model is None:
            return 0.0
        ak = abs(float(kappa))
        if ak < 1e-5:
            return 0.0
        R = 1.0 / ak
        ay_g = v * v * ak / G
        if ay_g < 0.05:
            return 0.0
        return float(self._ack_model.scrub_drag_N(R, ay_g, self._ack_pct))

    def _corner_vmax(self, kappa: float) -> float:
        """Speed ceiling for one curvature: v² κ = ay_max(v), iterated
        because downforce grows with v (fixed point, bounded)."""
        ak = abs(float(kappa))
        if ak < 1e-5:
            return 1e3                     # straight — no lateral cap
        v = np.sqrt(self._ay_max_local(0.0, kappa) / ak)
        for _ in range(8):
            v_new = np.sqrt(self._ay_max_local(v, kappa) / ak)
            if abs(v_new - v) < 1e-3:
                return float(v_new)
            v = 0.5 * (v + v_new)
        return float(v)

    def _ax_drive(self, v: float, ay: float, gear: int = 0,
                  torque_cut: bool = False, kappa: float = 0.0) -> float:
        """Available forward accel (m/s²): min(power, traction) on the
        remaining friction circle, minus drag, DIVIDED by the rotating-inertia
        factor (1 + m_eq/m).

        gear        1-based gear the car is actually in (0 = let the
                    instantaneous picker choose).  It sets both the drive
                    force AND the engine's referred inertia.
        torque_cut  True while a gearshift is in progress: no drive force at
                    all, the car coasts against drag."""
        m = self.veh.total_mass_kg
        ay_cap = self._ay_max_local(v, kappa)
        circle = np.sqrt(max(0.0, 1.0 - (ay / max(ay_cap, 1e-6)) ** 2))
        # traction (driven axle, weight transfer) — reuse the solver's fixed
        # point but with aero-augmented load handled via μ at speed
        a_trac = self.ss._traction_g_dynamic(0.0) * G * circle * self.grip_scale
        # aero load scales the traction ceiling (downforce raises it; net
        # LIFT lowers it — clamped non-negative)
        a_trac *= max(1.0 + self.downforce_N(v) / (m * G), 0.0)
        if torque_cut:
            F_pwr = 0.0
        elif v > 0.5:
            if gear and self._gears:
                F_pwr = self._force_in_gear(v, gear)[0]
            else:
                F_pwr = self._drive_force(v)   # gearbox-aware when set
        else:
            F_pwr = a_trac * m
        a_pwr = F_pwr / m
        # NOTE the drag term is OUTSIDE the min(): drag is a force on the car,
        # not a ceiling.  Aero drag plus FRONT-TYRE SCRUB — the steered front
        # tyres make force across the car's path as well as at the corner, and
        # that component only slows it down.  Everything is then divided by
        # the equivalent-mass factor, because the same net force has to spin
        # the wheels and the crank up as well as push the car.
        resist = (self.drag_N(v) + self.scrub_drag_N(v, kappa)) / m
        return max(0.0, (min(a_pwr, a_trac) - resist)
                   / self._inertia_div(gear))

    def _ax_brake(self, v: float, ay: float, gear: int = 0,
                  kappa: float = 0.0) -> float:
        """Available braking decel (m/s², positive) — all four tires on the
        remaining circle, plus drag and front-tyre scrub helping, divided by
        the rotating-inertia factor.  The brakes must absorb the wheels' and
        (clutch engaged) the engine's rotational energy too, so the car stops
        SLOWER, not faster."""
        m = self.veh.total_mass_kg
        # The CIRCLE is measured against the LOCAL cap: if the corner speed
        # was set by the front axle running out of steering-geometry
        # capability, the fronts are saturated and (at 0.65 front bias) they
        # are also the ones that do the braking, so nothing is left.
        # The MAGNITUDE stays on the car's own grip: Ackermann is a lateral
        # geometry effect and does not reduce a straight-line stop.  Charging
        # it against the brakes as well was double-counting.
        ay_cap = self._ay_max_local(v, kappa)
        circle = np.sqrt(max(0.0, 1.0 - (ay / max(ay_cap, 1e-6)) ** 2))
        return (self._ay_max(v) * circle
                + (self.drag_N(v) + self.scrub_drag_N(v, kappa)) / m) \
            / self._inertia_div(gear)

    # ── the three passes ─────────────────────────────────────────────────
    def simulate(self, track: Track, n_detail: int = 60,
                 progress_cb=None) -> LapResult:
        s = track.s_m
        ds = np.diff(s)
        n = track.n
        kap = track.kappa_1pm

        def _prog(msg, pct):
            if progress_cb:
                progress_cb(msg, pct)

        # pass 0 — build the load-transfer-aware lateral grip table from the
        # full per-corner solver (so the corner cap respects what the tires
        # can REALLY hold once the outer tire overloads — not the optimistic
        # point-mass μ).  A rough top-speed hint sizes the speed grid.
        _prog('Calibrating grip (load transfer)…', 2)
        kmin = float(np.min(np.abs(kap)[np.abs(kap) > 1e-5])) \
            if np.any(np.abs(kap) > 1e-5) else 1e-3
        v_hint = float(np.sqrt(self._ay_max_pointmass(30.0) / max(kmin, 1e-4)))
        try:
            self._build_grip_table(min(v_hint, 60.0))
        except Exception:
            self._grip_v = None    # fall back to point mass

        # pass 1 — corner ceiling
        _prog('Corner-speed ceiling…', 8)
        vcap = np.array([self._corner_vmax(k) for k in kap])

        # pass 2 — forward from standing start.  The loop now carries GEAR
        # STATE: which ratio the driver is in, how long since the last shift,
        # and how much torque-cut time is still to be served.  Without it the
        # picker chatters (measured: 8 up + 7 down shifts on a 41 s lap using
        # only two gears) and every shift is free and instantaneous.
        _prog('Forward (accel) pass…', 25)
        v_f = np.zeros(n)
        v_f[0] = 0.5                       # rolling out of the start box
        gear = max(self._gear_force_rpm(v_f[0])[2], 1) if self._gears else 0
        since, cut_left = 1e9, 0.0
        for i in range(n - 1):
            if self._gears:
                if cut_left <= 0.0:
                    g_new = self._shift_decide(v_f[i], gear, since)
                    if g_new != gear:
                        gear, since, cut_left = g_new, 0.0, self._shift_time
            ay = v_f[i] ** 2 * abs(kap[i])
            ax = self._ax_drive(v_f[i], ay, gear=gear, torque_cut=cut_left > 0,
                                kappa=kap[i])
            v_next = np.sqrt(max(v_f[i] ** 2 + 2.0 * ax * ds[i], 0.01))
            v_f[i + 1] = min(v_next, vcap[i + 1])
            dt_i = ds[i] / max(0.5 * (v_f[i] + v_f[i + 1]), 0.1)
            since += dt_i
            cut_left = max(cut_left - dt_i, 0.0)

        # pass 3 — backward braking.  The gear here is only used for the
        # engine's referred inertia (the brakes, not the engine, make the
        # force), so the instantaneous picker is the right source: it is the
        # gear the car is in at that speed.
        _prog('Backward (braking) pass…', 45)
        v_b = vcap.copy()
        v_b[-1] = vcap[-1]
        for i in range(n - 2, -1, -1):
            ay = v_b[i + 1] ** 2 * abs(kap[i + 1])
            gb = self._gear_force_rpm(v_b[i + 1])[2] if self._gears else 0
            ab = self._ax_brake(v_b[i + 1], ay, gear=gb, kappa=kap[i + 1])
            v_prev = np.sqrt(v_b[i + 1] ** 2 + 2.0 * ab * ds[i])
            v_b[i] = min(vcap[i], v_prev)

        v = np.minimum(v_f, v_b)
        v = np.maximum(v, 0.1)

        # which limit is active
        limit = np.zeros(n, int)           # 0 grip
        limit[np.isclose(v, v_f) & (v < vcap - 0.05)] = 1   # power/accel
        limit[np.isclose(v, v_b) & (v < vcap - 0.05) & (v_b < v_f)] = 2  # brake

        # time + accelerations
        v_mid = 0.5 * (v[1:] + v[:-1])
        dt = ds / np.maximum(v_mid, 0.1)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        lat_g = v ** 2 * np.abs(kap) / G
        lon_g = np.concatenate([[0.0], np.diff(v ** 2) / (2.0 * ds)]) / G

        res = LapResult(
            s_m=s, v_ms=v, t_s=t, lat_g=lat_g, lon_g=lon_g,
            v_corner_cap_ms=vcap,
            aero_down_N=self.downforce_N(v), aero_drag_N=self.drag_N(v),
            limit=limit,
            lap_time_s=float(t[-1]),
            avg_speed_kph=float(track.length_m / t[-1] * 3.6),
            max_speed_kph=float(v.max() * 3.6),
        )
        # ── corner / usage summary (time-weighted; binder-facing) ─────────
        _latm = 0.5 * (lat_g[1:] + lat_g[:-1])
        _lonm = 0.5 * (lon_g[1:] + lon_g[:-1])
        _corner = _latm > 0.30
        _t_c = float(np.sum(dt[_corner]))
        res.min_speed_kph = float(v.min() * 3.6)
        res.peak_lat_g = float(np.max(lat_g))
        res.avg_corner_lat_g = (float(np.sum(_latm[_corner] * dt[_corner])
                                      / _t_c) if _t_c > 0 else 0.0)
        res.time_cornering_pct = float(100.0 * _t_c / t[-1])
        res.peak_accel_g = float(np.max(_lonm))
        res.peak_brake_g = float(-np.min(_lonm))
        # ── per-station Ackermann channels ────────────────────────────────
        if self._ack_model is not None:
            res.scrub_N = np.array([self.scrub_drag_N(float(v[i]),
                                                      float(kap[i]))
                                    for i in range(n)])
            res.ay_cap_local_ms2 = np.array([
                self._ay_max_local(float(v[i]), float(kap[i]))
                for i in range(n)])
            base = np.array([self._ay_max(float(v[i])) for i in range(n)])
            bound = int(np.sum(res.ay_cap_local_ms2 < base - 1e-9))
            res.notes.append(
                f'Ackermann {self._ack_pct:+.0f}%: front-axle capability caps '
                f'the lateral limit at {bound}/{n} stations; scrub drag peaks '
                f'{np.max(res.scrub_N):.0f} N, lap-mean '
                f'{np.mean(res.scrub_N):.0f} N')
        else:
            res.scrub_N = np.zeros(n)
            res.ay_cap_local_ms2 = np.array([self._ay_max(float(vi))
                                             for vi in v])

        # ── ride height under downforce (PROGRESSIVE-MR aware) ─────────────
        # Aero downforce compresses the suspension (heave), dropping ride
        # height as speed builds.  The heave is found by integrating the REAL
        # wheel-rate curve k_wheel(t) = spring_rate · MR(t)² from the corner
        # solver — so a NONLINEAR (e.g. progressive) MR actually resists
        # compression and holds ride height, instead of a constant rate that
        # would ignore the rocker shape.  Tire deflection (load/tire-rate) adds
        # in series.  Constant-MR collapses to the old load/ride_rate formula.
        curves = self._build_rate_curves()
        down = self.downforce_N(v)
        heave_f_mm = np.array([self._aero_heave_mm(curves['F'],
                              float(d) * (1.0 - self.cop_rear) / 2.0) for d in down])
        heave_r_mm = np.array([self._aero_heave_mm(curves['R'],
                              float(d) * self.cop_rear / 2.0) for d in down])
        res.rh_front_mm = self.rh_front0 - heave_f_mm
        res.rh_rear_mm = self.rh_rear0 - heave_r_mm
        # Load-sensitivity note: aero can push a tire past the tested load
        # range.  The grip there DOES fall with load (load sensitivity, applied
        # via peak_mu's extended decline + grip-force ceiling) — the only
        # caveat is the exact decline RATE comes from extending your data's
        # last segment, not from measured points at those loads.
        try:
            fz_hi = float(self.tire.fz_range[1])
            m = self.veh.total_mass_kg
            wf = self.veh.front_weight_fraction
            Fz_tot_max = m * G + float(np.max(self.downforce_N(v)))
            Fz_tire_max = Fz_tot_max * max(wf, 1.0 - wf) / 2.0
            if Fz_tire_max > fz_hi:
                res.notes.append(
                    f'aero loads a tire to {Fz_tire_max:.0f} N, '
                    f'{(Fz_tire_max / fz_hi - 1) * 100:.0f}% past your TTC data '
                    f'({fz_hi:.0f} N) — grip falls with load there (load '
                    f'sensitivity extended from your data, not measured)')
        except Exception:
            pass
        # sectors (thirds by distance)
        thirds = [np.searchsorted(s, track.length_m * f) for f in (1/3, 2/3)]
        res.sector_times_s = [float(t[thirds[0]]),
                              float(t[thirds[1]] - t[thirds[0]]),
                              float(t[-1] - t[thirds[1]])]

        # powertrain channels (gearbox-aware: sawtooth rpm + gear trace).
        # power_used = the power the car ACTUALLY needs at each station
        # (m·ax + drag)·v when driving — NOT the available gear force, which
        # would overstate partial-throttle sections.
        m_tot = self.veh.total_mass_kg
        gear_i, rpm_arr, cut_arr, n_up, n_dn, t_lost = \
            self._gear_trace(v, ds)
        gear_arr = gear_i.astype(float)
        pwr_arr = np.zeros(n)
        for i in range(n):
            F_avail = (self._force_in_gear(float(v[i]), int(gear_i[i]))[0]
                       if self._gears else
                       self._gear_force_rpm(float(v[i]))[0])
            if cut_arr[i]:
                F_avail = 0.0
            if lon_g[i] > 0:
                # power the car ACTUALLY needs: the net force plus drag, plus
                # what the rotating parts absorb (m_eq is real mass to spin up)
                m_eff = m_tot + self.equivalent_mass_kg(int(gear_i[i]))
                F_used = m_eff * lon_g[i] * G + float(self.drag_N(v[i]))
                pwr_arr[i] = min(max(F_used, 0.0), max(F_avail, 0.0)) * v[i]
        res.engine_rpm = rpm_arr
        res.gear = gear_arr
        res.shifting = cut_arr
        res.power_used_W = pwr_arr
        res.n_upshifts, res.n_downshifts = int(n_up), int(n_dn)
        res.shift_time_lost_s = float(t_lost)
        res.engine_model = self.engine_model_name()
        res.equiv_mass_kg = {
            int(gi): float(self.equivalent_mass_kg(int(gi)))
            for gi in (np.unique(gear_i) if self._gears else [0])}
        # SAY which engine model ran, and what the rotating inertia is worth —
        # a silent default is how a 5% bias survives for months.
        if self._gears:
            g_lo = int(np.min(gear_i[gear_i > 0])) if np.any(gear_i > 0) else 1
            res.notes.append(
                f'engine: {res.engine_model}; rotating inertia adds '
                f'{self.equivalent_mass_kg(g_lo):.0f} kg equivalent in gear '
                f'{g_lo} ({100 * (self._inertia_div(g_lo) - 1):.0f}% of car '
                f'mass) — inertias are ASSUMED defaults until measured')
            if self._shift_time > 0 or self._shift_min_int > 0:
                res.notes.append(
                    f'shifts: {n_up} up / {n_dn} down, {t_lost:.2f} s of '
                    f'torque cut at {self._shift_time * 1000:.0f} ms each, '
                    f'min interval {self._shift_min_int:.2f} s, hysteresis '
                    f'{100 * self._shift_hyst:.0f}%')

        # ── differential yaw moment (corner entry/exit balance) ───────────
        # On power: drive torque through the diff -> power-ramp locking ->
        # understeer.  Off power: the clutch preload still acts (a stabilising
        # entry moment); the coast RAMP adds to it under engine braking, which
        # this lap model doesn't separate from the brakes — so coast shows the
        # preload baseline.  Sign: + = understeer (exit) / stabilising (entry).
        from vahan.differential import Differential
        diff = Differential.from_vehicle(self.veh)
        r_t = max(self.veh.tire_radius_m, 1e-3)
        track_r = float(getattr(self.veh, 'rear_track_m', 1.2))
        # grip cap on the force bias: the driven axle can't bias more force
        # than its tires can transmit (~ its peak μ × axle load).
        mu0 = self._mu_at_speed(15.0) / max(self.grip_scale, 1e-3)
        Fz_rear = self.veh.total_mass_kg * G * (1.0 - self.veh.front_weight_fraction)
        max_bias = mu0 * Fz_rear
        ratio = float(getattr(self.veh, 'total_drive_ratio', 10.0) or 10.0)
        T_overrun = float(getattr(self.veh, 'engine_braking_Nm', 0.0)) * ratio
        diff_arr = np.zeros(n)
        for i in range(n):
            on_power = lon_g[i] > 0.02
            if on_power:
                F_drive = pwr_arr[i] / max(v[i], 0.5)      # tractive force
                T_axle = F_drive * r_t
                cap = min(max_bias, F_drive)
            elif lon_g[i] < -0.02:
                T_axle = T_overrun        # engine braking -> coast ramp live
                cap = max_bias
            else:
                T_axle = 0.0              # neutral -> preload only
                cap = max_bias
            diff_arr[i] = diff.yaw_moment_Nm(T_axle, track_r, r_t, on_power,
                                             max_bias_N=cap)
        res.diff_yaw_Nm = diff_arr

        # ── detailed pass: full steady-state solve at sampled stations ────
        _prog('Suspension detail pass…', 60)
        idx = np.unique(np.linspace(0, n - 1, max(int(n_detail), 8)).astype(int))
        det = {k: [] for k in (
            'Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR',
            'util_FL', 'util_FR', 'util_RL', 'util_RR',
            'travel_FL', 'travel_FR', 'travel_RL', 'travel_RR',
            'shock_F_mm', 'shock_R_mm',
            'roll_deg', 'pitch_deg', 'lltd_pct', 'understeer_deg')}
        mr_f = max(self.veh.motion_ratio_front, 1e-3)
        mr_r = max(self.veh.motion_ratio_rear, 1e-3)
        done = 0
        self.ss._warm = {}
        # Make the full-solve utilization use the SAME grip the lap's speed/
        # lat-g were derated by, so the util plot and the lat-g agree (a util
        # of 1.0 = the tire is at the grip the sim actually assumed).
        _mu_scale_prev = getattr(self.ss, '_mu_scale', 1.0)
        self.ss._mu_scale = self.grip_scale
        for i in idx:
            lg = float(np.sign(kap[i]) * lat_g[i])
            xg = float(np.clip(lon_g[i], -3.0, 3.0))
            # per-corner aero load at this station's speed
            aero = None
            if self.cla > 1e-6:
                Fd = float(self.downforce_N(v[i]))
                fa = Fd * (1.0 - self.cop_rear) / 2.0
                ra = Fd * self.cop_rear / 2.0
                aero = {'FL': fa, 'FR': fa, 'RL': ra, 'RR': ra}
            try:
                r_ = self.ss.solve(abs(lg), xg, aero_Fz=aero)
                for c in ('FL', 'FR', 'RL', 'RR'):
                    det[f'Fz_{c}'].append(float(r_.Fz.get(c, np.nan)))
                    det[f'util_{c}'].append(float(r_.utilization.get(c, np.nan)))
                    det[f'travel_{c}'].append(float(r_.travel.get(c, np.nan)))
                det['shock_F_mm'].append(
                    float(r_.travel.get('FL', np.nan)) * mr_f)
                det['shock_R_mm'].append(
                    float(r_.travel.get('RL', np.nan)) * mr_r)
                det['roll_deg'].append(float(r_.roll_angle_deg))
                det['pitch_deg'].append(float(r_.pitch_angle_deg))
                el_f = r_.elastic_lt_front_N + r_.geometric_lt_front_N \
                    + r_.unsprung_lt_front_N
                el_r = r_.elastic_lt_rear_N + r_.geometric_lt_rear_N \
                    + r_.unsprung_lt_rear_N
                tot = el_f + el_r
                det['lltd_pct'].append(
                    float(el_f / tot * 100.0) if tot > 1.0 else np.nan)
                det['understeer_deg'].append(
                    float(r_.understeer_gradient_deg))
            except Exception:
                for k in det:
                    det[k].append(np.nan)
            done += 1
            if done % 10 == 0:
                _prog(f'Suspension detail {done}/{len(idx)}…',
                      60 + int(35 * done / len(idx)))

        self.ss._mu_scale = _mu_scale_prev   # restore for the dynamics page

        res.det_s_m = s[idx]
        res.det_channels = {k: np.asarray(vv, float) for k, vv in det.items()}
        # Consistency note: with util now on the same grip as the lap, the
        # peak util should sit near 1.0 at the grip-limited corners.  Flag if
        # the QSS point-mass cap and the full per-corner solve disagree by a
        # lot (real load-transfer effects the point mass can't see).
        try:
            ch = res.det_channels
            um = np.maximum.reduce([ch[f'util_{c}']
                                    for c in ('FL', 'FR', 'RL', 'RR')])
            lon_d = np.interp(res.det_s_m, s, lon_g)
            apex = np.abs(lon_d) < 0.2          # ~pure lateral
            apex_u = np.nanmax(um[apex]) if np.any(apex) else np.nan
            comb_u = np.nanmax(um[~apex]) if np.any(~apex) else np.nan
            if np.isfinite(comb_u) and comb_u > 1.1:
                res.notes.append(
                    f'lateral grip is load-transfer-honest (apex util '
                    f'{apex_u:.2f}); combined brake/accel-in-corner util peaks '
                    f'{comb_u:.2f} — point-mass friction circle is optimistic '
                    f'there (QSS limitation, lap slightly quick on entries/exits)')
        except Exception:
            pass
        _prog('Done', 100)
        return res
