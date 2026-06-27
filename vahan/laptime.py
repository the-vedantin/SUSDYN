"""
vahan/laptime.py — quasi-steady-state (QSS) lap-time simulator.

Three-pass point-mass-plus solver on a traced track centerline, driven by the
SAME models the rest of Vahan uses (single-model rule):

  * tire grip      — the loaded TTC TireModel / LinearTireModel peak_mu(Fz)
                     (load-sensitive, so aero downforce raises grip degressively)
  * traction       — SteadyStateSolver._traction_g_dynamic-style weight-transfer
                     fixed point for the driven axle, friction-circle combined
  * power          — VehicleParams.drive_force_at_v_N (ICE/EV aware) minus drag
  * suspension     — a detailed pass runs the full SteadyStateSolver at sampled
                     stations to extract Fz / utilization / travel / roll /
                     LLTD / shock actuation from the actual car

Passes:
  1. corner-speed ceiling   v_max(s) from lateral grip (+ aero, iterated)
  2. forward (accel)        from standing start, traction & power limited
  3. backward (braking)     friction-circle braking + drag assist
  v(s) = min of the three;  t = ∫ ds / v.
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
        return cls(name=d.get('name', path),
                   x_m=np.asarray(d['x_m'], float),
                   y_m=np.asarray(d['y_m'], float),
                   kappa_1pm=np.asarray(d['kappa_1pm'], float),
                   width_m=float(d.get('width_m', 3.5)))


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
    # detailed (sampled stations, full steady-state solve)
    det_s_m: np.ndarray = None
    det_channels: dict = field(default_factory=dict)
    # scalars
    lap_time_s: float = 0.0
    sector_times_s: list = field(default_factory=list)
    avg_speed_kph: float = 0.0
    max_speed_kph: float = 0.0
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

    # ── gearbox ──────────────────────────────────────────────────────────
    def set_gearbox(self, gear_ratios, primary_ratio: float,
                    final_drive: float, redline_rpm: float):
        """Per-gear drive model.  total_ratio(gear) = primary × gear × final.
        Engine: constant TORQUE below the peak-power rpm, constant POWER
        above it, nothing beyond redline (the standard 2-segment QSS model
        when only peak power + its rpm are known)."""
        gears = [float(g) for g in gear_ratios if float(g) > 0]
        if gears:
            self._gears = gears
            self._primary = max(float(primary_ratio), 1e-3)
            self._final = max(float(final_drive), 1e-3)
            self._redline_rpm = max(float(redline_rpm), 1000.0)

    def _gear_force_rpm(self, v: float):
        """(best_force_N, rpm_in_that_gear, gear_index 1-based) at speed v.
        Picks the highest-force gear whose rpm is under redline; if even top
        gear over-revs, returns top gear pinned at redline force=power model."""
        veh = self.veh
        P = veh.wheel_power_W
        if not self._gears or P <= 0:
            return float(veh.drive_force_at_v_N(max(v, 0.5))), \
                v / (2 * np.pi * max(veh.tire_radius_m, 1e-3)) * 60.0 \
                * float(getattr(veh, 'total_drive_ratio', 10.0) or 10.0), 0
        r = max(veh.tire_radius_m, 1e-3)
        rpm_peak = max(float(veh.engine_rpm), 1000.0)
        w_peak = rpm_peak * 2 * np.pi / 60.0
        T_peak = P / w_peak                       # engine torque plateau
        wheel_w = max(v, 0.05) / r
        best = None
        for gi, g in enumerate(self._gears):
            ratio = self._primary * g * self._final
            rpm = wheel_w * ratio * 60.0 / (2 * np.pi)
            if rpm > self._redline_rpm:
                if gi < len(self._gears) - 1:
                    continue                      # over-rev -> shift up
                # even TOP gear is on the limiter: no more drive force
                return (0.0, float(self._redline_rpm), gi + 1)
            if rpm < rpm_peak:
                F = T_peak * ratio / r            # torque plateau
            else:
                F = P / max(v, 0.5)               # constant power
            if best is None or F > best[0]:
                best = (float(F), float(rpm), gi + 1)
        return best if best else (0.0, float(self._redline_rpm),
                                  len(self._gears))

    def _drive_force(self, v: float) -> float:
        return self._gear_force_rpm(v)[0]

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

    def _corner_vmax(self, kappa: float) -> float:
        """Speed ceiling for one curvature: v² κ = ay_max(v), iterated
        because downforce grows with v (fixed point, bounded)."""
        ak = abs(float(kappa))
        if ak < 1e-5:
            return 1e3                     # straight — no lateral cap
        v = np.sqrt(self._ay_max(0.0) / ak)
        for _ in range(8):
            v_new = np.sqrt(self._ay_max(v) / ak)
            if abs(v_new - v) < 1e-3:
                return float(v_new)
            v = 0.5 * (v + v_new)
        return float(v)

    def _ax_drive(self, v: float, ay: float) -> float:
        """Available forward accel (m/s²): min(power, traction) on the
        remaining friction circle, minus drag."""
        m = self.veh.total_mass_kg
        ay_cap = self._ay_max(v)
        circle = np.sqrt(max(0.0, 1.0 - (ay / max(ay_cap, 1e-6)) ** 2))
        # traction (driven axle, weight transfer) — reuse the solver's fixed
        # point but with aero-augmented load handled via μ at speed
        a_trac = self.ss._traction_g_dynamic(0.0) * G * circle * self.grip_scale
        # aero load scales the traction ceiling (downforce raises it; net
        # LIFT lowers it — clamped non-negative)
        a_trac *= max(1.0 + self.downforce_N(v) / (m * G), 0.0)
        if v > 0.5:
            F_pwr = self._drive_force(v)   # gearbox-aware when set_gearbox()
        else:
            F_pwr = a_trac * m
        a_pwr = F_pwr / m
        return max(0.0, min(a_pwr, a_trac) - self.drag_N(v) / m)

    def _ax_brake(self, v: float, ay: float) -> float:
        """Available braking decel (m/s², positive) — all four tires on the
        remaining circle, plus drag helping."""
        m = self.veh.total_mass_kg
        ay_cap = self._ay_max(v)
        circle = np.sqrt(max(0.0, 1.0 - (ay / max(ay_cap, 1e-6)) ** 2))
        return self._ay_max(v) * circle + self.drag_N(v) / m

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

        # pass 2 — forward from standing start
        _prog('Forward (accel) pass…', 25)
        v_f = np.zeros(n)
        v_f[0] = 0.5                       # rolling out of the start box
        for i in range(n - 1):
            ay = v_f[i] ** 2 * abs(kap[i])
            ax = self._ax_drive(v_f[i], ay)
            v_next = np.sqrt(max(v_f[i] ** 2 + 2.0 * ax * ds[i], 0.01))
            v_f[i + 1] = min(v_next, vcap[i + 1])

        # pass 3 — backward braking
        _prog('Backward (braking) pass…', 45)
        v_b = vcap.copy()
        v_b[-1] = vcap[-1]
        for i in range(n - 2, -1, -1):
            ay = v_b[i + 1] ** 2 * abs(kap[i + 1])
            ab = self._ax_brake(v_b[i + 1], ay)
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
        rpm_arr = np.zeros(n)
        gear_arr = np.zeros(n)
        pwr_arr = np.zeros(n)
        for i in range(n):
            F_avail, rpm, gi = self._gear_force_rpm(float(v[i]))
            rpm_arr[i] = rpm
            gear_arr[i] = gi
            if lon_g[i] > 0:
                F_used = m_tot * lon_g[i] * G + float(self.drag_N(v[i]))
                pwr_arr[i] = min(max(F_used, 0.0), max(F_avail, 0.0)) * v[i]
        res.engine_rpm = rpm_arr
        res.gear = gear_arr
        res.power_used_W = pwr_arr

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
