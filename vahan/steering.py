"""
vahan/steering.py — Steering-wheel ↔ rack ↔ road-wheel geometry.

The dynamics / transient solvers work internally in **road-wheel angle**
(the actual angle the front tire makes with the car centreline — the
quantity the bicycle model cares about).  Drivers, though, rotate a
steering wheel.  The conversion chain is:

    steering_wheel_deg
        × (rack_travel_per_rev_mm / 360)     [rack geometry — driver lever]
        = rack_travel_mm
        × (dδ/d(rack)) from kinematics        [linkage geometry]
        = road_wheel_angle_rad

This file encapsulates that chain so the solver, the GUI, and the
skidpad path follower all share one consistent mapping — and so that
**changing rack mm/rev actually changes the simulation output**.

Public API:
    SteeringGeometry        — holds the mapping; supports both probed
                              and linearised constructors.

Construction modes
------------------
1. ``from_probe(front_solver, front_hp, rack_params, ...)`` — preferred.
   Builds a dense rack ↔ road-wheel lookup by solving the kinematics at
   a grid of rack travels.  Captures full nonlinearity (bump-steer,
   anti-Ackermann, etc.).

2. ``from_linear_ratio(overall_ratio_deg_deg, rack_per_rev_mm, ...)`` —
   fallback when no kinematic solver is available.  Pure linear model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SteeringGeometry:
    """
    Rack geometry + mapping between steering-wheel, rack, and road-wheel.

    All internal data is stored as monotonic lookup tables so forward
    and inverse lookups are vectorised and robust against non-linearity.
    The *front* axle is modelled symmetrically — the "representative"
    road-wheel angle used for conversions is the average of FL and FR
    toe magnitudes (signed so +rack → +road-wheel).

    Attributes
    ----------
    rack_m_grid : np.ndarray
        Monotonically increasing rack positions (m), symmetric about 0.
    road_wheel_rad_grid : np.ndarray
        Corresponding average road-wheel angle at each rack position.
    rack_travel_per_rev_mm : float
        Driver side of the conversion.
    max_rack_half_m : float
        Physical half-stroke of the rack (m).
    max_road_wheel_rad : float
        Maximum attainable road-wheel angle, given the rack limit.
        (Equal to the terminal value of ``road_wheel_rad_grid``.)
    overall_ratio_deg_per_deg : float
        Small-angle steering-wheel-to-road-wheel ratio (degree of
        hand-wheel rotation per degree of road-wheel rotation).
        Higher = slower steering (more turns to lock-to-lock).
    """

    rack_m_grid: np.ndarray
    road_wheel_rad_grid: np.ndarray
    rack_travel_per_rev_mm: float
    max_rack_half_m: float
    max_road_wheel_rad: float
    overall_ratio_deg_per_deg: float

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_probe(cls,
                   front_solver_factory,
                   front_hp_fl: dict,
                   front_hp_fr: dict,
                   rack_travel_per_rev_mm: float,
                   total_rack_travel_mm: float,
                   n_samples: int = 11) -> "SteeringGeometry":
        """
        Build the mapping by probing the kinematic solver at a grid of
        rack travels.

        Parameters
        ----------
        front_solver_factory : callable
            ``front_solver_factory(rack_m: float, side: str) -> float``
            returns the road-wheel angle (rad, signed ISO: +toe-out =
            positive for left corner under +rack).  ``side`` is 'FL' or
            'FR'.  Any solver failure should return ``np.nan`` — this
            routine masks NaNs and falls back to the nearest valid
            sample.
        front_hp_fl, front_hp_fr : dict
            Front corner hardpoint dicts (not used here — present so
            callers can construct the factory with the geometry baked in.
            Kept in the signature for symmetry with the legacy plumbing).
        rack_travel_per_rev_mm, total_rack_travel_mm
            Rack geometry straight from ``VehicleParams``.
        n_samples : int
            Grid density across the full rack range.  Odd numbers put a
            sample exactly at 0 (which gets zero road-wheel by symmetry).
        """
        max_rack_half_m = float(total_rack_travel_mm) / 2.0 / 1000.0

        # Sample rack positions symmetrically about zero
        rack_grid = np.linspace(-max_rack_half_m, +max_rack_half_m, int(n_samples))

        # Probe road-wheel at each rack position.  Sign convention:
        # positive rack → left turn → FL toe-in (-) / FR toe-in (+)?
        # We average |toe| and assign the sign of the rack input.
        rw = np.zeros_like(rack_grid)
        for i, rm in enumerate(rack_grid):
            try:
                a_fl = front_solver_factory(float(rm), 'FL')
                a_fr = front_solver_factory(float(rm), 'FR')
                if not (np.isfinite(a_fl) and np.isfinite(a_fr)):
                    rw[i] = np.nan
                    continue
                # Representative road-wheel angle = (|FL| + |FR|) / 2
                # with the sign taken from rack direction.
                mag = 0.5 * (abs(a_fl) + abs(a_fr))
                rw[i] = np.sign(rm) * mag if rm != 0 else 0.0
            except Exception:
                rw[i] = np.nan

        # Fill NaNs by nearest neighbour so the interpolator stays
        # monotonic.  If the whole probe failed, fall back to a linear
        # model: 0.02 rad per mm of rack (typical FSAE ~1.1°/mm).
        if np.all(np.isnan(rw)):
            return cls.from_linear_ratio(
                overall_ratio_deg_per_deg=6.0,          # generic FSAE fallback
                rack_travel_per_rev_mm=rack_travel_per_rev_mm,
                max_rack_half_m=max_rack_half_m,
            )

        mask = ~np.isnan(rw)
        rw = np.interp(rack_grid, rack_grid[mask], rw[mask])

        # Small-angle overall ratio from central finite difference
        dr = rack_grid[1] - rack_grid[0] if len(rack_grid) > 1 else 1e-3
        central = len(rw) // 2
        if central > 0 and central < len(rw) - 1:
            drw_drack = (rw[central + 1] - rw[central - 1]) / (2 * dr)
        else:
            drw_drack = 0.02  # fallback slope
        # Steering-wheel angle (deg) to rack (m): sw_deg × (per_rev_mm / 360) / 1000
        drack_dsw_deg = rack_travel_per_rev_mm / (360.0 * 1000.0)  # m per degree
        dsw_deg_drw_deg = 1.0 / max(abs(drw_drack * drack_dsw_deg * np.pi / 180.0),
                                    1e-9)

        max_rw = float(np.nanmax(np.abs(rw)))
        return cls(
            rack_m_grid=rack_grid,
            road_wheel_rad_grid=rw,
            rack_travel_per_rev_mm=float(rack_travel_per_rev_mm),
            max_rack_half_m=float(max_rack_half_m),
            max_road_wheel_rad=max_rw,
            overall_ratio_deg_per_deg=float(dsw_deg_drw_deg),
        )

    @classmethod
    def from_linear_ratio(cls,
                          overall_ratio_deg_per_deg: float,
                          rack_travel_per_rev_mm: float,
                          max_rack_half_m: float,
                          n_samples: int = 11) -> "SteeringGeometry":
        """
        Fallback constructor using a pure linear model.

        Useful when the GUI's corner solvers aren't available (e.g.
        headless tests or early init).  Given the overall steering
        ratio (hand-wheel degrees per road-wheel degree), derive the
        rack→road-wheel slope and build the lookup.
        """
        # Overall ratio: dsw_deg / drw_deg
        # sw_deg × rack_per_rev_mm / 360 = rack_mm  (driver side)
        # So drack_mm/dsw_deg = rack_per_rev_mm / 360
        # And drw_deg/drack_mm = (1/overall_ratio) × (360/rack_per_rev_mm)
        drw_deg_drack_mm = (1.0 / max(overall_ratio_deg_per_deg, 1e-6)) \
                           * (360.0 / max(rack_travel_per_rev_mm, 1e-6))
        drw_rad_drack_m  = np.radians(drw_deg_drack_mm) * 1000.0

        rack_grid = np.linspace(-max_rack_half_m, +max_rack_half_m, int(n_samples))
        rw = drw_rad_drack_m * rack_grid
        max_rw = float(abs(rw).max())
        return cls(
            rack_m_grid=rack_grid,
            road_wheel_rad_grid=rw,
            rack_travel_per_rev_mm=float(rack_travel_per_rev_mm),
            max_rack_half_m=float(max_rack_half_m),
            max_road_wheel_rad=max_rw,
            overall_ratio_deg_per_deg=float(overall_ratio_deg_per_deg),
        )

    # ── Forward lookups ──────────────────────────────────────────────────

    def road_wheel_from_rack(self, rack_m) -> np.ndarray:
        """Road-wheel angle (rad) at a given rack position (m)."""
        rack_m = np.asarray(rack_m, float)
        return np.interp(rack_m, self.rack_m_grid, self.road_wheel_rad_grid)

    def road_wheel_from_steering_wheel(self, sw_deg) -> np.ndarray:
        """Road-wheel angle (rad) at a given steering-wheel angle (deg)."""
        sw_deg = np.asarray(sw_deg, float)
        rack_m = sw_deg * (self.rack_travel_per_rev_mm / 360.0) / 1000.0
        return self.road_wheel_from_rack(rack_m)

    # ── Inverse lookups ──────────────────────────────────────────────────

    def rack_from_road_wheel(self, rw_rad) -> np.ndarray:
        """
        Rack position (m) required to produce the given road-wheel angle.

        Uses np.interp with the axes swapped.  Since the road-wheel grid
        is (nearly) monotonic with rack position, this is safe.
        """
        rw_rad = np.asarray(rw_rad, float)
        # Ensure monotonic sort for np.interp
        order = np.argsort(self.road_wheel_rad_grid)
        xs = self.road_wheel_rad_grid[order]
        ys = self.rack_m_grid[order]
        return np.interp(rw_rad, xs, ys)

    def steering_wheel_from_road_wheel(self, rw_rad) -> np.ndarray:
        """Steering-wheel angle (deg) required for the given road-wheel angle."""
        rack_m = self.rack_from_road_wheel(rw_rad)
        return rack_m * 1000.0 * 360.0 / max(self.rack_travel_per_rev_mm, 1e-6)

    def rack_mm_from_road_wheel(self, rw_rad) -> np.ndarray:
        """Convenience: rack position in mm (for per-step logging)."""
        return self.rack_from_road_wheel(rw_rad) * 1000.0

    # ── Saturation ──────────────────────────────────────────────────────

    def saturate_road_wheel(self, rw_rad: float) -> float:
        """Clamp a demanded road-wheel angle to the physical rack limit."""
        lim = self.max_road_wheel_rad
        return float(np.clip(rw_rad, -lim, +lim))


# ═════════════════════════════════════════════════════════════════════════════
# STEERING EFFORT — what the driver's hands feel, from the ONE solved model.
#
# Chain (virtual work), the binder-verified method:
#     F_rack  = SUM_i  M_kingpin,i * (d delta_i / d rack)          [N]
#     T_wheel = F_rack * C / (2*pi)                                 [N*m]
#       M_kingpin,i = |Mz_tire(alpha_i, Fz_i, camber_i)|  (TTC aligning torque)
#                   + |Fy_i| * mechanical_trail            (caster contribution)
#       d delta_i / d rack : per-wheel linkage ratio PROBED from the real front
#           solver; its inverse is the EFFECTIVE steering arm (mm/rad).
# Also reports the PHYSICAL arm (kingpin axis -> outboard tie-rod mount, the
# machined lever) and the tie-rod force on it.  The GUI Analysis plot and the
# binder generator both call compute_steering_effort — one implementation.
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_LAT_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.3)


def linkage_gains(front_hp: dict, pushrod_body: str):
    """Per-wheel d(road-wheel)/d(rack) at centre, rad/m, probed from the real
    front solver (translate tie_rod_inner in X, solve, read toe)."""
    from vahan.hardpoints import DoubleWishboneHardpoints
    from vahan.solver import SuspensionConstraints
    from vahan.kinematics import KinematicMetrics

    hp0 = {k: np.asarray(v, float).copy() for k, v in front_hp.items()
           if v is not None and np.all(np.isfinite(np.asarray(v, float)))}
    d_tr = hp0['tie_rod_outer'] - hp0['tie_rod_inner']
    tierod_sq = float(d_tr @ d_tr)

    def toe_at(rack_m, side):
        r = rack_m if side == 'L' else -rack_m
        hp = {k: v.copy() for k, v in hp0.items()}
        hp['tie_rod_inner'] = hp0['tie_rod_inner'] + np.array([r, 0., 0.])
        solver = SuspensionConstraints(
            DoubleWishboneHardpoints(**{k: np.array(v, float) for k, v in hp.items()}),
            tierod_len_sq=tierod_sq, pushrod_body=pushrod_body,
            damper_actuation='pushrod', damper_body=pushrod_body)
        st = solver.solve(0.0)
        m = KinematicMetrics(st, 'left')
        return np.radians(float(m.toe)) * (1.0 if side == 'L' else -1.0)

    h = 0.003                                     # +/-3 mm central difference
    gL = (toe_at(+h, 'L') - toe_at(-h, 'L')) / (2 * h)
    gR = (toe_at(+h, 'R') - toe_at(-h, 'R')) / (2 * h)
    return float(gL), float(gR)


def compute_steering_effort(ss_solver, front_hp: dict, pushrod_body: str,
                            tire_model, steer_cfg: Optional[dict] = None,
                            lat_grid=DEFAULT_LAT_GRID,
                            front_solver=None) -> dict:
    """Steering-effort report from the solved model.

    ss_solver: SteadyStateSolver (per-wheel Fy/Fz/camber at each lat g).
    front_hp:  front hardpoint dict.  pushrod_body: topology damper mount.
    tire_model: front TTC tire (Mz + slip_angle_for_Fy).
    steer_cfg: config 'steer' block {rack_travel_per_rev_mm, total_rack_travel_mm}.
    front_solver: solved FL corner solver, for trail/scrub + the physical arm.
    """
    from vahan.kinematics import KinematicMetrics

    steer_cfg = steer_cfg or {}
    C_mm = float(steer_cfg.get('rack_travel_per_rev_mm', 0.0) or 0.0)
    stroke_mm = float(steer_cfg.get('total_rack_travel_mm', 0.0) or 0.0)

    gL, gR = linkage_gains(front_hp, pushrod_body)
    arm_L = 1.0 / gL * 1000.0 if gL else float('nan')
    arm_R = 1.0 / gR * 1000.0 if gR else float('nan')
    arm_eff = 0.5 * (abs(arm_L) + abs(arm_R))

    t_mech, scrub = 0.0136, 0.0079
    st0 = None
    if front_solver is not None:
        try:
            st0 = front_solver.solve(0.0)
            m0 = KinematicMetrics(st0, 'left')
            if np.isfinite(m0.mechanical_trail):
                t_mech = float(m0.mechanical_trail)
            if np.isfinite(m0.scrub_radius):
                scrub = float(m0.scrub_radius)
        except Exception:
            st0 = None

    def mz_tire(alpha_deg, Fz, camber_deg):
        try:
            return abs(float(tire_model.Mz(alpha_deg, max(Fz, 1.0), camber_deg)))
        except Exception:
            return float('nan')

    def alpha_for(Fy, Fz, camber_deg):
        try:
            a = float(tire_model.slip_angle_for_Fy(abs(Fy), max(Fz, 1.0), camber_deg))
            return abs(a) if np.isfinite(a) else 6.0
        except Exception:
            return 6.0

    def F_rack_at(lat_g):
        r = ss_solver.solve(float(lat_g), 0.0)
        total = 0.0
        parts = {}
        for lbl, g in (('FL', gL), ('FR', gR)):
            Fy = float(r.Fy.get(lbl, 0.0)); Fz = float(r.Fz.get(lbl, 1.0))
            cam = float(r.camber.get(lbl, 0.0)) if hasattr(r, 'camber') else 0.0
            al = alpha_for(Fy, Fz, cam)
            mz = mz_tire(al, Fz, cam)
            if not np.isfinite(mz):
                mz = abs(Fy) * 0.012          # fallback pneumatic trail 12 mm
            mk = mz + abs(Fy) * t_mech
            total += mk * abs(g)
            parts[lbl] = dict(Fy=Fy, Fz=Fz, alpha_deg=al, Mz=mz, M_kp=mk)
        return total, parts

    curve, parts_at = [], {}
    for lg in lat_grid:
        F, parts = F_rack_at(lg)
        curve.append((float(lg), float(F)))
        parts_at[float(lg)] = parts
    F_arr = [f for _, f in curve]
    F_max = max(F_arr)
    peak_g = curve[int(np.argmax(F_arr))][0]

    arm_phys = arm_torq = f_tr_max = float('nan')
    if st0 is not None:
        P = lambda k: np.asarray(getattr(st0, k), float)
        lo, uo, tro, tri = P('lca_outer'), P('uca_outer'), P('tr_outer'), P('tr_inner')
        kp = (uo - lo) / np.linalg.norm(uo - lo)
        v = tro - lo
        arm_phys = float(np.linalg.norm(v - (v @ kp) * kp) * 1000.0)
        utr = (tri - tro) / np.linalg.norm(tri - tro)
        arm_torq = float(abs(np.cross(tro - lo, utr) @ kp) * 1000.0)
        mk_max = max((p['M_kp'] for p in parts_at[peak_g].values()), default=float('nan'))
        if np.isfinite(mk_max) and arm_torq > 0:
            f_tr_max = float(mk_max / (arm_torq / 1000.0))

    out = dict(
        arm_eff_mm_L=round(arm_L, 1), arm_eff_mm_R=round(arm_R, 1),
        arm_eff_mm=round(arm_eff, 1),
        arm_physical_mm=(round(arm_phys, 1) if np.isfinite(arm_phys) else None),
        arm_torque_mm=(round(arm_torq, 1) if np.isfinite(arm_torq) else None),
        tie_rod_force_max_N=(round(f_tr_max, 0) if np.isfinite(f_tr_max) else None),
        mech_trail_mm=round(t_mech * 1000.0, 1), scrub_mm=round(scrub * 1000.0, 1),
        F_rack_vs_latg=[(g, round(f, 1)) for g, f in curve],
        F_max_N=round(F_max, 1), peak_latg=peak_g,
        C_mm_per_rev=(C_mm or None), stroke_total_mm=(stroke_mm or None),
    )
    if C_mm:
        c = C_mm / (2.0 * np.pi * 1000.0)
        out['T_vs_latg'] = [(g, round(f * c, 2)) for g, f in curve]
        out['T_max_Nm'] = round(F_max * c, 2)
        out['ratio_to1'] = round((arm_eff / 1000.0) / (C_mm / 1000.0 / (2 * np.pi)), 2)
    if stroke_mm and np.isfinite(arm_eff):
        wb = float(getattr(getattr(ss_solver, '_veh', None), 'wheelbase_m', 1.537))
        delta_need = np.degrees(np.arctan(wb / 3.93)) + 3.0   # FSAE hairpin + slip margin
        need = (arm_eff / 1000.0) * np.radians(delta_need) * 1000.0
        out['delta_need_deg'] = round(delta_need, 1)
        out['stroke_need_side_mm'] = round(need, 1)
        out['stroke_side_mm'] = round(stroke_mm / 2.0, 1)
        out['stroke_fits'] = bool(need <= stroke_mm / 2.0)
        if C_mm:
            out['lock_to_lock_deg'] = round(2.0 * need / C_mm * 360.0, 0)
    return out
