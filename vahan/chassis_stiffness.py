"""
Chassis torsional stiffness vs lateral load transfer — the three-spring model.

The sprung mass is split into a FRONT and a REAR lump, each with its own
height and its own roll centre.  Each lump's roll moment is reacted by that
axle's roll stiffness, by the other axle THROUGH the chassis torsion spring,
or by both:

        ground ──k_f── [front body node] ──k_c── [rear body node] ──k_r── ground
                              ↑ M_f = m_sf·a_y·(h_f − z_F)   ↑ M_r = m_sr·a_y·(h_r − z_R)

Node balance (θ = roll angle of each end of the chassis, rad):

        k_f·θ_f + k_c·(θ_f − θ_r) = M_f
        k_r·θ_r + k_c·(θ_r − θ_f) = M_r

Elastic transfer per axle is k·θ/track; the geometric (through the roll
centre) and unsprung terms are the same as SteadyStateSolver._compute_load_transfer.
Solving the 2×2 gives, for the front axle, exactly the closed form the team uses:

        ΔF_ef = (1/t_f) · [ k_f / (k_f + k_r·k_c/(k_r+k_c)) · M_f
                          + (k_f·k_c/(k_f+k_c)) / (k_r + k_f·k_c/(k_f+k_c)) · M_r ]

(the 2×2 and the closed form are gated against each other in test_one_model.py).
k_c → ∞ recovers the rigid-chassis split k_f/(k_f+k_r); k_c → 0 makes each
axle react only its own lump.

Required chassis stiffness: `kc_for_lltd` inverts the model for a target total
front load-transfer share; `kc_for_tolerance` gives the smallest k_c whose LLTD
sits within a tolerance of the rigid value, evaluated at the LARGEST front/rear
roll-stiffness offset you intend to tune to (at the design split the chassis
carries almost nothing and every tolerance is met by any k_c — the classic trap).

All stiffnesses in N·m/rad, masses kg, lengths m, a_y in g.  Pure numpy, no GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math

import numpy as np

G = 9.81
NM_PER_RAD_PER_NM_PER_DEG = 180.0 / math.pi      # N·m/deg → N·m/rad
FTLBF_PER_NM = 0.737562149                       # 1 N·m = 0.7376 ft·lbf


def nm_deg_to_nm_rad(k):  return float(k) * NM_PER_RAD_PER_NM_PER_DEG
def nm_rad_to_nm_deg(k):  return float(k) / NM_PER_RAD_PER_NM_PER_DEG
def nm_deg_to_ftlbf_deg(k): return float(k) * FTLBF_PER_NM
def ftlbf_deg_to_nm_deg(k): return float(k) / FTLBF_PER_NM


@dataclass
class ChassisInputs:
    """Everything the three-spring model needs.  Build one with
    `inputs_from_vehicle` (defaults from the solved car) and edit the lumps."""
    m_sf_kg: float            # front sprung lump
    m_sr_kg: float            # rear sprung lump
    h_f_m: float              # height of the front lump CG
    h_r_m: float              # height of the rear lump CG
    z_f_m: float              # front roll-centre height
    z_r_m: float              # rear roll-centre height
    k_f_Nm_rad: float         # front roll stiffness (springs + bar)
    k_r_Nm_rad: float         # rear roll stiffness
    k_c_Nm_rad: float         # chassis torsional stiffness, hub to hub
    t_f_m: float
    t_r_m: float
    m_uf_kg: float = 0.0      # unsprung per AXLE (both wheels), as VehicleParams
    m_ur_kg: float = 0.0
    h_u_m: float = 0.0        # unsprung CG height
    wheelbase_m: float = 1.53
    a_s_m: float = 0.0        # sprung CG from the front axle (information only)

    @property
    def m_s_kg(self) -> float:
        return self.m_sf_kg + self.m_sr_kg

    @property
    def d_f_m(self) -> float:
        """Front lump height above the front roll centre (Liam's d_f)."""
        return self.h_f_m - self.z_f_m

    @property
    def d_r_m(self) -> float:
        return self.h_r_m - self.z_r_m

    @property
    def lltd_rigid(self) -> float:
        """Total front LT share with a rigid chassis (k_c → ∞)."""
        return solve(replace(self, k_c_Nm_rad=float('inf'))).lltd_front

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChassisResult:
    a_y_g: float
    theta_f_rad: float
    theta_r_rad: float
    twist_rad: float               # θ_f − θ_r (front end rolls more: +)
    chassis_torque_Nm: float       # k_c · twist (what the frame carries)
    M_f_Nm: float
    M_r_Nm: float
    elastic_front_N: float         # one-side delta, + = to the outside wheel
    elastic_rear_N: float
    geometric_front_N: float
    geometric_rear_N: float
    unsprung_front_N: float
    unsprung_rear_N: float

    @property
    def total_front_N(self):  return self.elastic_front_N + self.geometric_front_N + self.unsprung_front_N
    @property
    def total_rear_N(self):   return self.elastic_rear_N + self.geometric_rear_N + self.unsprung_rear_N
    @property
    def lltd_front(self) -> float:
        tot = self.total_front_N + self.total_rear_N
        return self.total_front_N / tot if abs(tot) > 1e-12 else 0.5
    @property
    def elastic_share_front(self) -> float:
        tot = self.elastic_front_N + self.elastic_rear_N
        return self.elastic_front_N / tot if abs(tot) > 1e-12 else 0.5
    @property
    def twist_deg(self):  return math.degrees(self.twist_rad)
    @property
    def roll_f_deg(self): return math.degrees(self.theta_f_rad)
    @property
    def roll_r_deg(self): return math.degrees(self.theta_r_rad)
    @property
    def alpha(self) -> float:
        """Chassis twist / mean body roll (the α of the team's rule of thumb)."""
        mean = 0.5 * (self.theta_f_rad + self.theta_r_rad)
        return self.twist_rad / mean if abs(mean) > 1e-12 else 0.0


def _node_angles(inp: ChassisInputs, M_f: float, M_r: float):
    """Solve the two node balances.  k_c = inf → both ends roll together."""
    k_f, k_r, k_c = float(inp.k_f_Nm_rad), float(inp.k_r_Nm_rad), float(inp.k_c_Nm_rad)
    if not math.isfinite(k_c):
        th = (M_f + M_r) / (k_f + k_r) if (k_f + k_r) > 0 else 0.0
        return th, th
    A = np.array([[k_f + k_c, -k_c], [-k_c, k_r + k_c]], float)
    if abs(np.linalg.det(A)) < 1e-12:
        return 0.0, 0.0
    th_f, th_r = np.linalg.solve(A, np.array([M_f, M_r], float))
    return float(th_f), float(th_r)


def solve(inp: ChassisInputs, a_y_g: float = 1.0) -> ChassisResult:
    """Load transfer at lateral acceleration a_y (g) with chassis compliance."""
    ay = float(a_y_g) * G
    M_f = inp.m_sf_kg * ay * inp.d_f_m
    M_r = inp.m_sr_kg * ay * inp.d_r_m
    th_f, th_r = _node_angles(inp, M_f, M_r)
    twist = th_f - th_r
    k_c = float(inp.k_c_Nm_rad)
    torque = (k_c * twist) if math.isfinite(k_c) else (M_f - inp.k_f_Nm_rad * th_f)
    return ChassisResult(
        a_y_g=float(a_y_g), theta_f_rad=th_f, theta_r_rad=th_r, twist_rad=twist,
        chassis_torque_Nm=float(torque), M_f_Nm=M_f, M_r_Nm=M_r,
        elastic_front_N=inp.k_f_Nm_rad * th_f / inp.t_f_m,
        elastic_rear_N=inp.k_r_Nm_rad * th_r / inp.t_r_m,
        geometric_front_N=inp.m_sf_kg * ay * inp.z_f_m / inp.t_f_m,
        geometric_rear_N=inp.m_sr_kg * ay * inp.z_r_m / inp.t_r_m,
        unsprung_front_N=inp.m_uf_kg * ay * inp.h_u_m / inp.t_f_m,
        unsprung_rear_N=inp.m_ur_kg * ay * inp.h_u_m / inp.t_r_m,
    )


def closed_form_front_elastic_N(inp: ChassisInputs, a_y_g: float = 1.0) -> float:
    """The team's closed form for the FRONT elastic transfer (one-side delta),
    kept as an independent check of `solve` — not used by the GUI."""
    ay = float(a_y_g) * G
    k_f, k_r, k_c = inp.k_f_Nm_rad, inp.k_r_Nm_rad, inp.k_c_Nm_rad
    M_f = inp.m_sf_kg * ay * inp.d_f_m
    M_r = inp.m_sr_kg * ay * inp.d_r_m
    if not math.isfinite(k_c):
        return (k_f / (k_f + k_r)) * (M_f + M_r) / inp.t_f_m
    s_rc = k_r * k_c / (k_r + k_c)          # rear springs through the chassis
    s_fc = k_f * k_c / (k_f + k_c)          # front springs through the chassis
    own = k_f / (k_f + s_rc)
    via = s_fc / (k_r + s_fc)
    return (own * M_f + via * M_r) / inp.t_f_m


def lltd_curve(inp: ChassisInputs, k_c_values, a_y_g: float = 1.0) -> np.ndarray:
    """Total front LT share for each chassis stiffness (N·m/rad)."""
    return np.array([solve(replace(inp, k_c_Nm_rad=float(k)), a_y_g).lltd_front
                     for k in np.asarray(k_c_values, float)])


def kc_for_lltd(inp: ChassisInputs, target_lltd_front: float,
                lo: float = 1.0, hi: float = 1e8) -> float:
    """Smallest k_c (N·m/rad) that puts the total front LT share at the target.
    LLTD is monotone in k_c between the k_c=0 value and the rigid value; NaN
    if the target lies outside that range (no chassis stiffness can reach it —
    change the bars, not the frame)."""
    f0 = solve(replace(inp, k_c_Nm_rad=0.0)).lltd_front
    f1 = inp.lltd_rigid
    t = float(target_lltd_front)
    lo_v, hi_v = (f0, f1) if f0 <= f1 else (f1, f0)
    if not (lo_v - 1e-9 <= t <= hi_v + 1e-9):
        return float('nan')
    if abs(f1 - f0) < 1e-12:
        return 0.0
    # bisect in log k_c on the signed residual
    def r(k): return solve(replace(inp, k_c_Nm_rad=k)).lltd_front - t
    a, b = math.log(lo), math.log(hi)
    ra, rb = r(math.exp(a)), r(math.exp(b))
    if ra == 0.0:
        return lo
    if ra * rb > 0:
        return float('nan')
    for _ in range(80):
        m = 0.5 * (a + b)
        rm = r(math.exp(m))
        if ra * rm <= 0:
            b, rb = m, rm
        else:
            a, ra = m, rm
    return float(math.exp(0.5 * (a + b)))


def kc_for_tolerance(inp: ChassisInputs, tol_lltd: float = 0.01) -> float:
    """Smallest k_c whose LLTD is within `tol_lltd` (fraction, 0.01 = 1 point)
    of the rigid-chassis LLTD for THESE inputs.  Evaluate on the inputs with
    the largest front/rear offset you will tune to (see module doc)."""
    f1 = inp.lltd_rigid
    f0 = solve(replace(inp, k_c_Nm_rad=0.0)).lltd_front
    if abs(f1 - f0) <= tol_lltd:
        return 0.0
    target = f1 - math.copysign(tol_lltd, f1 - f0)
    return kc_for_lltd(inp, target)


def k_series(inp: ChassisInputs) -> float:
    """Front and rear roll stiffness in SERIES (N·m/rad): what the chassis works
    against when it passes torque between the axles.  Not the car's roll
    stiffness — that is the parallel sum k_f + k_r."""
    kf, kr = float(inp.k_f_Nm_rad), float(inp.k_r_Nm_rad)
    return kf * kr / (kf + kr) if (kf + kr) > 0 else 0.0


def kc_for_gain(inp: ChassisInputs, target: float) -> float:
    """Chassis stiffness (N·m/rad) that gives gain = target, from the closed form
    gain = k_c / (k_c + k_series): k_c = target / (1 - target) · k_series."""
    g = float(target)
    if not (0.0 < g < 1.0):
        return float('nan') if g >= 1.0 else 0.0
    return g / (1.0 - g) * k_series(inp)


def gain(inp: ChassisInputs) -> float:
    """Fraction of a rigid-chassis LLTD offset (from the k_c = 0 split) that the
    compliant chassis delivers: (LLTD(k_c) − LLTD(0)) / (LLTD(∞) − LLTD(0))."""
    f0 = solve(replace(inp, k_c_Nm_rad=0.0)).lltd_front
    f1 = inp.lltd_rigid
    if abs(f1 - f0) < 1e-12:
        return 1.0
    return (solve(inp).lltd_front - f0) / (f1 - f0)


def inputs_from_vehicle(veh, rc_f_m: float, rc_r_m: float, k_c_Nm_rad: float,
                        m_sf_kg=None, m_sr_kg=None, h_f_m=None, h_r_m=None) -> ChassisInputs:
    """Defaults from a VehicleParams (vahan.dynamics): the sprung mass is split
    by the SPRUNG CG station (the same a_sprung SteadyStateSolver uses for the
    geometric term), lumps at veh.sprung_end_cg_heights_m.  Pass any of the four
    lump values to override."""
    L = float(veh.wheelbase_m)
    m_s = float(veh.sprung_mass_kg)
    if m_s > 1e-3:
        a_s = (float(veh.total_mass_kg) * float(veh.cg_to_front_axle_m)
               - float(veh.unsprung_mass_rear_kg) * L) / m_s
    else:
        a_s = float(veh.cg_to_front_axle_m)
    a_s = min(max(a_s, 0.0), L)
    h_s = float(veh.sprung_cg_height_m)
    h_sf, h_sr = (float(x) for x in getattr(veh, 'sprung_end_cg_heights_m', (h_s, h_s)))
    msf = m_s * (L - a_s) / L if m_sf_kg is None else float(m_sf_kg)
    msr = m_s * a_s / L if m_sr_kg is None else float(m_sr_kg)
    return ChassisInputs(
        m_sf_kg=msf, m_sr_kg=msr,
        h_f_m=h_sf if h_f_m is None else float(h_f_m),
        h_r_m=h_sr if h_r_m is None else float(h_r_m),
        z_f_m=float(rc_f_m), z_r_m=float(rc_r_m),
        k_f_Nm_rad=float(veh.roll_stiffness_front_Npm_rad),
        k_r_Nm_rad=float(veh.roll_stiffness_rear_Npm_rad),
        k_c_Nm_rad=float(k_c_Nm_rad),
        t_f_m=float(veh.front_track_m), t_r_m=float(veh.rear_track_m),
        m_uf_kg=float(veh.unsprung_mass_front_kg), m_ur_kg=float(veh.unsprung_mass_rear_kg),
        h_u_m=float(veh.unsprung_cg_height_m), wheelbase_m=L, a_s_m=a_s)


def summary(inp: ChassisInputs, a_y_g: float = 1.0, tune_offset: float = 0.10,
            tol_lltd: float = 0.01) -> dict:
    """One dict with everything the page shows: the result at a_y, the rigid
    and zero-chassis splits, the gain, and the required k_c for the tolerance
    at the design split AND at a bar split moved `tune_offset` toward the
    front (the k_f/k_r ratio scaled so the rigid elastic share rises by that
    much — the 'largest offset you will tune to')."""
    res = solve(inp, a_y_g)
    f_rigid = inp.lltd_rigid
    f_zero = solve(replace(inp, k_c_Nm_rad=0.0), a_y_g).lltd_front
    # tuned inputs: scale k_f so the rigid ELASTIC share is +tune_offset
    kf, kr = inp.k_f_Nm_rad, inp.k_r_Nm_rad
    e_rigid = kf / (kf + kr)
    e_tuned = min(max(e_rigid + tune_offset, 0.02), 0.98)
    kf_t = kr * e_tuned / (1.0 - e_tuned)
    tuned = replace(inp, k_f_Nm_rad=kf_t)
    return {
        'result': res, 'lltd': res.lltd_front, 'lltd_rigid': f_rigid, 'lltd_zero_kc': f_zero,
        'gain': gain(inp), 'alpha': res.alpha, 'twist_deg': res.twist_deg,
        'chassis_torque_Nm': res.chassis_torque_Nm,
        'kc_tol_design': kc_for_tolerance(inp, tol_lltd),
        'kc_tol_tuned': kc_for_tolerance(tuned, tol_lltd),
        'tuned_k_f_Nm_rad': kf_t, 'tuned_lltd_rigid': tuned.lltd_rigid,
        'tuned_gain': gain(tuned), 'tune_offset': tune_offset, 'tol_lltd': tol_lltd,
        'k_series_Nm_rad': k_series(inp),
    }
