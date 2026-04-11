"""
vahan/loads.py — Component force calculations for suspension members.

Given per-corner wheel loads (Fz, Fy, Fx) and the 3D geometry from the
kinematic solver, computes:
  - Axial forces in every suspension link (6×6 equilibrium)
  - Ball joint resultant forces in V (up+) and H (fwd+)
  - Bearing loads at inner/outer bearings in V and H
  - Brake caliper mounting bolt forces in V and H
  - Brake system forces (clamping, line pressure)

Axis convention: X=lateral(outboard+), Y=longitudinal(fwd+), Z=up(+)
Force sign convention throughout:
    V = vertical,     positive = UP
    H = longitudinal, positive = FORWARD (towards nose)
"""

from dataclasses import dataclass, field
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BrakeParams:
    """Brake system parameters (per caliper)."""
    pad_mu: float = 0.45            # pad friction coefficient
    piston_area_mm2: float = 793.5  # per caliper (1.23 in²)
    pad_radius_mm: float = 94.4     # effective radius from wheel center
    num_pistons: int = 1            # pistons per caliper side
    caliper_bolt_spacing_mm: float = 60.0  # vertical distance between two mounting bolts


@dataclass
class UprightParams:
    """Upright/bearing geometry."""
    bearing_spacing_mm: float = 50.0   # inner-to-outer bearing distance along spindle
    cp_offset_mm: float = 30.0        # contact patch plane offset from inner bearing (along spindle)
    caliper_angle_deg: float = 45.0   # caliper position: degrees from top of disc, CW from outboard view


@dataclass
class ComponentLoads:
    """Per-corner component forces.

    All directional forces use V/H convention:
        V = vertical,     positive = UP
        H = longitudinal, positive = FORWARD (towards nose)
    Axial member forces:
        Positive = tension (pulled apart)
        Negative = compression (pushed together)
    """
    # Wheel loads (inputs)
    Fz_N: float = 0.0    # vertical ground reaction (up+)
    Fy_N: float = 0.0    # lateral cornering force
    Fx_N: float = 0.0    # longitudinal force (fwd+)

    # ── Suspension member axial forces (N) — along each link ──────
    uca_front_N: float = 0.0
    uca_rear_N: float = 0.0
    lca_front_N: float = 0.0
    lca_rear_N: float = 0.0
    tierod_N: float = 0.0
    pushrod_N: float = 0.0

    # ── Ball joint resultants (V=up+, H=fwd+) ────────────────────
    # Vector sum of both arm forces at the ball joint
    uca_bj_V: float = 0.0
    uca_bj_H: float = 0.0
    lca_bj_V: float = 0.0
    lca_bj_H: float = 0.0
    tierod_bj_V: float = 0.0
    tierod_bj_H: float = 0.0
    pushrod_bj_V: float = 0.0
    pushrod_bj_H: float = 0.0

    # Spring force (N, compression positive)
    spring_force_N: float = 0.0

    # ── Bearing loads (V=up+, H=fwd+) ────────────────────────────
    bearing_inner_V: float = 0.0
    bearing_inner_H: float = 0.0
    bearing_outer_V: float = 0.0
    bearing_outer_H: float = 0.0

    # ── Caliper mounting bolt loads (V=up+, H=fwd+) ──────────────
    caliper_upper_V: float = 0.0
    caliper_upper_H: float = 0.0
    caliper_lower_V: float = 0.0
    caliper_lower_H: float = 0.0

    # ── Brake ─────────────────────────────────────────────────────
    brake_torque_Nm: float = 0.0
    caliper_clamp_N: float = 0.0
    line_pressure_MPa: float = 0.0

    # Solve quality
    residual: float = 0.0   # equilibrium residual (should be ~0)


# ═══════════════════════════════════════════════════════════════════════════
#  MEMBER FORCE SOLVER
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def compute_corner_loads(
    state,              # SolvedState from kinematic solver
    Fz: float,          # vertical ground reaction (N, positive up)
    Fy: float,          # lateral force at contact patch (N)
    Fx: float,          # longitudinal force at contact patch (N, positive fwd)
    brake_torque: float,    # brake torque at wheel (Nm)
    brake_params: BrakeParams,
    upright_params: UprightParams,
    wheel_radius_m: float = 0.203,
    motion_ratio: float = 1.0,
) -> ComponentLoads:
    """
    Compute all member forces at one suspension corner.

    Uses 3D static equilibrium on the upright free body.
    The upright is connected to 6 members (UCA front, UCA rear, LCA front,
    LCA rear, tie rod, pushrod), each carrying axial force only.
    6 unknowns, 6 equations (3 force + 3 moment) → exact solution.

    After solving the 6×6 system, decomposes forces into V/H at ball joints,
    computes bearing loads and caliper bolt forces with directional components.
    """
    result = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=brake_torque)

    # ── Extract 3D positions from solved state ──────────────────────
    uca_f = np.asarray(state.uca_front, dtype=float)
    uca_r = np.asarray(state.uca_rear, dtype=float)
    uca_o = np.asarray(state.uca_outer, dtype=float)
    lca_f = np.asarray(state.lca_front, dtype=float)
    lca_r = np.asarray(state.lca_rear, dtype=float)
    lca_o = np.asarray(state.lca_outer, dtype=float)
    tr_i  = np.asarray(state.tr_inner, dtype=float)
    tr_o  = np.asarray(state.tr_outer, dtype=float)
    push_o = np.asarray(state.pushrod_outer, dtype=float)
    push_i = np.asarray(state.pushrod_inner, dtype=float)
    wc     = np.asarray(state.wheel_center, dtype=float)

    # Contact patch = wheel center projected to ground (Z=0)
    cp = np.array([wc[0], wc[1], 0.0])

    # ── Unit vectors along each member (outboard ← inboard) ────────
    u = np.zeros((6, 3))
    u[0] = _normalize(uca_o - uca_f)     # UCA front member
    u[1] = _normalize(uca_o - uca_r)     # UCA rear member
    u[2] = _normalize(lca_o - lca_f)     # LCA front member
    u[3] = _normalize(lca_o - lca_r)     # LCA rear member
    u[4] = _normalize(tr_o  - tr_i)      # Tie rod
    u[5] = _normalize(push_o - push_i)   # Pushrod

    # ── Position vectors from LCA outer ball joint ──────────────────
    r_uca  = uca_o  - lca_o
    r_tr   = tr_o   - lca_o
    r_push = push_o - lca_o
    r_cp   = cp     - lca_o

    # ── Applied force at contact patch ──────────────────────────────
    F_app = np.array([Fx, Fy, Fz])

    # Brake torque as moment about wheel spin axis
    spin = _normalize(np.asarray(state.spin_axis, dtype=float))
    M_brake = brake_torque * spin

    # ── Build 6×6 system: A @ forces = b ────────────────────────────
    A = np.zeros((6, 6))

    # Force equilibrium (rows 0-2): ΣF = 0
    for j in range(6):
        A[0:3, j] = u[j]

    # Moment about LCA outer (rows 3-5): ΣM_lca_o = 0
    A[3:6, 0] = np.cross(r_uca, u[0])
    A[3:6, 1] = np.cross(r_uca, u[1])
    # LCA front/rear connect TO lca_outer → zero moment arm
    A[3:6, 4] = np.cross(r_tr, u[4])
    A[3:6, 5] = np.cross(r_push, u[5])

    # RHS
    b = np.zeros(6)
    b[0:3] = -F_app
    b[3:6] = -(np.cross(r_cp, F_app) + M_brake)

    # ── Solve ───────────────────────────────────────────────────────
    try:
        forces = np.linalg.solve(A, b)
        residual = np.linalg.norm(A @ forces - b)
    except np.linalg.LinAlgError:
        forces, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residual = np.linalg.norm(A @ forces - b)

    result.uca_front_N = float(forces[0])
    result.uca_rear_N  = float(forces[1])
    result.lca_front_N = float(forces[2])
    result.lca_rear_N  = float(forces[3])
    result.tierod_N    = float(forces[4])
    result.pushrod_N   = float(forces[5])
    result.residual    = float(residual)

    # ── Ball joint resultants in V (Z, up+) and H (Y, fwd+) ────────
    # UCA: vector sum of both arm forces at the outer ball joint
    F_uca_vec = forces[0] * u[0] + forces[1] * u[1]
    result.uca_bj_V = float(F_uca_vec[2])   # Z = vertical
    result.uca_bj_H = float(F_uca_vec[1])   # Y = longitudinal

    # LCA: vector sum of both arm forces at the outer ball joint
    F_lca_vec = forces[2] * u[2] + forces[3] * u[3]
    result.lca_bj_V = float(F_lca_vec[2])
    result.lca_bj_H = float(F_lca_vec[1])

    # Tie rod: single link force decomposed
    F_tr_vec = forces[4] * u[4]
    result.tierod_bj_V = float(F_tr_vec[2])
    result.tierod_bj_H = float(F_tr_vec[1])

    # Pushrod: single link force decomposed
    F_push_vec = forces[5] * u[5]
    result.pushrod_bj_V = float(F_push_vec[2])
    result.pushrod_bj_H = float(F_push_vec[1])

    # ── Spring force from pushrod × motion ratio ────────────────────
    if abs(motion_ratio) > 0.01:
        result.spring_force_N = abs(forces[5]) * motion_ratio

    # ── Bearing loads (V/H at inner and outer) ──────────────────────
    _compute_bearing_loads(result, brake_params, upright_params, wheel_radius_m)

    # ── Caliper bolt forces (V/H at upper and lower) ───────────────
    _compute_caliper_bolt_loads(result, brake_params, upright_params)

    # ── Brake system (clamp, pressure) ──────────────────────────────
    _compute_brake_forces(result, brake_params)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  BEARING LOADS  (moment equilibrium on the hub free body)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_bearing_loads(result: ComponentLoads, bp: BrakeParams,
                           up: UprightParams, wheel_radius_m: float):
    """
    Bearing loads from moment equilibrium on the hub/wheel assembly.

    Hub free body sees:
      1. Contact patch forces Fz (up) and Fx (fwd) at lateral offset d
         from inner bearing along the spindle, Rr below spindle axis.
      2. Brake pad friction on disc — tangential at caliper angle θ,
         magnitude = brake_torque / pad_radius.

    Outputs V (up+) and H (fwd+) at each bearing.
    """
    l1 = up.bearing_spacing_mm / 1000.0   # m (bearing spacing)
    d  = up.cp_offset_mm / 1000.0         # m (CP offset from inner bearing)
    theta = np.radians(up.caliper_angle_deg)
    r_pad = bp.pad_radius_mm / 1000.0     # m

    if l1 < 0.001:
        return

    Fz = result.Fz_N
    Fx = result.Fx_N
    T  = result.brake_torque_Nm

    # ── Brake friction on disc (tangential, acts on the hub) ─────
    # At caliper angle θ from top (CW from outboard), friction on disc
    # opposes forward rotation: direction = (-cos θ, sin θ) in (H, V)
    F_fric = T / r_pad if r_pad > 0.001 else 0.0
    F_fric_V = F_fric * np.sin(theta)     # vertical component on hub (up+)
    F_fric_H = -F_fric * np.cos(theta)    # longitudinal component on hub (fwd+)

    # ── Total forces on hub to distribute between bearings ───────
    total_V = Fz + F_fric_V
    total_H = Fx + F_fric_H

    # Simple beam: moment about inner bearing → outer bearing force
    result.bearing_outer_V = float(total_V * d / l1)
    result.bearing_inner_V = float(total_V * (l1 - d) / l1)
    result.bearing_outer_H = float(total_H * d / l1)
    result.bearing_inner_H = float(total_H * (l1 - d) / l1)


# ═══════════════════════════════════════════════════════════════════════════
#  CALIPER MOUNTING BOLT FORCES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_caliper_bolt_loads(result: ComponentLoads, bp: BrakeParams,
                                up: UprightParams):
    """
    Forces at the two caliper mounting bolts (upper and lower).

    The brake pads grip the disc with tangential friction.  The reaction
    on the caliper goes through two mounting bolts.

    Two contributions at each bolt:
      1. Direct shear — total friction shared equally.
      2. Torque couple — brake torque reacted as a horizontal force pair
         between the vertically-spaced bolts.
    """
    T = result.brake_torque_Nm
    r_pad = bp.pad_radius_mm / 1000.0
    s = bp.caliper_bolt_spacing_mm / 1000.0
    theta = np.radians(up.caliper_angle_deg)

    if r_pad < 0.001 or s < 0.001 or T < 1e-6:
        return  # no brake torque → bolts carry zero

    F_friction = T / r_pad

    # Reaction on caliper (Newton's 3rd — opposite of friction on disc)
    F_cal_H = F_friction * np.cos(theta)
    F_cal_V = -F_friction * np.sin(theta)

    # 1. Direct shear — equal split
    V_dir = F_cal_V / 2.0
    H_dir = F_cal_H / 2.0

    # 2. Torque couple — brake torque spins caliper forward at top,
    #    upper bolt pushed fwd, lower bolt pushed rearward
    H_couple = T / s

    result.caliper_upper_V = float(V_dir)
    result.caliper_upper_H = float(H_dir + H_couple)
    result.caliper_lower_V = float(V_dir)
    result.caliper_lower_H = float(H_dir - H_couple)


# ═══════════════════════════════════════════════════════════════════════════
#  BRAKE FORCES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_brake_forces(result: ComponentLoads, bp: BrakeParams):
    """
    Brake caliper clamping force and line pressure from known brake torque.

    brake_torque = clamp_force × pad_mu × pad_radius × 2 (both pads)
    line_pressure = clamp_force / piston_area
    """
    T = result.brake_torque_Nm
    r_pad = bp.pad_radius_mm / 1000  # m

    if bp.pad_mu > 0 and r_pad > 0:
        result.caliper_clamp_N = T / (bp.pad_mu * r_pad * 2)

        A_piston = bp.piston_area_mm2  # mm²
        if A_piston > 0:
            result.line_pressure_MPa = result.caliper_clamp_N / A_piston  # N/mm² = MPa


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: COMPUTE ALL 4 CORNERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_corners(
    solvers: dict,          # {'FL': SuspensionConstraints, ...}
    dyn_result,             # SteadyStateResult
    brake_params_f: BrakeParams,
    brake_params_r: BrakeParams,
    upright_params_f: UprightParams,
    upright_params_r: UprightParams,
    wheel_radius_m: float = 0.203,
    motion_ratio_f: float = 1.0,
    motion_ratio_r: float = 1.0,
) -> dict:
    """
    Compute component loads for all 4 corners from a dynamics result.

    Returns dict: {'FL': ComponentLoads, 'FR': ..., 'RL': ..., 'RR': ...}
    """
    results = {}
    for label in ['FL', 'FR', 'RL', 'RR']:
        Fz = dyn_result.Fz.get(label, 0)
        Fy = dyn_result.Fy.get(label, 0)
        Fx = dyn_result.Fx.get(label, 0)
        bt = dyn_result.brake_torque.get(label, 0)

        is_front = label[0] == 'F'
        bp = brake_params_f if is_front else brake_params_r
        up = upright_params_f if is_front else upright_params_r
        mr = motion_ratio_f if is_front else motion_ratio_r

        solver = solvers.get(label)
        if solver is None:
            # No kinematic solver — still compute bearing / brake forces
            cl = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=bt)
            _compute_bearing_loads(cl, bp, up, wheel_radius_m)
            _compute_caliper_bolt_loads(cl, bp, up)
            _compute_brake_forces(cl, bp)
            results[label] = cl
            continue

        # Get kinematic state at this corner's travel
        travel_m = dyn_result.travel.get(label, 0) / 1000  # mm → m
        try:
            state = solver.solve(travel_m)
        except Exception:
            # Kinematic solve failed — still compute bearing / brake forces
            cl = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=bt)
            _compute_bearing_loads(cl, bp, up, wheel_radius_m)
            _compute_caliper_bolt_loads(cl, bp, up)
            _compute_brake_forces(cl, bp)
            results[label] = cl
            continue

        results[label] = compute_corner_loads(
            state, Fz, Fy, Fx, bt, bp, up, wheel_radius_m, mr)

    return results
