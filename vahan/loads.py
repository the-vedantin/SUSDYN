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
    # ── Caliper MOUNT geometry, straight off the caliper drawing ──────────
    # Wilwood GP200-style radial mount.  On the drawing these are:
    #   MOUNT CENTER   2.38 in (60.5 mm)  -> caliper_bolt_spacing_mm  (l5)
    #   MOUNT HEIGHT   1.10 in (27.9 mm)  -> caliper_mount_height_mm
    #   MOUNT OFFSET   0.86 in (21.8 mm)  -> caliper_mount_offset_mm
    # and the drawing's own D1 = (disc diameter / 2) - MOUNT HEIGHT is the radius
    # from the wheel centre to the bolt LINE.  So the mount position is not a free
    # parameter: give it the rotor and these three numbers and it is determined.
    rotor_dia_mm: float = 240.0             # the disc this caliper is mounted to
    rotor_thickness_mm: float = 6.35        # disc width (0.25 in) — thermal mass, not torque
    caliper_bolt_spacing_mm: float = 60.5   # l5: spacing between the two mount bolts
    caliper_mount_height_mm: float = 27.9   # bolt line BELOW the disc OD
    caliper_mount_offset_mm: float = 21.8   # bolt plane offset from the disc face

    @property
    def bolt_line_radius_mm(self) -> float:
        """l3 — radius from the wheel centre to the caliper bolt line (drawing D1)."""
        return max(0.5 * float(self.rotor_dia_mm) - self.caliper_mount_height_mm, 1.0)

    @property
    def bolt_circle_radius_mm(self) -> float:
        """The drawing's tabulated 'A' — each bolt sits half the spacing off the
        radial centre line, so A = sqrt(D1^2 + (l5/2)^2).  Reproduces the table:
        10.00 in disc -> D1 3.90 in, A 4.07 in."""
        return float(np.hypot(self.bolt_line_radius_mm,
                              0.5 * self.caliper_bolt_spacing_mm))

    @property
    def caliper_l4_mm(self) -> float:
        """l4 — pad centre of area offset from the bolt line (Seward Fig 6.15).

        DERIVED, never typed.  Seward's own force balance is
        F_brake = W_long * R_r / (l3 + l4), and F_brake = brake_torque / R_pad,
        so l3 + l4 = R_pad identically.  With l3 fixed by the drawing, l4 follows.
        It used to be a free input defaulting to 25 mm, which contradicted both the
        drawing and that identity, and put the bolts 23 mm from where they sit."""
        return max(self.pad_radius_mm - self.bolt_line_radius_mm, 0.0)


@dataclass
class UprightParams:
    """Upright/bearing geometry."""
    bearing_spacing_mm: float = 50.8   # l1: bearing center-to-center along the spindle (2.00 in)
    # The NEAR (outer) bearing sits this far INBOARD of the wheel centre-line.
    # Both bearings are inboard of the wheel, so the tyre load is OVERHUNG outboard
    # of the pair — the outer bearing carries more than the wheel load and the
    # inner one reacts the other way.  Measured on the outgoing car: 1.55 in.
    bearing_inboard_offset_mm: float = 39.4   # 1.55 in
    cp_offset_mm: float = 30.0        # DEPRECATED: was overloaded as both the outer
                                      # bearing position AND the beam lever; kept so
                                      # old saved configs still load, no longer used.
    caliper_angle_deg: float = 45.0   # manual caliper clock (used only when vertical mounts are OFF)
    # Clock the caliper so its two mount bolts stack VERTICALLY, on the tie-rod
    # side of the wheel — the radial points fore-aft, so the lug spacing runs
    # up-down.  Simplifies the upright (one vertical mounting boss shared with the
    # steering arm face).  When False, caliper_angle_deg sets the clock instead.
    caliper_vertical_mounts: bool = True

    @property
    def cp_to_inner_bearing_mm(self) -> float:
        """Beam lever d: contact-patch plane (the wheel centre-line) to the INNER
        (far) bearing.  = near-bearing offset + spacing, since the patch is
        outboard of both bearings."""
        return self.bearing_inboard_offset_mm + self.bearing_spacing_mm


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
    bearing_axial_N: float = 0.0   # lateral thrust on the bearing pair (= Fy)

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
    LCA rear, tie rod, and the actuation link), each carrying axial force only.
    The 6th member is the pushrod for pushrod/pullrod corners, the live
    pushrod (to the cradle bellcrank) for DECOUPLED corners, or the damper
    line for DIRECT-acting corners.
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

    # ── DIRECT-damper corner: use the damper line as the actuation member ──
    # A direct-acting damper is mounted on the chassis and connected straight
    # to the upright/control-arm — there is NO pushrod.  The kinematic solver
    # therefore collapses the pushrod to a single point (push_o == push_i) and
    # reports the damper endpoints via rocker_spring_pt (moving end, == push_o)
    # and spring_chassis_pt (the fixed chassis end).  Using the zero-length
    # pushrod as the 6th force member makes the upright free body singular, so
    # substitute the real damper line (push_o → spring_chassis_pt): same attach
    # point on the moving body, real axial direction to the chassis.
    if np.linalg.norm(push_o - push_i) < 1e-9:
        chassis_pt = np.asarray(getattr(state, 'spring_chassis_pt', push_i),
                                dtype=float)
        if (chassis_pt.shape == (3,) and np.all(np.isfinite(chassis_pt))
                and np.linalg.norm(push_o - chassis_pt) > 1e-9):
            push_i = chassis_pt

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
    # FRAME: hardpoint coordinates are x = LATERAL, y = LONGITUDINAL,
    # z = up.  Fy (lateral tire force) therefore goes on x and Fx
    # (longitudinal) on y — the old [Fx, Fy, Fz] ordering applied the
    # braking force sideways and the cornering force fore-aft, which
    # broke left/right mirror symmetry under pure braking.
    # Sign: in steady cornering every tire's lateral force points toward
    # the turn centre; with the convention that positive lateral
    # acceleration has the LEFT (+x) side as the outside, that direction
    # is −x for all four corners.
    F_app = np.array([-Fy, Fx, Fz])

    # Brake torque as a moment on the upright.  Both wheels spin in the SAME
    # world sense (rolling forward: omega ~ +x for +y travel), so the brake
    # reaction torque on the upright is the same world vector on BOTH sides.
    # The state's spin_axis points outboard (flips sign left/right) — using
    # it raw gave the right-hand corners a wrong-signed brake moment and a
    # fake left/right asymmetry under pure braking.  Rectify to the
    # consistent world handedness (+x component positive).
    spin = _normalize(np.asarray(state.spin_axis, dtype=float))
    if spin[0] < 0.0:
        spin = -spin
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

    # Sign convention repair: with u pointing inboard->outboard and
    # A·f = -F_app, a POSITIVE solved f pushes the upright outboard, i.e.
    # the member is in COMPRESSION.  The documented (and reported)
    # convention is positive = TENSION, so negate here.  Physical check:
    # a pure-bump case must put the pushrod in compression (negative).
    forces = -forces

    result.uca_front_N = float(forces[0])
    result.uca_rear_N  = float(forces[1])
    result.lca_front_N = float(forces[2])
    result.lca_rear_N  = float(forces[3])
    result.tierod_N    = float(forces[4])
    result.pushrod_N   = float(forces[5])
    result.residual    = float(residual)

    # ── Ball joint resultants in V (Z, up+) and H (Y, fwd+) ────────
    # Force ON the upright from a member with tension T along û(in->out)
    # is −T·û.  (forces[] are tension-positive after the repair above.)
    F_uca_vec = -(forces[0] * u[0] + forces[1] * u[1])
    result.uca_bj_V = float(F_uca_vec[2])   # Z = vertical
    result.uca_bj_H = float(F_uca_vec[1])   # Y = longitudinal

    F_lca_vec = -(forces[2] * u[2] + forces[3] * u[3])
    result.lca_bj_V = float(F_lca_vec[2])
    result.lca_bj_H = float(F_lca_vec[1])

    F_tr_vec = -(forces[4] * u[4])
    result.tierod_bj_V = float(F_tr_vec[2])
    result.tierod_bj_H = float(F_tr_vec[1])

    F_push_vec = -(forces[5] * u[5])
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
    d  = up.cp_to_inner_bearing_mm / 1000.0   # m (patch plane -> inner bearing)
    theta = np.radians(up.caliper_angle_deg)
    r_pad = bp.pad_radius_mm / 1000.0     # m

    if l1 < 0.001:
        return

    Fz = result.Fz_N
    Fx = result.Fx_N
    Fy = result.Fy_N
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

    # ── Overturning moment from the LATERAL force ────────────────
    # Fy acts at the contact patch, one tire radius BELOW the spindle
    # axis: M = Fy·R_tire about the car's longitudinal axis.  Reacted as
    # a VERTICAL couple across the bearing spacing — in hard cornering
    # this term DOMINATES the bearing radial loads.  Fy itself is the
    # axial (thrust) load on the bearing pair.
    M_over = Fy * wheel_radius_m          # N·m
    dV_over = M_over / l1                 # vertical couple magnitude

    # Simple beam: moment about inner bearing → outer bearing force,
    # plus the overturning couple (+ on the outboard row, − inboard for
    # a laterally-loaded outside wheel).
    result.bearing_outer_V = float(total_V * d / l1 + dV_over)
    result.bearing_inner_V = float(total_V * (l1 - d) / l1 - dV_over)
    result.bearing_outer_H = float(total_H * d / l1)
    result.bearing_inner_H = float(total_H * (l1 - d) / l1)
    result.bearing_axial_N = float(Fy)


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
    s = bp.caliper_bolt_spacing_mm / 1000.0          # l5, bolt (lug) spacing
    l4 = max(float(bp.caliper_l4_mm) / 1000.0, 0.0)   # derived from the drawing
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

    # 2. Torque couple.  The couple the BOLT PAIR reacts is the moment of the pad
    #    friction about the BOLT LINE, so the lever is l4 — the offset of the pad
    #    centre of area from that line (Seward Ch.6 Fig 6.15):
    #        H_couple = F_friction * l4 / l5
    #    This used to be T / l5, i.e. the moment about the WHEEL AXLE divided by
    #    the bolt spacing.  That moment is reacted by the whole upright, not by the
    #    two bolts, and using it overstated the couple by r_pad / l4 — about 4.3x
    #    on this car (5,449 N against 1,178 N per bolt at 1.5 g braking).  It also
    #    disagreed with the Loads-page arrows, which already used Seward.
    H_couple = F_friction * l4 / s

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
#  BRAKE SYSTEM CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BrakeSystemParams:
    """System-level brake parameters (pedal + master cylinders)."""
    pedal_ratio: float = 5.0            # mechanical advantage of pedal lever
    mc_bore_front_mm: float = 15.87     # front master cylinder bore (5/8")
    mc_bore_rear_mm: float = 15.87      # rear master cylinder bore (5/8")
    bias_pct_front: float = 65.0        # brake bias % to front (from bias bar)
    tire_radius_m: float = 0.203        # loaded tire radius

    @property
    def mc_area_front_mm2(self) -> float:
        return np.pi * (self.mc_bore_front_mm / 2) ** 2

    @property
    def mc_area_rear_mm2(self) -> float:
        return np.pi * (self.mc_bore_rear_mm / 2) ** 2


@dataclass
class BrakeCornerResult:
    """Brake calculator output for one corner."""
    # Current operating point
    Fz_N: float = 0.0
    tire_mu: float = 0.0               # peak μ used (from tire model or fallback)
    brake_torque_Nm: float = 0.0
    line_pressure_MPa: float = 0.0
    clamp_force_N: float = 0.0
    # At lockup
    lockup_Fx_N: float = 0.0               # Fx that locks this tire
    lockup_torque_Nm: float = 0.0           # brake torque at lockup
    lockup_clamp_N: float = 0.0             # caliper clamp at lockup
    lockup_line_pressure_MPa: float = 0.0   # line pressure at lockup
    lockup_pedal_force_N: float = 0.0       # pedal force that locks this corner
    # Margins
    lockup_margin_pct: float = 0.0          # how far from lockup (100% = at limit)


def compute_brake_system(
    Fz: dict,
    brake_params_f: BrakeParams,
    brake_params_r: BrakeParams,
    system: BrakeSystemParams,
    tire_model=None,
    cambers: dict = None,
) -> dict:
    """
    Compute brake pressures, torques, and lockup limits for all 4 corners.

    Parameters
    ----------
    Fz : dict
        Per-corner vertical loads {'FL': N, 'FR': N, 'RL': N, 'RR': N}.
    brake_params_f, brake_params_r : BrakeParams
        Per-caliper parameters (front / rear).
    system : BrakeSystemParams
        System-level params (pedal, master cylinders, bias).
    tire_model : optional
        Tire model with ``peak_mu(Fz_N, camber_deg)`` — pulls μ per corner
        from TTC data (load-sensitive).  Falls back to 1.5 if None.
    cambers : dict, optional
        Per-corner camber {'FL': deg, ...} for peak_mu lookup.

    Returns
    -------
    dict of {corner_label: BrakeCornerResult}
    """
    results = {}
    bias_f = system.bias_pct_front / 100.0
    bias_r = 1.0 - bias_f
    r_tire = system.tire_radius_m
    if cambers is None:
        cambers = {}

    for label in ['FL', 'FR', 'RL', 'RR']:
        is_front = label[0] == 'F'
        bp = brake_params_f if is_front else brake_params_r
        fz = max(Fz.get(label, 0), 0.0)

        r_pad = bp.pad_radius_mm / 1000.0
        A_piston = bp.piston_area_mm2
        mu_pad = bp.pad_mu
        n_pistons = bp.num_pistons

        # Peak tire μ from tire model (load-sensitive) or fallback
        if tire_model is not None and hasattr(tire_model, 'peak_mu'):
            cam = abs(cambers.get(label, 0.0))
            mu_tire = float(tire_model.peak_mu(max(fz, 1.0), cam))
        else:
            mu_tire = 1.5

        cr = BrakeCornerResult(Fz_N=fz)
        cr.tire_mu = mu_tire

        # ── Lockup threshold ─────────────────────────────────────
        # The tire locks when Fx > mu_tire × Fz
        lockup_Fx = mu_tire * fz                            # N
        lockup_torque = lockup_Fx * r_tire                  # Nm

        # Caliper clamp to produce that torque
        # T = clamp × mu_pad × r_pad × 2 (both pads)
        if mu_pad > 0 and r_pad > 0:
            lockup_clamp = lockup_torque / (mu_pad * r_pad * 2)
        else:
            lockup_clamp = 0.0

        # Line pressure to produce that clamp
        # clamp = P × A_piston × n_pistons (floating caliper: ×2 for both sides)
        # For floating caliper with n_pistons on one side:
        #   clamp = P × A_piston × n_pistons
        # (the floating side mirrors the force → both pads grip equally)
        effective_piston_area = A_piston * n_pistons
        if effective_piston_area > 0:
            lockup_pressure = lockup_clamp / effective_piston_area  # N/mm² = MPa
        else:
            lockup_pressure = 0.0

        # Pedal force to generate that line pressure.
        #
        # The BALANCE BAR splits the pedal pushrod force between the two master
        # cylinders, so only a FRACTION of it reaches this circuit:
        #     F_pushrod = F_pedal × pedal_ratio
        #     F_mc      = F_pushrod × bias          (bias_f front, bias_r rear)
        #     P_line    = F_mc / A_mc
        # → F_pedal = P_line × A_mc / (pedal_ratio × bias)
        #
        # Omitting the bias term UNDERSTATES the pedal force (by 1/0.65 = 1.54x
        # front and 1/0.35 = 2.86x rear at a 65% bias) — it was previously
        # dropped here, which made the pedal-box sizing optimistic.
        mc_area = system.mc_area_front_mm2 if is_front else system.mc_area_rear_mm2
        bias = bias_f if is_front else bias_r
        if system.pedal_ratio > 0 and bias > 0:
            lockup_pedal = lockup_pressure * mc_area / (system.pedal_ratio * bias)
        else:
            lockup_pedal = 0.0

        cr.lockup_Fx_N = lockup_Fx
        cr.lockup_torque_Nm = lockup_torque
        cr.lockup_clamp_N = lockup_clamp
        cr.lockup_line_pressure_MPa = lockup_pressure
        cr.lockup_pedal_force_N = lockup_pedal

        results[label] = cr

    return results


def compute_brake_thermal(
    vehicle_mass_kg: float,
    bias_pct_front: float,
    speed_start_mph: float,
    speed_end_mph: float,
    rotor_mass_f_kg: float,
    rotor_mass_r_kg: float,
    rotor_cp: float = 460.0,
    ambient_C: float = 25.0,
) -> dict:
    """
    Single braking event adiabatic rotor temperature rise.

    Assumes 100% of kinetic energy goes into the rotors (worst case —
    no convective cooling during the stop).  Energy is split by brake
    bias, then divided equally between left/right per axle.

    Parameters
    ----------
    vehicle_mass_kg : float
        Total vehicle mass (driver included).
    bias_pct_front : float
        Brake bias % to front axle.
    speed_start_mph, speed_end_mph : float
        Braking from → to (mph).
    rotor_mass_f_kg, rotor_mass_r_kg : float
        Mass of ONE front / rear rotor.
    rotor_cp : float
        Specific heat of rotor material (J/kg·K).
    ambient_C : float
        Ambient / initial rotor temperature (°C).

    Returns
    -------
    dict of {corner_label: {'energy_kJ': float, 'delta_T_C': float, 'peak_T_C': float}}
    """
    mph_to_ms = 0.44704
    v1 = speed_start_mph * mph_to_ms
    v2 = speed_end_mph * mph_to_ms
    KE = 0.5 * vehicle_mass_kg * (v1 ** 2 - v2 ** 2)  # joules
    if KE < 0:
        KE = 0.0

    bias_f = bias_pct_front / 100.0
    energy_front_axle = KE * bias_f
    energy_rear_axle = KE * (1.0 - bias_f)

    results = {}
    for label in ['FL', 'FR', 'RL', 'RR']:
        is_front = label[0] == 'F'
        energy_per_rotor = (energy_front_axle if is_front else energy_rear_axle) / 2.0
        m_rotor = rotor_mass_f_kg if is_front else rotor_mass_r_kg

        if m_rotor > 0 and rotor_cp > 0:
            delta_T = energy_per_rotor / (m_rotor * rotor_cp)
        else:
            delta_T = 0.0

        results[label] = {
            'energy_kJ': energy_per_rotor / 1000.0,
            'delta_T_C': delta_T,
            'peak_T_C': ambient_C + delta_T,
        }

    return results


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
    cradle_solvers: dict | None = None,
    heave_tbar_solvers: dict | None = None,
) -> dict:
    """
    Compute component loads for all 4 corners from a dynamics result.

    Returns dict: {'FL': ComponentLoads, 'FR': ..., 'RL': ..., 'RR': ...}

    cradle_solvers (optional): {'front': TwinRockerDecoupledSolver|None,
    'rear': ...}.  For DECOUPLED corners the per-corner kinematic solver leaves
    pushrod_inner = NaN because the cradle (where the pushrod's inner end lives)
    is solved separately.  When the matching cradle solver is supplied, it is
    driven with the live left/right pushrod_outer to recover the true live
    pushrod_inner so the upright free body has a valid actuation member.
    """
    # ── Phase 1: solve every corner's kinematic state up front ───────────────
    states = {}
    for label in ['FL', 'FR', 'RL', 'RR']:
        solver = solvers.get(label)
        if solver is None:
            states[label] = None
            continue
        travel_m = dyn_result.travel.get(label, 0) / 1000  # mm → m
        try:
            states[label] = solver.solve(travel_m)
        except Exception:
            states[label] = None

    # ── Phase 2: fill DECOUPLED cradle-end (pushrod_inner) from cradle solver ─
    # The cradle solver needs BOTH wheels' live pushrod_outer to pose the
    # twin-rocker cradle, so this must happen after all corners are solved.
    if cradle_solvers:
        for axle_key, (lbl_L, lbl_R) in (('front', ('FL', 'FR')),
                                         ('rear',  ('RL', 'RR'))):
            cs = cradle_solvers.get(axle_key)
            sL, sR = states.get(lbl_L), states.get(lbl_R)
            if cs is None or sL is None or sR is None:
                continue
            try:
                cst = cs.solve(np.asarray(sL.pushrod_outer, dtype=float),
                               np.asarray(sR.pushrod_outer, dtype=float))
                sL.pushrod_inner = np.asarray(cst.pushrod_inner_L, dtype=float)
                sR.pushrod_inner = np.asarray(cst.pushrod_inner_R, dtype=float)
            except Exception:
                pass  # leave NaN → compute_corner_loads falls back to lstsq

    # ── Phase 2b: fill HEAVE-T-BAR pushrod_inner from the heave-T-bar rocker ──
    # Same idea as the cradle fill: cradle_link corners report pushrod_inner =
    # NaN because the pushrod's inner end lives on the external heave-T-bar
    # rocker.  The HeaveTBarRockerSolver models the LEFT side; the RIGHT corner
    # is the X-mirror (pose the left solver with the mirrored pushrod_outer and
    # mirror the result back).
    if heave_tbar_solvers:
        def _mir(p):
            q = np.asarray(p, dtype=float).copy(); q[0] *= -1.0; return q
        for axle_key, (lbl_L, lbl_R) in (('front', ('FL', 'FR')),
                                         ('rear',  ('RL', 'RR'))):
            hs = heave_tbar_solvers.get(axle_key)
            if hs is None:
                continue
            sL, sR = states.get(lbl_L), states.get(lbl_R)
            # Only fill when the corner left pushrod_inner as NaN (i.e. it is a
            # cradle_link corner).  A pushrod-mode heave-T-bar corner already has
            # a valid corner pushrod_inner — DON'T overwrite it with the rocker's.
            if sL is not None and not np.all(np.isfinite(np.asarray(sL.pushrod_inner, float))):
                try:
                    sL.pushrod_inner = np.asarray(
                        hs.pose(np.asarray(sL.pushrod_outer, dtype=float))['pushrod_inner'],
                        dtype=float)
                except Exception:
                    pass
            if sR is not None and not np.all(np.isfinite(np.asarray(sR.pushrod_inner, float))):
                try:
                    sR.pushrod_inner = _mir(
                        hs.pose(_mir(sR.pushrod_outer))['pushrod_inner'])
                except Exception:
                    pass

    # ── Phase 3: component loads per corner ──────────────────────────────────
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

        state = states.get(label)
        if state is None:
            # No solver or kinematic solve failed — still compute bearing/brake
            cl = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=bt)
            _compute_bearing_loads(cl, bp, up, wheel_radius_m)
            _compute_caliper_bolt_loads(cl, bp, up)
            _compute_brake_forces(cl, bp)
            results[label] = cl
            continue

        results[label] = compute_corner_loads(
            state, Fz, Fy, Fx, bt, bp, up, wheel_radius_m, mr)

    return results
