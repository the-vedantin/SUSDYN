"""
vahan/loads.py — Component force calculations for suspension members.

Given per-corner wheel loads (Fz, Fy, Fx) and the 3D geometry from the
kinematic solver, computes:
  - Axial forces in every suspension link (6×6 equilibrium)
  - Ball joint resultant forces in V (up+) and H (fwd+)
  - Bearing loads at inner/outer bearings in V and H
  - Brake caliper mounting bolt forces in V and H
  - Brake system forces (clamping, line pressure)

WORLD FRAME (the hardpoint frame, VERIFIED on the car 2026-09-22, not assumed):
    X = lateral,       +X = the car's LEFT   (FL/RL hardpoints have x > 0)
    Y = longitudinal,  +Y = REARWARD         (front axle y ~ 0, rear axle y ~ +1.54 m)
    Z = up
  Right-handed: X = Y x Z (rearward x up = left).  Every older comment in this file
  that said "Y = fwd+" was wrong about the frame; the audit of 2026-09-22 inherited
  that error.  Consequences that follow from the frame (derived, not assumed):
    * a wheel rolling FORWARD (-Y) spins about +X on BOTH sides:
        v_centre + omega x r_cp = 0,  r_cp = -R z,  v_centre = -V y
        => omega_x * R = V  => omega = +(V/R) x
    * a braking tyre force points REARWARD = +Y; a drive force points -Y.

SIGNED WHEEL FORCES used by every function here (ComponentLoads Fx_N/Fy_N/Fz_N):
    Fx_N  longitudinal, + = FORWARD (drive), - = braking     (world y = -Fx)
    Fy_N  lateral,      + = toward the car's LEFT (world +x) (world x = +Fy)
    Fz_N  vertical ground reaction, + = up                   (world z = +Fz)
  patch_force_world() is the ONE place this mapping lives; the member solver, the
  corner moments and the 3-D arrows all call it.  SteadyStateResult stores Fx/Fy
  as MAGNITUDES (dynamics.py); signed_wheel_forces() converts them.

Force sign convention for the V/H tables:
    V = vertical,     positive = UP
    H = longitudinal, positive = FORWARD (towards nose)  -> H = -(world y)
Axial member forces: positive = TENSION, negative = COMPRESSION.
"""

from dataclasses import dataclass, field
import numpy as np

G_STD = 9.81   # m/s^2 (same constant as vahan.ymd.G)

# Solve-validity threshold on the 2-norm condition number of the member system
# with its moment rows divided by a characteristic length (so every entry is a
# dimensionless direction cosine).  cond bounds the relative error amplification:
# forces can move by up to cond x (relative input error).  A 0.5 mm pickup error
# on a ~300 mm arm is ~0.2 %, so cond 1e3 already allows ~200 % worst-case error.
# Beyond it the member numbers are not engineering numbers -> valid=False, NaN.
# The real 2027 car sits at cond ~ 10-40 (see test_one_model 'loads validity').
COND_LIMIT = 1.0e3


# ═══════════════════════════════════════════════════════════════════════════
#  FRAME HELPERS — the ONE mapping from vehicle-signed forces to world vectors
# ═══════════════════════════════════════════════════════════════════════════

def patch_force_world(Fx_fwd: float, Fy_left: float, Fz_up: float) -> np.ndarray:
    """Contact-patch force ON the tyre, as a world vector (see module docstring).
    Fx_fwd + = forward (drive), Fy_left + = toward the car's left, Fz_up + = up."""
    return np.array([float(Fy_left), -float(Fx_fwd), float(Fz_up)])


def world_VH(vec) -> tuple:
    """(V, H) of a world vector: V = up+, H = forward+ (= -world y)."""
    v = np.asarray(vec, dtype=float)
    return float(v[2]), float(-v[1])


def rolling_axis(spin_axis) -> np.ndarray:
    """Unit wheel axis in the sense of the FORWARD-rolling angular velocity (+X
    component, both sides — derived in the module docstring)."""
    s = np.asarray(spin_axis, dtype=float)
    n = np.linalg.norm(s)
    s = s / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    return -s if s[0] < 0.0 else s


def contact_patch_point(wheel_center, spin_axis, radius_m: float) -> np.ndarray:
    """Contact patch = one loaded radius from the wheel centre, straight DOWN
    within the wheel plane (chassis-frame ground normal = +Z).

    The old member solver projected the wheel centre to Z = 0.  In a chassis-
    relative travel sweep the wheel centre moves in Z, so that made the tyre's
    moment arm grow with bump (a 12 mm roll bump = +6 % lever).  The loaded
    radius is a property of the tyre, not of the suspension travel.  Body roll
    (~1 deg) tilting the road normal in the chassis frame is neglected."""
    wc = np.asarray(wheel_center, dtype=float)
    s = rolling_axis(spin_axis)
    down = np.array([0.0, 0.0, -1.0])
    d = down - float(down @ s) * s
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else down
    return wc + float(radius_m) * d


def signed_wheel_forces(dyn_result, label: str) -> tuple:
    """(Fx_fwd, Fy_left, Fz_up) for one corner from a SteadyStateResult.

    CONTRACT (dynamics.py, 2026-09-22): result.Fx / result.Fy hold MAGNITUDES
    (abs(m*a) splits) whatever the direction; brake_torque > 0 only while
    braking.  Direction therefore comes from the operating point:
      * longitudinal_g < 0 (or brake torque booked) -> braking -> Fx_fwd = -|Fx|
      * lateral_g > 0 = RIGHT-hand turn (dynamics: 'left side gains load'), turn
        centre on the right (-X) -> every tyre pushes toward -X -> Fy_left = -|Fy|
    The net guards this contract (test_one_model 'loads sign contract')."""
    lat = float(getattr(dyn_result, 'lateral_g', 0.0) or 0.0)
    lon = float(getattr(dyn_result, 'longitudinal_g', 0.0) or 0.0)
    fx = abs(float(dyn_result.Fx.get(label, 0.0)))
    fy = abs(float(dyn_result.Fy.get(label, 0.0)))
    bt = float(dyn_result.brake_torque.get(label, 0.0))
    braking = (lon < 0.0) or (bt > 0.0)
    Fx_fwd = -fx if braking else fx
    Fy_left = -float(np.sign(lat)) * fy
    return Fx_fwd, Fy_left, float(dyn_result.Fz.get(label, 0.0))


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
        """l3 — radius from the wheel centre to the caliper bolt line (drawing D1).
        Raw drawing value, never clamped: an impossible layout (bolt line at or
        through the wheel centre) is reported by caliper_domain_error(), not
        silently replaced by a fictitious 1 mm radius."""
        return 0.5 * float(self.rotor_dia_mm) - float(self.caliper_mount_height_mm)

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
        drawing and that identity, and put the bolts 23 mm from where they sit.

        SIGNED (2026-09-22, Astra F-03): l4 < 0 means the bolt line sits radially
        OUTSIDE the pad centre.  That is a legitimate layout, and the caliper free
        body (_compute_caliper_bolt_loads) is exact for either sign — the radial
        couple simply reverses.  It used to be clamped to 0, which kept the larger
        bolt-line radius in the lug positions but dropped the couple: 50 N.m of
        unbalanced moment on a 120 mm disc / 90 mm pad / 100 mm bolt line at
        450 N.m, up to 751.8 N.m over a broad input sweep."""
        return float(self.pad_radius_mm) - self.bolt_line_radius_mm

    def caliper_domain_error(self) -> str:
        """'' when the caliper geometry is physically usable, else the reason.

        Rejected explicitly (never clamped into a plausible-looking number):
          * disc diameter, pad radius or bolt spacing not positive;
          * pad centre ON or OUTSIDE the disc (pad radius >= disc radius) —
            the pad cannot act on a disc it does not overlap;
          * bolt line at or through the wheel centre (l3 <= 0).
        A bolt line radially outside the pad centre (l4 < 0) is NOT rejected:
        it is supported by the signed free body."""
        r_disc = 0.5 * float(self.rotor_dia_mm)
        r_pad = float(self.pad_radius_mm)
        if not (np.isfinite(r_disc) and r_disc > 0.0):
            return f'disc diameter {self.rotor_dia_mm} mm is not positive'
        if not (np.isfinite(r_pad) and r_pad > 0.0):
            return f'pad radius {self.pad_radius_mm} mm is not positive'
        if r_pad >= r_disc:
            return (f'pad centre at {r_pad:.1f} mm lies on/outside the disc '
                    f'(disc radius {r_disc:.1f} mm)')
        if not (np.isfinite(self.caliper_bolt_spacing_mm)
                and float(self.caliper_bolt_spacing_mm) > 0.0):
            return (f'caliper bolt spacing {self.caliper_bolt_spacing_mm} mm '
                    f'is not positive')
        if not (self.bolt_line_radius_mm > 0.0):
            return (f'caliper bolt line at {self.bolt_line_radius_mm:.1f} mm from '
                    f'the wheel centre (mount height {self.caliper_mount_height_mm} '
                    f'mm >= disc radius {r_disc:.1f} mm)')
        return ''


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
    # Manual caliper clock (used only when vertical mounts are OFF): degrees from
    # the TOP of the disc toward the REAR (0 = top, 90 = trailing edge, 270 =
    # leading edge), mirror-symmetric left/right.  = "CW seen from outboard" on
    # the LEFT wheel.  caliper_frame() is the one implementation (table + 3-D).
    caliper_angle_deg: float = 45.0
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
    # Wheel loads (inputs) — SIGNED, see the module docstring
    Fz_N: float = 0.0    # vertical ground reaction (up+)
    Fy_N: float = 0.0    # lateral force, + = toward the car's LEFT
    Fx_N: float = 0.0    # longitudinal force, + = forward (drive), - = braking

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

    # Spring force (N, compression positive).  From the ROCKER's own moment
    # balance about its pivot axis (never pushrod x wheel motion ratio); with an
    # ARB the roll part is shared spring:bar by roll stiffness (compute_all_corners).
    spring_force_N: float = 0.0
    # Spring force the rocker would need with NO anti-roll bar (rocker-only
    # lever equilibrium, F_spring*b = -M_pushrod).  Diagnostic.
    spring_rocker_only_N: float = 0.0
    # Part of the rocker moment the ARB drop link carries, expressed as an
    # equivalent spring force (N).  Left and right are equal-and-opposite, which
    # is the bar's own torsion balance.  rocker_arb_freebody turns it into the
    # drop-link force with the real drop-link geometry.
    arb_spring_equiv_N: float = 0.0
    # ARB drop-link axial force (tension +) and the torque this side's arm puts
    # into the bar about the bar axis (+X), when compute_all_corners is given the
    # ARB geometry.  Bar balance: left torque + right torque = 0 exactly.
    arb_link_N: float = 0.0
    arb_bar_torque_Nm: float = 0.0
    spring_arb_note: str = ''

    # ── Pushrod-carrying ARM: leg SHEAR at the inboard pickup (N) ─────────
    # When the pushrod lands on a control arm away from its ball joint the arm
    # is a loaded frame, not two 2-force legs: each inboard pickup reaction has
    # an along-leg part (reported above as the leg force) AND a transverse part
    # (this) that the leg carries as shear/bending.  Zero for 2-force legs.
    uca_front_shear_N: float = 0.0
    uca_rear_shear_N: float = 0.0
    lca_front_shear_N: float = 0.0
    lca_rear_shear_N: float = 0.0

    # ── Bearing loads (V=up+, H=fwd+) ────────────────────────────
    bearing_inner_V: float = 0.0
    bearing_inner_H: float = 0.0
    bearing_outer_V: float = 0.0
    bearing_outer_H: float = 0.0
    bearing_axial_N: float = 0.0   # lateral thrust on the bearing pair (= Fy)
    # Per-SUBSYSTEM validity (Astra F-03, 2026-09-22).  `valid` below covers the
    # member solve only; an unusable bearing or caliper geometry must not come
    # back as plausible zero loads, so each sub-body carries its own status and
    # its outputs are NaN when invalid.
    bearing_valid: bool = True
    bearing_invalid_reason: str = ''

    # ── Caliper mounting bolt loads (V=up+, H=fwd+) ──────────────
    caliper_upper_V: float = 0.0
    caliper_upper_H: float = 0.0
    caliper_lower_V: float = 0.0
    caliper_lower_H: float = 0.0
    caliper_valid: bool = True
    caliper_invalid_reason: str = ''

    # ── Brake ─────────────────────────────────────────────────────
    brake_torque_Nm: float = 0.0
    caliper_clamp_N: float = 0.0
    line_pressure_MPa: float = 0.0

    # Solve quality — a result is an engineering number ONLY when valid is True.
    residual: float = 0.0            # |A x - b| of the member system (mixed N / N·m)
    force_residual_N: float = 0.0    # force closure of the upright (+arm) bodies
    moment_residual_Nm: float = 0.0  # moment closure of the upright (+arm) bodies
    cond_number: float = 0.0         # scaled condition number (see COND_LIMIT)
    valid: bool = True
    invalid_reason: str = ''
    pushrod_body: str = 'upright'    # which rigid body carries pushrod_outer
    travel_m: float = 0.0            # wheel travel the pose was solved at

    # World-vector outputs for the 3-D load view (same numbers as the table).
    # chassis_forces: force ON the chassis/rocker at each inboard pickup.
    # joint_forces:   force ON the upright (or pushrod-carrying arm) at each
    #                 outboard joint: 'uca', 'lca', 'tie', 'pushrod'.
    # caliper_lugs:   [(position, force ON the upright lug)] x 2.
    chassis_forces: dict = field(default_factory=dict, repr=False)
    joint_forces: dict = field(default_factory=dict, repr=False)
    caliper_lugs: list = field(default_factory=list, repr=False)
    # The solved kinematic pose these loads were computed on (ONE MODEL: the
    # load view draws THIS pose, never a re-solve at zero travel).
    state: object = field(default=None, repr=False, compare=False)


# ═══════════════════════════════════════════════════════════════════════════
#  MEMBER FORCE SOLVER
# ═══════════════════════════════════════════════════════════════════════════
#
# FREE-BODY DECLARATION (2026-09-22 repair; RCVD ch.17 fig 17.15, Seward ch.6)
#
# BODY U = wheel + tyre + hub + rotor + bearings + upright + caliper (outboard
# brakes: caliper on the upright, rotor on the hub).  The pad<->rotor friction
# pair is INTERNAL to U: equal, opposite, colinear -> zero net force and moment.
# External actions on U:
#   * tyre patch force F (world vector, patch_force_world) at the patch point
#   * unsprung weight + d'Alembert inertia  m_u (g - a)  at the wheel centre
#   * member forces at the outboard joints
#   * a HALF-SHAFT torque about the wheel axis when the wheel torque is carried
#     by a shaft to the chassis (driving through a chassis-mounted diff, or
#     inboard brakes).  Wheel spin-axis balance with no angular acceleration
#     (quasi-static; I*alpha of the wheel neglected):
#         T_shaft = -[(r_patch - r_wc) x F] . w      (w = rolling axis)
#     i.e. the tyre force then acts on the suspension AT THE WHEEL CENTRE for
#     the spin-axis component (RCVD fig 17.15b), and AT THE PATCH for outboard
#     brakes (fig 17.15a).
# The old code applied the patch force (which already carries the brake moment
# F*R about the axle) AND added the caliper's reaction couple T = F*R on top:
# that books one half of an internal action pair as external -> 2x the braking
# moment into the links.
#
# UNKNOWNS.  Every link with spherical joints at both ends is a 2-force member:
# force ON the upright = -T u   (u = unit inboard->outboard, T = tension +).
#   * pushrod on the UPRIGHT: six 2-force members meet U -> 6x6, determinate.
#   * pushrod on a CONTROL ARM (both axles of the 2027 car: UCA): the arm is
#     loaded at THREE places (chassis pivot axis, ball joint, pushrod mount) so
#     its legs are NOT 2-force members.  A spherical ball joint passes a force,
#     never a moment, so U only sees one unknown 3-D force R at that ball joint:
#         U unknowns: R (3) + other arm's 2 legs + tie rod = 6 -> 6x6
#     then the ARM body A (pivots f, r on axis e = unit(r - f)):
#         moment about the pivot axis:  e.[(bj-f) x (-R)] + e.[(po-f) x (-P u_p)] = 0
#             -> P = e.[(bj-f) x (-R)] / e.[(po-f) x u_p]
#         force:   R_f + R_r = R + P u_p                       (3 eqs)
#         moment about f:  (r-f) x R_r = (bj-f) x R + (po-f) x (P u_p)   (2 eqs)
#         the along-axis split R_f.e vs R_r.e is statically indeterminate (the
#         bushing pair can shuttle force along its common axis) -> equal split
#         (= equal axial bushing stiffness; also the minimum-norm solution).
#     Leg force = -R_leg . u_leg (tension +), leg SHEAR = transverse remainder.

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def rocker_required_spring(*, pushrod_inner, pushrod_outer, pushrod_N: float,
                           rocker_pivot, rocker_axis, rocker_spring_pt,
                           spring_chassis_pt) -> tuple:
    """Spring force (compression +) that closes the ROCKER's moment about its own
    pivot axis with the pushrod alone — no ARB.  Returns (F_spring, b_spring, m_push).

    Rocker free body (revolute pivot: reacts force + the two moments normal to
    its axis, NOTHING about the axis a):
        m_push = [(p_i - P) x (T_push * u_push)] . a,   u_push = unit(p_o - p_i)
        b_s    = [(s   - P) x u_c] . a,                 u_c    = unit(s - s_c)
        m_push + F_spring * b_s = 0   ->   F_spring = -m_push / b_s
    (a tension T_push pulls the rocker toward the pushrod's outer end; a
    compressed spring pushes the rocker point away from its chassis end.)
    This is the instantaneous lever ratio F_s = F_p * (a_p / b_s), NOT the wheel
    motion ratio: MR_wheel = (d push / d wheel) * (a_p / b_s)."""
    pi = np.asarray(pushrod_inner, float); po = np.asarray(pushrod_outer, float)
    P = np.asarray(rocker_pivot, float); a = _normalize(np.asarray(rocker_axis, float))
    sp = np.asarray(rocker_spring_pt, float); sc = np.asarray(spring_chassis_pt, float)
    if not all(np.all(np.isfinite(x)) for x in (pi, po, P, a, sp, sc)):
        return float('nan'), float('nan'), float('nan')
    u_push = _normalize(po - pi)
    m_push = float(np.cross(pi - P, float(pushrod_N) * u_push) @ a)
    u_c = _normalize(sp - sc)
    b_s = float(np.cross(sp - P, u_c) @ a)
    if abs(b_s) < 1e-6 or np.linalg.norm(u_push) < 0.5 or np.linalg.norm(u_c) < 0.5:
        return float('nan'), b_s, m_push
    return -m_push / b_s, b_s, m_push


def compute_corner_loads(
    state,              # SolvedState from kinematic solver
    Fz: float,          # vertical ground reaction (N, + up)
    Fy: float,          # lateral patch force (N, + toward the car's LEFT)
    Fx: float,          # longitudinal patch force (N, + forward/drive, - braking)
    brake_torque: float,    # brake torque at wheel (N·m, + retarding) — caliper/bearing
                            # sub-bodies only; INTERNAL to the member free body
    brake_params: BrakeParams,
    upright_params: UprightParams,
    wheel_radius_m: float = 0.203,
    motion_ratio: float | None = None,   # DEPRECATED, ignored (spring = rocker balance)
    *,
    pushrod_body: str = 'upright',
    rocker_axis=None,
    wheel_torque_via_shaft: bool | None = None,
    brakes_inboard: bool = False,
    unsprung_mass_kg: float = 0.0,
    accel_world=(0.0, 0.0, 0.0),
    cond_limit: float = COND_LIMIT,
) -> ComponentLoads:
    """Member forces at one suspension corner — see the FREE-BODY DECLARATION.

    pushrod_body       : 'upright' | 'uca' | 'lca' — the rigid body carrying
                         pushrod_outer (the solver's own `_pushrod_body`).
    rocker_axis        : world rocker pivot axis (solver `_rocker_axis`); gives the
                         rocker-only spring force.  None -> spring NaN.
    wheel_torque_via_shaft : None = auto: True when driving (Fx > 0, chassis diff +
                         half-shafts) or braking with inboard brakes; False for
                         braking with OUTBOARD brakes (caliper on the upright).
    unsprung_mass_kg, accel_world : corner unsprung mass lumped at the wheel
                         centre and the car's world acceleration (m/s^2); the
                         body carries m_u*(g - a).  Defaults 0 = massless corner.
    """
    result = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=brake_torque,
                            pushrod_body=str(pushrod_body or 'upright'),
                            state=state,
                            travel_m=float(getattr(state, 'travel', 0.0) or 0.0))
    nan = float('nan')
    reasons = []

    # ── Extract 3D positions from solved state ──────────────────────────────
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
    spin   = np.asarray(state.spin_axis, dtype=float)

    # ── DIRECT-damper corner: the damper line is the actuation member ──────
    # The solver collapses the pushrod to a point (push_o == push_i) and reports
    # the damper's chassis end as spring_chassis_pt.
    is_direct = bool(np.all(np.isfinite(push_o)) and np.all(np.isfinite(push_i))
                     and np.linalg.norm(push_o - push_i) < 1e-9)
    if is_direct:
        chassis_pt = np.asarray(getattr(state, 'spring_chassis_pt', push_i),
                                dtype=float)
        if (chassis_pt.shape == (3,) and np.all(np.isfinite(chassis_pt))
                and np.linalg.norm(push_o - chassis_pt) > 1e-9):
            push_i = chassis_pt

    body = str(pushrod_body or 'upright').lower()
    if body not in ('upright', 'uca', 'lca'):
        body = 'upright'

    # ── External actions on body U ──────────────────────────────────────────
    F_patch = patch_force_world(Fx, Fy, Fz)
    cp = contact_patch_point(wc, spin, wheel_radius_m)
    w = rolling_axis(spin)
    F_unspr = float(unsprung_mass_kg) * (np.array([0.0, 0.0, -G_STD])
                                        - np.asarray(accel_world, dtype=float))
    if wheel_torque_via_shaft is None:
        wheel_torque_via_shaft = (Fx > 0.0) or (Fx < 0.0 and bool(brakes_inboard))
    T_shaft = np.zeros(3)
    if wheel_torque_via_shaft:
        T_shaft = -float(np.cross(cp - wc, F_patch) @ w) * w
    O = lca_o.copy()                       # moment reference point
    F_ext = F_patch + F_unspr
    M_ext = np.cross(cp - O, F_patch) + np.cross(wc - O, F_unspr) + T_shaft

    # ── Unit vectors (inboard -> outboard) ─────────────────────────────────
    u_uf = _normalize(uca_o - uca_f); u_ur = _normalize(uca_o - uca_r)
    u_lf = _normalize(lca_o - lca_f); u_lr = _normalize(lca_o - lca_r)
    u_tr = _normalize(tr_o - tr_i);   u_p = _normalize(push_o - push_i)

    # ── Upright system: columns = force (rows 0-2) and moment about O (3-5) of a
    #    UNIT unknown.  Axial member: force on U = -T u.  Joint vector R: force on
    #    U = R (identity columns).
    def _axial(p, u):
        return np.concatenate([-u, np.cross(p - O, -u)])
    cols, names = [], []
    if body == 'uca':
        R_idx = 0
        for k in range(3):
            e = np.zeros(3); e[k] = 1.0
            cols.append(np.concatenate([e, np.cross(uca_o - O, e)]))
        names += ['R_x', 'R_y', 'R_z']
        cols += [_axial(lca_o, u_lf), _axial(lca_o, u_lr), _axial(tr_o, u_tr)]
        names += ['lca_front', 'lca_rear', 'tierod']
        arm = (uca_f, uca_r, uca_o, u_uf, u_ur)
    elif body == 'lca':
        cols += [_axial(uca_o, u_uf), _axial(uca_o, u_ur)]
        names += ['uca_front', 'uca_rear']
        R_idx = 2
        for k in range(3):
            e = np.zeros(3); e[k] = 1.0
            cols.append(np.concatenate([e, np.cross(lca_o - O, e)]))
        names += ['R_x', 'R_y', 'R_z']
        cols.append(_axial(tr_o, u_tr)); names.append('tierod')
        arm = (lca_f, lca_r, lca_o, u_lf, u_lr)
    else:
        cols = [_axial(uca_o, u_uf), _axial(uca_o, u_ur), _axial(lca_o, u_lf),
                _axial(lca_o, u_lr), _axial(tr_o, u_tr), _axial(push_o, u_p)]
        names = ['uca_front', 'uca_rear', 'lca_front', 'lca_rear', 'tierod', 'pushrod']
        arm = None
    A = np.column_stack(cols)
    b = -np.concatenate([F_ext, M_ext])

    # ── Validity: rank + scaled conditioning (moment rows / characteristic L) ─
    pts = [uca_o, lca_o, tr_o, push_o, cp]
    Lc = max([np.linalg.norm(p - O) for p in pts if np.all(np.isfinite(p))] + [1e-3])
    A_s = A.copy(); A_s[3:6, :] /= Lc
    if not np.all(np.isfinite(A_s)):
        rank, cond = 0, float('inf')
        reasons.append('non-finite geometry (a joint position or member direction is NaN)')
    else:
        sv = np.linalg.svd(A_s, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-10)) if sv[0] > 0 else 0
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf')
        if rank < 6:
            reasons.append(f'mechanism: member system rank {rank} < 6 '
                           f'(links do not fix the upright)')
        elif cond > cond_limit:
            reasons.append(f'ill-conditioned: cond {cond:.3g} > {cond_limit:.0f} '
                           f'(near-toggle geometry, forces not trustworthy)')
    result.cond_number = cond
    if rank == 6:
        x = np.linalg.solve(A, b)
    else:
        x = (np.linalg.lstsq(A, b, rcond=None)[0] if np.all(np.isfinite(A))
             else np.full(6, nan))
    r_sys = A @ x - b
    result.residual = float(np.linalg.norm(r_sys))
    f_res = float(np.linalg.norm(r_sys[0:3])); m_res = float(np.linalg.norm(r_sys[3:6]))
    sol = dict(zip(names, x))

    # ── Pushrod-carrying ARM body ───────────────────────────────────────────
    shear = {}
    chassis = {}
    if arm is not None:
        a_f, a_r, a_bj, u_af, u_ar = arm
        R = np.array([sol['R_x'], sol['R_y'], sol['R_z']])  # force ON U from the arm
        e_ax = a_r - a_f
        L2 = float(e_ax @ e_ax)
        e_hat = _normalize(e_ax)
        lever_p = float(np.cross(push_o - a_f, u_p) @ e_hat)
        m_bj = float(np.cross(a_bj - a_f, -R) @ e_hat)
        # |lever| relative to the pushrod-mount distance = sine of the pushrod's
        # angle to the arm's swing direction; ~0 = pushrod through the pivot axis
        # (it cannot hold the arm up).
        _ref = max(np.linalg.norm(push_o - a_f), 1e-6)
        if (not np.isfinite(lever_p)) or abs(lever_p) / _ref < 1e-3 or L2 < 1e-10:
            reasons.append('pushrod line (nearly) passes through the arm pivot axis '
                           '— it cannot react the arm; forces unbounded')
            P = nan
        else:
            P = m_bj / lever_p
        S = R + P * u_p                         # = R_f + R_r (forces ON the arm)
        M0 = np.cross(a_bj - a_f, R) + np.cross(push_o - a_f, P * u_p)
        R_r = (np.cross(M0, e_ax) / L2 if L2 > 1e-10 else np.full(3, nan)) \
            + 0.5 * float(S @ e_hat) * e_hat
        R_f = S - R_r
        # arm closure (independent check of the above algebra)
        F_arm = R_f + R_r - R - P * u_p
        M_arm = (np.cross(a_r - a_f, R_r) + np.cross(a_bj - a_f, -R)
                 + np.cross(push_o - a_f, -P * u_p))
        f_res = max(f_res, float(np.linalg.norm(F_arm)))
        m_res = max(m_res, float(np.linalg.norm(M_arm)))
        T_af = float(-R_f @ u_af); T_ar = float(-R_r @ u_ar)
        sh_f = float(np.linalg.norm(R_f + T_af * u_af))
        sh_r = float(np.linalg.norm(R_r + T_ar * u_ar))
        k = 'uca' if body == 'uca' else 'lca'
        sol[f'{k}_front'] = T_af; sol[f'{k}_rear'] = T_ar; sol['pushrod'] = P
        shear[f'{k}_front'] = sh_f; shear[f'{k}_rear'] = sh_r
        chassis[f'{k}_front'] = -R_f; chassis[f'{k}_rear'] = -R_r
        R_joint = {k: R}
    else:
        R_joint = {}

    result.force_residual_N = f_res
    result.moment_residual_Nm = m_res

    T = {n: float(sol.get(n, nan)) for n in
         ('uca_front', 'uca_rear', 'lca_front', 'lca_rear', 'tierod', 'pushrod')}
    ok = (not reasons) and all(np.isfinite(v) for v in T.values())
    if ok and max(f_res, m_res) > 1e-6 * max(1.0, float(np.linalg.norm(F_ext))):
        reasons.append(f'equilibrium does not close (force {f_res:.3g} N, '
                       f'moment {m_res:.3g} N·m)')
        ok = False
    if not ok and not reasons:
        reasons.append('non-finite member force')
    result.valid = bool(ok)
    result.invalid_reason = '; '.join(reasons)
    if not ok:                     # never report an invalid solve as numbers
        T = {n: nan for n in T}
        shear = {n: nan for n in ('uca_front', 'uca_rear', 'lca_front', 'lca_rear')}
        chassis = {}
        R_joint = {}

    result.uca_front_N = T['uca_front']; result.uca_rear_N = T['uca_rear']
    result.lca_front_N = T['lca_front']; result.lca_rear_N = T['lca_rear']
    result.tierod_N = T['tierod'];       result.pushrod_N = T['pushrod']
    result.uca_front_shear_N = float(shear.get('uca_front', 0.0))
    result.uca_rear_shear_N = float(shear.get('uca_rear', 0.0))
    result.lca_front_shear_N = float(shear.get('lca_front', 0.0))
    result.lca_rear_shear_N = float(shear.get('lca_rear', 0.0))

    # ── Joint resultants (force ON the upright / on the pushrod-carrying arm) ─
    F_uca = R_joint.get('uca', -(T['uca_front'] * u_uf + T['uca_rear'] * u_ur))
    F_lca = R_joint.get('lca', -(T['lca_front'] * u_lf + T['lca_rear'] * u_lr))
    F_tie = -T['tierod'] * u_tr
    F_push = -T['pushrod'] * u_p
    result.uca_bj_V, result.uca_bj_H = world_VH(F_uca)
    result.lca_bj_V, result.lca_bj_H = world_VH(F_lca)
    result.tierod_bj_V, result.tierod_bj_H = world_VH(F_tie)
    result.pushrod_bj_V, result.pushrod_bj_H = world_VH(F_push)
    if ok:
        result.joint_forces = {'uca': F_uca, 'lca': F_lca, 'tie': F_tie, 'pushrod': F_push}
        # force ON the chassis (or rocker) at each inboard pickup
        for n, u in (('uca_front', u_uf), ('uca_rear', u_ur), ('lca_front', u_lf),
                     ('lca_rear', u_lr), ('tierod', u_tr), ('pushrod', u_p)):
            chassis.setdefault(n, T[n] * u)
        result.chassis_forces = chassis

    # ── Spring force: the ROCKER's own moment balance (no ARB here; the
    #    left/right ARB share is applied by compute_all_corners) ────────────
    if is_direct:
        result.spring_force_N = -T['pushrod']        # damper line IS the spring
    elif rocker_axis is not None:
        S_req, _b, _m = rocker_required_spring(
            pushrod_inner=push_i, pushrod_outer=push_o, pushrod_N=T['pushrod'],
            rocker_pivot=getattr(state, 'rocker_pivot', np.full(3, nan)),
            rocker_axis=rocker_axis,
            rocker_spring_pt=getattr(state, 'rocker_spring_pt', np.full(3, nan)),
            spring_chassis_pt=getattr(state, 'spring_chassis_pt', np.full(3, nan)))
        result.spring_force_N = S_req
    else:
        result.spring_force_N = nan                  # no rocker geometry supplied
    result.spring_rocker_only_N = result.spring_force_N
    result.arb_spring_equiv_N = 0.0

    # ── Bearing, caliper and brake sub-bodies (hub / caliper free bodies) ───
    frame = caliper_frame(upright_params, spin, wc, tr_o)
    fy_in = -1.0 if float(wc[0]) >= 0.0 else 1.0      # Fy_left -> inboard sense
    _compute_bearing_loads(result, brake_params, upright_params, wheel_radius_m,
                           fy_inboard_sign=fy_in, frame=frame)
    _compute_caliper_bolt_loads(result, brake_params, upright_params,
                                frame=frame, wheel_center=wc)
    _compute_brake_forces(result, brake_params)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CALIPER CLOCKING — the ONE implementation (table AND 3-D arrows)
# ═══════════════════════════════════════════════════════════════════════════

def caliper_frame(up: UprightParams, spin_axis=(1.0, 0.0, 0.0),
                  wheel_center=(0.0, 0.0, 0.0), tr_outer=None) -> tuple:
    """(r_hat, t_hat, w): world unit vectors of the caliper.

    r_hat : wheel centre -> pad centre / bolt-line centre (radial)
    t_hat : direction the disc surface moves at the pad in FORWARD rolling
            = w x r_hat (w = rolling axis, +X both sides) — the bolt-spacing
            direction and the direction of the friction force ON the caliper.
    Vertical mounts (default): radial points FORE-AFT toward the tie-rod side,
    so the two bolts stack vertically.  Otherwise caliper_angle_deg measured
    from the top toward the REAR (UprightParams docstring)."""
    w = rolling_axis(spin_axis)
    wc = np.asarray(wheel_center, dtype=float)
    z = np.array([0.0, 0.0, 1.0]); y = np.array([0.0, 1.0, 0.0])
    vup = _normalize(z - float(z @ w) * w)
    rear = _normalize(y - float(y @ w) * w)       # +Y = rearward
    if bool(getattr(up, 'caliper_vertical_mounts', True)):
        dy = (float(np.asarray(tr_outer, float)[1] - wc[1])
              if tr_outer is not None else 0.0)
        r_hat = np.sign(dy) * rear if abs(dy) > 1e-6 else -rear   # default: leading
    else:
        phi = np.radians(float(getattr(up, 'caliper_angle_deg', 45.0)))
        r_hat = vup * np.cos(phi) + rear * np.sin(phi)
    r_hat = _normalize(r_hat)
    t_hat = _normalize(np.cross(w, r_hat))
    return r_hat, t_hat, w


# ═══════════════════════════════════════════════════════════════════════════
#  BEARING LOADS  (moment equilibrium on the hub free body)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_bearing_loads(result: ComponentLoads, bp: BrakeParams,
                           up: UprightParams, wheel_radius_m: float,
                           fy_inboard_sign: float = 1.0, frame=None):
    """
    Bearing loads from moment equilibrium on the HUB free body
    (wheel + tyre + hub + rotor; the upright is NOT in this body).

    Hub free body sees:
      1. Tyre patch force (signed Fx_N fwd+, Fz_N up+, Fy_N) at the patch plane,
         d outboard of the inner bearing, one radius below the spindle.
      2. Pad friction ON THE DISC = -(T/r_pad) * t_hat at the caliper clock
         (it opposes the disc's forward rolling motion t_hat).
      3. Half-shaft torque (pure couple along the axle) when driven.
      4. The two bearing reactions.
    Outputs the load each bearing takes from the hub, V (up+) and H (fwd+).
    Simple two-support beam; the pad force is lumped at the patch plane (the
    rotor sits slightly inboard) — first-order sizing model, not preload-resolved.
    """
    l1 = up.bearing_spacing_mm / 1000.0   # m (bearing spacing)
    d  = up.cp_to_inner_bearing_mm / 1000.0   # m (patch plane -> inner bearing)
    r_pad = bp.pad_radius_mm / 1000.0     # m

    # Domain gate (Astra F-03): an unusable spacing / radius used to `return`
    # here and leave the default 0.0 loads — indistinguishable from a real
    # unloaded bearing.  Now the sub-body is marked invalid and reads NaN.
    _why = ''
    if not (np.isfinite(l1) and l1 >= 0.001):
        _why = (f'bearing spacing {up.bearing_spacing_mm} mm < 1 mm '
                f'(two-support beam undefined)')
    elif not (np.isfinite(d)):
        _why = 'bearing position not finite'
    elif not (np.isfinite(wheel_radius_m) and wheel_radius_m > 0.0):
        _why = f'wheel radius {wheel_radius_m} m is not positive'
    elif abs(float(result.brake_torque_Nm)) > 1e-6 and not (r_pad > 0.001):
        _why = (f'pad radius {bp.pad_radius_mm} mm cannot carry '
                f'{result.brake_torque_Nm:.1f} N.m of brake torque')
    if _why:
        nan = float('nan')
        result.bearing_outer_V = result.bearing_inner_V = nan
        result.bearing_outer_H = result.bearing_inner_H = nan
        result.bearing_axial_N = nan
        result.bearing_valid = False
        result.bearing_invalid_reason = _why
        return
    result.bearing_valid = True
    result.bearing_invalid_reason = ''
    if frame is None:
        frame = caliper_frame(up)
    _r_hat, t_hat, _w = frame

    Fz = result.Fz_N
    Fx = result.Fx_N            # signed, fwd+
    Fy = result.Fy_N            # signed, left+
    T  = result.brake_torque_Nm

    # ── Brake friction on disc (tangential, acts on the hub) ─────
    F_fric = T / r_pad if r_pad > 0.001 else 0.0   # T == 0 here when r_pad tiny
    F_fric_V, F_fric_H = world_VH(-F_fric * t_hat)

    # ── Total forces on hub to distribute between bearings ───────
    total_V = Fz + F_fric_V
    total_H = Fx + F_fric_H

    # ── Overturning moment from the LATERAL force ────────────────
    # Fy acts at the contact patch, one tire radius BELOW the spindle
    # axis: M = Fy·R_tire about the car's longitudinal axis.  Reacted as
    # a VERTICAL couple across the bearing spacing — in hard cornering
    # this term DOMINATES the bearing radial loads.  Fy itself is the
    # axial (thrust) load on the bearing pair.
    # SIGN (fixed 2026-09-19, contested by the upright designer, checked against
    # Seward 6.4.1 Maxle = Wlat*Rr - Wvert*l2): on the loaded OUTSIDE wheel the
    # ground pushes the patch INBOARD, so Fy*R OPPOSES the overhung-Fz moment.
    # fy_inboard_sign converts the signed (left+) Fy into its INBOARD component
    # on THIS corner: -1 on the left wheels, +1 on the right (2026-09-22: it used
    # to come from the spin axis, which points +x on BOTH sides, so the right
    # wheels got the wrong sense).
    Fy_in = fy_inboard_sign * Fy
    M_over = Fy_in * wheel_radius_m      # N·m, + = inboard force
    dV_over = M_over / l1                 # vertical couple magnitude

    # Simple beam: moment about inner bearing → outer bearing force,
    # plus the overturning couple (+ on the outboard row, − inboard for
    # a laterally-loaded outside wheel).
    result.bearing_outer_V = float(total_V * d / l1 - dV_over)
    result.bearing_inner_V = float(total_V * (l1 - d) / l1 + dV_over)
    result.bearing_outer_H = float(total_H * d / l1)
    result.bearing_inner_H = float(total_H * (l1 - d) / l1)
    result.bearing_axial_N = float(Fy_in)     # + = inboard thrust


# ═══════════════════════════════════════════════════════════════════════════
#  CALIPER MOUNTING BOLT FORCES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_caliper_bolt_loads(result: ComponentLoads, bp: BrakeParams,
                                up: UprightParams, frame=None,
                                wheel_center=(0.0, 0.0, 0.0)):
    """
    Force ON THE UPRIGHT at each of the two caliper mount lugs (Seward ch.6
    Fig 6.15), from the CALIPER free body:

      pad friction ON the caliper  = +F t_hat at the pad centre (radius R_pad),
                                     F = T / R_pad
      bolts at  b± = l3 r_hat ± (l5/2) t_hat,  forces from the upright B±
      ΣF = 0  and in-plane moment about the wheel centre (derived 2026-09-22):
          B± = -(F/2) t_hat ± (F l4 / l5) r_hat,     l4 = R_pad - l3
      Force ON the upright lug = -B± = +(F/2) t_hat ∓ (F l4/l5) r_hat.
    The couple lever is l4 (pad centre to bolt line), not R_pad (that moment is
    reacted by the whole upright).  'upper' = the lug with the larger Z.

    SIGNED l4 (derivation, 2026-09-22).  Write B± = a± t_hat + b± r_hat.
      ΣF = 0:            a+ + a- = -F,   b+ + b- = 0
      ΣM_w about wc:     R_pad F + l3 (a+ + a-) - (l5/2)(b+ - b-) = 0
      equal tangential share (the bolt pair's split along t_hat is statically
      indeterminate; equal = equal bolt shear stiffness): a± = -F/2
      =>  b± = ± F (R_pad - l3) / l5 = ± F l4 / l5
    Nothing in that derivation needs l4 >= 0: with the bolt line OUTSIDE the
    pad centre (l4 < 0) the radial couple reverses and the balance is still
    exact.  The old max(l4, 0) clamp kept l3 in the lug positions but dropped
    the couple (Astra F-03: 50 N.m residual on a 90 mm pad / 100 mm bolt line
    at 450 N.m).  Pad-outside-disc / non-positive geometry is REJECTED
    (caliper_valid False, NaN loads), never clamped.
    """
    T = result.brake_torque_Nm
    r_pad = bp.pad_radius_mm / 1000.0
    s = bp.caliper_bolt_spacing_mm / 1000.0          # l5, bolt (lug) spacing
    l3 = float(bp.bolt_line_radius_mm) / 1000.0      # signed drawing value
    l4 = r_pad - l3                                  # SIGNED; l3 + l4 = R_pad exactly
    result.caliper_lugs = []

    _why = bp.caliper_domain_error() if hasattr(bp, 'caliper_domain_error') else ''
    if _why:
        nan = float('nan')
        result.caliper_upper_V = result.caliper_upper_H = nan
        result.caliper_lower_V = result.caliper_lower_H = nan
        result.caliper_valid = False
        result.caliper_invalid_reason = _why
        return
    result.caliper_valid = True
    result.caliper_invalid_reason = ''
    if T < 1e-6:
        return  # no brake torque → bolts carry zero (a real zero: geometry valid)
    if frame is None:
        frame = caliper_frame(up)
    r_hat, t_hat, _w = frame
    wc = np.asarray(wheel_center, dtype=float)

    F = T / r_pad
    H_couple = F * l4 / s
    lugs = []
    for sgn in (+1.0, -1.0):
        pos = wc + l3 * r_hat + sgn * 0.5 * s * t_hat
        f_up = 0.5 * F * t_hat - sgn * H_couple * r_hat
        lugs.append((pos, f_up))
    lugs.sort(key=lambda pf: -float(pf[0][2]))        # upper first
    (pu, fu), (pl, fl) = lugs
    result.caliper_upper_V, result.caliper_upper_H = world_VH(fu)
    result.caliper_lower_V, result.caliper_lower_H = world_VH(fl)
    result.caliper_lugs = lugs


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
#  CORNER MOMENTS  (the load-view moments — ONE place, read by GUI + binder)
# ═══════════════════════════════════════════════════════════════════════════
#
# These used to be computed inline in gui/wheel_package.py, which duplicated
# physics into the presentation layer.  They now live here so the 3-D load view
# AND the binder read the SAME moment values (ONE MODEL).  2026-09-22: signs now
# come from patch_force_world (the same mapping as the member solver).

def rocker_arb_freebody(
    *,
    pushrod_inner, pushrod_outer, pushrod_N: float,
    rocker_pivot, rocker_axis,
    rocker_spring_pt, spring_chassis_pt, spring_force_N: float,
    arb_drop_top, arb_arm_end, arb_pivot=None,
    bar_axis=(1.0, 0.0, 0.0),
    m0_opposite: float | None = None,
) -> dict:
    """Rocker / ARB bellcrank free body, in WORLD coordinates.

    Every point and the rocker axis are world-frame vectors the CALLER has
    already mirrored for the corner side (left/right).  Pure vector free body.

    Rocker (revolute pivot: reacts a force and the two moments NORMAL to its axis
    a, but NOTHING about a):
      * F_push  = pushrod_N * unit(pushrod_outer - pushrod_inner)   (tension +)
      * F_spr   = -spring_force_N * unit(spring_chassis_pt - rocker_spring_pt)
      * m0      = [(p_i - P) x F_push + (s - P) x F_spr] . a
      * F_arb   = -(m0 / lever) * u_arb,  lever = [(d - P) x u_arb] . a
                  -> THIS rocker's moment about its own axis closes exactly.
      * F_pivot = -(F_push + F_spr + F_arb)
    spring_force_N must come from compute_all_corners (rocker balance + the
    roll-stiffness spring:bar share) — then the two drop-link forces of an axle
    are equal-and-opposite, which IS the bar's torsion balance.

    2026-09-22: the drop-link force used to react only (m0 + m0_opposite)/2.  A
    free pivot cannot hold the other half: that left up to half of m0 unbalanced
    about the rocker axis (-270 N·m in the audit's example).  m0_opposite is now
    DIAGNOSTIC only: returned as 'bar_balance_Nm' = (m0 + m0_opposite)/2.

    Returns {'F_push','F_spr','F_arb','F_pivot','u_arb','arb_torsion_Nm','m0',
    'lever','axis_moment_residual_Nm','valid'} — forces N, moments N·m.
    """
    pi = np.asarray(pushrod_inner, dtype=float)
    po = np.asarray(pushrod_outer, dtype=float)
    P = np.asarray(rocker_pivot, dtype=float)
    axis = np.asarray(rocker_axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    sp = np.asarray(rocker_spring_pt, dtype=float)
    sc = np.asarray(spring_chassis_pt, dtype=float)
    dt = np.asarray(arb_drop_top, dtype=float)
    ae = np.asarray(arb_arm_end, dtype=float)

    u_push = (po - pi) / max(np.linalg.norm(po - pi), 1e-9)
    F_push = float(pushrod_N) * u_push
    u_sp = (sc - sp) / max(np.linalg.norm(sc - sp), 1e-9)
    F_spr = -float(spring_force_N) * u_sp
    u_arb = (ae - dt) / max(np.linalg.norm(ae - dt), 1e-9)
    m0 = float((np.cross(pi - P, F_push) + np.cross(sp - P, F_spr)) @ axis)
    lever = float(np.cross(dt - P, u_arb) @ axis)
    ok = bool(np.isfinite(m0) and np.isfinite(lever) and abs(lever) > 1e-4)
    F_arb = (-m0 / lever if ok else 0.0) * u_arb
    F_pivot = -(F_push + F_spr + F_arb)
    m_res = float((np.cross(pi - P, F_push) + np.cross(sp - P, F_spr)
                   + np.cross(dt - P, F_arb)) @ axis)

    out = {'F_push': F_push, 'F_spr': F_spr, 'F_arb': F_arb,
           'F_pivot': F_pivot, 'u_arb': u_arb, 'arb_torsion_Nm': None,
           'm0': m0, 'lever': lever, 'axis_moment_residual_Nm': m_res,
           'valid': bool(ok and abs(m_res) < 1e-6 * max(1.0, abs(m0))),
           'bar_balance_Nm': (0.5 * (m0 + float(m0_opposite))
                              if m0_opposite is not None else None)}

    # ── ARB BAR TORSION: the only moment on the car that does NOT act at the
    #    wheel.  Every link ends in a spherical joint (carries no moment); the
    #    anti-roll bar is the exception — it is a torsion spring, so the
    #    drop-link force at the arm end twists the bar about its own axis.
    if arb_pivot is not None and np.linalg.norm(F_arb) > 1.0:
        ap = np.asarray(arb_pivot, dtype=float)
        r_arm = ae - ap
        M_arb = np.cross(r_arm, -F_arb)
        out['arb_torsion_Nm'] = float(M_arb @ np.asarray(bar_axis, dtype=float))
    return out


def corner_moments(
    *,
    Fx: float, Fy: float, Fz: float, camber_deg: float = 0.0,
    wheel_center, spin_axis,
    lca_outer=None, uca_outer=None,
    tire_model=None,
    freebody: dict | None = None,
    rocker_arb: dict | None = None,
    wheel_radius_m: float | None = None,
) -> dict:
    """All five load-view moments at one corner, in N·m.

    Fx, Fy, Fz are the SIGNED wheel forces of the module docstring (Fx fwd+,
    Fy toward the car's left +, Fz up+) — the SAME ones the member solver uses
    (patch_force_world), so the kingpin moment and the member table describe
    one load case.  (Before 2026-09-22 this function put +Fy on world +X while
    the member solver put it on -X: mirror-image lateral loading.)

    Returned keys (present only when computable):
      hub_torque_Nm   = Fx * R_r  (drive +, braking -) about the wheel axis;
                        vector = hub_torque_Nm * hub_torque_axis.
      overturning_Nm  = Fy * R_r  about the fore-aft axis;
                        vector = overturning_Nm * overturning_axis.
      kingpin_Nm      = moment of the patch force about the steering axis through
                        the two ball joints (needs lca_outer and uca_outer).
      mz_Nm           = tyre self-aligning torque (needs a tire_model).
      arb_torsion_Nm  = ARB bar torsion (from rocker_arb_freebody).

    R_r = wheel_radius_m (the loaded radius the member solver uses); falls back
    to wheel_center z for old callers.  Patch = contact_patch_point().
    """
    wc = np.asarray(wheel_center, dtype=float)
    spin = np.asarray(spin_axis, dtype=float)
    spin = spin / max(np.linalg.norm(spin), 1e-9)
    Fx = float(Fx); Fy = float(Fy); Fz = float(Fz)

    R_r = (float(wheel_radius_m) if wheel_radius_m is not None
           else max(float(wc[2]), 1e-3))
    # Moment of the patch force about the wheel centre, with cp - wc = -R z:
    #   M = -R z x (Fy x - Fx y + Fz z) = -R Fx x - R Fy y
    # so the scalars Fx*R and Fy*R point along -x and -y respectively.
    out = {
        'hub_torque_Nm': Fx * R_r,
        'overturning_Nm': Fy * R_r,
        'hub_torque_axis': -rolling_axis(spin),
        'overturning_axis': np.array([0.0, -1.0, 0.0]),
    }

    # ── STEERING (kingpin) moment about the steering axis through the joints ──
    if lca_outer is not None and uca_outer is not None:
        try:
            kp_a = np.asarray(lca_outer, dtype=float)
            kp_b = np.asarray(uca_outer, dtype=float)
            k = kp_b - kp_a
            k = k / max(np.linalg.norm(k), 1e-9)
            patch = contact_patch_point(wc, spin, R_r)
            Fpatch = patch_force_world(Fx, Fy, Fz)
            out['kingpin_Nm'] = float(np.dot(k, np.cross(patch - kp_a, Fpatch)))
        except Exception:
            pass

    # ── TYRE self-aligning torque Mz, straight from the tyre model ──
    # The tyre model is queried with the lateral force MAGNITUDE (its reference
    # slip branch), exactly as before the sign repair.
    if tire_model is not None:
        try:
            sa = tire_model.slip_angle_for_Fy(abs(Fy), Fz, float(camber_deg))
            out['mz_Nm'] = float(tire_model.Mz(sa, Fz, float(camber_deg)))
        except Exception:
            pass

    # ── ARB bar torsion (from the rocker/ARB free body) ──
    if freebody is None and rocker_arb is not None:
        try:
            freebody = rocker_arb_freebody(**rocker_arb)
        except Exception:
            freebody = None
    if freebody is not None and freebody.get('arb_torsion_Nm') is not None:
        out['arb_torsion_Nm'] = float(freebody['arb_torsion_Nm'])

    return out


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
    grip_scale: float = None,
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
        from TTC data (load-sensitive).  REQUIRED — the old silent mu = 1.5
        fallback is gone (Astra review F-09).
    grip_scale : float
        THE project grip scale (MainWindow.grip_scale()); the lockup limit
        is mu_peak * grip_scale * Fz, the same road as every other limit.
    cambers : dict, optional
        Per-corner SIGNED tyre inclination {'FL': deg, ...} for the peak_mu
        lookup — pass SteadyStateResult.inclination, not .camber.

    Returns
    -------
    dict of {corner_label: BrakeCornerResult}
    """
    if tire_model is None or not hasattr(tire_model, 'peak_mu'):
        raise ValueError('compute_brake_system needs the selected tyre model '
                         '(no built-in mu)')
    if grip_scale is None:
        raise ValueError('compute_brake_system needs grip_scale (the project '
                         'grip scale)')
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

        # Peak tire μ from tire model (load-sensitive) or fallback.  `cambers`
        # is the SIGNED tyre inclination (SteadyStateResult.inclination);
        # straight-line braking has no turn hand, so take the peak over BOTH
        # slip branches (slip_sign=0) at that inclination — no |camber|.
        cam = float(cambers.get(label, 0.0))
        mu_tire = (float(tire_model.peak_mu(max(fz, 1.0), cam, 0))
                   * float(grip_scale))

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
    *,
    veh=None,
    topology=None,
    brakes_inboard_f: bool = False,
    brakes_inboard_r: bool = False,
    arb_geometry=None,
) -> dict:
    """
    Compute component loads for all 4 corners from a dynamics result.

    Returns dict: {'FL': ComponentLoads, 'FR': ..., 'RL': ..., 'RR': ...}

    motion_ratio_f/r are DEPRECATED and ignored: the spring force now comes from
    the rocker's own moment balance, not pushrod x wheel motion ratio.

    veh (VehicleParams, optional but what every GUI caller passes): supplies
      * the unsprung mass per corner (weight + d'Alembert inertia on the upright
        body — the spring never carries it, and neither does the pushrod), and
      * the roll-stiffness share of the ARB, k_arb / (k_wheel + k_arb) per axle,
        which splits each axle's ROLL (antisymmetric) rocker moment between the
        springs and the bar.  Without veh: massless corners, no ARB share.
    topology (SuspensionTopology, optional): a CONTROL-ARM ARB puts a drop-link
      force on the lower arm that this member model does not contain -> those
      corners are flagged invalid instead of silently loading the pushrod with it.
    brakes_inboard_f/r: False (default, this car) = caliper on the upright, so the
      brake torque is internal to the upright body; True = brake torque reaches
      the hub through the half-shaft (like the drive torque).
    arb_geometry (callable(label, state) -> {'drop_top','arm_end','pivot'} world
      points, or None): the bellcrank-ARB geometry at the solved pose (the GUI owns
      it: MainWindow._arb_drop_top_world + the mirrored arm/pivot).  With it the
      drop-link forces and the bar torque are solved exactly (Phase 4).

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
    # Car acceleration in the world frame (m/s^2): lateral_g > 0 is a right-hand
    # turn (centripetal toward -X); longitudinal_g > 0 is forward (-Y).
    lat_g = float(getattr(dyn_result, 'lateral_g', 0.0) or 0.0)
    lon_g = float(getattr(dyn_result, 'longitudinal_g', 0.0) or 0.0)
    a_world = np.array([-lat_g * G_STD, -lon_g * G_STD, 0.0])
    _incomplete = ''
    if veh is None:
        # Fail LOUDLY: a caller that skips veh gets massless corners and no
        # spring/ARB roll split — numbers that differ from the GUI's.
        import warnings
        _incomplete = ('computed WITHOUT veh: unsprung mass and ARB roll share '
                       'omitted (not the GUI load case)')
        warnings.warn('compute_all_corners: ' + _incomplete + ' — pass veh=, '
                      'topology=, arb_geometry= as gui.wheel_package.compute_case does',
                      RuntimeWarning, stacklevel=2)
    nan = float('nan')
    results = {}
    for label in ['FL', 'FR', 'RL', 'RR']:
        Fx, Fy, Fz = signed_wheel_forces(dyn_result, label)
        bt = float(dyn_result.brake_torque.get(label, 0))

        is_front = label[0] == 'F'
        bp = brake_params_f if is_front else brake_params_r
        up = upright_params_f if is_front else upright_params_r
        m_u = 0.0
        if veh is not None:
            m_u = 0.5 * float(getattr(veh, 'unsprung_mass_front_kg' if is_front
                                      else 'unsprung_mass_rear_kg', 0.0) or 0.0)

        state = states.get(label)
        if state is None:
            # No solver or kinematic solve failed — bearing/brake still follow
            # from the wheel forces, but there is NO member solution: say so.
            cl = ComponentLoads(Fz_N=Fz, Fy_N=Fy, Fx_N=Fx, brake_torque_Nm=bt)
            for k in ('uca_front_N', 'uca_rear_N', 'lca_front_N', 'lca_rear_N',
                      'tierod_N', 'pushrod_N', 'spring_force_N',
                      'spring_rocker_only_N', 'residual', 'force_residual_N',
                      'moment_residual_Nm', 'cond_number'):
                setattr(cl, k, nan)
            cl.valid = False
            cl.invalid_reason = ('no corner solver' if solvers.get(label) is None
                                 else 'kinematic solve failed at this travel')
            _compute_bearing_loads(cl, bp, up, wheel_radius_m,
                                   fy_inboard_sign=(-1.0 if label[1] == 'L' else 1.0))
            _compute_caliper_bolt_loads(cl, bp, up)
            _compute_brake_forces(cl, bp)
            results[label] = cl
            continue

        solver = solvers.get(label)
        act = str(getattr(solver, '_damper_actuation', 'pushrod'))
        body = (getattr(solver, '_damper_body', 'upright') if act == 'direct'
                else getattr(solver, '_pushrod_body', 'upright'))
        r_axis = getattr(solver, '_rocker_axis', None) if act in ('pushrod', 'pullrod') else None
        results[label] = compute_corner_loads(
            state, Fz, Fy, Fx, bt, bp, up, wheel_radius_m,
            pushrod_body=body, rocker_axis=r_axis,
            brakes_inboard=(brakes_inboard_f if is_front else brakes_inboard_r),
            unsprung_mass_kg=m_u, accel_world=a_world)
        if _incomplete:
            results[label].spring_arb_note = _incomplete

    # ── Phase 4: springs vs ARB on each axle (rocker balance + bar torsion) ───
    # Each rocker must close its own moment about its free pivot axis:
    #     m_push + F_spring * b_s + m_arb = 0.
    # Express every rocker's pushrod moment as the spring force that would close
    # it alone, S_req = -m_push / b_s (compression +, sign-convention free).
    # Split each axle into symmetric and antisymmetric (roll) parts:
    #     S_sym = (S_req_L + S_req_R)/2,   S_anti = (S_req_L - S_req_R)/2.
    # * A bellcrank ARB is a torsion bar in bushings: in heave/pitch it rotates
    #   rigidly and carries nothing -> the SYMMETRIC part is all springs.
    # * In roll both rockers rotate against spring AND bar in parallel (same
    #   rocker rotation), so the antisymmetric part splits by stiffness:
    #     share = k_arb / (k_wheel + k_arb)  (wheel rates; the common wheel->rocker
    #     kinematics cancels in the ratio) = ARB fraction of axle roll stiffness.
    # EXACT bar + rocker equilibrium (needs the ARB geometry at the solved pose):
    #   drop-link tension t (+ pulls the rocker toward the arm end), unit
    #   u = unit(arm_end - drop_top); on rocker side i:
    #     lambda_i = [(d_i - P_i) x u_i] . a_i        (drop-link lever on the rocker)
    #     rho_i    = [(e_i - p_i) x (-u_i)] . X       (lever on the bar about its axis)
    #   bar in bushings, no torque about its own axis:  t_L rho_L + t_R rho_R = 0
    #     -> t_L = tau / rho_L,  t_R = -tau / rho_R   (tau = bar torsion)
    #   each rocker:  m_push + S b_s + t lambda = 0
    #     -> S_L = S_req_L - tau g_L,  S_R = S_req_R + tau g_R,  g = lambda/(rho b_s)
    #   constitutive split (the ONLY non-equilibrium input): the bar carries its
    #   roll-stiffness share of the antisymmetric spring-equivalent load,
    #     tau (g_L + g_R) = 2 share S_anti.
    #   At a mirror-symmetric pose g_L = g_R and this reduces to the rule above;
    #   in a rolled pose it keeps BOTH rockers and the bar exactly balanced.
    # Without the geometry: the rocker-level split (bar balance only at a
    # symmetric pose) and arb_link_N = NaN.
    for axle_key, (lL, lR) in (('front', ('FL', 'FR')), ('rear', ('RL', 'RR'))):
        is_front = axle_key == 'front'
        cL, cR = results.get(lL), results.get(lR)
        if cL is None or cR is None:
            continue
        ax_top = getattr(topology, axle_key, None) if topology is not None else None
        arb_type = str(getattr(getattr(ax_top, 'arb_type', None), 'value', 'bellcrank'))
        share = 0.0
        if veh is not None:
            try:
                mode = str(getattr(veh, 'topology_mode_front' if is_front
                                   else 'topology_mode_rear', 'standard'))
                k_arb = float(veh.arb_rate_front_Npm if is_front else veh.arb_rate_rear_Npm)
                if mode != 'decoupled' and k_arb > 0.0 and arb_type != 'none':
                    t = float(veh.front_track_m if is_front else veh.rear_track_m)
                    K_roll = float(veh._roll_stiffness_for_axle(is_front))
                    share = (k_arb * t * t / 2.0) / K_roll if K_roll > 0 else 0.0
            except Exception:
                share = 0.0
        if arb_type == 'control_arm' and share > 0.0:
            for c in (cL, cR):
                if c.valid:
                    c.valid = False
                    c.invalid_reason = ('control-arm ARB: its drop-link force on the '
                                        'lower arm is not in this member model')
            continue
        if arb_type not in ('bellcrank', 'tbar'):
            share = 0.0
        sL, sR = cL.spring_rocker_only_N, cR.spring_rocker_only_N
        if not (np.isfinite(sL) and np.isfinite(sR)):
            continue
        s_sym = 0.5 * (sL + sR)
        s_anti = 0.5 * (sL - sR)
        if share <= 0.0:
            continue                       # no bar: rocker-only springs stand
        geo = {}
        if arb_geometry is not None and cL.state is not None and cR.state is not None:
            try:
                for lbl, c in ((lL, cL), (lR, cR)):
                    g = arb_geometry(lbl, c.state)
                    st = c.state
                    a = np.asarray(getattr(solvers.get(lbl), '_rocker_axis'), float)
                    a = a / np.linalg.norm(a)
                    P = np.asarray(st.rocker_pivot, float)
                    dt = np.asarray(g['drop_top'], float)
                    ae = np.asarray(g['arm_end'], float)
                    ap = np.asarray(g['pivot'], float)
                    u = _normalize(ae - dt)
                    lam = float(np.cross(dt - P, u) @ a)
                    rho = float(np.cross(ae - ap, -u) @ np.array([1.0, 0.0, 0.0]))
                    _S, b_s, _m = rocker_required_spring(
                        pushrod_inner=st.pushrod_inner, pushrod_outer=st.pushrod_outer,
                        pushrod_N=c.pushrod_N, rocker_pivot=P, rocker_axis=a,
                        rocker_spring_pt=st.rocker_spring_pt,
                        spring_chassis_pt=st.spring_chassis_pt)
                    geo[lbl] = (lam, rho, b_s)
            except Exception:
                geo = {}
        ok_geo = (len(geo) == 2 and all(np.isfinite(v) for t in geo.values() for v in t)
                  and all(abs(t[1]) > 1e-4 and abs(t[2]) > 1e-6 for t in geo.values()))
        if ok_geo:
            (lamL, rhoL, bL), (lamR, rhoR, bR) = geo[lL], geo[lR]
            gL, gR = lamL / (rhoL * bL), lamR / (rhoR * bR)
            if abs(gL + gR) > 1e-9:
                tau = 2.0 * share * s_anti / (gL + gR)
                cL.spring_force_N = sL - tau * gL
                cR.spring_force_N = sR + tau * gR
                cL.arb_link_N, cR.arb_link_N = tau / rhoL, -tau / rhoR
                cL.arb_bar_torque_Nm, cR.arb_bar_torque_Nm = tau, -tau
                cL.arb_spring_equiv_N = sL - cL.spring_force_N
                cR.arb_spring_equiv_N = sR - cR.spring_force_N
                continue
            note = ('ARB drop link has no lever on the rockers (g_L + g_R = 0): '
                    'bar load indeterminate')
            for c in (cL, cR):
                c.spring_force_N = nan; c.arb_link_N = nan; c.spring_arb_note = note
            continue
        # no usable ARB geometry: rocker-level split (bar balanced only at a
        # mirror-symmetric pose); drop-link force left to rocker_arb_freebody.
        cL.spring_force_N = s_sym + (1.0 - share) * s_anti
        cR.spring_force_N = s_sym - (1.0 - share) * s_anti
        cL.arb_spring_equiv_N = sL - cL.spring_force_N
        cR.arb_spring_equiv_N = sR - cR.spring_force_N
        cL.arb_link_N = cR.arb_link_N = nan
        for c in (cL, cR):
            c.spring_arb_note = ('ARB geometry not supplied: spring/bar split at the '
                                 'rocker level (bar balance exact only at a symmetric pose)')

    return results
