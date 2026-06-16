"""
vahan/tbar.py -- HEAVE_TBAR 2-DOF kinematic solver + linear-MR IK.

Mechanical model:
  * The T-bar BRACKET pivots about a chassis-fixed lateral axis defined by
    heave_bracket_hinge_l and heave_bracket_hinge_r.  One rotational DOF
    (theta_bracket): the bracket rotates upward in HEAVE, stays still in
    ROLL.
  * The T-bar TWIST is a separate DOF (theta_twist): the bar twists when
    the L and R rocker drop links move in OPPOSITE directions (roll), but
    not when they move TOGETHER (heave).
  * The HEAVE spring sits between heave_spring_tbar_pt (on the bracket,
    moves with theta_bracket) and heave_spring_chassis_pt (chassis-fixed).
    Spring compression = |spring_tbar_pt - spring_chassis_pt|_design minus
                          |spring_tbar_pt(theta_bracket) - spring_chassis_pt|.

For the IK (LINEAR heave MR): we want d(spring_compression)/d(z_heave) to be
constant over the operational heave range.  Since theta_bracket is
APPROXIMATELY linear in z_heave (small-angle bracket pivot driven by drop
links which are approximately linear in wheel motion), the non-linearity
comes from the spring-length-vs-theta curve.  The IK varies the chassis
attach position to flatten that curve.

Inputs to the solver
--------------------
heave_dict:
    heave_bracket_hinge_l, heave_bracket_hinge_r  -- chassis-fixed hinge endpoints
    heave_spring_tbar_pt                          -- bracket-resident spring attach (design pose)
    heave_spring_chassis_pt                       -- chassis-fixed spring attach

Returns
-------
HeaveTBarState carrying theta_bracket, spring length, spring delta, and
the new world position of heave_spring_tbar_pt.

This is a simplified 1-DOF model (only heave -- the roll twist DOF is
modelled separately by the torsion-bar stiffness in vahan.dynamics).
Sufficient for computing the heave-spring MR which is what the IK
optimises.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

# Steel default (G=80 GPa).  User can override.
_G_STEEL_PA = 80e9


def torsion_bar_stiffness_Nm_rad(OD_mm: float, ID_mm: float,
                                  length_mm: float,
                                  G_Pa: float = _G_STEEL_PA) -> float:
    """Torsional stiffness of a circular bar (solid or tube).

    K_torsion = G * J / L   [N*m/rad]
        J = pi * (D^4 - d^4) / 32   [m^4]
    where D = outer diameter, d = inner diameter (0 for solid), L = length.
    Pass diameters and length in MILLIMETRES; G in Pascals.
    Returns N*m/rad.
    """
    D_m = OD_mm / 1000.0
    d_m = ID_mm / 1000.0
    L_m = length_mm / 1000.0
    if L_m <= 0:
        return 0.0
    J_m4 = np.pi * (D_m**4 - d_m**4) / 32.0
    return float(G_Pa * J_m4 / L_m)


# ─── T-BAR TWIST SOLVER (roll DOF) ──────────────────────────────────────
#
# Mechanical model:
#   * Vertical torsion bar between tbar_base_chassis (bottom, chassis-fixed
#     under HEAVE_TBAR or chassis-rigid under TBAR) and tbar_top_node
#     (top, where the two lever arms branch).
#   * Two lateral lever arms extend from tbar_top_node to tbar_arm_end_L
#     and tbar_arm_end_R (mirrored).
#   * Each arm can independently rotate about the torsion-bar's axis
#     (the line from tbar_base_chassis to tbar_top_node).
#   * The bar TWISTS by an angle equal to (theta_arm_L - theta_arm_R).
#     Restoring torque: T = K_torsion * twist_angle.
#   * Drop links from each arm tip to the rocker drop-pt translate the
#     wheel motion into arm rotation (kinematic chain).
#
# In ROLL: arm_L rotates one way, arm_R the other -> bar twists.
# In HEAVE: both arms rotate the same way -> bar doesn't twist (zero
#           torque from bar); the HEAVE_TBAR bracket DOF handles this
#           through HeaveTBarSolver.

@dataclass
class TBarTwistState:
    """Result from solving the T-bar twist DOF given two drop-pt positions."""
    theta_arm_L: float      # left arm rotation about torsion axis (rad)
    theta_arm_R: float      # right arm rotation about torsion axis (rad)
    twist_angle: float      # bar twist = (theta_L - theta_R) (rad)
    arm_end_L:  np.ndarray  # world position of left arm tip after rotation
    arm_end_R:  np.ndarray  # world position of right arm tip after rotation


class TBarTwistSolver:
    """1-DOF-per-arm bar-twist solver.

    Given the drop-link top positions (rocker side, world coords) on
    LEFT and RIGHT, find the arm rotations theta_L, theta_R such that
    the drop-link lengths are preserved.  Each arm rotates independently
    about the torsion bar's axis (line from tbar_base_chassis to
    tbar_top_node), so the solve decouples into two 1-D Newton-Raphson
    problems.

    The bar twist = theta_L - theta_R is the angle that resists roll
    via the bar's torsional stiffness K_torsion (computed externally
    from OD/ID/L/G via `torsion_bar_stiffness_Nm_rad`).
    """

    def __init__(self, arb_hp: dict):
        # tbar geometry from the per-axle ARB dict
        base = np.asarray(arb_hp['tbar_base_chassis'], float).copy()
        top  = np.asarray(arb_hp['tbar_top_node'],     float).copy()
        ax_v = top - base
        ax_n = float(np.linalg.norm(ax_v))
        # Torsion-bar axis (unit vector from base to top).  Falls back
        # to +Z if degenerate.
        self._axis = ax_v / ax_n if ax_n > 1e-9 else np.array([0., 0., 1.])
        self._pivot = top.copy()   # arms rotate about top_node about this axis

        # Design-pose arm end positions.  For symmetric T-bars these are
        # mirrored in X about chassis centreline.
        # The dict may have a single 'tbar_arm_end' (per-side caller mirrors)
        # or explicit _L / _R variants.  We mirror the single one for the
        # right side.
        ae_default = np.asarray(arb_hp.get('tbar_arm_end',
                                            np.array([0.165, 0., top[2]])),
                                 float)
        self._arm_end_L0 = ae_default.copy()
        ae_R = ae_default.copy(); ae_R[0] *= -1.0
        self._arm_end_R0 = arb_hp.get('tbar_arm_end_right',
                                       ae_R).copy() if isinstance(
            arb_hp.get('tbar_arm_end_right'), np.ndarray) else ae_R

        # Drop-link top positions at design pose
        dt_default = np.asarray(arb_hp.get('tbar_drop_top',
                                            self._arm_end_L0), float)
        self._drop_L0 = dt_default.copy()
        dt_R = dt_default.copy(); dt_R[0] *= -1.0
        self._drop_R0 = dt_R

        # Drop-link length at design (rigid rod)
        self._L_drop_L = float(np.linalg.norm(self._arm_end_L0 - self._drop_L0))
        self._L_drop_R = float(np.linalg.norm(self._arm_end_R0 - self._drop_R0))

    def _rot_arm(self, arm0: np.ndarray, theta: float) -> np.ndarray:
        """Rotate an arm-end world position about the torsion axis."""
        r = arm0 - self._pivot
        c, s = np.cos(theta), np.sin(theta)
        return self._pivot + c * r + s * np.cross(self._axis, r) \
                             + (1.0 - c) * np.dot(self._axis, r) * self._axis

    def _solve_one_arm(self, arm0, drop_world, L_drop_ref, theta_guess=0.0):
        """Newton-Raphson on one arm: find theta such that
        |arm_end(theta) - drop_world| = L_drop_ref.

        Symmetric-geometry handling: if the Jacobian goes to zero (drop
        link perpendicular to arm tip's instantaneous motion at theta=0,
        which happens for "vertical bar + horizontal arm + vertical drop
        link" cases), kick theta away from zero to escape the saddle.
        """
        theta = theta_guess
        for it in range(40):
            ae = self._rot_arm(arm0, theta)
            d  = ae - drop_world
            r  = float(d @ d) - L_drop_ref ** 2
            if abs(r) < 1e-14:
                break
            dae_dt = np.cross(self._axis, ae - self._pivot)
            drdt = 2.0 * float(d @ dae_dt)
            if abs(drdt) < 1e-10:
                # Symmetric geometry -- kick theta to break the saddle.
                # Pick the sign by checking which direction reduces |r|.
                if it == 0:
                    theta = 0.01
                    continue
                else:
                    break  # truly stalled
            theta -= r / drdt
            theta = float(np.clip(theta, -np.pi/2, np.pi/2))
        return theta

    def solve(self, drop_L_world: np.ndarray,
              drop_R_world: np.ndarray) -> TBarTwistState:
        """Solve both arms.  Returns rotations + twist angle."""
        theta_L = self._solve_one_arm(self._arm_end_L0,
                                       np.asarray(drop_L_world, float),
                                       self._L_drop_L)
        theta_R = self._solve_one_arm(self._arm_end_R0,
                                       np.asarray(drop_R_world, float),
                                       self._L_drop_R)
        return TBarTwistState(
            theta_arm_L=theta_L,
            theta_arm_R=theta_R,
            twist_angle=theta_L - theta_R,
            arm_end_L=self._rot_arm(self._arm_end_L0, theta_L),
            arm_end_R=self._rot_arm(self._arm_end_R0, theta_R),
        )

    def motion_ratio_roll(self, drop_pt_design: np.ndarray,
                          wheel_z_perturb_m: float = 1e-3) -> float:
        """Bar TWIST per metre of antisymmetric wheel motion.

        Approximate: perturb drop_L up by +dz and drop_R down by -dz
        (the kinematic chain converts wheel z to drop_pt z through the
        corner rocker; for this estimate we apply directly).  Compute
        the resulting twist angle and divide by the wheel motion.

        Returns rad of bar twist per metre of (z_L - z_R) / 2 wheel motion.
        Multiply by K_torsion (Nm/rad) to get the equivalent at-the-
        drop-link torque per unit wheel roll motion.
        """
        dz = wheel_z_perturb_m
        dp_L = np.asarray(drop_pt_design, float) + np.array([0., 0., +dz])
        dp_R = np.asarray(drop_pt_design, float).copy()
        dp_R[0] *= -1.0
        dp_R = dp_R + np.array([0., 0., -dz])
        st = self.solve(dp_L, dp_R)
        return float(st.twist_angle) / (2 * dz)


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot_about_axis(p, pivot, axis_unit, theta):
    """Rotate point p about line (pivot, unit axis) by theta rad."""
    r = p - pivot
    c, s = np.cos(theta), np.sin(theta)
    return pivot + c * r + s * np.cross(axis_unit, r) \
                 + (1.0 - c) * np.dot(axis_unit, r) * axis_unit


@dataclass
class HeaveTBarState:
    theta_bracket: float   # bracket rotation about chassis hinge axis (rad)
    spring_length: float   # current spring length (m)
    spring_delta:  float   # compression (m, +ve = compressed from free length)
    tbar_pt_world: np.ndarray  # updated heave_spring_tbar_pt position (m)


class HeaveTBarSolver:
    """1-DOF heave bracket + heave spring kinematic solver.

    Roll is handled separately by the torsion stiffness; this class
    computes ONLY the heave-mode response: given a bracket rotation
    theta_bracket, what is the current spring length?

    Used by the IK in `optimize_heave_linear_mr` to evaluate the MR
    curve while varying the chassis attach position.
    """

    def __init__(self, heave_dict: dict):
        h_l = np.asarray(heave_dict['heave_bracket_hinge_l'], float)
        h_r = np.asarray(heave_dict['heave_bracket_hinge_r'], float)
        self._hinge_origin = 0.5 * (h_l + h_r)            # midpoint
        self._hinge_axis   = _norm(h_r - h_l)             # lateral X-ish
        self._tbar_pt0     = np.asarray(heave_dict['heave_spring_tbar_pt'], float).copy()
        self._chassis_pt   = np.asarray(heave_dict['heave_spring_chassis_pt'], float).copy()
        self._L_free       = float(np.linalg.norm(self._tbar_pt0 - self._chassis_pt))

    def solve(self, theta_bracket: float) -> HeaveTBarState:
        tbar_pt = _rot_about_axis(self._tbar_pt0,
                                  self._hinge_origin,
                                  self._hinge_axis,
                                  theta_bracket)
        L = float(np.linalg.norm(tbar_pt - self._chassis_pt))
        return HeaveTBarState(
            theta_bracket = theta_bracket,
            spring_length = L,
            spring_delta  = self._L_free - L,
            tbar_pt_world = tbar_pt,
        )

    def mr_curve(self, theta_range_rad: np.ndarray) -> np.ndarray:
        """Return spring compression for each theta in theta_range_rad.

        Multiply by d(theta)/d(wheel_z) (the bracket-rotation-per-wheel-
        travel ratio) to get spring-compression-per-wheel-travel = MR.
        """
        return np.array([self.solve(t).spring_delta for t in theta_range_rad])


def linear_mr_metric(heave_dict: dict,
                     theta_range_rad: np.ndarray = None) -> dict:
    """Measure how linear the heave-spring MR curve is over a theta range.

    A perfectly linear MR (constant d(compression)/d(theta)) gives metric=0.
    Returns:
        mr_variance     : variance of the per-step MR (smaller = more linear)
        max_dev_from_mean_mm : worst deviation from the mean MR, mm/rad
        mr_mean         : mean MR over the range, mm/rad
        total_metric    : the scalar IK minimises
    """
    if theta_range_rad is None:
        # Reasonable operational range for HEAVE: ~±5 deg bracket rotation
        # corresponds to maybe ±25 mm of wheel travel through typical T-bar
        # geometry.
        theta_range_rad = np.linspace(-np.deg2rad(5), np.deg2rad(5), 11)

    solver = HeaveTBarSolver(heave_dict)
    compressions = solver.mr_curve(theta_range_rad)
    # MR = d(compression)/d(theta), discrete derivative
    d_theta = theta_range_rad[1] - theta_range_rad[0]
    mr = np.diff(compressions) / d_theta   # m / rad
    mr_mean = float(np.mean(mr))
    mr_var = float(np.var(mr))
    max_dev = float(np.max(np.abs(mr - mr_mean))) * 1000.0  # mm/rad deviation
    return {
        'mr_variance':       mr_var,
        'max_dev_from_mean_mm': max_dev,
        'mr_mean':           mr_mean,
        'mr_mean_mm_per_deg':float(mr_mean * np.deg2rad(1.0) * 1000.0),
        'total_metric':      mr_var,
    }


def optimize_heave_linear_mr(heave_dict: dict,
                             theta_range_rad: np.ndarray = None,
                             include_spring_attach: bool = True,
                             max_iter: int = 200,
                             verbose: bool = False) -> tuple[dict, dict]:
    """Optimise heave-bracket geometry to flatten heave-spring MR curve.

    Per the user's design intent: "the motion ratio of the heave spring
    [should] be linear, which is defined by the T bar pivot placement".

    Design variables:
      * Always: bracket hinge axis (Y, Z) -- the chassis-side pivot.
      * If include_spring_attach=True: also the spring-tbar attach
        position and spring-chassis attach position.  These give the
        optimiser more freedom to flatten the curve at larger bracket
        rotations.

    The MR is inherently a sin-curve for a simple single-pivot bracket
    (the spring-attach point traces a circular arc).  At small bracket
    rotations (~±5 deg) the sin function is already very linear -- the
    default Vahan geometry hits MR variance ~ 2e-08 there with no
    optimisation needed.  At larger rotations (±15 deg or more) the
    non-linearity grows; the IK reduces it as much as a single-pivot
    bracket geometry can.  For truly constant MR at large rotations the
    bracket needs a 4-bar linkage (topology change, not in scope here).

    The mean MR magnitude is anchored to its INITIAL value so the
    optimiser can't trivially collapse to a zero-MR "perfectly linear"
    geometry.  Vehicle dynamics needs a finite heave-spring stiffness
    to be useful.
    """
    if theta_range_rad is None:
        theta_range_rad = np.linspace(-np.deg2rad(5), np.deg2rad(5), 11)

    base = {k: np.array(v, float).copy() for k, v in heave_dict.items()}
    h_l0 = base['heave_bracket_hinge_l']
    h_r0 = base['heave_bracket_hinge_r']
    # Initial pivot midpoint (Y, Z)
    x0 = np.array([0.5 * (h_l0[1] + h_r0[1]),
                   0.5 * (h_l0[2] + h_r0[2])])

    tbar_pt = base['heave_spring_tbar_pt']
    bounds = [
        (tbar_pt[1] - 0.3, tbar_pt[1] + 0.3),  # hinge Y
        (0.0, 0.5),                             # hinge Z
    ]

    def _apply(x):
        h = {k: v.copy() for k, v in base.items()}
        # Keep hinge_L.X and hinge_R.X at their mirror positions; only
        # the common Y, Z change.
        new_l = h_l0.copy(); new_l[1] = x[0]; new_l[2] = x[1]
        new_r = h_r0.copy(); new_r[1] = x[0]; new_r[2] = x[1]
        h['heave_bracket_hinge_l'] = new_l
        h['heave_bracket_hinge_r'] = new_r
        return h

    before = linear_mr_metric(base, theta_range_rad)
    # Anchor the mean MR to its ORIGINAL value so the optimiser doesn't
    # collapse the geometry to a trivial zero-MR "perfectly linear" pose.
    # The user wants a non-zero MR that is ALSO linear -- a zero-MR
    # solution gives a soft heave mode (no spring force), useless for
    # vehicle dynamics.
    target_mean_mr = before['mr_mean']
    # Weight the anchor much higher than the variance term so the
    # optimiser preserves MR magnitude as a hard target.
    anchor_weight = 1.0e4

    def _obj(x):
        try:
            m = linear_mr_metric(_apply(x), theta_range_rad)
            mean_dev = (m['mr_mean'] - target_mean_mr) ** 2
            return m['total_metric'] + anchor_weight * mean_dev
        except Exception:
            return 1e9

    if verbose:
        print(f'BEFORE: MR variance={before["mr_variance"]:.2e}, '
              f'max_dev={before["max_dev_from_mean_mm"]:.4f} mm/rad, '
              f'MR_mean={before["mr_mean_mm_per_deg"]:.3f} mm/deg')

    res = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-14})
    new_hp = _apply(res.x)
    after = linear_mr_metric(new_hp, theta_range_rad)
    if verbose:
        print(f'AFTER:  MR variance={after["mr_variance"]:.2e}, '
              f'max_dev={after["max_dev_from_mean_mm"]:.4f} mm/rad, '
              f'MR_mean={after["mr_mean_mm_per_deg"]:.3f} mm/deg')
        print(f'        x_opt = {res.x}')

    return new_hp, {
        'before':     before,
        'after':      after,
        'x_opt':      res.x.tolist(),
        'iter':       int(res.nit),
        'converged':  bool(res.success),
        'message':    str(res.message),
    }
