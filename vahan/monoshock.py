"""
vahan/monoshock.py — DECOUPLED twin-bellcrank suspension solver.

cross-car push-pull "Push-Pull Heave and Roll Dampers" layout
(common on Formula Student cars), NOT a single monoshock cradle.

──────────────────────────────────────────────────────────────────────
Geometry
──────────────────────────────────────────────────────────────────────
Per axle there are:
  * TWO bellcranks, one per side.  Each pivots about its OWN
    longitudinal axis (≈ chassis fore-aft direction).  The two pivots
    are NOT on a common axis — they're at mirrored Y positions on
    opposite sides of the chassis centreline.
  * Each pushrod runs wheel → its OWN side's bellcrank.
  * NO corner springs.
  * HEAVE coilover: cross-car damper between the two rockers.  Both
    attach points are at the SAME Z (above each rocker's pivot) and
    mirror-symmetric in X.  In HEAVE both rockers rotate in mirrored
    directions → the two heave attaches move toward each other in X
    → damper compresses.  In ROLL the rockers rotate oppositely (one
    +α, one −α in the world frame) → both ends translate in the SAME
    X direction → damper length is unchanged to first order.
  * ROLL coilover: also cross-car, but with ASYMMETRIC Z attach:
    LEFT-side attach BELOW its pivot, RIGHT-side attach ABOVE its
    pivot.  In HEAVE the two ends move in OPPOSITE Z directions →
    damper length unchanged to first order.  In ROLL the two ends
    move in the SAME Z direction → damper compresses.

This decouples the two modes: heave stiffness comes ENTIRELY from
the heave damper, roll stiffness ENTIRELY from the roll damper, and
each can be tuned independently with no compromise.

──────────────────────────────────────────────────────────────────────
Solve interface
──────────────────────────────────────────────────────────────────────
    solver = TwinRockerDecoupledSolver(hp_dict,
                                       pushrod_outer_left_design,
                                       pushrod_outer_right_design)
    state = solver.solve(pushrod_outer_left_world,
                         pushrod_outer_right_world)

`state` (TwinRockerDecoupledState) carries:
    theta_L, theta_R       rocker rotation angles (rad)
    heave_delta, roll_delta    spring compression (m, +ve = compressed)
    heave_length, roll_length  current cross-car damper lengths (m)
    heave_L, heave_R       updated heave-attach world positions
    roll_L,  roll_R        updated roll-attach world positions
    pushrod_inner_L, _R    updated pushrod-inner positions
    converged              bool

2-D Newton-Raphson with analytic 2×2 Jacobian.  Two unknowns
(theta_L, theta_R), two pushrod-length residuals (left + right).

──────────────────────────────────────────────────────────────────────
Backward-compat alias
──────────────────────────────────────────────────────────────────────
The previous (wrong) MonoshockCradleSolver class is kept as a thin
alias forwarding to the new solver, so save-file load paths that
reference the old name don't break.  Old save files containing
``cradle_pivot`` / ``lateral_spring_*`` / ``oblique_spring_*`` keys
will need to be re-built from the new defaults (set_topology after
load will detect missing keys and repopulate).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ─── helpers ────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot_about_axis(p: np.ndarray, pivot: np.ndarray, axis_unit: np.ndarray,
                    theta: float) -> np.ndarray:
    """Rotate point p about line (pivot, unit axis) by theta rad (Rodrigues)."""
    r = p - pivot
    c, s = np.cos(theta), np.sin(theta)
    return pivot + c * r + s * np.cross(axis_unit, r) \
                 + (1.0 - c) * np.dot(axis_unit, r) * axis_unit


# ─── state ──────────────────────────────────────────────────────────────────

@dataclass
class TwinRockerDecoupledState:
    """Result of a single solve call on a twin-rocker decoupled axle."""
    theta_L: float       # left rocker rotation (rad)
    theta_R: float       # right rocker rotation (rad)
    heave_delta: float   # heave damper compression (m, +ve = compressed)
    roll_delta: float    # roll damper compression (m, +ve = compressed)
    heave_length: float  # current heave damper length (m)
    roll_length: float   # current roll damper length (m)
    heave_L: np.ndarray  # updated heave attach on LEFT rocker (world m)
    heave_R: np.ndarray  # updated heave attach on RIGHT rocker (world m)
    roll_L:  np.ndarray  # updated roll attach on LEFT rocker (world m)
    roll_R:  np.ndarray  # updated roll attach on RIGHT rocker (world m)
    pushrod_inner_L: np.ndarray
    pushrod_inner_R: np.ndarray
    converged: bool


# ─── solver ─────────────────────────────────────────────────────────────────

class TwinRockerDecoupledSolver:
    """2-DOF (theta_L, theta_R) solver for the reference twin-bellcrank
    decoupled suspension.

    Construction freezes:
      * Both pivot axes (pivot point + unit axis) for LEFT and RIGHT
      * Design-pose positions of every rocker reference point
      * Pushrod lengths (computed from the supplied design-pose
        pushrod_outer positions)
      * Free lengths of both cross-car coilovers (computed at design
        pose so spring delta at design = 0)
    """

    def __init__(self,
                 hp_dict: dict,
                 pushrod_outer_left_design: np.ndarray,
                 pushrod_outer_right_design: np.ndarray):
        # LEFT rocker geometry
        self._pivot_L = np.asarray(hp_dict['rocker_pivot_left'], float).copy()
        self._axis_L  = _norm(np.asarray(hp_dict['rocker_axis_pt_left'], float)
                               - self._pivot_L)
        self._p_pin_L0   = np.asarray(hp_dict['pushrod_inner_left'], float).copy()
        self._p_heave_L0 = np.asarray(hp_dict['heave_damper_left'],  float).copy()
        self._p_roll_L0  = np.asarray(hp_dict['roll_damper_left'],   float).copy()

        # RIGHT rocker geometry
        self._pivot_R = np.asarray(hp_dict['rocker_pivot_right'], float).copy()
        self._axis_R  = _norm(np.asarray(hp_dict['rocker_axis_pt_right'], float)
                               - self._pivot_R)
        self._p_pin_R0   = np.asarray(hp_dict['pushrod_inner_right'], float).copy()
        self._p_heave_R0 = np.asarray(hp_dict['heave_damper_right'],  float).copy()
        self._p_roll_R0  = np.asarray(hp_dict['roll_damper_right'],   float).copy()

        # Fall back to world +X axis if a pivot axis is degenerate
        if float(np.linalg.norm(self._axis_L)) < 0.5:
            self._axis_L = np.array([1., 0., 0.])
        if float(np.linalg.norm(self._axis_R)) < 0.5:
            self._axis_R = np.array([1., 0., 0.])

        # Pushrod lengths at design pose — CONSTANT (rigid rods)
        self._L_push_L = float(np.linalg.norm(
            self._p_pin_L0 - np.asarray(pushrod_outer_left_design, float)))
        self._L_push_R = float(np.linalg.norm(
            self._p_pin_R0 - np.asarray(pushrod_outer_right_design, float)))

        # Free spring lengths at design pose (spring delta = 0 there)
        self._L_heave_free = float(np.linalg.norm(
            self._p_heave_L0 - self._p_heave_R0))
        self._L_roll_free  = float(np.linalg.norm(
            self._p_roll_L0  - self._p_roll_R0))

        # Warm-start cache
        self._last_tL = 0.0
        self._last_tR = 0.0

    # ── Pose helpers ─────────────────────────────────────────────────────
    def _pose_L(self, p0: np.ndarray, theta_L: float) -> np.ndarray:
        return _rot_about_axis(p0, self._pivot_L, self._axis_L, theta_L)

    def _pose_R(self, p0: np.ndarray, theta_R: float) -> np.ndarray:
        return _rot_about_axis(p0, self._pivot_R, self._axis_R, theta_R)

    def _residuals(self, theta_L: float, theta_R: float,
                   p_out_L: np.ndarray, p_out_R: np.ndarray) -> np.ndarray:
        pin_L = self._pose_L(self._p_pin_L0, theta_L)
        pin_R = self._pose_R(self._p_pin_R0, theta_R)
        d_L = pin_L - p_out_L
        d_R = pin_R - p_out_R
        return np.array([
            float(d_L @ d_L) - self._L_push_L ** 2,
            float(d_R @ d_R) - self._L_push_R ** 2,
        ])

    def _jacobian(self, theta_L: float, theta_R: float,
                  p_out_L: np.ndarray, p_out_R: np.ndarray) -> np.ndarray:
        """2×2 analytic Jacobian.

        d/dθ_L of |P_L − p_out_L|²  = 2 (P_L − p_out_L) · (∂P_L/∂θ_L)
        where ∂P_L/∂θ_L = axis_L × (P_L − pivot_L)  (rigid rotation).
        Off-diagonals are zero (LEFT residual doesn't depend on θ_R).
        """
        pin_L = self._pose_L(self._p_pin_L0, theta_L)
        pin_R = self._pose_R(self._p_pin_R0, theta_R)
        dpin_L_dtL = np.cross(self._axis_L, pin_L - self._pivot_L)
        dpin_R_dtR = np.cross(self._axis_R, pin_R - self._pivot_R)
        d_L = pin_L - p_out_L
        d_R = pin_R - p_out_R
        return np.array([
            [2.0 * (d_L @ dpin_L_dtL), 0.0],
            [0.0, 2.0 * (d_R @ dpin_R_dtR)],
        ])

    # ── Public solve ─────────────────────────────────────────────────────
    def solve(self,
              pushrod_outer_left_world: np.ndarray,
              pushrod_outer_right_world: np.ndarray,
              warm_start: bool = True,
              max_iter: int = 25,
              tol: float = 1e-12,
              max_abs_theta_rad: float = 1.047) -> TwinRockerDecoupledState:
        """Solve cradle pose for live pushrod_outer positions.

        Newton-Raphson typically converges in 4-6 iterations.

        max_abs_theta_rad: hard cap on the solution magnitude (default
            60 deg).  The bellcrank rotation has TWO valid roots for a
            given pushrod_outer (e.g. +16 deg AND +138 deg both satisfy
            |pin_rotated - po| = L_pushrod).  We want the SMALL-angle
            root.  If the warm-start landed near a large-angle root from
            a prior solve, the Newton iteration stays there.  Detect
            this and restart from theta=0 with warm_start=False.
        """
        p_out_L = np.asarray(pushrod_outer_left_world, float)
        p_out_R = np.asarray(pushrod_outer_right_world, float)

        def _newton(t_L0, t_R0):
            tL, tR = t_L0, t_R0
            conv = False
            for _ in range(max_iter):
                r = self._residuals(tL, tR, p_out_L, p_out_R)
                if float(np.linalg.norm(r)) < tol:
                    conv = True
                    break
                J = self._jacobian(tL, tR, p_out_L, p_out_R)
                try:
                    step = np.linalg.solve(J, -r)
                except np.linalg.LinAlgError:
                    step = -1e-3 * J.T @ r
                step = np.clip(step, -0.2, 0.2)   # cap to ~11° per iter
                tL += float(step[0])
                tR += float(step[1])
            return tL, tR, conv

        # Try warm-start first
        theta_L0 = self._last_tL if warm_start else 0.0
        theta_R0 = self._last_tR if warm_start else 0.0
        theta_L, theta_R, converged = _newton(theta_L0, theta_R0)

        # Reject large-angle alternate roots and retry from theta=0.
        # The bellcrank constraint has two valid solutions per pushrod_outer
        # (one with small theta, one ~180-2θ).  We want the small one.
        if (converged and (abs(theta_L) > max_abs_theta_rad
                            or abs(theta_R) > max_abs_theta_rad)):
            theta_L, theta_R, converged = _newton(0.0, 0.0)

        self._last_tL = theta_L
        self._last_tR = theta_R

        # Recover updated attach positions for visualisation + spring deltas
        pin_L   = self._pose_L(self._p_pin_L0,   theta_L)
        pin_R   = self._pose_R(self._p_pin_R0,   theta_R)
        heave_L = self._pose_L(self._p_heave_L0, theta_L)
        heave_R = self._pose_R(self._p_heave_R0, theta_R)
        roll_L  = self._pose_L(self._p_roll_L0,  theta_L)
        roll_R  = self._pose_R(self._p_roll_R0,  theta_R)
        L_heave = float(np.linalg.norm(heave_L - heave_R))
        L_roll  = float(np.linalg.norm(roll_L  - roll_R))

        return TwinRockerDecoupledState(
            theta_L=theta_L, theta_R=theta_R,
            heave_delta=self._L_heave_free - L_heave,
            roll_delta=self._L_roll_free  - L_roll,
            heave_length=L_heave, roll_length=L_roll,
            heave_L=heave_L, heave_R=heave_R,
            roll_L=roll_L,   roll_R=roll_R,
            pushrod_inner_L=pin_L, pushrod_inner_R=pin_R,
            converged=converged,
        )

    # ── Convenience for dynamics ─────────────────────────────────────────
    def wheel_rate_matrix(self,
                          pushrod_outer_left_design: np.ndarray,
                          pushrod_outer_right_design: np.ndarray,
                          k_heave_Npm: float, k_roll_Npm: float,
                          step_m: float = 1e-3) -> np.ndarray:
        """Finite-difference the 2×2 wheel-rate matrix at design pose.

        Returns K such that:
            [F_heave_mode, F_roll_mode]  =  K  @  [Δz_left_wheel, Δz_right_wheel]
        where F = spring force from each damper.

        Diagonal-dominant K means the modes are well-decoupled:
            K_heave ≈ heave_spring × (geometric ratio for heave)
            K_roll  ≈ roll_spring  × (geometric ratio for roll)
        with the off-diagonals quantifying any residual cross-coupling.
        """
        p_out_L0 = np.asarray(pushrod_outer_left_design, float)
        p_out_R0 = np.asarray(pushrod_outer_right_design, float)
        s0 = self.solve(p_out_L0, p_out_R0)
        F0 = np.array([k_heave_Npm * s0.heave_delta,
                       k_roll_Npm  * s0.roll_delta])

        K = np.zeros((2, 2))
        for col, side in enumerate(('L', 'R')):
            dL = np.array([0., 0., step_m])
            if side == 'L':
                s = self.solve(p_out_L0 + dL, p_out_R0)
            else:
                s = self.solve(p_out_L0, p_out_R0 + dL)
            F = np.array([k_heave_Npm * s.heave_delta,
                          k_roll_Npm  * s.roll_delta])
            K[:, col] = (F - F0) / step_m
        return K


# ─── backward-compat alias ──────────────────────────────────────────────────
# Old name (used by save/load code before the redesign).  Forwards to the
# new class so existing call sites don't break.  When constructed with
# legacy hardpoint dicts (cradle_pivot / lateral_spring_* / oblique_spring_*),
# we raise a clear error rather than silently producing wrong results.

class MonoshockCradleSolver(TwinRockerDecoupledSolver):
    """DEPRECATED alias — use TwinRockerDecoupledSolver directly.

    Old code paths that imported MonoshockCradleSolver will now get the
    twin-bellcrank solver instead.  If the hp_dict has legacy
    cradle/lateral/oblique keys (from the previous wrong implementation),
    we raise a clear ValueError so the caller knows to rebuild from the
    new defaults.
    """
    def __init__(self, hp_dict: dict, pushrod_outer_left, pushrod_outer_right):
        legacy_keys = {'cradle_pivot', 'lateral_spring_cradle',
                       'oblique_spring_cradle'}
        if legacy_keys & set(hp_dict.keys()):
            raise ValueError(
                'MonoshockCradleSolver was renamed to TwinRockerDecoupledSolver '
                'with new hardpoint set (rocker_pivot_left/right, '
                'heave_damper_left/right, roll_damper_left/right).  This '
                'hp_dict still has legacy cradle/lateral/oblique keys — '
                'rebuild it from DEFAULT_FRONT_DECOUPLED / DEFAULT_REAR_DECOUPLED.')
        super().__init__(hp_dict, pushrod_outer_left, pushrod_outer_right)


# Re-export old state-class name too, for callers that referenced it.
MonoshockCradleState = TwinRockerDecoupledState
