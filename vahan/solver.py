"""
3-D kinematic constraint solver for a double wishbone + pushrod/rocker corner.

──────────────────────────────────────────────────────────────────────────────
HOW IT WORKS
──────────────────────────────────────────────────────────────────────────────
The mechanism has one degree of freedom: wheel travel (Δz of wheel center).

Moving unknowns — 12 scalars packed as x[0..11]:
    x[0:3]  = uca_outer   (UCA outboard balljoint)
    x[3:6]  = lca_outer   (LCA outboard balljoint)
    x[6:9]  = tr_outer    (tie-rod upright pickup)
    x[9:12] = wheel_center

12 constraint equations:
    [0]  |uca_outer − uca_front|²  = L²  (UCA front arm length)
    [1]  |uca_outer − uca_rear|²   = L²  (UCA rear arm length)
    [2]  |lca_outer − lca_front|²  = L²  (LCA front arm length)
    [3]  |lca_outer − lca_rear|²   = L²  (LCA rear arm length)
    [4]  |lca_outer − uca_outer|²  = L²  (upright height)
    [5]  |tr_outer  − uca_outer|²  = L²  (upright: tie-rod to UCA BJ)
    [6]  |tr_outer  − lca_outer|²  = L²  (upright: tie-rod to LCA BJ)
    [7]  |tr_outer  − tr_inner|²   = L²  (tie-rod length)
    [8]  |wc        − uca_outer|²  = L²  (upright: WC to UCA BJ)
    [9]  |wc        − lca_outer|²  = L²  (upright: WC to LCA BJ)
    [10] |wc        − tr_outer|²   = L²  = (upright: WC to tie-rod pickup)
    [11] wc_z = wc0_z + travel             (drive constraint)

The Jacobian is derived analytically — no finite differences.

After the main solve, pushrod_outer (also fixed to the upright) is recovered
via a rigid-body frame transform.  Then the rocker angle is found with a
Newton-Raphson 1-D solve (Rodrigues rotation).
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from .hardpoints import DoubleWishboneHardpoints


# ─── small geometry helpers ──────────────────────────────────────────────────

def _d2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two 3-D points."""
    d = a - b
    return float(d @ d)


def _norm(v: np.ndarray) -> np.ndarray:
    """Unit vector."""
    return v / np.linalg.norm(v)


def _build_frame(origin: np.ndarray,
                 pt1: np.ndarray,
                 pt2: np.ndarray) -> np.ndarray:
    """
    Build a right-handed orthonormal frame (3×3 rotation matrix) from three
    non-collinear points.

        e1 = norm(pt1 − origin)
        e3 = norm(e1 × (pt2 − origin))
        e2 = e3 × e1

    Columns are the basis vectors expressed in the world frame.
    """
    e1 = _norm(pt1 - origin)
    v2 = pt2 - origin
    e3 = _norm(np.cross(e1, v2))
    e2 = np.cross(e3, e1)
    return np.column_stack([e1, e2, e3])   # shape (3, 3)


def _rodrigues(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v by angle theta (rad) about unit axis using Rodrigues."""
    c, s = np.cos(theta), np.sin(theta)
    return c * v + s * np.cross(axis, v) + (1.0 - c) * np.dot(axis, v) * axis


# ─── constraint system ───────────────────────────────────────────────────────

class SuspensionConstraints:
    """
    Encapsulates the 12 constraint equations for one suspension corner.

    Instantiate once per corner (at design position).
    Call .solve(travel_m) for each wheel-travel position.
    """

    def __init__(self, hp: DoubleWishboneHardpoints,
                 tierod_len_sq: float | None = None,
                 pushrod_body: str = 'upright'):
        """
        hp             : hardpoints at the current position (may have steered tie_rod_inner)
        tierod_len_sq  : override for the squared tie-rod length.
                         Must be supplied when tie_rod_inner has been moved by rack travel
                         so the physical rod length stays constant regardless of rack pos.
                         If None, computed from hp (correct for non-steered corners).
        pushrod_body   : which rigid body carries pushrod_outer.
                         'upright' (default) — pushrod mounts to the upright.
                         'uca'               — pushrod outer is fixed to the UCA.
                         'lca'               — pushrod outer is fixed to the LCA.
        """
        self.hp = hp
        self._pushrod_body = pushrod_body

        # Precompute squared link lengths from design hardpoints
        self._L2 = {
            'uca_front': _d2(hp.uca_outer, hp.uca_front),
            'uca_rear':  _d2(hp.uca_outer, hp.uca_rear),
            'lca_front': _d2(hp.lca_outer, hp.lca_front),
            'lca_rear':  _d2(hp.lca_outer, hp.lca_rear),
            'upright':   _d2(hp.lca_outer, hp.uca_outer),
            'tr_uca':    _d2(hp.tie_rod_outer, hp.uca_outer),
            'tr_lca':    _d2(hp.tie_rod_outer, hp.lca_outer),
            # Tierod length: use override if supplied (rack-steered case)
            'tierod':    tierod_len_sq if tierod_len_sq is not None
                         else _d2(hp.tie_rod_outer, hp.tie_rod_inner),
            'wc_uca':    _d2(hp.wheel_center,  hp.uca_outer),
            'wc_lca':    _d2(hp.wheel_center,  hp.lca_outer),
            'wc_tr':     _d2(hp.wheel_center,  hp.tie_rod_outer),
        }
        self._wc0_z = float(hp.wheel_center[2])

        # ── upright local frame at design ──────────────────────────────────
        # Origin at lca_outer; e1 toward uca_outer; plane spanned with tr_outer.
        self._upright_origin_0 = hp.lca_outer.copy()
        self._F0 = _build_frame(hp.lca_outer, hp.uca_outer, hp.tie_rod_outer)

        # Wheel spin axis in upright local frame.
        # At design position the axle is horizontal and lateral → [1, 0, 0].
        # This gives exactly 0° camber and 0° toe at the design hardpoints,
        # matching the physical assumption that the car is aligned at rest.
        # As the upright rigid-body rotates during travel, the stored local
        # direction rotates with it, reproducing the correct camber/toe changes.
        spin_world = np.array([1.0, 0.0, 0.0])
        self._spin_axis_local = self._F0.T @ spin_world   # upright-frame

        # ── pushrod outer: body-frame local coords ─────────────────────────
        # Which rigid body carries pushrod_outer determines which frame is used.
        if pushrod_body == 'uca':
            # UCA body: rotates about uca_front -> uca_rear axis
            F_push0 = _build_frame(hp.uca_front, hp.uca_rear, hp.uca_outer)
            self._pushrod_outer_local = F_push0.T @ (hp.pushrod_outer - hp.uca_front)
        elif pushrod_body == 'lca':
            # LCA body: rotates about lca_front -> lca_rear axis
            F_push0 = _build_frame(hp.lca_front, hp.lca_rear, hp.lca_outer)
            self._pushrod_outer_local = F_push0.T @ (hp.pushrod_outer - hp.lca_front)
        else:
            # 'upright' (default): pushrod outer fixed to upright
            self._pushrod_outer_local = self._F0.T @ (hp.pushrod_outer - self._upright_origin_0)

        # ── rocker geometry ────────────────────────────────────────────────
        pivot          = hp.rocker_pivot
        arm_push       = hp.pushrod_inner   - pivot
        arm_spring     = hp.rocker_spring_pt - pivot
        self._rocker_pivot       = pivot.copy()
        self._rocker_arm_push    = arm_push.copy()
        self._rocker_arm_spring  = arm_spring.copy()
        # Rocker rotation axis: use explicit axis point if available (from hardpoints),
        # otherwise fall back to the normal of the two-arm plane.
        axis_pt = getattr(hp, 'rocker_axis_pt', None)
        if axis_pt is not None and np.linalg.norm(axis_pt - pivot) > 1e-9:
            self._rocker_axis = _norm(axis_pt - pivot)
        else:
            self._rocker_axis = _norm(np.cross(arm_push, arm_spring))
        self._L_pushrod   = float(np.sqrt(_d2(hp.pushrod_inner, hp.pushrod_outer)))

    # ── residual vector ──────────────────────────────────────────────────────

    def _residuals(self, x: np.ndarray, travel: float) -> np.ndarray:
        uca = x[0:3]; lca = x[3:6]; tr = x[6:9]; wc = x[9:12]
        L2  = self._L2
        hp  = self.hp
        return np.array([
            _d2(uca, hp.uca_front)      - L2['uca_front'],
            _d2(uca, hp.uca_rear)       - L2['uca_rear'],
            _d2(lca, hp.lca_front)      - L2['lca_front'],
            _d2(lca, hp.lca_rear)       - L2['lca_rear'],
            _d2(lca, uca)               - L2['upright'],
            _d2(tr,  uca)               - L2['tr_uca'],
            _d2(tr,  lca)               - L2['tr_lca'],
            _d2(tr,  hp.tie_rod_inner)  - L2['tierod'],
            _d2(wc,  uca)               - L2['wc_uca'],
            _d2(wc,  lca)               - L2['wc_lca'],
            _d2(wc,  tr)                - L2['wc_tr'],
            wc[2] - (self._wc0_z + travel),
        ])

    # ── analytic Jacobian ────────────────────────────────────────────────────

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        uca = x[0:3]; lca = x[3:6]; tr = x[6:9]; wc = x[9:12]
        hp  = self.hp
        J   = np.zeros((12, 12))

        # f0: |uca − uca_front|²
        J[0, 0:3] = 2*(uca - hp.uca_front)
        # f1: |uca − uca_rear|²
        J[1, 0:3] = 2*(uca - hp.uca_rear)
        # f2: |lca − lca_front|²
        J[2, 3:6] = 2*(lca - hp.lca_front)
        # f3: |lca − lca_rear|²
        J[3, 3:6] = 2*(lca - hp.lca_rear)
        # f4: |lca − uca|²
        J[4, 0:3] = -2*(lca - uca);  J[4, 3:6] =  2*(lca - uca)
        # f5: |tr − uca|²
        J[5, 0:3] = -2*(tr  - uca);  J[5, 6:9] =  2*(tr  - uca)
        # f6: |tr − lca|²
        J[6, 3:6] = -2*(tr  - lca);  J[6, 6:9] =  2*(tr  - lca)
        # f7: |tr − tr_inner|²
        J[7, 6:9] = 2*(tr - hp.tie_rod_inner)
        # f8: |wc − uca|²
        J[8, 0:3]  = -2*(wc - uca);  J[8, 9:12]  =  2*(wc - uca)
        # f9: |wc − lca|²
        J[9, 3:6]  = -2*(wc - lca);  J[9, 9:12]  =  2*(wc - lca)
        # f10: |wc − tr|²
        J[10, 6:9]  = -2*(wc - tr);  J[10, 9:12] =  2*(wc - tr)
        # f11: wc_z = target  →  ∂/∂wc_z = 1
        J[11, 11] = 1.0

        return J

    # ── 1-D rocker solve ─────────────────────────────────────────────────────

    def _solve_rocker(self,
                      pushrod_outer_world: np.ndarray,
                      theta_0: float = 0.0,
                      theta_direction: float = 0.0,
                      spring_prev: float | None = None) -> tuple:
        """
        Find rocker angle θ such that:
            |pivot + R(axis, θ) @ arm_push − pushrod_outer_world|² = L_pushrod²

        Uses Newton-Raphson starting from theta_0 (previous step's angle).
        Returns (theta, pushrod_inner_world, spring_pt_world).
        """
        pivot  = self._rocker_pivot
        arm_p  = self._rocker_arm_push
        arm_s  = self._rocker_arm_spring
        axis   = self._rocker_axis
        L2_pr  = self._L_pushrod ** 2

        def _nr_solve(start):
            """Run Newton-Raphson from a given starting angle. Returns converged theta."""
            th = start
            for _ in range(80):
                arm_rot  = _rodrigues(arm_p, axis, th)
                pi_world = pivot + arm_rot
                r        = pi_world - pushrod_outer_world
                residual = float(r @ r) - L2_pr
                if abs(residual) < 1e-12:
                    break
                d_arm = np.cross(axis, arm_rot)
                drdt  = float(2.0 * r @ d_arm)
                if abs(drdt) < 1e-14:
                    break
                th -= residual / drdt
            # Verify it actually converged
            arm_rot  = _rodrigues(arm_p, axis, th)
            pi_world = pivot + arm_rot
            r        = pi_world - pushrod_outer_world
            converged = abs(float(r @ r) - L2_pr) < 1e-6
            return th, converged

        def _signed_delta(th):
            """Smallest signed angular step from theta_0 to th."""
            d = th - theta_0
            # Wrap to (-pi, pi]
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            return d

        # Try 4 starting points to find both branches robustly
        candidates = []
        for start in (theta_0, theta_0 + np.pi,
                      theta_0 + np.pi / 2, theta_0 - np.pi / 2):
            th, ok = _nr_solve(start)
            if ok:
                d = _signed_delta(th)
                candidates.append((d, th))

        if not candidates:
            # Nothing converged — stay put
            theta = theta_0
        elif len(candidates) == 1:
            theta = candidates[0][1]
        elif spring_prev is not None:
            # PRIMARY criterion: pick the branch whose spring length is most
            # continuous with the previous step.  This is far more reliable than
            # angle-based direction tracking because angle wrapping can cause
            # spurious branch switches while spring length must change smoothly.
            chassis_pt = self.hp.spring_chassis_pt
            def _spring_len_for(th):
                sp = self._rocker_pivot + _rodrigues(self._rocker_arm_spring,
                                                     self._rocker_axis, th)
                return float(np.sqrt(_d2(sp, chassis_pt)))
            theta = min([th for _, th in candidates],
                        key=lambda th: abs(_spring_len_for(th) - spring_prev))
        elif abs(theta_direction) < 1e-9:
            # No previous spring data, no direction: pick closest angle to theta_0
            theta = min(candidates, key=lambda x: abs(x[0]))[1]
        else:
            # Fall back to direction-based selection
            same_dir = [(d, th) for d, th in candidates
                        if d * theta_direction > 0]
            theta = (min(same_dir,    key=lambda x: abs(x[0]))[1] if same_dir
                     else min(candidates, key=lambda x: abs(x[0]))[1])

        arm_spring_rot = _rodrigues(arm_s, axis, theta)
        spring_pt      = pivot + arm_spring_rot
        pushrod_inner  = pivot + _rodrigues(arm_p, axis, theta)
        return theta, pushrod_inner, spring_pt

    # ── main solve ───────────────────────────────────────────────────────────

    def solve(self,
              travel: float,
              x0: np.ndarray | None = None,
              rocker_theta0: float = 0.0,
              rocker_direction: float = 0.0,
              rocker_spring_prev: float | None = None,
              tol: float = 1e-10,
              max_iter: int = 60) -> "SolvedState":
        """
        Solve all point positions for a given wheel travel (meters, + = bump).

        Args:
            travel:        wheel-center vertical displacement from design (m)
            x0:            initial guess for [uca, lca, tr, wc] (12 values).
                           Defaults to design position.  Pass the previous
                           step's solution for robust continuation.
            rocker_theta0: rocker angle warm-start (rad).
            tol:           max absolute residual to accept as converged.
            max_iter:      Newton-Raphson iteration cap.

        Returns:
            SolvedState with every point coordinate populated.
        """
        hp = self.hp

        if x0 is None:
            x0 = np.concatenate([hp.uca_outer, hp.lca_outer,
                                  hp.tie_rod_outer, hp.wheel_center])

        x = x0.copy()
        for i in range(max_iter):
            r = self._residuals(x, travel)
            if np.max(np.abs(r)) < tol:
                break
            J  = self._jacobian(x)
            dx = np.linalg.solve(J, -r)
            x += dx
        else:
            raise RuntimeError(
                f"Main solver did not converge at travel={travel*1000:.2f} mm "
                f"(max residual={np.max(np.abs(r)):.2e})."
            )

        uca_out = x[0:3]
        lca_out = x[3:6]
        tr_out  = x[6:9]
        wc      = x[9:12]
        hp      = self.hp

        # ── upright frame — always used for spin axis ──────────────────────
        F_upright = _build_frame(lca_out, uca_out, tr_out)
        spin_axis_world = _norm(F_upright @ self._spin_axis_local)

        # ── pushrod outer — body-dependent ────────────────────────────────
        if self._pushrod_body == 'uca':
            F_push = _build_frame(hp.uca_front, hp.uca_rear, uca_out)
            pushrod_outer_world = F_push @ self._pushrod_outer_local + hp.uca_front
        elif self._pushrod_body == 'lca':
            F_push = _build_frame(hp.lca_front, hp.lca_rear, lca_out)
            pushrod_outer_world = F_push @ self._pushrod_outer_local + hp.lca_front
        else:
            pushrod_outer_world = F_upright @ self._pushrod_outer_local + lca_out

        # ── rocker ────────────────────────────────────────────────────────
        rocker_theta, pushrod_inner, spring_pt = self._solve_rocker(
            pushrod_outer_world, rocker_theta0, rocker_direction,
            spring_prev=rocker_spring_prev,
        )

        spring_length = float(np.sqrt(
            _d2(spring_pt, self.hp.spring_chassis_pt)
        ))

        return SolvedState(
            travel=travel,
            # primary moving points
            uca_outer      = uca_out.copy(),
            lca_outer      = lca_out.copy(),
            tr_outer       = tr_out.copy(),
            wheel_center   = wc.copy(),
            # upright-derived
            pushrod_outer  = pushrod_outer_world,
            spin_axis      = spin_axis_world,
            # rocker
            pushrod_inner  = pushrod_inner,
            rocker_spring_pt = spring_pt,
            rocker_angle   = rocker_theta,
            spring_length  = spring_length,
            # chassis-fixed references (copied for convenience in kinematics)
            uca_front      = hp.uca_front.copy(),
            uca_rear       = hp.uca_rear.copy(),
            lca_front      = hp.lca_front.copy(),
            lca_rear       = hp.lca_rear.copy(),
            tr_inner       = hp.tie_rod_inner.copy(),
            rocker_pivot   = hp.rocker_pivot.copy(),
            spring_chassis_pt = hp.spring_chassis_pt.copy(),
        )


# ─── solved state ─────────────────────────────────────────────────────────────

class SolvedState:
    """
    All point coordinates (world frame) for a single wheel-travel position.

    Every attribute is a numpy array (3,) unless noted otherwise.
    Use SolvedState.all_moving_points() to get them as a dict.
    """

    __slots__ = [
        'travel',
        'uca_outer', 'lca_outer', 'tr_outer', 'wheel_center',
        'pushrod_outer', 'spin_axis',
        'pushrod_inner', 'rocker_spring_pt',
        'rocker_angle', 'spring_length',
        'uca_front', 'uca_rear', 'lca_front', 'lca_rear',
        'tr_inner', 'rocker_pivot', 'spring_chassis_pt',
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def x_vec(self) -> np.ndarray:
        """Pack [uca, lca, tr, wc] into a 12-vector (solver warm-start format)."""
        return np.concatenate([self.uca_outer, self.lca_outer,
                                self.tr_outer,  self.wheel_center])

    def all_moving_points(self) -> dict:
        """Dict of every point that changes with wheel travel."""
        return {
            'uca_outer':        self.uca_outer,
            'lca_outer':        self.lca_outer,
            'tr_outer':         self.tr_outer,
            'wheel_center':     self.wheel_center,
            'pushrod_outer':    self.pushrod_outer,
            'pushrod_inner':    self.pushrod_inner,
            'rocker_spring_pt': self.rocker_spring_pt,
        }
