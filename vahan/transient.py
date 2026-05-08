"""
vahan/transient.py - Time-domain transient vehicle dynamics.

Simulates turn-in, skidpad entry, step-steer, etc. as a 7-state
bicycle + roll model integrated with RK4.  Uses the existing
tire_model (TTC) for per-wheel lateral forces and leverages
VehicleParams for all vehicle-level constants.

Kinematic effects (camber change, roll-centre migration) are
captured via precomputed lookup tables — one evaluation per
corner at ~20 travel points at init, then cheap interpolation
inside the integration loop.

Coordinate convention (ISO 8855):
    x forward, y to the left, z up.
    positive yaw rate  = turning left
    positive ay        = to the left
    positive steer_rad = steering into a left turn
    positive roll phi  = body rolls to the right
                         (outside of a left turn — outside wheels compress)

Public API:
    TransientParams          dataclass — inertias, damping, ackermann %
    SteeringProfile          callables returning front-wheel steer(t)
    TransientInputs          dataclass — driver inputs over time
    TransientResult          dataclass — full time history + metrics
    TransientSolver          main solver class

Example
-------
    from vahan.transient import (TransientSolver, TransientParams,
                                  TransientInputs, SteeringProfile)
    solver = TransientSolver(vehicle, tire_model, corner_solvers)
    inp = TransientInputs(
        v_x_target_ms=10.4,
        steering=SteeringProfile.ramp(t_start=0.5, t_end=1.0, steer_rad=0.18),
        duration_s=5.0,
        dt_s=0.002,
    )
    result = solver.simulate(inp)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from .dynamics import VehicleParams
from .kinematics import KinematicMetrics
from .steering import SteeringGeometry

G = 9.81


# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransientParams:
    """Dynamics parameters specific to time-domain simulation.

    These are quantities the steady-state solver does not need
    (inertias and damping) and a handful of kinematic simplifications
    (Ackermann %, steering tau) that only matter during transients.
    """

    # Inertias (kg·m²)
    sprung_roll_inertia: float = 40.0      # Ixx about roll axis.  FSAE: ~40
    yaw_inertia: float = 120.0             # Izz about CG.         FSAE: ~100-150

    # Damping
    roll_damping_Nms_rad: float = 1200.0
    # Body roll damping coefficient (N·m·s/rad).  Should be DERIVED from
    # real damper dyno data, not set as a magic number — the GUI dispatch
    # in main_window builds it from the four damper bump/rebound rate
    # inputs as
    #     c_phi = Σ_axle (c_bump + c_rebound) · MR² · t² / 4
    # using the car's motion ratio (shock/wheel) and track width.  A
    # typical FSAE damper combo gives ~1500-2500 N·m·s/rad of body roll
    # damping; the default above is only a fallback for direct callers
    # who construct TransientParams without going through the GUI.

    # Steering kinematics
    ackermann_pct: float = 0.0             # 0 = parallel, 100 = perfect Ackermann,
                                           # negative = anti-Ackermann
    steering_tau_s: float = 0.02
    # 1st-order lag (s) between commanded and actual road-wheel steer.
    # Models actuator bandwidth (autonomous) or driver reaction +
    # rack/column compliance (human).  IMPORTANT: τ interacts with any
    # closed-loop steering controller — a Stanley path-follower expects
    # the actuator to be fast compared with the loop bandwidth, so
    # τ ≈ 0.01-0.02 s for autonomous use.  Larger values (0.10-0.25 s)
    # are realistic for HUMAN driver reaction time and are fine for
    # OPEN-LOOP test profiles (step, ramp, sine), but they introduce
    # significant phase lag that destabilises a closed-loop controller —
    # Stanley + τ=0.2 will limit-cycle.  τ=0 makes the ODE singular
    # (steer_dot = Δ/τ → ∞); the integrator clamps to a 1 ms floor to
    # avoid blow-up but you should never rely on that.

    # Throttle/brake routing
    # These are just a constant-Fx mode for MVP.  Phase 2 can add a proper
    # engine map.  Positive Fx = drive force (accel); negative = braking.
    longitudinal_control_mode: str = 'speed_hold'
    # 'speed_hold' — PID on vx to track v_x_target
    # 'open_loop'  — use TransientInputs.longitudinal_profile directly (N)

    # Speed-hold PI gains.  If None, TransientSolver derives them from
    # the VehicleParams.speed_hold_kp_per_kg × total_mass so heavier cars
    # get proportionally more restoring force for the same error.
    speed_hold_kp: Optional[float] = None   # N per (m/s) error
    speed_hold_ki: Optional[float] = None   # N per (m/s·s) error

    # Max longitudinal force as a multiple of weight.  1.5 ≈ 1.5 g of
    # drive or brake acceleration — a sane clamp well above what FSAE
    # tyres can sustain, so the sim doesn't blow up at large step inputs.
    fx_limit_g: float = 1.5

    # Kinematic lookup grid.  If ``kin_table_travel_mm`` is None, the
    # solver derives the range from the VehicleParams shock stroke
    # (± full stroke / max MR) so every shock size gets a sensible LUT.
    kin_table_travel_mm: Optional[tuple] = None
    kin_table_n: int = 25                        # number of LUT points

    # Physical saturation on the commanded road-wheel angle.  If None,
    # the solver uses the SteeringGeometry's physical limit (preferred);
    # if no geometry is supplied, it falls back to VehicleParams.max_steer_angle_deg.
    max_steer_rad: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Steering profiles
# ─────────────────────────────────────────────────────────────────────────────

class SteeringProfile:
    """Collection of steering-angle-vs-time generators.

    Each method returns a Callable[[float], float] that maps time (s) to
    front-wheel steer angle (rad).  Positive = left turn (ISO).
    """

    @staticmethod
    def constant(steer_rad: float) -> Callable[[float], float]:
        return lambda t: steer_rad

    @staticmethod
    def step(t_step: float, steer_rad: float) -> Callable[[float], float]:
        """Instantaneous step at t_step."""
        return lambda t: steer_rad if t >= t_step else 0.0

    @staticmethod
    def ramp(t_start: float, t_end: float,
             steer_rad: float) -> Callable[[float], float]:
        """Linear ramp from 0 at t_start to steer_rad at t_end, then hold."""
        dt = max(t_end - t_start, 1e-6)
        def _p(t):
            if t < t_start:
                return 0.0
            if t >= t_end:
                return steer_rad
            return steer_rad * (t - t_start) / dt
        return _p

    @staticmethod
    def sine(amplitude_rad: float, frequency_hz: float,
             t_start: float = 0.0,
             n_cycles: float = float('inf')) -> Callable[[float], float]:
        """Sinusoidal steer — useful for frequency-response tests."""
        omega = 2 * np.pi * frequency_hz
        T = n_cycles / frequency_hz if n_cycles != float('inf') else float('inf')
        def _p(t):
            if t < t_start or (t - t_start) > T:
                return 0.0
            return amplitude_rad * np.sin(omega * (t - t_start))
        return _p

    @staticmethod
    def skidpad(radius_m: float, wheelbase_m: float,
                t_entry: float = 0.5,
                ramp_duration: float = 0.5,
                direction: str = 'left') -> Callable[[float], float]:
        """
        FSAE skidpad steer profile (single-circle, steady-state hold).

        Uses the bicycle (Ackermann) approximation for required steer:
            delta = wheelbase / radius
        Ramps in over `ramp_duration` starting at `t_entry`.
        """
        sign = +1.0 if direction.lower().startswith('l') else -1.0
        delta_ss = sign * wheelbase_m / max(radius_m, 1e-6)
        return SteeringProfile.ramp(t_entry, t_entry + ramp_duration, delta_ss)

    @staticmethod
    def skidpad_full(radius_m: float,
                     wheelbase_m: float,
                     speed_ms: float,
                     t_entry: float = 1.0,
                     ramp_duration: float = 0.5,
                     transition_duration: float = 0.0,
                     first_direction: str = 'right',
                     n_laps_per_side: int = 2
                     ) -> tuple[Callable[[float], float], float]:
        """
        FULL FSAE skidpad: entry → N laps right → transition → N laps left → exit.

        Per FSAE rules: two tangent circles, 15.25 m inner cone diameter,
        3 m track width → path-centreline radius 9.125 m.  Laps 1 & 2 run
        the RIGHT circle, laps 3 & 4 run the LEFT circle.  Timed laps are
        the 2nd and 4th — lap 1 and 3 are "warm-up" laps per side.

        The steering input is a piecewise profile using the bicycle-model
        steady-state angle δ = wheelbase / radius.  Sign flips at the
        crossover; the transition is a linear ramp of ``transition_duration``
        seconds passing through zero steer at the crossing point.

        Parameters
        ----------
        radius_m : float
            Path centreline radius (9.125 m for standard FSAE skidpad).
        wheelbase_m : float
            Vehicle wheelbase (m) — for the bicycle-model steer angle.
        speed_ms : float
            Target forward speed (m/s) — sets the lap time
            T_lap = 2π·R / v.
        t_entry : float
            Straight-line entry time before the first ramp starts.
        ramp_duration : float
            Duration of the entry and exit ramps (steer → δ_ss and
            δ_ss → 0).
        transition_duration : float
            Duration of the mid-skidpad sign flip (δ_ss → −δ_ss).
            Default 0.0 — an instantaneous sign flip.  The solver's
            ``steering_tau_s`` lag smooths the step into a physical
            turn-around.  A non-zero value here lets the car drift
            forward during the flip, which breaks the tangent-circles
            geometry (the two circle centres would no longer sit on a
            line perpendicular to the entry straight), so leave it at 0
            for a true FSAE trajectory.
        first_direction : str
            'right' (FSAE regulation) or 'left'.
        n_laps_per_side : int
            2 for regulation.

        Returns
        -------
        (profile, total_duration_s) : tuple
            ``profile(t)`` returns front-wheel steer angle in rad.
            ``total_duration_s`` is the minimum sim time to capture the
            whole event (entry + all laps + transition + exit).
        """
        # FSAE regulation: RIGHT circle first.  ISO: +steer = left turn,
        # so 'right' first → start with NEGATIVE steer.
        sign = -1.0 if first_direction.lower().startswith('r') else +1.0
        delta_ss = sign * wheelbase_m / max(radius_m, 1e-6)

        T_lap = 2 * np.pi * radius_m / max(speed_ms, 0.1)
        t_enter_end = t_entry + ramp_duration
        t_side1_end = t_enter_end + n_laps_per_side * T_lap
        t_trans_end = t_side1_end + transition_duration
        t_side2_end = t_trans_end + n_laps_per_side * T_lap
        t_exit_end  = t_side2_end + ramp_duration

        def _p(t):
            if t < t_entry:
                return 0.0
            if t < t_enter_end:
                # Entry ramp: 0 → δ_ss
                frac = (t - t_entry) / max(ramp_duration, 1e-6)
                return delta_ss * frac
            if t < t_side1_end:
                # First side (right for regulation) — hold δ_ss
                return delta_ss
            if t < t_trans_end:
                # Transition: +δ_ss → −δ_ss through zero at midpoint
                frac = (t - t_side1_end) / max(transition_duration, 1e-6)
                return delta_ss * (1.0 - 2.0 * frac)
            if t < t_side2_end:
                # Second side (left) — hold −δ_ss
                return -delta_ss
            if t < t_exit_end:
                # Exit ramp: −δ_ss → 0
                frac = (t - t_side2_end) / max(ramp_duration, 1e-6)
                return -delta_ss * (1.0 - frac)
            return 0.0

        return _p, t_exit_end


# ─────────────────────────────────────────────────────────────────────────────
#  Closed-loop path followers
# ─────────────────────────────────────────────────────────────────────────────

class SkidpadPathFollower:
    """Closed-loop **Stanley** path-tracking controller for the FSAE skidpad.

    Replaces the original pure-pursuit law that drove the rear/CG toward
    a forward look-ahead point on the path.  Pure pursuit has two known
    failure modes that show up badly on the FSAE skidpad:

    1. **No heading-error damping.**  Pure pursuit only sees the angle to
       a forward target — if the car's heading is wrong but the target
       happens to be roughly ahead, the controller commands almost zero
       correction.  At the figure-8 crossover the car arrives heading
       ≈ +X but with non-zero yaw rate from the previous circle; pure
       pursuit can't damp that out.

    2. **Steady-state offset on tight curves.**  On a circle of radius R
       with a finite look-ahead L_d, pure pursuit settles to a path
       roughly L_d² / (8R) outside the ideal — for FSAE (R=9.125 m,
       L_d≈3 m), that's ~12 cm of bias even with perfect tires, and it
       compounds when the tires saturate.

    The Stanley controller [Hoffmann et al. 2007, Stanford DARPA] tracks
    the **front axle** to the path and combines two terms:

        δ = ψ_e  +  atan2(k · e, v_min)  +  atan(L · κ_path)
            ─┬─    ────────┬────────       ───────┬────────
            heading       cross-track            curvature
            error         feedback              feed-forward

    where:
        ψ_e   = path tangent heading − vehicle heading (wrapped to ±π)
        e     = signed lateral offset of front axle from path tangent
                (positive when car is to the LEFT of the path)
        v_min = max(v_x, ~1 m/s)  prevents blow-up at low speed
        κ     = path curvature at the closest point (1/R, signed)
        L     = wheelbase

    The cross-track term has 1/v scaling — high-speed corrections are
    naturally gentle, low-speed corrections sharp.  The feed-forward
    term ``atan(L·κ)`` is the kinematic Ackermann steer for the path's
    own curvature; without it Stanley still has a steady-state offset
    on tight circles.  With it, the FSAE skidpad tracks to single-cm
    error within grip.

    The figure-8 polyline is unchanged — entry → 2 × first circle → 2 ×
    opposite circle → exit — and the monotonic ``_last_idx`` state is
    still used to prevent the closest-point search from hopping back to
    the wrong lap at the crossover.

    Parameters
    ----------
    radius_m : float
        Path centreline radius (9.125 m for standard FSAE skidpad).
    wheelbase_m : float
        Vehicle wheelbase (m).  Used both for the front-axle reference
        point and for the curvature feed-forward.
    speed_ms : float
        Target forward speed (m/s).  Used for total-time estimate and as
        a fall-back when the live ``v_x`` is near zero.
    first_direction : str
        'right' (FSAE regulation) or 'left'.
    n_laps_per_side : int
        2 for regulation.
    t_entry_s : float
        Time spent on the entry straight before reaching (X_t, 0).
    exit_straight_m : float
        Length of the exit straight after the figure-8 (for plotting).
    lookahead_m : Optional[float]
        **Unused** — kept in the signature for backward compatibility
        with callers built around the old pure-pursuit version.  Stanley
        does not have a look-ahead distance.
    path_resolution_m : float
        Sampling density of the polyline representation (m).
    max_steer_rad / max_steer_deg : Optional[float]
        Saturation limit on the controller output.
    k_cross_track : float
        Stanley cross-track gain.  Larger = more aggressive lateral
        correction, smaller = gentler but more lateral offset.  For FSAE
        skidpad with L=1.53 m, v=12 m/s, e≈0.1 m: k=2 gives ~1°
        correction, k=4 gives ~2°.  Default 2.5 is a robust middle
        ground; raise to 4-5 for sharper tracking on high-grip surfaces.
    soft_v_min : float
        Lower bound on the v in the cross-track denominator (m/s).
        Stops the term blowing up at sim start when v_x ≈ 0.
    use_feedforward : bool
        Whether to add ``atan(L · κ_path)`` curvature feed-forward.
        On.  Off only for back-to-back comparison vs raw Stanley.
    """

    def __init__(self,
                 radius_m: float,
                 wheelbase_m: float,
                 speed_ms: float,
                 first_direction: str = 'right',
                 n_laps_per_side: int = 2,
                 t_entry_s: float = 1.0,
                 exit_straight_m: float = 10.0,
                 lookahead_m: Optional[float] = None,
                 path_resolution_m: float = 0.05,
                 max_steer_rad: Optional[float] = None,
                 max_steer_deg: Optional[float] = None,
                 k_cross_track: float = 2.5,
                 soft_v_min: float = 1.5,
                 use_feedforward: bool = True,
                 k_yaw_damp: float = 0.0):
        """See class docstring for parameter notes."""
        self.R = float(radius_m)
        self.L = float(wheelbase_m)
        self.v = max(float(speed_ms), 0.1)
        self.sign_first = -1.0 if first_direction.lower().startswith('r') else +1.0
        self.n_laps = int(n_laps_per_side)
        # Tangent point: where the car first touches both circles.
        self.X_t = max(t_entry_s * self.v, 0.5)
        # Kept for back-compat — Stanley doesn't use it.
        self.lookahead = float(lookahead_m) if lookahead_m is not None else 0.0
        if max_steer_rad is not None:
            self.max_steer = float(max_steer_rad)
        elif max_steer_deg is not None:
            self.max_steer = float(np.radians(max_steer_deg))
        else:
            # Generic fallback — the main-window dispatch should always
            # supply a rack-derived limit in practice.
            self.max_steer = np.radians(30.0)
        self.k_cross_track = float(k_cross_track)
        self.soft_v_min = float(soft_v_min)
        self.use_feedforward = bool(use_feedforward)
        # Optional explicit yaw-rate damping ( -k_yaw · r ).  Stanley's
        # heading-error term already provides proportional yaw control;
        # this term adds derivative-style suppression of residual yaw
        # oscillation when the tyres are saturating.  Default 0.0 keeps
        # behaviour identical to textbook Stanley until a user tunes it.
        self.k_yaw_damp = float(k_yaw_damp)

        # ── Build the ideal path as a polyline ──────────────────────────
        # Curvature κ is also stored per-point so the feed-forward term
        # can read it without a runtime derivative of psi.
        self._xs, self._ys, self._psi, self._kappa = self._build_path(path_resolution_m)
        self._path_len = len(self._xs)
        # Monotonic arc-length index state — never walk backwards.
        self._last_idx = 0

        # Estimated total duration for the caller's convenience
        T_lap = 2 * np.pi * self.R / self.v
        self.total_time_s = (
            t_entry_s                       # entry straight
            + 2 * self.n_laps * T_lap       # laps on both sides
            + exit_straight_m / self.v      # exit straight
        )

    # ── Path construction ───────────────────────────────────────────────
    def _build_path(self, res_m: float):
        """Build dense (X, Y, psi, kappa) polyline for the whole skidpad.

        Curvature κ is signed: positive when the path turns LEFT (CCW),
        negative for RIGHT (CW).  Used by the Stanley feed-forward term.

        Returns arrays sampled at ~res_m spacing along arc length.
        """
        Xs, Ys, Ps, Ks = [], [], [], []

        # 1) Entry straight: (0,0) → (X_t, 0), heading +X — κ = 0
        n = max(int(self.X_t / res_m), 2)
        xs = np.linspace(0.0, self.X_t, n, endpoint=False)
        Xs.append(xs); Ys.append(np.zeros_like(xs))
        Ps.append(np.zeros_like(xs))
        Ks.append(np.zeros_like(xs))

        # 2) Two circles.  Curvature on each is ±1/R signed by travel
        #    direction (+ for CCW, − for CW), so the feed-forward term
        #    atan(L·κ) gives the kinematic Ackermann steer with the
        #    correct sign automatically.
        for side in (+1, -1):  # side 1 = first direction, side 2 = opposite
            sign = self.sign_first * side
            cy = sign * self.R  # centre y
            if sign > 0:
                theta0 = -np.pi / 2
            else:
                theta0 = +np.pi / 2
            dtheta_sign = +1.0 if sign > 0 else -1.0
            total_sweep = dtheta_sign * 2 * np.pi * self.n_laps
            n_arc = max(int(abs(total_sweep) * self.R / res_m), 16)
            thetas = theta0 + np.linspace(0.0, total_sweep,
                                          n_arc, endpoint=False)
            xs_c = self.X_t + self.R * np.cos(thetas)
            ys_c = cy + self.R * np.sin(thetas)
            psi_c = np.arctan2(dtheta_sign * np.cos(thetas),
                               dtheta_sign * (-np.sin(thetas)))
            kappa_c = np.full_like(thetas, dtheta_sign / self.R)
            Xs.append(xs_c); Ys.append(ys_c); Ps.append(psi_c); Ks.append(kappa_c)

        # 3) Exit straight from (X_t, 0) heading +X — κ = 0
        exit_n = max(int(10.0 / res_m), 2)
        xs_e = self.X_t + np.linspace(0.0, 10.0, exit_n, endpoint=True)
        Xs.append(xs_e); Ys.append(np.zeros_like(xs_e))
        Ps.append(np.zeros_like(xs_e))
        Ks.append(np.zeros_like(xs_e))

        return (np.concatenate(Xs), np.concatenate(Ys),
                np.concatenate(Ps), np.concatenate(Ks))

    # ── Stanley steering law ────────────────────────────────────────────
    def __call__(self, t: float, state) -> float:
        # State unpack:  [vx, vy, r, phi, phi_dot, X, Y, psi, steer_actual]
        v_x = float(state[0])
        r   = float(state[2])
        X, Y, psi = float(state[5]), float(state[6]), float(state[7])

        # Stanley references the FRONT axle, not the CG.  Promote (X,Y)
        # by one wheelbase along the body x-axis.
        cos_p = np.cos(psi); sin_p = np.sin(psi)
        X_f = X + self.L * cos_p
        Y_f = Y + self.L * sin_p

        # ── Closest path index (monotonic) ──────────────────────────────
        # The figure-8 path crosses itself at (X_t, 0); a global argmin
        # would happily snap onto the wrong lap.  Restrict the search to
        # a forward window around the last index so progress is monotone.
        search_start = max(self._last_idx - 50, 0)
        search_end   = min(self._last_idx + 500, self._path_len)
        dxs = self._xs[search_start:search_end] - X_f
        dys = self._ys[search_start:search_end] - Y_f
        d2  = dxs * dxs + dys * dys
        if len(d2) == 0:
            return 0.0
        local_idx = int(np.argmin(d2))
        closest_idx = search_start + local_idx
        self._last_idx = closest_idx

        psi_p = float(self._psi[closest_idx])
        kappa = float(self._kappa[closest_idx])

        # ── Heading error  ψ_e = ψ_path − ψ_car  (wrapped to ±π) ────────
        psi_e = psi_p - psi
        psi_e = float(np.arctan2(np.sin(psi_e), np.cos(psi_e)))

        # ── Signed lateral (cross-track) error ──────────────────────────
        # Convention: e_lat > 0  ⇔  path is to the LEFT of the car
        #                      ⇔  car needs a LEFT (positive) steer.
        # Project (P − F) onto the path-left normal (−sin ψ_p, cos ψ_p).
        ex = self._xs[closest_idx] - X_f
        ey = self._ys[closest_idx] - Y_f
        e_lat = -np.sin(psi_p) * ex + np.cos(psi_p) * ey

        # ── Stanley command ─────────────────────────────────────────────
        #   δ = ψ_e + atan2(k·e, v)  +  atan(L·κ)  −  k_yaw·r
        v_eff = max(v_x, self.soft_v_min)
        delta = psi_e + float(np.arctan2(self.k_cross_track * e_lat, v_eff))
        if self.use_feedforward:
            delta += float(np.arctan(self.L * kappa))
        if self.k_yaw_damp > 0.0:
            # Yaw-rate damping: r > 0 ⇒ over-rotating left ⇒ steer right.
            delta -= self.k_yaw_damp * r

        # Saturate to physical rack travel limit.
        if delta >  self.max_steer:
            delta =  self.max_steer
        elif delta < -self.max_steer:
            delta = -self.max_steer
        return float(delta)

    # ── Convenience: ideal path for plotting ───────────────────────────
    def ideal_path(self):
        """Return (X, Y) arrays of the ideal skidpad path for overlay."""
        return self._xs.copy(), self._ys.copy()

    def circle_centres(self):
        """Return the two circle centres (right_centre, left_centre) as
        (X, Y) tuples.  For plotting the ideal FSAE cone layout."""
        cy1 = self.sign_first * self.R
        cy2 = -self.sign_first * self.R
        return (self.X_t, cy1), (self.X_t, cy2)


@dataclass
class TransientInputs:
    """Driver inputs + simulation settings."""

    v_x_target_ms: float = 10.0
    steering: Callable[[float], float] = field(
        default_factory=lambda: SteeringProfile.constant(0.0))
    # Closed-loop (state-dependent) steering controller.  Signature:
    #     f(t_s, state_vec) -> front-wheel steer angle (rad)
    # where state_vec = [vx, vy, r, phi, phi_dot, X, Y, psi, steer_actual].
    # When set, this OVERRIDES `steering`.  Used by the FSAE skidpad path
    # follower — the open-loop steering profile cannot correct for yaw-
    # reversal drift at the figure-8 crossover.
    steering_controller: Optional[Callable[[float, "np.ndarray"], float]] = None

    longitudinal_profile: Optional[Callable[[float], float]] = None
    # If mode='open_loop', this callable returns total Fx (N) at the driven axle.
    # If None or mode='speed_hold', vx is held to v_x_target by a PI controller.

    duration_s: float = 5.0
    dt_s: float = 0.002

    # Initial state
    initial_v_x_ms: Optional[float] = None   # default = v_x_target
    initial_v_y_ms: float = 0.0
    initial_yaw_rate: float = 0.0
    initial_yaw_rad: float = 0.0
    initial_roll_rad: float = 0.0
    initial_roll_rate: float = 0.0
    initial_X_m: float = 0.0
    initial_Y_m: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransientResult:
    """Time history + derived metrics from a transient simulation."""

    t: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # 8-state time histories
    v_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yaw_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(0))        # phi, rad
    roll_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))   # phi_dot, rad/s
    X: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Derived signals
    ay: np.ndarray = field(default_factory=lambda: np.zeros(0))    # lateral accel (m/s²)
    ax: np.ndarray = field(default_factory=lambda: np.zeros(0))    # longitudinal accel
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))  # sideslip at CG (rad)
    steer: np.ndarray = field(default_factory=lambda: np.zeros(0)) # commanded road-wheel steer (rad)
    steer_actual: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Driver-side steering chain (populated when a SteeringGeometry is
    # supplied to the solver; otherwise zero-filled).
    #
    # steer_wheel_deg : what the driver is rotating the hand-wheel (deg).
    # rack_travel_mm  : the rack tube translation from centre (mm).
    # Both are derived from ``steer_actual`` via the steering geometry.
    steer_wheel_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rack_travel_mm: np.ndarray  = field(default_factory=lambda: np.zeros(0))

    # Per-corner time histories (each np.ndarray of shape (n_steps,))
    Fz: dict = field(default_factory=dict)          # vertical load (N)
    Fy: dict = field(default_factory=dict)          # lateral force (N), body frame
    Fx: dict = field(default_factory=dict)          # long. force (N), body frame
    slip_angle: dict = field(default_factory=dict)  # slip angle (rad)
    camber: dict = field(default_factory=dict)      # dynamic camber (deg)

    # Scalar metrics
    yaw_rate_rise_time_s: float = 0.0        # 10-90% of steady-state r
    yaw_rate_settling_time_s: float = 0.0    # ±5% of steady-state r
    yaw_rate_overshoot_pct: float = 0.0
    peak_lateral_g: float = 0.0
    steady_state_lateral_g: float = 0.0
    peak_roll_deg: float = 0.0
    steady_state_roll_deg: float = 0.0
    peak_understeer_deg: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Solver
# ─────────────────────────────────────────────────────────────────────────────

_CORNERS = ('FL', 'FR', 'RL', 'RR')


class TransientSolver:
    """
    Time-domain 7-state vehicle dynamics integrator.

    States
    ------
        s = [vx, vy, r, phi, phi_dot, X, Y, psi, steer_actual]
    (steer_actual is an internal 1st-order lag on the commanded steer.)

    Equations
    ---------
        m · (v_x_dot - v_y·r)           = ΣFx_body
        m · (v_y_dot + v_x·r)           = ΣFy_body
        Izz · r_dot                      = ΣMz_body
        Ixx · phi_ddot + c·phi_dot + K_total·phi
                                         = M_s · ay_sprung · h_arm
        X_dot = vx·cos(psi) - vy·sin(psi)
        Y_dot = vx·sin(psi) + vy·cos(psi)
        psi_dot = r
        steer_dot = (steer_cmd - steer_actual) / tau

    Notes
    -----
        Fz per corner uses instantaneous static + geometric LT + unsprung LT
        + elastic LT from current roll angle.  This correctly captures the
        dynamic lag of load transfer through the springs — elastic LT lags
        ay by the roll dynamics, while geometric/unsprung LT respond instantly.
    """

    def __init__(self, vehicle: VehicleParams,
                 tire_model,
                 corner_solvers: Optional[dict] = None,
                 params: Optional[TransientParams] = None,
                 steering_geometry: Optional[SteeringGeometry] = None,
                 shock_stroke_mm: Optional[float] = None):
        """
        Parameters
        ----------
        vehicle : VehicleParams
        tire_model : TireModel or LinearTireModel (must have .Fy, .peak_mu)
        corner_solvers : dict {label: SuspensionConstraints} or None
            If provided, builds per-corner lookup tables of camber(travel)
            and RC height(travel).  If None, uses constant design-position
            values (fast but loses kinematic coupling).
        params : TransientParams
        steering_geometry : SteeringGeometry or None
            If provided, saturates commanded road-wheel angles to the
            physical rack limit AND lets the solver record the driver's
            steering-wheel angle + rack position per step.  When None,
            the solver falls back to ``vehicle.max_steer_angle_deg`` for
            saturation and reports steering-wheel angle as 0.
        shock_stroke_mm : float or None
            Full damper stroke in mm, used to auto-size the kinematic
            lookup-table travel range if TransientParams.kin_table_travel_mm
            is left at None.  The LUT spans ±stroke × max(MR_front, MR_rear)
            so the entire working range is covered without clamp-tails.
            Defaults to 50 mm if not supplied.
        """
        if tire_model is None:
            raise ValueError('TransientSolver requires a tire_model')
        self._veh = vehicle
        self._tire = tire_model
        self._params = params if params is not None else TransientParams()
        self._steering_geom = steering_geometry
        # Resolve speed-hold PI gains: per-kg values win when the raw
        # gains are None (encouraged — they scale with vehicle mass).
        p = self._params
        m = vehicle.total_mass_kg
        self._kp = (p.speed_hold_kp
                    if p.speed_hold_kp is not None
                    else vehicle.speed_hold_kp_per_kg * m)
        self._ki = (p.speed_hold_ki
                    if p.speed_hold_ki is not None
                    else vehicle.speed_hold_ki_per_kg * m)
        # Physical steering saturation — priority:
        #   1. explicit params.max_steer_rad
        #   2. steering_geometry.max_road_wheel_rad (rack-derived)
        #   3. vehicle.max_steer_angle_deg (geometry-hard limit)
        if p.max_steer_rad is not None:
            self._max_steer_rad = float(p.max_steer_rad)
        elif steering_geometry is not None:
            self._max_steer_rad = float(steering_geometry.max_road_wheel_rad)
        else:
            self._max_steer_rad = float(np.radians(vehicle.max_steer_angle_deg))
        # Resolve the kinematic LUT travel range
        if p.kin_table_travel_mm is not None:
            self._kin_lo_mm, self._kin_hi_mm = p.kin_table_travel_mm
        else:
            mr_max = max(vehicle.motion_ratio_front, vehicle.motion_ratio_rear, 0.5)
            half = (shock_stroke_mm or 50.0) / max(mr_max, 0.1) * 0.6
            # 0.6 factor: typical suspension doesn't reach full shock
            # stroke at roll alone, so keep LUT cost down.
            self._kin_lo_mm = -float(half)
            self._kin_hi_mm = +float(half)
        self._build_kinematic_lut(corner_solvers)

    # ── Kinematic lookup tables ──────────────────────────────────────────

    def _build_kinematic_lut(self, corner_solvers: Optional[dict]):
        """Precompute camber(travel) and RC(travel) per corner."""
        lo_mm, hi_mm = self._kin_lo_mm, self._kin_hi_mm
        n = self._params.kin_table_n
        self._travel_grid_m = np.linspace(lo_mm, hi_mm, n) / 1000.0
        self._camber_lut = {c: np.zeros(n) for c in _CORNERS}
        self._rc_lut = {c: np.zeros(n) for c in _CORNERS}

        if corner_solvers is None:
            return  # leave zeros → constant kinematics fallback

        for lbl in _CORNERS:
            solver = corner_solvers.get(lbl)
            if solver is None:
                continue
            side = 'left' if lbl.endswith('L') else 'right'
            for i, t_m in enumerate(self._travel_grid_m):
                try:
                    state = solver.solve(float(t_m))
                    m = KinematicMetrics(state, side)
                    self._camber_lut[lbl][i] = m.camber
                    self._rc_lut[lbl][i] = m.roll_center_height
                except Exception:
                    self._camber_lut[lbl][i] = 0.0
                    self._rc_lut[lbl][i] = 0.05

    def _camber(self, label: str, travel_m: float) -> float:
        return float(np.interp(travel_m,
                               self._travel_grid_m,
                               self._camber_lut[label]))

    def _rc_height(self, label: str, travel_m: float) -> float:
        rc = self._rc_lut[label]
        if np.all(rc == 0):
            return 0.05  # fallback if no solvers given
        return float(np.interp(travel_m, self._travel_grid_m, rc))

    # ── Per-corner Fz and travel from roll ───────────────────────────────

    def _per_corner_loads(self, v: VehicleParams,
                          phi: float, ay: float, ax: float) -> dict:
        """
        Per-corner Fz (N) and travel (m) at a given roll / ay / ax state.

        Breakdown:
            Fz = Fz_static
               + pitch LT (long.)
               + geometric LT (through current RC)
               + unsprung LT
               + elastic LT (via roll angle)

        Travel from roll: roll_travel = sin(phi) · track / 2
            - In ISO convention positive phi = body rolls to the right.
            - So in a LEFT turn: positive ay → positive phi → right side
              compresses (outside).
        """
        W = v.total_mass_kg * G
        Fz_static_f = W * v.front_weight_fraction / 2
        Fz_static_r = W * v.rear_weight_fraction / 2

        # Pitch LT (positive ax → load to rear)
        dFz_pitch = v.total_mass_kg * ax * v.cg_height_m / v.wheelbase_m
        Fz = {
            'FL': Fz_static_f - dFz_pitch / 2,
            'FR': Fz_static_f - dFz_pitch / 2,
            'RL': Fz_static_r + dFz_pitch / 2,
            'RR': Fz_static_r + dFz_pitch / 2,
        }

        # Per-corner travel from roll.  ISO: +phi = body tilts right →
        # right-side corners compress (positive bump travel).
        travel_f = np.sin(phi) * v.front_track_m / 2
        travel_r = np.sin(phi) * v.rear_track_m / 2
        travel = {
            'FL': -travel_f,
            'FR': +travel_f,
            'RL': -travel_r,
            'RR': +travel_r,
        }

        # Roll-centre heights at current travel
        rc_f = (self._rc_height('FL', travel['FL'])
                + self._rc_height('FR', travel['FR'])) / 2
        rc_r = (self._rc_height('RL', travel['RL'])
                + self._rc_height('RR', travel['RR'])) / 2

        # --- Geometric LT (through RC; one-side delta) ------------------
        geo_f = v.sprung_mass_kg * v.front_weight_fraction * ay * rc_f / v.front_track_m
        geo_r = v.sprung_mass_kg * v.rear_weight_fraction * ay * rc_r / v.rear_track_m

        # --- Unsprung LT -------------------------------------------------
        h_us = v.unsprung_cg_height_m
        uns_f = (v.unsprung_mass_front_kg / 2) * ay * h_us / v.front_track_m
        uns_r = (v.unsprung_mass_rear_kg / 2)  * ay * h_us / v.rear_track_m

        # --- Elastic LT (via roll angle, NOT ay!) ------------------------
        # This is the dynamics — elastic LT lags ay with roll dynamics.
        K_f = v.roll_stiffness_front_Npm_rad
        K_r = v.roll_stiffness_rear_Npm_rad
        # one-side delta = K_axle · phi / track
        el_f = K_f * phi / v.front_track_m
        el_r = K_r * phi / v.rear_track_m

        # Apply: positive ay = left turn → positive LT puts load on RIGHT
        # (outside) wheels.  ISO sign of LT delta: add to right, subtract from left.
        Fz['FL'] -= (geo_f + uns_f + el_f)
        Fz['FR'] += (geo_f + uns_f + el_f)
        Fz['RL'] -= (geo_r + uns_r + el_r)
        Fz['RR'] += (geo_r + uns_r + el_r)

        # Clamp so a wheel never has negative normal load
        for k in Fz:
            Fz[k] = max(Fz[k], 0.0)

        return {'Fz': Fz, 'travel': travel, 'rc_f': rc_f, 'rc_r': rc_r}

    # ── Per-corner slip angles (Ackermann + body motion) ─────────────────

    def _slip_angles(self, v: VehicleParams,
                     vx: float, vy: float, r: float,
                     steer_cmd: float) -> dict:
        """
        Slip angle per wheel (rad).  ISO: positive slip → tire produces
        positive (leftward) Fy.

        Body-frame velocity at wheel i with position (a_i, b_i from CG):
            vx_i = vx - r · b_i
            vy_i = vy + r · a_i
        Slip angle:  α_i = δ_i - atan2(vy_i, vx_i)
        """
        a = v.cg_to_front_axle_m          # +forward from CG
        b = -v.cg_to_rear_axle_m          # rear axle is behind CG
        tf2 = v.front_track_m / 2
        tr2 = v.rear_track_m / 2

        # Position of each wheel relative to CG: (a, b_lat) where b_lat > 0 = left
        pos = {
            'FL': ( a, +tf2),
            'FR': ( a, -tf2),
            'RL': ( b, +tr2),
            'RR': ( b, -tr2),
        }

        # Per-wheel steer (Ackermann): inner wheel steers more
        # Positive steer_cmd = left turn → inner is LEFT (FL)
        ackermann_frac = self._params.ackermann_pct / 100.0
        if abs(steer_cmd) < 1e-9 or ackermann_frac == 0.0:
            d_FL = d_FR = steer_cmd
        else:
            L = v.wheelbase_m
            # Perfect-Ackermann inner/outer for a given mean delta:
            # R = L / tan(delta) is the radius at CG track centreline
            R = L / np.tan(steer_cmd)
            # Inner wheel: R - t_f/2;   Outer wheel: R + t_f/2
            # (sign of steer_cmd carries which side is inner)
            if steer_cmd > 0:  # left turn → FL is inner
                d_in = np.arctan2(L, (R - tf2))
                d_out = np.arctan2(L, (R + tf2))
                d_FL_ack = d_in
                d_FR_ack = d_out
            else:              # right turn → FR is inner
                d_in = np.arctan2(L, (-R - tf2))   # R is negative
                d_out = np.arctan2(L, (-R + tf2))
                d_FL_ack = -d_out   # mirrored
                d_FR_ack = -d_in
            # Blend parallel ↔ Ackermann
            d_FL = (1 - ackermann_frac) * steer_cmd + ackermann_frac * d_FL_ack
            d_FR = (1 - ackermann_frac) * steer_cmd + ackermann_frac * d_FR_ack

        steer_per = {'FL': d_FL, 'FR': d_FR, 'RL': 0.0, 'RR': 0.0}

        alpha = {}
        for lbl in _CORNERS:
            ax_pos, b_pos = pos[lbl]
            vx_i = vx - r * b_pos
            vy_i = vy + r * ax_pos
            # Use a small floor on vx_i to avoid the atan2 jumping at v=0
            vx_i = max(vx_i, 0.1)
            alpha[lbl] = steer_per[lbl] - np.arctan2(vy_i, vx_i)

        return alpha, steer_per

    # ── Dynamics derivative ──────────────────────────────────────────────

    def _deriv(self, state: np.ndarray,
               steer_cmd: float,
               Fx_total: float) -> tuple:
        """
        Compute the state derivative at a given (state, inputs).

        state = [vx, vy, r, phi, phi_dot, X, Y, psi, steer_actual]

        Returns (dstate, diagnostics) where diagnostics holds the per-corner
        forces used so the caller can record them at the current step.
        """
        v = self._veh
        p = self._params

        vx, vy, r, phi, phi_dot, X, Y, psi, steer_actual = state

        # --- Apply 1st-order lag on steer command -----------------------
        # τ → 0 makes this derivative singular (and the ODE infinitely
        # stiff for any explicit integrator).  Clamp to 1 ms so a user
        # who types 0 in the GUI gets a fast-but-finite actuator instead
        # of a sim that explodes inside the first 0.2 s.  See the
        # ``steering_tau_s`` field doc for guidance on physical values.
        tau_eff = max(p.steering_tau_s, 1e-3)
        steer_dot = (steer_cmd - steer_actual) / tau_eff

        # --- Slip angles ------------------------------------------------
        alpha, steer_per = self._slip_angles(v, vx, vy, r, steer_actual)

        # --- Per-corner loads + tire forces (vectorized over 4 wheels) ─
        # Two-pass fixed-point: ay drives LT → LT drives Fz → Fz drives
        # Fy → Fy drives ay.  Two passes converge for well-posed cases.
        # Tire calls are batched as length-4 arrays to save scipy
        # RegularGridInterpolator per-call overhead (≈4× speedup).
        ay_est = 0.0
        # alpha as a 4-vector in FL,FR,RL,RR order
        alpha_arr = np.array([alpha['FL'], alpha['FR'], alpha['RL'], alpha['RR']])
        alpha_deg = np.degrees(alpha_arr)
        Fz = None
        travel = None
        Fy = {}
        for _pass in range(2):
            loads = self._per_corner_loads(v, phi, ay_est, Fx_total / v.total_mass_kg)
            Fz = loads['Fz']
            travel = loads['travel']
            fz_arr  = np.array([max(Fz[c], 0.0) for c in _CORNERS])
            cam_arr = np.array([self._camber(c, travel[c]) for c in _CORNERS])

            # One batched tire call for all 4 wheels
            fy_tire = np.asarray(self._tire.Fy(alpha_deg, fz_arr, cam_arr),
                                 dtype=float)
            if fy_tire.ndim == 0:
                fy_tire = np.array([float(fy_tire)] * 4)
            # TTC convention flip: positive alpha → negative Fy in raw data.
            # ISO expects positive alpha → positive Fy.  Flip where signs agree.
            sign_flip = (np.sign(fy_tire) == -np.sign(alpha_arr)) & (np.abs(alpha_arr) > 1e-6)
            fy_tire = np.where(sign_flip, -fy_tire, fy_tire)
            Fy = {c: float(fy_tire[i]) for i, c in enumerate(_CORNERS)}

            # Per-wheel Fx distribution
            # For MVP: Fx_total applied to driven axle, 50/50 left/right.
            dt_mode = v.drivetrain.upper()
            if Fx_total >= 0:   # drive
                if dt_mode == 'FWD':
                    driven = ('FL', 'FR')
                elif dt_mode == 'AWD':
                    driven = _CORNERS
                else:               # RWD default
                    driven = ('RL', 'RR')
            else:               # brake — use brake bias
                driven = _CORNERS
            Fx_per = {k: 0.0 for k in _CORNERS}
            if Fx_total >= 0:
                per = Fx_total / len(driven)
                for lbl in driven:
                    Fx_per[lbl] = per
            else:
                bb = v.front_brake_bias
                Fx_per['FL'] = Fx_total * bb / 2
                Fx_per['FR'] = Fx_total * bb / 2
                Fx_per['RL'] = Fx_total * (1 - bb) / 2
                Fx_per['RR'] = Fx_total * (1 - bb) / 2

            # Sum forces/moments in body frame.  Each wheel has Fx_wheel
            # along the wheel's rolling direction (tire x) and Fy_wheel
            # lateral.  Front wheels are steered by steer_per.
            Fx_body = 0.0
            Fy_body = 0.0
            Mz_body = 0.0
            pos = {
                'FL': ( v.cg_to_front_axle_m, +v.front_track_m / 2),
                'FR': ( v.cg_to_front_axle_m, -v.front_track_m / 2),
                'RL': (-v.cg_to_rear_axle_m,  +v.rear_track_m / 2),
                'RR': (-v.cg_to_rear_axle_m,  -v.rear_track_m / 2),
            }
            for lbl in _CORNERS:
                d = steer_per[lbl]
                fx_w = Fx_per[lbl]
                fy_w = Fy[lbl]
                # Rotate wheel-frame forces into body frame
                fx_b = fx_w * np.cos(d) - fy_w * np.sin(d)
                fy_b = fx_w * np.sin(d) + fy_w * np.cos(d)
                ax_pos, b_pos = pos[lbl]
                Fx_body += fx_b
                Fy_body += fy_b
                Mz_body += ax_pos * fy_b - b_pos * fx_b

            # Translational dynamics in body frame (ISO)
            m = v.total_mass_kg
            vx_dot = Fx_body / m + vy * r
            vy_dot = Fy_body / m - vx * r
            ay_est = vy_dot + vx * r   # = Fy_body / m  (for next pass)

        # --- Roll dynamics ---------------------------------------------
        # I_xx · phi_ddot + c · phi_dot + K · phi = M_s · ay_sprung · h_arm
        # h_arm is the lever arm from the roll axis up to the SPRUNG-mass
        # CG (NOT the whole-vehicle CG — the unsprung mass sits at the
        # wheel-centre height and pulls h_cg down, so using h_cg here
        # under-predicts the roll moment by a few percent).
        h_roll_axis = 0.5 * (loads['rc_f'] + loads['rc_r'])
        h_arm = v.sprung_cg_height_m - h_roll_axis
        # In ISO: +ay (left turn) should produce +phi (roll to right).
        # The sprung roll moment is m_s · ay · h_arm (positive ay → positive moment).
        M_roll = v.sprung_mass_kg * ay_est * h_arm
        K_total = v.roll_stiffness_total_Npm_rad
        phi_ddot = (M_roll
                    - p.roll_damping_Nms_rad * phi_dot
                    - K_total * phi) / max(p.sprung_roll_inertia, 1e-3)

        # --- Yaw dynamics ----------------------------------------------
        r_dot = Mz_body / max(p.yaw_inertia, 1e-3)

        # --- Global position -------------------------------------------
        X_dot = vx * np.cos(psi) - vy * np.sin(psi)
        Y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r

        dstate = np.array([vx_dot, vy_dot, r_dot,
                           phi_dot, phi_ddot,
                           X_dot, Y_dot, psi_dot,
                           steer_dot])

        diag = {
            'ay': ay_est,
            'ax': vx_dot - vy * r,
            'Fz': Fz,
            'Fy': Fy,
            'Fx': Fx_per,
            'alpha': alpha,
            'steer_per': steer_per,
            'travel': travel,
        }
        return dstate, diag

    # ── Top-level integrator ─────────────────────────────────────────────

    def simulate(self, inp: TransientInputs) -> TransientResult:
        """Run the simulation forward and return TransientResult."""
        p = self._params
        v = self._veh
        dt = inp.dt_s
        n = int(np.ceil(inp.duration_s / dt)) + 1
        t_arr = np.arange(n) * dt

        # ── Initial state ───────────────────────────────────────────────
        vx0 = inp.initial_v_x_ms if inp.initial_v_x_ms is not None else inp.v_x_target_ms
        state0_no_steer = np.array([vx0,
                                    inp.initial_v_y_ms,
                                    inp.initial_yaw_rate,
                                    inp.initial_roll_rad,
                                    inp.initial_roll_rate,
                                    inp.initial_X_m,
                                    inp.initial_Y_m,
                                    inp.initial_yaw_rad,
                                    0.0])
        if inp.steering_controller is not None:
            initial_steer = float(inp.steering_controller(0.0, state0_no_steer))
        else:
            initial_steer = float(inp.steering(0.0))
        # Saturate initial road-wheel demand to the physical rack limit
        # before it becomes the integration state.
        lim = self._max_steer_rad
        initial_steer = float(np.clip(initial_steer, -lim, +lim))
        state = state0_no_steer.copy()
        state[8] = initial_steer

        # Storage
        res = TransientResult(t=t_arr)
        res.v_x = np.zeros(n); res.v_y = np.zeros(n)
        res.yaw_rate = np.zeros(n); res.yaw = np.zeros(n)
        res.roll = np.zeros(n); res.roll_rate = np.zeros(n)
        res.X = np.zeros(n); res.Y = np.zeros(n)
        res.ay = np.zeros(n); res.ax = np.zeros(n)
        res.beta = np.zeros(n)
        res.steer = np.zeros(n); res.steer_actual = np.zeros(n)
        res.steer_wheel_deg = np.zeros(n); res.rack_travel_mm = np.zeros(n)
        for c in _CORNERS:
            res.Fz[c] = np.zeros(n)
            res.Fy[c] = np.zeros(n)
            res.Fx[c] = np.zeros(n)
            res.slip_angle[c] = np.zeros(n)
            res.camber[c] = np.zeros(n)

        # Speed-hold integrator state
        speed_err_int = 0.0

        lim = self._max_steer_rad
        for i in range(n):
            t = t_arr[i]
            if inp.steering_controller is not None:
                steer_cmd = float(inp.steering_controller(t, state))
            else:
                steer_cmd = float(inp.steering(t))

            # --- Physical rack saturation -------------------------------
            # Clamp the commanded road-wheel angle to the rack stroke
            # before it enters the 1st-order steer lag.  If no rack info
            # was supplied this falls back to VehicleParams.max_steer_angle_deg.
            steer_cmd = float(np.clip(steer_cmd, -lim, +lim))

            # --- Longitudinal control (speed hold PI or open loop) ─────
            if p.longitudinal_control_mode == 'open_loop' and inp.longitudinal_profile:
                Fx_total = float(inp.longitudinal_profile(t))
            else:
                err = inp.v_x_target_ms - state[0]
                speed_err_int += err * dt
                Fx_total = self._kp * err + self._ki * speed_err_int
                # Clamp to a sane fraction of tire capacity
                Fx_max = p.fx_limit_g * v.total_mass_kg * G
                Fx_total = float(np.clip(Fx_total, -Fx_max, Fx_max))

            # --- Record current state ──────────────────────────────────
            res.v_x[i] = state[0]
            res.v_y[i] = state[1]
            res.yaw_rate[i] = state[2]
            res.roll[i] = state[3]
            res.roll_rate[i] = state[4]
            res.X[i] = state[5]
            res.Y[i] = state[6]
            res.yaw[i] = state[7]
            res.steer[i] = steer_cmd
            res.steer_actual[i] = state[8]
            # Driver-side chain: drive steer_actual (the road-wheel angle
            # after the 1st-order lag) back through the steering geometry
            # so we can plot what the driver is doing at the hand-wheel.
            if self._steering_geom is not None:
                rw = float(state[8])
                res.rack_travel_mm[i] = float(
                    self._steering_geom.rack_mm_from_road_wheel(rw))
                res.steer_wheel_deg[i] = float(
                    self._steering_geom.steering_wheel_from_road_wheel(rw))

            # --- RK4 step ──────────────────────────────────────────────
            k1, diag = self._deriv(state, steer_cmd, Fx_total)
            k2, _ = self._deriv(state + 0.5 * dt * k1, steer_cmd, Fx_total)
            k3, _ = self._deriv(state + 0.5 * dt * k2, steer_cmd, Fx_total)
            k4, _ = self._deriv(state + dt * k3, steer_cmd, Fx_total)

            # Record diagnostics from k1 (state-of-record)
            res.ay[i] = diag['ay']
            res.ax[i] = diag['ax']
            if state[0] > 0.1:
                res.beta[i] = np.arctan2(state[1], state[0])
            for c in _CORNERS:
                res.Fz[c][i] = diag['Fz'][c]
                res.Fy[c][i] = diag['Fy'][c]
                res.Fx[c][i] = diag['Fx'][c]
                res.slip_angle[c][i] = diag['alpha'][c]
                res.camber[c][i] = self._camber(c, diag['travel'][c])

            if i < n - 1:
                state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # ── Derived metrics ─────────────────────────────────────────────
        self._compute_metrics(res, inp)
        return res

    # ── Post-processing ──────────────────────────────────────────────────

    def _compute_metrics(self, res: TransientResult, inp: TransientInputs):
        """Extract rise time / overshoot / steady-state metrics."""
        if len(res.t) < 10:
            return
        r_abs = np.abs(res.yaw_rate)
        ay_abs = np.abs(res.ay)

        # Steady state: last 10% of simulation
        n = len(res.t)
        tail_lo = int(0.85 * n)
        r_ss = float(np.mean(r_abs[tail_lo:]))
        ay_ss = float(np.mean(ay_abs[tail_lo:])) / G   # lateral g
        roll_ss = float(np.degrees(np.mean(np.abs(res.roll[tail_lo:]))))
        res.steady_state_lateral_g = ay_ss
        res.steady_state_roll_deg = roll_ss

        # Peaks
        res.peak_lateral_g = float(np.max(ay_abs)) / G
        res.peak_roll_deg = float(np.max(np.abs(res.roll)) * 180 / np.pi)

        # Overshoot (% above steady state)
        if r_ss > 1e-3:
            r_peak = float(np.max(r_abs))
            res.yaw_rate_overshoot_pct = 100.0 * max(r_peak - r_ss, 0.0) / r_ss

            # Rise time: 10% → 90% of steady-state
            try:
                i_10 = int(np.argmax(r_abs >= 0.1 * r_ss))
                i_90 = int(np.argmax(r_abs >= 0.9 * r_ss))
                if i_90 > i_10:
                    res.yaw_rate_rise_time_s = res.t[i_90] - res.t[i_10]
            except Exception:
                pass

            # Settling time: first time after which |r - r_ss| stays < 5 %·r_ss
            try:
                band = 0.05 * r_ss
                within = np.abs(r_abs - r_ss) <= band
                # walk back from the end — find first index where it
                # has been continuously within band
                last_out = 0
                for i in range(len(within)):
                    if not within[i]:
                        last_out = i
                res.yaw_rate_settling_time_s = res.t[last_out]
            except Exception:
                pass

        # Peak understeer proxy: |front avg slip| - |rear avg slip|
        try:
            front_alpha = 0.5 * (np.abs(res.slip_angle['FL'])
                                 + np.abs(res.slip_angle['FR']))
            rear_alpha  = 0.5 * (np.abs(res.slip_angle['RL'])
                                 + np.abs(res.slip_angle['RR']))
            diff = front_alpha - rear_alpha
            res.peak_understeer_deg = float(np.max(diff) * 180 / np.pi)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test (run: python -m vahan.transient)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from .dynamics import VehicleParams
    from .tire_model import LinearTireModel

    veh = VehicleParams()
    tire = LinearTireModel(C_alpha_N_per_deg=250.0, mu=1.6)
    solver = TransientSolver(veh, tire)

    # Standard FSAE skidpad: 9.125m path radius (approx).
    # At 1.2 g: v = sqrt(R·g·ay) ≈ 10.4 m/s
    inp = TransientInputs(
        v_x_target_ms=10.4,
        steering=SteeringProfile.skidpad(radius_m=9.125,
                                          wheelbase_m=veh.wheelbase_m,
                                          t_entry=0.5,
                                          ramp_duration=0.5,
                                          direction='left'),
        duration_s=5.0,
        dt_s=0.002,
    )
    res = solver.simulate(inp)

    print(f'Steady-state lateral g   : {res.steady_state_lateral_g:.3f}')
    print(f'Peak lateral g           : {res.peak_lateral_g:.3f}')
    print(f'Steady-state roll (deg)  : {res.steady_state_roll_deg:.3f}')
    print(f'Peak roll (deg)          : {res.peak_roll_deg:.3f}')
    print(f'Yaw rate rise time (s)   : {res.yaw_rate_rise_time_s:.3f}')
    print(f'Yaw rate overshoot (%)   : {res.yaw_rate_overshoot_pct:.1f}')
    print(f'Peak understeer angle    : {res.peak_understeer_deg:.2f} deg')
    print(f'Peak Fz (FL/FR/RL/RR)    : '
          + '/'.join(f'{np.max(res.Fz[c]):.0f}' for c in _CORNERS) + ' N')
