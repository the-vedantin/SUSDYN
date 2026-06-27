"""
vahan/differential.py — limited-slip differential model (Drexler FSAE LSD).

WHY IT MATTERS FOR VD
---------------------
The diff splits drive (and engine-overrun) torque between the two driven
wheels.  In a corner that split is asymmetric, which makes a YAW MOMENT:

  * ON POWER (corner exit): the clutch biases drive toward the slower INNER
    wheel, so the axle's force asymmetry RESISTS the turn -> understeer.
    More power-lock = more exit understeer (and more traction off the corner).
  * ON COAST (corner entry, trailing throttle / engine braking): the clutch
    biases overrun toward the inner wheel, which RESISTS the rear rotating ->
    corner-entry stability.  More coast-lock = less lift-off rotation; less
    coast-lock = a pointier, more rotation-happy entry.
  * PRELOAD acts at all times (the clutch is pre-clamped), so it adds a
    baseline of this stabilising effect even at neutral throttle.

DREXLER ADJUSTABLE FSAE LSD (2017 manual, "diffy.pdf")
------------------------------------------------------
Ramp angle -> theoretical lock-up torque (%):
    30 deg -> 88 %   40 -> 60   45 -> 51   50 -> 42   60 -> 29
Each ramp set has a POWER (drive) angle and a COAST (overrun) angle.  The 3
selectable options (power/coast):
    Option 1: 40/50  (basic/default)   Option 2: 45/60    Option 3: 30/45
Preload: 30-35 Nm new, settling to 25-30 Nm used.

MODEL
-----
locking torque transmitted by the clutch:
    T_lock = preload + locking_frac(ramp_deg) * |T_axle|        [N·m]
force bias across the driven axle and the resulting yaw moment:
    Mz_diff = T_lock * track / r_tire        [N·m]   (+ = understeer/stabilise)
(open diff -> frac=0, preload=0 -> Mz=0;  spool -> frac=1, fully locked.)

This is a steady, torque-bias model — it captures the balance TENDENCY the
diff sets, which is what the % is tuned for.  It does not model clutch slip
transients or the grip-limited cap on the force bias (a future refinement).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Drexler manual: ramp angle (deg) -> locking fraction (lock-up torque %)
_RAMP_DEG = np.array([30.0, 40.0, 45.0, 50.0, 60.0])
_RAMP_LOCK = np.array([0.88, 0.60, 0.51, 0.42, 0.29])

# Selectable ramp options (power_deg, coast_deg)
DREXLER_OPTIONS = {
    1: (40.0, 50.0),   # basic / default
    2: (45.0, 60.0),
    3: (30.0, 45.0),
}


def ramp_locking_fraction(ramp_deg: float) -> float:
    """Locking fraction for a Drexler ramp angle (linear interp of the
    manual's table; _RAMP_DEG is increasing as np.interp requires).  Smaller
    angle = more aggressive = more lock."""
    return float(np.interp(float(ramp_deg), _RAMP_DEG, _RAMP_LOCK))


@dataclass
class Differential:
    """Driven-axle differential.

    kind: 'open' | 'spool' | 'salisbury'
    power_ramp_deg / coast_ramp_deg: Drexler ramp angles (salisbury only)
    preload_Nm: clutch pre-clamp torque (baseline locking, both ramps)
    """
    kind: str = 'salisbury'
    power_ramp_deg: float = 40.0
    coast_ramp_deg: float = 50.0
    preload_Nm: float = 30.0

    # ── locking fractions ────────────────────────────────────────────────
    def locking_fraction(self, on_power: bool) -> float:
        if self.kind == 'open':
            return 0.0
        if self.kind == 'spool':
            return 1.0
        return ramp_locking_fraction(
            self.power_ramp_deg if on_power else self.coast_ramp_deg)

    @property
    def power_locking_pct(self) -> float:
        return 100.0 * self.locking_fraction(True)

    @property
    def coast_locking_pct(self) -> float:
        return 100.0 * self.locking_fraction(False)

    # ── torque + yaw moment ──────────────────────────────────────────────
    def locking_torque_Nm(self, axle_torque_Nm: float, on_power: bool) -> float:
        """Clutch locking torque transmitted across the plates.

        DEFINITION FIX: the Drexler "lock-up torque in percent" is the LOCK
        VALUE = (T_high - T_low)/(T_high + T_low) of the two wheel torques.
        Since T_high - T_low = 2*T_lock and T_high + T_low = T_axle:
            lock%  = 2*T_lock / T_axle   ->   T_lock = (lock%/2) * T_axle
        i.e. the inter-wheel FORCE bias = lock% * drive force (so 60% lock
        biases 60% of the drive to one wheel — bounded and intuitive).  The
        earlier model used lock% directly as the ramp factor, double-counting
        by 2x.  Preload adds a constant clutch torque on top."""
        if self.kind == 'open':
            return 0.0
        return self.preload_Nm + 0.5 * self.locking_fraction(on_power) \
            * abs(float(axle_torque_Nm))

    def yaw_moment_Nm(self, axle_torque_Nm: float, track_m: float,
                      tire_radius_m: float, on_power: bool,
                      max_bias_N: float | None = None) -> float:
        """Diff yaw moment (N·m).  Sign convention: POSITIVE = understeer/
        stabilising (resists the turn on power, resists rear rotation on
        coast).

        The clutch biases drive toward the slower wheel; the inter-wheel
        FORCE bias is ΔFx = 2·T_lock / r_tire, and Mz = ΔFx · track/2 =
        T_lock·track/r.  ΔFx is physically capped by ``max_bias_N`` (the
        tires can't transmit more force difference than their grip allows) —
        without it the model over-predicts at high drive torque / full lock.
        """
        r = max(float(tire_radius_m), 1e-3)
        T_lock = self.locking_torque_Nm(axle_torque_Nm, on_power)
        d_fx = 2.0 * T_lock / r
        if max_bias_N is not None and max_bias_N > 0.0:
            d_fx = min(d_fx, float(max_bias_N))
        return d_fx * float(track_m) / 2.0

    # ── description ──────────────────────────────────────────────────────
    def describe(self) -> str:
        if self.kind == 'open':
            return 'Open differential (no locking)'
        if self.kind == 'spool':
            return 'Spool / fully locked (both wheels same speed)'
        return (f'Drexler LSD — power {self.power_ramp_deg:.0f}° '
                f'({self.power_locking_pct:.0f}% lock) / coast '
                f'{self.coast_ramp_deg:.0f}° ({self.coast_locking_pct:.0f}% '
                f'lock), preload {self.preload_Nm:.0f} Nm')

    @classmethod
    def from_vehicle(cls, veh) -> 'Differential':
        """Build from VehicleParams diff_* fields (with sensible defaults)."""
        return cls(
            kind=getattr(veh, 'diff_kind', 'salisbury'),
            power_ramp_deg=float(getattr(veh, 'diff_power_ramp_deg', 40.0)),
            coast_ramp_deg=float(getattr(veh, 'diff_coast_ramp_deg', 50.0)),
            preload_Nm=float(getattr(veh, 'diff_preload_Nm', 30.0)),
        )
