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
