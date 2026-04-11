"""
Kinematic metrics derived from a SolvedState.

Axis convention (matches CAD environment):
    X  →  lateral   (outboard positive for the corner being modelled)
    Y  →  longitudinal (forward positive)
    Z  →  up (positive)

All angles in degrees. All lengths in metres unless stated otherwise.

Sign conventions (left corner default, right mirrors via _sign):
    Camber   negative = top of wheel leans inboard  (negative camber)
    Toe      positive = toe-in
    Caster   positive = top of kingpin tilts rearward (-Y)
    KPI      positive = top of kingpin tilts inboard (-X for left corner)
    Scrub radius    positive = KP ground point is inboard of contact patch
    Mechanical trail positive = contact patch is behind KP ground point (+Y behind)
    Roll-centre height positive = above ground
"""

import numpy as np
from .solver import SolvedState, _norm


def _intersect_2d(p1, p2, p3, p4):
    """2-D line intersection. Returns None if parallel."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(denom) < 1e-12:
        return None
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / denom
    return p1 + t * d1


class KinematicMetrics:
    """
    All suspension kinematic metrics from a single SolvedState.

    Args:
        state : SolvedState from SuspensionConstraints.solve()
        side  : 'left' or 'right' — controls sign conventions
    """

    def __init__(self, state: SolvedState, side: str = 'left'):
        if side not in ('left', 'right'):
            raise ValueError("side must be 'left' or 'right'")
        self._s    = state
        self._sign = 1.0 if side == 'left' else -1.0

    # ── kingpin axis ─────────────────────────────────────────────────────────

    @property
    def kingpin_axis(self) -> np.ndarray:
        """Unit vector of kingpin, pointing LCA BJ → UCA BJ."""
        return _norm(self._s.uca_outer - self._s.lca_outer)

    # ── camber ───────────────────────────────────────────────────────────────

    @property
    def camber(self) -> float:
        """
        Camber angle (deg).
        Front view (XZ plane). Negative = top of wheel leans inboard.
        Spin axis deviation from horizontal, measured in XZ plane.
        """
        spin  = self._s.spin_axis
        # angle of spin axis from horizontal; tilt in X direction
        angle = np.degrees(np.arctan2(spin[2], abs(spin[0])))
        return -angle * self._sign

    # ── toe ──────────────────────────────────────────────────────────────────

    @property
    def toe(self) -> float:
        """
        Toe angle (deg). Positive = toe-in.
        Top view (XY plane). Deviation of spin axis from pure lateral (X).
        """
        spin  = self._s.spin_axis
        angle = np.degrees(np.arctan2(spin[1], abs(spin[0])))
        return -angle * self._sign

    # ── caster ───────────────────────────────────────────────────────────────

    @property
    def caster(self) -> float:
        """
        Caster angle (deg). Positive = top of kingpin leans rearward (-Y).
        Side view (YZ plane).
        """
        kp = self.kingpin_axis
        return np.degrees(np.arctan2(-kp[1], kp[2]))

    # ── KPI ──────────────────────────────────────────────────────────────────

    @property
    def kpi(self) -> float:
        """
        Kingpin inclination (deg). Positive = top leans inboard.
        Front view (XZ plane).
        """
        kp = self.kingpin_axis
        return np.degrees(np.arctan2(-kp[0] * self._sign, kp[2]))

    # ── kingpin ground intersection ──────────────────────────────────────────

    def _kingpin_ground(self) -> np.ndarray:
        """Point where kingpin axis intersects the ground plane (Z=0)."""
        kp_dir = self.kingpin_axis
        kp_pt  = self._s.lca_outer
        if abs(kp_dir[2]) < 1e-9:
            return kp_pt.copy()
        t = -kp_pt[2] / kp_dir[2]
        return kp_pt + t * kp_dir

    @property
    def scrub_radius(self) -> float:
        """
        Scrub radius (m). Positive = KP ground point is inboard of contact patch.
        Lateral (X) distance.
        """
        kg   = self._kingpin_ground()
        cp_x = self._s.wheel_center[0]
        return (cp_x - kg[0]) * self._sign

    @property
    def mechanical_trail(self) -> float:
        """
        Mechanical trail (m). Positive = contact patch behind KP ground point.
        Longitudinal (Y) distance.
        """
        kg   = self._kingpin_ground()
        cp_y = self._s.wheel_center[1]
        return -(cp_y - kg[1])

    # ── roll centre ──────────────────────────────────────────────────────────

    @property
    def ic_front_view(self):
        """
        Instant-centre in the front view (XZ plane) for this corner.
        Returns a 2-element array [x_m, z_m], or None if arms are parallel.
        Used by the axle-level roll-centre computation (requires both corners).
        """
        s = self._s
        uca_in  = np.array([(s.uca_front[0]+s.uca_rear[0])/2,
                             (s.uca_front[2]+s.uca_rear[2])/2])
        lca_in  = np.array([(s.lca_front[0]+s.lca_rear[0])/2,
                             (s.lca_front[2]+s.lca_rear[2])/2])
        uca_out = np.array([s.uca_outer[0], s.uca_outer[2]])
        lca_out = np.array([s.lca_outer[0], s.lca_outer[2]])
        return _intersect_2d(uca_in, uca_out, lca_in, lca_out)

    @property
    def roll_center_height(self) -> float:
        """
        Roll-centre height (m) — instant-centre method, front view (XZ plane).

        1. Project arm inboard midpoints + outboard BJs into XZ plane.
        2. Find IC = intersection of arm lines.
        3. Line from IC → contact patch; intersect with X=0 (centreline).
        """
        s = self._s

        # Inboard midpoints in XZ plane (X=lateral, Z=height)
        uca_in = np.array([(s.uca_front[0]+s.uca_rear[0])/2,
                            (s.uca_front[2]+s.uca_rear[2])/2])
        lca_in = np.array([(s.lca_front[0]+s.lca_rear[0])/2,
                            (s.lca_front[2]+s.lca_rear[2])/2])

        uca_out = np.array([s.uca_outer[0], s.uca_outer[2]])
        lca_out = np.array([s.lca_outer[0], s.lca_outer[2]])

        ic = _intersect_2d(uca_in, uca_out, lca_in, lca_out)

        # Contact patch in XZ (Z=0, X=wheel centre lateral pos)
        cp = np.array([s.wheel_center[0], 0.0])

        if ic is None:
            return 0.0   # parallel arms → RC at ground

        if abs(ic[0]) < 1e-6:
            return float(ic[1])

        # Centreline in XZ is the vertical line X=0
        cl_a = np.array([0.0, -1.0])
        cl_b = np.array([0.0,  1.0])
        rc = _intersect_2d(ic, cp, cl_a, cl_b)
        return float(rc[1]) if rc is not None else float(ic[1])

    # ── spring / rocker ──────────────────────────────────────────────────────

    @property
    def spring_length(self) -> float:
        return self._s.spring_length

    @property
    def rocker_angle_deg(self) -> float:
        return float(np.degrees(self._s.rocker_angle))

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """All metrics. Lengths in mm, angles in degrees."""
        return {
            'travel_mm':             self._s.travel * 1000,
            'camber_deg':            self.camber,
            'toe_deg':               self.toe,
            'caster_deg':            self.caster,
            'kpi_deg':               self.kpi,
            'scrub_radius_mm':       self.scrub_radius * 1000,
            'mechanical_trail_mm':   self.mechanical_trail * 1000,
            'roll_center_height_mm': self.roll_center_height * 1000,
            'spring_length_mm':      self.spring_length * 1000,
            'rocker_angle_deg':      self.rocker_angle_deg,
        }
