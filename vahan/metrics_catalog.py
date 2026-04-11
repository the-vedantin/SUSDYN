"""
metrics_catalog.py
──────────────────
Defines every quantity Vahan can compute and plot.

Each entry in CATALOG is a dict:
    key      : unique string id
    label    : human-readable name
    unit     : display unit string
    category : grouping for the picker UI
    fn       : callable(KinematicMetrics, extra=None) -> float
               'extra' carries per-step data that isn't in KinematicMetrics
               (e.g. spring_length_prev for motion ratio)
"""

import numpy as np


# ── raw metric functions ──────────────────────────────────────────────────────

def _camber(m, **_):          return m.camber
def _toe(m, **_):             return m.toe
def _caster(m, **_):          return m.caster
def _kpi(m, **_):             return m.kpi
def _scrub(m, **_):           return m.scrub_radius * 1000
def _trail(m, **_):           return m.mechanical_trail * 1000
def _rc_height(m, **_):       return m.roll_center_height * 1000
def _spring_len(m, **_):      return m.spring_length * 1000
def _rocker_ang(m, **_):      return m.rocker_angle_deg
def _wc_x(m, **_):            return float(m._s.wheel_center[0]) * 1000
def _wc_y(m, **_):            return float(m._s.wheel_center[1]) * 1000
def _wc_z(m, **_):            return float(m._s.wheel_center[2]) * 1000
def _travel(m, **_):          return float(m._s.travel) * 1000

def _motion_ratio(m, spring_prev=None, travel_prev=None, **_):
    """MR = Δspring / Δwheel_travel  (both in metres → dimensionless)."""
    if spring_prev is None or travel_prev is None:
        return 0.0
    dt = float(m._s.travel) - travel_prev
    if abs(dt) < 1e-9:
        return 0.0
    ds = m.spring_length - spring_prev
    return abs(ds / dt)

def _ic_y(m, **_):
    """Instant-centre lateral position (mm from centreline, YZ plane)."""
    s = m._s
    uca_in = np.array([(s.uca_front[1]+s.uca_rear[1])/2,
                        (s.uca_front[2]+s.uca_rear[2])/2])
    lca_in = np.array([(s.lca_front[1]+s.lca_rear[1])/2,
                        (s.lca_front[2]+s.lca_rear[2])/2])
    from vahan.kinematics import _intersect_2d
    ic = _intersect_2d(uca_in,
                       np.array([s.uca_outer[1], s.uca_outer[2]]),
                       lca_in,
                       np.array([s.lca_outer[1], s.lca_outer[2]]))
    return float(ic[0]) * 1000 if ic is not None else 0.0

def _ic_z(m, **_):
    """Instant-centre height (mm)."""
    s = m._s
    uca_in = np.array([(s.uca_front[1]+s.uca_rear[1])/2,
                        (s.uca_front[2]+s.uca_rear[2])/2])
    lca_in = np.array([(s.lca_front[1]+s.lca_rear[1])/2,
                        (s.lca_front[2]+s.lca_rear[2])/2])
    from vahan.kinematics import _intersect_2d
    ic = _intersect_2d(uca_in,
                       np.array([s.uca_outer[1], s.uca_outer[2]]),
                       lca_in,
                       np.array([s.lca_outer[1], s.lca_outer[2]]))
    return float(ic[1]) * 1000 if ic is not None else 0.0

def _sv_ic_coeff(s):
    """
    Shared helper: side-view (YZ-plane) IC geometric anti coefficient.

    Uses the 3D virtual-arm method:
      1. Find the foot-of-perpendicular from the outer BJ to the pivot axis.
      2. arm = outer - foot
      3. vel = pivot_axis_hat × arm  (velocity of outer if arm rotates about pivot)
      4. Project vel to YZ plane → perpendicular gives the side-view arm direction.
      5. Intersect the two arm lines in the YZ plane.

    Returns Z_ic / (Y_ic - wc_y).  Positive when IC is forward of the axle
    (Y_ic < wc_y in Y+ = rearward convention), which is the anti position.
    Returns NaN on failure.
    """
    from vahan.kinematics import _intersect_2d

    def _sv_arm_line(pa, pb, outer):
        """YZ side-view arm line for one wishbone: (point_yz, direction_yz)."""
        pa = np.asarray(pa, float)
        pb = np.asarray(pb, float)
        outer = np.asarray(outer, float)
        d = pb - pa
        d_len = np.linalg.norm(d)
        if d_len < 1e-9:
            return None, None
        d_hat = d / d_len
        t = float(np.dot(outer - pa, d_hat))
        q = pa + t * d_hat                      # foot of perpendicular on pivot axis
        arm = outer - q
        vel = np.cross(d_hat, arm)              # velocity direction of outer BJ
        vel_yz = np.array([vel[1], vel[2]])     # project to side view (Y,Z)
        if np.linalg.norm(vel_yz) < 1e-9:
            return None, None
        # arm direction in side view is perpendicular to vel_yz
        arm_dir_yz = np.array([-vel_yz[1], vel_yz[0]])
        pt_yz = np.array([outer[1], outer[2]])
        return pt_yz, arm_dir_yz

    uca_pt, uca_dir = _sv_arm_line(s.uca_front, s.uca_rear, s.uca_outer)
    lca_pt, lca_dir = _sv_arm_line(s.lca_front, s.lca_rear, s.lca_outer)
    if uca_pt is None or lca_pt is None:
        return float('nan')

    ic = _intersect_2d(uca_pt, uca_pt + uca_dir, lca_pt, lca_pt + lca_dir)
    if ic is None:
        return float('nan')

    dy = ic[0] - s.wheel_center[1]   # Y_ic − wc_y  (negative when IC is forward)
    dz = ic[1]                        # IC height above ground
    if abs(dy) < 1e-3:
        return float('nan')
    return dz / dy


def _anti_dive(m, cg_height_m=0.28, wheelbase_m=1.524,
               front_brake_bias=0.65, **_):
    """
    Anti-Dive % (front suspension, braking).
    = (Z_ic / (wc_y − Y_ic)) × (wheelbase / h_cg) × front_brake_bias × 100
    100 % = no pitch under braking regardless of spring stiffness.
    """
    coeff = _sv_ic_coeff(m._s)
    if np.isnan(coeff):
        return float('nan')
    return coeff * (wheelbase_m / cg_height_m) * front_brake_bias * 100.0


def _anti_squat(m, cg_height_m=0.28, wheelbase_m=1.524,
                rear_drive_bias=1.0, **_):
    """
    Anti-Squat % (rear suspension, acceleration, RWD by default).
    = (Z_ic / (wc_y − Y_ic)) × (wheelbase / h_cg) × rear_drive_bias × 100
    100 % = no squat under full-throttle acceleration.

    Sign note: _sv_ic_coeff returns Z/(Y_ic - wc_y).  For the rear axle the
    anti position has IC *ahead* of the axle (Y_ic < wc_y → coeff < 0), so
    we negate to get a positive anti-squat percentage.
    """
    coeff = _sv_ic_coeff(m._s)
    if np.isnan(coeff):
        return float('nan')
    return -coeff * (wheelbase_m / cg_height_m) * rear_drive_bias * 100.0


def _anti_lift(m, cg_height_m=0.28, wheelbase_m=1.524,
               front_drive_bias=0.0, **_):
    """
    Anti-Lift % (front suspension, acceleration).
    = (Z_ic / (wc_y − Y_ic)) × (wheelbase / h_cg) × front_drive_bias × 100
    Zero for RWD (front_drive_bias = 0).  Non-zero for AWD or FWD.
    """
    coeff = _sv_ic_coeff(m._s)
    if np.isnan(coeff):
        return float('nan')
    return coeff * (wheelbase_m / cg_height_m) * front_drive_bias * 100.0


def _kingpin_len(m, **_):
    """Kingpin axis length (upright height) mm."""
    from vahan.solver import _d2
    return float(np.sqrt(_d2(m._s.uca_outer, m._s.lca_outer))) * 1000

def _steer_angle(m, **_):
    """
    Front wheel steer angle (deg) — angle of wheel spin axis in top view (XY).
    Same sign as toe but measured from the opposite side convention.
    """
    return m.toe  # reuses toe computation — positive = toe-in = steer inward

def _ackermann_pct(m, ackermann_pct_value=None, **_):
    """
    Ackermann percentage — populated by post-processing in compute_ackermann_post().

    Per-step fn just reads the pre-computed value injected via extra kwargs.
    Returns NaN if not yet computed (e.g. non-steer modes).
    """
    if ackermann_pct_value is not None:
        return float(ackermann_pct_value)
    return float('nan')


def compute_ackermann_post(toe_curve: np.ndarray,
                           steer_input: np.ndarray,
                           wheelbase_m: float = 1.530,
                           front_track_m: float = 1.222) -> np.ndarray:
    """
    Post-process a steer-sweep toe curve into Ackermann percentage.

    For a symmetric car, both corners share the same hardpoints (mirrored).
    A steer sweep on the left corner at input angles [-A ... 0 ... +A] gives
    the left-wheel steer angle at each step.  By mirror symmetry, the
    right-wheel steer angle at input +d equals the left-wheel angle at -d
    (with sign flip because toe-in is positive and the directions mirror).

    At each steer input d:
        delta_near  = |toe(+d)|   (this corner — nearer to turn centre)
        delta_far   = |toe(-d)|   (opposite corner — farther from turn centre)

    The ideal Ackermann relationship for turn radius R:
        inner_ideal = atan(L / (R - t/2))
        outer_ideal = atan(L / (R + t/2))
    where R = L / tan(delta_bicycle), delta_bicycle = (delta_near + delta_far) / 2.

    Ackermann % = (delta_near - delta_far) / (inner_ideal - outer_ideal) * 100

    At zero steer, returns NaN (0/0 indeterminate).

    Parameters
    ----------
    toe_curve : array, shape (n,)
        Toe angle in degrees at each steer step. Positive = toe-in.
        In steer mode, toe represents the wheel steer angle.
    steer_input : array, shape (n,)
        Steering input values (degrees) — the x-axis of the sweep.
        Must be symmetric about zero (e.g. linspace(-20, 20, n)).
    wheelbase_m : float
        Vehicle wheelbase in metres.
    front_track_m : float
        Front track width in metres.

    Returns
    -------
    ackermann : array, shape (n,)
        Ackermann percentage at each steer step. NaN where undefined.
    """
    n = len(toe_curve)
    ack = np.full(n, np.nan)
    L = wheelbase_m
    t = front_track_m

    # Build a lookup: for each steer input, find the index of the
    # mirror input (-steer).  The sweep is assumed to be a linspace
    # symmetric about zero, so mirror index is simply (n-1-i).
    for i in range(n):
        steer_deg = steer_input[i]
        # Skip near-zero steer (indeterminate)
        if abs(steer_deg) < 0.05:
            continue

        toe_this = toe_curve[i]
        # Mirror index: opposite end of the symmetric sweep
        mirror_i = n - 1 - i
        toe_mirror = toe_curve[mirror_i]

        # Actual steer angles (degrees).
        # In steer mode, positive steer input → positive toe on inner wheel.
        # Toe sign: positive = toe-in.  For a left-turn (positive rack input),
        # the left wheel (inner) toes-in more → positive toe.
        # The mirrored value has the opposite sign convention, so negate.
        delta_near = abs(toe_this)       # this corner
        delta_far  = abs(toe_mirror)     # opposite corner (by symmetry)

        # Identify inner (larger angle) vs outer (smaller angle)
        delta_inner = max(delta_near, delta_far)
        delta_outer = min(delta_near, delta_far)

        # Actual toe-difference (steer angle spread)
        actual_diff = delta_inner - delta_outer

        # Average steer angle → bicycle-model turn radius
        delta_avg_rad = np.radians((delta_inner + delta_outer) / 2.0)
        if abs(delta_avg_rad) < 1e-6:
            continue
        R = L / np.tan(delta_avg_rad)

        # Ideal Ackermann angles
        denom_inner = R - t / 2.0
        denom_outer = R + t / 2.0
        if abs(denom_inner) < 1e-6 or denom_outer < 1e-6:
            continue
        ideal_inner = np.degrees(np.arctan(L / denom_inner))
        ideal_outer = np.degrees(np.arctan(L / denom_outer))
        ideal_diff = ideal_inner - ideal_outer

        if abs(ideal_diff) < 1e-9:
            continue

        ack[i] = (actual_diff / ideal_diff) * 100.0

    return ack

# ── ARB metrics ───────────────────────────────────────────────────────────────
# These are computed externally (passed via extra kwargs) because they require
# ARB hardpoints not stored in KinematicMetrics.
# The fn receives arb_angle, arb_drop_travel, arb_mr via **_ kwargs.

def _arb_angle(m, arb_angle=None, **_):
    return float(np.degrees(arb_angle)) if arb_angle is not None else float('nan')

def _arb_drop_travel(m, arb_drop_travel=None, **_):
    return float(arb_drop_travel * 1000) if arb_drop_travel is not None else float('nan')

def _arb_mr(m, arb_mr=None, **_):
    return float(arb_mr) if arb_mr is not None else float('nan')


# ── catalog ───────────────────────────────────────────────────────────────────

CATALOG = [
    # ── Angles ───────────────────────────────────────────────────────────────
    dict(key='camber',       label='Camber Angle',         unit='°',    category='Angles',    fn=_camber),
    dict(key='toe',          label='Toe Angle',            unit='°',    category='Angles',    fn=_toe),
    dict(key='caster',       label='Caster Angle',         unit='°',    category='Angles',    fn=_caster),
    dict(key='kpi',          label='KPI Angle',            unit='°',    category='Angles',    fn=_kpi),
    dict(key='rocker_angle', label='Rocker Angle',         unit='°',    category='Angles',    fn=_rocker_ang),

    # ── Lengths ───────────────────────────────────────────────────────────────
    dict(key='scrub',        label='Scrub Radius',         unit='mm',   category='Lengths',   fn=_scrub),
    dict(key='trail',        label='Mechanical Trail',     unit='mm',   category='Lengths',   fn=_trail),
    dict(key='spring_len',   label='Spring/Damper Length', unit='mm',   category='Lengths',   fn=_spring_len),
    dict(key='kp_len',       label='Kingpin Length',       unit='mm',   category='Lengths',   fn=_kingpin_len),

    # ── Geometry ──────────────────────────────────────────────────────────────
    dict(key='rc_height',    label='Roll Centre Height',   unit='mm',   category='Geometry',  fn=_rc_height),
    dict(key='ic_y',         label='Instant Centre Y',     unit='mm',   category='Geometry',  fn=_ic_y),
    dict(key='ic_z',         label='Instant Centre Z',     unit='mm',   category='Geometry',  fn=_ic_z),

    # ── Anti Geometry ─────────────────────────────────────────────────────────
    # Requires CG height, wheelbase, and force-distribution biases (passed via
    # extra kwargs from _all_metrics).  Defaults: h_cg=280 mm, wb=1524 mm,
    # front_brake_bias=65 %, RWD (rear_drive=100 %, front_drive=0 %).
    dict(key='anti_dive',  label='Anti-Dive',  unit='%', category='Anti', fn=_anti_dive),
    dict(key='anti_squat', label='Anti-Squat', unit='%', category='Anti', fn=_anti_squat),
    dict(key='anti_lift',  label='Anti-Lift',  unit='%', category='Anti', fn=_anti_lift),

    # ── Wheel Centre ──────────────────────────────────────────────────────────
    dict(key='wc_x',         label='Wheel Centre X',       unit='mm',   category='Wheel Ctr', fn=_wc_x),
    dict(key='wc_y',         label='Wheel Centre Y',       unit='mm',   category='Wheel Ctr', fn=_wc_y),
    dict(key='wc_z',         label='Wheel Centre Z',       unit='mm',   category='Wheel Ctr', fn=_wc_z),

    # ── Ratios ────────────────────────────────────────────────────────────────
    dict(key='motion_ratio', label='Motion Ratio (damper/wheel)',    unit='-',   category='Ratios', fn=_motion_ratio),

    # ── Steering ──────────────────────────────────────────────────────────────
    dict(key='steer_angle',  label='Front Wheel Steer Angle',       unit='°',   category='Steering', fn=_steer_angle),
    dict(key='ackermann',    label='Ackermann %',                    unit='%',   category='Steering', fn=_ackermann_pct),

    # ── ARB ───────────────────────────────────────────────────────────────────
    dict(key='arb_angle',       label='ARB Angle',                  unit='°',   category='ARB', fn=_arb_angle),
    dict(key='arb_drop_travel', label='ARB Drop Link Travel',       unit='mm',  category='ARB', fn=_arb_drop_travel),
    dict(key='arb_mr',          label='ARB Motion Ratio',           unit='-',   category='ARB', fn=_arb_mr),
]

# Fast lookup by key
CATALOG_MAP = {e['key']: e for e in CATALOG}
CATALOG_KEYS = [e['key'] for e in CATALOG]

# Default graph selections
DEFAULT_Y_KEYS = ['camber', 'toe', 'rc_height', 'motion_ratio']
