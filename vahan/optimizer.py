"""
vahan/optimizer.py — Inverse Kinematics for Suspension Design

Given target metric curves, find hardpoint positions that produce them.
Uses a hybrid global-local optimization strategy:
  1. Differential Evolution (global search, coarse grid)
  2. Levenberg-Marquardt (local refinement, fine grid)

The existing forward solver (SuspensionConstraints → KinematicMetrics) is
used as a black box inside the optimization loop.
"""

import hashlib
import logging

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares, differential_evolution

from vahan import DoubleWishboneHardpoints
from vahan.solver import SuspensionConstraints, SolvedState
from vahan.kinematics import KinematicMetrics
from vahan.metrics_catalog import CATALOG_MAP, compute_ackermann_post

# Valid constructor kwargs for the double-wishbone corner solver.  A live GUI
# hp dict also carries topology-specific extras (tbar_*, arb_lca_attach,
# rocker_tbar_drop_pt, decoupled-cradle points) and may hold None-valued slots
# (e.g. DIRECT has no pushrod) — none of those may be forwarded to the
# dataclass, which rejects unknown kwargs.
_DWH_FIELD_NAMES = set(DoubleWishboneHardpoints.__dataclass_fields__)

log = logging.getLogger(__name__)


class IKEvaluationError(RuntimeError):
    """A defect in the IK forward evaluation itself (not a geometry that
    fails to close).  Raised instead of being converted to NaN so a coding
    error can never masquerade as "no solution at this station"."""


# Exceptions that indicate a bug in the evaluation code, never a geometry
# that legitimately fails to assemble at one travel station.
_CODE_DEFECTS = (NameError, AttributeError, TypeError, KeyError, ImportError)

# Rule 01 / Rule 04 static gates (mm) and Rule 02 axis gate (deg) -- the same
# numbers vahan.packaging.Tolerances uses for coplanar_mm / arb_inplane_mm /
# rocker_axis_deg.  Kept literal so the optimizer does not import the
# packaging module at load time; test_one_model.py asserts they stay equal.
CHAIN_COPLANAR_GATE_MM = 3.0
ARB_INPLANE_GATE_MM = 3.0
ROCKER_AXIS_GATE_DEG = 1e-4
# Residual normalisation: 1.0 == one design-target unit (Rule 01 design target
# is < 0.1 mm static out-of-plane), so the solver drives toward zero offset.
CHAIN_RESIDUAL_UNIT_MM = 0.1

# Points that define the rocker plate (Rule 01 plane) -- moving any of them
# re-derives rocker_axis_pt (Rule 02: axis = pivot + plane normal x length).
_PLATE_KEYS = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')
_CHAIN_KEYS = ('pushrod_outer', 'pushrod_inner', 'rocker_pivot',
               'rocker_spring_pt', 'spring_chassis_pt')
_DROP_LINK_KEYS = ('arb_drop_top', 'arb_arm_end')


def geometry_fingerprint(hp_dict: dict) -> str:
    """Stable stamp of a hardpoint dict (1 micrometre resolution).

    Bound to every IK result at SOLVE time; Apply refuses when the live
    geometry's stamp differs (the solution was computed for another car)."""
    h = hashlib.sha1()
    for k in sorted(hp_dict):
        v = hp_dict[k]
        h.update(str(k).encode())
        if v is None:
            h.update(b'<None>')
            continue
        a = np.asarray(v, float).ravel()
        # +0.0 folds -0.0 into 0.0 so the sign of a zero never changes the stamp
        h.update((np.round(a * 1e6) + 0.0).tobytes())
    return h.hexdigest()[:16]


def derive_rocker_axis(hp: dict, ref_hp: dict | None = None) -> dict:
    """Rule 02: regenerate ``rocker_axis_pt`` = pivot + (unit normal of the
    static rocker plate) x (reference axis length), keeping the reference
    axis sense.  Same construction as ``vahan.relocate.resolve_bundle``.
    Modifies and returns ``hp``; no-op when the plate is degenerate or any
    plate point is missing."""
    ref = hp if ref_hp is None else ref_hp
    if not all(hp.get(k) is not None for k in _PLATE_KEYS):
        return hp
    pv = np.asarray(hp['rocker_pivot'], float)
    n = np.cross(np.asarray(hp['pushrod_inner'], float) - pv,
                 np.asarray(hp['rocker_spring_pt'], float) - pv)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12 or not np.isfinite(nn):
        return hp
    n = n / nn
    L = 0.0254
    if ref.get('rocker_axis_pt') is not None and ref.get('rocker_pivot') is not None:
        old = (np.asarray(ref['rocker_axis_pt'], float)
               - np.asarray(ref['rocker_pivot'], float))
        lo = float(np.linalg.norm(old))
        if lo > 1e-9:
            L = lo
            if float(old @ n) < 0.0:
                n = -n
    hp['rocker_axis_pt'] = pv + n * L
    return hp


def static_chain_rule_metrics(hp: dict, side: str = 'left',
                              include_drop_link: bool = True) -> dict | None:
    """Rules 01/02/04 at STATIC for one corner's hardpoint dict.

    Thin wrapper over the shared checker
    ``vahan.packaging.actuation_chain_plate_metrics`` (the same function the
    packaging validator and the regression net use): the plane is the CURRENT
    static plate through rocker pivot / pushrod inner / rocker spring eye; the
    pushrod outer+inner, pivot, spring eye, spring chassis eye and (bellcrank
    ARB) both drop-link ends are measured against it; the rocker axis error is
    its angle from that plane's normal.  Returns None when the corner has no
    per-corner rocker chain (DIRECT, cradle topologies)."""
    if not all(hp.get(k) is not None
               and np.all(np.isfinite(np.asarray(hp[k], float)))
               for k in _CHAIN_KEYS):
        return None
    from vahan.packaging import actuation_chain_plate_metrics
    state = {k: np.asarray(hp[k], float) for k in _CHAIN_KEYS}
    axis = None
    if hp.get('rocker_axis_pt') is not None:
        axis = (np.asarray(hp['rocker_axis_pt'], float)
                - np.asarray(hp['rocker_pivot'], float))
    else:
        state['rocker_axis_pt'] = state['rocker_pivot'] + np.array([0., 0.0254, 0.])
    arb = None
    if include_drop_link and all(hp.get(k) is not None for k in _DROP_LINK_KEYS):
        arb = {k: np.asarray(hp[k], float) for k in _DROP_LINK_KEYS}
    out = actuation_chain_plate_metrics(
        [(0.0, state)], outboard_sign=(1.0 if side == 'left' else -1.0),
        rocker_axis=axis, arb=arb)
    signed = out.get('static_signed_mm', {})
    chain_mm = max((abs(signed[k]) for k in _CHAIN_KEYS if k in signed),
                   default=float('nan'))
    drop_mm = (max(abs(signed[k]) for k in _DROP_LINK_KEYS)
               if arb is not None else 0.0)
    out['chain_static_mm'] = float(chain_mm)
    out['drop_link_static_mm'] = float(drop_mm)
    out['drop_link_checked'] = arb is not None
    return out


def chain_rule_violations(m: dict | None) -> list[str]:
    """Plain-language reasons a static chain fails Rules 01/02/04."""
    if m is None:
        return []
    if not np.isfinite(m.get('coplanar_static_mm', np.nan)):
        return ['rocker plate is degenerate (pivot, pushrod and spring eye '
                'in a line) - no actuation plane']
    bad = []
    if m['chain_static_mm'] > CHAIN_COPLANAR_GATE_MM:
        bad.append(f"actuation chain {m['chain_static_mm']:.2f} mm out of its "
                   f"static plane (limit {CHAIN_COPLANAR_GATE_MM:.1f} mm), "
                   f"worst at {m.get('worst_point')}")
    if m.get('drop_link_checked') and m['drop_link_static_mm'] > ARB_INPLANE_GATE_MM:
        bad.append(f"ARB drop link {m['drop_link_static_mm']:.2f} mm out of the "
                   f"rocker plane (limit {ARB_INPLANE_GATE_MM:.1f} mm)")
    ax = m.get('rocker_axis_normal_error_deg', np.nan)
    if not (np.isfinite(ax) and ax <= ROCKER_AXIS_GATE_DEG):
        bad.append(f'rocker pivot axis {ax:.4f} deg off the plate normal')
    return bad


def _undefined_by_definition(metric_key: str, travel: np.ndarray,
                             motion: str) -> np.ndarray:
    """Stations where a metric is undefined by its own definition (not a
    solve failure): the ARB motion ratio is the secant (bar angle / wheel
    travel) and has no value at exactly zero travel."""
    t = np.asarray(travel, float)
    if metric_key == 'arb_mr' and motion != 'steer':
        return np.abs(t) < 1e-9
    return np.zeros(t.shape, bool)


# ─── Design variable specification ───────────────────────────────────────────

@dataclass
class DesignVar:
    """One adjustable hardpoint coordinate."""
    point:  str       # e.g. 'uca_front'
    coord:  int       # 0=X, 1=Y, 2=Z
    bound:  float     # max deviation in metres from current value

    @property
    def label(self):
        ax = 'XYZ'[self.coord]
        return f'{self.point}.{ax}'


# ─── Orthogonal variable groups ─────────────────────────────────────────────
# Each group targets variables that primarily affect ONE geometric aspect
# while minimally disturbing others.  Used for staged solving.
#
# KEY PRINCIPLE: front-view geometry (camber, RC) is controlled by arm
# heights & lateral positions.  Side-view geometry (anti-dive/squat) is
# controlled by fore-aft pivot axis TILT (Z-difference between front and
# rear inboard mounts).  Tie rods are nearly perfectly orthogonal to
# everything else.  Pushrod/rocker is completely independent.

ORTHO_GROUPS: dict[str, list[dict]] = {
    # Group 1 — Motion ratio: pushrod/rocker only. ZERO cross-contamination.
    'motion_ratio': [
        dict(point='pushrod_outer', coord=0), dict(point='pushrod_outer', coord=2),
        dict(point='pushrod_inner', coord=0), dict(point='pushrod_inner', coord=2),
        dict(point='rocker_spring_pt', coord=0), dict(point='rocker_spring_pt', coord=2),
    ],
    # Group 2 — Bump steer / toe: tie rod only. Near-zero cross-contamination.
    'toe': [
        dict(point='tie_rod_outer', coord=2),   # dominant: height rel to LCA arc
        dict(point='tie_rod_inner', coord=2),
        dict(point='tie_rod_inner', coord=1),
        dict(point='tie_rod_outer', coord=1),
    ],
    # Group 2b — Ackermann %: tie rod geometry (controls inner/outer steer split).
    # Same variables as toe — Ackermann is entirely determined by the steering
    # linkage geometry (tie rod length, height, and fore-aft position).
    'ackermann': [
        dict(point='tie_rod_outer', coord=2),   # dominant: height rel to LCA arc
        dict(point='tie_rod_inner', coord=2),
        dict(point='tie_rod_inner', coord=1),
        dict(point='tie_rod_outer', coord=1),
        dict(point='tie_rod_inner', coord=0),   # lateral position also affects Ackermann
        dict(point='tie_rod_outer', coord=0),
    ],
    # Group 2c — Rack position only: moves the inboard tie-rod pickup
    # (= rack end) in X/Y/Z WITHOUT touching tie_rod_outer.  This changes
    # Ackermann geometry by repositioning the rack while keeping the outer
    # ball joint (on the upright/knuckle) fixed — preserving steering ratio
    # and rack length.  Use when the dynamic ideal Ackermann target differs
    # from the current Ackermann curve.
    'rack_position': [
        dict(point='tie_rod_inner', coord=0),   # lateral (X) — rack offset
        dict(point='tie_rod_inner', coord=1),   # fore-aft (Y) — rack fore/aft
        dict(point='tie_rod_inner', coord=2),   # vertical (Z) — rack height
    ],
    # Group 3 — Anti-dive/squat/lift: side-view pivot axis TILT.
    # Only change the Z-difference between front & rear inboard mounts,
    # NOT their average Z (which would shift front-view geometry).
    'anti_dive': [
        dict(point='uca_front', coord=1), dict(point='uca_rear', coord=1),
        dict(point='lca_front', coord=1), dict(point='lca_rear', coord=1),
    ],
    'anti_squat': [
        dict(point='uca_front', coord=1), dict(point='uca_rear', coord=1),
        dict(point='lca_front', coord=1), dict(point='lca_rear', coord=1),
    ],
    'anti_lift': [
        dict(point='uca_front', coord=1), dict(point='uca_rear', coord=1),
        dict(point='lca_front', coord=1), dict(point='lca_rear', coord=1),
    ],
    # Group 4 — Camber gain: outboard BJ Z-heights + inboard lateral (X).
    # Outer Z changes FVSA length (camber gain). Inboard X changes arm
    # length ratio (camber gain rate). Avoids inboard Y (anti) and
    # outer X (caster).
    'camber': [
        dict(point='uca_outer', coord=2), dict(point='lca_outer', coord=2),
        dict(point='uca_front', coord=2), dict(point='uca_rear', coord=2),
        dict(point='lca_front', coord=2), dict(point='lca_rear', coord=2),
        dict(point='uca_front', coord=0), dict(point='lca_front', coord=0),
    ],
    # Group 5 — Roll centre height: coupled with camber via front-view IC.
    # Uses same front-view variables. Typically solved together with camber.
    'rc_height': [
        dict(point='uca_front', coord=2), dict(point='uca_rear', coord=2),
        dict(point='lca_front', coord=2), dict(point='lca_rear', coord=2),
        dict(point='uca_front', coord=0), dict(point='lca_front', coord=0),
        dict(point='uca_outer', coord=2), dict(point='lca_outer', coord=2),
    ],
    # Group 6 — Caster / trail: outer BJ fore-aft (X) offsets.
    # Minimal effect on front-view (camber/RC) or side-view (anti).
    'caster': [
        dict(point='uca_outer', coord=1),
        dict(point='lca_outer', coord=1),
    ],
    'trail': [
        dict(point='uca_outer', coord=1),
        dict(point='lca_outer', coord=1),
    ],
    # Group 7 — ARB motion ratio: ARB bellcrank geometry.
    # arb_drop_top connects to the rocker, arb_arm_end is the lever,
    # arb_pivot is the torsion bar rotation axis.
    'arb_mr': [
        dict(point='arb_drop_top', coord=0), dict(point='arb_drop_top', coord=2),
        dict(point='arb_arm_end', coord=0), dict(point='arb_arm_end', coord=2),
        dict(point='arb_pivot', coord=0), dict(point='arb_pivot', coord=2),
    ],
}

# Backward compat alias
PRESETS = ORTHO_GROUPS

# Solving priority: metrics that can be solved most independently go first.
# Each tuple: (metric_key, group_key) — group_key indexes ORTHO_GROUPS.
SOLVE_ORDER = [
    'motion_ratio',   # completely independent
    'arb_mr',         # ARB bellcrank — independent of suspension arms
    'toe',            # near-zero cross-contamination
    'ackermann',      # same variable group as toe (steer mode only)
    'anti_dive',      # side-view, minimal front-view effect
    'anti_squat',
    'anti_lift',
    'camber',         # front-view, coupled with RC
    'rc_height',      # coupled with camber — solve together or after
    'caster',         # minor tweaks last
    'trail',
]


# ─── Tube collision detection ──────────────────────────────────────────────
# Each suspension member is a tube with an outer diameter.  If two non-
# connected tubes overlap (centre-to-centre distance < sum of radii),
# the geometry is physically impossible and the solution is rejected.

SUSPENSION_MEMBERS = [
    # (point_a, point_b, member_name)
    ('uca_front', 'uca_outer',          'uca_front_arm'),
    ('uca_rear',  'uca_outer',          'uca_rear_arm'),
    ('lca_front', 'lca_outer',          'lca_front_arm'),
    ('lca_rear',  'lca_outer',          'lca_rear_arm'),
    ('tie_rod_inner', 'tie_rod_outer',  'tie_rod'),
    ('pushrod_outer', 'pushrod_inner',  'pushrod'),
    ('rocker_spring_pt', 'spring_chassis_pt', 'spring_damper'),
]

# Default outer diameters in metres (typical FSAE tubes)
DEFAULT_TUBE_OD: dict[str, float] = {
    'uca_front_arm': 0.0254,    # 1 in
    'uca_rear_arm':  0.0254,
    'lca_front_arm': 0.0254,
    'lca_rear_arm':  0.0254,
    'tie_rod':       0.0190,    # 3/4 in
    'pushrod':       0.0190,
    'spring_damper': 0.0508,    # 2 in  (spring + damper body)
}


def _segment_distance(p1, q1, p2, q2):
    """Minimum distance between 3-D line segments p1–q1 and p2–q2."""
    d1 = q1 - p1
    d2 = q2 - p2
    r  = p1 - p2
    a  = float(d1 @ d1)
    e  = float(d2 @ d2)
    f  = float(d2 @ r)
    EPS = 1e-12

    if a <= EPS and e <= EPS:          # both are points
        return float(np.linalg.norm(r))

    if a <= EPS:                       # segment 1 is a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPS:                   # segment 2 is a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:                          # general case
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = (np.clip((b * f - c * e) / denom, 0.0, 1.0)
                 if abs(denom) > EPS else 0.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def check_collisions(hp_dict: dict,
                     tube_od: dict[str, float] | None = None
                     ) -> list[dict]:
    """
    Check for physical collisions between suspension tube members.

    Returns a list of dicts for every colliding pair::

        {'member_a', 'member_b', 'distance_mm',
         'min_clearance_mm', 'overlap_mm'}

    Empty list → no collisions.
    """
    if tube_od is None:
        tube_od = DEFAULT_TUBE_OD

    collisions = []
    n = len(SUSPENSION_MEMBERS)
    for i in range(n):
        pa, qa, ma = SUSPENSION_MEMBERS[i]
        if pa not in hp_dict or qa not in hp_dict:
            continue
        ra = tube_od.get(ma, 0.025) / 2.0

        for j in range(i + 1, n):
            pb, qb, mb = SUSPENSION_MEMBERS[j]
            if pb not in hp_dict or qb not in hp_dict:
                continue
            # Members that share an endpoint are physically connected
            if {pa, qa} & {pb, qb}:
                continue
            rb = tube_od.get(mb, 0.025) / 2.0

            dist = _segment_distance(
                hp_dict[pa], hp_dict[qa],
                hp_dict[pb], hp_dict[qb])
            min_clear = ra + rb
            overlap = min_clear - dist

            if overlap > 0:
                collisions.append({
                    'member_a': ma,
                    'member_b': mb,
                    'distance_mm':      round(dist * 1000, 2),
                    'min_clearance_mm': round(min_clear * 1000, 2),
                    'overlap_mm':       round(overlap * 1000, 2),
                })
    return collisions


def _build_collision_pairs(hp_dict, tube_od):
    """Pre-compute (pa, qa, ra, pb, qb, rb) tuples for residual penalty."""
    pairs = []
    if tube_od is None:
        return pairs
    n = len(SUSPENSION_MEMBERS)
    for i in range(n):
        pa, qa, ma = SUSPENSION_MEMBERS[i]
        if pa not in hp_dict or qa not in hp_dict:
            continue
        ra = tube_od.get(ma, 0.025) / 2.0
        for j in range(i + 1, n):
            pb, qb, mb = SUSPENSION_MEMBERS[j]
            if pb not in hp_dict or qb not in hp_dict:
                continue
            if {pa, qa} & {pb, qb}:
                continue
            rb = tube_od.get(mb, 0.025) / 2.0
            pairs.append((pa, qa, ra, pb, qb, rb))
    return pairs


# ─── Design space: pack/unpack between hp dict and flat vector ───────────────

class DesignSpace:
    """Maps between a hardpoint dict and a flat optimisation vector."""

    def __init__(self, base_hp: dict, variables: list[DesignVar]):
        self.base_hp = {k: v.copy() for k, v in base_hp.items()}
        self.variables = list(variables)
        self.n = len(self.variables)
        for v in self.variables:
            if v.point == 'rocker_axis_pt':
                raise ValueError('rocker_axis_pt is DERIVED (pivot + plate '
                                 'normal, Rule 02) - move rocker_pivot, '
                                 'pushrod_inner or rocker_spring_pt instead')
        # Rule 02: when a plate point moves, the pivot axis is re-derived as
        # the new plate normal (the corner solver already rotates the rocker
        # about that normal; the stored point must agree with it).
        self._derive_axis = any(v.point in _PLATE_KEYS for v in self.variables)

    def pack(self, hp: dict) -> np.ndarray:
        """Extract variable coordinates from hp dict → flat array."""
        return np.array([hp[v.point][v.coord] for v in self.variables])

    def unpack(self, x: np.ndarray) -> dict:
        """Flat array → modified hp dict (copies base, applies x)."""
        hp = {k: v.copy() for k, v in self.base_hp.items()}
        for i, v in enumerate(self.variables):
            hp[v.point][v.coord] = x[i]
        if self._derive_axis:
            derive_rocker_axis(hp, self.base_hp)
        return hp

    def x0(self) -> np.ndarray:
        """Current (base) hardpoint values as a flat vector."""
        return self.pack(self.base_hp)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bound arrays."""
        x = self.x0()
        lo = np.array([x[i] - self.variables[i].bound for i in range(self.n)])
        hi = np.array([x[i] + self.variables[i].bound for i in range(self.n)])
        return lo, hi

    def bounds_list(self) -> list[tuple[float, float]]:
        """List of (lo, hi) tuples for DE."""
        lo, hi = self.bounds()
        return list(zip(lo.tolist(), hi.tolist()))


# ─── Target specification ────────────────────────────────────────────────────

def shaped_target(lo: float, hi: float, n: int,
                  shape: str = 'linear', curvature: float = 2.0) -> np.ndarray:
    """Build a target metric curve from ``lo`` (at MIN travel / droop) to
    ``hi`` (at MAX travel / bump) with a chosen nonlinearity in normalised
    travel τ ∈ [0,1].

        linear       f = τ
        progressive  f = τ^p          (p = curvature > 1: rises slowly then
                                       fast — most stiffening DEEP in bump;
                                       this is what holds ride height under
                                       downforce as the suspension compresses)
        digressive   f = 1 − (1−τ)^p  (rises fast then plateaus)
        exponential  f = (e^{kτ}−1)/(e^{k}−1)   (k = curvature; smooth)

    target(τ) = lo + (hi − lo)·f(τ).  Constant target -> lo == hi.
    """
    tau = np.linspace(0.0, 1.0, int(n))
    s = (shape or 'linear').lower()
    if s in ('progressive', 'power', 'rising'):
        p = max(float(curvature), 0.1)
        f = tau ** p
    elif s in ('digressive', 'falling'):
        p = max(float(curvature), 0.1)
        f = 1.0 - (1.0 - tau) ** p
    elif s in ('exponential', 'exp'):
        k = float(curvature)
        f = tau if abs(k) < 1e-6 else (np.exp(k * tau) - 1.0) / (np.exp(k) - 1.0)
    else:
        f = tau
    return float(lo) + (float(hi) - float(lo)) * f


@dataclass
class Target:
    """A target for one metric — either a constant value or a full curve."""
    metric_key: str           # e.g. 'camber', 'anti_dive'
    values:     np.ndarray    # target values at each travel step
    weight:     float = 1.0   # importance weight
    tolerance:  float = 0.0   # dead-band: no penalty inside ±tolerance


# ─── ARB bellcrank solver (for IK evaluation) ────────────────────────────────

def _rodrigues(v, k, theta):
    """Rodrigues' rotation: rotate v about axis k by angle theta."""
    ct, st_ = np.cos(theta), np.sin(theta)
    return v * ct + np.cross(k, v) * st_ + k * np.dot(k, v) * (1 - ct)


def _solve_arb_bellcrank(arb_drop_top_world, arb_hp):
    """
    Solve for the ARB bellcrank angle given the drop-link attachment position.

    Returns (arb_angle_rad, drop_link_travel_m).
    """
    pv = arb_hp['arb_pivot']
    ae0 = arb_hp['arb_arm_end']
    dt0 = arb_hp['arb_drop_top']

    bc_axis = np.array([1., 0., 0.])  # torsion bar runs lateral (X)
    arm_vec = ae0 - pv
    arm_len2 = float(arm_vec @ arm_vec)
    if arm_len2 < 1e-12:
        return 0., 0.

    dl_vec0 = dt0 - ae0
    dl_len2 = float(dl_vec0 @ dl_vec0)

    theta = 0.0
    for _ in range(60):
        arm_rot = _rodrigues(arm_vec, bc_axis, theta)
        ae_world = pv + arm_rot
        diff = ae_world - arb_drop_top_world
        res = float(diff @ diff) - dl_len2
        if abs(res) < 1e-14:
            break
        d_arm = np.cross(bc_axis, arm_rot)
        drdt = float(2.0 * diff @ d_arm)
        if abs(drdt) < 1e-14:
            break
        theta -= res / drdt
        # Clamp to physical range (no bellcrank rotates > 90 deg)
        theta = max(-np.pi / 2, min(np.pi / 2, theta))

    ae_world = pv + _rodrigues(arm_vec, bc_axis, theta)
    drop_link_travel = float(np.linalg.norm(ae_world - arb_drop_top_world)
                             - np.sqrt(dl_len2))
    return theta, drop_link_travel


# ─── Forward evaluation helper ───────────────────────────────────────────────

def _evaluate_sweep(hp_dict: dict, travel_arr: np.ndarray, side: str = 'left',
                    pushrod_body: str = 'uca',
                    metric_keys: list[str] | None = None,
                    anti_kwargs: dict | None = None,
                    motion: str = 'heave',
                    diagnostics: dict | None = None,
                    steer_params: dict | None = None) -> dict[str, np.ndarray]:
    """
    Run the forward solver over a travel array and return metric curves.

    A station that fails to assemble leaves NaN in the curves; when
    ``diagnostics`` (a dict) is given, each such failure is recorded in
    ``diagnostics['station_errors']`` as (index, stage, message).  A defect in
    the evaluation code itself raises :class:`IKEvaluationError`.

    motion controls what the travel_arr values mean:
        'heave'  — vertical wheel travel in metres
        'roll'   — treated as vertical travel on one corner (same as heave)
        'pitch'  — treated as vertical travel on one corner (same as heave)
        'steer'  — travel_arr is in degrees; converted to rack travel internally
    """
    hp_work = {k: v.copy() for k, v in hp_dict.items()}
    d = hp_work['tie_rod_outer'] - hp_work['tie_rod_inner']
    tierod_len_sq = float(d @ d)

    # Separate ARB points (not part of DoubleWishboneHardpoints)
    _ARB_KEYS = {'arb_drop_top', 'arb_arm_end', 'arb_pivot'}
    arb_hp = {k: hp_work[k] for k in _ARB_KEYS if k in hp_work
              and hp_work[k] is not None}
    has_arb = len(arb_hp) == 3
    # Forward ONLY recognised double-wishbone fields with real, finite values.
    # This silently ignores topology-specific extras (control-arm / T-bar ARB
    # points, decoupled-cradle nodes), None slots (e.g. DIRECT has no pushrod),
    # AND NaN slots (e.g. a cradle-resident pushrod_inner) so the corner solver
    # can be built for EVERY topology, not just bellcrank/corner cars.
    hp_solver = {k: v for k, v in hp_work.items()
                 if k in _DWH_FIELD_NAMES and v is not None
                 and np.all(np.isfinite(np.asarray(v, float)))}

    # ── Pick the damper-actuation mode the available hardpoints can build ──
    # Camber/toe/caster/RC-height/anti-geometry all come purely from the
    # 12-constraint wishbone+tierod solve and are INDEPENDENT of the damper
    # path; the mode only governs which optional pushrod/rocker/damper locals
    # SuspensionConstraints.__init__ constructs.  Letting it default to
    # 'pushrod' crashes DIRECT cars (no pushrod at all) and DECOUPLED cars
    # (pushrod_outer present, but the rocker/spring live on the external
    # cradle).  Detecting the buildable mode here lets IK geometry work for
    # EVERY topology — the solver returns valid wishbone geometry and flags
    # the unused spring path as NaN.
    if all(k in hp_solver for k in ('pushrod_outer', 'pushrod_inner',
                                    'rocker_pivot', 'rocker_spring_pt')):
        _damper_act = 'pushrod'        # full per-corner rocker (corner / heave-tbar)
    elif 'pushrod_outer' in hp_solver:
        _damper_act = 'cradle_link'    # DECOUPLED: rocker lives on the external cradle
    elif 'damper_chassis_pt' in hp_solver and 'damper_outer_pt' in hp_solver:
        _damper_act = 'direct'         # DIRECT: damper bolts straight to a moving body
    else:
        # No pushrod and no direct-damper points: synthesize a dummy
        # pushrod_outer (never read by geometry metrics) so the wishbone solve
        # can still run.  Defensive — unreached by the current topology space.
        hp_solver['pushrod_outer'] = np.asarray(hp_solver['uca_outer'], float)
        _damper_act = 'cradle_link'

    keys = metric_keys or list(CATALOG_MAP.keys())
    # Ackermann post-processing needs the toe curve — ensure it's computed
    _need_toe_for_ackermann = ('ackermann' in keys and 'toe' not in keys
                               and motion == 'steer')
    compute_keys = list(keys) + (['toe'] if _need_toe_for_ackermann else [])
    out = {k: np.full(len(travel_arr), np.nan) for k in compute_keys}
    extra = anti_kwargs or {}

    # Need ARB metrics?
    _arb_keys_needed = [k for k in compute_keys if k.startswith('arb_')]

    # For non-steer modes, build solver once (much faster)
    base_solver = None
    if motion != 'steer':
        hp_obj = DoubleWishboneHardpoints(
            **{k: np.array(v, float) for k, v in hp_solver.items()})
        base_solver = SuspensionConstraints(hp_obj,
                                             tierod_len_sq=tierod_len_sq,
                                             pushrod_body=pushrod_body,
                                             damper_actuation=_damper_act,
                                             damper_body=pushrod_body)

    # Two-pass sweep from center outward for warm-start continuity
    n = len(travel_arr)
    mid = np.argmin(np.abs(travel_arr))
    order = list(range(mid, n)) + list(range(mid - 1, -1, -1))

    spring_prev = None
    travel_prev = None

    for idx in order:
        t_raw = float(travel_arr[idx])
        try:
            if motion == 'steer':
                # Handwheel deg -> rack travel through the ONE conversion,
                # fed from the PROJECT steer block (mm/rev, stroke clamp,
                # rack direction).  This used to be a private 60 mm/rev.
                from vahan.steering import rack_travel_from_handwheel_deg
                if steer_params is None:
                    raise IKEvaluationError(
                        "steer-mode IK needs the project steer block "
                        "(steer_params) - no built-in rack")
                rack_m = rack_travel_from_handwheel_deg(t_raw, steer_params)
                hp_steer = {k: v.copy() for k, v in hp_solver.items()}
                hp_steer['tie_rod_inner'] = (hp_solver['tie_rod_inner']
                                             + np.array([rack_m, 0., 0.]))
                hp_obj = DoubleWishboneHardpoints(
                    **{k: np.array(v, float) for k, v in hp_steer.items()})
                solver = SuspensionConstraints(hp_obj,
                                               tierod_len_sq=tierod_len_sq,
                                               pushrod_body=pushrod_body,
                                               damper_actuation=_damper_act,
                                               damper_body=pushrod_body)
                t_solve = 0.0
            else:
                solver = base_solver
                t_solve = t_raw

            st = solver.solve(t_solve)
            m = KinematicMetrics(st, side)
            for k in compute_keys:
                entry = CATALOG_MAP.get(k)
                if entry is None:
                    continue
                try:
                    out[k][idx] = entry['fn'](m, spring_prev=spring_prev,
                                               travel_prev=travel_prev, **extra)
                except Exception:
                    pass

            # ARB metrics: compute from rocker angle + ARB bellcrank
            if has_arb and _arb_keys_needed:
                try:
                    pv = st.rocker_pivot
                    # REAL rocker axis, not a hardcoded +Y (84.4 deg off at the
                    # front).  This feeds IK target evaluation, so a wrong axis
                    # here optimises against a fictitious ARB.  Read from THIS
                    # sweep's hardpoints (hp_work) -- the same source the GUI's
                    # graph path uses.  (Was getattr(hp, ...) on an undefined
                    # name: a NameError swallowed below -> every ARB IK metric
                    # silently NaN, 2026-09-22 audit P0.)
                    _axp = hp_work.get('rocker_axis_pt')
                    ax_pt = (np.asarray(_axp, float) if _axp is not None
                             else pv + np.array([0., 0.0254, 0.]))
                    r_axis = ax_pt - pv
                    rn = np.linalg.norm(r_axis)
                    if rn > 1e-9:
                        r_axis = r_axis / rn
                    else:
                        r_axis = np.array([0., 1., 0.])
                    arm_dt = arb_hp['arb_drop_top'] - pv
                    dt_w = pv + _rodrigues(arm_dt, r_axis, st.rocker_angle)
                    ang, dl_t = _solve_arb_bellcrank(dt_w, arb_hp)
                    if 'arb_angle' in out:
                        out['arb_angle'][idx] = np.degrees(ang)
                    if 'arb_drop_travel' in out:
                        out['arb_drop_travel'][idx] = dl_t * 1000
                    if 'arb_mr' in out:
                        out['arb_mr'][idx] = min(abs(np.degrees(ang) / (t_raw * 1000)), 5.0) if abs(t_raw) > 1e-9 else float('nan')
                except _CODE_DEFECTS as e:
                    raise IKEvaluationError(
                        f'ARB metric evaluation defect: {type(e).__name__}: {e}') from e
                except Exception as e:      # geometry: bellcrank cannot close here
                    if diagnostics is not None:
                        diagnostics.setdefault('station_errors', []).append(
                            (int(idx), 'arb', f'{type(e).__name__}: {e}'))

            spring_prev = m.spring_length
            travel_prev = t_raw
        except IKEvaluationError:
            raise
        except Exception as e:              # corner does not assemble here
            if diagnostics is not None:
                diagnostics.setdefault('station_errors', []).append(
                    (int(idx), 'corner', f'{type(e).__name__}: {e}'))
            spring_prev = None
            travel_prev = None

    # ── Ackermann % post-processing (steer mode only) ────────────────────
    # Requires the full toe curve + vehicle params, so it runs after the loop.
    if 'ackermann' in keys and motion == 'steer':
        toe_curve = out.get('toe')
        if toe_curve is not None and not np.all(np.isnan(toe_curve)):
            wb = extra.get('wheelbase_m', 1.530)
            # Track width: prefer explicit param; fall back to 2 * |wc_x|
            ft = extra.get('front_track_m')
            if ft is None:
                wc_x = abs(float(hp_work.get('wheel_center',
                                              np.array([0.6, 0, 0]))[0]))
                ft = 2.0 * wc_x if wc_x > 0.1 else 1.222
            out['ackermann'] = compute_ackermann_post(
                toe_curve, travel_arr,
                wheelbase_m=wb, front_track_m=ft)

    # Remove auxiliary toe if it was only computed for ackermann
    if _need_toe_for_ackermann:
        out.pop('toe', None)

    return out


def _ls_diag(res, label: str) -> dict:
    """Retained scipy.optimize.least_squares termination diagnostics.
    ``success`` is False for status <= 0 (evaluation budget exhausted or an
    improper input) -- such a result is not applicable."""
    status = int(getattr(res, 'status', -1))
    ok = bool(getattr(res, 'success', False)) and status > 0
    ok = ok and bool(np.all(np.isfinite(getattr(res, 'x', [np.nan]))))
    return {'success': ok, 'status': status,
            'message': f'{label}: status {status}: {getattr(res, "message", "")}',
            'nfev': int(getattr(res, 'nfev', 0) or 0)}


# ─── Inverse solver ──────────────────────────────────────────────────────────

class InverseSolver:
    """
    Optimises hardpoint positions to match target metric curves.

    Workflow:
        solver = InverseSolver(hp_dict, side='left', pushrod_body='uca')
        solver.add_target('camber', target_values, weight=1.0)
        solver.set_variables(variables)
        result = solver.solve()
    """

    def __init__(self, hp_dict: dict, side: str = 'left',
                 pushrod_body: str = 'uca',
                 travel_mm: tuple[float, float] = (-40, 40),
                 n_points: int = 21,
                 anti_kwargs: dict | None = None,
                 motion: str = 'heave',
                 axle: str | None = None,
                 drop_link_in_plane: bool = True,
                 steer_params: dict | None = None):
        self.hp_dict = {k: v.copy() for k, v in hp_dict.items()}
        # Project steer block (rack mm/rev, stroke, direction) — required for
        # motion='steer'; the sweep converts handwheel deg with it.
        self.steer_params = dict(steer_params) if steer_params else None
        if motion == 'steer' and self.steer_params is None:
            raise ValueError("InverseSolver(motion='steer') needs steer_params "
                             "= the project steer block")
        # Solve context bound to every result: which axle this geometry is
        # and a stamp of it, so Apply can never write the solution onto the
        # other axle or onto geometry that changed after the solve.
        self.axle = axle
        self.geometry_stamp = geometry_fingerprint(self.hp_dict)
        # Rule 04 applies to bellcrank ARBs only (control-arm bars exempt).
        self.drop_link_in_plane = bool(drop_link_in_plane)
        self.side = side
        self.pushrod_body = pushrod_body
        self.motion = motion
        self.n_points = n_points
        self.targets: list[Target] = []
        self.ds: DesignSpace | None = None
        self.anti_kwargs = anti_kwargs or {}
        # Regularisation: penalise moving too far from the starting point
        self.regularisation = 0.1
        # Tube collision avoidance (set to dict of ODs to enable)
        self.tube_od: dict[str, float] | None = None
        self._collision_pairs: list[tuple] = []

        # Build the travel array — units depend on motion type
        # heave/roll/pitch: mm → metres (vertical wheel travel)
        # steer: degrees of steering angle (stays as-is, not divided)
        if motion == 'steer':
            # For steer mode, travel_arr stores the steer angle in degrees
            # _evaluate_sweep handles conversion to rack travel
            self.travel = np.linspace(travel_mm[0], travel_mm[1], n_points)
        else:
            # heave / roll / pitch — all reduce to vertical travel on one corner
            self.travel = np.linspace(travel_mm[0] / 1000, travel_mm[1] / 1000, n_points)

    def add_target(self, metric_key: str, target_values: np.ndarray | float,
                   weight: float = 1.0, tolerance: float = 0.0):
        """Add a target. If scalar, broadcasts to constant curve.

        tolerance: dead-band in metric units. No penalty for deviations
                   within ±tolerance of the target. Use for lock constraints
                   so they allow small drift without fighting the primary.
        """
        if np.isscalar(target_values):
            target_values = np.full(self.n_points, float(target_values))
        self.targets.append(Target(metric_key, np.asarray(target_values, float),
                                   weight, tolerance))

    def clear_targets(self):
        self.targets.clear()

    def set_variables(self, variables: list[DesignVar]):
        self.ds = DesignSpace(self.hp_dict, variables)

    def set_variables_from_preset(self, preset_key: str, bound_mm: float = 10.0):
        """Use a named preset group of design variables."""
        specs = PRESETS.get(preset_key)
        if specs is None:
            raise ValueError(f'Unknown preset: {preset_key}. '
                             f'Available: {list(PRESETS.keys())}')
        bound_m = bound_mm / 1000.0
        variables = []
        seen = set()
        for s in specs:
            key = (s['point'], s['coord'])
            if key not in seen and s['point'] in self.hp_dict:
                seen.add(key)
                variables.append(DesignVar(s['point'], s['coord'], bound_m))
        self.ds = DesignSpace(self.hp_dict, variables)

    def _metric_keys(self) -> list[str]:
        return list({t.metric_key for t in self.targets})

    def _eval(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Run forward sweep for a given design-variable vector."""
        hp = self.ds.unpack(x)
        return _evaluate_sweep(hp, self.travel, self.side, self.pushrod_body,
                                self._metric_keys(), self.anti_kwargs,
                                motion=self.motion,
                                steer_params=self.steer_params)

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        """Residual vector for least-squares (not squared)."""
        curves = self._eval(x)
        parts = []
        for t in self.targets:
            predicted = curves.get(t.metric_key, np.full(self.n_points, np.nan))
            diff = predicted - t.values
            # Replace NaN with a large penalty
            diff = np.where(np.isnan(diff), 10.0, diff)
            # Dead-band: zero penalty inside ±tolerance
            if t.tolerance > 0:
                diff = np.sign(diff) * np.maximum(np.abs(diff) - t.tolerance, 0.0)
            parts.append(np.sqrt(t.weight) * diff)

        # Regularisation toward starting position
        if self.regularisation > 0:
            x0 = self.ds.x0()
            # Normalise by bounds so all variables contribute equally
            _, hi = self.ds.bounds()
            span = hi - x0
            span = np.where(np.abs(span) < 1e-9, 1.0, span)
            reg = self.regularisation * (x - x0) / span
            parts.append(reg)

        # Collision avoidance penalty (fixed-length, one per pair)
        if self._collision_pairs:
            hp = self.ds.unpack(x)
            coll = np.zeros(len(self._collision_pairs))
            for k, (pa, qa, ra, pb, qb, rb) in enumerate(
                    self._collision_pairs):
                dist = _segment_distance(hp[pa], hp[qa], hp[pb], hp[qb])
                # Smooth ramp: penalty starts 1 mm before contact
                gap = dist - (ra + rb)
                margin = 0.001        # 1 mm safety buffer
                if gap < margin:
                    coll[k] = 2000.0 * (margin - gap)
            parts.append(coll)

        # Static actuation-chain rule (Rules 01/02/04, 2026-09-15): EVERY
        # chain point -- pushrod outer+inner, rocker pivot, rocker spring eye,
        # spring chassis eye and (bellcrank ARB) both drop-link ends -- in the
        # CURRENT static plate plane of this candidate, via the shared checker.
        # The plane is re-evaluated for every candidate (never the original
        # plane); the pivot axis is re-derived as that plane's normal in
        # DesignSpace.unpack, so axis normality holds by construction.
        # Residuals are signed mm / CHAIN_RESIDUAL_UNIT_MM (1.0 == 0.1 mm).
        # (Replaces a 3-point residual against the ORIGINAL plane that could
        # not see pushrod_outer / spring_chassis_pt / drop links -- audit P0.)
        hp_curr = self.ds.unpack(x) if not self._collision_pairs else hp
        m = static_chain_rule_metrics(hp_curr, self.side,
                                      include_drop_link=self.drop_link_in_plane)
        if m is not None:
            names = self._chain_residual_names(m)
            sm = m.get('static_signed_mm', {})
            vals = [sm.get(k, np.nan) for k in names]
            parts.append(np.array([v / CHAIN_RESIDUAL_UNIT_MM if np.isfinite(v)
                                   else 10.0 / CHAIN_RESIDUAL_UNIT_MM
                                   for v in vals]))

        return np.concatenate(parts)

    def _chain_residual_names(self, m: dict) -> list[str]:
        """Fixed-length list of chain points in the residual (least_squares
        needs a constant residual length across candidates)."""
        if not hasattr(self, '_chain_names'):
            self._chain_names = list(_CHAIN_KEYS) + (
                list(_DROP_LINK_KEYS) if m.get('drop_link_checked') else [])
        return self._chain_names

    def _cost(self, x: np.ndarray) -> float:
        r = self._residuals(x)
        return float(r @ r)

    def solve(self, method: str = 'hybrid',
              progress_cb=None,
              warm_start: np.ndarray | None = None) -> dict:
        """
        Run the inverse solver.

        Parameters
        ----------
        method : 'staged' | 'hybrid' | 'local' | 'global'
        progress_cb : callable(str) or None — status callback
        warm_start : optional initial x vector (skips DE, goes straight to LM)

        Returns
        -------
        dict with keys:
            'hp': optimised hardpoint dict
            'x': optimised variable vector
            'cost': final cost value
            'curves': dict of metric curves at the solution
            'variables': list of DesignVar
            'deltas_mm': per-variable change from start (in mm)
        """
        if self.ds is None:
            raise RuntimeError('Call set_variables() or set_variables_from_preset() first')
        if not self.targets:
            raise RuntimeError('No targets set. Call add_target() first')

        # Pre-compute collision pairs (fixed-length residual vector)
        self._collision_pairs = _build_collision_pairs(
            self.hp_dict, self.tube_od)

        x0 = self.ds.x0()
        lo, hi = self.ds.bounds()

        if warm_start is not None:
            # Clamp warm-start to new (wider) bounds
            x_start = np.clip(warm_start, lo, hi)
            if progress_cb:
                progress_cb('Refining from warm start (LM)...')
            res_lm = least_squares(
                self._residuals, x_start, bounds=(lo, hi),
                method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
            )
            x_final = res_lm.x
            cost = float(res_lm.cost)
            diag = _ls_diag(res_lm, 'warm-start LM')

        elif method == 'staged':
            # ── Priority-ordered staged solving ─────────────────────
            # Solve each primary target in isolation using ONLY its
            # orthogonal variable group (from ORTHO_GROUPS), then do a
            # final polish pass with all variables + all targets.
            #
            # WHY:  Different suspension metrics are controlled by
            # geometrically independent hardpoint subsets.  Solving
            # camber with front-view variables doesn't disturb anti-
            # geometry (side-view).  Solving toe with tie-rod variables
            # disturbs nothing else.  By solving each metric with its
            # own group first, we land in the correct neighbourhood
            # BEFORE the final polish, avoiding cross-contamination
            # that makes the monolithic solver struggle.
            # ────────────────────────────────────────────────────────

            hp_work = {k: v.copy() for k, v in self.hp_dict.items()}
            target_map = {t.metric_key: t for t in self.targets}
            user_vars = {(v.point, v.coord): v for v in self.ds.variables}

            # Recover travel range (mm) for sub-solver construction
            if self.motion == 'steer':
                travel_range = (float(self.travel[0]),
                                float(self.travel[-1]))
            else:
                travel_range = (float(self.travel[0] * 1000),
                                float(self.travel[-1] * 1000))

            # Collect stages: only PRIMARY targets (tolerance == 0)
            # in SOLVE_ORDER priority.  Lock constraints (tolerance > 0)
            # are deferred to the final polish — their target is "keep
            # current" so solving them in isolation is a no-op.
            stages = []
            ordered = set()
            for mk in SOLVE_ORDER:
                if mk in target_map and mk in ORTHO_GROUPS:
                    t = target_map[mk]
                    if t.tolerance <= 0:
                        stages.append((mk, t))
                        ordered.add(mk)
            # Any primary targets not in SOLVE_ORDER (future metrics)
            for t in self.targets:
                if (t.metric_key not in ordered
                        and t.tolerance <= 0
                        and t.metric_key in ORTHO_GROUPS):
                    stages.append((t.metric_key, t))

            n_stages = len(stages)
            stage_diags = []
            for i, (metric_key, target) in enumerate(stages):
                group = ORTHO_GROUPS[metric_key]

                # Intersect: only group variables the user also selected
                stage_vars = []
                seen = set()
                for g in group:
                    key = (g['point'], g['coord'])
                    if key in user_vars and key not in seen:
                        seen.add(key)
                        stage_vars.append(DesignVar(
                            g['point'], g['coord'],
                            user_vars[key].bound))

                if not stage_vars:
                    continue

                if progress_cb:
                    progress_cb(
                        f'Stage {i+1}/{n_stages}: {metric_key} '
                        f'({len(stage_vars)} vars)...')

                # Sub-solver: only this metric, only this group's vars
                stage_ik = InverseSolver(
                    hp_work, side=self.side,
                    pushrod_body=self.pushrod_body,
                    travel_mm=travel_range,
                    n_points=self.n_points,
                    anti_kwargs=self.anti_kwargs,
                    motion=self.motion,
                    steer_params=self.steer_params,
                )
                stage_ik.add_target(
                    metric_key, target.values, weight=1.0)
                stage_ik.set_variables(stage_vars)
                stage_ik.regularisation = 0.05  # light: stay close
                stage_ik.tube_od = self.tube_od

                stage_res = stage_ik.solve(method='local')
                stage_diags.append((metric_key, stage_res.get('solver_success'),
                                    stage_res.get('solver_message', '')))
                hp_work = {k: v.copy()
                           for k, v in stage_res['hp'].items()}

            # ── Final polish: all vars + all targets ────────────────
            if progress_cb:
                progress_cb('Final polish (all targets + all vars)...')

            polish_ik = InverseSolver(
                self.hp_dict,   # ORIGINAL base → correct bounds
                side=self.side,
                pushrod_body=self.pushrod_body,
                travel_mm=travel_range,
                n_points=self.n_points,
                anti_kwargs=self.anti_kwargs,
                motion=self.motion,
                steer_params=self.steer_params,
            )
            for t in self.targets:
                polish_ik.add_target(
                    t.metric_key, t.values, t.weight, t.tolerance)
            polish_ik.set_variables(list(self.ds.variables))
            polish_ik.regularisation = self.regularisation
            polish_ik.tube_od = self.tube_od

            # Warm-start from staged result, clipped to original bounds
            staged_x = polish_ik.ds.pack(hp_work)
            lo_p, hi_p = polish_ik.ds.bounds()
            warm_x = np.clip(staged_x, lo_p, hi_p)

            res_polish = least_squares(
                polish_ik._residuals, warm_x,
                bounds=(lo_p, hi_p),
                method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
            )
            x_final = res_polish.x
            cost = float(res_polish.cost)
            diag = _ls_diag(res_polish, 'staged final polish')
            diag['stages'] = stage_diags

        elif method == 'global':
            if progress_cb:
                progress_cb('Running global search (Differential Evolution)...')
            res_de = differential_evolution(
                self._cost,
                bounds=self.ds.bounds_list(),
                x0=x0,
                seed=42,
                maxiter=150,
                tol=1e-8,
                polish=False,
                mutation=(0.5, 1.5),
                recombination=0.9,
                workers=-1,
                updating='deferred',
            )
            x_final = res_de.x
            cost = self._cost(x_final)
            diag = {'success': bool(res_de.success), 'status': None,
                    'message': f'differential evolution: {res_de.message}',
                    'nfev': int(getattr(res_de, 'nfev', 0) or 0)}

        elif method == 'hybrid':
            # Multi-start LM: try N random starting points + the base x0,
            # keep the best.  Much faster than DE for most IK landscapes.
            n_starts = 5
            rng = np.random.default_rng(42)
            starts = [x0]    # always include the current hardpoints
            for _ in range(n_starts - 1):
                starts.append(rng.uniform(lo, hi))

            best_x = x0
            best_cost = float('inf')
            best_res = None
            start_errors = []
            for i, xs in enumerate(starts):
                if progress_cb:
                    progress_cb(f'Multi-start LM: {i+1}/{n_starts}...')
                try:
                    res = least_squares(
                        self._residuals, xs, bounds=(lo, hi),
                        method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
                    )
                except IKEvaluationError:
                    raise                     # code defect: never hide it
                except Exception as e:        # this start failed numerically
                    start_errors.append(f'start {i+1}: {type(e).__name__}: {e}')
                    log.warning('IK hybrid start %d failed: %s', i + 1, e)
                    continue
                # Prefer converged starts; among equals, the lower cost.
                if best_res is None or (
                        (bool(res.success), -float(res.cost))
                        > (bool(best_res.success), -best_cost)):
                    best_cost = float(res.cost)
                    best_x = res.x
                    best_res = res
            x_final = best_x
            cost = best_cost
            if best_res is None:
                diag = {'success': False, 'status': None, 'nfev': 0,
                        'message': 'every multi-start LM start failed: '
                                   + '; '.join(start_errors)}
            else:
                diag = _ls_diag(best_res, f'multi-start LM (best of {n_starts})')
                if start_errors:
                    diag['message'] += (f' [{len(start_errors)} start(s) '
                                        f'failed: ' + '; '.join(start_errors) + ']')

        else:   # 'local'
            if progress_cb:
                progress_cb('Running local solve (LM)...')
            res_lm = least_squares(
                self._residuals, x0, bounds=(lo, hi),
                method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
            )
            x_final = res_lm.x
            cost = float(res_lm.cost)
            diag = _ls_diag(res_lm, 'local LM')

        hp_final = self.ds.unpack(x_final)
        eval_diag = {}
        curves = _evaluate_sweep(hp_final, self.travel, self.side, self.pushrod_body,
                                  self._metric_keys(), self.anti_kwargs,
                                  motion=self.motion, diagnostics=eval_diag,
                                  steer_params=self.steer_params)
        deltas = (x_final - x0) * 1000  # metres → mm

        # travel_mm depends on motion type
        if self.motion == 'steer':
            travel_mm = self.travel  # already in degrees
        else:
            travel_mm = self.travel * 1000

        # ── Saturation analysis ──────────────────────────────────────────
        # Which variables hit their movement limit?  (≥85% of bound used)
        saturated = []
        for i, v in enumerate(self.ds.variables):
            delta_abs = abs(float(deltas[i]))
            bound_mm = v.bound * 1000
            pct = delta_abs / bound_mm if bound_mm > 1e-6 else 0.0
            if pct >= 0.85:
                saturated.append({
                    'index': i,
                    'label': v.label,
                    'delta_mm': float(deltas[i]),
                    'bound_mm': bound_mm,
                    'pct_used': pct,
                })

        # Primary target error (first target = the one the user asked for).
        # Evaluated only over stations where the metric is defined; a NaN at
        # any other station makes the whole error NaN (never nanmax-hidden).
        primary = self.targets[0]
        primary_curve = curves.get(primary.metric_key,
                                   np.full(self.n_points, np.nan))
        primary_errors = np.abs(primary_curve - primary.values)
        undefined = _undefined_by_definition(primary.metric_key, self.travel,
                                             self.motion)
        defined_err = primary_errors[~undefined]
        primary_max_error = (float(np.max(defined_err))
                             if defined_err.size and np.all(np.isfinite(defined_err))
                             else float('nan'))

        # ── Collision check ──────────────────────────────────────────────
        collisions = (check_collisions(hp_final, self.tube_od)
                      if self.tube_od else [])

        # ── Result contract: may this result be applied to the model? ────
        reasons = []
        if not diag.get('success', False):
            reasons.append(f"solver did not converge ({diag.get('message', '')})")
        if not np.isfinite(cost):
            reasons.append('final objective is not finite')
        for t in self.targets:
            c = np.asarray(curves.get(t.metric_key,
                                      np.full(self.n_points, np.nan)), float)
            bad = ~np.isfinite(c) & ~_undefined_by_definition(
                t.metric_key, self.travel, self.motion)
            if bad.any():
                tr = (self.travel if self.motion == 'steer'
                      else self.travel * 1000.0)
                unit = 'deg' if self.motion == 'steer' else 'mm'
                reasons.append(
                    f'{t.metric_key} not solved at {int(bad.sum())} of '
                    f'{len(c)} travel stations (e.g. {tr[bad][0]:+.1f} {unit})')
        if not np.isfinite(primary_max_error):
            reasons.append(f'{primary.metric_key} target error is not finite')
        # Rules 01/02/04 at static: refuse if the solve introduced or worsened
        # a violation; a pre-existing violation is reported, not blamed on IK.
        chain_base = static_chain_rule_metrics(
            self.hp_dict, self.side, include_drop_link=self.drop_link_in_plane)
        chain_final = static_chain_rule_metrics(
            hp_final, self.side, include_drop_link=self.drop_link_in_plane)
        base_bad = chain_rule_violations(chain_base)
        chain_warnings = []
        for msg in chain_rule_violations(chain_final):
            worse = True
            if chain_base is not None and base_bad:
                worse = (not np.isfinite(chain_final['coplanar_static_mm'])
                         or chain_final['chain_static_mm']
                         > chain_base['chain_static_mm'] + 0.01
                         or chain_final['drop_link_static_mm']
                         > chain_base['drop_link_static_mm'] + 0.01
                         or chain_final['rocker_axis_normal_error_deg']
                         > chain_base['rocker_axis_normal_error_deg'] + 1e-6)
            if worse:
                reasons.append(msg)
            else:
                chain_warnings.append(f'pre-existing: {msg}')
        applicable = not reasons

        return {
            'axle': self.axle,
            'geometry_stamp': self.geometry_stamp,
            'side': self.side,
            'motion': self.motion,
            'solver_success': bool(diag.get('success', False)),
            'solver_status': diag.get('status'),
            'solver_message': diag.get('message', ''),
            'solver_nfev': diag.get('nfev'),
            'solver_stages': diag.get('stages', []),
            'station_errors': eval_diag.get('station_errors', []),
            'chain_rules': chain_final,
            'chain_rules_base': chain_base,
            'chain_warnings': chain_warnings,
            'applicable': applicable,
            'reject_reasons': reasons,
            'hp': hp_final,
            'x': x_final,
            'cost': cost,
            'curves': curves,
            'targets': {t.metric_key: t.values for t in self.targets},
            'travel_mm': travel_mm,
            'variables': self.ds.variables,
            'deltas_mm': deltas,
            'saturated': saturated,
            'primary_max_error': primary_max_error,
            'primary_metric': primary.metric_key,
            'collisions': collisions,
        }


# ─── Parallel explore helper (module-level for pickling) ────────────────────

def _solve_at_bound(args: tuple) -> dict:
    """
    Solve one IK instance at a given bound level with warm-start.
    Designed to be called via multiprocessing.Pool.map().

    args = (solver_kwargs, bound_mm, warm_x, bound_label)
    solver_kwargs has everything needed to rebuild an InverseSolver.
    """
    solver_kwargs, bound_mm, warm_x, bound_label = args

    # Deserialise: lists → numpy arrays
    hp_dict = {k: np.array(v, float) for k, v in solver_kwargs['hp_dict'].items()}
    side = solver_kwargs['side']
    pushrod_body = solver_kwargs['pushrod_body']
    travel_mm = solver_kwargs['travel_mm']
    n_points = solver_kwargs['n_points']
    anti_kwargs = solver_kwargs['anti_kwargs']
    motion = solver_kwargs['motion']
    targets_spec = solver_kwargs['targets']       # list of (key, values, weight, tol)
    var_specs = solver_kwargs['var_specs']         # list of (point, coord)

    ik = InverseSolver(
        hp_dict, side=side, pushrod_body=pushrod_body,
        travel_mm=travel_mm, n_points=n_points,
        anti_kwargs=anti_kwargs, motion=motion,
        axle=solver_kwargs.get('axle'),
        drop_link_in_plane=solver_kwargs.get('drop_link_in_plane', True),
        steer_params=solver_kwargs.get('steer_params'),
    )

    for entry in targets_spec:
        key, values, weight = entry[0], entry[1], entry[2]
        tol = entry[3] if len(entry) > 3 else 0.0
        ik.add_target(key, np.array(values), weight, tolerance=tol)

    variables = [DesignVar(pt, coord, bound_mm / 1000) for pt, coord in var_specs]
    ik.set_variables(variables)

    # Collision avoidance
    tube_od = solver_kwargs.get('tube_od')
    if tube_od:
        ik.tube_od = tube_od

    warm = np.array(warm_x) if warm_x is not None else None
    if warm is not None and len(warm) != len(variables):
        warm = None   # shape mismatch — fall back to x0

    result = ik.solve(method='hybrid' if warm is None else 'local',
                      warm_start=warm)
    result['bound_label'] = bound_label
    return result
