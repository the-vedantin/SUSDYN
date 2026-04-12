"""
vahan/optimizer.py — Inverse Kinematics for Suspension Design

Given target metric curves, find hardpoint positions that produce them.
Uses a hybrid global-local optimization strategy:
  1. Differential Evolution (global search, coarse grid)
  2. Levenberg-Marquardt (local refinement, fine grid)

The existing forward solver (SuspensionConstraints → KinematicMetrics) is
used as a black box inside the optimization loop.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares, differential_evolution

from vahan import DoubleWishboneHardpoints
from vahan.solver import SuspensionConstraints, SolvedState
from vahan.kinematics import KinematicMetrics
from vahan.metrics_catalog import CATALOG_MAP, compute_ackermann_post


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

    def pack(self, hp: dict) -> np.ndarray:
        """Extract variable coordinates from hp dict → flat array."""
        return np.array([hp[v.point][v.coord] for v in self.variables])

    def unpack(self, x: np.ndarray) -> dict:
        """Flat array → modified hp dict (copies base, applies x)."""
        hp = {k: v.copy() for k, v in self.base_hp.items()}
        for i, v in enumerate(self.variables):
            hp[v.point][v.coord] = x[i]
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
                    motion: str = 'heave') -> dict[str, np.ndarray]:
    """
    Run the forward solver over a travel array and return metric curves.

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
    arb_hp = {k: hp_work[k] for k in _ARB_KEYS if k in hp_work}
    has_arb = len(arb_hp) == 3
    hp_solver = {k: v for k, v in hp_work.items() if k not in _ARB_KEYS}

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
                                             pushrod_body=pushrod_body)

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
                rack_mm_per_rev = 60.0
                rack_m = t_raw * rack_mm_per_rev / 360.0 / 1000.0
                hp_steer = {k: v.copy() for k, v in hp_solver.items()}
                hp_steer['tie_rod_inner'] = (hp_solver['tie_rod_inner']
                                             + np.array([rack_m, 0., 0.]))
                hp_obj = DoubleWishboneHardpoints(
                    **{k: np.array(v, float) for k, v in hp_steer.items()})
                solver = SuspensionConstraints(hp_obj,
                                               tierod_len_sq=tierod_len_sq,
                                               pushrod_body=pushrod_body)
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
                    ax_pt = pv + np.array([0., 0.0254, 0.])
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
                except Exception:
                    pass

            spring_prev = m.spring_length
            travel_prev = t_raw
        except Exception:
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
                 motion: str = 'heave'):
        self.hp_dict = {k: v.copy() for k, v in hp_dict.items()}
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
                                motion=self.motion)

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

        # Rocker coplanarity constraint: rocker_pivot, pushrod_inner, and
        # rocker_spring_pt must lie on the same plane (defined by the
        # design-position normal).  The rocker is a planar mechanism.
        hp_curr = self.ds.unpack(x) if not self._collision_pairs else hp
        if all(k in hp_curr for k in ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')):
            pv = hp_curr['rocker_pivot']
            pi = hp_curr['pushrod_inner']
            rs = hp_curr['rocker_spring_pt']
            # Design-position plane normal
            if not hasattr(self, '_rocker_plane_normal'):
                pv0 = self.hp_dict['rocker_pivot']
                a = self.hp_dict['pushrod_inner'] - pv0
                b = self.hp_dict['rocker_spring_pt'] - pv0
                n = np.cross(a, b)
                norm = np.linalg.norm(n)
                self._rocker_plane_normal = n / norm if norm > 1e-9 else np.array([0., 1., 0.])
            n_hat = self._rocker_plane_normal
            # Penalise out-of-plane deviation for both arm tips
            dev_pi = float(np.dot(pi - pv, n_hat))
            dev_rs = float(np.dot(rs - pv, n_hat))
            parts.append(np.array([5000.0 * dev_pi, 5000.0 * dev_rs]))

        return np.concatenate(parts)

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
                )
                stage_ik.add_target(
                    metric_key, target.values, weight=1.0)
                stage_ik.set_variables(stage_vars)
                stage_ik.regularisation = 0.05  # light: stay close
                stage_ik.tube_od = self.tube_od

                stage_res = stage_ik.solve(method='local')
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
            for i, xs in enumerate(starts):
                if progress_cb:
                    progress_cb(f'Multi-start LM: {i+1}/{n_starts}...')
                try:
                    res = least_squares(
                        self._residuals, xs, bounds=(lo, hi),
                        method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
                    )
                    if res.cost < best_cost:
                        best_cost = float(res.cost)
                        best_x = res.x
                except Exception:
                    pass
            x_final = best_x
            cost = best_cost

        else:   # 'local'
            if progress_cb:
                progress_cb('Running local solve (LM)...')
            res_lm = least_squares(
                self._residuals, x0, bounds=(lo, hi),
                method='trf', ftol=1e-10, xtol=1e-10, max_nfev=500,
            )
            x_final = res_lm.x
            cost = float(res_lm.cost)

        hp_final = self.ds.unpack(x_final)
        curves = _evaluate_sweep(hp_final, self.travel, self.side, self.pushrod_body,
                                  self._metric_keys(), self.anti_kwargs,
                                  motion=self.motion)
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

        # Primary target error (first target = the one the user asked for)
        primary = self.targets[0]
        primary_curve = curves.get(primary.metric_key,
                                   np.full(self.n_points, np.nan))
        primary_errors = np.abs(primary_curve - primary.values)
        primary_max_error = float(np.nanmax(primary_errors))

        # ── Collision check ──────────────────────────────────────────────
        collisions = (check_collisions(hp_final, self.tube_od)
                      if self.tube_od else [])

        return {
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
