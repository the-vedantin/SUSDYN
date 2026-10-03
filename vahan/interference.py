"""Geometric interference (clash) detection between suspension/driveline members.

Each member is a capsule: a line segment with a radius (tube OD/2, shaft OD/2,
etc.).  Two capsules clash when the distance between their centre-lines is less
than the sum of their radii (plus a safety margin).  Members that share an
endpoint (a real joint) are skipped — they are meant to touch.

Pure geometry (no Qt), so the GUI's Interference view mode and a headless
regression check both call the same code.  Feed it world-frame points in metres.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar


def seg_seg_distance(p1, q1, p2, q2) -> float:
    """Shortest distance between segments p1q1 and p2q2 (world units)."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def rim_barrel_gap(member, wheel_center, spin_axis, inner_radius_m,
                   half_width_m) -> float:
    """Capsule surface gap to an open finite cylindrical barrel, in metres.

    ``member`` uses the same ``a``, ``b``, ``r`` fields as ``full_members``.
    The barrel is centred at ``wheel_center``, along ``spin_axis``, with the
    supplied inner radius and axial half-width. Its surface includes the two
    circular lip edges, but no end-cap discs or assumed wall thickness.
    Positive means separated; negative means the capsule intersects that
    surface. No design clearance is subtracted: callers apply their margin.

    Points just outside the axial band can still contact a lip. For each
    centreline point the distance is hypot(radial_radius - inner_radius,
    max(abs(axial_position) - half_width, 0)). Minimise this along the entire
    segment, then subtract the capsule radius. All vectors must be finite
    3-vectors, dimensions positive and finite, and the axis nonzero.
    """
    def vector(value, name):
        try:
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite 3-vector') from exc
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f'{name} must be a finite 3-vector')
        return result

    def dimension(value, name):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be positive and finite') from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f'{name} must be positive and finite')
        return result

    a = vector(member['a'], 'member a')
    b = vector(member['b'], 'member b')
    center = vector(wheel_center, 'wheel center')
    axis = vector(spin_axis, 'spin axis')
    radius = dimension(member['r'], 'member radius')
    rim_radius = dimension(inner_radius_m, 'inner radius')
    half_width = dimension(half_width_m, 'half width')
    axis_scale = float(np.max(np.abs(axis)))
    if axis_scale == 0:
        raise ValueError('spin axis must be nonzero')
    axis = axis / axis_scale
    axis = axis / np.linalg.norm(axis)

    q, v = a - center, b - a
    axial_start = float(q @ axis)
    axial_delta = float(v @ axis)
    radial_start = q - axial_start * axis
    radial_delta = v - axial_delta * axis
    aa = float(radial_delta @ radial_delta)
    bb = 2.0 * float(radial_start @ radial_delta)
    cc = float(radial_start @ radial_start)
    if not all(math.isfinite(x) for x in (axial_start, axial_delta, aa, bb, cc)):
        raise ValueError('member coordinates exceed the supported numeric range')

    cuts = [0.0, 1.0]
    if axial_delta != 0:
        cuts.extend(((-half_width - axial_start) / axial_delta,
                     (half_width - axial_start) / axial_delta))
    if aa > 0:
        cuts.append(-bb / (2.0 * aa))
        discriminant = bb * bb - 4.0 * aa * (cc - rim_radius * rim_radius)
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            cuts.extend(((-bb - root) / (2.0 * aa),
                         (-bb + root) / (2.0 * aa)))
    cuts = sorted(set(t for t in cuts if 0.0 <= t <= 1.0))

    def distance_squared(t):
        radial_gap = math.sqrt(max(0.0, aa * t * t + bb * t + cc)) - rim_radius
        axial_gap = max(abs(axial_start + t * axial_delta) - half_width, 0.0)
        return radial_gap * radial_gap + axial_gap * axial_gap

    closest_squared = min(distance_squared(t) for t in cuts)
    if np.any(v != 0):
        # Split at axial-band edges, the radial minimum and shell crossings.
        # Keep endpoints as candidates so lip contacts and crossing zeros are
        # retained even when a bounded minimizer stops just short of an edge.
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo <= 1e-12:
                continue
            result = minimize_scalar(distance_squared, bounds=(lo, hi),
                                     method='bounded', options={'xatol': 1e-10})
            if not result.success or not math.isfinite(float(result.fun)):
                raise ValueError('rim barrel distance minimization failed')
            closest_squared = min(closest_squared, float(result.fun))
    return math.sqrt(max(0.0, closest_squared)) - radius


def _shares_endpoint(a, b, tol):
    for pa in (a['a'], a['b']):
        for pb in (b['a'], b['b']):
            if np.linalg.norm(np.asarray(pa, float) - np.asarray(pb, float)) < tol:
                return True
    return False


def _chassis_trimmed_segment(c, other, tol):
    """FSAE chassis-bay tube ``c`` (vahan.chassis) vs a member ``other``: when
    ``other`` has an endpoint ON one of the tube's bracket pickups (the arm leg
    bolted to that node's bracket, a toe link sharing the LCA rear pickup) the
    bracket zone — node offset + tube radius from that node — is trimmed off
    the tube.  Returns (a, b) of the tube segment to test, or None when the
    whole tube is bracket zone (a designed contact, never a clash)."""
    a = np.asarray(c['a'], float); b = np.asarray(c['b'], float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-12:
        return a, b
    u = (b - a) / L
    t0, t1 = 0.0, L
    trim = float(c.get('bracket_trim', 0.0))
    for pk, end in c.get('joints', ()):
        if any(np.linalg.norm(np.asarray(e, float) - pk) < tol for e in (other['a'], other['b'])):
            if end == 'a':
                t0 = max(t0, trim)
            else:
                t1 = min(t1, L - trim)
    if t1 <= t0:
        return None
    return a + u * t0, a + u * t1


def pair_gap_mm(mi, mj, share_tol_m: float = 0.006):
    """Surface gap (mm) of two capsule members, or None for a DESIGNED contact:
    pairs sharing an endpoint within ``share_tol_m`` (one joint), and chassis
    frame tube vs chassis frame tube (one welded frame).  A member bolted to a
    chassis tube's bracket pickup is tested against the tube minus its bracket
    zone (see _chassis_trimmed_segment)."""
    ci, cj = bool(mi.get('chassis')), bool(mj.get('chassis'))
    if ci and cj:
        return None
    if _shares_endpoint(mi, mj, share_tol_m):
        return None
    ai, bi, aj, bj = mi['a'], mi['b'], mj['a'], mj['b']
    if ci:
        seg = _chassis_trimmed_segment(mi, mj, share_tol_m)
        if seg is None:
            return None
        ai, bi = seg
    elif cj:
        seg = _chassis_trimmed_segment(mj, mi, share_tol_m)
        if seg is None:
            return None
        aj, bj = seg
    return (seg_seg_distance(ai, bi, aj, bj) - mi['r'] - mj['r']) * 1000.0


# Member pairs that are DESIGNED to be near each other (a real joint / pickup),
# so a small "overlap" between them is not a clash.  The pushrod picks up on the
# lower control arm, so its tube runs close to the lower-arm tubes at the tab.
DEFAULT_CONNECTED = frozenset({
    frozenset({'pushrod', 'lower arm rear'}),
    frozenset({'pushrod', 'lower arm front'}),
})


def clashes(members, margin_mm: float = 1.0, share_tol_mm: float = 6.0,
            connected=DEFAULT_CONNECTED) -> list:
    """Find clashing capsule pairs.

    members: list of dicts {'name', 'a', 'b', 'r'} (endpoints in metres, radius
    in metres).  Returns a list of dicts {'a','b','gap_mm'} for every pair whose
    surface gap (centre-line distance - r_a - r_b) is below margin_mm, sorted
    worst (most negative) first.  Pairs sharing an endpoint, or listed in
    ``connected`` (designed joints/pickups), are skipped.
    """
    out = []
    n = len(members)
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = members[i], members[j]
            if frozenset({mi['name'], mj['name']}) in connected:
                continue
            gap = pair_gap_mm(mi, mj, share_tol_mm / 1000.0)
            if gap is None:
                continue
            if gap < margin_mm:
                out.append({'a': mi['name'], 'b': mj['name'],
                            'gap_mm': round(gap, 1)})
    out.sort(key=lambda d: d['gap_mm'])
    return out


# Radii (metres) for the members we know about.  Tubes are a nominal FSAE
# a-arm / link OD; the driveshaft uses the real car-dict value.
_TUBE_R = 0.008           # ~16 mm rod-end tube (assumption)

# ── THE full member set (GUI interference view = packaging validator) ────────
# One spec list shared by gui/main_window.py's interference view mode and
# vahan/packaging.py's validity oracle, so "clash-free" always means the SAME
# members: arms + tie rod + pushrod + ball-joint spheres + coilover + ARB drop
# link + rocker hardware spheres (+ torsion bar + driveshaft added per corner).
# Radii are the user's stated hardware (0.625" tubes, 1" ball joints, 1.5"
# rocker bearing, 0.315"-radius drop-link ball joints); the coilover uses the
# car's own spring_od_mm — what view3d draws it with.
_FULL_TUBE_R = 0.5 * 0.625 * 25.4 / 1000.0   # 0.625" pushrod / arm tube radius
_BJ_R        = 0.0127                        # 1" ball-joint sphere radius
_LINK_R      = 0.006                         # ~12 mm drop link / ARB blade tube
_BRG_R       = 0.5 * 38.1 / 1000.0           # 1.5" rocker pivot bearing OD
_RE_R        = 0.315 * 25.4 / 1000.0         # drop-link ball joint radius


def arb_blade_envelope_radius(blade_w_mm: float, blade_t_mm: float) -> float | None:
    """Circumscribed radius of a declared rectangular ARB blade section."""
    w, t = float(blade_w_mm), float(blade_t_mm)
    if w <= 0.0 or t <= 0.0:
        return None
    return 0.0005 * math.hypot(w, t)


def arb_blade_polyline(pts: dict, pivot, side_offset_m: float = 0.0,
                       start_fraction: float = 0.43,
                       end_fraction: float = 0.94,
                       axial_offset_m: float = 0.0):
    """Return a straight or rigid three-segment ARB arm with fixed endpoints."""
    a = np.asarray(pivot, float); b = np.asarray(pts['arb_arm_end_world'], float)
    off = float(side_offset_m); axial_off = float(axial_offset_m)
    if abs(off) < 1e-12 and abs(axial_off) < 1e-12:
        return [a, b]
    f0, f1 = float(start_fraction), float(end_fraction)
    if not (0.0 < f0 < f1 < 1.0):
        raise ValueError('ARB blade dogleg fractions must satisfy 0 < start < end < 1')
    axis = b-a; axis /= np.linalg.norm(axis)
    # Bar-fixed frame: the torsion axis points from this corner toward vehicle
    # centre.  cross(bar_axis, blade_axis) rotates rigidly with the blade about
    # that axis and is independent of every rocker point.
    bar_axis = np.array([-1.0 if a[0] >= 0.0 else 1.0, 0.0, 0.0])
    route = (1.0 if a[0] >= 0.0 else -1.0) * np.cross(bar_axis, axis)
    lr = np.linalg.norm(route)
    if lr < 1e-9:
        raise ValueError('ARB blade dogleg is undefined for a blade parallel to the torsion axis')
    route /= lr
    dogleg = off*route + axial_off*bar_axis
    return [a, a+f0*(b-a)+dogleg, a+f1*(b-a)+dogleg, b]


def arb_blade_dogleg_params_for(car: dict, corner_or_axle: str):
    """Resolve axle dogleg offset (mm) and span fractions with global fallbacks."""
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    return (float(car.get(f'{axle}_arb_blade_dogleg_side_offset_mm',
                          car.get('arb_blade_dogleg_side_offset_mm', 0.0))),
            float(car.get(f'{axle}_arb_blade_dogleg_axial_offset_mm',
                          car.get('arb_blade_dogleg_axial_offset_mm', 0.0))),
            float(car.get(f'{axle}_arb_blade_dogleg_start_fraction',
                          car.get('arb_blade_dogleg_start_fraction', 0.43))),
            float(car.get(f'{axle}_arb_blade_dogleg_end_fraction',
                          car.get('arb_blade_dogleg_end_fraction', 0.94))))


def arb_member_kwargs(car: dict, corner_or_axle: str,
                      blade_w_mm: float = 0.0, blade_t_mm: float = 0.0):
    """All configured physical ARB-arm arguments consumed by ``full_members``."""
    dog = arb_blade_dogleg_params_for(car, corner_or_axle)
    return dict(arb_blade_w_mm=float(blade_w_mm),
                arb_blade_t_mm=float(blade_t_mm),
                arb_blade_dogleg_side_offset_mm=dog[0],
                arb_blade_dogleg_axial_offset_mm=dog[1],
                arb_blade_dogleg_start_fraction=dog[2],
                arb_blade_dogleg_end_fraction=dog[3])


def full_member_specs(spring_od_mm: float = 63.0, uca_cross_member_od_mm: float = 25.4) -> list:
    """(name, key_a, key_b, radius_m) specs against a corners-draw ``pts`` dict."""
    sr = 0.5 * float(spring_od_mm) / 1000.0
    uca_cross_r = 0.5 * float(uca_cross_member_od_mm) / 1000.0
    return [
        ('upper arm front',  'uca_front',        'uca_outer',         _FULL_TUBE_R),
        ('upper arm rear',   'uca_rear',         'uca_outer',         _FULL_TUBE_R),
        ('lower arm front',  'lca_front',        'lca_outer',         _FULL_TUBE_R),
        ('lower arm rear',   'lca_rear',         'lca_outer',         _FULL_TUBE_R),
        ('tie / toe rod',    'tie_rod_inner',    'tie_rod_outer',     _FULL_TUBE_R),
        # Same assumed one-inch joint-body envelope used at the upright;
        # a thin tie-rod tube alone can clear while its outer joint hits the lip.
        ('tie outer joint',  'tie_rod_outer',    'tie_rod_outer',     _BJ_R),
        ('pushrod',          'pushrod_outer',    'pushrod_inner',     _FULL_TUBE_R),
        # Over-arm pickup uses the documented 1-inch spherical body too;
        # checking only the narrower pushrod tube misses its rim/arm envelope.
        ('pushrod outer joint', 'pushrod_outer', 'pushrod_outer',    _BJ_R),
        ('lower ball joint', 'lca_outer',        'lca_outer',         _BJ_R),
        ('upper ball joint', 'uca_outer',        'uca_outer',         _BJ_R),
        ('coilover',         'rocker_spring_pt', 'spring_chassis_pt', sr),
        ('ARB drop link',    'arb_arm_end_world', 'arb_drop_top',     _LINK_R),
        # Rocker hardware as real volumes (zero-length capsules = spheres).
        ('rocker bearing',   'rocker_pivot',     'rocker_pivot',      _BRG_R),
        ('ARB rod end',      'arb_drop_top',     'arb_drop_top',      _RE_R),
        ('ARB arm-end rod end', 'arb_arm_end_world', 'arb_arm_end_world', _RE_R),
        ('spring rod end',   'rocker_spring_pt', 'rocker_spring_pt',  _RE_R),
        ('pushrod rod end',  'pushrod_inner',    'pushrod_inner',     _RE_R),
        # Chassis tube between the two upper-arm pickups (user 2026-09-21: "there is a
        # member there") — the rear one crosses the driveshaft.  OD from the car dict
        # (uca_cross_member_od_mm, default 25.4 mm = ASSUMED 1 in tube, not a measured part).
        ('UCA chassis cross member', 'uca_front', 'uca_rear', uca_cross_r),
    ]


ROCKER_PLATE_HALF_T = 0.003          # the 3D view's bellcrank plate: 6 mm machined plate (view3d._rocker_half_t)
_PLATE_ATTACH_EXCLUDE = 2.0 * _RE_R  # a member's own rod end sits ON the plate: skip its first 16 mm


def rocker_plate_style_for(car: dict, corner_or_axle: str) -> str:
    """Resolve an axle-specific rocker style, retaining the global fallback."""
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    return str(car.get(f'{axle}_rocker_plate_style',
                       car.get('rocker_plate_style', 'legacy')))


def rocker_pr_full_length_fork_for(car: dict, corner_or_axle: str) -> bool:
    """Resolve the opt-in full-length PR fork, retaining a global fallback."""
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    return bool(car.get(f'{axle}_rocker_pr_full_length_fork',
                        car.get('rocker_pr_full_length_fork', False)))


def rocker_pr_full_length_fork_clear_gap_for(car: dict, corner_or_axle: str) -> float:
    """Resolve full-length PR fork clear gap in metres; default remains 24 mm."""
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    return 0.001 * float(car.get(f'{axle}_rocker_pr_full_length_fork_clear_gap_mm',
                                 car.get('rocker_pr_full_length_fork_clear_gap_mm', 24.0)))


def rocker_pr_full_length_fork_jog_for(car: dict, corner_or_axle: str):
    """Resolve negative-cheek +normal jog (metres) and four path fractions."""
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    def get(name, default):
        return float(car.get(f'{axle}_{name}', car.get(name, default)))
    return (0.001*get('rocker_pr_full_length_fork_negative_cheek_jog_mm', 0.0),
            get('rocker_pr_full_length_fork_jog_start_fraction', 0.68),
            get('rocker_pr_full_length_fork_jog_hold_start_fraction', 0.76),
            get('rocker_pr_full_length_fork_jog_hold_end_fraction', 0.87),
            get('rocker_pr_full_length_fork_jog_end_fraction', 0.95))


def rocker_plate_physical_options_for(car: dict, corner_or_axle: str) -> dict:
    """Resolve the complete axle-specific solid-rocker option set.

    Collision callers must use this rather than a mixture of global defaults
    and hand-resolved fork values.  Otherwise a saved local-clevis/fork/jog
    rocker is tested against a different solid than the renderer displays.
    """
    tag = str(corner_or_axle).strip().lower()
    axle = 'front' if tag.startswith('f') else 'rear' if tag.startswith('r') else tag
    setback_key = f'{axle}_rocker_spring_clevis_setback_mm'
    jog, *fractions = rocker_pr_full_length_fork_jog_for(car, corner_or_axle)
    return {
        'style': rocker_plate_style_for(car, corner_or_axle),
        'clear_gap_m': 0.001 * float(car.get('rocker_plate_clear_gap_mm', 24.0)),
        'spring_clevis_clear_gap_m': 0.001 * float(car.get(
            'rocker_spring_clevis_clear_gap_mm',
            float(car.get('spring_od_mm', 63.0)) + 6.0)),
        'spring_clevis_setback_m': (0.001 * float(car[setback_key])
                                    if setback_key in car else None),
        'main_arm_width_m': 0.001 * float(car.get('rocker_plate_main_arm_width_mm', 38.1)),
        'arb_arm_width_m': 0.001 * float(car.get('rocker_plate_arb_arm_width_mm', 25.4)),
        'pr_full_length_fork': rocker_pr_full_length_fork_for(car, corner_or_axle),
        'pr_full_length_fork_clear_gap_m': rocker_pr_full_length_fork_clear_gap_for(
            car, corner_or_axle),
        'pr_full_length_fork_negative_cheek_jog_m': jog,
        'pr_full_length_fork_jog_fractions': tuple(fractions),
    }


def rocker_double_shear_polys(pts: dict, clear_gap_m: float = 0.024,
                              plate_t_m: float = 0.006,
                              spring_clevis_clear_gap_m: float = 0.069,
                              spring_clevis_setback_m: float = None,
                              main_arm_width_m: float = 0.0381,
                              arb_arm_width_m: float = 0.0254,
                              pivot_boss_radius_m: float = 0.0254,
                              main_boss_radius_m: float = 0.01905,
                              arb_boss_radius_m: float = 0.0127):
    """Plate-arm and boss outlines for a symmetric double-shear rocker.

    The linkage joint centres remain in the kinematic force plane.  Two plates
    sit symmetrically outside a real clear gap, and each plate has three
    load-bearing arms from the pivot to the pushrod, spring and ARB pickups.
    Returned polygons are already offset to the two plate centre planes and
    are shared by rendering and collision checks.
    """
    keys = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt', 'arb_drop_top')
    if not all(k in pts and pts[k] is not None and
               np.all(np.isfinite(np.asarray(pts[k], float))) for k in keys):
        return []
    pivot = np.asarray(pts['rocker_pivot'], float)
    push = np.asarray(pts['pushrod_inner'], float)
    spring = np.asarray(pts['rocker_spring_pt'], float)
    n = np.cross(push - pivot, spring - pivot); ln = np.linalg.norm(n)
    if ln < 1e-12:
        return []
    n /= ln
    # A cross product is a pseudovector: reflecting the right corner across X
    # reverses its raw sign.  Restore a true mirrored plate normal so named
    # +/- cheeks and an asymmetric formed-cheek route select matching physical
    # faces on both corners.
    if pivot[0] < 0.0:
        n = -n
    offset = 0.5 * (float(clear_gap_m) + float(plate_t_m))
    out = []
    arm_defs = []
    for end, width in ((push, main_arm_width_m), (spring, main_arm_width_m),
                       (np.asarray(pts['arb_drop_top'], float), arb_arm_width_m)):
        # Project the configured pickup to the force plane; Rule 04 normally
        # makes this correction numerical noise, while legacy standoffs do not
        # silently distort the structural arm.
        end = end - float((end - pivot) @ n) * n
        axis = end - pivot; L = np.linalg.norm(axis)
        if L < 1e-6:
            continue
        axis /= L
        side = np.cross(n, axis); side /= np.linalg.norm(side)
        h = 0.5 * float(width)
        base = np.array([pivot + h*side, end + h*side,
                         end - h*side, pivot - h*side], float)
        arm_defs.append(base)
    # Round bosses provide real ligament beyond every bolt centre.  The
    # 16-gons circumscribe the declared circular radii, so no physical boss
    # material is omitted by the shared render/collision approximation.
    ref = push - pivot; ref -= float(ref @ n)*n; ref /= np.linalg.norm(ref)
    tangent = np.cross(n, ref); tangent /= np.linalg.norm(tangent)
    boss_defs = []
    for center, radius in ((pivot, pivot_boss_radius_m), (push, main_boss_radius_m),
                           (spring, main_boss_radius_m),
                           (np.asarray(pts['arb_drop_top'], float), arb_boss_radius_m)):
        center = center - float((center - pivot) @ n)*n
        vertex_radius = float(radius) / np.cos(np.pi / 16.0)
        boss_defs.append(np.array([
            center + vertex_radius*(np.cos(a)*ref + np.sin(a)*tangent)
            for a in np.linspace(0.0, 2.0*np.pi, 16, endpoint=False)], float))
    for sign in (-1.0, 1.0):
        shift = sign * offset * n
        out.extend([poly + shift for poly in arm_defs])
        out.extend([poly + shift for poly in boss_defs])
    return out


def rocker_local_clevis_polys(pts: dict, clear_gap_m: float = 0.024,
                              plate_t_m: float = 0.006,
                              spring_clevis_clear_gap_m: float = 0.069,
                              spring_clevis_setback_m: float = None,
                              main_arm_width_m: float = 0.0381,
                              arb_arm_width_m: float = 0.0254,
                              pivot_boss_radius_m: float = 0.0254,
                              main_boss_radius_m: float = 0.01905,
                              arb_boss_radius_m: float = 0.0127,
                              scallop_radius_m: float = 0.012,
                              pr_full_length_fork: bool = False,
                              pr_full_length_fork_clear_gap_m: float = None,
                              pr_full_length_fork_negative_cheek_jog_m: float = 0.0,
                              pr_full_length_fork_jog_fractions=(0.68, 0.76, 0.87, 0.95)):
    """Convex solid outlines for a centre plate with a local pushrod clevis.

    The main plate stops before the pushrod joint instead of filling the
    four-point rocker quad.  Two short cheek plates surround the centre-plane
    rod end, and a transverse bridge overlaps both cheeks and the main arm so
    the returned solids form one connected rocker.  The ARB has a separate,
    bridged one-sided tab outside its centre-plane joint; pieces are returned individually because
    render and collision code deliberately share this convex decomposition.
    The spring clevis clear gap defaults to 69 mm: the declared 63 mm spring
    envelope plus the required 3 mm clearance on both sides.  Existing rod-end
    spheres define the ideal mating joints; no undeclared pin diameter or pin
    strength is implied.
    No hardpoint is moved or projected out of the static force plane.
    """
    keys = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt', 'arb_drop_top')
    if not all(k in pts and pts[k] is not None and
               np.all(np.isfinite(np.asarray(pts[k], float))) for k in keys):
        raise ValueError('local_clevis_arms requires finite pivot, pushrod, spring and ARB points')
    pivot, push, spring, arb = (np.asarray(pts[k], float) for k in keys)
    n = np.cross(push - pivot, spring - pivot); ln = np.linalg.norm(n)
    if ln < 1e-12:
        raise ValueError('local_clevis_arms static force plane is degenerate')
    n /= ln
    if pivot[0] < 0.0:
        n = -n

    def arm(a, b, width):
        axis = b - a; length = np.linalg.norm(axis)
        if length < 1e-9:
            return None
        axis /= length
        side = np.cross(n, axis); side /= np.linalg.norm(side)
        h = 0.5 * float(width)
        return np.array([a + h*side, b + h*side,
                         b - h*side, a - h*side], float)

    # Keep the centre plate outside a conservative circular pushrod-joint
    # void.  The bridge is another scallop radius inboard, leaving a real
    # straight ligament between its edge and the void rather than a tangent.
    push_axis = push - pivot; push_len = np.linalg.norm(push_axis)
    if push_len <= 2.0 * float(scallop_radius_m) + 1e-6:
        raise ValueError('local_clevis_arms pushrod lever is too short for its scallop and bridge')
    u = push_axis / push_len
    base = push - 2.0 * float(scallop_radius_m) * u
    arb_axis = arb - pivot; arb_len = np.linalg.norm(arb_axis)
    if arb_len <= 2.0 * float(scallop_radius_m) + 1e-6:
        raise ValueError('local_clevis_arms ARB lever is too short for its scallop and bridge')
    arb_u = arb_axis / arb_len
    arb_base = arb - 2.0 * float(scallop_radius_m) * arb_u
    spring_axis = spring - pivot; spring_len = np.linalg.norm(spring_axis)
    spring_offset = 0.5 * (float(spring_clevis_clear_gap_m) + float(plate_t_m))
    spring_setback = (spring_offset if spring_clevis_setback_m is None
                      else float(spring_clevis_setback_m))
    if spring_len <= spring_setback + 1e-6:
        raise ValueError('local_clevis_arms spring lever is too short for its scallop and bridge')
    spring_u = spring_axis / spring_len
    spring_base = spring - spring_setback * spring_u

    # Rectangular scallop in the pushrod centre arm around the fixed ARB eye.
    # The required 2-D edge distance is eye radius + plate half-thickness +
    # 3 mm clearance.  Split into convex pieces so render/collision share the
    # exact subtraction without concave fan triangulation.
    push_side = np.cross(n, u); push_side /= np.linalg.norm(push_side)
    arb_rel = arb - pivot
    ax = float(arb_rel @ u); sy = float(arb_rel @ push_side)
    half_w = 0.5 * float(main_arm_width_m)
    required = _RE_R + 0.5*float(plate_t_m) + 0.003
    original_edge = np.sign(sy or 1.0) * half_w
    lateral = abs(sy - original_edge)
    push_parts = []
    if 0.0 < ax < np.linalg.norm(base-pivot) and lateral < required:
        span = math.sqrt(max(required*required - lateral*lateral, 0.0))
        x0 = max(0.0, ax-span); x1 = min(np.linalg.norm(base-pivot), ax+span)
        relieved_edge = sy - np.sign(sy or 1.0)*required
        other_edge = -np.sign(sy or 1.0)*half_w
        def rect(xa, xb, ea, eb):
            return np.array([pivot+xa*u+ea*push_side, pivot+xb*u+ea*push_side,
                             pivot+xb*u+eb*push_side, pivot+xa*u+eb*push_side], float)
        if x0 > 1e-9: push_parts.append(rect(0.0, x0, original_edge, other_edge))
        push_parts.append(rect(x0, x1, relieved_edge, other_edge))
        if x1 < np.linalg.norm(base-pivot)-1e-9:
            push_parts.append(rect(x1, np.linalg.norm(base-pivot), original_edge, other_edge))
    else:
        push_parts.append(arm(pivot, base, main_arm_width_m))
    centre = ([] if pr_full_length_fork else push_parts) + [arm(pivot, spring_base, main_arm_width_m),
                           arm(pivot, arb_base, arb_arm_width_m)]
    out = [p for p in centre if p is not None]

    # Round centre-plate bosses.  Deliberately omit a pushrod boss: that is the
    # scalloped void occupied by the local clevis and its centre-plane joint.
    ref = u
    tangent = np.cross(n, ref); tangent /= np.linalg.norm(tangent)
    def boss(center, radius, normal=n, tangent_axis=tangent):
        vr = float(radius) / np.cos(np.pi / 16.0)
        radial = np.cross(normal, tangent_axis)
        return np.array([center + vr*(np.cos(a)*tangent_axis + np.sin(a)*radial)
                         for a in np.linspace(0.0, 2.0*np.pi, 16, endpoint=False)], float)
    out.append(boss(pivot, pivot_boss_radius_m))

    offset = 0.5 * (float(clear_gap_m) + float(plate_t_m))
    fork_gap = (float(clear_gap_m) if pr_full_length_fork_clear_gap_m is None
                else float(pr_full_length_fork_clear_gap_m))
    fork_offset = 0.5 * (fork_gap + float(plate_t_m))
    pr_offset = fork_offset if pr_full_length_fork else offset
    for sign in (-1.0, 1.0):
        shift = sign * pr_offset * n
        if pr_full_length_fork and sign < 0 and abs(float(pr_full_length_fork_negative_cheek_jog_m)) > 1e-12:
            f0, f1, f2, f3 = map(float, pr_full_length_fork_jog_fractions)
            if not (0.0 < f0 < f1 <= f2 < f3 < 1.0):
                raise ValueError('local_clevis_arms full-length PR fork jog fractions are invalid')
            jog = float(pr_full_length_fork_negative_cheek_jog_m)
            fs = (0.0, f0, f1, f2, f3, 1.0); js = (0.0, 0.0, jog, jog, 0.0, 0.0)
            axis = push-pivot
            for fa, fb, ja, jb in zip(fs[:-1], fs[1:], js[:-1], js[1:]):
                out.append(arm(pivot+fa*axis+shift+ja*n,
                               pivot+fb*axis+shift+jb*n, main_arm_width_m))
        else:
            out.append(arm(pivot if pr_full_length_fork else base,
                           push, main_arm_width_m) + shift)
        out.append(boss(push + shift, main_boss_radius_m))

    # A transverse 6 mm bridge at ``base`` spans the outer cheek faces.  Its
    # polygon lies in the (normal, in-plane-side) plane; the shared prism
    # extrusion supplies 6 mm along the pushrod-arm direction.  It overlaps
    # the centre arm and both cheeks by one plate thickness.
    side = np.cross(n, u); side /= np.linalg.norm(side)
    bridge_half_span = pr_offset + 0.5 * float(plate_t_m)
    bridge_half_width = 0.5 * float(main_arm_width_m)
    if not pr_full_length_fork:
        out.append(np.array([base - bridge_half_span*n + bridge_half_width*side,
                             base + bridge_half_span*n + bridge_half_width*side,
                             base + bridge_half_span*n - bridge_half_width*side,
                             base - bridge_half_span*n - bridge_half_width*side], float))

    if pr_full_length_fork:
        # The two 6 mm cheeks continue to the pivot, replacing the centre-plane
        # PR stem with a real 24 mm-clear corridor.  Side pivot bosses and a
        # short transverse bridge/sleeve connect both load paths to the central
        # pivot boss.  The bridge is local to the pivot; no material crosses the
        # drop-link corridor farther out on the PR branch.
        for sign in (-1.0, 1.0):
            shift = sign * pr_offset * n
            out.append(boss(pivot + shift, pivot_boss_radius_m))
        out.append(np.array([
            pivot - bridge_half_span*n + pivot_boss_radius_m*side,
            pivot + bridge_half_span*n + pivot_boss_radius_m*side,
            pivot + bridge_half_span*n - pivot_boss_radius_m*side,
            pivot - bridge_half_span*n - pivot_boss_radius_m*side], float))

    # The damper eye gets the same compact two-sided local clevis.  Its bridge
    # joins both cheeks to the centre spring arm.  The conservative full-span
    # 63 mm spring envelope remains collision checked; no axial neck is assumed.
    spring_side = np.cross(n, spring_u); spring_side /= np.linalg.norm(spring_side)
    spring_bridge_half_span = spring_offset + 0.5 * float(plate_t_m)
    spring_cheek = arm(spring_base, spring, main_arm_width_m)
    for sign in (-1.0, 1.0):
        shift = sign * spring_offset * n
        out.append(spring_cheek + shift)
        out.append(boss(spring + shift, main_boss_radius_m))
    out.append(np.array([
        spring_base - spring_bridge_half_span*n + bridge_half_width*spring_side,
        spring_base + spring_bridge_half_span*n + bridge_half_width*spring_side,
        spring_base + spring_bridge_half_span*n - bridge_half_width*spring_side,
        spring_base - spring_bridge_half_span*n - bridge_half_width*spring_side], float))

    # The ARB pickup uses its own one-sided tab rather than filling the centre
    # plate through the drop-link joint.  It uses the same 12 mm inner-face
    # offset as the pushrod clevis and a transverse bridge back to the centre
    # arm, so it is both collision-visible and structurally connected.
    arb_side = np.cross(n, arb_u); arb_side /= np.linalg.norm(arb_side)
    arb_shift = offset * n
    out.append(arm(arb_base, arb, arb_arm_width_m) + arb_shift)
    out.append(boss(arb + arb_shift, arb_boss_radius_m))
    arb_bridge_half_width = 0.5 * float(arb_arm_width_m)
    out.append(np.array([
        arb_base - 0.5*plate_t_m*n + arb_bridge_half_width*arb_side,
        arb_base + (offset + 0.5*plate_t_m)*n + arb_bridge_half_width*arb_side,
        arb_base + (offset + 0.5*plate_t_m)*n - arb_bridge_half_width*arb_side,
        arb_base - 0.5*plate_t_m*n - arb_bridge_half_width*arb_side], float))
    return out


def rocker_plate_poly(pts: dict):
    """The rocker PLATE polygon exactly as the 3D view draws it (view3d rk4 / rk3 order,
    fan-triangulated from the pivot): pivot, ARB drop top, pushrod attach, spring attach.
    Returns an (n, 3) array or None when the corner has no plate (direct damper / decoupled)."""
    keys4 = ('rocker_pivot', 'arb_drop_top', 'pushrod_inner', 'rocker_spring_pt')
    keys3 = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')
    def have(keys):
        return all(k in pts and pts[k] is not None and np.all(np.isfinite(np.asarray(pts[k], float))) for k in keys)
    if not have(keys3):
        return None
    P = {k: np.asarray(pts[k], float) for k in keys4 if k in pts}
    if np.linalg.norm(P['rocker_pivot'] - P['pushrod_inner']) < 1e-4:       # collapsed rocker (direct damper)
        return None
    if np.linalg.norm(P['pushrod_inner'] - P['rocker_spring_pt']) < 1e-4:   # degenerate plate
        return None
    if not have(keys4):
        return np.array([P[k] for k in keys3], float)
    # The plate is FLAT through pivot / pushrod attach / spring attach.  A drop top that
    # sits off that plane (a declared rod-end standoff, Rule 04 doc) is the STUD, not the
    # plate: its tab is the drop top projected onto the plate plane.
    A, B, C = P['rocker_pivot'], P['pushrod_inner'], P['rocker_spring_pt']
    n = np.cross(B - A, C - A); ln = np.linalg.norm(n)
    dt = P['arb_drop_top']
    if ln > 1e-12:
        n = n / ln; dt = dt - float((dt - A) @ n) * n
    return np.array([A, dt, B, C], float)


def segment_to_plate_gap(a, b, r, poly, half_t: float = ROCKER_PLATE_HALF_T,
                         skip_from_a: float = 0.0, samples: int = 41) -> float:
    """Clearance (m, negative = interference) between the capsule a-b (radius r) and the
    solid PLATE = polygon `poly` extruded +/-half_t along its normal (the same prism the
    3D view builds).  The polygon is fan-triangulated from its first vertex, like
    view3d.build_prism.  Sampled along the segment; `skip_from_a` metres at the a end are
    ignored (a rod end that is bolted to the plate by design)."""
    a = np.asarray(a, float); b = np.asarray(b, float); poly = np.asarray(poly, float)
    n = np.cross(poly[1] - poly[0], poly[2] - poly[0]); ln = np.linalg.norm(n)
    if ln < 1e-12:
        return float('inf')
    n = n / ln; c = poly.mean(0)
    L = np.linalg.norm(b - a)
    if L < 1e-9:
        if skip_from_a > 0.0:
            return float('inf')
        ts = np.array([0.0])
    else:
        t0 = min(max(skip_from_a / L, 0.0), 1.0)
        ts = np.linspace(t0, 1.0, samples)
    P = a[None, :] + (b - a)[None, :] * ts[:, None]
    dn = (P - c) @ n                                    # signed normal distance
    Q = P - dn[:, None] * n[None, :]                    # projections onto the plate plane
    inside = np.zeros(len(P), bool)
    for i in range(1, len(poly) - 1):                   # fan triangles (0, i, i+1)
        A, B, C = poly[0], poly[i], poly[i + 1]
        v0, v1 = C - A, B - A; v2 = Q - A[None, :]
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d02, d12 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-18:
            continue
        u = (d11 * d02 - d01 * d12) / den; v = (d00 * d12 - d01 * d02) / den
        inside |= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    gap = np.full(len(P), np.inf)
    gap[inside] = np.abs(dn[inside]) - half_t
    # outside the outline: nearest plate EDGE treated as a capsule of radius half_t
    if not inside.all():
        Po = P[~inside]; best = np.full(len(Po), np.inf)
        for i in range(len(poly)):
            E0, E1 = poly[i], poly[(i + 1) % len(poly)]
            e = E1 - E0; ee = e @ e
            tt = np.clip(((Po - E0[None, :]) @ e) / ee, 0.0, 1.0) if ee > 1e-18 else np.zeros(len(Po))
            d = np.linalg.norm(Po - (E0[None, :] + tt[:, None] * e[None, :]), axis=1) - half_t
            best = np.minimum(best, d)
        gap[~inside] = best
    return float(gap.min() - r)


def rocker_plate_gaps(pts: dict, members: list, half_t: float = ROCKER_PLATE_HALF_T,
                      label: str = '', style: str = 'legacy',
                      clear_gap_m: float = 0.024,
                      spring_clevis_clear_gap_m: float = 0.069,
                      spring_clevis_setback_m: float = None,
                      main_arm_width_m: float = 0.0381,
                      arb_arm_width_m: float = 0.0254,
                      pr_full_length_fork: bool = False,
                      pr_full_length_fork_clear_gap_m: float = None,
                      pr_full_length_fork_negative_cheek_jog_m: float = 0.0,
                      pr_full_length_fork_jog_fractions=(0.68, 0.76, 0.87, 0.95)) -> list:
    """[(member_name, gap_m)] of every capsule member against this corner's rocker plate.
    Members that attach to the plate (ARB drop link at the drop top, pushrod at
    pushrod_inner, coilover at rocker_spring_pt, and the rod ends / bearing that ARE the
    plate hardware) have their attached end excluded or are skipped."""
    if style == 'double_shear_arms':
        polys = rocker_double_shear_polys(
            pts, clear_gap_m=clear_gap_m, plate_t_m=2.0*half_t,
            main_arm_width_m=main_arm_width_m,
            arb_arm_width_m=arb_arm_width_m)
    elif style == 'local_clevis_arms':
        polys = rocker_local_clevis_polys(
            pts, clear_gap_m=clear_gap_m, plate_t_m=2.0*half_t,
            spring_clevis_clear_gap_m=spring_clevis_clear_gap_m,
            spring_clevis_setback_m=spring_clevis_setback_m,
            main_arm_width_m=main_arm_width_m,
            arb_arm_width_m=arb_arm_width_m,
            pr_full_length_fork=pr_full_length_fork,
            pr_full_length_fork_clear_gap_m=pr_full_length_fork_clear_gap_m,
            pr_full_length_fork_negative_cheek_jog_m=pr_full_length_fork_negative_cheek_jog_m,
            pr_full_length_fork_jog_fractions=pr_full_length_fork_jog_fractions)
    else:
        poly = rocker_plate_poly(pts)
        polys = [] if poly is None else [poly]
    if not polys:
        return []
    legacy_skip_names = ('rocker bearing', 'ARB rod end', 'spring rod end', 'pushrod rod end')
    out = []
    for m in members:
        nm = m['name']
        if style in ('double_shear_arms', 'local_clevis_arms'):
            if nm == 'rocker bearing':
                continue
        elif any(k in nm for k in legacy_skip_names) or 'torsion' in nm:
            continue
        a, b = np.asarray(m['a'], float), np.asarray(m['b'], float)
        skip_a = skip_b = 0.0
        if style in ('double_shear_arms', 'local_clevis_arms'):
            # The centre-plane links and joint spheres occupy the open clevis
            # gap.  No axial attachment exclusion is needed or allowed.
            pass
        elif 'ARB drop link' in nm:          # a = arm end, b = drop top (on the plate / its stud)
            skip_b = _PLATE_ATTACH_EXCLUDE
        elif nm.endswith('pushrod'):       # a = outer, b = pushrod_inner (on the plate)
            skip_b = _PLATE_ATTACH_EXCLUDE
        elif nm.endswith('coilover'):      # the coilover EYE is on the plate; the 63 mm spring body
            continue                       # around it is a detail-design item, not a capsule clash
        if skip_b > 0:
            g = min(segment_to_plate_gap(b, a, m['r'], poly, half_t,
                                         skip_from_a=skip_b) for poly in polys)
        else:
            g = min(segment_to_plate_gap(a, b, m['r'], poly, half_t,
                                         skip_from_a=skip_a) for poly in polys)
        out.append(((label + ' ' if label else '') + nm, float(g)))
    return out


def full_members(pts: dict, car: dict, arb_pivot=None, arb_od_mm: float = None,
                 driveshaft_seg=None, arb_blade_w_mm: float = 0.0,
                 arb_blade_t_mm: float = 0.0,
                 arb_blade_dogleg_side_offset_mm: float = 0.0,
                 arb_blade_dogleg_axial_offset_mm: float = 0.0,
                 arb_blade_dogleg_start_fraction: float = 0.43,
                 arb_blade_dogleg_end_fraction: float = 0.94) -> list:
    """Build the FULL capsule list for one corner's ``pts`` dict (world metres).

    pts: a corners-draw points dict (solved state points + live arb_drop_top /
    arb_arm_end_world).  Missing / non-finite points are skipped, never fatal.
    arb_pivot + arb_od_mm add the chassis-fixed ARB torsion tube (spans +x to
    -x through arb_pivot); driveshaft_seg=(inner, outer) metres adds the
    half-shaft with the car-dict OD.

    FSAE chassis bays (vahan.chassis, car['fsae_chassis'] enabled): the
    pickup-to-pickup 'UCA chassis cross member' stand-in is replaced by the
    corner's chassis-bay tubes (nodes 1.5 in inboard of each pickup along its
    arm leg, 1 in tubes, one diagonal).  Absent / disabled = unchanged.
    """
    from .chassis import settings as _chassis_settings, bay_members as _bay_members
    fsae = _chassis_settings(car)
    members = []
    for nm, ka, kb, rr in full_member_specs(car.get('spring_od_mm', 63.0),
                                            car.get('uca_cross_member_od_mm', 25.4)):
        if fsae is not None and nm == 'UCA chassis cross member':
            continue
        a, b = pts.get(ka), pts.get(kb)
        if a is None or b is None:
            continue
        a = np.asarray(a, float); b = np.asarray(b, float)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        members.append({'name': nm, 'a': a, 'b': b, 'r': rr})
    if fsae is not None:
        members.extend(_bay_members(pts, fsae))
    if arb_pivot is not None and arb_od_mm is not None:
        pv = np.asarray(arb_pivot, float)
        if np.all(np.isfinite(pv)):
            b2 = pv.copy(); b2[0] = -pv[0]
            members.append({'name': 'ARB torsion bar', 'a': pv, 'b': b2,
                            'r': 0.5 * float(arb_od_mm) / 1000.0})
        declared_blade_r = arb_blade_envelope_radius(arb_blade_w_mm, arb_blade_t_mm)
        blade_r = declared_blade_r or _FULL_TUBE_R
        ae = pts.get('arb_arm_end_world')
        if ae is not None and np.all(np.isfinite(ae)):
            bp = pv.copy()
            if float(np.asarray(ae)[0]) * float(bp[0]) < 0.0:
                bp[0] = -bp[0]
            blade_pts = arb_blade_polyline(
                pts, bp, 0.001*float(arb_blade_dogleg_side_offset_mm),
                arb_blade_dogleg_start_fraction, arb_blade_dogleg_end_fraction,
                0.001*float(arb_blade_dogleg_axial_offset_mm))
            for p0, p1 in zip(blade_pts[:-1], blade_pts[1:]):
                members.append({'name': 'ARB blade', 'a': p0, 'b': p1, 'r': blade_r,
                                'section_model': ('declared rectangular circumscribed'
                                                  if declared_blade_r is not None
                                                  else 'project-default circular 0.625 in OD')})
    if driveshaft_seg is not None:
        ds_r = 0.5 * float(car.get('driveshaft_dia_mm', 25.4)) / 1000.0
        members.append({'name': 'driveshaft',
                        'a': np.asarray(driveshaft_seg[0], float),
                        'b': np.asarray(driveshaft_seg[1], float), 'r': ds_r})
    return members


def cross_corner_clashes(members_by_corner: dict, margin_mm: float = 3.0) -> list:
    """Check different corners, including opposing coilovers near centreline.

    Each axle's torsion bar is emitted by both corner builders; retain one
    copy of identical bar geometry, then test it against the opposite corner.
    Same-corner joint exemptions remain the responsibility of ``clashes``.
    """
    groups = {}
    bars = []
    for corner, members in members_by_corner.items():
        groups[corner] = []
        for member in members:
            if member['name'] == 'ARB torsion bar':
                if any(corner[0] == old_corner[0] and member['r'] == old['r'] and
                       ((np.allclose(member['a'], old['a'], rtol=0, atol=1e-9)
                         and np.allclose(member['b'], old['b'], rtol=0, atol=1e-9))
                        or (np.allclose(member['a'], old['b'], rtol=0, atol=1e-9)
                            and np.allclose(member['b'], old['a'], rtol=0, atol=1e-9)))
                       for old_corner, old in bars):
                    continue
                bars.append((corner, member))
            groups[corner].append(member)
    result = []
    labels = list(groups)
    for i, left in enumerate(labels):
        for right in labels[i+1:]:
            for a in groups[left]:
                for b in groups[right]:
                    # Opposite-corner blade and the shared torsion tube are
                    # one ARB assembly and meet at the blade root.  Preserve
                    # that exact modeled joint across the corner grouping.
                    if ({a['name'], b['name']} == {'ARB blade', 'ARB torsion bar'}
                            and _shares_endpoint(a, b, 1e-6)):
                        continue
                    # Chassis-bay tubes of different corners are one frame.
                    if a.get('chassis') and b.get('chassis'):
                        continue
                    gap = 1000 * (seg_seg_distance(a['a'], a['b'], b['a'], b['b'])
                                  - a['r'] - b['r'])
                    # Proximity is not a joint between distinct corners.
                    # In particular, nearly coincident coilover ends must
                    # not inherit the same-corner shared-endpoint exemption.
                    if gap < margin_mm:
                        result.append(dict(a=a['name'], b=b['name'], gap_mm=round(gap, 1),
                                           a_corner=left, b_corner=right))
    return result


def connected_for(label: str):
    """Designed-joint exemptions for ``clashes()`` on this corner.

    Rear lower A-arm + toe link are ONE fabricated welded part (user DFM,
    2026-08-05) so their mutual interference is physically impossible; the
    front tie rod is the steering rack (a separate part) and stays checked.
    """
    # Front pushrod mounts to the UCA, so no LCA connection exemption applies.
    conn = frozenset() if label in ('FL', 'FR') else DEFAULT_CONNECTED
    if label in ('RL', 'RR'):
        conn = conn | {frozenset({'lower arm front', 'tie / toe rod'}),
                       frozenset({'lower arm rear', 'tie / toe rod'})}
    return conn


def corner_members(state, car, driveshaft_seg=None) -> list:
    """Build the capsule list for one solved corner.

    state: solved corner state (named world points).  car: the car dict (for the
    driveshaft OD).  driveshaft_seg: optional (inner, outer) tuple in metres for
    this corner's half-shaft; if given it is added as a fat capsule.
    """
    def P(name):
        return np.asarray(getattr(state, name), float)
    segs = [
        ('upper arm front', P('uca_front'), P('uca_outer'), _TUBE_R),
        ('upper arm rear',  P('uca_rear'),  P('uca_outer'), _TUBE_R),
        ('lower arm front', P('lca_front'), P('lca_outer'), _TUBE_R),
        ('lower arm rear',  P('lca_rear'),  P('lca_outer'), _TUBE_R),
        ('tie / toe rod',   P('tr_inner'),  P('tr_outer'),  _TUBE_R),
        ('pushrod',         P('pushrod_outer'), P('pushrod_inner'), _TUBE_R),
    ]
    members = [{'name': n, 'a': a, 'b': b, 'r': r} for n, a, b, r in segs]
    if driveshaft_seg is not None:
        ds_r = 0.5 * float(car.get('driveshaft_dia_mm', 25.4)) / 1000.0
        members.append({'name': 'driveshaft', 'a': np.asarray(driveshaft_seg[0], float),
                        'b': np.asarray(driveshaft_seg[1], float), 'r': ds_r})
    return members


def rack_members(car: dict, tr_inner_left, tr_inner_right, prefix: str = 'front steering rack') -> list:
    """Steering-rack capsules between the two live inner tie-rod joints (metres).

    Hardware is a HOUSING (OD ``rack_housing_od_mm``, default 38.1 = 1.5 in) that is
    ``rack_housing_length_mm`` long, centred between the joints, plus the rack BAR
    (``rack_bar_dia_mm``) protruding from the housing ends to the joints.  When the
    housing length is absent or not shorter than the joint spacing the whole span is
    the housing — the conservative proxy every audit used before 2026-09-10.  Both new
    keys are hardware ASSUMPTIONS until confirmed against the actual rack."""
    a = np.asarray(tr_inner_left, float); b = np.asarray(tr_inner_right, float)
    r_h = float(car.get('rack_housing_od_mm', 38.1)) / 2000.0
    span = float(np.linalg.norm(b - a))
    L = car.get('rack_housing_length_mm')
    L = float(L) / 1000.0 if L is not None else span
    if L >= span - 1e-6 or span <= 1e-9:
        return [{'name': f'{prefix} housing (1.5 in OD)', 'a': a, 'b': b, 'r': r_h}]
    # the housing is CHASSIS-FIXED on the car centreline (X = 0); only the bar slides with
    # rack travel, so the joints' midpoint is used for Y/Z and the bar stubs absorb the shift
    u = (b - a) / span; mid = 0.5 * (a + b); mid = np.array([0.0, mid[1], mid[2]])
    h0 = mid - u * (0.5 * L); h1 = mid + u * (0.5 * L)
    r_b = float(car.get('rack_bar_dia_mm', 25.4)) / 2000.0
    return [{'name': f'{prefix} housing (1.5 in OD)', 'a': h0, 'b': h1, 'r': r_h},
            {'name': f'{prefix} bar left', 'a': a, 'b': h0, 'r': r_b},
            {'name': f'{prefix} bar right', 'a': h1, 'b': b, 'r': r_b}]
