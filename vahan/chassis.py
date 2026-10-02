"""FSAE chassis-bay obstruction model (user request 2026-09-23).

The frame around each corner's inboard pickups, as clash bodies:

* NODES.  Every control-arm inboard pickup P (uca_front, uca_rear, lca_front,
  lca_rear) has a chassis frame node = the arm LEG line (outer ball joint O ->
  pickup P) continued ``node_offset_mm`` (default 38.1 mm = 1.5 in) past the
  pickup, toward the car centre:  node = P + offset * unit(P - O).  The node is
  the tube centre.  O is the DESIGN (static) ball joint, so the nodes are
  chassis-fixed: they do not move with wheel travel or steering.
* TUBES (``tube_od_mm``, default 25.4 mm = 1 in) per corner bay:
  UCA front node - UCA rear node, LCA front node - LCA rear node,
  UCA front node - LCA front node, UCA rear node - LCA rear node, plus ONE
  diagonal: UCA front - LCA rear ('ucaf_lcar') or UCA rear - LCA front
  ('ucar_lcaf').  'auto' = the diagonal with the larger worst-case clearance
  to every other member over full travel x steering (resolved by
  vahan.packaging.fsae_resolve_diagonals with the full-state audit).
  Left and right are mirror images (the right corner's hardpoints are the
  mirrored left ones).
* TRANSVERSE TUBES (user/chassis 2026-09-23): in the REAR, one tube per
  inboard control-arm point runs ACROSS the car from the left node to the
  mirrored right node (UCA front, UCA rear, LCA front, LCA rear), same OD and
  node rule.  Each corner carries its own HALF (node -> the car centreline
  X = 0) so the per-corner member sets stay self-contained; the two halves meet
  at X = 0 and their union is exactly the full tube.  Settings
  'transverse_rear' (default True) / 'transverse_front' (default False).
* CHASSIS-FIXED OBSTRUCTIONS for the tubes (fixed_obstructions): the sprocket
  disc derived from the imported "Diff and sprocket" STEP mesh (the X slab
  whose radial extent about the diff axis is the largest), the differential
  housing from the car dict (tripod-to-tripod span x tripod OD) and the
  revolved envelope of the rest of that mesh.  Tubes and these parts are all
  chassis-fixed, so the audit checks them once, not per travel state.
* DESIGNED CONTACTS.  A tube is welded at its node; the arm's inboard bearing
  bolts to the bracket that spans pickup -> node.  So a member with an
  endpoint ON one of the tube's bracket pickups (the arm leg, a rear toe link
  sharing the LCA rear pickup) is checked against the tube with the bracket
  zone (node_offset + tube radius from that node) trimmed off, never as a
  whole-tube clash.  Chassis tube vs chassis tube is one welded frame and is
  never a clash pair.

Settings live in the project car dict:
    car['fsae_chassis'] = {'enabled': bool, 'node_offset_mm': 38.1,
                           'tube_od_mm': 25.4,
                           'diagonal_front': 'auto', 'diagonal_rear': 'auto'}
The diagonal is chosen PER AXLE (user 2026-09-23): 'diagonal_front' for the
FL/FR bays, 'diagonal_rear' for the RL/RR bays.  A legacy single 'diagonal'
key (files saved before the per-axle choice) applies to both axles unless a
per-axle key is present.  An ABSENT key means OFF (old project files are
unchanged).  When OFF the
member set is exactly the pre-2026-09-23 one (pickup-to-pickup UCA chassis
cross member + LCA inner chassis member stand-ins).  Pure numpy, no Qt.
"""
import hashlib
import json

import numpy as np

DEFAULTS = {'enabled': True, 'node_offset_mm': 38.1, 'tube_od_mm': 25.4,
            'diagonal_front': 'auto', 'diagonal_rear': 'auto',
            'transverse_front': False, 'transverse_rear': True}
TRANSVERSE_KEYS = {'front': 'transverse_front', 'rear': 'transverse_rear'}
TRANSVERSE_PTS_KEY = 'chassis_transverse'   # pts key: this corner carries transverse halves
# (member name, node) — one transverse tube per inboard pickup, left node <-> right node
TRANSVERSE_TUBES = (
    ('chassis transverse UCA front', 'uca_front'),
    ('chassis transverse UCA rear', 'uca_rear'),
    ('chassis transverse LCA front', 'lca_front'),
    ('chassis transverse LCA rear', 'lca_rear'),
)
TRANSVERSE_NAME_PREFIX = 'chassis transverse'
LEGACY_DIAGONAL_KEY = 'diagonal'        # pre-per-axle files: one choice for both axles
DIAGONAL_KEYS = {'front': 'diagonal_front', 'rear': 'diagonal_rear'}
DIAGONALS = ('auto', 'ucaf_lcar', 'ucar_lcaf')
EXPLICIT_DIAGONALS = ('ucaf_lcar', 'ucar_lcaf')
DIAGONAL_LABELS = {'auto': 'auto (best packaging, full-state audit)',
                   'ucaf_lcar': 'UCA front node - LCA rear node',
                   'ucar_lcaf': 'UCA rear node - LCA front node'}

# pickup -> the outer ball joint of the SAME arm (defines the leg line)
LEG_OUTER = {'uca_front': 'uca_outer', 'uca_rear': 'uca_outer',
             'lca_front': 'lca_outer', 'lca_rear': 'lca_outer'}
NODE_PREFIX = 'chassis_node_'           # pts key prefix for the static nodes
DIAG_PTS_KEY = 'chassis_diagonal'       # pts key carrying the resolved diagonal

# (member name, node a, node b) — the four fixed bay tubes
BAY_TUBES = (
    ('chassis tube UCA front-rear', 'uca_front', 'uca_rear'),
    ('chassis tube LCA front-rear', 'lca_front', 'lca_rear'),
    ('chassis tube front UCA-LCA', 'uca_front', 'lca_front'),
    ('chassis tube rear UCA-LCA', 'uca_rear', 'lca_rear'),
)
DIAGONAL_TUBES = {
    'ucaf_lcar': ('chassis diagonal UCA front-LCA rear', 'uca_front', 'lca_rear'),
    'ucar_lcaf': ('chassis diagonal UCA rear-LCA front', 'uca_rear', 'lca_front'),
}
DIAGONAL_NAME_PREFIX = 'chassis diagonal'
TUBE_NAME_PREFIX = 'chassis '


def settings(car) -> dict | None:
    """Normalised settings when the feature is ON, else None (absent key = OFF)."""
    raw = (car or {}).get('fsae_chassis')
    if not isinstance(raw, dict) or not raw.get('enabled', False):
        return None
    return normalise(raw)


def normalise(raw: dict) -> dict:
    """Normalised copy of a raw car['fsae_chassis'] block regardless of its
    'enabled' flag: defaults filled, per-axle diagonal keys (a legacy single
    'diagonal' key applied to both axles), floats, validated."""
    out = dict(DEFAULTS)
    out['enabled'] = bool(raw.get('enabled', False))
    out.update({k: raw[k] for k in DEFAULTS if k in raw})
    # legacy single 'diagonal' key = both axles, unless a per-axle key is given
    if LEGACY_DIAGONAL_KEY in raw:
        for key in DIAGONAL_KEYS.values():
            if key not in raw:
                out[key] = raw[LEGACY_DIAGONAL_KEY]
    out['node_offset_mm'] = float(out['node_offset_mm'])
    out['tube_od_mm'] = float(out['tube_od_mm'])
    for axle, key in DIAGONAL_KEYS.items():
        if out[key] not in DIAGONALS:
            raise ValueError(f"fsae_chassis {key} must be one of {DIAGONALS}, got {out[key]!r}")
    for key in TRANSVERSE_KEYS.values():
        out[key] = bool(out[key])
    if not (out['tube_od_mm'] > 0.0):
        raise ValueError('fsae_chassis tube_od_mm must be positive')
    return out


def default_settings(enabled: bool = True) -> dict:
    d = dict(DEFAULTS)
    d['enabled'] = bool(enabled)
    return d


def axle_of(label: str) -> str:
    return 'front' if str(label).upper().startswith('F') else 'rear'


def diagonal_setting(s: dict, axle: str) -> str:
    """The saved diagonal choice ('auto' | 'ucaf_lcar' | 'ucar_lcaf') for one
    axle ('front' / 'rear', or a corner label) from normalised settings."""
    axle = axle if axle in DIAGONAL_KEYS else axle_of(axle)
    return s[DIAGONAL_KEYS[axle]]


def auto_axles(s: dict) -> tuple:
    """The axles whose diagonal is 'auto' (needs the full-state resolution)."""
    return tuple(ax for ax, key in DIAGONAL_KEYS.items() if s[key] == 'auto')


def any_auto(s: dict | None) -> bool:
    return bool(s) and bool(auto_axles(s))


def transverse_setting(s: dict, axle: str) -> bool:
    """Whether this axle ('front'/'rear' or a corner label) carries the
    left-node-to-right-node transverse tubes."""
    axle = axle if axle in TRANSVERSE_KEYS else axle_of(axle)
    return bool(s[TRANSVERSE_KEYS[axle]])


def chassis_nodes(hp: dict, node_offset_mm: float) -> dict:
    """{pickup: node (m)} from DESIGN (static) hardpoints: node = P + off*unit(P - O)."""
    off = float(node_offset_mm) / 1000.0
    out = {}
    for k, outer in LEG_OUTER.items():
        if k not in hp or outer not in hp or hp[k] is None or hp[outer] is None:
            continue
        p = np.asarray(hp[k], float); o = np.asarray(hp[outer], float)
        d = p - o; n = float(np.linalg.norm(d))
        if not (np.all(np.isfinite(d)) and n > 1e-9):
            continue
        out[k] = p + off * d / n
    return out


def node_pts(hp: dict, s: dict) -> dict:
    """pts entries ('chassis_node_<pickup>') for a corners-draw dict."""
    return {NODE_PREFIX + k: v for k, v in chassis_nodes(hp, s['node_offset_mm']).items()}


def diagonal_for(win, label: str) -> str:
    """The diagonal the live window uses for this corner, WITHOUT computing:
    an audit override ('both' while auto is being resolved, or a forced choice),
    else the explicit setting, else the cached auto resolution, else
    'ucaf_lcar' (unresolved fallback; the audits always resolve first)."""
    s = settings(getattr(win, '_car', {}))
    if s is None:
        return 'ucaf_lcar'
    ov = getattr(win, '_fsae_diag_override', None)
    if ov:
        return ov
    axle = axle_of(label)
    chosen = diagonal_setting(s, axle)
    if chosen in EXPLICIT_DIAGONALS:
        return chosen
    cache = getattr(win, '_fsae_chassis_cache', None) or {}
    return cache.get(axle) or 'ucaf_lcar'


def bay_members(pts: dict, s: dict, diagonal: str | None = None,
                axle: str | None = None) -> list:
    """Chassis-bay tube capsules for one corner (metres).

    Nodes come from pts['chassis_node_*'] (static design nodes injected by the
    window's corner assembly); if absent they are computed from pts' own
    pickups + ball joints, which is only correct for a STATIC pts dict.
    diagonal: 'ucaf_lcar' | 'ucar_lcaf' | 'both' (auto resolution only) |
    None/'auto' (= pts['chassis_diagonal'], else that axle's explicit setting
    when ``axle`` / pts['label'] says which axle, else 'ucaf_lcar')."""
    nodes = {k: np.asarray(pts[NODE_PREFIX + k], float) for k in LEG_OUTER
             if pts.get(NODE_PREFIX + k) is not None}
    if len(nodes) < 4:
        nodes = chassis_nodes(pts, s['node_offset_mm'])
    if len(nodes) < 4:
        return []
    if diagonal in (None, 'auto'):
        diagonal = pts.get(DIAG_PTS_KEY)
        if diagonal in (None, 'auto'):
            ax = axle or pts.get('label')
            chosen = diagonal_setting(s, ax) if ax else 'auto'
            diagonal = chosen if chosen in EXPLICIT_DIAGONALS else 'ucaf_lcar'
    r = 0.5 * float(s['tube_od_mm']) / 1000.0
    trim = float(s['node_offset_mm']) / 1000.0 + r      # bracket zone at each welded node
    specs = list(BAY_TUBES)
    if diagonal == 'both':
        specs += list(DIAGONAL_TUBES.values())
    else:
        specs.append(DIAGONAL_TUBES[diagonal])
    out = []
    for name, ka, kb in specs:
        pa, pb = pts.get(ka), pts.get(kb)
        joints = []
        if pa is not None:
            joints.append((np.asarray(pa, float), 'a'))
        if pb is not None:
            joints.append((np.asarray(pb, float), 'b'))
        out.append({'name': name, 'a': nodes[ka].copy(), 'b': nodes[kb].copy(), 'r': r,
                    'chassis': True, 'joints': joints, 'bracket_trim': trim})
    # transverse tubes: this corner's HALF, node -> car centreline (X = 0)
    tr = pts.get(TRANSVERSE_PTS_KEY)
    if tr is None:
        ax = axle or pts.get('label')
        tr = transverse_setting(s, ax) if ax else False
    if tr:
        for name, k in TRANSVERSE_TUBES:
            a = nodes[k].copy(); b = a.copy(); b[0] = 0.0
            pk = pts.get(k)
            joints = [(np.asarray(pk, float), 'a')] if pk is not None else []
            out.append({'name': name, 'a': a, 'b': b, 'r': r, 'chassis': True,
                        'joints': joints, 'bracket_trim': trim, 'transverse': True})
    return out


def is_transverse(m: dict) -> bool:
    return bool(m.get('transverse', False))


# ── chassis-fixed obstructions (sprocket disc, diff housing) ────────────────

def cylinder_point_distance_m(p, cyl) -> float:
    """Signed distance (m) from point p to a SOLID finite cylinder
    {'c': axis point, 'u': unit axis, 'h': half-length, 'R': radius} (all m).
    Negative inside = -(smallest way out)."""
    p = np.asarray(p, float); c = np.asarray(cyl['c'], float); u = np.asarray(cyl['u'], float)
    d = p - c
    ax = float(d @ u)
    rad = float(np.linalg.norm(d - ax * u))
    da = abs(ax) - float(cyl['h']); dr = rad - float(cyl['R'])
    if da <= 0.0 and dr <= 0.0:
        return max(da, dr)                       # inside: negative, the nearer face
    return float(np.hypot(max(da, 0.0), max(dr, 0.0)))


def segment_cylinder_gap_mm(a, b, r_m, cyl, samples: int = 201) -> float:
    """Surface gap (mm, negative = overlap) of capsule a-b (radius r_m) vs the
    solid finite cylinder: the point-distance is convex along the segment
    outside the body, so a coarse sample + golden-section refinement on the
    bracket around the best sample is exact to <1e-6 mm; inside the body the
    penetration depth of the deepest sample bracket is refined the same way."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ts = np.linspace(0.0, 1.0, int(samples))
    ds = np.array([cylinder_point_distance_m(a + (b - a) * t, cyl) for t in ts])
    i = int(np.argmin(ds))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    f = lambda t: cylinder_point_distance_m(a + (b - a) * t, cyl)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo); f1 = f(x1); f2 = f(x2)
    for _ in range(60):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = f(x2)
    best = min(float(ds[i]), f1, f2)
    return (best - float(r_m)) * 1000.0


def sprocket_disc_from_mesh(verts_mm, axis=(1.0, 0.0, 0.0)) -> dict | None:
    """The sprocket disc in a diff+sprocket mesh (mm, car coordinates): about
    the given axis direction, the vertices whose radial distance is within 10 %
    of the largest are the tooth ring; its axial extent is the disc thickness
    and its centroid the disc centre.  Returns {'c' (mm, 3), 'u', 'h_mm',
    'R_mm', 'x0_mm', 'x1_mm'} or None when the mesh is empty."""
    V = np.asarray(verts_mm, float)
    if V.ndim != 2 or len(V) < 3:
        return None
    u = np.asarray(axis, float); u = u / np.linalg.norm(u)
    c0 = V.mean(0)
    for _ in range(3):                       # centre refinement (ring centroid)
        d = V - c0
        ax = d @ u
        rad = np.linalg.norm(d - ax[:, None] * u[None, :], axis=1)
        rmax = float(rad.max())
        ring = rad >= 0.9 * rmax
        c0 = V[ring].mean(0)
    d = V - c0; ax = d @ u
    rad = np.linalg.norm(d - ax[:, None] * u[None, :], axis=1)
    rmax = float(rad.max()); ring = rad >= 0.9 * rmax
    x0, x1 = float(ax[ring].min()), float(ax[ring].max())
    c = c0 + u * 0.5 * (x0 + x1)
    return {'c': c, 'u': u, 'h_mm': 0.5 * (x1 - x0), 'R_mm': rmax,
            'x0_mm': float((c0 @ u) + x0), 'x1_mm': float((c0 @ u) + x1)}


def mesh_revolved_envelope(verts_mm, c_mm, axis, exclude_axial=None, bin_mm: float = 2.0) -> list:
    """Revolved envelope of a body-of-revolution mesh about the axis through
    c_mm: one finite cylinder (m) per axial bin of width bin_mm with the bin's
    largest radial extent.  exclude_axial=(lo, hi) drops that axial band (the
    sprocket disc, modelled on its own)."""
    V = np.asarray(verts_mm, float); c = np.asarray(c_mm, float)
    u = np.asarray(axis, float); u = u / np.linalg.norm(u)
    d = V - c; ax = d @ u
    rad = np.linalg.norm(d - ax[:, None] * u[None, :], axis=1)
    if exclude_axial is not None:
        keep = (ax < exclude_axial[0]) | (ax > exclude_axial[1])
        ax, rad = ax[keep], rad[keep]
    if len(ax) == 0:
        return []
    edges = np.arange(np.floor(ax.min()), ax.max() + bin_mm, bin_mm)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ax >= lo) & (ax < hi)
        if not m.any():
            continue
        out.append({'c': (c + u * 0.5 * (lo + hi)) / 1000.0, 'u': u,
                    'h': 0.5 * bin_mm / 1000.0, 'R': float(rad[m].max()) / 1000.0})
    return out


def fixed_obstructions(win) -> list:
    """Chassis-fixed parts the chassis tubes must clear, as
    [{'name', 'cyls': [cylinder (m)], 'source', ...}]:
    * 'rear sprocket disc (from STEP)' — derived from the first imported part
      whose name contains 'sprocket' (verts in mm, car coordinates, offset applied);
    * 'rear diff housing (car dict: tripod span x tripod OD)' — capsule-like
      cylinder along X between the two tripod centres (vahan.driveshaft), radius
      tripod_od/2;
    * 'rear diff+sprocket mesh envelope (revolved, from STEP)' — the rest of
      that mesh as a revolved envelope about the sprocket axis.
    Missing inputs simply leave that part out."""
    from .driveshaft import diff_center_m, tripod_inners_m
    out = []
    car = getattr(win, '_car', {}) or {}
    mesh = None
    for p in (getattr(win, '_imported_parts', None) or []):
        if 'sprocket' in str(p.get('name', '')).lower() and p.get('verts') is not None:
            mesh = p
            break
    if mesh is not None:
        V = np.asarray(mesh['verts'], float)
        disc = sprocket_disc_from_mesh(V, axis=(1.0, 0.0, 0.0))
        if disc is not None:
            out.append({'name': 'rear sprocket disc (from STEP)', 'source': mesh.get('name'),
                        'cyls': [{'c': disc['c'] / 1000.0, 'u': disc['u'],
                                  'h': disc['h_mm'] / 1000.0, 'R': disc['R_mm'] / 1000.0}],
                        'disc': disc})
            # the rest of the mesh, axial position measured from the disc centre
            env = mesh_revolved_envelope(V, disc['c'], disc['u'],
                                         exclude_axial=(-disc['h_mm'] - 0.5, disc['h_mm'] + 0.5))
            if env:
                out.append({'name': 'rear diff+sprocket mesh envelope (revolved, from STEP)',
                            'source': mesh.get('name'), 'cyls': env})
    try:
        # The yellow placeholder diff + tripods (car-dict proxy) is only an obstruction while it is shown
        # (user 2026-09-23: "delete that yellow thing from the native diff hub, it's obstructive" — the real
        # diff is the imported STEP, checked above).
        if not bool(car.get('show_diff_body', False)):
            raise StopIteration
        tri = tripod_inners_m(car); c = diff_center_m(car)
        span = float(np.linalg.norm(tri['L'] - tri['R']))
        if span > 0.0:
            out.append({'name': 'rear diff housing (car dict: tripod span x tripod OD)',
                        'source': 'diff_housing_width_mm / tripod_od_mm / diff position',
                        'cyls': [{'c': 0.5 * (tri['L'] + tri['R']), 'u': np.array([1.0, 0.0, 0.0]),
                                  'h': 0.5 * span, 'R': 0.5 * float(car.get('tripod_od_mm', 90.0)) / 1000.0}]})
    except (StopIteration, Exception):
        pass
    return out


def obstruction_gap_mm(member: dict, part: dict) -> float:
    """Tube capsule vs one obstruction = min over its cylinders (mm)."""
    return min(segment_cylinder_gap_mm(member['a'], member['b'], member['r'], cyl)
               for cyl in part['cyls'])


def fixed_part_gaps(tubes: list, parts: list) -> list:
    """Rows {'a': tube name, 'b': part name, 'gap_mm'} for every chassis tube
    vs every chassis-fixed obstruction (checked once: nothing here moves)."""
    rows = []
    for t in tubes:
        for p in parts:
            if not p.get('cyls'):
                continue
            rows.append({'a': t['name'], 'b': p['name'], 'gap_mm': float(obstruction_gap_mm(t, p))})
    return rows


def is_chassis(m: dict) -> bool:
    return bool(m.get('chassis', False))


def settings_signature(win) -> str:
    """Hash of every live input the auto-diagonal choice depends on: all four
    corners' design hardpoints, both ARB blocks, steering, the car dict and the
    panel ARB section values.  A change of any of them makes the cached auto
    choice stale."""
    h = hashlib.sha1()

    def put(obj):
        h.update(json.dumps(obj, sort_keys=True, default=_jsonable).encode())
    for attr in ('_front_hp', '_rear_hp', '_front_arb', '_rear_arb',
                 '_steer'):
        put({k: _jsonable(v) for k, v in (getattr(win, attr, {}) or {}).items()})
    car = dict(getattr(win, '_car', {}) or {})
    put({k: _jsonable(v) for k, v in car.items()
         if k not in _DISPLAY_ONLY_KEYS and not str(k).startswith('show_')})
    panel = getattr(win, '_dynamics_panel', None)
    vals = []
    for nm in ('_arb_OD_f', '_arb_OD_r', '_arb_blade_w_f', '_arb_blade_w_r',
               '_arb_blade_t_f', '_arb_blade_t_r'):
        c = getattr(panel, nm, None)
        try:
            vals.append(round(float(c.value()), 6) if c is not None else None)
        except Exception:
            vals.append(None)
    put(vals)
    mp = getattr(win, '_motion_panel', None)
    try:
        put([mp.min_val, mp.max_val, mp.stroke_mm])
    except Exception:
        pass
    return h.hexdigest()


# car-dict keys that only change what is DRAWN, never a member position/size
_DISPLAY_ONLY_KEYS = frozenset({'view_mode', 'load_vec_mode', 'wheel_pkg_corner'})


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 9) for x in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 9)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return str(v)
