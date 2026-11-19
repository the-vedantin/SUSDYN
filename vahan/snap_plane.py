"""Actuation "snap plane" per corner, for CAD.

The pushrod's OUTBOARD joint (pushrod_outer) rides on a control arm, so as the
wheel moves it swings on a circle about that arm's pivot axis (front pivot ->
rear pivot).  That circle lies in ONE plane: normal to the arm axis, through
pushrod_outer.  This is the plane the rest of the actuation chain (pushrod
inner end, rocker pivot, damper ends, anti-roll-bar drop link) should sit in.

For CAD we hand over three points that lie exactly on that circle -- the
pushrod outer joint at full droop, at static and at full bump, taken from the
ONE solved model (the corner's solver, at its own damper-stroke travel limits).
"Plane through 3 points" in SolidWorks then reproduces the plane exactly.

Units: the solver works in metres; this module returns metres and leaves unit
conversion to the caller (see ``sw_paste_rows``).
"""
from __future__ import annotations

import numpy as np

CORNERS = ('FL', 'FR', 'RL', 'RR')

# Actuation chain points whose distance from the plane is reported (static).
CHAIN_KEYS = ('pushrod_outer', 'pushrod_inner', 'rocker_pivot',
              'rocker_spring_pt', 'spring_chassis_pt',
              'arb_drop_top', 'arb_arm_end_world')

ARC_TAGS = ('droop', 'static', 'bump')


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * np.nan


def arm_axis(solver):
    """Unit pivot axis of the body that carries pushrod_outer (front -> rear
    pivot), or None when pushrod_outer is on the upright (no fixed axis)."""
    hp = solver.hp
    body = getattr(solver, '_pushrod_body', 'upright')
    if body == 'uca':
        return _unit(np.asarray(hp.uca_rear, float) - np.asarray(hp.uca_front, float)), 'uca'
    if body == 'lca':
        return _unit(np.asarray(hp.lca_rear, float) - np.asarray(hp.lca_front, float)), 'lca'
    return None, body


def corner_plane(solver, t_lo, t_hi, static_pts=None):
    """Plane of one corner's pushrod-outer arc.

    solver     : the corner's live SuspensionConstraints (ONE MODEL).
    t_lo, t_hi : full droop / full bump travel (m), e.g. from
                 MainWindow._spring_travel_range.
    static_pts : optional {name: xyz (m)} of the corner's static actuation
                 points (e.g. MainWindow._assemble_corners_draw pts).

    Returns a dict with metres:
      arc      : {'droop','static','bump'} -> pushrod_outer xyz
      travel   : {'droop','static','bump'} -> travel (m)
      normal   : unit normal of the 3-point plane (sign matched to arm axis)
      point    : point on the plane (static pushrod_outer)
      arm_axis : unit arm pivot axis (None if pushrod on upright)
      arm_body : 'uca' / 'lca' / 'upright'
      normal_vs_axis_deg : angle between plane normal and arm axis
      off_plane: {name: signed distance (m) of static point from the plane}
      rocker_axis_vs_normal_deg : angle between the rocker pivot axis and the
                  plane normal (0 = rocker swings in this plane)
    """
    trav = {'droop': float(t_lo), 'static': 0.0, 'bump': float(t_hi)}
    arc = {}
    for tag in ARC_TAGS:
        st = solver.solve(trav[tag])
        arc[tag] = np.asarray(st.pushrod_outer, float).copy()
    p0, p1, p2 = arc['droop'], arc['static'], arc['bump']
    n = _unit(np.cross(p1 - p0, p2 - p0))
    ax, body = arm_axis(solver)
    if ax is not None and float(np.dot(n, ax)) < 0:
        n = -n
    elif ax is None and n[1] < 0:      # no arm: point forward->rear (+Y) for a stable sign
        n = -n
    ang = (float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(n, ax)))))))
           if ax is not None else float('nan'))
    off = {}
    rax = float('nan')
    if static_pts:
        for k in CHAIN_KEYS:
            v = static_pts.get(k)
            if v is None:
                continue
            v = np.asarray(v, float)
            if v.shape != (3,) or not np.all(np.isfinite(v)):
                continue
            off[k] = float(np.dot(v - p1, n))
        rp, ra = static_pts.get('rocker_pivot'), static_pts.get('rocker_axis_pt')
        if rp is not None and ra is not None:
            a = _unit(np.asarray(ra, float) - np.asarray(rp, float))
            if np.all(np.isfinite(a)):
                rax = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, n)))))))
    return {'arc': arc, 'travel': trav, 'normal': n, 'point': p1.copy(),
            'arm_axis': ax, 'arm_body': body, 'normal_vs_axis_deg': ang,
            'off_plane': off, 'rocker_axis_vs_normal_deg': rax}


def actuation_planes(win):
    """All four corners' planes from a live MainWindow (its solvers, its
    damper-stroke travel limits and its static corner draw points)."""
    try:
        cd, _ = win._assemble_corners_draw({k: 0. for k in CORNERS}, 0.)
        static = {c['label']: c['pts'] for c in cd}
    except Exception:
        static = {}
    out = {}
    for lbl in CORNERS:
        s = win._solvers.get(lbl)
        if s is None:
            continue
        try:
            t_lo, t_hi = win._spring_travel_range(s, lbl)
            if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or abs(t_lo) >= 1.0 or abs(t_hi) >= 1.0:
                t_lo, t_hi = -0.025, 0.025        # stroke unknown: any 3 distinct arc points work
            out[lbl] = corner_plane(s, t_lo, t_hi, static.get(lbl))
        except Exception:
            continue
    return out


def sw_paste_rows(planes, transform):
    """Rows for the Vahan Hardpoints SolidWorks tool paste:
    'actplane_<tag>,FLx,FLy,FLz,FRx,...,RRz' (mm).  ``transform`` maps a Vahan
    xyz in mm to the 3 numbers the paste uses (the SAME transform as the rest
    of that paste).  Corners without a plane get 0,0,0 (skipped by the tool)."""
    rows = []
    for tag in ARC_TAGS:
        vals = []
        for lbl in CORNERS:
            pl = planes.get(lbl)
            if pl is None:
                vals.extend(['0', '0', '0'])
            else:
                vals.extend(f'{v:.3f}' for v in transform(pl['arc'][tag] * 1000.0))
        rows.append(f'actplane_{tag},' + ','.join(vals))
    return rows


def sw_xyz_inch(p_m):
    """Vahan xyz (m) -> SolidWorks assembly xyz (in): X = -Vahan Y,
    Y = Vahan X, Z = Vahan Z (the 'Copy for SW (Front = front, head up)'
    axis system)."""
    p = np.asarray(p_m, float) * 1000.0 / 25.4
    return np.array([-p[1], p[0], p[2]])


def sw_vec(v):
    """Direction vector in the same SolidWorks axes (unitless)."""
    v = np.asarray(v, float)
    return np.array([-v[1], v[0], v[2]])
