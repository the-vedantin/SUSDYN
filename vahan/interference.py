"""Geometric interference (clash) detection between suspension/driveline members.

Each member is a capsule: a line segment with a radius (tube OD/2, shaft OD/2,
etc.).  Two capsules clash when the distance between their centre-lines is less
than the sum of their radii (plus a safety margin).  Members that share an
endpoint (a real joint) are skipped — they are meant to touch.

Pure geometry (no Qt), so the GUI's Interference view mode and a headless
regression check both call the same code.  Feed it world-frame points in metres.
"""
import numpy as np


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


def _shares_endpoint(a, b, tol):
    for pa in (a['a'], a['b']):
        for pb in (b['a'], b['b']):
            if np.linalg.norm(np.asarray(pa, float) - np.asarray(pb, float)) < tol:
                return True
    return False


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
            if _shares_endpoint(mi, mj, share_tol_mm / 1000.0):
                continue
            cd = seg_seg_distance(mi['a'], mi['b'], mj['a'], mj['b'])
            gap = (cd - mi['r'] - mj['r']) * 1000.0
            if gap < margin_mm:
                out.append({'a': mi['name'], 'b': mj['name'],
                            'gap_mm': round(gap, 1)})
    out.sort(key=lambda d: d['gap_mm'])
    return out


# Radii (metres) for the members we know about.  Tubes are a nominal FSAE
# a-arm / link OD; the driveshaft uses the real car-dict value.
_TUBE_R = 0.008           # ~16 mm rod-end tube (assumption)


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
