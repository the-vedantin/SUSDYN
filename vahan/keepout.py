# -*- coding: utf-8 -*-
"""Keep-out volumes for packaging ("do not touch this area").

A keep-out is a closed planar-faced solid read from a STEP AP203/AP242 file
(Onshape export, metres, car frame: X lateral, Y rearward, Z up).  It is held
as a union of CONVEX pieces, each a list of outward half-spaces (n, d) with
n.p <= d inside.  A member is a capsule (segment + radius); its gap to the
keep-out is the smallest signed distance of any point of the segment to the
union minus the radius.  Negative = the member is inside the volume.

The point-to-convex-piece distance used here is the largest outward
half-space distance, which is exact when the closest feature is a face and a
CONSERVATIVE under-estimate near edges/corners (it can only report a smaller
gap than the true one).  That is the right bias for a hard packaging gate.

Only the STEP entities needed for planar solids are parsed (CARTESIAN_POINT,
DIRECTION, AXIS2_PLACEMENT_3D, PLANE, ADVANCED_FACE, CLOSED_SHELL, and the
edge loops that give each face its vertices).  Curved faces raise.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

__all__ = ['KeepOut', 'load_step_keepout', 'capsule_gap_mm', 'audit_members']


# ── minimal STEP reader ───────────────────────────────────────────────────────
_ENT = re.compile(r'^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*;\s*$', re.S)


def _split_args(s: str) -> list:
    """Split a STEP argument list at top-level commas."""
    out, depth, cur, quote = [], 0, [], False
    for ch in s:
        if ch == "'" and not quote:
            quote = True
        elif ch == "'" and quote:
            quote = False
        if not quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                out.append(''.join(cur).strip()); cur = []; continue
        cur.append(ch)
    if cur:
        out.append(''.join(cur).strip())
    return out


def _refs(s: str) -> list:
    return [int(x) for x in re.findall(r'#(\d+)', s)]


def _read_entities(text: str) -> dict:
    body = text.split('DATA;', 1)[1].split('ENDSEC;', 1)[0]
    # join continuation lines: entities end with ';'
    ents = {}
    for stmt in body.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        m = _ENT.match(stmt + ';')
        if not m:
            continue
        ents[int(m.group(1))] = (m.group(2), m.group(3))
    return ents


def _unit_scale(ents: dict) -> float:
    """metres per STEP length unit (1.0 for .METRE., 0.001 for .MILLI. .METRE.)."""
    for _, (typ, args) in ents.items():
        if 'LENGTH_UNIT' in args and 'SI_UNIT' in args:
            return 0.001 if '.MILLI.' in args else 1.0
    return 1.0


def _faces_from_step(text: str):
    ents = _read_entities(text)
    scale = _unit_scale(ents)
    pt = {k: np.array([float(x) for x in re.findall(r'[-+0-9.Ee]+', _split_args(a)[1])]) * scale
          for k, (t, a) in ents.items() if t == 'CARTESIAN_POINT'}
    dr = {k: np.array([float(x) for x in re.findall(r'[-+0-9.Ee]+', _split_args(a)[1])])
          for k, (t, a) in ents.items() if t == 'DIRECTION'}
    ax = {k: _refs(a) for k, (t, a) in ents.items() if t == 'AXIS2_PLACEMENT_3D'}
    plane = {k: _refs(a)[0] for k, (t, a) in ents.items() if t == 'PLANE'}
    vtx = {k: _refs(a)[0] for k, (t, a) in ents.items() if t == 'VERTEX_POINT'}
    edge = {k: _refs(a)[:2] for k, (t, a) in ents.items() if t == 'EDGE_CURVE'}
    oedge = {k: _refs(a)[0] for k, (t, a) in ents.items() if t == 'ORIENTED_EDGE'}
    loop = {k: _refs(a) for k, (t, a) in ents.items() if t == 'EDGE_LOOP'}
    bound = {k: _refs(a)[0] for k, (t, a) in ents.items() if t in ('FACE_BOUND', 'FACE_OUTER_BOUND')}
    faces = []
    for k, (t, a) in ents.items():
        if t != 'ADVANCED_FACE':
            continue
        args = _split_args(a); bounds = _refs(args[1]); surf = _refs(args[2])[0]; same = args[3].strip() == '.T.'
        if surf not in plane:
            raise ValueError(f'keep-out face #{k} is not planar ({ents[surf][0]}) — only planar solids are supported')
        origin, ndir = ax[plane[surf]][0], ax[plane[surf]][1]
        n = dr[ndir] / np.linalg.norm(dr[ndir])
        if not same:
            n = -n
        verts = []
        for b in bounds:
            # chain the loop's edges by shared vertex id so the polygon is walked in order
            pairs = [tuple(edge[oedge[oe]]) for oe in loop[bound[b]]]
            chain = [pairs[0][0], pairs[0][1]]; rest = pairs[1:]
            while rest:
                for i, (u, v) in enumerate(rest):
                    if u == chain[-1]:
                        chain.append(v); rest.pop(i); break
                    if v == chain[-1]:
                        chain.append(u); rest.pop(i); break
                else:
                    raise ValueError(f'keep-out face #{k}: edge loop does not chain')
            if chain[0] == chain[-1]:
                chain = chain[:-1]
            for v in chain:
                p = pt[vtx[v]]
                if not verts or np.linalg.norm(p - verts[-1]) > 1e-12:
                    verts.append(p)
        faces.append({'n': n, 'p0': pt[origin], 'verts': np.array(verts)})
    if not faces:
        raise ValueError('no ADVANCED_FACE entities found in the STEP file')
    return faces


# ── keep-out solid ────────────────────────────────────────────────────────────
@dataclass
class KeepOut:
    name: str
    pieces: list = field(default_factory=list)     # each: list of (n, d) half-spaces, n.p <= d inside
    verts: np.ndarray = None                       # all solid vertices (m), for drawing / bbox
    faces: list = field(default_factory=list)      # face dicts from the STEP (for drawing)

    def bbox_m(self):
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def point_depth_m(self, p) -> float:
        """Signed distance of a point to the union: > 0 outside (conservative), < 0 inside (depth)."""
        p = np.asarray(p, float); best = np.inf
        for hs in self.pieces:
            d = max(float(np.dot(n, p) - dd) for n, dd in hs)   # > 0 outside this piece
            best = min(best, d)
        return best

    def capsule_gap_m(self, a, b, r, samples: int = 41) -> float:
        a = np.asarray(a, float); b = np.asarray(b, float)
        ts = np.linspace(0.0, 1.0, samples)
        return min(self.point_depth_m(a + (b - a) * t) for t in ts) - float(r)


def _outward(faces, centroid):
    """Face normals pointing away from the solid centroid (STEP orientation flags are not trusted)."""
    out = []
    for f in faces:
        n = f['n'].copy()
        if np.dot(n, f['p0'] - centroid) < 0:
            n = -n
        out.append((n, float(np.dot(n, f['p0']))))
    return out


def _convex_split_yz_prism(faces, verts):
    """The solids we get are prisms along X (|X| <= w) with a YZ profile.  If the profile is convex the
    solid is one piece; otherwise split at every reflex profile vertex into horizontal Z-bands, each convex.
    Falls back to the single half-space set (which then over-rejects concave regions) when the solid is not
    an X-prism."""
    xs = np.unique(np.round(verts[:, 0], 9))
    if len(xs) != 2:
        return None
    xlo, xhi = float(xs[0]), float(xs[1])
    prof = np.unique(np.round(verts[:, 1:], 9), axis=0)
    # order the profile polygon by walking the YZ edges of the X = xlo face
    side = [f for f in faces if abs(abs(f['n'][0]) - 1) < 1e-6]
    if not side:
        return None
    poly = side[0]['verts'][:, 1:]
    # convexity test
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    n = len(poly); signs = [np.sign(cross(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])) for i in range(n)]
    signs = [s for s in signs if s != 0]
    convex = all(s == signs[0] for s in signs)
    zs = sorted(set(float(z) for z in prof[:, 1]))
    bands = [(zs[0], zs[-1])] if convex else [(zs[i], zs[i + 1]) for i in range(len(zs) - 1)]
    pieces = []
    for zlo, zhi in bands:
        zm = 0.5 * (zlo + zhi)
        # Y-extent of the profile at height zm: intersect the polygon with the horizontal line z = zm
        ys = []
        for i in range(n):
            (y1, z1), (y2, z2) = poly[i], poly[(i + 1) % n]
            if (z1 - zm) * (z2 - zm) < 0:
                ys.append(y1 + (zm - z1) * (y2 - y1) / (z2 - z1))
        if len(ys) < 2:
            continue
        ylo, yhi = min(ys), max(ys)
        # the band is the convex hull of the profile inside [zlo, zhi]: use its slanted edges as half-spaces
        hs = [(np.array([1., 0, 0]), xhi), (np.array([-1., 0, 0]), -xlo),
              (np.array([0, 0, 1.]), zhi), (np.array([0, 0, -1.]), -zlo)]
        for i in range(n):
            (y1, z1), (y2, z2) = poly[i], poly[(i + 1) % n]
            if max(z1, z2) <= zlo + 1e-9 or min(z1, z2) >= zhi - 1e-9:
                continue                                   # edge outside this band
            ey, ez = y2 - y1, z2 - z1
            nn = np.array([0.0, ez, -ey]); nn /= np.linalg.norm(nn)
            # orient outward: away from the band's interior point (ymid, zm)
            ymid = 0.5 * (ylo + yhi)
            if np.dot(nn[1:], np.array([ymid, zm]) - np.array([y1, z1])) > 0:
                nn = -nn
            hs.append((nn, float(np.dot(nn, np.array([0.0, y1, z1])))))
        pieces.append(hs)
    return pieces


def load_step_keepout(path: str, name: str = None) -> KeepOut:
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        text = fh.read()
    faces = _faces_from_step(text)
    verts = np.unique(np.round(np.vstack([f['verts'] for f in faces]), 9), axis=0)
    centroid = verts.mean(axis=0)
    pieces = _convex_split_yz_prism(faces, verts)
    if pieces is None:
        pieces = [_outward(faces, centroid)]
    ko = KeepOut(name=name or path.split('/')[-1].split('\\')[-1], pieces=pieces, verts=verts, faces=faces)
    return ko


def capsule_gap_mm(ko: KeepOut, a, b, r_m) -> float:
    return ko.capsule_gap_m(a, b, r_m) * 1000.0


def audit_members(ko: KeepOut, members: list, exempt_within_mm: float = 0.0) -> list:
    """members: dicts with name/a/b/r (metres).  Returns [(name, gap_mm)] sorted, worst first.
    exempt_within_mm: pickups that sit ON the keep-out wall (chassis tubes) may be excused up to this depth."""
    out = []
    for m in members:
        g = capsule_gap_mm(ko, m['a'], m['b'], m['r'])
        out.append((m['name'], float(g)))
    out.sort(key=lambda x: x[1])
    return out


# ── live-model audit (ONE MODEL: members come from the same assembly the GUI draws) ─────────
def window_members(win, travel_m: float, rack_m: float, front_only: bool = False) -> list:
    """Every chassis-side and wheel-side member of the live model at one state, as capsules
    (metres): vahan.interference.full_members per corner + LCA-inner chassis member proxies,
    the steering-rack housing (1.5 in OD between the live tie_rod_inner points), both torsion
    bars (pivot to mirrored pivot) and the rear driveshafts."""
    import types as _types
    from .interference import full_members, arb_member_kwargs
    from .driveshaft import package as ds_package
    corners, _ = win._assemble_corners_draw({l: float(travel_m) for l in ('FL', 'FR', 'RL', 'RR')}, float(rack_m), light=True)
    by = {c['label']: c for c in corners}; ms = []; rear_states = {}
    for label, c in by.items():
        if front_only and not label.startswith('F'):
            continue
        arb = win._front_arb if label.startswith('F') else win._rear_arb
        apv = np.asarray(arb['arb_pivot'], float)
        od = float(getattr(win._dynamics_panel, '_arb_OD_f' if label.startswith('F') else '_arb_OD_r').value())
        if label.startswith('R'):
            rear_states[label] = _types.SimpleNamespace(wheel_center=np.asarray(c['pts']['wheel_center']), spin_axis=np.asarray(c['spin_axis']))
        front = label.startswith('F')
        panel = win._dynamics_panel
        akw = arb_member_kwargs(win._car, label,
            float(getattr(panel, '_arb_blade_w_f' if front else '_arb_blade_w_r').value()),
            float(getattr(panel, '_arb_blade_t_f' if front else '_arb_blade_t_r').value()))
        for m in full_members(c['pts'], win._car, arb_pivot=apv, arb_od_mm=od, **akw):
            if m.get('chassis'):      # FSAE chassis-bay tubes ARE frame, not suspension members (Rule 18)
                continue
            m = dict(m); m['name'] = label + ' ' + m['name']; ms.append(m)
        hp = c['pts']
        from .chassis import settings as _fsae_settings
        if _fsae_settings(win._car) is None:     # FSAE chassis bays replace this stand-in when ON
            ms.append({'name': label + ' LCA inner chassis member', 'a': np.asarray(hp['lca_front']), 'b': np.asarray(hp['lca_rear']), 'r': 0.008})
    from .interference import rack_members
    ms.extend(rack_members(win._car, by['FL']['pts']['tie_rod_inner'], by['FR']['pts']['tie_rod_inner']))
    for axle, arb in (('front', win._front_arb), ('rear', win._rear_arb)):
        if front_only and axle == 'rear':
            continue
        p = np.asarray(arb['arb_pivot'], float); q = p.copy(); q[0] = -p[0]
        od = float(getattr(win._dynamics_panel, '_arb_OD_f' if axle == 'front' else '_arb_OD_r').value())
        ms.append({'name': axle + ' ARB torsion bar', 'a': p, 'b': q, 'r': od / 2000.0})
    if not front_only:
        pkg = ds_package(win._car, rear_states)
        for label in ('RL', 'RR'):
            if label in pkg:
                ms.append({'name': label + ' driveshaft', 'a': np.asarray(pkg[label]['inner']), 'b': np.asarray(pkg[label]['outer']), 'r': float(win._car.get('driveshaft_dia_mm', 25.4)) / 2000.0})
    return ms


def audit_window(win, ko: KeepOut, n_rack: int = 3, front_only: bool = False) -> dict:
    """Keep-out gaps of every member over droop / static / bump x n_rack rack positions
    (-lock .. +lock).  Returns {'worst': {member: (gap_mm, travel_mm, rack_mm)}, 'inside': [...]}."""
    lo, hi = win._spring_travel_range(win._solvers['FL'], 'FL'); lo = min(lo, -0.025); hi = max(hi, 0.025)
    rh = float(win._steer['total_rack_travel_mm']) / 2000.0
    worst = {}
    for t in (lo, 0.0, hi):
        for rt in np.linspace(-rh, rh, n_rack):
            for m in window_members(win, t, rt, front_only=front_only):
                g = ko.capsule_gap_m(m['a'], m['b'], m['r']) * 1000.0
                if m['name'] not in worst or g < worst[m['name']][0]:
                    worst[m['name']] = (float(g), float(t * 1000), float(rt * 1000))
    inside = sorted([(k, v) for k, v in worst.items() if v[0] < 0], key=lambda kv: kv[1][0])
    return {'worst': worst, 'inside': inside}


def keepout_for_window(win):
    """The keep-out named in the loaded project (car['keepout_step'], repo-relative) or None."""
    import os
    rel = win._car.get('keepout_step') if hasattr(win, '_car') else None
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f'keep-out STEP not found: {rel}')
    return load_step_keepout(path)
