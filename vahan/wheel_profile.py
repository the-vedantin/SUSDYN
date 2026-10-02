"""Real wheel inner profile from the manufacturer's STEP, and member clearance against it.

Replaces the assumed straight barrel (tire_rim_dia_mm cylinder, rim_barrel_width_mm wide) with
the wheel's actual inner surface: barrel radius vs axial position (drop centre, bead seats,
lips) AND the centre disc / spokes, which no member may cross.

Frame: axial position ``d`` is measured INBOARD from the flange centre (the tyre centre
plane, which is where Vahan's wheel_center sits), in mm.  The wheel is axisymmetric for this
purpose (spokes are treated as a solid disc between the hub and the barrel — conservative).

Everything here is geometry from the STEP mesh (vahan.step_import) — no hardware numbers are
typed in.  Backspacing is READ from the mesh (mounting pad to the inboard flange) and reported so
the user can confirm it against the catalogue.
"""
import json
import numpy as np


def barrel_profile_from_step(path, source='onshape', bin_mm=2.0):
    """Tessellate the wheel STEP and reduce it to an inner profile.

    Returns a dict:
      d_mm        : axial stations, positive INBOARD from the flange centre
      r_inner_mm  : smallest surface radius at each station (barrel wall, or the disc where it exists)
      r_barrel_mm : same but only where the station is in the open barrel (NaN across the disc)
      disc_d_mm   : axial position of the centre disc's inboard face (members must stay inboard of it)
      flange_d_mm : (outboard, inboard) flange stations
      flange_r_mm : flange radius (rim OD / 2)
      backspacing_mm : mounting pad to the inboard flange
      width_in    : flange-to-flange span in inches
    """
    from vahan.step_import import load_step_mesh
    v, _, _ = load_step_mesh(path, source=source)
    v = np.asarray(v, float)
    c = v.mean(axis=0); vc = v - c
    step = max(1, len(vc) // 20000)
    _, _, vt = np.linalg.svd(vc[::step], full_matrices=False)
    ax = vt[2]                                   # least-spread principal axis = wheel axis
    z = vc @ ax
    rad = np.linalg.norm(vc - np.outer(z, ax), axis=1)
    edges = np.arange(z.min(), z.max() + bin_mm, bin_mm)
    zc, rmin, rmax = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() < 5:
            continue
        zc.append(0.5 * (lo + hi)); rmin.append(rad[m].min()); rmax.append(rad[m].max())
    zc, rmin, rmax = map(np.asarray, (zc, rmin, rmax))
    half = len(zc) // 2
    i_a, i_b = int(np.argmax(rmax[:half])), half + int(np.argmax(rmax[half:]))
    mid = 0.5 * (zc[i_a] + zc[i_b])
    # mounting pad: the axial station with the most near-axis surface (the flat hub face)
    hub = rad < 60.0
    hist, hedges = np.histogram(z[hub], bins=60)
    zpad = 0.5 * (hedges[np.argmax(hist)] + hedges[np.argmax(hist) + 1])
    # inboard = the flange FARTHER from the pad
    inboard_sign = 1.0 if abs(zc[i_b] - zpad) > abs(zc[i_a] - zpad) else -1.0
    d = (zc - mid) * inboard_sign
    order = np.argsort(d)
    d, rmin, rmax = d[order], rmin[order], rmax[order]
    flange_r = float(max(rmax[i_a], rmax[i_b]))
    # the open barrel is where the smallest radius is still "rim wall" (> 0.7 of the flange radius);
    # the disc / spokes drop the smallest radius far below that
    barrel = rmin > 0.7 * flange_r
    r_barrel = np.where(barrel, rmin, np.nan)
    # disc inboard face: the most inboard station that is NOT open barrel, on the outboard side of centre
    disc_stations = d[~barrel & (d < 0.5 * max(abs(d)))]
    disc_d = float(disc_stations.max()) if len(disc_stations) else float('nan')
    fl = sorted([(zc[i_a] - mid) * inboard_sign, (zc[i_b] - mid) * inboard_sign])
    return {
        'source_step': str(path),
        'd_mm': [float(x) for x in d], 'r_inner_mm': [float(x) for x in rmin],
        'r_barrel_mm': [float(x) for x in r_barrel],
        'disc_d_mm': disc_d, 'flange_d_mm': [float(fl[0]), float(fl[1])], 'flange_r_mm': flange_r,
        'backspacing_mm': float(abs((zpad - mid) * inboard_sign - fl[1])),
        'width_in': float((fl[1] - fl[0]) / 25.4),
    }


def load_profile(path):
    return json.load(open(path, encoding='utf-8'))


def barrel_radius_at(profile, d_mm):
    """Inner barrel radius (mm) at axial station d (inboard +); NaN across the disc / outside the wheel."""
    d = np.asarray(profile['d_mm'], float); r = np.asarray(profile['r_barrel_mm'], float)
    ok = np.isfinite(r)
    dq = np.asarray(d_mm, float)
    out = np.interp(dq, d[ok], r[ok], left=np.nan, right=np.nan)
    inside = (dq >= d[ok].min()) & (dq <= d[ok].max())
    return np.where(inside, out, np.nan)


def member_clearance(member, wheel_center, spin_axis_inboard, profile, samples=60):
    """Smallest surface clearance (mm) of a capsule member to the real wheel, and where.

    member: {'name','a','b','r'} in METRES (vahan.interference convention).
    spin_axis_inboard: unit vector along the wheel axis pointing INBOARD (toward the car centreline).
    Returns dict(gap_mm, station_d_mm, radial_mm, kind) where kind is 'barrel' (radial gap to the wall),
    'disc' (the member is axially inside the centre disc: gap is negative by its radial overlap) or
    'outside' (never within the wheel's axial span).
    """
    a = np.asarray(member['a'], float) * 1000.0; b = np.asarray(member['b'], float) * 1000.0
    wc = np.asarray(wheel_center, float) * 1000.0
    ax = np.asarray(spin_axis_inboard, float); ax = ax / np.linalg.norm(ax)
    rr = float(member['r']) * 1000.0
    d_lo, d_hi = float(min(profile['d_mm'])), float(max(profile['d_mm']))
    best = {'gap_mm': float('inf'), 'station_d_mm': float('nan'), 'radial_mm': float('nan'), 'kind': 'outside'}
    disc_d = profile.get('disc_d_mm', float('nan'))
    r_at_disc = barrel_radius_at(profile, [disc_d + 1.0])[0] if np.isfinite(disc_d) else float('nan')
    for s in np.linspace(0.0, 1.0, samples):
        p = a + (b - a) * s - wc
        dd = float(np.dot(p, ax)); radial = float(np.linalg.norm(p - dd * ax))
        if dd < d_lo or dd > d_hi:
            continue
        if np.isfinite(disc_d) and dd <= disc_d:
            # in the centre disc: anything beyond the hub is a clash; report the radial overlap
            gap = -(radial + rr) if radial + rr > 60.0 else float('inf')
            kind = 'disc'
        else:
            rb = barrel_radius_at(profile, [dd])[0]
            if not np.isfinite(rb):
                continue
            gap = float(rb) - radial - rr
            kind = 'barrel'
        if gap < best['gap_mm']:
            best = {'gap_mm': float(gap), 'station_d_mm': dd, 'radial_mm': radial, 'kind': kind}
    return best
