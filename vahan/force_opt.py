# -*- coding: utf-8 -*-
"""force_opt.py — pushrod force-transfer (loads) optimizer.

WHY: the pushrod should stay TANGENTIAL to the motion arc of its arm-side
attachment point through travel.  Any off-tangency angle theta means the
pushrod's axial force splits: cos(theta) does useful work rotating the
rocker; sin(theta) is a PARASITIC side load that bends the control arm and
loads its pivots (parasitic/axial = tan(theta); member-load amplification
for a given wheel load = 1/cos(theta)).  Perfect tangency through the whole
stroke is impossible (the arc direction rotates); the optimizer minimizes
the mean-square off-tangency over the operating travel.

KNOBS: pushrod outer position (along-arm + fore/aft) and the rocker group
(pivot + axis + spring pickup + pushrod inner, translated/rotated together).
The WISHBONE PLANES ARE NEVER TOUCHED — front-view arm inclination is a
designed property (it sets the roll-centre height; the low-RC geometry is
why the arms look 'inclined').  Camber, RC, and toe are therefore invariant
by construction; the constraints below guard MR band, damper stroke,
coplanarity and link clearance.
"""
import numpy as np


def tangency_deg(w, lbl, z, dz=0.001):
    """Off-tangency angle (deg) between the pushrod axis and the velocity of
    its arm-side attachment point at travel z.  0 = perfect force transfer."""
    s1 = w._solvers[lbl].solve(z); s2 = w._solvers[lbl].solve(z + dz)
    po1 = np.asarray(s1.pushrod_outer); po2 = np.asarray(s2.pushrod_outer)
    v = po2 - po1
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return float('nan')
    a = np.asarray(s1.pushrod_inner) - po1
    na = np.linalg.norm(a)
    if na < 1e-12:
        return float('nan')
    c = abs(float(np.dot(v / nv, a / na)))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def tangency_profile(w, lbl, zs=(-0.025, -0.012, 0.0, 0.012, 0.025)):
    return [tangency_deg(w, lbl, z) for z in zs]


def force_amplification(w, lbl, z, dz=0.001):
    """TRUE pushrod force per unit vertical wheel load, by virtual work:
        F_pr / F_wheel = v_z(wheel) / (v(po) . a_hat)
    where v(po) is the velocity of the pushrod's arm-side attachment and
    a_hat the pushrod axis.  This captures BOTH the off-tangency angle AND
    the attachment moment-arm — minimizing the angle alone lets an optimizer
    slide the attachment inboard and make forces WORSE (caught 2026-07-13)."""
    s1 = w._solvers[lbl].solve(z); s2 = w._solvers[lbl].solve(z + dz)
    po1 = np.asarray(s1.pushrod_outer); po2 = np.asarray(s2.pushrod_outer)
    wc1 = np.asarray(s1.wheel_center); wc2 = np.asarray(s2.wheel_center)
    v_po = po2 - po1
    v_wz = abs(wc2[2] - wc1[2])
    a = np.asarray(s1.pushrod_inner) - po1
    na = np.linalg.norm(a)
    if na < 1e-12 or v_wz < 1e-12:
        return float('nan')
    axial = abs(float(np.dot(v_po, a / na)))
    if axial < 1e-12:
        return float('inf')
    return float(v_wz / axial)


def amplification_profile(w, lbl, zs=(-0.025, -0.012, 0.0, 0.012, 0.025)):
    return [force_amplification(w, lbl, z) for z in zs]


def _mr_at(w, lbl, z=0.0, dz=0.004):
    s1 = w._solvers[lbl].solve(z - dz); s2 = w._solvers[lbl].solve(z + dz)
    return abs((s2.spring_length - s1.spring_length) / (2 * dz))


def _clearance(w, lbl, zs=(-0.025, 0.0, 0.025)):
    """Min pushrod-to-other-links distance (mm) — the pair the optimizer can
    actually change; full interference checks stay in the city pipeline."""
    def segdist(a1, a2, b1, b2):
        A = np.asarray(a2) - np.asarray(a1); B = np.asarray(b2) - np.asarray(b1)
        return min(np.linalg.norm(np.asarray(a1) + t1 * A - (np.asarray(b1) + t2 * B))
                   for t1 in np.linspace(0, 1, 9) for t2 in np.linspace(0, 1, 9))
    worst = 1e9
    for z in zs:
        try:
            st = w._solvers[lbl].solve(z)
        except Exception:
            continue
        pr = (st.pushrod_inner, st.pushrod_outer)
        for a, b in (('uca_front', 'uca_outer'), ('uca_rear', 'uca_outer'),
                     ('lca_front', 'lca_outer'), ('lca_rear', 'lca_outer'),
                     ('tie_rod_inner', 'tie_rod_outer')):
            pa, pb = getattr(st, a, None), getattr(st, b, None)
            if pa is None or pb is None:
                continue
            worst = min(worst, segdist(pr[0], pr[1], pa, pb) * 1000)
    return worst


def optimize_corner(w, lbl, max_iter=160, seed=3, verbose=True):
    """Minimize mean-square pushrod off-tangency over travel for one corner.

    Design variables (all metres, deltas from current):
      x[0] pushrod_outer along-arm slide  (fraction, 0.85..1.15 of current)
      x[1] pushrod_outer fore/aft (Y)     (+-15 mm)
      x[2..4] rocker group translate X/Y/Z (+-40 mm)  [pivot, axis_pt,
              spring_pt, pushrod_inner move together — rigid rocker move]
    Constraints (penalty): MR within +-0.06 of baseline (springs are sized
    to it), damper length inside stroke, pushrod clearance >= 12 mm,
    solver convergence at all probe travels.
    """
    rng = np.random.default_rng(seed)
    hp = w._front_hp if lbl[0] == 'F' else w._rear_hp
    arb = w._front_arb if lbl[0] == 'F' else w._rear_arb
    # rigid rocker GROUP: pivot + axis + spring pickup + pushrod inner + the
    # damper's chassis mount (must stay in the rocker plane for coplanarity;
    # carrying it keeps the damper geometry rigid so MR moves only through
    # the pushrod side) + the ARB DROP-LINK PICKUP: arb_drop_top is
    # rocker-mounted hardware — leaving it behind while the rocker moves
    # silently re-tunes the ARB motion ratio and flips the car's balance
    # (caught 2026-07-13).
    keys_group = ('rocker_pivot', 'rocker_axis_pt', 'rocker_spring_pt',
                  'pushrod_inner', 'spring_chassis_pt')
    keys_group = tuple(k for k in keys_group if k in hp)
    base = {k: hp[k].copy() for k in list(keys_group) + ['pushrod_outer', 'uca_outer', 'lca_outer']}
    base_droptop = (np.asarray(arb['arb_drop_top'], float).copy()
                    if isinstance(arb, dict) and 'arb_drop_top' in arb else None)
    mr0 = _mr_at(w, lbl)

    def _rot(p, pivot, ang_y, ang_x):
        """Rotate point about the pivot: first about global Y (tilts the
        rocker plane in front view — redirects pushrod-inner travel where
        tangency lives), then about global X."""
        v = p - pivot
        cy, sy = np.cos(ang_y), np.sin(ang_y)
        v = np.array([cy * v[0] + sy * v[2], v[1], -sy * v[0] + cy * v[2]])
        cx, sx = np.cos(ang_x), np.sin(ang_x)
        v = np.array([v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]])
        return pivot + v

    def apply(x):
        for k in base:
            hp[k] = base[k].copy()
        po = base['pushrod_outer']
        d_lo = np.linalg.norm(po - base['lca_outer'])
        d_uo = np.linalg.norm(po - base['uca_outer'])
        anchor = base['lca_outer'] if d_lo <= d_uo else base['uca_outer']
        hp['pushrod_outer'] = anchor + x[0] * (po - anchor) + np.array([0, x[1], 0])
        t = np.array([x[2], x[3], x[4]])
        piv = base['rocker_pivot'] + t
        for k in keys_group:
            p = base[k] + t
            if k != 'rocker_pivot':
                p = _rot(p, piv, x[5], x[6])
            hp[k] = p
        if base_droptop is not None:
            arb['arb_drop_top'] = _rot(base_droptop + t, piv, x[5], x[6])
        try:
            w._enforce_actuation_coplanar()
        except Exception:
            pass
        w._rebuild_solvers(0.)

    def cost(x):
        try:
            apply(x)
            amps = amplification_profile(w, lbl)
            if any(not np.isfinite(a_) for a_ in amps):
                return 1e6
            # objective = mean squared TRUE force amplification (F_pr/F_wheel)
            J = float(np.mean(np.square(amps)))
            mr = _mr_at(w, lbl)
            if abs(mr - mr0) > 0.025:
                # TIGHT: MR drift re-splits roll stiffness and flipped the
                # balance to front-first on the first attempt.  Springs and
                # ARB are sized to the baseline MR.
                J += 1e4 * (abs(mr - mr0) - 0.025)
            clr = _clearance(w, lbl)
            if clr < 12.0:
                J += 60 * (12.0 - clr)
            # damper length sanity across stroke
            for z in (-0.025, 0.025):
                st = w._solvers[lbl].solve(z)
                if not np.isfinite(st.spring_length):
                    return 1e6
            return J
        except Exception:
            return 1e6

    x = np.zeros(7); x[0] = 1.0
    lo = np.array([0.85, -0.015, -0.050, -0.050, -0.050, -0.61, -0.35])
    hi = np.array([1.15,  0.015,  0.050,  0.050,  0.050,  0.61,  0.35])
    J = cost(x)
    J0 = J
    step = (hi - lo) / 8.0
    n_since = 0
    for it in range(max_iter):
        cand = np.clip(x + rng.normal(0, 1, 7) * step, lo, hi)
        Jc = cost(cand)
        if Jc < J:
            x, J = cand, Jc
            n_since = 0
        else:
            n_since += 1
        if n_since == 25:
            step *= 0.5
            n_since = 0
        if np.max(step) < 1e-4:
            break
    apply(x)
    out = dict(J0=J0, J=J, x=list(map(float, x)),
               amp0=float(np.sqrt(J0)), amp=float(np.sqrt(min(J, J0))),
               mr0=mr0, mr=_mr_at(w, lbl), clearance=_clearance(w, lbl),
               amp_profile=amplification_profile(w, lbl),
               theta_profile=tangency_profile(w, lbl))
    if verbose:
        print('  %s: rms force amplification F_pr/F_wheel %.2f -> %.2f | MR %.3f -> %.3f | clearance %.1f mm'
              % (lbl, out['amp0'], out['amp'], mr0, out['mr'], out['clearance']))
    return out
