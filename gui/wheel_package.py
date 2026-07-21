"""Wheel Package Load Analysis — a live, Seward-style loads view inside Vahan.

Two SEPARATE analyses the user asked to keep apart:
  * UPRIGHT  — bearings + brake caliper + ball-joint loads ONLY (the upright
               free body), drawn after Derek Seward, Race Car Design Ch.6.
  * CONTROL ARMS — the six suspension-member axial forces, on their own.

A dropdown picks the load case and corner; a toggle switches the upright arrows
between COMPONENT vectors (lateral / fore-aft / vertical shown separately) and a
single RESULTANT vector per point.  Everything is computed from the one solved
model (compute_all_corners), so it always matches the 3-D view and the graphs.

Rendering is matplotlib (not the GL view) precisely because these are load
SCHEMATICS — the same clean diagram style as the binder loads chapter — and so
they stay readable (labels + leaders, never text shoved over the drawing).
"""
import numpy as np
import matplotlib
from matplotlib.figure import Figure

INK, RED, AMBER, GRAY, LGREY = '#1b1b1e', '#cc1f2d', '#b8860b', '#9a9a9e', '#c9c4bb'

# (label, lat g, lon g)   +lon = accel, -lon = braking
CASES = [('Max cornering 2.0 g', 2.0, 0.0),
         ('Full braking 1.6 g', 0.0, -1.6),
         ('Full acceleration 1.0 g', 0.0, 1.0),
         ('Cornering 1.4 g + braking 1.0 g', 1.4, -1.0)]

MEMBERS = [('Upper arm - front leg', 'uca_front', 'uca_outer', 'uca_front_N'),
           ('Upper arm - rear leg',  'uca_rear',  'uca_outer', 'uca_rear_N'),
           ('Lower arm - front leg', 'lca_front', 'lca_outer', 'lca_front_N'),
           ('Lower arm - rear leg',  'lca_rear',  'lca_outer', 'lca_rear_N'),
           ('Tie / toe link',        'tr_inner',  'tr_outer',  'tierod_N'),
           ('Pushrod',               'pushrod_inner', 'pushrod_outer', 'pushrod_N')]


def compute_case(win, lat_g, lon_g):
    """Return (loads_dict, veh, upright_params) for one case, exactly as the
    Loads panel does (ONE MODEL)."""
    from vahan.loads import compute_all_corners
    solver = win._build_dynamics_solver()
    result = solver.solve(lat_g, lon_g)
    veh = solver._veh
    up = win._loads_panel.get_upright_params()
    bp_f = win._loads_panel.get_brake_params_front()
    bp_r = win._loads_panel.get_brake_params_rear()
    cradle = {'front': win._decoupled_solver(True), 'rear': win._decoupled_solver(False)}
    htb = {'front': win._heave_tbar_solver(True), 'rear': win._heave_tbar_solver(False)}
    loads = compute_all_corners(
        win._solvers, result, brake_params_f=bp_f, brake_params_r=bp_r,
        upright_params_f=up, upright_params_r=up, wheel_radius_m=veh.tire_radius_m,
        motion_ratio_f=veh.motion_ratio_front, motion_ratio_r=veh.motion_ratio_rear,
        cradle_solvers=cradle, heave_tbar_solvers=htb)
    return loads, veh, up, result


def _u(st, a, b):
    v = np.asarray(getattr(st, b), float) - np.asarray(getattr(st, a), float)
    return v / max(np.linalg.norm(v), 1e-9)


# Colours for the dark 3-D view (colourblind-safe: white/red/amber/grey).
_C_TEN = (0.92, 0.92, 0.96, 1.0)   # tension (white)
_C_COMP = (0.95, 0.25, 0.25, 1.0)  # compression (red)
_C_UP = (0.66, 0.66, 0.72, 1.0)    # upright / ball-joint (grey)
_C_RX = (0.95, 0.72, 0.12, 1.0)    # ground / caliper reaction (amber)


def load_arrows(win, lat_g, lon_g, mode='resultant', only_corner=None):
    """Force-vector arrows for the 3-D Load mode.

    Returns a list of (p_world, tip_world, rgba).  mode='resultant' draws one
    arrow per load point along the true force direction; mode='components'
    splits each into lateral(X)/fore-aft(Y)/vertical(Z) arrows.  only_corner
    isolates a single corner (for the wheel-package view).
    """
    loads, veh, up, res = compute_case(win, lat_g, lon_g)
    corners = [only_corner] if only_corner else ['FL', 'FR', 'RL', 'RR']
    items = []          # (p, vec, rgba)
    for lbl in corners:
        c = loads.get(lbl)
        if c is None:
            continue
        st = win._solvers[lbl].solve(0.)
        # control-arm member axial forces at their inboard pickups
        for ik, ok, attr in (('uca_front', 'uca_outer', 'uca_front_N'),
                             ('uca_rear', 'uca_outer', 'uca_rear_N'),
                             ('lca_front', 'lca_outer', 'lca_front_N'),
                             ('lca_rear', 'lca_outer', 'lca_rear_N'),
                             ('tr_inner', 'tr_outer', 'tierod_N'),
                             ('pushrod_inner', 'pushrod_outer', 'pushrod_N')):
            F = float(getattr(c, attr))
            u = _u(st, ik, ok)
            p = np.asarray(getattr(st, ik), float)
            items.append((p, F * u, _C_TEN if F >= 0 else _C_COMP))
        # upright ball-joint resultants + contact patch + caliper
        R_uca = -(c.uca_front_N * _u(st, 'uca_front', 'uca_outer')
                  + c.uca_rear_N * _u(st, 'uca_rear', 'uca_outer'))
        R_lca = -(c.lca_front_N * _u(st, 'lca_front', 'lca_outer')
                  + c.lca_rear_N * _u(st, 'lca_rear', 'lca_outer'))
        R_tie = -(c.tierod_N * _u(st, 'tr_inner', 'tr_outer'))
        items.append((np.asarray(st.uca_outer, float), R_uca, _C_UP))
        items.append((np.asarray(st.lca_outer, float), R_lca, _C_UP))
        items.append((np.asarray(st.tr_outer, float), R_tie, _C_UP))
        wc = np.asarray(st.wheel_center, float)
        patch = np.array([wc[0], wc[1], 0.0])
        Fpatch = np.array([float(res.Fy.get(lbl, 0.0)),
                           float(res.Fx.get(lbl, 0.0)),
                           float(res.Fz.get(lbl, 0.0))])
        items.append((patch, Fpatch, _C_RX))
        bt = float(getattr(c, 'brake_torque_Nm', 0.0))
        if abs(bt) > 1.0:
            side = 1.0 if wc[0] >= 0 else -1.0
            rrad = 0.62 * float(veh.tire_radius_m)
            cal_pt = wc + np.array([-side * 0.03, 0.0, rrad])
            items.append((cal_pt, np.array([0.0, np.sign(bt) * abs(bt) / rrad, 0.0]), _C_RX))
    if not items:
        return []
    fmax = max(float(np.linalg.norm(v)) for _, v, _ in items) or 1.0
    SC = 0.14 / fmax
    MINL = 0.028
    arrows = []
    for p, v, col in items:
        if mode == 'components':
            for ax in range(3):
                comp = float(v[ax])
                if abs(comp) < 1.0:
                    continue
                d = np.zeros(3); d[ax] = comp
                L = max(abs(comp) * SC, MINL)
                arrows.append((p, p + np.sign(comp) * (np.abs(d) / max(abs(comp), 1e-9)) * L, col))
        else:
            mag = float(np.linalg.norm(v))
            if mag < 1.0:
                continue
            L = max(mag * SC, MINL)
            arrows.append((p, p + (v / mag) * L, col))
    return arrows


def _iso(p, ctr, az=-0.98, el=0.42):
    x, y, z = p[0] - ctr[0], p[1] - ctr[1], p[2] - ctr[2]
    x1 = x * np.cos(az) - y * np.sin(az)
    y1 = x * np.sin(az) + y * np.cos(az)
    z2 = y1 * np.sin(el) + z * np.cos(el)
    return np.array([x1, z2])


def _varrow(ax, x, y, comp, SC, col, horiz, lab, boxes, picks):
    """One component arrow with a collision-dodged label; records a pick target."""
    if abs(comp) < 1:
        return
    L = float(np.clip(abs(comp) * SC, 0.012, 0.085))
    dx, dy = (np.sign(comp) * L, 0.0) if horiz else (0.0, np.sign(comp) * L)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y), zorder=8,
                arrowprops=dict(arrowstyle='-|>', color=col, lw=2.4),
                annotation_clip=False)
    picks.append((x + dx, y + dy, lab))
    bx, by = x + dx, y + dy
    step, BW, BH = 0.030, 0.235, 0.017
    for kk in range(40):
        cand = (bx + (0.022 if dx >= 0 else -0.022),
                by + step * ((kk + 1) // 2) * (1 if kk % 2 else -1))
        box = (cand[0] - 0.002, cand[0] + BW, cand[1] - BH, cand[1] + BH)
        if not any(box[0] < q[1] and q[0] < box[1] and box[2] < q[3]
                   and q[2] < box[3] for q in boxes):
            boxes.append(box)
            ax.annotate(lab, (bx, by), xytext=cand, textcoords='data', ha='left',
                        va='center', fontsize=8.0, color=col, fontweight='bold',
                        zorder=9, bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                            ec=col, lw=0.7),
                        arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.7,
                                        shrinkA=0, shrinkB=2), annotation_clip=False)
            return


def _resarrow(ax, x, y, hcomp, vcomp, full3, SC, col, lab, boxes, picks):
    """A single RESULTANT arrow (in-plane) with the 3-axis breakdown labelled."""
    mag = float(np.hypot(hcomp, vcomp))
    if mag < 1:
        return
    L = float(np.clip(mag * SC, 0.015, 0.10))
    ux, uy = hcomp / mag, vcomp / mag
    ax.annotate('', xy=(x + ux * L, y + uy * L), xytext=(x, y), zorder=8,
                arrowprops=dict(arrowstyle='-|>', color=col, lw=3.0),
                annotation_clip=False)
    picks.append((x + ux * L, y + uy * L, lab + f'  |R| {full3:,.0f} N'))
    bx, by = x + ux * L, y + uy * L
    step, BW, BH = 0.032, 0.27, 0.020
    for kk in range(40):
        cand = (bx + (0.024 if ux >= 0 else -0.024),
                by + step * ((kk + 1) // 2) * (1 if kk % 2 else -1))
        box = (cand[0] - 0.002, cand[0] + BW, cand[1] - BH, cand[1] + BH)
        if not any(box[0] < q[1] and q[0] < box[1] and box[2] < q[3]
                   and q[2] < box[3] for q in boxes):
            boxes.append(box)
            ax.annotate(f'{lab}\n{full3:,.0f} N', (bx, by), xytext=cand,
                        textcoords='data', ha='left', va='center', fontsize=8.2,
                        color=col, fontweight='bold', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=col, lw=0.9),
                        arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.7,
                                        shrinkA=0, shrinkB=2), annotation_clip=False)
            return


def draw_upright(fig, win, loads, veh, up, res, lbl, case_tag, mode='components'):
    """Seward-style upright free body: bearings + caliper + ball-joint loads."""
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    fig.clear()
    st = win._solvers[lbl].solve(0.)
    c = loads[lbl]
    braking = abs(float(c.brake_torque_Nm)) > 1.0
    W_vert = float(res.Fz.get(lbl, 0.0)); W_lat = float(res.Fy.get(lbl, 0.0))
    W_long = float(res.Fx.get(lbl, 0.0))
    R_uca = -(c.uca_front_N * _u(st, 'uca_front', 'uca_outer')
              + c.uca_rear_N * _u(st, 'uca_rear', 'uca_outer'))
    R_lca = -(c.lca_front_N * _u(st, 'lca_front', 'lca_outer')
              + c.lca_rear_N * _u(st, 'lca_rear', 'lca_outer'))
    R_tie = -(c.tierod_N * _u(st, 'tr_inner', 'tr_outer'))
    Rr = float(veh.tire_radius_m)
    wc = np.asarray(st.wheel_center, float)
    uca = np.asarray(st.uca_outer, float); lca = np.asarray(st.lca_outer, float)
    tro = np.asarray(st.tr_outer, float)
    lateral_case = abs(W_lat) >= abs(W_long)
    i2 = 0 if lateral_case else 1
    Hname = 'lateral' if lateral_case else 'longitudinal'

    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.0, 1.05],
                          hspace=0.32, left=0.07, right=0.98, top=0.9, bottom=0.06)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, :]); axt.axis('off')
    picks = []

    # (a) wheel edge-on + horizontal axle + the two bearings ON it (Seward 6.10)
    side = 1.0 if wc[0] >= 0 else -1.0
    inb = -side
    xc, zc = wc[0], wc[2]
    l1 = up.bearing_spacing_mm / 1000.0
    l2 = 0.045; tw = 0.085
    x_out = xc + inb * l2; x_in = xc + inb * (l2 + l1)
    axa.plot([x_in - 0.07, xc + tw + 0.06], [0, 0], color=INK, lw=2.5)
    axa.add_patch(Rectangle((xc - tw, 0.0), 2 * tw, 2 * Rr, facecolor='#eceae4',
                            edgecolor=GRAY, lw=1.6, alpha=0.85, zorder=1))
    axa.plot([xc, xc], [-0.03, 2 * Rr + 0.03], color=GRAY, lw=1, ls=(0, (6, 4)))
    axa.text(xc, 2 * Rr + 0.05, 'wheel centre-line', ha='center', fontsize=8, color='#55555a')
    axa.plot([x_in - 0.055, xc], [zc, zc], color=INK, lw=5, alpha=0.9,
             solid_capstyle='round', zorder=4)
    axa.text(x_in - 0.03, zc + 0.016, 'axle', fontsize=8, ha='center', zorder=5)
    for xb, nm, ytxt in ((x_out, 'outer bearing', zc + 0.09),
                         (x_in, 'inner bearing', zc + 0.055)):
        axa.add_patch(Rectangle((xb - 0.011, zc - 0.02), 0.022, 0.04,
                                facecolor='#6d6d72', edgecolor=INK, lw=1.3, zorder=6))
        axa.annotate(nm, (xb, zc + 0.02), xytext=(xb, ytxt), ha='center',
                     fontsize=7.8, zorder=6, arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.7))
    axa.annotate('', xy=(x_out, zc - 0.05), xytext=(x_in, zc - 0.05),
                 arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=1))
    axa.text((x_out + x_in) / 2, zc - 0.066, f'bearing spacing l1 = {l1*1000:.0f} mm',
             ha='center', fontsize=7.8)
    xd = xc + tw + 0.03
    axa.annotate('', xy=(xd, 0), xytext=(xd, zc),
                 arrowprops=dict(arrowstyle='<|-|>', color='#55555a', lw=1))
    axa.text(xd + 0.008, zc / 2, f'rolling radius\n{Rr*1000:.0f} mm', fontsize=7.8,
             va='center', color='#55555a')
    SCa = 0.09 / max(abs(W_vert), abs(W_lat), abs(W_long), 1.0)
    axa.annotate('', xy=(xc, W_vert * SCa), xytext=(xc, 0),
                 arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.6))
    axa.text(xc + 0.006, W_vert * SCa, f'vertical {W_vert:,.0f} N', fontsize=8,
             color=RED, fontweight='bold')
    axa.set_aspect('equal'); axa.set_xlim(min(x_in, xc) - 0.12, xc + tw + 0.15)
    axa.set_ylim(-0.16, 2 * Rr + 0.12); axa.grid(alpha=0.18)
    axa.set_title('(a) wheel edge-on: axle horizontal, both bearings coaxial\n'
                  'with the wheel (Seward Fig 6.10)', fontsize=9.2)
    axa.set_xlabel('lateral (m)'); axa.set_ylabel('height (m)')

    # (b) upright silhouette + loads (components OR single resultant)
    poly = np.array([[uca[i2], uca[2]], [tro[i2], tro[2]], [lca[i2], lca[2]]])
    axb.fill(poly[:, 0], poly[:, 1], color='#efece6', ec=INK, lw=2.0, zorder=2)
    axb.add_patch(plt.Circle((wc[i2], wc[2]), 0.030, fill=False, color=INK, lw=2.0, zorder=5))
    SCb = 0.070 / max(abs(R_uca[i2]), abs(R_uca[2]), abs(R_lca[i2]), abs(R_lca[2]),
                      abs(c.bearing_outer_V), abs(c.bearing_inner_V),
                      abs(c.caliper_upper_V), 1.0)
    boxes = []
    for p_ in (uca, lca, tro):
        axb.scatter(p_[i2], p_[2], s=40, color=INK, zorder=6)
    if mode == 'resultant':
        _resarrow(axb, uca[i2], uca[2], R_uca[i2], R_uca[2],
                  float(np.linalg.norm(R_uca)), SCb, AMBER,
                  'upper BJ resultant', boxes, picks)
        _resarrow(axb, lca[i2], lca[2], R_lca[i2], R_lca[2],
                  float(np.linalg.norm(R_lca)), SCb, RED,
                  'lower BJ resultant', boxes, picks)
        _resarrow(axb, tro[i2], tro[2], R_tie[i2], R_tie[2],
                  float(np.linalg.norm(R_tie)), SCb, INK,
                  'tie/toe BJ resultant', boxes, picks)
    else:
        _varrow(axb, uca[i2], uca[2], R_uca[2], SCb, AMBER, False,
                f'upper BJ vertical {R_uca[2]:+,.0f} N', boxes, picks)
        _varrow(axb, uca[i2], uca[2], R_uca[i2], SCb, AMBER, True,
                f'upper BJ {Hname} {R_uca[i2]:+,.0f} N', boxes, picks)
        _varrow(axb, lca[i2], lca[2], R_lca[2], SCb, RED, False,
                f'lower BJ vertical {R_lca[2]:+,.0f} N', boxes, picks)
        _varrow(axb, lca[i2], lca[2], R_lca[i2], SCb, RED, True,
                f'lower BJ {Hname} {R_lca[i2]:+,.0f} N', boxes, picks)
    bear_H = (float(c.bearing_axial_N) if lateral_case
              else float(c.bearing_outer_H + c.bearing_inner_H))
    _varrow(axb, wc[i2] - 0.016, wc[2], c.bearing_outer_V, SCb, INK, False,
            f'outer bearing vert {c.bearing_outer_V:+,.0f} N', boxes, picks)
    _varrow(axb, wc[i2] + 0.016, wc[2], c.bearing_inner_V, SCb, '#6d6d72', False,
            f'inner bearing vert {c.bearing_inner_V:+,.0f} N', boxes, picks)
    if abs(bear_H) > 1:
        _varrow(axb, wc[i2], wc[2] + 0.024, bear_H, SCb, INK, True,
                f'bearing {Hname} {bear_H:+,.0f} N', boxes, picks)
    if braking:
        _varrow(axb, wc[i2], wc[2] - 0.045, float(c.caliper_upper_V), SCb, RED, False,
                f'caliper lug {c.caliper_upper_V:+,.0f} N (x2)', boxes, picks)
    axb.set_aspect('equal')
    xs = [uca[i2], lca[i2], tro[i2], wc[i2]]; zs = [uca[2], lca[2], tro[2], wc[2]]
    axb.set_xlim(min(xs) - 0.22, max(xs) + 0.36); axb.set_ylim(min(zs) - 0.16, max(zs) + 0.16)
    axb.grid(alpha=0.2)
    axb.set_title(f'(b) upright free body - {"rear view (lateral)" if lateral_case else "side view (fore-aft)"}\n'
                  f'{"single RESULTANT per joint" if mode=="resultant" else "COMPONENT vectors"} '
                  f'(Seward Fig 6.14/6.15)', fontsize=9.2)
    axb.set_xlabel(f'{Hname} (m)'); axb.set_ylabel('height (m)')

    tie_nm = 'Tie-rod end' if lbl[0] == 'F' else 'Toe-link end'
    rows = [('Contact patch (tyre)', f'{W_lat:+,.0f}', f'{W_long:+,.0f}', f'{W_vert:+,.0f}'),
            ('Upper ball joint', f'{R_uca[0]:+,.0f}', f'{R_uca[1]:+,.0f}', f'{R_uca[2]:+,.0f}'),
            ('Lower ball joint', f'{R_lca[0]:+,.0f}', f'{R_lca[1]:+,.0f}', f'{R_lca[2]:+,.0f}'),
            (tie_nm, f'{R_tie[0]:+,.0f}', f'{R_tie[1]:+,.0f}', f'{R_tie[2]:+,.0f}'),
            ('Outer bearing', f'{c.bearing_axial_N:+,.0f} ax', f'{c.bearing_outer_H:+,.0f}',
             f'{c.bearing_outer_V:+,.0f}'),
            ('Inner bearing', '-', f'{c.bearing_inner_H:+,.0f}', f'{c.bearing_inner_V:+,.0f}')]
    if braking:
        rows.append(('Brake caliper lug (each)', '-', '-',
                     f'{c.caliper_upper_V:+,.0f}  (torque {c.brake_torque_Nm:,.0f} Nm)'))
    tt = axt.table(cellText=rows,
                   colLabels=['Load on the upright (N)', 'lateral', 'longitudinal', 'vertical'],
                   colWidths=[0.30, 0.22, 0.22, 0.26], cellLoc='left', loc='center')
    tt.auto_set_font_size(False); tt.set_fontsize(9); tt.scale(1, 1.5)
    fig.suptitle(f'{lbl} UPRIGHT loads (bearings + caliper + ball joints) - {case_tag}',
                 fontsize=12, fontweight='bold')
    return picks


def draw_arms(fig, win, loads, lbl, case_tag):
    """The six control-arm axial forces for one corner (isometric)."""
    fig.clear()
    st = win._solvers[lbl].solve(0.)
    c = loads[lbl]
    tire_r = 0.203
    try:
        tire_r = float(win._build_dynamics_solver()._veh.tire_radius_m)
    except Exception:
        pass
    pts3 = ([np.asarray(getattr(st, m[1]), float) for m in MEMBERS]
            + [np.asarray(getattr(st, m[2]), float) for m in MEMBERS])
    ctr = np.mean(pts3, axis=0)
    wc = np.asarray(st.wheel_center, float)
    Fmax = max(abs(float(getattr(c, m[3]))) for m in MEMBERS) or 1.0
    SC = 0.14 / Fmax
    ax = fig.add_subplot(111); ax.axis('off'); ax.set_aspect('equal', adjustable='box')
    allpts = []; picks = []
    th = np.linspace(0, 2 * np.pi, 60)
    ring = np.array([_iso([wc[0], wc[1] + tire_r * np.cos(t), wc[2] + tire_r * np.sin(t)], ctr)
                     for t in th])
    ax.plot(ring[:, 0], ring[:, 1], color=LGREY, lw=1.3, zorder=1)
    allpts += list(ring)
    placed = []
    for name, ik, ok, fattr in MEMBERS:
        p_in = np.asarray(getattr(st, ik), float); p_out = np.asarray(getattr(st, ok), float)
        F = float(getattr(c, fattr)); col = RED if F < 0 else INK
        a, b = _iso(p_in, ctr), _iso(p_out, ctr)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col,
                lw=float(np.clip(abs(F) / 1500.0, 2.4, 8.5)), solid_capstyle='round',
                zorder=3, alpha=0.92)
        ax.scatter(b[0], b[1], s=30, facecolor='white', edgecolor=INK, lw=1.2, zorder=5)
        u = (p_out - p_in) / max(np.linalg.norm(p_out - p_in), 1e-9)
        tip = _iso(p_in + F * u * SC, ctr)
        ax.annotate('', xy=(tip[0], tip[1]), xytext=(a[0], a[1]), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6))
        ax.scatter(a[0], a[1], s=74, color=col, zorder=7)
        allpts += [a, b, tip]; placed.append((a, col, name, F))
        picks.append((a[0], a[1], f'{name}: {F:+,.0f} N'))
    P = np.array([pp[0] for pp in placed]); C2 = P.mean(axis=0)
    g = np.array(allpts)
    gx0, gx1, gy0, gy1 = g[:, 0].min(), g[:, 0].max(), g[:, 1].min(), g[:, 1].max()
    gw, gh = gx1 - gx0, gy1 - gy0
    rx, ry = 0.66 * gw, 0.60 * gh
    ang = np.array([np.arctan2(p[1] - C2[1], p[0] - C2[0]) for p in P])
    base = np.array([[C2[0] + rx * np.cos(t), C2[1] + ry * np.sin(t)] for t in ang])
    dymin = 0.15 * gh
    for sidemask in (base[:, 0] < C2[0], base[:, 0] >= C2[0]):
        idx = np.where(sidemask)[0]; idx = idx[np.argsort(-base[idx, 1])]
        for j in range(1, len(idx)):
            if base[idx[j - 1], 1] - base[idx[j], 1] < dymin:
                base[idx[j], 1] = base[idx[j - 1], 1] - dymin
    for k, (a, col, name, F) in enumerate(placed):
        lp = base[k]; left = lp[0] < C2[0]; tag = 'tension' if F >= 0 else 'compression'
        ax.annotate(f'{name}\n{F:+,.0f} N ({tag})', (a[0], a[1]), xytext=(lp[0], lp[1]),
                    textcoords='data', ha='right' if left else 'left', va='center',
                    fontsize=10, color=col, fontweight='bold', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.30', fc='white', ec=col, lw=1.1),
                    arrowprops=dict(arrowstyle='-', color=GRAY, lw=1.0, shrinkA=3, shrinkB=6,
                                    connectionstyle='arc3,rad=0.06'))
        allpts.append(lp + np.array([(-0.20 * gw if left else 0.20 * gw), 0]))
    g = np.array(allpts)
    padx = 0.05 * (g[:, 0].max() - g[:, 0].min()); pady = 0.05 * (g[:, 1].max() - g[:, 1].min())
    ax.set_xlim(g[:, 0].min() - padx, g[:, 0].max() + padx)
    ax.set_ylim(g[:, 1].min() - pady, g[:, 1].max() + pady)
    fig.suptitle(f'{lbl} CONTROL-ARM axial forces (analysed separately) - {case_tag}\n'
                 f'black = tension, red = compression, thickness proportional to force',
                 fontsize=11, fontweight='bold')
    return picks


def build_dialog(win):
    """Create the non-modal Wheel Package Load Analysis dialog."""
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                 QComboBox, QPushButton, QSizePolicy, QRadioButton,
                                 QButtonGroup, QWidget)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    dlg = QDialog(win)
    dlg.setWindowTitle('Wheel Package Load Analysis')
    dlg.resize(1180, 800)
    dlg.setModal(False)
    root = QHBoxLayout(dlg); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(10)

    ctrl = QVBoxLayout(); ctrl.setSpacing(8)
    def _hdr(t):
        l = QLabel(t); l.setStyleSheet('font-weight:bold;color:#b8860b'); return l
    ctrl.addWidget(_hdr('Analysis'))
    view_up = QRadioButton('Upright (bearings + caliper)')
    view_arm = QRadioButton('Control arms (separate)')
    view_up.setChecked(True)
    vg = QButtonGroup(dlg); vg.addButton(view_up); vg.addButton(view_arm)
    ctrl.addWidget(view_up); ctrl.addWidget(view_arm)
    ctrl.addSpacing(6); ctrl.addWidget(_hdr('Load case'))
    case_cb = QComboBox(); case_cb.addItems([c[0] for c in CASES]); ctrl.addWidget(case_cb)
    ctrl.addWidget(_hdr('Corner'))
    corner_cb = QComboBox(); corner_cb.addItems(['FL', 'FR', 'RL', 'RR']); ctrl.addWidget(corner_cb)
    ctrl.addWidget(_hdr('Upright vectors'))
    vec_comp = QRadioButton('Components (lat / long / vert)')
    vec_res = QRadioButton('Single resultant')
    vec_comp.setChecked(True)
    vgm = QButtonGroup(dlg); vgm.addButton(vec_comp); vgm.addButton(vec_res)
    ctrl.addWidget(vec_comp); ctrl.addWidget(vec_res)
    status = QLabel(''); status.setWordWrap(True); status.setStyleSheet('color:#555;font-size:11px')
    ctrl.addSpacing(6); ctrl.addWidget(status)
    ctrl.addStretch(1)
    close_b = QPushButton('Close'); close_b.clicked.connect(dlg.hide); ctrl.addWidget(close_b)
    cw = QWidget(); cw.setLayout(ctrl); cw.setFixedWidth(240); root.addWidget(cw)

    fig = Figure(facecolor='white'); canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    root.addWidget(canvas, stretch=1)

    ann = canvas.figure.text(0.5, 0.01, '', ha='center', fontsize=9, color='#b8860b')
    state = {'picks': []}

    def refresh():
        try:
            lat_g, lon_g = next((c[1], c[2]) for c in CASES if c[0] == case_cb.currentText())
            loads, veh, up, res = compute_case(win, lat_g, lon_g)
            lbl = corner_cb.currentText()
            case_tag = case_cb.currentText()
            en_up = view_up.isChecked()
            vec_comp.setEnabled(en_up); vec_res.setEnabled(en_up)
            if en_up:
                mode = 'resultant' if vec_res.isChecked() else 'components'
                state['picks'] = draw_upright(fig, win, loads, veh, up, res, lbl, case_tag, mode)
            else:
                state['picks'] = draw_arms(fig, win, loads, lbl, case_tag)
            status.setText(f'{case_tag}\n{lat_g:g} g lateral, {lon_g:g} g longitudinal\n'
                           'Hover the drawing for values.')
            ann.set_text('')
            canvas.draw_idle()
        except Exception as e:
            status.setText(f'Error: {e}')

    def on_move(ev):
        if ev.inaxes is None or not state['picks']:
            return
        best = None; bd = 1e9
        for px, py, lab in state['picks']:
            tp = ev.inaxes.transData.transform((px, py))
            d = np.hypot(tp[0] - ev.x, tp[1] - ev.y)
            if d < bd:
                bd = d; best = lab
        ann.set_text(best if bd < 40 else '')
        canvas.draw_idle()

    canvas.mpl_connect('motion_notify_event', on_move)
    for wgt in (view_up, view_arm, vec_comp, vec_res):
        wgt.toggled.connect(refresh)
    case_cb.currentTextChanged.connect(refresh)
    corner_cb.currentTextChanged.connect(refresh)
    refresh()
    return dlg
