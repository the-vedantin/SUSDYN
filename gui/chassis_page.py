"""
Chassis page — torsional stiffness vs lateral load transfer (three-spring model).

The sprung mass is split into a FRONT lump and a REAR lump, each at its own
height over its own roll centre.  Front springs, the chassis torsion spring and
the rear springs form the three-spring chain of vahan.chassis_stiffness; the
page shows what the frame carries, how much of a bar change reaches the tyres
(the gain) and the chassis stiffness a given LLTD tolerance needs.

Three tabs:
  Model      — the two bays drawn from the car's own chassis tubes (front
               view), rolled by their solved angles (exaggerated), with the
               lumps, roll centres, moment arms and the chassis torque.
  LLTD vs Kc — total front load-transfer share and the gain against chassis
               stiffness, for the design bar split and for the largest split
               you intend to tune to.
  Numbers    — every input and output with units, plus the closed form.

Headless use (the regression net): `page_inputs(win, state)` and
`bay_outline(win, axle)` are module-level; `ChassisPage(win).refresh()` draws.
Lump overrides and the chassis stiffness live in win._car['chassis_model'] so
they save with the project.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                             QPushButton, QScrollArea, QCheckBox, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox,
                             QSplitter)

from vahan import chassis_stiffness as CS
from vahan import chassis as CH
from vahan.kinematics import KinematicMetrics

_INK, _MUTED, _ACCENT = '#ECECEE', '#9A9AA2', '#E23B48'
_PANEL, _LINE, _PAPER = '#141416', '#2A2A2E', '#0B0B0D'
_YEL, _RED, _WHT, _BLU = '#FFD600', '#E53935', '#FFFFFF', '#42A5F5'
_GRY = '#8a8177'
_MONO = ("font-family:'Consolas','DejaVu Sans Mono',monospace; font-size:11px; "
         f"color:{_INK}; background:#0e0e10; padding:6px; border:1px solid {_LINE}; "
         "border-radius:4px;")

STATE_KEY = 'chassis_model'
DEFAULT_STATE = {
    'kc_ftlbf_deg': 2000.0,      # RoseLap-era figure, the only number the team has
    'a_y_g': 1.5,
    'tol_pts': 1.0,              # LLTD tolerance, percentage points
    'tune_pts': 10.0,            # largest front-ward bar offset you will tune to
    'gain_target': 0.75,         # team decision 2026-10-09: 0.95 is unrealistic, 0.75 is the target
    'exag': 8.0,                 # roll drawn × this in the Model tab
    'override': False,
    'apply_dynamics': False,     # feed k_c into the vehicle dynamics (VehicleParams.chassis_compliance)
    'm_sf_kg': None, 'm_sr_kg': None, 'h_f_mm': None, 'h_r_mm': None,
    'z_f_mm': None, 'z_r_mm': None,
}


# ── model access (headless) ────────────────────────────────────────────────────
def design_rc_m(win) -> tuple:
    """Static roll-centre heights (m) front / rear the way the Dynamics panel
    reads them: KinematicMetrics at travel 0 on the left corners."""
    out = []
    for lbl in ('FL', 'RL'):
        s = win._solvers.get(lbl) if getattr(win, '_solvers', None) else None
        if s is None:
            out.append(0.05); continue
        try:
            out.append(float(KinematicMetrics(s.solve(0.0), 'left').roll_center_height))
        except Exception:
            out.append(0.05)
    return out[0], out[1]


def page_inputs(win, state: dict | None = None):
    """(ChassisInputs, defaults ChassisInputs) for the live car.  `state` is the
    page state dict (see DEFAULT_STATE); overrides apply only when
    state['override'] is true."""
    st = dict(DEFAULT_STATE); st.update(state or {})
    solver = win._build_dynamics_solver()
    veh = solver._veh
    # roll centres the Dynamics solver itself uses at this a_y (left/right mean
    # at the rolled travel) so the rigid-chassis LLTD here IS the solver's;
    # static design RCs if that solve refuses (wheel lift at a silly a_y).
    try:
        r = solver.solve(float(st['a_y_g']), 0.0)
        rc_f, rc_r = float(r.rc_height_front_m), float(r.rc_height_rear_m)
        if not (np.isfinite(rc_f) and np.isfinite(rc_r)):
            raise ValueError('rc nan')
    except Exception:
        rc_f, rc_r = design_rc_m(win)
    kc = CS.nm_deg_to_nm_rad(CS.ftlbf_deg_to_nm_deg(float(st['kc_ftlbf_deg'])))
    base = CS.inputs_from_vehicle(veh, rc_f, rc_r, kc)
    if not st.get('override'):
        return base, base
    def _ov(key, default, scale=1.0):
        v = st.get(key)
        return default if v is None else float(v) * scale
    inp = CS.replace(base,
                     m_sf_kg=_ov('m_sf_kg', base.m_sf_kg),
                     m_sr_kg=_ov('m_sr_kg', base.m_sr_kg),
                     h_f_m=_ov('h_f_mm', base.h_f_m, 1e-3),
                     h_r_m=_ov('h_r_mm', base.h_r_m, 1e-3),
                     z_f_m=_ov('z_f_mm', base.z_f_m, 1e-3),
                     z_r_m=_ov('z_r_mm', base.z_r_m, 1e-3))
    return inp, base


def bay_members_3d(win, axle: str) -> list:
    """The chassis bay tubes on one axle (vahan.chassis.bay_members capsules:
    dicts with 'a', 'b', 'r' in metres + 'transverse'), left corner + mirrored
    right.  Uses the project's FSAE chassis settings when enabled, else the
    defaults — it is a drawing, not a clash check."""
    s = CH.settings(getattr(win, '_car', {})) or CH.default_settings(True)
    hp_all = win._all_corner_hp()
    out = []
    for lbl in (('FL', 'FR') if axle == 'front' else ('RL', 'RR')):
        hp = hp_all.get(lbl)
        if not hp:
            continue
        pts = dict(hp)
        pts.update(CH.node_pts(hp, s))
        try:
            diag = CH.diagonal_for(win, lbl)
        except Exception:
            diag = 'ucaf_lcar'
        for m in CH.bay_members(pts, s, diagonal=diag, axle=axle):
            a, b = np.asarray(m['a'], float), np.asarray(m['b'], float)
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                out.append({'a': a, 'b': b, 'r': float(m.get('r', 0.0127)),
                            'transverse': bool(m.get('transverse', False))})
    return out


def bay_outline(win, axle: str) -> list:
    """3-D (a, b, transverse) segments of the bay tubes (see bay_members_3d)."""
    return [(m['a'], m['b'], m['transverse']) for m in bay_members_3d(win, axle)]


# ── 3-D model scene (data only, so the net can check it headless) ─────────────
_RGBA_YEL = (1.00, 0.84, 0.00, 1.0)
_RGBA_RED = (0.90, 0.22, 0.21, 1.0)
_RGBA_BLU = (0.26, 0.65, 0.96, 1.0)
_RGBA_WHT = (0.92, 0.92, 0.92, 1.0)
_RGBA_GRY = (0.55, 0.55, 0.55, 0.30)


def _roll3(p, z_rc, phi):
    """Rotate a chassis-frame point about the axle's roll centre (x = 0, z = z_rc)
    by phi (rad); + leans the body to −X, the inertia force of a +X a_y."""
    p = np.asarray(p, float)
    x, z = _roll_xz(p[0], p[2], z_rc, phi)
    return np.array([x, p[1], z], float)


def model_scene(inp: CS.ChassisInputs, res: CS.ChassisResult, mem_f: list, mem_r: list,
                exag: float) -> dict:
    """Everything the 3-D Model tab draws over the solved car, as plain arrays
    (chassis frame, metres: +X car left, +Y rearward, +Z up, ground z = 0)."""
    def bay(mem, z_rc, h, m_lump, theta, M, name, rgba):
        phi = float(theta) * float(exag)
        ys = [q[1] for mm in mem for q in (mm['a'], mm['b'])]
        zs = [q[2] for mm in mem for q in (mm['a'], mm['b'])]
        y_bay = float(np.mean(ys)) if ys else 0.0
        z_mid = float(np.mean(zs)) if zs else 0.25
        rolled = [{'a': _roll3(mm['a'], z_rc, phi), 'b': _roll3(mm['b'], z_rc, phi), 'r': mm['r']}
                  for mm in mem]
        rc = np.array([0.0, y_bay, z_rc], float)
        lump = _roll3((0.0, y_bay, h), z_rc, phi)
        return {'name': name, 'rgba': rgba, 'rolled': rolled, 'rc': rc, 'lump': lump,
                'centroid': np.array([0.0, y_bay, z_mid], float), 'phi_true': float(theta),
                'phi_drawn': phi, 'M_Nm': float(M), 'm_kg': float(m_lump), 'h_m': float(h),
                'd_m': float(h - z_rc), 'F_N': float(m_lump) * CS.G * float(res.a_y_g)}
    f = bay(mem_f, inp.z_f_m, inp.h_f_m, inp.m_sf_kg, res.theta_f_rad, res.M_f_Nm, 'FRONT BAY', _RGBA_YEL)
    r = bay(mem_r, inp.z_r_m, inp.h_r_m, inp.m_sr_kg, res.theta_r_rad, res.M_r_Nm, 'REAR BAY', _RGBA_RED)
    a_y = float(res.a_y_g)
    F_max = max(f['F_N'], r['F_N'], 1.0)
    arrows, lines, texts, spheres = [], [], [], []
    for b in (f, r):
        # inertia force on the lump (−X for a +X a_y) + its moment about the RC
        L = 0.10 + 0.25 * b['F_N'] / F_max
        tip = b['lump'] + np.array([-L, 0.0, 0.0])
        arrows.append((b['lump'], tip, b['rgba'],
                       f"{b['name'].title()} lump {b['m_kg']:.0f} kg × {a_y:.2f} g = {b['F_N']:,.0f} N"))
        m_len = 0.06 + 0.16 * abs(b['M_Nm']) / max(abs(f['M_Nm']), abs(r['M_Nm']), 1.0)
        arrows.append((b['rc'], b['rc'] + np.array([0.0, m_len, 0.0]), b['rgba'],
                       f"M = m·a_y·d = {b['M_Nm']:,.0f} N·m ({b['name'].title()})"))
        lines.append((b['rc'], b['lump']))                     # roll arm d
        spheres.append((b['lump'], 0.030, b['rgba']))
        spheres.append((b['rc'], 0.016, _RGBA_BLU))
        texts.append((b['lump'] + np.array([0.0, 0.0, 0.24]),
                      f"{b['name']}  {b['m_kg']:.0f} kg @ {b['h_m'] * 1000:.0f} mm\n"
                      f"d = {b['d_m'] * 1000:.0f} mm   M = {b['M_Nm']:,.0f} N·m\n"
                      f"roll {math.degrees(b['phi_true']):.3f}°  (drawn ×{exag:g})", b['rgba']))
        texts.append((b['rc'] + np.array([0.03, 0.0, -0.09]), f"RC {b['rc'][2] * 1000:.0f} mm", _RGBA_BLU))
    # roll axis + the chassis torsion spring between the bay centroids
    lines.append((f['rc'], r['rc']))
    c0, c1 = f['centroid'], r['centroid']
    n = 18
    pts = [c0]
    for i in range(1, n):
        s = i / n
        amp = 0.045 if 1 < i < n - 1 else 0.0
        pts.append(c0 + (c1 - c0) * s + np.array([amp * (1 if i % 2 else -1), 0.0, 0.0]))
    pts.append(c1)
    spring = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    mid = 0.5 * (c0 + c1)
    kc_deg = CS.nm_rad_to_nm_deg(inp.k_c_Nm_rad)
    arrows.append((mid + np.array([0.0, 0.0, 0.09]), mid + np.array([0.0, 0.18, 0.09]), _RGBA_BLU,
                   f"torque through the frame {res.chassis_torque_Nm:+,.0f} N·m"))
    texts.append((mid + np.array([0.0, 0.0, 0.62]),
                  f"CHASSIS k_c = {kc_deg:,.0f} N·m/deg ({CS.nm_deg_to_ftlbf_deg(kc_deg):,.0f} ft·lbf/deg)\n"
                  f"twist θ_f − θ_r = {res.twist_deg:+.4f}°   torque {res.chassis_torque_Nm:+,.0f} N·m\n"
                  f"α = twist / mean roll = {res.alpha:+.3f}", _RGBA_BLU))
    return {'front': f, 'rear': r, 'static': list(mem_f) + list(mem_r), 'spring': spring,
            'arrows': arrows, 'lines': lines, 'texts': texts, 'spheres': spheres,
            'wheelbase_m': float(inp.wheelbase_m)}


def draw_model_3d(win, v3d, sc: dict) -> None:
    """Feed a View3D with the ONE solved car (static design pose, like the Loads
    page) and the chassis-model overlay from model_scene()."""
    cp = win._car
    try:
        v3d.set_tire_params(outer_r=cp['tire_outer_dia_mm'] / 2000., rim_r=cp['tire_rim_dia_mm'] / 2000.,
                            half_w=cp['tire_width_mm'] / 2000.)
        v3d.set_wheel_profile(win._wheel_profile_for_view())
    except Exception:
        pass
    corners_draw, _ = win._assemble_corners_draw({l: 0.0 for l in ('FL', 'FR', 'RL', 'RR')}, 0.0)
    try:
        thick = cp.get('show_shock_thickness', True)
        v3d.set_spring_dims(cp.get('spring_od_mm', 63.0) if thick else 2.0,
                            cp.get('damper_od_mm', 50.0) if thick else 2.0)
        v3d.set_thickness(thick)
    except Exception:
        pass
    try:
        # Load mode: desaturated links so the coloured bays read, and the force
        # arrows answer to hover like on the Loads page.
        v3d.set_view_mode('load'); v3d.sync_view_controls(view_mode='load')
        v3d.set_isolate_corner(None); v3d.set_brakes(cp.get('show_brakes', True))
    except Exception:
        pass
    try:
        arb = win._assemble_arb_segs(corners_draw)
    except Exception:
        arb = []
    if corners_draw:
        v3d.update_scene(corners_draw, arb)
    v3d.set_chassis_tubes(None)                       # the overlay draws the bays
    v3d.set_overlay_tubes('static', sc['static'], _RGBA_GRY)
    v3d.set_overlay_tubes('front', sc['front']['rolled'], _RGBA_YEL)
    v3d.set_overlay_tubes('rear', sc['rear']['rolled'], _RGBA_RED)
    v3d.set_overlay_spheres('lumps', sc['spheres'])
    v3d.set_overlay_lines('arms', sc['lines'], _RGBA_WHT, width=1.5)
    v3d.set_overlay_lines('spring', sc['spring'], _RGBA_BLU, width=2.5, on_top=True)
    v3d.set_overlay_text('labels', sc['texts'], font_size=7.0)
    v3d.set_load_vectors(sc['arrows'])


# ── drawing ───────────────────────────────────────────────────────────────────
_K_OBL = 0.38   # cabinet projection: screen = (x + K·y, z + K·y), y rearward


def _roll_xz(x, z, z_rc, phi):
    """Rotate (x, z) about the roll centre (0, z_rc) by roll phi (rad, + = the
    body leans to −X, i.e. the inertia force of a +X acceleration)."""
    zr = z - z_rc
    return x * math.cos(phi) - zr * math.sin(phi), z_rc + zr * math.cos(phi) + x * math.sin(phi)


def _proj(x, y, z):
    return x + _K_OBL * y, z + _K_OBL * y


def _draw_bay(ax, name, segs, t_m, z_rc, h_lump, m_lump, phi_true, exag, a_y_g, M_Nm, col, alpha):
    """One bay in the oblique view, rolled about its roll centre.  Returns the
    projected (x, y) of the bay centroid for the chassis-spring glyph."""
    phi = phi_true * exag
    half = 0.5 * t_m
    ys = [a[1] for a, b, _ in segs] + [b[1] for a, b, _ in segs]
    y_bay = float(np.mean(ys)) if ys else 0.0
    zs = [a[2] for a, b, _ in segs] + [b[2] for a, b, _ in segs]
    z_mid = float(np.mean(zs)) if zs else 0.25
    # ground + contact patches (not rolled)
    g0, g1 = _proj(-half - 0.25, y_bay, 0.0), _proj(half + 0.25, y_bay, 0.0)
    ax.plot([g0[0], g1[0]], [g0[1], g1[1]], color=_GRY, lw=1.2, alpha=alpha, label='_g')
    for sx in (-half, half):
        px, py = _proj(sx, y_bay, 0.0)
        ax.plot([px, px], [py, py + 0.03], color=_GRY, lw=5, solid_capstyle='butt', alpha=alpha, label='_cp')
    # tubes
    for (a, b, tr) in segs:
        xa, za = _roll_xz(a[0], a[2], z_rc, phi)
        xb, zb = _roll_xz(b[0], b[2], z_rc, phi)
        sa, sb = _proj(xa, a[1], za), _proj(xb, b[1], zb)
        ax.plot([sa[0], sb[0]], [sa[1], sb[1]], color=_WHT, lw=1.4 if tr else 2.2,
                alpha=alpha * (0.5 if tr else 1.0), label='_t')
    # roll centre + lump
    rx, ry = _proj(0.0, y_bay, z_rc)
    ax.plot([rx], [ry], marker='o', ms=7, color=_BLU, mec=_BLU, alpha=alpha, label='_rc')
    ax.text(rx + 0.04, ry - 0.02, f'RC {z_rc * 1000:.0f} mm', color=_BLU, fontsize=8, ha='left', va='top', alpha=alpha)
    xl, zl = _roll_xz(0.0, h_lump, z_rc, phi)
    lx, ly = _proj(xl, y_bay, zl)
    ax.plot([rx, lx], [ry, ly], color=col, lw=1.0, ls='--', alpha=alpha, label='_d')
    ax.text(0.5 * (rx + lx) + 0.04, 0.5 * (ry + ly), f'd = {(h_lump - z_rc) * 1000:.0f} mm',
            color=col, fontsize=8, ha='left', va='center', alpha=alpha)
    ax.plot([lx], [ly], marker='o', ms=11, color=col, mec='#000', alpha=alpha, label='_m')
    F = m_lump * CS.G * a_y_g
    ax.annotate('', xy=(lx - 0.22, ly), xytext=(lx, ly),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=2.0, mutation_scale=16, alpha=alpha))
    ax.text(lx - 0.24, ly, f'{m_lump:.0f} kg × {a_y_g:.2f} g = {F:,.0f} N', color=col, fontsize=8,
            ha='right', va='center', alpha=alpha)
    ax.text(lx, ly + 0.07, f'{name}\nM = m·a_y·d = {M_Nm:,.0f} N·m\nθ = {math.degrees(phi_true):.3f}° (×{exag:g})',
            color=_INK, fontsize=8, ha='center', va='bottom', alpha=alpha)
    return _proj(0.0, y_bay, z_mid)


def _spring_glyph(ax, p0, p1, n=14, amp=0.035):
    """Zig-zag torsion-spring glyph between two screen points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = float(np.linalg.norm(d))
    if L < 1e-6:
        return
    u = d / L; nrm = np.array([-u[1], u[0]])
    pts = [p0, p0 + u * 0.08 * L]
    core = 0.84 * L
    for i in range(1, n):
        s = 0.08 * L + core * i / n
        pts.append(p0 + u * s + nrm * amp * (1 if i % 2 else -1))
    pts += [p0 + u * 0.92 * L, p1]
    ax.plot([q[0] for q in pts], [q[1] for q in pts], color=_BLU, lw=1.8, label='_spring')


def draw_model(slot, inp: CS.ChassisInputs, res: CS.ChassisResult, segs_f, segs_r, exag: float):
    ax = slot.axes(1)
    c_r = _draw_bay(ax, 'REAR BAY', segs_r, inp.t_r_m, inp.z_r_m, inp.h_r_m, inp.m_sr_kg,
                    res.theta_r_rad, exag, res.a_y_g, res.M_r_Nm, _RED, 0.75)
    c_f = _draw_bay(ax, 'FRONT BAY', segs_f, inp.t_f_m, inp.z_f_m, inp.h_f_m, inp.m_sf_kg,
                    res.theta_f_rad, exag, res.a_y_g, res.M_f_Nm, _YEL, 1.0)
    _spring_glyph(ax, c_f, c_r)
    mid = 0.5 * (np.asarray(c_f) + np.asarray(c_r))
    kc_deg = CS.nm_rad_to_nm_deg(inp.k_c_Nm_rad)
    ax.text(mid[0] + 0.06, mid[1] - 0.02,
            f'k_c = {kc_deg:,.0f} N·m/deg ({CS.nm_deg_to_ftlbf_deg(kc_deg):,.0f} ft·lbf/deg)\n'
            f'twist θ_f − θ_r = {res.twist_deg:+.4f}°\ntorque through the frame = {res.chassis_torque_Nm:+,.0f} N·m\n'
            f'α = twist / mean roll = {res.alpha:+.3f}', color=_BLU, fontsize=8, ha='left', va='top')
    # scale bar + axis hint
    sx0, sy0 = _proj(-0.5 * inp.t_f_m - 0.25, 0.0, -0.12)
    ax.plot([sx0, sx0 + 0.5], [sy0, sy0], color=_GRY, lw=2, label='_sb')
    ax.text(sx0 + 0.25, sy0 - 0.02, '0.5 m', color=_GRY, fontsize=7, ha='center', va='top')
    ax.annotate('', xy=(sx0 + 0.9, sy0), xytext=(sx0 + 0.7, sy0),
                arrowprops=dict(arrowstyle='-|>', color=_GRY, lw=1))
    ax.text(sx0 + 0.92, sy0, '+X car left', color=_GRY, fontsize=7, ha='left', va='center')
    slot.fig.suptitle('Front and rear bays (oblique view, rearward = up-right), each rolled about its roll centre',
                      color='#d8d2c8', fontsize=9, x=0.01, ha='left')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.autoscale(); ax.margins(0.06, 0.22)
    slot.finish()



def draw_curve(slot, inp: CS.ChassisInputs, summ: dict, wf: float, gain_target: float = 0.75):
    ax1, ax2 = slot.axes(2)
    kc_ft = np.logspace(math.log10(100.0), math.log10(30000.0), 160)
    kc_rad = np.array([CS.nm_deg_to_nm_rad(CS.ftlbf_deg_to_nm_deg(k)) for k in kc_ft])
    tuned = CS.replace(inp, k_f_Nm_rad=summ['tuned_k_f_Nm_rad'])
    l_des = CS.lltd_curve(inp, kc_rad) * 100.0
    l_tun = CS.lltd_curve(tuned, kc_rad) * 100.0
    g_des = np.array([CS.gain(CS.replace(inp, k_c_Nm_rad=k)) for k in kc_rad])
    g_tun = np.array([CS.gain(CS.replace(tuned, k_c_Nm_rad=k)) for k in kc_rad])
    kc_now = CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(inp.k_c_Nm_rad))
    kc_tol = summ['kc_tol_tuned']
    kc_tol_ft = CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(kc_tol)) if np.isfinite(kc_tol) and kc_tol > 0 else None
    five_x = 5.0 * CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(inp.k_f_Nm_rad + inp.k_r_Nm_rad))

    ax1.plot(kc_ft, l_des, color=_YEL, lw=2, label='design split')
    ax1.plot(kc_ft, l_tun, color=_RED, lw=2, label=f'+{summ["tune_offset"] * 100:.0f} pt split')
    ax1.axhline(inp.lltd_rigid * 100, color=_WHT, ls='--', lw=1, label='_r')
    ax1.axhline(tuned.lltd_rigid * 100, color=_RED, ls='--', lw=1, alpha=0.7, label='_rt')
    ax1.axhline(wf * 100, color=_GRY, ls=':', lw=1, label='_wf')
    ax1.text(kc_ft[0] * 1.05, wf * 100, f'front weight {wf * 100:.1f} %', color=_GRY, fontsize=7, va='bottom')
    ax1.text(kc_ft[0] * 1.05, inp.lltd_rigid * 100, f'rigid {inp.lltd_rigid * 100:.1f} %', color=_WHT, fontsize=7, va='bottom')
    ax1.axvline(kc_now, color=_BLU, lw=1.2, label='_now')
    ax1.text(kc_now * 1.04, l_tun.max(), f'k_c now\n{kc_now:,.0f}', color=_BLU, fontsize=7, va='top')
    if kc_tol_ft:
        ax1.axvline(kc_tol_ft, color=_BLU, ls=':', lw=1.2, label='_tol')
        ax1.text(kc_tol_ft * 1.04, tuned.lltd_rigid * 100 - 0.6 * summ['tol_lltd'] * 100,
                 f'±{summ["tol_lltd"] * 100:.1f} pt needs\n{kc_tol_ft:,.0f}', color=_BLU, fontsize=7, va='top')
    ax1.axvline(five_x, color=_GRY, ls='-.', lw=0.9, label='_5x')
    ax1.text(five_x * 1.04, 0.5 * (l_tun.max() + l_des.min()), f'5× roll\n{five_x:,.0f}', color=_GRY, fontsize=7, va='center')
    ax1.set_xscale('log')
    ax1.set_xlabel('chassis torsional stiffness k_c (ft·lbf/deg, log)', fontsize=8)
    ax1.set_ylabel('total front load-transfer share (%)', fontsize=8)
    ax1.set_title('LLTD vs chassis stiffness', color='#d8d2c8', fontsize=10, loc='left')
    slot.legend(ax1, loc='center right')

    ax2.plot(kc_ft, g_des, color=_YEL, lw=2, label='design split')
    ax2.plot(kc_ft, g_tun, color=_RED, lw=2, label=f'+{summ["tune_offset"] * 100:.0f} pt split')
    for g, txt in ((0.90, 'driver cannot feel the loss'), (0.95, 'bars one-for-one')):
        ax2.axhline(g, color=_GRY, ls=':', lw=1, label='_g')
        ax2.text(kc_ft[0] * 1.05, g + 0.004, f'{g:.2f}  {txt}', color=_GRY, fontsize=7, va='bottom')
    kc_g = CS.kc_for_gain(inp, gain_target)
    if np.isfinite(kc_g) and kc_g > 0:
        kc_g_ft = CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(kc_g))
        ax2.axhline(gain_target, color=_WHT, ls='--', lw=1.2, label=f'target gain {gain_target:.2f}')
        ax2.axvline(kc_g_ft, color=_WHT, ls='--', lw=1.2, label=f'k_c for it: {kc_g_ft:,.0f} ft·lbf/deg')
        ax2.plot([kc_g_ft], [gain_target], marker='o', ms=6, color=_WHT, label='_t')
        ax1.axvline(kc_g_ft, color=_WHT, ls='--', lw=1.2, label=f'k_c for gain {gain_target:.2f}: {kc_g_ft:,.0f}')
    ax2.axvline(kc_now, color=_BLU, lw=1.2, label='_now')
    ax2.set_xscale('log')
    ax2.set_ylim(min(0.5, gain_target - 0.1), 1.02)
    ax2.set_xlabel('chassis torsional stiffness k_c (ft·lbf/deg, log)', fontsize=8)
    ax2.set_ylabel('gain = share of a bar change that reaches the tyres', fontsize=8)
    ax2.set_title('Gain vs k_c', color='#d8d2c8', fontsize=10, loc='left')
    slot.legend(ax2, loc='lower right')
    slot.finish()


# ── the page ──────────────────────────────────────────────────────────────────
class ChassisPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._inp = None
        self._scene = None
        self._base = None
        self._summ = None
        self._loading = False
        self._build()
        self._load_state()
        self.refresh()

    # state ------------------------------------------------------------------
    def _state(self) -> dict:
        return {
            'kc_ftlbf_deg': float(self._kc.value()), 'a_y_g': float(self._ay.value()),
            'tol_pts': float(self._tol.value()), 'tune_pts': float(self._tune.value()),
            'gain_target': float(self._gain.value()),
            'exag': float(self._exag.value()), 'override': bool(self._ovr.isChecked()),
            'apply_dynamics': bool(self._apply_dyn.isChecked()),
            'm_sf_kg': float(self._msf.value()), 'm_sr_kg': float(self._msr.value()),
            'h_f_mm': float(self._hf.value()), 'h_r_mm': float(self._hr.value()),
            'z_f_mm': float(self._zf.value()), 'z_r_mm': float(self._zr.value()),
        }

    def _load_state(self):
        st = dict(DEFAULT_STATE); st.update((getattr(self._main, '_car', {}) or {}).get(STATE_KEY, {}) or {})
        # k_c + the dynamics toggle are owned by the Dynamics panel; mirror them here
        dp = getattr(self._main, '_dynamics_panel', None)
        if dp is not None and hasattr(dp, '_chassis_kc'):
            st['kc_ftlbf_deg'] = CS.nm_deg_to_ftlbf_deg(float(dp._chassis_kc.value()))
            st['apply_dynamics'] = bool(dp._chassis_on.isChecked())
        self._loading = True
        try:
            self._kc.setValue(float(st['kc_ftlbf_deg'])); self._ay.setValue(float(st['a_y_g']))
            self._tol.setValue(float(st['tol_pts'])); self._tune.setValue(float(st['tune_pts']))
            self._gain.setValue(float(st.get('gain_target', 0.75)))
            self._exag.setValue(float(st['exag'])); self._ovr.setChecked(bool(st['override']))
            self._apply_dyn.setChecked(bool(st.get('apply_dynamics', False)))
            for w, k in ((self._msf, 'm_sf_kg'), (self._msr, 'm_sr_kg'), (self._hf, 'h_f_mm'),
                         (self._hr, 'h_r_mm'), (self._zf, 'z_f_mm'), (self._zr, 'z_r_mm')):
                if st.get(k) is not None:
                    w.setValue(float(st[k]))
        finally:
            self._loading = False

    def _save_state(self):
        car = getattr(self._main, '_car', None)
        if isinstance(car, dict):
            car[STATE_KEY] = self._state()
        dp = getattr(self._main, '_dynamics_panel', None)
        if dp is not None and hasattr(dp, '_chassis_kc'):      # one k_c: write it back to the Dynamics panel
            dp.set_state({'chassis_kc_Nm_deg': CS.ftlbf_deg_to_nm_deg(float(self._kc.value())),
                          'chassis_compliance': bool(self._apply_dyn.isChecked())})

    # ui -----------------------------------------------------------------------
    def _build(self):
        from gui.ackermann_page import _Slot
        from gui.panels import _spin, BTN_PRIMARY, BTN_SECONDARY
        self.setStyleSheet(
            f'QWidget{{background:{_PAPER};color:{_INK};font-size:13px}}'
            f'QGroupBox{{border:1px solid {_LINE};border-radius:6px;margin-top:10px;padding-top:8px}}'
            f'QGroupBox::title{{color:{_ACCENT};subcontrol-origin:margin;left:8px;padding:0 4px}}'
            f'QDoubleSpinBox,QComboBox{{background:#1e1e24;color:{_INK};border:1px solid {_LINE};padding:2px}}'
            f'QTableWidget{{background:{_PANEL};gridline-color:{_LINE};}}'
            f'QHeaderView::section{{background:#1e1e24;color:{_INK};padding:4px;border:0;}}'
            f'QTabBar::tab{{background:{_PANEL};color:{_MUTED};padding:6px 12px;border:1px solid {_LINE}}}'
            f'QTabBar::tab:selected{{color:{_INK};background:#1e1e24}}')
        root = QHBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 6, 0); ll.setSpacing(6)
        eyebrow = QLabel('CHASSIS'); eyebrow.setStyleSheet(f'color:{_ACCENT};font-size:11px;font-weight:650;letter-spacing:2px')
        title = QLabel('Torsional stiffness vs load transfer'); title.setStyleSheet(f'color:{_INK};font-size:15px;font-weight:bold')
        cap = QLabel('Front lump · front springs · chassis torsion spring · rear springs · rear lump. '
                     'What the frame carries, how much of a bar change reaches the tyres, and the '
                     'stiffness an LLTD tolerance needs.')
        cap.setWordWrap(True); cap.setStyleSheet(f'color:{_MUTED};font-size:11px')
        ll.addWidget(eyebrow); ll.addWidget(title); ll.addWidget(cap)
        self._note = QLabel(''); self._note.setStyleSheet(_MONO); self._note.setWordWrap(True)
        ll.addWidget(self._note)

        def row2(grid, r, text, w, tip=''):
            lab = QLabel(text); lab.setStyleSheet(f'color:{_MUTED};font-size:12px')
            if tip:
                lab.setToolTip(tip); w.setToolTip(tip)
            grid.addWidget(lab, r, 0); grid.addWidget(w, r, 1)

        g1 = QGroupBox('Chassis + case'); gl1 = QGridLayout(g1); gl1.setVerticalSpacing(4)
        self._kc = _spin(50.0, 50000.0, 2000.0, ' ft·lbf/deg', dec=0, step=100.0)
        self._kc_nm = QLabel(''); self._kc_nm.setStyleSheet(f'color:{_MUTED};font-size:11px')
        self._ay = _spin(0.1, 3.0, 1.5, ' g', dec=2, step=0.1)
        self._tol = _spin(0.1, 10.0, 1.0, ' pt', dec=1, step=0.5)
        self._tune = _spin(0.0, 30.0, 10.0, ' pt', dec=1, step=1.0)
        self._exag = _spin(1.0, 40.0, 8.0, ' ×', dec=0, step=1.0)
        self._gain = _spin(0.10, 0.99, 0.75, '', dec=2, step=0.05)
        row2(gl1, 0, 'k_c (hub to hub)', self._kc, 'Chassis torsional stiffness as the suspension sees it, hub to hub. '
             'A frame-only FEA number overstates it.')
        gl1.addWidget(self._kc_nm, 1, 1)
        self._apply_dyn = QCheckBox('apply k_c to the vehicle dynamics')
        self._apply_dyn.setToolTip('On: the steady-state solver (skidpad, G-G, lap sim, loads) uses this k_c — the front and '
                                   'rear of the body roll by their own angles, so each axle gets its own camber and the elastic '
                                   'load-transfer split moves.  Off: rigid chassis.  Kinematics are unchanged either way.')
        gl1.addWidget(self._apply_dyn, 7, 0, 1, 2)
        row2(gl1, 2, 'lateral a_y', self._ay, 'Load case for the Model tab and the twist / torque numbers.')
        row2(gl1, 3, 'LLTD tolerance', self._tol, 'Allowed deviation of the total front share from the rigid-chassis value, '
             'percentage points.  ~1 pt is where the driver stops noticing.')
        row2(gl1, 4, 'largest bar offset', self._tune, 'Largest front-ward move of the ELASTIC split you intend to tune to '
             '(points).  The required k_c is evaluated there — at the design split the frame carries nothing and any k_c passes.')
        row2(gl1, 5, 'target gain', self._gain, 'Share of a bar change that must reach the tyres. gain = k_c / (k_c + k_series). '
             'Team target 0.75 (0.95 was judged unrealistic); the k_c it needs is reported below and marked on the curve.')
        row2(gl1, 6, 'draw roll ×', self._exag, 'Exaggeration of the roll angles in the Model tab.')
        ll.addWidget(g1)

        g2 = QGroupBox('Localized sprung masses'); gl2 = QGridLayout(g2); gl2.setVerticalSpacing(4)
        self._ovr = QCheckBox('override the model split')
        self._ovr.setToolTip('Off: the sprung mass is split at the sprung CG station, both lumps at the sprung CG '
                             'height, roll centres from the kinematics.  On: use the six numbers below.')
        gl2.addWidget(self._ovr, 0, 0, 1, 2)
        self._msf = _spin(0.0, 500.0, 100.0, ' kg', dec=1, step=1.0)
        self._msr = _spin(0.0, 500.0, 100.0, ' kg', dec=1, step=1.0)
        self._hf = _spin(0.0, 800.0, 280.0, ' mm', dec=1, step=5.0)
        self._hr = _spin(0.0, 800.0, 280.0, ' mm', dec=1, step=5.0)
        self._zf = _spin(-200.0, 400.0, 50.0, ' mm', dec=1, step=5.0)
        self._zr = _spin(-200.0, 400.0, 50.0, ' mm', dec=1, step=5.0)
        row2(gl2, 1, 'front lump m_sf', self._msf, 'Sprung mass reacted at the front bay.')
        row2(gl2, 2, 'rear lump m_sr', self._msr, 'Sprung mass reacted at the rear bay.')
        row2(gl2, 3, 'front lump height h_f', self._hf, 'Height of the front lump CG above ground.')
        row2(gl2, 4, 'rear lump height h_r', self._hr, 'Height of the rear lump CG above ground.')
        row2(gl2, 5, 'front RC z_F', self._zf, "Front roll-centre height (the Dynamics solver's at this a_y when not overridden).")
        row2(gl2, 6, 'rear RC z_R', self._zr, 'Rear roll-centre height.')
        self._reset = QPushButton('From model'); self._reset.setStyleSheet(BTN_SECONDARY)
        self._reset.setToolTip('Copy the model split into the six fields.')
        gl2.addWidget(self._reset, 7, 0, 1, 2)
        ll.addWidget(g2)

        self._recalc = QPushButton('Recalculate'); self._recalc.setStyleSheet(BTN_PRIMARY)
        ll.addWidget(self._recalc)
        ll.addStretch(1)

        from gui.side_panel import SideScroll, side_split
        scroll = SideScroll(left, 340)   # resizable, never clipped

        self._tabs = QTabWidget()
        # Model = the real 3-D view (second View3D fed from the one solved car,
        # built on first refresh because its GL canvas needs a shown window).
        self._v3d = None
        self._model_host = QWidget()
        self._model_lay = QVBoxLayout(self._model_host)
        self._model_lay.setContentsMargins(0, 0, 0, 0); self._model_lay.setSpacing(4)
        cap = QLabel('The solved car with its front and rear bay tubes rolled about their own roll centres '
                     '(yellow front, red rear, grey = unrolled), the two sprung lumps with their inertia '
                     'forces and moments, and the chassis drawn as the torsion spring between the bays. '
                     'Drag to orbit, wheel to zoom, hover an arrow for its value.')
        cap.setWordWrap(True); cap.setStyleSheet(f'color:{_MUTED}; font-size:11px;')
        self._view_ph = QLabel('3-D view builds when the page is shown.')
        self._view_ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view_ph.setStyleSheet(f'color:{_MUTED}; background:#000; border:1px solid {_LINE};')
        self._model_sum = QLabel('—'); self._model_sum.setWordWrap(True)
        self._model_sum.setStyleSheet(
            "color:#e8e8ec; font-family:'Consolas','SF Mono',monospace; font-size:11px; "
            "background:#0e0e10; padding:5px; border:1px solid #2a2a2a; border-radius:4px;")
        self._model_lay.addWidget(cap)
        self._model_lay.addWidget(self._view_ph, stretch=1)
        self._model_lay.addWidget(self._model_sum)
        self._slot_model = _Slot('The two bays from the car\'s chassis tubes, rolled by their solved angles. '
                                 'Each lump\'s moment m·a_y·d is reacted by its own springs, by the other axle '
                                 'through the chassis spring, or both.')
        self._slot_curve = _Slot('Total front load-transfer share and the bar-change gain against chassis '
                                 'stiffness, for the design split (yellow) and the largest split you will tune to (red).')
        self._tbl = QTableWidget(0, 4)
        self._tbl.setHorizontalHeaderLabels(['Quantity', 'Value', 'Unit', 'Note'])
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.verticalHeader().setVisible(False)
        self._formula = QLabel(''); self._formula.setStyleSheet(_MONO); self._formula.setWordWrap(True)
        self._formula.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        num = QWidget(); nl = QVBoxLayout(num); nl.setContentsMargins(0, 0, 0, 0)
        spl = QSplitter(Qt.Orientation.Vertical); spl.addWidget(self._tbl); spl.addWidget(self._formula)
        spl.setStretchFactor(0, 3); spl.setStretchFactor(1, 1)
        nl.addWidget(spl)
        self._tabs.addTab(self._model_host, 'Model')
        self._tabs.addTab(self._slot_curve, 'LLTD vs Kc')
        self._tabs.addTab(num, 'Numbers')
        self._tabs.addTab(self._slot_model, 'Sketch')
        self._side_split = side_split(scroll, self._tabs, 340, key='chassis')
        root.addWidget(self._side_split, stretch=1)

        for w in (self._kc, self._ay, self._tol, self._tune, self._exag, self._gain,
                  self._msf, self._msr, self._hf, self._hr, self._zf, self._zr):
            w.valueChanged.connect(self._on_input)
        self._ovr.toggled.connect(self._on_input)
        self._apply_dyn.toggled.connect(self._on_input)
        self._reset.clicked.connect(self._copy_model_split)
        self._recalc.clicked.connect(self.refresh)

    # behaviour ------------------------------------------------------------------
    def _ensure_view(self):
        """Build the embedded View3D on first use (same deferral as the Loads page)."""
        if self._v3d is not None:
            return
        try:
            from gui.view3d import View3D
            self._v3d = View3D()
            self._v3d.set_on_view_controls(self._on_v3d_controls)
            self._model_lay.removeWidget(self._view_ph); self._view_ph.hide()
            self._model_lay.insertWidget(1, self._v3d.native, stretch=1)
            wb = float(self._main._car.get('wheelbase_mm', 1530.0)) / 1000.0
            self._v3d.set_camera(azimuth=-52, elevation=20, scale_factor=1.9, center=(0.0, 0.5 * wb, 0.30))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._v3d = None
            self._view_ph.setText(f'3-D view unavailable: {type(e).__name__}: {e}')

    def _on_v3d_controls(self, d):
        v = self._v3d
        if v is None:
            return
        try:
            if 'perspective' in d:
                v.set_perspective(bool(d['perspective']))
            if 'floor' in d and hasattr(v, '_ground'):
                v._ground.visible = bool(d['floor'])
            if 'thickness' in d:
                self._main._car['show_shock_thickness'] = bool(d['thickness'])
                self.refresh()
            v._canvas.update()
        except Exception:
            pass

    def _on_input(self, *_):
        if self._loading:
            return
        self._save_state()
        self._push_to_dynamics()
        self.refresh()

    def _push_to_dynamics(self):
        """The Dynamics panel owns k_c + the 'apply' toggle (the solver reads it there);
        this page only mirrored it, so unchecking the box here changed nothing.  Push
        both into the Dynamics panel; its own signals then re-run the dynamics."""
        dp = getattr(self._main, '_dynamics_panel', None)
        if dp is None or not hasattr(dp, '_chassis_on') or not hasattr(dp, '_chassis_kc'):
            return
        kc_nm_deg = CS.ftlbf_deg_to_nm_deg(float(self._kc.value()))
        if abs(float(dp._chassis_kc.value()) - kc_nm_deg) > 1e-6:
            dp._chassis_kc.setValue(kc_nm_deg)
        on = bool(self._apply_dyn.isChecked())
        if bool(dp._chassis_on.isChecked()) != on:
            dp._chassis_on.setChecked(on)

    def _copy_model_split(self):
        if self._base is None:
            return
        b = self._base
        self._loading = True
        try:
            self._msf.setValue(b.m_sf_kg); self._msr.setValue(b.m_sr_kg)
            self._hf.setValue(b.h_f_m * 1000); self._hr.setValue(b.h_r_m * 1000)
            self._zf.setValue(b.z_f_m * 1000); self._zr.setValue(b.z_r_m * 1000)
        finally:
            self._loading = False
        self._save_state(); self.refresh()

    def refresh(self, *_):
        try:
            st = self._state()
            inp, base = page_inputs(self._main, st)
        except Exception as e:
            self._note.setText(f'model unavailable: {type(e).__name__}: {e}')
            return
        self._inp, self._base = inp, base
        ovr = bool(st['override'])
        for w in (self._msf, self._msr, self._hf, self._hr, self._zf, self._zr):
            w.setEnabled(ovr)
        if not ovr:
            # show the model split in the (disabled) fields without triggering a recompute
            self._loading = True
            try:
                self._msf.setValue(base.m_sf_kg); self._msr.setValue(base.m_sr_kg)
                self._hf.setValue(base.h_f_m * 1000); self._hr.setValue(base.h_r_m * 1000)
                self._zf.setValue(base.z_f_m * 1000); self._zr.setValue(base.z_r_m * 1000)
            finally:
                self._loading = False
        kc_deg = CS.nm_rad_to_nm_deg(inp.k_c_Nm_rad)
        self._kc_nm.setText(f'= {kc_deg:,.0f} N·m/deg = {inp.k_c_Nm_rad:,.0f} N·m/rad')
        summ = CS.summary(inp, a_y_g=float(st['a_y_g']), tune_offset=float(st['tune_pts']) / 100.0,
                          tol_lltd=float(st['tol_pts']) / 100.0)
        self._summ = summ
        res = summ['result']
        veh = self._main._build_dynamics_solver()._veh
        wf = float(veh.front_weight_fraction)
        kf_d, kr_d = CS.nm_rad_to_nm_deg(inp.k_f_Nm_rad), CS.nm_rad_to_nm_deg(inp.k_r_Nm_rad)
        ks_d = CS.nm_rad_to_nm_deg(summ['k_series_Nm_rad'])
        kc_tol = summ['kc_tol_tuned']
        kc_g = CS.kc_for_gain(inp, float(st['gain_target']))
        self._kc_gain = kc_g
        kc_tol_txt = ('any (0)' if kc_tol == 0 else
                      f'{CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(kc_tol)):,.0f} ft·lbf/deg' if np.isfinite(kc_tol) else 'n/a')
        self._note.setText(
            f'k_f {kf_d:.0f}  k_r {kr_d:.0f}  k_series {ks_d:.0f} N·m/deg\n'
            f'front weight {wf * 100:.1f} %   LLTD rigid {summ["lltd_rigid"] * 100:.1f} %   with k_c {summ["lltd"] * 100:.1f} %\n'
            f'gain {summ["gain"]:.3f} (design)  {summ["tuned_gain"]:.3f} (+{st["tune_pts"]:.0f} pt)\n'
            f'twist {res.twist_deg:+.4f}°  frame torque {res.chassis_torque_Nm:+,.0f} N·m  α {res.alpha:+.3f}\n'
            f'k_c for ±{st["tol_pts"]:.1f} pt at +{st["tune_pts"]:.0f} pt: {kc_tol_txt}\n'
            f'k_c for gain {st["gain_target"]:.2f}: {CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(kc_g)):,.0f} ft·lbf/deg '
            f'({CS.nm_rad_to_nm_deg(kc_g):,.0f} N·m/deg)')
        # drawings
        try:
            mem_f, mem_r = bay_members_3d(self._main, 'front'), bay_members_3d(self._main, 'rear')
        except Exception:
            mem_f, mem_r = [], []
        segs_f = [(m['a'], m['b'], m['transverse']) for m in mem_f]
        segs_r = [(m['a'], m['b'], m['transverse']) for m in mem_r]
        meaning = (
            f'WHAT IT MEANS: at {res.a_y_g:.2f} g the front lump makes {res.M_f_Nm:,.0f} N·m and the rear '
            f'{res.M_r_Nm:,.0f} N·m. The frame carries {res.chassis_torque_Nm:+,.0f} N·m of that and twists '
            f'{res.twist_deg:+.4f}°, so the total front share is {summ["lltd"] * 100:.1f} % against '
            f'{summ["lltd_rigid"] * 100:.1f} % rigid. At the design split the chassis does almost nothing; its job '
            f'shows up when you move the bars.')
        try:
            self._scene = model_scene(inp, res, mem_f, mem_r, float(st['exag']))
            self._ensure_view()
            if self._v3d is not None:
                draw_model_3d(self._main, self._v3d, self._scene)
            self._model_sum.setText(meaning)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._model_sum.setText(f'3-D model draw failed: {type(e).__name__}: {e}')
        try:
            draw_model(self._slot_model, inp, res, segs_f, segs_r, float(st['exag']))
            self._slot_model.summary(meaning)
            draw_curve(self._slot_curve, inp, summ, wf, gain_target=float(st['gain_target']))
            tuned_lltd_kc = CS.solve(CS.replace(inp, k_f_Nm_rad=summ['tuned_k_f_Nm_rad'])).lltd_front
            self._slot_curve.summary(
                f'WHAT IT MEANS: moving the bars +{st["tune_pts"]:.0f} pt front-ward asks for a rigid LLTD of '
                f'{summ["tuned_lltd_rigid"] * 100:.1f} %; with k_c = {self._kc.value():,.0f} ft·lbf/deg the car gets '
                f'{tuned_lltd_kc * 100:.1f} % '
                f'(gain {summ["tuned_gain"]:.2f}). Holding that within ±{st["tol_pts"]:.1f} pt needs k_c ≥ {kc_tol_txt}; '
                f'the 5× rule of thumb is {5 * CS.nm_deg_to_ftlbf_deg(kf_d + kr_d):,.0f} ft·lbf/deg.')
        except Exception as e:
            import traceback; traceback.print_exc()
            self._note.setText(self._note.text() + f'\n(draw failed: {type(e).__name__}: {e})')
        self._fill_table(inp, summ, wf)

    def _fill_table(self, inp, summ, wf):
        res = summ['result']
        ft = lambda k_rad: CS.nm_deg_to_ftlbf_deg(CS.nm_rad_to_nm_deg(k_rad))
        rows = [
            ('front lump m_sf', f'{inp.m_sf_kg:.1f}', 'kg', 'sprung mass reacted at the front bay'),
            ('rear lump m_sr', f'{inp.m_sr_kg:.1f}', 'kg', 'sprung mass reacted at the rear bay'),
            ('front lump height h_f', f'{inp.h_f_m * 1000:.1f}', 'mm', ''),
            ('rear lump height h_r', f'{inp.h_r_m * 1000:.1f}', 'mm', ''),
            ('front RC z_F', f'{inp.z_f_m * 1000:.1f}', 'mm', "the Dynamics solver's roll centre at this a_y (L/R mean)"),
            ('rear RC z_R', f'{inp.z_r_m * 1000:.1f}', 'mm', ''),
            ('front arm d_f = h_f − z_F', f'{inp.d_f_m * 1000:.1f}', 'mm', ''),
            ('rear arm d_r = h_r − z_R', f'{inp.d_r_m * 1000:.1f}', 'mm', ''),
            ('front roll stiffness k_f', f'{CS.nm_rad_to_nm_deg(inp.k_f_Nm_rad):.1f}', 'N·m/deg', f'{ft(inp.k_f_Nm_rad):.0f} ft·lbf/deg, springs + bar'),
            ('rear roll stiffness k_r', f'{CS.nm_rad_to_nm_deg(inp.k_r_Nm_rad):.1f}', 'N·m/deg', f'{ft(inp.k_r_Nm_rad):.0f} ft·lbf/deg'),
            ('k_series = k_f k_r / (k_f + k_r)', f'{CS.nm_rad_to_nm_deg(summ["k_series_Nm_rad"]):.1f}', 'N·m/deg', 'what the chassis spring competes with'),
            ('chassis k_c', f'{CS.nm_rad_to_nm_deg(inp.k_c_Nm_rad):.0f}', 'N·m/deg', f'{ft(inp.k_c_Nm_rad):.0f} ft·lbf/deg, hub to hub'),
            ('k_c / (k_f + k_r)', f'{inp.k_c_Nm_rad / (inp.k_f_Nm_rad + inp.k_r_Nm_rad):.2f}', '×', 'the rule-of-thumb ratio'),
            ('lateral a_y', f'{res.a_y_g:.2f}', 'g', ''),
            ('front moment M_f = m_sf a_y d_f', f'{res.M_f_Nm:.1f}', 'N·m', ''),
            ('rear moment M_r = m_sr a_y d_r', f'{res.M_r_Nm:.1f}', 'N·m', ''),
            ('front roll θ_f', f'{res.roll_f_deg:.4f}', '°', ''),
            ('rear roll θ_r', f'{res.roll_r_deg:.4f}', '°', ''),
            ('twist θ_f − θ_r', f'{res.twist_deg:+.4f}', '°', ''),
            ('frame torque k_c·twist', f'{res.chassis_torque_Nm:+.1f}', 'N·m', 'the moment the chassis carries between the bays'),
            ('α = twist / mean roll', f'{res.alpha:+.4f}', '', 'the team rule-of-thumb number'),
            ('elastic LT front / rear', f'{res.elastic_front_N:.1f} / {res.elastic_rear_N:.1f}', 'N', 'one-side delta'),
            ('geometric LT front / rear', f'{res.geometric_front_N:.1f} / {res.geometric_rear_N:.1f}', 'N', 'through the roll centres'),
            ('unsprung LT front / rear', f'{res.unsprung_front_N:.1f} / {res.unsprung_rear_N:.1f}', 'N', ''),
            ('total LT front / rear', f'{res.total_front_N:.1f} / {res.total_rear_N:.1f}', 'N', ''),
            ('LLTD front (with k_c)', f'{summ["lltd"] * 100:.2f}', '%', ''),
            ('LLTD front (rigid)', f'{summ["lltd_rigid"] * 100:.2f}', '%', 'equals the Dynamics solver'),
            ('LLTD front (k_c = 0)', f'{summ["lltd_zero_kc"] * 100:.2f}', '%', 'each bay reacts only its own lump'),
            ('front weight share', f'{wf * 100:.2f}', '%', ''),
            ('gain, design split', f'{summ["gain"]:.3f}', '', '(LLTD(k_c) − LLTD(0)) / (LLTD(∞) − LLTD(0))'),
            (f'gain, +{summ["tune_offset"] * 100:.0f} pt split', f'{summ["tuned_gain"]:.3f}', '', f'rigid LLTD there {summ["tuned_lltd_rigid"] * 100:.1f} %'),
            (f'k_c for ±{summ["tol_lltd"] * 100:.1f} pt at +{summ["tune_offset"] * 100:.0f} pt',
             ('0 (any)' if summ['kc_tol_tuned'] == 0 else f'{CS.nm_rad_to_nm_deg(summ["kc_tol_tuned"]):.0f}' if np.isfinite(summ['kc_tol_tuned']) else 'n/a'),
             'N·m/deg', (f'{ft(summ["kc_tol_tuned"]):.0f} ft·lbf/deg' if np.isfinite(summ['kc_tol_tuned']) and summ['kc_tol_tuned'] > 0 else '')),
            (f'k_c for target gain {self._state()["gain_target"]:.2f}', f'{CS.nm_rad_to_nm_deg(self._kc_gain):.0f}', 'N·m/deg',
             f'{ft(self._kc_gain):.0f} ft·lbf/deg = gain/(1−gain) · k_series'),
            ('5× rule of thumb', f'{5 * CS.nm_rad_to_nm_deg(inp.k_f_Nm_rad + inp.k_r_Nm_rad):.0f}', 'N·m/deg',
             f'{5 * ft(inp.k_f_Nm_rad + inp.k_r_Nm_rad):.0f} ft·lbf/deg'),
        ]
        self._tbl.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(str(txt))
                if c == 1:
                    it.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
                self._tbl.setItem(r, c, it)
        self._formula.setText(
            'node balance (θ = roll of each end of the chassis):\n'
            '  k_f·θ_f + k_c·(θ_f − θ_r) = M_f = m_sf·a_y·(h_f − z_F)\n'
            '  k_r·θ_r + k_c·(θ_r − θ_f) = M_r = m_sr·a_y·(h_r − z_R)\n'
            'elastic transfer = k·θ / track;  geometric = m·a_y·z / track;  unsprung = m_u·a_y·h_u / track\n'
            'closed form (front elastic):\n'
            '  ΔF_ef = (1/t_f)·[ k_f/(k_f + k_r k_c/(k_r+k_c)) · M_f  +  (k_f k_c/(k_f+k_c))/(k_r + k_f k_c/(k_f+k_c)) · M_r ]\n'
            'k_c → ∞ gives k_f/(k_f+k_r) (the Dynamics solver);  k_c → 0 gives each bay its own lump.\n'
            'required k_c: invert LLTD(k_c) for the target share, at the LARGEST bar offset you will tune to.')
