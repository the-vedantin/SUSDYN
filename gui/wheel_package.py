"""Wheel Package Load view — the car GUI, isolated to one corner, with the
loads drawn as real 3-D force vectors in the GL viewer (NOT a matplotlib PNG).

compute_case() gets the per-corner loads from the one solved model
(compute_all_corners); load_arrows() turns them into 3-D arrows (the six member
axial forces, the upright ball-joint resultants, the tyre contact patch and the
brake caliper) in either single-resultant or lateral/fore-aft/vertical component
form; build_dialog() is a small control that drives the MAIN 3-D view into
Load mode with a chosen corner isolated.
"""
import numpy as np

# (label, lat g, lon g)   +lon = accel, -lon = braking
CASES = [('Max cornering 2.0 g', 2.0, 0.0),
         ('Full braking 1.6 g', 0.0, -1.6),
         ('Full acceleration 1.0 g', 0.0, 1.0),
         ('Cornering 1.4 g + braking 1.0 g', 1.4, -1.0)]


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


def build_dialog(win):
    """Non-modal control that drives the MAIN 3D GL view into a corner-isolated
    Load view (force vectors + caliper) - NO matplotlib.  This IS the wheel
    package view: the car GUI, one corner, with rendered force vectors."""
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox,
                                 QPushButton, QRadioButton, QButtonGroup)
    dlg = QDialog(win)
    dlg.setWindowTitle('Wheel Package Load View')
    dlg.setModal(False)
    lay = QVBoxLayout(dlg); lay.setSpacing(8)

    def hdr(t):
        l = QLabel(t); l.setStyleSheet('font-weight:bold;color:#b8860b'); return l
    lay.addWidget(hdr('Isolate corner'))
    corner_cb = QComboBox(); corner_cb.addItems(['All corners', 'FL', 'FR', 'RL', 'RR'])
    lay.addWidget(corner_cb)
    lay.addWidget(hdr('Load case'))
    case_cb = QComboBox(); case_cb.addItems([c[0] for c in CASES]); lay.addWidget(case_cb)
    lay.addWidget(hdr('Force vectors'))
    r_res = QRadioButton('Single resultant'); r_comp = QRadioButton('Components (X / Y / Z)')
    r_res.setChecked(True)
    bg = QButtonGroup(dlg); bg.addButton(r_res); bg.addButton(r_comp)
    lay.addWidget(r_res); lay.addWidget(r_comp)
    info = QLabel('Drives the main 3D view: Load mode + corner isolation + force '
                  'vectors + brake caliper.\nwhite = tension · red = compression · '
                  'grey = ball-joint · amber = ground / caliper.')
    info.setWordWrap(True); info.setStyleSheet('color:#888;font-size:11px')
    lay.addWidget(info)

    def apply():
        lat, lon = next((c[1], c[2]) for c in CASES if c[0] == case_cb.currentText())
        try:
            win._dynamics_panel._lat_g.setValue(lat); win._dynamics_panel._lon_g.setValue(lon)
        except Exception:
            pass
        win._car['view_mode'] = 'load'
        win._car['load_vec_mode'] = 'components' if r_comp.isChecked() else 'resultant'
        cc = corner_cb.currentText()
        win._car['wheel_pkg_corner'] = None if cc == 'All corners' else cc
        try:
            win._car_panel._view_mode_combo.setCurrentText('Load')
        except Exception:
            pass
        win._update_3d()
    for wdg in (corner_cb, case_cb):
        wdg.currentTextChanged.connect(lambda _: apply())
    for wdg in (r_res, r_comp):
        wdg.toggled.connect(lambda _: apply())

    def close():
        win._car['wheel_pkg_corner'] = None; win._car['view_mode'] = 'normal'
        try:
            win._car_panel._view_mode_combo.setCurrentText('Normal')
        except Exception:
            pass
        win._update_3d(); dlg.hide()
    cb = QPushButton('Close (back to Normal view)'); cb.clicked.connect(close)
    lay.addWidget(cb)
    apply()
    return dlg
