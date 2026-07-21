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


# Colours by LOAD CATEGORY (colourblind-safe: white/red/amber/grey-white).
# The category is what the user asked to keep distinct.
_C_TEN = (0.92, 0.92, 0.96, 1.0)   # CHASSIS reaction, tension (white)
_C_COMP = (0.95, 0.25, 0.25, 1.0)  # CHASSIS reaction, compression (red)
_C_UP = (0.95, 0.72, 0.12, 1.0)    # UPRIGHT ball joints + bearings (amber)
_C_RK = (0.80, 0.82, 0.92, 1.0)    # ROCKER (pushrod / spring / pivot) grey-white
_C_ARB = (0.20, 0.45, 1.00, 1.0)   # ARB drop-link / arm / chassis mount (solid blue)
_C_CAL = (0.35, 0.90, 0.95, 1.0)   # brake CALIPER mount (cyan — distinct from blue + amber)


def _load_items(win, lat_g, lon_g, only_corner=None):
    """Every load in the wheel package, as (point, force_vec, rgba, label).

    Categories the user asked to keep separate:
      CHASSIS  = the reaction each control-arm/tie/pushrod pushes into its
                 INBOARD frame pickup (tension -> pulls the chassis outboard).
      UPRIGHT  = the loads the UPRIGHT carries: the two ball joints, the two
                 wheel BEARINGS (radial + the outer bearing's axial), the brake
                 CALIPER lugs, and the tyre contact patch (Seward Ch.6).
      ROCKER/ARB = the bellcrank free body (pushrod, spring, ARB drop-link -
                 all AXIAL - and the only moment reaction, at the rocker PIVOT).
    """
    loads, veh, up, res = compute_case(win, lat_g, lon_g)
    corners = [only_corner] if only_corner else ['FL', 'FR', 'RL', 'RR']
    items = []
    for lbl in corners:
        c = loads.get(lbl)
        if c is None:
            continue
        st = win._solvers[lbl].solve(0.)
        wc = np.asarray(st.wheel_center, float)
        spin = np.asarray(st.spin_axis, float)
        spin = spin / max(np.linalg.norm(spin), 1e-9)

        # ── CHASSIS: reaction at each inboard pickup (force ON the frame) ──
        MEM = (('uca_front', 'uca_outer', 'uca_front_N', 'upper arm front'),
               ('uca_rear', 'uca_outer', 'uca_rear_N', 'upper arm rear'),
               ('lca_front', 'lca_outer', 'lca_front_N', 'lower arm front'),
               ('lca_rear', 'lca_outer', 'lca_rear_N', 'lower arm rear'),
               ('tr_inner', 'tr_outer', 'tierod_N', 'tie / toe rod'),
               ('pushrod_inner', 'pushrod_outer', 'pushrod_N', 'pushrod'))
        for ik, ok, attr, nm in MEM:
            F = float(getattr(c, attr))
            u = _u(st, ik, ok)                       # inboard -> outboard
            p = np.asarray(getattr(st, ik), float)
            tag = 'tension' if F >= 0 else 'compression'
            items.append((p, F * u, _C_TEN if F >= 0 else _C_COMP,
                          f'{lbl} CHASSIS · {nm} pickup · {abs(F):,.0f} N {tag} '
                          f'(the frame is {"pulled outboard" if F >= 0 else "pushed inboard"})'))

        # ── UPRIGHT: ball joints (resultant of the arm legs) ──
        R_uca = -(c.uca_front_N * _u(st, 'uca_front', 'uca_outer')
                  + c.uca_rear_N * _u(st, 'uca_rear', 'uca_outer'))
        R_lca = -(c.lca_front_N * _u(st, 'lca_front', 'lca_outer')
                  + c.lca_rear_N * _u(st, 'lca_rear', 'lca_outer'))
        R_tie = -(c.tierod_N * _u(st, 'tr_inner', 'tr_outer'))
        for pt, v, nm in ((st.uca_outer, R_uca, 'upper ball joint'),
                          (st.lca_outer, R_lca, 'lower ball joint'),
                          (st.tr_outer, R_tie, 'tie / toe ball joint')):
            items.append((np.asarray(pt, float), v, _C_UP,
                          f'{lbl} UPRIGHT · {nm} · {np.linalg.norm(v):,.0f} N'))

        # ── UPRIGHT: wheel BEARINGS (radial + axial), along the spin axis ──
        try:
            l2 = float(up.cp_offset_mm) / 1000.0
            l1 = float(up.bearing_spacing_mm) / 1000.0
            p_out = wc - spin * l2                    # inboard of the hub
            p_in = wc - spin * (l2 + l1)
            radial_out = np.array([0.0, float(c.bearing_outer_H), float(c.bearing_outer_V)])
            radial_in = np.array([0.0, float(c.bearing_inner_H), float(c.bearing_inner_V)])
            items.append((p_out, radial_out, _C_UP,
                          f'{lbl} UPRIGHT · outer bearing RADIAL · {np.linalg.norm(radial_out):,.0f} N'))
            items.append((p_in, radial_in, _C_UP,
                          f'{lbl} UPRIGHT · inner bearing RADIAL · {np.linalg.norm(radial_in):,.0f} N'))
            ax = float(c.bearing_axial_N)
            if abs(ax) > 1.0:                          # only the OUTER bearing takes axial
                items.append((p_out, ax * spin, _C_UP,
                              f'{lbl} UPRIGHT · outer bearing AXIAL · {abs(ax):,.0f} N'))
        except Exception:
            pass

        # ── UPRIGHT: brake CALIPER mount lugs (Seward Ch.6, Fig 6.15) ──
        # The single tangential pad-friction force F_brake = brake_torque / R_pad
        # is reacted at the TWO mounting lugs as BOTH a shared tangential force
        # V_brake = F_brake/2 (same direction on both lugs) AND a horizontal
        # COUPLE H_brake = F_brake·l4/l5 (equal-and-opposite on the two lugs),
        # because the pad centre-of-area is offset l4 from the bolt line.  The
        # OLD code applied only the tangential force and dropped H_brake — that
        # was the "no horizontal caliper force" bug.  The caliper MOUNT ANGLE
        # (up.caliper_angle_deg) sets where it sits around the disc.
        bt = float(getattr(c, 'brake_torque_Nm', 0.0))
        if abs(bt) > 1.0:
            vup = np.array([0., 0., 1.]) - np.dot([0., 0., 1.], spin) * spin
            vup = vup / max(np.linalg.norm(vup), 1e-9)          # vertical in wheel plane
            fwd = np.array([0., 1., 0.]) - np.dot([0., 1., 0.], spin) * spin
            fwd = fwd / max(np.linalg.norm(fwd), 1e-9)          # fore-aft in wheel plane
            phi = np.radians(float(getattr(up, 'caliper_angle_deg', 45.0)))
            r_hat = vup * np.cos(phi) + fwd * np.sin(phi)       # radial to the caliper
            t_hat = np.cross(spin, r_hat); t_hat /= max(np.linalg.norm(t_hat), 1e-9)
            R_pad = max(0.5 * float(win._car.get('rotor_dia_mm', 240.0)) / 1000.0 - 0.015, 0.03)
            l4 = float(getattr(up, 'caliper_l4_mm', 22.0)) / 1000.0   # pad-centre to bolt line
            l5 = max(float(getattr(up, 'caliper_l5_mm', 50.0)) / 1000.0, 0.01)  # lug spacing
            F_brake = bt / R_pad
            V_brake = 0.5 * F_brake                             # shared tangential (both lugs)
            H_brake = F_brake * l4 / l5                         # couple (opposite on the lugs)
            bolt_r = max(R_pad - l4, 0.02)
            s = float(np.sign(bt))
            for sgn in (+1.0, -1.0):
                pos = wc + r_hat * bolt_r + t_hat * (sgn * 0.5 * l5)
                F = s * V_brake * t_hat + sgn * H_brake * r_hat
                items.append((pos, F, _C_CAL,
                              f'{lbl} CALIPER · mount lug · V {V_brake:,.0f} N + H {H_brake:,.0f} N '
                              f'(brake torque {bt:,.0f} Nm)'))

        # ── UPRIGHT / TYRE: contact-patch load into the hub ──
        patch = np.array([wc[0], wc[1], 0.0])
        Fpatch = np.array([float(res.Fy.get(lbl, 0.0)),
                           float(res.Fx.get(lbl, 0.0)),
                           float(res.Fz.get(lbl, 0.0))])
        items.append((patch, Fpatch, _C_UP,
                      f'{lbl} TYRE · contact patch into hub · {np.linalg.norm(Fpatch):,.0f} N'))

        # ── ROCKER / ARB free body: pushrod, spring, ARB drop-link (axial),
        #    and the rocker PIVOT reaction (the only moment reaction) ──
        try:
            arb = win._front_arb if lbl[0] == 'F' else win._rear_arb
            hp = win._front_hp if lbl[0] == 'F' else win._rear_hp
            P = np.asarray(st.rocker_pivot, float)
            mir = np.array([-1.0, 1.0, 1.0]) if lbl in ('FR', 'RR') else 1.0
            axis = (np.asarray(hp['rocker_axis_pt'], float)
                    - np.asarray(hp['rocker_pivot'], float)) * mir
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            pi = np.asarray(st.pushrod_inner, float); po = np.asarray(st.pushrod_outer, float)
            u_push = (po - pi) / max(np.linalg.norm(po - pi), 1e-9)
            F_push = float(c.pushrod_N) * u_push
            sp = np.asarray(st.rocker_spring_pt, float); sc = np.asarray(st.spring_chassis_pt, float)
            u_sp = (sc - sp) / max(np.linalg.norm(sc - sp), 1e-9)
            F_spr = -float(c.spring_force_N) * u_sp
            dt = win._arb_drop_top_world(lbl, st)
            ae = np.asarray(arb['arb_arm_end'], float)
            if lbl in ('FR', 'RR'):
                ae = ae * np.array([-1.0, 1.0, 1.0])
            u_arb = (ae - dt) / max(np.linalg.norm(ae - dt), 1e-9)
            m0 = (np.cross(pi - P, F_push) + np.cross(sp - P, F_spr)) @ axis
            lever = np.cross(dt - P, u_arb) @ axis
            F_arb = (-m0 / lever if abs(lever) > 1e-9 else 0.0) * u_arb
            F_pivot = -(F_push + F_spr + F_arb)
            # ARB blade reacts the drop-link force into the CHASSIS at its pivot
            # mount (the torsion bar carries the moment; the pivot carries the
            # force) — sum of forces on the blade => pivot force = +F_arb.
            arb_piv = arb.get('arb_pivot')
            if arb_piv is not None:
                arb_piv = np.asarray(arb_piv, float)
                if lbl in ('FR', 'RR'):
                    arb_piv = arb_piv * np.array([-1.0, 1.0, 1.0])
            rows = [(pi, F_push, 'ROCKER · pushrod force', _C_RK),
                    (sp, F_spr, 'ROCKER · spring force', _C_RK),
                    (sc, -F_spr, 'CHASSIS · spring mount', _C_TEN),
                    (dt, F_arb, 'ARB · drop-link (axial)', _C_ARB),
                    (ae, -F_arb, 'ARB · arm end (axial)', _C_ARB),
                    (P, F_pivot, 'ROCKER · PIVOT reaction (moment)', _C_RK)]
            if arb_piv is not None:
                rows.append((arb_piv, F_arb, 'ARB · chassis pivot mount', _C_ARB))
            for pt, v, nm, col in rows:
                if np.linalg.norm(v) < 1.0 or not np.all(np.isfinite(pt)):
                    continue
                items.append((np.asarray(pt, float), v, col,
                              f'{lbl} {nm} · {np.linalg.norm(v):,.0f} N'))
        except Exception:
            pass
    return items


def load_arrows(win, lat_g, lon_g, mode='resultant', only_corner=None):
    """Force-vector arrows for Load mode.  Returns (p, tip, rgba, label).
    mode='resultant' = one arrow per load in the true direction; 'components'
    = split into lateral(X)/fore-aft(Y)/vertical(Z)."""
    items = _load_items(win, lat_g, lon_g, only_corner)
    if not items:
        return []
    fmax = max(float(np.linalg.norm(v)) for _, v, _, _ in items) or 1.0
    LMAX, MINL = 0.14, 0.030
    # COMPRESSED (sqrt) length scale: the biggest load is LMAX, but small loads
    # (ARB ~130 N, caliper ~900 N) keep a readable length instead of collapsing to
    # a stub next to the 5 kN bearing arrows.  Hover still gives the exact number.
    def _len(mag):
        return MINL + (LMAX - MINL) * (max(mag, 0.0) / fmax) ** 0.5
    AX = ('lateral', 'fore-aft', 'vertical')
    arrows = []
    for p, v, col, lab in items:
        if mode == 'components':
            for a in range(3):
                comp = float(v[a])
                if abs(comp) < 1.0:
                    continue
                d = np.zeros(3); d[a] = np.sign(comp)
                arrows.append((p, p + d * _len(abs(comp)), col,
                               f'{lab.split(" · ")[0]} · {AX[a]} {comp:+,.0f} N'))
        else:
            mag = float(np.linalg.norm(v))
            if mag < 1.0:
                continue
            arrows.append((p, p + (v / mag) * _len(mag), col, lab))
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
