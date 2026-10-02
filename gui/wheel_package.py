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
from vahan import loads as _loads

_G_EARTH = 9.80665

# (label, lat g, lon g, speed rule)   +lon = accel, -lon = braking
# Every load case has a SPEED (user 2026-09-29): the aero downforce on the
# corners is Cl·A·½ρv², so a case without a speed silently had no aero.  The
# speed rule is explicit per case and resolved by case_speed_kph():
#   'corner_radius' : the speed that gives this lateral g on the Dynamics-panel
#                     turn radius,  v = sqrt(|lat g| · g · R)
#   'aero_V_ref'    : the aero package's reference speed (Dynamics panel
#                     V_ref) — a straight-line case has no radius to imply one
#   'static'        : 0 g lateral AND 0 g longitudinal = the car at rest,
#                     0 km/h, no aero (the static reference of every cycle)
SPEED_CORNER, SPEED_VREF, SPEED_STATIC = 'corner_radius', 'aero_V_ref', 'static'
CASES = [('Max cornering 2.0 g', 2.0, 0.0, SPEED_CORNER),
         ('Full braking 1.6 g', 0.0, -1.6, SPEED_VREF),
         ('Full acceleration 1.0 g', 0.0, 1.0, SPEED_VREF),
         ('Cornering 1.4 g + braking 1.0 g', 1.4, -1.0, SPEED_CORNER)]


def case_speed_rule(lat_g, lon_g=0.0) -> str:
    """The speed rule a (lat g, lon g) case falls under: cornering cases sit on
    the panel's turn radius, braking / acceleration at the aero reference
    speed, 0 g / 0 g is the car at rest (static, no aero)."""
    if abs(float(lat_g)) > 1e-9:
        return SPEED_CORNER
    return SPEED_VREF if abs(float(lon_g)) > 1e-9 else SPEED_STATIC


def case_speed_kph(win, lat_g, lon_g=0.0, rule=None) -> float:
    """Speed (km/h) of one load case under its rule (see CASES)."""
    rule = rule or case_speed_rule(lat_g, lon_g)
    dp = win._dynamics_panel
    if rule == SPEED_CORNER:
        R = float(dp._turn_radius.value())
        return float(np.sqrt(abs(float(lat_g)) * _G_EARTH * max(R, 0.0))) * 3.6
    if rule == SPEED_STATIC:
        return 0.0
    return float(dp.get_custom_aero_params()['V_ref_kph'])


def aero_package(win) -> dict | None:
    """The aero package the load cases use, as Cl·A (m²) + CoP + ρ, or None
    when Apply Aero is OFF on the Dynamics panel or the package is empty.
    ONE path with the rest of the app: source 'custom' (and 'solved' before
    the aero-target solver has run) = the panel's F_ref at V_ref split by CoP
    (gui.build_tolerance_page.aero_package); source 'solved' with a result =
    that solver's axle needs at its g on the panel radius, i.e. the same
    downforce the sweep applies at that g, expressed as a Cl·A."""
    if not getattr(win, '_aero_active', False):
        return None
    dp = win._dynamics_panel
    source = dp.get_aero_source() if hasattr(dp, 'get_aero_source') else 'solved'
    r = getattr(win, '_last_aero_result', None)
    if source == 'solved' and r is not None:
        fn, rn = float(r.front_axle_need_N), float(r.rear_axle_need_N)
        g_ref = float(r.lateral_g)
        if fn + rn < 0.1 or g_ref < 0.01:
            return None
        R = float(dp._turn_radius.value())
        rho = float(dp.get_custom_aero_params()['air_density'])
        v2 = g_ref * _G_EARTH * R
        if R <= 0.0 or rho <= 0.0:
            return None
        return dict(cla_m2=2.0 * (fn + rn) / (rho * v2), cop_rear=rn / (fn + rn), rho=rho,
                    label=f'solved aero need {fn + rn:.0f} N at {g_ref:.2f} g on {R:.1f} m '
                          f'({np.sqrt(v2) * 3.6:.1f} km/h), {100.0 * rn / (fn + rn):.0f} % rear')
    from gui.build_tolerance_page import aero_package as _pkg
    pk = _pkg(win)
    if pk['cla_m2'] <= 0.0:
        return None
    return pk


def case_aero(win, lat_g, lon_g=0.0, speed_kph=None) -> dict:
    """The aero load of one case: {'speed_kph', 'rule', 'aero_Fz' (per-corner N
    or None), 'package' (label or None), 'total_N'}.  speed_kph None = the
    case rule.  Respects the panel's Apply Aero toggle (off -> aero_Fz None)."""
    from vahan.aero_ride import aero_per_corner_N
    rule = case_speed_rule(lat_g, lon_g)
    v = float(speed_kph) if speed_kph is not None else case_speed_kph(win, lat_g, lon_g, rule)
    pk = aero_package(win)
    af = aero_per_corner_N(pk['cla_m2'], pk['cop_rear'], pk['rho'], v / 3.6) if (pk and v > 0.0) else None
    return dict(speed_kph=v, rule=rule, aero_Fz=af, package=pk['label'] if pk else None,
                total_N=float(sum(af.values())) if af else 0.0)


def case_aero_text(ca: dict) -> str:
    """One-line human readout of case_aero()."""
    why = {SPEED_CORNER: 'on the Dynamics-panel turn radius', SPEED_VREF: 'the aero reference speed',
           SPEED_STATIC: 'at rest'}.get(ca['rule'], ca['rule'])
    if ca['rule'] == SPEED_STATIC and ca['speed_kph'] <= 0.0:
        return '0 km/h (0 g / 0 g = at rest), no aero'
    if not ca['aero_Fz']:
        return f"{ca['speed_kph']:.1f} km/h ({why}), aero OFF (Apply Aero on the Dynamics panel)"
    a = ca['aero_Fz']
    return (f"{ca['speed_kph']:.1f} km/h ({why}), aero {ca['total_N']:.0f} N = "
            f"{a['FL'] + a['FR']:.0f} N front / {a['RL'] + a['RR']:.0f} N rear "
            f"[{ca['package']}]")


def arb_geometry_fn(win):
    """callable(label, state) -> bellcrank-ARB points in WORLD at that pose
    (drop-link top on the rocker via MainWindow._arb_drop_top_world, arm end and
    bar pivot mirrored for the right side) — what compute_all_corners needs to
    solve the drop-link forces and the bar torque exactly.  None if no ARB data."""
    def _geo(lbl, st):
        arb = win._front_arb if lbl[0] == 'F' else win._rear_arb
        if not arb or arb.get('arb_pivot') is None:
            raise KeyError('no ARB pivot')
        mir = np.array([-1.0, 1.0, 1.0]) if lbl in ('FR', 'RR') else np.ones(3)
        dt = win._arb_drop_top_world(lbl, st)
        if dt is None:
            raise KeyError('no ARB drop top')
        return {'drop_top': np.asarray(dt, float),
                'arm_end': np.asarray(arb['arb_arm_end'], float) * mir,
                'pivot': np.asarray(arb['arb_pivot'], float) * mir}
    return _geo


def compute_case(win, lat_g, lon_g, speed_kph=None):
    """Return (loads, veh, upright_params, result, bp_f, bp_r, solver) for one
    case — THE load path (the Loads page, the Loads panel, the Bearings page
    and the report all call this; ONE MODEL).  The case is solved WITH the
    aero load at the case speed (case_aero: speed_kph None = the case rule,
    Apply Aero off = no aero)."""
    from vahan.loads import compute_all_corners
    solver = win._build_dynamics_solver()
    result = solver.solve(lat_g, lon_g, aero_Fz=case_aero(win, lat_g, lon_g, speed_kph)['aero_Fz'])
    veh = solver._veh
    up = win._loads_panel.get_upright_params()
    bp_f = win._loads_panel.get_brake_params_front()
    bp_r = win._loads_panel.get_brake_params_rear()
    cradle = {'front': win._decoupled_solver(True), 'rear': win._decoupled_solver(False)}
    htb = {'front': win._heave_tbar_solver(True), 'rear': win._heave_tbar_solver(False)}
    loads = compute_all_corners(
        win._solvers, result, brake_params_f=bp_f, brake_params_r=bp_r,
        upright_params_f=up, upright_params_r=up, wheel_radius_m=veh.tire_radius_m,
        cradle_solvers=cradle, heave_tbar_solvers=htb,
        veh=veh, topology=getattr(win, '_topology', None),
        arb_geometry=arb_geometry_fn(win))
    # bp_f/bp_r carry the Seward caliper geometry (pad radius R_pad, bolt spacing
    # l5, pad offset l4); `solver` carries the tyre models for the aligning
    # moment Mz.  Returned so the load view uses the SAME objects (one model).
    return loads, veh, up, result, bp_f, bp_r, solver


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
_C_MOM = (1.00, 0.55, 0.00, 1.0)   # MOMENTS (N·m) — orange, drawn double-headed


def _load_items(win, lat_g, lon_g, only_corner=None, speed_kph=None):
    """Every load in the wheel package, as (point, force_vec, rgba, label).
    speed_kph: the case speed for the aero load (None = the case rule).

    Categories the user asked to keep separate:
      CHASSIS  = the reaction each control-arm/tie/pushrod pushes into its
                 INBOARD frame pickup (tension -> pulls the chassis outboard).
      UPRIGHT  = the loads the UPRIGHT carries: the two ball joints, the two
                 wheel BEARINGS (radial + the outer bearing's axial), the brake
                 CALIPER lugs, and the tyre contact patch (Seward Ch.6).
      ROCKER/ARB = the bellcrank free body (pushrod, spring, ARB drop-link -
                 all AXIAL - and the only moment reaction, at the rocker PIVOT).
    """
    loads, veh, up, res, bp_f, bp_r, dyn = compute_case(win, lat_g, lon_g, speed_kph)
    corners = [only_corner] if only_corner else ['FL', 'FR', 'RL', 'RR']
    items = []
    # The ARB drop-link force closes EACH rocker's own moment about its pivot
    # axis (core: rocker_arb_freebody).  compute_all_corners already split each
    # axle's rocker moment into springs (heave + their share of roll) and bar
    # (its roll share, equal-and-opposite L/R), so no pairing happens here.
    # ONE MODEL: every corner is drawn at the POSE its loads were solved at
    # (ComponentLoads.state, the dynamic roll travel) — never a re-solve at 0.
    def _pose(lbl, c):
        st_ = getattr(c, 'state', None)
        if st_ is None:
            st_ = win._solvers[lbl].solve(float(res.travel.get(lbl, 0.0)) / 1000.0)
        return st_

    for lbl in corners:
        c = loads.get(lbl)
        if c is None:
            continue
        st = _pose(lbl, c)
        wc = np.asarray(st.wheel_center, float)
        spin = np.asarray(st.spin_axis, float)
        spin = spin / max(np.linalg.norm(spin), 1e-9)
        if not getattr(c, 'valid', True):
            # An invalid member solve is NOT drawn as forces (it would be drawn
            # as numbers that mean nothing).  The label says why.
            items.append((wc, np.zeros(3), _C_COMP,
                          f'{lbl} MEMBER SOLVE INVALID · {c.invalid_reason}'))

        # ── CHASSIS: reaction at each inboard pickup (force ON the frame) ──
        #    Vectors straight from the core solver (chassis_forces).  On the arm
        #    that carries the pushrod they are NOT along the leg (the leg also
        #    carries shear) — the label gives both parts.
        MEM = (('uca_front', 'uca_front', 'uca_front_N', 'upper arm front'),
               ('uca_rear', 'uca_rear', 'uca_rear_N', 'upper arm rear'),
               ('lca_front', 'lca_front', 'lca_front_N', 'lower arm front'),
               ('lca_rear', 'lca_rear', 'lca_rear_N', 'lower arm rear'),
               ('tierod', 'tr_inner', 'tierod_N', 'tie / toe rod'),
               ('pushrod', 'pushrod_inner', 'pushrod_N', 'pushrod'))
        cf = getattr(c, 'chassis_forces', {}) or {}
        for key, ik, attr, nm in MEM:
            F = float(getattr(c, attr))
            v = cf.get(key)
            p = np.asarray(getattr(st, ik), float)
            if v is None or not np.all(np.isfinite(v)) or not np.isfinite(F):
                continue
            tag = 'tension' if F >= 0 else 'compression'
            sh = float(getattr(c, attr.replace('_N', '_shear_N'), 0.0) or 0.0)
            sh_txt = f' + {sh:,.0f} N shear (arm carries the pushrod)' if sh > 1.0 else ''
            items.append((p, np.asarray(v, float), _C_TEN if F >= 0 else _C_COMP,
                          f'{lbl} CHASSIS · {nm} pickup · {abs(F):,.0f} N {tag}{sh_txt} '
                          f'(the frame is {"pulled outboard" if F >= 0 else "pushed inboard"})'))

        # ── UPRIGHT: ball joints — force ON the upright, from the core solver ──
        jf = getattr(c, 'joint_forces', {}) or {}
        for key, pt, nm in (('uca', st.uca_outer, 'upper ball joint'),
                            ('lca', st.lca_outer, 'lower ball joint'),
                            ('tie', st.tr_outer, 'tie / toe ball joint')):
            v = jf.get(key)
            if v is None or not np.all(np.isfinite(v)):
                continue
            items.append((np.asarray(pt, float), np.asarray(v, float), _C_UP,
                          f'{lbl} UPRIGHT · {nm} · {np.linalg.norm(v):,.0f} N'))

        # ── UPRIGHT: wheel BEARINGS (radial + axial), along the spin axis ──
        try:
            # Both bearings sit INBOARD of the wheel centre-line (the tyre load is
            # overhung outboard of the pair).  The near/outer bearing is
            # bearing_inboard_offset in; the far/inner one a spacing further.
            # This used to reuse cp_offset (30 mm) as the outer-bearing position,
            # which put the bearing out near the hub face instead of down the
            # spindle where it lives.
            off = float(getattr(up, 'bearing_inboard_offset_mm', 39.4)) / 1000.0
            l1 = float(up.bearing_spacing_mm) / 1000.0
            # INBOARD unit vector along the spindle: toward the car centreline
            # (x = 0).  The spin axis is NOT mirrored left-to-right — it points +x
            # on BOTH sides — so `wc - spin*off` moved inboard on one side and
            # OUTBOARD on the other.  Pick the sign that points toward the
            # centreline on every corner: flip spin when it points to the same
            # lateral side as the wheel.
            nin = spin * (-1.0 if spin[0] * wc[0] > 0 else 1.0)
            p_out = wc + nin * off                    # near (outer) bearing, inboard
            p_in = wc + nin * (off + l1)              # far (inner) bearing
            # Table V/H are up+/FORWARD+; the world frame is +Y = REARWARD, so the
            # world vector is (0, -H, V).  (Drawing +H on +Y pointed every
            # fore-aft bearing arrow backwards.)
            radial_out = np.array([0.0, -float(c.bearing_outer_H), float(c.bearing_outer_V)])
            radial_in = np.array([0.0, -float(c.bearing_inner_H), float(c.bearing_inner_V)])
            items.append((p_out, radial_out, _C_UP,
                          f'{lbl} UPRIGHT · outer bearing RADIAL · {np.linalg.norm(radial_out):,.0f} N'))
            items.append((p_in, radial_in, _C_UP,
                          f'{lbl} UPRIGHT · inner bearing RADIAL · {np.linalg.norm(radial_in):,.0f} N'))
            ax = float(c.bearing_axial_N)              # + = INBOARD thrust
            if abs(ax) > 1.0:                          # only the OUTER bearing takes axial
                items.append((p_out, ax * nin, _C_UP,
                              f'{lbl} UPRIGHT · outer bearing AXIAL · {abs(ax):,.0f} N'))
        except Exception:
            pass

        # ── UPRIGHT: brake CALIPER mount lugs (Seward Ch.6, Fig 6.15) ──
        # Positions and forces come from the core caliper free body
        # (vahan.loads._compute_caliper_bolt_loads via caliper_frame): the SAME
        # clocking and the SAME numbers as the Loads table.  Per lug, force ON the
        # upright = shared tangential F/2 (along the disc motion) -/+ the couple
        # F*l4/l5 along the radial (pad centre offset l4 from the bolt line).
        bt = float(getattr(c, 'brake_torque_Nm', 0.0))
        _bp = bp_f if lbl[0] == 'F' else bp_r
        for pos, F in (getattr(c, 'caliper_lugs', None) or []):
            F = np.asarray(F, float)
            _R_pad = max(float(_bp.pad_radius_mm) / 1000.0, 0.03)
            _Fb = bt / _R_pad
            # l4 is SIGNED (bolt line outside the pad centre -> couple reverses);
            # the label quotes its magnitude, the arrow is the core's signed vector.
            _Hc = _Fb * abs(float(_bp.caliper_l4_mm)) / max(float(_bp.caliper_bolt_spacing_mm), 1e-3)
            items.append((np.asarray(pos, float), F, _C_CAL,
                          f'{lbl} CALIPER · mount lug · V {0.5 * _Fb:,.0f} N + H {_Hc:,.0f} N '
                          f'(brake torque {bt:,.0f} Nm)'))

        # ── UPRIGHT / TYRE: contact-patch load into the hub ──
        #    Same patch point and the same signed world force the member solver
        #    used (vahan.loads.contact_patch_point / patch_force_world).
        _R = float(getattr(veh, 'tire_radius_m', wc[2]))
        patch = _loads.contact_patch_point(wc, spin, _R)
        Fpatch = _loads.patch_force_world(c.Fx_N, c.Fy_N, c.Fz_N)
        items.append((patch, Fpatch, _C_UP,
                      f'{lbl} TYRE · contact patch into hub · {np.linalg.norm(Fpatch):,.0f} N'))

        # ── ROCKER / ARB free body: pushrod, spring, ARB drop-link (axial), the
        #    rocker PIVOT reaction, AND the ARB bar torsion — ALL computed in the
        #    core solver (vahan.loads.rocker_arb_freebody).  This view only
        #    assembles the world-frame geometry (mirroring per corner) and DRAWS
        #    the forces + torsion it reads back; no free-body physics runs here.
        fb = None
        rows = []
        arb_piv = None
        try:
            arb = win._front_arb if lbl[0] == 'F' else win._rear_arb
            hp = win._front_hp if lbl[0] == 'F' else win._rear_hp
            P = np.asarray(st.rocker_pivot, float)
            # The rocker's REAL pivot axis is the one the kinematic solver turns
            # it about (normal of the rocker plate), not the raw axis point —
            # the core spring balance uses the same axis.
            axis = getattr(win._solvers[lbl], '_rocker_axis', None)
            if axis is None:
                mir = np.array([-1.0, 1.0, 1.0]) if lbl in ('FR', 'RR') else 1.0
                axis = (np.asarray(hp['rocker_axis_pt'], float)
                        - np.asarray(hp['rocker_pivot'], float)) * mir
            pi = np.asarray(st.pushrod_inner, float); po = np.asarray(st.pushrod_outer, float)
            sp = np.asarray(st.rocker_spring_pt, float); sc = np.asarray(st.spring_chassis_pt, float)
            dt = win._arb_drop_top_world(lbl, st)
            ae = np.asarray(arb['arb_arm_end'], float)
            if lbl in ('FR', 'RR'):
                ae = ae * np.array([-1.0, 1.0, 1.0])
            arb_piv = arb.get('arb_pivot')
            if arb_piv is not None:
                arb_piv = np.asarray(arb_piv, float)
                if lbl in ('FR', 'RR'):
                    arb_piv = arb_piv * np.array([-1.0, 1.0, 1.0])
            fb = _loads.rocker_arb_freebody(
                pushrod_inner=pi, pushrod_outer=po, pushrod_N=float(c.pushrod_N),
                rocker_pivot=P, rocker_axis=axis,
                rocker_spring_pt=sp, spring_chassis_pt=sc,
                spring_force_N=float(c.spring_force_N),
                arb_drop_top=dt, arb_arm_end=ae, arb_pivot=arb_piv)
            F_push = fb['F_push']; F_spr = fb['F_spr']
            F_arb = fb['F_arb']; F_pivot = fb['F_pivot']
            rows = [(pi, F_push, 'ROCKER · pushrod force', _C_RK),
                    (sp, F_spr, 'ROCKER · spring force', _C_RK),
                    (sc, -F_spr, 'CHASSIS · spring mount', _C_TEN),
                    (dt, F_arb, 'ARB · drop-link (axial)', _C_ARB),
                    (ae, -F_arb, 'ARB · arm end (axial)', _C_ARB),
                    # The rocker PIVOT bolts to the chassis, so its reaction is a
                    # CHASSIS mount load — the frame has to react it.  Coloured as
                    # a chassis load now (was grey/rocker) so it shows when the
                    # chassis group is on, per the request.
                    (P, F_pivot, 'CHASSIS · rocker pivot mount', _C_TEN)]
            if arb_piv is not None:
                # ARB blade reacts the drop-link force into the CHASSIS at its
                # pivot mount (the torsion bar carries the moment; the pivot
                # carries the force) — sum of forces on the blade => +F_arb.
                rows.append((arb_piv, F_arb, 'ARB · chassis pivot mount', _C_ARB))
        except Exception:
            pass
        for pt, v, nm, col in rows:
            if np.linalg.norm(v) < 1.0 or not np.all(np.isfinite(pt)):
                continue
            items.append((np.asarray(pt, float), v, col,
                          f'{lbl} {nm} · {np.linalg.norm(v):,.0f} N'))

        # ── MOMENTS (N·m, drawn double-headed along the moment axis) — ALL five
        #    read from the core solver (vahan.loads.corner_moments).  The load
        #    view only supplies the drawing DIRECTIONS (spin/fore-aft/kingpin/
        #    vertical/bar axes); every VALUE comes from core (ONE MODEL), so the
        #    binder and this view show identical numbers.
        # SIGNED wheel forces straight off the ComponentLoads (the member
        # solver's own inputs), so moments, arrows and member table agree.
        _tm = getattr(dyn, '_tire' if lbl[0] == 'F' else '_tire_rear', None)
        mom = _loads.corner_moments(
            Fx=float(c.Fx_N), Fy=float(c.Fy_N), Fz=float(c.Fz_N),
            camber_deg=float((getattr(res, 'inclination', None) or {}).get(lbl, 0.0)),
            wheel_center=wc, spin_axis=spin,
            lca_outer=st.lca_outer, uca_outer=st.uca_outer,
            tire_model=_tm, freebody=fb,
            wheel_radius_m=float(getattr(veh, 'tire_radius_m', wc[2])))

        # HUB brake/drive torque about the spin axis (reacted by the caliper
        # couple front / the driveshaft rear).
        T_hub = float(mom.get('hub_torque_Nm', 0.0))
        if abs(T_hub) > 1.0:
            # vector = value * axis from core (moment of the patch force about
            # the wheel centre; the scalar alone has no drawing direction)
            items.append((wc, T_hub * np.asarray(mom['hub_torque_axis'], float), _C_MOM,
                          f'{lbl} HUB · brake/drive torque (about axle) · {abs(T_hub):,.0f} N·m'))
        # OVERTURNING moment about the fore-aft axis (reacted by the two wheel
        # bearings as a vertical force couple).
        M_ot = float(mom.get('overturning_Nm', 0.0))
        if abs(M_ot) > 1.0:
            items.append((wc, M_ot * np.asarray(mom['overturning_axis'], float), _C_MOM,
                          f'{lbl} BEARINGS · overturning moment · {abs(M_ot):,.0f} N·m'))
        # STEERING moment about the kingpin axis through the two ball joints.
        M_kp = mom.get('kingpin_Nm')
        if M_kp is not None and abs(M_kp) > 1.0:
            kp_a = np.asarray(st.lca_outer, float)
            k = np.asarray(st.uca_outer, float) - kp_a
            k = k / max(np.linalg.norm(k), 1e-9)
            items.append((kp_a, float(M_kp) * k, _C_MOM,
                          f'{lbl} STEERING · moment about kingpin axis · '
                          f'{abs(M_kp):,.0f} N·m'))
        # TYRE self-aligning torque Mz about the vertical.
        _mz = mom.get('mz_Nm')
        if _mz is not None and abs(_mz) > 1.0:
            items.append((patch, np.array([0.0, 0.0, float(_mz)]), _C_MOM,
                          f'{lbl} TYRE · self-aligning torque Mz · '
                          f'{abs(_mz):,.0f} N·m'))
        # ARB bar TORSION about its own (lateral) axis — the only moment on the
        # car that does not act at the wheel.
        T_arb = mom.get('arb_torsion_Nm')
        if T_arb is not None and abs(T_arb) > 1.0 and arb_piv is not None:
            bar_axis = np.array([1.0, 0.0, 0.0])   # the bar runs laterally
            items.append((arb_piv, float(T_arb) * bar_axis, _C_MOM,
                          f'{lbl} ARB · bar TORSION about its own axis '
                          f'· {abs(T_arb):,.0f} N·m'))
    return items


def load_arrows(win, lat_g, lon_g, mode='resultant', only_corner=None, speed_kph=None):
    """Force-vector arrows for Load mode.  Returns (p, tip, rgba, label).
    mode='resultant' = one arrow per load in the true direction; 'components'
    = split into lateral(X)/fore-aft(Y)/vertical(Z).  speed_kph: the case
    speed for the aero load (None = the case rule)."""
    items = _load_items(win, lat_g, lon_g, only_corner, speed_kph)
    if not items:
        return []
    # Moments (N·m) are a different UNIT from forces (N) — scale each group by its
    # OWN max so a 270 N·m moment isn't dwarfed by a 5 kN force.  Moments are
    # tagged by 'N·m' in the label and drawn double-headed in view3d.
    forces = [it for it in items if 'N·m' not in it[3]]
    moments = [it for it in items if 'N·m' in it[3]]
    fmax = max((float(np.linalg.norm(v)) for _, v, _, _ in forces), default=1.0) or 1.0
    mmax = max((float(np.linalg.norm(v)) for _, v, _, _ in moments), default=1.0) or 1.0
    LMAX, MINL = 0.14, 0.030
    # COMPRESSED (sqrt) length scale: the biggest load is LMAX, but small loads
    # keep a readable length instead of collapsing to a stub.  Hover gives the number.
    def _len(mag, mx):
        return MINL + (LMAX - MINL) * (max(mag, 0.0) / mx) ** 0.5
    AX = ('lateral', 'fore-aft', 'vertical')
    arrows = []
    for p, v, col, lab in forces:
        if mode == 'components':
            for a in range(3):
                comp = float(v[a])
                if abs(comp) < 1.0:
                    continue
                d = np.zeros(3); d[a] = np.sign(comp)
                arrows.append((p, p + d * _len(abs(comp), fmax), col,
                               f'{lab.split(" · ")[0]} · {AX[a]} {comp:+,.0f} N'))
        else:
            mag = float(np.linalg.norm(v))
            if mag < 1.0:
                continue
            arrows.append((p, p + (v / mag) * _len(mag, fmax), col, lab))
    # moments: one arrow along the moment axis (right-hand rule), both modes.
    for p, v, col, lab in moments:
        mag = float(np.linalg.norm(v))
        if mag < 1.0:
            continue
        arrows.append((p, p + (v / mag) * _len(mag, mmax) * 0.9, col, lab))
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
    speed_lab = QLabel(''); speed_lab.setWordWrap(True)
    speed_lab.setStyleSheet('color:#9A9AA2;font-size:11px')
    lay.addWidget(speed_lab)

    def apply():
        lat, lon = next((c[1], c[2]) for c in CASES if c[0] == case_cb.currentText())
        try:
            win._dynamics_panel._lat_g.setValue(lat); win._dynamics_panel._lon_g.setValue(lon)
        except Exception:
            pass
        try:
            speed_lab.setText('Case speed ' + case_aero_text(case_aero(win, lat, lon)))
        except Exception as e:
            speed_lab.setText(f'Case speed: {e}')
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
