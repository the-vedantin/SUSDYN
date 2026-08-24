"""Packaging design system — hold every parameter, move the suspension points.

The idea: the wheel-side geometry (control-arm pickups, tie rod, wheel centre,
pushrod foot) NEVER moves, so every wheel parameter — camber / caster / KPI /
scrub / toe / bump steer / roll-centre — is held EXACTLY, by construction.
The inboard ACTUATION slice (pushrod inner, rocker, spring, damper chassis
end) plus the ARB assembly is moved as a rigid body with isometries that
provably preserve the rates (motion ratio, ARB rate), or re-tuned back to
them with the lever-scale bisection.  Every candidate is then judged by ONE
oracle — ``validate()`` — which re-measures everything through the SAME
solvers the GUI runs (ONE MODEL: nothing here re-implements physics; it calls
the corner solvers, KinematicMetrics, the dynamics VehicleParams build and
vahan.interference).

Two consumers:
  * MANUAL   — gui/packaging_page.py applies transforms to the live model and
               shows the validate() readout after every move.
  * GENERATOR— generate_solutions() samples transform compositions, validates
               each, and saves the survivors as configs/experiments/pkg_*.vahan
               (NEVER configs/ root — the binder auto-selects the highest
               configs/2027_v* and must not pick up experiments).

All coordinates are chassis-frame METRES (X lateral, Y longitudinal rearward+,
Z up), matching the hardpoint dicts on MainWindow.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .interference import clashes, full_members, connected_for
from .kinematics import KinematicMetrics

# ── which points belong to whom ──────────────────────────────────────────────
# Wheel-locating points: NEVER touched by any transform in this module.  That
# is the whole mechanism by which wheel parameters are held EXACTLY.
WHEEL_HP_KEYS = ('uca_front', 'uca_rear', 'uca_outer',
                 'lca_front', 'lca_rear', 'lca_outer',
                 'tie_rod_inner', 'tie_rod_outer', 'wheel_center',
                 'pushrod_outer')
# Inboard actuation slice: moved rigidly by the transforms.
ACTUATION_HP_KEYS = ('pushrod_inner', 'rocker_pivot', 'rocker_axis_pt',
                     'rocker_spring_pt', 'spring_chassis_pt',
                     'damper_chassis_pt')
ARB_KEYS = ('arb_drop_top', 'arb_arm_end', 'arb_pivot')

_LABEL = {'front': 'FL', 'rear': 'RL'}


# ═════════════════════════════════════════════════════════════════════════════
#  Tolerances — every held parameter, user-editable
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Tolerances:
    """How far a candidate may drift from the captured baseline and still count
    as "the same car".  Units in each field name.  Wheel-side entries are
    normally moot (points identical -> deltas are exactly zero); they exist so
    a hand edit that DOES touch the wheel side is still judged honestly."""
    camber_deg:        float = 0.05
    caster_deg:        float = 0.05
    kpi_deg:           float = 0.05
    toe_deg:           float = 0.05
    bump_steer_deg:    float = 0.02   # toe band over +-25 mm travel
    scrub_mm:          float = 1.0
    trail_mm:          float = 1.0
    rc_height_mm:      float = 2.0
    motion_ratio_pct:  float = 1.0
    arb_rate_pct:      float = 2.0
    coplanar_mm:       float = 3.0    # hard geometric law (<0.1 at design)
    arb_inplane_mm:    float = 3.0    # ARB drop link in rocker plane at static
    triad_deg:         float = 1.0    # bar/blade/drop mutual angles vs baseline
    clash_worsen_mm:   float = 0.25   # standing near-miss may not get worse


# ═════════════════════════════════════════════════════════════════════════════
#  Bundle helpers — one axle's (hp, arb) point dicts
# ═════════════════════════════════════════════════════════════════════════════
def get_bundle(win, axle: str) -> dict:
    """Deep-copy the left-side hardpoint + ARB dicts for 'front' or 'rear'."""
    hp = win._front_hp if axle == 'front' else win._rear_hp
    arb = win._front_arb if axle == 'front' else win._rear_arb
    return {'hp': {k: np.array(v, float) for k, v in hp.items()},
            'arb': {k: np.array(v, float) for k, v in (arb or {}).items()}}


def set_bundle(win, axle: str, bundle: dict, rebuild: bool = True):
    """Write a bundle back into the live MainWindow dicts (+ solver rebuild)."""
    hp = win._front_hp if axle == 'front' else win._rear_hp
    arb = win._front_arb if axle == 'front' else win._rear_arb
    hp.clear(); hp.update({k: np.array(v, float) for k, v in bundle['hp'].items()})
    if arb is not None:
        arb.clear(); arb.update({k: np.array(v, float) for k, v in bundle['arb'].items()})
    if rebuild:
        win._rebuild_solvers()


def _slice_items(bundle):
    """Yield (dict_name, key, point) for every movable actuation/ARB point."""
    for k in ACTUATION_HP_KEYS:
        if k in bundle['hp'] and bundle['hp'][k] is not None:
            yield 'hp', k, np.asarray(bundle['hp'][k], float)
    for k in ARB_KEYS:
        if k in bundle['arb'] and bundle['arb'][k] is not None:
            yield 'arb', k, np.asarray(bundle['arb'][k], float)


def _apply_pointwise(bundle, fn) -> dict:
    """New bundle with fn(point) applied to every actuation/ARB point.
    Wheel-locating points are copied through UNTOUCHED (held exactly)."""
    out = {'hp': {k: np.array(v, float) for k, v in bundle['hp'].items()},
           'arb': {k: np.array(v, float) for k, v in bundle['arb'].items()}}
    for d, k, p in _slice_items(bundle):
        out[d][k] = fn(p)
    return out


def rocker_plane(bundle):
    """(origin, unit normal) of the rocker plane: normal = rocker axis."""
    hp = bundle['hp']
    p0 = np.asarray(hp['rocker_pivot'], float)
    n = np.asarray(hp['rocker_axis_pt'], float) - p0
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        raise ValueError('degenerate rocker axis')
    return p0, n / ln


# ── transform primitives ─────────────────────────────────────────────────────
# Each returns a NEW bundle; wheel-locating points are never touched — that is
# how the wheel parameters are held EXACTLY (see WHEEL_HP_KEYS).

def mirror_about_pushrod_plane(bundle: dict) -> dict:
    """Mirror the actuation slice + ARB about the VERTICAL plane containing the
    pushrod line.  The pushrod itself lies in the plane, so pushrod_outer and
    pushrod_inner are fixed points -> pushrod length/direction unchanged -> all
    rates (MR, ARB) preserved exactly (isometry).  Involution: applying twice
    is the identity."""
    hp = bundle['hp']
    po = np.asarray(hp['pushrod_outer'], float)
    d = np.asarray(hp['pushrod_inner'], float) - po
    horiz = d.copy(); horiz[2] = 0.0
    if np.linalg.norm(horiz) < 1e-9:
        raise ValueError('pushrod is vertical — mirror plane undefined')
    n = np.cross(d, np.array([0.0, 0.0, 1.0]))
    n /= np.linalg.norm(n)
    return _apply_pointwise(bundle, lambda p: p - 2.0 * np.dot(p - po, n) * n)


def rotate_about_pushrod_line(bundle: dict, angle_deg: float) -> dict:
    """Rotate the actuation slice + ARB about the pushrod LINE by angle_deg
    (Rodrigues).  pushrod_inner is ON the axis -> pushrod unchanged -> rates
    preserved exactly (isometry)."""
    hp = bundle['hp']
    po = np.asarray(hp['pushrod_outer'], float)
    a = np.asarray(hp['pushrod_inner'], float) - po
    ln = np.linalg.norm(a)
    if ln < 1e-9:
        raise ValueError('zero-length pushrod')
    a = a / ln
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)

    def rot(p):
        v = p - po
        return po + v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1 - c)
    return _apply_pointwise(bundle, rot)


def translate_actuation(bundle: dict, dxyz_m) -> dict:
    """Rigid translation of the actuation slice + ARB by dxyz (metres).
    Moves pushrod_inner but NOT pushrod_outer, so the pushrod length/direction
    changes -> MR drifts; follow with retune_mr().  A component along the
    rocker-plane normal also moves the plane off the (fixed) pushrod_outer, so
    keep normal translations small (coplanarity law <3 mm)."""
    dv = np.asarray(dxyz_m, float)
    return _apply_pointwise(bundle, lambda p: p + dv)


def scale_rocker_lever(bundle: dict, k: float) -> dict:
    """Scale the (pushrod_inner - rocker_pivot) lever by k, within the rocker
    plane (both ends already lie in it).  This is the ONE knob retune_mr()
    bisects to restore the motion ratio; ARB rate follows MR^2 automatically
    because the drop-link radius on the rocker is untouched."""
    out = {'hp': {kk: np.array(v, float) for kk, v in bundle['hp'].items()},
           'arb': {kk: np.array(v, float) for kk, v in bundle['arb'].items()}}
    pv = np.asarray(out['hp']['rocker_pivot'], float)
    out['hp']['pushrod_inner'] = pv + float(k) * (
        np.asarray(out['hp']['pushrod_inner'], float) - pv)
    return out


def _chain_plane(bundle):
    """Best-fit (centroid, unit normal) of the static actuation chain
    [pushrod_outer, pushrod_inner, rocker_pivot, rocker_spring_pt,
    spring_chassis_pt] — the same 5-point plane the drop-link law uses."""
    hp = bundle['hp']
    pts = np.array([np.asarray(hp[k], float) for k in
                    ('pushrod_outer', 'pushrod_inner', 'rocker_pivot',
                     'rocker_spring_pt', 'spring_chassis_pt')])
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[-1]


def refit_arb(bundle: dict, ref_bundle: dict) -> dict:
    """Re-hang the ARB after a slice transform.

    The torsion bar is chassis-fixed ALONG X, so the ARB assembly cannot
    follow an arbitrary rotation of the actuation slice — its only physical
    freedoms are a rotation about the bar (X) axis and a re-mount translation.
    This keeps the transformed arb_drop_top (it rides ON the rocker — drop
    radius and in-plane position are rate-critical and already correct) and
    re-places arb_arm_end + arb_pivot by taking the BASELINE ARB triangle
    (drop_top/arm_end/pivot — all lengths and triad angles preserved exactly),
    rotating it about X so the arm end lands in the new rocker plane (the
    drop-link law), in either chirality, choosing the pose closest to the
    rigidly-transformed one.  Bar half-length |pivot.x| shifts only by the
    drop point's own x shift (checked by the rates gate downstream)."""
    D0 = np.asarray(ref_bundle['arb']['arb_drop_top'], float)
    E0 = np.asarray(ref_bundle['arb']['arb_arm_end'], float)
    P0 = np.asarray(ref_bundle['arb']['arb_pivot'], float)
    D1 = np.asarray(bundle['arb']['arb_drop_top'], float)
    E_rig = np.asarray(bundle['arb']['arb_arm_end'], float)
    P_rig = np.asarray(bundle['arb']['arb_pivot'], float)
    c1, n1 = _chain_plane(bundle)
    # target: preserve the BASELINE arm-end offset from the chain plane (not
    # snap to exactly 0) so the identity transform round-trips exactly.  The
    # SVD normal's sign is arbitrary between the two planes: try both signs.
    c0, n0 = _chain_plane(ref_bundle)
    off0 = float(np.dot(E0 - c0, n0))

    vE0, vP0 = E0 - D0, P0 - D0
    best = None
    M = np.array([1.0, -1.0, 1.0])   # reflection through xz (chirality flip)
    for vE, vP in ((vE0, vP0), (vE0 * M, vP0 * M)):
        for target in (off0, -off0):
            # solve (D1 + R_x(psi) vE - c1) . n1 = target
            nx, ny, nz = n1
            K = float(np.dot(D1 - c1, n1)) + nx * vE[0] - target
            A = ny * vE[1] + nz * vE[2]
            B = nz * vE[1] - ny * vE[2]
            R = float(np.hypot(A, B))
            phi = float(np.arctan2(B, A))
            if R < 1e-12:
                psis = [0.0]
            elif abs(K) <= R:
                d = float(np.arccos(np.clip(-K / R, -1.0, 1.0)))
                psis = [phi + d, phi - d]
            else:  # no exact in-plane solution — minimize the residual
                psis = [phi + (np.pi if K > 0 else 0.0)]
            for psi in psis:
                c, s = np.cos(psi), np.sin(psi)
                Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                E1 = D1 + Rx @ vE
                P1 = D1 + Rx @ vP
                cost = float(np.linalg.norm(E1 - E_rig) + np.linalg.norm(P1 - P_rig))
                if best is None or cost < best[0]:
                    best = (cost, E1, P1)
    out = {'hp': {k: np.array(v, float) for k, v in bundle['hp'].items()},
           'arb': {k: np.array(v, float) for k, v in bundle['arb'].items()}}
    out['arb']['arb_arm_end'] = best[1]
    out['arb']['arb_pivot'] = best[2]
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Fast solver-level measurements (pure; no GUI state mutated)
# ═════════════════════════════════════════════════════════════════════════════
def _corner_solver(win, axle: str, bundle: dict):
    """Fresh corner solver for a candidate bundle via the ONE construction
    path (MainWindow._build_corner_solvers — static + pure)."""
    label = _LABEL[axle]
    corners = {label: {k: np.asarray(v, float) for k, v in bundle['hp'].items()}}
    return type(win)._build_corner_solvers(corners, win._steer,
                                           win._topology, 0.0)[label]


def solver_mr(solver, dt: float = 0.001) -> float:
    """MR = |d spring_length / d travel| by the SAME +-1 mm central difference
    the dynamics build uses (main_window._build_dynamics_solver)."""
    sp = solver.solve(+dt); sm = solver.solve(-dt)
    return abs(sp.spring_length - sm.spring_length) / (2 * dt)


def retune_mr(win, bundle: dict, axle: str, target_mr: float,
              k_lo: float = 0.5, k_hi: float = 2.0, iters: int = 18) -> tuple:
    """Bisect a scale factor k on the (pushrod_inner - rocker_pivot) lever
    until the corner MR matches target_mr (the known-good re-tune recipe).
    Returns (retuned_bundle, k, mr_achieved).  Raises if the target is not
    bracketed in [k_lo, k_hi]."""
    def mr_of(k):
        return solver_mr(_corner_solver(win, axle, scale_rocker_lever(bundle, k)))
    f_lo = mr_of(k_lo) - target_mr
    f_hi = mr_of(k_hi) - target_mr
    if f_lo * f_hi > 0:
        raise ValueError(f'MR target {target_mr:.4f} not bracketed: '
                         f'k={k_lo}->{f_lo+target_mr:.4f}, k={k_hi}->{f_hi+target_mr:.4f}')
    lo, hi = k_lo, k_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = mr_of(mid) - target_mr
        if f_lo * fm <= 0:
            hi = mid
        else:
            lo, f_lo = mid, fm
    k = 0.5 * (lo + hi)
    out = scale_rocker_lever(bundle, k)
    return out, k, mr_of(k)


def scale_arb_drop_radius(bundle: dict, m: float) -> dict:
    """Scale the drop-link radius on the rocker: move arb_drop_top toward /
    away from rocker_pivot by factor m (both points lie in the rocker plane,
    so the drop point stays in-plane).  THE physical ARB-rate knob (the v35
    lesson: motion ratio = drop-link radius on the rocker)."""
    out = {'hp': {k: np.array(v, float) for k, v in bundle['hp'].items()},
           'arb': {k: np.array(v, float) for k, v in bundle['arb'].items()}}
    pv = np.asarray(out['hp']['rocker_pivot'], float)
    out['arb']['arb_drop_top'] = pv + float(m) * (
        np.asarray(out['arb']['arb_drop_top'], float) - pv)
    return out


def panel_arb_rate(win, axle: str) -> float:
    """ARB wheel rate (N/m) through the ONE panel formula the dynamics use:
    kinematic-derived arm/half/MR pushed into the panel, then get_params."""
    win._refresh_arb_geometry_into_panel()
    return float(win._dynamics_panel.get_params()[f'arb_rate_{axle}_Npm'])


def retune_arb(win, bundle: dict, axle: str, target_rate: float,
               ref_bundle: dict, m_lo: float = 0.5, m_hi: float = 2.0,
               iters: int = 16) -> tuple:
    """Bisect the drop-link radius scale m until the panel ARB wheel rate
    matches target_rate.  Each trial re-hangs the bar (refit_arb) and applies
    the bundle to the window (the rate formula needs the live solvers).
    Returns (bundle, m, rate).  The caller owns restoring window state."""
    def rate_of(m):
        b = refit_arb(scale_arb_drop_radius(bundle, m), ref_bundle)
        set_bundle(win, axle, b)
        return panel_arb_rate(win, axle), b
    r_lo, _ = rate_of(m_lo)
    r_hi, _ = rate_of(m_hi)
    f_lo, f_hi = r_lo - target_rate, r_hi - target_rate
    if f_lo * f_hi > 0:
        raise ValueError(f'ARB rate target {target_rate:.0f} not bracketed: '
                         f'{r_lo:.0f}..{r_hi:.0f} N/m')
    lo, hi = m_lo, m_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm, _ = rate_of(mid)
        fm -= target_rate
        if f_lo * fm <= 0:
            hi = mid
        else:
            lo, f_lo = mid, fm
    m = 0.5 * (lo + hi)
    rate, b = rate_of(m)
    return b, m, rate


# ═════════════════════════════════════════════════════════════════════════════
#  Baseline capture + the ONE validity oracle
# ═════════════════════════════════════════════════════════════════════════════
def travel_range_m(win) -> tuple:
    """(droop, bump) wheel travel in METRES: the config's real damper-derived
    range (motion panel min/max), widened to at least +-25 mm."""
    lo = min(float(win._motion_panel.min_val), -25.0) / 1000.0
    hi = max(float(win._motion_panel.max_val), 25.0) / 1000.0
    return lo, hi


def _axle_wheel_metrics(win, axle: str) -> dict:
    """Wheel-side kinematics for one axle from the LIVE solvers (ONE MODEL):
    static summary + camber/toe curves across the real travel + bump steer."""
    label = _LABEL[axle]
    solver = win._solvers[label]
    m0 = KinematicMetrics(solver.solve(0.0), 'left').summary()
    out = {k: float(m0[k]) for k in
           ('camber_deg', 'toe_deg', 'caster_deg', 'kpi_deg',
            'scrub_radius_mm', 'mechanical_trail_mm', 'roll_center_height_mm')}
    lo, hi = travel_range_m(win)
    stations = np.linspace(lo, hi, 7)
    cam, toe = [], []
    for t in stations:
        try:
            s = KinematicMetrics(solver.solve(float(t)), 'left').summary()
            cam.append(float(s['camber_deg'])); toe.append(float(s['toe_deg']))
        except Exception:
            cam.append(float('nan')); toe.append(float('nan'))
    out['camber_curve_deg'] = cam
    out['toe_curve_deg'] = toe
    out['travel_stations_mm'] = [float(t) * 1000 for t in stations]
    # bump steer: toe band over +-25 mm
    toes = []
    for t in np.linspace(-0.025, 0.025, 5):
        try:
            toes.append(float(KinematicMetrics(solver.solve(float(t)),
                                               'left').summary()['toe_deg']))
        except Exception:
            pass
    out['bump_steer_deg'] = (max(toes) - min(toes)) if len(toes) >= 2 else float('nan')
    return out


def _rates(win) -> dict:
    """MR f/r + ARB rate f/r from the dynamics build (the numbers the roll
    gradient actually runs on)."""
    veh = win._build_dynamics_solver()._veh
    return {'motion_ratio_front': float(veh.motion_ratio_front),
            'motion_ratio_rear': float(veh.motion_ratio_rear),
            'arb_rate_front_Npm': float(veh.arb_rate_front_Npm),
            'arb_rate_rear_Npm': float(veh.arb_rate_rear_Npm)}


def _axle_geometry_laws(win, axle: str) -> dict:
    """Hard geometric laws for one axle, same math as the regression net's
    'design actuation' gate: (1) full-chain coplanarity across travel INCLUDING
    pushrod_outer; (2) ARB drop link in the rocker plane at static (both ends);
    (3) triad angles bar/blade/drop at static."""
    label = _LABEL[axle]
    solver = win._solvers[label]
    arb = win._front_arb if axle == 'front' else win._rear_arb
    out = {}
    # (1) coplanarity: SVD plane through the whole chain at 3 travels
    cop = 0.0
    for t in (-0.025, 0.0, 0.025):
        st = solver.solve(t)
        P = lambda k: np.asarray(getattr(st, k), float) * 1000.0
        pts = np.array([P('pushrod_outer'), P('pushrod_inner'), P('rocker_pivot'),
                        P('rocker_spring_pt'), P('spring_chassis_pt'),
                        np.asarray(arb['arb_drop_top'], float) * 1000.0])
        c = pts.mean(0); _, _, vt = np.linalg.svd(pts - c)
        cop = max(cop, float(np.abs((pts - c) @ vt[-1]).max()))
    out['coplanar_mm'] = cop
    # (2) drop link in-plane at static, both ends, against the chain plane
    st = solver.solve(0.0)
    P = lambda k: np.asarray(getattr(st, k), float) * 1000.0
    pl = np.array([P('pushrod_outer'), P('pushrod_inner'), P('rocker_pivot'),
                   P('rocker_spring_pt'), P('spring_chassis_pt')])
    c0 = pl.mean(0); _, _, vt = np.linalg.svd(pl - c0)
    n = vt[-1]
    dt_off = float(abs((np.asarray(arb['arb_drop_top'], float) * 1000 - c0) @ n))
    ae_off = float(abs((np.asarray(arb['arb_arm_end'], float) * 1000 - c0) @ n))
    out['arb_drop_top_inplane_mm'] = dt_off
    out['arb_arm_end_inplane_mm'] = ae_off
    # (3) triad: torsion bar (X axis) / blade (pivot->arm end) / drop link
    pv = np.asarray(arb['arb_pivot'], float)
    ae = np.asarray(arb['arb_arm_end'], float)
    dtp = np.asarray(arb['arb_drop_top'], float)
    bar = np.array([1.0, 0.0, 0.0])
    blade = ae - pv
    drop = dtp - ae

    def ang(u, v):
        cu = np.linalg.norm(u); cv = np.linalg.norm(v)
        if cu < 1e-12 or cv < 1e-12:
            return float('nan')
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (cu * cv), -1, 1))))
    out['triad_bar_blade_deg'] = ang(bar, blade)
    out['triad_blade_drop_deg'] = ang(blade, drop)
    out['triad_bar_drop_deg'] = ang(bar, drop)
    return out


def _clash_stations(win) -> list:
    """[('droop', t), ('static', 0), ('bump', t)] in metres."""
    lo, hi = travel_range_m(win)
    return [('full droop', lo), ('static', 0.0), ('full bump', hi)]


def _clash_sweep(win) -> dict:
    """Full-member clash lists at static, full bump and full droop, using the
    SAME member set the GUI interference view runs (vahan.interference
    .full_members: arms + tie rod + pushrod + ball-joint spheres + coilover +
    ARB drop link/torsion bar + rocker hardware + driveshaft)."""
    out = {}
    for name, t in _clash_stations(win):
        travels = {l: float(t) for l in ('FL', 'FR', 'RL', 'RR')}
        corners_draw, _ = win._assemble_corners_draw(travels, 0.0)
        # rear half-shafts from the LIVE solved states (ONE MODEL)
        ds = {}
        try:
            import types as _types
            from .driveshaft import package as _ds_package
            rear = {c['label']: _types.SimpleNamespace(
                        wheel_center=np.asarray(c['pts']['wheel_center'], float),
                        spin_axis=np.asarray(c['spin_axis'], float))
                    for c in corners_draw if c['label'] in ('RL', 'RR')
                    and 'wheel_center' in c['pts']}
            pkg = _ds_package(win._car, rear) if len(rear) == 2 else None
            if pkg:
                ds = {l: (np.asarray(pkg[l]['inner'], float),
                          np.asarray(pkg[l]['outer'], float))
                      for l in ('RL', 'RR') if pkg.get(l)}
        except Exception:
            ds = {}
        found = []
        for c in corners_draw:
            label = c['label']
            arb = win._front_arb if label[0] == 'F' else win._rear_arb
            apv = aod = None
            try:
                if arb and 'arb_pivot' in arb:
                    apv = np.asarray(arb['arb_pivot'], float)
                    aod = float(getattr(win._dynamics_panel,
                                        '_arb_OD_f' if label[0] == 'F'
                                        else '_arb_OD_r').value())
            except Exception:
                pass
            mem = full_members(c['pts'], win._car, arb_pivot=apv, arb_od_mm=aod,
                               driveshaft_seg=ds.get(label))
            for cl in clashes(mem, connected=connected_for(label)):
                found.append({'corner': label, 'a': cl['a'], 'b': cl['b'],
                              'gap_mm': cl['gap_mm']})
        out[name] = found
    return out


def capture_baseline(win) -> dict:
    """Snapshot of EVERY held parameter from the currently loaded model.
    This is the reference validate() judges candidates against; the untouched
    model must always validate PASS against its own baseline."""
    base = {'wheel_points': {}, 'wheel_metrics': {}, 'geometry': {}}
    for axle in ('front', 'rear'):
        hp = win._front_hp if axle == 'front' else win._rear_hp
        base['wheel_points'][axle] = {
            k: np.asarray(hp[k], float).tolist()
            for k in WHEEL_HP_KEYS if k in hp and hp[k] is not None}
        base['wheel_metrics'][axle] = _axle_wheel_metrics(win, axle)
        base['geometry'][axle] = _axle_geometry_laws(win, axle)
    base['rates'] = _rates(win)
    base['clashes'] = _clash_sweep(win)
    lo, hi = travel_range_m(win)
    base['travel_mm'] = [lo * 1000, hi * 1000]
    return base


@dataclass
class ValidationResult:
    ok: bool = True
    checks: list = field(default_factory=list)   # dicts: name/axle/value/ref/tol/ok
    clashes: dict = field(default_factory=dict)  # station -> list of clash dicts
    aborted_after: str = ''                      # set when stop_early cut it short

    def add(self, name, axle, value, ref, tol, ok, unit=''):
        self.checks.append({'name': name, 'axle': axle, 'value': value,
                            'ref': ref, 'tol': tol, 'ok': bool(ok), 'unit': unit})
        if not ok:
            self.ok = False

    def failures(self):
        return [c for c in self.checks if not c['ok']]

    def summary(self) -> str:
        n_fail = len(self.failures())
        head = ('PASS — all parameters held' if self.ok
                else f'FAIL — {n_fail} parameter(s) out of tolerance')
        lines = [head]
        for c in self.failures():
            lines.append(f"  X {c['axle']} {c['name']}: {c['value']:+.4f} "
                         f"vs {c['ref']:+.4f} (tol {c['tol']}{c['unit']})")
        return '\n'.join(lines)


def validate(win, baseline: dict, tol: Tolerances = None,
             axles=('front', 'rear'), stop_early: bool = False) -> ValidationResult:
    """THE single validity oracle.  Judges the model CURRENTLY loaded in `win`
    against `baseline` within `tol`.  Cheap checks run first; the full-travel
    clash sweep runs last; with stop_early=True the first failing group aborts
    the rest (generator throughput).

    Checks, in order:
      1. wheel-locating points — if byte-identical to baseline, every wheel
         parameter is held EXACTLY by construction (recorded as zero-delta);
         if ANY moved, the wheel metrics are re-measured and compared.
      2. geometric laws: full-chain coplanarity (<tol.coplanar_mm), ARB drop
         link in the rocker plane at static, triad angles vs baseline.
      3. rates: MR f/r and ARB rate f/r vs baseline (percent).
      4. clash sweep at static / full bump / full droop with the FULL member
         set: no new pair vs baseline, no standing pair worse by
         tol.clash_worsen_mm.
    """
    tol = tol or Tolerances()
    res = ValidationResult()

    # ── 1. wheel side ────────────────────────────────────────────────────────
    for axle in axles:
        hp = win._front_hp if axle == 'front' else win._rear_hp
        moved = 0.0
        for k, ref in baseline['wheel_points'][axle].items():
            cur = np.asarray(hp.get(k), float)
            moved = max(moved, float(np.abs(cur - np.asarray(ref)).max()))
        res.add('wheel points moved', axle, moved * 1000, 0.0, 1e-9, moved < 1e-12,
                ' mm')
        bm = baseline['wheel_metrics'][axle]
        if moved < 1e-12:
            # held EXACTLY by construction — record zero deltas
            for name, t in (('camber_deg', tol.camber_deg),
                            ('caster_deg', tol.caster_deg),
                            ('kpi_deg', tol.kpi_deg), ('toe_deg', tol.toe_deg),
                            ('scrub_radius_mm', tol.scrub_mm),
                            ('mechanical_trail_mm', tol.trail_mm),
                            ('roll_center_height_mm', tol.rc_height_mm),
                            ('bump_steer_deg', tol.bump_steer_deg)):
                res.add(name, axle, float(bm[name]), float(bm[name]), t, True)
        else:
            cm = _axle_wheel_metrics(win, axle)
            for name, t in (('camber_deg', tol.camber_deg),
                            ('caster_deg', tol.caster_deg),
                            ('kpi_deg', tol.kpi_deg), ('toe_deg', tol.toe_deg),
                            ('scrub_radius_mm', tol.scrub_mm),
                            ('mechanical_trail_mm', tol.trail_mm),
                            ('roll_center_height_mm', tol.rc_height_mm),
                            ('bump_steer_deg', tol.bump_steer_deg)):
                res.add(name, axle, float(cm[name]), float(bm[name]), t,
                        abs(float(cm[name]) - float(bm[name])) <= t)
            # curve shapes (camber/toe across the real travel)
            for key, t in (('camber_curve_deg', tol.camber_deg),
                           ('toe_curve_deg', tol.toe_deg)):
                dmax = float(np.nanmax(np.abs(np.asarray(cm[key]) -
                                              np.asarray(bm[key]))))
                res.add(key, axle, dmax, 0.0, t, dmax <= t)
    if stop_early and not res.ok:
        res.aborted_after = 'wheel side'
        return res

    # ── 2. geometric laws ────────────────────────────────────────────────────
    for axle in axles:
        g = _axle_geometry_laws(win, axle)
        bg = baseline['geometry'][axle]
        res.add('coplanar_mm', axle, g['coplanar_mm'], 0.0, tol.coplanar_mm,
                g['coplanar_mm'] <= tol.coplanar_mm, ' mm')
        for k in ('arb_drop_top_inplane_mm', 'arb_arm_end_inplane_mm'):
            res.add(k, axle, g[k], 0.0, tol.arb_inplane_mm,
                    g[k] <= tol.arb_inplane_mm, ' mm')
        for k in ('triad_bar_blade_deg', 'triad_blade_drop_deg',
                  'triad_bar_drop_deg'):
            d = abs(g[k] - bg[k])
            res.add(k, axle, g[k], bg[k], tol.triad_deg, d <= tol.triad_deg,
                    ' deg')
    if stop_early and not res.ok:
        res.aborted_after = 'geometric laws'
        return res

    # ── 3. rates ─────────────────────────────────────────────────────────────
    r = _rates(win)
    br = baseline['rates']
    for key, t, ax in (('motion_ratio_front', tol.motion_ratio_pct, 'front'),
                       ('motion_ratio_rear', tol.motion_ratio_pct, 'rear'),
                       ('arb_rate_front_Npm', tol.arb_rate_pct, 'front'),
                       ('arb_rate_rear_Npm', tol.arb_rate_pct, 'rear')):
        if ax not in axles:
            continue
        ref = br[key]
        pct = abs(r[key] - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0
        res.add(key, ax, r[key], ref, t, pct <= t, ' %')
    if stop_early and not res.ok:
        res.aborted_after = 'rates'
        return res

    # ── 4. clash sweep (most expensive — last) ───────────────────────────────
    cur = _clash_sweep(win)
    res.clashes = cur
    # pool baseline pairs (corner+pair -> worst standing gap)
    base_gap = {}
    for lst in baseline['clashes'].values():
        for cl in lst:
            key = (cl['corner'], frozenset({cl['a'], cl['b']}))
            base_gap[key] = min(base_gap.get(key, 1e9), cl['gap_mm'])
    worst_new = None
    ok = True
    for station, lst in cur.items():
        for cl in lst:
            key = (cl['corner'], frozenset({cl['a'], cl['b']}))
            if key not in base_gap:
                if cl['gap_mm'] < 0.0:          # NEW real clash
                    ok = False
                    worst_new = min(worst_new or 0.0, cl['gap_mm'])
            elif cl['gap_mm'] < base_gap[key] - tol.clash_worsen_mm:
                ok = False                       # standing near-miss got worse
                worst_new = min(worst_new or 0.0, cl['gap_mm'] - base_gap[key])
    res.add('clash sweep (static/bump/droop)', 'both',
            0.0 if worst_new is None else worst_new, 0.0,
            tol.clash_worsen_mm, ok, ' mm')
    return res


# ═════════════════════════════════════════════════════════════════════════════
#  Packaging score — how far the actuation envelope stays from a keep-out plane
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_KEEPOUT = {'point_mm': [0.0, 123.0, 0.0], 'normal': [0.0, -1.0, 0.0],
                   'name': 'front-hoop plane y=+123 mm'}


def packaging_score(bundle: dict, keep_out: dict = None) -> float:
    """Min clearance (mm) of the actuation/ARB points to the keep-out plane.
    `normal` points toward the ALLOWED side; positive score = clear."""
    ko = keep_out or DEFAULT_KEEPOUT
    p0 = np.asarray(ko['point_mm'], float) / 1000.0
    n = np.asarray(ko['normal'], float)
    n = n / np.linalg.norm(n)
    return float(min(np.dot(p - p0, n) for _, _, p in _slice_items(bundle))) * 1000.0


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATOR — hundreds of valid packaging solutions
# ═════════════════════════════════════════════════════════════════════════════
def candidate_grid(rng, n: int) -> list:
    """Sample n transform recipes: mirror on/off x rotation angle x small rigid
    translation (in metres).  Includes the identity-ish neighbourhood."""
    recipes = []
    for _ in range(n):
        recipes.append({
            'mirror': bool(rng.integers(0, 2)),
            'rotate_deg': float(rng.uniform(-15.0, 15.0)),
            # x shifts the whole slice laterally: the ARB bar half-length
            # |pivot.x| moves with it, and the bar rate ~ 1/half-length, so
            # big x excursions die at the rates gate — sample it tighter.
            'translate_mm': [float(rng.uniform(-3.0, 3.0)),
                             float(rng.uniform(-12.0, 12.0)),
                             float(rng.uniform(-12.0, 12.0))],
        })
    return recipes


def apply_recipe(win, base_bundle: dict, axle: str, recipe: dict,
                 target_mr: float, target_arb: float = None) -> tuple:
    """base bundle -> (candidate bundle, lever k, drop-radius m) via
    mirror/rotate/translate + ARB re-hang + MR re-tune (lever bisection) +
    ARB-rate re-tune (drop-radius bisection).  Raises on degenerate geometry
    or an unbracketable target.  Leaves the window on the candidate when
    target_arb is given (retune_arb applies it live); caller restores."""
    b = base_bundle
    if recipe.get('mirror'):
        b = mirror_about_pushrod_plane(b)
    if abs(recipe.get('rotate_deg', 0.0)) > 1e-9:
        b = rotate_about_pushrod_line(b, recipe['rotate_deg'])
    tr = np.asarray(recipe.get('translate_mm', [0, 0, 0]), float) / 1000.0
    if np.linalg.norm(tr) > 1e-12:
        b = translate_actuation(b, tr)
    if b is not base_bundle:
        b = refit_arb(b, base_bundle)   # bar is chassis-fixed along X
    k = 1.0
    mr = solver_mr(_corner_solver(win, axle, b))
    if abs(mr - target_mr) / target_mr > 1e-4:
        b, k, mr = retune_mr(win, b, axle, target_mr)
    m = 1.0
    if target_arb is not None:
        set_bundle(win, axle, b)
        if abs(panel_arb_rate(win, axle) - target_arb) / target_arb > 1e-3:
            b, m, _ = retune_arb(win, b, axle, target_arb, base_bundle)
    return b, k, m


def generate_solutions(win, axle: str = 'front', n_target: int = 100,
                       seed: int = 0, keep_out: dict = None,
                       tol: Tolerances = None, out_dir: str = None,
                       max_candidates: int = 2000, progress=None,
                       save: bool = True) -> dict:
    """Produce up to n_target VALID packaging solutions for one axle.

    Samples transform recipes, re-tunes MR, then runs the ONE oracle
    (validate) with stop_early.  Survivors are ranked by packaging_score and
    (when save=True) written to configs/experiments/pkg_NNN.vahan + an index
    JSON.  The window is restored to its original state afterwards.

    progress: optional callable(done, tried, found) -> False to cancel.
    Returns {'solutions': [...], 'tried': int, 'elapsed_s': float, ...}.
    """
    tol = tol or Tolerances()
    ko = keep_out or DEFAULT_KEEPOUT
    rng = np.random.default_rng(seed)
    baseline = capture_baseline(win)
    base_bundle = get_bundle(win, axle)
    other = 'rear' if axle == 'front' else 'front'
    other_bundle = get_bundle(win, other)
    target_mr = baseline['rates'][f'motion_ratio_{axle}']
    target_arb = baseline['rates'][f'arb_rate_{axle}_Npm']

    out_dir = out_dir or os.path.join('configs', 'experiments')
    if save:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    solutions, tried = [], 0
    fail_stage = {}
    seen = set()   # de-duplicate near-identical point sets
    try:
        while len(solutions) < n_target and tried < max_candidates:
            recipe = candidate_grid(rng, 1)[0]
            tried += 1
            try:
                cand, k, m = apply_recipe(win, base_bundle, axle, recipe,
                                          target_mr, target_arb)
            except Exception:
                set_bundle(win, axle, base_bundle)   # retune may have left state
                fail_stage['retune'] = fail_stage.get('retune', 0) + 1
                continue
            # distinctness: round the movable points to 0.5 mm
            sig = tuple(np.round(p * 2000).astype(int).tobytes()
                        for _, _, p in _slice_items(cand))
            if sig in seen:
                continue
            seen.add(sig)
            set_bundle(win, axle, cand)
            try:
                res = validate(win, baseline, tol, axles=(axle,), stop_early=True)
            except Exception:
                res = None
            if res is not None and not res.ok:
                stage = res.aborted_after or 'clash sweep'
                fail_stage[stage] = fail_stage.get(stage, 0) + 1
            if res is not None and res.ok:
                score = packaging_score(cand, ko)
                deltas = {c['name']: c['value'] - c['ref']
                          for c in res.checks if isinstance(c['value'], float)
                          and isinstance(c['ref'], float)}
                sol = {'recipe': recipe, 'lever_k': k, 'drop_radius_m': m,
                       'score_mm': score, 'param_deltas': deltas}
                if save:
                    fn = os.path.join(out_dir, f'pkg_{len(solutions):03d}.vahan')
                    win._save_project_to_path(fn)
                    sol['file'] = fn
                solutions.append(sol)
            # restore for the next candidate
            set_bundle(win, axle, base_bundle)
            if progress is not None:
                if progress(tried, tried, len(solutions)) is False:
                    break
    finally:
        set_bundle(win, axle, base_bundle)
        set_bundle(win, other, other_bundle)
    solutions.sort(key=lambda s: -s['score_mm'])
    result = {'axle': axle, 'tried': tried, 'found': len(solutions),
              'fail_stage': fail_stage, 'elapsed_s': time.time() - t0,
              'keep_out': ko, 'tolerances': asdict(tol),
              'solutions': solutions}
    if save:
        with open(os.path.join(out_dir, 'pkg_index.json'), 'w') as f:
            json.dump(result, f, indent=1, default=float)
    return result
