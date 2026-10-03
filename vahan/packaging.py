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

from .interference import (clashes, full_members, connected_for, rim_barrel_gap,
                           cross_corner_clashes, rocker_plate_gaps,
                           rocker_plate_style_for, rocker_pr_full_length_fork_for,
                           rocker_pr_full_length_fork_clear_gap_for)
from .interference import arb_blade_dogleg_params_for
from .interference import arb_member_kwargs
from .interference import rocker_pr_full_length_fork_jog_for
from .interference import rocker_plate_physical_options_for
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
    rocker_axis_deg:   float = 1e-4   # Rule 02: axis normal to physical plate
    arb_inplane_mm:    float = 3.0    # ARB drop link in rocker plane at static
    triad_deg:         float = 1.0    # bar/blade/drop mutual angles from 90 degrees
    clash_worsen_mm:   float = 0.25   # standing near-miss may not get worse
    arb_mount_shift_mm: float = 100.0 # bar re-mount may move at most this far
                                      # from the baseline chassis mount (a bar
                                      # only mounts where structure exists)


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


def _physical_rocker_plane(hp, outboard_sign: float = 1.0):
    """Origin and stable unit normal of the three-attach rocker plate."""
    pivot = np.asarray(hp['rocker_pivot'], float)
    pushrod = np.asarray(hp['pushrod_inner'], float)
    spring = np.asarray(hp['rocker_spring_pt'], float)
    normal = np.cross(pushrod - pivot, spring - pivot)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-12:
        return pivot, np.full(3, np.nan)
    normal /= nn
    if float(normal @ np.array([float(outboard_sign), 0.0, 0.0])) < 0.0:
        normal = -normal
    return pivot, normal


def actuation_chain_plate_metrics(states, outboard_sign: float = 1.0,
                                  rocker_axis=None, arb=None) -> dict:
    """Rule 01/02 metrics for one corner at its static design position.

    ``states`` retains its historical iterable interface, but only the state
    nearest zero travel is acceptance data.  The physical plane is the corner's
    static rocker plane through pivot, pushrod-inner and spring pickup.  The
    pushrod outer, both spring eyes and, when ``arb`` is supplied, both drop-link
    endpoints are measured against that same plane.  Left and right corners are
    evaluated independently by :func:`_axle_geometry_laws`; wheel travel does
    not redefine or extend this static assembly rule.
    """
    states = list(states)
    if not states:
        raise ValueError('at least one solved state is required')

    def point(state, key):
        return np.asarray(state[key] if isinstance(state, dict)
                          else getattr(state, key), float)

    static_travel, static = min(states, key=lambda item: abs(float(item[0])))
    if abs(float(static_travel)) > 1e-12:
        raise ValueError('static actuation-plane metrics require an explicit zero-travel state')
    static_hp = {k: point(static, k) for k in
                 ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')}
    origin, normal = _physical_rocker_plane(static_hp, outboard_sign)
    keys = ('pushrod_outer', 'pushrod_inner', 'rocker_pivot',
            'rocker_spring_pt', 'spring_chassis_pt')
    named = [(k, point(static, k)) for k in keys]
    if arb is not None:
        named.extend((('arb_drop_top', np.asarray(arb['arb_drop_top'], float)),
                      ('arb_arm_end', np.asarray(arb['arb_arm_end'], float))))
    if not np.all(np.isfinite(normal)):
        return {'coplanar_mm': float('nan'), 'coplanar_static_mm': float('nan'),
                'chain_best_fit_mm': float('nan'),
                'rocker_axis_normal_error_deg': float('nan'),
                'worst_point': None, 'worst_travel_m': float('nan'),
                'static_signed_mm': {k: float('nan') for k, _ in named}}
    names = [k for k, _ in named]
    chain = np.array([p for _, p in named])
    signed = (chain - origin) @ normal * 1000.0
    static_signed = {k: float(v) for k, v in zip(names, signed)}
    i = int(np.argmax(np.abs(signed)))
    worst = (abs(float(signed[i])), names[i], 0.0)
    c = chain.mean(0)
    _, _, vt = np.linalg.svd(chain - c)
    best_fit = float(np.abs((chain - c) @ vt[-1]).max()) * 1000.0
    axis = (point(static, 'rocker_axis_pt') - point(static, 'rocker_pivot')
            if rocker_axis is None else np.asarray(rocker_axis, float))
    axis_n = float(np.linalg.norm(axis))
    axis_error = (float('nan') if axis_n < 1e-12 else
                  float(np.degrees(np.arccos(np.clip(abs(float((axis / axis_n) @ normal)),
                                                     -1.0, 1.0)))))
    return {'coplanar_mm': worst[0],
            'coplanar_static_mm': worst[0],
            'chain_best_fit_mm': best_fit,
            'rocker_axis_normal_error_deg': axis_error,
            'worst_point': worst[1], 'worst_travel_m': worst[2],
            'static_signed_mm': static_signed}


def arb_drop_link_plate_metrics(hp, arb, outboard_sign: float = 1.0) -> dict:
    """Measure a drop link against the physical rocker plate.

    The plate is the plane through ``rocker_pivot``, ``pushrod_inner`` and
    ``rocker_spring_pt``.  Inputs are in metres; reported distances are in
    millimetres.  The normal is oriented toward the corner's outboard X
    direction so endpoint signs are stable between left and right corners.
    ``direction_deg`` is the signed angle from the plate of the vector from
    drop top to arm end (zero means its direction is parallel to the plate).

    Project metadata is deliberately not an input.  A spacer or waiver may
    describe noncompliant hardware, but cannot change Rule 04's geometry.
    """
    pivot, normal = _physical_rocker_plane(hp, outboard_sign)
    if not np.all(np.isfinite(normal)):
        return {'drop_top_signed_mm': float('nan'),
                'arm_end_signed_mm': float('nan'),
                'direction_deg': float('nan')}
    drop_top = np.asarray(arb['arb_drop_top'], float)
    arm_end = np.asarray(arb['arb_arm_end'], float)
    link = arm_end - drop_top
    ln = float(np.linalg.norm(link))
    direction = (float('nan') if ln < 1e-12 else
                 float(np.degrees(np.arcsin(np.clip(float((link / ln) @ normal),
                                                    -1.0, 1.0)))))
    return {'drop_top_signed_mm': float((drop_top - pivot) @ normal) * 1000.0,
            'arm_end_signed_mm': float((arm_end - pivot) @ normal) * 1000.0,
            'direction_deg': direction}


def arb_drop_link_plate_compliant(metrics: dict, tolerance_mm: float,
                                  control_arm_arb: bool = False) -> bool:
    """Rule 04 acceptance from the shared physical-plate measurements."""
    if control_arm_arb:
        return True
    distances = (metrics.get('drop_top_signed_mm'),
                 metrics.get('arm_end_signed_mm'))
    return all(v is not None and np.isfinite(v) and abs(float(v)) <= tolerance_mm
               for v in distances)


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


def refit_arb_variants(bundle: dict, ref_bundle: dict) -> list:
    """EVERY discrete re-hang branch of refit_arb (2 chiralities x 2 plane-
    offset signs x up-to-2 rotation roots -> up to 8 bundles), sorted by
    closeness to the rigidly-transformed pose.  refit_arb() returns just the
    first of these; a packaging search may need the others — a different
    branch places the BAR on the other side of the rocker without touching
    the drop-top or the drive geometry."""
    D0 = np.asarray(ref_bundle['arb']['arb_drop_top'], float)
    E0 = np.asarray(ref_bundle['arb']['arb_arm_end'], float)
    P0 = np.asarray(ref_bundle['arb']['arb_pivot'], float)
    D1 = np.asarray(bundle['arb']['arb_drop_top'], float)
    E_rig = np.asarray(bundle['arb']['arb_arm_end'], float)
    P_rig = np.asarray(bundle['arb']['arb_pivot'], float)
    c1, n1 = _chain_plane(bundle)
    c0, n0 = _chain_plane(ref_bundle)
    off0 = float(np.dot(E0 - c0, n0))
    vE0, vP0 = E0 - D0, P0 - D0
    M = np.array([1.0, -1.0, 1.0])
    cands = []
    for vE, vP in ((vE0, vP0), (vE0 * M, vP0 * M)):
        for target in (off0, -off0):
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
            else:
                psis = [phi + (np.pi if K > 0 else 0.0)]
            for psi in psis:
                c, sn = np.cos(psi), np.sin(psi)
                Rx = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
                E1 = D1 + Rx @ vE
                P1 = D1 + Rx @ vP
                cost = float(np.linalg.norm(E1 - E_rig)
                             + np.linalg.norm(P1 - P_rig))
                out = {'hp': {k: np.array(v, float)
                              for k, v in bundle['hp'].items()},
                       'arb': {k: np.array(v, float)
                               for k, v in bundle['arb'].items()}}
                out['arb']['arb_arm_end'] = E1
                out['arb']['arb_pivot'] = P1
                cands.append((cost, out))
    cands.sort(key=lambda t: t[0])
    return [b for _c, b in cands]


def retune_arb_blade(win, bundle: dict, axle: str, target_rate: float,
                     ref_bundle: dict, s_lo: float = 0.5, s_hi: float = 2.5,
                     iters: int = 18) -> tuple:
    """Bisect the BLADE length (arm_end scaled about arb_pivot in the
    reference triangle) until the panel ARB wheel rate matches target_rate.
    The alternative knob to retune_arb(): a LONGER blade softens the rate
    without shrinking the drop radius into the rocker bearing.  Blade
    direction is preserved, so the triad angles survive; the drop link is an
    adjustable rod, so its length change is real hardware.
    Returns (bundle, s, rate).  Caller owns restoring window state."""
    E0 = np.asarray(ref_bundle['arb']['arb_arm_end'], float)
    P0 = np.asarray(ref_bundle['arb']['arb_pivot'], float)

    def rate_of(sc):
        ref2 = {'hp': {k: np.array(v, float)
                       for k, v in ref_bundle['hp'].items()},
                'arb': {k: np.array(v, float)
                        for k, v in ref_bundle['arb'].items()}}
        ref2['arb']['arb_arm_end'] = P0 + sc * (E0 - P0)
        b = refit_arb(bundle, ref2)
        set_bundle(win, axle, b)
        return panel_arb_rate(win, axle), b
    r_lo, _ = rate_of(s_lo)
    r_hi, _ = rate_of(s_hi)
    f_lo = r_lo - target_rate
    if f_lo * (r_hi - target_rate) > 0:
        raise ValueError(f'ARB rate target {target_rate:.0f} not bracketed by '
                         f'blade length: {r_lo:.0f}..{r_hi:.0f} N/m')
    lo, hi = s_lo, s_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm, _b = rate_of(mid)
        fm -= target_rate
        if f_lo * fm <= 0:
            hi = mid
        else:
            lo, f_lo = mid, fm
    sc = 0.5 * (lo + hi)
    rate, b = rate_of(sc)
    return b, sc, rate


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
               ref_bundle: dict, m_lo: float = 0.15, m_hi: float = 3.5,
               iters: int = 18) -> tuple:
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
    """Hard physical geometry laws shared with the design-actuation net.

    Rules 01/02/04 are static assembly checks performed independently on the
    left and right corners. Rule 03 checks the bar/blade/drop triad.
    """
    arb = win._front_arb if axle == 'front' else win._rear_arb
    out = {}
    labels = ('FL', 'FR') if axle == 'front' else ('RL', 'RR')
    corner_metrics, plate_metrics, states = {}, {}, {}
    for label in labels:
        solver = win._solvers[label]
        st = solver.solve(0.0); states[label] = st
        mirror = (np.array([1.0, 1.0, 1.0]) if label[1] == 'L'
                  else np.array([-1.0, 1.0, 1.0]))
        corner_arb = {k: np.asarray(v, float) * mirror for k, v in arb.items()
                      if k in ('arb_drop_top', 'arb_arm_end', 'arb_pivot')}
        corner_metrics[label] = actuation_chain_plate_metrics(
            [(0.0, st)], outboard_sign=mirror[0], arb=corner_arb,
            rocker_axis=np.asarray(solver.hp.rocker_axis_pt, float)
            - np.asarray(solver.hp.rocker_pivot, float))
        static_hp = {k: np.asarray(getattr(st, k), float) for k in
                     ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')}
        plate_metrics[label] = arb_drop_link_plate_metrics(
            static_hp, corner_arb, outboard_sign=mirror[0])

    worst_label = max(labels, key=lambda x: corner_metrics[x]['coplanar_mm'])
    out.update(corner_metrics[worst_label])
    out['coplanar_mm'] = max(corner_metrics[x]['coplanar_mm'] for x in labels)
    out['coplanar_static_mm'] = out['coplanar_mm']
    out['chain_best_fit_mm'] = max(corner_metrics[x]['chain_best_fit_mm'] for x in labels)
    out['rocker_axis_normal_error_deg'] = max(
        corner_metrics[x]['rocker_axis_normal_error_deg'] for x in labels)
    out['corner_static_mm'] = {x: corner_metrics[x]['coplanar_mm'] for x in labels}
    out['corner_axis_error_deg'] = {
        x: corner_metrics[x]['rocker_axis_normal_error_deg'] for x in labels}

    # Rule 04 uses each corner's own physical static rocker plane. Only an
    # explicit CONTROL_ARM topology is exempt; metadata remains diagnostic.
    st = states[labels[0]]
    P = lambda k: np.asarray(getattr(st, k), float) * 1000.0
    axle_topology = getattr(getattr(win, '_topology', None), axle, None)
    arb_type = getattr(getattr(axle_topology, 'arb_type', None), 'value', None)
    is_bottom_arb = (arb_type == 'control_arm')
    out['arb_is_bottom'] = bool(is_bottom_arb)
    def signed_worst(key):
        return max((plate_metrics[x][key] for x in labels), key=abs)
    out['arb_drop_top_plate_signed_mm'] = signed_worst('drop_top_signed_mm')
    out['arb_arm_end_plate_signed_mm'] = signed_worst('arm_end_signed_mm')
    out['arb_link_to_plate_deg'] = signed_worst('direction_deg')
    # Retain the established absolute-distance keys for existing consumers.
    out['arb_drop_top_inplane_mm'] = max(
        abs(plate_metrics[x]['drop_top_signed_mm']) for x in labels)
    out['arb_arm_end_inplane_mm'] = max(
        abs(plate_metrics[x]['arm_end_signed_mm']) for x in labels)
    out['arb_drop_standoff_mm'] = float(
        getattr(win, '_car', {}).get(f'{axle}_arb_drop_standoff_mm', 0.0) or 0.0)
    out['arb_link_plane_offset_mm'] = max(
        (0.5 * (plate_metrics[x]['drop_top_signed_mm']
                + plate_metrics[x]['arm_end_signed_mm']) for x in labels), key=abs)
    # (4) triad: torsion bar (X axis) / blade (pivot->arm end) / drop link
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
    # Preserve waiver metadata for diagnostics only; it never changes compliance.
    out['arb_inplane_waived'] = bool(getattr(win, '_car', {}).get(f'{axle}_arb_rule04_waiver'))
    out['triad_bar_blade_deg'] = ang(bar, blade)
    out['triad_blade_drop_deg'] = ang(blade, drop)
    out['triad_bar_drop_deg'] = ang(bar, drop)
    # (4) damper fore-aft cant at static (|Y rocker_spring_pt - Y spring_chassis_pt|).
    #     The net limits it to 25 mm on the REAR (damper lies across the car);
    #     the front damper runs fore-aft by design, so callers gate rear only.
    out['damper_cant_fore_aft_mm'] = float(abs(P('rocker_spring_pt')[1] - P('spring_chassis_pt')[1]))
    return out


def front_arb_hoop_line_gap_mm(win, margin_travel: bool = True) -> dict:
    """Rule 19 (user, 2026-09-14): every FRONT ARB member (torsion bar, blades, drop
    links, drop-top rod ends) stays AHEAD of the front-hoop line — the line through
    the LCA-aft and UCA-aft chassis pickups in the side view, extended upward.
    For each ARB point the line's Y at that height is Y_line(Z) = y_lca + (Z - z_lca)
    * (y_uca - y_lca) / (z_uca - z_lca) (a vertical line when the two pickups share
    Y, as on v101+).  gap = Y_line(Z) - (Y_point + radius), mm, positive = ahead.
    Evaluated on the assembled corners at droop / static / bump (the arm end and drop
    top move with the rocker and the bar).  Returns {'gap_mm', 'worst', 'line'}."""
    hp = win._front_hp
    yl, zl = (float(hp['lca_rear'][1]) * 1000.0, float(hp['lca_rear'][2]) * 1000.0)
    yu, zu = (float(hp['uca_rear'][1]) * 1000.0, float(hp['uca_rear'][2]) * 1000.0)
    slope = (yu - yl) / (zu - zl) if abs(zu - zl) > 1e-6 else 0.0
    y_line = lambda z: yl + (z - zl) * slope
    od = float(win._dynamics_panel._arb_OD_f.value())
    apv = np.asarray(win._front_arb['arb_pivot'], float) * 1000.0
    lo, hi = travel_range_m(win); lo = min(lo, -0.025); hi = max(hi, 0.025)
    RE, LINK = 8.0, 6.35
    worst = (float('inf'), '')
    stations = [(lo, 'droop'), (0.0, 'static'), (hi, 'bump')] if margin_travel else [(0.0, 'static')]
    for t, nm in stations:
        try:
            corners, _ = win._assemble_corners_draw({l: float(t) for l in ('FL', 'FR', 'RL', 'RR')}, 0.0, light=True)
        except Exception:
            return {'gap_mm': float('nan'), 'worst': 'assembly failed at %s' % nm, 'line': (yl, zl, yu, zu)}
        p = [c for c in corners if c['label'] == 'FL'][0]['pts']
        dt = np.asarray(p['arb_drop_top'], float) * 1000.0
        ae = np.asarray(p.get('arb_arm_end_world', p.get('arb_arm_end', apv / 1000.0)), float) * 1000.0
        for label, pt, r in (('torsion bar', apv, od / 2.0), ('arm end', ae, RE), ('drop top', dt, RE)):
            g = y_line(pt[2]) - (pt[1] + r)
            if g < worst[0]:
                worst = (float(g), '%s at %s (%s Y %.1f Z %.1f)' % (label, nm, 'front ARB', pt[1], pt[2]))
        # Blade and link bodies: use the same routed physical segments as all
        # other production collision consumers, rather than a straight chord.
        akw = arb_member_kwargs(win._car, 'front',
            float(win._dynamics_panel._arb_blade_w_f.value()),
            float(win._dynamics_panel._arb_blade_t_f.value()))
        fm = full_members(p, win._car, arb_pivot=np.asarray(win._front_arb['arb_pivot']),
                          arb_od_mm=od, **akw)
        for member in (m for m in fm if m['name'] in ('ARB blade', 'ARB drop link')):
            label = 'blade' if member['name'] == 'ARB blade' else 'drop link'
            a = np.asarray(member['a'])*1000.0; b = np.asarray(member['b'])*1000.0
            r = float(member['r'])*1000.0
            for f in np.linspace(0.0, 1.0, 9):
                q = a + (b - a) * f; g = y_line(q[2]) - (q[1] + r)
                if g < worst[0]:
                    worst = (float(g), '%s at %s (Y %.1f Z %.1f)' % (label, nm, q[1], q[2]))
    return {'gap_mm': worst[0], 'worst': worst[1], 'line': (yl, zl, yu, zu)}


def _clash_stations(win) -> list:
    """[('droop', t), ('static', 0), ('bump', t)] in metres."""
    lo, hi = travel_range_m(win)
    return [('full droop', lo), ('static', 0.0), ('full bump', hi)]


def _clash_sweep(win, travel_stations_m=None) -> dict:
    """Full-member clash lists at static, full bump and full droop (or supplied
    travel stations), using the current solver steering/rack position and the
    SAME member set the GUI interference view runs (vahan.interference
    .full_members: arms + tie rod + pushrod + ball-joint spheres + coilover +
    ARB drop link/torsion bar + rocker hardware + driveshaft).
    Incomplete or non-closing geometry raises instead of reporting no clashes.
    """
    out = {}
    fsae_ensure_resolved(win)       # FSAE chassis bays: 'auto' diagonal current (no-op when off)
    stations = (_clash_stations(win) if travel_stations_m is None else
                [(f'{float(t) * 1000:+.3f} mm', float(t)) for t in travel_stations_m])
    for name, t in stations:
        travels = {l: float(t) for l in ('FL', 'FR', 'RL', 'RR')}
        corners_draw, _ = win._assemble_corners_draw(
            travels, getattr(win, '_solver_rack_travel_m', 0.0))
        if len(corners_draw) != 4:
            raise ValueError(f'Incomplete corner geometry at {name}')
        for corner in corners_draw:
            if corner.get('geometry_errors'):
                raise ValueError(f"{corner['label']} at {name}: " +
                                 '; '.join(corner['geometry_errors']))
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
        members_by_corner = {}
        for c in corners_draw:
            label = c['label']
            arb = win._front_arb if label[0] == 'F' else win._rear_arb
            apv = aod = None
            panel = getattr(win, '_dynamics_panel', None)

            def _panel_value(name, default=0.0):
                control = getattr(panel, name, None)
                return float(control.value()) if control is not None else float(default)
            try:
                if arb and 'arb_pivot' in arb:
                    apv = np.asarray(arb['arb_pivot'], float)
                    aod = _panel_value('_arb_OD_f' if label[0] == 'F' else '_arb_OD_r')
            except Exception:
                pass
            _akw = arb_member_kwargs(win._car, label,
                _panel_value('_arb_blade_w_f' if label[0] == 'F' else '_arb_blade_w_r'),
                _panel_value('_arb_blade_t_f' if label[0] == 'F' else '_arb_blade_t_r'))
            mem = full_members(c['pts'], win._car, arb_pivot=apv, arb_od_mm=aod,
                               driveshaft_seg=ds.get(label),
                               **_akw)
            members_by_corner[label] = mem
            # Include the same physical plate test used by the live renderer.
            # Capsule-to-capsule checks alone miss rods passing through its face.
            for member_name, plate_gap in rocker_plate_gaps(
                    c['pts'], mem,
                    half_t=float(win._car.get(
                        'rocker_plate_thickness_mm', 6.0))/2000.0,
                    **rocker_plate_physical_options_for(win._car, label)):
                # Geometry designed at exactly 3 mm can evaluate a few
                # femtometres low after repeated rigid transforms.  A 1 nm
                # numeric tolerance prevents false failures without relaxing
                # the physical clearance requirement.
                if plate_gap < 0.003 - 1e-9:
                    found.append({'corner': label, 'a': member_name,
                                  'b': 'rocker plate',
                                  'gap_mm': float(plate_gap) * 1000.0 - 3.0,
                                  'surface_gap_mm': float(plate_gap) * 1000.0})
            for cl in clashes(mem, connected=connected_for(label)):
                found.append({'corner': label, 'a': cl['a'], 'b': cl['b'],
                              'gap_mm': cl['gap_mm']})
            if 'rim_barrel_width_mm' in win._car:
                for member in mem:
                    clearance = rim_barrel_gap(member, c['pts']['wheel_center'],
                        c['spin_axis'], float(win._car['tire_rim_dia_mm']) / 2000.,
                        float(win._car['rim_barrel_width_mm']) / 2000.) * 1000.
                    if clearance < 3.0:
                        found.append({'corner': label, 'a': member['name'],
                            'b': 'rim barrel + 3 mm clearance',
                            'gap_mm': clearance - 3.0,
                            'surface_gap_mm': clearance})
        for hit in cross_corner_clashes(members_by_corner):
            found.append({'corner': hit['a_corner']+'/'+hit['b_corner'],
                          'a': hit['a_corner']+' '+hit['a'],
                          'b': hit['b_corner']+' '+hit['b'], 'gap_mm': hit['gap_mm']})
        out[name] = found
    return out


def damper_motion_sign(win, axle: str) -> float:
    """Sign of d(spring_length)/d(travel): a proper PUSHROD COMPRESSES the
    damper in bump, so spring_length falls as travel rises -> negative.  A
    sign-inverted rocker turns the pushrod into a PULLROD (damper EXTENDS in
    bump, +).  solver_mr() takes an absolute value and is blind to this, so
    the sign must be judged separately or a pullrod passes as 'rate matched'
    (v73 shipped inverted before this guard existed)."""
    label = _LABEL[axle]
    solver = win._solvers[label]
    lo, hi = travel_range_m(win)
    L_hi = float(solver.solve(hi).spring_length)
    L_lo = float(solver.solve(lo).spring_length)
    d = L_hi - L_lo
    return float(np.sign(d)) if abs(d) > 1e-9 else 0.0


def capture_baseline(win) -> dict:
    """Snapshot of EVERY held parameter from the currently loaded model.
    This is the reference validate() judges relative parameters against.
    Absolute physical laws can reject a baseline that is itself defective."""
    base = {'wheel_points': {}, 'wheel_metrics': {}, 'geometry': {},
            'damper_sign': {}}
    for axle in ('front', 'rear'):
        hp = win._front_hp if axle == 'front' else win._rear_hp
        base['wheel_points'][axle] = {
            k: np.asarray(hp[k], float).tolist()
            for k in WHEEL_HP_KEYS if k in hp and hp[k] is not None}
        base['wheel_metrics'][axle] = _axle_wheel_metrics(win, axle)
        base['geometry'][axle] = _axle_geometry_laws(win, axle)
        base['damper_sign'][axle] = damper_motion_sign(win, axle)
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
             axles=('front', 'rear'), stop_early: bool = False,
             allow_wheel_motion: bool = False,
             skip_clash: bool = False) -> ValidationResult:
    """THE single validity oracle.  Judges the model CURRENTLY loaded in `win`
    against `baseline` within `tol`.  Cheap checks run first; the full-travel
    clash sweep runs last; with stop_early=True the first failing group aborts
    the rest (generator throughput).

    Checks, in order:
      1. wheel-locating points — if byte-identical to baseline, every wheel
         parameter is held EXACTLY by construction (recorded as zero-delta);
         if ANY moved, the wheel metrics are re-measured and compared.
      2. geometric laws: independent left/right static full-chain coplanarity,
         rocker-axis normality, static ARB link endpoints, and triad angles.
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
        # In relocation mode (allow_wheel_motion) a moved wheel point is the
        # POINT of the exercise — the re-measured metric/curve comparisons
        # below then carry the whole judgement; the distance is informational.
        res.add('wheel points moved', axle, moved * 1000, 0.0, 1e-9,
                allow_wheel_motion or moved < 1e-12, ' mm')
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
                np.isfinite(g['coplanar_mm']) and g['coplanar_mm'] <= tol.coplanar_mm,
                ' mm')
        res.add('rocker_axis_normal_error_deg', axle,
                g['rocker_axis_normal_error_deg'], 0.0, tol.rocker_axis_deg,
                np.isfinite(g['rocker_axis_normal_error_deg'])
                and g['rocker_axis_normal_error_deg'] <= tol.rocker_axis_deg,
                ' deg')
        _rule04_ok = arb_drop_link_plate_compliant(
            {'drop_top_signed_mm': g['arb_drop_top_plate_signed_mm'],
             'arm_end_signed_mm': g['arb_arm_end_plate_signed_mm']},
            tol.arb_inplane_mm, g.get('arb_is_bottom', False))
        for k in ('arb_drop_top_inplane_mm', 'arb_arm_end_inplane_mm'):
            res.add(k, axle, g[k], 0.0, tol.arb_inplane_mm,
                    _rule04_ok,
                    ' mm' + (' (control-arm ARB: Rule 04 exempt)'
                              if g.get('arb_is_bottom', False) else ''))
        for k in ('triad_bar_blade_deg', 'triad_blade_drop_deg',
                  'triad_bar_drop_deg'):
            # A non-square baseline is not permission to preserve that defect.
            d = abs(g[k] - 90.0)
            res.add(k, axle, g[k], 90.0, tol.triad_deg, d <= tol.triad_deg,
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
    # damper motion SIGN — a matched |motion ratio| can still be a sign-
    # inverted rocker (pushrod acting as pullrod: damper EXTENDS in bump).
    # The percent checks above are blind to it (abs), so guard it explicitly.
    for ax in axles:
        bs = baseline.get('damper_sign', {}).get(ax)
        if bs is None or bs == 0.0:
            continue
        cs = damper_motion_sign(win, ax)
        ok = (cs == bs)
        res.add('damper acts as pushrod', ax, cs, bs, 0.5, ok,
                ' (sign of d spring_length / d travel; must match baseline)')
    if stop_early and not res.ok:
        res.aborted_after = 'rates'
        return res

    # ── 4. clash sweep (most expensive — last) ───────────────────────────────
    if skip_clash:
        return res
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


# ═════════════════════════════════════════════════════════════════════════════
#  DESIGN CITY primitives (2026-09-09) — the packaging search used by
#  design_city.py.  Everything below is a MEASUREMENT or a GEOMETRIC
#  TRANSFORM on the ONE model; nothing re-implements physics.
# ═════════════════════════════════════════════════════════════════════════════
# Rocker hardware (user-supplied 2026-07-25, same numbers as the regression
# net's design-actuation gate): 1.5 in OD pivot bearing, drop-link / rod-end
# ball joints 0.315 in RADIUS.
ROCKER_BEARING_R_MM = 0.5 * 38.1
ROD_END_R_MM = 0.315 * 25.4
RACK_HOUSING_R_M = 0.01905          # 1.5 in OD steering rack housing proxy
LCA_MEMBER_R_M = 0.008              # LCA-inner chassis member proxy radius


def retarget_side_view_ic(hp: dict, ic_y_m: float, ic_z_m: float) -> dict:
    """Aim both wishbone planes at a new SIDE-VIEW instant centre (Y, Z) by moving
    the four chassis pickups in HEIGHT only.

    Each new arm plane is spanned by (a) the arm's present FRONT-VIEW line through its
    ball joint (plane cut at the ball joint's Y station - what sets camber gain and the
    roll-centre construction) and (b) the side-view line ball joint -> new instant
    centre. Pickup X and Y (the chassis node in plan view) and every outboard point are
    untouched. A point coincident with a moved pickup (toe link on the arm's bracket)
    moves with it. Returns a new hardpoint dict.
    """
    out = {k: np.array(v, float) for k, v in hp.items()}
    y_hat = np.array([0., 1., 0.])
    for arm in ('uca', 'lca'):
        pa, pb, po = (np.asarray(hp[f'{arm}_{e}'], float) for e in ('front', 'rear', 'outer'))
        normal = np.cross(pb - pa, po - pa)
        front_view = np.cross(normal, y_hat)                 # in the arm plane, in the plane Y = const
        side_view = np.array([0., float(ic_y_m) - po[1], float(ic_z_m) - po[2]])
        new_normal = np.cross(front_view, side_view)
        if abs(new_normal[2]) < 1e-9 or np.linalg.norm(side_view) < 1e-6:
            raise ValueError(f'{arm}: degenerate plane for that instant centre')
        for end in ('front', 'rear'):
            key = f'{arm}_{end}'
            old = np.asarray(hp[key], float)
            new = old.copy()
            new[2] = po[2] - (new_normal[0] * (old[0] - po[0]) + new_normal[1] * (old[1] - po[1])) / new_normal[2]
            for other, val in hp.items():
                if other != key and np.allclose(np.asarray(val, float), old, atol=1e-9):
                    out[other] = new.copy()
            out[key] = new
    return out


def _copy_bundle(bundle: dict) -> dict:
    return {'hp': {k: np.array(v, float) for k, v in bundle['hp'].items()},
            'arb': {k: np.array(v, float) for k, v in bundle['arb'].items()}}


def rotate_arb_drop_top(bundle: dict, angle_deg: float) -> dict:
    """Swing arb_drop_top about the rocker pivot IN THE CHAIN PLANE, radius
    held (the v99 front-ARB pose knob).  Follow with an ARB re-hang
    (refit_arb / refit_arb_variants) and a rate retune."""
    c, n = _chain_plane(bundle)
    piv = np.asarray(bundle['hp']['rocker_pivot'], float)
    r = np.asarray(bundle['arb']['arb_drop_top'], float) - piv
    r = r - n * float(np.dot(r, n))
    L = float(np.linalg.norm(r))
    if L < 1e-9:
        raise ValueError('drop top coincides with the rocker pivot')
    u = r / L
    a = np.radians(angle_deg)
    out = _copy_bundle(bundle)
    out['arb']['arb_drop_top'] = piv + L * (u * np.cos(a) + np.cross(n, u) * np.sin(a))
    return out


def set_drop_link_length(bundle: dict, length_m: float) -> dict:
    """Lengthen / shorten the drop link along its own line; the blade (pivot -
    arm end) is carried rigidly so the triad angles survive."""
    out = _copy_bundle(bundle)
    d = out['arb']['arb_drop_top']; e0 = out['arb']['arb_arm_end']; p0 = out['arb']['arb_pivot']
    u = (e0 - d) / np.linalg.norm(e0 - d)
    e1 = d + float(length_m) * u
    out['arb']['arb_arm_end'] = e1
    out['arb']['arb_pivot'] = e1 + (p0 - e0)
    return out


def scale_spring_lever(bundle: dict, s: float, hold_damper_length: bool = True) -> dict:
    """Scale the (rocker_spring_pt - rocker_pivot) lever by s inside the rocker
    plane.  With hold_damper_length the chassis damper eye slides along the
    damper's own line so the static damper length is unchanged (a different
    rocker shape, same damper).  Follow with retune_mr() on the pushrod lever
    so the motion ratio is restored."""
    out = _copy_bundle(bundle)
    pv = out['hp']['rocker_pivot']
    sp0 = out['hp']['rocker_spring_pt']
    sc0 = out['hp']['spring_chassis_pt']
    sp1 = pv + float(s) * (sp0 - pv)
    out['hp']['rocker_spring_pt'] = sp1
    if hold_damper_length:
        d = sc0 - sp0
        out['hp']['spring_chassis_pt'] = sp1 + d       # same vector -> same length
        if 'damper_chassis_pt' in out['hp'] and out['hp']['damper_chassis_pt'] is not None:
            out['hp']['damper_chassis_pt'] = out['hp']['damper_chassis_pt'] + (sp1 - sp0)
    return out


def slide_pickup(bundle: dict, key: str, d_mm: float) -> dict:
    """Slide one inboard arm pickup (uca_front/uca_rear/lca_front/lca_rear)
    along ITS OWN arm line toward (+) / away from (-) the outer ball joint.
    This DOES touch a wheel-locating point (the pivot axis direction moves),
    so the parameter gate decides whether the wheel curves survived."""
    out = _copy_bundle(bundle)
    outer = 'uca_outer' if key.startswith('uca') else 'lca_outer'
    a = out['hp'][key]; b = out['hp'][outer]
    u = (b - a) / np.linalg.norm(b - a)
    out['hp'][key] = a + u * (float(d_mm) / 1000.0)
    return out


SPRING_PAIR_KEYS = ('rocker_spring_pt', 'spring_chassis_pt', 'damper_chassis_pt')


def swing_spring_about_rocker_axis(bundle: dict, angle_deg: float) -> dict:
    """Rotate the SPRING pair (rocker_spring_pt + spring_chassis_pt, and the
    damper chassis eye when present) about the rocker AXIS by angle_deg.
    The rocker turns about that same axis, so spring length vs rocker angle
    is unchanged: |R(theta) Q s - Q c| = |R(theta) s - c| (rotations about one
    axis commute).  Motion-ratio curve, damper lengths, travel range and
    every rate are therefore held BY CONSTRUCTION; only the rocker shape and
    the coilover's chassis mount move (both stay in the rocker plane).
    The exact family the Design City search is built on."""
    p0, n = rocker_plane(bundle)
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    out = _copy_bundle(bundle)
    for k in SPRING_PAIR_KEYS:
        if k in out['hp'] and out['hp'][k] is not None:
            v = out['hp'][k] - p0
            out['hp'][k] = p0 + v * c + np.cross(n, v) * s + n * np.dot(n, v) * (1 - c)
    return out


PICKUP_PARTNER = {'uca_front': 'uca_rear', 'uca_rear': 'uca_front',
                  'lca_front': 'lca_rear', 'lca_rear': 'lca_front'}


def slide_pickup_along_axis(bundle: dict, key: str, d_mm: float,
                            min_spacing_mm: float = 60.0) -> dict:
    """Slide ONE inboard arm pickup (uca_front/uca_rear/lca_front/lca_rear)
    along the arm's PIVOT AXIS — the line through the two inboard pickups.
    The axis line is unchanged, so the outer ball joint sweeps the SAME
    circle: every wheel curve is identical by construction (the 0.1 % gate
    still re-measures it).  What changes is the chassis bracket location
    and the arm's own tube geometry (clash-checked downstream).
    +d moves the pickup TOWARD its partner (shorter pickup spacing);
    raises if the spacing would fall under min_spacing_mm."""
    partner = PICKUP_PARTNER[key]
    out = _copy_bundle(bundle)
    a = out['hp'][key]; b = out['hp'][partner]
    span = float(np.linalg.norm(b - a))
    u = (b - a) / span
    new = a + u * (float(d_mm) / 1000.0)
    if float(np.linalg.norm(b - new)) * 1000.0 < min_spacing_mm:
        raise ValueError(f'{key} slide {d_mm:+.1f} mm leaves {np.linalg.norm(b - new) * 1000:.1f} mm '
                         f'to {partner} (< {min_spacing_mm} mm)')
    out['hp'][key] = new
    return out


def rocker_hw_gaps(bundle: dict) -> dict:
    """Rocker HARDWARE separations (mm, negative = overlap) exactly as the
    regression net's design-actuation gate computes them: every pickup's
    rod end vs the pivot bearing, and rod ends vs each other.  The v99
    +60 deg pose passed every proxy audit with the pushrod and drop rod
    ends 2.3 mm apart — both audits skip pairs whose endpoints are within
    6 mm — so this gate must run on every accepted pose."""
    hp, arb = bundle['hp'], bundle['arb']
    pv = np.asarray(hp['rocker_pivot'], float) * 1000
    picks = {'pushrod': np.asarray(hp['pushrod_inner'], float) * 1000,
             'spring': np.asarray(hp['rocker_spring_pt'], float) * 1000,
             'ARB': np.asarray(arb['arb_drop_top'], float) * 1000}
    out = {}
    for n, p in picks.items():
        out[f'{n}_rodend_vs_bearing'] = float(np.linalg.norm(p - pv)) - ROCKER_BEARING_R_MM - ROD_END_R_MM
    ks = list(picks)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            out[f'{ks[i]}_vs_{ks[j]}_rodends'] = float(np.linalg.norm(picks[ks[i]] - picks[ks[j]])) - 2 * ROD_END_R_MM
    return out


def full_state_audit(win, n_rack: int = 13, near_mm: float = 10.0,
                     focus: str = None, chassis_diagonal: str = None) -> dict:
    """The 39-state proxy interference audit from the v99 builder, now a
    reusable feature: droop / static / bump x n_rack rack positions, every
    corner's full member set (vahan.interference.full_members) PLUS the
    LCA-inner chassis member proxies, the steering-rack housing (1.5 in OD),
    both torsion bars (pivot to mirrored pivot) and the rear driveshafts
    (vahan.driveshaft), all pairs cross-corner.  Pairs sharing an endpoint
    within 6 mm are one joint and skipped (so run rocker_hw_gaps too).

    FSAE chassis bays ON (car['fsae_chassis']): full_members carries the bay
    tubes and the LCA-inner stand-in is dropped; the 'auto' diagonal is
    resolved first (fsae_ensure_resolved).  ``chassis_diagonal`` forces a
    diagonal for this audit only ('ucaf_lcar' / 'ucar_lcaf' / 'both');
    ``focus`` keeps only pairs whose member names contain that text.
    Returns {'states', 'negatives', 'warnings', 'closure_errors',
    'worst_per_pair', 'min_gap_mm'}."""
    import types as _types
    from .interference import pair_gap_mm
    from .driveshaft import package as ds_package
    from .chassis import settings as _fsae_settings
    fsae = _fsae_settings(win._car)
    _prev_override = getattr(win, '_fsae_diag_override', None)
    if fsae is not None and chassis_diagonal is None and not _prev_override:
        fsae_ensure_resolved(win)
    if chassis_diagonal is not None:
        win._fsae_diag_override = chassis_diagonal
    try:
        return _full_state_audit(win, n_rack, near_mm, focus, fsae, _types, pair_gap_mm, ds_package)
    finally:
        if chassis_diagonal is not None:
            win._fsae_diag_override = _prev_override


def _full_state_audit(win, n_rack, near_mm, focus, fsae, _types, pair_gap_mm, ds_package):
    lo, hi = win._spring_travel_range(win._solvers['FL'], 'FL')
    lo = min(lo, -0.025); hi = max(hi, 0.025)
    rack_half = float(win._steer['total_rack_travel_mm']) / 2000.0
    racks = np.linspace(-rack_half, rack_half, int(n_rack))
    worst = {}
    closure = []
    n_states = 0
    for tname, t in (('full_droop', lo), ('static', 0.0), ('full_bump', hi)):
        for rt in racks:
            n_states += 1
            corners, _ = win._assemble_corners_draw(
                {l: float(t) for l in ('FL', 'FR', 'RL', 'RR')}, float(rt), light=True)
            by = {c['label']: c for c in corners}
            errs = [c['label'] for c in corners if c.get('geometry_errors')]
            if errs or len(by) != 4:
                closure.append({'travel': tname, 'rack_mm': float(rt * 1000),
                                'corners': errs or ['missing corner']})
                continue
            members = []; rear_states = {}
            for label, c in by.items():
                arb = win._front_arb if label.startswith('F') else win._rear_arb
                apv = np.asarray(arb['arb_pivot'], float)
                od = float(getattr(win._dynamics_panel,
                                   '_arb_OD_f' if label.startswith('F') else '_arb_OD_r').value())
                if label.startswith('R'):
                    rear_states[label] = _types.SimpleNamespace(
                        wheel_center=np.asarray(c['pts']['wheel_center']),
                        spin_axis=np.asarray(c['spin_axis']))
                front = label.startswith('F')
                akw = arb_member_kwargs(win._car, label,
                    float(getattr(win._dynamics_panel, '_arb_blade_w_f' if front else '_arb_blade_w_r').value()),
                    float(getattr(win._dynamics_panel, '_arb_blade_t_f' if front else '_arb_blade_t_r').value()))
                for m in full_members(c['pts'], win._car, arb_pivot=apv, arb_od_mm=od, **akw):
                    m = dict(m); m['name'] = label + ' ' + m['name']; members.append(m)
                hp = c['pts']
                if fsae is None:        # FSAE chassis bays replace this stand-in when ON
                    members.append({'name': label + ' LCA inner chassis member',
                                    'a': np.asarray(hp['lca_front']), 'b': np.asarray(hp['lca_rear']),
                                    'r': LCA_MEMBER_R_M})
            from .interference import rack_members as _rack_members
            members.extend(_rack_members(win._car, by['FL']['pts']['tie_rod_inner'], by['FR']['pts']['tie_rod_inner']))
            for axle, arb in (('front', win._front_arb), ('rear', win._rear_arb)):
                p = np.asarray(arb['arb_pivot'], float); q = p.copy(); q[0] = -p[0]
                od = float(getattr(win._dynamics_panel,
                                   '_arb_OD_f' if axle == 'front' else '_arb_OD_r').value())
                members.append({'name': axle + ' ARB torsion bar', 'a': p, 'b': q, 'r': od / 2000.0})
            try:
                pkg = ds_package(win._car, rear_states) if len(rear_states) == 2 else {}
            except Exception:
                pkg = {}
            for label in ('RL', 'RR'):
                if pkg and label in pkg:
                    members.append({'name': label + ' driveshaft',
                                    'a': np.asarray(pkg[label]['inner']), 'b': np.asarray(pkg[label]['outer']),
                                    'r': float(win._car.get('driveshaft_dia_mm', 25.4)) / 2000.0})
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if focus and focus not in a['name'] and focus not in b['name']:
                        continue
                    # the front tie rod hangs off the rack bar: rod, joint and rack are one mechanism, never a clash pair
                    if ('steering rack' in a['name'] + b['name']) and any(k in a['name'] + b['name'] for k in ('tie / toe rod', 'tie inner', 'tie rod')) and (a['name'][:1] == 'F' or b['name'][:1] == 'F' or a['name'].startswith('front') or b['name'].startswith('front')): continue
                    # shared endpoint within 6 mm = one joint; chassis frame vs frame and
                    # bracket zones of the FSAE chassis bays are designed contacts
                    gap = pair_gap_mm(a, b, 0.006)
                    if gap is None:
                        continue
                    if gap < near_mm:
                        k = (a['name'], b['name'])
                        if k not in worst or gap < worst[k][0]:
                            worst[k] = (float(gap), tname, float(rt * 1000))
    # FSAE chassis bays: the tubes vs the CHASSIS-FIXED obstructions (sprocket
    # disc from the imported STEP, diff housing) — nothing in these pairs moves,
    # so they are checked ONCE from the static assembly, not per state.
    fixed_rows = []
    if fsae is not None:
        from .chassis import fixed_obstructions, fixed_part_gaps, is_chassis
        parts = fixed_obstructions(win)
        if parts:
            corners, _ = win._assemble_corners_draw(
                {l: 0.0 for l in ('FL', 'FR', 'RL', 'RR')}, 0.0, light=True)
            tubes = []
            for c in corners:
                for m in full_members(c['pts'], win._car):
                    if is_chassis(m):
                        m = dict(m); m['name'] = c['label'] + ' ' + m['name']; tubes.append(m)
            for row in fixed_part_gaps(tubes, parts):
                if focus and focus not in row['a'] and focus not in row['b']:
                    continue
                fixed_rows.append(row)
                if row['gap_mm'] < near_mm:
                    k = (row['a'], row['b'])
                    if k not in worst or row['gap_mm'] < worst[k][0]:
                        worst[k] = (float(row['gap_mm']), 'chassis-fixed', 0.0)
    rows = [{'a': k[0], 'b': k[1], 'gap_mm': v[0], 'travel': v[1], 'rack_mm': v[2]}
            for k, v in sorted(worst.items(), key=lambda kv: kv[1][0])]
    return {'states': n_states, 'closure_errors': closure, 'fixed_part_gaps': fixed_rows,
            'negatives': [r for r in rows if r['gap_mm'] < 0.0],
            'warnings': [r for r in rows if 0.0 <= r['gap_mm'] < 3.0],
            'worst_per_pair': rows,
            'min_gap_mm': rows[0]['gap_mm'] if rows else float(near_mm)}


def fsae_resolve_diagonals(win) -> dict:
    """Resolve the FSAE chassis-bay 'auto' diagonal per axle (left/right are
    mirrored) with the full-state audit machinery: ONE full_state_audit pass
    with BOTH candidate diagonals present, pairs restricted to the diagonals,
    every other member (arms, links, rocker hardware, coilover, ARB, rack,
    torsion bars, driveshafts) over droop/static/bump x 13 rack positions.
    Chassis tube vs chassis tube is one frame and never counts.  Per axle the
    diagonal with the LARGER worst-case clearance wins; a tie within 1e-6 mm
    goes to 'ucaf_lcar' (deterministic).  Caches the result on the window
    (win._fsae_chassis_cache, keyed by vahan.chassis.settings_signature) and
    returns it: {'sig', 'front', 'rear', 'gaps': {axle: {diag: row}}}."""
    from .chassis import (settings_signature, DIAGONAL_TUBES, EXPLICIT_DIAGONALS)
    res = full_state_audit(win, near_mm=1e9, focus='chassis diagonal',
                           chassis_diagonal='both')
    gaps = {'front': {}, 'rear': {}}
    for row in res['worst_per_pair']:
        for side in ('a', 'b'):
            nm = row[side]
            for diag, (dname, _ka, _kb) in DIAGONAL_TUBES.items():
                if nm.endswith(dname):
                    axle = 'front' if nm[:1] == 'F' else 'rear'
                    other = row['b' if side == 'a' else 'a']
                    cur = gaps[axle].get(diag)
                    if cur is None or row['gap_mm'] < cur['gap_mm']:
                        gaps[axle][diag] = {'gap_mm': float(row['gap_mm']), 'diagonal': nm,
                                            'other': other, 'travel': row['travel'],
                                            'rack_mm': row['rack_mm']}
    out = {'sig': settings_signature(win), 'gaps': gaps,
           'closure_errors': res['closure_errors']}
    for axle in ('front', 'rear'):
        g1 = gaps[axle].get('ucaf_lcar', {}).get('gap_mm', float('inf'))
        g2 = gaps[axle].get('ucar_lcaf', {}).get('gap_mm', float('inf'))
        out[axle] = 'ucar_lcaf' if g2 > g1 + 1e-6 else 'ucaf_lcar'
    win._fsae_chassis_cache = out
    return out


def fsae_ensure_resolved(win) -> dict | None:
    """Return the live auto-diagonal resolution, recomputing it only when the
    feature is ON, at least one axle's diagonal is 'auto', and any input changed
    since the cached one.  None when the feature is off or both axles' diagonals
    are explicit (per-axle keys diagonal_front / diagonal_rear)."""
    from .chassis import settings as _fsae_settings, settings_signature, any_auto
    s = _fsae_settings(win._car)
    if not any_auto(s):                 # off, or both axles chosen explicitly
        return None
    if getattr(win, '_fsae_diag_override', None):
        return getattr(win, '_fsae_chassis_cache', None)
    cache = getattr(win, '_fsae_chassis_cache', None)
    if cache and cache.get('sig') == settings_signature(win):
        return cache
    return fsae_resolve_diagonals(win)


def fsae_auto_is_stale(win) -> bool:
    """True when the feature is ON with 'auto' on at least one axle and the
    cached choice no longer matches the live inputs (the 3D view keeps drawing
    the cached choice)."""
    from .chassis import settings as _fsae_settings, settings_signature, any_auto
    s = _fsae_settings(win._car)
    if not any_auto(s):                 # off, or both axles chosen explicitly
        return False
    cache = getattr(win, '_fsae_chassis_cache', None)
    return not cache or cache.get('sig') != settings_signature(win)


def handwheel_lock_deg(win) -> float:
    """Handwheel angle at full rack lock (deg) from the config's stroke and
    rack travel per turn — the value the net's lock sweeps use."""
    return (float(win._steer['total_rack_travel_mm'])
            / float(win._steer['rack_travel_per_rev_mm'])) * 180.0


def clash_negatives_at_locks(win) -> dict:
    """_clash_sweep (droop/static/bump, full member set + rim guard) at rack
    centre and both full locks.  Returns {'negatives': n, 'worst_mm', 'pairs'};
    raises if any station fails to close (the sweep refuses to report
    no-clash on geometry that did not solve)."""
    hand = handwheel_lock_deg(win)
    negs, worst, pairs = 0, 1e9, []
    fsae_ensure_resolved(win)       # resolve 'auto' at the design steer, before the lock rebuilds
    try:
        for h in (0.0, hand, -hand):
            win._rebuild_solvers(h)
            for station, hits in _clash_sweep(win).items():
                for hh in hits:
                    worst = min(worst, hh['gap_mm'])
                    if hh['gap_mm'] < 0.0:
                        negs += 1
                        pairs.append({'lock_deg': h, 'station': station, **hh})
    finally:
        win._rebuild_solvers(0.0)
    return {'negatives': negs, 'worst_mm': worst if worst < 1e9 else None, 'pairs': pairs}


def rate_match_arb(win, bundle: dict, axle: str, target_rate: float,
                   ref_bundle: dict, tol: float = 2e-4) -> tuple:
    """Hold the ARB wheel rate to target_rate within tol (relative): first
    the BLADE-length knob (keeps the drop top where the pose put it, so the
    rocker hardware gaps survive), then the drop-radius knob as fallback.
    Returns (bundle, rate, knob_used).  Leaves the window on the result."""
    set_bundle(win, axle, bundle)
    r0 = panel_arb_rate(win, axle)
    if abs(r0 - target_rate) / target_rate <= tol:
        return bundle, r0, 'none'
    try:
        b2, _sc, r2 = retune_arb_blade(win, bundle, axle, target_rate, ref_bundle,
                                       s_lo=0.35, s_hi=3.5, iters=26)
        Lb = float(np.linalg.norm(b2['arb']['arb_pivot'] - b2['arb']['arb_arm_end']) * 1000)
        if 50.0 <= Lb <= 220.0 and abs(r2 - target_rate) / target_rate <= tol:
            set_bundle(win, axle, b2)
            return b2, r2, 'blade'
    except Exception:
        pass
    b2, _m, r2 = retune_arb(win, bundle, axle, target_rate, ref_bundle,
                            m_lo=0.05, m_hi=6.0, iters=26)
    set_bundle(win, axle, b2)
    return b2, r2, 'drop_radius'


# ── the parameter vector: EVERY number the gate holds ────────────────────────
REL_TOL = 1e-3          # 0.1 % — the Design City gate
# Physical scale per unit family: the absolute floor is REL_TOL * scale, so a
# parameter whose baseline sits near zero (toe, MR slope, a toe-curve station)
# is judged against 0.1 % of its physical scale instead of 0.1 % of ~0.
PARAM_SCALES = {
    'deg': 1.0,          # angles (camber/toe/caster/KPI, curve stations, bump steer)
    'deg/mm': 0.01,      # camber gain
    'mm': 10.0,          # scrub, trail, RC height, damper lengths, sag, travel
    'N/m': 1000.0,       # wheel / ride / ARB rates
    'Nm/rad': 1000.0,    # roll stiffness
    '%': 10.0,           # anti-dive/squat, Ackermann, LLTD, roll distribution, stroke use
    'Hz': 1.0,           # ride frequency
    'deg/g': 1.0,        # roll gradient, understeer gradient, roll angle at 1 g
    'ratio': 0.1,        # motion ratio (spring / wheel)
    '1/m': 1.0,          # motion-ratio slope
    'sign': 1.0,         # damper motion sign (exact)
}
CURVE_STATIONS_MM = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0)


def _tangent_mr(solver, t_m: float, dt: float = 0.001) -> float:
    sp = solver.solve(t_m + dt); sm = solver.solve(t_m - dt)
    return float(abs(sp.spring_length - sm.spring_length) / (2 * dt))


def parameter_vector(win) -> dict:
    """EVERY parameter the Design City gate holds, measured from the LIVE
    model in `win` (ONE MODEL: corner solvers, KinematicMetrics, the metrics
    catalog's anti-dive/squat, the dynamics VehicleParams build, the panel's
    ride-frequency and roll-gradient formulas, the steady-state solver at
    1 g).  Returns an ordered {name: {'value', 'unit', 'axle', 'scale'}}."""
    from .metrics_catalog import _anti_dive, _anti_squat
    out = {}

    def add(name, value, unit, axle):
        out[name] = {'value': float(value) if value is not None and np.isfinite(float(value)) else float('nan'),
                     'unit': unit, 'axle': axle, 'scale': PARAM_SCALES[unit]}
    car = win._car
    anti_kw = {'cg_height_m': car.get('cg_z_mm', 280.) / 1000.,
               'wheelbase_m': car.get('wheelbase_mm', 1537.) / 1000.,
               'front_brake_bias': car.get('front_brake_bias_pct', 65.) / 100.,
               'rear_drive_bias': 1.0}
    stroke_mm = float(win._motion_panel.stroke_mm)
    for axle, label in (('front', 'FL'), ('rear', 'RL')):
        solver = win._solvers[label]
        m0 = KinematicMetrics(solver.solve(0.0), 'left')
        s0 = m0.summary()
        for key, unit in (('camber_deg', 'deg'), ('toe_deg', 'deg'), ('caster_deg', 'deg'),
                          ('kpi_deg', 'deg'), ('scrub_radius_mm', 'mm'),
                          ('mechanical_trail_mm', 'mm'), ('roll_center_height_mm', 'mm')):
            add(f'{axle}.{key}', s0[key], unit, axle)
        if axle == 'front':
            add('front.anti_dive_pct', _anti_dive(m0, **anti_kw), '%', axle)
        else:
            add('rear.anti_squat_pct', _anti_squat(m0, **anti_kw), '%', axle)
        # curves across fixed stations (independent of the travel range)
        cam, toe = {}, {}
        for t in CURVE_STATIONS_MM:
            try:
                s = KinematicMetrics(solver.solve(t / 1000.0), 'left').summary()
                cam[t] = float(s['camber_deg']); toe[t] = float(s['toe_deg'])
            except Exception:
                cam[t] = float('nan'); toe[t] = float('nan')
            add(f'{axle}.camber_at_{t:+.0f}mm_deg', cam[t], 'deg', axle)
            add(f'{axle}.toe_at_{t:+.0f}mm_deg', toe[t], 'deg', axle)
        add(f'{axle}.camber_gain_deg_per_mm', (cam[20.0] - cam[-20.0]) / 40.0, 'deg/mm', axle)
        toes = [toe[t] for t in (-20.0, -10.0, 0.0, 10.0, 20.0)]
        try:
            toes += [float(KinematicMetrics(solver.solve(t), 'left').toe) for t in (-0.025, 0.025)]
        except Exception:
            toes.append(float('nan'))
        add(f'{axle}.bump_steer_deg', max(toes) - min(toes), 'deg', axle)
        # motion ratio, its slope and curve; damper lengths and stroke use
        mr0 = _tangent_mr(solver, 0.0)
        add(f'{axle}.motion_ratio', mr0, 'ratio', axle)
        add(f'{axle}.mr_slope_per_m', (_tangent_mr(solver, 0.010) - _tangent_mr(solver, -0.010)) / 0.020, '1/m', axle)
        for t in (-30.0, -20.0, 20.0, 40.0, 50.0):
            try:
                add(f'{axle}.motion_ratio_at_{t:+.0f}mm', _tangent_mr(solver, t / 1000.0), 'ratio', axle)
            except Exception:
                add(f'{axle}.motion_ratio_at_{t:+.0f}mm', float('nan'), 'ratio', axle)
        L = {}
        for t in (-30.0, -25.0, 0.0, 25.0, 50.0):
            try:
                L[t] = float(solver.solve(t / 1000.0).spring_length) * 1000.0
            except Exception:
                L[t] = float('nan')
        add(f'{axle}.damper_length_static_mm', L[0.0], 'mm', axle)
        add(f'{axle}.damper_length_at_-30mm_mm', L[-30.0], 'mm', axle)
        add(f'{axle}.damper_length_at_+50mm_mm', L[50.0], 'mm', axle)
        add(f'{axle}.stroke_used_over_pm25mm_mm', L[-25.0] - L[25.0], 'mm', axle)
        add(f'{axle}.stroke_used_over_pm25mm_pct', (L[-25.0] - L[25.0]) / stroke_mm * 100.0 if stroke_mm > 0 else float('nan'), '%', axle)
        lo, hi = win._spring_travel_range(solver, label)
        add(f'{axle}.travel_droop_mm', lo * 1000.0, 'mm', axle)
        add(f'{axle}.travel_bump_mm', hi * 1000.0, 'mm', axle)
        add(f'{axle}.damper_motion_sign', damper_motion_sign(win, axle), 'sign', axle)
        # ARB motion ratio (wheel travel / arm-tip travel, the panel's own
        # derivation) at static AND at +-25 mm: a drop link swung off its arc
        # tangent keeps the static rate and loses it in bump (v74 rate cliff),
        # so the static rate alone cannot certify an ARB re-hang.
        for t in (-25.0, 0.0, 25.0):
            try:
                g = win._compute_arb_geometry_from_kinematics(axle[0].upper(), travel_m=t / 1000.0)
                add(f'{axle}.arb_motion_ratio_at_{t:+.0f}mm', g['mr'] if g else float('nan'), 'ratio', axle)
            except Exception:
                add(f'{axle}.arb_motion_ratio_at_{t:+.0f}mm', float('nan'), 'ratio', axle)
    # steering (front)
    try:
        add('front.ackermann_at_lock_pct', float(win._probe_static_ackermann()), '%', 'front')
    except Exception:
        add('front.ackermann_at_lock_pct', float('nan'), '%', 'front')
    # rates / ride / roll from the dynamics build (the numbers the roll
    # gradient and load transfer actually run on)
    ss = win._build_dynamics_solver()
    veh = ss._veh
    for axle, sfx in (('front', 'front'), ('rear', 'rear')):
        add(f'{axle}.spring_rate_Npm', getattr(veh, f'spring_rate_{sfx}_Npm'), 'N/m', axle)
        wr = getattr(veh, f'wheel_rate_{sfx}_Npm'); rr = getattr(veh, f'ride_rate_{sfx}_Npm')
        add(f'{axle}.wheel_rate_Npm', wr, 'N/m', axle)
        add(f'{axle}.ride_rate_Npm', rr, 'N/m', axle)
        frac = veh.front_weight_fraction if axle == 'front' else veh.rear_weight_fraction
        m_c = veh.sprung_mass_kg * frac / 2.0
        add(f'{axle}.ride_frequency_Hz', np.sqrt(rr / m_c) / (2 * np.pi) if (m_c > 0 and rr > 0) else 0.0, 'Hz', axle)
        add(f'{axle}.arb_rate_Npm', getattr(veh, f'arb_rate_{sfx}_Npm'), 'N/m', axle)
        add(f'{axle}.roll_stiffness_Nm_per_rad', getattr(veh, f'roll_stiffness_{sfx}_Npm_rad'), 'Nm/rad', axle)
    rs_f, rs_r = veh.roll_stiffness_front_Npm_rad, veh.roll_stiffness_rear_Npm_rad
    rs_t = rs_f + rs_r
    add('car.roll_stiffness_front_pct', rs_f / rs_t * 100.0 if rs_t > 0 else 50.0, '%', 'car')
    rc_f = out['front.roll_center_height_mm']['value'] / 1000.0
    rc_r = out['rear.roll_center_height_mm']['value'] / 1000.0
    a_frac = veh.cg_to_front_axle_m / max(veh.wheelbase_m, 1e-6)
    h_ra = rc_f + (rc_r - rc_f) * a_frac
    h_arm = max(veh.sprung_cg_height_m - h_ra, 0.0)
    add('car.roll_gradient_deg_per_g', np.degrees(veh.sprung_mass_kg * 9.80665 * h_arm / rs_t) if rs_t > 0 else float('nan'), 'deg/g', 'car')
    try:
        sag = veh.static_sag(preload_front_mm=float(win._motion_panel.preload_front_mm),
                             preload_rear_mm=float(win._motion_panel.preload_rear_mm),
                             stroke_mm=stroke_mm, mr_front=veh.motion_ratio_front,
                             mr_rear=veh.motion_ratio_rear)
        add('front.static_sag_shock_mm', sag['sag_shock_front_mm'], 'mm', 'front')
        add('rear.static_sag_shock_mm', sag['sag_shock_rear_mm'], 'mm', 'rear')
    except Exception:
        add('front.static_sag_shock_mm', float('nan'), 'mm', 'front')
        add('rear.static_sag_shock_mm', float('nan'), 'mm', 'rear')
    try:
        r1 = ss.solve(1.0)
        tot_f = r1.elastic_lt_front_N + r1.geometric_lt_front_N + r1.unsprung_lt_front_N
        tot_r = r1.elastic_lt_rear_N + r1.geometric_lt_rear_N + r1.unsprung_lt_rear_N
        add('car.lltd_front_at_1g_pct', tot_f / (tot_f + tot_r) * 100.0 if (tot_f + tot_r) > 0 else 50.0, '%', 'car')
        add('car.roll_angle_at_1g_deg', r1.roll_angle_deg, 'deg/g', 'car')
        add('car.understeer_gradient_deg_per_g', r1.understeer_gradient_deg, 'deg/g', 'car')
    except Exception:
        for k in ('car.lltd_front_at_1g_pct', 'car.roll_angle_at_1g_deg', 'car.understeer_gradient_deg_per_g'):
            add(k, float('nan'), '%' if 'pct' in k else 'deg/g', 'car')
    return out


def compare_parameters(base: dict, cur: dict, rel: float = REL_TOL) -> list:
    """Row per parameter: baseline, value, absolute tolerance
    (max(rel*|baseline|, rel*scale)), deviation in % of max(|baseline|,
    scale) and the verdict.  A NaN on either side fails unless both are NaN
    (an unavailable metric may not silently pass)."""
    rows = []
    for name, b in base.items():
        c = cur.get(name, {'value': float('nan')})
        bv, cv = float(b['value']), float(c['value'])
        scale = float(b['scale'])
        tol = max(rel * abs(bv), rel * scale)
        if np.isnan(bv) and np.isnan(cv):
            ok, dev, dev_pct = True, 0.0, 0.0
        elif np.isnan(bv) or np.isnan(cv):
            ok, dev, dev_pct = False, float('nan'), float('nan')
        else:
            dev = cv - bv
            ok = abs(dev) <= tol
            dev_pct = abs(dev) / max(abs(bv), scale) * 100.0
        rows.append({'name': name, 'unit': b['unit'], 'axle': b['axle'], 'baseline': bv,
                     'value': cv, 'deviation': dev, 'deviation_pct': dev_pct,
                     'tolerance_abs': tol, 'floor_abs': rel * scale,
                     'floor_active': rel * scale > rel * abs(bv), 'ok': bool(ok)})
    return rows
