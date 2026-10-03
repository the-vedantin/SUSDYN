"""Control-arm inboard spherical plain bearings — the inputs the SKF plain-bearing
calculator asks for (radial / axial force, oscillation half-angle, oscillation
time, load direction), per pickup, in two bore orientations.

Geometry (exact, no small-angle assumption): an A-arm hangs on the chassis by
two spherical joints, so its only motion is a rotation about the line through
its two inboard pickups (the PIVOT AXIS u).  The ball/bolt of each inboard
bearing is fixed to the chassis clevis (bore axis b fixed in the chassis); the
outer ring is fixed in the arm, so between two poses the outer ring turns by the
arm rotation R(u, dtheta).  That relative rotation is split exactly
(swing-twist) into
  * TURNING about the bore  = what SKF calls the angle of oscillation (sliding
    of the ring around the bore), and
  * TILT of the ring against the ball = misalignment (must stay inside the
    bearing's rated tilt).

Bore orientations (user 2026-09-23):
  'normal'  - bore NORMAL to the arm plane (plane through the two inboard
              pickups and the outer ball joint, at static).  u lies in that
              plane, so the arm rotation is pure TILT; turning ~ 0.  The tilt
              changes with wheel travel (0 at static).
  'pivot'   - bore ALONG the pivot line (front pickup -> rear pickup).  The arm
              rotation is pure TURNING.  The rod end is INLINE with its leg
              (user 2026-09-26), its eye square to the leg, so the bolt sits at
              a BUILT-IN tilt of 90 deg minus the leg-to-pivot-line angle,
              constant through travel.

Forces: the force the arm puts on the chassis at the pickup
(loads.ComponentLoads.chassis_forces), split along the bore (axial Fa) and
across it (radial Fr).  The same chassis-fixed bore axis is used in every pose.

Pure numpy, no Qt.  Vectors in metres, forces in N (converted to kN only where
the SKF field is in kN)."""
from __future__ import annotations

import numpy as np

ARMS = ('uca', 'lca')
PICKUPS = ('front', 'rear')
ORIENTATIONS = ('normal', 'pivot')
ORIENTATION_LABEL = {'normal': 'bore normal to the arm plane',
                     'pivot': 'bore along the front-rear pickup line'}
# Rod-end swivel limit (user 2026-09-29): the quantity that matters is the
# BUILT-IN angle of the rod end = 90 deg - (angle between the arm leg and the
# arm's pivot line), i.e. tilt_installed_deg with the bolt along the pickup
# line ('pivot'; the plane-normal bolt has 0 built-in).  A rod end swivels
# about 27 deg, so the built-in angle must stay under 27 deg.  The worst tilt
# over full travel is reported for information, not flagged.  The limit is an
# editable input on the Bearings page.
SWIVEL_LIMIT_DEG = 27.0


def over_swivel_limit(tilt_installed_deg: float, limit_deg: float = SWIVEL_LIMIT_DEG) -> bool:
    """True when the rod end's built-in angle exceeds its swivel limit."""
    return float(tilt_installed_deg) > float(limit_deg) + 1e-9


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v * np.nan


def _get(st, k):
    return np.asarray(st[k] if isinstance(st, dict) else getattr(st, k), float)


def pivot_axis(st, arm: str) -> np.ndarray:
    """Unit vector front pickup -> rear pickup (chassis-fixed)."""
    return _unit(_get(st, f'{arm}_rear') - _get(st, f'{arm}_front'))


def bore_axis(st_static, arm: str, orientation: str) -> np.ndarray:
    """Chassis-fixed bore axis for one arm at the STATIC pose."""
    u = pivot_axis(st_static, arm)
    if orientation == 'pivot':
        return u
    if orientation == 'normal':
        f = _get(st_static, f'{arm}_front')
        o = _get(st_static, f'{arm}_outer')
        return _unit(np.cross(u, o - f))
    raise ValueError(orientation)


def arm_angle_rad(st_ref, st, arm: str) -> float:
    """Signed rotation of the arm about its pivot axis from pose st_ref to st
    (right-hand about u = front->rear), measured on the outer ball joint."""
    u = pivot_axis(st_ref, arm)
    f = _get(st_ref, f'{arm}_front')
    a = _get(st_ref, f'{arm}_outer') - f
    b = _get(st, f'{arm}_outer') - f
    a = a - (a @ u) * u
    b = b - (b @ u) * u
    return float(np.arctan2(np.cross(a, b) @ u, a @ b))


def swing_twist_deg(u, dtheta_rad: float, b) -> tuple[float, float]:
    """Split a rotation by dtheta about unit axis u into (turning about unit
    axis b, tilt of b) in degrees — exact quaternion swing-twist split."""
    u = _unit(u); b = _unit(b)
    w = np.cos(0.5 * dtheta_rad)
    v = np.sin(0.5 * dtheta_rad) * u
    p = float(v @ b)
    twist = 2.0 * np.arctan2(abs(p), w) * (1.0 if p >= 0 else -1.0)
    # swing = q * twist^-1; its angle = 2*acos(|w_swing|), w_swing = (w*wt + p*pt)
    nt = np.hypot(w, p)
    ws = (w * w + p * p) / nt if nt > 0 else 1.0
    tilt = 2.0 * np.arccos(min(1.0, abs(ws)))
    return float(np.degrees(twist)), float(np.degrees(tilt))


def split_force(F, b) -> tuple[float, float]:
    """(radial, axial) magnitude in N of force F against unit bore axis b."""
    F = np.asarray(F, float); b = _unit(b)
    fa = float(F @ b)
    return float(np.linalg.norm(F - fa * b)), abs(fa)


def load_direction(F_a, F_b, b, rel_tol: float = 0.05) -> tuple[str, str]:
    """SKF 'Load direction' class of the radial load between the two extremes of
    a motion cycle.  Returns (class, reason).
      Alternating  - the radial load reverses direction (dot of the two radial
                     vectors < 0);
      Pulsating    - same direction, magnitude changes by more than rel_tol;
      Constant     - same direction and magnitude within rel_tol."""
    b = _unit(b)
    ra = np.asarray(F_a, float) - (np.asarray(F_a, float) @ b) * b
    rb = np.asarray(F_b, float) - (np.asarray(F_b, float) @ b) * b
    na, nb = float(np.linalg.norm(ra)), float(np.linalg.norm(rb))
    if na < 1e-9 or nb < 1e-9:
        return 'Pulsating', 'radial load drops to ~0 at one end of the cycle'
    cosang = float(ra @ rb) / (na * nb)
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    if cosang < 0.0:
        return 'Alternating', f'radial load turns {ang:.0f} deg between the two ends'
    if abs(na - nb) > rel_tol * max(na, nb):
        return 'Pulsating', f'same direction (within {ang:.0f} deg), magnitude {min(na, nb) / 1000:.2f}-{max(na, nb) / 1000:.2f} kN'
    return 'Constant', f'direction within {ang:.0f} deg, magnitude within {100 * rel_tol:.0f} %'


def housing_axis(st_static, arm: str, pickup: str, orientation: str) -> np.ndarray:
    """Axis of the rod-end EYE at static.  The rod end is INLINE with its arm
    leg (user 2026-09-26: shank threaded along the leg), so the eye axis is
    square to the leg.  'normal': the arm-plane normal (already square to the
    leg - the leg lies in the plane).  'pivot': the direction square to the leg
    that is closest to the pivot line (the bolt), i.e. the pivot line with its
    along-the-leg part removed."""
    leg = _unit(_get(st_static, f'{arm}_outer') - _get(st_static, f'{arm}_{pickup}'))
    b = bore_axis(st_static, arm, orientation)
    if orientation == 'normal':
        return b
    return _unit(b - (b @ leg) * leg)


def _rot(u, th, v):
    u = _unit(u); v = np.asarray(v, float)
    return v * np.cos(th) + np.cross(u, v) * np.sin(th) + u * (u @ v) * (1 - np.cos(th))


def tilt_deg(st_static, arm: str, pickup: str, orientation: str, theta_rad: float) -> float:
    """Misalignment of the eye against the bolt with the arm rotated theta from
    static: angle between the eye axis (turning with the arm about the pivot
    line) and the chassis-fixed bolt axis.  Unsigned, 0..90 deg."""
    u = pivot_axis(st_static, arm)
    h = _rot(u, theta_rad, housing_axis(st_static, arm, pickup, orientation))
    b = bore_axis(st_static, arm, orientation)
    return float(np.degrees(np.arccos(np.clip(abs(h @ b), 0.0, 1.0))))


def case_rows(st_static, st_a, st_b, F_a: dict, F_b: dict, *, corner: str,
              case: str, ends: str, osc_time_s: float, temperature_C: float,
              full_travel_states=(), swivel_limit_deg: float = SWIVEL_LIMIT_DEG) -> list[dict]:
    """SKF inputs for every inboard pickup of one corner for one motion cycle
    that swings between poses st_a and st_b (with chassis forces F_a / F_b,
    keyed 'uca_front', ...).  full_travel_states = (full droop, full bump):
    the whole-travel arm swing and worst tilt are reported from them.  One row
    per (pickup, orientation).  Tilt includes the rod end's BUILT-IN tilt
    (inline with the leg), constant through travel for the 'pivot' bore.
    swivel_limit_deg: the rod end's swivel limit; each row carries
    over_limit = its built-in angle (tilt_installed_deg) exceeds it."""
    rows = []
    for arm in ARMS:
        u = pivot_axis(st_static, arm)
        th_a = arm_angle_rad(st_static, st_a, arm)
        th_b = arm_angle_rad(st_static, st_b, arm)
        th_full = [arm_angle_rad(st_static, s, arm) for s in full_travel_states] or [th_a, th_b]
        for orient in ORIENTATIONS:
            b = bore_axis(st_static, arm, orient)
            turn_span, tilt_span = swing_twist_deg(u, th_b - th_a, b)   # SIGNED cycle swing split
            for pk in PICKUPS:
                key = f'{arm}_{pk}'
                if key not in F_a or key not in F_b:
                    continue
                t_inst = tilt_deg(st_static, arm, pk, orient, 0.0)
                t_a, t_b = tilt_deg(st_static, arm, pk, orient, th_a), tilt_deg(st_static, arm, pk, orient, th_b)
                t_full = [tilt_deg(st_static, arm, pk, orient, t) for t in th_full]
                fr_a, fa_a = split_force(F_a[key], b)
                fr_b, fa_b = split_force(F_b[key], b)
                cls, why = load_direction(F_a[key], F_b[key], b)
                t_worst = max([t_inst, t_a, t_b] + t_full)
                rows.append(dict(
                    corner=corner, pickup=key, orientation=orient, case=case, ends=ends,
                    Fr_kN=max(fr_a, fr_b) / 1000.0, Fa_kN=max(fa_a, fa_b) / 1000.0,
                    Fr_ends_kN=(fr_a / 1000.0, fr_b / 1000.0), Fa_ends_kN=(fa_a / 1000.0, fa_b / 1000.0),
                    half_angle_deg=abs(turn_span) / 2.0,          # SKF "half the angle of oscillation"
                    tilt_installed_deg=t_inst,                    # rod end inline with the leg, at static
                    tilt_half_deg=abs(tilt_span) / 2.0,           # tilt swing within this cycle (+- about its mean)
                    tilt_worst_deg=t_worst,                       # worst misalignment, full travel
                    swivel_limit_deg=float(swivel_limit_deg),     # the rod end's swivel limit
                    over_limit=over_swivel_limit(t_inst, swivel_limit_deg),   # built-in angle over it
                    arm_swing_deg=float(np.degrees(abs(th_b - th_a))),
                    arm_swing_full_deg=float(np.degrees(max(th_full) - min(th_full))),
                    osc_time_s=float(osc_time_s), temperature_C=float(temperature_C),
                    load_direction=cls, load_direction_reason=why))
    return rows
