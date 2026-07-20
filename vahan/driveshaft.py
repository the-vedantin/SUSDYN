"""Rear driveshaft / differential packaging geometry.

Pure geometry — no Qt, no solver internals — so it is unit-testable headless
and reused by both the 3-D view and any interference check.

Frame (this car, metres): x = LATERAL (centreline x=0, +x OUTBOARD), y =
LONGITUDINAL (front axle y=0, +y REARWARD), z = VERTICAL (+z UP).  FL/RL sit on
the +x side, FR/RR on the -x side (spin_axis points +x on the left, -x on the
right).

The differential is chassis-fixed and SHARED between the two driven corners, so
its position comes from the car dict (diff_long/vert/lateral_offset), NOT the
per-corner hardpoints.  "Effective spacing" = diff HOUSING WIDTH: the two output
flanges sit at the housing faces (± housing_width/2 from the diff centre) and the
tripod joint centres a small STUB further out.  The lateral offset shifts the
whole diff sideways, so the two half-shafts come out UNEQUAL length — exactly the
asymmetry the user wants to model.

ONE MODEL: the OUTBOARD end of each half-shaft is the LIVE solved wheel_center
(it already moves with the upright every travel step); never a re-declared point.
"""
import numpy as np

# diff_housing_width_mm now carries the INBOARD SHAFT PIVOT SPACING directly
# (tripod-centre to tripod-centre), because that is what the user measures on
# the car ("the inboard drive shaft pivot points are 11.5 in apart").  So the
# tripods sit at +/- spacing/2 from the diff centre (no extra stub).  The diff
# BODY is drawn shorter than this span in view3d.
STUB_M = 0.0


def _mm(car, key, default):
    return float(car.get(key, default)) / 1000.0


def diff_center_m(car) -> np.ndarray:
    """Differential centre in world metres from the car dict.

    +x is the LEFT side of the car (RL/FL sit on +x), so a positive
    diff_lateral_offset_mm shifts the diff toward the LEFT.
    """
    return np.array([_mm(car, 'diff_lateral_offset_mm', 0.0),
                     _mm(car, 'diff_long_mm', 1537.0),
                     _mm(car, 'diff_vert_mm', 150.0)], float)


def tripod_inners_m(car) -> dict:
    """Left (+x, feeds RL) and right (-x, feeds RR) tripod joint centres.

    Placed at +/- half the inboard pivot spacing from the diff centre.
    """
    c = diff_center_m(car)
    half = 0.5 * _mm(car, 'diff_housing_width_mm', 292.1) + STUB_M
    return {'L': c + np.array([+half, 0.0, 0.0]),
            'R': c + np.array([-half, 0.0, 0.0])}


def package(car, rear_states: dict) -> dict:
    """Full rear driveshaft package.

    rear_states: {'RL': solved_state, 'RR': solved_state} (each exposes
    .wheel_center and .spin_axis in world metres).  Returns a dict keyed 'RL'/
    'RR' with inner (tripod at diff), outer (hub = live wheel_center), the shaft
    axis, its length (mm), and the CV articulation angle to the spin axis (deg).
    Also returns 'diff_center', the tripod inners, and the length asymmetry.
    """
    inners = tripod_inners_m(car)
    side_of = {'RL': 'L', 'RR': 'R'}
    out = {'diff_center': diff_center_m(car),
           'tripod_inner': inners,
           'housing_width_mm': float(car.get('diff_housing_width_mm', 180.0)),
           'tripod_od_mm': float(car.get('tripod_od_mm', 90.0)),
           'shaft_dia_mm': float(car.get('driveshaft_dia_mm', 25.0))}
    for corner, st in rear_states.items():
        inner = inners[side_of[corner]]
        outer = np.asarray(st.wheel_center, float)      # LIVE solved hub point
        spin = np.asarray(st.spin_axis, float)
        v = outer - inner
        L = float(np.linalg.norm(v))
        axis = v / L if L > 1e-9 else v
        # CV articulation = angle between the shaft and the wheel spin axis
        # (0 deg would be a perfectly straight, in-line shaft).
        cosang = abs(float(np.dot(axis, spin / max(np.linalg.norm(spin), 1e-9))))
        artic = float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
        out[corner] = dict(inner=inner, outer=outer, axis=axis,
                           length_mm=L * 1000.0, articulation_deg=artic)
    if 'RL' in out and 'RR' in out:
        out['length_asymmetry_mm'] = abs(out['RL']['length_mm']
                                         - out['RR']['length_mm'])
    return out
