"""Wheel travel and ride height vs speed under aero — straight line and steady
corners (user 2026-09-26: "ride height shrink over velocity caused by downforce
in straights and corners").  No lap simulation, and NO physics of its own: every
number is the steady-state solver's (ONE MODEL).  Since 2026-09-26 the solver
sinks the body under the aero load (SteadyStateSolver.aero_heave, through
vahan.heave_curve), so its corner `travel` already carries

    aero sink (progressive wheel rate) + roll travel + jacking (if fed back)

This module only sweeps speed, builds the aero load per corner from the car's
aero package (downforce = Cl·A·½ρv², split by the centre of pressure) and reads
the solver back:
    travel_mm       suspension travel of each wheel, + = bump (compressed)
    ride_drop_mm    chassis height lost at each corner = travel + tyre squash,
                    tyre squash = (Fz - Fz_static) / tyre rate (solver Fz)
    aero_sink_mm    the aero part of the travel, per axle (solver)
Straight = 0 g lateral, steady or at a braking / accelerating longitudinal g
(the solver's pitch travel).  Corner = steady cornering on radius R at speed v, lateral
g = v^2 / (g R), up to the car's grip limit on that radius."""
from __future__ import annotations

import numpy as np

G = 9.81
CORNERS = ('FL', 'FR', 'RL', 'RR')


def aero_per_corner_N(cla_m2: float, cop_rear: float, rho: float, v_ms: float) -> dict:
    D = float(cla_m2) * 0.5 * float(rho) * v_ms * v_ms
    f, r = D * (1.0 - cop_rear) / 2.0, D * cop_rear / 2.0
    return {'FL': f, 'FR': f, 'RL': r, 'RR': r}


def ride_sweep(ss, cla_m2: float, cop_rear: float, rho: float, speeds_kph,
               radius_m: float | None = None, grip_limit_g: float | None = None,
               longitudinal_g: float = 0.0) -> dict:
    """longitudinal_g < 0 = braking, > 0 = accelerating (straight line); the
    solver's pitch travel (anti-dive / anti-lift / anti-squat) is in `travel`."""
    kt = max(float(ss._veh.tire_rate_Npm), 1.0)
    r0 = ss.solve(0.0, 0.0)
    fz0 = {c: float(r0.Fz[c]) for c in CORNERS}
    sp = np.asarray(speeds_kph, float); n = len(sp)
    travel = {c: np.full(n, np.nan) for c in CORNERS}
    drop = {c: np.full(n, np.nan) for c in CORNERS}
    sink = {'F': np.full(n, np.nan), 'R': np.full(n, np.nan)}
    lat = np.zeros(n); dfN = np.zeros(n)
    for i, v_kph in enumerate(sp):
        v = v_kph / 3.6
        ay = 0.0 if not radius_m else v * v / (G * float(radius_m))
        lat[i] = ay
        if grip_limit_g is not None and ay > grip_limit_g:
            continue
        aero = aero_per_corner_N(cla_m2, cop_rear, rho, v)
        dfN[i] = 2.0 * (aero['FL'] + aero['RL'])
        r = ss.solve(ay, float(longitudinal_g), aero_Fz=aero)
        for c in CORNERS:
            travel[c][i] = float(r.travel[c])
            drop[c][i] = travel[c][i] + (float(r.Fz[c]) - fz0[c]) / kt * 1000.0
        sink['F'][i], sink['R'][i] = r.aero_heave_front_mm, r.aero_heave_rear_mm
    wb = float(ss._veh.wheelbase_m)
    df = 0.5 * (drop['FL'] + drop['FR']); dr = 0.5 * (drop['RL'] + drop['RR'])
    return dict(speed_kph=sp, radius_m=radius_m, lateral_g=lat, longitudinal_g=float(longitudinal_g), downforce_N=dfN, travel_mm=travel,
                ride_drop_mm=drop, aero_sink_mm=sink, pitch_deg=np.degrees(np.arctan((df - dr) / 1000.0 / wb)),
                cla_m2=float(cla_m2), cop_rear=float(cop_rear), rho=float(rho), grip_limit_g=grip_limit_g)
