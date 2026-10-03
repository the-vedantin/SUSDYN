"""The ONE vertical-heave model: incremental wheel force vs bump travel from the
corner solver's spring length (progressive motion ratio + preload term), with
the tyre in series.  Used by the steady-state solver (aero heave in the corner
travel), the lap simulator's ride-height trace and the aero ride-height page.
Pure numpy; no imports from the rest of vahan (no cycles)."""
from __future__ import annotations

import numpy as np

_CACHE: dict = {}


def wheel_force_curves(veh, solvers) -> dict:
    """Per axle ('F', 'R'), the INCREMENTAL wheel force vs bump travel from the
    corner solver's spring length (progressive MR included) + the tyre rate:
    {'ts': travel m, 'fcum': N, 'kt': N/m}.  The ONE heave model — the lap
    sim's ride-height trace (LapSimulator._build_rate_curves) and the aero
    ride-height page (vahan.aero_ride) both use it.  See the LapSimulator
    method docstring for the virtual-work derivation (RCVD 16.3)."""
    out = {}
    for ax, label, mr0, spr, fs0 in (
        ('F', 'FL', veh.motion_ratio_front, veh.spring_rate_front_Npm,
         getattr(veh, 'static_spring_force_front_N', 0.0)),
        ('R', 'RL', veh.motion_ratio_rear, veh.spring_rate_rear_Npm,
         getattr(veh, 'static_spring_force_rear_N', 0.0))):
        ts = np.linspace(-0.05, 0.08, 66)            # 50 mm droop .. 80 mm bump (2 mm steps)
        sv = solvers.get(label)
        # CACHE (2026-09-26: 132 kinematic solves per fresh SteadyStateSolver made the
        # net 2.5x slower).  Key = this solver object + a GEOMETRY FINGERPRINT (its
        # spring length at -20 / 0 / +20 mm, 3 solves) + every rate input, so an
        # in-place hardpoint edit or a new spring can never hit a stale curve.
        key = None
        if sv is not None:
            try:
                fp = tuple(round(float(sv.solve(t).spring_length), 10) for t in (-0.02, 0.0, 0.02))
                key = (id(sv), fp, float(mr0), float(spr), float(fs0 or 0.0), float(veh.tire_rate_Npm))
            except Exception:
                key = None
        if key is not None and key in _CACHE:
            out[ax] = _CACHE[key]
            continue
        c = float(mr0) * ts                          # constant-MR fallback
        if sv is not None:
            try:
                sl = np.array([sv.solve(float(t)).spring_length for t in ts])
                i0 = int(np.argmin(np.abs(ts)))      # static sample
                comp = sl[i0] - sl                   # spring COMPRESSION from static
                # sign so bump travel reads as POSITIVE compression
                # whatever the solver's length convention
                if comp[-1] < 0:
                    comp = -comp
                if np.all(np.isfinite(comp)):
                    c = comp
            except Exception:
                pass
        # MR = dc/dq; second-order edges so MR(0) is the true static
        # tangent (first-order edges read MR(0) half a step into bump)
        mr = np.clip(np.gradient(c, ts, edge_order=2), 0.05, 5.0)
        F_s = max(float(fs0 or 0.0), 0.0) + float(spr) * c
        i0 = int(np.argmin(np.abs(ts)))
        fcum = F_s * mr - F_s[i0] * mr[i0]      # 0 at static; < 0 in droop
        # must be monotonic for the inverse lookup (load -> travel)
        fcum = np.maximum.accumulate(fcum)
        out[ax] = dict(ts=ts, fcum=fcum,
                       kt=max(float(veh.tire_rate_Npm), 1.0))
        if key is not None:
            if len(_CACHE) > 256:
                _CACHE.clear()
            _CACHE[key] = out[ax]
    return out


def heave_split_mm(curve: dict, load_per_wheel_N: float) -> tuple:
    """(suspension, tyre) compression in mm of one wheel under an ADDITIONAL
    vertical load, through the nonlinear wheel rate and the tyre in series."""
    load = float(load_per_wheel_N)            # < 0 = unloaded (droop), e.g. the front under acceleration
    return (float(np.interp(load, curve['fcum'], curve['ts'])) * 1000.0,
            load / curve['kt'] * 1000.0)
