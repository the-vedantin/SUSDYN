"""
vahan/ik_decoupled.py -- Inverse kinematics for DECOUPLED topology.

Goal: find cradle hardpoint positions such that
  * the HEAVE damper compresses ONLY in pure heave (zero length-change
    under pure roll)
  * the ROLL damper compresses ONLY in pure roll (zero length-change
    under pure heave)

Mathematically: the decoupling between modes is governed by the
GEOMETRIC RATIO that maps wheel motion to damper compression.  Define

  c_h(z_h, z_r)  =  heave-damper compression  as fn of heave+roll wheel motion
  c_r(z_h, z_r)  =  roll-damper  compression  ditto

with  z_h = (z_L + z_R)/2   (symmetric / heave mode)
      z_r = (z_L - z_R)/2   (antisymmetric / roll mode).

Perfect decoupling means  ∂c_h/∂z_r ≡ 0  AND  ∂c_r/∂z_h ≡ 0  at every
operating point.  In the reference twin-bellcrank design these
partials are not identically zero (the cradle has asymmetric Z), but
they CAN be made very small by tuning a handful of attach Z values.

We optimise over a SMALL design vector (kept short so the inverse
problem is well-posed):

   x[0]  = roll_damper_left.Z   offset from its rocker pivot Z
   x[1]  = roll_damper_right.Z  offset from its rocker pivot Z
   x[2]  = heave_damper_left.Z  offset from its rocker pivot Z
   x[3]  = heave_damper_right.Z offset (mirrored from x[2] by default)

Everything else (pushrod attach, pivot location, arm lengths) is held
fixed -- those are physically constrained by the bellcrank geometry
that the user already designed.  If the user wants to vary more, they
can edit BOUNDS and FIXED dicts at the top of `optimize_decoupling`.

Returns the OPTIMISED hardpoint dict + a small report of the residual
cross-coupling at the design pose and at ±10 mm operational sweep.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .monoshock import TwinRockerDecoupledSolver


def cross_coupling_metric(hp_dict: dict,
                          p_out_L_design: np.ndarray,
                          p_out_R_design: np.ndarray,
                          travels_mm=(2.0, 5.0, 10.0)) -> dict:
    """Measure how much cross-coupling the current cradle geometry has.

    Returns a dict with:
        heave_leakage_in_roll_mm  : RMS  |c_h|  under pure roll across travels
        roll_leakage_in_heave_mm  : RMS  |c_r|  under pure heave across travels
        worst_heave_leak_mm       : max |c_h| in roll
        worst_roll_leak_mm        : max |c_r| in heave
        total_metric              : the scalar we minimise (sum of squares)
    """
    solver = TwinRockerDecoupledSolver(hp_dict, p_out_L_design, p_out_R_design)
    heave_leaks = []
    roll_leaks  = []
    for t_mm in travels_mm:
        dz = t_mm * 1e-3
        # Pure heave: both wheels +dz
        st_h = solver.solve(p_out_L_design + np.array([0,0,dz]),
                            p_out_R_design + np.array([0,0,dz]))
        # Pure roll: L +dz, R -dz
        st_r = solver.solve(p_out_L_design + np.array([0,0, dz]),
                            p_out_R_design + np.array([0,0,-dz]))
        roll_leaks.append(abs(st_h.roll_delta))    # roll damper sees motion in heave -- LEAK
        heave_leaks.append(abs(st_r.heave_delta))  # heave damper sees motion in roll  -- LEAK
    heave_arr = np.asarray(heave_leaks) * 1000.0
    roll_arr  = np.asarray(roll_leaks)  * 1000.0
    total = float(np.sum(heave_arr**2 + roll_arr**2))
    return {
        'heave_leakage_in_roll_mm': float(np.sqrt(np.mean(heave_arr**2))),
        'roll_leakage_in_heave_mm': float(np.sqrt(np.mean(roll_arr**2))),
        'worst_heave_leak_mm':      float(np.max(heave_arr)),
        'worst_roll_leak_mm':       float(np.max(roll_arr)),
        'total_metric':             total,
    }


def optimize_decoupling(hp_dict: dict,
                        p_out_L_design: np.ndarray,
                        p_out_R_design: np.ndarray,
                        travels_mm=(2.0, 5.0, 10.0),
                        full_3d: bool = True,
                        max_iter: int = 400,
                        verbose: bool = False) -> tuple[dict, dict]:
    """Inverse-kinematics for the asymmetric monoshock cradle.

    Per the user: "bellcrank points are free estate".  The pushrod
    attach on each rocker MUST stay close to the outboard end of the
    upper control arm path (to prevent the A-arm from carrying side
    bending loads) -- but the damper attaches can move freely in 3-D
    on each bellcrank plate.

    Design vector (if full_3d=True, default):
        12 scalars -- (X, Y, Z) of each of the 4 damper attaches:
            x[0:3]   = heave_damper_left  (X, Y, Z)
            x[3:6]   = heave_damper_right (X, Y, Z)
            x[6:9]   = roll_damper_left   (X, Y, Z)
            x[9:12]  = roll_damper_right  (X, Y, Z)
        With full-3D freedom the optimiser can find geometries where
        the heave damper's axis is aligned with the bellcrank's swept
        path under symmetric motion (and the roll damper's axis aligned
        with the path under antisymmetric motion) -- which is the
        textbook way to achieve perfect decoupling.

    Design vector (if full_3d=False):
        4 scalars -- only the Z of each attach (legacy mode).

    pushrod_inner_{left,right} are NEVER varied -- they are
    structurally constrained to the outboard end of the upper control
    arm to keep load transmission through tension/compression rather
    than bending.

    Bounds: damper attaches stay within ±150 mm of the rocker pivot in
    every axis -- a realistic envelope for a FSAE bellcrank plate.
    """
    base = {k: np.array(v, float).copy() for k, v in hp_dict.items()}
    piv_L = base['rocker_pivot_left']
    piv_R = base['rocker_pivot_right']

    # Maximum offset from pivot in any direction (m) -- envelope for a
    # physically reasonable bellcrank plate.
    R_ENV = 0.15

    if full_3d:
        x0 = np.concatenate([
            base['heave_damper_left'],
            base['heave_damper_right'],
            base['roll_damper_left'],
            base['roll_damper_right'],
        ])
        bounds = []
        for piv in (piv_L, piv_R, piv_L, piv_R):
            bounds.extend([
                (piv[0] - R_ENV, piv[0] + R_ENV),
                (piv[1] - R_ENV, piv[1] + R_ENV),
                (piv[2] - R_ENV, piv[2] + R_ENV),
            ])

        def _apply(x: np.ndarray) -> dict:
            h = {k: v.copy() for k, v in base.items()}
            h['heave_damper_left']  = x[0:3].copy()
            h['heave_damper_right'] = x[3:6].copy()
            h['roll_damper_left']   = x[6:9].copy()
            h['roll_damper_right']  = x[9:12].copy()
            return h
    else:
        # Legacy mode: vary Z only (4-D design vector)
        x0 = np.array([
            base['heave_damper_left'][2],
            base['heave_damper_right'][2],
            base['roll_damper_left'][2],
            base['roll_damper_right'][2],
        ])
        bounds = [
            (piv_L[2] - R_ENV, piv_L[2] + R_ENV),
            (piv_R[2] - R_ENV, piv_R[2] + R_ENV),
            (piv_L[2] - R_ENV, piv_L[2] + R_ENV),
            (piv_R[2] - R_ENV, piv_R[2] + R_ENV),
        ]

        def _apply(x: np.ndarray) -> dict:
            h = {k: v.copy() for k, v in base.items()}
            h['heave_damper_left'][2]  = x[0]
            h['heave_damper_right'][2] = x[1]
            h['roll_damper_left'][2]   = x[2]
            h['roll_damper_right'][2]  = x[3]
            return h

    def _objective(x: np.ndarray) -> float:
        h = _apply(x)
        try:
            return cross_coupling_metric(h, p_out_L_design, p_out_R_design,
                                         travels_mm)['total_metric']
        except Exception:
            return 1e9   # huge penalty on solver failure

    before = cross_coupling_metric(base, p_out_L_design, p_out_R_design,
                                   travels_mm)
    if verbose:
        print(f'before: heave_leak={before["heave_leakage_in_roll_mm"]:.4f} mm  '
              f'roll_leak={before["roll_leakage_in_heave_mm"]:.4f} mm  '
              f'metric={before["total_metric"]:.6f}')

    res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-12,
                            'gtol': 1e-10})
    x_opt = res.x
    new_hp = _apply(x_opt)
    after = cross_coupling_metric(new_hp, p_out_L_design, p_out_R_design,
                                  travels_mm)
    if verbose:
        print(f'after:  heave_leak={after["heave_leakage_in_roll_mm"]:.4f} mm  '
              f'roll_leak={after["roll_leakage_in_heave_mm"]:.4f} mm  '
              f'metric={after["total_metric"]:.6f}')
        print(f'        x_opt = {x_opt}')

    return new_hp, {
        'before': before, 'after': after,
        'x_opt': x_opt.tolist(),
        'iter':  int(res.nit),
        'converged': bool(res.success),
        'message': str(res.message),
    }
