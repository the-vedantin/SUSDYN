"""vahan/ymd.py — THE yaw-moment state engine (Milliken Moment Method).

WHY THIS FILE EXISTS
--------------------
The tool already answers "which Ackermann?" three ways (per-wheel heading
alignment, the Fz-Fy map, RCVD ch.7 pair analysis).  The missing criterion is
the Milliken-canon one (Big-12, 2026-07): sweep Ackermann and score each
setting by the highest lateral acceleration the car can hold at ZERO net yaw
moment.  A point with N != 0 is a car that is still rotating — a transient the
driver cannot park on the skidpad — so only trimmed (N = 0) points count.
At each such trim the two moment sensitivities are read off:

    control    N_delta = dN/d(steer)      — how hard the car rotates per
                                            degree of front wheel; want > 0
    stability  N_beta  = dN/d(body slip)  — how hard it self-corrects per
                                            degree of attitude; want < 0

ONE ENGINE.  Every yaw-moment number in the tool comes from `ymd_state` below.
plot_mmd (vahan/analysis_plots.py) used to carry its own private state
iteration with a real physics flaw: rear slip was set equal to body slip —
only true for a car that is not rotating.  That iteration is deleted and
rerouted through here; do not grow another one.

THE YAW-RATE TERM IS THE POINT
------------------------------
In a steady turn the car rotates at r = Ay*g/V, and the axles do NOT see the
same sideslip: the rotation adds  l_f*r/V  of slip at the front axle and
l_r*r/V  at the rear (RCVD Eqs. 5.3/5.4, printed p147-148).  On an 8 m corner
that is degrees(l_r/R) ~ 5 deg of rear slip that the old plot_mmd iteration
simply did not have — its rear axle always understated slip at high Ay.

SIGN CONVENTION (fixed here, used consistently by every consumer)
-----------------------------------------------------------------
Left turn positive:  delta >= 0 = steer left = FL is the inner wheel;
Ay > 0 = accelerating left; r > 0 = yawing nose-left; N > 0 = moment that
rotates the nose INTO the left turn.

beta (body slip) is the ATTITUDE of the nose relative to the velocity vector,
positive = nose pointed INTO the turn (nose-left of the path in a left turn).
This is the modern moment-method convention (the one behind "positive control
and negative stability derivatives are always desirable") and it is the
MIRROR of RCVD ch.5's velocity-sideslip beta: RCVD's own CN-Ay build-up
(printed p304-305) runs its right-hand-turn example at beta NEGATIVE —
"rotating the whole vehicle to the right, as in a RH turn" — i.e. nose-in is
the negative direction of THEIR axis for THEIR hand.  Flip the hand and the
axis and you get this file's convention; the physics (RCVD Eqs. 5.3/5.4 slip
build-up, stability read as the slope of a constant-delta line through trim,
printed p308) is identical.  With nose-in beta a statically stable car shows
N_beta < 0 is the meaningful readout.  N_delta AT A MAXIMUM-TRIM POINT is 0
by construction (else it was not the max); finite computed values there are
step-size + tyre-grid artifact (refuted 2026-07-28, sign unstable with h) and
must not be presented as physics.  Away from the max-trim point N_delta is the
Milliken-desirable signs come out directly.

Forces are resolved into the BODY frame per wheel (RCVD ch.7 resolves the
axle by cos(reference steer); same idea):  F_y = F_tire*cos(steer) corners,
F_x = -F_tire*sin(steer) is induced drag, and

    N = sum( x_i * F_y,i  -  y_i * F_x,i )

The second term — the inner/outer drag split acting at +-track/2 — is how
Ackermann itself yaws the car (pro-Ackermann drags the inside, pulling the
nose in) and how the heavily-loaded outer wheel's drag pushes the nose OUT.
Measured on the 2027 car (v56, 12 psi, x0.70, 2026-07-28): without this term
the trimmed limit read 1.62 g, rear-limited, Ackermann-blind to 1e-5 g (at
trim with delta free, Ay is set by the rear axle alone — the front split can
only matter through saturation or these projection terms); with it the limit
is front-moment-limited, stable (N_beta < 0), and the sweep separates by
0.035 g at 8 m / 0.14 g at 4.5 m — mildly PRO-Ackermann at both radii, the
inside wheel's drag buying the front the last few Nm of nose-in moment at
the edge of the trimmable attitude window.  Note this criterion scores TRIM
CAPABILITY only; RCVD ch.19's case for reverse (tyre temperature, drag,
p715-716) is bought at part-throttle corners this metric does not see —
compare all four Ackermann methods, that is what they are for.

Slip angles (degrees, into-turn positive to match every tyre lookup in this
codebase — the TTC curves are SAE, +slip -> -Fy, negated once at the lookup):

    front wheel i :  s_i = delta_i + beta - degrees(l_f * r / V)
    rear  wheel i :  s_i = toe_i   + beta + degrees(l_r * r / V)

Equivalently, against the RCVD/SAE textbook form alpha_F = beta_v + l_f*r/V -
delta, alpha_R = beta_v - l_r*r/V (beta_v = velocity sideslip): s = -alpha and
beta = -beta_v.  Either way |rear slip - beta| = degrees(l_r*r/V), which is
the regression gate on the yaw-rate term.

LOADS come from the steady solver — the ONE solved model — tabulated once
over Ay (solver.solve(ay).Fz, the same pattern plot_ackermann_fz_fy uses) and
mirrored for the other hand by LOAD, never by label: the solver's positive
lat_g happens to load FL on this car, so wheels are matched light<->inner
(same lesson as vahan/ackermann.py).  Optional aero (the per-corner
N-per-lateral-g dict from MainWindow._get_aero_Fz_per_g) is handed to the
solver's own aero_Fz input scaled by that row's Ay, so load transfer and
camber react to it — no second aero model here.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

G = 9.81            # matches vahan/dynamics.py
_CORNERS = ('FL', 'FR', 'RL', 'RR')


def _dims(solver):
    """Mass and lever arms from the solver's own VehicleParams (or any
    duck-typed car exposing the same attributes on ._veh)."""
    veh = getattr(solver, '_veh', solver)
    m = float(veh.total_mass_kg)
    L = float(veh.wheelbase_m)
    wf = float(veh.front_weight_fraction)
    l_f = L * (1.0 - wf)          # front axle to CG
    l_r = L * wf                  # CG to rear axle
    t_f = float(veh.front_track_m)
    t_r = float(getattr(veh, 'rear_track_m', t_f))
    return m, L, l_f, l_r, t_f, t_r


def build_loads_table(solver, aero_Fz_per_g=None, ay_max=3.0, n=13):
    """Per-corner Fz and camber vs |Ay|, solved once and reused.

    The steady solver costs real time per solve (it iterates the kinematics),
    and one trim sweep evaluates thousands of operating points, so loads are
    tabulated on an Ay grid and interpolated — the identical trade
    plot_ackermann_fz_fy makes, just cached.  Wheels are stored as
    inner(light)/outer(heavy) per axle so the table serves BOTH turn hands.

    aero_Fz_per_g: falsy, or the per-corner N-at-1-lateral-g dict; each grid
    row's solve receives dict*ay through the solver's own aero_Fz input
    (V^2-scaling on a fixed radius — same as vahan/ackermann.py).
    """
    ays, rows = [], []
    for ay in np.linspace(0.0, float(ay_max), int(n)):
        aero_row = ({k: float(aero_Fz_per_g.get(k, 0.0)) * float(ay)
                     for k in _CORNERS} if aero_Fz_per_g else None)
        try:
            res = solver.solve(float(ay), 0.0, aero_Fz=aero_row)
        except Exception:
            break                       # wheel-lift refusal etc.: stop the table
        fz = {k: max(float(res.Fz[k]), 0.0) for k in _CORNERS}
        cam = {k: abs(float((getattr(res, 'camber', None) or {}).get(k, 0.0)))
               for k in _CORNERS}
        row = {}
        for axle, (a, b) in (('F', ('FL', 'FR')), ('R', ('RL', 'RR'))):
            # match by LOAD, never by label (ackermann.py lesson): the solver
            # does not share this file's idea of which way the car is turning.
            light, heavy = (a, b) if fz[a] <= fz[b] else (b, a)
            row['fz_in_' + axle] = fz[light]
            row['fz_out_' + axle] = fz[heavy]
            row['cam_in_' + axle] = cam[light]
            row['cam_out_' + axle] = cam[heavy]
        ays.append(float(ay))
        rows.append(row)
    if len(ays) < 3:
        raise RuntimeError('loads table: steady solver refused nearly every '
                           'Ay point — check the model')
    table = {'ay': np.asarray(ays, float)}
    for k in rows[0]:
        table[k] = np.asarray([r[k] for r in rows], float)
    return table


def _ackermann_split(delta_deg, ackermann_pct, t_f, L):
    """Front steer split.  The 100% (kinematic) spread grows with the square
    of steer:  spread ~ (t_f/L)*delta^2  [radians]; ackermann_pct scales it,
    negative = reverse (outer steered more).  delta >= 0 = left turn = FL
    inner."""
    d = float(delta_deg)
    spread = (math.degrees((t_f / L) * math.radians(abs(d)) ** 2)
              * (float(ackermann_pct) / 100.0) * (1.0 if d >= 0.0 else -1.0))
    d_in, d_out = d + spread / 2.0, d - spread / 2.0
    return (d_in, d_out) if d >= 0.0 else (d_out, d_in)     # (FL, FR)


def ymd_state(tire_model, solver, beta_deg, delta_deg, *,
              radius_m=None, V_mps=None, ackermann_pct=0.0,
              grip_multiplier=1.0, aero_Fz_per_g=None,
              toe_front_deg=0.0, toe_rear_deg=0.0,
              loads_table=None, max_iter=40, tol_g=1e-4, Ay0=0.0):
    """One quasi-steady operating point (beta, delta) -> converged Ay and N.

    Modes (exactly one):
      radius_m — constant-RADIUS (skidpad): V^2 = |Ay|*g*R, re-coupled every
                 iteration, so speed follows the cornering level.
      V_mps    — constant-SPEED (the MMD's frame).

    grip_multiplier: belt->asphalt derate applied at the Fy lookup (forward
    evaluation, same as ackermann.py PART 2); Ay stays ROAD g.
    toe_*: static toe, + = toe-in, applied as fixed hardware steer per wheel
    (left wheel -toe, right wheel +toe in this left-positive convention) — it
    does not flip with turn direction, because the car's toe doesn't.
    Ay0: starting guess for the fixed-point iteration.  The map can hold TWO
    attractors (one per turn hand — the hand flips the load mirroring and the
    yaw-term signs), and strongly nose-out states seeded at 0 were measured
    falling into the MIRROR hand (4.5 m, beta=-9: converged to Ay=-0.21 with
    rear slip -18 deg instead of the real left-turn state).  Seed on the hand
    you are analysing; the trim search seeds +0.75.

    Returns dict: Ay_g, N_Nm, V_mps, yaw_rate_radps, per-wheel slip/Fy/Fz,
    delta per front wheel, converged flag.
    """
    if (radius_m is None) == (V_mps is None):
        raise ValueError('give exactly one of radius_m or V_mps')
    gm = min(max(float(grip_multiplier), 0.05), 1.5)
    m, L, l_f, l_r, t_f, t_r = _dims(solver)
    table = loads_table if loads_table is not None else \
        build_loads_table(solver, aero_Fz_per_g)

    d_fl, d_fr = _ackermann_split(delta_deg, ackermann_pct, t_f, L)
    toe_f, toe_r = float(toe_front_deg), float(toe_rear_deg)
    beta = float(beta_deg)
    # Wheel steer angles in the BODY frame (left +): fronts get the Ackermann
    # split, every wheel gets its static toe (left wheel -toe, right +toe).
    ang = np.array([d_fl - toe_f, d_fr + toe_f, -toe_r, +toe_r])
    cos_a, sin_a = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    # wheel positions for the moment sum: x ahead of CG, y left of CG
    x_w = np.array([l_f, l_f, -l_r, -l_r])
    y_w = np.array([t_f / 2.0, -t_f / 2.0, t_r / 2.0, -t_r / 2.0])

    def _eval(Ay):
        V = (math.sqrt(max(abs(Ay), 1e-3) * G * float(radius_m))
             if radius_m is not None else float(V_mps))
        V = max(V, 0.5)
        r = Ay * G / V
        yaw_f = math.degrees(l_f * r / V)
        yaw_r = math.degrees(l_r * r / V)
        # Loads: interpolate at |Ay|; inner(light) side is the LEFT side of
        # the car for Ay >= 0 (left turn transfers load to the right).
        a_abs = abs(Ay)
        left, right = ('in', 'out') if Ay >= 0.0 else ('out', 'in')
        fz = np.array([np.interp(a_abs, table['ay'], table['fz_' + left + '_F']),
                       np.interp(a_abs, table['ay'], table['fz_' + right + '_F']),
                       np.interp(a_abs, table['ay'], table['fz_' + left + '_R']),
                       np.interp(a_abs, table['ay'], table['fz_' + right + '_R'])])
        cam = np.array([np.interp(a_abs, table['ay'], table['cam_' + left + '_F']),
                        np.interp(a_abs, table['ay'], table['cam_' + right + '_F']),
                        np.interp(a_abs, table['ay'], table['cam_' + left + '_R']),
                        np.interp(a_abs, table['ay'], table['cam_' + right + '_R'])])
        # into-turn slip per wheel: FL, FR, RL, RR (see module docstring)
        s = np.array([ang[0] + beta - yaw_f,
                      ang[1] + beta - yaw_f,
                      ang[2] + beta + yaw_r,
                      ang[3] + beta + yaw_r])
        # SAE curves: +slip -> -Fy, negated once so + = perpendicular to the
        # WHEEL, toward the inside of the turn.
        ft = -np.atleast_1d(np.asarray(
            tire_model.Fy(s, fz, cam), float)).ravel() * gm
        return V, r, s, fz, ft

    def _sums(ft):
        # Resolve each wheel's force into the BODY frame (RCVD ch.7 resolves
        # axle force by cos(reference steer); same thing per wheel here):
        #   F_y,i = F_t,i cos(ang_i)         lateral, does the cornering
        #   F_x,i = -F_t,i sin(ang_i)        induced drag along the car
        # N about the CG = sum(x_i*F_y,i - y_i*F_x,i).  The second term is the
        # inner/outer DRAG SPLIT at +-t/2 — the mechanism by which Ackermann
        # itself yaws the car (pro-Ackermann drags the inside of the car,
        # pulling the nose in) — and it is exactly what a front-force-only
        # moment sum cannot see.
        fy_body = ft * cos_a
        fx_body = -ft * sin_a
        Ay_g = float(fy_body.sum()) / (m * G)
        N = float((x_w * fy_body - y_w * fx_body).sum())
        return Ay_g, N, fy_body

    Ay, converged = float(Ay0), False
    for _ in range(int(max_iter)):
        V, r, s, fz, ft = _eval(Ay)
        Ay_new, _, _ = _sums(ft)
        if abs(Ay_new - Ay) < tol_g:
            Ay = Ay_new
            converged = True
            break
        Ay = 0.5 * Ay + 0.5 * Ay_new        # damped, same scheme plot_mmd used
    V, r, s, fz, ft = _eval(Ay)
    _, N, fy_body = _sums(ft)
    lab = dict(zip(_CORNERS, range(4)))
    return {
        'Ay_g': float(Ay), 'N_Nm': float(N),
        'V_mps': float(V), 'yaw_rate_radps': float(r),
        'beta_deg': beta, 'delta_deg': float(delta_deg),
        'delta_wheel_deg': {'FL': d_fl, 'FR': d_fr},
        'slip_deg': {k: float(s[i]) for k, i in lab.items()},
        'Fy_N': {k: float(fy_body[i]) for k, i in lab.items()},
        'Fz_N': {k: float(fz[i]) for k, i in lab.items()},
        'converged': bool(converged),
    }


def trim_point(tire_model, solver, beta_deg, *,
               radius_m=None, V_mps=None, ackermann_pct=0.0,
               grip_multiplier=1.0, aero_Fz_per_g=None, loads_table=None,
               delta_range=None, n_scan=14):
    """Find delta where N = 0 at this beta; return that trimmed state.

    delta_range default is geometry-aware: the scan must reach past the
    corner's kinematic wheel angle atan(L/R) (a 4.5 m FSAE hairpin already
    needs ~19 deg before any slip), so the ceiling is max(30, atan(L/R)+18).

    N(delta) rises with steer while the front axle still has grip and rolls
    over once it saturates, so there can be several zero crossings.  Only the
    FIRST rising crossing counts (N going - to + as steer increases, i.e.
    positive control): that is the equilibrium a skidpad driver reaches by
    winding on steer, and its limit as the fronts saturate IS the maximum
    trimmed point (RCVD's point T, printed p306, fronts saturated).  Later
    crossings are deep-drift equilibria at 13-15+ deg of front slip — past
    the tyre rig's measured +/-12 deg sweep, reachable only through a
    control-reversed region, and (measured on this car's data, 2026-07-28)
    sitting on an all-wheels-saturated grip plateau that reads THE SAME Ay
    for every Ackermann setting, so keeping any of them silently erased the
    criterion's whole discrimination.  No rising crossing -> has_trim=False,
    honestly, with `reason` saying WHY so the sweep can steer its search:
      'below'     — branch peaks with N still negative: the front cannot null
                    the rear's moment at this beta (too nose-in)
      'above'     — branch starts with N already positive: the rear never
                    engages at this beta (too nose-out)
      'no_branch' — no same-hand states at all in the scan window
    """
    kw = dict(radius_m=radius_m, V_mps=V_mps, ackermann_pct=ackermann_pct,
              grip_multiplier=grip_multiplier, aero_Fz_per_g=aero_Fz_per_g,
              loads_table=loads_table, Ay0=0.75)
    if delta_range is None:
        _, L, _, _, _, _ = _dims(solver)
        d_hi = 30.0 if radius_m is None else \
            max(30.0, math.degrees(math.atan2(L, float(radius_m))) + 18.0)
        delta_range = (-4.0, d_hi)
    deltas = np.linspace(float(delta_range[0]), float(delta_range[1]),
                         int(n_scan))
    sts = [ymd_state(tire_model, solver, beta_deg, d, **kw) for d in deltas]
    Ns = np.array([st['N_Nm'] for st in sts])
    Ays = np.array([st['Ay_g'] for st in sts])

    def _no_trim(reason):
        return {'has_trim': False, 'reason': reason,
                'beta_deg': float(beta_deg), 'Ay_g': float('nan'),
                'delta_deg': float('nan'), 'converged': False}

    # Same-hand (left-turn) states only.  The scan's low-delta end converges
    # to the MIRROR hand (Ay < 0) where N is hugely positive — an unfiltered
    # argmax landing there made every small-beta trim vanish (measured: at
    # beta=0 the delta=-1.4 state reads N=+3094 while the real left-turn trim
    # sits at delta=10.6 under a branch peak of only +258).
    ok = np.isfinite(Ns) & np.isfinite(Ays) & (Ays > 0.02)
    if not ok.any():
        return _no_trim('no_branch')
    idx = np.flatnonzero(ok)
    # First rising crossing along the branch.  (Taking the HIGHEST-Ay
    # crossing instead let one setting hop to a later crossing beyond a local
    # dip and report an off-trend number: measured, +100% read 1.49 g at a
    # 2nd crossing while every other setting read 1.42 at its 1st.)
    i_cross = None
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a == 1 and Ns[a] < 0.0 <= Ns[b]:
            i_cross = a
            break
    if i_cross is None:
        # branch starting positive = rear never engages (too nose-out);
        # starting negative and never reaching 0 = front short (too nose-in)
        return _no_trim('above' if Ns[int(idx[0])] >= 0.0 else 'below')
    lo, hi = float(deltas[i_cross]), float(deltas[i_cross + 1])
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ymd_state(tire_model, solver, beta_deg, mid, **kw)['N_Nm'] < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    st = ymd_state(tire_model, solver, beta_deg, 0.5 * (lo + hi), **kw)
    st['has_trim'] = True
    st['reason'] = 'trim'
    return st



def control_at_point(tire_model, solver, ackermann_list, *, radius_m=None,
                     V_mps=None, beta_deg=-1.0, delta_deg=None, lat_g=None,
                     grip_multiplier=1.0, aero_Fz_per_g=None,
                     loads_table=None, steps=(0.10, 0.25, 0.50)):
    """Control and stability derivatives at ONE COMMON operating point.

    WHY THIS EXISTS.  `trim_sweep_ackermann` reports dN/ddelta at each
    setting's own maximum-trim point, where it is zero by construction —
    finite values there are numerical artifact (see this module's header).
    Comparing settings needs the derivatives read at the SAME (beta, delta)
    for every setting, away from the maximum, with the difference step shown
    not to change the answer.

    Two ways to fix the point:
      * `delta_deg` given  — that steer angle, that body slip, all settings.
      * `lat_g` given      — the steer angle that puts THE FIRST setting in
                             the list at that lateral g (bisection), then the
                             SAME angle for every other setting, so the
                             comparison is at one operating point rather than
                             one performance level.

    Returns one row per setting: N_delta, N_beta, Ay_g, plus `step_spread`
    (max minus min of N_delta across `steps`).  A step_spread comparable to
    the between-setting spread means the number is not resolvable — the
    caller must say so rather than rank on it.
    """
    kw = dict(grip_multiplier=grip_multiplier, aero_Fz_per_g=aero_Fz_per_g,
              loads_table=loads_table)
    if (radius_m is None) == (V_mps is None):
        raise ValueError('give exactly one of radius_m or V_mps')
    kw['radius_m' if V_mps is None else 'V_mps'] = (radius_m if V_mps is None
                                                    else V_mps)

    def _ay(pct, b, d, seed=0.75):
        st = ymd_state(tire_model, solver, b, d, ackermann_pct=float(pct),
                       Ay0=seed, **kw)
        return st['Ay_g'], st['N_Nm']

    if delta_deg is None:
        if lat_g is None:
            raise ValueError('give delta_deg or lat_g')
        lo, hi = 0.5, 25.0
        ref = float(ackermann_list[0])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = ((mid, hi) if _ay(ref, beta_deg, mid)[0] < lat_g
                      else (lo, mid))
        delta_deg = 0.5 * (lo + hi)

    out = []
    for pct in ackermann_list:
        ay0, _ = _ay(pct, beta_deg, delta_deg)
        nd = []
        for h in steps:
            _, n_p = _ay(pct, beta_deg, delta_deg + h, ay0)
            _, n_m = _ay(pct, beta_deg, delta_deg - h, ay0)
            nd.append((n_p - n_m) / (2.0 * h))
        hb = steps[1] if len(steps) > 1 else steps[0]
        _, nb_p = _ay(pct, beta_deg + hb, delta_deg, ay0)
        _, nb_m = _ay(pct, beta_deg - hb, delta_deg, ay0)
        out.append({'pct': float(pct), 'Ay_g': ay0,
                    'beta_deg': float(beta_deg), 'delta_deg': float(delta_deg),
                    'N_delta': float(np.median(nd)),
                    'N_delta_by_step': dict(zip(steps, [float(x) for x in nd])),
                    'step_spread': float(max(nd) - min(nd)),
                    'N_beta': float((nb_p - nb_m) / (2.0 * hb))})
    return out


def mmm_metrics(tire_model, solver, ackermann_pct, *, radius_m=None,
                V_mps=None, grip_multiplier=1.0, aero_Fz_per_g=None,
                loads_table=None, control_frac=0.85):
    """The four quantities RCVD Chapter 8 reads off a CN-Ay moment diagram,
    computed the way the book defines them (docs/rcvd_ref/ch08 + ch05).

    Every earlier Ackermann "control" number was dN/ddelta read AT the
    maximum-trim point, where the book states it is zero by construction
    (control moment goes to zero at max trim, printed p320-321).  This
    function replaces that with the book's own diagram readings:

      1. ay_trim_max        RCVD point T (printed p306): the largest lateral g
                            on the trim line CN=0 (fronts saturated).  The grip
                            number.
      2. stability_index    RCVD printed p308: slope of a constant-STEER line
                            through trim, dCN/dAy.  Negative = stable.  Read at
                            a well-conditioned SUB-LIMIT trim, not at T.
      3. control_moment_avail  RCVD printed p320-321: from the trim point, the
                            CN height gained up the constant-BODY-SLIP line to
                            the front-tyre boundary — how much yaw the steering
                            can still command.  Read at the same sub-limit trim.
                            Returned as a coefficient (CN) and as N*m.
      4. limit_character    RCVD printed p306: sign of the apex point P
                            (max untrimmed force) relative to the Ay axis.
                            apex CN < 0 = PLOW (final understeer), ~0 = neutral
                            drift, > 0 = SPIN (final oversteer).  The car's Ay>0
                            is a left turn (module header): nose-out at the apex
                            (CN<0) is the plow direction.

    CN = N / (W*wheelbase), matching the book (printed p301) and plot_mmd.

    Physics is all ymd_state; this only sweeps the diagram and reads it.
    """
    kw = dict(radius_m=radius_m, V_mps=V_mps, ackermann_pct=float(ackermann_pct),
              grip_multiplier=grip_multiplier, aero_Fz_per_g=aero_Fz_per_g,
              loads_table=loads_table)
    m, L, l_f, l_r, t_f, t_r = _dims(solver)
    W = m * 9.80665
    WL = W * L

    # ---- point T: sweep body slip, trim at each, take the max-Ay trim -------
    kin = (math.degrees(l_r / float(radius_m)) if radius_m else 3.0)
    betas = np.arange(-kin - 8.0, 3.01, 0.5)
    trims = []
    for b in betas:
        tp = trim_point(tire_model, solver, float(b),
                        radius_m=radius_m, V_mps=V_mps,
                        ackermann_pct=float(ackermann_pct),
                        grip_multiplier=grip_multiplier,
                        aero_Fz_per_g=aero_Fz_per_g, loads_table=loads_table)
        if tp.get('has_trim') and np.isfinite(tp['Ay_g']):
            trims.append((tp['Ay_g'], tp['beta_deg'], tp['delta_deg']))
    if not trims:
        return {'ackermann_pct': float(ackermann_pct),
                'ay_trim_max': float('nan'), 'stability_index': float('nan'),
                'control_moment_avail_CN': float('nan'),
                'control_moment_avail_Nm': float('nan'),
                'apex_ay': float('nan'), 'apex_cn': float('nan'),
                'limit_character': 'no trim'}
    trims.sort()
    ay_T, b_T, d_T = trims[-1]

    # ---- sub-limit operating trim (well-conditioned) -----------------------
    ay_op_target = control_frac * ay_T
    ay_op, b_op, d_op = min(trims, key=lambda x: abs(x[0] - ay_op_target))

    def _state(b, d):
        st = ymd_state(tire_model, solver, float(b), float(d), Ay0=ay_op, **kw)
        return st['Ay_g'], st['N_Nm']

    # ---- (2) stability index: dCN/dAy along constant STEER through trim -----
    hb = 0.5
    ay_bp, n_bp = _state(b_op + hb, d_op)
    ay_bm, n_bm = _state(b_op - hb, d_op)
    d_ay = ay_bp - ay_bm
    stability_index = ((n_bp - n_bm) / d_ay / WL) if abs(d_ay) > 1e-9         else float('nan')

    # ---- (3) control moment available: up the constant-BODY-SLIP line ------
    # from the trim point (CN=0), increase steer at fixed body slip until the
    # front saturates; the peak CN reached is the control moment available.
    d_hi = d_op + max(8.0, abs(d_op))
    ds = np.linspace(d_op, d_hi, 22)
    cns = []
    for d in ds:
        _, n = _state(b_op, d)
        cns.append(n / WL)
    cn_peak = float(np.max(cns)) if cns else float('nan')
    control_cn = max(cn_peak, 0.0)          # gained above the CN=0 trim
    control_nm = control_cn * WL

    # ---- (4) apex P + limit character --------------------------------------
    # max-Ay state over a beta x steer grid = vector sum of front+rear
    # saturation (the book's point P).
    ap_ay, ap_cn = -1e9, float('nan')
    for b in np.arange(b_T - 4.0, b_T + 4.01, 1.0):
        for d in np.arange(d_T - 2.0, d_T + 8.01, 1.0):
            ay, n = _state(b, d)
            if ay > ap_ay:
                ap_ay, ap_cn = ay, n / WL
    if not np.isfinite(ap_cn):
        char = 'unknown'
    elif ap_cn < -0.004:
        char = 'PLOW (final understeer)'
    elif ap_cn > 0.004:
        char = 'SPIN (final oversteer)'
    else:
        char = 'NEUTRAL drift (race target)'

    return {'ackermann_pct': float(ackermann_pct),
            'ay_trim_max': float(ay_T),
            'beta_at_T': float(b_T), 'delta_at_T': float(d_T),
            'ay_op': float(ay_op), 'beta_op': float(b_op), 'delta_op': float(d_op),
            'stability_index': float(stability_index),
            'control_moment_avail_CN': float(control_cn),
            'control_moment_avail_Nm': float(control_nm),
            'apex_ay': float(ap_ay), 'apex_cn': float(ap_cn),
            'limit_character': char}


def mmm_metrics_sweep(tire_model, solver, ackermann_list, *, radius_m=None,
                      V_mps=None, grip_multiplier=1.0, aero_Fz_per_g=None,
                      loads_table=None, control_frac=0.85):
    """mmm_metrics for each Ackermann setting; shares one loads table."""
    if loads_table is None:
        loads_table = build_loads_table(solver, aero_Fz_per_g)
    return [mmm_metrics(tire_model, solver, float(p), radius_m=radius_m,
                        V_mps=V_mps, grip_multiplier=grip_multiplier,
                        aero_Fz_per_g=aero_Fz_per_g, loads_table=loads_table,
                        control_frac=control_frac)
            for p in ackermann_list]


def trim_sweep_ackermann(tire_model, solver, radius_m, ackermann_list,
                         grip_multiplier=1.0, aero_Fz_per_g=None,
                         beta_range=None, beta_step=1.5,
                         loads_table=None):
    """Daniel's criterion: for each Ackermann %, the highest trimmed (N = 0)
    lateral acceleration on a fixed radius, plus the control/stability
    derivatives AT that trim.

    beta is scanned coarsely, extended while the maximum sits on a scan edge
    (adaptive), then refined around the best point.  The default range is
    GEOMETRY-AWARE, not fixed: on a fixed radius the rear axle's kinematic
    crab is degrees(l_r/R) regardless of speed (r/V = 1/R), so rear slip =
    beta + degrees(l_r/R) and the trims live at beta in roughly
    [-2, +12] - degrees(l_r/R).  A fixed 0..10 window misses every trim on a
    tight corner (4.5 m: crab is 9.2 deg, trims sit nose-OUT near beta -5).
    Derivatives are central differences of quasi-steady N (Ay re-converged
    at every probe):
        N_delta = dN/d(steer)     Nm/deg   — want positive (control)
        N_beta  = dN/d(body slip) Nm/deg   — want negative (stability)

    Returns one row per pct: pct, Ay_trim_max, beta_at, delta_at, N_delta,
    N_beta, converged.
    """
    table = loads_table if loads_table is not None else \
        build_loads_table(solver, aero_Fz_per_g)
    if beta_range is None:
        _, _, _, l_r, _, _ = _dims(solver)
        crab = math.degrees(l_r / float(radius_m))
        beta_range = (max(-2.0 - crab, -25.0), min(12.0 - crab, 25.0))
    kw = dict(radius_m=float(radius_m), grip_multiplier=grip_multiplier,
              aero_Fz_per_g=aero_Fz_per_g, loads_table=table)
    rows = []
    for pct in ackermann_list:
        pct = float(pct)

        def _trim(b):
            return trim_point(tire_model, solver, b,
                              ackermann_pct=pct, **kw)

        lo, hi = float(beta_range[0]), float(beta_range[1])
        betas = list(np.arange(lo, hi + 1e-9, float(beta_step)))
        seen = {b: _trim(b) for b in betas}

        def _best():
            return max(seen, key=lambda b: (seen[b]['has_trim'],
                                            np.nan_to_num(seen[b]['Ay_g'],
                                                          nan=-1e9)))

        best_b = _best()
        if not seen[best_b]['has_trim']:
            # The trim band can be NARROWER than the coarse step: measured at
            # 4.5 m, every beta <= -5 reads 'above' (rear disengaged) and
            # every beta >= -4 reads 'below' (front moment short) — if a band
            # exists it hides between adjacent samples.  Refine across each
            # above/below transition before declaring no trim.
            def _kind(b):
                r = seen[b].get('reason')
                return 'above' if r == 'no_branch' else r
            bs = sorted(seen)
            for a, b in zip(bs[:-1], bs[1:]):
                if {_kind(a), _kind(b)} == {'above', 'below'}:
                    for nb in np.linspace(a, b, 7)[1:-1]:
                        seen[float(nb)] = _trim(float(nb))
            best_b = _best()
        # adaptive: chase the max off a scan edge (bounded, so a pathological
        # curve cannot run away)
        for _ in range(8):
            if best_b <= min(seen) + 1e-9:
                nb = min(seen) - float(beta_step)
            elif best_b >= max(seen) - 1e-9:
                nb = max(seen) + float(beta_step)
            else:
                break
            if abs(nb) > 25.0:
                break
            seen[nb] = _trim(nb)
            if seen[nb]['has_trim'] and (not seen[best_b]['has_trim']
                                         or seen[nb]['Ay_g'] >
                                         seen[best_b]['Ay_g']):
                best_b = nb
            else:
                break
        # refine around the winner
        for b in np.arange(best_b - float(beta_step),
                           best_b + float(beta_step) + 1e-9,
                           float(beta_step) / 3.0):
            b = float(b)
            if any(abs(b - k) < 1e-6 for k in seen):
                continue
            seen[b] = _trim(b)
            if seen[b]['has_trim'] and (not seen[best_b]['has_trim']
                                        or seen[b]['Ay_g'] >
                                        seen[best_b]['Ay_g']):
                best_b = b
        # The maximum usually sits at the EDGE of the trimmable beta window
        # (measured: Ay_trim climbs ~0.13 g/deg toward it at 8 m), so a grid
        # answer is quantised by step*slope ~ 0.06 g — worse than the
        # differences being compared.  Bisect the window edge(s).
        if seen[best_b]['has_trim']:
            bs = sorted(seen)
            i0 = bs.index(best_b)
            for side in (+1, -1):
                j = i0 + side
                if not (0 <= j < len(bs)) or seen[bs[j]]['has_trim']:
                    continue
                b_ok, b_bad = best_b, bs[j]
                for _ in range(6):
                    mid = 0.5 * (b_ok + b_bad)
                    st_m = _trim(mid)
                    seen[mid] = st_m
                    if st_m['has_trim']:
                        b_ok = mid
                        if st_m['Ay_g'] > seen[best_b]['Ay_g']:
                            best_b = mid
                    else:
                        b_bad = mid
        best = seen[best_b]
        if not best['has_trim']:
            rows.append({'pct': pct, 'Ay_trim_max': float('nan'),
                         'beta_at': float('nan'), 'delta_at': float('nan'),
                         'N_delta': float('nan'), 'N_beta': float('nan'),
                         'converged': False})
            continue
        b0, d0 = best['beta_deg'], best['delta_deg']
        h = 0.25
        # Seed every probe at the trim's own Ay: with the default seed the
        # probe can fall into the OTHER hand's attractor and hand back a
        # different branch's moment (measured at 4.5 m: N_delta read -19.8
        # unseeded where the true local slope is +30).
        dkw = dict(kw, Ay0=best['Ay_g'])
        N_dp = ymd_state(tire_model, solver, b0, d0 + h,
                         ackermann_pct=pct, **dkw)['N_Nm']
        N_dm = ymd_state(tire_model, solver, b0, d0 - h,
                         ackermann_pct=pct, **dkw)['N_Nm']
        N_bp = ymd_state(tire_model, solver, b0 + h, d0,
                         ackermann_pct=pct, **dkw)['N_Nm']
        N_bm = ymd_state(tire_model, solver, b0 - h, d0,
                         ackermann_pct=pct, **dkw)['N_Nm']
        rows.append({'pct': pct,
                     'Ay_trim_max': best['Ay_g'],
                     'beta_at': b0, 'delta_at': d0,
                     # KEPT FOR CONTINUITY, NOT FOR PHYSICS.  At a maximum-trim
                     # point dN/ddelta is 0 by construction (else it was not the
                     # maximum), so this number is step-size and tyre-grid
                     # artifact — the docstring at the top of this file has said
                     # so since 2026-07-28, and it was quoted as a design
                     # discriminator anyway (+24 vs +7 Nm/deg "control collapse
                     # at deep reverse", retracted 2026-08-02: at a common
                     # SUB-LIMIT point the ordering reverses).  Read
                     # control_at_point() instead.
                     'N_delta': (N_dp - N_dm) / (2.0 * h),
                     'N_delta_valid': False,
                     'N_beta': (N_bp - N_bm) / (2.0 * h),
                     'converged': bool(best['converged'])})
    return rows


class LLTDCar:
    """Closed-form corner-load 'car' for the MMD's what-if knobs.

    plot_mmd lets the user perturb roll-stiffness distribution (ARB lever
    deltas) and see the diagram move — a parameter STUDY, deliberately not
    tied to one solved chassis state.  This class owns that closed-form load
    model (rigid car, LLTD split by roll-stiffness fraction — exactly the
    formula plot_mmd carried privately) and duck-types the two things the ymd
    engine reads from a car: `.solve(ay).Fz/.camber` and the `._veh` dims.
    The STATE ITERATION stays in ymd_state; only the loads source differs.
    """

    def __init__(self, total_mass_kg, weight_dist_front, wheelbase_m,
                 cg_height_m, track_front_m, track_rear_m,
                 roll_stiffness_front_Npm_rad=0.0,
                 roll_stiffness_rear_Npm_rad=0.0,
                 delta_arb_front_Npm=0.0, delta_arb_rear_Npm=0.0):
        m = float(total_mass_kg)
        wf = float(weight_dist_front)
        self._m, self._h = m, float(cg_height_m)
        self._t_f, self._t_r = float(track_front_m), float(track_rear_m)
        # K_roll = K_wheel * t^2 / 2; the delta knobs add/remove ARB
        # wheel-rate to explore LLTD (moved here verbatim from plot_mmd).
        K_f = max(float(roll_stiffness_front_Npm_rad)
                  + float(delta_arb_front_Npm) * self._t_f ** 2 / 2.0, 0.0)
        K_r = max(float(roll_stiffness_rear_Npm_rad)
                  + float(delta_arb_rear_Npm) * self._t_r ** 2 / 2.0, 0.0)
        self.lltd_front = K_f / (K_f + K_r) if (K_f + K_r) > 1.0 else 0.5
        self._fz_f0 = m * G * wf / 2.0            # static front, per corner
        self._fz_r0 = m * G * (1.0 - wf) / 2.0
        self._veh = SimpleNamespace(
            total_mass_kg=m, wheelbase_m=float(wheelbase_m),
            front_weight_fraction=wf,
            front_track_m=self._t_f, rear_track_m=self._t_r)

    def solve(self, lateral_g, longitudinal_g=0.0, aero_Fz=None):
        ay = float(lateral_g)
        lt_f = self.lltd_front * self._m * G * ay * self._h / self._t_f
        lt_r = (1.0 - self.lltd_front) * self._m * G * ay * self._h / self._t_r
        fz = {'FL': self._fz_f0 - lt_f, 'FR': self._fz_f0 + lt_f,
              'RL': self._fz_r0 - lt_r, 'RR': self._fz_r0 + lt_r}
        if aero_Fz:
            for k in _CORNERS:
                fz[k] += float(aero_Fz.get(k, 0.0))
        fz = {k: max(v, 10.0) for k, v in fz.items()}     # same floor plot_mmd used
        return SimpleNamespace(Fz=fz, camber={k: 0.0 for k in _CORNERS})
