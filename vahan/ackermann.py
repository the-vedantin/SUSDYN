"""Solving for the right Ackermann percentage.  ONE procedure, two parts.

This file exists because the same question kept getting answered a different way
every time it was asked.  There is now exactly one method here, it lives in
Vahan (not in a report script), and every consumer calls it.

THE IDEA, IN PLAIN TERMS
------------------------
The car is going round a circle.  Every wheel is at a different place on the
car, so every wheel goes round its OWN circle of its own size.  A wheel that is
further from the middle of the turn travels a bigger circle than one nearer to
it, and a wheel further forward is swung round differently again.

Wherever a wheel is, at that instant it is travelling along the tangent to its
own circle.  That is the direction it is actually going.

A tyre only makes side force if it is pointed slightly away from the direction
it is going.  That angle is the slip angle.  So:

    where the wheel must POINT  =  the direction it is GOING  +  its slip angle

Do that for both front wheels.  They are going in slightly different directions
(different circles) and they need different slip angles (different loads, and a
more heavily loaded tyre needs a bigger slip angle).  So the two front wheels
must point at different angles.  How different they point IS the Ackermann.

    Ackermann %  =  (how much further the inner wheel is turned than the outer)
                    -------------------------------------------------------- x100
                    (the difference between their two travel directions)

100% means both wheels are pointed exactly along their own travel direction.

WHY IT CHANGES WITH SPEED ON THE SAME CIRCLE
--------------------------------------------
Two reasons, and both are in here:
  1. Faster round the same circle means more load thrown onto the outer wheels,
     so the two front tyres need more different slip angles from each other.
  2. Faster also means the car sits more sideways to its own path (it crabs),
     which swings every wheel's travel direction round.

PART 1 uses travel direction only.  Slip angles come from the solved car, which
already works them out for all four wheels.  No tyre curves are touched here.

PART 2 asks a different question about the same corner: of all the Ackermann
settings, which one puts the most force in the direction that actually holds the
car on the circle?  A tyre's force pushes sideways to the WHEEL, and the two
front wheels point different ways, so their two forces point different ways too.
Only the part of each force aimed at the middle of the turn does any good; the
rest just drags.  A wheel turned the wrong way makes force that pulls the car
OUT of the corner, and that is subtracted, not ignored.

Run both and you get a bucket of ideal percentages across lateral g.
"""
from __future__ import annotations

import math
import numpy as np

G = 9.80665


def _turn_geometry(veh, radius_m, beta_deg, front=True):
    """Which way each wheel is travelling, and how far it is from the middle
    of the turn.  Angles in degrees, measured from straight ahead, positive
    turning left."""
    L = float(veh.wheelbase_m)
    wf = float(veh.front_weight_fraction)
    l_f = L * (1.0 - wf)          # front axle ahead of the balance point
    l_r = L * wf                  # rear axle behind it
    t_f = float(veh.front_track_m)
    t_r = float(veh.rear_track_m)
    b = math.radians(beta_deg)
    # Middle of the turn: straight out to the left of where the car is going.
    ox, oy = -radius_m * math.sin(b), radius_m * math.cos(b)
    out = {}
    for lbl, (px, py) in (
            ('FL', (l_f, +t_f / 2.0)), ('FR', (l_f, -t_f / 2.0)),
            ('RL', (-l_r, +t_r / 2.0)), ('RR', (-l_r, -t_r / 2.0))):
        dx, dy = px - ox, py - oy          # from middle of turn to the wheel
        # travelling at right angles to that, going round the turn
        out[lbl] = {'heading_deg': math.degrees(math.atan2(dx, -dy)),
                    'radius_m': math.hypot(dx, dy)}
    return out


def solve_ackermann_geometry(solver, tire_model, radius_m, lat_g,
                             grip_multiplier: float = 1.0, aero=False):
    """PART 1 — point each front wheel where it needs to point, and read off
    the Ackermann that requires.

    AERO (optional).  ``aero`` is either falsy (no downforce — unchanged
    behaviour) or the per-corner downforce dict NORMALISED TO 1 LATERAL G
    that the GUI already applies everywhere else
    (MainWindow._get_aero_Fz_per_g — solved-deficit or custom-CFD source).
    On a fixed corner radius, sweeping lateral g IS sweeping speed
    (V^2 = lat_g * 9.81 * R) and downforce goes as V^2, so this row's load
    is simply dict * lat_g — the identical scaling the dynamics view uses
    (_get_active_aero_Fz), handed to the steady-state solver's OWN aero_Fz
    input so load transfer, camber and the Fz split all react to it.  No
    second aero model lives here.

    Downforce adds VERTICAL load but NO lateral demand — the car's mass is
    unchanged.  So the per-wheel demand is generalised: each AXLE must hold
    its static weight share times lat_g (front axle reacts m*a*l_r/L =
    m*a*wf, the split PART 2 already uses), and each wheel takes the axle's
    demand in proportion to its (aero-inclusive) vertical load:

        Fy_i = (W_axle * lat_g) * Fz_i / sum(Fz over that axle)

    With aero OFF this reduces EXACTLY to the old per-wheel Fy = Fz*lat_g:
    solve() renormalises each axle pair back to its static weight share
    (dynamics.py, per-axle clamp + renorm), so sum(Fz_axle) = W_axle, the
    ratio W_axle/sum(Fz_axle) is 1, and Fy_i = Fz_i * lat_g identically —
    gated to 1e-9 deg in test_one_model.py.  With aero ON the denominator
    grows (sum = W_axle + downforce on that axle) while the numerator does
    not, so every wheel's demanded mu sits BELOW lat_g — downforce is grip
    for free, and the inverted slip angles come out SMALLER at the same g.

    GRIP MULTIPLIER (belt -> asphalt).  The tyre data is belt-rig (Calspan
    TIRF) and belt grip is 0.65-0.75x asphalt-real — the project's derate
    band.  lat_g stays ROAD g throughout: the corner demand is what the car
    really asks for on asphalt.  A multiplier < 1 derates the belt curve so
    saturation appears at the ROAD limit (~1.5-1.7 g, slips at their 7-9 deg
    peaks) instead of the belt limit (~2.3 g, slips reading low).  The
    mechanism is inversion-side: an asphalt tyre making X newtons is a belt
    tyre making X / multiplier, so we invert the UNMODIFIED belt curve for
    fy_needed / multiplier — the demand is untouched, the curve is just
    worth less.  Applied identically to all four wheels, rears included
    (the crab solve uses their slips too).  1.0 = raw belt curves.

    THE METHOD (the user's, 2026-07-27, replacing every earlier attempt):

    Slip angle is a STATE, not a steering output.  It starts at zero and grows
    as the cornering gets more aggressive, on a different trajectory for each
    wheel, because each wheel does a different amount of work.  Steering does
    not set it — steering's job is to keep each wheel's slip angle sitting
    tangential to that wheel's own circle around the turn centre.

    So, per wheel:
        work     = each wheel corners ITS OWN vertical load:
                       Fy_needed = Fz * lateral_g
        slip     = whatever that wheel's OWN MEASURED CURVE requires to make
                   that force at that load — inverted straight from the TTC
                   data (rising branch, the part the rig measured well;
                   above the data ceiling the same measured-trend
                   extrapolation as every other lookup in the tool)
        point it = its own travel tangent + its slip angle

    NOTHING equalises the wheels.  No shared slip angle, no shared
    utilisation, no arriving at the limit together — the outer runs out of
    grip first, because it works at a higher fraction of a lower-mu curve.
    "Slip rises with load" comes OUT of the inversion; it is not an input.
    The earlier "placement rules" (peak_frac / util / slip / own_peak) are
    deleted: they existed only because a broken force split had erased the
    per-wheel information, and each was an invented equalisation the user
    explicitly rejected.  The peak-slip-trend knob drops out with them —
    inversion never asks where a curve peaks, so the one quantity this
    dataset cannot measure is no longer needed below saturation.

    Sanity anchor for free: as lat_g -> 0 every slip -> 0 and this returns
    exactly 100% Ackermann — pure geometry.

    The car's attitude (crab) is still not a choice: the rear wheels are
    bolted straight, so the car sits at exactly the attitude that leaves the
    rears at the slip angle their own inversion demands.

    Per-wheel Fy = Fz*lat_g is self-consistent by construction: summed over
    an axle it reproduces the axle's share of m*a (the axle's vertical load
    is its weight share), so no separate force-balance step is needed.
    """
    veh = solver._veh
    # Clamp rather than raise: a fat-fingered 0 would divide fy_needed by
    # zero, and anything past 1.5 claims asphalt beats the belt by more than
    # any surface pairing in the literature.
    gm = min(max(float(grip_multiplier), 0.05), 1.5)
    if aero and not isinstance(aero, dict):
        # A bare `aero=True` carries no package data and would silently mean
        # "no aero" if allowed through — fail loudly instead.
        raise TypeError('aero must be falsy or the per-corner N-per-lateral-g '
                        'dict from MainWindow._get_aero_Fz_per_g()')
    # This row's downforce: the canonical per-g package scaled by this row's
    # g (V^2 at fixed radius — see docstring), through the solver's own
    # aero_Fz path (the same input AeroDownforceSolver exercises).
    aero_row = ({k: float(aero.get(k, 0.0)) * float(lat_g)
                 for k in ('FL', 'FR', 'RL', 'RR')} if aero else None)
    res = solver.solve(float(lat_g), 0.0, aero_Fz=aero_row)
    fz_all = {k: max(float(res.Fz[k]), 0.0) for k in ('FL', 'FR', 'RL', 'RR')}
    # Camber matters: the light front wheel runs ~+1 deg where camber thrust
    # cancels a real fraction of the slip force on this tyre.
    cam_all = {k: abs(float((res.camber or {}).get(k, 0.0)))
               for k in ('FL', 'FR', 'RL', 'RR')}
    LIFTED_N = 5.0          # below this a wheel is off the ground

    # Axle lateral demand comes from the CAR'S MASS, never from Fz: the
    # front axle holds W*wf*lat_g and the rear W*(1-wf)*lat_g whether or not
    # a wing is pressing down.  Each wheel takes its axle's demand in
    # proportion to its vertical load; with aero off the ratio collapses to
    # exactly Fz*lat_g (see docstring), so nothing moves on the old path.
    W = float(veh.total_mass_kg) * G
    wf_veh = float(veh.front_weight_fraction)
    axle_W = {'F': W * wf_veh, 'R': W * (1.0 - wf_veh)}
    axle_fz = {'F': fz_all['FL'] + fz_all['FR'],
               'R': fz_all['RL'] + fz_all['RR']}

    sa = {}
    saturated = {}
    util = {}
    for lb in ('FL', 'FR', 'RL', 'RR'):
        fz = fz_all[lb]
        if fz < LIFTED_N:
            # A wheel in the air does no work and needs no slip angle.
            sa[lb] = 0.0
            saturated[lb] = False
            util[lb] = 0.0
            continue
        # Its axle's weight share cornered, split by vertical load (ROAD).
        # max() guards the one-wheel-airborne axle, where the grounded wheel
        # takes the whole axle demand (fz/axle_fz -> 1).
        fy_needed = axle_W[lb[0]] * float(lat_g) * fz / max(axle_fz[lb[0]],
                                                            1e-9)
        # Belt derate at the inversion: the road demand stays fy_needed, but
        # a derated curve must be asked for fy_needed / gm to represent the
        # same asphalt force.  All four wheels, rears included — the crab
        # solve below reads sa['RL']/sa['RR'].
        sa[lb] = float(tire_model.slip_angle_for_Fy(
            fy_needed / gm, fz, cam_all[lb]))
        # slip_angle_for_Fy sets this when the demand exceeds what the wheel
        # can make at ANY angle — the wheel is past its limit and the returned
        # angle is its peak, not a solution.  Surfaced, never hidden.
        saturated[lb] = bool(getattr(tire_model, 'last_lookup_saturated',
                                     False))
        # How hard this wheel is being asked to work, as a fraction of what it
        # can actually make.  Reported so an impossible ask is VISIBLE as a
        # number, not inferred from a footnote.
        _cap = abs(float(tire_model.peak_Fy(max(fz, 1.0), cam_all[lb]))) * gm
        util[lb] = fy_needed / max(_cap, 1e-9)

    def rear_mismatch(beta):
        g = _turn_geometry(veh, radius_m, beta)
        # rear wheels are unsteered: pointing straight ahead is 0 deg
        return 0.5 * ((g['RL']['heading_deg'] + sa['RL'])
                      + (g['RR']['heading_deg'] + sa['RR']))

    # rear_mismatch RISES with crab angle, so if it is positive the answer is
    # BELOW mid.  (Had this the wrong way round first time: the search ran
    # straight to its +/-30 deg bound and every number downstream was noise.)
    lo, hi = -30.0, 30.0
    f_lo, f_hi = rear_mismatch(lo), rear_mismatch(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f'no crab angle in +/-30 deg puts the rear wheels at the slip they '
            f'need (mismatch {f_lo:+.2f} to {f_hi:+.2f} deg) — check radius/g')
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if rear_mismatch(mid) > 0 else (mid, hi)
    beta = 0.5 * (lo + hi)

    g = _turn_geometry(veh, radius_m, beta)
    # The geometry above is written for a LEFT turn, so FL is the wheel on the
    # small circle.  The solver does not have to agree about which way the car
    # is turning — on this car it reports FL as the HEAVY wheel — so match the
    # two by LOAD, never by label: the inner wheel is the lightly loaded one.
    # (Matching by label first time put 1101 N on the inner wheel and 193 N on
    # the outer, which inverts the whole answer.)
    lighter = 'FL' if res.Fz['FL'] <= res.Fz['FR'] else 'FR'
    heavier = 'FR' if lighter == 'FL' else 'FL'
    inner, outer = 'FL', 'FR'                # geometric inner / outer
    sa_in, sa_out = sa[lighter], sa[heavier]
    fz_in, fz_out = float(res.Fz[lighter]), float(res.Fz[heavier])
    head_in, head_out = g[inner]['heading_deg'], g[outer]['heading_deg']
    point_in = head_in + sa_in
    point_out = head_out + sa_out
    spread_travel = head_in - head_out          # travel-alignment reference
    spread_point = point_in - point_out
    # TWO DIFFERENT "100%" REFERENCES — do not conflate them (caught by the
    # user 2026-07-31 when a 2.5 deg demand printed as -376%):
    #
    # STANDARD Ackermann % (the industry definition, and the one every
    # consumer should display): 0% = parallel steer, 100% = both wheels
    # converging on the LOW-SPEED turn centre (on the rear-axle line).  Its
    # denominator is the STATIC geometric spread at this radius — a pure
    # function of wheelbase/track/radius that does NOT change with lateral g.
    #
    # The travel-based ratio (kept as ackermann_pct_dynamic) divides by the
    # body-slip-included travel spread, which collapses toward zero as crab
    # pushes the turn centre level with the front axle — mixing a slip-physics
    # numerator with a collapsing geometric denominator.  Useful as an
    # alignment diagnostic, meaningless as "Ackermann %".
    # Static construction = turn centre ON THE REAR-AXLE LINE, which is the
    # crab angle that points both rear wheels straight ahead (their headings
    # zero) — NOT beta = 0, which parks the centre level with the CG and
    # halves the spread (net gate caught exactly that, 2026-07-31).
    _b_lo, _b_hi = -30.0, 30.0
    def _rear_head(b):
        _g = _turn_geometry(veh, radius_m, b)
        return 0.5 * (_g['RL']['heading_deg'] + _g['RR']['heading_deg'])
    if _rear_head(_b_lo) * _rear_head(_b_hi) < 0:
        for _ in range(60):
            _bm = 0.5 * (_b_lo + _b_hi)
            _b_lo, _b_hi = ((_b_lo, _bm) if _rear_head(_b_lo) * _rear_head(_bm) <= 0
                            else (_bm, _b_hi))
    _beta0 = 0.5 * (_b_lo + _b_hi)
    g0 = _turn_geometry(veh, radius_m, _beta0)
    spread_static = (g0[inner]['heading_deg'] - g0[outer]['heading_deg'])
    pct = (100.0 * spread_point / spread_static
           if abs(spread_static) > 0.05 else float('nan'))
    pct_dyn = (100.0 * spread_point / spread_travel
               if spread_travel > 0.05 else float('nan'))
    _peak_bound = spread_travel - (
        float(tire_model.peak_slip_angle(fz_out))
        - float(tire_model.peak_slip_angle(fz_in)))
    V = math.sqrt(max(lat_g, 1e-6) * G * radius_m)
    # Downforce actually inside THIS row's solve, read back from the
    # solver's own conservation (solve() renormalises SumFz to weight +
    # aero, dynamics.py) — never recomputed from an aero formula here.
    downforce_N = ((sum(float(res.Fz[k]) for k in ('FL', 'FR', 'RL', 'RR'))
                    - W) if aero_row else 0.0)
    return {
        'lat_g': float(lat_g), 'radius_m': float(radius_m),
        'speed_mps': V, 'speed_kph': V * 3.6,
        'crab_deg': beta,
        # These are the GEOMETRIC inner/outer wheels (FL/FR for a left turn),
        # which is what every *_travel/_point value below belongs to.  Reporting
        # the load-matched labels here made result['inner'] say 'FR' while
        # result['inner_travel_deg'] held FL's heading.
        'inner': inner, 'outer': outer,
        'inner_load_wheel': lighter, 'outer_load_wheel': heavier,
        'inner_lifted': bool(fz_in < LIFTED_N),
        # True when that wheel is asked for more force than it can make at
        # ANY angle — its slip is its peak, not a solved point.
        'inner_saturated': bool(saturated[lighter]),
        'outer_saturated': bool(saturated[heavier]),
        # How hard each front wheel is asked to work vs what it can make.
        # >1 means the corner is IMPOSSIBLE at this g on this surface.
        'inner_util': float(util.get(lighter, float('nan'))),
        'outer_util': float(util.get(heavier, float('nan'))),
        # THE TRUST FLAG.  A saturated wheel's slip angle is the PEAK-ANGLE
        # FALLBACK from slip_angle_for_Fy, not a solved value — so any toe
        # difference built from it is meaningless.  Worse, when BOTH fronts
        # saturate they return the SAME fallback angle and the toe difference
        # collapses to the bare tangent difference, i.e. exactly 100%
        # Ackermann.  That is what made this solver look erratic across g:
        # -1.27 deg at 1.5 g, -4.39 at 1.7 g, then +0.145 at 1.9 and 2.0 g —
        # the last two were pure geometry wearing a result's clothes.
        # Consumers MUST check this before using point_spread_deg.
        'valid': not (bool(saturated[lighter]) or bool(saturated[heavier])),
        'inner_travel_deg': head_in, 'outer_travel_deg': head_out,
        'inner_slip_deg': sa_in, 'outer_slip_deg': sa_out,
        'inner_point_deg': point_in, 'outer_point_deg': point_out,
        'travel_spread_deg': spread_travel,
        # The STANDARD 100%-Ackermann reference at this radius (static
        # construction, turn centre on the rear-axle line).  g-independent.
        'static_spread_deg': spread_static,
        # THE ANSWER, in degrees.  Read this before the percentage: the
        # percentage divides by travel_spread, and on a 10 m circle with any
        # crab the turn centre creeps up level with the front axle and that
        # spread collapses toward zero, so the percentage runs away while the
        # degrees stay perfectly well behaved.  Degrees are also what a
        # steering arm is actually built to.
        'point_spread_deg': (spread_point
                             if not (saturated[lighter] or saturated[heavier])
                             else float('nan')),
        'point_spread_raw_deg': spread_point,   # the fallback-built value
        # STANDARD definition: spread_point / static geometric spread.
        'ackermann_pct': (pct if not (saturated[lighter]
                                      or saturated[heavier])
                          else float('nan')),
        # Travel-alignment diagnostic (old denominator) — NOT Ackermann %.
        'ackermann_pct_dynamic': (pct_dyn if not (saturated[lighter]
                                                  or saturated[heavier])
                                  else float('nan')),
        # PHYSICAL CAP on the demand (user 2026-07-31): at the limit both
        # wheels sit at their OWN peaks, so the demanded split can never be
        # more reverse than travel_spread - (peak_slip(outer) - peak_slip
        # (inner)).  The equal-load-share prescription overshoots this once
        # the outer rides the flat top of its curve (measured: -2.52 deg
        # asked at 1.6 g vs -1.20 deg physical bound).  Consumers should
        # display the capped value; the raw one stays for diagnostics.
        'at_peak_bound_deg': _peak_bound,
        'at_peak_capped': bool(spread_point < _peak_bound - 0.05),
        'point_spread_capped_deg': (max(spread_point, _peak_bound)
                                    if not (saturated[lighter]
                                            or saturated[heavier])
                                    else float('nan')),
        'ackermann_pct_capped': (100.0 * max(spread_point, _peak_bound)
                                 / spread_static
                                 if abs(spread_static) > 0.05
                                 and not (saturated[lighter]
                                          or saturated[heavier])
                                 else float('nan')),
        'Fz_inner': fz_in, 'Fz_outer': fz_out,
        # The (clamped) belt->asphalt derate this row was solved with, so a
        # consumer can label its g axis honestly (road g vs raw belt).
        'grip_multiplier': gm,
        # Whether downforce was in this solve, and how much at this row's
        # speed — 0.0 when off, so consumers can always print the field.
        'aero': bool(aero_row),
        'downforce_N': float(downforce_N),
    }


def solve_ackermann_force(tire_model, solver, radius_m, lat_g,
                          ack_range=(-1000.0, 1000.0), n=401,
                          grip_multiplier: float = 1.0, aero=False):
    """PART 2 — of all the Ackermann settings, which one puts the most force
    in the direction that holds the car on the circle?

    The force a tyre makes pushes sideways to THE WHEEL.  Ackermann points the
    two front wheels different ways, so their forces point different ways too.
    Each force is split into the part aimed at the middle of the turn (useful)
    and the part aimed along the car's path (drag, which only slows it).  A
    wheel turned past its travel direction makes force the wrong way round and
    is subtracted.

    grip_multiplier: same belt->asphalt derate as PART 1 (0.65-0.75 band, see
    solve_ackermann_geometry).  Here the tyre is evaluated FORWARD, not
    inverted, so the derate scales each wheel's belt Fy down at the lookup —
    demand_N stays the car's real road demand.

    aero: same per-corner N-per-lateral-g dict as PART 1 (or falsy).  It
    raises the wheel loads the tyre is evaluated at (fz_in/fz_out below come
    from PART 1's aero-inclusive solve), but demand_N stays m*a*wf — the
    wing presses down, it does not add mass to corner.
    """
    # Same clamp as PART 1 so both parts of one call agree on the derate.
    gm = min(max(float(grip_multiplier), 0.05), 1.5)
    geo = solve_ackermann_geometry(solver, tire_model, radius_m, lat_g,
                                   grip_multiplier=gm, aero=aero)
    veh = solver._veh
    m = float(veh.total_mass_kg)
    wf = float(veh.front_weight_fraction)
    demand = m * float(lat_g) * G * wf          # what the front pair must hold
    head_in, head_out = geo['inner_travel_deg'], geo['outer_travel_deg']
    fz_in, fz_out = geo['Fz_inner'], geo['Fz_outer']
    beta = geo['crab_deg']
    travel_spread = geo['travel_spread_deg']
    # sweep the setting; for each, let the driver turn the wheel as far as
    # needed and take the best that setting can do
    steer0 = 0.5 * (geo['inner_point_deg'] + geo['outer_point_deg'])
    steers = np.linspace(steer0 - 6.0, steer0 + 16.0, 141)
    accs = np.linspace(ack_range[0], ack_range[1], int(n))
    best = {'useful_N': -1e18}
    rows = []
    for ack in accs:
        half = 0.5 * travel_spread * float(ack) / 100.0
        bu, brec = -1e18, None
        u_of_s, d_of_s = [], []
        for s in steers:
            p_in, p_out = s + half, s - half         # where each wheel points
            a_in, a_out = p_in - head_in, p_out - head_out   # slip angles
            # gm derates the belt curve to asphalt at the point of evaluation
            fy_in = -float(tire_model.Fy(a_in, fz_in, 0.0)) * gm
            fy_out = -float(tire_model.Fy(a_out, fz_out, 0.0)) * gm
            # resolve each force onto the direction that holds the circle
            u = (fy_in * math.cos(math.radians(p_in - beta))
                 + fy_out * math.cos(math.radians(p_out - beta)))
            d = (fy_in * math.sin(math.radians(p_in - beta))
                 + fy_out * math.sin(math.radians(p_out - beta)))
            u_of_s.append(u)
            d_of_s.append(d)
            if u > bu:
                bu, brec = u, dict(useful_N=u, drag_N=d, steer_deg=s,
                                   inner_slip_deg=a_in, outer_slip_deg=a_out,
                                   inner_N=fy_in * math.cos(math.radians(p_in - beta)),
                                   outer_N=fy_out * math.cos(math.radians(p_out - beta)))
        # ── THE TRIM POINT: where this setting JUST holds the corner ───────
        # The best-steer record above is the setting's CEILING — the driver
        # turning as far as the tyre allows.  That is the right number for
        # "can this setting hold the corner at all", and the wrong one for
        # "what does it cost", because the scrub at maximum steer is not the
        # scrub a driver pays while merely holding the circle.  So read the
        # SAME swept curve at the first steer angle whose useful force reaches
        # the demand, linearly interpolated between grid points (the objective
        # is steep there, unlike at the flat peak where the argmax wanders).
        ua = np.asarray(u_of_s, float)
        da = np.asarray(d_of_s, float)
        k = np.argmax(ua >= demand) if np.any(ua >= demand) else -1
        if k > 0:
            f = ((demand - ua[k - 1]) / (ua[k] - ua[k - 1])
                 if ua[k] != ua[k - 1] else 0.0)
            brec['trim_steer_deg'] = float(steers[k - 1]
                                           + f * (steers[k] - steers[k - 1]))
            brec['trim_drag_N'] = float(da[k - 1] + f * (da[k] - da[k - 1]))
            brec['trim_ok'] = True
        elif k == 0:
            brec['trim_steer_deg'] = float(steers[0])
            brec['trim_drag_N'] = float(da[0])
            brec['trim_ok'] = True
        else:
            # no steer angle in the window reaches the demand — this setting
            # cannot hold this corner; the ceiling is the honest fallback.
            brec['trim_steer_deg'] = float(brec['steer_deg'])
            brec['trim_drag_N'] = float(brec['drag_N'])
            brec['trim_ok'] = False
        brec['ackermann_pct'] = float(ack)
        # CENSORED-DATA GUARD.  The rig sweeps +/-12-13 deg of slip and the
        # LIGHT-load curves are still rising when the sweep ends — their
        # peak was never measured.  If either wheel's slip at this setting's
        # best point sits within CENSOR_MARGIN of the sweep edge, the number
        # rests on unmeasured tyre — flag it so consumers can refuse it like
        # a saturated state.  (Measured 2026-07-30: the >100% argmax rows
        # are NOT censored — that artifact is the flat plateau below, not
        # this guard.)
        CENSOR_MARGIN = 0.75
        _sa_edge = float(np.max(np.abs(tire_model.sa_range))) - CENSOR_MARGIN
        brec['ceiling_censored'] = bool(
            abs(brec.get('inner_slip_deg', 0.0)) >= _sa_edge
            or abs(brec.get('outer_slip_deg', 0.0)) >= _sa_edge)
        rows.append(brec)
        if bu > best['useful_N']:
            best = brec
    u = np.array([r['useful_N'] for r in rows])
    accs_arr = np.array([r['ackermann_pct'] for r in rows])
    # IS THE OPTIMUM EVEN REAL?  Maximising total force over a free steer angle
    # is a nearly flat objective: the driver can always turn a bit further to
    # make up what the split cost, so the curve has a long plateau and its
    # argmax wanders.  Say how much force actually separates best from worst,
    # and refuse to present a bound-hugging argmax as a design answer.
    spread_N = float(u.max() - u.min())
    spread_pct = 100.0 * spread_N / max(abs(float(u.max())), 1e-9)
    k = int(np.argmax(u))
    at_bound = k <= 1 or k >= len(u) - 2
    decisive = (spread_pct >= 1.0) and not at_bound
    PLATEAU_TOL_FRAC = 0.005
    _ua_all = np.array([float(r.get('useful_N', np.nan)) for r in rows])
    _pc_all = np.array([float(r.get('ackermann_pct', np.nan)) for r in rows])
    _umax = np.nanmax(_ua_all) if np.isfinite(_ua_all).any() else np.nan
    if np.isfinite(_umax):
        _tie = _ua_all >= _umax * (1.0 - PLATEAU_TOL_FRAC)
        _plat_lo = float(np.min(_pc_all[_tie]))
        _plat_hi = float(np.max(_pc_all[_tie]))
    else:
        _plat_lo = _plat_hi = float('nan')

    return {'geometry': geo, 'demand_N': demand,
            'ackermann_pct': accs_arr,
            'useful_N': u,
            'drag_N': np.array([r['drag_N'] for r in rows]),
            'inner_N': np.array([r['inner_N'] for r in rows]),
            'outer_N': np.array([r['outer_N'] for r in rows]),
            # The trim operating point: the steer that JUST holds this corner,
            # and the SCRUB DRAG the car pays there.  `useful_N` above is the
            # ceiling (can it hold it at all); these are the running cost.
            'trim_steer_deg': np.array([r['trim_steer_deg'] for r in rows]),
            'trim_drag_N': np.array([r['trim_drag_N'] for r in rows]),
            'trim_ok': np.array([r['trim_ok'] for r in rows], bool),
            'best': best,
            'ceiling_censored': np.array([bool(r.get('ceiling_censored'))
                                          for r in rows]),
            # True when the argmax itself rests on unmeasured tyre — treat
            # the winner as UNSUPPORTED, exactly like a saturated row.
            'best_censored': bool(best.get('ceiling_censored', False)),
            # PLATEAU HONESTY.  Past ~100% the lightly loaded inner wheel is
            # already at the edge of its (small) friction circle, so extra
            # split buys fractions of a newton and the useful-force curve
            # goes FLAT (measured: +112% beats +100% by 0.01% at R=2.5 m,
            # +212% by 0.17% at R=8 m).  An argmax on a flat curve crowns
            # numerical crumbs — same disease as a noise-floor YMD winner.
            # Report the TIE BAND instead: every setting within
            # PLATEAU_TOL_FRAC of the max is indistinguishable, and the
            # honest headline is the LOWEST such setting (least extra split,
            # least scrub drag, same force).
            'plateau_pct_lo': _plat_lo,
            'plateau_pct_hi': _plat_hi,
            'plateau_tol_frac': PLATEAU_TOL_FRAC,
            'best_supported_pct': _plat_lo,
            'spread_N': spread_N, 'spread_pct': spread_pct,
            'at_bound': at_bound, 'decisive': decisive,
            # `meets_demand` on the BEST setting was vacuously True at every
            # g (even the worst setting cleared the demand), so it told the
            # user nothing.  Report the margin of the WORST setting instead:
            # that is the question worth asking - can any Ackermann choice
            # fail to hold this corner?
            'worst_useful_N': float(u.min()),
            'demand_margin_worst': float(u.min() - demand),
            'any_setting_fails': bool(u.min() < demand)}


def ackermann_bucket(tire_model, solver, radius_m=10.0,
                     lat_g_list=(1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 2.0),
                     include_force=False, grip_multiplier: float = 1.0,
                     aero=False):
    # include_force gates PART 2: its sweep is ~56k tyre lookups PER g
    # and per rule, which turned the GUI button into a multi-minute
    # hang.  Part 1 (the geometry answer, in degrees) is what the
    # solver popup shows; ask for Part 2 explicitly when you want it.
    """Both parts across a list of lateral g — the bucket of ideal settings.

    grip_multiplier: belt->asphalt derate, threaded to both parts (0.65-0.75
    is the project band; see solve_ackermann_geometry).  Every g in
    lat_g_list stays ROAD g.

    aero: falsy, or the per-corner N-per-lateral-g dict (see
    solve_ackermann_geometry) — threaded to both parts, so every row's
    downforce follows that row's speed on the fixed radius."""
    out = []
    for g in lat_g_list:
        p1 = solve_ackermann_geometry(solver, tire_model, radius_m, g,
                                      grip_multiplier=grip_multiplier,
                                      aero=aero)
        try:
            if not include_force:
                raise RuntimeError('skipped')
            p2 = solve_ackermann_force(tire_model, solver, radius_m, g,
                                       grip_multiplier=grip_multiplier,
                                       aero=aero)
            best2, dem, met = (p2['best']['ackermann_pct'], p2['demand_N'],
                               p2['meets_demand'])
            uz = p2['best']['useful_N']
            extra = {'force_spread_pct': p2['spread_pct'],
                     'force_decisive': p2['decisive'],
                     'force_at_bound': p2['at_bound']}
        except Exception:
            best2, dem, met, uz = float('nan'), float('nan'), False, float('nan')
            extra = {'force_spread_pct': float('nan'),
                     'force_decisive': False, 'force_at_bound': False}
        out.append({**p1, 'force_best_pct': best2, 'demand_N': dem,
                    'best_useful_N': uz, 'meets_demand': met, **extra})
    return out
