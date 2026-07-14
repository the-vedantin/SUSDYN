# -*- coding: utf-8 -*-
"""design_city.py — local, token-free population design engine ("agent city").

The LLM-agent roles become algorithmic components so thousands of candidates
cost zero tokens:
    Dictator    -> objective functions (per-island weighting personalities)
    Simulation  -> the verified Vahan eval pipeline (kinematics + steady state)
    Faults      -> penalty functions (bump steer, scrub, roll, travel budget)
    Interference-> segment-distance constraint (reject < 10 mm clearance)
    Improvement -> the evolutionary loop itself (tournament + blend + mutate)
    Thorough    -> the metrics dict every candidate must produce
Two populations per user spec: FOUNDERS (Latin-hypercube from knowns bounds,
"start from scratch") and LINEAGE (seeded from existing configs, iterating).
Islands ("buildings") have different objective personalities; the archipelago
("city") shares a global Pareto archive. Every survivor is written out as a
loadable .vahan config + metrics.json + a card PNG for the GUI gallery.

CLI:  python design_city.py --out designs_city/run1 --islands 4 --pop 12 \
        --gens 4 --workers 4 --seed-configs configs/2027_v11_*.vahan
Balance metric: AXLE-AGGREGATE utilization (total axle demand / total axle
capacity) — per the 2026-07-11 correction; per-corner max is NOT used for
limits (inner-tire sub-data-floor artifact).
"""
import os, sys, json, argparse, random, itertools, traceback

# ── GENOME v3: EVERY persistable knob ────────────────────────────────────────
# Per-point XYZ deltas for every suspension hardpoint on both axles, plus
# structured group moves (track, wheelbase), ARB tube + pickup + lever, setup
# (springs/alignment), and vehicle-level (final drive, CG height).  Diff ramp
# settings are EXCLUDED: they do not round-trip through the .vahan schema, so
# a candidate could not reproduce its own numbers on load.
_PT_BOUNDS = {   # per-axis delta bounds (mm) by hardpoint
    'uca_front': 12., 'uca_rear': 12., 'uca_outer': 8.,
    'lca_front': 12., 'lca_rear': 12., 'lca_outer': 8.,
    'tie_rod_inner': 8., 'tie_rod_outer': 8.,
    'pushrod_inner': 8., 'pushrod_outer': 8.,
    'rocker_pivot': 6., 'rocker_spring_pt': 6., 'spring_chassis_pt': 10.,
}

def _build_genes():
    g = [
        # setup + vehicle
        ('spring_F_lbfin',   150.0, 325.0, 'step25'),
        ('spring_R_lbfin',   150.0, 325.0, 'step25'),
        ('static_camber_F',   -3.0,   0.0, 'float'),
        ('static_camber_R',   -3.0,   0.0, 'float'),
        ('static_toe_F',      -0.3,   0.3, 'float'),
        ('static_toe_R',      -0.3,   0.3, 'float'),
        ('final_drive_ratio', 10.0,  14.0, 'float'),
        ('cg_z_mm',          255.0, 285.0, 'float'),
        # ARB: lever length, tube OD + wall, drop-top pickup height (ARB-MR knob)
        ('lever_scale_F',      0.70,  1.60, 'float'),
        ('lever_scale_R',      0.60,  1.60, 'float'),
        ('arb_od_F_mm',       10.0,  17.0, 'float'),
        ('arb_od_R_mm',       10.0,  17.0, 'float'),
        ('arb_wall_F_mm',      1.2,   2.4, 'float'),
        ('arb_wall_R_mm',      1.2,   2.4, 'float'),
        ('arb_droptop_dz_F',  -8.0,   8.0, 'float'),
        ('arb_droptop_dz_R',  -8.0,   8.0, 'float'),
        # structured geometry
        ('track_dx_F_mm',    -40.0,  40.0, 'float'),   # outboard group lateral shift (half-track)
        ('track_dx_R_mm',    -40.0,  40.0, 'float'),
        ('wheelbase_dy_mm',  -25.0,  25.0, 'float'),   # whole rear corner group shift
    ]
    # every hardpoint coordinate, both axles
    for ax in ('F', 'R'):
        for pt, b in _PT_BOUNDS.items():
            for i, c in enumerate('xyz'):
                g.append(('hp_%s_%s_%s' % (ax, pt, c), -b, b, 'float'))
    return g

GENES = _build_genes()

ISLANDS = {   # objective personalities ("buildings"); weights over normalized metrics
    # 'aero' score = balance CONSISTENCY under the prospective 350 N package
    # at weight-split CoP: a car whose rear-first character survives aero.
    'grip':      dict(gmax=3.0, margin=0.5, feel=0.5, endur=0.5, aero=0.5),
    'rotation':  dict(gmax=1.0, margin=3.0, feel=0.5, endur=0.5, aero=1.0),
    'endurance': dict(gmax=1.0, margin=0.8, feel=0.5, endur=3.0, aero=1.0),
    'balanced':  dict(gmax=1.5, margin=1.5, feel=1.5, endur=1.0, aero=1.5),
}

def _quant(name, val):
    lo, hi, kind = next((l, h, k) for n, l, h, k in GENES if n == name)
    val = min(max(val, lo), hi)
    if kind == 'step25':
        val = round(val / 25.0) * 25.0
    return float(val)

def random_genome(rng):
    return {n: _quant(n, rng.uniform(lo, hi)) for n, lo, hi, k in GENES}

def lhs_genomes(n, rng):
    """Latin-hypercube founders — even coverage of the knowns box."""
    out = [dict() for _ in range(n)]
    for gname, lo, hi, kind in GENES:
        cells = list(range(n)); rng.shuffle(cells)
        for i, c in enumerate(cells):
            out[i][gname] = _quant(gname, lo + (hi - lo) * (c + rng.random()) / n)
    return out

def mutate(g, rng, scale=0.25, k=None):
    """SPARSE mutation: with ~100 genes, perturbing half of them per child is
    a random walk; touching a Poisson-few keeps offspring near the parent."""
    out = dict(g)
    names = [n for n, *_ in GENES]
    if k is None:
        k = max(1, min(len(names), int(rng.expovariate(1 / 5.0)) + 1))
    for gname in rng.sample(names, k):
        lo, hi, kind = next((l, h, kk) for n, l, h, kk in GENES if n == gname)
        out[gname] = _quant(gname, out[gname] + rng.gauss(0, scale * (hi - lo)))
    return out

def blend(a, b, rng):
    return {n: _quant(n, a[n] if rng.random() < 0.5 else b[n]) for n, *_ in GENES}

# ─────────────────────────── worker-side evaluation ──────────────────────────
_W = {}   # per-process cache: {'win': MainWindow, 'base': snapshots}

def _worker_init(base_config):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from PyQt6.QtWidgets import QApplication
    _W['app'] = QApplication.instance() or QApplication([])
    from gui.main_window import MainWindow
    w = MainWindow(); w._load_project_from_path(base_config)
    import copy
    _W['w'] = w
    _W['np'] = np
    _W['base'] = dict(
        front_hp={k: np.asarray(v).copy() for k, v in w._front_hp.items()},
        rear_hp={k: np.asarray(v).copy() for k, v in w._rear_hp.items()},
        front_arb=copy.deepcopy(w._front_arb), rear_arb=copy.deepcopy(w._rear_arb),
        alignment=dict(w._alignment), dyn_state=w._dynamics_panel.get_state(),
        car=dict(w._car))

def _apply(genome):
    np = _W['np']; w = _W['w']; base = _W['base']
    import copy
    for k, v in base['front_hp'].items(): w._front_hp[k] = v.copy()
    for k, v in base['rear_hp'].items():  w._rear_hp[k] = v.copy()
    w._front_arb = copy.deepcopy(base['front_arb']); w._rear_arb = copy.deepcopy(base['rear_arb'])
    w._alignment.update(base['alignment'])
    if 'car' not in base: base['car'] = dict(w._car)
    w._car.update(base['car'])

    OUTBOARD = ('uca_outer', 'lca_outer', 'tie_rod_outer', 'pushrod_outer', 'wheel_center')
    # 1) structured group moves: track (outboard group X) + wheelbase (rear group Y)
    for axd, gk, tk in ((w._front_hp, 'track_dx_F_mm', 'track_f_mm'),
                        (w._rear_hp, 'track_dx_R_mm', 'track_r_mm')):
        dx = genome[gk] / 1000.0
        for pt in OUTBOARD:
            if pt in axd: axd[pt] = axd[pt] + np.array([dx, 0, 0])
        w._car[tk] = base['car'][tk] + 2 * genome[gk]
    dy = genome['wheelbase_dy_mm'] / 1000.0
    for pt in list(w._rear_hp.keys()):
        w._rear_hp[pt] = w._rear_hp[pt] + np.array([0, dy, 0])
    for arbk in ('arb_pivot', 'arb_arm_end', 'arb_drop_top'):
        w._rear_arb[arbk] = np.asarray(w._rear_arb[arbk]) + np.array([0, dy, 0])
    w._car['wheelbase_mm'] = base['car']['wheelbase_mm'] + genome['wheelbase_dy_mm']
    w._car['axle_spacing_mm'] = w._car['wheelbase_mm']

    # 2) every-hardpoint per-axis deltas; rocker_pivot delta also translates
    #    rocker_axis_pt (rigid rocker translation, axis direction preserved)
    for ax, axd in (('F', w._front_hp), ('R', w._rear_hp)):
        for pt in _PT_BOUNDS:
            d = np.array([genome['hp_%s_%s_%s' % (ax, pt, c)] for c in 'xyz']) / 1000.0
            if pt in axd:
                axd[pt] = axd[pt] + d
                if pt == 'rocker_pivot' and 'rocker_axis_pt' in axd:
                    axd['rocker_axis_pt'] = axd['rocker_axis_pt'] + d
    try:
        w._enforce_actuation_coplanar()
    except Exception:
        pass

    # 3) ARB: lever length, drop-top pickup height (ARB-MR knob)
    for arb, ls, dz in ((w._front_arb, genome['lever_scale_F'], genome['arb_droptop_dz_F']),
                        (w._rear_arb, genome['lever_scale_R'], genome['arb_droptop_dz_R'])):
        piv = np.asarray(arb['arb_pivot']); tip = np.asarray(arb['arb_arm_end'])
        arb['arb_arm_end'] = piv + ls * (tip - piv)
        arb['arb_drop_top'] = np.asarray(arb['arb_drop_top']) + np.array([0, 0, dz / 1000.0])

    # 4) setup + vehicle knobs through their real storage
    st = dict(base['dyn_state'])
    st['spring_front_lbfin'] = genome['spring_F_lbfin']
    st['spring_rear_lbfin'] = genome['spring_R_lbfin']
    st['arb_OD_f_mm'] = genome['arb_od_F_mm']
    st['arb_ID_f_mm'] = max(genome['arb_od_F_mm'] - 2 * genome['arb_wall_F_mm'], 0.0)
    st['arb_OD_r_mm'] = genome['arb_od_R_mm']
    st['arb_ID_r_mm'] = max(genome['arb_od_R_mm'] - 2 * genome['arb_wall_R_mm'], 0.0)
    st['final_drive_ratio'] = genome['final_drive_ratio']
    w._dynamics_panel.set_state(st)
    w._alignment['front_camber_deg'] = genome['static_camber_F']
    w._alignment['rear_camber_deg'] = genome['static_camber_R']
    w._alignment['front_toe_deg'] = genome['static_toe_F']
    w._alignment['rear_toe_deg'] = genome['static_toe_R']
    w._car['cg_z_mm'] = genome['cg_z_mm']
    w._rebuild_solvers(0.)

def _axle_util(ss, r):
    """SINGLE MODEL: delegate to the canonical criterion in the solver.
    (This used to be a local reimplementation — the exact duplicate-model
    creep that let the aero solver ship a different limit metric.)"""
    return ss.axle_utilization(r)

def evaluate(args):
    """Full Thorough/Faults/Interference pipeline on one genome."""
    genome, cand_id, out_dir = args
    np = _W['np']; w = _W['w']
    M = {'id': cand_id, 'genome': genome, 'ok': False}
    try:
        _apply(genome)
        ss = w._build_dynamics_solver(); v = ss._veh
        # kinematics (heave sweep both axles)
        t = np.linspace(-0.028, 0.028, 29)
        kin = {}
        _flip = np.array([-1., 1., 1.])
        for lbl in ('FL', 'RL'):
            src = w._front_arb if lbl[0] == 'F' else w._rear_arb
            kin[lbl] = w._do_sweep(w._solvers[lbl], t, 'left', arb_hp=src,
                                   is_front=lbl[0] == 'F', label=lbl)
        def mx(lbl, key):
            a = np.asarray(kin[lbl][key], float); a = a[np.isfinite(a)]
            return float(np.max(np.abs(a))) if len(a) else float('nan')
        def at0(lbl, key):
            a = np.asarray(kin[lbl][key], float)
            i = len(a) // 2
            return float(a[i]) if np.isfinite(a[i]) else float(np.nanmedian(a))
        M['bump_steer_F'] = mx('FL', 'toe'); M['bump_steer_R'] = mx('RL', 'toe')
        M['scrub_F'] = at0('FL', 'scrub'); M['caster_F'] = at0('FL', 'caster')
        M['rc_F'] = at0('FL', 'rc_height'); M['rc_R'] = at0('RL', 'rc_height')
        cg = np.asarray(kin['FL']['camber'], float)
        M['camber_gain_F'] = float(cg[-1] - cg[len(cg)//2]) if np.isfinite(cg[-1]) else float('nan')
        M['MR_F'] = at0('FL', 'motion_ratio'); M['MR_R'] = at0('RL', 'motion_ratio')
        M['anti_squat'] = at0('RL', 'anti_squat')
        # dynamics: axle-aggregate limit + margin + roll + LLTD
        r10 = ss.solve(1.0)
        fa = r10.elastic_lt_front_N + r10.geometric_lt_front_N + r10.unsprung_lt_front_N
        re_ = r10.elastic_lt_rear_N + r10.geometric_lt_rear_N + r10.unsprung_lt_rear_N
        M['LLTD_F'] = 100 * fa / max(fa + re_, 1.)
        M['roll_deg_g'] = r10.roll_angle_deg
        lo_, hi_ = 0.5, 3.0
        for _ in range(14):
            mid = (lo_ + hi_) / 2
            try:
                u = _axle_util(ss, ss.solve(mid))
                lo_, hi_ = (mid, hi_) if max(u.values()) < 1.0 else (lo_, mid)
            except Exception:
                hi_ = mid
        M['gmax'] = lo_
        r15 = ss.solve(min(1.5, lo_ * 0.9)); u15 = _axle_util(ss, r15)
        M['margin'] = u15['R'] - u15['F']
        M['first_axle'] = 'REAR' if M['margin'] > 0 else 'FRONT'
        M['minFz_15'] = float(min(r15.Fz.values()))
        # AERO CASE: prospective 350 N package at weight-split CoP (honest band
        # centre — the neutral point is not determinable from the tire data).
        wfrac = v.front_weight_fraction
        AF = {'FL': 350*wfrac/2, 'FR': 350*wfrac/2, 'RL': 350*(1-wfrac)/2, 'RR': 350*(1-wfrac)/2}
        la, ha = 0.5, 3.4
        for _ in range(12):
            mid = (la + ha) / 2
            try:
                u = _axle_util(ss, ss.solve(mid, aero_Fz=AF))
                la, ha = (mid, ha) if max(u.values()) < 1.0 else (la, mid)
            except Exception:
                ha = mid
        M['gmax_aero'] = la
        ua = _axle_util(ss, ss.solve(min(1.5, la * 0.9), aero_Fz=AF))
        M['margin_aero'] = ua['R'] - ua['F']
        M['first_axle_aero'] = 'REAR' if M['margin_aero'] > 0 else 'FRONT'
        # frequencies + sag
        mF = v.sprung_mass_kg * v.front_weight_fraction / 2
        mR = v.sprung_mass_kg * v.rear_weight_fraction / 2
        M['freq_F'] = float((1/(2*np.pi))*np.sqrt(v.ride_rate_front_Npm/mF))
        M['freq_R'] = float((1/(2*np.pi))*np.sqrt(v.ride_rate_rear_Npm/mR))
        M['dsag_F'] = float(mF*9.80665/v.ride_rate_front_Npm*1000*v.motion_ratio_front)
        M['dsag_R'] = float(mR*9.80665/v.ride_rate_rear_Npm*1000*v.motion_ratio_rear)
        # interference (quick: RL + FL at 3 travels, key pairs)
        M['clearance_mm'] = _clearance(w, np)
        M['ok'] = all(np.isfinite([M['gmax'], M['margin'], M['LLTD_F'], M['bump_steer_F']]))
        # scores (normalized 0..1-ish) + penalties
        pen = 0.
        if M['clearance_mm'] < 10: pen += 5.
        if not (0.28 <= M['dsag_F']/55 <= 0.50 and 0.28 <= M['dsag_R']/55 <= 0.50): pen += 1.0
        if M['bump_steer_F'] > 0.30 or M['bump_steer_R'] > 0.25: pen += 1.5
        if M['roll_deg_g'] > 1.3: pen += 1.0
        if abs(M['scrub_F']) > 30: pen += 0.8
        # rules guards (genome can now move track/wheelbase/caster)
        wb = w._car.get('wheelbase_mm', 1537.)
        tf = w._car.get('track_f_mm', 1117.6); tr_ = w._car.get('track_r_mm', 1117.6)
        M['wheelbase_mm'] = wb; M['track_f_mm'] = tf; M['track_r_mm'] = tr_
        if wb < 1525.: pen += 5.
        if min(tf, tr_) / max(tf, tr_) < 0.75: pen += 5.
        if M['caster_F'] < 0.5: pen += 1.5     # keep self-centering positive
        ssf = min(tf, tr_) / (2. * w._car.get('cg_z_mm', 270.))
        M['SSF'] = ssf
        if ssf < 1.8: pen += 3.               # rollover margin guard (tilt 1.7g + margin)
        feel = 1.0 - min(M['bump_steer_F']/0.3, 1) * 0.5 - min(abs(M['scrub_F'])/30, 1) * 0.5
        endur = 1.0 - min(max(M['rc_F'], M['rc_R'], 0)/80, 1)*0.5 - min(M['roll_deg_g']/1.5, 1)*0.5
        # aero score: character survives the package (rear-first held, margin
        # shift small); a flip to FRONT-first under aero is aero-induced push.
        if M['first_axle_aero'] == 'FRONT' and M['first_axle'] == 'REAR':
            pen += 1.5
        aero_sc = (1.0 - min(abs(M['margin_aero'] - M['margin'])/0.05, 1) * 0.5
                   + (0.5 if M['first_axle_aero'] == 'REAR' else 0.0))
        M['scores'] = dict(gmax=(M['gmax']-1.6)/0.6, margin=min(max(M['margin'],0)/0.12,1.2),
                           feel=feel, endur=endur, aero=aero_sc)
        M['penalty'] = pen
        # write artifacts
        cdir = os.path.join(out_dir, cand_id); os.makedirs(cdir, exist_ok=True)
        _save_config(w, os.path.join(cdir, 'config.vahan'), np)
        _card(kin, ss, M, os.path.join(cdir, 'card.png'), np)
        json.dump(M, open(os.path.join(cdir, 'metrics.json'), 'w'), indent=1, default=str)
    except Exception as e:
        M['error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    return M

def _clearance(w, np):
    def segdist(a1, a2, b1, b2):
        A = np.asarray(a2)-np.asarray(a1); B = np.asarray(b2)-np.asarray(b1)
        return min(np.linalg.norm(np.asarray(a1)+ta*A-(np.asarray(b1)+tb*B))
                   for ta in np.linspace(0, 1, 9) for tb in np.linspace(0, 1, 9))
    worst = 1e9
    for lbl in ('FL', 'RL'):
        for tv in (-0.025, 0.0, 0.025):
            try: st = w._solvers[lbl].solve(tv)
            except Exception: continue
            segs = []
            for a, b in (('uca_front','uca_outer'),('uca_rear','uca_outer'),
                         ('lca_front','lca_outer'),('lca_rear','lca_outer'),
                         ('pushrod_inner','pushrod_outer'),('tie_rod_inner','tie_rod_outer')):
                pa, pb = getattr(st, a, None), getattr(st, b, None)
                if pa is not None and pb is not None: segs.append((pa, pb))
            try:
                dt = w._arb_drop_top_world(lbl, st)
                arb = w._front_arb if lbl[0] == 'F' else w._rear_arb
                if dt is not None: segs.append((np.asarray(arb['arb_arm_end']), dt))
            except Exception: pass
            for i in range(len(segs)):
                for j in range(i+1, len(segs)):
                    p1, p2 = segs[i]; q1, q2 = segs[j]
                    pts = [tuple(np.round(np.asarray(x), 4)) for x in (p1, p2, q1, q2)]
                    if len(set(pts)) < 4: continue
                    worst = min(worst, segdist(p1, p2, q1, q2)*1000)
    return float(worst)

def _save_config(w, path, np):
    mp = w._motion_panel
    def C(o):
        if isinstance(o, dict): return {k: C(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [C(x) for x in o]
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    data = {'version': 3,
            'front_hp': {k: np.asarray(v).tolist() for k, v in w._front_hp.items()},
            'rear_hp': {k: np.asarray(v).tolist() for k, v in w._rear_hp.items()},
            'front_arb': {k: np.asarray(v).tolist() for k, v in w._front_arb.items()},
            'rear_arb': {k: np.asarray(v).tolist() for k, v in w._rear_arb.items()},
            'front_heave': {k: np.asarray(v).tolist() for k, v in getattr(w, '_front_heave', {}).items()},
            'rear_heave': {k: np.asarray(v).tolist() for k, v in getattr(w, '_rear_heave', {}).items()},
            'front_decoupled': {k: np.asarray(v).tolist() for k, v in getattr(w, '_front_decoupled', {}).items()},
            'rear_decoupled': {k: np.asarray(v).tolist() for k, v in getattr(w, '_rear_decoupled', {}).items()},
            'car': w._car.copy(), 'steer': w._steer.copy(), 'alignment': w._alignment.copy(),
            'topology': w._topology.to_dict(),
            'motion': {'type': mp.motion, 'min': mp.min_val, 'max': mp.max_val,
                       'stroke_mm': mp.stroke_mm, 'preload_front_mm': mp.preload_front_mm,
                       'preload_rear_mm': mp.preload_rear_mm, 'fully_extended_mm': mp.fully_extended_mm},
            'panels': {'dynamics': w._dynamics_panel.get_state(), 'skidpad': w._skidpad_panel.get_state(),
                       'loads': w._loads_panel.get_state(), 'aero': w._aero_panel.get_state(),
                       'brake_calc': w._brake_calc_panel.get_state()}}
    json.dump(C(data), open(path, 'w'), indent=1)

def _card(kin, ss, M, path, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE='#0057B8'; RED='#D62828'; OCHRE='#E8A000'
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.6), dpi=90)
    fig.patch.set_facecolor('white')
    t = np.linspace(-28, 28, len(np.asarray(kin['FL']['toe'])))
    # 1 balance: axle util vs g
    gs = np.linspace(0.4, min(M['gmax'], 2.4), 8)
    uf = []; ur = []
    for g in gs:
        try:
            u = _axle_util(ss, ss.solve(float(g)))
            uf.append(u['F']); ur.append(u['R'])
        except Exception:
            uf.append(np.nan); ur.append(np.nan)
    axs[0,0].plot(gs, uf, color=BLUE, label='front'); axs[0,0].plot(gs, ur, color=RED, label='rear')
    axs[0,0].axhline(1, color=OCHRE, lw=0.8)
    axs[0,0].set_title('axle util vs g  (gmax %.2f, %s-first)' % (M['gmax'], M['first_axle']), fontsize=7)
    axs[0,0].legend(fontsize=6)
    # 2 camber+toe vs travel
    axs[0,1].plot(t, kin['FL']['camber'], color=BLUE, label='camber F')
    axs[0,1].plot(t, kin['FL']['toe'], color=RED, label='toe F')
    axs[0,1].plot(t, kin['RL']['toe'], color=OCHRE, label='toe R')
    axs[0,1].set_title('camber/toe vs travel (mm)', fontsize=7); axs[0,1].legend(fontsize=6)
    # 3 RC height vs travel
    axs[1,0].plot(t, kin['FL']['rc_height'], color=BLUE, label='RC F')
    axs[1,0].plot(t, kin['RL']['rc_height'], color=RED, label='RC R')
    axs[1,0].set_title('RC height (mm) vs travel', fontsize=7); axs[1,0].legend(fontsize=6)
    # 4 numbers panel
    axs[1,1].axis('off')
    txt = ('LLTD %.1f%%F  roll %.2f deg/g\nmargin %+0.3f  minFz %.0f N\n'
           'bumpsteer %.2f/%.2f  scrub %.1f\nfreq %.2f/%.2f Hz  sag %.1f/%.1f\n'
           'springs %.0f/%.0f lbf/in\nclearance %.1f mm   pen %.1f') % (
        M['LLTD_F'], M['roll_deg_g'], M['margin'], M['minFz_15'],
        M['bump_steer_F'], M['bump_steer_R'], M['scrub_F'],
        M['freq_F'], M['freq_R'], M['dsag_F'], M['dsag_R'],
        M['genome']['spring_F_lbfin'], M['genome']['spring_R_lbfin'],
        M['clearance_mm'], M['penalty'])
    axs[1,1].text(0.02, 0.95, txt, fontsize=8, va='top', family='monospace')
    for a in axs.ravel():
        a.tick_params(labelsize=6)
    fig.suptitle(M['id'], fontsize=8)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)

# ───────────────────────────── evolutionary city ─────────────────────────────
def fitness(M, weights):
    if not M.get('ok'): return -99.
    s = M['scores']
    return sum(w * s.get(k, 0.5) for k, w in weights.items()) - M['penalty']

def dominates(a, b, eps=0.03):
    """Epsilon-dominance: scores are compared on an eps grid so the archive
    stays a SHORT list of meaningfully-different designs (plain multi-objective
    dominance kept ~90% of candidates — useless as a shortlist).  .get(...,0.5)
    keeps pre-aero-era archives loadable."""
    KEYS = ('gmax', 'margin', 'feel', 'endur', 'aero')
    q = lambda s: tuple(round(s.get(k, 0.5) / eps) for k in KEYS)
    ka, kb = q(a['scores']), q(b['scores'])
    return all(x >= y for x, y in zip(ka, kb)) and any(x > y for x, y in zip(ka, kb))

def run_city(out_dir, base_config, seed_configs, islands, pop, gens, workers, seed=7):
    import multiprocessing as mp_
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    isl_names = list(ISLANDS.keys())[:islands]
    # founders (LHS from scratch) + lineage (mutations of seeds -> baseline genome = all-neutral)
    neutral = {n: 0.0 for n, *_ in GENES}
    neutral.update(spring_F_lbfin=200., spring_R_lbfin=250., static_camber_F=-0.8,
                   static_camber_R=-0.7, final_drive_ratio=12.59, cg_z_mm=270.,
                   lever_scale_F=1.0, lever_scale_R=1.0,
                   arb_od_F_mm=12.7, arb_od_R_mm=12.7, arb_wall_F_mm=1.525, arb_wall_R_mm=1.525)
    def temper(g):
        """Founders: full-range on setup genes, half-range on geometry genes —
        a uniform 100-D corner sample is almost always an unbuildable car."""
        out = dict(g)
        for n, lo, hi, k in GENES:
            if n.startswith(('hp_', 'track_', 'wheelbase_')):
                out[n] = _quant(n, neutral[n] + 0.5 * (g[n] - neutral[n]))
        return out
    pops = {}
    for i, nm in enumerate(isl_names):
        founders = [temper(g) for g in lhs_genomes(pop // 2, rng)]  # from-scratch set
        lineage = [mutate(neutral, rng, 0.12) for _ in range(pop - len(founders))]  # iterating set
        pops[nm] = founders + lineage
    pool = mp_.Pool(workers, initializer=_worker_init, initargs=(base_config,))
    archive = []; hist = []
    cand_n = itertools.count()
    for gen in range(gens):
        jobs = []
        for nm in isl_names:
            for g in pops[nm]:
                cid = 'g%02d_%s_c%04d' % (gen, nm, next(cand_n))
                jobs.append((g, cid, out_dir))
        results = pool.map(evaluate, jobs)
        ok = [m for m in results if m.get('ok')]
        print('[gen %d] evaluated %d, ok %d' % (gen, len(results), len(ok)))
        # archive: global non-dominated
        for m in ok:
            if m['penalty'] >= 5: continue          # interference reject
            if not any(dominates(a, m) for a in archive):
                archive = [a for a in archive if not dominates(m, a)] + [m]
        hist.extend([{k: m.get(k) for k in ('id','ok','penalty','gmax','margin','LLTD_F','error')} for m in results])
        # next generation per island: tournament + blend + mutate (+ migration)
        by_isl = {nm: [m for m in ok if ('_%s_' % nm) in m['id']] for nm in isl_names}
        for nm in isl_names:
            ranked = sorted(by_isl[nm], key=lambda m: fitness(m, ISLANDS[nm]), reverse=True)
            elite = [m['genome'] for m in ranked[:max(2, pop//5)]]
            migrant = []
            others = [m for x in isl_names if x != nm for m in by_isl[x]]
            if others:
                migrant = [max(others, key=lambda m: fitness(m, ISLANDS[nm]))['genome']]
            nxt = list(elite) + migrant
            while len(nxt) < pop:
                if len(ranked) >= 2:
                    a, b = rng.sample(ranked[:max(4, len(ranked)//2)], 2)
                    nxt.append(mutate(blend(a['genome'], b['genome'], rng), rng, 0.15))
                else:
                    nxt.append(random_genome(rng))
            pops[nm] = nxt[:pop]
    pool.close(); pool.join()
    json.dump({'archive': [m['id'] for m in archive],
               'archive_metrics': archive, 'history': hist,
               'islands': isl_names, 'pop': pop, 'gens': gens},
              open(os.path.join(out_dir, 'city.json'), 'w'), indent=1, default=str)
    print('CITY DONE: %d Pareto survivors -> %s' % (len(archive), out_dir))
    return archive


def _rescore_one(args):
    """Re-derive the DYNAMICS metrics of one existing candidate from its own
    config file (tire-model/criterion fixes change limit-zone numbers, e.g.
    the 2026-07-12 marginal-slope extrapolation).  Kinematic metrics are
    unaffected by tire changes but the card is redrawn so its utilization
    curve and number strip match."""
    cand_dir = args
    np = _W['np']; w = _W['w']
    mj = os.path.join(cand_dir, 'metrics.json')
    try:
        M = json.load(open(mj))
        w._load_project_from_path(os.path.join(cand_dir, 'config.vahan'))
        w._rebuild_solvers(0.)
        ss = w._build_dynamics_solver()
        t = np.linspace(-0.028, 0.028, 29)
        kin = {}
        for lbl in ('FL', 'RL'):
            src = w._front_arb if lbl[0] == 'F' else w._rear_arb
            kin[lbl] = w._do_sweep(w._solvers[lbl], t, 'left', arb_hp=src,
                                   is_front=lbl[0] == 'F', label=lbl)
        r10 = ss.solve(1.0)
        fa = r10.elastic_lt_front_N + r10.geometric_lt_front_N + r10.unsprung_lt_front_N
        re_ = r10.elastic_lt_rear_N + r10.geometric_lt_rear_N + r10.unsprung_lt_rear_N
        M['LLTD_F'] = 100 * fa / max(fa + re_, 1.)
        M['roll_deg_g'] = r10.roll_angle_deg
        lo_, hi_ = 0.5, 3.2
        for _ in range(14):
            mid = (lo_ + hi_) / 2
            try:
                u = ss.axle_utilization(ss.solve(mid))
                lo_, hi_ = (mid, hi_) if max(u.values()) < 1.0 else (lo_, mid)
            except Exception:
                hi_ = mid
        M['gmax'] = lo_
        r15 = ss.solve(min(1.5, lo_ * 0.9)); u15 = ss.axle_utilization(r15)
        M['margin'] = u15['R'] - u15['F']
        M['first_axle'] = 'REAR' if M['margin'] > 0 else 'FRONT'
        M['minFz_15'] = float(min(r15.Fz.values()))
        M['scores'] = dict(gmax=(M['gmax'] - 1.6) / 0.6,
                           margin=min(max(M['margin'], 0) / 0.12, 1.2),
                           feel=M.get('scores', {}).get('feel', 0.5),
                           endur=M.get('scores', {}).get('endur', 0.5))
        M['rescored_v3_tire'] = True
        _card(kin, ss, M, os.path.join(cand_dir, 'card.png'), np)
        json.dump(M, open(mj, 'w'), indent=1, default=str)
        return M
    except Exception as e:
        return {'id': os.path.basename(cand_dir), 'ok': False,
                'error': 'rescore: %s' % str(e)[:150]}


def rescore_run(run_dir, workers=5, base_config='configs/2027_v11_(raised_arb_free_springs).vahan'):
    import multiprocessing as mp_
    import glob as glob_
    cands = sorted(os.path.dirname(p) for p in
                   glob_.glob(os.path.join(run_dir, '*', 'metrics.json')))
    if not cands:
        print('no candidates in', run_dir); return
    pool = mp_.Pool(workers, initializer=_worker_init, initargs=(base_config,))
    res = pool.map(_rescore_one, cands)
    pool.close(); pool.join()
    ok = [m for m in res if m.get('ok')]
    # rebuild the archive with the fresh scores
    archive = []
    for m in ok:
        if m.get('penalty', 0) >= 5: continue
        if not any(dominates(a, m) for a in archive):
            archive = [a for a in archive if not dominates(m, a)] + [m]
    cj = os.path.join(run_dir, 'city.json')
    city = json.load(open(cj)) if os.path.exists(cj) else {}
    city['archive'] = [m['id'] for m in archive]
    city['archive_metrics'] = archive
    city['rescored_v3_tire'] = True
    json.dump(city, open(cj, 'w'), indent=1, default=str)
    print('RESCORED %s: %d/%d ok, archive %d' % (run_dir, len(ok), len(res), len(archive)))


def lap_filter(out_dir, top_n=8):
    """Final-stage product check: run the external lap sim on the best archive
    members (per-island fitness leaders + overall) and write lap_s into their
    metrics + city.json.  ~10-15 s per candidate, so archive-only by design."""
    import subprocess
    cj = os.path.join(out_dir, 'city.json')
    city = json.load(open(cj))
    arch = city.get('archive_metrics', [])
    picks = {}
    for nm, wts in ISLANDS.items():
        ok = [m for m in arch if m.get('ok')]
        if ok:
            best = max(ok, key=lambda m: fitness(m, wts))
            picks[best['id']] = best
    for m in sorted([m for m in arch if m.get('ok')],
                    key=lambda m: m.get('gmax', 0), reverse=True)[:top_n]:
        picks[m['id']] = m
    print('lap filter on %d candidates' % len(picks))
    for cid, m in picks.items():
        cfg = os.path.join(out_dir, cid, 'config.vahan')
        if not os.path.exists(cfg): continue
        try:
            r = subprocess.run([sys.executable, 'DESIGN_2027/scripts/run_roselap.py', cfg],
                               capture_output=True, text=True, timeout=300,
                               env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
            out = (r.stdout or '') + (r.stderr or '')
            lap = None
            for tok in out.replace('|', ' ').split():
                pass
            import re
            mm = re.search(r'LAP\s+([0-9.]+)\s*s', out)
            if mm: lap = float(mm.group(1))
            m['lap_s'] = lap
            mj = os.path.join(out_dir, cid, 'metrics.json')
            mfull = json.load(open(mj)); mfull['lap_s'] = lap
            json.dump(mfull, open(mj, 'w'), indent=1, default=str)
            print('  %s -> lap %s s' % (cid, lap))
        except Exception as e:
            print('  %s lap FAIL %s' % (cid, str(e)[:80]))
    json.dump(city, open(cj, 'w'), indent=1, default=str)


def full_report(cand_dir, base_config=None):
    """Binder-style figure set for ONE shortlisted candidate (on demand from
    the GUI).  Sheets: kinematics, dynamics, MODEL VIEWS, G-G, tire util,
    aero DF-required, transient step/ramp steer, MMD."""
    _worker_init(os.path.join(cand_dir, 'config.vahan'))
    np = _W['np']; w = _W['w']
    import inspect
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE='#0057B8'; RED='#D62828'; OCHRE='#E8A000'; BLACK='#222222'
    C4={'FL':BLUE,'FR':RED,'RL':OCHRE,'RR':BLACK}
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 8, 'axes.grid': True,
                         'grid.alpha': 0.3, 'figure.autolayout': True})
    def lightify(fig):
        fig.patch.set_facecolor('white')
        for a in fig.get_axes():
            a.set_facecolor('white')
            for sp in a.spines.values(): sp.set_color('#888888')
            a.tick_params(colors='#333333')
            a.xaxis.label.set_color('#222222'); a.yaxis.label.set_color('#222222')
            a.title.set_color('#222222')
            leg = a.get_legend()
            if leg:
                leg.get_frame().set_facecolor('#f4f4f4')
                for tx in leg.get_texts(): tx.set_color('#222222')
        return fig
    ss = w._build_dynamics_solver(); v = ss._veh
    t = np.linspace(-0.028, 0.028, 41)
    kin = {}
    for lbl in ('FL', 'RL'):
        src = w._front_arb if lbl[0] == 'F' else w._rear_arb
        kin[lbl] = w._do_sweep(w._solvers[lbl], t, 'left', arb_hp=src,
                               is_front=lbl[0] == 'F', label=lbl)
    x = t * 1000

    def fig_kin():
        fig, axs = plt.subplots(2, 3, figsize=(11, 6))
        panels = [('toe', 'bump steer (deg)'), ('camber', 'camber (deg)'),
                  ('caster', 'caster (deg)'), ('scrub', 'scrub (mm)'),
                  ('motion_ratio', 'motion ratio'), ('rc_height', 'RC height (mm)')]
        for ax, (k, ttl) in zip(axs.ravel(), panels):
            for lbl, c in (('FL', BLUE), ('RL', RED)):
                ax.plot(x, kin[lbl][k], color=c, label=lbl)
            ax.set_title(ttl, fontsize=8); ax.legend(fontsize=6); ax.tick_params(labelsize=6)
        fig.savefig(os.path.join(cand_dir, 'report_kinematics.png')); plt.close(fig)

    def fig_dyn():
        sw = ss.sweep_lateral_g((0.0, 1.9), 21)
        fig, axs = plt.subplots(2, 2, figsize=(9, 6))
        g = sw['lateral_g']
        axs[0,0].plot(g, sw['roll_angle_deg'], color=BLUE); axs[0,0].set_title('roll (deg) vs g', fontsize=8)
        axs[0,1].plot(g, sw['understeer_gradient_deg'], color=RED); axs[0,1].axhline(0, ls='--', lw=0.7, color=BLACK)
        axs[0,1].set_title('understeer gradient (deg/g)', fontsize=8)
        for cn in ('FL','FR','RL','RR'):
            axs[1,0].plot(g, sw['Fz_%s' % cn], color=C4[cn], label=cn)
        axs[1,0].set_title('per-corner Fz (N)', fontsize=8); axs[1,0].legend(fontsize=6)
        ef = np.asarray(sw['elastic_lt_front_N']); gf = np.asarray(sw['geometric_lt_front_N'])
        axs[1,1].stackplot(g, ef, gf, colors=[BLUE, RED], labels=['elastic', 'geometric'])
        axs[1,1].set_title('front LT paths (N)', fontsize=8); axs[1,1].legend(fontsize=6)
        for a in axs.ravel(): a.tick_params(labelsize=6)
        fig.savefig(os.path.join(cand_dir, 'report_dynamics.png')); plt.close(fig)

    def _segments(lbl):
        """All drawable segments for one corner incl. rocker, spring, ARB.
        The corner SOLVERS already return correct per-side coordinates; only
        the chassis-side DICT points (rocker/spring/ARB, stored left-side)
        need the X flip for right corners."""
        st = w._solvers[lbl].solve(0.)
        hp = w._front_hp if lbl[0] == 'F' else w._rear_hp
        arb = w._front_arb if lbl[0] == 'F' else w._rear_arb
        flip = np.array([-1, 1, 1]) if lbl in ('FR', 'RR') else np.array([1, 1, 1])
        S = []
        def seg(a, b, f=1):
            if a is None or b is None: return
            fa = flip if f else np.array([1, 1, 1])
            S.append((np.asarray(a) * fa, np.asarray(b) * fa))
        for a, b in (('uca_front','uca_outer'),('uca_rear','uca_outer'),
                     ('lca_front','lca_outer'),('lca_rear','lca_outer'),
                     ('uca_outer','lca_outer'),('pushrod_inner','pushrod_outer'),
                     ('tie_rod_inner','tie_rod_outer')):
            seg(getattr(st, a, None), getattr(st, b, None), f=0)   # solver: already sided
        seg(hp.get('rocker_pivot'), hp.get('rocker_spring_pt'))
        seg(hp.get('rocker_spring_pt'), hp.get('spring_chassis_pt'))       # coilover
        try:
            seg(arb.get('arb_pivot'), arb.get('arb_arm_end'))
            seg(arb.get('arb_arm_end'), arb.get('arb_drop_top'))
        except Exception: pass
        return S, np.asarray(st.wheel_center)

    def fig_model():
        fig = plt.figure(figsize=(11, 8))
        ax3 = fig.add_subplot(221, projection='3d')
        axf = fig.add_subplot(222); axs_ = fig.add_subplot(223); axt = fig.add_subplot(224)
        allpts = []
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            try: S, wc = _segments(lbl)
            except Exception: continue
            c = C4[lbl]
            for a, b in S:
                a = a * 1000; b = b * 1000
                allpts += [a, b]
                ax3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=c, lw=1.1)
                axf.plot([a[0], b[0]], [a[2], b[2]], color=c, lw=0.9)
                axs_.plot([a[1], b[1]], [a[2], b[2]], color=c, lw=0.9)
                axt.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=0.9)
            wcm = wc * 1000    # solver wheel_center is already correctly sided
            th = np.linspace(0, 2*np.pi, 36); r = v.tire_radius_m * 1000
            axf.plot(wcm[0]+0*th, wcm[2]+r*np.sin(th), color=c, lw=0.5, alpha=0.5)
            axs_.plot(wcm[1]+r*np.cos(th), wcm[2]+r*np.sin(th), color=c, lw=0.5, alpha=0.5)
        P = np.asarray(allpts)
        if len(P):   # EQUAL aspect so the iso is not morphed
            rng = P.max(0) - P.min(0)
            ax3.set_box_aspect(tuple(np.maximum(rng, 1)))
        ax3.set_title('iso (equal aspect)', fontsize=8); ax3.view_init(20, -55)
        axf.set_title('front view XZ', fontsize=8); axf.set_aspect('equal')
        axs_.set_title('side view YZ', fontsize=8); axs_.set_aspect('equal')
        axt.set_title('top view XY', fontsize=8); axt.set_aspect('equal')
        fig.suptitle('model: wishbones/pushrod/tie/rocker/coilover/ARB (design position)', fontsize=9)
        fig.savefig(os.path.join(cand_dir, 'report_model.png')); plt.close(fig)

    def axle_util(r):
        return ss.axle_utilization(r)   # SINGLE MODEL: canonical criterion

    def fig_gg():
        """Paired envelopes: mechanical AND with the prospective 350 N package
        at weight-split CoP — the aero/non-aero flagship comparison."""
        fig, ax = plt.subplots(figsize=(5.5, 5))
        wfrac = v.front_weight_fraction
        AF = {'FL': 350*wfrac/2, 'FR': 350*wfrac/2,
              'RL': 350*(1-wfrac)/2, 'RR': 350*(1-wfrac)/2}
        for tag, af, c in (('mechanical', None, BLUE), ('with 350 N aero', AF, RED)):
            lons = np.linspace(-1.5, 1.1, 12); lats = []
            for lon in lons:
                lo_, hi_ = 0.2, 3.2
                for _ in range(9):
                    mid = (lo_ + hi_) / 2
                    try:
                        u = axle_util(ss.solve(mid, float(lon), aero_Fz=af))
                        lo_, hi_ = (mid, hi_) if max(u.values()) < 1.0 else (lo_, mid)
                    except Exception: hi_ = mid
                lats.append(lo_)
            lats = np.asarray(lats)
            ax.plot(lats, lons, color=c, label=tag); ax.plot(-lats, lons, color=c)
        ax.axhline(0, color=BLACK, lw=0.5); ax.axvline(0, color=BLACK, lw=0.5)
        ax.set_xlabel('lateral g'); ax.set_ylabel('longitudinal g (+accel)')
        ax.set_title('G-G envelope — mechanical vs 350 N aero (weight-split CoP)\n'
                     'axle-aggregate limit; >1139 N zone = extrapolation band', fontsize=8)
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(cand_dir, 'report_gg.png')); plt.close(fig)

    def fig_tireutil():
        import vahan.analysis_plots as ap
        try:
            f = ap.plot_friction_circle(ss, max_ay_g=1.9)
            lightify(f).savefig(os.path.join(cand_dir, 'report_tireutil.png')); plt.close(f)
        except Exception as e:
            print('tireutil FAIL', e)

    def fig_aero():
        from vahan.dynamics import AeroDownforceSolver
        aero = AeroDownforceSolver(ss)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for tgt, c in ((0.85, BLUE), (0.95, RED)):
            gs = np.linspace(1.5, 2.4, 13); tot = []; fshare = []
            for g in gs:
                try:
                    ar = aero.solve(float(g), target_util=tgt)
                    df = ar.downforce if isinstance(ar.downforce, dict) else {}
                    s = sum(df.values()); tot.append(s)
                    fshare.append(100*(df.get('FL',0)+df.get('FR',0))/s if s > 1 else float('nan'))
                except Exception:
                    tot.append(float('nan')); fshare.append(float('nan'))
            axs[0].plot(gs, tot, color=c, marker='o', ms=3, label='target util %.2f' % tgt)
            axs[1].plot(gs, fshare, color=c, marker='o', ms=3, label='target util %.2f' % tgt)
        axs[0].set_title('downforce REQUIRED vs lateral g (N)', fontsize=8)
        axs[0].set_xlabel('lateral g'); axs[0].legend(fontsize=6)
        axs[1].set_title('required DF front share (%)', fontsize=8)
        axs[1].set_xlabel('lateral g'); axs[1].axhline(46, ls='--', lw=0.7, color=BLACK); axs[1].legend(fontsize=6)
        fig.savefig(os.path.join(cand_dir, 'report_aero.png')); plt.close(fig)

    def fig_transient():
        try:
            from vahan.transient import (TransientSolver, TransientInputs,
                                          TransientParams, SteeringProfile)
            mF = v.sprung_mass_kg
            Ixx = mF * (v.roll_gyradius_track_frac * (v.front_track_m + v.rear_track_m) / 2) ** 2
            Izz = v.yaw_inertia_factor * v.total_mass_kg * v.cg_to_front_axle_m * v.cg_to_rear_axle_m
            k_roll = v.roll_stiffness_total_Npm_rad
            zeta = 0.5   # NO shock dyno data — assumed ratio, same flag as the binder
            c_phi = 2 * zeta * (k_roll * Ixx) ** 0.5
            tp = TransientParams(sprung_roll_inertia=Ixx, yaw_inertia=Izz,
                                 roll_damping_Nms_rad=c_phi)
            try: sg = w._build_steering_geometry()
            except Exception: sg = None
            sol = TransientSolver(v, _W['w']._tire_model, corner_solvers=w._solvers,
                                  params=tp, steering_geometry=sg)
            vx = 12.0
            steer_rad = np.radians(6.0)   # ~6 deg road-wheel step (moderate, sub-limit at 12 m/s)
            profiles = {'step': SteeringProfile.step(0.5, steer_rad),
                        'ramp': SteeringProfile.ramp(0.5, 2.5, steer_rad)}
            fig, axs = plt.subplots(2, 3, figsize=(11, 6))
            for row, kind in enumerate(('step', 'ramp')):
                try:
                    res = sol.simulate(TransientInputs(v_x_target_ms=vx,
                                                       steering=profiles[kind], duration_s=4.0))
                    tt = np.asarray(res.t)
                    axs[row,0].plot(tt, np.degrees(np.asarray(res.yaw_rate)), color=BLUE)
                    axs[row,0].set_title('%s steer: yaw rate (deg/s)' % kind, fontsize=8)
                    axs[row,1].plot(tt, np.asarray(res.ay)/9.80665, color=RED)
                    axs[row,1].set_title('lateral g', fontsize=8)
                    axs[row,2].plot(tt, np.degrees(np.asarray(res.roll)), color=OCHRE, label='roll (deg)')
                    axs[row,2].plot(tt, np.degrees(np.asarray(res.steer)), color=BLACK, lw=0.8,
                                    label='road-wheel steer (deg)')
                    axs[row,2].set_title('roll + input', fontsize=8); axs[row,2].legend(fontsize=6)
                except Exception as e:
                    axs[row,0].text(0.05, 0.5, '%s FAIL %s' % (kind, str(e)[:60]), fontsize=7)
            for a in axs.ravel(): a.tick_params(labelsize=6); a.set_xlabel('t (s)', fontsize=7)
            fig.suptitle('transient steering response @ %.0f m/s — damping ASSUMED zeta=0.5 (no dyno data)' % vx,
                         fontsize=9)
            fig.savefig(os.path.join(cand_dir, 'report_transient.png')); plt.close(fig)
        except Exception as e:
            print('transient FAIL:', e)

    def fig_mmd():
        import vahan.analysis_plots as ap
        try:
            wf = v.cg_to_rear_axle_m / v.wheelbase_m
            f = ap.plot_mmd(_W['w']._tire_model, v.total_mass_kg, wf, v.wheelbase_m,
                            v.cg_height_m, v.front_track_m, v.rear_track_m,
                            roll_stiffness_front_Npm_rad=v.roll_stiffness_front_Npm_rad,
                            roll_stiffness_rear_Npm_rad=v.roll_stiffness_rear_Npm_rad)
            lightify(f).savefig(os.path.join(cand_dir, 'report_mmd.png')); plt.close(f)
        except Exception as e:
            print('mmd FAIL', e)

    fig_kin(); fig_dyn(); fig_model(); fig_gg(); fig_tireutil(); fig_aero()
    fig_transient(); fig_mmd()
    print('report figures written ->', cand_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='designs_city/run1')
    ap.add_argument('--base', default='configs/2027_v11_(raised_arb_free_springs).vahan')
    ap.add_argument('--islands', type=int, default=4)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--gens', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--lap-top', type=int, default=0, help='after evolution, lap-sim the top N archive members')
    ap.add_argument('--report', default=None, help='generate full report figures for one candidate dir and exit')
    ap.add_argument('--rescore', default=None, help='re-derive dynamics metrics of an existing run dir and exit')
    a = ap.parse_args()
    if a.rescore:
        rescore_run(a.rescore, workers=a.workers, base_config=a.base)
    elif a.report:
        full_report(a.report)
    else:
        run_city(a.out, a.base, [], a.islands, a.pop, a.gens, a.workers, a.seed)
        if a.lap_top > 0:
            lap_filter(a.out, a.lap_top)
