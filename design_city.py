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

GENES = [
    # name,                lo,     hi,    kind
    ('spring_F_lbfin',     150.0,  300.0, 'step25'),
    ('spring_R_lbfin',     175.0,  325.0, 'step25'),
    ('lever_scale_F',      0.70,   1.60,  'float'),   # ARB lever length scale (rate ~ 1/A^2.5)
    ('lever_scale_R',      0.60,   1.60,  'float'),
    ('static_camber_F',   -2.5,    0.0,   'float'),
    ('static_camber_R',   -2.5,    0.0,   'float'),
    ('uca_in_dz_F_mm',   -15.0,   15.0,   'float'),   # front upper inboard plane height (RC knob)
    ('uca_in_dz_R_mm',   -15.0,   15.0,   'float'),
    ('lca_in_dz_F_mm',   -10.0,   10.0,   'float'),
    ('lca_in_dz_R_mm',   -10.0,   10.0,   'float'),
    ('tie_in_dz_mm',      -6.0,    6.0,   'float'),   # front bump-steer knob
]

ISLANDS = {   # objective personalities ("buildings"); weights over normalized metrics
    'grip':      dict(gmax=3.0, margin=0.5, feel=0.5, endur=0.5),
    'rotation':  dict(gmax=1.0, margin=3.0, feel=0.5, endur=0.5),
    'endurance': dict(gmax=1.0, margin=0.8, feel=0.5, endur=3.0),
    'balanced':  dict(gmax=1.5, margin=1.5, feel=1.5, endur=1.0),
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

def mutate(g, rng, scale=0.25):
    out = dict(g)
    for gname, lo, hi, kind in GENES:
        if rng.random() < 0.45:
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
        alignment=dict(w._alignment), dyn_state=w._dynamics_panel.get_state())

def _apply(genome):
    np = _W['np']; w = _W['w']; base = _W['base']
    import copy
    for k, v in base['front_hp'].items(): w._front_hp[k] = v.copy()
    for k, v in base['rear_hp'].items():  w._rear_hp[k] = v.copy()
    w._front_arb = copy.deepcopy(base['front_arb']); w._rear_arb = copy.deepcopy(base['rear_arb'])
    w._alignment.update(base['alignment'])
    st = dict(base['dyn_state'])
    st['spring_front_lbfin'] = genome['spring_F_lbfin']
    st['spring_rear_lbfin'] = genome['spring_R_lbfin']
    w._dynamics_panel.set_state(st)
    w._alignment['front_camber_deg'] = genome['static_camber_F']
    w._alignment['rear_camber_deg'] = genome['static_camber_R']
    # hardpoint deltas (m)
    for hp, key, dz in (('front_hp', 'uca_front', genome['uca_in_dz_F_mm']),
                        ('front_hp', 'uca_rear',  genome['uca_in_dz_F_mm']),
                        ('rear_hp',  'uca_front', genome['uca_in_dz_R_mm']),
                        ('rear_hp',  'uca_rear',  genome['uca_in_dz_R_mm']),
                        ('front_hp', 'lca_front', genome['lca_in_dz_F_mm']),
                        ('front_hp', 'lca_rear',  genome['lca_in_dz_F_mm']),
                        ('rear_hp',  'lca_front', genome['lca_in_dz_R_mm']),
                        ('rear_hp',  'lca_rear',  genome['lca_in_dz_R_mm']),
                        ('front_hp', 'tie_rod_inner', genome['tie_in_dz_mm'])):
        d = getattr(w, '_' + hp)
        if key in d: d[key] = d[key] + np.array([0, 0, dz / 1000.0])
    # ARB lever scales
    for arb, s in ((w._front_arb, genome['lever_scale_F']), (w._rear_arb, genome['lever_scale_R'])):
        piv = np.asarray(arb['arb_pivot']); tip = np.asarray(arb['arb_arm_end'])
        arb['arb_arm_end'] = piv + s * (tip - piv)
    w._rebuild_solvers(0.)

def _axle_util(ss, r):
    np = _W['np']; out = {}
    for ax, (a, b) in (('F', ('FL', 'FR')), ('R', ('RL', 'RR'))):
        dem = abs(r.Fy[a]) + abs(r.Fy[b]); cap = 0.
        for c in (a, b):
            fz = max(r.Fz[c], 0.)
            t = ss._tire_for(c)
            fr = t.fz_range() if callable(t.fz_range) else t.fz_range
            lo = float(np.asarray(fr).ravel()[0])
            mu = float(t.peak_mu(max(fz, lo), abs(r.camber.get(c, 0.)))) * ss._mu_scale
            cap += mu * fz
        out[ax] = dem / max(cap, 1.)
    return out

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
        feel = 1.0 - min(M['bump_steer_F']/0.3, 1) * 0.5 - min(abs(M['scrub_F'])/30, 1) * 0.5
        endur = 1.0 - min(max(M['rc_F'], M['rc_R'], 0)/80, 1)*0.5 - min(M['roll_deg_g']/1.5, 1)*0.5
        M['scores'] = dict(gmax=(M['gmax']-1.6)/0.6, margin=min(max(M['margin'],0)/0.12,1.2),
                           feel=feel, endur=endur)
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
    return sum(weights[k] * s[k] for k in weights) - M['penalty']

def dominates(a, b):
    ka = (a['scores']['gmax'], a['scores']['margin'], a['scores']['feel'], a['scores']['endur'])
    kb = (b['scores']['gmax'], b['scores']['margin'], b['scores']['feel'], b['scores']['endur'])
    return all(x >= y for x, y in zip(ka, kb)) and any(x > y for x, y in zip(ka, kb))

def run_city(out_dir, base_config, seed_configs, islands, pop, gens, workers, seed=7):
    import multiprocessing as mp_
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    isl_names = list(ISLANDS.keys())[:islands]
    # founders (LHS from scratch) + lineage (mutations of seeds -> baseline genome = all-neutral)
    neutral = dict(spring_F_lbfin=200., spring_R_lbfin=250., lever_scale_F=1.0, lever_scale_R=1.0,
                   static_camber_F=-0.8, static_camber_R=-0.7, uca_in_dz_F_mm=0., uca_in_dz_R_mm=0.,
                   lca_in_dz_F_mm=0., lca_in_dz_R_mm=0., tie_in_dz_mm=0.)
    pops = {}
    for i, nm in enumerate(isl_names):
        founders = lhs_genomes(pop // 2, rng)                       # from-scratch set
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='designs_city/run1')
    ap.add_argument('--base', default='configs/2027_v11_(raised_arb_free_springs).vahan')
    ap.add_argument('--islands', type=int, default=4)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--gens', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    run_city(a.out, a.base, [], a.islands, a.pop, a.gens, a.workers, a.seed)
