"""MAP-Elites quality-diversity engine for rocker/ARB PACKAGING.

The wheel side never moves -> every kinematic curve is held EXACTLY (0%).
The rocker + coilover + ARB are a rigid inboard assembly with huge freedom;
this fills a MAP-Elites archive of DIVERSE valid packagings, each with:
  - actuation coplanarity EXACT (all chain points built in one plane)
  - motion ratio re-tuned to baseline (<= tol)
  - damper acting as a PUSHROD (compresses in bump) -- hard sign guard
  - ARB rate re-tuned to baseline via bar OD (K_wheel ~ OD^4-ID^4)
  - triad law by construction; clash-free across full travel
Descriptor (diversity axes) = packaging positions of the moved hardware.
Quality (rank within a cell) = worst clearance margin (most buildable wins).

Genome (all continuous, mapped from [0,1)):
  t1,t2   plane tilt about pushrod_outer (contains the fixed pushrod foot)
  pv_a,pv_r  rocker pivot polar offset from the pushrod foot, in-plane
  in_a    pushrod-inner angle on the rocker (input lever; RADIUS = MR knob)
  eye_a,eye_r spring eye polar on the rocker
  sc_a    damper direction from the eye (damper length fixed) -> spring chassis
  dr_a,dr_r  ARB drop-top polar on the rocker
  chir    blade chirality (0/1)
"""
import os, sys, json, glob, time
import numpy as np
from . import packaging as PK
from .interference import seg_seg_distance as ssd

# clearance radii (mm)
SPR, TUBE, RE, BRG, LINK = 31.5, 7.94, 8.0, 19.05, 6.0


class PackagingQD:
    def __init__(self, win, axle='front', cfg=None, seed=0,
                 sc_pin_mm=None, plane_tilt_deg=28.0, pivot_spread_mm=90.0):
        self.w = win
        self.axle = axle
        self.rng = np.random.default_rng(seed)
        self.b0 = PK.get_bundle(win, axle)
        hp0 = self.b0['hp']
        self.P0 = np.asarray(hp0['pushrod_outer'], float)   # fixed foot
        self.base = PK.capture_baseline(win)
        self.tgt_mr = float(self.base['rates'][
            'motion_ratio_front' if axle == 'front' else 'motion_ratio_rear'])
        self.tgt_arb = float(self.base['rates'][
            'arb_rate_front_Npm' if axle == 'front' else 'arb_rate_rear_Npm'])
        self.sign0 = float(self.base['damper_sign'][axle])
        c0, n0 = PK._chain_plane(self.b0)
        self.n0 = n0
        # in-plane basis at the foot
        e1 = np.asarray(hp0['pushrod_inner'], float) - self.P0
        e1 = e1 - np.dot(e1, n0) * n0
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(n0, self.e1)
        self.pv0 = np.asarray(hp0['rocker_pivot'], float)
        self.D_DMP = float(np.linalg.norm(
            np.asarray(hp0['rocker_spring_pt'], float)
            - np.asarray(hp0['spring_chassis_pt'], float)))
        self.AX = float(np.linalg.norm(
            np.asarray(hp0['rocker_axis_pt'], float) - self.pv0))
        # ARB baseline invariants
        dt0 = np.asarray(self.b0['arb']['arb_drop_top'], float)
        ae0 = np.asarray(self.b0['arb']['arb_arm_end'], float)
        pv0a = np.asarray(self.b0['arb']['arb_pivot'], float)
        self.L_LINK = float(np.linalg.norm(ae0 - dt0))
        self.BETA = float(np.linalg.norm(ae0 - pv0a))
        X = np.array([1.0, 0, 0])
        bl0 = (ae0 - pv0a) / self.BETA
        dr0 = (dt0 - ae0) / self.L_LINK
        self.ANG_BB = float(np.arccos(np.clip(np.dot(bl0, X), -1, 1)))
        self.ANG_BD = float(np.arccos(np.clip(np.dot(bl0, dr0), -1, 1)))
        self.ANG_XD = float(np.arccos(np.clip(np.dot(dr0, X), -1, 1)))
        self.OD0 = float(self._od_spin().value())
        self.ID0 = float((win._dynamics_panel._arb_ID_f
                          if axle == 'front'
                          else win._dynamics_panel._arb_ID_r).value())
        self.sc_pin = (np.asarray(sc_pin_mm, float) / 1000.0
                       if sc_pin_mm is not None else None)
        self.plane_tilt = np.radians(plane_tilt_deg)
        self.pivot_spread = pivot_spread_mm / 1000.0
        self.arm_dmp = 188.0
        self.X = X

    def _od_spin(self):
        return (self.w._dynamics_panel._arb_OD_f if self.axle == 'front'
                else self.w._dynamics_panel._arb_OD_r)

    # ── genome -> geometry ───────────────────────────────────────────────────
    def _plane(self, t1, t2):
        # rotate the baseline in-plane basis about the foot by two angles
        def rot(v, axis, ang):
            axis = axis / np.linalg.norm(axis)
            c, s = np.cos(ang), np.sin(ang)
            return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
        n = rot(self.n0, self.e1, t2)
        n = rot(n, np.array([0.0, 0, 1.0]), t1)
        n = n / np.linalg.norm(n)
        e1 = self.e1 - np.dot(self.e1, n) * n
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        return n, e1, e2

    def build(self, g, in_r=None):
        t1 = (g[0] - 0.5) * 2 * self.plane_tilt
        t2 = (g[1] - 0.5) * 2 * self.plane_tilt
        n, e1, e2 = self._plane(t1, t2)
        # rocker pivot: polar offset from the foot's baseline pivot projection
        pv_a = g[2] * 2 * np.pi
        pv_r = 0.10 + g[3] * self.pivot_spread   # 100 mm .. 100+spread
        piv = self.P0 + pv_r * (np.cos(pv_a) * e1 + np.sin(pv_a) * e2)
        # rocker-local frame
        u, v = e1, e2
        in_a = g[4] * 2 * np.pi
        ir = (0.045 if in_r is None else in_r)
        pin = piv + ir * (np.cos(in_a) * u + np.sin(in_a) * v)
        eye_a = g[5] * 2 * np.pi
        eye_r = 0.040 + g[6] * 0.055
        eye = piv + eye_r * (np.cos(eye_a) * u + np.sin(eye_a) * v)
        if self.sc_pin is not None:
            sc = self.sc_pin.copy()
        else:
            sc_a = g[7] * 2 * np.pi
            sc = eye + self.D_DMP * (np.cos(sc_a) * e1 + np.sin(sc_a) * e2)
        b = {'hp': {k: np.array(x, float) for k, x in self.b0['hp'].items()},
             'arb': {k: np.array(x, float)
                     for k, x in self.b0['arb'].items()}}
        b['hp']['pushrod_inner'] = pin
        b['hp']['rocker_pivot'] = piv
        b['hp']['rocker_spring_pt'] = eye
        b['hp']['spring_chassis_pt'] = sc
        b['hp']['rocker_axis_pt'] = piv + n * self.AX
        return b, n, e1, e2, u, v

    def add_arb(self, b, n, u, v, g):
        piv = np.asarray(b['hp']['rocker_pivot'], float)
        dr_a = g[8] * 2 * np.pi
        dr_r = 0.032 + g[9] * 0.030
        D = piv + dr_r * (np.cos(dr_a) * u + np.sin(dr_a) * v)
        # link direction pinned by the bar-drop-angle law (2 roots x2), pick
        # by chirality bit + a sub-choice folded into g[10]
        c1x, c2x = float(np.dot(u, self.X)), float(np.dot(v, self.X))
        Rm = float(np.hypot(c1x, c2x)); ph0 = float(np.arctan2(c2x, c1x))
        K = np.cos(self.ANG_XD)
        if Rm < 1e-9 or abs(K) > Rm:
            return None
        dd = float(np.arccos(np.clip(K / Rm, -1.0, 1.0)))
        # ONLY the two roots that put dot(drop, bar) = +cos(bar-drop angle);
        # the +pi roots flip the drop to 180-angle and break the triad.
        roots = [ph0 + dd, ph0 - dd]
        psi = roots[int(g[10] * 2) % 2]
        dhat = np.cos(psi) * u + np.sin(psi) * v
        E = D - self.L_LINK * dhat
        g2 = dhat - np.dot(dhat, self.X) * self.X
        ng2 = np.linalg.norm(g2)
        if ng2 < 1e-9:
            return None
        g2 = g2 / ng2
        g3 = np.cross(self.X, g2)
        pp = np.cos(self.ANG_BB)
        qq = (np.cos(self.ANG_BD) - pp * np.dot(dhat, self.X)) / ng2
        rr2 = 1.0 - pp * pp - qq * qq
        if rr2 < 0:
            return None
        sgn = 1.0 if g[11] < 0.5 else -1.0
        B = pp * self.X + qq * g2 + sgn * np.sqrt(rr2) * g3
        bb = {'hp': {k: np.array(x, float) for k, x in b['hp'].items()},
              'arb': {k: np.array(x, float) for k, x in b['arb'].items()}}
        bb['arb']['arb_drop_top'] = D
        bb['arb']['arb_arm_end'] = E
        bb['arb']['arb_pivot'] = E - self.BETA * B
        return bb

    # ── cheap static clearance (pure geometry) ───────────────────────────────
    def static_margin(self, bb, od_mm):
        hp, arb = bb['hp'], bb['arb']
        rs = np.asarray(hp['rocker_spring_pt'], float)
        sc = np.asarray(hp['spring_chassis_pt'], float)
        piv = np.asarray(hp['rocker_pivot'], float)
        pin = np.asarray(hp['pushrod_inner'], float)
        D = np.asarray(arb['arb_drop_top'], float)
        E = np.asarray(arb['arb_arm_end'], float)
        Pb = np.asarray(arb['arb_pivot'], float)
        bar_b = Pb.copy(); bar_b[0] = -Pb[0]
        barR = 0.5 * od_mm / 1000.0 * 1000.0
        return min(
            ssd(rs, sc, D, E) * 1000 - (SPR + LINK),
            ssd(rs, sc, D, D) * 1000 - (SPR + RE),
            ssd(rs, sc, Pb, bar_b) * 1000 - (SPR + barR),
            ssd(rs, sc, self.P0, pin) * 1000 - (SPR + TUBE),
            ssd(rs, sc, piv, piv) * 1000 - (SPR + BRG),
            ssd(D, E, piv, piv) * 1000 - (BRG + LINK),
            ssd(D, E, self.P0, pin) * 1000 - (TUBE + LINK),
            ssd(Pb, bar_b, self.P0, pin) * 1000 - (barR + TUBE),
            np.linalg.norm(D - piv) * 1000 - (BRG + RE))

    # ── rate knobs ───────────────────────────────────────────────────────────
    def mr_tune(self, g):
        """Bisect the pushrod input lever (in_r) to hit the baseline MR."""
        def mr_of(ir):
            b, n, e1, e2, u, v = self.build(g, in_r=ir)
            return PK.solver_mr(PK._corner_solver(self.w, self.axle, b))
        lo, hi = 0.020, 0.110
        f_lo = mr_of(lo) - self.tgt_mr
        f_hi = mr_of(hi) - self.tgt_mr
        if f_lo * f_hi > 0:
            return None
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if f_lo * (mr_of(mid) - self.tgt_mr) <= 0:
                hi = mid
            else:
                lo, f_lo = mid, mr_of(lo) - self.tgt_mr
        return 0.5 * (lo + hi)

    def od_for_rate(self, bb):
        spin = self._od_spin()

        def rate_at(od):
            spin.setValue(float(od))
            PK.set_bundle(self.w, self.axle, bb)
            return float(PK.panel_arb_rate(self.w, self.axle))
        lo, hi = 5.0, 34.0
        r_lo, r_hi = rate_at(lo), rate_at(hi)
        if (r_lo - self.tgt_arb) * (r_hi - self.tgt_arb) > 0:
            return None
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if (r_lo - self.tgt_arb) * (rate_at(mid) - self.tgt_arb) <= 0:
                hi = mid
            else:
                lo = mid; r_lo = rate_at(lo)
        od = 0.5 * (lo + hi)
        return od, rate_at(od)

    # ── fast travel-swept clash (cheap; no full _clash_sweep) ────────────────
    def travel_margin(self, bb, od_mm):
        """Worst clearance across droop/static/bump for the members that
        actually swing (coilover, pushrod, ARB link/bar/rod-end, bearing).
        Solves the FL corner at 3 travels and rigidly rotates the ARB drop
        top with the rocker.  ~6 solver calls; far cheaper than _clash_sweep.
        """
        PK.set_bundle(self.w, self.axle, bb)
        try:
            lo, hi = PK.travel_range_m(self.w)
            solver = self.w._solvers['FL' if self.axle == 'front' else 'RL']
        except Exception:
            return -1e9
        piv0 = np.asarray(bb['hp']['rocker_pivot'], float)
        eye0 = np.asarray(bb['hp']['rocker_spring_pt'], float)
        _, n_pl = PK._chain_plane(bb)
        D0 = np.asarray(bb['arb']['arb_drop_top'], float)
        E0 = np.asarray(bb['arb']['arb_arm_end'], float)
        Pb = np.asarray(bb['arb']['arb_pivot'], float)
        bar_b = Pb.copy(); bar_b[0] = -Pb[0]
        sc = np.asarray(bb['hp']['spring_chassis_pt'], float)
        barR = 0.5 * od_mm / 1000.0 * 1000.0
        u0 = eye0 - piv0
        u0p = u0 - np.dot(u0, n_pl) * n_pl
        u0p = u0p / max(np.linalg.norm(u0p), 1e-12)
        worst = 1e9
        for t in (lo, 0.0, hi):
            try:
                st = solver.solve(float(t))
                eye = np.asarray(st.rocker_spring_pt, float)
                pin = np.asarray(st.pushrod_inner, float)
                po = np.asarray(st.pushrod_outer, float)
                piv = np.asarray(st.rocker_pivot, float)                     if hasattr(st, 'rocker_pivot') else piv0
            except Exception:
                return -1e9
            # rocker rotation this travel -> spin the drop-top about the pivot
            u1 = eye - piv
            u1p = u1 - np.dot(u1, n_pl) * n_pl
            u1p = u1p / max(np.linalg.norm(u1p), 1e-12)
            th = float(np.arctan2(np.dot(np.cross(u0p, u1p), n_pl),
                                  np.dot(u0p, u1p)))
            c, sn = np.cos(th), np.sin(th)
            rel = D0 - piv0
            D = piv + (rel * c + np.cross(n_pl, rel) * sn
                       + n_pl * np.dot(n_pl, rel) * (1 - c))
            m = min(
                ssd(eye, sc, D, E0) * 1000 - (SPR + LINK),
                ssd(eye, sc, D, D) * 1000 - (SPR + RE),
                ssd(eye, sc, Pb, bar_b) * 1000 - (SPR + barR),
                ssd(eye, sc, po, pin) * 1000 - (SPR + TUBE),
                ssd(eye, sc, piv, piv) * 1000 - (SPR + BRG),
                ssd(D, E0, po, pin) * 1000 - (TUBE + LINK),
                ssd(D, E0, piv, piv) * 1000 - (BRG + LINK),
                ssd(Pb, bar_b, po, pin) * 1000 - (barR + TUBE))
            worst = min(worst, float(m))
        return worst

    # ── full evaluation of one genome ────────────────────────────────────────
    def evaluate(self, g, tol, full_clash=True):
        ir = self.mr_tune(g)
        if ir is None:
            return None
        b, n, e1, e2, u, v = self.build(g, in_r=ir)
        # damper sign guard — reject pullrods immediately
        PK.set_bundle(self.w, self.axle, b)
        if PK.damper_motion_sign(self.w, self.axle) != self.sign0:
            return None
        bb = self.add_arb(b, n, u, v, g)
        if bb is None:
            return None
        odr = self.od_for_rate(bb)
        if odr is None:
            return None
        od, rate = odr
        m = self.static_margin(bb, od)
        if m < 2.0:
            self._od_spin().setValue(self.OD0)
            return None
        if full_clash:
            # FULL oracle (final validation of an archive member): rates,
            # laws, damper sign, and the complete travel clash sweep.
            self._od_spin().setValue(float(od))
            PK.set_bundle(self.w, self.axle, bb)
            res = PK.validate(self.w, self.base, tol, allow_wheel_motion=True)
            self._od_spin().setValue(self.OD0)
            if res.failures():
                return None
            worst = max((PK_check_frac(c) for c in res.checks
                         if c['tol'] > 1e-12
                         and c['name'] != 'wheel points moved'
                         and not c['name'].startswith('clash')), default=0.0)
            m = self.travel_margin(bb, od)
        else:
            # CHEAP archive eval.  Coplanarity is only STATIC-exact by
            # construction; ACROSS TRAVEL the chain drifts, so the geometric
            # laws (coplanar / ARB in-plane / triad, all swept) MUST be gated
            # here or the archive fills with candidates that die at full
            # validation (the pinned-target 0-solution bug).
            self._od_spin().setValue(float(od))
            PK.set_bundle(self.w, self.axle, bb)
            gl = PK._axle_geometry_laws(self.w, self.axle)
            bg = self.base['geometry'][self.axle]
            ok_law = (gl['coplanar_mm'] <= tol.coplanar_mm
                      and gl['arb_drop_top_inplane_mm'] <= tol.arb_inplane_mm
                      and gl['arb_arm_end_inplane_mm'] <= tol.arb_inplane_mm
                      and all(abs(gl[k] - bg[k]) <= tol.triad_deg for k in
                              ('triad_bar_blade_deg', 'triad_blade_drop_deg',
                               'triad_bar_drop_deg')))
            if not ok_law:
                self._od_spin().setValue(self.OD0)
                return None
            tm = self.travel_margin(bb, od)
            self._od_spin().setValue(self.OD0)
            if tm < 2.0:
                return None
            m = tm
            worst = 0.0
        sc = np.asarray(bb['hp']['spring_chassis_pt'], float) * 1000
        Pb = np.asarray(bb['arb']['arb_pivot'], float) * 1000
        piv = np.asarray(bb['hp']['rocker_pivot'], float) * 1000
        return {'genome': list(map(float, g)), 'in_r_mm': ir * 1000,
                'bar_OD_mm': od, 'arb_rate': rate, 'static_margin_mm': m,
                'worst_tol_frac': worst,
                'sc_mm': sc.tolist(), 'bar_yz': [Pb[1], Pb[2]],
                'bar_half_mm': abs(Pb[0]), 'pivot_mm': piv.tolist(),
                'bundle': {'hp': {k: np.asarray(x, float).tolist()
                                  for k, x in bb['hp'].items()},
                           'arb': {k: np.asarray(x, float).tolist()
                                   for k, x in bb['arb'].items()}}}


from .relocate import _check_frac as PK_check_frac


def run_mapelites(qd, tol, n_init=800, n_iter=6000, cell_mm=18.0,
                  iso=0.10, line=0.35, n_show=None, progress=None, log_every=500):
    """MAP-Elites: fill an archive of DIVERSE valid rocker/ARB packagings.
    Descriptor (diversity axes) = spring_chassis (y,z) + ARB bar y, each binned
    at cell_mm; a chirality/root bit splits topologies.  Quality within a cell
    = static clearance margin (most buildable wins).  Elites are mutated by
    iso+line (Vassiliades/Mouret): child = elite + N(0,iso) + line*U*(e2-e1),
    which travels ALONG the feasible corridor between two elites.  Returns the
    archive as a list of solutions (genomes + resolved bundles)."""
    import numpy as np
    arch = {}                      # cell -> (quality, solution)
    genomes = {}                   # cell -> genome (for mutation parents)

    def cell_of(sol):
        sc = sol['sc_mm']; by = sol['bar_yz'][0]
        root_bit = int(round(sol['genome'][10]))
        chir_bit = int(sol['genome'][11] >= 0.5)
        return (int(np.floor(sc[1] / cell_mm)), int(np.floor(sc[2] / cell_mm)),
                int(np.floor(by / cell_mm)), root_bit, chir_bit)

    def consider(g):
        sol = qd.evaluate(g, tol, full_clash=False)   # cheap: archive tier
        if sol is None:
            return False
        c = cell_of(sol)
        q = sol['static_margin_mm']
        if c not in arch or q > arch[c][0]:
            arch[c] = (q, sol, np.asarray(g, float))
            genomes[c] = np.asarray(g, float)
            return True
        return False

    n_eval = 0
    # ── init: random ──
    for _ in range(int(n_init)):
        consider(qd.rng.random(12)); n_eval += 1
        if progress and n_eval % log_every == 0:
            progress('init %d/%d evals, archive %d cells'
                     % (n_eval, n_init, len(arch)))
    # ── iterate: iso+line mutation of elites ──
    for _ in range(int(n_iter)):
        n_eval += 1
        if len(genomes) >= 2 and qd.rng.random() < 0.9:
            keys = list(genomes)
            e1 = genomes[keys[qd.rng.integers(len(keys))]]
            e2 = genomes[keys[qd.rng.integers(len(keys))]]
            child = e1 + qd.rng.normal(0, iso, 12) + line * qd.rng.normal() * (e2 - e1)
            child = np.clip(child, 0.0, 1.0)
        else:
            child = qd.rng.random(12)
        consider(child)
        if progress and n_eval % log_every == 0:
            progress('iter %d, archive %d cells' % (n_eval, len(arch)))
    # ── FINAL TIER: full oracle (rates, laws, damper sign, full clash sweep)
    # on every archive member; keep only the ones that truly pass.  Tally the
    # binding constraint of each REJECTED cell so a 0-result run can name what
    # bound it (the GUI needs reject_reasons to explain "why 0"). ────────────
    if progress:
        progress('cheap archive %d cells; running full oracle on each...'
                 % len(arch))
    import numpy as np
    final = []
    reject_reasons = {}
    for _q, sol, g in arch.values():
        full = qd.evaluate(g, tol, full_clash=True)
        if full is not None:
            final.append(full)
            continue
        # rejected at full validation — re-run once to capture the reason
        try:
            b = {'hp': {k: np.array(v, float)
                        for k, v in sol['bundle']['hp'].items()},
                 'arb': {k: np.array(v, float)
                         for k, v in sol['bundle']['arb'].items()}}
            qd._od_spin().setValue(float(sol['bar_OD_mm']))
            PK.set_bundle(qd.w, qd.axle, b)
            res = PK.validate(qd.w, qd.base, tol, allow_wheel_motion=True)
            qd._od_spin().setValue(qd.OD0)
            for c in res.failures():
                why = '%s %s' % (c['axle'], c['name'])
                reject_reasons[why] = reject_reasons.get(why, 0) + 1
        except Exception:
            pass
    final.sort(key=lambda s: (-s['static_margin_mm']))
    if n_show:
        final = final[:int(n_show)]
    return final, len(arch), n_eval, reject_reasons


def qd_relocate_search(win, axle, dict_name, key, target_mm, baseline=None,
                       tol=None, n_init=900, n_iter=2600, min_sep_frac=0.20,
                       seed=0, n_show=40, progress=None):
    """GUI-facing packaging-QD relocation: PIN (dict_name.key) at target_mm and
    fill a MAP-Elites archive of DIVERSE rocker/ARB assemblies that hold every
    wheel curve exactly and keep the dynamics within tol.  Returns the same
    result shape the Relocate tab already renders (solutions[] with pos_mm /
    dist_from_* / worst_tol_frac / clash_checked / bundle).  Only the front/rear
    spring_chassis_pt is supported today (the point the rocker packages around);
    callers fall back to generative_search for other keys."""
    import numpy as np
    tol = tol or PK.Tolerances()
    if not (dict_name == 'hp' and key == 'spring_chassis_pt'):
        return {'error': 'packaging-QD supports spring_chassis_pt today; use '
                         'the generative search for %s.%s' % (dict_name, key),
                'solutions': []}
    tgt = np.asarray(target_mm, float)
    qd = PackagingQD(win, axle, seed=seed, plane_tilt_deg=20.0,
                     pivot_spread_mm=120.0, sc_pin_mm=tuple(tgt))
    if baseline is not None:
        qd.base = baseline
    p0 = np.asarray((win._front_hp if axle == 'front'
                     else win._rear_hp)[key], float) * 1000.0
    sols, ncell, nev, rej = run_mapelites(qd, tol, n_init=n_init,
                                          n_iter=n_iter, cell_mm=12.0,
                                          n_show=n_show, progress=progress)
    qd._od_spin().setValue(qd.OD0)
    out = []
    for s in sols:
        out.append({
            'pos_mm': [round(float(x), 3) for x in tgt],
            'delta_mm': [round(float(a - b), 3) for a, b in zip(tgt, p0)],
            'dist_from_og_mm': round(float(np.linalg.norm(tgt - p0)), 2),
            'dist_from_target_mm': 0.0,
            'worst_tol_frac': round(float(s['worst_tol_frac']), 3),
            'clash_checked': True,
            'bar_OD_mm': round(float(s['bar_OD_mm']), 2),
            'resolved': {'bar_OD_mm': round(float(s['bar_OD_mm']), 2),
                         'bar_half_mm': round(float(s['bar_half_mm']), 1)},
            'bundle': s['bundle']})
    # If the cheap archive never found a single feasible packaging, the reason
    # is upstream of the full oracle (no coplanar/rate/sign-valid rocker at
    # this pinned point) — say so; otherwise report the full-oracle rejects.
    if not rej and ncell == 0:
        rej = {'%s no coplanar/rate-valid rocker at this pinned point' % axle:
               nev}
    return {'point': '%s %s.%s' % (axle, dict_name, key),
            'method': 'packaging MAP-Elites (rocker/ARB quality-diversity)',
            'original_mm': [round(float(x), 3) for x in p0],
            'target_mm': [round(float(x), 2) for x in tgt],
            'tree_size': nev, 'span_mm': 0.0,
            'best_approach_mm': 0.0,
            'target_reached': bool(out),          # honest: only if solutions
            'min_sep_mm': 0.0, 'n_final_rejects': ncell - len(out),
            'reject_reasons': rej,
            'n_cells': ncell, 'n_evals': nev, 'solutions': out}
