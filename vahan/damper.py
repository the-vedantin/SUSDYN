"""Damper force-velocity model from manufacturer dyno curves.

ONE MODEL: the project declares the damper in car['damper'] (see
`project_damper_from_car`); every consumer (Ride page, ride study, transient
roll damping, skidpad panel) gets it through MainWindow._damper_model() and
converts to the wheel with the motion ratio of the solved kinematics:

    v_damper = MR * v_wheel          F_wheel = F_damper(v_damper) * MR

(MR = damper / wheel travel, VehicleParams.motion_ratio_*; the static MR —
the ride/roll models are linearised about static, so is the conversion).

Dyno file schema (JSON; velocities mm/s, forces N, POSITIVE magnitudes for
both compression and rebound):
    {"damper": ..., "delivered_valve_spec": "C12/R12",
     "adjusters": {"LS": "clicks from fully closed (0 = stiffest)",
                   "HS": "turns from fully open (0 = softest, 4.3 = stiffest)"},
     "valve_specs": {"C12/R12": {
         "ls_sweep": {"hs_turns": 4.3, "curves": {"0": {"comp": [[v, F], ...],
                                                         "reb": [[v, F], ...]}, "2": ...}},
         "hs_sweep": {"ls_clicks": 0, "curves": {"4.3": {...}, "3": ..., "0": ...}}}, ...}}

COMBINATION ASSUMPTION (parallel flow).  The dyno gives two one-parameter
sweeps per valve spec, not the full LS x HS map.  The low-speed bleed and the
high-speed shim stack are treated as two PARALLEL flow paths seeing the same
pressure (force), so at a given force their velocities add:

    v(F; ls, hs) = v_bleed(F; ls) + v_stack(F; hs)
    v_stack(F; hs) = inverse of the HS-sweep curve at hs (measured with the
                     LS bleed fully closed, ls = 0)
    v_bleed(F; ls) = v_LSsweep(F; ls) - v_stack(F; hs_ref)   (hs_ref = 4.3,
                     the HS setting the LS sweep was run at), clipped >= 0
                     and made non-decreasing in F (bleed flow cannot fall as
                     the pressure rises).

PATH LAYOUT (the 2008 and 2011 TTX25 sheets).  Some sheets do not sweep the
adjusters separately: every curve moves BOTH (settings "0-5", "4-4.5",
"7-4", ... = LS clicks - HS turns), stored as
    "valve_specs": {"C12/R12": {"settings": {"4-4.5": {"ls_clicks": 4,
                    "hs_turns": 4.5, "comp": [...], "reb": [...]}, ...}}}
One path cannot separate two adjusters, so such a valve spec is modelled
along its path only: v(F) interpolated linearly in LS clicks between the
measured settings at fixed force, and the HS turns are the path's own at
those clicks (`normalize_setting` snaps HS to the path and lists the snapped
keys in 'off_path').  Every measured setting is reproduced exactly.

The sheet measures the setting (ls 0, hs 4.3) TWICE (first curve of each
sweep) and the two digitised traces differ (`DamperDyno.consistency()`; up
to ~15 % at a few mm/s on the TTX25 sheets).  The closed-bleed curve of the
LS sweep therefore stands for v_stack(F; 4.3) inside v_bleed, and the
duplicate difference D(F) = v_LS(F; 0) - v_HS(F; 4.3) is added with weight 1
at hs 4.3 fading linearly to 0 at the next measured HS setting.  Result:
every LS-sweep curve and every HS-sweep curve except the duplicate is
reproduced exactly at its own setting; on a self-consistent sheet D = 0 and
this IS the formula above.  Between measured settings the bleed v_bleed(F)
is interpolated LINEARLY in clicks at fixed FORCE (orifice flow at one
pressure), and the stack curve linearly in turns at fixed VELOCITY
(F_stack(v) = mix of the neighbouring HS curves, then inverted): the HS
curves are flat above their blow-off knee, so a fixed-force mix of two of
them averages velocities that differ by an order of magnitude.  Leave-one-
out on the TTX25 MkII sheet (each interior curve withheld and predicted from
its neighbours, `leave_one_out`): HS rms 89 N -> 23 N with this choice, LS
9 N either way.  The superposed v(F) of a setting is clipped at 0 and made
non-decreasing in F.  F(v) is the inverse of v(F),
found by bisection.  F(0) = 0 (gas force and seal friction are not in the
dyno curves and are not modelled).

Above the dyno velocity range (the last measured velocity, 250 mm/s for the
TTX25 sheets) the force is extended linearly with the secant slope of the
last 50 mm/s of the combined curve and the excursion is FLAGGED
(`beyond_dyno_mps`).  Inside the range, a component curve needed at a force
above its own last point (a soft bleed at a stiff total force) is extended
the same way — that is internal to the decomposition, not flagged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

MM = 1e-3
SECANT_SPAN_MPS = 0.05          # last 50 mm/s of the measured range
DIRECTIONS = ('comp', 'reb')
SETTING_KEYS = ('lsc', 'hsc', 'lsr', 'hsr')


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_dyno_path(path, project_path=None):
    """Absolute path of a project-declared dyno file.  Relative paths are
    tried against the repository root, then the project file's folder, then
    the working directory; the first that exists wins (else the repo-root
    candidate is returned so the error names it)."""
    if not path:
        raise ValueError('no dyno file declared')
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    cands = [os.path.join(repo_root(), path)]
    if project_path:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(project_path)), path))
    cands.append(os.path.abspath(path))
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _lin_ext(x, xp, fp):
    """np.interp with LINEAR extrapolation above xp[-1] (last two knots)."""
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y + np.maximum(x - xp[-1], 0.) * slope


class _Curve:
    """One measured curve, SI, made monotone; v(F) and F(v)."""

    def __init__(self, pts, label):
        a = np.asarray(pts, float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a)):
            raise ValueError(f'{label}: need >= 2 finite [v, F] points')
        v, F = a[:, 0] * MM, np.abs(a[:, 1])
        if np.any(v < 0):
            raise ValueError(f'{label}: velocities must be >= 0 (magnitudes)')
        order = np.argsort(v, kind='stable')
        v, F = v[order], F[order]
        keep = np.r_[True, np.diff(v) > 0]          # drop duplicate velocities
        v, F = v[keep], F[keep]
        if v[0] > 0:
            v, F = np.r_[0., v], np.r_[0., F]
        F[0] = 0.                                   # F(0) = 0 (see module doc)
        F = np.maximum.accumulate(F)
        if F[-1] <= 0:
            raise ValueError(f'{label}: no positive force')
        F = F + 1e-9 * F[-1] * np.arange(len(F))    # strictly increasing for inversion
        self.label, self.v, self.F = label, v, F
        self.v_max = float(v[-1])
        span = min(SECANT_SPAN_MPS, self.v_max)
        f_back = float(np.interp(self.v_max - span, v, F))
        self.slope = max((F[-1] - f_back) / span, 1e-6 * F[-1] / self.v_max)

    def v_of_F(self, F):
        F = np.asarray(F, float)
        return np.where(F <= self.F[-1], np.interp(F, self.F, self.v),
                        self.v_max + (F - self.F[-1]) / self.slope)

    def F_of_v(self, v):
        v = np.asarray(v, float)
        return np.interp(v, self.v, self.F) + np.maximum(v - self.v_max, 0.) * self.slope


def _bracket(x, xs):
    """Indices and weight of x between sorted xs (clamped)."""
    x = float(np.clip(x, xs[0], xs[-1]))
    j = int(np.searchsorted(xs, x, side='right'))
    j = min(max(j, 1), len(xs) - 1)
    i = j - 1
    w = 0. if xs[j] == xs[i] else (x - xs[i]) / (xs[j] - xs[i])
    return i, j, w


class _Direction:
    """Compression or rebound of one valve spec: the two sweeps.

    Every component is piecewise linear in F with knots at the measured
    curves' forces, so all of them are tabulated on ONE force grid holding
    every knot (plus a uniform fill): linear interpolation on that grid is
    exact, not an approximation.  The superposed v(F) of a setting is made
    non-decreasing (running max) — at a measured setting it already is (it
    IS the measured curve); between settings digitising noise around a
    stack's crack knee could otherwise make it fold back."""

    def __init__(self, ls_curves, hs_curves, hs_ref, ls_ref, label):
        self.label = label
        self.ls_x = np.array(sorted(ls_curves))
        self.hs_x = np.array(sorted(hs_curves))
        self.ls = [ls_curves[k] for k in self.ls_x]
        self.hs = [hs_curves[k] for k in self.hs_x]
        self.hs_ref, self.ls_ref = float(hs_ref), float(ls_ref)
        f_top = max(c.F[-1] for c in self.ls + self.hs)
        self.F_grid = np.unique(np.r_[np.linspace(0., 1.5 * f_top, 2001),
                                      np.concatenate([c.F for c in self.ls + self.hs])])
        self.stack_tab = [c.v_of_F(self.F_grid) for c in self.hs]
        # velocity grid for the fixed-velocity HS mix: every knot of the HS
        # curves (piecewise-linear exact) up to past the top of F_grid
        v_top = max(float(c.v_of_F(self.F_grid[-1])) for c in self.hs)
        self._stack_v_grid = np.unique(np.r_[np.linspace(0., v_top, 4001),
                                             np.concatenate([c.v for c in self.hs])])
        stack_ref = self._stack_tab(self.hs_ref)
        ls_tab = [c.v_of_F(self.F_grid) for c in self.ls]
        i_closed = np.flatnonzero(np.isclose(self.ls_x, self.ls_ref))
        if len(i_closed):
            # v_bleed = v_LS(ls) - v_stack(hs_ref), with the sheet's own
            # closed-bleed LS curve standing for v_stack(hs_ref): the dyno
            # measured that setting (ls_ref, hs_ref) TWICE, once per sweep,
            # and the two traces differ (consistency()).  The difference
            # D(F) = v_LS(ls_ref) - v_HS(hs_ref) is added back with a weight
            # 1 at hs_ref fading to 0 at the next measured HS setting, so
            # every LS-sweep curve AND every other HS-sweep curve is
            # reproduced exactly; on consistent data D = 0 and this is the
            # plain formula of the module docstring.
            closed = ls_tab[i_closed[0]]
            self.bleed_tab = [t - closed for t in ls_tab]
            self.dup_tab = closed - stack_ref
        else:
            self.bleed_tab = [t - stack_ref for t in ls_tab]
            self.dup_tab = np.zeros_like(stack_ref)
        others = self.hs_x[~np.isclose(self.hs_x, self.hs_ref)]
        below, above = others[others < self.hs_ref], others[others > self.hs_ref]
        self._dup_lo = float(below.max()) if len(below) else None
        self._dup_hi = float(above.min()) if len(above) else None
        self._cache = {}
        # the dyno velocity range used for the flag: the smallest last
        # measured velocity among the curves of this direction
        self.v_dyno_max = min(c.v_max for c in self.ls + self.hs)

    @staticmethod
    def _mix(x, xs, tabs):
        i, j, w = _bracket(x, xs)
        return tabs[i] if w == 0. else (1 - w) * tabs[i] + w * tabs[j]

    def _stack_tab(self, hs):
        """v_stack(F; hs) on F_grid: measured curve at a measured setting,
        else the neighbouring HS curves mixed at fixed velocity (docstring)."""
        i, j, w = _bracket(hs, self.hs_x)
        if w == 0.:
            return self.stack_tab[i]
        if w == 1.:
            return self.stack_tab[j]
        vg = self._stack_v_grid
        F = (1. - w) * self.hs[i].F_of_v(vg) + w * self.hs[j].F_of_v(vg)
        return _lin_ext(self.F_grid, F, vg)

    def _dup_weight(self, hs):
        """1 at hs_ref, linearly 0 at the neighbouring measured HS settings."""
        hs = float(hs)
        if np.isclose(hs, self.hs_ref):
            return 1.
        if hs < self.hs_ref:
            lo = self._dup_lo
            return 0. if lo is None else float(np.clip((hs - lo) / (self.hs_ref - lo), 0., 1.))
        hi = self._dup_hi
        return 0. if hi is None else float(np.clip((hi - hs) / (hi - self.hs_ref), 0., 1.))

    def table(self, ls, hs):
        """v(F) of one setting on F_grid, non-decreasing."""
        key = (float(ls), float(hs))
        tab = self._cache.get(key)
        if tab is None:
            v = (self._mix(ls, self.ls_x, self.bleed_tab) + self._stack_tab(hs)
                 + self._dup_weight(hs) * self.dup_tab)
            v = np.maximum.accumulate(np.maximum(v, 0.))
            tab = v + 1e-12 * np.arange(len(v))     # strictly increasing
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = tab
        return tab

    def velocity(self, F, ls, hs):
        F = np.maximum(np.asarray(F, float), 0.)
        return _lin_ext(F, self.F_grid, self.table(ls, hs))

    def force(self, speed, ls, hs, *, iters=60):
        """Force magnitude at speed magnitudes (m/s) inside the range: bisection
        on the monotone v(F); beyond v_dyno_max the 50 mm/s secant extension."""
        s = np.atleast_1d(np.abs(np.asarray(speed, float)))
        vmax = self.v_dyno_max
        targets = np.r_[np.minimum(s, vmax), vmax, vmax - min(SECANT_SPAN_MPS, vmax)]
        lo = np.zeros_like(targets)
        hi = np.full_like(targets, self.F_grid[-1])
        for _ in range(60):
            short = self.velocity(hi, ls, hs) < targets
            if not short.any():
                break
            hi[short] *= 2.
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.velocity(mid, ls, hs) < targets
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        F = 0.5 * (lo + hi)
        F[targets <= 0.] = 0.
        f_in, f_max, f_back = F[:-2], F[-2], F[-1]
        slope = (f_max - f_back) / min(SECANT_SPAN_MPS, vmax)
        return f_in + np.maximum(s - vmax, 0.) * slope



class _PathDirection(_Direction):
    """Compression or rebound of a valve spec measured along ONE combined
    LS/HS path (module docstring, PATH LAYOUT): v(F) linear in LS clicks
    between the measured settings at fixed force; HS follows the path."""

    def __init__(self, points, label):
        self.label = label
        keys = sorted(points)
        self.ls_x = np.array([k[0] for k in keys], float)
        self.path_hs = np.array([k[1] for k in keys], float)
        if np.any(np.diff(self.ls_x) <= 0):
            raise ValueError(f'{label}: path settings need distinct LS clicks')
        self.hs_x = np.unique(self.path_hs)
        self.ls = [points[k] for k in keys]
        self.hs = []
        self.hs_ref = self.ls_ref = float('nan')
        f_top = max(c.F[-1] for c in self.ls)
        self.F_grid = np.unique(np.r_[np.linspace(0., 1.5 * f_top, 2001),
                                      np.concatenate([c.F for c in self.ls])])
        self.path_tab = [c.v_of_F(self.F_grid) for c in self.ls]
        self._cache = {}
        self.v_dyno_max = min(c.v_max for c in self.ls)

    def hs_on_path(self, ls):
        return float(np.interp(float(ls), self.ls_x, self.path_hs))

    def table(self, ls, hs):
        key = float(ls)
        tab = self._cache.get(key)
        if tab is None:
            v = np.maximum.accumulate(np.maximum(self._mix(ls, self.ls_x, self.path_tab), 0.))
            tab = v + 1e-12 * np.arange(len(v))
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = tab
        return tab


def _setting_value(x, name):
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError(f'damper setting {name} must be a number, got {x!r}')
    if not np.isfinite(x):
        raise ValueError(f'damper setting {name} must be finite')
    return x


class DamperDyno:
    """All valve specs of one damper's dyno sheet (see module docstring)."""

    def __init__(self, data, source=''):
        if not isinstance(data, dict) or not isinstance(data.get('valve_specs'), dict) \
                or not data['valve_specs']:
            raise ValueError('dyno file needs a non-empty "valve_specs" object')
        units = data.get('units', {}) or {}
        if units.get('velocity', 'mm/s') != 'mm/s' or units.get('force', 'N') != 'N':
            raise ValueError(f'dyno units must be mm/s and N, got {units}')
        self.source = str(source)
        self.name = str(data.get('damper', 'damper'))
        self.delivered_valve_spec = data.get('delivered_valve_spec')
        self.adjusters = dict(data.get('adjusters', {}) or {})
        self.raw = data
        self._valves = {}
        self._layout = {}
        for valve, spec in data['valve_specs'].items():
            if isinstance(spec, dict) and 'settings' in spec and 'ls_sweep' not in spec:
                try:
                    dirs = {}
                    for d in DIRECTIONS:
                        pts = {(float(c['ls_clicks']), float(c['hs_turns'])):
                               _Curve(c[d], f'{valve} {k} {d}')
                               for k, c in spec['settings'].items() if c.get(d)}
                        if len(pts) < 1:
                            raise ValueError(f'{valve} {d}: no curves')
                        dirs[d] = _PathDirection(pts, f'{valve} {d}')
                except (KeyError, TypeError, AttributeError) as e:
                    raise ValueError(f'valve spec {valve!r} malformed: {type(e).__name__}: {e}')
                self._valves[str(valve)] = dirs
                self._layout[str(valve)] = 'path'
                continue
            self._layout[str(valve)] = 'sweeps'
            try:
                ls_sw, hs_sw = spec['ls_sweep'], spec['hs_sweep']
                hs_ref = float(ls_sw.get('hs_turns', 4.3))
                ls_ref = float(hs_sw.get('ls_clicks', 0))
                dirs = {}
                for d in DIRECTIONS:
                    ls_c = {float(k): _Curve(c[d], f'{valve} LS {k} {d}')
                            for k, c in ls_sw['curves'].items() if c.get(d)}
                    hs_c = {float(k): _Curve(c[d], f'{valve} HS {k} {d}')
                            for k, c in hs_sw['curves'].items() if c.get(d)}
                    if len(ls_c) < 1 or len(hs_c) < 1:
                        raise ValueError(f'{valve} {d}: both sweeps need curves')
                    dirs[d] = _Direction(ls_c, hs_c, hs_ref, ls_ref, f'{valve} {d}')
                self._valves[str(valve)] = dirs
            except (KeyError, TypeError, AttributeError) as e:
                raise ValueError(f'valve spec {valve!r} malformed: {type(e).__name__}: {e}')

    @classmethod
    def from_file(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f), source=path)

    # ── catalogue ──────────────────────────────────────────────────────
    @property
    def valve_specs(self):
        return list(self._valves)

    def layout(self, valve):
        """'sweeps' (separate LS and HS sweeps, parallel flow) or 'path'
        (combined LS/HS settings; HS follows the path)."""
        self._dir(valve, 'comp')
        return self._layout[valve]

    def ls_range(self, valve, direction='comp'):
        d = self._dir(valve, direction)
        return float(d.ls_x[0]), float(d.ls_x[-1])

    def hs_range(self, valve, direction='comp'):
        d = self._dir(valve, direction)
        return float(d.hs_x[0]), float(d.hs_x[-1])

    def _dir(self, valve, direction):
        if valve not in self._valves:
            raise ValueError(f'valve spec {valve!r} not in the dyno file '
                             f'(has {", ".join(self._valves)})')
        return self._valves[valve][direction]

    def normalize_setting(self, setting):
        """{valve, lsc, hsc, lsr, hsr} with adjusters clamped to the measured
        range; 'clamped' lists the ones that were outside."""
        s = dict(setting or {})
        valve = s.get('valve') or self.delivered_valve_spec or self.valve_specs[0]
        out = {'valve': str(valve), 'clamped': []}
        for key, d, kind in (('lsc', 'comp', 'ls'), ('hsc', 'comp', 'hs'),
                             ('lsr', 'reb', 'ls'), ('hsr', 'reb', 'hs')):
            if key not in s:
                raise ValueError(f'damper setting needs {key}')
            x = _setting_value(s[key], key)
            lo, hi = (self.ls_range if kind == 'ls' else self.hs_range)(out['valve'], d)
            if x < lo or x > hi:
                out['clamped'].append(key)
                x = min(max(x, lo), hi)
            out[key] = x
        if self._layout.get(out['valve']) == 'path':
            out['off_path'] = []
            for ls_key, hs_key, d in (('lsc', 'hsc', 'comp'), ('lsr', 'hsr', 'reb')):
                hs = self._dir(out['valve'], d).hs_on_path(out[ls_key])
                if abs(out[hs_key] - hs) > 5e-3:
                    out['off_path'].append(hs_key)
                out[hs_key] = hs
        return out

    def _adjusters(self, setting, direction):
        s = self.normalize_setting(setting)
        d = self._dir(s['valve'], direction)
        return d, (s['lsc'], s['hsc']) if direction == 'comp' else (s['lsr'], s['hsr'])

    # ── force law ──────────────────────────────────────────────────────
    def velocity(self, F_N, setting, direction):
        """Damper velocity magnitude (m/s) at force magnitude F (N) — the
        superposed v(F; ls, hs)."""
        d, (ls, hs) = self._adjusters(setting, direction)
        return d.velocity(F_N, ls, hs)

    def dyno_max_velocity(self, setting, direction='comp'):
        d, _ = self._adjusters(setting, direction)
        return float(d.v_dyno_max)

    def force(self, v_mps, setting):
        """Signed damper force (N) at damper velocity v (m/s); v > 0 is
        compression -> F > 0, v < 0 rebound -> F < 0."""
        v = np.asarray(v_mps, float)
        flat = np.atleast_1d(v).ravel()
        out = np.zeros_like(flat)
        for direction, sign, mask in (('comp', 1., flat > 0), ('reb', -1., flat < 0)):
            if mask.any():
                d, (ls, hs) = self._adjusters(setting, direction)
                out[mask] = sign * d.force(np.abs(flat[mask]), ls, hs)
        return float(out[0]) if v.ndim == 0 else out.reshape(v.shape)

    def beyond_dyno(self, v_mps, setting):
        """Largest |v| beyond the measured range (m/s, 0 = inside), per direction."""
        v = np.atleast_1d(np.asarray(v_mps, float))
        out = {}
        for direction, part in (('comp', v[v > 0]), ('reb', -v[v < 0])):
            vmax = self.dyno_max_velocity(setting, direction)
            out[direction] = float(max(0., part.max() - vmax)) if len(part) else 0.
        return out

    def curve(self, setting, n=1001):
        """Tabulated F(v) of one setting for fast repeated evaluation."""
        return DamperCurve(self, setting, n=n)

    def equivalent_linear_c(self, setting, v_rms_mps):
        """Statistical linearisation, secant form: c = F(v)/v at the RMS
        damper velocity, compression and rebound separately and averaged
        (N·s/m, damper frame)."""
        v = float(v_rms_mps)
        if not np.isfinite(v) or v <= 0:
            raise ValueError('RMS velocity must be positive')
        comp = self.force(v, setting) / v
        reb = -self.force(-v, setting) / v
        flag = self.beyond_dyno([v, -v], setting)
        return {'c_Nspm': 0.5 * (comp + reb), 'comp_Nspm': comp, 'reb_Nspm': reb,
                'v_rms_mps': v, 'beyond_dyno_mps': max(flag.values())}

    # ── data checks ────────────────────────────────────────────────────
    def measured_curves(self, valve):
        """[(sweep, key, setting, direction, v_mps, F_N)] of one valve spec —
        each measured curve with the full setting it was measured at."""
        out = []
        spec = self.raw['valve_specs'][valve]
        if self._layout.get(valve) == 'path':
            for key, c in spec['settings'].items():
                ls, hs = float(c['ls_clicks']), float(c['hs_turns'])
                setting = {'valve': valve, 'lsc': ls, 'hsc': hs, 'lsr': ls, 'hsr': hs}
                for d in DIRECTIONS:
                    if c.get(d):
                        a = np.asarray(c[d], float)
                        out.append(('path', key, setting, d, a[:, 0] * MM, np.abs(a[:, 1])))
            return out
        hs_ref = float(spec['ls_sweep'].get('hs_turns', 4.3))
        ls_ref = float(spec['hs_sweep'].get('ls_clicks', 0))
        for sweep, sub in (('ls', spec['ls_sweep']), ('hs', spec['hs_sweep'])):
            for key, c in sub['curves'].items():
                x = float(key)
                ls, hs = (x, hs_ref) if sweep == 'ls' else (ls_ref, x)
                setting = {'valve': valve, 'lsc': ls, 'hsc': hs, 'lsr': ls, 'hsr': hs}
                for d in DIRECTIONS:
                    if c.get(d):
                        a = np.asarray(c[d], float)
                        out.append((sweep, key, setting, d, a[:, 0] * MM, np.abs(a[:, 1])))
        return out

    def consistency(self, valve):
        """Max |difference| between the two dyno curves that should be the
        same physical setting (LS sweep at ls_ref vs HS sweep at hs_ref),
        relative to the larger force, per direction."""
        out = {}
        for d in DIRECTIONS:
            dd = self._dir(valve, d)
            i_ls = np.flatnonzero(np.isclose(dd.ls_x, dd.ls_ref))
            i_hs = np.flatnonzero(np.isclose(dd.hs_x, dd.hs_ref))
            if not len(i_ls) or not len(i_hs):
                out[d] = float('nan'); continue
            a, b = dd.ls[i_ls[0]], dd.hs[i_hs[0]]
            v = np.linspace(0., min(a.v_max, b.v_max), 101)[1:]
            fa, fb = a.F_of_v(v), b.F_of_v(v)
            out[d] = float(np.max(np.abs(fa - fb) / np.maximum(np.maximum(fa, fb), 1e-9)))
        return out



# ─────────────────────────────────────────────────────────────────────────────
#  Every published curve, recreated by the model
# ─────────────────────────────────────────────────────────────────────────────
CHECK_V_MIN_MPS = 0.025      # error statistics from 25 mm/s up (digitising
                             # error ~±3 N there; ±20 N hysteresis below)


def sheet_curves(raw):
    """Every curve printed on a dyno sheet (raw JSON dict), any layout:
    [{valve, group, key, ls, hs, comp, reb}] with group 'LS sweep',
    'HS sweep', 'path' or 'comparison' (the comparison page re-plots one
    setting of several valve specs).  comp / reb = [[v mm/s, F N], ...]."""
    out = []
    for valve, spec in raw.get('valve_specs', {}).items():
        if 'settings' in spec and 'ls_sweep' not in spec:
            for key, c in spec['settings'].items():
                out.append(dict(valve=valve, group='path', key=key, ls=float(c['ls_clicks']),
                                hs=float(c['hs_turns']), comp=c.get('comp'), reb=c.get('reb')))
            continue
        hs_ref = float(spec['ls_sweep'].get('hs_turns', 4.3))
        ls_ref = float(spec['hs_sweep'].get('ls_clicks', 0))
        for key, c in spec['ls_sweep']['curves'].items():
            out.append(dict(valve=valve, group='LS sweep', key=key, ls=float(key), hs=hs_ref,
                            comp=c.get('comp'), reb=c.get('reb')))
        for key, c in spec['hs_sweep']['curves'].items():
            out.append(dict(valve=valve, group='HS sweep', key=key, ls=ls_ref, hs=float(key),
                            comp=c.get('comp'), reb=c.get('reb')))
    for key, c in (raw.get('comparison_page') or {}).items():
        valve = key.split(' ')[0]
        out.append(dict(valve=valve, group='comparison', key=key, ls=float(c['ls_clicks']),
                        hs=float(c['hs_turns']), comp=c.get('comp'), reb=c.get('reb')))
    return out


def compare_curve(dyno, curve, v_min_mps=CHECK_V_MIN_MPS):
    """The model's F(v) at the digitised velocities of one sheet curve
    (sheet_curves item) at that curve's setting.  Per direction:
    v_mps, F_meas_N, F_model_N (magnitudes) and, from v_min up, rms_N,
    max_abs_N, max_rel (fraction of the measured force).  'clamped' /
    'off_path' say when the model could not sit at the printed setting —
    such a curve is then NOT a like-for-like check."""
    setting = dyno.normalize_setting({'valve': curve['valve'], 'lsc': curve['ls'], 'hsc': curve['hs'],
                                      'lsr': curve['ls'], 'hsr': curve['hs']})
    rec = dict(valve=curve['valve'], group=curve['group'], key=curve['key'], ls=curve['ls'],
               hs=curve['hs'], clamped=list(setting.get('clamped', [])),
               off_path=list(setting.get('off_path', [])), setting=setting)
    for d, sign in (('comp', 1.), ('reb', -1.)):
        pts = curve.get(d)
        if not pts:
            continue
        a = np.asarray(pts, float)
        a = a[a[:, 0] > 0]
        v, Fm = a[:, 0] * MM, np.abs(a[:, 1])
        Fp = sign * dyno.force(sign * v, setting)
        sel = v >= v_min_mps - 1e-12
        e = Fp[sel] - Fm[sel]
        rec[d] = dict(v_mps=v, F_meas_N=Fm, F_model_N=Fp,
                      rms_N=float(np.sqrt(np.mean(e ** 2))) if sel.any() else float('nan'),
                      max_abs_N=float(np.max(np.abs(e))) if sel.any() else float('nan'),
                      max_rel=float(np.max(np.abs(e) / np.maximum(Fm[sel], 1e-9))) if sel.any() else float('nan'))
    return rec


def reproduce_sheet(dyno, raw=None, v_min_mps=CHECK_V_MIN_MPS):
    """compare_curve for every curve of a sheet against a model.  raw = the
    model's own sheet (default) -> the reproduction check; another sheet ->
    the cross-sheet check (valve specs the model does not have are skipped)."""
    raw = dyno.raw if raw is None else raw
    return [compare_curve(dyno, c, v_min_mps) for c in sheet_curves(raw)
            if c['valve'] in dyno.valve_specs]


def leave_one_out(raw, v_min_mps=CHECK_V_MIN_MPS):
    """Interpolation honesty: every INTERIOR curve (not the end of a sweep
    or path, not the shared reference setting) is removed from its sweep and
    predicted by the model built from the rest of the sheet."""
    import copy
    out = []
    for c in sheet_curves(raw):
        if c['group'] == 'comparison':
            continue
        spec = raw['valve_specs'][c['valve']]
        if c['group'] == 'path':
            tab = spec['settings']; xs = sorted(float(v['ls_clicks']) for v in tab.values()); x = c['ls']
        else:
            sw = spec['ls_sweep' if c['group'] == 'LS sweep' else 'hs_sweep']
            tab = sw['curves']; xs = sorted(float(k) for k in tab); x = float(c['key'])
            ref = float(spec['hs_sweep'].get('ls_clicks', 0)) if c['group'] == 'LS sweep' \
                else float(spec['ls_sweep'].get('hs_turns', 4.3))
            if np.isclose(x, ref):
                continue
        if len(xs) < 3 or np.isclose(x, xs[0]) or np.isclose(x, xs[-1]):
            continue
        sub = copy.deepcopy({k: v for k, v in raw.items() if k != 'comparison_page'})
        sub['valve_specs'] = {c['valve']: copy.deepcopy(spec)}
        held = sub['valve_specs'][c['valve']]
        (held['settings'] if c['group'] == 'path' else
         held['ls_sweep' if c['group'] == 'LS sweep' else 'hs_sweep']['curves']).pop(c['key'])
        rec = compare_curve(DamperDyno(sub), c, v_min_mps)
        rec['held_out'] = True
        out.append(rec)
    return out


def check_summary(records, direction=None):
    """Worst rms / abs / rel over records (like-for-like ones only)."""
    rms, ab, rel = [], [], []
    for r in records:
        if r['clamped'] or r['off_path']:
            continue
        for d in ((direction,) if direction else DIRECTIONS):
            if d in r and np.isfinite(r[d]['max_abs_N']):
                rms.append(r[d]['rms_N']); ab.append(r[d]['max_abs_N']); rel.append(r[d]['max_rel'])
    if not ab:
        return dict(n=0, rms_N=float('nan'), max_abs_N=float('nan'), max_rel=float('nan'))
    return dict(n=len(ab), rms_N=float(np.sqrt(np.mean(np.square(rms)))),
                max_abs_N=float(max(ab)), max_rel=float(max(rel)))


class DamperCurve:
    """F(v) of one setting tabulated in the damper frame (signed), with the
    same linear extension beyond the dyno range as DamperDyno.force."""

    def __init__(self, dyno, setting, n=1001):
        self.setting = dyno.normalize_setting(setting)
        self.vmax_c = dyno.dyno_max_velocity(self.setting, 'comp')
        self.vmax_r = dyno.dyno_max_velocity(self.setting, 'reb')
        vc = np.linspace(0., self.vmax_c, n)
        vr = np.linspace(0., self.vmax_r, n)
        fc = dyno.force(vc, self.setting)
        fr = dyno.force(-vr, self.setting)
        span_c, span_r = min(SECANT_SPAN_MPS, self.vmax_c), min(SECANT_SPAN_MPS, self.vmax_r)
        self.slope_c = float((fc[-1] - dyno.force(self.vmax_c - span_c, self.setting)) / span_c)
        self.slope_r = float((-fr[-1] + dyno.force(-(self.vmax_r - span_r), self.setting)) / span_r)
        self.v = np.r_[-vr[::-1], vc[1:]]
        self.F = np.r_[fr[::-1], fc[1:]]
        slopes = np.diff(self.F) / np.diff(self.v)
        self.slope_max = float(max(slopes.max(), self.slope_c, self.slope_r))

    def __call__(self, v):
        v = np.asarray(v, float)
        return (np.interp(v, self.v, self.F)
                + np.maximum(v - self.vmax_c, 0.) * self.slope_c
                + np.minimum(v + self.vmax_r, 0.) * self.slope_r)

    def beyond_dyno_mps(self, v):
        v = np.asarray(v, float)
        if v.size == 0:
            return 0.
        return float(max(0., v.max() - self.vmax_c, -v.min() - self.vmax_r))


class WheelDampers:
    """The four corner dampers seen AT THE WHEEL (FL FR RL RR):
    F_wheel(v_wheel) = F_damper(MR * v_wheel) * MR with the solved MRs.
    Positive wheel velocity = bump (compression)."""

    def __init__(self, dyno, setting, motion_ratios):
        mr = np.asarray(motion_ratios, float)
        if mr.shape != (4,) or not np.all(np.isfinite(mr)) or np.any(mr <= 0):
            raise ValueError('four positive motion ratios (FL FR RL RR) required')
        self.dyno = dyno
        self.mr = mr
        self.curve = dyno.curve(setting)
        self.setting = self.curve.setting

    def __call__(self, v_wheel_mps):
        """Wheel-frame forces for wheel velocities (..., 4)."""
        v = np.asarray(v_wheel_mps, float)
        return self.curve(v * self.mr) * self.mr

    @property
    def max_wheel_slope_Nspm(self):
        """Largest wheel-frame dF/dv per corner (integrator step sizing)."""
        return self.curve.slope_max * self.mr ** 2

    def equivalent_wheel_c(self, damper_v_rms_mps):
        """(c_avg, c_comp, c_reb) wheel-frame N·s/m per corner at per-corner
        RMS DAMPER velocities (secant linearisation, x MR²)."""
        vr = np.broadcast_to(np.asarray(damper_v_rms_mps, float), (4,))
        if np.any(~np.isfinite(vr)) or np.any(vr <= 0):
            raise ValueError('positive RMS damper velocities required')
        comp = self.curve(vr) / vr
        reb = -self.curve(-vr) / vr
        m2 = self.mr ** 2
        return 0.5 * (comp + reb) * m2, comp * m2, reb * m2

    def beyond_dyno_mps(self, v_wheel_mps):
        return self.curve.beyond_dyno_mps(np.asarray(v_wheel_mps, float) * self.mr)

    def roll_moment_fn(self, track_front_m, track_rear_m):
        """Body roll damping moment M(phi_dot) (N·m, same sign as phi_dot) in
        pure roll: per axle the bump-side wheel moves at +t/2·phi_dot and the
        rebound side at -t/2·phi_dot, M = t/2·(F_w(+s) - F_w(-s))."""
        tf, tr = float(track_front_m), float(track_rear_m)
        mr_f, mr_r = float(self.mr[0]), float(self.mr[2])
        curve = self.curve

        def moment(phi_dot):
            p = np.asarray(phi_dot, float)
            m = 0.
            for t, mr in ((tf, mr_f), (tr, mr_r)):
                s = 0.5 * t * p * mr
                m = m + 0.5 * t * mr * (curve(s) - curve(-s))
            return float(m) if p.ndim == 0 else m

        moment.equivalent_c = lambda rate: (moment(float(rate)) / float(rate))
        return moment


def damping_ratios(c_wheel_front, c_wheel_rear, *, sprung_corner_mass_front_kg,
                   sprung_corner_mass_rear_kg, unsprung_corner_mass_front_kg,
                   unsprung_corner_mass_rear_kg, ride_rate_front_Npm, ride_rate_rear_Npm,
                   wheel_rate_front_Npm, wheel_rate_rear_Npm, tire_rate_Npm,
                   track_front_m, track_rear_m, roll_stiffness_Nm_per_rad, roll_inertia_kgm2):
    """Damping ratios of the uncoupled modes for wheel-frame coefficients:
      ride (body)  zeta = c / (2 sqrt(K_ride m_s,corner))   (ride rate = wheel
                   rate in series with the tyre, as ride_frequency_Hz uses)
      wheel hop    zeta = c / (2 sqrt((k_wheel + k_tyre) m_u,corner))
      roll         zeta = C_phi / (2 sqrt(K_phi I_xx)),  C_phi = sum c t²/2
                   (both wheels of the axle at t/2·phi_dot), K_phi the
                   model's suspension roll stiffness (springs + ARB)."""
    out = {}
    for axle, c, ms, mu, kr, kw in (
            ('front', c_wheel_front, sprung_corner_mass_front_kg, unsprung_corner_mass_front_kg,
             ride_rate_front_Npm, wheel_rate_front_Npm),
            ('rear', c_wheel_rear, sprung_corner_mass_rear_kg, unsprung_corner_mass_rear_kg,
             ride_rate_rear_Npm, wheel_rate_rear_Npm)):
        out[f'ride_{axle}'] = float(c) / (2. * np.sqrt(float(kr) * float(ms)))
        out[f'hop_{axle}'] = float(c) / (2. * np.sqrt((float(kw) + float(tire_rate_Npm)) * float(mu)))
    c_phi = (float(c_wheel_front) * float(track_front_m) ** 2 / 2.
             + float(c_wheel_rear) * float(track_rear_m) ** 2 / 2.)
    out['roll_damping_Nms_rad'] = c_phi
    out['roll'] = c_phi / (2. * np.sqrt(float(roll_stiffness_Nm_per_rad) * float(roll_inertia_kgm2)))
    return out


def vehicle_damping_ratios(veh, c_wheel_front, c_wheel_rear, roll_inertia_kgm2):
    """damping_ratios on a VehicleParams (masses split as the ride model does)."""
    from vahan.ride_solve import sprung_corner_masses_kg
    mf, mr = sprung_corner_masses_kg(veh)
    return damping_ratios(
        c_wheel_front, c_wheel_rear,
        sprung_corner_mass_front_kg=mf, sprung_corner_mass_rear_kg=mr,
        unsprung_corner_mass_front_kg=veh.unsprung_mass_front_kg / 2.,
        unsprung_corner_mass_rear_kg=veh.unsprung_mass_rear_kg / 2.,
        ride_rate_front_Npm=veh.ride_rate_front_Npm, ride_rate_rear_Npm=veh.ride_rate_rear_Npm,
        wheel_rate_front_Npm=veh.wheel_rate_front_Npm, wheel_rate_rear_Npm=veh.wheel_rate_rear_Npm,
        tire_rate_Npm=veh.tire_rate_Npm, track_front_m=veh.front_track_m,
        track_rear_m=veh.rear_track_m, roll_stiffness_Nm_per_rad=veh.roll_stiffness_total_Npm_rad,
        roll_inertia_kgm2=roll_inertia_kgm2)


# ─────────────────────────────────────────────────────────────────────────────
#  Project hookup: car['damper']
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProjectDamper:
    """The project's damper: the loaded dyno sheet + the declared setting."""
    dyno: DamperDyno
    setting: dict            # normalised {valve, lsc, hsc, lsr, hsr, clamped}
    dyno_file: str           # as declared (repo-relative)
    path: str                # resolved

    def wheel(self, motion_ratio_front, motion_ratio_rear):
        mf, mr = float(motion_ratio_front), float(motion_ratio_rear)
        return WheelDampers(self.dyno, self.setting, [mf, mf, mr, mr])

    def wheel_for_vehicle(self, veh):
        return self.wheel(veh.motion_ratio_front, veh.motion_ratio_rear)

    @property
    def label(self):
        s = self.setting
        return (f'{self.dyno.name} {s["valve"]}  LSC {s["lsc"]:g} / HSC {s["hsc"]:g}  '
                f'LSR {s["lsr"]:g} / HSR {s["hsr"]:g}')


_DYNO_CACHE = {}


def load_dyno(path):
    """DamperDyno for a resolved path, cached on (path, mtime)."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    dyno = _DYNO_CACHE.get(key)
    if dyno is None:
        dyno = DamperDyno.from_file(path)
        _DYNO_CACHE.clear()
        _DYNO_CACHE[key] = dyno
    return dyno


def project_damper_from_car(car, project_path=None):
    """ProjectDamper from car['damper'] or None when absent / not 'dyno'.

    car['damper'] = {"model": "dyno", "dyno_file": "<repo-relative path>",
                     "valve": "C12/R12", "lsc": int, "hsc": float,
                     "lsr": int, "hsr": float}
    Any other model value (e.g. "linear", kept so a toggle does not lose the
    adjuster values) or no block = the assumed linear coefficients.
    Raises ValueError when the block says dyno but cannot be honoured."""
    block = (car or {}).get('damper')
    if not isinstance(block, dict) or block.get('model') != 'dyno':
        return None
    path = resolve_dyno_path(block.get('dyno_file'), project_path)
    if not os.path.isfile(path):
        raise ValueError(f'damper dyno file not found: {block.get("dyno_file")!r}')
    dyno = load_dyno(path)
    setting = dyno.normalize_setting(block)
    return ProjectDamper(dyno, setting, str(block.get('dyno_file')), path)
