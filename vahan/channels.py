"""Plot-anything channel registry (user 2026-11-10: "a list of every measurable variable and
measured-against variable ... so I can view anything against anything ... with adjustable units").

Three data sources, each the ONE model's own output — nothing here re-models anything:
  kin  one kinematic sweep from MainWindow._compute_sweep for one motion (heave / roll / pitch /
       steer): the per-corner metric arrays of vahan.metrics_catalog.CATALOG, plus wheel rate /
       ride rate / ride frequency / spring roll stiffness derived from those metrics with the
       SAME VehicleParams rates the dynamics solver uses
  dyn  the SteadyStateSolver sweep dict the Dynamics panel produces (lateral g, longitudinal g
       or speed sweep) plus what MainWindow._attach_dyn_extras / _attach_wheel_rate /
       _attach_steering_effort add to it
  lap  a vahan.laptime.LapResult with its Track (every per-station channel, the detail-pass
       channels resampled onto the lap distance)

A Channel is a label, a unit FAMILY and a getter: getter(data) -> {series name: array in the
family's base unit}, every series on the source's own station grid, so any channel can be the X
axis of any other from the same source.  Units: UNITS lists each family's display units with the
factor FROM the base unit; convert() applies it.  Pure numpy, no Qt — gui/free_plot.py lists,
evaluates and draws.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

CORNERS = ('FL', 'FR', 'RL', 'RR')

# family -> (base unit, {display unit: factor from base})
UNITS: Dict[str, tuple] = {
    'angle':      ('°',       {'°': 1.0, 'rad': math.pi / 180.0, 'mrad': math.pi / 180.0 * 1000.0, "arcmin": 60.0}),
    'length':     ('mm',      {'mm': 1.0, 'in': 1.0 / 25.4, 'cm': 0.1, 'm': 1e-3}),
    'distance':   ('m',       {'m': 1.0, 'ft': 3.280839895, 'km': 1e-3, 'mi': 1.0 / 1609.344}),
    'force':      ('N',       {'N': 1.0, 'lbf': 0.2248089431, 'kgf': 0.1019716213, 'kN': 1e-3}),
    'speed':      ('m/s',     {'km/h': 3.6, 'mph': 2.2369362921, 'm/s': 1.0, 'ft/s': 3.280839895}),
    'stiffness':  ('N/mm',    {'N/mm': 1.0, 'lbf/in': 5.710147155, 'N/m': 1000.0, 'kN/m': 1.0}),
    'roll_stiff': ('N·m/deg', {'N·m/deg': 1.0, 'lbf·ft/deg': 0.7375621493, 'N·m/rad': 57.29577951,
                               'kN·m/rad': 0.05729577951}),
    'torque':     ('N·m',     {'N·m': 1.0, 'lbf·ft': 0.7375621493, 'lbf·in': 8.850745792}),
    'frequency':  ('Hz',      {'Hz': 1.0, 'cpm': 60.0, 'rad/s': 2.0 * math.pi}),
    'accel':      ('g',       {'g': 1.0, 'm/s²': 9.80665, 'ft/s²': 32.17404856}),
    'percent':    ('%',       {'%': 1.0, 'fraction': 0.01}),
    'ratio':      ('-',       {'-': 1.0}),
    'power':      ('W',       {'kW': 1e-3, 'hp': 1.0 / 745.6998716, 'W': 1.0}),
    'time':       ('s',       {'s': 1.0, 'ms': 1000.0, 'min': 1.0 / 60.0}),
    'rpm':        ('rpm',     {'rpm': 1.0, 'rad/s': 2.0 * math.pi / 60.0, 'Hz': 1.0 / 60.0}),
    'deg_per_g':  ('°/g',     {'°/g': 1.0, 'rad/g': math.pi / 180.0}),
    'deg_per_mm': ('°/mm',    {'°/mm': 1.0, '°/in': 25.4, 'rad/m': math.pi / 180.0 * 1000.0}),
    'curvature':  ('1/m',     {'1/m': 1.0, '1/ft': 0.3048}),
    'count':      ('-',       {'-': 1.0}),
}

_CATALOG_UNIT_FAMILY = {'°': 'angle', 'mm': 'length', '%': 'percent', '-': 'ratio', 'm': 'distance',
                        'N': 'force', '°/mm': 'deg_per_mm', '°/mm': 'deg_per_mm', '°': 'angle'}


def unit_family(unit: str) -> str:
    """Family for a CATALOG-style unit string ('°', 'mm', '%', '-', 'm', 'N', '°/mm')."""
    return _CATALOG_UNIT_FAMILY.get(str(unit).strip(), 'ratio')


def base_unit(family: str) -> str:
    return UNITS[family][0]


def units_for(family: str) -> List[str]:
    return list(UNITS[family][1].keys())


def unit_factor(family: str, unit: str) -> float:
    """Multiply a base-unit value by this to show it in `unit`."""
    fam = UNITS[family][1]
    return float(fam.get(unit, 1.0))


def convert(arr, family: str, unit: str):
    a = np.asarray(arr, float)
    return a * unit_factor(family, unit)


@dataclass
class Channel:
    key: str
    label: str
    family: str
    source: str                       # 'kin' | 'dyn' | 'lap'
    category: str = ''
    scope: str = 'vehicle'            # 'corner' | 'axle' | 'front' | 'rear' | 'vehicle'
    getter: Optional[Callable] = None  # getter(data) -> {series: ndarray (base units)}
    note: str = ''

    @property
    def unit(self) -> str:
        return base_unit(self.family)

    def units(self) -> List[str]:
        return units_for(self.family)

    def series(self, data) -> Dict[str, np.ndarray]:
        """{series name: base-unit array}; {} when the data does not carry it."""
        if self.getter is None or data is None:
            return {}
        try:
            out = self.getter(data)
        except Exception:
            return {}
        res = {}
        for k, v in (out or {}).items():
            if v is None:
                continue
            a = np.asarray(v, float).ravel()
            if a.size:
                res[str(k)] = a
        return res


# ─────────────────────────────────────────────────────────────────────────────
#  KINEMATIC SWEEP  (data = MainWindow._compute_sweep result + rate extras)
#     data keys: motion, x_arr, x_label, sweep_results {corner: {metric: arr}},
#                spring_rate_Nmm {'F','R'}, tire_rate_Nmm, corner_mass_kg {'F','R'},
#                track_m {'F','R'}
# ─────────────────────────────────────────────────────────────────────────────
KIN_MOTIONS = [
    ('heave', 'Wheel travel, heave (+ bump, − droop)', 'length', (-50.0, 50.0)),
    ('roll',  'Roll angle (chassis, + = left side down)', 'angle', (-5.0, 5.0)),
    ('pitch', 'Pitch travel (+ = nose down / tail up)', 'length', (-30.0, 30.0)),
    ('steer', 'Steering wheel angle', 'angle', (-360.0, 360.0)),
]
KIN_MOTION_DEFAULTS = {m: rng for m, _l, _f, rng in KIN_MOTIONS}


def _corner_series(d, key, scope):
    sr = d.get('sweep_results') or {}
    out = {}
    if scope == 'axle':
        for ax_, c in (('F', 'FL'), ('R', 'RL')):
            if c in sr and key in sr[c]:
                out[ax_] = sr[c][key]
    elif scope == 'front':
        for c in ('FL', 'FR'):
            if c in sr and key in sr[c]:
                out[c] = sr[c][key]
    elif scope == 'rear':
        for c in ('RL', 'RR'):
            if c in sr and key in sr[c]:
                out[c] = sr[c][key]
    elif scope == 'vehicle':
        for c in CORNERS:
            if c in sr and key in sr[c]:
                out[''] = sr[c][key]
                break
    else:
        for c in CORNERS:
            if c in sr and key in sr[c]:
                out[c] = sr[c][key]
    return out


def _kin_wheel_rate(d):
    """spring wheel rate k_spring x MR^2 (N/mm) per corner from the swept motion ratio
    (damper/wheel) and the dynamics VehicleParams spring rates."""
    k = d.get('spring_rate_Nmm') or {}
    out = {}
    for c, mr in _corner_series(d, 'motion_ratio', 'corner').items():
        ks = k.get(c[0])
        if ks is None:
            continue
        out[c] = float(ks) * np.asarray(mr, float) ** 2
    return out


def _kin_ride_rate(d):
    kt = d.get('tire_rate_Nmm')
    if kt is None:
        return {}
    out = {}
    for c, wr in _kin_wheel_rate(d).items():
        with np.errstate(divide='ignore', invalid='ignore'):
            out[c] = wr * float(kt) / (wr + float(kt))
    return out


def _kin_ride_freq(d):
    m = d.get('corner_mass_kg') or {}
    out = {}
    for c, rr in _kin_ride_rate(d).items():
        mc = m.get(c[0])
        if not mc:
            continue
        out[c] = np.sqrt(np.maximum(rr * 1000.0, 0.0) / float(mc)) / (2.0 * np.pi)
    return out


def _kin_roll_stiff_springs(d):
    """spring part of each axle's roll stiffness at the swept travel: mean(L,R wheel rate) x t^2/2
    (N·m/deg) — the same expression MainWindow._attach_dyn_extras uses."""
    t = d.get('track_m') or {}
    wr = _kin_wheel_rate(d)
    out = {}
    for ax_ in ('F', 'R'):
        l_, r_ = wr.get(ax_ + 'L'), wr.get(ax_ + 'R')
        tm = t.get(ax_)
        if l_ is None or r_ is None or not tm:
            continue
        out[ax_] = 0.5 * (l_ + r_) * 1000.0 * float(tm) ** 2 / 2.0 * math.pi / 180.0
    return out


def kin_channels(topology_flavors=('standard',)) -> List[Channel]:
    """Every kinematic-sweep channel: the four motion inputs (what is swept) + every CATALOG metric
    the active topology has + the derived rate channels."""
    from vahan.metrics_catalog import CATALOG
    flav = set(topology_flavors or ('standard',))
    out = []
    for m, label, fam, _rng in KIN_MOTIONS:
        def _g(d, m=m):
            return {'': d['x_arr']} if d.get('motion') == m else {}
        out.append(Channel(f'motion:{m}', label, fam, 'kin', 'Sweep input', 'vehicle', _g,
                           note='the swept input itself'))
    for e in CATALOG:
        topos = e.get('topologies')
        if topos is not None and not (topos & flav):
            continue
        key, scope = e['key'], e.get('scope', 'corner')
        out.append(Channel(key, e['label'], unit_family(e['unit']), 'kin', e.get('category', ''), scope,
                           lambda d, key=key, scope=scope: _corner_series(d, key, scope)))
    out += [
        Channel('wheel_rate', 'Wheel rate (spring)', 'stiffness', 'kin', 'Rates', 'corner', _kin_wheel_rate,
                note='spring rate from the dynamics car × the swept motion ratio squared'),
        Channel('ride_rate', 'Ride rate (wheel rate in series with the tyre)', 'stiffness', 'kin', 'Rates', 'corner', _kin_ride_rate),
        Channel('ride_freq', 'Ride frequency', 'frequency', 'kin', 'Rates', 'corner', _kin_ride_freq),
        Channel('roll_stiff_springs', 'Roll stiffness from the springs (per axle, at this travel)', 'roll_stiff', 'kin',
                'Rates', 'axle', _kin_roll_stiff_springs),
    ]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMICS SWEEP  (data = the sweep dict)
# ─────────────────────────────────────────────────────────────────────────────
_DYN_SCALAR = {
    'lateral_g':               ('Lateral acceleration', 'accel', 'Sweep input'),
    'longitudinal_g':          ('Longitudinal acceleration (+ accelerating)', 'accel', 'Sweep input'),
    'speed_kph':               ('Speed', 'speed', 'Sweep input'),
    'lon_g_applied':           ('Longitudinal acceleration held (after the grip limit)', 'accel', 'Sweep input'),
    'lat_g_applied':           ('Lateral acceleration applied', 'accel', 'Sweep input'),
    'roll_angle_deg':          ('Roll angle', 'angle', 'Body'),
    'roll_front_deg':          ('Roll angle at the front axle', 'angle', 'Body'),
    'roll_rear_deg':           ('Roll angle at the rear axle', 'angle', 'Body'),
    'chassis_twist_deg':       ('Chassis twist (front roll − rear roll)', 'angle', 'Body'),
    'pitch_angle_deg':         ('Pitch angle (+ nose down)', 'angle', 'Body'),
    'understeer_gradient_deg': ('Understeer (extra steer needed beyond Ackermann)', 'angle', 'Handling'),
    'turn_radius_m':           ('Turn radius', 'distance', 'Handling'),
    'yaw_rate_degps':          ('Yaw rate', 'angle', 'Handling'),
    'steer_angle_deg':         ('Road-wheel steer angle', 'angle', 'Steering'),
    'steering_wheel_deg':      ('Steering wheel angle', 'angle', 'Steering'),
    'steer_mean_deg':          ('Front steer the turn needs (mean of both wheels, into the turn)', 'angle', 'Steering'),
    'steer_inner_deg':         ('Inside front wheel steer (into the turn)', 'angle', 'Steering'),
    'steer_outer_deg':         ('Outside front wheel steer (into the turn)', 'angle', 'Steering'),
    'steer_rack_mm':           ('Rack travel', 'length', 'Steering'),
    'steer_FL_deg':            ('Front-left wheel steer (into the turn)', 'angle', 'Steering'),
    'steer_FR_deg':            ('Front-right wheel steer (into the turn)', 'angle', 'Steering'),
    'downforce_N':             ('Downforce', 'force', 'Aero'),
    'aero_Fz_front_N':         ('Downforce on the front axle', 'force', 'Aero'),
    'aero_Fz_rear_N':          ('Downforce on the rear axle', 'force', 'Aero'),
}
_DYN_PAIR = {   # <base>_front_<u> / <base>_rear_<u> -> one axle channel
    'rc_height':     ('Roll centre height', 'length', 'Geometry'),
    'elastic_lt':    ('Elastic (spring + bar) load transfer', 'force', 'Load transfer'),
    'geometric_lt':  ('Geometric (roll-centre) load transfer', 'force', 'Load transfer'),
    'jacking_force': ('Jacking force', 'force', 'Jacking'),
    'jacking_heave': ('Jacking heave (body lift)', 'length', 'Jacking'),
    'aero_heave':    ('Aero heave', 'length', 'Aero'),
    'pitch_travel':  ('Pitch travel', 'length', 'Body'),
}
_DYN_CORNER = {   # <base>_<FL|FR|RL|RR>
    'Fz':            ('Tyre vertical load', 'force', 'Tyres'),
    'travel':        ('Suspension travel (+ bump)', 'length', 'Suspension'),
    'camber':        ('Camber to the chassis', 'angle', 'Tyres'),
    'camber_ground': ('Camber to the road', 'angle', 'Tyres'),
    'toe':           ('Toe to the chassis (+ toe-in, includes steer)', 'angle', 'Tyres'),
    'inclination':   ('Inclination angle (tyre to road)', 'angle', 'Tyres'),
    'utilization':   ('Tyre grip used (1 = at the limit)', 'ratio', 'Tyres'),
    'wheel_rate':    ('Wheel rate (spring, at the solved travel)', 'stiffness', 'Rates'),
    'slip_angle':    ('Slip angle', 'angle', 'Tyres'),
    'Fy':            ('Tyre lateral force', 'force', 'Tyres'),
    'Fx':            ('Tyre longitudinal force', 'force', 'Tyres'),
    'ride_rate':     ('Ride rate (wheel + tyre, at the solved travel)', 'stiffness', 'Rates'),
    'ride_freq':     ('Ride frequency (at the solved travel)', 'frequency', 'Rates'),
}
_SUFFIX_FAMILY = [('_degps', 'angle'), ('_deg', 'angle'), ('_mm', 'length'), ('_N', 'force'), ('_g', 'accel'),
                  ('_Npm', 'stiffness'), ('_kph', 'speed'), ('_mph', 'speed'), ('_Nm', 'torque'),
                  ('_pct', 'percent'), ('_Hz', 'frequency'), ('_m', 'distance'), ('_s', 'time')]
_SUFFIX_SCALE = {'_Npm': 1e-3, '_kph': 1.0 / 3.6, '_mph': 1.0 / 2.2369362921}


def guess_family(key: str) -> tuple:
    """(family, scale-to-base) from a key's unit suffix; ('ratio', 1) when unknown."""
    for suf, fam in _SUFFIX_FAMILY:
        if key.endswith(suf):
            return fam, _SUFFIX_SCALE.get(suf, 1.0)
    return 'ratio', 1.0


def _strip_suffix(key: str) -> str:
    for suf, _f in _SUFFIX_FAMILY:
        if key.endswith(suf):
            return key[:-len(suf)]
    return key


def _arr_like(v, n=None):
    if v is None or isinstance(v, (dict, str, bool)):
        return None
    a = np.asarray(v, dtype=float) if not np.isscalar(v) else np.asarray([float(v)])
    if a.ndim != 1:
        return None
    if n is not None and a.size == 1 and n > 1:
        a = np.full(n, float(a[0]))
    return a


def dyn_channels(sweep: Optional[dict]) -> List[Channel]:
    """Channels of a Dynamics-panel sweep dict.  With sweep=None the KNOWN channels are listed
    (so the picker is never empty before a sweep has run); with a sweep every numeric array it
    carries is listed, known ones with their proper label/unit, the rest by their key."""
    out: List[Channel] = []
    seen = set()

    def _scalar(key, label, fam, cat, scale=1.0):
        seen.add(key)
        out.append(Channel(key, label, fam, 'dyn', cat, 'vehicle',
                           lambda d, key=key, scale=scale: ({'': _arr_like(d.get(key)) * scale}
                                                            if _arr_like(d.get(key)) is not None else {})))

    def _pair(base, label, fam, cat, scale=1.0):
        def _g(d, base=base, scale=scale):
            r = {}
            for ax_, nm in (('F', 'front'), ('R', 'rear')):
                for k in d:
                    if k.startswith(f'{base}_{nm}') and _arr_like(d.get(k)) is not None:
                        r[ax_] = _arr_like(d.get(k)) * scale
                        break
            return r
        out.append(Channel(base, label, fam, 'dyn', cat, 'axle', _g))

    def _corner(base, label, fam, cat, scale=1.0):
        def _g(d, base=base, scale=scale):
            r = {}
            for c in CORNERS:
                a = _arr_like(d.get(f'{base}_{c}'))
                if a is not None:
                    r[c] = a * scale
            return r
        out.append(Channel(base, label, fam, 'dyn', cat, 'corner', _g))

    keys = list(sweep.keys()) if isinstance(sweep, dict) else []
    # known scalars (listed if present, or all of them when no sweep yet)
    for k, (label, fam, cat) in _DYN_SCALAR.items():
        if sweep is None or k in sweep:
            _scalar(k, label, fam, cat, 1.0 / 3.6 if k == 'speed_kph' else 1.0)
    for base, (label, fam, cat) in _DYN_PAIR.items():
        if sweep is None or any(k.startswith(base + '_front') for k in keys):
            _pair(base, label, fam, cat)
            seen.update(k for k in keys if k.startswith(base + '_front') or k.startswith(base + '_rear'))
    for base, (label, fam, cat) in _DYN_CORNER.items():
        if sweep is None or any(k == f'{base}_{c}' for c in CORNERS for k in keys):
            _corner(base, label, fam, cat)
            seen.update(f'{base}_{c}' for c in CORNERS)
    # the handling extras MainWindow._attach_dyn_extras stashes
    def _ex(d, key, scale=1.0):
        ex = d.get('dyn_extras') or {}
        a = _arr_like(ex.get(key), n=len(d.get('lateral_g', d.get('longitudinal_g', d.get('speed_kph', [])))))
        return {'': a * scale} if a is not None else {}
    EX = [('roll_grad_dyn', 'Roll gradient, slope along the sweep (d roll / d lateral g)', 'deg_per_g', 'Handling'),
          ('roll_grad_secant', 'Roll gradient, roll ÷ lateral g', 'deg_per_g', 'Handling'),
          ('pitch_grad_dyn', 'Pitch gradient, slope along the sweep', 'deg_per_g', 'Handling'),
          ('pitch_grad_secant', 'Pitch gradient, pitch ÷ longitudinal g', 'deg_per_g', 'Handling'),
          ('pitch_stiff_eff', 'Pitch stiffness (effective, moment ÷ solved pitch)', 'roll_stiff', 'Rates'),
          ('pitch_stiff_springs', 'Pitch stiffness from the springs (solver formula)', 'roll_stiff', 'Rates'),
          ('lt_share_front_pct', 'Load-transfer split, front share', 'percent', 'Load transfer'),
          ('elastic_share_front_pct', 'Elastic load-transfer split, front share', 'percent', 'Load transfer')]
    ex = (sweep or {}).get('dyn_extras') if isinstance(sweep, dict) else None
    for k, label, fam, cat in EX:
        if sweep is None or (ex and k in ex):
            out.append(Channel('ex:' + k, label, fam, 'dyn', cat, 'vehicle', lambda d, k=k: _ex(d, k)))
    if sweep is None or (ex and ('geo_share_front_pct' in ex or 'geo_share_rear_pct' in ex)):
        out.append(Channel('ex:geo_share', "Geometric share of each axle's load transfer", 'percent', 'dyn', 'Load transfer', 'axle',
                           lambda d: {ax_: a for ax_, a in (('F', _ex(d, 'geo_share_front_pct').get('')),
                                                             ('R', _ex(d, 'geo_share_rear_pct').get(''))) if a is not None}))
    if sweep is None or (ex and 'roll_stiff' in ex):
        def _rs(d, names):
            rs = (d.get('dyn_extras') or {}).get('roll_stiff') or {}
            n = len(d.get('lateral_g', d.get('longitudinal_g', d.get('speed_kph', []))))
            r = {}
            for lab, k in names:
                a = _arr_like(rs.get(k), n=n)
                if a is not None:
                    r[lab] = a
            return r
        out.append(Channel('ex:roll_stiff_static', 'Roll stiffness the solver uses (static)', 'roll_stiff', 'dyn', 'Rates', 'vehicle',
                           lambda d: _rs(d, [('F', 'F'), ('R', 'R'), ('total', 'total'), ('F springs', 'F_springs'),
                                             ('F bar', 'F_bar'), ('R springs', 'R_springs'), ('R bar', 'R_bar')])))
        out.append(Channel('ex:roll_stiff_travel', 'Roll stiffness at the solved travel (swept)', 'roll_stiff', 'dyn', 'Rates', 'vehicle',
                           lambda d: _rs(d, [('F', 'F_total_travel'), ('R', 'R_total_travel'), ('total', 'total_travel'),
                                             ('F springs', 'F_springs_travel'), ('F bar', 'F_bar_travel'),
                                             ('R springs', 'R_springs_travel'), ('R bar', 'R_bar_travel')])))
        out.append(Channel('ex:roll_stiff_share', 'Roll stiffness front share', 'percent', 'dyn', 'Rates', 'vehicle',
                           lambda d: _rs(d, [('static', 'share_f_pct'), ('at travel', 'share_f_travel')])))
    # steering effort chain (MainWindow._attach_steering_effort), on its own lateral-g grid -> interpolated
    def _se(d, key):
        se = d.get('steering_effort') or {}
        cur = se.get(key)
        lat = _arr_like(d.get('lateral_g'))
        if not cur or lat is None:
            return {}
        g = np.array([float(p[0]) for p in cur]); t = np.array([float(p[1]) for p in cur])
        o = np.argsort(g)
        return {'': np.interp(lat, g[o], t[o], left=np.nan, right=np.nan)}
    if sweep is None or 'lateral_g' in (sweep or {}):
        out.append(Channel('steer:T', 'Steering torque at the hand wheel', 'torque', 'dyn', 'Steering', 'vehicle',
                           lambda d: _se(d, 'T_vs_latg'), note='needs the steering-effort chain (computed on demand)'))
        out.append(Channel('steer:F_rim', 'Push at the steering-wheel rim', 'force', 'dyn', 'Steering', 'vehicle',
                           lambda d: _se(d, 'F_rim_vs_latg'), note='needs the steering-effort chain (computed on demand)'))
    # anything else numeric the sweep carries, by key
    if isinstance(sweep, dict):
        groups = {}
        for k in keys:
            if k in seen or k in _DYN_SCALAR or _arr_like(sweep.get(k)) is None:
                continue
            if k in ('speed_mph',):
                continue
            parts = k.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in CORNERS:
                groups.setdefault(parts[0], set()).add(parts[1])
            else:
                fam, sc = guess_family(k)
                _scalar(k, _strip_suffix(k).replace('_', ' '), fam, 'Other', sc)
        for base in groups:
            if base in _DYN_CORNER:
                continue
            fam, sc = guess_family(base)
            _corner(base, _strip_suffix(base).replace('_', ' '), fam, 'Other', sc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  LAP SIM  (data = {'res': LapResult, 'track': Track})
# ─────────────────────────────────────────────────────────────────────────────
def _lap_attr(d, name, scale=1.0):
    a = getattr(d['res'], name, None)
    a = _arr_like(a)
    return {'': a * scale} if a is not None else {}


def _lap_kappa(d):
    res, tr = d['res'], d.get('track')
    s = np.asarray(res.s_m, float)
    if tr is None:
        return None
    k = np.asarray(tr.kappa_1pm, float)
    if len(k) != len(s):
        k = np.interp(s, np.linspace(s[0], s[-1], len(k)), k)
    return k


def _lap_segment(d):
    """1 on corner stations (isolate_corners intervals), 0 on straights."""
    res = d['res']
    s = np.asarray(res.s_m, float)
    seg = np.zeros(len(s))
    for c in getattr(res, 'corners', None) or []:
        seg[int(c['i_in']):int(c['i_out']) + 1] = 1.0
    return {'': seg}


def _lap_det(d, key, scale=1.0):
    res = d['res']
    ch = (getattr(res, 'det_channels', None) or {}).get(key)
    ds = getattr(res, 'det_s_m', None)
    if ch is None or ds is None:
        return None
    s = np.asarray(res.s_m, float)
    a = np.asarray(ch, float); ds = np.asarray(ds, float)
    if len(a) != len(ds) or len(a) < 2:
        return None
    return np.interp(s, ds, a) * scale


def lap_channels(res=None) -> List[Channel]:
    """Channels of a LapResult, all on the lap-distance grid (the detail-pass channels are
    resampled onto it)."""
    out = [
        Channel('lap:s', 'Distance along the lap', 'distance', 'lap', 'Lap', 'vehicle', lambda d: _lap_attr(d, 's_m')),
        Channel('lap:t', 'Time into the lap', 'time', 'lap', 'Lap', 'vehicle', lambda d: _lap_attr(d, 't_s')),
        Channel('lap:v', 'Speed', 'speed', 'lap', 'Lap', 'vehicle', lambda d: _lap_attr(d, 'v_ms')),
        Channel('lap:v_cap', 'Fastest the grip allows here (corner speed cap)', 'speed', 'lap', 'Lap', 'vehicle',
                lambda d: _lap_attr(d, 'v_corner_cap_ms')),
        Channel('lap:lat_g', 'Sideways g (unsigned)', 'accel', 'lap', 'Lap', 'vehicle', lambda d: _lap_attr(d, 'lat_g')),
        Channel('lap:lat_g_signed', 'Sideways g (+ left-hand turn, − right-hand)', 'accel', 'lap', 'Lap', 'vehicle',
                lambda d: ({'': np.sign(_lap_kappa(d)) * _arr_like(d['res'].lat_g)} if _lap_kappa(d) is not None else {})),
        Channel('lap:lon_g', 'Forward g (+ speeding up, − braking)', 'accel', 'lap', 'Lap', 'vehicle', lambda d: _lap_attr(d, 'lon_g')),
        Channel('lap:lon_tyre_g', 'Forward g the tyres make (drag and scrub taken out)', 'accel', 'lap', 'Lap', 'vehicle',
                lambda d: _lap_attr(d, 'lon_tyre_g')),
        Channel('lap:ay_cap', 'Sideways grip limit here', 'accel', 'lap', 'Lap', 'vehicle',
                lambda d: _lap_attr(d, 'ay_cap_local_ms2', 1.0 / 9.80665)),
        Channel('lap:kappa', 'Path curvature (+ left)', 'curvature', 'lap', 'Track', 'vehicle',
                lambda d: ({'': _lap_kappa(d)} if _lap_kappa(d) is not None else {})),
        Channel('lap:radius', 'Path radius', 'distance', 'lap', 'Track', 'vehicle',
                lambda d: ({'': 1.0 / np.maximum(np.abs(_lap_kappa(d)), 1e-4)} if _lap_kappa(d) is not None else {})),
        Channel('lap:segment', 'Segment (1 = corner, 0 = straight)', 'count', 'lap', 'Track', 'vehicle', _lap_segment),
        Channel('lap:limit', 'What limits the speed here (0 grip, 1 power, 2 braking)', 'count', 'lap', 'Lap', 'vehicle',
                lambda d: _lap_attr(d, 'limit')),
        Channel('lap:aero_down', 'Downforce', 'force', 'lap', 'Aero', 'vehicle', lambda d: _lap_attr(d, 'aero_down_N')),
        Channel('lap:aero_drag', 'Aero drag', 'force', 'lap', 'Aero', 'vehicle', lambda d: _lap_attr(d, 'aero_drag_N')),
        Channel('lap:scrub', 'Front-tyre scrub drag', 'force', 'lap', 'Aero', 'vehicle', lambda d: _lap_attr(d, 'scrub_N')),
        Channel('lap:rh', 'Ride height', 'length', 'lap', 'Aero', 'axle',
                lambda d: {k: v for k, v in (('F', _arr_like(getattr(d['res'], 'rh_front_mm', None))),
                                             ('R', _arr_like(getattr(d['res'], 'rh_rear_mm', None)))) if v is not None}),
        Channel('lap:rpm', 'Engine speed', 'rpm', 'lap', 'Powertrain', 'vehicle', lambda d: _lap_attr(d, 'engine_rpm')),
        Channel('lap:gear', 'Gear', 'count', 'lap', 'Powertrain', 'vehicle', lambda d: _lap_attr(d, 'gear')),
        Channel('lap:power', 'Power used', 'power', 'lap', 'Powertrain', 'vehicle', lambda d: _lap_attr(d, 'power_used_W')),
        Channel('lap:diff', 'Differential yaw moment (+ pushes straight)', 'torque', 'lap', 'Powertrain', 'vehicle',
                lambda d: _lap_attr(d, 'diff_yaw_Nm')),
    ]
    det = (getattr(res, 'det_channels', None) or {}) if res is not None else {}
    groups = {}
    singles = []
    for k in det:
        parts = k.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in CORNERS:
            groups.setdefault(parts[0], set()).add(parts[1])
        else:
            singles.append(k)
    _DET = {'util': ('Tyre grip used (1 = limit)', 'ratio'), 'Fz': ('Tyre vertical load', 'force'),
            'camber': ('Camber', 'angle'), 'travel': ('Suspension travel', 'length'),
            'util_axle': ('Axle grip used', 'ratio')}
    for base in sorted(groups):
        label, fam = _DET.get(base, (base.replace('_', ' '), guess_family(base)[0]))
        sc = 1.0 if base in _DET else guess_family(base)[1]
        out.append(Channel('det:' + base, f'{label} (detail pass)', fam, 'lap', 'Full solve along the lap', 'corner',
                           lambda d, base=base, sc=sc: {c: a for c in CORNERS
                                                        for a in [_lap_det(d, f'{base}_{c}', sc)] if a is not None}))
    for k in sorted(singles):
        if k.startswith('util_axle_'):
            continue
        fam, sc = guess_family(k)
        lab = {'roll': ('Roll angle (detail pass)', 'angle'), 'lltd': ('Load-transfer front share (detail pass)', 'percent')}.get(k)
        if lab:
            label, fam, sc = lab[0], lab[1], 1.0
        else:
            label = k.replace('_', ' ') + ' (detail pass)'
        out.append(Channel('det:' + k, label, fam, 'lap', 'Full solve along the lap', 'vehicle',
                           lambda d, k=k, sc=sc: ({'': _lap_det(d, k, sc)} if _lap_det(d, k, sc) is not None else {})))
    if 'util_axle_F' in det or 'util_axle_R' in det:
        out.append(Channel('det:util_axle', 'Axle grip used, the pair of tyres (detail pass)', 'ratio', 'lap',
                           'Full solve along the lap', 'axle',
                           lambda d: {ax_: a for ax_ in ('F', 'R') for a in [_lap_det(d, f'util_axle_{ax_}')] if a is not None}))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  pairing X with Y
# ─────────────────────────────────────────────────────────────────────────────
_AXLE_OF = {'FL': 'F', 'FR': 'F', 'RL': 'R', 'RR': 'R'}


def pair_series(x_series: Dict[str, np.ndarray], y_series: Dict[str, np.ndarray],
                corners=CORNERS) -> List[tuple]:
    """[(series name, x, y)] for every drawable pair: a single X series is shared by every Y
    series; per-corner X pairs with the same corner (or that corner's axle); corner filtering
    applies to corner / axle series names."""
    keep = set(corners or CORNERS)
    axles = {_AXLE_OF[c] for c in keep}

    def _ok(name):
        if name in CORNERS:
            return name in keep
        if name in ('F', 'R'):
            return name in axles
        return True

    out = []
    xs = {k: v for k, v in x_series.items()}
    if not xs or not y_series:
        return out
    single_x = list(xs.values())[0] if len(xs) == 1 else None
    for name, y in y_series.items():
        if not _ok(name):
            continue
        x = None
        if name in xs:
            x = xs[name]
        elif name in CORNERS and _AXLE_OF[name] in xs:
            x = xs[_AXLE_OF[name]]
        elif name in ('F', 'R') and (name + 'L') in xs:
            x = xs[name + 'L']
        elif single_x is not None:
            x = single_x
        elif '' in xs:
            x = xs['']
        if x is None or len(x) != len(y):
            continue
        out.append((name, np.asarray(x, float), np.asarray(y, float)))
    return out
