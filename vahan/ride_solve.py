"""Ride-rate solve on the road model (ONE MODEL: the vehicle is the GUI's
solved VehicleParams; this module computes, the Ride page only calls/plots).

What it does
------------
* builds ISO 8608 class A/B synthetic roads ONCE in distance (stored seeds),
  replays them at each speed of a bracket with the same-path wheelbase delay
  and a left/right coherence bracket (identical tracks vs independent seeds);
* evaluates the 7-DOF linear RideModel on every road case: per-corner dynamic
  load coefficient DLC = RMS(Fz-mean)/mean, contact-loss indicator, body
  acceleration RMS, wheel-travel and damper-velocity peaks against the
  bump-stop margin;
* checks the occasional single bump (half-sine, both wheels of an axle, rear
  delayed by wheelbase/speed) for front/rear settling;
* sweeps front x rear RIDE RATE, picks the pair that minimises the worst-corner
  DLC subject to the travel margin and the settling check, and converts the
  chosen ride rates to SPRING rates through VehicleParams' own wheel-rate
  relation (tire in series, motion ratio, geometric MR-slope term) plus the
  nearest standard 25 lbf/in springs with the preload that keeps ride height.

Explicit assumptions (all shown on the page, none inferred from hardware):
* inertias and the four wheel-frame damping coefficients are USER INPUTS;
* the occasional-bump check is the RCVD flat-ride rule (Milliken, RCVD
  p.410: "the car lands flat after crossing a bump ... slightly higher
  undamped natural frequency on the rear than on the front, i.e.
  approximately 10% stiffer on the rear") evaluated on the FULL 7-DOF model
  with the wheelbase delay instead of as a frequency rule of thumb: from the
  instant the REAR wheels leave the bump to the end of the record, the body
  motion is split into BOUNCE (mean of the two axle-centreline motions) and
  PITCH (half their difference, i.e. pitch angle x wheelbase/2). "Lands
  flat" = pitch RMS / bounce RMS <= flat_ride_max_ratio (default 1.0: the
  pitch motion at the axles is no larger than the bounce motion). This
  ratio is smooth in the ride rates and, at light damping, is minimal at a
  rear/front frequency ratio of 1.0-1.2 — RCVD's rule recovered from the
  simulation. The sweep gates on the ratio AVERAGED over the speed bracket
  (the bump is spatial; the arrival delay wheelbase/speed makes slow
  speeds pitch more) because the project brief uses this check to judge the
  front/rear SPLIT, not as the primary ride-rate justification
  (DESIGN_2027/binder_run/product_goals.md, "do not tune to one occasional
  bump"): if no grid point meets the threshold, the best travel-feasible
  point is reported with that status instead of no answer. Threshold settle
  times (settle_fraction of own peak) are information; the record must
  show both axles settled (validity of the window);
* the independent right-hand track uses seed + INDEPENDENT_SEED_OFFSET;
* seeds: the road synthesis fixes the amplitude spectrum and randomises only
  the phases, so by Parseval every RMS metric of the linear response (DLC,
  acceleration RMS, travel RMS) is seed-independent on coherent tracks;
  peaks (travel, min load, damper peak, contact-loss fraction) and the
  independent-track cases (left/right cross terms) do depend on the seed;
* aero (RideInputs.aero_cla_m2 > 0, the Dynamics panel package when aero is
  applied): every road case runs about the LOADED state at its own speed
  (loaded_model) — tyre loads include the downforce and the wheel travel
  to the stops is counted from rest, i.e. the aero sink plus the road;
* the linear model reports the fraction of samples where its predicted load
  reaches zero as a contact-loss INDICATOR — the physics there is outside the
  linear regime; RideModel.contact_response is the nonlinear check.

Dyno dampers (RideInputs.damper = vahan.damper.WheelDampers, the project's
car['damper'] at the solved static MR).  Absent = the four assumed linear
wheel coefficients above, every number unchanged.  Present:
* frequency domain: each road case uses the EQUIVALENT LINEAR wheel
  coefficient (secant F(v)/v, comp/reb averaged) at that case's own RMS
  damper velocity, found by fixed-point iteration (<= damper_max_iter,
  relative change <= damper_tol; convergence reported per case);
* the current-car study ALSO integrates every case in the time domain with
  the nonlinear asymmetric damper law and the unilateral tyre
  (RideModel.contact_response, RK4) and reports ITS DLC / travel / damper
  velocity / contact numbers as the case metrics (the first
  time_domain_discard_s of each record are the start-from-rest transient and
  are excluded); the equivalent-linear metrics are kept alongside;
* transfer curves use the equivalent coefficients of the reference case
  (middle speed, coherent, first seed), PSDs and the bump check those of the
  case at the same speed.
"""
from dataclasses import dataclass, replace, field
import numpy as np

from vahan.ride import RideModel, CORNERS
from vahan.road import RoadSpectrum, synthesize_profile, delayed_profile
from vahan.laptime import wheel_rate_from_ride_rate_Npm

LBF_PER_IN_TO_NPM = 175.127
INDEPENDENT_SEED_OFFSET = 7919      # right track seed when tracks are independent
G_MPS2 = 9.81
COHERENCE_CHOICES = ('coherent', 'independent', 'both')
AXLES = ('front', 'rear')


# ─────────────────────────────────────────────────────────────────────────────
#  Inputs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RideInputs:
    """Every user-owned assumption of the ride study, SI units."""
    pitch_inertia_kgm2: float
    roll_inertia_kgm2: float
    corner_damping_Nspm: tuple           # FL, FR, RL, RR — wheel frame
    road_class: str = 'A'
    speeds_mps: tuple = (10., 20., 30.)
    seeds: tuple = (1,)
    coherence: str = 'both'              # coherent | independent | both
    stop_margin_m: float = 0.025         # wheel travel to the stop (either way)
    record_length_m: float = 1000.
    spacing_m: float = 0.02
    bump_height_m: float = 0.025
    bump_length_m: float = 0.5
    settle_fraction: float = 0.05
    settle_record_s: float = 6.0
    flat_ride_max_ratio: float = 1.0    # pitch RMS / bounce RMS after the bump
    # nonlinear dyno dampers at the wheel (vahan.damper.WheelDampers) or None
    damper: object = field(default=None, compare=False)
    damper_max_iter: int = 10           # equivalent-linear fixed-point iterations
    damper_tol: float = 1e-3            # relative change in c that counts as converged
    time_domain_discard_s: float = 2.0  # start-from-rest transient excluded from stats
    # Aero package (downforce = ClA x rho/2 x v^2, split by the centre of
    # pressure). 0 = no aero (every number as before). With aero, each road
    # case runs about the LOADED state at its own speed: the mean tyre loads
    # (DLC denominator, unilateral-contact threshold) include the downforce
    # and the travel to the stops starts from the aero sink, not from rest.
    aero_cla_m2: float = 0.0
    aero_cop_rear: float = 0.5           # rear share of the downforce (0..1)
    air_density: float = 1.225
    # Wheel travel from rest to the BUMP stop (m); None = stop_margin_m (the
    # old symmetric margin).  With aero the sink moves the car toward the
    # bump stop, so the two sides have to be judged separately:
    # usage = max(bump peak / bump margin, droop peak / stop_margin_m).
    bump_stop_margin_m: float = None

    def __post_init__(self):
        if self.coherence not in COHERENCE_CHOICES:
            raise ValueError(f'coherence must be one of {COHERENCE_CHOICES}')
        if not self.speeds_mps or any(not np.isfinite(v) or v <= 0 for v in self.speeds_mps):
            raise ValueError('speeds must be positive and finite')
        if not self.seeds:
            raise ValueError('at least one seed required')
        for name in ('stop_margin_m', 'record_length_m', 'spacing_m',
                     'bump_height_m', 'bump_length_m', 'settle_record_s',
                     'flat_ride_max_ratio'):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 < self.settle_fraction < 1:
            raise ValueError('settle_fraction must lie in (0,1)')
        if self.bump_stop_margin_m is not None and not float(self.bump_stop_margin_m) > 0:
            raise ValueError('bump_stop_margin_m must be positive')
        if (not np.isfinite(self.aero_cla_m2) or self.aero_cla_m2 < 0
                or not 0 <= self.aero_cop_rear <= 1 or not self.air_density > 0):
            raise ValueError('aero: ClA >= 0, centre of pressure in [0,1], density > 0')

    @property
    def coherence_values(self):
        return ('coherent', 'independent') if self.coherence == 'both' else (self.coherence,)


# ─────────────────────────────────────────────────────────────────────────────
#  Ride rate <-> frequency <-> spring rate (VehicleParams' own relation)
# ─────────────────────────────────────────────────────────────────────────────
# What the road model leaves out — shown word for word on the Ride page and in
# the binder so nobody has to guess (user 2026-10-23: "everything very
# explicit as 'this is not being considered'").
NOT_CONSIDERED = (
    'Tyre size: the tyre touches the road at one point, so it does not roll over or swallow '
    'short sharp bumps (a real contact patch smooths them).',
    'Tyre damping (rubber losses): the tyre is a pure spring.',
    'Bump stops and droop limits as forces: travel past them is only flagged.',
    'Motion ratio and wheel rate changing with travel: held at the static values.',
    'Frame flex: the body is rigid.',
    'Downforce changing with ride height or pitch: downforce depends on speed only.',
    'Cornering, braking or throttle at the same time: straight-line driving only.',
    'Damper friction, gas force and heat fade: the damper is its dyno curve only.',
    'The real track surface: the road is a made-up ISO 8608 road, nobody measured an FSAE track.',
)


def estimated_inertias(veh):
    """Body pitch and roll inertia ESTIMATED from the car (no measurement),
    the same rules the transient model uses (gui/main_window auto inertias):
      roll  = sprung mass x (roll_gyradius_track_frac x mean track)^2
      pitch = 0.9 x yaw, yaw = yaw_inertia_factor x total mass x a x b
    Returns values (kg.m^2) and the plain-words method of each."""
    track = 0.5 * (float(veh.front_track_m) + float(veh.rear_track_m))
    k_roll = float(veh.roll_gyradius_track_frac) * track
    roll = float(veh.sprung_mass_kg) * k_roll ** 2
    m = float(veh.total_mass_kg)
    yaw = float(veh.yaw_inertia_factor) * m * float(veh.cg_to_front_axle_m) * float(veh.cg_to_rear_axle_m)
    pitch = 0.9 * yaw
    return {'pitch_kgm2': pitch, 'roll_kgm2': roll, 'yaw_kgm2': yaw,
            'pitch_method': (f'0.9 x yaw inertia {yaw:.0f} kg·m² (yaw = {veh.yaw_inertia_factor:.2f} x '
                             f'{m:.0f} kg x front {veh.cg_to_front_axle_m*1000:.0f} mm x rear '
                             f'{veh.cg_to_rear_axle_m*1000:.0f} mm)'),
            'roll_method': (f'sprung mass {veh.sprung_mass_kg:.0f} kg x ({veh.roll_gyradius_track_frac:.2f} x '
                            f'mean track {track*1000:.0f} mm)²')}


def sprung_corner_masses_kg(veh):
    """(front, rear) sprung mass per corner — the same axle moment balance
    RideModel.from_vehicle uses, so ride frequency and the model agree."""
    ms = float(veh.sprung_mass_kg)
    muf, mur = float(veh.unsprung_mass_front_kg), float(veh.unsprung_mass_rear_kg)
    length, a = float(veh.wheelbase_m), float(veh.cg_to_front_axle_m)
    mt = ms + muf + mur
    front_sprung = mt * (length - a) / length - muf
    rear_sprung = mt * a / length - mur
    if min(front_sprung, rear_sprung) <= 0:
        raise ValueError('sprung axle support masses must be positive')
    return front_sprung / 2., rear_sprung / 2.


def ride_frequency_Hz(ride_rate_Npm, corner_mass_kg):
    """Undamped natural frequency of the sprung corner on its RIDE rate
    (wheel rate in series with the tire), Hz."""
    return float(np.sqrt(float(ride_rate_Npm) / float(corner_mass_kg)) / (2 * np.pi))


def ride_rate_from_frequency_Npm(frequency_Hz, corner_mass_kg):
    return float((2 * np.pi * float(frequency_Hz)) ** 2 * float(corner_mass_kg))


def spring_rate_for_ride_rate_Npm(veh, axle, ride_rate_Npm):
    """Corner SPRING rate that gives `ride_rate_Npm` on this VehicleParams.

    Uses the model's own chain: ride -> wheel (tire in series, the laptime
    inversion) -> spring through VehicleParams.wheel_rate_<axle>_Npm, which
    is affine in the spring rate (MR², plus the static-force x MR-slope
    geometric term). Verified by round trip; raises when the topology has no
    corner spring or the demanded ride rate exceeds the tire rate.
    """
    if axle not in AXLES:
        raise ValueError('axle must be front or rear')
    target = float(ride_rate_Npm)
    if not np.isfinite(target) or target <= 0:
        raise ValueError('ride rate must be positive and finite')
    wheel = wheel_rate_from_ride_rate_Npm(target, veh.tire_rate_Npm)
    if not np.isfinite(wheel):
        raise ValueError('ride rate at or above the tire rate is unreachable by any spring')
    key, wprop = f'spring_rate_{axle}_Npm', f'wheel_rate_{axle}_Npm'
    k0 = float(getattr(veh, key)); w0 = float(getattr(veh, wprop))
    k1 = k0 + 10000.; w1 = float(getattr(replace(veh, **{key: k1}), wprop))
    slope = (w1 - w0) / (k1 - k0)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError('corner spring does not set the wheel rate on this topology')
    k = k0 + (wheel - w0) / slope
    if k <= 0:
        raise ValueError('ride rate below the geometric wheel-rate term: no positive spring')
    got = float(getattr(replace(veh, **{key: k}), f'ride_rate_{axle}_Npm'))
    if abs(got - target) > 1e-6 * target:
        raise ValueError(f'ride/spring round trip failed: {got} vs {target}')
    return float(k)


def vehicle_with_ride_rates(veh, front_ride_rate_Npm, rear_ride_rate_Npm):
    """VehicleParams copy whose springs realise the two ride rates."""
    return replace(veh,
                   spring_rate_front_Npm=spring_rate_for_ride_rate_Npm(veh, 'front', front_ride_rate_Npm),
                   spring_rate_rear_Npm=spring_rate_for_ride_rate_Npm(veh, 'rear', rear_ride_rate_Npm))


def standard_spring_options(veh, axle, spring_rate_Npm, *, preload_mm, stroke_mm,
                            step_lbf_in=25., n_each_side=2):
    """Nearby catalogue springs (multiples of `step_lbf_in`) for one axle.

    Each row: the standard rate, the wheel/ride rate and ride frequency it
    gives on THIS car, its deviation from the exact solved ride rate, and the
    collar preload (mm of spring) that keeps the static ride height the exact
    spring would give with the current preload — via VehicleParams.static_sag
    (sprung corner weight / (MR x k)). A negative preload means the stiffer
    spring already sits higher at zero preload; the ride-height change at the
    wheel is reported in that case.
    """
    if axle not in AXLES:
        raise ValueError('axle must be front or rear')
    key = f'spring_rate_{axle}_Npm'
    exact = replace(veh, **{key: float(spring_rate_Npm)})
    kw = dict(preload_front_mm=preload_mm, preload_rear_mm=preload_mm, stroke_mm=stroke_mm)
    sag_exact = exact.static_sag(**kw)
    target_ride = float(getattr(exact, f'ride_rate_{axle}_Npm'))
    mass = sprung_corner_masses_kg(veh)[0 if axle == 'front' else 1]
    mr = float(getattr(veh, f'motion_ratio_{axle}'))
    base_lbf = float(spring_rate_Npm) / LBF_PER_IN_TO_NPM
    lower = np.floor(base_lbf / step_lbf_in) * step_lbf_in
    rows = []
    for step in range(-n_each_side + 1, n_each_side + 1):
        lbf = lower + step * step_lbf_in
        if lbf <= 0:
            continue
        k_std = lbf * LBF_PER_IN_TO_NPM
        cand = replace(veh, **{key: k_std})
        ride = float(getattr(cand, f'ride_rate_{axle}_Npm'))
        sag_std = cand.static_sag(**kw)
        required_std = sag_std[f'required_spring_compression_{axle}_mm']
        preload_needed = required_std - sag_exact[f'sag_shock_{axle}_mm']
        rows.append({
            'spring_lbf_in': float(lbf),
            'spring_Npm': float(k_std),
            'wheel_rate_Npm': float(getattr(cand, f'wheel_rate_{axle}_Npm')),
            'ride_rate_Npm': ride,
            'ride_frequency_Hz': ride_frequency_Hz(ride, mass),
            'ride_rate_deviation_pct': 100. * (ride / target_ride - 1.),
            'preload_to_hold_ride_height_mm': float(preload_needed),
            'ride_height_change_at_wheel_mm': (float(-preload_needed / mr)
                                               if preload_needed < 0 else 0.),
            'sag_shock_mm_at_that_preload': float(min(stroke_mm, max(0., required_std - max(0., preload_needed)))),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Road cases
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RoadCase:
    speed_mps: float
    seed: int
    coherence: str
    dt_s: float
    road_heights_m: np.ndarray       # (n, 4) FL FR RL RR at each time sample

    @property
    def label(self):
        return f'{self.speed_mps*3.6:.0f} km/h, seed {self.seed}, {self.coherence}'


def _profiles(inputs, seed):
    spectrum = RoadSpectrum(inputs.road_class)
    _, left = synthesize_profile(spectrum, length_m=inputs.record_length_m,
                                 spacing_m=inputs.spacing_m, seed=int(seed))
    _, right = synthesize_profile(spectrum, length_m=inputs.record_length_m,
                                  spacing_m=inputs.spacing_m,
                                  seed=int(seed) + INDEPENDENT_SEED_OFFSET)
    return left, right


def build_road_cases(inputs, wheelbase_m):
    """Roads are generated once in DISTANCE per seed and replayed at every
    speed (dt = spacing / v); the rear wheels see the front profile delayed
    by the wheelbase (exact periodic band-limited shift)."""
    if not np.isfinite(wheelbase_m) or wheelbase_m <= 0:
        raise ValueError('wheelbase must be positive and finite')
    cases = []
    for seed in inputs.seeds:
        left, right = _profiles(inputs, seed)
        for coherence in inputs.coherence_values:
            fr = left if coherence == 'coherent' else right
            rear_l = delayed_profile(left, inputs.spacing_m, wheelbase_m)
            rear_r = delayed_profile(fr, inputs.spacing_m, wheelbase_m)
            heights = np.column_stack([left, fr, rear_l, rear_r])
            for v in inputs.speeds_mps:
                cases.append(RoadCase(float(v), int(seed), coherence,
                                      inputs.spacing_m / float(v), heights))
    return cases


def aero_corner_loads_N(inputs, speed_mps):
    """Downforce per corner FL FR RL RR (N) of the inputs' aero package."""
    d = float(inputs.aero_cla_m2) * 0.5 * float(inputs.air_density) * float(speed_mps) ** 2
    f, r = d * (1. - float(inputs.aero_cop_rear)) / 2., d * float(inputs.aero_cop_rear) / 2.
    return np.array([f, f, r, r])


def loaded_model(model, inputs, speed_mps):
    """(model about the aero-loaded state at this speed, static suspension
    sink in m FL FR RL RR, + = bump).  The sink is the same model's own
    static solution K q = F under the corner downforce (linear wheel rates,
    tyre in series); the returned model's baseline tyre loads include the
    downforce. No aero in the inputs -> (model, zeros)."""
    fa = aero_corner_loads_N(inputs, speed_mps)
    if not np.any(fa > 0):
        return model, np.zeros(4)
    force = np.zeros(7)
    force[:3] = -model.body_corner_map.T @ fa        # downforce pushes the body corners down
    sink = model.suspension_map @ np.linalg.solve(model.stiffness, force)
    return replace(model, baseline_loads=model.baseline_loads + fa), sink


def _stop_usage(travel_from_rest, droop_margin_m, inputs):
    """(peak |travel| from rest, usage) per corner; bump side against
    inputs.bump_stop_margin_m when set, droop side against the stop margin."""
    bump = float(droop_margin_m)
    if inputs is not None and inputs.bump_stop_margin_m is not None:
        bump = float(inputs.bump_stop_margin_m)
    up = np.max(travel_from_rest, axis=0)
    down = np.max(-travel_from_rest, axis=0)
    usage = np.maximum(up / bump, down / float(droop_margin_m))
    return np.maximum(np.abs(up), np.abs(down)), usage


def road_psd_response(model, inputs, speed_mps, wheelbase_m, track_correlation,
                      frequencies_Hz=None):
    """Spectral (PSD) response of the model to the ISO class road at one
    speed and one left/right correlation — the RideModel.spectral_response
    wrapper the page plots. Returns (frequencies, outputs). The load
    coefficient is relative to the tyre load WITH the downforce at that speed."""
    from vahan.road import corner_cross_psd
    model, _ = loaded_model(model, inputs, speed_mps)
    if frequencies_Hz is None:
        spectrum = RoadSpectrum(inputs.road_class)
        f_hi = spectrum.high_cycles_per_m * float(speed_mps)
        frequencies_Hz = np.linspace(0.05, max(f_hi, 1.), 1500)
    road = corner_cross_psd(RoadSpectrum(inputs.road_class), frequencies_Hz,
                            float(speed_mps), wheelbase_m,
                            track_correlation=track_correlation)
    return np.asarray(frequencies_Hz, float), model.spectral_response(frequencies_Hz, road)


def transfer_curves(model, frequencies_Hz):
    """Magnitudes the page plots, per unit road input:
    heave: all four wheels in phase (m per m); pitch: front in phase, rear
    anti-phase (rad per m); roll: left/right anti-phase (rad per m);
    wheel: FL wheel per FL road (m per m); load: FL dynamic load per FL road (N per m)."""
    h = model.transfer(frequencies_Hz)
    q = h['coordinates']
    return {
        'frequency_Hz': np.asarray(frequencies_Hz, float),
        'heave_m_per_m': np.abs(q[:, 0, :] @ np.ones(4)),
        'pitch_rad_per_m': np.abs(q[:, 1, :] @ np.array([1., 1., -1., -1.])),
        'roll_rad_per_m': np.abs(q[:, 2, :] @ np.array([1., -1., 1., -1.])),
        'wheel_m_per_m': np.abs(q[:, 3, 0]),
        'load_N_per_m': np.abs(h['dynamic_tire_load_N'][:, 0, 0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RideMetrics:
    dlc: np.ndarray                     # (4) RMS(Fz-mean)/mean
    load_rms_N: np.ndarray              # (4)
    min_load_N: np.ndarray              # (4)
    contact_loss_fraction_linear: np.ndarray   # (4) linear-model indicator
    travel_peak_m: np.ndarray           # (4) max |wheel travel|
    travel_rms_m: np.ndarray            # (4)
    travel_usage: np.ndarray            # (4) peak / stop margin
    damper_velocity_rms_mps: np.ndarray  # (4) wheel velocity x MR
    damper_velocity_peak_mps: np.ndarray  # (4)
    body_heave_acc_rms_mps2: float
    body_pitch_acc_rms_radps2: float
    body_roll_acc_rms_radps2: float
    body_corner_acc_rms_mps2: np.ndarray  # (4)
    kind: str = 'linear periodic (frequency domain)'
    beyond_dyno_mps: float = 0.0        # damper |v| past the dyno range (dyno dampers only)
    # static aero sink (m, + = bump) the travel peak/usage start from;
    # travel_rms_m is the road part only
    aero_sink_m: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def worst_dlc(self):
        return float(np.max(self.dlc))


def evaluate_case(model, case, *, motion_ratios, stop_margin_m, inputs=None):
    """Metrics of one road case from the steady periodic linear response.
    With `inputs` carrying an aero package the case runs about the loaded
    state at its speed (loaded_model): loads include the downforce and the
    travel peak / stop usage are measured from rest, i.e. aero sink + road."""
    mr = np.asarray(motion_ratios, float)
    if mr.shape != (4,) or np.any(mr <= 0):
        raise ValueError('four positive motion ratios (FL FR RL RR) required')
    if stop_margin_m <= 0:
        raise ValueError('stop margin must be positive')
    sink = np.zeros(4)
    if inputs is not None:
        model, sink = loaded_model(model, inputs, case.speed_mps)
    resp = model.periodic_response(case.dt_s, case.road_heights_m)
    load = resp['dynamic_tire_load_N'] + model.baseline_loads
    mean = load.mean(axis=0)
    rms = np.sqrt(np.mean((load - mean) ** 2, axis=0))
    travel = resp['suspension_displacement_m']
    from_rest, usage = _stop_usage(travel + sink, stop_margin_m, inputs)
    vel = resp['suspension_velocity_mps'] * mr
    acc = resp['body_acceleration']
    return RideMetrics(
        dlc=rms / mean, load_rms_N=rms, min_load_N=load.min(axis=0),
        contact_loss_fraction_linear=np.mean(load <= 0., axis=0),
        travel_peak_m=from_rest,
        travel_rms_m=np.sqrt(np.mean(travel ** 2, axis=0)),
        travel_usage=usage,
        aero_sink_m=sink,
        damper_velocity_rms_mps=np.sqrt(np.mean(vel ** 2, axis=0)),
        damper_velocity_peak_mps=np.max(np.abs(vel), axis=0),
        body_heave_acc_rms_mps2=float(np.sqrt(np.mean(acc[:, 0] ** 2))),
        body_pitch_acc_rms_radps2=float(np.sqrt(np.mean(acc[:, 1] ** 2))),
        body_roll_acc_rms_radps2=float(np.sqrt(np.mean(acc[:, 2] ** 2))),
        body_corner_acc_rms_mps2=np.sqrt(np.mean(resp['body_corner_acceleration_mps2'] ** 2, axis=0)),
    )


def bump_settling(model, inputs, speed_mps, wheelbase_m, dt_s=1e-3):
    """Occasional single bump: half-sine of bump_height x bump_length under
    both wheels of an axle, rear delayed by wheelbase/speed, linear response.

    Returns the body traces over each axle centreline, their threshold
    settle instants (information: last time |motion| exceeds
    settle_fraction of its own peak, absolute from the front wheels entering
    the bump and relative to that axle's own entry), and the flat-ride
    number: pitch_to_bounce_ratio = RMS(half the axle difference) /
    RMS(axle mean) from the rear wheels leaving the bump to the end of the
    record. passes = both axles settled in the record AND the ratio is at
    most inputs.flat_ride_max_ratio (RCVD p.410 "lands flat").
    """
    count = int(round(inputs.settle_record_s / dt_s))
    count += 1 - count % 2                      # odd: no Nyquist bin
    t = np.arange(count) * dt_s
    duration = inputs.bump_length_m / float(speed_mps)
    if duration >= inputs.settle_record_s / 4:
        raise ValueError('bump lasts too long for the settle record')
    front = np.where(t < duration, inputs.bump_height_m * np.sin(np.pi * t / duration), 0.)
    delay = wheelbase_m / float(speed_mps)
    rear = delayed_profile(front, dt_s, delay)
    resp = model.periodic_response(dt_s, np.column_stack([front, front, rear, rear]))
    q = resp['coordinates']
    x_front, x_rear = model.body_corner_map[0, 1], model.body_corner_map[2, 1]
    out = {'time_s': t, 'delay_s': delay, 'pitch_rad': q[:, 1], 'heave_m': q[:, 0],
           'road_front_m': front, 'road_rear_m': rear}
    settled = True
    for name, x in (('front', x_front), ('rear', x_rear)):
        z = q[:, 0] + x * q[:, 1]
        peak = float(np.max(np.abs(z)))
        threshold = inputs.settle_fraction * peak
        above = np.nonzero(np.abs(z) > threshold)[0]
        last = float(t[above[-1]]) if len(above) else 0.
        tail = np.abs(z[-max(1, count // 20):]).max() <= threshold
        settled &= bool(tail)
        out[f'body_at_{name}_axle_m'] = z
        out[f'{name}_peak_m'] = peak
        out[f'{name}_settle_abs_s'] = last
    out['front_settle_s'] = out['front_settle_abs_s']
    out['rear_settle_s'] = out['rear_settle_abs_s'] - delay
    out['settled_in_record'] = settled
    out['pitch_peak_rad'] = float(np.max(np.abs(q[:, 1])))
    zf, zr = out['body_at_front_axle_m'], out['body_at_rear_axle_m']
    window = t >= delay + duration                 # rear wheels have left the bump
    out['window_start_s'] = float(delay + duration)
    bounce = 0.5 * (zf[window] + zr[window])
    pitch = 0.5 * (zf[window] - zr[window])        # = pitch angle x wheelbase / 2
    bounce_rms = float(np.sqrt(np.mean(bounce ** 2)))
    pitch_rms = float(np.sqrt(np.mean(pitch ** 2)))
    out['bounce_rms_after_bump_m'] = bounce_rms
    out['pitch_rms_after_bump_m'] = pitch_rms
    out['pitch_to_bounce_ratio'] = pitch_rms / bounce_rms if bounce_rms > 0 else np.inf
    out['flat_ride_ok'] = bool(out['pitch_to_bounce_ratio'] <= inputs.flat_ride_max_ratio)
    out['passes'] = bool(settled and out['flat_ride_ok'])
    return out


def launch_load_lag(veh, inputs, *, accel_g, thrust_rise_time_s, rear_slope_from_wheel_centre,
                    front_slope_from_wheel_centre, rear_wheel_centre_height_m,
                    front_wheel_centre_height_m, duration_s=1.0, dt_s=5e-4):
    """How fast the rear tyres RECEIVE the launch load transfer, vs anti-squat.

    Rear-wheel drive, differential on the CHASSIS (half-shafts). Per unit thrust
    Fx = m*a (RCVD p.617-619, fig 17.15b), forces on this ride model:
      rear links push the chassis forward with Fx - m_unsprung_rear*a along the line
        wheel centre -> side-view instant centre: vertical part J = that * slope lifts
        the body at the rear axle and pushes the rear wheels DOWN (the anti-squat path,
        immediate);
      the diff mounts put the drive-torque reaction Fx * wheel-centre height on the
        chassis, nose-up;
      the rest of the nose-up moment comes from the link force acting below the sprung
        CG; the front unsprung mass is dragged along by the front links (same rule).
    Whatever the links do not carry reaches the tyre only as the body squats on the
    springs/dampers (the lagging path). Thrust rises as a half-cosine over
    thrust_rise_time_s (0 = step). Wheel spin inertia and tyre relaxation are NOT modelled.
    Returns per-REAR-WHEEL load gain vs time and the lag numbers.
    """
    if accel_g <= 0 or thrust_rise_time_s < 0 or duration_s <= 0 or dt_s <= 0:
        raise ValueError('positive acceleration/duration/step and nonnegative rise time required')
    model = ride_model_for(veh, inputs)
    t = np.arange(0., float(duration_s) + dt_s / 2, float(dt_s))
    rise = float(thrust_rise_time_s)
    frac = np.ones_like(t) if rise == 0 else np.where(t < rise, 0.5 * (1 - np.cos(np.pi * t / rise)), 1.)
    if rise == 0:
        frac[0] = 0.                                    # at rest at t=0, full thrust from the first step
    a = float(accel_g) * G_MPS2
    ms, muf, mur = float(veh.sprung_mass_kg), float(veh.unsprung_mass_front_kg), float(veh.unsprung_mass_rear_kg)
    fx = (ms + muf + mur) * a
    hs = float(veh.sprung_cg_height_m)
    rr, rf = float(rear_wheel_centre_height_m), float(front_wheel_centre_height_m)
    x_front, x_rear = float(model.body_corner_map[0, 1]), float(model.body_corner_map[2, 1])
    link_rear, link_front = fx - mur * a, muf * a       # horizontal link force on the chassis (rear fwd, front aft)
    jack_rear = link_rear * float(rear_slope_from_wheel_centre)
    jack_front = link_front * float(front_slope_from_wheel_centre)
    unit = np.zeros(7)
    unit[0] = jack_rear + jack_front
    unit[1] = (link_rear * (hs - rr) + fx * rr - link_front * (hs - rf)
               + x_rear * jack_rear + x_front * jack_front)
    unit[3:5] = -jack_front / 2
    unit[5:7] = -jack_rear / 2
    resp = model.forced_response(t, frac[:, None] * unit[None, :])
    load = resp['dynamic_tire_load_N'][:, 2]            # RL = RR (symmetric)
    final = float(-np.linalg.solve(model.stiffness, unit)[5] * model.tire_rates[2])
    textbook = fx * float(veh.cg_height_m) / float(veh.wheelbase_m) / 2

    def first_time(series, level):
        hit = np.nonzero(series >= level)[0]
        return float(t[hit[0]]) if len(hit) else float('nan')

    t50, t90 = first_time(load, 0.5 * final), first_time(load, 0.9 * final)
    thrust50, thrust90 = first_time(frac, 0.5), first_time(frac, 0.9)
    return {
        'time_s': t, 'thrust_fraction': frac, 'rear_wheel_load_gain_N': load,
        'rear_squat_m': resp['suspension_displacement_m'][:, 2], 'pitch_rad': resp['coordinates'][:, 1],
        'final_load_gain_N': final, 'textbook_load_gain_N': textbook,
        'link_share': jack_rear / 2 / final if final else float('nan'),
        't50_s': t50, 't90_s': t90, 'lag50_s': t50 - thrust50, 'lag90_s': t90 - thrust90,
        # load the tyre is short of what the thrust already asks of it, integrated (N.s per wheel)
        'load_shortfall_Ns': float(np.trapezoid(np.clip(frac * final - load, 0., None), t)),
        'peak_overshoot_N': float(load.max() - final),
    }


def contact_patch_vs_ride_frequency(veh, inputs, frequencies_Hz, *, curve_frequencies_Hz=(2.0, 3.0, 3.5),
                                    hold_damping_ratio=False, transfer_frequencies_Hz=None):
    """Contact-patch (tyre normal load) deviation as ONE axle's ride frequency is varied,
    the other axle left as it is in the model. Road class / speeds / seeds from `inputs`.

    hold_damping_ratio=False keeps the wheel damping coefficients of `inputs`;
    True scales the varied axle's coefficient with sqrt(ride rate) so its damping RATIO
    stays that of the current car. Per axle returns RMS load deviation (% of static and N,
    mean over road cases), the lowest load seen (N), and the load-per-mm-of-road transfer
    curve for the current car and each curve frequency.
    """
    freqs = np.asarray(frequencies_Hz, float)
    if freqs.ndim != 1 or not len(freqs) or np.any(freqs <= 0):
        raise ValueError('positive ride-frequency vector required')
    fq = (np.linspace(0.3, 30., 600) if transfer_frequencies_Hz is None
          else np.asarray(transfer_frequencies_Hz, float))
    cases = build_road_cases(inputs, veh.wheelbase_m)
    mr = corner_motion_ratios(veh)
    masses = dict(zip(AXLES, sprung_corner_masses_kg(veh)))
    k_now = {'front': float(veh.ride_rate_front_Npm), 'rear': float(veh.ride_rate_rear_Npm)}
    out = {'frequencies_Hz': freqs, 'transfer_frequency_Hz': fq, 'hold_damping_ratio': bool(hold_damping_ratio),
           'road_class': inputs.road_class}
    for axle, idx, ids in (('front', 0, (0, 1)), ('rear', 2, (2, 3))):
        f_now = ride_frequency_Hz(k_now[axle], masses[axle])

        def model_at(f_hz):
            k = dict(k_now); k[axle] = ride_rate_from_frequency_Npm(float(f_hz), masses[axle])
            damp = list(inputs.corner_damping_Nspm)
            if hold_damping_ratio:
                for i in ids:
                    damp[i] *= (k[axle] / k_now[axle]) ** 0.5
            from dataclasses import replace
            return ride_model_for(vehicle_with_ride_rates(veh, k['front'], k['rear']),
                                  replace(inputs, corner_damping_Nspm=tuple(damp)))

        pct, rms, low = [], [], []
        for f_hz in freqs:
            model = model_at(f_hz)
            ms = [evaluate_case(model, c, motion_ratios=mr, stop_margin_m=inputs.stop_margin_m, inputs=inputs) for c in cases]
            pct.append(100. * float(np.mean([m.dlc[idx] for m in ms])))
            rms.append(float(np.mean([m.load_rms_N[idx] for m in ms])))
            low.append(float(np.min([m.min_load_N[idx] for m in ms])))
        curves = {}
        for f_hz in [f_now] + [float(v) for v in curve_frequencies_Hz]:
            h = model_at(f_hz).transfer(fq)['dynamic_tire_load_N'][:, idx, idx]
            curves[round(float(f_hz), 3)] = np.abs(h) / 1000.           # N per mm of road
        out[axle] = {'current_Hz': f_now, 'static_load_N': float(model_at(f_now).baseline_loads[idx]),
                     'rms_pct_of_static': np.array(pct), 'rms_N': np.array(rms), 'min_load_N': np.array(low),
                     'load_per_mm_road_N': curves}
    return out


def launch_lag_study(veh, inputs, *, rear_slope_from_wheel_centre, front_slope_from_wheel_centre,
                     rear_wheel_centre_height_m, front_wheel_centre_height_m, accel_g=1.0,
                     thrust_rise_times_s=(0., 0.1, 0.25), anti_squat_pcts=(0., 30., 60., 100.)):
    """launch_load_lag for the car AS BUILT plus reference anti-squat percentages, per thrust rise time.
    A reference percentage p uses the slope p/100 * CG height / wheelbase (RCVD fig 17.15b)."""
    ratio = float(veh.cg_height_m) / float(veh.wheelbase_m)
    as_built = float(rear_slope_from_wheel_centre) / ratio * 100.
    slopes = [('as built', as_built, float(rear_slope_from_wheel_centre))]
    slopes += [(f'{p:.0f} %', float(p), float(p) / 100. * ratio) for p in anti_squat_pcts]
    out = {'as_built_anti_squat_pct': as_built, 'accel_g': float(accel_g), 'cases': []}
    for rise in thrust_rise_times_s:
        for label, pct, slope in slopes:
            r = launch_load_lag(veh, inputs, accel_g=accel_g, thrust_rise_time_s=float(rise),
                                rear_slope_from_wheel_centre=slope,
                                front_slope_from_wheel_centre=front_slope_from_wheel_centre,
                                rear_wheel_centre_height_m=rear_wheel_centre_height_m,
                                front_wheel_centre_height_m=front_wheel_centre_height_m, duration_s=0.8)
            r.update(label=label, anti_squat_pct=pct, thrust_rise_time_s=float(rise))
            out['cases'].append(r)
    return out


def flat_ride_ratio(veh):
    """Undamped rear/front ride-frequency ratio on the current ride rates —
    RCVD p.410 wants it slightly above 1 (about 1.10)."""
    mf, mr = sprung_corner_masses_kg(veh)
    return (ride_frequency_Hz(veh.ride_rate_rear_Npm, mr)
            / ride_frequency_Hz(veh.ride_rate_front_Npm, mf))


def ride_rate_grid_Npm(veh, axle, freq_min_Hz, freq_max_Hz, count):
    """Evenly spaced ride FREQUENCIES converted to ride rates for one axle
    (the sweep axis the page asks for). Returns (rates_Npm, freqs_Hz)."""
    if axle not in AXLES:
        raise ValueError('axle must be front or rear')
    lo, hi, n = float(freq_min_Hz), float(freq_max_Hz), int(count)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi < lo or n < 1:
        raise ValueError('need 0 < f_min <= f_max and at least one point')
    mass = sprung_corner_masses_kg(veh)[0 if axle == 'front' else 1]
    freqs = np.linspace(lo, hi, n) if n > 1 else np.array([lo])
    return np.array([ride_rate_from_frequency_Npm(f, mass) for f in freqs]), freqs


def current_car_study(veh, inputs, transfer_frequencies_Hz=None):
    """Everything the page shows for the CURRENT springs: per-case metrics,
    the bump-settle traces per speed, transfer curves and the road PSD
    responses per speed and coherence. All on the one VehicleParams."""
    if inputs.damper is not None:
        return _current_car_study_dyno(veh, inputs, transfer_frequencies_Hz)
    model = ride_model_for(veh, inputs)
    cases = build_road_cases(inputs, veh.wheelbase_m)
    mr = corner_motion_ratios(veh)
    metrics = [evaluate_case(model, c, motion_ratios=mr, stop_margin_m=inputs.stop_margin_m, inputs=inputs)
               for c in cases]
    bumps = {float(v): bump_settling(model, inputs, v, veh.wheelbase_m) for v in inputs.speeds_mps}
    if transfer_frequencies_Hz is None:
        transfer_frequencies_Hz = np.logspace(-1, np.log10(60.), 600)
    psd = {}
    for v in inputs.speeds_mps:
        for coh in inputs.coherence_values:
            f, out = road_psd_response(model, inputs, v, veh.wheelbase_m,
                                       1. if coh == 'coherent' else 0.)
            psd[(float(v), coh)] = (f, out)
    mf, mr_mass = sprung_corner_masses_kg(veh)
    return {
        'model': model, 'cases': cases, 'metrics': metrics, 'bumps': bumps,
        'transfer': transfer_curves(model, transfer_frequencies_Hz), 'psd': psd,
        'front_ride_frequency_Hz': ride_frequency_Hz(veh.ride_rate_front_Npm, mf),
        'rear_ride_frequency_Hz': ride_frequency_Hz(veh.ride_rate_rear_Npm, mr_mass),
        'flat_ride_ratio': flat_ride_ratio(veh),
    }


def _current_car_study_dyno(veh, inputs, transfer_frequencies_Hz=None):
    """current_car_study with dyno dampers (see the module docstring)."""
    cases = build_road_cases(inputs, veh.wheelbase_m)
    mr = corner_motion_ratios(veh)
    eq = [equivalent_damping_for_case(veh, inputs, c) for c in cases]
    metrics = [time_domain_case_metrics(e['model'], c, inputs.damper, motion_ratios=mr,
                                        stop_margin_m=inputs.stop_margin_m,
                                        discard_s=inputs.time_domain_discard_s, inputs=inputs)
               for e, c in zip(eq, cases)]
    ref = reference_case_index(cases, inputs.speeds_mps)
    model = eq[ref]['model']
    bumps = {}
    for v in inputs.speeds_mps:
        bumps[float(v)] = bump_settling(eq[_speed_case_index(cases, v)]['model'], inputs, v, veh.wheelbase_m)
    if transfer_frequencies_Hz is None:
        transfer_frequencies_Hz = np.logspace(-1, np.log10(60.), 600)
    psd = {}
    for v in inputs.speeds_mps:
        for coh in inputs.coherence_values:
            at = [i for i, c in enumerate(cases)
                  if np.isclose(c.speed_mps, v) and c.coherence == coh]
            f, out = road_psd_response(eq[at[0]]['model'], inputs, v, veh.wheelbase_m,
                                       1. if coh == 'coherent' else 0.)
            psd[(float(v), coh)] = (f, out)
    mf, mr_mass = sprung_corner_masses_kg(veh)
    return {
        'model': model, 'cases': cases, 'metrics': metrics, 'bumps': bumps,
        'transfer': transfer_curves(model, transfer_frequencies_Hz), 'psd': psd,
        'front_ride_frequency_Hz': ride_frequency_Hz(veh.ride_rate_front_Npm, mf),
        'rear_ride_frequency_Hz': ride_frequency_Hz(veh.ride_rate_rear_Npm, mr_mass),
        'flat_ride_ratio': flat_ride_ratio(veh),
        'metrics_equivalent_linear': [e['metrics'] for e in eq],
        'damper_equivalent': [{k: e[k] for k in ('c_wheel_Nspm', 'c_comp_Nspm', 'c_reb_Nspm',
                                                  'damper_v_rms_mps', 'iterations', 'converged',
                                                  'change')} for e in eq],
        'reference_case_index': ref,
        'damping_kind': 'dyno',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep + selection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RideSweep:
    front_rates_Npm: np.ndarray
    rear_rates_Npm: np.ndarray
    case_labels: list
    speeds_mps: tuple
    dlc: np.ndarray                    # (nf, nr, ncase, 4)
    travel_usage: np.ndarray           # (nf, nr, ncase, 4)
    travel_peak_m: np.ndarray          # (nf, nr, ncase, 4)
    damper_velocity_rms_mps: np.ndarray   # (nf, nr, ncase, 4)
    damper_velocity_peak_mps: np.ndarray  # (nf, nr, ncase, 4)
    contact_loss_fraction_linear: np.ndarray  # (nf, nr, ncase, 4)
    min_load_N: np.ndarray             # (nf, nr, ncase, 4)
    body_heave_acc_rms_mps2: np.ndarray   # (nf, nr, ncase)
    settle_pass: np.ndarray            # (nf, nr, nspeed) bool: both axles settled inside the record
    pitch_to_bounce: np.ndarray        # (nf, nr, nspeed) RMS pitch / RMS bounce after the bump
    front_settle_s: np.ndarray         # (nf, nr, nspeed) information only
    rear_settle_s: np.ndarray          # (nf, nr, nspeed) information only
    stop_margin_m: float
    flat_ride_max_ratio: float = 1.0
    case_speeds_mps: tuple = ()        # speed of each case (same order as case_labels)
    # TYRE-CONTACT acceptance (audit 2026-09-22 item 27).  The linear model
    # lets a tyre load go negative (it "pulls the road down"); such a point
    # is not a valid ride setup, however good its DLC.  A pair is contact-OK
    # only if every corner in every case keeps min load >= min_tire_load_N
    # AND the fraction of samples at Fz <= 0 is <= max_contact_loss_fraction.
    min_tire_load_N: float = 0.0
    max_contact_loss_fraction: float = 0.0

    def cases_at_speed(self, speed_mps):
        """Indices of the road cases (seeds x coherence) run at one speed."""
        return [c for c, v in enumerate(self.case_speeds_mps) if np.isclose(v, speed_mps)]

    @property
    def worst_dlc(self):
        """Worst corner over every case, per (front, rear) grid point."""
        return self.dlc.max(axis=(2, 3))

    @property
    def worst_travel_usage(self):
        return self.travel_usage.max(axis=(2, 3))

    @property
    def travel_ok(self):
        return self.worst_travel_usage <= 1.

    @property
    def settle_ok(self):
        """Record long enough: both axles settled at every speed (validity)."""
        return self.settle_pass.all(axis=2)

    @property
    def mean_pitch_to_bounce(self):
        """Flat-ride number averaged over the speed bracket."""
        return self.pitch_to_bounce.mean(axis=2)

    @property
    def flat_ride_ok(self):
        return (self.mean_pitch_to_bounce <= self.flat_ride_max_ratio) & self.settle_ok

    @property
    def worst_min_load_N(self):
        """Lowest tyre load (linear model) over every case and corner."""
        return self.min_load_N.min(axis=(2, 3))

    @property
    def worst_contact_loss_fraction(self):
        return self.contact_loss_fraction_linear.max(axis=(2, 3))

    @property
    def contact_ok(self):
        """Every tyre stays on the road in the linear screening."""
        return ((self.worst_min_load_N >= self.min_tire_load_N)
                & (self.worst_contact_loss_fraction <= self.max_contact_loss_fraction))

    @property
    def feasible(self):
        return self.travel_ok & self.flat_ride_ok & self.contact_ok

    def select(self):
        """Grid point minimising worst-corner DLC among points that meet the
        travel margin, the flat-ride threshold AND tyre contact; if none
        meets flat ride, the best travel+contact point (status says so); if
        no travel-feasible point keeps contact, the best travel-feasible
        point flagged CONTACT LOST (never 'feasible'); if nothing fits the
        travel margin, the unconstrained minimum."""
        worst = self.worst_dlc
        feasible = self.feasible
        travel_contact = self.travel_ok & self.contact_ok
        if feasible.any():
            mask, status = feasible, 'feasible'
        elif travel_contact.any():
            mask = travel_contact
            status = ('flat-ride threshold not met anywhere on the grid at these '
                      'dampers/inertias — best travel-feasible point shown')
        elif self.travel_ok.any():
            mask = self.travel_ok
            _lo = float(np.where(self.travel_ok, self.worst_min_load_N, -np.inf).max())
            status = ('TYRE CONTACT LOST at every travel-feasible point (linear model: '
                      f'best minimum tyre load {_lo:.0f} N) — best travel-feasible point '
                      'shown, NOT a valid setup')
        else:
            mask = np.ones_like(feasible)
            status = 'NO point fits the travel margin — unconstrained minimum shown'
        masked = np.where(mask, worst, np.inf)
        i, j = np.unravel_index(int(np.argmin(masked)), worst.shape)
        return {'i_front': int(i), 'j_rear': int(j), 'status': status,
                'front_ride_rate_Npm': float(self.front_rates_Npm[i]),
                'rear_ride_rate_Npm': float(self.rear_rates_Npm[j]),
                'worst_dlc': float(worst[i, j]),
                'dlc_grid_min': float(worst.min()), 'dlc_grid_max': float(worst.max()),
                'worst_travel_usage': float(self.worst_travel_usage[i, j]),
                'mean_pitch_to_bounce': float(self.mean_pitch_to_bounce[i, j]),
                'worst_pitch_to_bounce': float(self.pitch_to_bounce[i, j].max()),
                'flat_ride_ok': bool(self.flat_ride_ok[i, j]),
                'settle_ok': bool(self.settle_ok[i, j]),
                'travel_ok': bool(self.travel_ok[i, j]),
                'contact_ok': bool(self.contact_ok[i, j]),
                'worst_min_load_N': float(self.worst_min_load_N[i, j]),
                'worst_contact_loss_fraction': float(self.worst_contact_loss_fraction[i, j]),
                'n_feasible': int(feasible.sum()), 'n_grid': int(feasible.size)}


def ride_model_for(veh, inputs, corner_damping_Nspm=None):
    damping = inputs.corner_damping_Nspm if corner_damping_Nspm is None else corner_damping_Nspm
    return RideModel.from_vehicle(veh, pitch_inertia_kgm2=inputs.pitch_inertia_kgm2,
                                  roll_inertia_kgm2=inputs.roll_inertia_kgm2,
                                  corner_damping_Nspm=list(damping))


def equivalent_damping_for_case(veh, inputs, case):
    """Equivalent linear wheel coefficients of inputs.damper on one road
    case: c = c_eq(RMS damper velocity of THIS case) x MR², iterated from
    inputs.corner_damping_Nspm until the relative change is <= damper_tol
    (at most damper_max_iter solves).  Returns a dict: c_wheel_Nspm (4),
    c_comp/c_reb (4), damper_v_rms_mps (4), iterations, converged,
    change, history, model, metrics (frequency-domain at the final c)."""
    damper = inputs.damper
    if damper is None:
        raise ValueError('no damper model in the ride inputs')
    mr = corner_motion_ratios(veh)
    c = np.asarray(inputs.corner_damping_Nspm, float).copy()
    history, converged, change = [], False, np.inf
    comp = reb = v_rms = None
    for it in range(1, int(inputs.damper_max_iter) + 1):
        model = ride_model_for(veh, inputs, c)
        m = evaluate_case(model, case, motion_ratios=mr, stop_margin_m=inputs.stop_margin_m, inputs=inputs)
        v_rms = np.maximum(m.damper_velocity_rms_mps, 1e-5)
        c_new, comp, reb = damper.equivalent_wheel_c(v_rms)
        change = float(np.max(np.abs(c_new - c) / c_new))
        history.append({'c_wheel_Nspm': c.copy(), 'damper_v_rms_mps': v_rms.copy(), 'change': change})
        c = c_new
        if change <= inputs.damper_tol:
            converged = True
            break
    model = ride_model_for(veh, inputs, c)
    metrics = evaluate_case(model, case, motion_ratios=mr, stop_margin_m=inputs.stop_margin_m, inputs=inputs)
    metrics.kind = 'equivalent linear (frequency domain)'
    metrics.beyond_dyno_mps = float(damper.curve.beyond_dyno_mps(
        np.r_[metrics.damper_velocity_peak_mps, -metrics.damper_velocity_peak_mps]))
    return {'c_wheel_Nspm': c, 'c_comp_Nspm': comp, 'c_reb_Nspm': reb,
            'damper_v_rms_mps': metrics.damper_velocity_rms_mps, 'iterations': len(history),
            'converged': converged, 'change': change, 'history': history,
            'model': model, 'metrics': metrics}


def time_domain_case_metrics(model, case, damper, *, motion_ratios, stop_margin_m, discard_s=2.0,
                             inputs=None):
    """RideMetrics of one road case from the time-domain integration with
    the NONLINEAR wheel damper law and the unilateral tyre (RK4). The first
    discard_s seconds (start from rest) are excluded from every statistic.
    `inputs` with aero: about the loaded state at the case speed (evaluate_case)."""
    mr = np.asarray(motion_ratios, float)
    sink = np.zeros(4)
    if inputs is not None:
        model, sink = loaded_model(model, inputs, case.speed_mps)
    road = case.road_heights_m
    t = np.arange(len(road)) * case.dt_s
    res = model.contact_response(t, road, damper_force_fn=damper, integrator='rk4')
    keep = t >= min(float(discard_s), 0.5 * t[-1])
    load = res['normal_load_N'][keep]
    mean = load.mean(axis=0)
    rms = np.sqrt(np.mean((load - mean) ** 2, axis=0))
    travel = res['suspension_displacement_m'][keep]
    from_rest, usage = _stop_usage(travel + sink, stop_margin_m, inputs)
    vel = (res['velocity'] @ model.suspension_map.T)[keep] * mr
    acc = res['body_acceleration'][keep]
    return RideMetrics(
        dlc=rms / mean, load_rms_N=rms, min_load_N=load.min(axis=0),
        contact_loss_fraction_linear=np.mean(load <= 0., axis=0),
        travel_peak_m=from_rest,
        travel_rms_m=np.sqrt(np.mean(travel ** 2, axis=0)),
        travel_usage=usage,
        aero_sink_m=sink,
        damper_velocity_rms_mps=np.sqrt(np.mean(vel ** 2, axis=0)),
        damper_velocity_peak_mps=np.max(np.abs(vel), axis=0),
        body_heave_acc_rms_mps2=float(np.sqrt(np.mean(acc[:, 0] ** 2))),
        body_pitch_acc_rms_radps2=float(np.sqrt(np.mean(acc[:, 1] ** 2))),
        body_roll_acc_rms_radps2=float(np.sqrt(np.mean(acc[:, 2] ** 2))),
        body_corner_acc_rms_mps2=np.sqrt(np.mean(res['body_corner_acceleration_mps2'][keep] ** 2, axis=0)),
        kind='time domain, nonlinear dampers + unilateral tyre',
        beyond_dyno_mps=float(damper.curve.beyond_dyno_mps(vel.ravel())))


def load_swing_vs_speed(veh, inputs, speeds_mps, road_classes=('A', 'B')):
    """Tyre-load swing against SPEED (the page's headline plot and the
    binder's table): for every road class and speed, the worse of the
    coherent/independent tracks per corner.  Frequency domain on the
    equivalent linear dampers (dyno: iterated per case; no dyno: the four
    coefficients).  Loads include the downforce at that speed when the
    inputs carry an aero package.  Returns {class: {'speed_kph', 'mean_N',
    'rms_N', 'pct', 'min_N', 'c_wheel_Nspm'}} with (n_speed, 4) arrays."""
    mr = corner_motion_ratios(veh)
    out = {}
    for cls in road_classes:
        inp = replace(inputs, road_class=cls, speeds_mps=tuple(float(v) for v in speeds_mps),
                      seeds=inputs.seeds[:1], coherence='both')
        cases = build_road_cases(inp, veh.wheelbase_m)
        rows = {k: np.zeros((len(speeds_mps), 4)) for k in ('mean_N', 'rms_N', 'min_N', 'c_wheel_Nspm')}
        rows['rms_N'][:] = -np.inf; rows['min_N'][:] = np.inf
        for case in cases:
            s = int(np.argmin(np.abs(np.asarray(speeds_mps, float) - case.speed_mps)))
            if inp.damper is not None:
                e = equivalent_damping_for_case(veh, inp, case)
                m, c = e['metrics'], e['c_wheel_Nspm']
            else:
                m = evaluate_case(ride_model_for(veh, inp), case, motion_ratios=mr,
                                  stop_margin_m=inp.stop_margin_m, inputs=inp)
                c = np.asarray(inp.corner_damping_Nspm, float)
            worse = m.load_rms_N > rows['rms_N'][s]
            rows['rms_N'][s] = np.where(worse, m.load_rms_N, rows['rms_N'][s])
            rows['c_wheel_Nspm'][s] = np.where(worse, c, rows['c_wheel_Nspm'][s])
            rows['mean_N'][s] = m.load_rms_N / m.dlc
            rows['min_N'][s] = np.minimum(rows['min_N'][s], m.min_load_N)
        rows['pct'] = 100. * rows['rms_N'] / rows['mean_N']
        rows['speed_kph'] = np.asarray(speeds_mps, float) * 3.6
        out[cls] = rows
    return out


def quarter_car_load_rms_N(sprung_kg, unsprung_kg, wheel_rate_Npm, wheel_damping_Nspm,
                           tire_rate_Npm, road_class, speed_mps, n=200000):
    """INDEPENDENT cross-check of the tyre-load swing: one corner alone (body
    on spring + damper, wheel on the tyre), written out here from the two
    equations of motion and integrated against the ISO road PSD — shares no
    matrices with RideModel.  RMS dynamic tyre load (N)."""
    sp = RoadSpectrum(road_class)
    v = float(speed_mps)
    f = np.linspace(sp.low_cycles_per_m * v, sp.high_cycles_per_m * v, int(n))
    w = 2 * np.pi * f
    S = sp.temporal_psd(f, v)
    k, c, kt = float(wheel_rate_Npm), float(wheel_damping_Nspm), float(tire_rate_Npm)
    a11 = -w ** 2 * float(sprung_kg) + 1j * w * c + k
    a12 = -(1j * w * c + k)
    a22 = -w ** 2 * float(unsprung_kg) + 1j * w * c + k + kt
    zu = a11 * kt / (a11 * a22 - a12 ** 2)          # wheel motion per unit road
    return float(np.sqrt(np.trapezoid(np.abs(kt * (1. - zu)) ** 2 * S, f)))


# ─────────────────────────────────────────────────────────────────────────────
#  Springs x dampers x speed: tyre load AND contact-patch location
# ─────────────────────────────────────────────────────────────────────────────
# The fixed reference road (user 2026-10-23: "consistent bumps"): one stored
# seed per track, built once in DISTANCE, so every spring, damper and speed
# drives over the very same bumps.  Road B uses the same seed, i.e. the same
# bumps twice as tall (ISO 8608 B = 4x the A power, same n^-2 slope).
REFERENCE_ROAD_SEED = 1
ROAD_A_NOTE = ('Road A = ISO 8608 class A (very good road). Used as the STAND-IN for a smooth FSAE '
               'site such as MIS - it is NOT a measured MIS surface.')


def reference_road(road_class, record_length_m=1000., spacing_m=0.02):
    """The fixed reference road of one class: distance (m), left and right
    track heights (m), and the bump content per spatial-frequency band
    (octaves from the ISO band edge up): band edges in cycles/m and the RMS
    height of each band (m).  Same arrays the road cases replay."""
    sp = RoadSpectrum(road_class)
    inp = RideInputs(pitch_inertia_kgm2=1., roll_inertia_kgm2=1., corner_damping_Nspm=(1., 1., 1., 1.),
                     road_class=road_class, record_length_m=record_length_m, spacing_m=spacing_m)
    left, right = _profiles(inp, REFERENCE_ROAD_SEED)
    x = np.arange(len(left)) * float(spacing_m)
    edges = [sp.low_cycles_per_m]
    while edges[-1] * 2. < sp.high_cycles_per_m:
        edges.append(edges[-1] * 2.)
    if sp.high_cycles_per_m / edges[-1] < 1.5:       # no sliver band at the top
        edges.pop()
    edges.append(sp.high_cycles_per_m)
    edges = np.asarray(edges)
    zf = np.fft.rfft(left); n = np.fft.rfftfreq(len(left), spacing_m)
    band_rms = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (n >= lo) & (n < hi)
        band_rms.append(float(np.sqrt(2. * np.sum(np.abs(zf[keep]) ** 2)) / len(left)))
    return {'distance_m': x, 'left_m': left, 'right_m': right, 'band_edges_cpm': edges,
            'band_rms_m': np.asarray(band_rms), 'band_rms_theory_m': np.sqrt(sp.band_power(edges[:-1], edges[1:])),
            'rms_m': float(np.std(left)), 'seed': REFERENCE_ROAD_SEED}


def contact_patch_table(solver, tire_radius_m, travel_limit_m=0.06, n=61):
    """Contact-patch position against wheel travel from the SOLVED corner
    (vahan.solver.SuspensionConstraints.solve, the same solve the 3-D view
    draws), chassis frame: travel (m, + = bump), lateral and fore-aft
    movement from the static position (m; lateral + = outboard, fore-aft
    + = rearward).  Solved outward from static with warm starts; stops at
    the first travel the solver cannot reach."""
    from vahan.loads import contact_patch_point
    st0 = solver.solve(0.)
    p0 = contact_patch_point(st0.wheel_center, st0.spin_axis, tire_radius_m)
    side = 1. if p0[0] >= 0 else -1.
    rows = {0.: np.zeros(2)}
    for sign in (1., -1.):
        x0 = st0.x_vec()
        for t in np.linspace(0., sign * travel_limit_m, n // 2 + 1)[1:]:
            try:
                st = solver.solve(float(t), x0=x0)
            except Exception:
                break
            p = contact_patch_point(st.wheel_center, st.spin_axis, tire_radius_m)
            if not np.all(np.isfinite(p)):
                break
            rows[float(t)] = np.array([side * (p[0] - p0[0]), p[1] - p0[1]])
            x0 = st.x_vec()
    t = np.array(sorted(rows))
    xy = np.array([rows[k] for k in t])
    return {'travel_m': t, 'lateral_m': xy[:, 0], 'fore_aft_m': xy[:, 1]}


def ride_rate_options(veh, front_frequencies_Hz):
    """Ride-rate choices for the spring x damper study: each front ride
    frequency with the rear at the CURRENT rear/front frequency ratio.
    Returns dicts with Hz, ride rates (N/m) and the corner spring (N/m and
    lbf/in) on this car's own wheel-rate chain."""
    mf, mr = sprung_corner_masses_kg(veh)
    ff0 = ride_frequency_Hz(veh.ride_rate_front_Npm, mf)
    ratio = ride_frequency_Hz(veh.ride_rate_rear_Npm, mr) / ff0
    out = []
    for ff in front_frequencies_Hz:
        fr = float(ff) * ratio
        rf, rr = ride_rate_from_frequency_Npm(ff, mf), ride_rate_from_frequency_Npm(fr, mr)
        kf = spring_rate_for_ride_rate_Npm(veh, 'front', rf)
        kr = spring_rate_for_ride_rate_Npm(veh, 'rear', rr)
        out.append({'front_Hz': float(ff), 'rear_Hz': fr, 'ride_front_Npm': rf, 'ride_rear_Npm': rr,
                    'spring_front_Npm': kf, 'spring_rear_Npm': kr,
                    'spring_front_lbfin': kf / LBF_PER_IN_TO_NPM, 'spring_rear_lbfin': kr / LBF_PER_IN_TO_NPM,
                    'is_current': bool(abs(ff - ff0) < 1e-3)})
    return out


def spring_damper_speed_study(veh, inputs, rate_options, dampers, speeds_mps, cp_tables,
                              road_classes=('A', 'B'), progress=None):
    """Tyre-load swing AND contact-patch movement on the fixed reference road
    for every spring choice x damper setting x speed x road class.

    rate_options: ride_rate_options(); dampers: [(label, WheelDampers)];
    cp_tables: four contact_patch_table() dicts FL FR RL RR.  Per case the
    equivalent linear dampers are iterated (equivalent_damping_for_case),
    the steady periodic response gives each corner's wheel travel (plus the
    aero sink at that speed) and the solved kinematic table turns it into
    contact-patch lateral / fore-aft position.  Worse of same/different
    left-right bumps per corner.  Returns arrays (n_rate, n_damper, n_speed, 4)
    per class: load_rms_N, load_pct, min_load_N, lat_rms_mm, lat_p99_mm,
    long_rms_mm, long_p99_mm, travel_rms_mm, c_wheel_Nspm; plus speed_kph.
    *_p99 = the movement exceeded only 1 % of the time (steadier than the
    single biggest sample of one record).  min_load_N <= 0 = the tyre leaves
    the road somewhere in the run: the straight-line (equivalent linear)
    model is not valid there and the page marks it."""
    speeds = tuple(float(v) for v in speeds_mps)
    keys = ('load_rms_N', 'load_pct', 'min_load_N', 'lat_rms_mm', 'lat_p99_mm',
            'long_rms_mm', 'long_p99_mm', 'travel_rms_mm', 'c_wheel_Nspm')
    shape = (len(rate_options), len(dampers), len(speeds), 4)
    total = len(road_classes) * shape[0] * shape[1] * len(speeds) * 2
    done = 0
    out = {}
    for cls in road_classes:
        res = {k: np.full(shape, -np.inf) for k in keys}
        res['min_load_N'][:] = np.inf
        base = replace(inputs, road_class=cls, speeds_mps=speeds, seeds=(REFERENCE_ROAD_SEED,), coherence='both')
        cases = build_road_cases(base, veh.wheelbase_m)
        for i, ro in enumerate(rate_options):
            vk = replace(veh, spring_rate_front_Npm=ro['spring_front_Npm'], spring_rate_rear_Npm=ro['spring_rear_Npm'])
            for j, (_, wd) in enumerate(dampers):
                inp = replace(base, damper=wd)
                for case in cases:
                    s = speeds.index(case.speed_mps)
                    e = equivalent_damping_for_case(vk, inp, case)
                    model, sink = loaded_model(e['model'], inp, case.speed_mps)
                    resp = model.periodic_response(case.dt_s, case.road_heights_m)
                    travel = resp['suspension_displacement_m'] + sink
                    m = e['metrics']
                    vals = {'load_rms_N': m.load_rms_N, 'load_pct': 100. * m.dlc,
                            'travel_rms_mm': 1000. * m.travel_rms_m, 'c_wheel_Nspm': e['c_wheel_Nspm']}
                    lat = np.zeros(4); latp = np.zeros(4); lon = np.zeros(4); lonp = np.zeros(4)
                    for c in range(4):
                        tb = cp_tables[c]
                        y = np.interp(travel[:, c], tb['travel_m'], tb['lateral_m'])
                        z = np.interp(travel[:, c], tb['travel_m'], tb['fore_aft_m'])
                        lat[c], latp[c] = np.std(y), np.percentile(np.abs(y - y.mean()), 99.)
                        lon[c], lonp[c] = np.std(z), np.percentile(np.abs(z - z.mean()), 99.)
                    vals.update(lat_rms_mm=1000. * lat, lat_p99_mm=1000. * latp,
                                long_rms_mm=1000. * lon, long_p99_mm=1000. * lonp)
                    worse = m.load_rms_N > res['load_rms_N'][i, j, s]
                    for k, v in vals.items():
                        if k in ('lat_rms_mm', 'lat_p99_mm', 'long_rms_mm', 'long_p99_mm', 'travel_rms_mm'):
                            res[k][i, j, s] = np.maximum(res[k][i, j, s], v)
                        else:
                            res[k][i, j, s] = np.where(worse, v, res[k][i, j, s])
                    res['min_load_N'][i, j, s] = np.minimum(res['min_load_N'][i, j, s], m.min_load_N)
                    done += 1
                    if progress is not None:
                        progress(done, total)
        res['speed_kph'] = np.asarray(speeds) * 3.6
        out[cls] = res
    return out


def reference_case_index(cases, speeds_mps):
    """The case whose equivalent coefficients stand for the car in the
    speed-independent outputs: middle speed of the bracket, coherent if
    present, first seed."""
    v_mid = sorted(float(v) for v in speeds_mps)[(len(speeds_mps) - 1) // 2]
    at = [i for i, c in enumerate(cases) if np.isclose(c.speed_mps, v_mid)]
    coh = [i for i in at if cases[i].coherence == 'coherent']
    return (coh or at or [0])[0]


def _speed_case_index(cases, speed_mps):
    at = [i for i, c in enumerate(cases) if np.isclose(c.speed_mps, speed_mps)]
    coh = [i for i in at if cases[i].coherence == 'coherent']
    return (coh or at)[0]


def corner_motion_ratios(veh):
    mf, mr = float(veh.motion_ratio_front), float(veh.motion_ratio_rear)
    return np.array([mf, mf, mr, mr])


def sweep_ride_rates(veh, inputs, front_rates_Npm, rear_rates_Npm, *,
                     progress=None, cancelled=None):
    """Evaluate every (front, rear) ride-rate pair on every road case and
    speed. `progress(done, total)` is called after each grid point;
    `cancelled()` returning True aborts with RuntimeError."""
    fr = np.asarray(front_rates_Npm, float); rr = np.asarray(rear_rates_Npm, float)
    if fr.ndim != 1 or rr.ndim != 1 or not len(fr) or not len(rr):
        raise ValueError('ride-rate grids must be non-empty vectors')
    if np.any(fr <= 0) or np.any(rr <= 0):
        raise ValueError('ride rates must be positive')
    cases = build_road_cases(inputs, veh.wheelbase_m)
    mr = corner_motion_ratios(veh)
    nf, nr, nc, ns = len(fr), len(rr), len(cases), len(inputs.speeds_mps)
    shape4 = (nf, nr, nc, 4)
    out = dict(dlc=np.zeros(shape4), travel_usage=np.zeros(shape4), travel_peak_m=np.zeros(shape4),
               damper_velocity_rms_mps=np.zeros(shape4), damper_velocity_peak_mps=np.zeros(shape4),
               contact_loss_fraction_linear=np.zeros(shape4), min_load_N=np.zeros(shape4),
               body_heave_acc_rms_mps2=np.zeros((nf, nr, nc)),
               settle_pass=np.zeros((nf, nr, ns), bool), pitch_to_bounce=np.zeros((nf, nr, ns)),
               front_settle_s=np.zeros((nf, nr, ns)), rear_settle_s=np.zeros((nf, nr, ns)))
    done, total = 0, nf * nr
    for i, kf in enumerate(fr):
        for j, kr in enumerate(rr):
            if cancelled is not None and cancelled():
                raise RuntimeError('ride sweep cancelled')
            point_veh = vehicle_with_ride_rates(veh, kf, kr)
            if inputs.damper is None:
                model = ride_model_for(point_veh, inputs)
                case_models = None
            else:      # equivalent linear dampers per case (time domain is the study's)
                eq = [equivalent_damping_for_case(point_veh, inputs, case) for case in cases]
                case_models = [e['model'] for e in eq]
            for c, case in enumerate(cases):
                if case_models is None:
                    m = evaluate_case(model, case, motion_ratios=mr, stop_margin_m=inputs.stop_margin_m, inputs=inputs)
                else:
                    m = eq[c]['metrics']
                out['dlc'][i, j, c] = m.dlc
                out['travel_usage'][i, j, c] = m.travel_usage
                out['travel_peak_m'][i, j, c] = m.travel_peak_m
                out['damper_velocity_rms_mps'][i, j, c] = m.damper_velocity_rms_mps
                out['damper_velocity_peak_mps'][i, j, c] = m.damper_velocity_peak_mps
                out['contact_loss_fraction_linear'][i, j, c] = m.contact_loss_fraction_linear
                out['min_load_N'][i, j, c] = m.min_load_N
                out['body_heave_acc_rms_mps2'][i, j, c] = m.body_heave_acc_rms_mps2
            for s, v in enumerate(inputs.speeds_mps):
                bmodel = model if case_models is None else case_models[_speed_case_index(cases, v)]
                b = bump_settling(bmodel, inputs, v, veh.wheelbase_m)
                out['settle_pass'][i, j, s] = b['settled_in_record']
                out['pitch_to_bounce'][i, j, s] = b['pitch_to_bounce_ratio']
                out['front_settle_s'][i, j, s] = b['front_settle_s']
                out['rear_settle_s'][i, j, s] = b['rear_settle_s']
            done += 1
            if progress is not None:
                progress(done, total)
    return RideSweep(fr, rr, [c.label for c in cases], tuple(inputs.speeds_mps),
                     stop_margin_m=float(inputs.stop_margin_m),
                     flat_ride_max_ratio=float(inputs.flat_ride_max_ratio),
                     case_speeds_mps=tuple(c.speed_mps for c in cases), **out)


def describe_selection(sweep, veh, inputs, *, preload_front_mm, preload_rear_mm, stroke_mm):
    """Numbers the page prints for the chosen pair: ride rates, frequencies,
    exact spring rates through the model's MR relation, standard springs."""
    sel = sweep.select()
    mass_f, mass_r = sprung_corner_masses_kg(veh)
    kf, kr = sel['front_ride_rate_Npm'], sel['rear_ride_rate_Npm']
    chosen = vehicle_with_ride_rates(veh, kf, kr)
    sel.update({
        'front_ride_frequency_Hz': ride_frequency_Hz(kf, mass_f),
        'rear_ride_frequency_Hz': ride_frequency_Hz(kr, mass_r),
        'flat_ride_ratio': ride_frequency_Hz(kr, mass_r) / ride_frequency_Hz(kf, mass_f),
        'front_wheel_rate_Npm': float(chosen.wheel_rate_front_Npm),
        'rear_wheel_rate_Npm': float(chosen.wheel_rate_rear_Npm),
        'front_spring_rate_Npm': float(chosen.spring_rate_front_Npm),
        'rear_spring_rate_Npm': float(chosen.spring_rate_rear_Npm),
        'front_spring_rate_lbf_in': float(chosen.spring_rate_front_Npm) / LBF_PER_IN_TO_NPM,
        'rear_spring_rate_lbf_in': float(chosen.spring_rate_rear_Npm) / LBF_PER_IN_TO_NPM,
        'motion_ratio_front': float(veh.motion_ratio_front),
        'motion_ratio_rear': float(veh.motion_ratio_rear),
        'current_front_ride_rate_Npm': float(veh.ride_rate_front_Npm),
        'current_rear_ride_rate_Npm': float(veh.ride_rate_rear_Npm),
        'current_front_spring_rate_Npm': float(veh.spring_rate_front_Npm),
        'current_rear_spring_rate_Npm': float(veh.spring_rate_rear_Npm),
        'sprung_corner_mass_front_kg': mass_f, 'sprung_corner_mass_rear_kg': mass_r,
        'standard_springs_front': standard_spring_options(
            veh, 'front', chosen.spring_rate_front_Npm, preload_mm=preload_front_mm, stroke_mm=stroke_mm),
        'standard_springs_rear': standard_spring_options(
            veh, 'rear', chosen.spring_rate_rear_Npm, preload_mm=preload_rear_mm, stroke_mm=stroke_mm),
    })
    return sel


def verify_contact(veh, inputs, front_ride_rate_Npm, rear_ride_rate_Npm, *, case_index=0,
                   length_m=None):
    """Nonlinear (compression-only tire) re-run of one road case at the
    chosen pair: the check the linear indicator cannot make. Returns the
    RideModel.contact_response dict plus per-corner DLC on the actual load."""
    cases = build_road_cases(inputs, veh.wheelbase_m)
    case = cases[int(case_index)]
    model = ride_model_for(vehicle_with_ride_rates(veh, front_ride_rate_Npm, rear_ride_rate_Npm), inputs)
    model, sink = loaded_model(model, inputs, case.speed_mps)
    n = len(case.road_heights_m) if length_m is None else int(round(float(length_m) / inputs.spacing_m))
    n = max(4, min(n, len(case.road_heights_m)))
    t = np.arange(n) * case.dt_s
    margin = float(inputs.stop_margin_m)
    bump = margin if inputs.bump_stop_margin_m is None else float(inputs.bump_stop_margin_m)
    result = model.contact_response(t, case.road_heights_m[:n],
                                    travel_limits_m=np.array([-margin - sink, bump - sink]),
                                    damper_force_fn=inputs.damper)
    load = result['normal_load_N']
    mean = load.mean(axis=0)
    result['dlc'] = np.sqrt(np.mean((load - mean) ** 2, axis=0)) / mean
    result['case_label'] = case.label
    return result
