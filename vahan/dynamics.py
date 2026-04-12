"""
vahan/dynamics.py — Steady-state vehicle dynamics

Computes lateral load transfer, body roll, per-corner vertical loads,
and suspension travel under steady-state cornering (and optionally
longitudinal acceleration).

Couples with the kinematic solver to capture geometry-dependent
effects: roll centre migration, camber change, motion ratio variation.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import uniform_filter1d
from .solver import SuspensionConstraints
from .kinematics import KinematicMetrics


# ─────────────────────────────────────────────────────────────────────────────
#  Vehicle parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleParams:
    """All vehicle-level parameters for dynamics computations."""

    # Mass (kg)
    total_mass_kg: float = 290.35
    sprung_mass_kg: float = 223.8
    unsprung_mass_front_kg: float = 26.5    # per axle (both wheels)
    unsprung_mass_rear_kg: float = 40.05

    # Geometry (m)
    wheelbase_m: float = 1.530
    front_track_m: float = 1.2218
    rear_track_m: float = 1.200
    cg_height_m: float = 0.26063
    cg_to_front_axle_m: float = 0.84118

    # Unsprung CG height (approximation: wheel center height)
    unsprung_cg_height_m: float = 0.203

    # Springs (N/m at the spring, not at the wheel)
    spring_rate_front_Npm: float = 22000.0
    spring_rate_rear_Npm: float = 22000.0
    motion_ratio_front: float = 0.97
    motion_ratio_rear: float = 0.82

    # Tire (N/m)
    tire_rate_Npm: float = 159100.0

    # ARB (N/m at the wheel)
    arb_rate_front_Npm: float = 18850.0
    arb_rate_rear_Npm: float = 8500.0

    # Powertrain
    power_hp: float = 0.0           # peak wheel horsepower
    engine_rpm: float = 0.0         # engine RPM (for speed + torque calc)
    total_drive_ratio: float = 10.0 # overall gear ratio (engine → wheel)
    tire_radius_m: float = 0.203    # loaded tire radius (m)
    drivetrain: str = 'RWD'         # 'RWD', 'FWD', or 'AWD'
    max_steer_angle_deg: float = 28.0  # max front wheel steer angle
    front_brake_bias: float = 0.65  # fraction of brake force on front axle (0-1)

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def cg_to_rear_axle_m(self):
        return self.wheelbase_m - self.cg_to_front_axle_m

    @property
    def front_weight_fraction(self):
        return self.cg_to_rear_axle_m / self.wheelbase_m

    @property
    def rear_weight_fraction(self):
        return self.cg_to_front_axle_m / self.wheelbase_m

    @property
    def wheel_rate_front_Npm(self):
        """Spring rate at the wheel = spring_rate * MR^2."""
        return self.spring_rate_front_Npm * self.motion_ratio_front ** 2

    @property
    def wheel_rate_rear_Npm(self):
        return self.spring_rate_rear_Npm * self.motion_ratio_rear ** 2

    @property
    def ride_rate_front_Npm(self):
        """Series combination of wheel rate and tire rate."""
        kw, kt = self.wheel_rate_front_Npm, self.tire_rate_Npm
        return (kw * kt) / (kw + kt)

    @property
    def ride_rate_rear_Npm(self):
        kw, kt = self.wheel_rate_rear_Npm, self.tire_rate_Npm
        return (kw * kt) / (kw + kt)

    @property
    def roll_stiffness_front_Npm_rad(self):
        """Roll stiffness from springs + ARB, front axle (N·m/rad).

        K_roll = (K_wheel + K_arb) * t^2 / 2
        Both wheels contribute: 2 * K * (t/2)^2 = K * t^2 / 2
        """
        t = self.front_track_m
        return (self.wheel_rate_front_Npm + self.arb_rate_front_Npm) * t ** 2 / 2

    @property
    def roll_stiffness_rear_Npm_rad(self):
        t = self.rear_track_m
        return (self.wheel_rate_rear_Npm + self.arb_rate_rear_Npm) * t ** 2 / 2

    @property
    def roll_stiffness_total_Npm_rad(self):
        return self.roll_stiffness_front_Npm_rad + self.roll_stiffness_rear_Npm_rad

    # ── Powertrain computed properties ───────────────────────────────

    @property
    def speed_ms(self):
        """Vehicle speed (m/s) from engine RPM, total drive ratio, tire radius."""
        if self.engine_rpm <= 0 or self.total_drive_ratio <= 0:
            return 0.0
        # v = (RPM × 2π × r_tire) / (ratio × 60)
        return (self.engine_rpm * 2 * np.pi * self.tire_radius_m
                / (self.total_drive_ratio * 60))

    @property
    def speed_kph(self):
        return self.speed_ms * 3.6

    @property
    def engine_torque_Nm(self):
        """Torque from P = T × ω  →  T = P / ω."""
        if self.engine_rpm <= 0 or self.power_hp <= 0:
            return 0.0
        omega = self.engine_rpm * 2 * np.pi / 60  # rad/s
        return self.power_hp * 745.7 / omega

    @property
    def wheel_torque_Nm(self):
        return self.engine_torque_Nm * self.total_drive_ratio

    @property
    def drive_force_N(self):
        """Longitudinal drive force at the contact patch."""
        if self.tire_radius_m <= 0:
            return 0.0
        return self.wheel_torque_Nm / self.tire_radius_m

    @property
    def min_turn_radius_m(self):
        """Minimum turn radius from max steer angle and wheelbase."""
        if self.max_steer_angle_deg <= 0:
            return float('inf')
        return self.wheelbase_m / np.tan(np.radians(self.max_steer_angle_deg))

    def lateral_g_at_radius(self, turn_radius_m: float) -> float:
        """Lateral g = v² / (R × g) at current speed."""
        v = self.speed_ms
        if v <= 0 or turn_radius_m <= 0:
            return 0.0
        return v ** 2 / (turn_radius_m * G)

    def accel_g_from_engine(self) -> float:
        """Longitudinal g available from engine (force / weight)."""
        if self.drive_force_N <= 0:
            return 0.0
        return self.drive_force_N / (self.total_mass_kg * G)

    @classmethod
    def from_car_dict(cls, car: dict) -> "VehicleParams":
        """
        Construct from the GUI's self._car dict + dynamics panel params.
        Falls back to defaults for missing keys.
        """
        kw = {}
        _map = {
            'total_mass_kg':          'total_mass_kg',
            'sprung_mass_kg':         'sprung_mass_kg',
            'unsprung_mass_front_kg': 'unsprung_mass_front_kg',
            'unsprung_mass_rear_kg':  'unsprung_mass_rear_kg',
            'front_track_m':          'front_track_m',
            'rear_track_m':           'rear_track_m',
            'cg_height_m':            'cg_height_m',
            'cg_to_front_axle_m':     'cg_to_front_axle_m',
            'spring_rate_front_Npm':  'spring_rate_front_Npm',
            'spring_rate_rear_Npm':   'spring_rate_rear_Npm',
            'motion_ratio_front':     'motion_ratio_front',
            'motion_ratio_rear':      'motion_ratio_rear',
            'tire_rate_Npm':          'tire_rate_Npm',
            'arb_rate_front_Npm':     'arb_rate_front_Npm',
            'arb_rate_rear_Npm':      'arb_rate_rear_Npm',
            'power_hp':               'power_hp',
            'engine_rpm':             'engine_rpm',
            'total_drive_ratio':      'total_drive_ratio',
            'tire_radius_m':          'tire_radius_m',
            'drivetrain':             'drivetrain',
            'max_steer_angle_deg':    'max_steer_angle_deg',
            'front_brake_bias':       'front_brake_bias',
        }
        for src, dst in _map.items():
            if src in car:
                kw[dst] = car[src]
        # GUI stores track in mm — new keys are track_f_mm / track_r_mm
        if 'track_f_mm' in car and 'front_track_m' not in car:
            kw['front_track_m'] = car['track_f_mm'] / 1000
            kw['rear_track_m'] = car['track_r_mm'] / 1000
        elif 'track_mm' in car and 'front_track_m' not in car:
            kw['front_track_m'] = car['track_mm'] / 1000
            kw['rear_track_m'] = car.get('rear_track_mm', car['track_mm']) / 1000
        if 'wheelbase_mm' in car and 'wheelbase_m' not in kw:
            kw['wheelbase_m'] = car['wheelbase_mm'] / 1000
        if 'cg_z_mm' in car and 'cg_height_m' not in kw:
            kw['cg_height_m'] = car['cg_z_mm'] / 1000
        return cls(**kw)


# ─────────────────────────────────────────────────────────────────────────────
#  Steady-state result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SteadyStateResult:
    """Output of the steady-state cornering solver."""
    lateral_g: float
    longitudinal_g: float = 0.0

    # Per-corner vertical loads (N, positive = compression)
    Fz: dict = field(default_factory=dict)  # {'FL': ..., 'FR': ..., 'RL': ..., 'RR': ...}

    # Roll / pitch
    roll_angle_deg: float = 0.0
    pitch_angle_deg: float = 0.0

    # Per-corner suspension travel (m)
    travel: dict = field(default_factory=dict)

    # Per-corner camber at operating point (deg)
    camber: dict = field(default_factory=dict)

    # Roll centre heights at operating point (m)
    rc_height_front_m: float = 0.0
    rc_height_rear_m: float = 0.0

    # Load transfer breakdown (N, per axle, one-side delta)
    elastic_lt_front_N: float = 0.0
    elastic_lt_rear_N: float = 0.0
    geometric_lt_front_N: float = 0.0
    geometric_lt_rear_N: float = 0.0
    unsprung_lt_front_N: float = 0.0
    unsprung_lt_rear_N: float = 0.0

    # Tire utilization (0-1 or higher if beyond grip)
    utilization: dict = field(default_factory=dict)

    # Per-corner lateral force (N, positive = outboard)
    Fy: dict = field(default_factory=dict)

    # Per-corner longitudinal force (N, positive = forward)
    Fx: dict = field(default_factory=dict)

    # Per-corner brake torque (Nm, positive = retarding)
    brake_torque: dict = field(default_factory=dict)

    # Understeer gradient (front avg SA - rear avg SA, positive = understeer)
    understeer_gradient_deg: float = 0.0

    # Convergence info
    iterations: int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Steady-state solver
# ─────────────────────────────────────────────────────────────────────────────

G = 9.81  # m/s^2


class SteadyStateSolver:
    """
    Steady-state cornering equilibrium solver.

    Given lateral_g (and optionally longitudinal_g), iterates to find
    the equilibrium roll angle, per-corner loads, and suspension travel.
    Queries the kinematic solver at each iteration to capture RC migration,
    camber change, and motion ratio variation with travel.
    """

    def __init__(self,
                 vehicle: VehicleParams,
                 solvers: dict,
                 tire_model=None):
        """
        Parameters
        ----------
        vehicle : VehicleParams
        solvers : dict
            {'FL': SuspensionConstraints, 'FR': ..., 'RL': ..., 'RR': ...}
        tire_model : TireModel | LinearTireModel | None
            If None, tire utilization is not computed.
        """
        self._veh = vehicle
        self._solvers = solvers
        self._tire = tire_model
        self._warm = {}  # per-corner warm start cache

    def solve(self, lateral_g: float,
              longitudinal_g: float = 0.0,
              max_iter: int = 15,
              tol_deg: float = 0.002) -> SteadyStateResult:
        """
        Solve steady-state equilibrium.

        Algorithm:
        1. Compute static corner loads
        2. Initial roll estimate from total roll stiffness
        3. Iterate: roll → per-corner travel → kinematics → load transfer → new roll
        4. Converge when roll angle change < tol_deg
        """
        v = self._veh
        ay = lateral_g * G  # m/s^2
        ax = longitudinal_g * G

        # ── Step 1: Static loads ─────────────────────────────────────────
        W = v.total_mass_kg * G
        Fz_static_front = W * v.front_weight_fraction / 2  # per corner
        Fz_static_rear = W * v.rear_weight_fraction / 2

        # Longitudinal load transfer (pitch)
        delta_Fz_pitch = v.total_mass_kg * ax * v.cg_height_m / v.wheelbase_m
        # Positive ax = acceleration → load transfers to rear
        Fz_static = {
            'FL': Fz_static_front - delta_Fz_pitch / 2,
            'FR': Fz_static_front - delta_Fz_pitch / 2,
            'RL': Fz_static_rear + delta_Fz_pitch / 2,
            'RR': Fz_static_rear + delta_Fz_pitch / 2,
        }

        # ── Step 2: Initial roll estimate ────────────────────────────────
        # Get design-position RC heights
        rc_f = self._query_rc_height('FL', 0.0)
        rc_r = self._query_rc_height('RL', 0.0)

        roll_rad = self._compute_roll(ay, v, rc_f, rc_r)
        roll_prev = roll_rad

        # ── Step 3: Iterate ──────────────────────────────────────────────
        result = SteadyStateResult(lateral_g=lateral_g,
                                   longitudinal_g=longitudinal_g)

        for iteration in range(max_iter):
            # 3a. Per-corner travel from roll
            # Positive roll = body leans right = left side compresses, right extends
            # Convention: positive travel = bump (compression)
            roll_travel_front = np.sin(roll_rad) * v.front_track_m / 2
            roll_travel_rear = np.sin(roll_rad) * v.rear_track_m / 2

            travels = {
                'FL': +roll_travel_front,
                'FR': -roll_travel_front,
                'RL': +roll_travel_rear,
                'RR': -roll_travel_rear,
            }

            # 3b. Solve kinematics at each corner
            rc_heights = {}
            cambers = {}
            for label, travel_m in travels.items():
                side = 'left' if label.endswith('L') else 'right'
                try:
                    state = self._solve_corner(label, travel_m)
                    m = KinematicMetrics(state, side)
                    rc_heights[label] = m.roll_center_height
                    cambers[label] = m.camber
                except Exception:
                    rc_heights[label] = rc_f if label[0] == 'F' else rc_r
                    cambers[label] = 0.0

            # 3c. Axle-level RC height (average of left/right)
            rc_f = (rc_heights['FL'] + rc_heights['FR']) / 2
            rc_r = (rc_heights['RL'] + rc_heights['RR']) / 2

            # 3d. Load transfer
            lt = self._compute_load_transfer(ay, v, rc_f, rc_r)

            # Per-corner Fz (positive = compression)
            # Positive lateral_g = turning right = body rolls left
            # Left side (FL, RL) gains load, right side loses
            Fz = {
                'FL': Fz_static['FL'] + lt['total_front'],
                'FR': Fz_static['FR'] - lt['total_front'],
                'RL': Fz_static['RL'] + lt['total_rear'],
                'RR': Fz_static['RR'] - lt['total_rear'],
            }

            # 3e. Update roll angle
            roll_rad_new = self._compute_roll(ay, v, rc_f, rc_r)

            # Check convergence
            if abs(np.degrees(roll_rad_new - roll_prev)) < tol_deg:
                roll_rad = roll_rad_new
                break
            roll_prev = roll_rad
            roll_rad = roll_rad_new

        # ── Step 4: Build result ─────────────────────────────────────────
        result.roll_angle_deg = np.degrees(roll_rad)

        # Pitch angle from longitudinal load transfer
        # Pitch stiffness = 2 * (K_wheel_front * a^2 + K_wheel_rear * b^2)
        # where a,b = CG distance to front/rear axle
        a = v.cg_to_front_axle_m
        b = v.cg_to_rear_axle_m
        K_pitch = 2 * (v.wheel_rate_front_Npm * a**2 + v.wheel_rate_rear_Npm * b**2)
        pitch_moment = v.sprung_mass_kg * ax * (v.cg_height_m - v.unsprung_cg_height_m)
        result.pitch_angle_deg = np.degrees(pitch_moment / K_pitch) if K_pitch > 0 else 0.0
        result.Fz = Fz
        result.travel = {k: v * 1000 for k, v in travels.items()}  # mm
        result.camber = cambers
        result.rc_height_front_m = rc_f
        result.rc_height_rear_m = rc_r
        result.elastic_lt_front_N = lt['elastic_front']
        result.elastic_lt_rear_N = lt['elastic_rear']
        result.geometric_lt_front_N = lt['geometric_front']
        result.geometric_lt_rear_N = lt['geometric_rear']
        result.unsprung_lt_front_N = lt['unsprung_front']
        result.unsprung_lt_rear_N = lt['unsprung_rear']
        result.iterations = iteration + 1

        # Tire utilization: friction circle — combined Fy + Fx
        #
        # Fy distribution:
        #   Front/rear split from yaw equilibrium (= static weight fraction).
        #   Left/right split within each axle from cornering stiffness at
        #   the dynamic Fz — this is what makes ARB changes affect utilization.
        #   Tire is degressive: more LT variation → lower average C_alpha
        #   → front saturates earlier → understeer.
        #
        # Fx distribution:
        #   Accel → driven axle only (RWD/FWD/AWD), 50/50 left/right.
        #   Braking → brake bias front/rear, 50/50 left/right.
        if self._tire is not None:
            total_fy = abs(v.total_mass_kg * ay)
            fy_front_axle = total_fy * v.front_weight_fraction  # total for axle
            fy_rear_axle  = total_fy * v.rear_weight_fraction

            # Get cornering stiffness at each corner's dynamic Fz
            # Use abs() — TTC sign convention gives negative C_alpha
            C_a = {}
            for label in ['FL', 'FR', 'RL', 'RR']:
                fz_c = max(Fz[label], 1.0)
                C_a[label] = abs(float(self._tire.cornering_stiffness(
                    fz_c, abs(cambers.get(label, 0)))))

            # Distribute Fy within each axle by cornering stiffness
            fy_per_corner = {}
            C_front = C_a['FL'] + C_a['FR']
            C_rear  = C_a['RL'] + C_a['RR']
            if C_front > 0:
                fy_per_corner['FL'] = fy_front_axle * C_a['FL'] / C_front
                fy_per_corner['FR'] = fy_front_axle * C_a['FR'] / C_front
            else:
                fy_per_corner['FL'] = fy_front_axle / 2
                fy_per_corner['FR'] = fy_front_axle / 2
            if C_rear > 0:
                fy_per_corner['RL'] = fy_rear_axle * C_a['RL'] / C_rear
                fy_per_corner['RR'] = fy_rear_axle * C_a['RR'] / C_rear
            else:
                fy_per_corner['RL'] = fy_rear_axle / 2
                fy_per_corner['RR'] = fy_rear_axle / 2

            # Longitudinal force demand per corner
            total_fx = abs(v.total_mass_kg * ax)
            fx_per_corner = {}
            if ax > 0:
                dt = v.drivetrain.upper()
                for lbl in ['FL', 'FR', 'RL', 'RR']:
                    if dt == 'RWD' and lbl[0] == 'R':
                        fx_per_corner[lbl] = total_fx / 2
                    elif dt == 'FWD' and lbl[0] == 'F':
                        fx_per_corner[lbl] = total_fx / 2
                    elif dt == 'AWD':
                        fx_per_corner[lbl] = total_fx / 4
                    else:
                        fx_per_corner[lbl] = 0.0
            else:
                bb_f = v.front_brake_bias
                for lbl in ['FL', 'FR']:
                    fx_per_corner[lbl] = total_fx * bb_f / 2
                for lbl in ['RL', 'RR']:
                    fx_per_corner[lbl] = total_fx * (1 - bb_f) / 2

            for label in ['FL', 'FR', 'RL', 'RR']:
                fz_corner = Fz[label]
                if fz_corner <= 0:
                    # Wheel lifted — cap at 1.0 (fully saturated)
                    result.utilization[label] = 1.0
                    continue
                mu = float(self._tire.peak_mu(fz_corner, abs(cambers.get(label, 0))))
                grip_budget = mu * fz_corner
                fy_req = fy_per_corner.get(label, 0.0)
                fx_req = fx_per_corner.get(label, 0.0)
                combined = np.sqrt(fy_req ** 2 + fx_req ** 2)
                result.utilization[label] = combined / grip_budget if grip_budget > 0 else 0.0

            # Store per-corner forces for component load analysis
            result.Fy = dict(fy_per_corner)
            result.Fx = dict(fx_per_corner)
            # Brake torque per corner = Fx × tire_radius
            tire_r = self._veh.tire_radius_m
            for label in ['FL', 'FR', 'RL', 'RR']:
                result.brake_torque[label] = abs(fx_per_corner.get(label, 0)) * tire_r

            # Understeer gradient: back-calculate slip angles from tire model
            if abs(ay) > 0.1 and hasattr(self._tire, 'slip_angle_for_Fy'):
                try:
                    sa_fl = self._tire.slip_angle_for_Fy(
                        fy_per_corner['FL'], max(Fz['FL'], 1.0), abs(cambers.get('FL', 0)))
                    sa_fr = self._tire.slip_angle_for_Fy(
                        fy_per_corner['FR'], max(Fz['FR'], 1.0), abs(cambers.get('FR', 0)))
                    sa_rl = self._tire.slip_angle_for_Fy(
                        fy_per_corner['RL'], max(Fz['RL'], 1.0), abs(cambers.get('RL', 0)))
                    sa_rr = self._tire.slip_angle_for_Fy(
                        fy_per_corner['RR'], max(Fz['RR'], 1.0), abs(cambers.get('RR', 0)))
                    sa_front = (sa_fl + sa_fr) / 2
                    sa_rear = (sa_rl + sa_rr) / 2
                    result.understeer_gradient_deg = sa_front - sa_rear
                except Exception:
                    pass
        else:
            # No tire model — still compute basic Fy/Fx from equilibrium
            total_fy = abs(v.total_mass_kg * ay)
            total_fx = abs(v.total_mass_kg * ax)
            fy_f = total_fy * v.front_weight_fraction
            fy_r = total_fy * v.rear_weight_fraction
            result.Fy = {'FL': fy_f/2, 'FR': fy_f/2, 'RL': fy_r/2, 'RR': fy_r/2}
            bb = v.front_brake_bias
            if ax < 0:
                result.Fx = {'FL': total_fx*bb/2, 'FR': total_fx*bb/2,
                             'RL': total_fx*(1-bb)/2, 'RR': total_fx*(1-bb)/2}
            else:
                result.Fx = {'FL': 0, 'FR': 0, 'RL': total_fx/2, 'RR': total_fx/2}
            tire_r = v.tire_radius_m
            for label in ['FL', 'FR', 'RL', 'RR']:
                result.brake_torque[label] = abs(result.Fx.get(label, 0)) * tire_r

        return result

    def sweep_lateral_g(self,
                        g_range: tuple = (0.0, 2.0),
                        n_points: int = 41,
                        longitudinal_g: float = 0.0) -> dict:
        """
        Sweep lateral acceleration and return arrays of all outputs.

        Returns dict with numpy arrays keyed by output name.
        """
        g_arr = np.linspace(g_range[0], g_range[1], n_points)
        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {'lateral_g': g_arr}
        for k in keys:
            out[k] = np.zeros(n_points)
        for ck in corner_keys:
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'{ck}_{lbl}'] = np.zeros(n_points)

        self._warm = {}  # reset warm starts

        for i, lg in enumerate(g_arr):
            r = self.solve(lg, longitudinal_g)
            out['roll_angle_deg'][i] = r.roll_angle_deg
            out['pitch_angle_deg'][i] = r.pitch_angle_deg
            out['rc_height_front_mm'][i] = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i] = r.rc_height_rear_m * 1000
            out['elastic_lt_front_N'][i] = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i] = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i] = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'Fz_{lbl}'][i] = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i] = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i] = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)

        # Smooth understeer gradient — inverse tire lookup is noisy at grid edges
        us = out['understeer_gradient_deg']
        if len(us) >= 5:
            out['understeer_gradient_deg'] = uniform_filter1d(us, size=5, mode='nearest')

        return out

    def sweep_longitudinal_g(self,
                             g_range: tuple = (-2.0, 2.0),
                             n_points: int = 41,
                             lateral_g: float = 0.0) -> dict:
        """
        Sweep longitudinal acceleration and return arrays of all outputs.

        Negative g = braking, positive g = acceleration.
        Returns dict with numpy arrays keyed by output name.
        """
        g_arr = np.linspace(g_range[0], g_range[1], n_points)
        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {'longitudinal_g': g_arr}
        for k in keys:
            out[k] = np.zeros(n_points)
        for ck in corner_keys:
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'{ck}_{lbl}'] = np.zeros(n_points)

        self._warm = {}

        for i, lg in enumerate(g_arr):
            r = self.solve(lateral_g, lg)
            out['roll_angle_deg'][i] = r.roll_angle_deg
            out['pitch_angle_deg'][i] = r.pitch_angle_deg
            out['rc_height_front_mm'][i] = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i] = r.rc_height_rear_m * 1000
            out['elastic_lt_front_N'][i] = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i] = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i] = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'Fz_{lbl}'][i] = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i] = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i] = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)

        # Smooth understeer gradient
        us = out['understeer_gradient_deg']
        if len(us) >= 5:
            out['understeer_gradient_deg'] = uniform_filter1d(us, size=5, mode='nearest')

        return out

    def sweep_combined(self,
                       lat_range: tuple = (0.0, 2.0),
                       lon_g: float = -0.5,
                       n_points: int = 41) -> dict:
        """
        Sweep lateral g while simultaneously applying longitudinal g.

        This is the combined cornering + braking/accel condition — the real
        peak dynamic load case. Sweeps lateral g as the x-axis while
        holding longitudinal g constant (e.g. -0.5g braking while cornering).

        Returns dict with 'lateral_g' as x-axis and all outputs.
        The title/metadata indicates the fixed longitudinal g.
        """
        g_arr = np.linspace(lat_range[0], lat_range[1], n_points)
        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {'lateral_g': g_arr, 'fixed_longitudinal_g': lon_g}
        for k in keys:
            out[k] = np.zeros(n_points)
        for ck in corner_keys:
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'{ck}_{lbl}'] = np.zeros(n_points)

        self._warm = {}

        for i, lat_g in enumerate(g_arr):
            r = self.solve(lat_g, lon_g)
            out['roll_angle_deg'][i] = r.roll_angle_deg
            out['pitch_angle_deg'][i] = r.pitch_angle_deg
            out['rc_height_front_mm'][i] = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i] = r.rc_height_rear_m * 1000
            out['elastic_lt_front_N'][i] = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i] = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i] = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'Fz_{lbl}'][i] = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i] = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i] = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)

        # Smooth understeer gradient
        us = out['understeer_gradient_deg']
        if len(us) >= 5:
            out['understeer_gradient_deg'] = uniform_filter1d(us, size=5, mode='nearest')

        return out

    def max_accel_g(self, speed_kph: float = 0.0, lateral_g: float = 0.0) -> dict:
        """
        Compute maximum longitudinal acceleration at a given speed.

        Returns dict with:
            traction_g   — grip-limited max g (mu * Fz_driven / m_total)
            power_g      — power-limited max g at given speed (P / m / v / g)
            effective_g  — min(traction, power) = actual max accel
            braking_g    — max braking g (all 4 tires, mu * total_Fz / m)
        """
        v = self._veh
        W = v.total_mass_kg * G
        r_tire = v.tire_radius_m

        # Get tire mu from tire model, or use default
        if self._tire is not None and hasattr(self._tire, 'peak_mu'):
            # Use average Fz per corner for mu estimate
            Fz_front = W * v.front_weight_fraction / 2
            Fz_rear = W * v.rear_weight_fraction / 2
            mu_f = float(self._tire.peak_mu(Fz_front, 0.0))
            mu_r = float(self._tire.peak_mu(Fz_rear, 0.0))
        else:
            mu_f = mu_r = 1.5  # conservative default

        # Traction limit (depends on driven axle)
        dt = v.drivetrain.upper()
        if dt == 'RWD':
            Fz_driven = W * v.rear_weight_fraction
            mu_driven = mu_r
        elif dt == 'FWD':
            Fz_driven = W * v.front_weight_fraction
            mu_driven = mu_f
        else:  # AWD
            Fz_driven = W
            mu_driven = (mu_f + mu_r) / 2

        traction_force = mu_driven * Fz_driven
        traction_g = traction_force / (v.total_mass_kg * G)

        # Power limit: F = P / v
        power_W = v.power_hp * 745.7  # HP → watts
        if speed_kph > 1.0 and power_W > 0:
            v_ms = speed_kph / 3.6
            power_force = power_W / v_ms
            power_g = power_force / (v.total_mass_kg * G)
        else:
            power_g = float('inf') if power_W > 0 else 0.0

        # Braking: all 4 tires, average mu
        mu_avg = (mu_f * v.front_weight_fraction + mu_r * v.rear_weight_fraction)
        braking_g = mu_avg  # mu * m * g / (m * g) = mu

        effective_g = min(traction_g, power_g) if power_W > 0 else traction_g

        return {
            'traction_g': traction_g,
            'power_g': power_g if power_g != float('inf') else 0.0,
            'effective_g': effective_g if effective_g != float('inf') else traction_g,
            'braking_g': braking_g,
            'mu_front': mu_f,
            'mu_rear': mu_r,
        }

    # ── Internals ────────────────────────────────────────────────────────

    def _solve_corner(self, label: str, travel_m: float):
        """Solve kinematics for one corner with warm-start caching."""
        solver = self._solvers[label]
        warm = self._warm.get(label)
        if warm is not None:
            state = solver.solve(travel_m, x0=warm['x0'],
                                 rocker_theta0=warm['theta'],
                                 rocker_spring_prev=warm['spring_len'])
        else:
            state = solver.solve(travel_m)
        self._warm[label] = {
            'x0': state.x_vec(),
            'theta': state.rocker_angle,
            'spring_len': state.spring_length,
        }
        return state

    def _query_rc_height(self, label: str, travel_m: float) -> float:
        """Get roll centre height at a single corner's travel."""
        side = 'left' if label.endswith('L') else 'right'
        try:
            state = self._solvers[label].solve(travel_m)
            m = KinematicMetrics(state, side)
            return m.roll_center_height
        except Exception:
            return 0.05  # 50mm fallback

    def _compute_roll(self, ay: float, v: VehicleParams,
                      rc_f: float, rc_r: float) -> float:
        """Compute roll angle (rad) from lateral acceleration and RC heights."""
        # Roll axis height at CG longitudinal position
        b = v.cg_to_front_axle_m / v.wheelbase_m  # fraction from front
        h_roll_axis = rc_f * (1 - b) + rc_r * b

        # Sprung mass roll moment
        h_arm = v.cg_height_m - h_roll_axis
        roll_moment = v.sprung_mass_kg * ay * h_arm  # N·m

        # Roll stiffness resists
        K_total = v.roll_stiffness_total_Npm_rad
        if K_total < 1.0:
            return 0.0
        return roll_moment / K_total

    def _compute_load_transfer(self, ay: float, v: VehicleParams,
                               rc_f: float, rc_r: float) -> dict:
        """
        Compute all load transfer components (N, one-side delta per axle).

        Positive = load added to the outside wheel in a right turn.
        """
        K_f = v.roll_stiffness_front_Npm_rad
        K_r = v.roll_stiffness_rear_Npm_rad
        K_total = K_f + K_r

        # Roll axis height at CG
        b = v.cg_to_front_axle_m / v.wheelbase_m
        h_roll_axis = rc_f * (1 - b) + rc_r * b

        # Geometric (direct through roll centre, no body roll needed)
        geo_front = v.sprung_mass_kg * v.front_weight_fraction * ay * rc_f / v.front_track_m
        geo_rear = v.sprung_mass_kg * v.rear_weight_fraction * ay * rc_r / v.rear_track_m

        # Elastic (through springs + ARB, proportional to roll stiffness dist)
        h_arm = v.cg_height_m - h_roll_axis
        roll_moment = v.sprung_mass_kg * ay * h_arm

        if K_total > 0:
            elastic_front = roll_moment * (K_f / K_total) / v.front_track_m
            elastic_rear = roll_moment * (K_r / K_total) / v.rear_track_m
        else:
            elastic_front = elastic_rear = 0.0

        # Unsprung mass (directly through axle height)
        h_us = v.unsprung_cg_height_m
        unsprung_front = (v.unsprung_mass_front_kg / 2) * ay * h_us / v.front_track_m
        unsprung_rear = (v.unsprung_mass_rear_kg / 2) * ay * h_us / v.rear_track_m

        return {
            'geometric_front': geo_front,
            'geometric_rear': geo_rear,
            'elastic_front': elastic_front,
            'elastic_rear': elastic_rear,
            'unsprung_front': unsprung_front,
            'unsprung_rear': unsprung_rear,
            'total_front': geo_front + elastic_front + unsprung_front,
            'total_rear': geo_rear + elastic_rear + unsprung_rear,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamics Sensitivity Analyzer
# ─────────────────────────────────────────────────────────────────────────────

# Output metrics the optimizer tracks
SENSITIVITY_OUTPUTS = [
    'understeer_gradient_deg',
    'roll_angle_deg',
    'pitch_angle_deg',
    'lltd_pct',            # elastic LT front / total elastic LT × 100
    'utilization_max',     # max of all 4 corners
    'utilization_spread',  # max - min across corners (balance)
]

# Tunable parameters:
#   (key, display_name, unit, delta_for_finitediff, category, practical_step)
# category: 'parameter' = bolt-on changes; 'kinematic' = hardpoint / geometry changes
# practical_step: the realistic increment a user would actually make (display units)
SENSITIVITY_KNOBS = [
    # ── Parameter knobs (shop-adjustable) ────────────────────────────
    ('spring_rate_front_Npm', 'Spring rate F',  'lbf/in', 175.127 * 10, 'parameter', 25),
    ('spring_rate_rear_Npm',  'Spring rate R',  'lbf/in', 175.127 * 10, 'parameter', 25),
    ('arb_rate_front_Npm',    'ARB rate F',     'lbf/in', 175.127 * 10, 'parameter', 25),
    ('arb_rate_rear_Npm',     'ARB rate R',     'lbf/in', 175.127 * 10, 'parameter', 25),
    ('cg_to_front_axle_m',    'Weight dist (CG fwd)', 'mm', 0.010,      'parameter', 25),
    ('front_brake_bias',      'Brake bias',     '%',      0.02,          'parameter', 5),
    # ── Kinematic knobs (hardpoint / geometry changes) ────────────────
    ('motion_ratio_front',    'Motion ratio F', '',       0.02,          'kinematic', 0.05),
    ('motion_ratio_rear',     'Motion ratio R', '',       0.02,          'kinematic', 0.05),
    # RC height and Ackermann are not direct VehicleParams fields —
    # they are injected by overriding the kinematic solver query.
    # We handle them specially via _perturb_rc and _perturb_ackermann.
]


def _extract_outputs(result: SteadyStateResult) -> dict:
    """Pull the tracked output metrics from a SteadyStateResult."""
    el_f = result.elastic_lt_front_N
    el_r = result.elastic_lt_rear_N
    el_tot = el_f + el_r
    lltd = (el_f / el_tot * 100) if el_tot > 0 else 50.0

    utils = [result.utilization.get(c, 0) for c in ('FL', 'FR', 'RL', 'RR')]
    return {
        'understeer_gradient_deg': result.understeer_gradient_deg,
        'roll_angle_deg': result.roll_angle_deg,
        'pitch_angle_deg': result.pitch_angle_deg,
        'lltd_pct': lltd,
        'utilization_max': max(utils) if utils else 0,
        'utilization_spread': (max(utils) - min(utils)) if utils else 0,
    }


class DynamicsSensitivity:
    """
    Numerical sensitivity analyzer for vehicle dynamics.

    Perturbs each tunable parameter by a small delta, re-solves the
    steady-state equilibrium, and computes ∂output / ∂input for every
    output metric.  Also handles kinematic pseudo-knobs (RC height,
    Ackermann) by building modified solvers.

    Usage:
        sens = DynamicsSensitivity(base_veh_params, solvers, tire_model)
        table = sens.analyze(lateral_g=1.2, longitudinal_g=-0.5)
        # table = {
        #   'baseline': {metric: value, ...},
        #   'sensitivities': [
        #     {'knob': 'Spring rate F', 'unit': 'lbf/in', 'category': 'parameter',
        #      'delta_input': 10.0,  # in display units
        #      'effects': {metric: d_metric_per_unit_input, ...},
        #      'implementations': ['Swap spring: 200 → 210 lbf/in']},
        #     ...
        #   ]
        # }
    """

    def __init__(self, vehicle: VehicleParams, solvers: dict, tire_model=None):
        self._base_veh = vehicle
        self._solvers = solvers
        self._tire = tire_model

    def analyze(self, lateral_g: float = 1.2,
                longitudinal_g: float = 0.0) -> dict:
        """
        Run sensitivity analysis at the given operating point.

        Returns dict with 'baseline' outputs and 'sensitivities' list.
        """
        # ── Baseline solve ───────────────────────────────────────────
        base_solver = SteadyStateSolver(self._base_veh, self._solvers, self._tire)
        base_result = base_solver.solve(lateral_g, longitudinal_g)
        baseline = _extract_outputs(base_result)

        sensitivities = []

        # ── Standard VehicleParams knobs ─────────────────────────────
        for key, name, unit, delta, category, practical_step in SENSITIVITY_KNOBS:
            base_val = getattr(self._base_veh, key)

            # Perturb up
            veh_up = self._perturb_veh(key, base_val + delta)
            solver_up = SteadyStateSolver(veh_up, self._solvers, self._tire)
            result_up = solver_up.solve(lateral_g, longitudinal_g)
            out_up = _extract_outputs(result_up)

            # Perturb down (central difference)
            veh_dn = self._perturb_veh(key, base_val - delta)
            solver_dn = SteadyStateSolver(veh_dn, self._solvers, self._tire)
            result_dn = solver_dn.solve(lateral_g, longitudinal_g)
            out_dn = _extract_outputs(result_dn)

            # Convert delta to display units
            if unit == 'lbf/in':
                delta_display = delta / 175.127
            elif unit == 'mm':
                delta_display = delta * 1000
            elif unit == '%':
                delta_display = delta * 100
            else:
                delta_display = delta

            # ∂output/∂input (per display unit)
            effects = {}
            for metric in SENSITIVITY_OUTPUTS:
                d_out = out_up[metric] - out_dn[metric]
                d_in = 2 * delta_display  # central difference
                effects[metric] = d_out / d_in if abs(d_in) > 1e-12 else 0.0

            # Current value in display units
            if unit == 'lbf/in':
                current_display = base_val / 175.127
            elif unit == 'mm':
                current_display = base_val * 1000
            elif unit == '%':
                current_display = base_val * 100
            else:
                current_display = base_val

            impls = self._implementation_hints(key, name, unit, current_display)

            # effects_per_step: what actually happens if you change by one practical step
            effects_per_step = {m: effects[m] * practical_step for m in effects}

            sensitivities.append({
                'knob': name,
                'key': key,
                'unit': unit,
                'category': category,
                'current_value': current_display,
                'delta_input': delta_display,
                'practical_step': practical_step,
                'effects': effects,              # per 1 display-unit
                'effects_per_step': effects_per_step,  # per practical step
                'implementations': impls,
            })

        # ── RC height pseudo-knobs (not a VehicleParams field) ────────
        # We simulate RC height change by directly adjusting the
        # geometric LT component — equivalent to raising/lowering RC.
        for axle, label in [('front', 'RC height F'), ('rear', 'RC height R')]:
            rc_delta_m = 0.005   # 5mm perturbation
            effects = self._rc_sensitivity(
                base_result, baseline, lateral_g, longitudinal_g,
                axle, rc_delta_m)

            current_rc = (base_result.rc_height_front_m if axle == 'front'
                          else base_result.rc_height_rear_m) * 1000

            rc_practical_step = 10  # 10mm is a realistic RC height adjustment
            effects_per_step = {m: effects[m] * rc_practical_step for m in effects}

            sensitivities.append({
                'knob': label,
                'key': f'rc_height_{axle}',
                'unit': 'mm',
                'category': 'kinematic',
                'current_value': current_rc,
                'delta_input': rc_delta_m * 1000,
                'practical_step': rc_practical_step,
                'effects': effects,
                'effects_per_step': effects_per_step,
                'implementations': [
                    f'Move side-view IC \u2192 adjust lca/uca pickup heights',
                    f'Current: {current_rc:.1f} mm',
                ],
            })

        return {
            'baseline': baseline,
            'baseline_result': base_result,
            'sensitivities': sensitivities,
            'vehicle_params': self._base_veh,
        }

    def recommend(self, analysis: dict, target_metric: str,
                  target_delta: float) -> list:
        """
        Given an analysis result and a desired change in one metric,
        return ranked list of ways to achieve it.

        Each entry: {knob, unit, category, change_needed, current, new_value,
                     side_effects: {other_metric: delta}, implementations}
        """
        # Ensure _base_veh is set (may be called via __new__ without __init__)
        if not hasattr(self, '_base_veh') or self._base_veh is None:
            self._base_veh = analysis.get('vehicle_params')

        recommendations = []
        for s in analysis['sensitivities']:
            effect = s['effects'].get(target_metric, 0)
            if abs(effect) < 1e-6:
                continue

            change_needed = target_delta / effect  # how much to change this knob

            # Compute side effects on other metrics
            side_effects = {}
            for metric in SENSITIVITY_OUTPUTS:
                if metric == target_metric:
                    continue
                other_effect = s['effects'].get(metric, 0)
                side_effects[metric] = other_effect * change_needed

            new_val = s['current_value'] + change_needed

            # Regenerate hints with actual change_needed for roll stiffness info
            impls = self._implementation_hints(
                s['key'], s['knob'], s['unit'], s['current_value'],
                change_needed=change_needed)
            if not impls:
                impls = s.get('implementations', [])

            recommendations.append({
                'knob': s['knob'],
                'key': s['key'],
                'unit': s['unit'],
                'category': s['category'],
                'current': s['current_value'],
                'change_needed': change_needed,
                'new_value': new_val,
                'side_effects': side_effects,
                'implementations': impls,
                'effectiveness': abs(effect),  # for sorting
            })

        # Sort by effectiveness (biggest effect first)
        recommendations.sort(key=lambda r: r['effectiveness'], reverse=True)
        return recommendations

    # ── Internals ────────────────────────────────────────────────────

    def _perturb_veh(self, key: str, new_val: float) -> VehicleParams:
        """Create a copy of VehicleParams with one field changed."""
        from dataclasses import asdict
        d = asdict(self._base_veh)
        d[key] = new_val
        return VehicleParams(**d)

    def _rc_sensitivity(self, base_result, baseline, lateral_g, longitudinal_g,
                        axle, rc_delta_m):
        """
        Estimate sensitivity to roll centre height change.

        Instead of rebuilding the kinematic solver (expensive), we use the
        analytical load transfer equations to compute what would change if
        RC height shifted.  This is accurate for small perturbations.
        """
        v = self._base_veh
        ay = lateral_g * G

        rc_f = base_result.rc_height_front_m
        rc_r = base_result.rc_height_rear_m

        effects = {}
        for sign in [+1, -1]:
            rc_f_p = rc_f + (rc_delta_m * sign if axle == 'front' else 0)
            rc_r_p = rc_r + (rc_delta_m * sign if axle == 'rear' else 0)

            # Recompute roll with perturbed RC
            b = v.cg_to_front_axle_m / v.wheelbase_m
            h_roll = rc_f_p * (1 - b) + rc_r_p * b
            h_arm = v.cg_height_m - h_roll
            roll_moment = v.sprung_mass_kg * ay * h_arm
            K_total = v.roll_stiffness_total_Npm_rad
            roll_rad = roll_moment / K_total if K_total > 0 else 0

            # Recompute LT
            K_f = v.roll_stiffness_front_Npm_rad
            K_r = v.roll_stiffness_rear_Npm_rad
            geo_f = v.sprung_mass_kg * v.front_weight_fraction * ay * rc_f_p / v.front_track_m
            geo_r = v.sprung_mass_kg * v.rear_weight_fraction * ay * rc_r_p / v.rear_track_m
            if K_total > 0:
                el_f = roll_moment * (K_f / K_total) / v.front_track_m
                el_r = roll_moment * (K_r / K_total) / v.rear_track_m
            else:
                el_f = el_r = 0

            el_tot = el_f + el_r
            lltd = (el_f / el_tot * 100) if el_tot > 0 else 50

            out = {
                'roll_angle_deg': np.degrees(roll_rad),
                'pitch_angle_deg': baseline['pitch_angle_deg'],  # RC doesn't affect pitch
                'lltd_pct': lltd,
                'understeer_gradient_deg': baseline['understeer_gradient_deg'],  # approx
                'utilization_max': baseline['utilization_max'],
                'utilization_spread': baseline['utilization_spread'],
            }

            if sign == +1:
                out_up = out
            else:
                out_dn = out

        delta_display = rc_delta_m * 1000  # mm
        for metric in SENSITIVITY_OUTPUTS:
            d_out = out_up[metric] - out_dn[metric]
            effects[metric] = d_out / (2 * delta_display)

        return effects

    def _implementation_hints(self, key, name, unit, current_val,
                              change_needed=0.0) -> list:
        """Return human-readable implementation suggestions for a knob."""
        hints = []
        v = self._base_veh
        if 'spring_rate' in key:
            axle = 'front' if 'front' in key else 'rear'
            t = v.front_track_m if 'front' in key else v.rear_track_m
            mr = v.motion_ratio_front if 'front' in key else v.motion_ratio_rear
            arb = v.arb_rate_front_Npm if 'front' in key else v.arb_rate_rear_Npm
            # Current and new roll stiffness contribution from this spring
            new_rate_lbf = current_val + change_needed
            new_rate_Npm = new_rate_lbf * 175.127
            old_wheel = current_val * 175.127 * mr ** 2
            new_wheel = new_rate_Npm * mr ** 2
            old_roll = (old_wheel + arb) * t ** 2 / 2
            new_roll = (new_wheel + arb) * t ** 2 / 2
            delta_roll = new_roll - old_roll
            hints.append(f'Swap {axle} spring: {current_val:.0f} -> {new_rate_lbf:.0f} {unit}')
            hints.append(f'Roll stiffness {axle}: {old_roll:.0f} -> {new_roll:.0f} N\u00b7m/rad '
                         f'(\u0394{delta_roll:+.0f})')
        elif 'arb_rate' in key:
            axle = 'front' if 'front' in key else 'rear'
            t = v.front_track_m if 'front' in key else v.rear_track_m
            wheel_rate = v.wheel_rate_front_Npm if 'front' in key else v.wheel_rate_rear_Npm
            # Current and new roll stiffness
            new_rate_lbf = current_val + change_needed
            new_rate_Npm = new_rate_lbf * 175.127
            old_Npm = current_val * 175.127
            old_roll = (wheel_rate + old_Npm) * t ** 2 / 2
            new_roll = (wheel_rate + new_rate_Npm) * t ** 2 / 2
            delta_roll = new_roll - old_roll
            hints.append(f'{axle.title()} ARB rate: {current_val:.0f} -> {new_rate_lbf:.0f} {unit}')
            hints.append(f'Roll stiffness {axle}: {old_roll:.0f} -> {new_roll:.0f} N\u00b7m/rad '
                         f'(\u0394{delta_roll:+.0f})')
            # Blade length guidance: ARB stiffness ~ 1/L^3
            if old_Npm > 1 and new_rate_Npm > 1:
                ratio = (old_Npm / new_rate_Npm) ** (1.0 / 3.0)
                hints.append(f'Blade length ratio: \u00d7{ratio:.3f} '
                             f'(stiffer = shorter blade)')
        elif 'motion_ratio' in key:
            axle = 'front' if 'front' in key else 'rear'
            hints.append(f'Adjust {axle} rocker geometry (pushrod/rocker points)')
            hints.append(f'Currently MR = {current_val:.3f}')
        elif 'cg_to_front' in key:
            hints.append(f'Move battery, radiator, or ballast')
            hints.append(f'CG currently {current_val:.0f} mm from front axle')
        elif 'brake_bias' in key:
            hints.append(f'Adjust brake bias bar / proportioning valve')
            hints.append(f'Currently {current_val:.0f}% front')
        return hints
