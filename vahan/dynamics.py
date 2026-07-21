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
    """All vehicle-level parameters for dynamics computations.

    Mass is split into sprung + unsprung-front + unsprung-rear; total is a
    derived property (= sum).  Earlier versions exposed total_mass_kg as an
    independent input — that allowed the three to disagree, which is a
    poka-yoke violation.  Total is now always exactly consistent.
    """

    # Mass (kg) — sprung + unsprung_front + unsprung_rear are independent;
    # total_mass_kg is a derived @property below.
    sprung_mass_kg: float = 223.8
    unsprung_mass_front_kg: float = 26.5    # per axle (both wheels)
    unsprung_mass_rear_kg: float = 40.05

    # Geometry (m)
    wheelbase_m: float = 1.530
    front_track_m: float = 1.2218
    rear_track_m: float = 1.200
    cg_height_m: float = 0.26063
    cg_to_front_axle_m: float = 0.845

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
    #   power_hp                 — peak ENGINE/MOTOR power (hp).  Drivetrain
    #                              losses are taken via drivetrain_efficiency.
    #   engine_rpm               — RPM at which peak power occurs.  Only
    #                              used to compute the GEARED top speed
    #                              (not the drive force — that uses the
    #                              constant-power model below).
    #   total_drive_ratio        — final drive ratio = engine_omega / wheel_omega.
    #                              Single number — no separate primary/sprocket.
    #   drivetrain_efficiency    — wheel_power / engine_power (0.85-0.95 typ).
    #   powertrain_type          — 'ICE' (constant-power above peak RPM) or
    #                              'EV' (constant torque below motor base
    #                              speed, constant power above).
    #   peak_torque_Nm           — EV only: motor peak torque at the shaft.
    #                              Sets the low-speed traction-limit force.
    #                              Ignored for ICE.
    power_hp: float = 0.0
    engine_rpm: float = 0.0
    total_drive_ratio: float = 10.0
    tire_radius_m: float = 0.203
    drivetrain: str = 'RWD'         # 'RWD', 'FWD', or 'AWD'
    drivetrain_efficiency: float = 0.92
    powertrain_type: str = 'ICE'    # 'ICE' or 'EV'
    peak_torque_Nm: float = 0.0     # EV only
    # Limited-slip differential (Drexler FSAE LSD by default — see
    # vahan/differential.py).  Sets corner entry/exit balance via locking %.
    diff_kind: str = 'salisbury'    # 'open' | 'spool' | 'salisbury'
    diff_power_ramp_deg: float = 40.0   # drive ramp  (-> power locking %)
    diff_coast_ramp_deg: float = 50.0   # overrun ramp (-> coast locking %)
    diff_preload_Nm: float = 30.0
    # Engine-braking torque at the CRANK (Nm) on a closed throttle — the
    # overrun torque that flows through the diff and engages the COAST ramp.
    # Without it the coast ramp would be a dead input.  ×total_drive_ratio
    # gives the axle overrun torque.
    engine_braking_Nm: float = 12.0
    max_steer_angle_deg: float = 28.0
    front_brake_bias: float = 0.65

    # Aerodynamic drag — bounds terminal speed during longitudinal
    # acceleration trajectories.  CdA is the lumped drag coefficient ×
    # frontal area (m²).  Without these, the steady-state power-limit
    # speed v = P/(m·g·g_e) goes to infinity at low g.
    cda_m2:                float = 1.0      # Cd · A (m²)  — FSAE no-aero ≈ 1.0
    air_density_kg_m3:     float = 1.225    # ρ at ISA sea level / 15 °C

    # Steering rack geometry — drives the mapping between the driver's
    # steering-wheel rotation and the tie-rod translation at the front axle.
    # Used by `vahan.steering.SteeringGeometry` to convert commanded
    # road-wheel angle ↔ steering-wheel angle, and to derive the physical
    # max road-wheel angle imposed by the rack travel limit.
    rack_travel_per_rev_mm: float = 60.0    # rack translation per 360° of wheel
    total_rack_travel_mm:   float = 120.0   # full stroke bump-to-bump

    # Auto-inertia coefficients — used by the GUI when Ixx/Izz aren't
    # measured directly.  Documented so they show up in from_car_dict.
    yaw_inertia_factor:        float = 1.2   # Izz ≈ k · m · a · b
    roll_gyradius_track_frac:  float = 0.35  # k_roll ≈ frac · track_avg

    # Speed-hold PI controller (normalised by mass so the same gains give
    # similar response on 200 kg and 800 kg cars).  Actual gain =
    # per_kg × total_mass_kg inside TransientSolver.
    speed_hold_kp_per_kg: float = 0.69   # → 200 N/(m/s) on a 290 kg FSAE
    speed_hold_ki_per_kg: float = 0.17

    # ── Topology-specific spring/MR overrides ────────────────────────────
    # The default wheel_rate / roll_stiffness formulas assume one spring
    # per corner driven by a single rocker (MR² law).  For topologies
    # whose spring set differs (HEAVE_TBAR adds a 3rd element; DECOUPLED
    # replaces corner springs with cross-car heave + roll dampers) those
    # formulas don't apply.  The host (MainWindow.set_topology) writes
    # the per-axle topology_mode tag + the necessary rates and motion
    # ratios into these fields, and the property overrides below branch
    # on them.
    #
    #   topology_mode_*  ∈ {'standard', 'heave_tbar', 'decoupled'}
    #   *_3rd_*          additional 3rd-element heave spring (HEAVE_TBAR)
    #   decoupled_*      heave + roll spring rates and the GEOMETRIC ratios
    #                    (cross-car damper compression per unit symmetric /
    #                    antisymmetric wheel motion) from the twin-bellcrank
    #                    kinematic solver.
    topology_mode_front: str = 'standard'
    topology_mode_rear:  str = 'standard'

    heave_3rd_rate_front_Npm: float = 0.0
    heave_3rd_MR_front:       float = 0.0   # 3rd-spring compression per Δz (symmetric)
    heave_3rd_rate_rear_Npm:  float = 0.0
    heave_3rd_MR_rear:        float = 0.0

    decoupled_heave_rate_front_Npm: float = 0.0
    decoupled_heave_MR_front:       float = 0.0   # damper Δ per symmetric wheel Δz
    decoupled_roll_rate_front_Npm:  float = 0.0
    decoupled_roll_MR_front:        float = 0.0   # damper Δ per antisymmetric Δz
    decoupled_heave_rate_rear_Npm:  float = 0.0
    decoupled_heave_MR_rear:        float = 0.0
    decoupled_roll_rate_rear_Npm:   float = 0.0
    decoupled_roll_MR_rear:         float = 0.0

    # ── RCVD item 3: geometric IR-rate wheel-rate correction ───────────
    # Per RCVD section 16.3 (p595-598) the wheel rate has TWO terms:
    #   K_wheel  =  F_s * (dIR/d_delta)  +  K_s * IR^2
    # where F_s is the static spring force (= F_corner / IR_static) and
    # dIR/d_delta is how the installation ratio changes with wheel
    # travel.  The first term is usually <2 % for stiffly-sprung FSAE
    # cars but grows for softer setups; ignoring it ~always under-
    # predicts wheel rate when IR rises into bump.
    #
    # Set by MainWindow._apply_topology_to_dyn_params from the kinematic
    # sweep:
    #   mr_slope_*  =  (MR_at_+5mm  -  MR_at_-5mm) / 0.010  in 1/m
    #   static_spring_force_*  =  static corner load / MR_static  in N
    mr_slope_front_per_m: float = 0.0
    mr_slope_rear_per_m:  float = 0.0
    static_spring_force_front_N: float = 0.0
    static_spring_force_rear_N:  float = 0.0

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def total_mass_kg(self) -> float:
        """Derived total vehicle mass = sprung + unsprung_front + unsprung_rear.

        Was an independent input field — removed so the three mass terms
        can never silently disagree.  All downstream callers (which use
        ``veh.total_mass_kg`` for lateral/longitudinal load-transfer
        formulae, total weight, etc.) still work because this is a drop-in
        @property replacement for the old field.
        """
        return (self.sprung_mass_kg
                + self.unsprung_mass_front_kg
                + self.unsprung_mass_rear_kg)

    @property
    def cg_to_rear_axle_m(self):
        return self.wheelbase_m - self.cg_to_front_axle_m

    @property
    def sprung_cg_height_m(self):
        """Height of the *sprung-mass* CG above ground (m).

        The user-supplied ``cg_height_m`` is the WHOLE-vehicle CG.  Most
        chassis-roll / pitch moment expressions need the sprung-mass CG
        instead — the unsprung mass sits at the wheel-centre height and
        pulls the whole-vehicle CG down.  Conservation of mass-times-
        height gives:

            m · h_cg = m_s · h_s + m_u · h_u
            ⇒  h_s = (m · h_cg − m_u · h_u) / m_s

        Falls back to ``cg_height_m`` if sprung mass is degenerate
        (ill-posed inputs) so callers never crash on missing data.
        """
        m_u = self.total_mass_kg - self.sprung_mass_kg
        if self.sprung_mass_kg <= 1e-3:
            return self.cg_height_m
        return ((self.total_mass_kg * self.cg_height_m
                 - m_u * self.unsprung_cg_height_m)
                / self.sprung_mass_kg)

    @property
    def front_weight_fraction(self):
        return self.cg_to_rear_axle_m / self.wheelbase_m

    @property
    def rear_weight_fraction(self):
        return self.cg_to_front_axle_m / self.wheelbase_m

    # ── Topology-aware wheel rate / roll stiffness ───────────────────────
    # Branches on topology_mode_{front,rear}.  See the field declarations
    # above and the derivation in the module docstring.
    #
    # Per-axle "wheel rate" semantics:
    #   K_wheel = effective per-wheel vertical stiffness in PURE HEAVE
    #             (i.e. the spring force per metre of symmetric wheel-
    #             pair displacement, divided by 2).
    # Standard topology:   K_wheel  =  K_spring × MR²       (corner spring)
    # HEAVE_TBAR topology: K_wheel  =  K_spring × MR² + ½ × K_3rd × MR_3rd²
    #                                (corner spring + shared 3rd element)
    # DECOUPLED topology:  K_wheel  =  ½ × K_heave × MR_heave²
    #                                (NO corner spring; the shared cross-
    #                                 car heave damper is split between
    #                                 both wheels — factor of ½).

    def _wheel_rate_for_axle(self, is_front: bool) -> float:
        if is_front:
            mode = self.topology_mode_front
            k_corner = self.spring_rate_front_Npm
            mr_corner = self.motion_ratio_front
            k_3rd, mr_3rd = self.heave_3rd_rate_front_Npm, self.heave_3rd_MR_front
            k_h,   mr_h   = self.decoupled_heave_rate_front_Npm, self.decoupled_heave_MR_front
            mr_slope = self.mr_slope_front_per_m
            Fs       = self.static_spring_force_front_N
        else:
            mode = self.topology_mode_rear
            k_corner = self.spring_rate_rear_Npm
            mr_corner = self.motion_ratio_rear
            k_3rd, mr_3rd = self.heave_3rd_rate_rear_Npm, self.heave_3rd_MR_rear
            k_h,   mr_h   = self.decoupled_heave_rate_rear_Npm, self.decoupled_heave_MR_rear
            mr_slope = self.mr_slope_rear_per_m
            Fs       = self.static_spring_force_rear_N
        if mode == 'decoupled':
            return 0.5 * k_h * mr_h ** 2
        # RCVD section 16.3 correction (item 3): K_wheel = Fs * dIR/dδ + K_s * IR^2.
        # Adds a typically-small Fs * mr_slope term that grows for soft
        # suspensions.  Defaults to zero when mr_slope or Fs aren't set
        # (the caller didn't run a kinematic sweep), so backwards-compat.
        geometric_term = Fs * mr_slope    # N/m
        if mode == 'heave_tbar':
            return (k_corner * mr_corner ** 2 + 0.5 * k_3rd * mr_3rd ** 2
                    + geometric_term)
        # standard
        return k_corner * mr_corner ** 2 + geometric_term

    @property
    def wheel_rate_front_Npm(self):
        return self._wheel_rate_for_axle(is_front=True)

    @property
    def wheel_rate_rear_Npm(self):
        return self._wheel_rate_for_axle(is_front=False)

    @property
    def ride_rate_front_Npm(self):
        """Series combination of wheel rate and tire rate."""
        kw, kt = self.wheel_rate_front_Npm, self.tire_rate_Npm
        if kw + kt <= 0:
            return 0.0
        return (kw * kt) / (kw + kt)

    @property
    def ride_rate_rear_Npm(self):
        kw, kt = self.wheel_rate_rear_Npm, self.tire_rate_Npm
        if kw + kt <= 0:
            return 0.0
        return (kw * kt) / (kw + kt)

    def _roll_stiffness_for_axle(self, is_front: bool) -> float:
        """Per-axle roll stiffness in N·m/rad.

        Standard:    K_roll = (K_wheel + K_arb) × t² / 2
        HEAVE_TBAR:  same as standard — 3rd element sees zero force in
                     pure roll because both drop links push the bracket
                     in opposite directions, so the heave spring doesn't
                     compress.
        DECOUPLED:   K_roll = ¼ × K_roll_spring × MR_roll² × t²
                     (no corner spring + no separate ARB; entire roll
                     stiffness comes from the cross-car roll coilover).
        """
        if is_front:
            mode  = self.topology_mode_front
            t     = self.front_track_m
            k_arb = self.arb_rate_front_Npm
            k_r, mr_r = self.decoupled_roll_rate_front_Npm, self.decoupled_roll_MR_front
        else:
            mode  = self.topology_mode_rear
            t     = self.rear_track_m
            k_arb = self.arb_rate_rear_Npm
            k_r, mr_r = self.decoupled_roll_rate_rear_Npm, self.decoupled_roll_MR_rear

        if mode == 'decoupled':
            # ARB rate field is ignored — decoupled has no ARB.
            return 0.25 * k_r * mr_r ** 2 * t ** 2
        # standard + heave_tbar: corner-spring contribution + ARB
        k_wheel = self._wheel_rate_for_axle(is_front)
        # HEAVE_TBAR's wheel rate already includes the heave 3rd element,
        # but the 3rd element sees no force in roll.  Subtract its
        # contribution from the wheel rate used here.
        if mode == 'heave_tbar':
            k_3rd, mr_3rd = ((self.heave_3rd_rate_front_Npm,
                              self.heave_3rd_MR_front) if is_front
                              else (self.heave_3rd_rate_rear_Npm,
                                    self.heave_3rd_MR_rear))
            k_wheel -= 0.5 * k_3rd * mr_3rd ** 2
        return (k_wheel + k_arb) * t ** 2 / 2

    @property
    def roll_stiffness_front_Npm_rad(self):
        return self._roll_stiffness_for_axle(is_front=True)

    @property
    def roll_stiffness_rear_Npm_rad(self):
        return self._roll_stiffness_for_axle(is_front=False)

    @property
    def roll_stiffness_total_Npm_rad(self):
        return self.roll_stiffness_front_Npm_rad + self.roll_stiffness_rear_Npm_rad

    # ── Static sag computation ───────────────────────────────────────
    #
    # Sag = how far the damper has compressed from its fully-extended
    # position when the car is sitting at rest.  Depends on corner
    # load, spring rate, motion ratio, and any collar preload.
    #
    # Force balance at the shock (virtual work: F_wheel·δ_w = F_shock·δ_s):
    #     F_shock_static = F_wheel_static / MR      (MR = δ_shock/δ_wheel)
    #     F_shock_static = k_spring × (preload_mm + sag_shock_mm)
    # → sag_shock_mm = F_wheel/(MR·k_spring) − preload_mm
    #   (clamped to [0, stroke]; if negative, preload alone holds the car up
    #    and the damper rests at full extension with sag = 0.)

    def static_sag(self,
                   preload_front_mm: float = 0.0,
                   preload_rear_mm:  float = 0.0,
                   stroke_mm:        float = 55.0,
                   mr_front:         float = None,
                   mr_rear:          float = None) -> dict:
        """
        Compute per-corner static sag (damper compression from full droop).

        Parameters
        ----------
        preload_front_mm, preload_rear_mm : float
            Collar preload in mm of spring compression.
        stroke_mm : float
            Total damper stroke (shock-frame travel available).
        mr_front, mr_rear : float or None
            Override motion ratios (e.g. from live kinematics). If None,
            uses self.motion_ratio_front/rear.

        Returns
        -------
        dict with keys:
            'sag_shock_front_mm', 'sag_shock_rear_mm'     — shock frame
            'sag_wheel_front_mm', 'sag_wheel_rear_mm'     — wheel frame
            'sag_front_pct', 'sag_rear_pct'               — % of stroke
            'topped_out_front', 'topped_out_rear'         — preload so high
                                                            damper sits at
                                                            full extension
            'bottomed_out_front', 'bottomed_out_rear'     — spring too soft
                                                            for the load
        """
        g = 9.81
        mr_f = mr_front if mr_front is not None else self.motion_ratio_front
        mr_r = mr_rear  if mr_rear  is not None else self.motion_ratio_rear

        # Static corner wheel loads (including unsprung weight at wheel).
        # Weight fractions come from CG, then split left/right 50/50.
        w_f = self.front_weight_fraction
        Fz_sprung_f = self.sprung_mass_kg * w_f * g / 2.0
        Fz_sprung_r = self.sprung_mass_kg * (1 - w_f) * g / 2.0
        Fz_us_f     = self.unsprung_mass_front_kg * g / 2.0
        Fz_us_r     = self.unsprung_mass_rear_kg  * g / 2.0
        Fz_f        = Fz_sprung_f + Fz_us_f
        Fz_r        = Fz_sprung_r + Fz_us_r

        # Spring compression required at static (shock frame, mm)
        # k_spring is in N/m; convert to N/mm and use mm throughout.
        k_f = self.spring_rate_front_Npm / 1000.0  # N/mm
        k_r = self.spring_rate_rear_Npm  / 1000.0
        if k_f <= 0 or mr_f <= 0:
            required_f = 0.0
        else:
            required_f = Fz_f / (mr_f * k_f)  # mm of shock compression
        if k_r <= 0 or mr_r <= 0:
            required_r = 0.0
        else:
            required_r = Fz_r / (mr_r * k_r)

        # Sag = spring compression minus preload (clamped to [0, stroke])
        raw_f = required_f - preload_front_mm
        raw_r = required_r - preload_rear_mm
        sag_f = max(0.0, min(stroke_mm, raw_f))
        sag_r = max(0.0, min(stroke_mm, raw_r))

        return {
            'sag_shock_front_mm': sag_f,
            'sag_shock_rear_mm':  sag_r,
            'sag_wheel_front_mm': sag_f / mr_f if mr_f > 0 else 0.0,
            'sag_wheel_rear_mm':  sag_r / mr_r if mr_r > 0 else 0.0,
            'sag_front_pct':      (sag_f / stroke_mm * 100.0) if stroke_mm > 0 else 0.0,
            'sag_rear_pct':       (sag_r / stroke_mm * 100.0) if stroke_mm > 0 else 0.0,
            'topped_out_front':   raw_f < 0,
            'topped_out_rear':    raw_r < 0,
            'bottomed_out_front': raw_f > stroke_mm,
            'bottomed_out_rear':  raw_r > stroke_mm,
            'required_spring_compression_front_mm': required_f,
            'required_spring_compression_rear_mm':  required_r,
            'mr_front_used': mr_f,
            'mr_rear_used':  mr_r,
        }

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

    # ── Constant-power model ─────────────────────────────────────────────
    # Drive force at any velocity comes from `drive_force_at_v_N(v)`.  The
    # old engine_torque / wheel_torque / drive_force_N properties are
    # legacy fixed-RPM snapshots — kept for backward compat but their
    # results equal `drive_force_at_v_N(top speed)` (i.e. force at peak
    # RPM, which is the smallest force in a constant-power regime).

    @property
    def wheel_power_W(self) -> float:
        """Peak power at the contact patch  =  engine power × drivetrain eff."""
        return self.power_hp * 745.7 * self.drivetrain_efficiency

    @property
    def ev_max_traction_force_N(self) -> float:
        """For EV only: the constant low-speed wheel force set by motor torque
        and gearing (motor peak torque × final ratio × eff / tire radius).
        Returns 0 for ICE or if motor torque isn't set."""
        if (self.powertrain_type != 'EV'
                or self.peak_torque_Nm <= 0
                or self.tire_radius_m <= 0):
            return 0.0
        return (self.peak_torque_Nm * self.total_drive_ratio
                * self.drivetrain_efficiency / self.tire_radius_m)

    def drive_force_at_v_N(self, v_ms: float) -> float:
        """Longitudinal drive force at velocity v (m/s).

        ICE: constant-power model — F = P_wheel / v.  Floored at v=1 m/s
            to avoid blowup at standstill (the true low-speed limit is
            tyre grip, not engine torque, for an ICE in a low gear).
        EV : min(EV motor-torque limit, constant-power limit).  Gives the
            classic flat-then-falling EV traction curve.
        """
        v_eff = max(float(v_ms), 1.0)
        F_power = self.wheel_power_W / v_eff
        if self.powertrain_type == 'EV':
            F_torque = self.ev_max_traction_force_N
            if F_torque > 0:
                return min(F_torque, F_power)
        return F_power

    # ── Legacy properties (kept for backward compat) ─────────────────────
    @property
    def engine_torque_Nm(self):
        """[Legacy] Engine torque at peak RPM.  Use drive_force_at_v_N(v)."""
        if self.engine_rpm <= 0 or self.power_hp <= 0:
            return 0.0
        omega = self.engine_rpm * 2 * np.pi / 60
        return self.power_hp * 745.7 / omega

    @property
    def wheel_torque_Nm(self):
        """[Legacy] Wheel torque at peak RPM (= engine torque × ratio × eff)."""
        return self.engine_torque_Nm * self.total_drive_ratio * self.drivetrain_efficiency

    @property
    def drive_force_N(self):
        """[Legacy] Drive force at geared top speed.  Use drive_force_at_v_N(v)."""
        return self.drive_force_at_v_N(self.speed_ms)

    @property
    def min_turn_radius_m(self):
        """Minimum turn radius from max steer angle and wheelbase."""
        if self.max_steer_angle_deg <= 0:
            return float('inf')
        return self.wheelbase_m / np.tan(np.radians(self.max_steer_angle_deg))

    @property
    def max_rack_half_travel_m(self) -> float:
        """
        Physical half-stroke of the rack in metres (symmetric about centre).
        Simply ``total_rack_travel_mm / 2`` converted to metres.
        """
        return float(self.total_rack_travel_mm) / 2.0 / 1000.0

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
        # NOTE: 'total_mass_kg' is intentionally NOT in this map — it's a
        # derived @property (sum of sprung + both unsprungs).  Old files
        # may still have it; we silently ignore those entries.
        _map = {
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
            'drivetrain_efficiency':  'drivetrain_efficiency',
            'powertrain_type':        'powertrain_type',
            'peak_torque_Nm':         'peak_torque_Nm',
            'max_steer_angle_deg':    'max_steer_angle_deg',
            'front_brake_bias':       'front_brake_bias',
            # Rack / steering geometry
            'rack_travel_per_rev_mm': 'rack_travel_per_rev_mm',
            'total_rack_travel_mm':   'total_rack_travel_mm',
            # Auto-inertia & speed-hold overrides (optional)
            'yaw_inertia_factor':       'yaw_inertia_factor',
            'roll_gyradius_track_frac': 'roll_gyradius_track_frac',
            'speed_hold_kp_per_kg':     'speed_hold_kp_per_kg',
            'speed_hold_ki_per_kg':     'speed_hold_ki_per_kg',
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

    # Tire utilization (0–1 at grip limit; >1 if demand exceeds μ·Fz)
    utilization: dict = field(default_factory=dict)

    # Per-corner lateral force (N, positive = outboard)
    Fy: dict = field(default_factory=dict)

    # Per-corner longitudinal force (N, positive = forward)
    Fx: dict = field(default_factory=dict)

    # Per-corner brake torque (Nm, positive = retarding)
    brake_torque: dict = field(default_factory=dict)

    # Understeer gradient (front avg SA - rear avg SA, positive = understeer)
    understeer_gradient_deg: float = 0.0
    diff_yaw_Nm: float = 0.0         # differential yaw moment at this point

    # Convergence info
    iterations: int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Steady-state solver
# ─────────────────────────────────────────────────────────────────────────────

G = 9.81  # m/s^2

# Minimum grip (N) for utilization ratio — avoids div-by-zero; keeps values
# order-1 when Fz is tiny (unlike a 1e-9 floor, which blows up to 1e12).
_UTIL_GRIP_FLOOR_N = 1.0


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
                 tire_model=None,
                 tire_model_rear=None):
        """
        Parameters
        ----------
        vehicle : VehicleParams
        solvers : dict
            {'FL': SuspensionConstraints, 'FR': ..., 'RL': ..., 'RR': ...}
        tire_model : TireModel | LinearTireModel | None
            Front-axle tire.  If None, the parametric fallback is used.
        tire_model_rear : TireModel | LinearTireModel | None
            Rear-axle tire.  None => same tire as the front (the usual case).
            Set it to run a SPLIT tire setup (different compound front/rear);
            every per-corner grip query then uses the axle's own tire.
        """
        self._veh = vehicle
        self._solvers = solvers
        # Always have a tire model so peak_mu / cornering_stiffness /
        # slip_angle_for_Fy never fall through to a hardcoded literal.
        # The user's loaded TTC data is preferred when available; if
        # nothing's loaded we use the parametric LinearTireModel —
        # whose own defaults (mu, C_alpha, Fz_ref) live as named
        # arguments on its __init__, not magic numbers buried in the
        # dynamics solver.
        if tire_model is None:
            from vahan.tire_model import LinearTireModel
            tire_model = LinearTireModel()
        self._tire = tire_model                       # front (and default)
        self._tire_rear = tire_model_rear or tire_model
        self._warm = {}  # per-corner warm start cache
        # Optional grip derate applied to the tire μ when computing
        # utilization (the friction-circle budget).  1.0 = raw belt μ
        # (default; the dynamics page is "what the tire data says").  The
        # lap-time sim sets this to its grip_scale so the utilization plot
        # is consistent with the lap's grip-derated speed/lat-g — otherwise
        # the two channels silently disagree by the derate factor.
        self._mu_scale = 1.0

    def _tire_for(self, label: str):
        """Tire model for a corner — front tire for FL/FR, rear for RL/RR.
        With no split set, both are the same object."""
        return self._tire if label and label[0] == 'F' else self._tire_rear

    def _diff_yaw_moment(self, ax, Fz: dict, C_a: dict):
        """Differential yaw moment (N·m, + = understeer/stabilising) at this
        operating point, and whether it's the power or coast ramp.

          * on power  (ax>0): drive torque  T = m·ax·r  through the diff
          * on coast  (ax<0): engine-overrun torque = engine_braking × ratio
            (so the COAST RAMP is a live input); brakes act at the wheels,
            outside the diff, so they're excluded.
          * the inter-wheel force bias is capped by the INNER (less-loaded)
            driven wheel's grip — it can't transmit more than μ·Fz_inner.
        """
        from vahan.differential import Differential
        v = self._veh
        diff = Differential.from_vehicle(v)
        if diff.kind == 'open':
            return 0.0, (ax > 0)
        r = max(v.tire_radius_m, 1e-3)
        dt = v.drivetrain.upper()
        driven = ['RL', 'RR'] if dt == 'RWD' else \
                 ['FL', 'FR'] if dt == 'FWD' else ['FL', 'FR', 'RL', 'RR']
        track = (v.rear_track_m if dt == 'RWD' else
                 v.front_track_m if dt == 'FWD' else
                 0.5 * (v.front_track_m + v.rear_track_m))
        on_power = ax > 0.02
        if on_power:
            T_axle = v.total_mass_kg * ax * r          # tractive torque
        elif ax < -0.02:
            ratio = float(getattr(v, 'total_drive_ratio', 10.0) or 10.0)
            T_axle = float(getattr(v, 'engine_braking_Nm', 0.0)) * ratio
        else:
            T_axle = 0.0                                # preload only
        # inner driven wheel grip cap
        fz_inner = min(max(Fz.get(c, 0.0), 0.0) for c in driven)
        tire = self._tire_for(driven[0])
        mu_inner = float(tire.peak_mu(max(fz_inner, 1.0), 0.0)) * self._mu_scale
        max_bias = mu_inner * fz_inner
        mz = diff.yaw_moment_Nm(T_axle, track, r, on_power, max_bias_N=max_bias)
        return mz, on_power

    def solve(self, lateral_g: float,
              longitudinal_g: float = 0.0,
              max_iter: int = 15,
              tol_deg: float = 0.002,
              aero_Fz: dict = None) -> SteadyStateResult:
        """
        Solve steady-state equilibrium.

        Algorithm:
        1. Compute static corner loads (+ optional aero downforce)
        2. Initial roll estimate from total roll stiffness
        3. Iterate: roll → per-corner travel → kinematics → load transfer → new roll
        4. Converge when roll angle change < tol_deg

        Parameters
        ----------
        aero_Fz : dict, optional
            Per-corner additional vertical load from aerodynamics (N).
            {'FL': N, 'FR': N, 'RL': N, 'RR': N}
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

        # Add aerodynamic downforce to static loads
        if aero_Fz:
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                Fz_static[lbl] += aero_Fz.get(lbl, 0.0)

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

        # Physical Fz floor from spring forces: the analytical LT formula
        # can predict Fz < 0 while the spring is still compressed.  The real
        # minimum Fz at each corner = spring_force_at_travel + unsprung_weight.
        # Spring force = wheel_rate × max(static_sag + travel, 0).
        #
        # The SPRING carries only the SPRUNG share of the corner load; the
        # unsprung mass (wheel/tyre/upright/hub) hangs below the spring and
        # never deflects it.  So the static sag must be computed from
        # (Fz_static − unsprung_weight), NOT the whole corner load.  Using the
        # whole load over-states the sag, hence the spring force, hence the
        # floor — by ~one unsprung weight — which then clamps even the LOADED
        # outside corner upward and collapses the realised load transfer to
        # ~44 % of analytic.  With the sprung-only sag the floor equals the
        # true static load at rest (spring_force = Fz_static − unspr_w, +unspr_w
        # back ⇒ Fz_static) and only bites the inside corner near lift-off,
        # exactly as intended.
        for (a, b) in (('FL', 'FR'), ('RL', 'RR')):
            ax_load = Fz_static[a] + Fz_static[b]
            Fz[a] = max(Fz[a], 0.0)
            Fz[b] = max(Fz[b], 0.0)
            s = Fz[a] + Fz[b]
            if s > 1e-6 and ax_load > 0:
                k = ax_load / s
                Fz[a] *= k
                Fz[b] *= k

        # Renormalize so Σ Fz = vehicle weight + aero (total vertical load is conserved)
        W_total = v.total_mass_kg * G
        if aero_Fz:
            W_total += sum(aero_Fz.values())
        Fz_sum = sum(Fz.values())
        if Fz_sum > 0:
            scale = W_total / Fz_sum
            Fz = {k: v * scale for k, v in Fz.items()}

        # ── Step 4: Build result ─────────────────────────────────────────
        result.roll_angle_deg = np.degrees(roll_rad)

        # Pitch angle from longitudinal load transfer.  The pitch couple
        # the suspension reacts is m_s · ax · h_s where h_s is the
        # sprung-mass CG height (NOT (h_cg − h_us), which has no clean
        # physical interpretation — see VehicleParams.sprung_cg_height_m).
        # Pitch stiffness = 2 · (K_wheel_front · a² + K_wheel_rear · b²)
        # with a, b = CG distance to each axle.
        a = v.cg_to_front_axle_m
        b = v.cg_to_rear_axle_m
        K_pitch = 2 * (v.wheel_rate_front_Npm * a**2 + v.wheel_rate_rear_Npm * b**2)
        pitch_moment = v.sprung_mass_kg * ax * v.sprung_cg_height_m
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

            # Get cornering stiffness at each corner's dynamic Fz.
            # Below the tire data range, scale C_a linearly → 0 so that
            # both demand (∝ C_a) and grip (∝ Fz) vanish together,
            # keeping utilization = demand/grip smooth through wheel lift.
            # `fz_range` is a TTC-specific attribute (lowest test load
            # for cornering-stiffness extrapolation).  LinearTireModel
            # is parametric and doesn't have a "test range" — treat its
            # min as 0 so we never bypass any data.
            C_a = {}
            for label in ['FL', 'FR', 'RL', 'RR']:
                tire = self._tire_for(label)          # split-tire aware
                fz_data_min = float(getattr(tire, 'fz_range', (0.0,))[0])
                fz_raw = max(Fz[label], 0.0)
                fz_c = max(fz_raw, fz_data_min)
                ca = abs(float(tire.cornering_stiffness(
                    fz_c, abs(cambers.get(label, 0)))))
                # Linear ramp below data range: C_a(Fz) → 0 as Fz → 0
                if fz_raw < fz_data_min:
                    ca *= fz_raw / fz_data_min
                C_a[label] = ca

            # Distribute Fy within each axle by cornering stiffness alone.
            # Both tires on an axle share the same slip angle α, so each
            # produces Fy = Cα(Fz)·α  →  left/right split ∝ Cα.
            fy_per_corner = {}
            Wf = C_a['FL'] + C_a['FR']
            Wr = C_a['RL'] + C_a['RR']
            if Wf > 0:
                fy_per_corner['FL'] = fy_front_axle * C_a['FL'] / Wf
                fy_per_corner['FR'] = fy_front_axle * C_a['FR'] / Wf
            else:
                fy_per_corner['FL'] = fy_front_axle / 2
                fy_per_corner['FR'] = fy_front_axle / 2
            if Wr > 0:
                fy_per_corner['RL'] = fy_rear_axle * C_a['RL'] / Wr
                fy_per_corner['RR'] = fy_rear_axle * C_a['RR'] / Wr
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
                tire = self._tire_for(label)          # split-tire aware
                fz_lo = float(getattr(tire, 'fz_range', (0.0,))[0])
                fz_raw = max(Fz[label], 0.0)
                fz_for_mu = max(fz_raw, fz_lo)        # stable mu eval
                mu = float(tire.peak_mu(
                    fz_for_mu, abs(cambers.get(label, 0.0)))) * self._mu_scale
                grip_budget = mu * max(fz_raw, 0.01)  # → 0 smoothly
                fy_req = fy_per_corner.get(label, 0.0)
                fx_req = fx_per_corner.get(label, 0.0)
                combined = np.sqrt(fy_req ** 2 + fx_req ** 2)
                result.utilization[label] = combined / grip_budget

            # Store per-corner forces for component load analysis
            result.Fy = dict(fy_per_corner)
            result.Fx = dict(fx_per_corner)
            # Brake torque per corner = Fx × tire_radius — ONLY while braking.
            # Under power (ax > 0) the driven-axle Fx is DRIVE force: its hub
            # torque is reacted through the DRIVESHAFT into the diff, not by the
            # brake caliper (the pads are not clamping).  Booking it as brake
            # torque put ~1.9 kN of phantom load into the caliper mount lugs at
            # full acceleration.  The hub torque itself is still reported — see
            # the "brake/drive torque (about axle)" moment in wheel_package.
            tire_r = self._veh.tire_radius_m
            braking = ax < 0
            for label in ['FL', 'FR', 'RL', 'RR']:
                result.brake_torque[label] = (
                    abs(fx_per_corner.get(label, 0)) * tire_r if braking else 0.0)

            # Understeer gradient: back-calculate slip angles from tire model
            # (per-axle tire so a split setup shifts the balance correctly).
            if abs(ay) > 0.1 and hasattr(self._tire, 'slip_angle_for_Fy') \
                    and hasattr(self._tire_rear, 'slip_angle_for_Fy'):
                try:
                    sa_fl = self._tire.slip_angle_for_Fy(
                        fy_per_corner['FL'], max(Fz['FL'], 1.0), abs(cambers.get('FL', 0)))
                    sa_fr = self._tire.slip_angle_for_Fy(
                        fy_per_corner['FR'], max(Fz['FR'], 1.0), abs(cambers.get('FR', 0)))
                    sa_rl = self._tire_rear.slip_angle_for_Fy(
                        fy_per_corner['RL'], max(Fz['RL'], 1.0), abs(cambers.get('RL', 0)))
                    sa_rr = self._tire_rear.slip_angle_for_Fy(
                        fy_per_corner['RR'], max(Fz['RR'], 1.0), abs(cambers.get('RR', 0)))
                    sa_front = (sa_fl + sa_fr) / 2
                    sa_rear = (sa_rl + sa_rr) / 2
                    result.understeer_gradient_deg = sa_front - sa_rear
                except Exception:
                    pass

            # ── DIFFERENTIAL: yaw moment -> understeer-gradient shift ──────
            # The driven-axle drive/overrun torque, biased by the diff, makes
            # a yaw moment that the tires must react with a front/rear lateral-
            # force couple ΔFy = Mz_diff / wheelbase -> a slip-angle shift on
            # each axle -> a real understeer-gradient change.  This is what
            # closes the loop: the diff now MOVES the balance, not just a plot.
            try:
                mz_diff, on_power = self._diff_yaw_moment(ax, Fz, C_a)
                result.diff_yaw_Nm = mz_diff
                L = max(v.wheelbase_m, 1e-6)
                Ca_f = max(C_a['FL'] + C_a['FR'], 1e-6)   # N/deg
                Ca_r = max(C_a['RL'] + C_a['RR'], 1e-6)
                d_understeer = (mz_diff / L) * (1.0 / Ca_f + 1.0 / Ca_r)
                result.understeer_gradient_deg += d_understeer
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
            for label in ['FL', 'FR', 'RL', 'RR']:   # brake torque only under braking
                result.brake_torque[label] = (
                    abs(result.Fx.get(label, 0)) * tire_r if ax < 0 else 0.0)

        return result

    def sweep_by_speed(self,
                       v_min_mph: float,
                       v_max_mph: float,
                       turn_radius_m: float,
                       n_points: int = 41,
                       longitudinal_g: float = 0.0,
                       aero_Fz: dict = None) -> dict:
        """Sweep speed (X-axis) at fixed turn radius — derive lat-g.

        Companion to ``sweep_lateral_g`` for the "Sweep by: Speed"
        option.  At constant R, lat-g and v are linked by
            v² = a_y · g_e · R
            ⇒  a_y = v² / (g_e · R)
        Each sweep step picks v on a uniform grid, computes the
        implied lat-g, and runs the same steady-state dynamics solve.
        Returns a dict keyed like ``sweep_lateral_g`` but with
        ``speed_mph`` and ``speed_kph`` as the primary X array and
        ``lateral_g`` as the **derived** values.

        ``longitudinal_g`` is held fixed (just like sweep_combined) —
        if non-zero, it shows up in pitch / load transfer / utilization
        but doesn't change the centripetal speed.
        """
        v_arr_mph = np.linspace(v_min_mph, v_max_mph, n_points)
        v_arr_ms  = v_arr_mph / 2.23694
        if turn_radius_m > 1e-6:
            lat_arr = (v_arr_ms ** 2) / (9.81 * turn_radius_m)
        else:
            lat_arr = np.zeros(n_points)

        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {
            'speed_mph':       v_arr_mph,
            'speed_kph':       v_arr_mph * 1.609344,
            'lateral_g':       lat_arr,        # derived from v + R
            'turn_radius_m':   turn_radius_m,
        }
        for k in keys:
            out[k] = np.zeros(n_points)
        for ck in corner_keys:
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                out[f'{ck}_{lbl}'] = np.zeros(n_points)

        self._warm = {}
        mu = self._effective_mu()
        # Track friction-clamp like sweep_combined does so the user can
        # see when the implied lat-g exceeds the circle.
        out['lat_g_applied']  = np.zeros(n_points)
        out['lon_g_applied']  = np.zeros(n_points)
        out['friction_clamp'] = np.zeros(n_points, dtype=bool)

        for i, lat_g in enumerate(lat_arr):
            lat_c, lon_c, clamped = self._clamp_to_friction_circle(
                float(lat_g), float(longitudinal_g), mu)
            r = self.solve(lat_c, lon_c, aero_Fz=aero_Fz)
            out['lat_g_applied'][i]   = lat_c
            out['lon_g_applied'][i]   = lon_c
            out['friction_clamp'][i]  = clamped
            out['roll_angle_deg'][i]       = r.roll_angle_deg
            out['pitch_angle_deg'][i]      = r.pitch_angle_deg
            out['rc_height_front_mm'][i]   = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i]    = r.rc_height_rear_m  * 1000
            out['elastic_lt_front_N'][i]   = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i]    = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i]  = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                out[f'Fz_{lbl}'][i]          = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i]      = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i]      = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)

        for k in ['understeer_gradient_deg']:
            if len(out[k]) >= 5:
                out[k] = uniform_filter1d(out[k], size=5, mode='nearest')
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = uniform_filter1d(out[uk], size=3, mode='nearest')

        return out

    def sweep_acceleration_trajectory(self,
                                       start_speed_mph: float = 0.0,
                                       lateral_g: float = 0.0,
                                       a_threshold_g: float = 0.01,
                                       max_steps: int = 5000,
                                       direction: str = 'accel',
                                       end_speed_mph: float = 0.0,
                                       target_lon_g: float = 0.0) -> dict:
        """Time-domain longitudinal trajectory.

        ``target_lon_g`` (signed): the longitudinal-g the driver is
        applying.  Sign drives direction — no separate toggle:
            > 0  → acceleration; per-step g = min(target, traction, P/(m·v))
            < 0  → braking;       per-step |g| = min(|target|, μ-circle)
            = 0  → coast (drag only; from rest → stationary).
        Per-step physics still clamps to what the tires + engine can
        actually deliver.  Drag (CdA, ρ) is always subtracted from the
        achievable accel.

        End conditions:
            • Accel: terminate when achievable g decays below
              ``a_threshold_g · traction_g`` (≈ at terminal speed).
            • Brake: terminate when v reaches 0 (or end_speed_mph).
        """
        v = self._veh
        # Use WHEEL power (= engine power × drivetrain efficiency).  Drag
        # balances WHEEL power at terminal speed, not engine power, so use
        # the new derived property.  For EV the low-speed torque limit is
        # applied per-step via drive_force_at_v_N below.
        P  = v.wheel_power_W
        M  = v.total_mass_kg
        rho = v.air_density_kg_m3
        CdA = v.cda_m2
        gE  = G                                      # 9.80665

        # Drivetrain-aware traction limit (driven axle saturation) — includes
        # longitudinal weight transfer onto the driven axle (fixed point with
        # load-sensitive μ; friction circle vs the lateral_g argument).
        mu = self._effective_mu()
        traction_g = self._traction_g_dynamic(lateral_g)
        dt_kind = v.drivetrain.upper()

        # Terminal speed: drag balances engine power.
        if rho > 0 and CdA > 0 and P > 0:
            v_terminal_ms = (2 * P / (rho * CdA)) ** (1.0 / 3.0)
        else:
            # Without drag the trajectory is unbounded — fall back to
            # the user's start_speed plus the traction-accel scale, so
            # the loop terminates after a sensible distance.  Same code
            # path is fine; just won't asymptote naturally.
            v_terminal_ms = max(start_speed_mph / 2.23694, 1.0) * 10.0

        # Time step from physics: 1 % of (terminal / max accel).
        if traction_g > 1e-6:
            dt = 0.01 * v_terminal_ms / (traction_g * gE)
        else:
            dt = 0.05
        dt = max(dt, 1e-3)                           # numerical floor

        # Integration loop
        v_start_ms = max(start_speed_mph / 2.23694, 0.0)
        times   = [0.0]
        speeds  = [v_start_ms]
        g_appl  = []
        a_term_threshold_si = a_threshold_g * traction_g * gE   # m/s²

        # Direction is ALWAYS derived from the sign of target_lon_g.
        # The UI exposes a single signed lon-g spinner — positive means
        # the driver is on the throttle, negative means brakes, zero
        # means coasting (drag only).  There is no separate "direction"
        # dropdown; the sign IS the input.
        target_lon = float(target_lon_g)
        is_braking = target_lon < -1e-9
        target_mag = abs(target_lon)          # 0 = coast (not full throttle)

        # Optional speed cap.  When end_speed_mph > 0, terminate the run
        # once v crosses it (in either direction depending on accel/brake).
        # Default 0 = "no cap, run to natural endpoint".
        end_v_ms = float(end_speed_mph) / 2.23694 if end_speed_mph > 0 else None

        # For braking, the friction circle uses full 4-tire μ (not the
        # drivetrain-derived traction_g, which is only relevant for
        # acceleration).  Lateral demand eats into the available
        # longitudinal grip via √(μ² − a_y²).
        brake_mu = mu
        if is_braking:
            ay_g = abs(float(lateral_g))
            if ay_g >= brake_mu:
                brake_g = 0.0                         # no grip left for brakes
            else:
                brake_g = float(np.sqrt(brake_mu * brake_mu - ay_g * ay_g))

        for _ in range(max_steps):
            v_now = speeds[-1]

            if is_braking:
                # Constant max-grip deceleration; no power/drag terms
                # — the brakes are dissipating, not the engine doing
                # work.  Drag still helps slow the car but is small
                # compared to brake force at typical FSAE speeds.
                F_drag = 0.5 * rho * CdA * v_now * v_now
                # Cap brake-g by user target.  brake_g is the full
                # circle-limited maximum; user can ask for less (e.g.
                # 0.5g brake when 1.5g is available).  target_mag = 0
                # → no braking force (coast / drag only).
                brake_eff = min(brake_g, target_mag)
                a_brake_si = brake_eff * gE
                a_drag_si  = F_drag / M
                a_signed   = -(a_brake_si + a_drag_si)   # m/s², negative
                g_appl.append(a_signed / gE)             # signed (-)

                # Terminate when v reaches 0 (or the user-set floor).
                stop_v = end_v_ms if end_v_ms is not None else 0.0
                if v_now <= max(stop_v, 1e-3) and len(times) > 5:
                    break
                v_next = max(stop_v, v_now + a_signed * dt)
                times.append(times[-1] + dt)
                speeds.append(v_next)
            else:
                # Acceleration: engine/motor vs drag, traction-limited at low v.
                # drive_force_at_v_N handles both ICE (constant power) and EV
                # (constant torque at low v, constant power above).
                if v_now > 0.5:                       # 0.5 m/s ≈ 1 mph
                    F_pwr = v.drive_force_at_v_N(v_now)
                else:
                    F_pwr = traction_g * M * gE       # very low v → traction
                F_drag = 0.5 * rho * CdA * v_now * v_now
                F_net  = F_pwr - F_drag
                a_pwr  = F_net / M
                a_trac = traction_g * gE
                a      = max(0.0, min(a_pwr, a_trac))
                # Cap by user target (in m/s²).  User can ask for less
                # than the achievable max (e.g. 0.3g instead of full-
                # throttle 0.83g), but not more — physics already
                # capped above.  target_mag = 0 → zero throttle (coast).
                a = min(a, target_mag * gE)
                g_appl.append(a / gE)

                # Terminate when achievable accel decays toward terminal,
                # or when v hits the user-set ceiling.
                if a < a_term_threshold_si and len(times) > 5:
                    break
                if end_v_ms is not None and v_now >= end_v_ms and len(times) > 1:
                    break
                v_next = v_now + a * dt
                times.append(times[-1] + dt)
                speeds.append(v_next)

        # The last sample missed g; copy the final value.
        if len(g_appl) < len(times):
            g_appl.append(g_appl[-1] if g_appl else 0.0)

        t_arr  = np.asarray(times, float)
        v_ms   = np.asarray(speeds, float)
        g_arr  = np.asarray(g_appl, float)
        n      = len(t_arr)

        out = {
            'time_s':         t_arr,
            'speed_mph':      v_ms * 2.23694,
            'speed_kph':      v_ms * 3.6,
            'longitudinal_g': g_arr,
        }
        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        for k in keys:
            out[k] = np.zeros(n)
        for ck in ('Fz', 'travel', 'camber', 'utilization'):
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                out[f'{ck}_{lbl}'] = np.zeros(n)

        # Run the steady-state dynamics solver at each (lat, lon) point.
        self._warm = {}
        for i in range(n):
            r = self.solve(lateral_g, float(g_arr[i]))
            out['roll_angle_deg'][i]       = r.roll_angle_deg
            out['pitch_angle_deg'][i]      = r.pitch_angle_deg
            out['rc_height_front_mm'][i]   = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i]    = r.rc_height_rear_m  * 1000
            out['elastic_lt_front_N'][i]   = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i]    = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i]  = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                out[f'Fz_{lbl}'][i]          = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i]      = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i]      = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)
        return out

    def _effective_mu(self) -> float:
        """Vehicle-level peak μ used by the friction-circle clamp.

        Always pulled from the tire model — TTC data when loaded, or
        the LinearTireModel parametric fallback installed by __init__
        when none.  Average front + rear at the static-Fz operating
        point (zero camber).  Used as the radius of the friction circle
        √(a_x² + a_y²) ≤ μ for clamping impossible (lat, lon)
        operating points in the combined sweep.
        """
        v = self._veh
        W = v.total_mass_kg * G
        Fz_f = W * v.front_weight_fraction / 2
        Fz_r = W * v.rear_weight_fraction / 2
        mu_f = float(self._tire.peak_mu(Fz_f, 0.0))        # front tire
        mu_r = float(self._tire_rear.peak_mu(Fz_r, 0.0))   # rear tire
        # Conservative: use the lower of the two so we don't pretend the
        # rear can save the front when the front saturates first.
        return min(mu_f, mu_r)

    def _clamp_to_friction_circle(self, lat_g: float,
                                   lon_g: float,
                                   mu: float) -> tuple:
        """Drivetrain-aware friction-circle clamp on (lat, lon).

        Lateral force comes from all 4 tires; longitudinal force comes
        from whatever subset is doing the work.  Per-tire saturation:
            (F_y/F_z)² + (F_x/F_z)²  ≤  μ²
        Re-arranged in g-units with each tire taking its weight share:
            a_x ≤ k_drive · √(μ² − a_y²)
        where k_drive is the **fraction of vehicle weight on the
        force-applying axle/tires** — drivetrain-dependent for accel,
        always 1.0 for braking (all 4 tires brake).

            RWD accel : k_drive = rear_weight_fraction
            FWD accel : k_drive = front_weight_fraction
            AWD accel : k_drive = 1.0
            Braking   : k_drive = 1.0

        For the user's "1g lat + 1g lon RWD-accel" case at μ=1.5:
            k_drive = 0.55  (rear weight fraction)
            a_x_max = 0.55 · √(1.5² − 1²) = 0.615 g
        The requested 1g of acceleration gets clamped to 0.615g, and
        pitch drops with it — visible in the combined sweep instead of
        sitting flat past the lat-g where the rear tires actually
        saturate.

        Returns (lat_clamped, lon_clamped, was_clamped: bool).
        """
        mu = max(float(mu), 1e-3)
        lat = float(lat_g)
        lon = float(lon_g)
        lat_abs = abs(lat)

        # Pure-cornering limit — all 4 tires can contribute laterally,
        # so this uses the full μ.
        if lat_abs >= mu:
            return (np.sign(lat) * mu, 0.0, True)

        # Drivetrain- and brake-bias-aware longitudinal capacity.
        # k_drive accounts for the per-tire friction circle on the
        # axle that bites first.  Derivation:
        #   per-tire saturation : (F_y/F_z)² + (F_x/F_z)² ≤ μ²
        # which in g-units gives, for the limiting tire:
        #   a_x_max = (axle_weight_frac / axle_force_frac) · √(μ² − a_y²)
        # k_drive = (axle_weight_frac / axle_force_frac) is the ratio of
        # weight on that axle to its share of the longitudinal force.
        # Smaller k_drive ⇒ tire is over-loaded laterally vs. it's
        # carrying the same lon share ⇒ clamp bites earlier.
        v = self._veh
        if lon >= 0:
            # Acceleration: driven tires take the lon force.  Use the SAME
            # dynamic-weight-transfer fixed point as max_accel_g / the
            # trajectory (load-sensitive μ + transfer onto the driven axle +
            # the lateral-demand circle), so the combined sweep's accel clamp
            # and the launch model can never disagree.
            max_lon_dyn = self._traction_g_dynamic(lat)
            if abs(lon) <= max_lon_dyn:
                return (lat, lon, False)
            return (lat, np.sign(lon) * max_lon_dyn, True)
        else:
            # Braking: all 4 tires brake, but brake-bias usually puts
            # MORE longitudinal share on the front (e.g. 65 %) than
            # the front carries by static weight (e.g. 45 %).  Front
            # tires saturate first.  k_drive = front_frac / brake_bias
            # measures that lopsidedness.
            #   45 % weight + 65 % brake bias  →  k = 0.45/0.65 = 0.692
            #   So clamp bites well before the naive μ-circle.
            bb = float(v.front_brake_bias)
            if bb >= 1e-3:
                k_drive_front = v.front_weight_fraction / bb
            else:
                k_drive_front = float('inf')          # no front brakes
            # Rear-tire side, for completeness:
            rb = 1.0 - bb
            if rb >= 1e-3:
                k_drive_rear = v.rear_weight_fraction / rb
            else:
                k_drive_rear = float('inf')           # no rear brakes
            # Whichever axle saturates first sets the limit.
            k_drive = min(k_drive_front, k_drive_rear)

        circle_remaining = np.sqrt(mu * mu - lat_abs * lat_abs)
        max_lon = k_drive * circle_remaining
        if abs(lon) <= max_lon:
            return (lat, lon, False)
        return (lat, np.sign(lon) * max_lon, True)

    def sweep_lateral_g(self,
                        g_range: tuple = (0.0, 2.0),
                        n_points: int = 41,
                        longitudinal_g: float = 0.0,
                        aero_Fz: dict = None) -> dict:
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
            r = self.solve(lg, longitudinal_g, aero_Fz=aero_Fz)
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

        # Smooth noisy signals — tire inverse lookup + kinematic solver edges
        for k in ['understeer_gradient_deg']:
            if len(out[k]) >= 5:
                out[k] = uniform_filter1d(out[k], size=5, mode='nearest')
        for lbl in ['FL', 'FR', 'RL', 'RR']:
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = uniform_filter1d(out[uk], size=3, mode='nearest')

        return out

    def sweep_longitudinal_g(self,
                             g_range: tuple = (-2.0, 2.0),
                             n_points: int = 41,
                             lateral_g: float = 0.0,
                             aero_Fz: dict = None) -> dict:
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
            r = self.solve(lateral_g, lg, aero_Fz=aero_Fz)
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

        for k in ['understeer_gradient_deg']:
            if len(out[k]) >= 5:
                out[k] = uniform_filter1d(out[k], size=5, mode='nearest')
        for lbl in ['FL', 'FR', 'RL', 'RR']:
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = uniform_filter1d(out[uk], size=3, mode='nearest')

        return out

    def sweep_combined(self,
                       lat_range: tuple = (0.0, 2.0),
                       lon_g: float = -0.5,
                       n_points: int = 41,
                       aero_Fz: dict = None) -> dict:
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
        # Track the actually-applied (lat, lon) after the friction
        # clamp so plots can show the *real* operating point, not the
        # one the user typed.  This is what makes pitch drop with lat-g
        # in a combined cornering+braking sweep — past the circle, the
        # car can no longer hold the requested lon-g and the actual
        # value (and therefore pitch) decays.
        out['lon_g_applied']  = np.zeros(n_points)
        out['lat_g_applied']  = np.zeros(n_points)
        out['friction_clamp'] = np.zeros(n_points, dtype=bool)

        self._warm = {}
        mu = self._effective_mu()

        for i, lat_g in enumerate(g_arr):
            lat_c, lon_c, clamped = self._clamp_to_friction_circle(lat_g, lon_g, mu)
            r = self.solve(lat_c, lon_c, aero_Fz=aero_Fz)
            out['lat_g_applied'][i]  = lat_c
            out['lon_g_applied'][i]  = lon_c
            out['friction_clamp'][i] = clamped
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
        for lbl in ['FL', 'FR', 'RL', 'RR']:
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = uniform_filter1d(out[uk], size=3, mode='nearest')

        return out

    def sweep_acceleration(self,
                           v_min_kph: float = 0.0,
                           v_max_kph: float = 200.0,
                           n_points: int = 41,
                           lateral_g: float = 0.0,
                           aero_Fz: dict = None) -> dict:
        """
        Trajectory sweep: accelerate from rest along a real driving curve.

        At each sample speed v, the longitudinal-g is whatever the car can
        actually deliver:
            g(v) = min(traction_limit, P/(m·g·v))           — accel
            g(v) = min(braking_limit, ...)                  — would be if
                   v_max < v_min (i.e. decel sweep)
        Speed grows monotonically from v_min to v_max on the X-axis, and
        the longitudinal-g traces the traction-then-power-limited envelope
        that an actual driver would experience under full throttle.

        This replaces the old "sweep g from g_min to g_max" semantics for
        longitudinal — that one was an envelope of operating-point steady-
        states, not a trajectory, and confused everyone because the
        speed–g relationship is inverse (high-g only at low-speed in the
        power-limited regime).

        Returns a dict identical in shape to ``sweep_longitudinal_g`` but
        with the X-axis as **speed**:
            'speed_kph'      — primary X array (kph)
            'speed_mph'      — same in mph for plot convenience
            'longitudinal_g' — g actually achieved at each speed (the
                               traction/power envelope)
        """
        v_arr_kph = np.linspace(v_min_kph, v_max_kph, n_points)

        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {
            'speed_kph':      v_arr_kph,
            'speed_mph':      v_arr_kph / 1.609344,
            'longitudinal_g': np.zeros(n_points),
        }
        for k in keys:
            out[k] = np.zeros(n_points)
        for ck in corner_keys:
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'{ck}_{lbl}'] = np.zeros(n_points)

        self._warm = {}

        for i, vk in enumerate(v_arr_kph):
            # Achievable g at this speed (min of traction and power)
            accel = self.max_accel_g(speed_kph=vk, lateral_g=lateral_g)
            g_eff = accel['effective_g']
            out['longitudinal_g'][i] = g_eff

            # Steady-state dynamics at that g
            r = self.solve(lateral_g, g_eff, aero_Fz=aero_Fz)
            out['roll_angle_deg'][i]       = r.roll_angle_deg
            out['pitch_angle_deg'][i]      = r.pitch_angle_deg
            out['rc_height_front_mm'][i]   = r.rc_height_front_m * 1000
            out['rc_height_rear_mm'][i]    = r.rc_height_rear_m * 1000
            out['elastic_lt_front_N'][i]   = r.elastic_lt_front_N
            out['elastic_lt_rear_N'][i]    = r.elastic_lt_rear_N
            out['geometric_lt_front_N'][i] = r.geometric_lt_front_N
            out['geometric_lt_rear_N'][i]  = r.geometric_lt_rear_N
            out['understeer_gradient_deg'][i] = r.understeer_gradient_deg
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'Fz_{lbl}'][i]          = r.Fz.get(lbl, 0)
                out[f'travel_{lbl}'][i]      = r.travel.get(lbl, 0)
                out[f'camber_{lbl}'][i]      = r.camber.get(lbl, 0)
                out[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)

        # Smoothing — same convention as sweep_longitudinal_g
        for k in ['understeer_gradient_deg']:
            if len(out[k]) >= 5:
                out[k] = uniform_filter1d(out[k], size=5, mode='nearest')
        for lbl in ['FL', 'FR', 'RL', 'RR']:
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = uniform_filter1d(out[uk], size=3, mode='nearest')

        return out

    def _traction_g_dynamic(self, lateral_g: float = 0.0) -> float:
        """Grip-limited forward accel INCLUDING longitudinal weight transfer
        onto the driven axle (the README's documented static-vs-dynamic gap:
        static rear_frac·μ ≈ 0.83 g; with transfer a/g = μ·rear/(1 − μ·h/L)
        ≈ 1.16 g for the default car).  Solved as a small fixed point so the
        load-sensitive μ(Fz) from the tire model is honoured, with the
        friction circle (lateral demand) reducing available longitudinal μ."""
        v = self._veh
        W = v.total_mass_kg * G
        h_L = v.cg_height_m / max(v.wheelbase_m, 1e-6)
        dt_kind = v.drivetrain.upper()
        ay = abs(float(lateral_g))
        a_g = 0.0
        for _ in range(15):
            transfer = a_g * h_L                  # ΔW/W shifted rearward
            if dt_kind == 'RWD':
                frac = v.rear_weight_fraction + transfer
            elif dt_kind == 'FWD':
                frac = v.front_weight_fraction - transfer
            else:                                  # AWD: all wheels driven
                frac = 1.0
            frac = min(max(frac, 0.0), 1.0)
            Fz_per = W * frac / (2.0 if dt_kind != 'AWD' else 4.0)
            # μ of the DRIVEN axle's tire (rear for RWD, front for FWD, the
            # softer of the two for AWD so it's not optimistic on a split).
            if dt_kind == 'RWD':
                mu = float(self._tire_rear.peak_mu(Fz_per, 0.0))
            elif dt_kind == 'FWD':
                mu = float(self._tire.peak_mu(Fz_per, 0.0))
            else:
                mu = min(float(self._tire.peak_mu(Fz_per, 0.0)),
                         float(self._tire_rear.peak_mu(Fz_per, 0.0)))
            mu_eff = float(np.sqrt(max(mu * mu - ay * ay, 0.0)))
            a_new = mu_eff * frac
            if abs(a_new - a_g) < 1e-6:
                a_g = a_new
                break
            a_g = a_new
        return float(a_g)

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
        # Tire model is guaranteed by SteadyStateSolver.__init__ —
        # either the user's loaded TTC data or the parametric
        # LinearTireModel fallback.  No hardcoded mu literal here.
        Fz_front = W * v.front_weight_fraction / 2
        Fz_rear  = W * v.rear_weight_fraction / 2
        mu_f = float(self._tire.peak_mu(Fz_front, 0.0))       # front tire
        mu_r = float(self._tire_rear.peak_mu(Fz_rear,  0.0))  # rear tire

        # Traction limit (depends on driven axle) — WITH longitudinal weight
        # transfer onto the driven axle (fixed-point incl. load-sensitive μ).
        # The old static-distribution number under-predicted RWD launch by
        # ~30 % (0.83 g vs 1.16 g for the default car — see README note).
        traction_g = self._traction_g_dynamic(lateral_g)

        # Power-limit force: routed through drive_force_at_v_N so EV
        # motors get the constant-torque low-speed cap, and drivetrain
        # efficiency is applied uniformly.
        power_W = v.wheel_power_W
        if speed_kph > 1.0 and power_W > 0:
            v_ms = speed_kph / 3.6
            power_force = v.drive_force_at_v_N(v_ms)
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

    def axle_utilization(self, result: 'SteadyStateResult') -> dict:
        """CANONICAL grip-limit criterion: per-AXLE aggregate utilization.

            util_axle = (sum of demanded planar tire force on the axle)
                        / (sum of mu(Fz, camber) * mu_scale * Fz on the axle)

        Per-corner MAX utilization is NOT a limit criterion: at high lateral g
        the unloaded inner tire sits below the tire-data floor where its
        utilization is an interpolation artifact (2026-07-11 refuter finding
        — it produced fake grip 'collapses' and a 6 kN downforce-required
        plateau).  An axle with an LSD works as a pair, so the pair budget is
        the physical limit.  Uses the SAME floor-clamped, belt->track-scaled
        mu path as the solve() grip budget so every consumer (G-G, aero
        solver, design-city, reports) shares ONE definition — the single-model
        rule applied to the limit metric itself.
        """
        out = {}
        for ax, pair in (('F', ('FL', 'FR')), ('R', ('RL', 'RR'))):
            dem = 0.0
            cap = 0.0
            for c in pair:
                fy = abs(float(result.Fy.get(c, 0.0))) if result.Fy else 0.0
                fx = abs(float(result.Fx.get(c, 0.0))) if result.Fx else 0.0
                dem += float(np.hypot(fy, fx))
                fz = max(float(result.Fz.get(c, 0.0)), 0.0)
                tire = self._tire_for(c)
                if tire is None:
                    continue
                fz_lo = float(np.asarray(tire.fz_range).ravel()[0])
                mu = float(tire.peak_mu(max(fz, fz_lo),
                                        abs(result.camber.get(c, 0.0))))
                cap += mu * self._mu_scale * fz
            out[ax] = dem / max(cap, 1e-6)
        return out

    def _clamp_and_renormalize_fz(self, Fz: dict, W_total: float) -> dict:
        """
        Enforce physically plausible normal loads (no tension on the ground).

        The linear load-transfer model can predict inside Fz < 0 at high lateral
        g (wheel lift). Clamp each corner to ≥ 0, then scale so the four corners
        still sum to total vehicle weight.
        """
        labels = ('FL', 'FR', 'RL', 'RR')
        pos = np.array([max(0.0, float(Fz[l])) for l in labels])
        s = float(pos.sum())
        if s <= 1e-9:
            v = W_total / 4.0
            return {l: v for l in labels}
        scale = W_total / s
        return {l: float(pos[i] * scale) for i, l in enumerate(labels)}

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
        """Compute roll angle (rad) from lateral acceleration and RC heights.

        Per RCVD eq. (p682):
            phi / ay  =  -W_s * h2  /  (K_F + K_R  -  W_s * h2)

        The ``-W_s * h2`` term in the denominator is the **gravity
        stabilisation** — once the body has rolled, the sprung weight
        acts at a moment arm of ``h2 * sin(phi) ~= h2 * phi`` about the
        roll axis, providing a destabilising torque.  Omitting it (as
        Vahan did before this fix) over-estimates roll stiffness by
        ``W_s * h2`` and under-predicts roll angle by ~5 % at FSAE-scale
        cornering forces (1.5 g, K_total ~ 3000 N*m/rad).
        """
        # Roll axis height at CG longitudinal position
        b = v.cg_to_front_axle_m / v.wheelbase_m  # fraction from front
        h_roll_axis = rc_f * (1 - b) + rc_r * b

        # Sprung-mass roll moment.  h_arm is the lever from the roll
        # axis up to the SPRUNG-mass CG -- see VehicleParams.sprung_cg_height_m
        # for why this differs from the whole-vehicle CG.
        h_arm = v.sprung_cg_height_m - h_roll_axis
        roll_moment = v.sprung_mass_kg * ay * h_arm  # N·m

        # Roll stiffness MINUS gravity-stabilisation term.  W_s * h_arm
        # = sprung weight times lever to roll axis.
        #
        # Negative K_eff means the gravity-induced overturning moment
        # exceeds the suspension's restoring stiffness -- physically the
        # car would tip over before reaching this lateral g.  For static
        # analysis we cap K_eff at a small positive number to avoid
        # huge/infinite roll angles in the output; the underlying
        # condition (K_total < W_s * h_arm) indicates an under-sprung
        # car for that CG height.
        K_grav = v.sprung_mass_kg * G * h_arm   # N*m/rad
        K_total = v.roll_stiffness_total_Npm_rad
        K_eff  = K_total - K_grav
        if K_eff < 1.0:
            # Static analysis breaks down here -- the car is unstable
            # in roll at this combination of h_arm + roll stiffness.
            # Return a large but finite roll value (saturated) rather
            # than 0 (which would silently look stable).  Use the
            # un-stabilised K_total so the answer is still meaningful
            # as a "soft" upper-bound estimate.
            if K_total < 1.0:
                return 0.0
            return roll_moment / K_total
        return roll_moment / K_eff

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

        # Geometric (direct through roll centre, no body roll needed).
        # Per RCVD section 18.4 the geometric LT uses the SPRUNG-mass front
        # fraction (Wsp / Ws), not the whole-vehicle front fraction (W_F / W).
        # The sprung CG is at a_s longitudinally; under the rigid-body
        # assumption m_s * a_s + m_u_F * 0 + m_u_R * L = m * a, so:
        #     a_s = (m * a - m_u_R * L) / m_s
        # which differs from `a` (whole-vehicle) by the unsprung-mass
        # distribution.  At FSAE scale this changes geometric LT by ~2 %
        # but it is the only place Vahan failed to be self-consistent
        # about sprung-vs-whole-vehicle CG (see VehicleParams.sprung_cg_height_m
        # for the analogous Z-direction correction).
        m_s = v.sprung_mass_kg
        if m_s > 1e-3:
            a_sprung = (v.total_mass_kg * v.cg_to_front_axle_m
                        - v.unsprung_mass_rear_kg * v.wheelbase_m) / m_s
            sprung_front_frac = max(0.0, min(1.0,
                                    (v.wheelbase_m - a_sprung) / v.wheelbase_m))
        else:
            sprung_front_frac = v.front_weight_fraction
        sprung_rear_frac = 1.0 - sprung_front_frac
        geo_front = m_s * sprung_front_frac * ay * rc_f / v.front_track_m
        geo_rear  = m_s * sprung_rear_frac  * ay * rc_r / v.rear_track_m

        # Elastic (through springs + ARB, proportional to roll stiffness dist)
        h_arm = v.sprung_cg_height_m - h_roll_axis
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
    'lltd_pct',            # TOTAL LT front share (elastic+geometric+unsprung) × 100
    'utilization_max',     # max of all 4 corners
    'utilization_spread',  # max - min across corners (balance)
    'ideal_ackermann_pct', # dynamic ideal Ackermann from tire-model inversion
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


def knobs_for_vehicle(veh: 'VehicleParams') -> list:
    """Topology-aware knob list (single-model): each axle exposes the spring
    elements its mechanism ACTUALLY has, instead of the generic corner-spring
    + ARB pair for every car.

      standard    -> corner spring rate + ARB rate (+ motion ratio)
      tbar        -> corner coil rate + T-bar roll rate (+ coil motion ratio)
      heave_tbar  -> corner coil rate + 3rd (heave) spring rate + T-bar roll
                     rate (+ coil motion ratio)
      decoupled   -> heave spring rate + roll spring rate (the two cross-car
                     coilovers; no corner spring, no bar, MR is geometric)
    """
    K = 175.127 * 10   # ≈10 lbf/in in N/m
    knobs = []
    for suffix, mode in (('front', getattr(veh, 'topology_mode_front', 'standard')),
                         ('rear',  getattr(veh, 'topology_mode_rear',  'standard'))):
        tag = suffix[0].upper()
        if mode == 'decoupled':
            knobs += [
                (f'decoupled_heave_rate_{suffix}_Npm', f'Heave spring rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'decoupled_roll_rate_{suffix}_Npm', f'Roll spring rate {tag}',
                 'lbf/in', K, 'parameter', 25),
            ]
        elif mode == 'heave_tbar':
            knobs += [
                (f'spring_rate_{suffix}_Npm', f'Corner coil rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'heave_3rd_rate_{suffix}_Npm', f'3rd (heave) spring rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'arb_rate_{suffix}_Npm', f'T-bar roll rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'motion_ratio_{suffix}', f'Coil motion ratio {tag}',
                 '', 0.02, 'kinematic', 0.05),
            ]
        else:   # standard (incl. plain T-bar ARB — its roll rate is arb_rate)
            knobs += [
                (f'spring_rate_{suffix}_Npm', f'Spring rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'arb_rate_{suffix}_Npm', f'ARB / T-bar rate {tag}',
                 'lbf/in', K, 'parameter', 25),
                (f'motion_ratio_{suffix}', f'Motion ratio {tag}',
                 '', 0.02, 'kinematic', 0.05),
            ]
    knobs += [
        ('cg_to_front_axle_m', 'Weight dist (CG fwd)', 'mm', 0.010, 'parameter', 25),
        ('front_brake_bias',   'Brake bias',           '%',  0.02,  'parameter', 5),
    ]
    return knobs


def _extract_outputs(result: SteadyStateResult,
                     tire=None, vehicle: VehicleParams = None,
                     turn_radius_m: float = None) -> dict:
    """Pull the tracked output metrics from a SteadyStateResult.

    Optional tire / vehicle / turn_radius_m enable ideal Ackermann
    computation.  Without them the metric defaults to NaN.
    """
    # LLTD = TOTAL lateral load-transfer distribution (elastic + geometric +
    # unsprung).  The car's balance responds to the TOTAL — an elastic-only
    # number misleads ARB tuning whenever roll-centre heights or unsprung
    # masses differ front/rear (industry-standard definition).
    tot_f = (result.elastic_lt_front_N + result.geometric_lt_front_N
             + result.unsprung_lt_front_N)
    tot_r = (result.elastic_lt_rear_N + result.geometric_lt_rear_N
             + result.unsprung_lt_rear_N)
    tot = tot_f + tot_r
    lltd = (tot_f / tot * 100) if tot > 0 else 50.0

    utils = [result.utilization.get(c, 0) for c in ('FL', 'FR', 'RL', 'RR')]

    ack = _ideal_ackermann_pct(result, tire, vehicle, turn_radius_m)

    return {
        'understeer_gradient_deg': result.understeer_gradient_deg,
        'roll_angle_deg': result.roll_angle_deg,
        'pitch_angle_deg': result.pitch_angle_deg,
        'lltd_pct': lltd,
        'utilization_max': max(utils) if utils else 0,
        'utilization_spread': (max(utils) - min(utils)) if utils else 0,
        'ideal_ackermann_pct': ack,
    }


def _ideal_ackermann_pct(result: SteadyStateResult,
                         tire=None, vehicle: VehicleParams = None,
                         turn_radius_m: float = None) -> float:
    """Compute the Ackermann % the tires WANT at a given operating point.

    The idea: at a specific turn radius and Fz distribution, each front
    tire needs a different slip angle to produce its share of lateral
    force.  The difference between those slip angles — on top of the
    geometric Ackermann split — defines what the tires "want".

    Returns
    -------
    ideal_ackermann_pct : float
        100 = pure Ackermann (inner steers more by exactly the geometric
        amount).  >100 = tires want MORE inner steer than geometry gives
        (common at high-g where the inner tire is unloaded).  <100 = the
        tires want less.  NaN if we can't compute (no tire model, etc.).

    Formula
    -------
        geo_inner = atan(L / (R − t/2))
        geo_outer = atan(L / (R + t/2))
        geo_diff  = geo_inner − geo_outer          (always positive)

        SA_inner = tire.slip_angle_for_Fy(Fy_inner, Fz_inner, camber_inner)
        SA_outer = tire.slip_angle_for_Fy(Fy_outer, Fz_outer, camber_outer)

        required_steer_diff = geo_diff + (SA_inner − SA_outer)
        ideal_ack%  = required_steer_diff / geo_diff × 100
    """
    if tire is None or vehicle is None or turn_radius_m is None:
        return float('nan')
    if not hasattr(tire, 'slip_angle_for_Fy'):
        return float('nan')
    if turn_radius_m < 1.0:
        return float('nan')

    L = vehicle.wheelbase_m
    t = vehicle.front_track_m
    R = turn_radius_m

    # Geometric steer angles for inner / outer front wheels
    geo_inner = np.arctan(L / (R - t / 2))
    geo_outer = np.arctan(L / (R + t / 2))
    geo_diff = geo_inner - geo_outer      # always > 0
    if geo_diff < 1e-9:
        return float('nan')

    # We assume a left turn → FL = outer, FR = inner.
    # (The math is symmetric; the sign convention doesn't change the
    # Ackermann % because we use magnitudes.)
    Fy = getattr(result, 'Fy', {})
    Fz = getattr(result, 'Fz', {})
    camber = getattr(result, 'camber', {})

    Fy_outer = abs(Fy.get('FL', 0))
    Fy_inner = abs(Fy.get('FR', 0))
    Fz_outer = max(Fz.get('FL', 1.0), 1.0)
    Fz_inner = max(Fz.get('FR', 1.0), 1.0)
    cam_outer = abs(camber.get('FL', 0))
    cam_inner = abs(camber.get('FR', 0))

    try:
        SA_outer = tire.slip_angle_for_Fy(Fy_outer, Fz_outer, cam_outer)
        SA_inner = tire.slip_angle_for_Fy(Fy_inner, Fz_inner, cam_inner)
    except Exception:
        return float('nan')

    # Convert slip angles to radians for consistent units with geo_diff
    SA_outer_rad = np.radians(SA_outer)
    SA_inner_rad = np.radians(SA_inner)

    # The tires require this steer-angle difference (inner − outer)
    required_diff = geo_diff + (SA_inner_rad - SA_outer_rad)

    return float(required_diff / geo_diff * 100)


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
                longitudinal_g: float = 0.0,
                turn_radius_m: float = None) -> dict:
        """
        Run sensitivity analysis averaged over a g range (±0.3g around
        the operating point, 5 samples).  Averaging eliminates artifacts
        from single-point noise in the tire model / kinematic solver.

        Parameters
        ----------
        turn_radius_m : float, optional
            Turn radius for dynamic ideal Ackermann computation.
            If None, ideal_ackermann_pct will be NaN in outputs.

        Returns dict with 'baseline' outputs and 'sensitivities' list.
        """
        # Sample at multiple g points around the operating point
        g_spread = 0.3
        n_samples = 5
        g_samples = np.linspace(
            max(0.1, lateral_g - g_spread),
            lateral_g + g_spread,
            n_samples)

        # Shared kwargs for _extract_outputs (enables ideal Ackermann)
        _eo_kw = dict(tire=self._tire, vehicle=self._base_veh,
                      turn_radius_m=turn_radius_m)

        # ── Baseline: average over samples ──────────────────────────
        base_solver = SteadyStateSolver(self._base_veh, self._solvers, self._tire)
        baselines = [_extract_outputs(base_solver.solve(g, longitudinal_g), **_eo_kw)
                     for g in g_samples]
        baseline = {k: float(np.nanmean([b[k] for b in baselines]))
                    for k in baselines[0]}
        # Also keep the center-point result for display
        base_result = base_solver.solve(lateral_g, longitudinal_g)
        baseline_center = _extract_outputs(base_result, **_eo_kw)
        # Use center point for display values, averaged for sensitivities
        baseline.update({f'_display_{k}': v for k, v in baseline_center.items()})

        sensitivities = []

        # ── VehicleParams knobs — TOPOLOGY-AWARE (single-model): each axle
        # perturbs the spring elements its mechanism actually has.
        for key, name, unit, delta, category, practical_step in knobs_for_vehicle(self._base_veh):
            base_val = getattr(self._base_veh, key)

            # Average perturbed outputs over the same g samples
            veh_up = self._perturb_veh(key, base_val + delta)
            solver_up = SteadyStateSolver(veh_up, self._solvers, self._tire)
            _eo_up = dict(tire=self._tire, vehicle=veh_up,
                          turn_radius_m=turn_radius_m)
            outs_up = [_extract_outputs(solver_up.solve(g, longitudinal_g), **_eo_up)
                       for g in g_samples]
            out_up = {k: float(np.nanmean([o[k] for o in outs_up]))
                      for k in outs_up[0]}

            veh_dn = self._perturb_veh(key, base_val - delta)
            solver_dn = SteadyStateSolver(veh_dn, self._solvers, self._tire)
            _eo_dn = dict(tire=self._tire, vehicle=veh_dn,
                          turn_radius_m=turn_radius_m)
            outs_dn = [_extract_outputs(solver_dn.solve(g, longitudinal_g), **_eo_dn)
                       for g in g_samples]
            out_dn = {k: float(np.nanmean([o[k] for o in outs_dn]))
                      for k in outs_dn[0]}

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
                val_up = out_up[metric]
                val_dn = out_dn[metric]
                if np.isnan(val_up) or np.isnan(val_dn):
                    effects[metric] = float('nan')
                    continue
                d_out = val_up - val_dn
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

        # Physical bounds: (key → (min_value, max_value)) in SI units
        _BOUNDS = {
            'spring_rate_front_Npm': (1750, 175000),    # 10–1000 lbf/in
            'spring_rate_rear_Npm':  (1750, 175000),
            'arb_rate_front_Npm':    (0, 87500),         # 0–500 lbf/in
            'arb_rate_rear_Npm':     (0, 87500),
            'motion_ratio_front':    (0.3, 2.0),
            'motion_ratio_rear':     (0.3, 2.0),
            'cg_to_front_axle_m':    (0.3, 2.5),
            'front_brake_bias':      (0.4, 0.85),
        }

        recommendations = []
        for s in analysis['sensitivities']:
            effect = s['effects'].get(target_metric, 0)
            if np.isnan(effect) or abs(effect) < 1e-6:
                continue

            change_needed = target_delta / effect  # how much to change this knob

            # Clamp to physical bounds
            key = s['key']
            if key in _BOUNDS:
                lo, hi = _BOUNDS[key]
                new_clamped = max(lo, min(hi, s['current_value'] + change_needed))
                change_needed = new_clamped - s['current_value']
                if abs(change_needed) < 1e-9:
                    continue  # already at bound, skip

            # Compute side effects on other metrics
            side_effects = {}
            for metric in SENSITIVITY_OUTPUTS:
                if metric == target_metric:
                    continue
                other_effect = s['effects'].get(metric, 0)
                if np.isnan(other_effect):
                    continue
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
            h_arm = v.sprung_cg_height_m - h_roll
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

            # TOTAL LLTD (matches _extract_outputs' definition): elastic +
            # geometric + unsprung.  The RC knob acts mostly through the
            # GEOMETRIC share — an elastic-only ratio would show the WRONG
            # SIGN for this sensitivity (raising RC lowers elastic, raises
            # geometric more).
            us_f = (v.unsprung_mass_front_kg * ay
                    * v.unsprung_cg_height_m / v.front_track_m)
            us_r = (v.unsprung_mass_rear_kg * ay
                    * v.unsprung_cg_height_m / v.rear_track_m)
            tot_f = el_f + geo_f + us_f
            tot_r = el_r + geo_r + us_r
            tot = tot_f + tot_r
            lltd = (tot_f / tot * 100) if tot > 0 else 50

            out = {
                'roll_angle_deg': np.degrees(roll_rad),
                'pitch_angle_deg': baseline['pitch_angle_deg'],  # RC doesn't affect pitch
                'lltd_pct': lltd,
                'understeer_gradient_deg': baseline['understeer_gradient_deg'],  # approx
                'utilization_max': baseline['utilization_max'],
                'utilization_spread': baseline['utilization_spread'],
                'ideal_ackermann_pct': baseline.get('ideal_ackermann_pct', float('nan')),  # RC doesn't directly change Ackermann
            }

            if sign == +1:
                out_up = out
            else:
                out_dn = out

        delta_display = rc_delta_m * 1000  # mm
        for metric in SENSITIVITY_OUTPUTS:
            val_up = out_up[metric]
            val_dn = out_dn[metric]
            if np.isnan(val_up) or np.isnan(val_dn):
                effects[metric] = float('nan')
                continue
            d_out = val_up - val_dn
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


# ─────────────────────────────────────────────────────────────────────────────
#  Aero downforce solver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AeroResult:
    """Per-corner additional Fz required to hit target utilization."""
    lateral_g: float
    longitudinal_g: float = 0.0
    target_util: float = 0.0

    # Per-corner required additional Fz (N) — remaining deficit after aero
    downforce: dict = field(default_factory=dict)
    # Per-corner utilization after adding Fz
    utilization_aero: dict = field(default_factory=dict)
    # Corners that hit the cap
    capped: list = field(default_factory=list)

    # Axle-level: max(left, right) per axle — remaining deficit
    front_axle_need_N: float = 0.0
    rear_axle_need_N: float = 0.0
    # Total remaining deficit
    total_downforce_N: float = 0.0
    # Rear bias (% of total on rear axle)
    rear_aero_bias_pct: float = 50.0


class AeroDownforceSolver:
    """Per-corner Fz deficit solver with axle-level summary."""

    def __init__(self, steady_solver: SteadyStateSolver):
        self._ss = steady_solver

    def solve(self, lateral_g: float,
              longitudinal_g: float = 0.0,
              target_util: float = 0.80,
              max_total_downforce_N: float = 12000.0,
              max_iter: int = 60) -> AeroResult:
        """Downforce required per AXLE to bring the axle-aggregate utilization
        down to ``target_util`` at the given g-state.

        REWRITTEN 2026-07-12 after the '6 kN cap gang' finding.  The old
        implementation had three defects that produced a 6000 N plateau:
        (1) it targeted PER-CORNER utilization, so the unloaded inner tire
        (below the tire-data floor = interpolation-artifact zone) demanded
        infinite help and slammed each corner into its 3000 N cap;
        (2) it used RAW belt mu with no belt->track scale, disagreeing with
        the steady-state solver's own grip budget inside one tool;
        (3) it sized downforce per individual corner — a wing cannot push on
        one wheel; aero loads an axle symmetrically.
        Now: per-axle bisection on symmetric axle downforce, judged by the
        canonical SteadyStateSolver.axle_utilization (mu-scaled), with the
        steady state RE-SOLVED at each step so load transfer and camber react
        to the added load.
        """
        ss = self._ss
        if ss._tire is None:
            raise ValueError('Tire model required for aero solver')

        result = AeroResult(lateral_g=lateral_g,
                            longitudinal_g=longitudinal_g,
                            target_util=target_util)
        D_axle_cap = float(max(max_total_downforce_N, 100.0)) / 2.0
        pair = {'F': ('FL', 'FR'), 'R': ('RL', 'RR')}

        def aero_fz(dF, dR):
            return {'FL': dF / 2, 'FR': dF / 2, 'RL': dR / 2, 'RR': dR / 2}

        base = ss.solve(lateral_g, longitudinal_g)
        u0 = ss.axle_utilization(base)
        need = {'F': 0.0, 'R': 0.0}
        for ax in ('F', 'R'):
            if u0[ax] <= target_util + 1e-9:
                continue
            # feasibility at the cap first
            def _u_at(d):
                r = ss.solve(lateral_g, longitudinal_g,
                             aero_Fz=aero_fz(d if ax == 'F' else need['F'],
                                             d if ax == 'R' else need['R']))
                return ss.axle_utilization(r)[ax]
            if _u_at(D_axle_cap) > target_util:
                need[ax] = D_axle_cap
                result.capped.extend(pair[ax])
                continue
            lo, hi = 0.0, D_axle_cap
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if _u_at(mid) <= target_util:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 20.0:      # N — aero packages aren't built to 0.1 N
                    break
            need[ax] = hi

        # final combined state for reporting (per-corner utils are DISPLAY
        # values; the limit criterion is the axle aggregate)
        final = ss.solve(lateral_g, longitudinal_g,
                         aero_Fz=aero_fz(need['F'], need['R']))
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            result.downforce[lbl] = need[lbl[0]] / 2.0
            result.utilization_aero[lbl] = float(final.utilization.get(lbl, 0.0))

        result.front_axle_need_N = need['F']
        result.rear_axle_need_N = need['R']
        result.total_downforce_N = need['F'] + need['R']
        result.rear_aero_bias_pct = (
            need['R'] / result.total_downforce_N * 100.0) if result.total_downforce_N > 0 else 50.0
        return result

    def sweep(self, g_range: np.ndarray,
              longitudinal_g: float = 0.0,
              target_util: float = 0.80,
              max_total_downforce_N: float = 12000.0) -> dict:
        """Sweep lateral g — deficit vs g."""
        gs = np.asarray(g_range, float)
        out = {
            'lateral_g': gs,
            'front_need': np.zeros(len(gs)),
            'rear_need': np.zeros(len(gs)),
            'total': np.zeros(len(gs)),
            'rear_bias_pct': np.full(len(gs), 50.0),
        }
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            out[f'dF_{lbl}'] = np.zeros(len(gs))

        for i, g in enumerate(gs):
            try:
                r = self.solve(g, longitudinal_g, target_util,
                               max_total_downforce_N=max_total_downforce_N)
                out['front_need'][i] = r.front_axle_need_N
                out['rear_need'][i]  = r.rear_axle_need_N
                out['total'][i]      = r.total_downforce_N
                out['rear_bias_pct'][i] = r.rear_aero_bias_pct
                for lbl in ('FL', 'FR', 'RL', 'RR'):
                    out[f'dF_{lbl}'][i] = r.downforce.get(lbl, 0.0)
            except Exception:
                pass
        return out
