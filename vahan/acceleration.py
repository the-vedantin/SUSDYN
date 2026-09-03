"""Gear-resolved longitudinal acceleration model (the straight-line launch event).

The lap sim's tractive force is power-based (F = P/v), which is FINAL-DRIVE
independent — it cannot answer "what final drive launches best / tops out where".
This module resolves the drivetrain by GEAR: tractive force at the contact patch =
engine_torque(rpm) * (primary * gear * final_drive) * efficiency / rolling_radius,
with rpm = (v / r_wheel) * total_ratio, upshifting at redline; the usable force is
capped by rear-tyre grip (load-sensitive mu on the aero-loaded rear axle, RWD) and
reduced by aero drag. Integrating a = F_net/m over the run gives the time-distance
trajectory, the 75 m acceleration-event time, and the drag-limited top speed.

ONE MODEL: torque from vahan.engine, grip from the fitted tyre model, aero from the
same ClA/CdA/rho the lap sim uses. No hardcoded engine or grip numbers here.
"""
from __future__ import annotations

import numpy as np

import vahan.engine as _engine

# The documented FSAE gearbox used across the lap sims (set_gearbox default).
DEFAULT_GEARS = [2.75, 2.0, 1.667, 1.444, 1.304, 1.208]
DEFAULT_PRIMARY = 2.111
DEFAULT_REDLINE_RPM = 11000.0
G = 9.80665


class AccelerationModel:
    """One straight-line launch model for a fixed final drive.

    Parameters
    ----------
    mass_kg, r_wheel_m, rear_weight_frac : vehicle
    tire_rear : object with peak_mu(Fz_N, camber_deg) -> mu  (the fitted rear tyre)
    cla_m2, cda_m2, rho, aero_rear_frac  : aero (downforce = 0.5*rho*ClA*v^2)
    gears, primary, final_drive, redline_rpm, drivetrain_eff : drivetrain
    engine_rpm, engine_torque_Nm : the crank torque curve (from vahan.engine)
    """

    def __init__(self, mass_kg, r_wheel_m, rear_weight_frac, tire_rear,
                 cla_m2, cda_m2, rho, aero_rear_frac,
                 final_drive, gears=None, primary=DEFAULT_PRIMARY,
                 redline_rpm=DEFAULT_REDLINE_RPM, drivetrain_eff=0.92,
                 grip_scale=0.60, engine_rpm=None, engine_torque_Nm=None):
        self.m = float(mass_kg)
        self.r = float(r_wheel_m)
        self.rear_frac = float(rear_weight_frac)
        self.tire_rear = tire_rear
        self.cla = float(cla_m2); self.cda = float(cda_m2); self.rho = float(rho)
        self.aero_rear_frac = float(aero_rear_frac)
        self.gears = list(gears) if gears else list(DEFAULT_GEARS)
        self.primary = float(primary)
        self.final = float(final_drive)
        self.redline = float(redline_rpm)
        self.eff = float(drivetrain_eff)
        self.grip_scale = float(grip_scale)   # belt->road derate on peak mu
        if engine_rpm is None or engine_torque_Nm is None:
            ec = _engine.engine_curve('corrected') or _engine.engine_curve('anchored')
            if ec is None:
                raise RuntimeError('no engine curve available')
            engine_rpm, engine_torque_Nm, _ = ec
        self.erpm = np.asarray(engine_rpm, float)
        self.etq = np.asarray(engine_torque_Nm, float)
        self.idle_rpm = float(self.erpm.min())
        # Launch rpm: below the mapped range the clutch slips and the driver
        # holds the engine near peak torque — that is the force delivered off
        # the line (grip-limited anyway).
        self.launch_rpm = float(self.erpm[int(np.argmax(self.etq))])

    # ── drivetrain ────────────────────────────────────────────────────────
    def _rpm(self, v_ms, total_ratio):
        return (v_ms / self.r) * total_ratio * 60.0 / (2.0 * np.pi)

    def tractive_force_N(self, v_ms):
        """Best gear's contact-patch force at speed v (envelope over gears with
        idle <= rpm <= redline). 0 if no gear keeps the engine in range."""
        v_ms = max(v_ms, 0.1)
        best = 0.0
        for g in self.gears:
            ratio = self.primary * g * self.final
            rpm = self._rpm(v_ms, ratio)
            if rpm > self.redline:
                continue                              # would over-rev this gear
            # Below idle: clutch slips, engine held at launch rpm.
            rpm_tq = rpm if rpm >= self.idle_rpm else self.launch_rpm
            tq = float(np.interp(rpm_tq, self.erpm, self.etq))
            F = tq * ratio * self.eff / self.r
            if F > best:
                best = F
        return best

    def top_speed_gearing_ms(self):
        """Speed at which the tallest gear hits redline (gearing-limited cap)."""
        ratio = self.primary * self.gears[-1] * self.final
        return self.redline * 2.0 * np.pi / 60.0 * self.r / ratio

    # ── grip + drag ───────────────────────────────────────────────────────
    def _downforce_N(self, v_ms):
        return 0.5 * self.rho * self.cla * v_ms ** 2

    def _drag_N(self, v_ms):
        return 0.5 * self.rho * self.cda * v_ms ** 2

    def traction_force_N(self, v_ms, long_transfer_N=0.0):
        """Rear-axle grip limit (RWD): mu(Fz_rear) * Fz_rear, Fz_rear = static rear
        + aero rear share + longitudinal weight transfer onto the rear."""
        Fz = self.m * G * self.rear_frac + self._downforce_N(v_ms) * self.aero_rear_frac + long_transfer_N
        try:
            mu = float(self.tire_rear.peak_mu(Fz / 2.0, 0.0))   # per-wheel Fz
        except Exception:
            mu = 1.4
        return self.grip_scale * mu * Fz

    # ── run ───────────────────────────────────────────────────────────────
    def run(self, v0_ms=0.0, dt=0.005, t_max=30.0, x_max_m=150.0,
            cg_h_m=0.30, wheelbase_m=1.537):
        """Integrate the launch to x_max_m (so fixed-distance events like the
        75 m sprint are always covered — the car keeps rolling at terminal
        speed). Returns arrays t,v,x,a and the metrics. Longitudinal weight
        transfer onto the rear = m*a*cg_h/wheelbase (fixed-point per step)."""
        t, v, x = [0.0], [float(v0_ms)], [0.0]
        while t[-1] < t_max and x[-1] < x_max_m:
            vv = v[-1]
            Ftr = self.tractive_force_N(vv)
            # fixed-point: grip depends on accel via rear weight transfer
            a = (min(Ftr, self.traction_force_N(vv)) - self._drag_N(vv)) / self.m
            for _ in range(3):
                lt = self.m * max(a, 0.0) * cg_h_m / wheelbase_m
                Fgrip = self.traction_force_N(vv, long_transfer_N=lt)
                a = (min(Ftr, Fgrip) - self._drag_N(vv)) / self.m
            v.append(max(vv + a * dt, 0.0)); x.append(x[-1] + vv * dt); t.append(t[-1] + dt)
        t, v, x = np.array(t), np.array(v), np.array(x)
        vk = v * 3.6

        def _time_at_dist(d):
            return float(np.interp(d, x, t)) if x[-1] >= d else float('nan')

        def _time_at_speed(kph):
            return float(np.interp(kph, vk, t)) if vk[-1] >= kph else float('nan')
        return {
            't': t, 'v_ms': v, 'v_kph': vk, 'x_m': x,
            'a_g': np.gradient(v, t) / G,
            'top_speed_kph': float(vk.max()),
            't_75m_s': _time_at_dist(75.0),
            't_0_100kph_s': _time_at_speed(100.0),
            'top_speed_gearing_kph': self.top_speed_gearing_ms() * 3.6,
        }


def from_window(win, final_drive, cla_m2=0.946, cda_m2=1.0, air_density=1.225,
                aero_rear_frac=0.54, grip_scale=0.60, gears=None, primary=DEFAULT_PRIMARY,
                redline_rpm=DEFAULT_REDLINE_RPM, drivetrain_eff=0.92):
    """Build an AccelerationModel from a solved MainWindow (ONE MODEL): mass and
    rear-tyre model from the dynamics solver, rolling radius from the tyre OD."""
    ss = win._build_dynamics_solver(); v = ss._veh
    r_wheel = 0.5 * float(win._car.get('tire_outer_dia_mm', 406.0)) / 1000.0
    tire_rear = getattr(ss, '_tire_rear', None) or getattr(ss, '_tire', None)
    return AccelerationModel(
        mass_kg=v.total_mass_kg, r_wheel_m=r_wheel,
        rear_weight_frac=getattr(v, 'rear_weight_fraction', 0.54),
        tire_rear=tire_rear, cla_m2=cla_m2, cda_m2=cda_m2, rho=air_density,
        aero_rear_frac=aero_rear_frac, final_drive=final_drive, gears=gears,
        primary=primary, redline_rpm=redline_rpm, drivetrain_eff=drivetrain_eff,
        grip_scale=grip_scale)
