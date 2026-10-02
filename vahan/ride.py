"""Linear vertical road response of the current solved vehicle.

Seven coordinates: body heave, nose-up pitch, left-up roll, FL/FR/RL/RR
wheel vertical displacement. SI units throughout. Inertias and linearized
wheel-coordinate damping are required inputs, never inferred hardware facts.
Road spectra are one-sided height cross spectra per Hz, not spatial PSDs.
Spectral/FFT response is bilateral. Time integration optionally uses unilateral
tyre contact; suspension remains linearized and hardware stops are not modeled.
"""
from dataclasses import dataclass
import numpy as np

CORNERS = ('FL', 'FR', 'RL', 'RR')


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


@dataclass
class RideModel:
    mass: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    road_force: np.ndarray
    suspension_map: np.ndarray
    body_corner_map: np.ndarray
    tire_rates: np.ndarray
    baseline_loads: np.ndarray

    @classmethod
    def from_vehicle(cls, vehicle, *, pitch_inertia_kgm2,
                     roll_inertia_kgm2, corner_damping_Nspm,
                     baseline_loads_N=None):
        """Build from VehicleParams (standard corner-spring topology only).

        Damping is FOUR coefficients at the WHEEL, FL/FR/RL/RR, >=0.
        Convert a shock coefficient using the solved instantaneous MR squared
        outside this function. Asymmetric bump/rebound needs an explicitly
        chosen equivalent linear coefficient; frequency response is linear.
        Vehicle wheel rates include its configured static geometric term.
        Optional baseline tire loads permit a stated loaded operating point;
        callers must also provide rates linearized at that same point.
        """
        if any(getattr(vehicle, f'topology_mode_{a}', 'standard') != 'standard'
               for a in ('front', 'rear')):
            raise ValueError('Road model currently supports standard topology only')
        ms = _positive(vehicle.sprung_mass_kg, 'sprung mass')
        muf = _positive(vehicle.unsprung_mass_front_kg, 'front unsprung mass')
        mur = _positive(vehicle.unsprung_mass_rear_kg, 'rear unsprung mass')
        length = _positive(vehicle.wheelbase_m, 'wheelbase')
        tf = _positive(vehicle.front_track_m, 'front track')
        tr = _positive(vehicle.rear_track_m, 'rear track')
        a = float(vehicle.cg_to_front_axle_m)
        if not np.isfinite(a) or not 0 < a < length:
            raise ValueError('whole-car CG must lie between the axles')
        mt = ms + muf + mur
        front_total = mt * (length-a) / length
        rear_total = mt * a / length
        front_sprung, rear_sprung = front_total-muf, rear_total-mur
        if min(front_sprung, rear_sprung) <= 0:
            raise ValueError('sprung axle support masses must be positive')
        # Moment balance locates sprung CG independently of whole-car CG.
        sprung_a = length * rear_sprung / ms
        x = np.array([sprung_a, sprung_a, sprung_a-length, sprung_a-length])
        y = np.array([tf/2, -tf/2, tr/2, -tr/2])
        body = np.column_stack([np.ones(4), x, y])
        incidence = np.column_stack([-body, np.eye(4)])  # wheel minus body = bump
        kw = np.array([vehicle.wheel_rate_front_Npm]*2
                      + [vehicle.wheel_rate_rear_Npm]*2, dtype=float)
        if not np.all(np.isfinite(kw)) or np.any(kw <= 0):
            raise ValueError('wheel stiffness must be finite and positive')
        ct = np.asarray(corner_damping_Nspm, dtype=float)
        if ct.shape != (4,) or not np.all(np.isfinite(ct)) or np.any(ct < 0):
            raise ValueError('four finite nonnegative wheel damping coefficients required')
        kt = np.full(4, _positive(vehicle.tire_rate_Npm, 'tire rate'))
        spring_matrix = np.diag(kw)
        for axle, ids in [('front', [0, 1]), ('rear', [2, 3])]:
            arb = float(getattr(vehicle, f'arb_rate_{axle}_Npm'))
            if not np.isfinite(arb) or arb < 0:
                raise ValueError('ARB rates must be finite and nonnegative')
            # Gives Kphi = Karb*track^2/2, same as VehicleParams.
            spring_matrix[np.ix_(ids, ids)] += arb/2 * np.array([[1., -1.], [-1., 1.]])
        mass = np.diag([ms, _positive(pitch_inertia_kgm2, 'pitch inertia'),
                        _positive(roll_inertia_kgm2, 'roll inertia'),
                        muf/2, muf/2, mur/2, mur/2])
        stiffness = incidence.T @ spring_matrix @ incidence
        stiffness[3:, 3:] += np.diag(kt)
        damping = incidence.T @ np.diag(ct) @ incidence
        force = np.zeros((7, 4)); force[3:] = np.diag(kt)
        loads = (np.array([front_total, front_total, rear_total, rear_total])*9.81/2
                 if baseline_loads_N is None else np.asarray(baseline_loads_N, dtype=float))
        if loads.shape != (4,) or not np.all(np.isfinite(loads)) or np.any(loads <= 0):
            raise ValueError('four positive finite baseline tire loads required')
        return cls(mass, stiffness, damping, force, incidence, body, kt, loads)

    def transfer(self, frequencies_Hz):
        """Complex output transfer matrices, axes [frequency, output, road].

        Normal load is the perturbation about baseline, positive compression.
        Body acceleration contains heave (m/s²), pitch and roll (rad/s²).
        """
        freq = np.asarray(frequencies_Hz, dtype=float)
        if freq.ndim != 1 or not len(freq) or not np.all(np.isfinite(freq)) or np.any(freq < 0):
            raise ValueError('frequency must be a finite nonnegative vector')
        omega = 2*np.pi*freq
        dynamic = (self.stiffness[None] - omega[:, None, None]**2*self.mass
                   + 1j*omega[:, None, None]*self.damping)
        response = np.linalg.solve(dynamic, np.broadcast_to(self.road_force, (len(freq), 7, 4)))
        suspension = self.suspension_map[None] @ response
        return {
            'coordinates': response,
            'dynamic_tire_load_N': self.tire_rates[None, :, None]*(np.eye(4)[None]-response[:, 3:]),
            'suspension_displacement_m': suspension,
            # Wheel-frame bump velocity (damper velocity = this x instantaneous MR).
            'suspension_velocity_mps': 1j*omega[:, None, None]*suspension,
            'body_acceleration': -omega[:, None, None]**2*response[:, :3],
            'body_corner_acceleration_mps2': -omega[:, None, None]**2*(self.body_corner_map[None] @ response[:, :3]),
        }

    def spectral_response(self, frequencies_Hz, road_cross_psd_m2_per_Hz):
        """Output PSDs/RMS from a validated Hermitian PSD road matrix.

        Stochastic PSDs alone do not determine minimum tire load/contact-loss
        events. RMS/baseline is reported; no Gaussian tail or zero-loss claim
        is invented. For deterministic response use contact_validity below.
        """
        freq = np.asarray(frequencies_Hz, dtype=float)
        if freq.ndim != 1 or len(freq) < 2 or np.any(np.diff(freq) <= 0):
            raise ValueError('at least two strictly increasing frequencies required')
        road = np.asarray(road_cross_psd_m2_per_Hz, dtype=complex)
        if road.shape != (len(freq), 4, 4) or not np.all(np.isfinite(road)):
            raise ValueError('road cross PSD must have shape (n_frequencies,4,4) and finite entries')
        scale = max(float(np.max(np.abs(road))), np.finfo(float).tiny)
        if np.max(np.abs(road-road.conj().transpose(0, 2, 1))) > 1e-10*scale:
            raise ValueError('road cross PSD must be Hermitian')
        eigenvalues, eigenvectors = np.linalg.eigh(road)
        if np.min(eigenvalues) < -1e-10*scale:
            raise ValueError('road cross PSD must be positive semidefinite')
        # Factor the PSD to avoid subtractive cancellation for coherent roads.
        # Reject substantive negative eigenvalues above; remove only roundoff.
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
        outputs = {}
        for name, h in self.transfer(freq).items():
            amplitudes = h @ factor
            psd = np.sum(np.abs(amplitudes)**2, axis=2)
            outputs[name] = {'psd': psd, 'rms': np.sqrt(np.trapezoid(psd, freq, axis=0))}
        outputs['dynamic_load_coefficient'] = outputs['dynamic_tire_load_N']['rms']/self.baseline_loads
        outputs['contact_loss_status'] = 'not determined from second-order spectra'
        return outputs

    def periodic_response(self, dt_s, road_heights_m):
        """Steady periodic response by FFT; no startup or transient settling.

        Samples span [0, n*dt), excluding the repeated endpoint. The implicit
        period joins last sample to first; callers must supply a periodic road.
        An even-length road must have negligible Nyquist content because a
        complex mechanical response at the self-conjugate Nyquist bin cannot
        be represented uniquely as a real sampled waveform. Prefer odd n or
        band-limit the road below Nyquist. DC and Nyquist output bins are real.
        """
        dt = _positive(dt_s, 'sample interval')
        road = np.asarray(road_heights_m)
        if np.iscomplexobj(road):
            raise ValueError('road heights must be real')
        road = np.asarray(road, dtype=float)
        if road.ndim != 2 or road.shape[1] != 4 or len(road) < 3 or not np.all(np.isfinite(road)):
            raise ValueError('finite real (n>=3,4) road-height array required')
        count = len(road)
        spectrum = np.fft.rfft(road, axis=0)
        if count % 2 == 0:
            scale = max(float(np.max(np.abs(spectrum))), np.finfo(float).tiny)
            if np.max(np.abs(spectrum[-1])) > 1e-10*scale:
                raise ValueError('even-length road has Nyquist energy; band-limit or use odd sample count')
            spectrum[-1] = 0.
        result = {'time_s': np.arange(count)*dt,
                  'response_kind': 'steady periodic; no startup transient'}
        # Bins with no road input produce exactly zero output, so the transfer
        # is solved only where the road spectrum is nonzero (band-limited
        # synthetic roads excite a few thousand of the bins). Same numbers.
        active = np.any(spectrum != 0, axis=1)
        active[0] = True
        h_active = self.transfer(np.fft.rfftfreq(count, dt)[active])
        for name, h in h_active.items():
            transformed = np.zeros((len(spectrum), h.shape[1]), dtype=complex)
            transformed[active] = np.einsum('noi,ni->no', h, spectrum[active])
            transformed[0] = transformed[0].real
            if count % 2 == 0:
                transformed[-1] = transformed[-1].real
            result[name] = np.fft.irfft(transformed, n=count, axis=0)
        result['contact'] = self.contact_validity(result['dynamic_tire_load_N'])
        return result

    def forced_response(self, time_s, generalized_force_N):
        """Linear time response to forces applied INSIDE the car (flat road).

        generalized_force_N is (n, 7) in this model's coordinates: heave force
        (N, up), nose-up pitch moment (N.m), left-up roll moment (N.m) and four
        vertical wheel forces (N, up). Starts at rest at static equilibrium;
        force is piecewise linear between samples. Bilateral tyre contact.
        """
        from scipy.signal import lsim
        t = np.asarray(time_s, dtype=float)
        force = np.asarray(generalized_force_N, dtype=float)
        if t.ndim != 1 or len(t) < 3 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)):
            raise ValueError('time must be a finite increasing vector (n>=3)')
        if force.shape != (len(t), 7) or not np.all(np.isfinite(force)):
            raise ValueError('finite (n,7) generalized force required')
        inv_mass = np.linalg.inv(self.mass)
        a = np.block([[np.zeros((7, 7)), np.eye(7)],
                      [-inv_mass @ self.stiffness, -inv_mass @ self.damping]])
        b = np.vstack([np.zeros((7, 7)), inv_mass])
        _, out, _ = lsim((a, b, np.eye(14), np.zeros((14, 7))), force, t)
        q, velocity = out[:, :7], out[:, 7:]
        return {'time_s': t, 'coordinates': q, 'velocity': velocity,
                'dynamic_tire_load_N': -self.tire_rates * q[:, 3:],
                'suspension_displacement_m': q @ self.suspension_map.T,
                'response_kind': 'transient from rest; linear; bilateral tyre; flat road'}

    def contact_validity(self, dynamic_tire_load_N):
        """Flag invalid bilateral-contact predictions; never clip forces."""
        delta = np.asarray(dynamic_tire_load_N, dtype=float)
        if delta.ndim != 2 or delta.shape[1] != 4 or len(delta) == 0 or not np.all(np.isfinite(delta)):
            raise ValueError('finite (n_samples,4) dynamic tire-load array required')
        minimum = np.min(delta+self.baseline_loads, axis=0)
        return {'minimum_load_N': minimum, 'linear_contact_valid': bool(np.all(minimum > 0)),
                'invalid_corners': [c for c, f in zip(CORNERS, minimum) if f <= 0]}

    def contact_response(self, time_s, road_heights_m, *, initial_state=None,
                         travel_limits_m=None, max_step_s=None,
                         rtol=1e-7, atol=1e-9):
        """Time integration with compression-only elastic tyre contact.

        Uses this model's same mass, suspension stiffness and damping. Only
        tyre contact is nonlinear: Fz=max(0,Fz0+Kt*(road-wheel)). Removing
        contact changes the equations of motion, not just the reported force.
        Coordinates are perturbations about loaded static equilibrium; zero
        initial state is at rest on zero-height road. Road is piecewise linear.

        Suspension remains linearized (no changing MR, spring unseating,
        asymmetric damping or aero). Optional four-corner travel bounds flag
        invalid results; they do NOT invent a bump-stop force law. No final
        hardware selection may rely on a trace that violates those bounds.
        Contact fractions are sample statistics, not exact event durations.
        """
        from scipy.integrate import solve_ivp

        def real_array(value, name):
            if np.iscomplexobj(value):
                raise ValueError(f'{name} must be real')
            return np.asarray(value, dtype=float)

        t = real_array(time_s, 'time')
        road = real_array(road_heights_m, 'road heights')
        if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t))
                or np.any(np.diff(t) <= 0)):
            raise ValueError('strictly increasing finite time vector required')
        if road.shape != (len(t), 4) or not np.all(np.isfinite(road)):
            raise ValueError('finite road heights with shape (n_times,4) required')
        # np.interp otherwise copies each strided column on every RHS call.
        tracks = np.ascontiguousarray(road.T)
        y0 = np.zeros(14) if initial_state is None else real_array(initial_state, 'initial state')
        if y0.shape != (14,) or not np.all(np.isfinite(y0)):
            raise ValueError('initial state must contain seven positions and seven velocities')
        bounds = None
        if travel_limits_m is not None:
            bounds = real_array(travel_limits_m, 'travel limits')
            if (bounds.shape != (2, 4) or not np.all(np.isfinite(bounds))
                    or np.any(bounds[0] >= bounds[1])):
                raise ValueError('travel limits must be finite (2,4), lower then upper')
        step = min(np.diff(t)) if max_step_s is None else _positive(max_step_s, 'max step')
        _positive(rtol, 'relative tolerance'); _positive(atol, 'absolute tolerance')
        suspension_k = self.stiffness.copy()
        suspension_k[3:, 3:] -= np.diag(self.tire_rates)
        inv_mass = np.linalg.inv(self.mass)

        def rhs(now, state):
            q, velocity = state[:7], state[7:]
            height = np.array([np.interp(now, t, track) for track in tracks])
            load = np.maximum(0., self.baseline_loads + self.tire_rates*(height-q[3:]))
            force = -suspension_k @ q - self.damping @ velocity
            force[3:] += load-self.baseline_loads
            return np.concatenate((velocity, inv_mass @ force))

        solution = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t,
                             max_step=step, rtol=rtol, atol=atol, method='DOP853')
        if not solution.success or solution.y.shape[1] != len(t):
            raise RuntimeError(f'contact integration failed: {solution.message}')
        q, velocity = solution.y[:7].T, solution.y[7:].T
        load = np.maximum(0., self.baseline_loads + self.tire_rates*(road-q[:, 3:]))
        travel = q @ self.suspension_map.T
        acceleration = np.array([rhs(now, state)[7:] for now, state in zip(t, solution.y.T)])
        result = dict(time_s=t, coordinates=q, velocity=velocity,
                      normal_load_N=load, dynamic_tire_load_N=load-self.baseline_loads,
                      suspension_displacement_m=travel,
                      body_acceleration=acceleration[:, :3],
                      body_corner_acceleration_mps2=acceleration[:, :3] @ self.body_corner_map.T,
                      contact_loss_sample_fraction=np.mean(load == 0., axis=0),
                      minimum_load_N=load.min(axis=0),
                      response_kind='transient unilateral tyre; linearized suspension; no stops',
                      travel_bounds_checked=bounds is not None)
        if bounds is not None:
            invalid = (travel < bounds[0]) | (travel > bounds[1])
            result['within_travel_bounds'] = not bool(np.any(invalid))
            result['invalid_travel_corners'] = [c for c, bad in zip(CORNERS, invalid.any(axis=0)) if bad]
        return result
