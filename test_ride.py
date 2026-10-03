"""Analytic checks for the vertical road model; python -m unittest test_ride."""
import unittest
import numpy as np
from vahan.dynamics import VehicleParams
from vahan.ride import RideModel


class RideChecks(unittest.TestCase):
    def setUp(self):
        self.v = VehicleParams(sprung_mass_kg=200, unsprung_mass_front_kg=20,
                               unsprung_mass_rear_kg=20, wheelbase_m=2,
                               cg_to_front_axle_m=1, front_track_m=1.2,
                               rear_track_m=1.2, spring_rate_front_Npm=30000,
                               spring_rate_rear_Npm=30000, motion_ratio_front=1,
                               motion_ratio_rear=1, tire_rate_Npm=150000,
                               arb_rate_front_Npm=5000, arb_rate_rear_Npm=5000)
        self.m = RideModel.from_vehicle(self.v, pitch_inertia_kgm2=60,
                                        roll_inertia_kgm2=30,
                                        corner_damping_Nspm=[1000]*4)

    def test_common_static_road_is_rigid_translation(self):
        h = self.m.transfer([0])
        expected = np.array([1, 0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(h['coordinates'][0] @ np.ones(4), expected, atol=1e-12)
        np.testing.assert_allclose(h['dynamic_tire_load_N'][0] @ np.ones(4), 0, atol=1e-9)

    def test_roll_stiffness_matches_vehicle_and_arb_has_no_heave(self):
        expected = self.v.roll_stiffness_front_Npm_rad+self.v.roll_stiffness_rear_Npm_rad
        self.assertAlmostEqual(self.m.stiffness[2, 2], expected)
        self.assertAlmostEqual(self.m.stiffness[0, 0], 4*30000)

    def test_symmetric_heave_equals_independent_quarter_car(self):
        f = 3.; w = 2*np.pi*f
        # Independent 2DOF quarter-car analytic matrix, body then wheel.
        k, c, kt, ms, mu = 30000, 1000, 150000, 50, 10
        a = np.array([[k+1j*w*c-w*w*ms, -k-1j*w*c],
                      [-k-1j*w*c, k+kt+1j*w*c-w*w*mu]])
        q = np.linalg.solve(a, [0, kt])
        actual = self.m.transfer([f])['coordinates'][0] @ np.ones(4)
        np.testing.assert_allclose(actual[[0, 3]], q, rtol=1e-12)
        np.testing.assert_allclose(actual[1:3], 0, atol=1e-12)

    def test_cross_spectrum_and_contact_are_not_silently_repaired(self):
        freq = np.array([1., 2.])
        bad = np.broadcast_to(np.diag([1, 1, 1, -1]), (2, 4, 4))
        with self.assertRaises(ValueError): self.m.spectral_response(freq, bad)
        result = self.m.contact_validity(np.array([[-1000, 0, 0, 0]]))
        self.assertFalse(result['linear_contact_valid'])
        self.assertEqual(result['invalid_corners'], ['FL'])

    def test_psd_response_preserves_correlated_road(self):
        freq = np.linspace(.1, 10, 200)
        road = np.broadcast_to(np.ones((4, 4))*1e-6, (len(freq), 4, 4))
        result = self.m.spectral_response(freq, road)
        # Rank-one input PSD eigenvalues carry roundoff after factorization.
        self.assertLess(result['body_acceleration']['rms'][1], 1e-6)
        self.assertGreater(result['dynamic_load_coefficient'][0], 0)
        self.assertEqual(result['contact_loss_status'], 'not determined from second-order spectra')

    def test_periodic_response_against_quarter_car_harmonic(self):
        n, dt, frequency = 1001, .001, 10/(1001*.001)
        time = np.arange(n)*dt
        road = np.sin(2*np.pi*frequency*time)[:, None]*np.ones((1, 4))*.001
        response = self.m.periodic_response(dt, road)
        w = 2*np.pi*frequency
        a = np.array([[30000+1j*w*1000-w*w*50, -30000-1j*w*1000],
                      [-30000-1j*w*1000, 180000+1j*w*1000-w*w*10]])
        quarter = np.linalg.solve(a, [0, 150000])
        expected = .001*np.imag(quarter[0]*np.exp(1j*w*time))
        np.testing.assert_allclose(response['coordinates'][:, 0], expected, atol=1e-14)
        self.assertTrue(response['contact']['linear_contact_valid'])

    def test_periodic_rejects_unrepresentable_nyquist(self):
        road = np.tile((-1.)**np.arange(100), (4, 1)).T
        with self.assertRaises(ValueError): self.m.periodic_response(.01, road)

    def test_contact_time_integration_matches_small_harmonic(self):
        # Start on the analytic steady-state orbit; interpolation is the only
        # input approximation. No settling window or fitted amplitude needed.
        f, amplitude = 3., .0001
        time = np.linspace(0., .5, 1001)
        w = 2*np.pi*f
        h = self.m.transfer([f])['coordinates'][0] @ np.ones(4)
        state = np.r_[amplitude*h.imag, amplitude*w*h.real]
        road = amplitude*np.sin(w*time)[:, None]*np.ones((1, 4))
        response = self.m.contact_response(time, road, initial_state=state)
        expected = amplitude*np.imag(np.exp(1j*w*time[:, None])*h)
        np.testing.assert_allclose(response['coordinates'], expected, atol=3e-9)
        np.testing.assert_array_equal(response['contact_loss_sample_fraction'], 0.)

    def test_airborne_force_balance_and_recontact(self):
        # All tyres initially above the road: external total vertical force is
        # gravity. Internal spring/damper forces must cancel in COM momentum.
        state = np.zeros(14); state[[0, 3, 4, 5, 6]] = .03
        time = np.linspace(0., .2, 1001)
        response = self.m.contact_response(time, np.zeros((len(time), 4)),
            initial_state=state, travel_limits_m=np.array([[-.001]*4, [.001]*4]))
        np.testing.assert_array_equal(response['normal_load_N'][0], 0.)
        masses = np.diag(self.m.mass)[[0, 3, 4, 5, 6]]
        momentum = response['velocity'][:, [0, 3, 4, 5, 6]] @ masses
        np.testing.assert_allclose((momentum[1]-momentum[0])/(time[1]-time[0]),
                                   -masses.sum()*9.81, rtol=1e-8)
        self.assertTrue(np.any(response['normal_load_N'][-1] > 0))
        self.assertFalse(response['within_travel_bounds'])
        self.assertTrue(np.all(response['normal_load_N'] >= 0))

    def test_contact_input_validation(self):
        for time, road, kwargs in [
                (np.array([0, 1], complex), np.zeros((2, 4)), {}),
                ([0, 1], np.ones((2, 4), complex)*1j, {}),
                ([0, 1], np.zeros((2, 4)), {'initial_state': np.zeros(14, complex)}),
                ([0, 1], np.zeros((2, 4)), {'travel_limits_m': np.zeros((2, 4), complex)})]:
            with self.assertRaises(ValueError):
                self.m.contact_response(time, road, **kwargs)
        with self.assertRaises(ValueError):
            self.m.contact_response([0, 0], np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            self.m.contact_response([0, 1], np.zeros((2, 4)), initial_state=np.zeros(7))
        with self.assertRaises(ValueError):
            self.m.contact_response([0, 1], np.zeros((2, 4)), travel_limits_m=np.zeros((2, 4)))

from dataclasses import replace
from vahan import ride_solve as RS


def _solve_vehicle(**kw):
    """Standard-topology car with a real motion ratio and the geometric
    wheel-rate term switched on, so the spring inversion is exercised."""
    base = dict(sprung_mass_kg=200, unsprung_mass_front_kg=20, unsprung_mass_rear_kg=20,
                wheelbase_m=2, cg_to_front_axle_m=1, front_track_m=1.2, rear_track_m=1.2,
                spring_rate_front_Npm=30000, spring_rate_rear_Npm=30000,
                motion_ratio_front=0.7, motion_ratio_rear=0.7, tire_rate_Npm=150000,
                arb_rate_front_Npm=5000, arb_rate_rear_Npm=5000,
                mr_slope_front_per_m=0.2, mr_slope_rear_per_m=0.2,
                static_spring_force_front_N=700., static_spring_force_rear_N=700.)
    base.update(kw)
    return VehicleParams(**base)


def _fast_inputs(**kw):
    """Short records so the solve checks run in seconds."""
    base = dict(pitch_inertia_kgm2=60, roll_inertia_kgm2=30, corner_damping_Nspm=(800.,)*4,
                road_class='A', speeds_mps=(15.,), seeds=(3,), coherence='coherent',
                stop_margin_m=0.03, record_length_m=200., spacing_m=0.05, settle_record_s=3.)
    base.update(kw)
    return RS.RideInputs(**base)


class RideSolveChecks(unittest.TestCase):
    def setUp(self):
        self.v = _solve_vehicle()

    def test_ride_frequency_round_trip(self):
        k, m = 13388.2, 47.5
        f = RS.ride_frequency_Hz(k, m)
        self.assertAlmostEqual(RS.ride_rate_from_frequency_Npm(f, m), k, places=6)
        self.assertAlmostEqual(f, np.sqrt(k/m)/(2*np.pi))

    def test_sprung_corner_masses_match_ride_model_split(self):
        mf, mr = RS.sprung_corner_masses_kg(self.v)
        self.assertAlmostEqual(2*(mf+mr), self.v.sprung_mass_kg)
        model = RideModel.from_vehicle(self.v, pitch_inertia_kgm2=60, roll_inertia_kgm2=30,
                                       corner_damping_Nspm=[0]*4)
        # Moment balance: the sprung CG sits where RideModel put it.
        sprung_a = self.v.wheelbase_m*2*mr/self.v.sprung_mass_kg
        self.assertAlmostEqual(model.body_corner_map[0, 1], sprung_a)

    def test_spring_rate_for_ride_rate_round_trips_through_vehicle_params(self):
        for axle in RS.AXLES:
            k_now = getattr(self.v, f'spring_rate_{axle}_Npm')
            r_now = getattr(self.v, f'ride_rate_{axle}_Npm')
            self.assertAlmostEqual(RS.spring_rate_for_ride_rate_Npm(self.v, axle, r_now), k_now, places=6)
            target = 1.3*r_now
            k = RS.spring_rate_for_ride_rate_Npm(self.v, axle, target)
            got = getattr(replace(self.v, **{f'spring_rate_{axle}_Npm': k}), f'ride_rate_{axle}_Npm')
            self.assertAlmostEqual(got, target, places=6)
            # Analytic: wheel = ride*tire/(tire-ride); spring = (wheel - Fs*slope)/MR^2
            wheel = target*self.v.tire_rate_Npm/(self.v.tire_rate_Npm-target)
            self.assertAlmostEqual(k, (wheel-700.*0.2)/0.7**2, places=6)
        with self.assertRaises(ValueError):
            RS.spring_rate_for_ride_rate_Npm(self.v, 'front', self.v.tire_rate_Npm)
        both = RS.vehicle_with_ride_rates(self.v, 12000., 9000.)
        self.assertAlmostEqual(both.ride_rate_front_Npm, 12000., places=6)
        self.assertAlmostEqual(both.ride_rate_rear_Npm, 9000., places=6)

    def test_ride_rate_grid_is_even_in_frequency(self):
        rates, freqs = RS.ride_rate_grid_Npm(self.v, 'rear', 1.5, 3.5, 5)
        np.testing.assert_allclose(freqs, [1.5, 2., 2.5, 3., 3.5])
        m = RS.sprung_corner_masses_kg(self.v)[1]
        np.testing.assert_allclose(rates, [(2*np.pi*f)**2*m for f in freqs])

    def test_class_B_road_doubles_every_linear_response(self):
        inp_a, inp_b = _fast_inputs(), _fast_inputs(road_class='B')
        model = RS.ride_model_for(self.v, inp_a)
        mr = RS.corner_motion_ratios(self.v)
        case_a = RS.build_road_cases(inp_a, self.v.wheelbase_m)[0]
        case_b = RS.build_road_cases(inp_b, self.v.wheelbase_m)[0]
        np.testing.assert_allclose(case_b.road_heights_m, 2*case_a.road_heights_m, atol=1e-12)
        ma = RS.evaluate_case(model, case_a, motion_ratios=mr, stop_margin_m=inp_a.stop_margin_m)
        mb = RS.evaluate_case(model, case_b, motion_ratios=mr, stop_margin_m=inp_b.stop_margin_m)
        for name in ('dlc', 'travel_peak_m', 'damper_velocity_rms_mps', 'body_corner_acc_rms_mps2'):
            np.testing.assert_allclose(getattr(mb, name), 2*getattr(ma, name), rtol=1e-9)
        self.assertAlmostEqual(mb.body_heave_acc_rms_mps2, 2*ma.body_heave_acc_rms_mps2)
        # travel usage is peak / margin
        half = RS.evaluate_case(model, case_a, motion_ratios=mr, stop_margin_m=inp_a.stop_margin_m/2)
        np.testing.assert_allclose(half.travel_usage, 2*ma.travel_usage, rtol=1e-12)

    def test_rear_wheels_see_front_road_delayed_by_wheelbase(self):
        inp = _fast_inputs()
        case = RS.build_road_cases(inp, self.v.wheelbase_m)[0]
        shift = int(round(self.v.wheelbase_m/inp.spacing_m))
        np.testing.assert_allclose(case.road_heights_m[:, 2], np.roll(case.road_heights_m[:, 0], shift), atol=1e-12)
        np.testing.assert_array_equal(case.road_heights_m[:, 0], case.road_heights_m[:, 1])
        indep = RS.build_road_cases(_fast_inputs(coherence='independent'), self.v.wheelbase_m)[0]
        self.assertGreater(np.max(np.abs(indep.road_heights_m[:, 0]-indep.road_heights_m[:, 1])), 1e-4)

    def test_sweep_is_deterministic_and_seed_dependent(self):
        inp = _fast_inputs()
        fr, _ = RS.ride_rate_grid_Npm(self.v, 'front', 2., 3., 2)
        rr, _ = RS.ride_rate_grid_Npm(self.v, 'rear', 2., 3., 2)
        a = RS.sweep_ride_rates(self.v, inp, fr, rr)
        b = RS.sweep_ride_rates(self.v, inp, fr, rr)
        np.testing.assert_array_equal(a.dlc, b.dlc)
        np.testing.assert_array_equal(a.pitch_to_bounce, b.pitch_to_bounce)
        # Parseval: random-phase synthesis fixes the amplitude spectrum, so
        # every RMS metric of the linear response is seed-INDEPENDENT on
        # coherent tracks; only the peaks (travel, min load) move with the seed.
        c = RS.sweep_ride_rates(self.v, _fast_inputs(seeds=(4,)), fr, rr)
        np.testing.assert_allclose(c.dlc, a.dlc, rtol=1e-9)
        self.assertGreater(np.max(np.abs(c.travel_peak_m-a.travel_peak_m)), 1e-5)
        self.assertGreater(np.max(np.abs(c.min_load_N-a.min_load_N)), 1e-2)
        self.assertEqual(a.case_speeds_mps, (15.,))
        self.assertTrue(np.all(a.dlc > 0) and np.all(np.isfinite(a.dlc)))
        sel = RS.describe_selection(a, self.v, inp, preload_front_mm=0, preload_rear_mm=0, stroke_mm=55)
        i, j = sel['i_front'], sel['j_rear']
        self.assertAlmostEqual(sel['front_ride_rate_Npm'], fr[i]); self.assertAlmostEqual(sel['rear_ride_rate_Npm'], rr[j])
        chosen = RS.vehicle_with_ride_rates(self.v, fr[i], rr[j])
        self.assertAlmostEqual(sel['front_spring_rate_Npm'], chosen.spring_rate_front_Npm)
        self.assertAlmostEqual(sel['rear_ride_frequency_Hz']/sel['front_ride_frequency_Hz'], sel['flat_ride_ratio'])

    def test_selection_prefers_feasible_then_travel_then_unconstrained(self):
        def sweep(travel, ratio):
            dlc = np.array([[[0.30]*4], [[0.10]*4]])[:, :, None, :]   # (2,1,1,4): stiff row has the lower DLC
            z4 = np.zeros((2, 1, 1, 4))
            return RS.RideSweep(np.array([1., 2.]), np.array([1.]), ['c'], (10.,), dlc,
                                travel_usage=z4+np.array(travel)[:, None, None, None],
                                travel_peak_m=z4, damper_velocity_rms_mps=z4, damper_velocity_peak_mps=z4,
                                contact_loss_fraction_linear=z4, min_load_N=z4,
                                body_heave_acc_rms_mps2=np.zeros((2, 1, 1)), settle_pass=np.ones((2, 1, 1), bool),
                                pitch_to_bounce=np.array(ratio)[:, None, None], front_settle_s=np.zeros((2, 1, 1)),
                                rear_settle_s=np.zeros((2, 1, 1)), stop_margin_m=0.03, flat_ride_max_ratio=1.0,
                                case_speeds_mps=(10.,))
        s = sweep([0.5, 0.5], [0.8, 0.8]).select()
        self.assertEqual((s['i_front'], s['status'], s['n_feasible']), (1, 'feasible', 2))
        s = sweep([0.5, 0.5], [0.8, 1.5]).select()          # stiff row fails flat ride -> soft row wins
        self.assertEqual((s['i_front'], s['status']), (0, 'feasible'))
        s = sweep([0.5, 0.5], [1.5, 1.5]).select()          # nothing meets flat ride -> best travel-feasible
        self.assertEqual(s['i_front'], 1); self.assertIn('flat-ride threshold not met', s['status'])
        s = sweep([1.5, 1.5], [0.8, 0.8]).select()          # nothing fits the travel margin
        self.assertEqual(s['i_front'], 1); self.assertIn('travel margin', s['status'])

    def test_standard_springs_include_the_exact_catalogue_rate(self):
        k = 300.*RS.LBF_PER_IN_TO_NPM
        veh = replace(self.v, spring_rate_front_Npm=k)
        rows = RS.standard_spring_options(veh, 'front', k, preload_mm=5., stroke_mm=55.)
        exact = [r for r in rows if r['spring_lbf_in'] == 300.][0]
        self.assertAlmostEqual(exact['ride_rate_deviation_pct'], 0.)
        self.assertAlmostEqual(exact['preload_to_hold_ride_height_mm'], 5.)
        self.assertAlmostEqual(exact['ride_rate_Npm'], veh.ride_rate_front_Npm)
        stiffer = [r for r in rows if r['spring_lbf_in'] == 325.][0]
        self.assertLess(stiffer['preload_to_hold_ride_height_mm'], 5.)
        self.assertGreater(stiffer['ride_rate_deviation_pct'], 0.)
        self.assertEqual([r['spring_lbf_in'] for r in rows], [275., 300., 325., 350.])

    def test_bump_flat_ride_penalises_stiff_front(self):
        # RCVD p.410: a front stiffer than the rear pitches after a bump.
        # Lightly damped so the pitch mode is visible in the post-bump window.
        inp = _fast_inputs(corner_damping_Nspm=(300.,)*4, speeds_mps=(20.,))
        even = RS.ride_model_for(RS.vehicle_with_ride_rates(self.v, 12000., 12000.), inp)
        stiff_front = RS.ride_model_for(RS.vehicle_with_ride_rates(self.v, 22000., 12000.), inp)
        b_even = RS.bump_settling(even, inp, 20., self.v.wheelbase_m)
        b_stiff = RS.bump_settling(stiff_front, inp, 20., self.v.wheelbase_m)
        self.assertGreater(b_stiff['pitch_to_bounce_ratio'], b_even['pitch_to_bounce_ratio'])
        self.assertTrue(b_even['settled_in_record'])
        self.assertAlmostEqual(b_even['window_start_s'], self.v.wheelbase_m/20.+inp.bump_length_m/20.)
        # rear road = front road delayed by wheelbase/speed
        dt = b_even['time_s'][1]-b_even['time_s'][0]
        shift = int(round(self.v.wheelbase_m/20./dt))
        np.testing.assert_allclose(b_even['road_rear_m'], np.roll(b_even['road_front_m'], shift), atol=1e-9)


class LaunchLagChecks(unittest.TestCase):
    def setUp(self):
        from vahan import ride_solve as RS
        self.RS = RS
        self.v = VehicleParams(sprung_mass_kg=200, unsprung_mass_front_kg=20, unsprung_mass_rear_kg=20,
                               wheelbase_m=2, cg_to_front_axle_m=1, front_track_m=1.2, rear_track_m=1.2,
                               spring_rate_front_Npm=30000, spring_rate_rear_Npm=30000, motion_ratio_front=1,
                               motion_ratio_rear=1, tire_rate_Npm=150000, arb_rate_front_Npm=5000,
                               arb_rate_rear_Npm=5000)
        self.inp = RS.RideInputs(pitch_inertia_kgm2=60, roll_inertia_kgm2=30, corner_damping_Nspm=(1000.,) * 4)

    def lag(self, pct, rise=0.):
        slope = pct / 100 * self.v.cg_height_m / self.v.wheelbase_m
        return self.RS.launch_load_lag(self.v, self.inp, accel_g=1.0, thrust_rise_time_s=rise,
                                       rear_slope_from_wheel_centre=slope, front_slope_from_wheel_centre=0.,
                                       rear_wheel_centre_height_m=self.v.unsprung_cg_height_m,
                                       front_wheel_centre_height_m=self.v.unsprung_cg_height_m)

    def test_steady_load_gain_is_textbook_transfer_for_any_anti_squat(self):
        for pct in (0., 30., 100.):
            r = self.lag(pct)
            self.assertAlmostEqual(r['final_load_gain_N'], r['textbook_load_gain_N'], delta=0.5)
            self.assertAlmostEqual(r['rear_wheel_load_gain_N'][-1], r['final_load_gain_N'],
                                   delta=0.05 * r['final_load_gain_N'])

    def test_more_anti_squat_delivers_the_load_sooner_and_squats_less(self):
        t50 = [self.lag(p)['t50_s'] for p in (0., 30., 60., 100.)]
        self.assertTrue(all(a > b for a, b in zip(t50, t50[1:])), t50)
        self.assertLess(abs(self.lag(100.)['rear_squat_m'][-1]), 0.15 * abs(self.lag(0.)['rear_squat_m'][-1]))

    def test_contact_patch_sweep_shapes_and_current_point(self):
        RS = self.RS
        inp = RS.RideInputs(pitch_inertia_kgm2=60, roll_inertia_kgm2=30, corner_damping_Nspm=(1000.,) * 4,
                            speeds_mps=(20.,), coherence='coherent', record_length_m=200.)
        out = RS.contact_patch_vs_ride_frequency(self.v, inp, [2.0, 3.0])
        for axle in RS.AXLES:
            self.assertEqual(out[axle]['rms_pct_of_static'].shape, (2,))
            self.assertTrue(np.all(out[axle]['rms_pct_of_static'] > 0))
            self.assertIn(round(out[axle]['current_Hz'], 3), out[axle]['load_per_mm_road_N'])


if __name__ == '__main__': unittest.main()
