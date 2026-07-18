"""
vahan/analysis_plots.py — Higher-level analysis plots that go beyond the
standard kinematic / dynamics graphs. Each function takes the existing
solver/tire/vehicle state and returns a matplotlib Figure.

Functions:
  - plot_brake_capacity(...)     : required line pressure vs deceleration g
  - plot_ride_freq_bump(...)     : pitch over a single-wheel bump at varying speeds
  - plot_rc_vs_body_roll(...)    : per-axle RC height vs body roll (from dynamics sweep)
  - plot_steering_torque(...)    : steering-wheel torque vs steer angle
  - plot_ackermann_loads(...)    : Fy vs slip angle at inner/outer front Fz
  - plot_ackermann_demand(...)   : tire SA demand vs Ackermann geometry supply
  - plot_mmd(...)                : Milliken Moment Diagram (pure cornering)

All plots use the existing "dark theme" facecolor so they match the rest
of the GUI; the PlotDialog provides a light-theme export option.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


_DARK_BG = '#0e0e10'
_PANEL_BG = '#101013'
_GRID = '#222'
_TICK = '#888'
_TEXT = '#cccccc'

_BLUE = '#42A5F5'
_RED = '#EF5350'
_YELLOW = '#FFD600'
_WHITE = '#ffffff'
_TEAL = '#26C6DA'


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _styled_fig(figsize=(9, 5), n=1, m=1):
    fig = Figure(figsize=figsize, facecolor=_DARK_BG)
    axes = []
    for i in range(n * m):
        ax = fig.add_subplot(n, m, i + 1)
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_TICK, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, lw=0.5)
        ax.title.set_color(_TEXT)
        ax.xaxis.label.set_color(_TICK)
        ax.yaxis.label.set_color(_TICK)
        axes.append(ax)
    if n * m == 1:
        return fig, axes[0]
    return fig, axes


# ═════════════════════════════════════════════════════════════════════════════
# 1. Brake capacity validation
# ═════════════════════════════════════════════════════════════════════════════
def plot_brake_capacity(
    total_mass_kg: float, weight_dist_front: float, cg_height_m: float,
    wheelbase_m: float, tire_radius_m: float, brake_bias_front: float,
    pad_mu: float, piston_area_per_caliper_mm2: float,
    rotor_outer_dia_in: float, pad_height_in: float = 0.85,
    typical_max_line_psi: float = 1500,
    tire_model=None,
) -> Figure:
    """Required line pressure vs deceleration g, both axles, compared to typical max.

    When *tire_model* is provided the plot computes the effective peak
    braking deceleration from tire grip (peak_Fy as proxy for μ_x via the
    friction-circle assumption) and forward weight transfer.
    """
    m = total_mass_kg
    g_acc = 9.81
    in_to_m = 0.0254
    wf = weight_dist_front
    r_eff = (rotor_outer_dia_in * in_to_m) / 2 - (pad_height_in * in_to_m) / 2
    A_piston_m2 = piston_area_per_caliper_mm2 * 1e-6   # one-side total

    g_arr = np.linspace(0.05, 2.5, 50)
    psi = 6894.76

    # ── Required line pressure per axle ─────────────────────────────────────
    Pf, Pr = [], []
    for gx in g_arr:
        F_total = m * gx * g_acc
        Ff = brake_bias_front * F_total
        Fr = (1 - brake_bias_front) * F_total
        Tf = (Ff / 2) * tire_radius_m
        Tr = (Fr / 2) * tire_radius_m
        denom = 2 * pad_mu * A_piston_m2 * r_eff
        Pf.append(Tf / denom / psi)
        Pr.append(Tr / denom / psi)
    Pf, Pr = np.array(Pf), np.array(Pr)

    # ── Tire-limited peak decel ─────────────────────────────────────────────
    # For each trial decel ax (g):
    #   Fz_front = m·g·(wf + ax·h/WB)/2    (weight transfers forward)
    #   Fz_rear  = m·g·((1-wf) - ax·h/WB)/2
    #   Grip available = peak_Fy(Fz)  (friction-circle: μ_x ≈ μ_y)
    #   Demand per tire = bias_fraction · m·ax·g / 2
    #   Lockup when demand > available at either axle.
    peak_g = None
    lock_axle = ''
    ideal_peak_g = None
    if tire_model is not None and hasattr(tire_model, 'peak_Fy'):
        ax_sweep = np.linspace(0.05, 3.0, 300)
        for ax in ax_sweep:
            h_wb = cg_height_m / wheelbase_m
            Fz_f = m * g_acc * (wf + ax * h_wb) / 2.0    # per tire
            Fz_r = m * g_acc * ((1.0 - wf) - ax * h_wb) / 2.0
            if Fz_r < 10.0:          # rear unloads completely
                peak_g = peak_g or ax
                lock_axle = lock_axle or 'rear (unloads)'
                break
            grip_f = abs(float(tire_model.peak_Fy(max(Fz_f, 10.0))))
            grip_r = abs(float(tire_model.peak_Fy(max(Fz_r, 10.0))))
            demand_f = brake_bias_front * m * ax * g_acc / 2.0
            demand_r = (1.0 - brake_bias_front) * m * ax * g_acc / 2.0
            if peak_g is None:
                if demand_f > grip_f:
                    peak_g = ax; lock_axle = 'front'
                elif demand_r > grip_r:
                    peak_g = ax; lock_axle = 'rear'
            # Ideal-bias peak (total grip vs total demand)
            if ideal_peak_g is None:
                total_grip = 2.0 * (grip_f + grip_r)
                total_demand = m * ax * g_acc
                if total_demand > total_grip:
                    ideal_peak_g = ax

    fig, ax = _styled_fig(figsize=(9.5, 5))
    ax.plot(g_arr, Pf, color=_BLUE, lw=2.4, label='Front required')
    ax.plot(g_arr, Pr, color=_RED, lw=2.4, ls='--', label='Rear required')
    ax.axhline(typical_max_line_psi, color=_TICK, lw=1.2, ls=':',
               label=f'Typical FSAE max ({typical_max_line_psi:.0f} psi)')

    # Peak decel markers
    if peak_g is not None:
        Pf_at_peak = float(np.interp(peak_g, g_arr, Pf))
        Pr_at_peak = float(np.interp(peak_g, g_arr, Pr))
        ax.axvline(peak_g, color=_YELLOW, lw=1.8, ls='--',
                   label=f'Tire-limited peak: {peak_g:.2f} g ({lock_axle} locks)')
        note = (f'Tire-limited peak: {peak_g:.2f} g  ({lock_axle} locks)\n'
                f'Front {Pf_at_peak:.0f} psi  '
                f'({typical_max_line_psi / max(Pf_at_peak, 1):.1f}× margin)   '
                f'Rear {Pr_at_peak:.0f} psi')
        if ideal_peak_g is not None and abs(ideal_peak_g - peak_g) > 0.02:
            ax.axvline(ideal_peak_g, color=_WHITE, lw=1.0, ls=':',
                       label=f'Ideal-bias peak: {ideal_peak_g:.2f} g')
            note += f'\nIdeal-bias peak: {ideal_peak_g:.2f} g  '
            note += f'(bias would need ≈ {_ideal_bias(tire_model, m, g_acc, wf, cg_height_m, wheelbase_m, ideal_peak_g):.0f}% F)'
    else:
        note = 'Load tire data to compute effective braking decel'

    ax.set_xlabel('Deceleration (g)', fontsize=10)
    ax.set_ylabel('Required brake line pressure (psi)', fontsize=10)
    ax.set_title(f'Brake-Torque Capacity Validation\n'
                 f'pad μ={pad_mu}, A_piston={piston_area_per_caliper_mm2:.0f} mm², '
                 f'rotor OD={rotor_outer_dia_in}", r_eff={r_eff*1000:.0f} mm',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT,
              edgecolor=_GRID, fontsize=9, loc='upper left')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, max(typical_max_line_psi * 1.15, Pf.max() * 1.15))

    ax.text(0.98, 0.02, note,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=_TEXT,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID))

    fig.tight_layout()
    return fig


def _ideal_bias(tire_model, m, g_acc, wf, h_cg, wb, ax_g):
    """Front brake bias that locks both axles simultaneously at *ax_g*."""
    h_wb = h_cg / wb
    Fz_f = m * g_acc * (wf + ax_g * h_wb) / 2.0
    Fz_r = m * g_acc * ((1.0 - wf) - ax_g * h_wb) / 2.0
    grip_f = abs(float(tire_model.peak_Fy(max(Fz_f, 10.0))))
    grip_r = abs(float(tire_model.peak_Fy(max(Fz_r, 10.0))))
    return 100.0 * grip_f / max(grip_f + grip_r, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Ride freq / bump response
# ═════════════════════════════════════════════════════════════════════════════
def plot_ride_freq_bump(
    sprung_total_kg: float, weight_dist_front: float, wheelbase_m: float,
    k_wheel_front_Npm: float, k_wheel_rear_Npm: float,
    damping_ratio: float = 0.5,
    speeds_mps: list = (5, 10, 15, 20),
    bump_height_m: float = 0.020, bump_duration_s: float = 0.06,
) -> Figure:
    """Pitch response over a single-wheel bump at varying lap speeds."""
    m_F = sprung_total_kg * weight_dist_front / 2
    m_R = sprung_total_kg * (1 - weight_dist_front) / 2

    f_F = np.sqrt(k_wheel_front_Npm / m_F) / (2 * np.pi)
    f_R = np.sqrt(k_wheel_rear_Npm / m_R) / (2 * np.pi)

    c_F = 2 * damping_ratio * np.sqrt(k_wheel_front_Npm * m_F)
    c_R = 2 * damping_ratio * np.sqrt(k_wheel_rear_Npm * m_R)

    def make_bump(t_start):
        def bump(t):
            if t < t_start or t > t_start + bump_duration_s:
                return 0.0
            return bump_height_m * np.sin(np.pi * (t - t_start) / bump_duration_s)
        return bump

    def axle_rhs(t, state, m, k, c, y_fn):
        x, xdot = state
        y = y_fn(t)
        eps = 1e-5
        ydot = (y_fn(t + eps) - y) / eps if y > 0 else 0
        return [xdot, (k * (y - x) + c * (ydot - xdot)) / m]

    def simulate(m, k, c, y_fn, t_eval):
        sol = solve_ivp(axle_rhs, (t_eval[0], t_eval[-1]), [0, 0],
                        t_eval=t_eval, args=(m, k, c, y_fn),
                        method='RK45', max_step=0.002, rtol=1e-6, atol=1e-8)
        return sol.y[0]

    n = len(speeds_mps)
    fig = Figure(figsize=(11, 7), facecolor=_DARK_BG)
    t_eval = np.linspace(0, 1.0, 3000)
    for i, V in enumerate(speeds_mps):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_TICK, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, lw=0.5)

        t_R = wheelbase_m / V
        bF = make_bump(0.0); bR = make_bump(t_R)
        xF = simulate(m_F, k_wheel_front_Npm, c_F, bF, t_eval)
        xR = simulate(m_R, k_wheel_rear_Npm, c_R, bR, t_eval)
        pitch = np.degrees(np.arctan((xF - xR) / wheelbase_m))

        ax.plot(t_eval * 1000, pitch, color=_BLUE, lw=2.0, label='Pitch')
        ax.plot(t_eval * 1000, xF * 1000, color=_TEAL, lw=1.0, ls='--', alpha=0.6,
                label='Front disp (mm)')
        ax.plot(t_eval * 1000, xR * 1000, color=_RED, lw=1.0, ls=':', alpha=0.6,
                label='Rear disp (mm)')
        ax.axhline(0, color=_TICK, lw=0.4)
        ax.axvline(t_R * 1000, color=_YELLOW, lw=1.0, ls='--', alpha=0.7,
                   label=f'Rear arrival ({t_R*1000:.0f} ms)')
        ax.set_ylabel(f'V={V*3.6:.0f} km/h\n({V} m/s)', fontsize=9,
                      color=_TICK)
        if i == 0:
            ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
                      fontsize=7, loc='upper right', ncol=2)

    fig.axes[-1].set_xlabel('Time (ms)', fontsize=10, color=_TICK)
    fig.suptitle(f'Pitch over a {bump_height_m*1000:.0f} mm Single-Wheel Bump  —  '
                 f'f_F = {f_F:.2f} Hz, f_R = {f_R:.2f} Hz, ζ = {damping_ratio}',
                 fontsize=11, color=_TEXT, y=0.995)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. RC height vs body roll (from dynamics sweep)
# ═════════════════════════════════════════════════════════════════════════════
def plot_rc_vs_body_roll(steady_solver, max_lat_g: float = 2.0) -> Figure:
    """Per-axle RC height vs body roll, from the dynamics solver lat-g sweep."""
    sweep = steady_solver.sweep_lateral_g(g_range=(0, max_lat_g), n_points=41,
                                           longitudinal_g=0.0)
    roll = np.asarray(sweep['roll_angle_deg'])
    rc_f = np.asarray(sweep['rc_height_front_mm'])
    rc_r = np.asarray(sweep['rc_height_rear_mm'])

    fig, ax = _styled_fig(figsize=(9.5, 5))
    ax.plot(roll, rc_f, color=_BLUE, lw=2.4, label='Front RC')
    ax.plot(roll, rc_r, color=_RED, lw=2.4, ls='--', label='Rear RC')

    # Mark operating max body roll
    op_max = float(roll.max())
    rc_f_at_max = float(np.interp(op_max, roll, rc_f))
    rc_r_at_max = float(np.interp(op_max, roll, rc_r))
    ax.axvline(op_max, color=_YELLOW, lw=1.4, ls=':',
               label=f'Max body roll achieved ({op_max:.2f}°)')

    ax.set_xlabel('Body roll angle (°)', fontsize=10)
    ax.set_ylabel('RC height above ground (mm)', fontsize=10)
    ax.set_title(f'Roll-Centre Height vs Body Roll  —  from dynamics sweep '
                 f'(0 → {max_lat_g:.1f} g)',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID, fontsize=9,
              loc='best')
    ax.text(0.02, 0.02,
            f'At {op_max:.2f}° body roll:  Front Δ = {rc_f_at_max - rc_f[0]:+.2f} mm,  '
            f'Rear Δ = {rc_r_at_max - rc_r[0]:+.2f} mm',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color=_TEXT,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID))
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. Steering torque vs steer angle
# ═════════════════════════════════════════════════════════════════════════════
def plot_steering_torque(
    tire_model, vehicle_params,
    scrub_radius_m: float, kpi_deg: float,
    steering_ratio: float = 6.0,
    speeds_mps: list = (10, 15, 20),
    wheelbase_m: float = 1.537, weight_dist_front: float = 0.45,
) -> Figure:
    """Estimated steering-wheel torque vs front-wheel steer angle at constant V."""
    m = vehicle_params.total_mass_kg
    g_acc = 9.81
    a_dist = wheelbase_m * (1 - weight_dist_front)  # CG to front
    b_dist = wheelbase_m * weight_dist_front         # CG to rear
    steering_arm_m = scrub_radius_m * np.cos(np.radians(kpi_deg))

    delta_arr = np.linspace(0.5, 25, 30)
    fig, ax = _styled_fig(figsize=(9.5, 5))
    colors = [_BLUE, _YELLOW, _RED]

    Fz_front_static = m * g_acc * (b_dist / wheelbase_m) / 2
    for V, col in zip(speeds_mps, colors):
        ts = []
        for d in delta_arr:
            R = wheelbase_m / np.tan(np.radians(d))
            Ay = V * V / R / g_acc
            if Ay > 2.5: continue
            Fy_demand = m * Ay * g_acc * (b_dist / wheelbase_m) / 2
            try:
                alpha = tire_model.slip_angle_for_Fy(Fy_demand, Fz_front_static, 0.0)
            except Exception:
                alpha = 5.0
            if alpha is None or np.isnan(alpha):
                alpha = 5.0
            Mz = float(tire_model.Mz(alpha, Fz_front_static, 0.0))
            T_kingpin = abs(Mz) + abs(Fy_demand) * abs(steering_arm_m)
            T_wheel = 2 * T_kingpin / steering_ratio
            ts.append((d, T_wheel))
        if ts:
            xs, ys = zip(*ts)
            ax.plot(xs, ys, color=col, lw=2.0, label=f'V = {V*3.6:.0f} km/h')

    ax.set_xlabel('Front-wheel steer angle (°)', fontsize=10)
    ax.set_ylabel('Steering-wheel torque (N·m)', fontsize=10)
    ax.set_title(f'Steering-Wheel Torque vs Steer Angle\n'
                 f'(theoretical, from TTC Mz + scrub-radius geometry)',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID, fontsize=9,
              loc='lower right')
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 5. Ackermann inner/outer load reasoning
# ═════════════════════════════════════════════════════════════════════════════
def plot_ackermann_loads(
    tire_model, total_mass_kg: float, weight_dist_front: float,
    wheelbase_m: float, cg_height_m: float, track_front_m: float,
    lat_g: float = 1.0,
) -> Figure:
    """Fy vs slip angle for inner vs outer front tire at given lateral g."""
    g = 9.81
    b_dist = wheelbase_m * weight_dist_front  # CG to rear
    Fz_static_front = total_mass_kg * g * (b_dist / wheelbase_m) / 2
    # Front-axle lateral LT (assume equal-split inner/outer for simplicity)
    LT_total = total_mass_kg * lat_g * g * cg_height_m / track_front_m
    Fz_inner = max(Fz_static_front - LT_total / 2, 10.0)
    Fz_outer = Fz_static_front + LT_total / 2

    sa = np.linspace(0, 12, 25)
    Fy_inner = np.array([abs(float(tire_model.Fy(s, Fz_inner, 0.0))) for s in sa])
    Fy_outer = np.array([abs(float(tire_model.Fy(s, Fz_outer, 0.0))) for s in sa])

    pk_i = np.argmax(Fy_inner); pk_o = np.argmax(Fy_outer)

    fig, ax = _styled_fig(figsize=(9.5, 5))
    ax.plot(sa, Fy_inner, color=_BLUE, lw=2.4,
            label=f'INNER  (Fz = {Fz_inner:.0f} N)')
    ax.plot(sa, Fy_outer, color=_RED, lw=2.4,
            label=f'OUTER  (Fz = {Fz_outer:.0f} N)')
    ax.plot(sa[pk_i], Fy_inner[pk_i], 'o', color=_BLUE, markersize=9,
            markeredgecolor=_WHITE)
    ax.plot(sa[pk_o], Fy_outer[pk_o], 'o', color=_RED, markersize=9,
            markeredgecolor=_WHITE)
    ax.annotate(f'Inner peak\nα = {sa[pk_i]:.1f}°\nFy = {Fy_inner[pk_i]:.0f} N',
                xy=(sa[pk_i], Fy_inner[pk_i]),
                xytext=(sa[pk_i] - 3, Fy_inner[pk_i] + 200),
                fontsize=9, color=_BLUE)
    ax.annotate(f'Outer peak\nα = {sa[pk_o]:.1f}°\nFy = {Fy_outer[pk_o]:.0f} N',
                xy=(sa[pk_o], Fy_outer[pk_o]),
                xytext=(sa[pk_o] + 0.3, Fy_outer[pk_o] - 250),
                fontsize=9, color=_RED)
    ax.set_xlabel('Slip angle |α| (°)', fontsize=10)
    ax.set_ylabel('Lateral force |Fy| (N)', fontsize=10)
    ax.set_title(f'Why Partial Ackermann: Inner vs Outer Front-Tire Fy at {lat_g:.1f} g Lateral',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
              fontsize=9, loc='lower right')
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6. Ackermann slip-angle demand vs geometry supply
# ═════════════════════════════════════════════════════════════════════════════
def plot_ackermann_demand(
    tire_model, total_mass_kg: float, weight_dist_front: float,
    wheelbase_m: float, cg_height_m: float, track_front_m: float,
    ackermann_pct: float = 33.0,
    velocity_mps: float = 13.4,
) -> Figure:
    """Tire peak-Fy slip-angle demand vs Ackermann geometry supply.

    Top panel  — peak-|Fy| slip angle for inner & outer front tires vs lateral g.
    Bottom panel — Δα (inner−outer) from TTC demand vs geometry supply at the
                   car's Ackermann % and at 100 % (reference).
    """
    g_acc = 9.81
    m = total_mass_kg
    b_dist = wheelbase_m * weight_dist_front        # CG to rear
    Fz_static_front = m * g_acc * (b_dist / wheelbase_m) / 2.0
    L = wheelbase_m
    t = track_front_m

    lat_g_arr = np.linspace(0.1, 2.0, 40)
    sa_sweep = np.linspace(0, 14, 70)               # for peak search

    sa_pk_in, sa_pk_out = [], []
    fz_in_arr, fz_out_arr = [], []

    for ag in lat_g_arr:
        LT = m * ag * g_acc * cg_height_m / t
        fz_i = max(Fz_static_front - LT / 2.0, 10.0)
        fz_o = Fz_static_front + LT / 2.0
        fz_in_arr.append(fz_i)
        fz_out_arr.append(fz_o)

        Fy_i = np.array([abs(float(tire_model.Fy(s, fz_i, 0.0))) for s in sa_sweep])
        Fy_o = np.array([abs(float(tire_model.Fy(s, fz_o, 0.0))) for s in sa_sweep])
        sa_pk_in.append(float(sa_sweep[np.argmax(Fy_i)]))
        sa_pk_out.append(float(sa_sweep[np.argmax(Fy_o)]))

    sa_pk_in  = np.array(sa_pk_in)
    sa_pk_out = np.array(sa_pk_out)
    demand    = sa_pk_in - sa_pk_out                 # + = inner wants more SA

    # ── Ackermann geometry supply ──
    supply_100  = np.full_like(lat_g_arr, np.nan)
    supply_act  = np.full_like(lat_g_arr, np.nan)
    steer_avg   = np.full_like(lat_g_arr, np.nan)
    V = velocity_mps
    for k, ag in enumerate(lat_g_arr):
        R = V * V / (ag * g_acc)
        if R < t:                                    # tighter than track — skip
            continue
        d_avg = np.degrees(np.arctan(L / R))
        d_inner_100 = np.degrees(np.arctan(L / max(R - t / 2.0, 0.01)))
        d_outer_100 = np.degrees(np.arctan(L / (R + t / 2.0)))
        s100 = d_inner_100 - d_outer_100
        supply_100[k] = s100
        supply_act[k] = s100 * ackermann_pct / 100.0
        steer_avg[k]  = d_avg

    # ── Figure: two vertically stacked subplots ──
    fig = Figure(figsize=(10, 8), facecolor=_DARK_BG)

    # ── Top panel: individual peak SAs ──────────────────────────────────
    ax1 = fig.add_subplot(2, 1, 1)
    for sp in ax1.spines.values():
        sp.set_edgecolor(_GRID)
    ax1.set_facecolor(_PANEL_BG)
    ax1.tick_params(colors=_TICK, labelsize=8)
    ax1.grid(True, color=_GRID, lw=0.5)

    ax1.plot(lat_g_arr, sa_pk_in,  color=_BLUE, lw=2.4, label='Inner peak SA')
    ax1.plot(lat_g_arr, sa_pk_out, color=_RED,  lw=2.4, label='Outer peak SA')
    ax1.fill_between(lat_g_arr, sa_pk_in, sa_pk_out, color=_YELLOW, alpha=0.12,
                     label='Optimal SA gap')

    ax1.set_ylabel('Slip angle at peak |Fy| (°)', fontsize=10, color=_TICK)
    ax1.set_title('Peak-Fy Slip Angle for Inner & Outer Front Tires',
                  fontsize=10, color=_TEXT)
    ax1.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
               fontsize=9, loc='upper left')

    # Secondary y-axis: Fz for each tire
    ax1b = ax1.twinx()
    fz_in_np  = np.array(fz_in_arr)
    fz_out_np = np.array(fz_out_arr)
    ax1b.plot(lat_g_arr, fz_in_np,  color=_BLUE, lw=1.0, ls=':', alpha=0.5)
    ax1b.plot(lat_g_arr, fz_out_np, color=_RED,  lw=1.0, ls=':', alpha=0.5)
    ax1b.set_ylabel('Normal load Fz (N)', fontsize=9, color=_TICK)
    ax1b.tick_params(colors=_TICK, labelsize=8)
    ax1b.spines['right'].set_edgecolor(_GRID)
    # Label the Fz lines
    ax1b.text(lat_g_arr[-1], fz_in_np[-1], f'  {fz_in_np[-1]:.0f} N',
              fontsize=8, color=_BLUE, va='center', alpha=0.7)
    ax1b.text(lat_g_arr[-1], fz_out_np[-1], f'  {fz_out_np[-1]:.0f} N',
              fontsize=8, color=_RED, va='center', alpha=0.7)

    # ── Bottom panel: operating SA vs peak SA per tire ────────────────
    ax2 = fig.add_subplot(2, 1, 2)
    for sp in ax2.spines.values():
        sp.set_edgecolor(_GRID)
    ax2.set_facecolor(_PANEL_BG)
    ax2.tick_params(colors=_TICK, labelsize=8)
    ax2.grid(True, color=_GRID, lw=0.5)

    # Compute actual operating SA for each tire in steady-state cornering.
    # Bicycle model: Fy_front = m * a_y * g * (weight_dist_front).
    # Ackermann splits the steer angle → splits the slip angle (approx).
    # Solve for alpha_avg where |Fy(α+Δδ/2, Fz_in)| + |Fy(α-Δδ/2, Fz_out)| = Fy_front.
    op_inner = np.full_like(lat_g_arr, np.nan)
    op_outer = np.full_like(lat_g_arr, np.nan)
    op_inner_par = np.full_like(lat_g_arr, np.nan)  # parallel (0%) for comparison

    for k, ag in enumerate(lat_g_arr):
        fz_i = fz_in_arr[k]
        fz_o = fz_out_arr[k]
        Fy_front = m * ag * g_acc * weight_dist_front

        R = V * V / (ag * g_acc)
        if R < t:
            continue
        d_i_100 = np.degrees(np.arctan(L / max(R - t / 2.0, 0.01)))
        d_o_100 = np.degrees(np.arctan(L / (R + t / 2.0)))
        half = (d_i_100 - d_o_100) * ackermann_pct / 100.0 / 2.0

        def _solve_avg(h):
            """Find avg SA where inner+outer Fy = Fy_front, given half-split h."""
            def resid(a):
                fi = abs(float(tire_model.Fy(max(a + h, 0.01), fz_i, 0.0)))
                fo = abs(float(tire_model.Fy(max(a - h, 0.01), fz_o, 0.0)))
                return fi + fo - Fy_front
            try:
                return brentq(resid, 0.01, 14.0)
            except ValueError:
                # Tires saturated — find SA that maximises total Fy
                best_a, best_fy = 0.0, 0.0
                for a in np.linspace(0.5, 14, 50):
                    fi = abs(float(tire_model.Fy(max(a + h, 0.01), fz_i, 0.0)))
                    fo = abs(float(tire_model.Fy(max(a - h, 0.01), fz_o, 0.0)))
                    if fi + fo > best_fy:
                        best_a, best_fy = a, fi + fo
                return best_a

        # With Ackermann
        a_avg = _solve_avg(half)
        op_inner[k] = a_avg + half
        op_outer[k] = max(a_avg - half, 0.0)

        # Parallel (0% Ackermann) — same SA on both tires
        a_par = _solve_avg(0.0)
        op_inner_par[k] = a_par

    # Plot: peak (solid) vs operating (dashed) for each tire
    ax2.plot(lat_g_arr, sa_pk_in,  color=_BLUE, lw=2.4,
             label='Inner peak SA (max grip)')
    ax2.plot(lat_g_arr, sa_pk_out, color=_RED,  lw=2.4,
             label='Outer peak SA (max grip)')
    ax2.plot(lat_g_arr, op_inner,  color=_BLUE, lw=2.0, ls='--',
             label=f'Inner operating SA ({ackermann_pct:.0f}% Ack)')
    ax2.plot(lat_g_arr, op_outer,  color=_RED,  lw=2.0, ls='--',
             label=f'Outer operating SA ({ackermann_pct:.0f}% Ack)')
    ax2.plot(lat_g_arr, op_inner_par, color=_WHITE, lw=1.0, ls=':',
             alpha=0.5, label='Both tires at 0% (parallel)')

    ax2.set_xlabel('Lateral acceleration (g)', fontsize=10, color=_TICK)
    ax2.set_ylabel('Slip angle (°)', fontsize=10, color=_TICK)
    ax2.set_title(f'Tire Operating Point vs Peak Grip — '
                  f'{ackermann_pct:.0f}% Ackermann at {V*3.6:.0f} km/h',
                  fontsize=10, color=_TEXT)
    ax2.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
               fontsize=8, loc='best')

    # Annotate gaps at a reference g
    ref_g = 1.75
    ri = np.argmin(np.abs(lat_g_arr - ref_g))
    if np.isfinite(op_inner[ri]) and np.isfinite(op_outer[ri]):
        gap_i = sa_pk_in[ri] - op_inner[ri]
        gap_o = sa_pk_out[ri] - op_outer[ri]
        ax2.axvline(lat_g_arr[ri], color=_YELLOW, lw=1.2, ls='--', alpha=0.5)
        ax2.text(0.98, 0.02,
                 f'At {ref_g:.2f} g:\n'
                 f'  Inner: operating {op_inner[ri]:.1f}° vs peak {sa_pk_in[ri]:.1f}° '
                 f'(gap {gap_i:+.1f}°)\n'
                 f'  Outer: operating {op_outer[ri]:.1f}° vs peak {sa_pk_out[ri]:.1f}° '
                 f'(gap {gap_o:+.1f}°)',
                 transform=ax2.transAxes, ha='right', va='bottom',
                 fontsize=8, color=_TEXT,
                 bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID, alpha=0.9))

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7. Ackermann Fz–Fy operating map
# ═════════════════════════════════════════════════════════════════════════════
def plot_ackermann_fz_fy(
    tire_model, total_mass_kg: float, weight_dist_front: float,
    wheelbase_m: float, cg_height_m: float, track_front_m: float,
    ackermann_pct: float = 33.0,
    velocity_mps: float = 13.4,
) -> Figure:
    """Fz–Fy operating map with iso-slip-angle lines.

    Gray background lines show tire Fy at constant slip angles (2°–12°).
    Overlaid: demand paths (equal-utilisation Fy split) and Ackermann-supply
    paths for inner & outer front tires as lateral g sweeps.
    """
    g_acc = 9.81
    m = total_mass_kg
    wf = weight_dist_front
    Fz_static = m * g_acc * (wf) / 2.0  # static front-corner Fz
    L = wheelbase_m
    t = track_front_m
    V = velocity_mps

    fig, ax = _styled_fig(figsize=(10, 7))

    # ── Iso-slip-angle lines (tire capability background) ───────────
    fz_range = np.linspace(50, 1500, 80)
    sa_levels = [2, 4, 6, 8, 10, 12]
    for sa in sa_levels:
        fy_line = np.array([abs(float(tire_model.Fy(sa, fz, 0.0)))
                            for fz in fz_range])
        ax.plot(fz_range, fy_line, color=_TICK, lw=0.8, alpha=0.35)
        ax.text(fz_range[-1] + 15, fy_line[-1], f'α={sa}°',
                fontsize=7, color=_TICK, va='center', alpha=0.6)

    # Peak-Fy envelope
    peak_fy = np.array([abs(float(tire_model.peak_Fy(fz, 0.0)))
                        for fz in fz_range])
    ax.plot(fz_range, peak_fy, color=_YELLOW, lw=1.5, ls='--', alpha=0.6,
            label='Peak Fy envelope')

    # ── Sweep lateral g ─────────────────────────────────────────────
    lat_g_arr = np.linspace(0.2, 2.0, 35)

    fz_ip, fz_op = [], []
    fy_dem_i, fy_dem_o = [], []
    fy_ack_i, fy_ack_o = [], []

    for ag in lat_g_arr:
        LT = m * ag * g_acc * cg_height_m / t
        fz_i = max(Fz_static - LT / 2.0, 10.0)
        fz_o = Fz_static + LT / 2.0
        fz_ip.append(fz_i);  fz_op.append(fz_o)

        Fy_front = m * ag * g_acc * wf          # total front-axle Fy needed

        # ── Demand: equal-utilisation split ──
        pk_i = abs(float(tire_model.peak_Fy(fz_i, 0.0)))
        pk_o = abs(float(tire_model.peak_Fy(fz_o, 0.0)))
        tot_pk = max(pk_i + pk_o, 1.0)
        fy_d_i = Fy_front * pk_i / tot_pk
        fy_d_o = Fy_front * pk_o / tot_pk
        fy_dem_i.append(fy_d_i);  fy_dem_o.append(fy_d_o)

        # ── Ackermann supply (geometry → SA → Fy, no demand coupling) ──
        R = V * V / (ag * g_acc)
        if R < t:
            fy_ack_i.append(np.nan); fy_ack_o.append(np.nan)
            continue
        d_i100 = np.degrees(np.arctan(L / max(R - t / 2, 0.01)))
        d_o100 = np.degrees(np.arctan(L / (R + t / 2)))
        geo_half = (d_i100 - d_o100) / 2.0          # half geometric diff
        ack_half = geo_half * ackermann_pct / 100.0  # how much is recovered
        # At 0% Ackermann (parallel): outer gets geo_half MORE SA than avg,
        #   inner gets geo_half LESS — the outer is underturned geometrically.
        # At 100% Ackermann (ideal): both get equal SA (geo mismatch zeroed).
        # sa_i = a_avg - geo_half*(1-ack%), sa_o = a_avg + geo_half*(1-ack%)
        Ca_i = abs(float(tire_model.Fy(1.0, fz_i, 0.0)))   # N/deg
        Ca_o = abs(float(tire_model.Fy(1.0, fz_o, 0.0)))   # N/deg
        a_avg = min(m * ag * g_acc * wf / max(Ca_i + Ca_o, 1.0), 14.0)
        sa_i = max(a_avg - geo_half + ack_half, 0.01)   # inner: less SA at low ack%
        sa_o = max(a_avg + geo_half - ack_half, 0.01)   # outer: more SA at low ack%
        fy_ack_i.append(abs(float(tire_model.Fy(sa_i, fz_i, 0.0))))
        fy_ack_o.append(abs(float(tire_model.Fy(sa_o, fz_o, 0.0))))

    fz_ip = np.array(fz_ip);       fz_op = np.array(fz_op)
    fy_dem_i = np.array(fy_dem_i);  fy_dem_o = np.array(fy_dem_o)
    fy_ack_i = np.array(fy_ack_i);  fy_ack_o = np.array(fy_ack_o)

    # ── Plot demand & supply paths ──────────────────────────────────
    ax.plot(fz_ip, fy_dem_i, color=_BLUE, lw=2.4,
            label='Inner demand (equal util.)')
    ax.plot(fz_op, fy_dem_o, color=_RED,  lw=2.4,
            label='Outer demand (equal util.)')
    ax.plot(fz_ip, fy_ack_i, color=_BLUE, lw=2.0, ls='--',
            label=f'Inner at {ackermann_pct:.0f}% Ackermann')
    ax.plot(fz_op, fy_ack_o, color=_RED,  lw=2.0, ls='--',
            label=f'Outer at {ackermann_pct:.0f}% Ackermann')

    # Mark static Fz
    ax.axvline(Fz_static, color=_YELLOW, lw=1.0, ls=':', alpha=0.4,
               label=f'Static Fz ({Fz_static:.0f} N)')

    # Annotate a few lateral-g markers along the outer path
    for g_mark in [0.5, 1.0, 1.5]:
        idx = np.argmin(np.abs(lat_g_arr - g_mark))
        ax.plot(fz_op[idx], fy_dem_o[idx], 'o', color=_RED,
                markersize=5, markeredgecolor=_WHITE, zorder=5)
        ax.text(fz_op[idx] + 20, fy_dem_o[idx],
                f'{g_mark:.1f}g', fontsize=7, color=_TEXT)

    ax.set_xlabel('Normal load Fz (N)', fontsize=10, color=_TICK)
    ax.set_ylabel('Lateral force |Fy| (N)', fontsize=10, color=_TICK)
    ax.set_title(f'Fz–Fy Operating Map — {ackermann_pct:.0f}% Ackermann '
                 f'at {V * 3.6:.0f} km/h\n'
                 f'(gray lines = constant slip angle)',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
              fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 8. Milliken Moment Diagram
# ═════════════════════════════════════════════════════════════════════════════
def plot_mmd(
    tire_model, total_mass_kg: float, weight_dist_front: float,
    wheelbase_m: float, cg_height_m: float,
    track_front_m: float, track_rear_m: float,
    roll_stiffness_front_Npm_rad: float = 0.0,
    roll_stiffness_rear_Npm_rad: float = 0.0,
    delta_arb_front_Npm: float = 0.0,
    delta_arb_rear_Npm: float = 0.0,
    toe_front_deg: float = 0.0,
    toe_rear_deg: float = 0.0,
    velocity_mps: float = 13.4,
    beta_range_deg: tuple = (-8, 8), beta_n: int = 9,
    delta_range_deg: tuple = (-20, 20), delta_n: int = 11,
) -> Figure:
    """Pure-cornering MMD: yaw moment N vs lateral acceleration Ay,
    with constant-β and constant-δ lines.

    Roll stiffness (springs + ARB) drives the lateral load transfer
    distribution (LLTD) between front and rear axles.  Static toe angles
    (+ = toe-in) shift the individual corner slip angles.
    """
    g_acc = 9.81
    m = total_mass_kg
    # l_f = distance from front axle to CG;  l_r = CG to rear axle
    l_f = wheelbase_m * (1.0 - weight_dist_front)   # a_d
    l_r = wheelbase_m * weight_dist_front            # b_d
    Fz_F0 = m * g_acc * (l_r / wheelbase_m) / 2     # static front Fz per corner
    Fz_R0 = m * g_acc * (l_f / wheelbase_m) / 2     # static rear  Fz per corner

    # ── Lateral load transfer distribution (LLTD) ───────────────────────────
    # K_roll = (K_wheel + K_arb) * t^2 / 2  (from VehicleParams property)
    # delta_arb_* lets the user add or remove ARB wheel-rate to explore LLTD.
    K_f = max(roll_stiffness_front_Npm_rad
              + delta_arb_front_Npm * track_front_m ** 2 / 2, 0.0)
    K_r = max(roll_stiffness_rear_Npm_rad
              + delta_arb_rear_Npm  * track_rear_m  ** 2 / 2, 0.0)
    K_tot = K_f + K_r
    LLTD_f = K_f / K_tot if K_tot > 1.0 else 0.5   # front fraction; default 50 %

    def compute_state(beta_deg, delta_deg, max_iter=20, tol=1e-4):
        Ay = 0.0
        for _ in range(max_iter):
            # Lateral load transfer — corrected formula (was missing g_acc)
            # Each axle's share of the roll couple is LLTD_f / (1-LLTD_f).
            LT_F = LLTD_f       * m * g_acc * Ay * cg_height_m / track_front_m
            LT_R = (1.0 - LLTD_f) * m * g_acc * Ay * cg_height_m / track_rear_m
            Fz_FL = max(Fz_F0 - LT_F, 10.0); Fz_FR = max(Fz_F0 + LT_F, 10.0)
            Fz_RL = max(Fz_R0 - LT_R, 10.0); Fz_RR = max(Fz_R0 + LT_R, 10.0)

            # Per-corner slip angles: toe-in shifts inner vs outer SA
            # sign(β) tracks which side is inner (inner = lower load).
            t_sgn = 1.0 if beta_deg >= 0 else -1.0
            # front toe-in: inner reduces SA, outer increases SA
            a_FL = beta_deg - delta_deg - toe_front_deg * t_sgn
            a_FR = beta_deg - delta_deg + toe_front_deg * t_sgn
            # rear toe-in: inner reduces SA, outer increases SA
            a_RL = beta_deg - toe_rear_deg * t_sgn
            a_RR = beta_deg + toe_rear_deg * t_sgn

            Fy_FL = -float(tire_model.Fy(a_FL, Fz_FL, 0.0))
            Fy_FR = -float(tire_model.Fy(a_FR, Fz_FR, 0.0))
            Fy_RL = -float(tire_model.Fy(a_RL, Fz_RL, 0.0))
            Fy_RR = -float(tire_model.Fy(a_RR, Fz_RR, 0.0))
            Ay_new = (Fy_FL + Fy_FR + Fy_RL + Fy_RR) / (m * g_acc)
            if abs(Ay_new - Ay) < tol:
                Ay = Ay_new
                break
            Ay = 0.5 * Ay + 0.5 * Ay_new
        N = l_f * (Fy_FL + Fy_FR) - l_r * (Fy_RL + Fy_RR)
        return Ay, N

    # Isoline VALUES stay at the user's chosen spacing (beta_n / delta_n
    # lines), but each line is SAMPLED densely along its sweep axis so the
    # curves are smooth instead of kinked polylines through the coarse grid.
    line_n = 41
    betas  = np.linspace(*beta_range_deg, beta_n)
    deltas = np.linspace(*delta_range_deg, delta_n)
    betas_dense  = np.linspace(*beta_range_deg, line_n)
    deltas_dense = np.linspace(*delta_range_deg, line_n)

    # constant-β lines: sweep δ densely
    Ay_b = np.zeros((beta_n, line_n)); N_b = np.zeros_like(Ay_b)
    for i, b in enumerate(betas):
        for j, d in enumerate(deltas_dense):
            Ay_b[i, j], N_b[i, j] = compute_state(b, d)
    # constant-δ lines: sweep β densely
    Ay_d = np.zeros((line_n, delta_n)); N_d = np.zeros_like(Ay_d)
    for i, b in enumerate(betas_dense):
        for j, d in enumerate(deltas):
            Ay_d[i, j], N_d[i, j] = compute_state(b, d)

    # ── dN/dβ at trim (β=0, δ=0): directional stability slope ──────────────
    # Use the dense constant-δ=0 line for a fine central difference.
    mid_d = delta_n // 2
    mid_b = line_n // 2
    if 0 < mid_b < line_n - 1:
        dN_db = ((N_d[mid_b + 1, mid_d] - N_d[mid_b - 1, mid_d]) /
                 (betas_dense[mid_b + 1] - betas_dense[mid_b - 1]))  # N·m / deg
    else:
        dN_db = float('nan')
    stable = dN_db < 0 if np.isfinite(dN_db) else None

    fig, ax = _styled_fig(figsize=(10, 8))
    # Constant β — solid blue
    # Label only the extreme isolines so tip labels never crowd/overprint.
    b_lab = {betas.min(), betas.max()}
    d_lab = {deltas.min(), deltas.max()}
    for i, b in enumerate(betas):
        lw = 2.2 if b == 0 else 1.2
        alpha = 1.0 if b == 0 else 0.75
        ax.plot(Ay_b[i, :], N_b[i, :], color=_BLUE, lw=lw, alpha=alpha,
                label=f'β={b:.0f}°')
        if b in b_lab:
            ax.text(Ay_b[i, -1] + 0.03, N_b[i, -1], f'β={b:.0f}°',
                    color=_BLUE, fontsize=8, va='center', ha='left')
    # Constant δ — dashed red
    for j, d in enumerate(deltas):
        lw = 2.2 if d == 0 else 1.2
        alpha = 1.0 if d == 0 else 0.75
        ax.plot(Ay_d[:, j], N_d[:, j], color=_RED, lw=lw, alpha=alpha, ls='--',
                label=f'δ={d:.0f}°')
        if d in d_lab:
            ax.text(Ay_d[-1, j], N_d[-1, j] + 60, f'δ={d:.0f}°',
                    color=_RED, fontsize=8, ha='center', va='bottom')

    ax.axhline(0, color=_TICK, lw=0.8)
    ax.axvline(0, color=_TICK, lw=0.8)
    # Trim point marker
    ay_trim, n_trim = compute_state(0.0, 0.0)
    trim_handle, = ax.plot(ay_trim, n_trim, 'o',
            color=_YELLOW, markersize=11, markeredgecolor=_WHITE,
            label='Trim point (β=0, δ=0)')

    ax.set_xlabel('Lateral acceleration  A_y  (g)', fontsize=10)
    ax.set_ylabel('Yaw moment  N  about CG  (N·m)', fontsize=10)
    ax.set_title(
        f'Milliken Moment Diagram — Pure Cornering  |  '
        f'{velocity_mps * 3.6:.0f} km/h  ({velocity_mps / 0.44704:.0f} mph)',
        fontsize=11, color=_TEXT)
    # Only the trim point in the legend — the isolines are labelled for the
    # hover readout, not the legend (20 entries would swamp it).
    ax.legend(handles=[trim_handle], facecolor=_PANEL_BG, labelcolor=_TEXT,
              edgecolor=_GRID, fontsize=9, loc='lower left')

    # ── Legend box (top-left) ───────────────────────────────────────────────
    ax.text(0.02, 0.98,
            'Solid line:   constant β (sweep δ)\n'
            'Dashed line:  constant δ (sweep β)\n'
            'Equilibrium turn → on the A_y axis (N = 0)\n'
            'Max grip   → outermost reach in A_y',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color=_TEXT,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID, pad=4))

    # ── Setup / stability box (top-right) ──────────────────────────────────
    lltd_pct_f = LLTD_f * 100
    if np.isfinite(dN_db):
        stab_str = ('Stable ✓' if stable
                    else 'Unstable ✗' if stable is False else '—')
        slope_str = f'dN/dβ = {dN_db:+.0f} N·m/° → {stab_str}'
    else:
        slope_str = 'dN/dβ = —'
    toe_f_str = f'{toe_front_deg:+.1f}°' if toe_front_deg else '0° (neutral)'
    toe_r_str = f'{toe_rear_deg:+.1f}°' if toe_rear_deg else '0° (neutral)'
    info_lines = (
        f'LLTD:  F {lltd_pct_f:.0f}% / R {100 - lltd_pct_f:.0f}%\n'
        f'Toe F: {toe_f_str}   Toe R: {toe_r_str}\n'
        f'{slope_str}'
    )
    stability_color = (_YELLOW if stable else _RED) if stable is not None else _TEXT
    ax.text(0.98, 0.98, info_lines,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color=stability_color,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID, pad=4))

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 9. Wheel-rate linearity across travel
# ═════════════════════════════════════════════════════════════════════════════
def plot_wheel_rate_linearity(
    solvers: dict,
    k_spring_front_Npm: float,
    k_spring_rear_Npm: float,
    travel_range_mm: tuple = (-50, 50),
    operating_range_mm: tuple = (-25, 25),
    n_points: int = 41,
) -> Figure:
    """Wheel rate K_w(τ) = K_spring · MR(τ)² as a function of wheel travel.

    MR(τ) is computed numerically from each corner's kinematic solver via
    central difference of the spring length about τ.  This captures the
    geometric variation of the motion ratio across the travel range
    (rocker rotation + linkage geometry).

    Parameters
    ----------
    solvers : dict
        {'FL', 'FR', 'RL', 'RR'} → SuspensionConstraints
    k_spring_front_Npm, k_spring_rear_Npm : float
        Coil spring rates (N/m at the spring).
    travel_range_mm : (low, high)
        Wheel travel sweep.  Negative = droop, positive = bump.
    operating_range_mm : (low, high)
        Highlighted "typical operating travel" band.
    target_rate_Npm : float
        Reference horizontal line.
    n_points : int
        Sweep resolution.
    """
    import numpy as _np

    travels_mm = _np.linspace(travel_range_mm[0], travel_range_mm[1], n_points)
    d_tau = 5e-4   # ±0.5 mm finite-difference step for MR

    def _sweep_axle(label, k_spring):
        solver = solvers.get(label)
        if solver is None:
            return _np.full(n_points, _np.nan)
        kw = _np.full(n_points, _np.nan)
        x_warm = None; th_warm = 0.0; sl_warm = None
        for i, t_mm in enumerate(travels_mm):
            t_m = t_mm * 1e-3
            try:
                s_p = solver.solve(t_m + d_tau,
                                   x0=x_warm, rocker_theta0=th_warm,
                                   rocker_spring_prev=sl_warm)
                s_m = solver.solve(t_m - d_tau,
                                   x0=x_warm, rocker_theta0=th_warm,
                                   rocker_spring_prev=sl_warm)
                mr = (s_p.spring_length - s_m.spring_length) / (2.0 * d_tau)
                kw[i] = float(k_spring) * mr * mr
                # Warm start
                x_warm = _np.concatenate([s_p.uca_outer, s_p.lca_outer,
                                          s_p.tr_outer, s_p.wheel_center])
                th_warm = s_p.rocker_angle
                sl_warm = s_p.spring_length
            except Exception:
                continue
        return kw

    kw_F = _sweep_axle('FL', k_spring_front_Npm)
    kw_R = _sweep_axle('RL', k_spring_rear_Npm)

    fig, ax = _styled_fig(figsize=(9.5, 5))

    # Operating-travel band
    ax.axvspan(operating_range_mm[0], operating_range_mm[1],
               color=_YELLOW, alpha=0.10,
               label=f'Typical operating travel (±{operating_range_mm[1]:.0f} mm)')

    # Curves
    ax.plot(travels_mm, kw_F, color=_BLUE, lw=2.4, label='Front')
    ax.plot(travels_mm, kw_R, color=_RED, lw=2.4, ls='--', label='Rear')

    # Design-position dots (τ = 0)
    kw_F0 = float(_np.interp(0.0, travels_mm, kw_F))
    kw_R0 = float(_np.interp(0.0, travels_mm, kw_R))
    ax.plot(0, kw_F0, 'o', color=_BLUE, markersize=7, markeredgecolor=_WHITE, zorder=5)
    ax.plot(0, kw_R0, 'o', color=_RED,  markersize=7, markeredgecolor=_WHITE, zorder=5)

    # Operating-range stats
    op_mask = (travels_mm >= operating_range_mm[0]) & (travels_mm <= operating_range_mm[1])
    def _stats(kw, kw0):
        vals = kw[op_mask]
        vals = vals[~_np.isnan(vals)]
        if len(vals) == 0:
            return 0.0, 0.0, 0.0
        lo, hi = float(vals.min()), float(vals.max())
        pct = (hi - lo) / kw0 * 100.0 if kw0 > 0 else 0.0
        return lo, hi, pct
    fmin, fmax, fpct = _stats(kw_F, kw_F0)
    rmin, rmax, rpct = _stats(kw_R, kw_R0)

    note = (
        f'At design (τ=0):  Front {kw_F0:.0f} N/m   Rear {kw_R0:.0f} N/m\n'
        f'Operating band (±{operating_range_mm[1]:.0f} mm):\n'
        f'  Front {fmin:.0f}–{fmax:.0f} N/m   (range {fpct:.1f}% of design)\n'
        f'  Rear  {rmin:.0f}–{rmax:.0f} N/m   (range {rpct:.1f}% of design)\n'
        f'F/R ratio at design: {kw_F0 / max(kw_R0, 1.0):.2f}'
    )
    ax.text(0.02, 0.98, note,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color=_TEXT,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID, pad=4))

    ax.set_xlabel('Wheel travel (mm)  —  negative = droop, positive = bump',
                  fontsize=10)
    ax.set_ylabel(r'Wheel rate  $K_{wheel}$  (N/m)', fontsize=10)
    ax.set_title(
        f'Wheel-Rate Linearity Across Travel  '
        f'(F spring {k_spring_front_Npm/175.127:.0f} lbf/in, '
        f'R spring {k_spring_rear_Npm/175.127:.0f} lbf/in)',
        fontsize=10, color=_TEXT,
    )
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
              fontsize=9, loc='lower right')
    ax.set_xlim(travel_range_mm[0], travel_range_mm[1])

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 10. Lateral load transfer distribution
# ═════════════════════════════════════════════════════════════════════════════
def plot_lateral_load_transfer(
    solver,                      # SteadyStateSolver
    ay_max: float = 2.5,
    n_points: int = 51,
) -> Figure:
    """Per-corner wheel loads and LLT breakdown vs lateral acceleration.

    Left panel — per-corner Fz:
        Shows how each wheel's vertical load evolves with cornering.
        Inside wheels (FL, RL in a right turn) lose load; outside gain.
        Wheel-lift threshold shown at Fz = 0.

    Right panel — load transfer per axle:
        Total ΔFz at front and rear, stacked by source:
        geometric (through links), elastic (roll stiffness), unsprung mass.
        LLTD % annotation shows front fraction of total LLT.
    """
    sweep = solver.sweep_lateral_g(g_range=(0.0, ay_max), n_points=n_points)
    ay = sweep['lateral_g']
    g_acc = 9.81

    # Static loads (Ay = 0 point)
    Fz0 = {lbl: sweep[f'Fz_{lbl}'][0] for lbl in ('FL', 'FR', 'RL', 'RR')}

    # Per-corner loads (positive = compression)
    Fz_FL = sweep['Fz_FL']
    Fz_FR = sweep['Fz_FR']
    Fz_RL = sweep['Fz_RL']
    Fz_RR = sweep['Fz_RR']

    # LLT components: ΔFz relative to static (N, one-side delta, absolute values)
    geo_F = sweep['geometric_lt_front_N']
    ela_F = sweep['elastic_lt_front_N']
    geo_R = sweep['geometric_lt_rear_N']
    ela_R = sweep['elastic_lt_rear_N']

    # Unsprung LLT = total - geometric - elastic (derived, always ≥ 0)
    total_F = Fz_FR - Fz0['FR']              # outside gains
    total_R = Fz_RR - Fz0['RR']
    uns_F = np.maximum(total_F - geo_F - ela_F, 0.0)
    uns_R = np.maximum(total_R - geo_R - ela_R, 0.0)

    # LLTD at design lateral g (last point) and across sweep
    total_lt = total_F + total_R
    lltd_pct = np.where(total_lt > 1.0, total_F / total_lt * 100.0, 50.0)

    fig, (ax_fz, ax_lt) = _styled_fig(figsize=(12, 5), n=1, m=2)

    # ── Left: per-corner Fz ───────────────────────────────────────────────
    # Right turn: FR/RR = outside (gain), FL/RL = inside (lose)
    ax_fz.plot(ay, Fz_FL / g_acc, color=_YELLOW,  lw=2.0, ls='--',
               label='FL (inner front)')
    ax_fz.plot(ay, Fz_FR / g_acc, color=_YELLOW,  lw=2.0, ls='-',
               label='FR (outer front)')
    ax_fz.plot(ay, Fz_RL / g_acc, color=_BLUE,    lw=2.0, ls='--',
               label='RL (inner rear)')
    ax_fz.plot(ay, Fz_RR / g_acc, color=_BLUE,    lw=2.0, ls='-',
               label='RR (outer rear)')

    # Wheel-lift line
    ax_fz.axhline(0, color=_RED, lw=1.2, ls=':', alpha=0.8)
    ax_fz.text(ay_max * 0.02, 2, 'Wheel lift', color=_RED, fontsize=8)

    # Static load annotations
    ax_fz.axhline(Fz0['FL'] / g_acc, color=_YELLOW, lw=0.5, ls=':', alpha=0.4)
    ax_fz.axhline(Fz0['RL'] / g_acc, color=_BLUE,   lw=0.5, ls=':', alpha=0.4)

    ax_fz.set_xlabel('Lateral acceleration  (g)', fontsize=10)
    ax_fz.set_ylabel('Vertical load  (kg-force)', fontsize=10)
    ax_fz.set_title('Per-Corner Wheel Loads', fontsize=10, color=_TEXT)
    ax_fz.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
                 fontsize=8, loc='upper right')
    ax_fz.set_xlim(0, ay_max)

    # ── Right: LLT breakdown (stacked fill) ──────────────────────────────
    # Front axle — stacked upward
    ax_lt.fill_between(ay, 0,             geo_F,
                       color=_YELLOW, alpha=0.30, label='Front geometric')
    ax_lt.fill_between(ay, geo_F,         geo_F + ela_F,
                       color=_YELLOW, alpha=0.55, label='Front elastic')
    ax_lt.fill_between(ay, geo_F + ela_F, geo_F + ela_F + uns_F,
                       color=_YELLOW, alpha=0.80, label='Front unsprung')

    # Rear axle — stacked downward (negative so it reads as "below")
    ax_lt.fill_between(ay, 0,              -geo_R,
                       color=_BLUE, alpha=0.30, label='Rear geometric')
    ax_lt.fill_between(ay, -geo_R,         -(geo_R + ela_R),
                       color=_BLUE, alpha=0.55, label='Rear elastic')
    ax_lt.fill_between(ay, -(geo_R+ela_R), -(geo_R + ela_R + uns_R),
                       color=_BLUE, alpha=0.80, label='Rear unsprung')

    # LLTD % curve as secondary axis
    ax2 = ax_lt.twinx()
    ax2.plot(ay, lltd_pct, color=_WHITE, lw=1.5, ls='--', alpha=0.7)
    ax2.set_ylabel('LLTD  (front %)', color=_TICK, fontsize=9)
    ax2.tick_params(colors=_TICK, labelsize=8)
    ax2.set_ylim(30, 70)
    ax2.axhline(50, color=_TICK, lw=0.5, ls=':', alpha=0.5)

    ax_lt.axhline(0, color=_TICK, lw=0.6)
    ax_lt.set_xlabel('Lateral acceleration  (g)', fontsize=10)
    ax_lt.set_ylabel('ΔFz  (N)', fontsize=10)
    ax_lt.set_title('Load Transfer Breakdown', fontsize=10, color=_TEXT)
    ax_lt.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
                 fontsize=7.5, loc='lower left', ncol=2)
    ax_lt.set_xlim(0, ay_max)

    # Summary annotation
    lltd_at_1g_idx = int(np.argmin(np.abs(ay - 1.0)))
    lltd_1g = float(lltd_pct[lltd_at_1g_idx])
    veh = solver._veh
    wf_pct = veh.front_weight_fraction * 100
    K_f = veh.roll_stiffness_front_Npm_rad
    K_r = veh.roll_stiffness_rear_Npm_rad
    K_tot = K_f + K_r
    ela_lltd = K_f / K_tot * 100 if K_tot > 0 else 50
    note = (
        f'Weight dist:       {wf_pct:.1f}% front\n'
        f'Elastic LLTD:   {ela_lltd:.1f}% front\n'
        f'LLTD @ 1g:       {lltd_1g:.1f}% front'
    )
    delta_col = _YELLOW if lltd_1g > wf_pct + 1 else (_BLUE if lltd_1g < wf_pct - 1 else _WHITE)
    ax_lt.text(0.98, 0.98, note,
               transform=ax_lt.transAxes, ha='right', va='top',
               fontsize=8.5, color=delta_col,
               bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID, pad=4))

    fig.tight_layout()
    return fig



# =============================================================================
# Friction circle (g-g diagram) -- per-tyre force envelope + observed Fx/Fy
# operating points across cornering / braking / accel cases.
# Source: RCVD ch 9.3 (Vehicle Capability) + ch 14.4 (friction circle).
# =============================================================================
import matplotlib.pyplot as plt


def plot_friction_circle(steady_solver, max_ay_g=1.8, n_pts=21):
    """Friction-circle / g-g diagram.

    Left panel: per-axle (Ay, Ax) envelope = the achievable operating
                region after per-tyre friction-circle clamp + LT.
    Right panel: per-tyre Fx/Fy normalised by mu*Fz, with operating
                 points for canonical g cases (1g cornering, 1g brake,
                 1g accel, 0.7g+0.7g combined).

    Catches: asymmetric envelope (LT or anti-geometry bug);
             operating points outside the circle (clamp not working).
    """
    fig, axes = _styled_fig(figsize=(13, 6), n=1, m=2)
    ax_gg, ax_fc = axes

    veh = steady_solver._veh
    g_acc = 9.81
    W_total = veh.total_mass_kg * g_acc

    # ----- Left: g-g envelope -----
    ay_arr = np.linspace(-max_ay_g, max_ay_g, n_pts)
    ax_max = np.full_like(ay_arr, float("nan"))
    ax_min = np.full_like(ay_arr, float("nan"))
    mu_default = 1.6

    def _mu_at(k, Fz_k):
        """Per-tyre mu from the solver's ACTUAL tire (split-tire aware, camber-0
        row, solver's belt->track scale).  The old lookup read a nonexistent
        `tire_model` attribute (the solver stores `_tire`) and called
        peak_mu with one argument (it takes fz, camber) — both silently fell
        back to mu_default=1.6, so the whole figure ignored the TTC data."""
        try:
            t = steady_solver._tire_for(k)
            return float(t.peak_mu(Fz_k, 0.0)) * getattr(steady_solver, '_mu_scale', 1.0)
        except Exception:
            return mu_default

    def tyre_util_max(result):
        umax = 0.0
        for k in result.Fz:
            Fz_k = float(result.Fz[k])
            if Fz_k <= 0:
                continue
            mu = _mu_at(k, Fz_k)
            Fx_k = float(result.Fx.get(k, 0.0)) if hasattr(result, "Fx") else 0.0
            Fy_k = float(result.Fy.get(k, 0.0)) if hasattr(result, "Fy") else 0.0
            util = (Fx_k**2 + Fy_k**2) / max((mu * Fz_k) ** 2, 1e-6)
            if util > umax:
                umax = util
        return umax

    for i, ay in enumerate(ay_arr):
        for sign in (+1, -1):
            ax_test = 0.0
            ax_last_ok = 0.0
            step = 0.05
            for _ in range(40):
                try:
                    r = steady_solver.solve(lateral_g=ay,
                                            longitudinal_g=ax_test)
                    if tyre_util_max(r) > 1.05:
                        break
                    ax_last_ok = ax_test
                    ax_test += sign * step
                except Exception:
                    break
            if sign > 0:
                ax_max[i] = ax_last_ok
            else:
                ax_min[i] = ax_last_ok

    valid = np.isfinite(ax_max) & np.isfinite(ax_min)
    if valid.any():
        ax_gg.fill_between(ay_arr[valid], ax_min[valid], ax_max[valid],
                            color=_BLUE, alpha=0.18,
                            label="Achievable envelope")
        ax_gg.plot(ay_arr[valid], ax_max[valid], color=_BLUE, lw=1.4,
                    label="Max accel @ ay")
        ax_gg.plot(ay_arr[valid], ax_min[valid], color=_RED, lw=1.4,
                    label="Max brake @ ay")

    theta = np.linspace(0, 2*np.pi, 200)
    ax_gg.plot(mu_default * np.cos(theta), mu_default * np.sin(theta),
                color=_YELLOW, lw=1.0, alpha=0.5, linestyle="--",
                label=f"Reference circle (mu={mu_default})")

    ax_gg.axhline(0, color=_TICK, lw=0.5, alpha=0.5)
    ax_gg.axvline(0, color=_TICK, lw=0.5, alpha=0.5)
    ax_gg.set_aspect("equal", "box")
    ax_gg.set_xlabel("Lateral acceleration  Ay  (g)", fontsize=10)
    ax_gg.set_ylabel("Longitudinal acceleration  Ax  (g, +accel / -brake)",
                      fontsize=10)
    ax_gg.set_title("g-g Diagram (friction-circle envelope)",
                     fontsize=10, color=_TEXT)
    ax_gg.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
                  fontsize=7.5, loc="upper right")
    ax_gg.set_xlim(-max_ay_g*1.1, max_ay_g*1.1)
    ax_gg.set_ylim(-max_ay_g*1.1, max_ay_g*1.1)

    # ----- Right: per-tyre friction circles IN FORCE (newtons) -----
    # Circle radius = mu(Fz)*Fz at that corner's load in the case — the actual
    # grip budget in N.  Operating point = the actual (Fy, Fx) in N.
    #   COLOR  = corner (FL/FR/RL/RR)
    #   MARKER = load case
    corners = ["FL", "FR", "RL", "RR"]
    corner_colors = {"FL": _BLUE, "FR": _TEAL, "RL": _RED, "RR": _YELLOW}
    cases = {
        "1g cornering":    (1.0, 0.0,  "o"),
        "1g braking":      (0.0, -1.0, "s"),
        "1g accel":        (0.0, +1.0, "^"),
        "0.7g+0.7g":       (0.7, +0.7, "D"),
    }
    r_max = 1.0
    # Draw the grip-budget circles at the 1g-cornering loads (one per corner —
    # the most-referenced case) so the plot shows real newton radii.
    try:
        r_ref = steady_solver.solve(lateral_g=1.0, longitudinal_g=0.0)
        for k in corners:
            Fz_k = float(r_ref.Fz.get(k, 0))
            if Fz_k <= 1:
                continue
            cap = _mu_at(k, Fz_k) * Fz_k
            r_max = max(r_max, cap)
            circle = plt.Circle((0, 0), cap, fill=False,
                                color=corner_colors[k], lw=1.2, alpha=0.55)
            ax_fc.add_patch(circle)
    except Exception:
        pass

    for case_name, (ay, ax_long, mk) in cases.items():
        try:
            r = steady_solver.solve(lateral_g=ay, longitudinal_g=ax_long)
            for k in corners:
                Fz_k = float(r.Fz.get(k, 0))
                if Fz_k <= 1:
                    continue
                Fy_k = float(r.Fy.get(k, 0)) if hasattr(r, "Fy") else 0
                Fx_k = float(r.Fx.get(k, 0)) if hasattr(r, "Fx") else 0
                r_max = max(r_max, abs(Fy_k), abs(Fx_k))
                ax_fc.scatter(Fy_k, Fx_k, marker=mk,
                                color=corner_colors[k], s=60, alpha=0.75,
                                edgecolors=_WHITE, linewidths=0.5)
        except Exception:
            continue

    # Composite legend: corner colors + case markers (grey proxies)
    from matplotlib.lines import Line2D
    proxies  = [Line2D([0], [0], marker='o', ls='', color=corner_colors[k],
                       label=k) for k in corners]
    proxies += [Line2D([0], [0], marker=mk, ls='', color=_TICK,
                       label=case_name)
                for case_name, (_, _, mk) in cases.items()]
    ax_fc.legend(handles=proxies, facecolor=_PANEL_BG, labelcolor=_TEXT,
                  edgecolor=_GRID, fontsize=7, loc="lower right", ncols=2)

    ax_fc.set_xlabel("Lateral force Fy (N)", fontsize=10)
    ax_fc.set_ylabel("Longitudinal force Fx (N)", fontsize=10)
    ax_fc.set_title("Per-Tyre Friction Circles (N) — circle = grip budget "
                    "mu(Fz)·Fz at 1 g", fontsize=10, color=_TEXT)
    ax_fc.axhline(0, color=_TICK, lw=0.5, alpha=0.5)
    ax_fc.axvline(0, color=_TICK, lw=0.5, alpha=0.5)
    ax_fc.set_aspect("equal", "box")
    lim = r_max * 1.15
    ax_fc.set_xlim(-lim, lim)
    ax_fc.set_ylim(-lim, lim)

    fig.tight_layout()
    return fig
