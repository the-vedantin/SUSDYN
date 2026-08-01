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
import math
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
    # Below the axes so the note can never occlude the RC curves.
    ax.text(0.5, -0.16,
            f'At {op_max:.2f}° body roll:  front RC height change '
            f'{rc_f_at_max - rc_f[0]:+.2f} mm,  rear {rc_r_at_max - rc_r[0]:+.2f} mm',
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color=_TEXT,
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_GRID))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. Steering torque vs steer angle
# ═════════════════════════════════════════════════════════════════════════════
def plot_steering_torque(effort: dict) -> Figure:
    """Steering effort vs lateral g from vahan.steering.compute_steering_effort
    — the ONE implementation shared with the binder (probed linkage + per-wheel
    solved loads + TTC aligning torque + mechanical trail).  Replaces the old
    constant-speed estimate, which used static Fz and the wrong lever (scrub)
    for the side-force term."""
    fig, ax = _styled_fig(figsize=(9.5, 5.2))
    have_T = 'T_vs_latg' in effort
    if have_T:
        xs = [g for g, _ in effort['T_vs_latg']]
        ys = [t for _, t in effort['T_vs_latg']]
        ax.set_ylabel('Steering-wheel torque (N·m)', fontsize=10)
    else:
        xs = [g for g, _ in effort['F_rack_vs_latg']]
        ys = [f for _, f in effort['F_rack_vs_latg']]
        ax.set_ylabel('Rack force (N)  [no pinion mm/rev in config]', fontsize=10)
    ax.plot(xs, ys, '-o', color=_YELLOW, lw=2.2, ms=5, mfc=_RED, mec=_RED,
            label='steering effort (solved per-wheel loads)')
    ipk = int(np.argmax(ys))
    ax.annotate(f'peak {ys[ipk]:.1f} at {xs[ipk]:.1f} g\n(before the grip limit — '
                'pneumatic trail collapses)',
                xy=(xs[ipk], ys[ipk]), xytext=(xs[0] + 0.1, ys[ipk] * 0.86),
                arrowprops=dict(arrowstyle='->', color=_TEXT, lw=1.1),
                fontsize=9, color=_TEXT)
    ax.set_xlabel('Lateral acceleration (g)', fontsize=10)
    bits = [f'effective arm {effort.get("arm_eff_mm", float("nan")):.1f} mm']
    if effort.get('arm_physical_mm') is not None:
        bits.append(f'physical arm (KP→tie-rod outer) {effort["arm_physical_mm"]:.1f} mm')
    if effort.get('ratio_to1') is not None:
        bits.append(f'ratio {effort["ratio_to1"]:.2f}:1 '
                    f'({effort.get("C_mm_per_rev", 0):.1f} mm/rev)')
    if effort.get('lock_to_lock_deg') is not None:
        bits.append(f'lock-to-lock {effort["lock_to_lock_deg"]:.0f}°')
    if effort.get('tie_rod_force_max_N') is not None:
        bits.append(f'tie-rod {effort["tie_rod_force_max_N"]:.0f} N max')
    half = (len(bits) + 1) // 2
    ax.set_title('Steering effort vs lateral acceleration\n'
                 + '  ·  '.join(bits[:half]) + '\n' + '  ·  '.join(bits[half:]),
                 fontsize=9, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID, fontsize=9,
              loc='lower center')
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
def _peak_sa(sa, fy):
    """Slip angle at peak Fy, interpolated BETWEEN grid points.

    A bare argmax snaps the peak to the sweep grid (0.203 deg at 70 points over
    0-14 deg).  The Ackermann demand curve is the DIFFERENCE of two such peaks,
    and the whole effect being measured is 0.02-0.27 deg — smaller than one grid
    step.  Quantised that way the difference collapses to 0 or +/-1 step and
    flips sign on rounding, which is exactly what made the demand curve look
    like meaningless noise.  Fit a parabola through the peak and its neighbours
    so the answer is continuous in load instead.
    """
    i = int(np.argmax(fy))
    if i <= 0 or i >= len(sa) - 1:
        return float(sa[i])
    y0, y1, y2 = float(fy[i - 1]), float(fy[i]), float(fy[i + 1])
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-12:
        return float(sa[i])
    # vertex offset in grid units, clamped to the bracketing interval
    off = 0.5 * (y0 - y2) / den
    off = max(-1.0, min(1.0, off))
    return float(sa[i] + off * (sa[i + 1] - sa[i]))


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

        # Peak slip angle from the TIRE MODEL's fitted value, not a local
        # argmax over this sweep.  These curves are flat near the peak, so an
        # argmax reads noise: on the R20 data it scatters with R2 = 0.07 and
        # slopes the WRONG WAY, which made this demand curve point backwards.
        # The model's fitted peak gives R2 = 0.83, rising with load.
        Fy_i = np.array([abs(float(tire_model.Fy(s, fz_i, 0.0))) for s in sa_sweep])
        Fy_o = np.array([abs(float(tire_model.Fy(s, fz_o, 0.0))) for s in sa_sweep])
        sa_pk_in.append(_peak_sa(sa_sweep, Fy_i))
        sa_pk_out.append(_peak_sa(sa_sweep, Fy_o))

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

        def _axle_fy(a, h):
            """Signed total axle Fy at average slip a with half-split h.

            SIGNED, and NOT floored at +0.01.  A large Ackermann split can drive
            one wheel past straight-ahead into negative slip, where it pulls
            AGAINST the turn.  Clamping that wheel to +0.01 deg made a wrong-way
            wheel free, so the solve below could satisfy the force balance with a
            split that is physically a penalty.  Same clamp class as the tyre
            model's flat extrapolation — see the module note there.
            """
            si, so = a + h, a - h
            fi = abs(float(tire_model.Fy(si, fz_i, 0.0))) * (1.0 if si >= 0 else -1.0)
            fo = abs(float(tire_model.Fy(so, fz_o, 0.0))) * (1.0 if so >= 0 else -1.0)
            return fi + fo

        def _solve_avg(h):
            """Find avg SA where inner+outer Fy = Fy_front, given half-split h."""
            try:
                return brentq(lambda a: _axle_fy(a, h) - Fy_front, 0.01, 14.0)
            except ValueError:
                # Tires saturated — find SA that maximises total Fy
                best_a, best_fy = 0.0, -np.inf
                for a in np.linspace(0.5, 14, 50):
                    tot = _axle_fy(a, h)
                    if tot > best_fy:
                        best_a, best_fy = a, tot
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
# 7. Ackermann Fz–Fy operating map — demand vs delivered, per front wheel
# ═════════════════════════════════════════════════════════════════════════════
def ackermann_fz_fy_data(
    tire_model, solver, radius_m: float = 8.0,
    ackermann_pct: float = 45.0, grip_multiplier: float = 1.0,
    lat_g_list=None,
) -> dict:
    """THE COMPUTE HALF of the Fz-Fy map — arrays only, no plotting.

    Split out of ``plot_ackermann_fz_fy`` (which now calls it) so a second
    consumer — the Ackermann page — can draw the same numbers on its own
    canvas in its own palette WITHOUT owning a second copy of the physics.
    Duplicating this loop is exactly the mechanism that produced rival
    Ackermann answers before; there is one loop and it lives here.

    Returns, all as numpy arrays indexed by the swept lateral g:
        lat_g            lateral g of each row (ROAD g)
        fz_inner/outer   per-wheel vertical load from solver.solve (N)
        cap_inner/outer  MOST that wheel can make at its own load (N)
        del_inner/outer  what it DELIVERS at ackermann_pct (N), length
                         n_delivered — a PREFIX of lat_g, stopped at the
                         grip limit rather than fabricated past it
        n_delivered      how many rows the delivered paths cover
        g_limit          first g where the axle demand is unreachable at any
                         steer angle, or None
        spread_deg       the 100%-Ackermann kinematic toe split at this radius
        split_deg        the as-built toe split (spread x pct/100)
    """
    veh = solver._veh
    L = float(veh.wheelbase_m)
    t = float(veh.front_track_m)
    R = float(radius_m)

    # ── Kinematic steer at this radius, ZERO-CRAB approximation ─────────────
    # Chassis slip angle taken as 0: each wheel's zero-slip heading is the
    # tangent of its own circle about a common centre on the rear-axle line.
    # Good enough HERE because both wheels sit at the SAME radius — a common
    # crab angle shifts both tangents together and largely cancels out of the
    # inner-outer split that Ackermann acts on.
    th_in = math.degrees(math.atan(L / max(R - t / 2.0, 0.05)))
    th_out = math.degrees(math.atan(L / (R + t / 2.0)))
    spread = th_in - th_out                      # 100%-Ackermann toe split
    split = spread * float(ackermann_pct) / 100.0  # as-built toe split
    th_mean = 0.5 * (th_in + th_out)

    # 0.1-g steps put the 0.5 / 1.0 / 1.5 g labels exactly on grid points
    # (17 points, inside the 25-point speed budget).
    if lat_g_list is not None:
        lat_g_arr = np.asarray(list(lat_g_list), float)[:25]
    else:
        lat_g_arr = np.linspace(0.2, 1.8, 17)

    def _pair(d_mean, fz_i, fz_o):
        """Signed per-wheel force toward the corner at mean steer *d_mean*.

        Slip = the wheel's OWN steer minus its OWN tangent direction.  The
        tyre data is SAE (+slip → −Fy), so negate once: positive = pulls into
        the corner.  A wheel left short of its tangent (under-Ackermann'd
        inner at low g) goes to NEGATIVE slip and comes back negative — it
        plots below zero instead of being clamped free, same lesson as the
        pair analysis above.
        """
        s_i = (d_mean + split / 2.0) - th_in
        s_o = (d_mean - split / 2.0) - th_out
        f_i = -float(tire_model.Fy(s_i, fz_i, 0.0)) * grip_multiplier
        f_o = -float(tire_model.Fy(s_o, fz_o, 0.0)) * grip_multiplier
        return f_i, f_o

    # ── Sweep lateral g: demand always, delivered until the grip limit ──────
    g_ok, fz_ip, fz_op = [], [], []
    fy_dem_i, fy_dem_o = [], []
    n_del = 0                       # delivered points kept (prefix of g_ok)
    fy_del_i, fy_del_o = [], []
    g_limit = None                  # first g where demand is unreachable

    scan = np.linspace(th_mean - 2.0, th_mean + 16.0, 25)  # mean-steer grid
    for ag in lat_g_arr:
        try:
            res = solver.solve(float(ag), 0.0)
        except Exception:
            continue                # solver refused (e.g. wheel lift) — skip
        fz_fl, fz_fr = float(res.Fz['FL']), float(res.Fz['FR'])
        # Inner front = the LIGHTER of FL/FR: the solver decides which side
        # unloads, this plot assumes no turn direction of its own.
        fz_i, fz_o = min(fz_fl, fz_fr), max(fz_fl, fz_fr)

        g_ok.append(float(ag))
        fz_ip.append(fz_i);  fz_op.append(fz_o)
        # PER-WHEEL CAPABILITY, not a per-wheel "demand".
        # This used to plot fz*ag per wheel and call it demand.  That is the
        # EQUAL-UTILISATION assumption (same mu on both wheels) — and 100%
        # Ackermann is the EQUAL-SLIP geometry, which on this tyre lands in
        # almost the same place.  So the plot was comparing an assumption
        # against its own twin and 100% "won" by construction, every time.
        # The circular reference is gone: each wheel is now drawn against the
        # MOST IT CAN ACTUALLY MAKE at its own load, which is a measured
        # property of the tyre and owes nothing to any Ackermann setting.  The
        # axle requirement (which IS real: the front pair must total the
        # front's share of m*a) is annotated instead of split by fiat.
        fy_dem_i.append(abs(float(tire_model.peak_Fy(max(fz_i, 1.0), 0.0)))
                        * grip_multiplier)
        fy_dem_o.append(abs(float(tire_model.peak_Fy(max(fz_o, 1.0), 0.0)))
                        * grip_multiplier)

        if g_limit is not None:
            continue                        # delivered path already stopped

        demand_axle = (fz_i + fz_o) * ag
        totals = np.array([sum(_pair(d, fz_i, fz_o)) for d in scan])
        k_pk = int(np.argmax(totals))
        if float(totals[k_pk]) < demand_axle:
            g_limit = float(ag)             # unreachable at ANY steer — stop
            continue
        # Bracket the crossing on the RISING side only (up to the peak):
        # past the peak the same total repeats at silly steer angles, and the
        # low-steer solution is the one the driver actually reaches first.
        k = 1
        while k <= k_pk and totals[k] < demand_axle:
            k += 1
        lo, hi = float(scan[k - 1]), float(scan[min(k, k_pk)])
        for _ in range(40):                 # ≤ 40 bisection steps (budget)
            mid = 0.5 * (lo + hi)
            if sum(_pair(mid, fz_i, fz_o)) < demand_axle:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        f_i, f_o = _pair(0.5 * (lo + hi), fz_i, fz_o)
        fy_del_i.append(f_i);  fy_del_o.append(f_o)
        n_del += 1

    return {
        'lat_g': np.array(g_ok),
        'fz_inner': np.array(fz_ip), 'fz_outer': np.array(fz_op),
        'cap_inner': np.array(fy_dem_i), 'cap_outer': np.array(fy_dem_o),
        'del_inner': np.array(fy_del_i), 'del_outer': np.array(fy_del_o),
        'n_delivered': int(n_del), 'g_limit': g_limit,
        'radius_m': R, 'ackermann_pct': float(ackermann_pct),
        'grip_multiplier': float(grip_multiplier),
        'spread_deg': float(spread), 'split_deg': float(split),
    }


def plot_ackermann_fz_fy(
    tire_model, solver, radius_m: float = 8.0,
    ackermann_pct: float = 45.0, grip_multiplier: float = 1.0,
    lat_g_list=None,
) -> Figure:
    """Fz–Fy operating map: DEMAND vs DELIVERED per front wheel vs lateral g.

    x = vertical load Fz, y = SIGNED lateral force toward the corner, on the
    usual tyre-space background (gray iso-slip lines + peak-Fy envelope).

      DEMAND (solid) — each wheel corners ITS OWN vertical load:
          Fy = Fz · a_y.   Fz is already in newtons and a_y is a dimensionless
          ratio of accelerations (a/g), so there is NO extra 9.81 here — the
          gravity constants cancel:  m_corner·a = (Fz/g)·(a_y·g) = Fz·a_y.
      DELIVERED (dashed) — what the tyres actually give at the as-built
          Ackermann %, once the MEAN steer angle is bisected so the axle
          total matches the axle demand.  Where no steer angle can reach the
          demand, the delivered path STOPS and the last point gets an 'x' —
          no fabricated points past the grip limit.

    Per-corner loads come from ``solver.solve(lat_g, 0).Fz`` — the ONE solved
    model.  The previous version carried its own rigid-car load-transfer
    formula (m·a·h/t); its own comment admitted it was a SECOND model, which
    is exactly what the ONE-MODEL rule forbids, so it is gone.

    The numbers come from ``ackermann_fz_fy_data`` — this function is the
    DRAW half only, so the Ackermann page can plot the identical arrays in
    its own palette without a second copy of the physics.
    """
    R = float(radius_m)
    _d = ackermann_fz_fy_data(tire_model, solver, radius_m=R,
                              ackermann_pct=ackermann_pct,
                              grip_multiplier=grip_multiplier,
                              lat_g_list=lat_g_list)
    g_ok = _d['lat_g']
    fz_ip, fz_op = _d['fz_inner'], _d['fz_outer']
    fy_dem_i, fy_dem_o = _d['cap_inner'], _d['cap_outer']
    fy_del_i, fy_del_o = _d['del_inner'], _d['del_outer']
    n_del, g_limit = _d['n_delivered'], _d['g_limit']

    fig, ax = _styled_fig(figsize=(10, 7))

    # ── Iso-slip-angle lines (tire capability background) ───────────────────
    # Scaled by grip_multiplier so the background matches the delivered
    # curves — an unscaled envelope would show delivered points floating
    # impossibly above the very capability lines they came from.
    fz_hi = max(1500.0, float(fz_op.max()) * 1.10) if len(fz_op) else 1500.0
    fz_range = np.linspace(50, fz_hi, 80)
    sa_levels = [2, 4, 6, 8, 10, 12]
    # Dodge the edge labels: near the top the iso-lines converge (saturated
    # tire), so stacked labels would overprint.  Skip a label if it would sit
    # within 3% of the previous one's height.
    _placed_y = []
    for sa in sa_levels:
        fy_line = np.array([abs(float(tire_model.Fy(sa, fz, 0.0)))
                            for fz in fz_range]) * grip_multiplier
        ax.plot(fz_range, fy_line, color=_TICK, lw=0.8, alpha=0.35)
        y_end = float(fy_line[-1])
        span = max(abs(float(np.nanmax(fy_line))), 1.0)
        if all(abs(y_end - yp) > 0.03 * span for yp in _placed_y):
            ax.text(fz_range[-1] + 15, y_end, f'α={sa}°',
                    fontsize=7, color=_TICK, va='center', alpha=0.6)
            _placed_y.append(y_end)

    peak_fy = np.array([abs(float(tire_model.peak_Fy(fz, 0.0)))
                        for fz in fz_range]) * grip_multiplier
    ax.plot(fz_range, peak_fy, color=_YELLOW, lw=1.5, ls='--', alpha=0.6,
            label=f'Peak Fy envelope (×{grip_multiplier:.2f} grip)')

    # ── Demand paths (solid) ────────────────────────────────────────────────
    ax.plot(fz_op, fy_dem_o, color=_WHITE, lw=2.4,
            label='Outer CAPABILITY (most it can make)')
    ax.plot(fz_ip, fy_dem_i, color=_YELLOW, lw=2.4,
            label='Inner CAPABILITY (most it can make)')

    # ── Delivered paths (dashed), stopped at the grip limit ─────────────────
    if n_del > 0:
        ax.plot(fz_op[:n_del], fy_del_o, color=_RED, lw=2.0, ls='--',
                label=f'Outer delivered at {ackermann_pct:.0f}% Ackermann')
        ax.plot(fz_ip[:n_del], fy_del_i, color=_BLUE, lw=2.0, ls='--',
                label=f'Inner delivered at {ackermann_pct:.0f}% Ackermann')
        if g_limit is not None:
            g_last = float(g_ok[n_del - 1])  # last g the axle still reached
            ax.plot(fz_op[n_del - 1], fy_del_o[-1], 'x', color=_RED,
                    markersize=10, markeredgewidth=2.2, zorder=6)
            ax.plot(fz_ip[n_del - 1], fy_del_i[-1], 'x', color=_BLUE,
                    markersize=10, markeredgewidth=2.2, zorder=6)
            # The ✕ itself says WHERE on the g sweep it sits, per wheel —
            # not only the summary box.  Dropped a full text-height clear of
            # the ✕: at the limit the delivered point rejoins the demand
            # curve, so the demand curve's own g tag lives right next door.
            ax.annotate(f'limit {g_last:.2f} g',
                        xy=(float(fz_op[n_del - 1]), float(fy_del_o[-1])),
                        xytext=(12, -24), textcoords='offset points',
                        ha='left', va='top', fontsize=7, color=_RED)
            ax.annotate(f'limit {g_last:.2f} g',
                        xy=(float(fz_ip[n_del - 1]), float(fy_del_i[-1])),
                        xytext=(-9, -22), textcoords='offset points',
                        ha='right', va='top', fontsize=7, color=_BLUE)
            ax.text(fz_op[n_del - 1], fy_del_o[-1] * 1.04,
                    f'grip limit — axle demand\nunreachable from {g_limit:.2f} g',
                    fontsize=8, color=_RED, ha='right', va='bottom')

    # ── g-sweep markers: every curve is a PATH over lateral g ───────────────
    # The x-axis is Fz, but the swept variable is LATERAL g — without dots the
    # four curves read as four functions of load.  So: major dots + tiny
    # 'X.X g' tags at 0.5 / 1.0 / 1.5 g on ALL FOUR curves, plus small
    # unlabelled dots every 0.25 g so the direction of travel reads anywhere
    # on a path.  Tag dodge: inner-cluster tags hang LEFT of their points,
    # outer-cluster tags sit RIGHT; within a cluster the demand and delivered
    # points share the same Fz at a given g (same solved load), so their only
    # possible collision is vertical — the upper point of the pair tags
    # upward, the lower downward, and they can never pile up.
    clusters = {
        'outer': [(fz_op, fy_dem_o, g_ok, _WHITE, False)],
        'inner': [(fz_ip, fy_dem_i, g_ok, _YELLOW, False)],
    }
    if n_del > 1:
        clusters['outer'].append(
            (fz_op[:n_del], fy_del_o, g_ok[:n_del], _RED, True))
        clusters['inner'].append(
            (fz_ip[:n_del], fy_del_i, g_ok[:n_del], _BLUE, True))
    xmark = {}                      # ✕ position per cluster, for tag dodging
    if n_del > 0 and g_limit is not None:
        xmark['outer'] = (float(fz_op[n_del - 1]), float(fy_del_o[-1]))
        xmark['inner'] = (float(fz_ip[n_del - 1]), float(fy_del_i[-1]))
    xr = float(np.diff(ax.get_xlim())[0])
    yr = float(np.diff(ax.get_ylim())[0])
    for cluster, members in clusters.items():
        sx = 9 if cluster == 'outer' else -9        # tag side, offset points
        ha = 'left' if cluster == 'outer' else 'right'
        for x_a, y_a, g_a, col, _delv in members:
            if len(g_a) < 2:
                continue
            # minor dots every 0.25 g, interpolated onto the path
            for gm in np.arange(0.25, float(g_a[-1]) + 1e-9, 0.25):
                if gm < float(g_a[0]) or any(abs(gm - m) < 1e-9
                                             for m in (0.5, 1.0, 1.5)):
                    continue        # majors get their own bigger dot below
                ax.plot(float(np.interp(gm, g_a, x_a)),
                        float(np.interp(gm, g_a, y_a)), 'o', color=col,
                        markersize=2.2, markeredgewidth=0, zorder=4)
        for g_mark in (0.5, 1.0, 1.5):
            pts = []
            for x_a, y_a, g_a, col, delv in members:
                if len(g_a) == 0:
                    continue
                idx = int(np.argmin(np.abs(g_a - g_mark)))
                if abs(float(g_a[idx]) - g_mark) > 0.051:
                    continue        # this curve's sweep never hit this g
                near_x = (delv and g_limit is not None
                          and idx >= len(g_a) - 2)
                if delv and g_limit is not None and idx == len(g_a) - 1:
                    continue        # that point IS the ✕ — already g-tagged
                ax.plot(float(x_a[idx]), float(y_a[idx]), 'o', color=col,
                        markersize=4, markeredgecolor=_DARK_BG, zorder=5)
                if near_x:
                    continue        # dot only: one step from the ✕ a text
                                    # tag piles onto 'limit X.XX g'
                pts.append((float(x_a[idx]), float(y_a[idx]), col))
            pts.sort(key=lambda p: p[1])            # bottom → top
            for rank, (x_p, y_p, col) in enumerate(pts):
                if len(pts) == 2:                   # pair: point away from twin
                    up = (rank == 1)
                else:                               # alone: cluster default…
                    up = (cluster == 'outer')
                    # …unless the ✕ sits practically ON this point (delivered
                    # rejoins demand at the limit) — then tag away from it
                    if cluster in xmark:
                        mx, my = xmark[cluster]
                        if (abs(x_p - mx) / xr < 0.025
                                and abs(y_p - my) / yr < 0.03):
                            up = (y_p >= my)
                ax.annotate(f'{g_mark:.1f} g', xy=(x_p, y_p),
                            xytext=(sx, 5 if up else -5),
                            textcoords='offset points', ha=ha,
                            va='bottom' if up else 'top',
                            fontsize=7, color=col, zorder=7)

    # ── Direction of travel: one arrow per cluster, along its demand curve ──
    # Outer GAINS load as g rises, so its path walks up-right; inner SHEDS
    # load, so its path walks left.  Drawn just above each curve between the
    # 1.0 and 1.5 g tags, offset enough to share no space with them.
    if len(g_ok) >= 6:
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        i0 = int(np.argmin(np.abs(g_ok - 1.05)))
        i1 = int(np.argmin(np.abs(g_ok - 1.45)))
        if i0 != i1:
            arrows = (
                # outer: sideways offset — the white curve is steep here, so
                # a vertical offset would land the arrow on the red delivered
                # path; the space just RIGHT of the white curve is empty
                (fz_op, fy_dem_o, _WHITE, 0.07 * (x_hi - x_lo), 0.0),
                # inner: the curve is nearly flat, float the arrow above it
                (fz_ip, fy_dem_i, _YELLOW, 0.0, 0.06 * (y_hi - y_lo)),
            )
            for x_a, y_a, col, dx, dy in arrows:
                tail = (float(x_a[i0]) + dx, float(y_a[i0]) + dy)
                head = (float(x_a[i1]) + dx, float(y_a[i1]) + dy)
                top = max(tail[1], head[1])
                if top > y_hi:                      # keep the arrow on-axes
                    y_hi = top + 0.04 * (y_hi - y_lo)
                    ax.set_ylim(y_lo, y_hi)
                ax.annotate('g increases', xy=head, xytext=tail,
                            fontsize=7, color=col, va='center', ha='left',
                            annotation_clip=False,
                            arrowprops=dict(arrowstyle='->', color=col,
                                            lw=1.1))

    # Signed axis: a wheel delivering wrong-way force sits BELOW this line.
    ax.axhline(0, color=_TICK, lw=0.8)

    ax.set_xlabel('Vertical load Fz (N)  —  from the solved car, per corner',
                  fontsize=10, color=_TICK)
    ax.set_ylabel('Lateral force toward the corner (N, signed)',
                  fontsize=10, color=_TICK)
    ax.set_title(f'Fz–Fy Map at R = {R:.1f} m — {ackermann_pct:.0f}% '
                 f'Ackermann, grip ×{grip_multiplier:.2f}\n'
                 f'solid = MOST that wheel can make at its own load  ·  '
                 f'dashed = what it delivers  ·  '
                 f'swept over lateral g  ·  below zero = fights the turn',
                 fontsize=10, color=_TEXT)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
              fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7b. Steady-State Pair Analysis  (RCVD Chapter 7, Figures 7.2 / 7.3)
# ═════════════════════════════════════════════════════════════════════════════
def ackermann_pair_potential(
    tire_model, solver, lat_g: float = 1.5,
    ackermann_list=(-100.0, -50.0, 0.0, 50.0, 100.0),
    steer_max_deg: float = 18.0, n_steer: int = 91,
    front: bool = True,
):
    """Milliken's steady-state PAIR ANALYSIS for one axle, swept over Ackermann.

    THE METHOD IS NOT MINE — it is RCVD Chapter 7 (printed p.279-291).  Milliken:
    *"In a pair analysis, steady-state lateral force performance is calculated
    for a pair of tires on a single track/axle, using data measured on a single
    tire"*, and it exists precisely for our question: *"the objective is usually
    one of increasing the lateral force capability of either the front or rear
    track"* (p.280).  The procedure, p.285: fix the wheel loads from the load
    transfer, choose a series of REFERENCE STEER angles (the average of the two
    wheels, since *"no absolute steer reference exists for a single track"*),
    let the two wheels differ by toe and Ackermann, and enter the tyre data with
    each wheel's OWN angle.  The output is the "lateral force potential diagram",
    RCVD Fig. 7.2/7.3 — cornering coefficient against reference steer angle, one
    curve per design variable.  Fig. 7.3 runs exactly this format for camber,
    roll-rate distribution, roll-centre height, toe and CG height; Ackermann is
    the same kind of test.

    WHY THIS AND NOT THE Fz-Fy MAP.  On the Fz-Fy map the x-axis is vertical
    load, which does not depend on Ackermann at all, and the whole Ackermann
    knob is the small slip-angle split it produces at the ONE steer angle that
    corner implies.  At the map's default 48 km/h the entire kinematic spread is
    0.30 deg, so -50% and -60% differ by 0.015 deg of slip and draw the same
    picture.  Ackermann authority grows as steer SQUARED, so the honest
    independent variable is STEER ANGLE — sweep it and the settings separate.

    DIRECTION IS ACCOUNTED FOR, TWICE OVER:
      * each wheel's force is taken SIGNED from the tyre model, so a wheel
        dragged to negative slip returns a negative number and subtracts
        (never abs(Fy)*sign(slip) — those disagree wherever the curve's own
        zero crossing is not exactly at zero slip);
      * the force acts perpendicular to ITS OWN WHEEL, and Ackermann points the
        two wheels differently, so each is resolved onto the car axes:
            useful  = Fy * cos(delta)     pulls the car round
            scrub   = Fy * sin(delta)     pure drag, slows the car
        Adding raw magnitudes credits a steered wheel with force it never
        delivers.

    Loads come from the SOLVED CAR (`solver.solve(lat_g).Fz`), not a private
    load-transfer formula — RCVD p.284 says to do exactly this when analysing a
    given vehicle rather than a bare wheel pair.

    Returns a dict (no plotting) so the GUI, the binder and the regression net
    all read the same numbers.
    """
    veh = solver._veh
    L = float(veh.wheelbase_m)
    t = float(veh.front_track_m if front else veh.rear_track_m)
    res = solver.solve(float(lat_g), 0.0)
    ka, kb = ('FL', 'FR') if front else ('RL', 'RR')
    fz_a, fz_b = float(res.Fz[ka]), float(res.Fz[kb])
    fz_out, fz_in = max(fz_a, fz_b), max(min(fz_a, fz_b), 0.0)
    w_axle = fz_in + fz_out
    # RCVD Eq. 7.1: fraction of the axle load moved from inner to outer
    flt = (fz_out - fz_in) / max(w_axle, 1e-9)

    steer = np.linspace(0.0, float(steer_max_deg), int(n_steer))
    out = {'steer_deg': steer, 'lat_g': float(lat_g), 'flt': flt,
           'fz_in': fz_in, 'fz_out': fz_out, 'w_axle': w_axle, 'curves': {}}
    for ack in ackermann_list:
        d_rad = np.radians(steer)
        # kinematic 100% Ackermann toe-out between the wheels grows as steer^2
        spread = np.degrees((t / L) * d_rad ** 2) * (float(ack) / 100.0)
        d_in = steer + spread / 2.0          # inner turns MORE when ack > 0
        d_out = steer - spread / 2.0
        fy_in = np.array([-float(tire_model.Fy(a, fz_in, 0.0)) for a in d_in])
        fy_out = np.array([-float(tire_model.Fy(a, fz_out, 0.0)) for a in d_out])
        useful = (fy_in * np.cos(np.radians(d_in))
                  + fy_out * np.cos(np.radians(d_out)))
        scrub = (fy_in * np.sin(np.radians(d_in))
                 + fy_out * np.sin(np.radians(d_out)))
        k = int(np.argmax(useful))
        out['curves'][float(ack)] = {
            'useful_N': useful, 'scrub_N': scrub,
            'cornering_coeff': useful / max(w_axle, 1e-9),
            'inner_N': fy_in * np.cos(np.radians(d_in)),
            'outer_N': fy_out * np.cos(np.radians(d_out)),
            'peak_N': float(useful[k]), 'peak_steer_deg': float(steer[k]),
            'scrub_at_peak_N': float(scrub[k]),
        }
    return out


def plot_ackermann_pair_potential(
    tire_model, solver, lat_g: float = 1.5,
    ackermann_pct: float = 33.0,
    ackermann_list=(-100.0, -50.0, 0.0, 50.0, 100.0),
    steer_max_deg: float = 18.0,
) -> Figure:
    """RCVD Fig. 7.2/7.3 lateral-force potential diagram, family = Ackermann."""
    d = ackermann_pair_potential(
        tire_model, solver, lat_g=lat_g, steer_max_deg=steer_max_deg,
        ackermann_list=tuple(sorted(set(list(ackermann_list)
                                        + [float(ackermann_pct)]))))
    steer = d['steer_deg']
    fig, (ax1, ax2) = _styled_fig(figsize=(12.5, 6.2), n=1, m=2)
    styles = [(_WHITE, '-'), (_YELLOW, '--'), (_RED, '-.'),
              (_BLUE, ':'), (_TICK, '-'), (_YELLOW, ':')]
    for (ack, c), (col, ls) in zip(sorted(d['curves'].items()), styles):
        lw = 2.6 if abs(ack - ackermann_pct) < 0.51 else 1.7
        lab = f'{ack:+.0f}%' + ('  (this car)' if lw > 2 else '')
        ax1.plot(steer, c['useful_N'], color=col, ls=ls, lw=lw, label=lab)
        ax1.plot([c['peak_steer_deg']], [c['peak_N']], 'o', color=col, ms=6)
        ax2.plot(steer, c['scrub_N'], color=col, ls=ls, lw=lw, label=lab)
    ax1.set_xlabel('Reference steer angle (deg)  —  average of the two wheels',
                   fontsize=9, color=_TICK)
    ax1.set_ylabel('Axle lateral force toward the corner (N)',
                   fontsize=9, color=_TICK)
    ax1.set_title(f'Pair analysis (RCVD ch.7) at {d["lat_g"]:.2f} g\n'
                  f'inner {d["fz_in"]:.0f} N / outer {d["fz_out"]:.0f} N, '
                  f'load transfer {100*d["flt"]:.0f}%   ·  dot = best steer',
                  fontsize=9.5, color=_TEXT)
    ax2.set_xlabel('Reference steer angle (deg)', fontsize=9, color=_TICK)
    ax2.set_ylabel('Scrub drag from the front tyres (N)', fontsize=9,
                   color=_TICK)
    ax2.set_title('What it costs: force resolved along the car, not across it\n'
                  '(the part of the tyre force that only slows you down)',
                  fontsize=9.5, color=_TEXT)
    for ax in (ax1, ax2):
        ax.axhline(0, color=_GRID, lw=0.8)
        ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
                  fontsize=8, title='Ackermann', loc='best')
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
    ackermann_pct: float = 0.0,
    beta_range_deg: tuple = (-8, 8), beta_n: int = 9,
    delta_range_deg: tuple = (-20, 20), delta_n: int = 11,
) -> Figure:
    """Pure-cornering MMD: yaw moment N vs lateral acceleration Ay,
    with constant-β and constant-δ lines.

    Roll stiffness (springs + ARB) drives the lateral load transfer
    distribution (LLTD) between front and rear axles.  Static toe angles
    (+ = toe-in) shift the individual corner slip angles.
    """
    # ── State computation: rerouted through vahan.ymd (the ONE yaw-moment
    # engine).  This function used to carry its own private iteration with
    # rear slip set EQUAL to body slip (aRL = aRR = beta) — only true for a
    # car that is not rotating.  The engine adds the missing yaw-rate term
    # (rear slip differs from beta by degrees(l_r*r/V); front by l_f*r/V, RCVD
    # Eqs. 5.3/5.4), so ABSOLUTE NUMBERS HAVE MOVED versus older MMD figures:
    # the old diagrams understated rear slip — and therefore rear force — at
    # high Ay.  The LLTD what-if knobs (ARB lever deltas) stay a closed-form
    # loads model by design (a parameter study), owned by ymd.LLTDCar; only
    # the loads source differs from the trim-sweep path, never the physics.
    from vahan import ymd
    car = ymd.LLTDCar(
        total_mass_kg=total_mass_kg, weight_dist_front=weight_dist_front,
        wheelbase_m=wheelbase_m, cg_height_m=cg_height_m,
        track_front_m=track_front_m, track_rear_m=track_rear_m,
        roll_stiffness_front_Npm_rad=roll_stiffness_front_Npm_rad,
        roll_stiffness_rear_Npm_rad=roll_stiffness_rear_Npm_rad,
        delta_arb_front_Npm=delta_arb_front_Npm,
        delta_arb_rear_Npm=delta_arb_rear_Npm)
    LLTD_f = car.lltd_front                     # front fraction, for the box
    # dense enough that the 10 N wheel-lift clamp doesn't kink the interp
    _loads = ymd.build_loads_table(car, ay_max=3.0, n=61)

    # Warm-start each state from the previous one along the sweep: the
    # fixed-point map holds one attractor per turn hand, and cold 0-starts
    # let states at the extreme corners (big beta against big opposite
    # delta) hop hands mid-line, scribbling the diagram tips.  Continuation
    # keeps each isoline on its own branch (and converges faster).
    _prev_ay = {'ay': 0.0}

    def compute_state(beta_deg, delta_deg):
        st = ymd.ymd_state(
            tire_model, car, beta_deg, delta_deg, V_mps=velocity_mps,
            ackermann_pct=ackermann_pct, toe_front_deg=toe_front_deg,
            toe_rear_deg=toe_rear_deg, loads_table=_loads,
            Ay0=_prev_ay['ay'])
        _prev_ay['ay'] = st['Ay_g']
        return st['Ay_g'], st['N_Nm']

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

    # ── THE NUMBERS, from the same states the curves are drawn from ────────
    # Trimmed limit: walk every constant-beta line, find its N = 0 crossings
    # by sign change + linear interpolation, keep the one with the highest
    # positive-Ay (left-turn) crossing.  Control/stability at that exact
    # point come from two extra engine calls each (central differences) —
    # not read off the coarse grid.
    # The trimmed limit can sit OUTSIDE the drawn attitude window (post
    # tyre-clamp it does: the outer N = 0 crossings live beyond beta = -8
    # at this speed), so the search sweeps a WIDER attitude range than the
    # figure shows.  The drawn window stays the user's choice.
    lim = {'ay': float('nan'), 'beta': float('nan'), 'delta': float('nan'),
           'ctrl': float('nan'), 'stab': float('nan')}
    _b_lo = min(beta_range_deg) - 10.0
    _b_hi = max(beta_range_deg) + 4.0
    for b in np.arange(_b_lo, _b_hi + 0.5, 1.0):
        Nrow = np.empty(line_n); Arow = np.empty(line_n)
        for j, d in enumerate(deltas_dense):
            Arow[j], Nrow[j] = compute_state(float(b), float(d))
        for j in range(line_n - 1):
            if Nrow[j] == 0 or (Nrow[j] > 0) != (Nrow[j + 1] > 0):
                f = Nrow[j] / (Nrow[j] - Nrow[j + 1] + 1e-12)
                ayc = Arow[j] + f * (Arow[j + 1] - Arow[j])
                dc = deltas_dense[j] + f * (deltas_dense[j + 1]
                                            - deltas_dense[j])
                if np.isfinite(ayc) and (not np.isfinite(lim['ay'])
                                         or ayc > lim['ay']):
                    lim.update(ay=float(ayc), beta=float(b), delta=float(dc))
    # Second transect family: at high Ay the N = 0 contour can run parallel
    # to the constant-beta lines (no crossing along them at all) — the
    # crossings then only show up sweeping BETA at constant steer.
    _bd = np.arange(_b_lo, _b_hi + 0.4, 0.75)
    for d in deltas:
        Nrow = np.empty(len(_bd)); Arow = np.empty(len(_bd))
        for j, b in enumerate(_bd):
            Arow[j], Nrow[j] = compute_state(float(b), float(d))
        for j in range(len(_bd) - 1):
            if Nrow[j] == 0 or (Nrow[j] > 0) != (Nrow[j + 1] > 0):
                f = Nrow[j] / (Nrow[j] - Nrow[j + 1] + 1e-12)
                ayc = Arow[j] + f * (Arow[j + 1] - Arow[j])
                bc = _bd[j] + f * (_bd[j + 1] - _bd[j])
                if np.isfinite(ayc) and (not np.isfinite(lim['ay'])
                                         or ayc > lim['ay']):
                    lim.update(ay=float(ayc), beta=float(bc), delta=float(d))
    # Local refinement: the family sweeps quantize the crossing to the
    # coarse grids (a 4-deg steer bin moved the cross-setting comparison
    # more than the physics did).  Re-sweep finely around the found point,
    # both directions, keeping the best N = 0 crossing.
    if np.isfinite(lim['ay']):
        for _pass in range(2):
            _df = np.arange(lim['delta'] - 2.5, lim['delta'] + 2.51, 0.25)
            Nr = np.empty(len(_df)); Ar = np.empty(len(_df))
            for j, d in enumerate(_df):
                Ar[j], Nr[j] = compute_state(lim['beta'], float(d))
            for j in range(len(_df) - 1):
                if (Nr[j] > 0) != (Nr[j + 1] > 0):
                    f = Nr[j] / (Nr[j] - Nr[j + 1] + 1e-12)
                    ayc = Ar[j] + f * (Ar[j + 1] - Ar[j])
                    if ayc > lim['ay']:
                        lim.update(ay=float(ayc),
                                   delta=float(_df[j] + f * 0.25))
            _bf = np.arange(lim['beta'] - 1.5, lim['beta'] + 1.51, 0.25)
            Nr = np.empty(len(_bf)); Ar = np.empty(len(_bf))
            for j, b in enumerate(_bf):
                Ar[j], Nr[j] = compute_state(float(b), lim['delta'])
            for j in range(len(_bf) - 1):
                if (Nr[j] > 0) != (Nr[j + 1] > 0):
                    f = Nr[j] / (Nr[j] - Nr[j + 1] + 1e-12)
                    ayc = Ar[j] + f * (Ar[j + 1] - Ar[j])
                    if ayc > lim['ay']:
                        lim.update(ay=float(ayc),
                                   beta=float(_bf[j] + f * 0.25))
    if np.isfinite(lim['ay']):
        h = 0.5
        _, n_dp = compute_state(lim['beta'], lim['delta'] + h)
        _, n_dm = compute_state(lim['beta'], lim['delta'] - h)
        _, n_bp = compute_state(lim['beta'] + h, lim['delta'])
        _, n_bm = compute_state(lim['beta'] - h, lim['delta'])
        lim['ctrl'] = float((n_dp - n_dm) / (2 * h))
        lim['stab'] = float((n_bp - n_bm) / (2 * h))
    ay_reach = float(np.nanmax([np.nanmax(Ay_b), np.nanmax(Ay_d)]))

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

    # ── Numbers box (bottom-right): the readable results ────────────────
    if np.isfinite(lim['ay']):
        num_lines = (
            f"AT THE TRIMMED LIMIT (N = 0, driven clean)\n"
            f"  lateral grip     {lim['ay']:+.3f} g\n"
            f"  body slip        {lim['beta']:+.1f} deg\n"
            f"  mean steer       {lim['delta']:+.1f} deg\n"
            f"  control  dN/dd   {lim['ctrl']:+.0f} N·m/deg\n"
            f"  stability dN/db  {lim['stab']:+.0f} N·m/deg\n"
            f"MAX REACH (untrimmed, spinning) {ay_reach:+.3f} g"
        )
    else:
        num_lines = 'no N = 0 crossing found in the swept range'
    ax.text(0.98, 0.02, num_lines,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=_TEXT, family='monospace',
            bbox=dict(facecolor=_PANEL_BG, edgecolor=_YELLOW, pad=6))
    if np.isfinite(lim['ay']):
        ax.plot([lim['ay']], [0.0], 'o', ms=12, mfc='none',
                mec=_YELLOW, mew=2.0, zorder=6)

    fig._mmd_numbers = dict(ackermann_pct=float(ackermann_pct),
                            ay_trim_limit_g=lim['ay'],
                            beta_at_limit_deg=lim['beta'],
                            delta_at_limit_deg=lim['delta'],
                            control_Nm_per_deg=lim['ctrl'],
                            stability_Nm_per_deg=lim['stab'],
                            ay_reach_g=ay_reach)
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


# ═════════════════════════════════════════════════════════════════════════════
# 12. Slip angle vs vertical load vs lateral force  (the tyre, seen directly)
# ═════════════════════════════════════════════════════════════════════════════
def plot_slip_load_force(tire_model, solver=None, grip_multiplier: float = 1.0,
                         lat_g_list=(0.5, 1.0, 1.5, 2.0)) -> Figure:
    """Slip angle against lateral force, one curve per vertical load — plus
    where the CAR's four wheels actually sit on that family.

    This is the diagnostic the user asked for and the one every Ackermann
    argument rests on: how much side force a tyre makes at a given slip angle
    AT ITS OWN LOAD.  It exists as a Vahan feature (button + this function)
    rather than a report script because a number nobody can pull up and drive
    is not a result.

    Everything is read from the tyre model and the solved car — no physics is
    computed here.  The right panel is the FRICTION CHECK the user demanded:
    peak available force against load, with the car's own operating points on
    top, so an impossible demand is visible instead of implied.
    """
    fig, (ax1, ax2) = _styled_fig(figsize=(13, 5.6), n=1, m=2)
    gm = max(float(grip_multiplier), 1e-6)
    fz_lo, fz_hi = (float(tire_model.fz_range[0]),
                    float(tire_model.fz_range[1]))
    loads = np.linspace(max(fz_lo * 0.5, 50.0), fz_hi, 6)
    sa_hi = float(np.max(np.abs(tire_model.sa_range)))
    sa = np.linspace(0.0, sa_hi, 120)
    # Load is the series variable, so vary it by SHADE within one hue family
    # (never a second colour) — the in-app palette stays yellow/red/white.
    shades = ['#ffffff', '#ffe082', '#ffca28', '#ff8f00', '#e64a19', '#b71c1c']
    for fz, col in zip(loads, shades):
        fy = np.array([abs(float(tire_model.Fy(a, float(fz), 0.0))) * gm
                       for a in sa])
        ax1.plot(sa, fy, color=col, lw=1.9, label=f'{fz:.0f} N')
        k = int(np.argmax(fy))
        ax1.plot([sa[k]], [fy[k]], 'o', color=col, ms=4)
    ax1.set_xlabel('Slip angle (deg)', fontsize=10, color=_TICK)
    ax1.set_ylabel(f'Lateral force (N){"" if gm >= 1 else f"  x{gm:.2f} asphalt"}',
                   fontsize=10, color=_TICK)
    ax1.set_title('What the tyre makes: slip angle vs load\n'
                  'dot = that load\'s peak (later at higher load)',
                  fontsize=10, color=_TEXT)
    ax1.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
               fontsize=8, title='vertical load', loc='lower right')
    ax1.grid(alpha=0.25, lw=0.5)

    # ── right: the friction check ────────────────────────────────────────
    fzs = np.linspace(50.0, fz_hi * 1.25, 90)
    pk = np.array([abs(float(tire_model.peak_Fy(float(f), 0.0))) * gm
                   for f in fzs])
    ax2.plot(fzs, pk, color=_YELLOW, lw=2.2,
             label='MOST the tyre can make (peak)')
    ax2.axvline(fz_lo, color=_TICK, lw=0.8, ls=':', alpha=0.7)
    ax2.axvline(fz_hi, color=_TICK, lw=0.8, ls=':', alpha=0.7)
    ax2.text(fz_hi, pk.max() * 0.05, ' measured data ends',
             fontsize=7.5, color=_TICK, rotation=90, va='bottom')
    if solver is not None:
        marks = ['o', 's', '^', 'D']
        over = 0
        for gi, g in enumerate(lat_g_list):
            try:
                r = solver.solve(float(g), 0.0)
            except Exception:
                continue
            for c in ('FL', 'FR', 'RL', 'RR'):
                fz = max(float(r.Fz[c]), 0.0)
                fy = abs(float(r.Fy[c]))
                cap = abs(float(tire_model.peak_Fy(max(fz, 1.0), 0.0))) * gm
                bad = fy > cap * 1.001
                over += int(bad)
                ax2.plot([fz], [fy], marks[gi % 4],
                         color=(_RED if bad else _WHITE), ms=6,
                         markeredgecolor=_RED if bad else _TICK, mew=1.0,
                         label=(f'{g:.1f} g' if c == 'FL' else None))
        ax2.set_title('Friction check: is any wheel asked for more than the\n'
                      'tyre can give?  ' + ('RED = IMPOSSIBLE DEMAND'
                                            if over else
                                            'all inside the envelope'),
                      fontsize=10, color=(_RED if over else _TEXT))
    ax2.set_xlabel('Vertical load Fz (N)', fontsize=10, color=_TICK)
    ax2.set_ylabel('Lateral force (N)', fontsize=10, color=_TICK)
    ax2.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, edgecolor=_GRID,
               fontsize=8, loc='upper left')
    ax2.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    return fig


def plot_ymd_ackermann_grid(tire_model, solver, radius_m=8.0,
                            pct_list=(100, 70, 50, 30, 0, -30, -70),
                            grip_multiplier=0.70, aero_Fz_per_g=None,
                            fig=None):
    """Ackermann vs stability/control — READABLE version.

    The first version drew nine full moment diagrams in a grid; they were
    near-identical postage stamps (that near-identity IS the physics), so
    the figure carried no readable information.  This version shows only
    what differs:

      LEFT  — the trim region, zoomed and overlaid: for each Ackermann
              setting, the constant-steer line through its own trimmed
              limit, in the last tenths of a g before the limit.  Where a
              curve crosses CN = 0 is that setting's trimmed limit; the
              slope through the crossing is how gradually the car
              approaches it.
      RIGHT — the metrics: max trimmed lateral g (with the measured
              0.006 g noise band), control dN/d(steer) and stability
              dN/d(body slip) at trim, per setting.

    Returns (fig, rows) where rows is vahan.ymd.trim_sweep_ackermann
    output enriched with a plain-language verdict per metric.  All physics
    from vahan.ymd — this function only sweeps and draws.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from . import ymd as YMD

    m, L, l_f, l_r, t_f, t_r = YMD._dims(solver)
    W = m * 9.80665
    table = YMD.build_loads_table(solver, aero_Fz_per_g)

    trim = YMD.trim_sweep_ackermann(
        tire_model, solver, radius_m, list(pct_list),
        grip_multiplier=grip_multiplier, aero_Fz_per_g=aero_Fz_per_g,
        loads_table=table)

    if fig is None:
        fig = plt.figure(figsize=(13.0, 6.0))
    axL, axR = fig.subplots(1, 2)
    NOISE_G = 0.006

    # LEFT: one curve per setting — beta swept through the setting's own
    # trim attitude at its trim steer, plotted only near the limit.
    styles = ['-', '--', '-.', ':', '-', '--', '-.']
    widths = [2.6, 2.2, 2.2, 2.2, 2.6, 2.2, 2.2]
    cols = ['#1a1a1a', '#1a1a1a', '#8a8a8a', '#8a8a8a',
            '#C43B3B', '#C43B3B', '#D99000']
    ay_lo = 1e9
    for k, r in enumerate(trim):
        if not np.isfinite(r.get('Ay_trim_max', float('nan'))):
            continue
        b0, d0 = r['beta_at'], r['delta_at']
        bs = np.linspace(b0 - 2.5, b0 + 2.5, 21)
        pts = []
        ay0 = max(r['Ay_trim_max'] - 0.1, 0.5)
        for b in bs:
            try:
                st = YMD.ymd_state(tire_model, solver, float(b), float(d0),
                                   radius_m=radius_m,
                                   ackermann_pct=float(r['pct']),
                                   grip_multiplier=grip_multiplier,
                                   aero_Fz_per_g=aero_Fz_per_g,
                                   loads_table=table, Ay0=ay0)
            except Exception:
                continue
            if st.get('converged', True):
                pts.append((st['Ay_g'], st['N_Nm'] / (W * L)))
                ay0 = st['Ay_g']
        if len(pts) > 2:
            a = np.array(pts)
            axL.plot(a[:, 0], a[:, 1], styles[k % 7], color=cols[k % 7],
                     lw=widths[k % 7],
                     label=f"{r['pct']:+.0f}%  (trim {r['Ay_trim_max']:.3f} g)")
            ay_lo = min(ay_lo, float(np.nanmin(a[:, 0])))
            # RCVD Fig 8.26 limit parameters (printed p334-335), read from
            # the same swept states:
            #   SI — slope of the steer line through trim, dCN/dAy
            #        (negative = stable; ~0 = divergence boundary, p205)
            #   apex — max untrimmed Ay and its CN: below the axis = PLOW,
            #        on it = NEUTRAL drift, above = SPIN (p306, p320);
            #        race cars are tuned neutral at the limit (p320).
            _near = np.argsort(np.abs(a[:, 1]))[:5]
            if len(_near) >= 2:
                _aa, _cc = a[_near, 0], a[_near, 1]
                _den = (_aa.max() - _aa.min())
                r['SI_CN_per_g'] = (float(np.polyfit(_aa, _cc, 1)[0])
                                    if _den > 1e-4 else float('nan'))
            _kk = int(np.nanargmax(a[:, 0]))
            r['Ay_apex_g'] = float(a[_kk, 0])
            r['CN_apex'] = float(a[_kk, 1])
            r['limit_character'] = ('PLOW (safe push)' if r['CN_apex'] < -0.004
                                    else 'SPIN (nose-in past limit)'
                                    if r['CN_apex'] > 0.004
                                    else 'NEUTRAL (race target, RCVD p320)')
        axL.plot([r['Ay_trim_max']], [0.0], 'o', ms=7,
                 color=cols[k % 7], zorder=5)
    axL.axhline(0.0, color='#D99000', lw=1.4)
    ays = [r['Ay_trim_max'] for r in trim
           if np.isfinite(r.get('Ay_trim_max', float('nan')))]
    if ays:
        axL.axvspan(max(ays) - NOISE_G, max(ays), color='#D99000',
                    alpha=0.25)
        axL.set_xlim(min(min(ays) - 0.02, ay_lo), max(ays) + 0.012)
    axL.set_ylim(-0.05, 0.05)
    axL.set_xlabel('lateral g (road)')
    axL.set_ylabel('CN = N / (W L)')
    axL.set_title('The trim region, zoomed — dots on CN = 0 are each '
                  'setting\'s limit;\nshaded band = 0.006 g noise floor',
                  fontsize=10)
    axL.grid(alpha=0.25)
    axL.legend(fontsize=8.5, loc='lower left')

    # RIGHT: metrics vs Ackermann
    pc = np.array([r['pct'] for r in trim], float)
    ay = np.array([r.get('Ay_trim_max', float('nan')) for r in trim], float)
    ct = np.array([r.get('N_delta', float('nan')) for r in trim], float)
    sb = np.array([r.get('N_beta', float('nan')) for r in trim], float)
    o = np.argsort(pc)
    axR.plot(pc[o], ay[o], 'o-', color='#1a1a1a', lw=2.2,
             label='max trimmed g')
    if np.isfinite(ay).any():
        axR.axhspan(np.nanmax(ay) - NOISE_G, np.nanmax(ay),
                    color='#D99000', alpha=0.25,
                    label='tie band (0.006 g noise)')
    ax2 = axR.twinx()
    ax2.plot(pc[o], ct[o], 's--', color='#C43B3B', lw=1.6,
             label='control dN/d(steer) @trim')
    ax2.plot(pc[o], sb[o], '^:', color='#8a8a8a', lw=1.6,
             label='stability dN/d(body slip) @trim')
    ax2.set_ylabel('N·m per deg', fontsize=9)
    axR.set_xlabel('Ackermann %')
    axR.set_ylabel('trimmed lateral g')
    axR.set_title('Metrics vs Ackermann — inside the band = tie',
                  fontsize=10)
    axR.grid(alpha=0.25)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower center')

    fig.suptitle(
        f'Ackermann vs stability and control — constant radius '
        f'{radius_m:.0f} m, asphalt {grip_multiplier:.2f}x rig grip, '
        f'{"with" if aero_Fz_per_g else "no"} aero', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, trim
