"""
Regenerate the panel + transient screenshots referenced by the README:

  screenshots/skidpad_panel.png    — Skidpad / Transient panel UI
  screenshots/dynamics_panel.png   — Dynamics panel with hollow ARB inputs
                                     and the auto-derived ARB readout
  screenshots/transient_sim.png    — Time-history plot from a step-steer sim
                                     with inertias + roll damping derived
                                     from VehicleParams + panel damper rates

Run from the repo root:
    PYTHONIOENCODING=utf-8 py tools/gen_screenshots.py
"""
from __future__ import annotations

import os
import sys

# Use the Windows platform plugin so we get system font cache + ClearType.
# We never *show* a window — we only call .grab() — so no display is needed.
# (offscreen plugin works but loses font discovery → tofu glyphs.)
os.environ.pop('QT_QPA_PLATFORM', None)

# Script lives in <repo>/tools/, so the repo root is its parent.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import QApplication, QScrollArea, QWidget, QVBoxLayout

from gui.panels import DynamicsPanel, SkidpadPanel


SHOTS = os.path.join(ROOT, 'screenshots')
os.makedirs(SHOTS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Panel rendering helper
# ─────────────────────────────────────────────────────────────────────────────

def render_panel(panel: QWidget, width: int, out: str) -> None:
    """Force-show a collapsible panel, expand it, lay it out at `width` px,
    and grab to a PNG file.  Works with QT_QPA_PLATFORM=offscreen so no
    real display is required."""
    # Wrap so the section's expand-collapse animation has a parent.
    host = QWidget()
    host.setObjectName('host')
    host.setStyleSheet(
        'QWidget#host { background: #000; }'
        'QLabel { color: #e0e0e0; }'
    )
    lay = QVBoxLayout(host)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.addWidget(panel)
    lay.addStretch(1)

    # If the panel is a CollapsibleSection, force it expanded.
    for attr in ('expand', 'set_expanded', 'setExpanded', 'toggle'):
        fn = getattr(panel, attr, None)
        if callable(fn):
            try:
                fn(True)
                break
            except TypeError:
                try:
                    fn()
                    break
                except Exception:
                    pass

    host.setFixedWidth(width)
    host.adjustSize()
    # Let layout settle (process events twice — first pass sizes children,
    # second pass resizes the host to fit).
    QApplication.processEvents()
    host.adjustSize()
    QApplication.processEvents()

    pm: QPixmap = host.grab()
    if pm.isNull():
        raise RuntimeError(f'grab() returned null pixmap for {out}')
    if not pm.save(out, 'PNG'):
        raise RuntimeError(f'save() failed for {out}')
    print(f'  wrote {out}  ({pm.width()}x{pm.height()})')


# ─────────────────────────────────────────────────────────────────────────────
#  Skidpad panel
# ─────────────────────────────────────────────────────────────────────────────

def gen_skidpad_panel():
    print('[1/3] Rendering SkidpadPanel ...')
    panel = SkidpadPanel()
    # Bump test type to skidpad_full so the panel shows the full mode UI
    for i in range(panel._test_combo.count()):
        if panel._test_combo.itemData(i) == 'skidpad_full':
            panel._test_combo.setCurrentIndex(i)
            break
    panel._target_speed.setValue(23.3)
    panel._target_lat_g.setValue(1.50)
    panel._dt_ms.setValue(2.0)
    # Populate the auto-info readout so the screenshot looks complete
    panel.set_auto_info(
        yaw_Izz=110.5, sprung_Ixx=70.2, ackermann_pct=12.0,
        sim_duration_s=12.5, peak_steer_deg=9.6,
        derived_speed_ms=10.4, derived_roll_damping=2870.0,
    )
    # Populate a fake result so the readouts are non-empty
    class _FakeResult:
        peak_lateral_g = 1.52
        steady_state_lateral_g = 1.48
        peak_roll_deg = 1.18
        steady_state_roll_deg = 1.05
        yaw_rate_rise_time_s = 0.182
        yaw_rate_overshoot_pct = 4.6
        yaw_rate_settling_time_s = 0.495
        peak_understeer_deg = 0.86
    panel.show_result(_FakeResult())
    panel.set_status('Sim complete — 12.5 s.')
    render_panel(panel, width=420,
                 out=os.path.join(SHOTS, 'skidpad_panel.png'))


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamics panel
# ─────────────────────────────────────────────────────────────────────────────

def gen_dynamics_panel():
    print('[2/3] Rendering DynamicsPanel ...')
    panel = DynamicsPanel()
    # Show a hollow rear bar so the new ID input is non-zero in the screenshot
    panel._arb_OD_f.setValue(12.7)
    panel._arb_ID_f.setValue(0.0)
    panel._arb_OD_r.setValue(12.7)
    panel._arb_ID_r.setValue(9.75)
    # Populate the derived-ARB readout
    panel.set_derived_arb_geometry(
        front={'arm_length_mm': 84.33, 'half_length_mm': 249.43, 'mr': 2.500},
        rear={'arm_length_mm': 104.78, 'half_length_mm': 286.26, 'mr': 3.000},
    )
    # Trigger the dyn-constants block so it has content
    panel._on_driving_changed()
    render_panel(panel, width=420,
                 out=os.path.join(SHOTS, 'dynamics_panel.png'))


# ─────────────────────────────────────────────────────────────────────────────
#  Transient simulation result plot
# ─────────────────────────────────────────────────────────────────────────────

def gen_transient_sim():
    print('[3/3] Running transient sim + plotting ...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from vahan.dynamics import VehicleParams
    from vahan.tire_model import LinearTireModel
    from vahan.transient import (TransientSolver, TransientParams,
                                 TransientInputs, SteeringProfile)

    veh = VehicleParams()
    tire = LinearTireModel()

    # ── Derive inertias + roll damping the same way main_window does ────
    # Yaw inertia (bicycle-style):    Izz = k · m · a · b   (k=1.2 default)
    # Roll inertia (gyradius rule):   Ixx = m_s · (frac · track_avg)²
    a_cg = veh.cg_to_front_axle_m
    b_cg = veh.wheelbase_m - a_cg
    Izz  = veh.yaw_inertia_factor * veh.total_mass_kg * a_cg * b_cg
    track_avg = 0.5 * (veh.front_track_m + veh.rear_track_m)
    k_roll    = veh.roll_gyradius_track_frac * track_avg
    Ixx  = veh.sprung_mass_kg * (k_roll ** 2)

    # Roll damping from SkidpadPanel default damper bump/rebound rates.
    # c_phi_axle = (c_bump + c_rebound) · MR² · t² / 4
    c_F_bump, c_F_reb = 1305.0, 2936.0      # SkidpadPanel defaults
    c_R_bump, c_R_reb = 1483.0, 3338.0
    c_F = c_F_bump + c_F_reb
    c_R = c_R_bump + c_R_reb
    c_phi_F = c_F * veh.motion_ratio_front**2 * veh.front_track_m**2 / 4
    c_phi_R = c_R * veh.motion_ratio_rear **2 * veh.rear_track_m **2 / 4
    c_phi   = c_phi_F + c_phi_R

    print(f'  Derived: Izz={Izz:.1f} kg·m²  Ixx={Ixx:.1f} kg·m²  '
          f'c_phi={c_phi:.0f} N·m·s/rad')

    # 50 ms ramp-step to ~10 deg road-wheel angle, target speed 12 m/s,
    # 5 s sim — that captures the full yaw rise + settle window for a
    # FSAE-class car cleanly.
    steer = SteeringProfile.ramp(
        t_start=0.5, t_end=0.55,
        steer_rad=np.radians(10.0),
    )
    tparams = TransientParams(
        sprung_roll_inertia=Ixx,
        yaw_inertia=Izz,
        roll_damping_Nms_rad=c_phi,
        ackermann_pct=10.0,
        steering_tau_s=0.02,
    )
    solver = TransientSolver(veh, tire, params=tparams)

    inputs = TransientInputs(
        v_x_target_ms=12.0,
        steering=steer,
        duration_s=5.0,
        dt_s=0.002,
    )
    result = solver.simulate(inputs)

    # ── Plot ────────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 5.4),
                            sharex=True, dpi=120)
    fig.patch.set_facecolor('#000000')

    t = result.t

    # 1. Steer command + actual + yaw rate
    ax0 = axs[0]
    ax0.set_facecolor('#0a0a0a')
    ax0.plot(t, np.degrees(result.steer), color='#888888',
             lw=1.2, label='steer cmd', linestyle='--')
    ax0.plot(t, np.degrees(result.steer_actual), color='#4FC3F7',
             lw=1.6, label='steer actual')
    ax0.set_ylabel('steer (deg)', color='#cccccc', fontsize=10)
    ax0.legend(loc='lower right', fontsize=8, framealpha=0.7)
    ax0r = ax0.twinx()
    ax0r.plot(t, np.degrees(result.yaw_rate), color='#FFB74D',
              lw=1.6, label='yaw rate')
    ax0r.set_ylabel('yaw rate (deg/s)', color='#FFB74D', fontsize=10)
    ax0r.tick_params(axis='y', colors='#FFB74D')
    ax0.set_title('Step-steer transient — 10° at t = 0.5 s, target 12 m/s',
                  color='#e0e0e0', fontsize=11, pad=8)

    # 2. Lateral g
    ax1 = axs[1]
    ax1.set_facecolor('#0a0a0a')
    ay_g = result.ay / 9.80665
    ax1.plot(t, ay_g, color='#81C784', lw=1.8)
    ax1.axhline(result.steady_state_lateral_g, color='#444', lw=0.8,
                linestyle=':', label=f'ss = {result.steady_state_lateral_g:.2f} g')
    ax1.set_ylabel('lateral (g)', color='#cccccc', fontsize=10)
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.7)

    # 3. Roll
    ax2 = axs[2]
    ax2.set_facecolor('#0a0a0a')
    ax2.plot(t, np.degrees(result.roll), color='#CE93D8', lw=1.8)
    ax2.axhline(result.steady_state_roll_deg, color='#444', lw=0.8,
                linestyle=':',
                label=f'ss = {result.steady_state_roll_deg:.2f}°')
    ax2.set_ylabel('roll (deg)', color='#cccccc', fontsize=10)
    ax2.set_xlabel('t (s)', color='#cccccc', fontsize=10)
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.7)

    for ax in (ax0, ax1, ax2):
        ax.tick_params(colors='#cccccc')
        ax.grid(True, color='#1a1a1a', lw=0.5)
        for s in ax.spines.values():
            s.set_color('#333333')
    for s in ax0r.spines.values():
        s.set_color('#333333')

    fig.tight_layout()
    out = os.path.join(SHOTS, 'transient_sim.png')
    fig.savefig(out, dpi=120, facecolor='#000000')
    plt.close(fig)
    print(f'  wrote {out}')
    print(f'  ay peak  = {result.peak_lateral_g:.3f} g   '
          f'ay steady = {result.steady_state_lateral_g:.3f} g')
    print(f'  roll peak  = {result.peak_roll_deg:.2f}°    '
          f'roll steady = {result.steady_state_roll_deg:.2f}°')
    print(f'  yaw rise = {result.yaw_rate_rise_time_s*1000:.0f} ms   '
          f'overshoot = {result.yaw_rate_overshoot_pct:.1f} %')


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Force a Windows system font so text renders as glyphs, not tofu boxes,
    # even when we're running from a non-interactive shell.
    app.setFont(QFont('Segoe UI', 9))
    gen_skidpad_panel()
    gen_dynamics_panel()
    gen_transient_sim()
    print('\nAll three screenshots written to screenshots/')


if __name__ == '__main__':
    main()
