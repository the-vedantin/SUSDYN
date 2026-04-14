"""
gui/main_window.py -- Vahan Main Window

Axis convention: X=lateral(outboard+), Y=longitudinal(fwd+), Z=up(+)

Corners:
    FL = left-front   (default hardpoints, outboard = +X)
    FR = mirror of FL (outboard = -X)
    RL = left-rear    (absolute Y coords -- no wheelbase offset applied)
    RR = mirror of RL

Steering (front only):
    Rack translates in X. Both steer-rod inners move by the same rack_travel.
    rack_travel = steer_wheel_angle * rack_mm_per_rev / 360
    Clamped by both total_rack_travel_mm and max_rack_travel_in.
"""

import sys
import json
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QSplitter, QStatusBar, QSizePolicy, QScrollArea,
    QGroupBox, QCheckBox, QMenuBar, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from vahan import DoubleWishboneHardpoints
from vahan.solver import SuspensionConstraints, SolvedState, _norm, _rodrigues
from vahan.kinematics import KinematicMetrics, _intersect_2d
from vahan.metrics_catalog import CATALOG, CATALOG_MAP, DEFAULT_Y_KEYS

from gui.view3d import View3D, HP_NAMES
from gui.panels import (
    MotionPanel, CarParamsPanel, HardpointPanel,
    ValuesPanel, GraphPickerPanel, SteeringPanel, AlignmentPanel,
    CollapsibleSection, InverseKinematicsPanel, DynamicsPanel, DynamicsOptPanel,
    LoadsPanel, AeroPanel, AeroGeomPanel,
)
from vahan.optimizer import InverseSolver, DesignVar
from vahan.dynamics import (VehicleParams, SteadyStateSolver, SteadyStateResult,
                            DynamicsSensitivity, AeroDownforceSolver, AeroResult)

# ==============================================================================
#  DEFAULT HARDPOINTS  (X=lateral outboard+, Y=fwd+, Z=up+)
#  All values in metres, converted from inches (1 in = 0.0254 m).
#  Right-side / FL convention: outboard = +X.
#  Front Y values are small offsets from the front axle centre.
#  Rear  Y values are ABSOLUTE (already include the ~60 in wheelbase).
# ==============================================================================

# Front axle -- FL corner (outboard = +X)
DEFAULT_FRONT_HP = {
    'uca_front':         np.array([ 0.26353, -0.12700,  0.26353]),
    'uca_rear':          np.array([ 0.23243,  0.12700,  0.24877]),
    'uca_outer':         np.array([ 0.48260,  0.00912,  0.28598]),
    'lca_front':         np.array([ 0.21590, -0.11748,  0.12065]),
    'lca_rear':          np.array([ 0.21590,  0.12342,  0.12700]),
    'lca_outer':         np.array([ 0.53340, -0.00318,  0.11913]),
    'tie_rod_inner':     np.array([ 0.21908, -0.06985,  0.15199]),  # rack end (steered)
    'tie_rod_outer':     np.array([ 0.54293, -0.07303,  0.17145]),  # steer rod upright end
    'wheel_center':      np.array([ 0.55880,  0.00000,  0.20320]),
    'pushrod_outer':     np.array([ 0.43815, -0.00318,  0.31953]),  # fixed to UCA
    'pushrod_inner':     np.array([ 0.25740, -0.00318,  0.64683]),
    'rocker_pivot':      np.array([ 0.21293, -0.00318,  0.62230]),
    'rocker_spring_pt':  np.array([ 0.20749, -0.00318,  0.67919]),
    'spring_chassis_pt': np.array([ 0.01588, -0.00318,  0.66091]),
    # Axis point = pivot + 1 in Y  => rocker pivots about Y-parallel axis
    'rocker_axis_pt':    np.array([ 0.21293,  0.02222,  0.62230]),
}

# Front ARB (left / FL side; right is X-mirrored)
DEFAULT_FRONT_ARB = {
    'arb_drop_top':  np.array([ 0.23833, -0.00318,  0.62149]),
    'arb_arm_end':   np.array([ 0.23833, -0.00318,  0.55779]),
    'arb_pivot':     np.array([ 0.23833, -0.08758,  0.55779]),
}

# Rear axle -- RL corner (outboard = +X)
# Y coords are ABSOLUTE -- do NOT apply _offset_y (no wheelbase shift).
DEFAULT_REAR_HP = {
    'uca_front':         np.array([ 0.27940,  1.44780,  0.26975]),
    'uca_rear':          np.array([ 0.24778,  1.65895,  0.27148]),
    'uca_outer':         np.array([ 0.48895,  1.54940,  0.28075]),
    'lca_front':         np.array([ 0.28158,  1.44958,  0.14356]),
    'lca_rear':          np.array([ 0.22860,  1.65895,  0.12700]),
    'lca_outer':         np.array([ 0.53340,  1.53670,  0.11913]),
    'tie_rod_inner':     np.array([ 0.28158,  1.44958,  0.14356]),  # chassis toe link (same point as lca_front)
    'tie_rod_outer':     np.array([ 0.53340,  1.46086,  0.12631]),  # 21.000/57.514/4.973 in
    'wheel_center':      np.array([ 0.55880,  1.53670,  0.20320]),
    'pushrod_outer':     np.array([ 0.48260,  1.54623,  0.14448]),  # fixed to LCA
    'pushrod_inner':     np.array([ 0.28110,  1.54623,  0.38765]),
    'rocker_pivot':      np.array([ 0.23708,  1.54623,  0.35118]),
    'rocker_spring_pt':  np.array([ 0.22585,  1.54623,  0.42657]),
    'spring_chassis_pt': np.array([ 0.03545,  1.54623,  0.39817]),
    # Axis point = pivot + 1 in Y  => rocker pivots about Y-parallel axis
    'rocker_axis_pt':    np.array([ 0.23708,  1.57163,  0.35118]),
}

# Rear ARB (left / RL side; absolute Y coords, no offset)
DEFAULT_REAR_ARB = {
    'arb_drop_top':  np.array([ 0.27518,  1.54623,  0.35118]),
    'arb_arm_end':   np.array([ 0.27518,  1.54623,  0.23178]),
    'arb_pivot':     np.array([ 0.27518,  1.65100,  0.23178]),
}

# Per-corner plot colors — yellow/red/white/blue (user preference).
CORNER_PLOT_COLORS = {
    'FL': '#FFD600',   # yellow
    'FR': '#E53935',   # red
    'RL': '#FFFFFF',   # white
    'RR': '#42A5F5',   # blue
}


# ==============================================================================
#  HELPERS
# ==============================================================================

def _hp_obj(d: dict) -> DoubleWishboneHardpoints:
    """Build a DoubleWishboneHardpoints from a plain dict (metres)."""
    return DoubleWishboneHardpoints(**{k: np.array(v, float) for k, v in d.items()})


def _mirror_x(d: dict) -> dict:
    """Negate X of all points -> opposite side of car."""
    return {k: v * np.array([-1., 1., 1.]) for k, v in d.items()}


def _offset_y(d: dict, dy: float) -> dict:
    """Shift all points in Y (kept for legacy / front ARB use)."""
    return {k: v + np.array([0., dy, 0.]) for k, v in d.items()}


def _state_to_pts(state: SolvedState, hp_dict: dict) -> dict:
    pts = {k: v.copy() for k, v in hp_dict.items()}
    mp  = state.all_moving_points()
    pts.update({
        'uca_outer':        mp['uca_outer'],
        'lca_outer':        mp['lca_outer'],
        'tie_rod_outer':    mp['tr_outer'],
        'wheel_center':     mp['wheel_center'],
        'pushrod_outer':    mp['pushrod_outer'],
        'pushrod_inner':    mp['pushrod_inner'],
        'rocker_spring_pt': mp['rocker_spring_pt'],
    })
    return pts


def _all_metrics(state: SolvedState, side: str,
                 spring_prev=None, travel_prev=None, **extra) -> dict:
    m   = KinematicMetrics(state, side)
    out = {}
    for entry in CATALOG:
        try:
            out[entry['key']] = entry['fn'](
                m, spring_prev=spring_prev, travel_prev=travel_prev, **extra)
        except Exception:
            out[entry['key']] = float('nan')
    return out


def _rack_travel_from_angle(steer_wheel_deg: float, steer_params: dict) -> float:
    """
    Rack translation in metres from steering wheel angle.
    Clamped by both total_rack_travel_mm (symmetric) and max_rack_travel_in.
    """
    ratio    = steer_params.get('rack_travel_per_rev_mm', 60.0)
    total    = steer_params.get('total_rack_travel_mm', 120.0)
    max_in   = steer_params.get('max_rack_travel_in', 2.5)
    max_mm   = max_in * 25.4   # inches -> mm
    # effective half-travel is the smaller of the two limits
    half     = min(total / 2.0, max_mm)
    travel_mm = steer_wheel_deg * ratio / 360.0
    travel_mm = float(np.clip(travel_mm, -half, half))
    return travel_mm / 1000.0   # -> metres


# ==============================================================================
#  CURVES CANVAS
# ==============================================================================

class CurvesCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor='#000000')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._vlines:    list = []   # axvline Line2D per subplot
        self._all_axes:  list = []   # current subplot axes
        self._plot_data: list = []   # [(ax, [(x, y, label, color), ...]), ...]
        self._hover_ann        = None
        self._bg               = None  # blitting background cache
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        # Cache background after every full redraw (handles resize automatically)
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)

    def plot(self, x_arr, x_label, results_per_corner, selected_keys, title,
             corners=None):
        self._hover_ann = None  # cleared by fig.clf() below
        self.fig.clf()
        n = len(selected_keys)
        if n == 0:
            self.draw()
            return

        # Filter by selected corners
        if corners is not None:
            results_per_corner = {k: v for k, v in results_per_corner.items()
                                  if k in corners}

        # ── Compute valid x-range from spring_len (stroke limits) ─────────────
        # spring_len is NaN outside the stroke → use it to find the trimmed range.
        x_lo, x_hi = x_arr[0], x_arr[-1]   # full sweep extent
        v_lo, v_hi = x_arr[-1], x_arr[0]   # valid data extent (inverted start)
        for res in results_per_corner.values():
            sp = res.get('spring_len')
            if sp is None:
                continue
            valid = ~np.isnan(sp)
            if not valid.any():
                continue
            idxs = np.where(valid)[0]
            v_lo = min(v_lo, x_arr[idxs[0]])
            v_hi = max(v_hi, x_arr[idxs[-1]])
        # If we found a tighter range, use it; otherwise fall back to full range
        if v_lo < v_hi:
            xlim = (v_lo - (x_hi - x_lo) * 0.02,
                    v_hi + (x_hi - x_lo) * 0.02)
            range_txt = f'[{v_lo:+.0f}, {v_hi:+.0f}]'
        else:
            xlim = (x_lo, x_hi)
            range_txt = None

        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        self.fig.subplots_adjust(
            hspace=0.72, wspace=0.40,
            left=0.09, right=0.97, top=0.90, bottom=0.10)

        styles = {
            'FL': (CORNER_PLOT_COLORS['FL'], '-'),
            'FR': (CORNER_PLOT_COLORS['FR'], '--'),
            'RL': (CORNER_PLOT_COLORS['RL'], '-.'),
            'RR': (CORNER_PLOT_COLORS['RR'], ':'),
        }

        for idx, key in enumerate(selected_keys):
            entry = CATALOG_MAP.get(key)
            if not entry:
                continue
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor('#080808')
            ax.tick_params(colors='#777777', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#222222')
            ax.set_ylabel(f'{entry["label"]}\n({entry["unit"]})',
                          color='#888888', fontsize=8, labelpad=2)
            ax.set_xlabel(x_label, color='#888888', fontsize=8, labelpad=2)
            ax.grid(True, color='#1a1a1a', lw=0.5)
            ax.axvline(0, color='#333333', lw=0.8, ls=':')
            ax.set_xlim(*xlim)

            for lbl, (color, ls) in styles.items():
                if lbl not in results_per_corner:
                    continue
                res = results_per_corner[lbl]
                if key in res:
                    ax.plot(x_arr, res[key], color=color, lw=1.8,
                            ls=ls, label=lbl)
            if len(results_per_corner) > 1:
                ax.legend(fontsize=7, facecolor='#06060e', labelcolor='white',
                          framealpha=0.7, loc='best', handlelength=1.0, ncol=2)

            # Range annotation below each subplot (stroke limits)
            if range_txt:
                ax.annotate(range_txt,
                            xy=(0.5, -0.28), xycoords='axes fraction',
                            ha='center', va='top',
                            fontsize=7, color='#555555')

        self.fig.suptitle(title, color='#cccccc', fontsize=9, y=0.98)

        # ── Vertical snap line (one per subplot, initially hidden) ────────
        self._all_axes  = []
        self._vlines    = []
        self._plot_data = []
        for ax in self.fig.axes:
            self._all_axes.append(ax)
            vl = ax.axvline(x=float('nan'), color='#ffffff', lw=0.8,
                            ls='--', alpha=0.5, zorder=10)
            self._vlines.append(vl)
            # Collect curve data for hover
            series = []
            for line in ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith('_'):
                    continue
                xd = line.get_xdata()
                yd = line.get_ydata()
                series.append((xd, yd, lbl, line.get_color()))
            self._plot_data.append((ax, series))

        self.draw()

    def _on_draw(self, event):
        """Cache the fully-rendered background for blitting (fires after every draw)."""
        try:
            self._bg = self.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg = None

    def _blit_overlay(self):
        """
        Fast update: restore cached background then draw only vlines + annotation.
        Falls back to draw_idle() if the cache is stale/missing.
        """
        if self._bg is None or not self._all_axes:
            self.draw_idle()
            return
        try:
            self.restore_region(self._bg)
            for ax, vl in zip(self._all_axes, self._vlines):
                ax.draw_artist(vl)
            if self._hover_ann is not None:
                self._hover_ann.axes.draw_artist(self._hover_ann)
            self.blit(self.fig.bbox)
        except Exception:
            self._bg = None
            self.draw_idle()

    def set_vline(self, x_val):
        """Move the vertical snap line — uses blitting for zero-lag response."""
        for vl in self._vlines:
            vl.set_xdata([x_val, x_val])
        self._blit_overlay()

    def _on_hover(self, event):
        """Show value annotation when hovering over a curve."""
        if event.inaxes is None:
            if self._hover_ann is not None:
                try:
                    self._hover_ann.remove()
                except Exception:
                    pass
                self._hover_ann = None
                self._blit_overlay()
            return

        ax = event.inaxes
        series = None
        for stored_ax, s in self._plot_data:
            if stored_ax is ax:
                series = s
                break
        if not series:
            return

        x_mouse = event.xdata
        if x_mouse is None:
            return
        nearest_idx = None
        for xd, yd, *_ in series:
            if len(xd) == 0:
                continue
            nearest_idx = int(np.argmin(np.abs(np.asarray(xd, float) - x_mouse)))
            break
        if nearest_idx is None:
            return

        lines = []
        xd_ref = None
        for xd, yd, lbl, color in series:
            if nearest_idx < len(yd):
                yv = yd[nearest_idx]
                if not np.isnan(yv):
                    lines.append(f'{lbl}: {yv:.3g}')
                    if xd_ref is None:
                        xd_ref = xd
        if not lines:
            return

        x_ann = float(xd_ref[nearest_idx]) if xd_ref is not None else x_mouse
        txt   = '\n'.join(lines)

        if self._hover_ann is not None:
            try:
                self._hover_ann.remove()
            except Exception:
                pass
            self._hover_ann = None

        self._hover_ann = ax.annotate(
            txt,
            xy=(x_ann, event.ydata),
            xytext=(8, 8), textcoords='offset points',
            fontsize=7, color='#e0e0e0',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a', ec='#444444', alpha=0.85),
            zorder=20,
        )
        self._blit_overlay()

    def plot_dynamics(self, sweep: dict, graphs: list | None = None,
                      corners: list | None = None,
                      turn_radius_m: float = 0.0,
                      wheelbase_m: float = 1.53,
                      steer_ratio: float = 0.0,
                      power_W: float = 0.0,
                      mass_kg: float = 290.0):
        """Plot dynamics sweep results with selectable graphs and corners."""
        self._hover_ann = None
        self.fig.clf()

        # Determine x axis (lateral or longitudinal sweep)
        is_longitudinal = 'longitudinal_g' in sweep and 'lateral_g' not in sweep
        if is_longitudinal:
            g_arr = sweep['longitudinal_g']
            x_label = 'Longitudinal g'
        else:
            g_arr = sweep['lateral_g']
            x_label = 'Lateral g'

        if corners is None:
            corners = ['FL', 'FR', 'RL', 'RR']
        if graphs is None:
            graphs = ['fz', 'roll', 'travel', 'lt', 'utilization']

        _C = dict(CORNER_PLOT_COLORS)
        _LS = {'FL': '-', 'FR': '--', 'RL': '-.', 'RR': ':'}

        # Understeer data (needed for steer_correction / path_deviation)
        us = sweep.get('understeer_gradient_deg')

        # Build list of (title, ylabel, series) based on selected graphs
        plots = []

        if 'fz' in graphs:
            series = [(c, sweep[f'Fz_{c}'], _C[c], _LS[c]) for c in corners]
            plots.append(('Corner Loads', 'Fz (N)', series))

        if 'roll' in graphs:
            plots.append(('Roll Angle', 'Roll (deg)', [
                ('Roll', sweep['roll_angle_deg'], '#4FC3F7', '-'),
            ]))

        if 'pitch' in graphs:
            pa = sweep.get('pitch_angle_deg')
            if pa is not None:
                plots.append(('Pitch Angle', 'Pitch (deg)', [
                    ('Pitch', pa, '#AB47BC', '-'),
                ]))

        if 'travel' in graphs:
            series = [(c, sweep[f'travel_{c}'], _C[c], _LS[c]) for c in corners]
            plots.append(('Suspension Travel', 'Travel (mm)', series))

        if 'camber' in graphs:
            series = [(c, sweep[f'camber_{c}'], _C[c], _LS[c]) for c in corners]
            plots.append(('Camber', 'Camber (deg)', series))

        if 'lt' in graphs:
            plots.append(('Load Transfer', 'LT (N)', [
                ('Elastic F', sweep['elastic_lt_front_N'], '#64B5F6', '-'),
                ('Elastic R', sweep['elastic_lt_rear_N'], '#0D47A1', '--'),
                ('Geo F', sweep['geometric_lt_front_N'], '#E53935', '-.'),
                ('Geo R', sweep['geometric_lt_rear_N'], '#AD1457', ':'),
            ]))

        if 'rc' in graphs:
            plots.append(('Roll Centre Height', 'RC (mm)', [
                ('Front', sweep['rc_height_front_mm'], '#42A5F5', '-'),
                ('Rear', sweep['rc_height_rear_mm'], '#4527A0', '--'),
            ]))

        if 'utilization' in graphs:
            series = [(c, sweep.get(f'utilization_{c}', np.zeros_like(g_arr)),
                        _C[c], _LS[c]) for c in corners]
            plots.append(('Tire Utilization', 'Utilization', series))
            # Add 1.0 reference line flag for this plot
            self._util_plot_idx = len(plots) - 1

        if 'understeer' in graphs:
            if us is not None and np.any(us):
                plots.append(('Understeer Gradient', 'SA_front − SA_rear (deg)', [
                    ('US Gradient', us, '#9575CD', '-'),
                ]))

        if 'steer_correction' in graphs:
            if us is not None and np.any(us) and wheelbase_m > 0:
                if turn_radius_m > 0:
                    ack_deg = np.degrees(wheelbase_m / turn_radius_m)
                    total_steer = ack_deg + us
                    if steer_ratio > 0:
                        # Show handwheel angle (what the driver actually turns)
                        hw_ack = np.full_like(g_arr, ack_deg * steer_ratio)
                        hw_req = total_steer * steer_ratio
                        plots.append(('Handwheel Angle', 'Steering wheel (deg)', [
                            ('Ackermann', hw_ack, '#555555', '--'),
                            ('Required', hw_req, '#4FC3F7', '-'),
                            ('Extra (US)', us * steer_ratio, '#BA68C8', '-.'),
                        ]))
                    else:
                        plots.append(('Steer Correction', 'Front wheel angle (deg)', [
                            ('Ackermann', np.full_like(g_arr, ack_deg), '#555555', '--'),
                            ('Required', total_steer, '#4FC3F7', '-'),
                            ('Extra (US)', us, '#BA68C8', '-.'),
                        ]))
                else:
                    plots.append(('Steer Correction', 'Extra steer (deg)', [
                        ('Extra (US)', us, '#BA68C8', '-'),
                    ]))

        if 'path_deviation' in graphs:
            if us is not None and np.any(us) and turn_radius_m > 0 and wheelbase_m > 0:
                # If driver inputs only Ackermann steer:
                # R_actual = R / (1 - R * Δα_rad / L)
                us_rad = np.radians(us)
                denom = 1.0 - turn_radius_m * us_rad / wheelbase_m
                denom = np.where(np.abs(denom) < 0.01, 0.01, denom)  # avoid div-by-0
                r_actual = turn_radius_m / denom
                deviation = r_actual - turn_radius_m
                # Clamp extreme values for readability
                deviation = np.clip(deviation, -50, 50)
                plots.append(('Path Deviation', f'Drift from {turn_radius_m:.0f}m radius (m)', [
                    ('Deviation', deviation, '#90CAF9', '-'),
                ]))

        if not plots:
            self.draw()
            return

        n = len(plots)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        # Extra top margin for speed axis
        show_speed = (not is_longitudinal and turn_radius_m > 0) or is_longitudinal
        top_margin = 0.86 if show_speed else 0.90
        self.fig.subplots_adjust(
            hspace=0.72, wspace=0.40,
            left=0.09, right=0.97, top=top_margin, bottom=0.10)

        self._all_axes  = []
        self._vlines    = []
        self._plot_data = []

        for idx, (title, ylabel, series) in enumerate(plots):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor('#080808')
            ax.tick_params(colors='#777777', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#222222')
            ax.set_ylabel(ylabel, color='#888888', fontsize=8, labelpad=2)
            ax.set_xlabel(x_label, color='#888888', fontsize=8, labelpad=2)
            ax.grid(True, color='#1a1a1a', lw=0.5)

            for lbl, ydata, color, ls in series:
                ax.plot(g_arr, ydata, color=color, lw=1.8, ls=ls, label=lbl)

            # Reference lines
            if title == 'Understeer Gradient':
                ax.axhline(y=0, color='#555555', lw=0.8, ls='--', alpha=0.6)
            if title == 'Tire Utilization':
                ax.axhline(y=1.0, color='#B0BEC5', lw=1.0, ls='--', alpha=0.7,
                            label='_grip limit')
            if title == 'Path Deviation':
                ax.axhline(y=0, color='#555555', lw=0.8, ls='--', alpha=0.6)

            ax.legend(fontsize=7, facecolor='#06060e', labelcolor='white',
                      framealpha=0.7, loc='best', handlelength=1.0, ncol=2)

            # Speed secondary x-axis on all subplots
            if show_speed:
                if not is_longitudinal and turn_radius_m > 0:
                    R = turn_radius_m
                    def _g_to_mph(g, R=R):
                        return np.sqrt(np.maximum(g, 0) * 9.81 * R) * 2.23694
                    def _mph_to_g(mph, R=R):
                        v = mph / 2.23694
                        return v**2 / (9.81 * R) if R > 0 else 0.0
                elif is_longitudinal and power_W > 0 and mass_kg > 0:
                    P, M = power_W, mass_kg
                    def _g_to_mph(g, P=P, M=M):
                        g = np.asarray(g, float)
                        F = np.maximum(g, 0.01) * M * 9.81
                        return np.where(F > 0, (P / F) * 2.23694, 0.0)
                    def _mph_to_g(mph, P=P, M=M):
                        v = max(mph / 2.23694, 0.01)
                        return P / (M * 9.81 * v) if v > 0 else 0.0
                else:
                    _g_to_mph = _mph_to_g = None
                try:
                    if _g_to_mph is not None:
                        secax = ax.secondary_xaxis('top',
                                                   functions=(_g_to_mph, _mph_to_g))
                        secax.set_xlabel('Speed (mph)', color='#4FC3F7',
                                        fontsize=7, labelpad=2)
                        secax.tick_params(colors='#4FC3F7', labelsize=7)
                except Exception:
                    pass  # older matplotlib may not support this

            self._all_axes.append(ax)
            vl = ax.axvline(x=float('nan'), color='#ffffff', lw=0.8,
                            ls='--', alpha=0.5, zorder=10)
            self._vlines.append(vl)
            ax_series = []
            for line in ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith('_'):
                    continue
                ax_series.append((line.get_xdata(), line.get_ydata(),
                                  lbl, line.get_color()))
            self._plot_data.append((ax, ax_series))

        fixed_lon = sweep.get('fixed_longitudinal_g')
        if fixed_lon is not None:
            sweep_type = f'Combined (lon={fixed_lon:+.1f}g)'
        elif is_longitudinal:
            sweep_type = 'Longitudinal'
        else:
            sweep_type = 'Lateral'
        self.fig.suptitle(f'Dynamics Sweep ({sweep_type})',
                          color='#cccccc', fontsize=9, y=0.98)
        self.draw()


# ==============================================================================
#  IK SOLVER WORKER (runs in a background QThread)
# ==============================================================================

class _IKWorker(QThread):
    """Runs InverseSolver.solve() off the main thread."""
    finished = Signal(dict)    # result dict on success
    failed   = Signal(str)     # error string on failure
    status   = Signal(str)     # progress messages

    def __init__(self, solver: InverseSolver, method: str):
        super().__init__()
        self._solver = solver
        self._method = method

    def run(self):
        try:
            result = self._solver.solve(
                method=self._method,
                progress_cb=lambda msg: self.status.emit(msg),
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class _IKExploreWorker(QThread):
    """Runs multiple IK solves in parallel using warm-start LM."""
    finished = Signal(list)    # list of result dicts
    failed   = Signal(str)
    status   = Signal(str)

    def __init__(self, solver_kwargs: dict, bound_levels: list[float],
                 warm_x: np.ndarray):
        """
        solver_kwargs: serialisable dict with all info to rebuild InverseSolver
        bound_levels: list of bound_mm values to try
        warm_x: best x from initial solve (used as LM starting point)
        """
        super().__init__()
        self._solver_kwargs = solver_kwargs
        self._bounds = bound_levels
        self._warm_x = warm_x

    def run(self):
        from concurrent.futures import ThreadPoolExecutor
        from vahan.optimizer import _solve_at_bound

        try:
            tasks = [
                (self._solver_kwargs, bnd, self._warm_x.tolist(),
                 f'+-{bnd:.0f}mm')
                for bnd in self._bounds
            ]

            self.status.emit(
                f'Solving {len(tasks)} bound levels in parallel...')

            # ThreadPool avoids Windows multiprocessing spawn overhead.
            # numpy/scipy release the GIL during C-level math so
            # threads get real parallelism for the heavy computation.
            with ThreadPoolExecutor() as pool:
                solutions = list(pool.map(_solve_at_bound, tasks))

            solutions.sort(key=lambda r: r['cost'])
            self.finished.emit(solutions)
        except Exception as e:
            self.failed.emit(str(e))


class _DynamicsSolveWorker(QThread):
    """Runs SteadyStateSolver.solve() off the main thread."""
    finished = Signal(object)  # SteadyStateResult
    failed   = Signal(str)

    def __init__(self, solver: SteadyStateSolver, lateral_g: float,
                 longitudinal_g: float = 0.0):
        super().__init__()
        self._solver = solver
        self._lat_g = lateral_g
        self._lon_g = longitudinal_g

    def run(self):
        try:
            result = self._solver.solve(self._lat_g, self._lon_g)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class _DynamicsSweepWorker(QThread):
    """Runs lateral or longitudinal sweep off the main thread."""
    finished = Signal(dict)   # sweep arrays
    failed   = Signal(str)

    def __init__(self, solver: SteadyStateSolver, g_min: float, g_max: float,
                 n_points: int, longitudinal_g: float = 0.0,
                 mode: str = 'lateral', lateral_g: float = 0.0):
        super().__init__()
        self._solver = solver
        self._g_min = g_min
        self._g_max = g_max
        self._n = n_points
        self._lon_g = longitudinal_g
        self._lat_g = lateral_g
        self._mode = mode

    def run(self):
        try:
            if self._mode == 'combined':
                result = self._solver.sweep_combined(
                    lat_range=(self._g_min, self._g_max),
                    lon_g=self._lon_g,
                    n_points=self._n)
            elif self._mode == 'longitudinal':
                result = self._solver.sweep_longitudinal_g(
                    g_range=(self._g_min, self._g_max),
                    n_points=self._n,
                    lateral_g=self._lat_g)
            else:
                result = self._solver.sweep_lateral_g(
                    g_range=(self._g_min, self._g_max),
                    n_points=self._n,
                    longitudinal_g=self._lon_g)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class _SensitivityWorker(QThread):
    """Runs dynamics sensitivity analysis off the main thread."""
    finished = Signal(dict)
    failed   = Signal(str)

    def __init__(self, sens: DynamicsSensitivity,
                 lateral_g: float, longitudinal_g: float):
        super().__init__()
        self._sens = sens
        self._lat_g = lateral_g
        self._lon_g = longitudinal_g

    def run(self):
        try:
            result = self._sens.analyze(self._lat_g, self._lon_g)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


# ==============================================================================
#  MAIN WINDOW
# ==============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Vahan -- Suspension Kinematics')
        self.resize(1500, 900)

        # state
        self._front_hp  = {k: v.copy() for k, v in DEFAULT_FRONT_HP.items()}
        self._rear_hp   = {k: v.copy() for k, v in DEFAULT_REAR_HP.items()}
        self._front_arb = {k: v.copy() for k, v in DEFAULT_FRONT_ARB.items()}
        self._rear_arb  = {k: v.copy() for k, v in DEFAULT_REAR_ARB.items()}
        self._car       = {'axle_spacing_mm': 1537., 'wheelbase_mm': 1537.,
                           'track_f_mm': 1222., 'track_r_mm': 1200.,
                           'wheel_offset_f_mm': 25., 'wheel_offset_r_mm': 25.,
                           'tire_outer_dia_mm': 406., 'tire_rim_dia_mm': 330.,
                           'tire_width_mm': 200., 'show_ground': True,
                           'cg_x_mm': 0., 'cg_y_mm': 1100., 'cg_z_mm': 280.,
                           'front_brake_bias_pct': 65.}
        self._steer     = {'rack_travel_per_rev_mm': 60.,
                           'total_rack_travel_mm': 100.,
                           'max_rack_travel_in': 2.0}
        self._selected_keys    = list(DEFAULT_Y_KEYS)
        self._selected_corners = ['FL', 'FR', 'RL', 'RR']
        self._solvers: dict[str, SuspensionConstraints] = {}
        self._sweep_results: dict[str, dict] = {}
        self._x_arr   = np.zeros(2)
        self._x_label = 'Wheel Travel (mm)'
        self._alignment   = {'front_toe_deg': 0., 'front_camber_deg': 0.,
                              'rear_toe_deg':  0., 'rear_camber_deg':  0.}
        self._last_valid_st: dict = {}   # label → last SolvedState within spring limits
        self._show_rc        = True
        self._show_roll_axis = True
        self._3d_pending     = False     # deferred 3D update flag
        self._tire_model     = None      # TireModel or LinearTireModel
        self._dyn_sweep_data = None      # last dynamics sweep dict
        self._dyn_worker     = None      # active dynamics worker thread

        self._build_ui()
        self._apply_style()
        # Centre the camera orbit pivot at the car midpoint
        wb_half = self._car['axle_spacing_mm'] / 2000.  # half axle spacing in metres
        self.view3d.set_camera_center((0., wb_half, 0.2))
        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()
        self._try_autoload_tire()
        self._update_min_turn_radius()
        # Push initial aero geometry to 3D viewer
        self._on_aero_geom(self._aero_geom_panel.params())

    # ==========================================================================
    #  BUILD UI
    # ==========================================================================

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('File')
        save_act = fm.addAction('Save Project…')
        load_act = fm.addAction('Load Project…')
        save_act.triggered.connect(self._save_project)
        load_act.triggered.connect(self._load_project)

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Vahan Project', '', 'Vahan Project (*.vahan);;JSON (*.json)')
        if not path:
            return
        mp = self._motion_panel
        data = {
            'version': 1,
            'front_hp':  {k: v.tolist() for k, v in self._front_hp.items()},
            'rear_hp':   {k: v.tolist() for k, v in self._rear_hp.items()},
            'front_arb': {k: v.tolist() for k, v in self._front_arb.items()},
            'rear_arb':  {k: v.tolist() for k, v in self._rear_arb.items()},
            'car':       self._car.copy(),
            'steer':     self._steer.copy(),
            'alignment': self._alignment.copy(),
            'motion': {
                'type':      mp.motion,
                'min':       mp.min_val,
                'max':       mp.max_val,
                'stroke_mm': self._motion_panel._stroke.value(),
                'sag_pct':   self._motion_panel._sag.value(),
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f'Saved: {path}', 4000)

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Vahan Project', '', 'Vahan Project (*.vahan);;JSON (*.json)')
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)

            def _arr(d, key):
                return {k: np.array(v, float) for k, v in d[key].items()}

            self._front_hp  = _arr(data, 'front_hp')
            self._rear_hp   = _arr(data, 'rear_hp')
            self._front_arb = _arr(data, 'front_arb')
            self._rear_arb  = _arr(data, 'rear_arb')
            car_data = data.get('car', {})
            # backward compat: old files have cg_height_mm → cg_z_mm
            if 'cg_height_mm' in car_data and 'cg_z_mm' not in car_data:
                car_data['cg_z_mm'] = car_data.pop('cg_height_mm')
                car_data.setdefault('cg_x_mm', 0.)
                car_data.setdefault('cg_y_mm', 1100.)
            # backward compat: old files without axle_spacing / wheel_offset
            if 'axle_spacing_mm' not in car_data:
                car_data['axle_spacing_mm'] = car_data.get('wheelbase_mm', 1537.)
            car_data.setdefault('wheel_offset_f_mm', 25.)
            car_data.setdefault('wheel_offset_r_mm', 25.)
            self._car.update(car_data)
            self._steer.update(data.get('steer', {}))
            self._alignment.update(data.get('alignment', {}))

            self._front_hp_panel.refresh(self._front_hp, self._front_arb)
            self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb)
            self._car_panel.set_params(self._car)

            self._rebuild_solvers()
            self._run_sweep()
            self._update_3d()
            self.statusBar().showMessage(f'Loaded: {path}', 4000)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    def _build_ui(self):
        self._build_menu()
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 3D + curves
        self.view3d = View3D()
        self.view3d.set_on_pick(self._on_pick)

        self.curves = CurvesCanvas()

        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(self.view3d.native)
        left_split.addWidget(self.curves)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 2)

        # right sidebar
        self._motion_panel   = MotionPanel()
        self._steer_panel    = SteeringPanel()
        self._car_panel      = CarParamsPanel()
        self._alignment_panel = AlignmentPanel()
        self._graph_panel    = GraphPickerPanel()
        self._front_hp_panel = HardpointPanel('Front Hardpoints', self._front_hp, self._front_arb)
        self._rear_hp_panel  = HardpointPanel('Rear Hardpoints',  self._rear_hp,  self._rear_arb)
        self._values_panel   = ValuesPanel()

        # 3D overlay toggles (collapsible)
        self._overlay_box = CollapsibleSection('3D Overlays', header_color='#cccccc')
        self._chk_rc   = QCheckBox('Roll Centres')
        self._chk_ra   = QCheckBox('Roll Axis')
        self._chk_rc.setChecked(True)
        self._chk_ra.setChecked(True)
        self._overlay_box.add_widget(self._chk_rc)
        self._overlay_box.add_widget(self._chk_ra)
        self._chk_rc.toggled.connect(lambda v: (
            setattr(self, '_show_rc', v),        self._update_3d()))
        self._chk_ra.toggled.connect(lambda v: (
            setattr(self, '_show_roll_axis', v), self._update_3d()))

        # Left sidebar (existing controls)
        sidebar_inner = QWidget()
        sv = QVBoxLayout(sidebar_inner)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(4)
        for w in [self._motion_panel, self._steer_panel, self._alignment_panel,
                  self._car_panel, self._overlay_box, self._graph_panel,
                  self._front_hp_panel, self._rear_hp_panel, self._values_panel]:
            sv.addWidget(w)

        left_scroll = QScrollArea()
        left_scroll.setWidget(sidebar_inner)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(220)

        # Right sidebar (IK + Dynamics panels)
        self._ik_panel = InverseKinematicsPanel()
        self._dynamics_panel = DynamicsPanel()
        self._dynamics_opt_panel = DynamicsOptPanel()
        self._aero_panel = AeroPanel()
        self._aero_geom_panel = AeroGeomPanel()
        self._loads_panel = LoadsPanel()
        ik_inner = QWidget()
        ik_layout = QVBoxLayout(ik_inner)
        ik_layout.setContentsMargins(0, 0, 0, 0)
        ik_layout.setSpacing(4)
        ik_layout.addWidget(self._ik_panel)
        ik_layout.addWidget(self._dynamics_panel)
        ik_layout.addWidget(self._dynamics_opt_panel)
        ik_layout.addWidget(self._aero_panel)
        ik_layout.addWidget(self._aero_geom_panel)
        ik_layout.addWidget(self._loads_panel)
        ik_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidget(ik_inner)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(200)

        # Layout: [left sidebar | 3D+curves | right sidebar]
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(left_scroll)
        h_split.addWidget(left_split)
        h_split.addWidget(right_scroll)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        h_split.setStretchFactor(2, 0)
        h_split.setSizes([300, 900, 280])
        root.addWidget(h_split)

        self.setStatusBar(QStatusBar())

        # signals
        self._motion_panel.motion_changed.connect(self._on_sweep_trigger)
        self._motion_panel.range_changed.connect(self._on_sweep_trigger)
        self._motion_panel.position_changed.connect(self._on_position)
        self._steer_panel.steering_changed.connect(self._on_steer)
        self._car_panel.params_changed.connect(self._on_car)
        self._front_hp_panel.hp_changed.connect(lambda d: self._on_hp(d, 'front'))
        self._rear_hp_panel.hp_changed.connect(lambda d: self._on_hp(d, 'rear'))
        self._front_hp_panel.row_selected.connect(self._on_row)
        self._rear_hp_panel.row_selected.connect(self._on_row)
        self._graph_panel.selection_changed.connect(self._on_graph_sel)
        self._graph_panel.corners_changed.connect(self._on_corners_sel)
        self._alignment_panel.alignment_changed.connect(self._on_alignment)
        self._ik_panel.solve_requested.connect(self._on_ik_solve)
        self._ik_panel.apply_requested.connect(self._on_ik_apply)
        self._dynamics_panel.solve_requested.connect(self._on_dynamics_solve)
        self._dynamics_panel.sweep_requested.connect(self._on_dynamics_sweep)
        self._dynamics_panel.tire_file_changed.connect(self._on_tire_file)
        self._dynamics_panel.graph_selection_changed.connect(self._on_dyn_graph_sel)
        self._dynamics_panel.corners_changed.connect(self._on_dyn_corners_sel)
        self._dynamics_opt_panel.analyze_requested.connect(self._on_sensitivity_analyze)
        self._aero_panel.solve_requested.connect(self._on_aero_solve)
        self._aero_panel.sweep_requested.connect(self._on_aero_sweep)
        self._aero_geom_panel.aero_geom_changed.connect(self._on_aero_geom)
        self._loads_panel.loads_requested.connect(self._on_compute_loads)
        self._motion_panel.damper_params_changed.connect(self._on_damper_limits)
        # Push initial damper limits to IK panel
        self._on_damper_limits({
            'stroke_mm': self._motion_panel._stroke.value(),
            'sag_pct':   self._motion_panel._sag.value(),
        })

    # ==========================================================================
    #  CORNER HP DICTS IN WORLD FRAME
    # ==========================================================================

    def _all_corner_hp(self) -> dict[str, dict]:
        """
        Return world-frame hardpoint dicts for all four corners.
        Front: small Y offsets from axle centre.
        Rear:  absolute Y coords -- NO wheelbase offset applied.
        Alignment (camber/toe) is applied as metric offsets post-solve,
        not as hardpoint modifications.
        """
        fl = self._front_hp
        fr = _mirror_x(fl)
        rl = self._rear_hp
        rr = _mirror_x(rl)
        return {'FL': fl, 'FR': fr, 'RL': rl, 'RR': rr}

    def _steered_hp(self, hp: dict, rack_travel_m: float, is_front: bool,
                    mirror: bool = False) -> dict:
        """
        Apply rack translation to tie_rod_inner on front axle only.

        The rack is a rigid body -- both ends move the same amount in world X.
        FL: outboard = +X, tie_rod_inner.x += rack_travel_m
        FR: hardpoints already have X negated, but the rack still moves +rack_travel_m
            in world X, so FR tie_rod_inner.x += rack_travel_m too.
        Both corners get the identical shift -- no sign flip on FR.
        """
        if not is_front:
            return hp
        out = {k: v.copy() for k, v in hp.items()}
        out['tie_rod_inner'] = hp['tie_rod_inner'] + np.array([rack_travel_m, 0., 0.])
        return out

    # ==========================================================================
    #  SOLVERS
    # ==========================================================================

    def _spring_limits(self, solver: SuspensionConstraints) -> tuple[float, float]:
        """
        Return (spring_min_m, spring_max_m) based on stroke and static sag.

        At design position (travel=0):
            spring_0 = design spring length
        Static sag: car sits at sag_pct % of stroke from full droop.
            full_droop_spring = spring_0 + sag_pct/100 * stroke
            full_bump_spring  = full_droop_spring - stroke
        """
        try:
            st0 = solver.solve(0.)
            spring_0  = st0.spring_length
            stroke_m  = self._motion_panel._stroke.value() / 1000.
            sag_pct   = self._motion_panel._sag.value() / 100.
            droop_len = spring_0 + sag_pct * stroke_m
            bump_len  = droop_len - stroke_m
            return bump_len, droop_len
        except Exception:
            return 0., 1.

    def _rebuild_solvers(self, steer_angle_deg: float = 0.0):
        """
        Rebuild all 4 corner solvers.
        steer_angle_deg: current steering wheel angle (used in Steer sweep mode).
        At design (heave/roll/pitch), this is 0.
        """
        try:
            corners = self._all_corner_hp()
            rt = _rack_travel_from_angle(steer_angle_deg, self._steer)
            for label, hp_d in corners.items():
                is_front = label in ('FL', 'FR')
                steered  = self._steered_hp(hp_d, rt, is_front)

                # Always use the DESIGN tie-rod length (before any rack travel).
                # Moving tie_rod_inner with the rack must not change the rod length.
                d = hp_d['tie_rod_outer'] - hp_d['tie_rod_inner']
                design_tierod_len_sq = float(d @ d)

                # Front: pushrod is mounted to UCA.  Rear: pushrod on LCA.
                pushrod_body = 'uca' if is_front else 'lca'

                self._solvers[label] = SuspensionConstraints(
                    _hp_obj(steered),
                    tierod_len_sq=design_tierod_len_sq,
                    pushrod_body=pushrod_body,
                )
            cp = self._car
            self.view3d.set_tire_params(
                outer_r = cp['tire_outer_dia_mm'] / 2000.,
                rim_r   = cp['tire_rim_dia_mm']   / 2000.,
                half_w  = cp['tire_width_mm']     / 2000.,
            )
        except Exception as e:
            self.statusBar().showMessage(f'Solver init: {e}', 6000)

    @staticmethod
    def _solve_arb_bellcrank(arb_drop_top_world: np.ndarray,
                              arb_hp: dict) -> tuple:
        """
        Solve for the ARB bell-crank angle given the current position of the
        drop-link rocker attachment point (arb_drop_top_world).

        The bell crank rotates about a lateral axis through arb_pivot.
        Its arm (arb_arm_end) traces a circle in the plane perpendicular to
        the torsion-bar axis.

        Returns (arb_angle_rad, arb_arm_end_world, drop_link_travel_m).
        """
        pv   = arb_hp['arb_pivot']        # chassis fixed
        ae0  = arb_hp['arb_arm_end']      # design position of arm end
        dt0  = arb_hp['arb_drop_top']     # design position of drop-link end on rocker

        # Torsion bar axis: lateral (X), derived from pivot toward car centre
        # (flip_x convention means left-side pivot has +X and right-side −X)
        bc_axis = np.array([0., 1., 0.])  # Y axis (bar runs longitudinally)
        # Actually ARB torsion bar is lateral (X axis) in our convention
        bc_axis = np.array([1., 0., 0.])

        arm_vec  = ae0 - pv                       # design arm in world
        arm_len2 = float(arm_vec @ arm_vec)
        if arm_len2 < 1e-12:
            return 0., ae0.copy(), 0.

        # Drop-link length is fixed (design)
        dl_vec0  = dt0 - ae0
        dl_len2  = float(dl_vec0 @ dl_vec0)

        # 1-D Newton: find angle θ such that
        # |pv + R(bc_axis, θ)@arm_vec - arb_drop_top_world|² = dl_len²
        theta = 0.0
        for _ in range(60):
            arm_rot  = _rodrigues(arm_vec, bc_axis, theta)
            ae_world = pv + arm_rot
            diff     = ae_world - arb_drop_top_world
            res      = float(diff @ diff) - dl_len2
            if abs(res) < 1e-14:
                break
            d_arm = np.cross(bc_axis, arm_rot)
            drdt  = float(2.0 * diff @ d_arm)
            if abs(drdt) < 1e-14:
                break
            theta -= res / drdt
            theta = max(-np.pi / 2, min(np.pi / 2, theta))

        ae_world      = pv + _rodrigues(arm_vec, bc_axis, theta)
        drop_link_travel = float(np.linalg.norm(ae_world - arb_drop_top_world)
                                 - np.sqrt(dl_len2))
        return theta, ae_world, drop_link_travel

    def _run_sweep(self):
        mp     = self._motion_panel
        motion = mp.motion
        lo, hi = mp.min_val, mp.max_val
        n      = 81
        try:
            _flip_x = np.array([-1., 1., 1.])
            def _arb(lbl):
                src = self._front_arb if lbl in ('FL', 'FR') else self._rear_arb
                if lbl in ('FR', 'RR'):
                    return {k: v * _flip_x for k, v in src.items()}
                return src

            # Alignment offsets (applied post-solve as measurement shifts).
            a = self._alignment
            def _calign(lbl):
                return a['front_camber_deg'] if lbl in ('FL','FR') else a['rear_camber_deg']
            def _talign(lbl):
                return a['front_toe_deg']    if lbl in ('FL','FR') else a['rear_toe_deg']

            def _sweep(lbl, t):
                return self._do_sweep(
                    self._solvers[lbl], t,
                    'left' if lbl in ('FL', 'RL') else 'right',
                    arb_hp=_arb(lbl),
                    camber_off=_calign(lbl), toe_off=_talign(lbl),
                    is_front=lbl in ('FL', 'FR'),
                )

            if motion == 'heave':
                t_arr   = np.linspace(lo/1000, hi/1000, n)
                x_arr   = t_arr * 1000
                x_label = 'Wheel Travel (mm)'
                sweeps  = {lbl: t_arr for lbl in ('FL', 'FR', 'RL', 'RR')}
                self._rebuild_solvers(0.)
                self._sweep_results = {
                    lbl: _sweep(lbl, t)
                    for lbl, t in sweeps.items() if lbl in self._solvers
                }

            elif motion == 'roll':
                angles  = np.linspace(lo, hi, n)
                th      = self._front_hp['wheel_center'][0]
                t_l     =  np.sin(np.radians(angles)) * th
                t_r     = -t_l
                x_arr   = angles
                x_label = 'Roll Angle (deg)'
                sweeps  = {'FL': t_l, 'FR': t_r, 'RL': t_l, 'RR': t_r}
                self._rebuild_solvers(0.)
                self._sweep_results = {
                    lbl: _sweep(lbl, t)
                    for lbl, t in sweeps.items() if lbl in self._solvers
                }

            elif motion == 'pitch':
                t_arr   = np.linspace(lo/1000, hi/1000, n)
                x_arr   = t_arr * 1000
                x_label = 'Pitch Travel (mm)'
                sweeps  = {'FL': t_arr, 'FR': t_arr, 'RL': -t_arr, 'RR': -t_arr}
                self._rebuild_solvers(0.)
                self._sweep_results = {
                    lbl: _sweep(lbl, t)
                    for lbl, t in sweeps.items() if lbl in self._solvers
                }

            else:  # steer -- vary steering wheel angle, zero heave
                steer_angles = np.linspace(lo, hi, n)   # steering wheel deg
                x_arr        = steer_angles
                x_label      = 'Steering Wheel Angle (deg)'
                res_fl = {e['key']: np.full(n, float('nan')) for e in CATALOG}
                res_fr = {e['key']: np.full(n, float('nan')) for e in CATALOG}
                for i, ang in enumerate(steer_angles):
                    self._rebuild_solvers(ang)
                    for lbl, res, side in [('FL', res_fl, 'left'),
                                           ('FR', res_fr, 'right')]:
                        solver = self._solvers.get(lbl)
                        if not solver:
                            continue
                        try:
                            st = solver.solve(0.)
                            vals = _all_metrics(st, side)
                        except Exception:
                            continue
                        for key in res:
                            val = vals.get(key, float('nan'))
                            if key == 'camber' and not np.isnan(val):
                                val += _calign(lbl)
                            elif key == 'toe' and not np.isnan(val):
                                val += _talign(lbl)
                            res[key][i] = val
                self._sweep_results = {'FL': res_fl, 'FR': res_fr}
                # Restore solvers at current steer position
                cur_angle = mp.position if mp.motion == 'steer' else 0.
                self._rebuild_solvers(cur_angle)

            self._x_arr   = x_arr
            self._x_label = x_label

            # ── Post-process: compute proper axle roll centre ─────────────────
            # Replace per-corner rc_height (which diverges in roll mode because
            # each corner independently projects its IC to X=0) with the correct
            # axle roll centre: intersection of the left IC-to-CP line with the
            # right IC-to-CP line in the front view (XZ plane).
            for left_lbl, right_lbl in [('FL', 'FR'), ('RL', 'RR')]:
                lr = self._sweep_results.get(left_lbl, {})
                rr = self._sweep_results.get(right_lbl, {})
                l_ic_x = lr.get('_ic_fv_x')
                l_ic_z = lr.get('_ic_fv_z')
                l_cp_x = lr.get('_cp_x')
                r_ic_x = rr.get('_ic_fv_x')
                r_ic_z = rr.get('_ic_fv_z')
                r_cp_x = rr.get('_cp_x')
                if any(a is None for a in [l_ic_x, l_ic_z, l_cp_x,
                                            r_ic_x, r_ic_z, r_cp_x]):
                    continue
                n_steps = len(l_ic_x)
                axle_rc = np.full(n_steps, float('nan'))
                for i in range(n_steps):
                    vs = (l_ic_x[i], l_ic_z[i], l_cp_x[i],
                          r_ic_x[i], r_ic_z[i], r_cp_x[i])
                    if any(np.isnan(v) for v in vs):
                        continue
                    l_ic = np.array([l_ic_x[i], l_ic_z[i]])
                    r_ic = np.array([r_ic_x[i], r_ic_z[i]])
                    l_cp = np.array([l_cp_x[i], 0.0])
                    r_cp = np.array([r_cp_x[i], 0.0])
                    rc = _intersect_2d(l_ic, l_cp, r_ic, r_cp)
                    if rc is not None:
                        axle_rc[i] = rc[1] * 1000.  # Z in mm
                if left_lbl in self._sweep_results:
                    self._sweep_results[left_lbl]['rc_height'] = axle_rc.copy()
                if right_lbl in self._sweep_results:
                    self._sweep_results[right_lbl]['rc_height'] = axle_rc.copy()

            self._replot()
        except Exception as e:
            self.statusBar().showMessage(f'Sweep: {e}', 6000)
            import traceback; traceback.print_exc()

    # Keys whose valid values depend on the spring/rocker being within limits.
    # All other metrics (camber, toe, RC height, etc.) come from the main
    # Newton solver and are independent of whether the rocker/spring is OOB.
    _SPRING_KEYS = frozenset({'motion_ratio', 'spring_len', 'rocker_angle'})

    def _do_sweep(self, solver, travels, side, arb_hp=None,
                  camber_off=0., toe_off=0., is_front=True):
        """
        Sweep over wheel travel positions and record all kinematic metrics.

        Rocker branch-flip fix: sweep in TWO passes starting from t≈0 (design
        position) outward in each direction. This keeps the warm-start always
        incremental so the rocker Newton solver always tracks the correct
        geometric branch. A cold start from the most extreme droop/bump position
        frequently converges to the wrong branch, producing a near-constant
        (wrong) spring_length that makes MR ≈ 0 with spike artefacts.

        MR = |Δspring_length / Δwheel_travel|  (dimensionless, ≈ 0.5–1.0 typical).
        """
        out = {e['key']: np.full(len(travels), float('nan')) for e in CATALOG}
        # Hidden arrays for axle-level roll-centre post-processing
        out['_ic_fv_x'] = np.full(len(travels), float('nan'))  # front-view IC X (m)
        out['_ic_fv_z'] = np.full(len(travels), float('nan'))  # front-view IC Z (m)
        out['_cp_x']    = np.full(len(travels), float('nan'))  # contact-patch X (m)
        spring_min, spring_max = self._spring_limits(solver)
        spring_lens  = np.full(len(travels), float('nan'))
        travels_arr  = np.array([float(t) for t in travels])

        # Find index closest to t=0 (design position) to use as the warm-start seed
        mid_idx = int(np.argmin(np.abs(travels_arr)))

        def _sweep_pass(indices):
            x_w = None; th_w = 0.0; th_prev2 = None
            spring_prev = travel_prev = None
            rocker_spring_prev = None   # previous spring length for branch continuity

            for i in indices:
                t  = travels[i]
                direction = 0.0
                if th_prev2 is not None:
                    direction = float(np.sign(th_w - th_prev2))

                try:
                    st = solver.solve(float(t), x0=x_w,
                                      rocker_theta0=th_w,
                                      rocker_direction=direction,
                                      rocker_spring_prev=rocker_spring_prev)
                except Exception:
                    continue   # solver failed — keep warm-start from last success

                x_w = st.x_vec(); th_prev2 = th_w; th_w = st.rocker_angle
                rocker_spring_prev = st.spring_length   # update for next step

                spring_ok = spring_min <= st.spring_length <= spring_max
                if spring_ok:
                    spring_lens[i] = st.spring_length   # only record within stroke

                # Outside stroke limits: leave ALL metrics as NaN (trim the curve).
                # Warm-start variables still update so the solver stays on-track.
                if not spring_ok:
                    spring_prev = travel_prev = None
                    continue

                # ── ARB bell-crank ────────────────────────────────────────
                arb_kwargs = {}
                if arb_hp is not None:
                    try:
                        pivot  = solver.hp.rocker_pivot
                        ax_pt  = getattr(solver.hp, 'rocker_axis_pt',
                                         pivot + np.array([0., 0.0254, 0.]))
                        r_axis = _norm(ax_pt - pivot)
                        arm_dt = arb_hp['arb_drop_top'] - pivot
                        dt_w   = pivot + _rodrigues(arm_dt, r_axis, st.rocker_angle)
                        ang, _, dl_t = self._solve_arb_bellcrank(dt_w, arb_hp)
                        arb_kwargs = {
                            'arb_angle':       ang,
                            'arb_drop_travel': dl_t,
                            'arb_mr': min(abs(np.degrees(ang) / (float(t) * 1000)), 5.0) if abs(float(t)) > 1e-9 else float('nan'),
                        }
                    except Exception:
                        pass

                anti_kwargs = {
                    'cg_height_m':      self._car.get('cg_z_mm', 280.) / 1000.,
                    'wheelbase_m':      self._car.get('wheelbase_mm', 1537.) / 1000.,
                    'front_brake_bias': self._car.get('front_brake_bias_pct', 65.) / 100.,
                    'rear_drive_bias':  1.0,   # RWD assumed
                    'front_drive_bias': 0.0,   # RWD = no front drive
                }
                vals = _all_metrics(st, side, spring_prev, travel_prev,
                                    **arb_kwargs, **anti_kwargs)
                for key in out:
                    if key.startswith('_'):
                        continue
                    val = vals.get(key, float('nan'))
                    if key == 'camber' and not np.isnan(val):
                        val += camber_off
                    elif key == 'toe' and not np.isnan(val):
                        val += toe_off
                    out[key][i] = val

                # Store front-view IC for axle-level roll-centre post-processing.
                # Computed directly from SolvedState (same formula as roll_center_height).
                uca_in_xz = np.array([(st.uca_front[0]+st.uca_rear[0])/2,
                                       (st.uca_front[2]+st.uca_rear[2])/2])
                lca_in_xz = np.array([(st.lca_front[0]+st.lca_rear[0])/2,
                                       (st.lca_front[2]+st.lca_rear[2])/2])
                ic_fv = _intersect_2d(uca_in_xz,
                                      np.array([st.uca_outer[0], st.uca_outer[2]]),
                                      lca_in_xz,
                                      np.array([st.lca_outer[0], st.lca_outer[2]]))
                if ic_fv is not None:
                    out['_ic_fv_x'][i] = float(ic_fv[0])
                    out['_ic_fv_z'][i] = float(ic_fv[1])
                out['_cp_x'][i] = float(st.wheel_center[0])

                spring_prev = st.spring_length
                travel_prev = float(t)

        # Pass 1: mid → end  (positive travel direction, warm-start from design)
        _sweep_pass(range(mid_idx, len(travels)))
        # Pass 2: mid → start (negative travel direction)
        _sweep_pass(range(mid_idx, -1, -1))

        # ── Post-process MR ───────────────────────────────────────────────────
        # Cumulative MR = |Δdamper_length / Δwheel_travel| from design (t=0).
        # Using the cumulative ratio is far more stable than np.gradient:
        # no numerical differentiation noise, no branch-flip spikes, and gives
        # directly the "how many mm of damper per mm of wheel" value the user wants.
        valid = ~np.isnan(spring_lens)
        if valid.sum() >= 2:
            # Spring length at design position (t≈0)
            spring_0 = (spring_lens[mid_idx]
                        if not np.isnan(spring_lens[mid_idx])
                        else np.nanmedian(spring_lens))

            mr_full = np.full(len(travels_arr), float('nan'))
            nz = np.abs(travels_arr) > 1e-6   # avoid division by zero at t=0
            mr_full[valid & nz] = np.abs(
                (spring_lens[valid & nz] - spring_0) / travels_arr[valid & nz]
            )
            out['motion_ratio'] = mr_full

        return out

    def _update_3d(self):
        if not self._solvers:
            return
        try:
            mp     = self._motion_panel
            pos    = mp.position
            motion = mp.motion

            # ── rack travel (needed both for solver steer and for visual) ─────
            rt_m = _rack_travel_from_angle(
                pos if motion == 'steer' else 0., self._steer)

            if motion == 'steer':
                self._rebuild_solvers(pos)
                travels = {lbl: 0. for lbl in ('FL','FR','RL','RR')}
            elif motion == 'heave':
                travels = {lbl: pos/1000 for lbl in ('FL','FR','RL','RR')}
            elif motion == 'roll':
                th = self._front_hp['wheel_center'][0]
                t  = np.sin(np.radians(pos)) * th
                travels = {'FL': t, 'FR': -t, 'RL': t, 'RR': -t}
            else:  # pitch
                travels = {'FL': pos/1000, 'FR': pos/1000,
                           'RL': -pos/1000, 'RR': -pos/1000}

            corners_draw = []
            all_corner_values = {}
            hp_dicts     = self._all_corner_hp()
            flip_x       = np.array([-1., 1., 1.])

            for label in ('FL', 'FR', 'RL', 'RR'):
                solver = self._solvers.get(label)
                if not solver:
                    continue
                t  = travels.get(label, 0.)
                try:
                    st = solver.solve(float(t))
                except Exception:
                    try:
                        st = solver.solve(0.)
                    except Exception:
                        continue

                # ── spring-limit check: FREEZE at last valid, don't reset ──
                s_min, s_max = self._spring_limits(solver)
                if s_min <= st.spring_length <= s_max:
                    self._last_valid_st[label] = st   # cache valid state
                else:
                    cached = self._last_valid_st.get(label)
                    if cached is not None:
                        st = cached   # freeze at last valid geometry
                    else:
                        continue      # no cache yet (startup), skip corner

                hp_d = hp_dicts[label]

                # ── steering visual: show tie_rod_inner at steered position ──
                is_front = label in ('FL', 'FR')
                steered_hp_d = self._steered_hp(hp_d, rt_m, is_front)
                pts = _state_to_pts(st, steered_hp_d)

                # rocker_pivot is chassis-fixed
                pts['rocker_pivot'] = hp_d['rocker_pivot']

                # ── arb_drop_top: rotate with rocker from its design arm ────
                # For right corners (FR, RR) the ARB point is the X-mirror of
                # the stored left-side value. The rocker pivot is already
                # mirrored (comes from _mirror_x), so we must mirror the
                # design arb_drop_top too before computing the arm vector.
                arb_hp = self._front_arb if is_front else self._rear_arb
                try:
                    pivot    = hp_d['rocker_pivot']
                    axis_pt  = hp_d.get('rocker_axis_pt', pivot + np.array([0., 0.0254, 0.]))
                    r_axis   = _norm(axis_pt - pivot)
                    dt_design = arb_hp['arb_drop_top'].copy()
                    if label in ('FR', 'RR'):
                        dt_design = dt_design * flip_x
                    arm_dt   = dt_design - pivot
                    arm_rot  = _rodrigues(arm_dt, r_axis, st.rocker_angle)
                    pts['arb_drop_top'] = pivot + arm_rot

                    # ── Bell-crank: find rotated arm_end (blade end) ─────────
                    # The drop link is fixed-length rigid rod. arb_drop_top
                    # (on rocker) pulls/pushes the blade, rotating it about
                    # the torsion-bar axis. Solve for the new arm_end position
                    # so the drop-link segment has constant length.
                    arb_hp_vis = (arb_hp if label not in ('FR', 'RR')
                                  else {k: v * flip_x for k, v in arb_hp.items()})
                    _, ae_world, _ = self._solve_arb_bellcrank(
                        pts['arb_drop_top'], arb_hp_vis)
                    pts['arb_arm_end_world'] = ae_world
                except Exception:
                    pass   # if geometry invalid, rocker quad falls back to triangle

                # ── camber visual: rotate spin axis by alignment offset ────────
                # Equivalent to adding a shim between hub and upright.
                # Left corners: rotate spin axis around Y by -camber_off_rad
                # Right corners: rotate by +camber_off_rad
                # (derived from camber = -arctan2(spin[2], |spin[0]|) * sign)
                camber_vis = self._alignment.get(
                    'front_camber_deg' if is_front else 'rear_camber_deg', 0.)
                is_left = label in ('FL', 'RL')
                rot_rad = np.radians(camber_vis) * (-1. if is_left else 1.)
                spin_vis = (_rodrigues(st.spin_axis, np.array([0., 1., 0.]), rot_rad)
                            if abs(rot_rad) > 1e-9 else st.spin_axis)

                corners_draw.append({
                    'pts': pts, 'spin_axis': spin_vis, 'label': label})

                # Compute metrics for this corner
                # Two-point solve for MR: solve at t - epsilon first
                side = 'left' if label in ('FL', 'RL') else 'right'
                _dt = 0.001  # 1mm perturbation
                t_prev = float(t) - _dt
                spring_prev = travel_prev = None
                try:
                    st_prev = solver.solve(t_prev)
                    spring_prev = float(np.sqrt(
                        (st_prev.rocker_spring_pt[0] - st_prev.spring_chassis_pt[0])**2 +
                        (st_prev.rocker_spring_pt[1] - st_prev.spring_chassis_pt[1])**2 +
                        (st_prev.rocker_spring_pt[2] - st_prev.spring_chassis_pt[2])**2))
                    travel_prev = t_prev
                except Exception:
                    pass
                corner_vals = _all_metrics(st, side,
                    spring_prev=spring_prev, travel_prev=travel_prev,
                    cg_height_m=self._car.get('cg_z_mm', 280.) / 1000.,
                    wheelbase_m=self._car.get('wheelbase_mm', 1537.) / 1000.,
                    front_brake_bias=self._car.get('front_brake_bias_pct', 65.) / 100.,
                    rear_drive_bias=1.0, front_drive_bias=0.0,
                )
                # Add alignment offsets
                cam_key = 'front_camber_deg' if is_front else 'rear_camber_deg'
                toe_key = 'front_toe_deg' if is_front else 'rear_toe_deg'
                corner_vals['camber'] = (corner_vals.get('camber', 0.)
                                         + self._alignment.get(cam_key, 0.))
                corner_vals['toe']    = (corner_vals.get('toe', 0.)
                                         + self._alignment.get(toe_key, 0.))

                # ARB metrics for this corner
                try:
                    pivot  = st.rocker_pivot
                    ax_pt  = pivot + np.array([0., 0.0254, 0.])
                    r_axis = _norm(ax_pt - pivot)
                    arb_d  = arb_hp['arb_drop_top'].copy()
                    if label in ('FR', 'RR'):
                        arb_d = arb_d * flip_x
                    arm_dt = arb_d - pivot
                    dt_w   = pivot + _rodrigues(arm_dt, r_axis, st.rocker_angle)
                    arb_vis = (arb_hp if label not in ('FR', 'RR')
                               else {k: v * flip_x for k, v in arb_hp.items()})
                    ang, _, dl_t = self._solve_arb_bellcrank(dt_w, arb_vis)
                    corner_vals['arb_angle'] = float(np.degrees(ang))
                    corner_vals['arb_drop_travel'] = float(dl_t * 1000)
                    corner_vals['arb_mr'] = min(abs(np.degrees(ang) / (float(t) * 1000)), 5.0) if abs(float(t)) > 1e-9 else float('nan')
                except Exception:
                    pass

                all_corner_values[label] = corner_vals

            # ── ARB visual ────────────────────────────────────────────────────
            # Topology: arb_drop_top (on rocker, moving)
            #           → arb_arm_end (blade tip, rotates about torsion-bar axis)
            #           → arb_pivot (fixed)
            #           arb_pivot_L → arb_pivot_R  (torsion bar, lateral)
            # arb_arm_end_world stored in pts by the bell-crank solve above.
            arb_segs = []
            for axle_l, axle_r, arb_hp in [
                ('FL', 'FR', self._front_arb),
                ('RL', 'RR', self._rear_arb),
            ]:
                pv_l = arb_hp['arb_pivot'].copy()
                pv_r = pv_l * flip_x
                # Fallback: design arm_end if bell-crank solve was unavailable
                ae_l_design = arb_hp['arb_arm_end'].copy()
                ae_r_design = ae_l_design * flip_x

                for c in corners_draw:
                    dt = c['pts'].get('arb_drop_top')
                    ae_w = c['pts'].get('arb_arm_end_world')
                    if c['label'] == axle_l and dt is not None:
                        ae = ae_w if ae_w is not None else ae_l_design
                        arb_segs += [(dt, ae), (ae, pv_l)]
                    if c['label'] == axle_r and dt is not None:
                        ae = ae_w if ae_w is not None else ae_r_design
                        arb_segs += [(dt, ae), (ae, pv_r)]

                arb_segs += [(pv_l, pv_r)]   # torsion bar

            # ── Rack visual ───────────────────────────────────────────────────
            hp_world = self._all_corner_hp()
            rack_l = self._steered_hp(hp_world['FL'], rt_m, True)['tie_rod_inner']
            rack_r = self._steered_hp(hp_world['FR'], rt_m, True)['tie_rod_inner']
            self.view3d.update_rack(rack_l, rack_r)

            self.view3d.toggle_ground(self._car.get('show_ground', True))
            self.view3d.update_scene(corners_draw, arb_segs)

            # ── Roll-centre spheres (axle-level, proper IC intersection) ─────
            def _axle_rc(left_lbl, right_lbl):
                """Compute roll-centre 3-D position for an axle from corner pts."""
                l_c = next((c for c in corners_draw if c['label'] == left_lbl),  None)
                r_c = next((c for c in corners_draw if c['label'] == right_lbl), None)
                if l_c is None or r_c is None:
                    return None

                def _ic_from_pts(pts):
                    """Front-view (XZ) IC + contact-patch X from a pts dict."""
                    uca_in = np.array([(pts['uca_front'][0]+pts['uca_rear'][0])/2,
                                       (pts['uca_front'][2]+pts['uca_rear'][2])/2])
                    lca_in = np.array([(pts['lca_front'][0]+pts['lca_rear'][0])/2,
                                       (pts['lca_front'][2]+pts['lca_rear'][2])/2])
                    ic = _intersect_2d(uca_in,
                                       np.array([pts['uca_outer'][0], pts['uca_outer'][2]]),
                                       lca_in,
                                       np.array([pts['lca_outer'][0], pts['lca_outer'][2]]))
                    return ic, float(pts['wheel_center'][0])

                l_ic, l_cpx = _ic_from_pts(l_c['pts'])
                r_ic, r_cpx = _ic_from_pts(r_c['pts'])
                if l_ic is None or r_ic is None:
                    return None
                rc = _intersect_2d(l_ic, np.array([l_cpx, 0.0]),
                                   r_ic, np.array([r_cpx, 0.0]))
                if rc is None:
                    return None
                y_axle = (l_c['pts']['wheel_center'][1] +
                          r_c['pts']['wheel_center'][1]) / 2.
                return np.array([float(rc[0]), y_axle, float(rc[1])], float)

            front_rc = _axle_rc('FL', 'FR')
            rear_rc  = _axle_rc('RL', 'RR')
            if self._show_rc:
                self.view3d.update_rc(front_rc, rear_rc)
            self.view3d.set_rc_visible(self._show_rc)
            self.view3d.set_roll_axis_visible(self._show_roll_axis)

            # ── CG sphere ────────────────────────────────────────────────
            cg_x = self._car.get('cg_x_mm', 0.) / 1000.
            cg_y = self._car.get('cg_y_mm', 1100.) / 1000.
            cg_z = self._car.get('cg_z_mm', 280.) / 1000.
            self.view3d.update_cg((cg_x, cg_y, cg_z))

            self._values_panel.update_values(all_corner_values)

            unit = 'deg' if motion in ('roll', 'steer') else ' mm'
            self.statusBar().showMessage(
                f'{motion.title()} = {pos:+.2f}{unit}', 2000)

        except Exception as e:
            self.statusBar().showMessage(f'3D: {e}', 5000)
            import traceback; traceback.print_exc()

    # ==========================================================================
    #  ALIGNMENT
    # ==========================================================================
    #
    # Camber and toe alignment is implemented as post-solve measurement offsets:
    #
    #   Camber: equivalent to adding a shim between the hub and upright.
    #           The kinematic linkage geometry is unchanged; the spin axis is
    #           rotated by the target angle after each solve for metrics and
    #           for the 3-D tire visual.
    #
    #   Toe:    equivalent to a rod-end adjustment (threading in/out).
    #           Same offset approach for simplicity and reliability.
    #
    # This avoids the cold-start issue where solver.solve(0.) always returns
    # the design-position geometry (residuals are identically zero there), so
    # any hardpoint-perturbation Newton solve for camber/toe would measure 0
    # and never converge to a non-trivial result.

    # ==========================================================================
    #  EVENT HANDLERS
    # ==========================================================================

    def _on_position(self, pos):
        """
        Slider moved: snap the vline immediately (blit, zero cost),
        and defer the heavy 3D solve to the next idle event-loop cycle.
        Multiple rapid slider events collapse to a single 3D update.
        """
        self.curves.set_vline(pos)
        if not self._3d_pending:
            self._3d_pending = True
            QTimer.singleShot(0, self._deferred_3d)

    def _deferred_3d(self):
        self._3d_pending = False
        self._update_3d()

    def _on_sweep_trigger(self, *_):
        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()

    def _on_steer(self, params: dict):
        self._steer = params
        cur_angle = self._motion_panel.position if self._motion_panel.motion == 'steer' else 0.
        self._rebuild_solvers(cur_angle)
        self._run_sweep()
        self._update_3d()
        self._update_min_turn_radius()

    def _update_min_turn_radius(self):
        """Compute min turn radius from steering geometry and update readout."""
        try:
            steer_params = self._steer
            total_mm = steer_params.get('total_rack_travel_mm', 120.0)
            max_in   = steer_params.get('max_rack_travel_in', 2.5)
            half_mm  = min(total_mm / 2.0, max_in * 25.4)
            rack_m   = half_mm / 1000.0

            hp_raw = {k: v.copy() for k, v in self._front_hp.items()}
            hp_steered = self._steered_hp(hp_raw, rack_m, is_front=True)

            # Design tie-rod length (before rack moves)
            d = hp_raw['tie_rod_outer'] - hp_raw['tie_rod_inner']
            design_tierod_len_sq = float(d @ d)

            steered_solver = SuspensionConstraints(
                _hp_obj(hp_steered),
                tierod_len_sq=design_tierod_len_sq,
                pushrod_body='uca')
            state = steered_solver.solve(0.0)
            m = KinematicMetrics(state, 'left')
            max_steer_deg = abs(m.toe)
            if max_steer_deg > 0.5:
                wb = self._car['wheelbase_mm'] / 1000
                r_min = wb / np.tan(np.radians(max_steer_deg))
                self._dynamics_panel._cached_r_min = r_min
                self._dynamics_panel._cached_max_steer = max_steer_deg
                # Steering ratio: handwheel degrees / front wheel degrees
                rack_per_rev = steer_params.get('rack_travel_per_rev_mm', 60.0)
                if rack_per_rev > 0:
                    hw_deg = (half_mm / rack_per_rev) * 360.0
                    self._dynamics_panel._cached_steer_ratio = hw_deg / max_steer_deg
                self._dynamics_panel._on_driving_changed()
        except Exception:
            pass

    def _on_car(self, params: dict):
        old = self._car

        # ── axle spacing delta → shift ALL rear hardpoints in Y ───────────
        # Axle spacing = distance between front/rear hardpoint clusters.
        das = (params.get('axle_spacing_mm', old.get('axle_spacing_mm', 1537.))
               - old.get('axle_spacing_mm', old.get('wheelbase_mm', 1537.))) / 1000.
        if abs(das) > 1e-9:
            dy = np.array([0., das, 0.])
            for k in list(self._rear_hp):
                self._rear_hp[k] = self._rear_hp[k] + dy
            for k in list(self._rear_arb):
                self._rear_arb[k] = self._rear_arb[k] + dy

        # ── wheelbase delta → dynamics only, NO hardpoint shift ───────────
        # Wheelbase = contact-patch distance, used for load transfer,
        # Ackermann, understeer gradient, etc.  Does not move geometry.

        # ── track delta → shift outboard pickups + wheel_center in X ──────
        # "Outboard" = the upright pickup points and wheel centre.
        # Inboard chassis mounts stay fixed (bolted to frame).
        _OUTBOARD = {'uca_outer', 'lca_outer', 'tie_rod_outer',
                     'wheel_center', 'pushrod_outer'}
        dt_f = (params['track_f_mm'] - old['track_f_mm']) / 2000.  # half-track Δ (m)
        if abs(dt_f) > 1e-9:
            dx = np.array([dt_f, 0., 0.])
            for k in _OUTBOARD:
                if k in self._front_hp:
                    self._front_hp[k] = self._front_hp[k] + dx
        dt_r = (params['track_r_mm'] - old['track_r_mm']) / 2000.
        if abs(dt_r) > 1e-9:
            dx = np.array([dt_r, 0., 0.])
            for k in _OUTBOARD:
                if k in self._rear_hp:
                    self._rear_hp[k] = self._rear_hp[k] + dx

        # ── wheel offset delta → shift ONLY wheel_center in X ────────────
        # Wheel offset = how far the wheel sits beyond the outboard pickups.
        dof = (params.get('wheel_offset_f_mm', old.get('wheel_offset_f_mm', 25.))
               - old.get('wheel_offset_f_mm', 25.)) / 1000.
        if abs(dof) > 1e-9:
            dx = np.array([dof, 0., 0.])
            if 'wheel_center' in self._front_hp:
                self._front_hp['wheel_center'] = self._front_hp['wheel_center'] + dx
        dor = (params.get('wheel_offset_r_mm', old.get('wheel_offset_r_mm', 25.))
               - old.get('wheel_offset_r_mm', 25.)) / 1000.
        if abs(dor) > 1e-9:
            dx = np.array([dor, 0., 0.])
            if 'wheel_center' in self._rear_hp:
                self._rear_hp['wheel_center'] = self._rear_hp['wheel_center'] + dx

        self._car = params

        # Refresh the hardpoint table UIs so the user sees the shifted values
        self._front_hp_panel.refresh(self._front_hp, self._front_arb)
        self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb)

        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()
        self._update_min_turn_radius()

        # Update dynamics readout (weight distribution, etc.)
        self._dynamics_panel._on_driving_changed()

    def _on_hp(self, hp_dict: dict, axle: str):
        # Split combined dict back into suspension HP and ARB HP
        arb_keys = ['arb_drop_top', 'arb_arm_end', 'arb_pivot']
        hp  = {k: v for k, v in hp_dict.items() if k not in arb_keys}
        arb = {k: v for k, v in hp_dict.items() if k in arb_keys}
        if axle == 'front':
            self._front_hp  = hp
            self._front_arb = arb
        else:
            self._rear_hp  = hp
            self._rear_arb = arb
        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()

    def _on_row(self, name: str):
        self.view3d.set_selected(name)
        self._update_3d()

    def _on_pick(self, name: str, corner: str = 'FL'):
        if corner in ('FL', 'FR'):
            self._front_hp_panel.highlight_row(name)
        else:
            self._rear_hp_panel.highlight_row(name)

    # ── Inverse Kinematics ───────────────────────────────────────────────────

    _ik_thread: _IKWorker | None = None
    _ik_explore_thread: _IKExploreWorker | None = None

    def _build_ik_solver(self, spec: dict, bound_mm: float) -> InverseSolver:
        """Create a configured InverseSolver from a UI spec dict."""
        from vahan.optimizer import _evaluate_sweep

        axle = spec['axle']
        hp = dict(self._front_hp if axle == 'front' else self._rear_hp)
        side = 'left'
        pushrod_body = 'uca' if axle == 'front' else 'lca'

        # Merge ARB points into hp dict so optimizer can adjust them
        arb = self._front_arb if axle == 'front' else self._rear_arb
        for k, v in arb.items():
            hp[k] = v.copy()

        variables = []
        for hp_name in spec['hp_names']:
            for coord in spec['coords']:
                if hp_name in hp:
                    variables.append(DesignVar(hp_name, coord, bound_mm / 1000))

        anti_kwargs = {
            'cg_height_m':      self._car.get('cg_z_mm', 280.) / 1000.,
            'wheelbase_m':      self._car.get('wheelbase_mm', 1537.) / 1000.,
            'front_brake_bias': self._car.get('front_brake_bias_pct', 65.) / 100.,
            'rear_drive_bias':  1.0,
            'front_drive_bias': 0.0,
        }

        lo_mm = spec.get('range_lo', -30)
        hi_mm = spec.get('range_hi', 30)
        motion = spec.get('motion', 'heave')
        n_pts = 21

        ik = InverseSolver(
            hp, side=side, pushrod_body=pushrod_body,
            travel_mm=(lo_mm, hi_mm), n_points=n_pts,
            anti_kwargs=anti_kwargs,
            motion=motion,
        )

        # Primary target: linear ramp
        target_lo = spec.get('target_lo', spec.get('target', 0.0))
        target_hi = spec.get('target_hi', target_lo)
        target_ramp = np.linspace(float(target_lo), float(target_hi), n_pts)

        # Auto-balance: primary weight scales with number of locks
        # so the primary isn't drowned out by lock penalties
        lock_metrics = spec.get('lock_metrics', [])
        n_locks = max(len(lock_metrics), 1)
        primary_weight = float(n_locks) * 10.0   # 10x per lock
        lock_weight = 1.0                          # locks are soft

        ik.add_target(spec['metric_key'], target_ramp, weight=primary_weight)

        # Lock constraints with tolerance dead-band
        lock_tol = spec.get('lock_tol', 5.0)
        if lock_metrics:
            current_curves = _evaluate_sweep(
                hp, ik.travel, side, pushrod_body,
                metric_keys=lock_metrics,
                anti_kwargs=anti_kwargs,
                motion=motion,
            )
            for lk in lock_metrics:
                curve = current_curves.get(lk)
                if curve is not None and not np.all(np.isnan(curve)):
                    ik.add_target(lk, curve, weight=lock_weight,
                                  tolerance=lock_tol)

        ik.set_variables(variables)

        # Enable tube collision avoidance
        ik.tube_od = spec.get('tube_od') or {}

        return ik

    def _on_ik_solve(self, spec: dict):
        """Spawn a background QThread to run the IK solver."""
        busy = ((self._ik_thread is not None and self._ik_thread.isRunning()) or
                (self._ik_explore_thread is not None and self._ik_explore_thread.isRunning()))
        if busy:
            self.statusBar().showMessage('IK already running — please wait', 3000)
            return
        try:
            # ── Explore mode: parallel warm-start LM at wider bounds ─────
            if spec.get('explore'):
                last = self._ik_panel._last_result
                if last is None or 'x' not in last:
                    self._ik_panel.show_result(None,
                        'Run a normal Solve first to get a starting point.')
                    return

                warm_x_raw = np.array(last['x'])
                axle = spec['axle']
                hp = dict(self._front_hp if axle == 'front' else self._rear_hp)
                arb = self._front_arb if axle == 'front' else self._rear_arb
                for k, v in arb.items():
                    hp[k] = v.copy()
                side = 'left'
                pushrod_body = 'uca' if axle == 'front' else 'lca'
                lo_mm = spec.get('range_lo', -30)
                hi_mm = spec.get('range_hi', 30)
                motion = spec.get('motion', 'heave')
                n_pts = 21

                anti_kwargs = {
                    'cg_height_m':      self._car.get('cg_z_mm', 280.) / 1000.,
                    'wheelbase_m':      self._car.get('wheelbase_mm', 1537.) / 1000.,
                    'front_brake_bias': self._car.get('front_brake_bias_pct', 65.) / 100.,
                    'rear_drive_bias':  1.0,
                    'front_drive_bias': 0.0,
                }

                # Build target list (serialisable for multiprocessing)
                from vahan.optimizer import _evaluate_sweep
                target_lo = spec.get('target_lo', 0.0)
                target_hi = spec.get('target_hi', target_lo)
                target_ramp = np.linspace(float(target_lo), float(target_hi), n_pts)

                # Auto-balanced weights (same logic as _build_ik_solver)
                lock_metrics = spec.get('lock_metrics', [])
                n_locks = max(len(lock_metrics), 1)
                primary_weight = float(n_locks) * 10.0
                lock_weight = 1.0
                lock_tol = spec.get('lock_tol', 5.0)

                # (key, values, weight, tolerance)
                targets_spec = [(spec['metric_key'], target_ramp.tolist(),
                                 primary_weight, 0.0)]

                # Compute travel array to match what InverseSolver would use
                if motion == 'steer':
                    travel_arr = np.linspace(lo_mm, hi_mm, n_pts)
                else:
                    travel_arr = np.linspace(lo_mm / 1000, hi_mm / 1000, n_pts)

                if lock_metrics:
                    current_curves = _evaluate_sweep(
                        hp, travel_arr, side, pushrod_body,
                        metric_keys=lock_metrics,
                        anti_kwargs=anti_kwargs, motion=motion,
                    )
                    for lk in lock_metrics:
                        curve = current_curves.get(lk)
                        if curve is not None and not np.all(np.isnan(curve)):
                            targets_spec.append((lk, curve.tolist(),
                                                 lock_weight, lock_tol))

                # Variable specs (point, coord) — bounds vary per level
                var_specs = []
                for hp_name in spec['hp_names']:
                    for coord in spec['coords']:
                        if hp_name in hp:
                            var_specs.append((hp_name, coord))

                # Validate warm start matches current variable selection
                if len(warm_x_raw) != len(var_specs):
                    self._ik_panel.show_result(None,
                        f'Variable selection changed since last solve '
                        f'({len(warm_x_raw)} → {len(var_specs)}). '
                        f'Run Solve again first.')
                    return

                solver_kwargs = {
                    'hp_dict':      {k: v.tolist() for k, v in hp.items()},
                    'side':         side,
                    'pushrod_body': pushrod_body,
                    'travel_mm':    (lo_mm, hi_mm),
                    'n_points':     n_pts,
                    'anti_kwargs':  anti_kwargs,
                    'motion':       motion,
                    'targets':      targets_spec,
                    'var_specs':    var_specs,
                    'tube_od':      spec.get('tube_od', {}),
                }

                base = spec['bound_mm']
                levels = [base * m for m in (2, 4, 7, 10)]

                self._ik_explore_thread = _IKExploreWorker(
                    solver_kwargs, levels, warm_x_raw)
                self._ik_explore_thread.status.connect(
                    lambda msg: self.statusBar().showMessage(msg, 0))
                self._ik_explore_thread.finished.connect(self._on_ik_explore_done)
                self._ik_explore_thread.failed.connect(self._on_ik_fail)
                self.statusBar().showMessage('Searching for solutions in parallel...', 0)
                self._ik_explore_thread.start()
                return

            # ── Normal single solve ──────────────────────────────────────
            hp_check = dict(self._front_hp if spec['axle'] == 'front' else self._rear_hp)
            arb_check = self._front_arb if spec['axle'] == 'front' else self._rear_arb
            hp_check.update(arb_check)
            has_vars = any(hp_name in hp_check
                          for hp_name in spec['hp_names']
                          for _ in spec['coords'])
            if not has_vars:
                self._ik_panel.show_result(None, 'No valid variables selected.')
                return

            ik = self._build_ik_solver(spec, spec['bound_mm'])

            self._ik_thread = _IKWorker(ik, spec['method'])
            self._ik_thread.status.connect(
                lambda msg: self.statusBar().showMessage(msg, 0))
            self._ik_thread.finished.connect(self._on_ik_done)
            self._ik_thread.failed.connect(self._on_ik_fail)
            self.statusBar().showMessage('IK solving...', 0)
            self._ik_thread.start()

        except Exception as e:
            self._ik_panel.show_result(None, str(e))
            import traceback; traceback.print_exc()

    def _on_damper_limits(self, params: dict):
        """Forward damper stroke/sag to IK panel and values panel."""
        stroke = params.get('stroke_mm', 60.0)
        sag = params.get('sag_pct', 30.0)
        self._ik_panel.set_damper_limits(stroke, sag)
        self._values_panel.update_damper_params(stroke, sag)

    def _on_ik_done(self, result: dict):
        self._ik_panel.show_result(result)
        self.statusBar().showMessage(f'IK done — cost {result["cost"]:.4f}', 5000)

    def _on_ik_explore_done(self, solutions: list[dict]):
        # Filter out solutions with tube collisions
        total = len(solutions)
        solutions = [s for s in solutions if not s.get('collisions')]
        dropped = total - len(solutions)
        msg = f'Found {len(solutions)} solutions'
        if dropped:
            msg += f' ({dropped} rejected — tube collision)'
        self.statusBar().showMessage(msg, 5000)
        self._ik_panel.show_solutions(solutions)

    def _on_ik_fail(self, msg: str):
        self._ik_panel.show_result(None, msg)
        self.statusBar().showMessage(f'IK failed: {msg}', 5000)

    def _on_ik_apply(self, data: dict):
        """Apply IK-optimised hardpoints to the model."""
        axle = data['axle']
        new_hp = data['hp']

        # Separate ARB points from suspension hardpoints
        _ARB_KEYS = {'arb_drop_top', 'arb_arm_end', 'arb_pivot'}
        sus_hp = {k: v.copy() for k, v in new_hp.items() if k not in _ARB_KEYS}
        arb_hp = {k: v.copy() for k, v in new_hp.items() if k in _ARB_KEYS}

        if axle == 'front':
            self._front_hp = sus_hp
            if arb_hp:
                self._front_arb = arb_hp
            self._front_hp_panel.refresh(self._front_hp, self._front_arb)
        else:
            self._rear_hp = sus_hp
            if arb_hp:
                self._rear_arb = arb_hp
            self._rear_hp_panel.refresh(self._rear_hp, self._rear_arb)

        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()
        self.statusBar().showMessage(f'Applied IK result to {axle} suspension', 4000)

    def _replot(self):
        """Re-draw curves with current keys and corner selection."""
        title = (f'{self._motion_panel.motion.title()}  '
                 f'[{self._motion_panel.min_val:+.0f} -> '
                 f'{self._motion_panel.max_val:+.0f}]')
        self.curves.plot(self._x_arr, self._x_label,
                         self._sweep_results, self._selected_keys, title,
                         corners=self._selected_corners)

    def _on_graph_sel(self, keys: list):
        self._selected_keys = keys
        self._replot()

    def _on_corners_sel(self, corners: list):
        self._selected_corners = corners
        self._replot()

    def _on_alignment(self, params: dict):
        self._alignment = params
        self._run_sweep()    # rebuilds sweep with new camber/toe offsets
        self._update_3d()   # rotates tire spin axis visually
        self.statusBar().showMessage(
            f'Alignment: front {params["front_camber_deg"]:+.2f}° camber  '
            f'{params["front_toe_deg"]:+.2f}° toe  |  '
            f'rear {params["rear_camber_deg"]:+.2f}° camber  '
            f'{params["rear_toe_deg"]:+.2f}° toe', 5000)

    # ==========================================================================
    #  DYNAMICS
    # ==========================================================================

    def _build_dynamics_solver(self) -> SteadyStateSolver:
        """Build a SteadyStateSolver from current GUI state.

        - Motion ratios are queried from the kinematic solver at design
          position (travel=0), not manually entered.
        - CG and track come from the Car Parameters panel.
        - Unsprung CG height = wheel center Z at design (from geometry).
        """
        dyn_params = self._dynamics_panel.get_params()
        car = self._car

        # Geometry from Car Params panel
        dyn_params['front_track_m'] = car['track_f_mm'] / 1000
        dyn_params['rear_track_m'] = car['track_r_mm'] / 1000
        dyn_params['wheelbase_m'] = car['wheelbase_mm'] / 1000
        dyn_params['cg_height_m'] = car['cg_z_mm'] / 1000
        dyn_params['cg_to_front_axle_m'] = car.get('cg_y_mm', 1100) / 1000
        if 'front_brake_bias_pct' in car:
            dyn_params['front_brake_bias'] = car['front_brake_bias_pct'] / 100

        # Motion ratio from kinematic model: MR = d(spring_length)/d(travel)
        # Computed via central difference at design position (±1mm)
        dt = 0.001  # 1mm perturbation
        for label, param_key in [('FL', 'motion_ratio_front'), ('RL', 'motion_ratio_rear')]:
            solver = self._solvers.get(label)
            if solver:
                try:
                    s_plus = solver.solve(+dt)
                    s_minus = solver.solve(-dt)
                    mr = abs(s_plus.spring_length - s_minus.spring_length) / (2 * dt)
                    if 0.1 < mr < 3.0:  # sanity check
                        dyn_params[param_key] = mr
                except Exception:
                    pass  # keep default if solver fails

        # Unsprung CG height = wheel center Z at design position
        for label in ('FL', 'RL'):
            solver = self._solvers.get(label)
            if solver:
                try:
                    state = solver.solve(0.0)
                    dyn_params['unsprung_cg_height_m'] = float(state.wheel_center[2])
                    break
                except Exception:
                    pass

        # Max steer angle from geometry (cached by _update_min_turn_radius)
        cached_steer = getattr(self._dynamics_panel, '_cached_max_steer', None)
        if cached_steer and cached_steer > 1.0:
            dyn_params['max_steer_angle_deg'] = cached_steer

        veh = VehicleParams(**dyn_params)
        # Update computed constants display
        self._dynamics_panel.update_constants(veh)
        tire = self._tire_model
        if tire is None:
            from vahan.tire_model import LinearTireModel
            tire = LinearTireModel()
        return SteadyStateSolver(veh, self._solvers, tire)

    def _try_autoload_tire(self):
        """Auto-load default tire data if tire_data/ directory exists."""
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Prefer run5 (14 PSI operating pressure, 7" rim)
        candidates = [
            os.path.join(base, 'tire_data', 'B2356run5.mat'),
            os.path.join(base, 'tire_data', 'B2356run5.csv'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                self._on_tire_file(path)
                return

    def _on_tire_file(self, path: str):
        """Load tire data from .mat, .csv, or .xlsx."""
        try:
            from vahan.tire_model import TireModel
            self._tire_model = TireModel.from_file(path)
            name = path.split('/')[-1].split('\\')[-1]
            self._dynamics_panel._tire_path = path
            self._dynamics_panel._tire_label.setText(name)
            self._dynamics_panel._tire_label.setStyleSheet(
                'color: #e0e0e0; font-size: 11px;')
            psi_str = f'  P: {self._tire_model.pressure_psi:.1f} psi' if self._tire_model.pressure_psi > 0 else ''
            self._dynamics_panel.set_status(
                f'Loaded: {self._tire_model.tire_id}  '
                f'SA: {self._tire_model.sa_range[0]:.0f} to {self._tire_model.sa_range[1]:.0f} deg  '
                f'Fz: {self._tire_model.fz_range[0]:.0f}-{self._tire_model.fz_range[1]:.0f} N'
                f'{psi_str}')
            self.statusBar().showMessage(f'Tire model loaded: {path}', 4000)
        except Exception as e:
            self._dynamics_panel.set_status(f'Error: {e}')
            self.statusBar().showMessage(f'Tire load error: {e}', 6000)

    def _on_dynamics_solve(self, spec: dict):
        """Single-point steady-state solve."""
        try:
            ss = self._build_dynamics_solver()
            self._dynamics_panel.set_solving(True)
            self._dynamics_panel.set_status('Solving...')

            worker = _DynamicsSolveWorker(
                ss, spec['lateral_g'], spec.get('longitudinal_g', 0.0))
            worker.finished.connect(self._on_dynamics_solve_done)
            worker.failed.connect(self._on_dynamics_failed)
            self._dyn_worker = worker
            worker.start()
        except Exception as e:
            self._dynamics_panel.set_status(f'Error: {e}')

    def _on_dynamics_solve_done(self, result):
        self._dynamics_panel.set_solving(False)
        self._dynamics_panel.show_result(result)
        # Compute max g and min turn radius if power/steer is set
        try:
            ss = self._build_dynamics_solver()
            veh = ss._veh
            max_g_info = ss.max_accel_g(speed_kph=veh.speed_kph)
            if max_g_info.get('traction_g', 0) > 0:
                self._dynamics_panel.show_max_g(max_g_info)
            if veh.min_turn_radius_m < 100:
                max_g_info['min_turn_radius_m'] = veh.min_turn_radius_m
                self._dynamics_panel.show_max_g(max_g_info)
        except Exception:
            pass
        self._dynamics_panel.set_status('Done.')
        self.statusBar().showMessage(
            f'Dynamics: {result.roll_angle_deg:.3f} deg roll at '
            f'{result.lateral_g:.2f}g', 4000)

    def _on_dynamics_sweep(self, spec: dict):
        """Lateral or longitudinal g sweep."""
        try:
            ss = self._build_dynamics_solver()
            self._dynamics_panel.set_solving(True)
            mode = spec.get('mode', 'lateral')
            self._dynamics_panel.set_status(
                f'Sweeping ({mode})...')

            worker = _DynamicsSweepWorker(
                ss, spec['g_min'], spec['g_max'],
                spec.get('n_points', 41),
                longitudinal_g=spec.get('longitudinal_g', 0.0),
                mode=mode,
                lateral_g=spec.get('lateral_g', 0.0))
            worker.finished.connect(self._on_dynamics_sweep_done)
            worker.failed.connect(self._on_dynamics_failed)
            self._dyn_worker = worker
            worker.start()
        except Exception as e:
            self._dynamics_panel.set_status(f'Error: {e}')

    def _on_dynamics_sweep_done(self, sweep: dict):
        self._dynamics_panel.set_solving(False)
        self._dyn_sweep_data = sweep  # stash for re-plot on graph/corner change

        # Determine mode from which x-axis key is present
        is_longitudinal = 'longitudinal_g' in sweep and 'lateral_g' not in sweep
        g_key = 'longitudinal_g' if is_longitudinal else 'lateral_g'
        g_arr = sweep[g_key]

        self._dynamics_panel.set_status(
            f'Sweep complete: {len(g_arr)} points')

        graphs = self._dynamics_panel.get_selected_graphs()
        corners = self._dynamics_panel.get_selected_corners()
        turn_r = self._dynamics_panel._turn_radius.value()
        wb = self._car.get('wheelbase_mm', 1530) / 1000
        sr = getattr(self._dynamics_panel, '_cached_steer_ratio', 0.0)
        hp_w = self._dynamics_panel._power_hp.value() * 745.7
        mass = self._dynamics_panel._total_mass.value()
        self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                 turn_radius_m=turn_r, wheelbase_m=wb,
                                 steer_ratio=sr,
                                 power_W=hp_w, mass_kg=mass)

        # Show the 1g (lateral) or 0g (longitudinal) point in the table
        ref_g = 0.0 if is_longitudinal else 1.0
        idx_ref = np.argmin(np.abs(g_arr - ref_g))
        if abs(g_arr[idx_ref] - ref_g) < 0.15:
            result = SteadyStateResult(
                lateral_g=g_arr[idx_ref] if not is_longitudinal else 0.0,
                longitudinal_g=g_arr[idx_ref] if is_longitudinal else 0.0)
            result.roll_angle_deg = sweep['roll_angle_deg'][idx_ref]
            result.pitch_angle_deg = sweep.get('pitch_angle_deg', np.zeros(1))[min(idx_ref, len(sweep.get('pitch_angle_deg', [0]))-1)]
            result.rc_height_front_m = sweep['rc_height_front_mm'][idx_ref] / 1000
            result.rc_height_rear_m = sweep['rc_height_rear_mm'][idx_ref] / 1000
            result.elastic_lt_front_N = sweep['elastic_lt_front_N'][idx_ref]
            result.elastic_lt_rear_N = sweep['elastic_lt_rear_N'][idx_ref]
            result.geometric_lt_front_N = sweep['geometric_lt_front_N'][idx_ref]
            result.geometric_lt_rear_N = sweep['geometric_lt_rear_N'][idx_ref]
            result.understeer_gradient_deg = sweep.get('understeer_gradient_deg', np.zeros(1))[min(idx_ref, len(sweep.get('understeer_gradient_deg', [0]))-1)]
            result.iterations = 0
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                result.Fz[lbl] = sweep[f'Fz_{lbl}'][idx_ref]
                result.travel[lbl] = sweep[f'travel_{lbl}'][idx_ref]
                result.camber[lbl] = sweep[f'camber_{lbl}'][idx_ref]
                result.utilization[lbl] = sweep.get(f'utilization_{lbl}', np.zeros(1))[idx_ref]
            self._dynamics_panel.show_result(result)

    def _on_dynamics_failed(self, msg: str):
        self._dynamics_panel.set_solving(False)
        self._dynamics_panel.set_status(f'Error: {msg}')
        self.statusBar().showMessage(f'Dynamics error: {msg}', 6000)

    def _on_dyn_graph_sel(self, graphs: list):
        """Re-plot dynamics with new graph selection."""
        sweep = getattr(self, '_dyn_sweep_data', None)
        if sweep is not None:
            corners = self._dynamics_panel.get_selected_corners()
            turn_r = self._dynamics_panel._turn_radius.value()
            wb = self._car.get('wheelbase_mm', 1530) / 1000
            hp_w = self._dynamics_panel._power_hp.value() * 745.7
            mass = self._dynamics_panel._total_mass.value()
            self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                     turn_radius_m=turn_r, wheelbase_m=wb,
                                     power_W=hp_w, mass_kg=mass)

    def _on_dyn_corners_sel(self, corners: list):
        """Re-plot dynamics with new corner selection."""
        sweep = getattr(self, '_dyn_sweep_data', None)
        if sweep is not None:
            graphs = self._dynamics_panel.get_selected_graphs()
            turn_r = self._dynamics_panel._turn_radius.value()
            wb = self._car.get('wheelbase_mm', 1530) / 1000
            hp_w = self._dynamics_panel._power_hp.value() * 745.7
            mass = self._dynamics_panel._total_mass.value()
            self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                     turn_radius_m=turn_r, wheelbase_m=wb,
                                     power_W=hp_w, mass_kg=mass)

    # ── Dynamics Optimizer ───────────────────────────────────────────────

    _sens_worker: _SensitivityWorker | None = None

    def _on_sensitivity_analyze(self, spec: dict):
        """Run sensitivity analysis in a background thread."""
        try:
            solver = self._build_dynamics_solver()
            tire = self._tire_model
            if tire is None:
                from vahan.tire_model import LinearTireModel
                tire = LinearTireModel()
            sens = DynamicsSensitivity(solver._veh, self._solvers, tire)

            self._sens_worker = _SensitivityWorker(
                sens, spec['lateral_g'], spec['longitudinal_g'])
            self._sens_worker.finished.connect(self._on_sensitivity_done)
            self._sens_worker.failed.connect(self._on_sensitivity_failed)
            self._sens_worker.start()
        except Exception as e:
            self._dynamics_opt_panel._opt_status.setText(f'Error: {e}')
            self._dynamics_opt_panel._analyze_btn.setEnabled(True)

    def _on_sensitivity_done(self, analysis: dict):
        self._dynamics_opt_panel.show_analysis(analysis)
        self.statusBar().showMessage('Sensitivity analysis complete', 4000)

    def _on_sensitivity_failed(self, msg: str):
        self._dynamics_opt_panel._opt_status.setText(f'Error: {msg}')
        self._dynamics_opt_panel._analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f'Sensitivity error: {msg}', 6000)

    # ==========================================================================
    #  COMPONENT LOADS
    # ==========================================================================

    def _on_compute_loads(self):
        """Compute component forces for all 4 corners at current dynamics state."""
        try:
            from vahan.loads import compute_all_corners

            solver = self._build_dynamics_solver()
            dyn_params = self._dynamics_panel.get_params()
            lat_g = dyn_params.get('_lat_g', 1.2)
            lon_g = dyn_params.get('_lon_g', 0.0)

            # Get lat/lon g from the dynamics panel spinners
            try:
                lat_g = self._dynamics_panel._lat_g.value()
                lon_g = self._dynamics_panel._lon_g.value()
            except AttributeError:
                pass

            result = solver.solve(lat_g, lon_g)

            # Separate front/rear brake params + shared upright params
            bp_f = self._loads_panel.get_brake_params_front()
            bp_r = self._loads_panel.get_brake_params_rear()
            up = self._loads_panel.get_upright_params()

            veh = solver._veh
            wheel_r = veh.tire_radius_m

            loads = compute_all_corners(
                self._solvers, result,
                brake_params_f=bp_f, brake_params_r=bp_r,
                upright_params_f=up, upright_params_r=up,
                wheel_radius_m=wheel_r,
                motion_ratio_f=veh.motion_ratio_front,
                motion_ratio_r=veh.motion_ratio_rear,
            )

            self._loads_panel.show_loads(loads, lat_g=lat_g, lon_g=lon_g)
            self.statusBar().showMessage(
                f'Component loads computed at {lat_g:.1f}g lat, {lon_g:.1f}g lon', 4000)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._loads_panel._loads_status.setText(f'Error: {e}')

    # ==========================================================================
    #  AERO DOWNFORCE
    # ==========================================================================

    def _get_device_positions(self) -> dict:
        """Extract device CoP positions from the aero geometry panel."""
        gp = self._aero_geom_panel.params()
        return {
            'fw_y':   gp['fw_y'],                                   # m from front axle
            'rw_y':   gp['rw_y'],                                   # m from front axle
            'diff_y': (gp['diff_y_start'] + gp['diff_y_end']) / 2,  # diffuser centroid
        }

    def _on_aero_solve(self, params: dict):
        try:
            self._aero_panel._status.setText('Solving...')
            ss = self._build_dynamics_solver()
            aero = AeroDownforceSolver(ss)
            result = aero.solve(
                params['lateral_g'], params['longitudinal_g'],
                params['target_util'],
                front_aero_fraction=params.get('front_aero_fraction', 0.5),
                device_positions=self._get_device_positions(),
            )
            self._aero_panel.show_result(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._aero_panel._status.setText(f'Error: {e}')

    def _on_aero_sweep(self, params: dict):
        try:
            self._aero_panel._status.setText('Sweeping...')
            ss = self._build_dynamics_solver()
            aero = AeroDownforceSolver(ss)
            g_range = np.linspace(0.1, params['lateral_g'], 21)
            sweep = aero.sweep(
                g_range, params['longitudinal_g'], params['target_util'],
                front_aero_fraction=params.get('front_aero_fraction', 0.5),
                device_positions=self._get_device_positions(),
            )

            # ── Plot in dynamics figure ──
            # Consistent per-corner styles (matches dynamics sweep exactly)
            _styles = {
                'FL': (CORNER_PLOT_COLORS['FL'], '-'),   # yellow solid
                'FR': (CORNER_PLOT_COLORS['FR'], '--'),  # red dashed
                'RL': (CORNER_PLOT_COLORS['RL'], '-.'),  # white dash-dot
                'RR': (CORNER_PLOT_COLORS['RR'], ':'),   # blue dotted
            }
            _leg_kw = dict(fontsize=7, facecolor='#06060e',
                           labelcolor='white', framealpha=0.7,
                           loc='best', handlelength=1.5, ncol=2)

            fig = self.curves.fig
            fig.clear()
            gs = sweep['lateral_g']

            # ── Subplot 1: per-corner Fz deficit ──
            ax1 = fig.add_subplot(1, 2, 1)
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                col, ls = _styles[lbl]
                ax1.plot(gs, sweep[f'dF_{lbl}'], label=lbl,
                         color=col, ls=ls, lw=1.8)
            ax1.set_xlabel('Lateral g')
            ax1.set_ylabel('Additional Fz required (N)')
            ax1.set_title('Per-corner load deficit',
                          color='white', fontsize=10)
            ax1.legend(**_leg_kw)
            ax1.grid(True, alpha=0.2)

            # ── Subplot 2: per-device force allocation ──
            ax2 = fig.add_subplot(1, 2, 2)
            F_fw   = sweep['F_fw']
            F_rw   = sweep['F_rw']
            F_diff = sweep['F_diff']
            F_total = F_fw + F_rw + F_diff

            ax2.plot(gs, F_fw, color='#FFD600', linewidth=2.2,
                     linestyle='-', marker='v', markersize=4,
                     markevery=3, label='Front wing')
            ax2.plot(gs, F_rw, color='#42A5F5', linewidth=2.2,
                     linestyle='-', marker='^', markersize=4,
                     markevery=3, label='Rear wing')
            ax2.plot(gs, F_diff, color='#E53935', linewidth=2.2,
                     linestyle='-', marker='s', markersize=3,
                     markevery=3, label='Diffuser')
            ax2.plot(gs, F_total, color='#FFFFFF', linewidth=2.0,
                     linestyle=':', marker='o', markersize=2,
                     markevery=3, label='Total', alpha=0.7)
            ax2.set_xlabel('Lateral g')
            ax2.set_ylabel('Device force (N)')
            ax2.set_title(f'Device allocation (util\u2264{params["target_util"]:.0%})',
                          color='white', fontsize=10)
            ax2.legend(**{**_leg_kw, 'ncol': 1, 'loc': 'upper left'})
            ax2.grid(True, alpha=0.2)

            # Annotate rear bias at final g
            bias_final = sweep['rear_aero_bias_pct'][-1]
            if F_total[-1] > 0:
                ax2.annotate(
                    f'Rear bias: {bias_final:.0f}%',
                    xy=(gs[-1], F_total[-1]),
                    xytext=(-60, 12), textcoords='offset points',
                    color='#aaa', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#666'))

            for ax in fig.get_axes():
                ax.set_facecolor('#000000')
                ax.tick_params(colors='#888')
                ax.xaxis.label.set_color('#aaa')
                ax.yaxis.label.set_color('#aaa')

            fig.tight_layout()
            self.curves.draw()
            self._aero_panel._status.setText(
                f'Sweep done: 0.1\u2013{params["lateral_g"]:.1f}g, {len(g_range)} pts')
        except Exception as e:
            import traceback; traceback.print_exc()
            self._aero_panel._status.setText(f'Error: {e}')

    # ==========================================================================
    #  AERO GEOMETRY (3D overlay)
    # ==========================================================================

    def _on_aero_geom(self, params: dict):
        """Push aero package geometry to the 3D viewer."""
        try:
            self.view3d.update_aero(params)
        except Exception as e:
            import traceback; traceback.print_exc()

    # ==========================================================================
    #  STYLE
    # ==========================================================================

    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow, QWidget, QScrollArea {
            background-color: #000000;
            color: #e0e0e0;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QTableWidget {
            background-color: #0a0a0a;
            alternate-background-color: #0f0f0f;
            gridline-color: #2a2a2a;
            color: #e0e0e0;
            border: none;
            font-size: 12px;
        }
        QHeaderView::section {
            background-color: #111111;
            color: #cccccc;
            border: 1px solid #2a2a2a;
            padding: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        QTableWidget::item:selected { background-color: #333333; color: white; }
        QListWidget {
            background-color: #0a0a0a;
            alternate-background-color: #0f0f0f;
            color: #e0e0e0;
            border: 1px solid #2a2a2a;
            font-size: 12px;
        }
        QListWidget::item:selected { background-color: #333333; }
        QRadioButton { spacing: 5px; font-size: 13px; }
        QCheckBox    { spacing: 5px; font-size: 13px; }
        QLabel       { color: #e0e0e0; font-size: 13px; }
        QSlider::groove:horizontal {
            height: 5px; background: #2a2a2a; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #888888; width: 15px; height: 15px;
            margin: -5px 0; border-radius: 8px;
        }
        QSlider::sub-page:horizontal { background: #666666; border-radius: 2px; }
        QDoubleSpinBox, QSpinBox {
            background-color: #0a0a0a;
            border: 1px solid #2a2a2a;
            color: #e0e0e0;
            padding: 3px 5px;
            border-radius: 3px;
            font-size: 12px;
        }
        QComboBox {
            background-color: #0a0a0a;
            border: 1px solid #2a2a2a;
            color: #e0e0e0;
            padding: 3px 5px;
            border-radius: 3px;
            font-size: 12px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #1a1a1a;
            color: #e0e0e0;
            selection-background-color: #333333;
        }
        QPushButton {
            background-color: #1a1a1a;
            border: 1px solid #444444;
            color: #e0e0e0;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 12px;
        }
        QPushButton:hover { background-color: #333333; }
        QScrollBar:vertical   { background: #050505; width: 8px; }
        QScrollBar::handle:vertical { background: #2a2a2a; border-radius: 4px; }
        QStatusBar { color: #888888; font-size: 11px; }
        QSplitter::handle { background: #2a2a2a; }
        QGroupBox {
            border: 1px solid #2a2a2a;
            margin-top: 6px;
            padding-top: 6px;
        }
        QGroupBox::title {
            color: #cccccc;
        }
        """)


# -- entry point ---------------------------------------------------------------

def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
