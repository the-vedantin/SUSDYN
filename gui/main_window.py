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
    Clamped symmetrically by total_rack_travel_mm.
"""

import sys
import json
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QSplitter, QStatusBar, QSizePolicy, QScrollArea,
    QGroupBox, QCheckBox, QMenuBar, QFileDialog, QMessageBox,
    QDialog, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize as plt_Normalize

from vahan import DoubleWishboneHardpoints
from vahan.solver import SuspensionConstraints, SolvedState, _norm, _rodrigues
from vahan.kinematics import KinematicMetrics, _intersect_2d
from vahan.metrics_catalog import (CATALOG, CATALOG_MAP, DEFAULT_Y_KEYS,
                                    compute_ackermann_post)

from gui.view3d import View3D, HP_NAMES
from gui.panels import (
    MotionPanel, CarParamsPanel, HardpointPanel,
    ValuesPanel, GraphPickerPanel, SteeringPanel, AlignmentPanel,
    CollapsibleSection, InverseKinematicsPanel, DynamicsPanel, DynamicsOptPanel,
    LoadsPanel, AeroPanel, SkidpadPanel, BrakeCalcPanel,
)
from vahan.optimizer import InverseSolver, DesignVar
from vahan.dynamics import (VehicleParams, SteadyStateSolver, SteadyStateResult,
                            DynamicsSensitivity, AeroDownforceSolver, AeroResult)
from vahan.transient import (TransientSolver, TransientParams, TransientInputs,
                             TransientResult, SteeringProfile,
                             SkidpadPathFollower)
from vahan.steering import SteeringGeometry

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


def _ackermann_from_pair(toe_left_deg: float, toe_right_deg: float,
                         wheelbase_m: float, front_track_m: float) -> float:
    """
    Ackermann % from a single (FL, FR) steer pair.

    Inputs are the absolute steer angles of each front wheel (deg).
    The larger-magnitude wheel is the inner (nearer turn centre); the bicycle
    model then gives the ideal Ackermann angle split for that turn radius.

    Returns NaN at / near zero steer (indeterminate).
    """
    d_L = abs(float(toe_left_deg))
    d_R = abs(float(toe_right_deg))
    if np.isnan(d_L) or np.isnan(d_R):
        return float('nan')

    d_inner = max(d_L, d_R)
    d_outer = min(d_L, d_R)

    # Near-zero steer: indeterminate.  Require a couple tenths of a degree
    # on the inner wheel before the geometry is meaningful.
    if d_inner < 0.2:
        return float('nan')

    avg_rad = np.radians((d_inner + d_outer) / 2.0)
    if abs(avg_rad) < 1e-6:
        return float('nan')

    R = wheelbase_m / np.tan(avg_rad)
    denom_inner = R - front_track_m / 2.0
    denom_outer = R + front_track_m / 2.0
    if abs(denom_inner) < 1e-6 or denom_outer < 1e-6:
        return float('nan')

    ideal_inner = np.degrees(np.arctan(wheelbase_m / denom_inner))
    ideal_outer = np.degrees(np.arctan(wheelbase_m / denom_outer))
    ideal_diff  = ideal_inner - ideal_outer
    if abs(ideal_diff) < 1e-9:
        return float('nan')

    return (d_inner - d_outer) / ideal_diff * 100.0


def _rack_travel_from_angle(steer_wheel_deg: float, steer_params: dict) -> float:
    """
    Rack translation in metres from steering wheel angle.
    Clamped symmetrically by total_rack_travel_mm.
    """
    ratio    = steer_params.get('rack_travel_per_rev_mm', 60.0)
    total    = steer_params.get('total_rack_travel_mm', 120.0)
    half     = total / 2.0
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
                    lines.append(f'{lbl}: {_fmt_num(float(yv))}')
                    if xd_ref is None:
                        xd_ref = xd
        if not lines:
            return

        x_ann = float(xd_ref[nearest_idx]) if xd_ref is not None else x_mouse
        xlabel = ax.get_xlabel() or 'x'
        lines.insert(0, f'{xlabel}: {_fmt_num(x_ann)}')
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
                      max_hw_deg: float = 0.0,
                      power_W: float = 0.0,
                      mass_kg: float = 290.0):
        """Plot dynamics sweep results with selectable graphs and corners."""
        self._hover_ann = None
        self.fig.clf()

        # Reset per-plot refs that the render loop checks (set by specific
        # plot blocks below when those plots are actually built)
        self._util_plot_idx = None
        self._swa_plot_idx  = None
        self._swa_max_hw    = None

        # Grip onset: lateral-g (or speed for accel-trajectory sweeps)
        # where any tire first saturates (util >= 1.0).  Past this point
        # the steady-state solver is extrapolating into a region the
        # car cannot physically hold, so any steering/US metrics
        # computed there are not trustworthy and should be marked
        # visually.
        self._g_grip_limit = None
        try:
            _x_ref = sweep.get('speed_mph',
                               sweep.get('lateral_g',
                                         sweep.get('longitudinal_g')))
            util_max = np.zeros_like(_x_ref)
            for c in ('FL', 'FR', 'RL', 'RR'):
                u = sweep.get(f'utilization_{c}')
                if u is not None:
                    util_max = np.maximum(util_max, u)
            _sat = np.where(util_max >= 1.0)[0]
            if len(_sat) > 0:
                self._g_grip_limit = float(_x_ref[_sat[0]])
        except Exception:
            pass

        # Determine x axis.
        #   • Longitudinal trajectory                : X = time (s)
        #   • Speed-sweep (sweep_by_speed result)    : X = speed (mph)
        #   • Lateral / combined g-sweep             : X = lat-g
        #   • Pure long-g sweep (legacy)             : X = lon-g
        is_longitudinal = 'longitudinal_g' in sweep and 'lateral_g' not in sweep
        is_acceleration = is_longitudinal and 'time_s' in sweep
        # Speed-sweep marker: 'speed_mph' present AND not the time-domain
        # acceleration trajectory (which also has speed_mph but uses
        # time_s as primary X).
        is_speed_sweep  = ('speed_mph' in sweep and 'time_s' not in sweep
                            and not is_longitudinal)
        if is_acceleration:
            g_arr = sweep['time_s']
            x_label = 'Time (s)'
        elif is_speed_sweep:
            g_arr = sweep['speed_mph']
            x_label = 'Speed (mph)'
        elif is_longitudinal:
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

        if 'speed' in graphs:
            # Speedometer speed at each sweep sample.  Pure physics, no
            # over-constraint:
            #
            #   • Longitudinal trajectory : speed_mph already in the
            #     result, grows monotonically over real time.  Initial
            #     condition for the integrator is start_speed_mph.
            #
            #   • Lateral / combined sweep : at a given turn radius R
            #     and lateral-g a_y, speed is fully determined by
            #         v = √(a_y · g_earth · R)
            #     start_speed and turn_radius together would over-
            #     constrain (3 vars, 2 equations) — so for these sweeps
            #     start_speed plays no role; turn_radius from the
            #     CarParams panel sets R.
            if is_acceleration and 'speed_mph' in sweep:
                v_arr = np.asarray(sweep['speed_mph'], float)
            else:
                lat_arr = np.asarray(sweep.get('lateral_g', g_arr), float)
                if turn_radius_m > 0:
                    v_ms = np.sqrt(np.maximum(lat_arr, 0.0) * 9.81
                                    * turn_radius_m)
                    v_arr = v_ms * 2.23694
                else:
                    # No turn radius defined — can't derive speed from g
                    v_arr = np.full_like(g_arr, float('nan'), dtype=float)
            plots.append(('Speed', 'Speed (mph)', [
                ('Speed', v_arr, '#4FC3F7', '-'),
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
            # Same constant-speed-cornering interpretation as the SWA
            # plot (above): R = v²/(a_y·g) so ack scales with lat-g
            # (zero at lat=0, growing with corner sharpness).
            start_v_mph = float(sweep.get('start_speed_mph', 0.0))
            v_ms = start_v_mph / 2.23694
            if us is not None and np.any(us) and wheelbase_m > 0 and v_ms > 1e-3 and not is_longitudinal:
                ack_rad = wheelbase_m * np.maximum(g_arr, 0.0) * 9.81 / (v_ms ** 2)
                ack_deg = np.degrees(ack_rad)
                total_steer = ack_deg + us
                if steer_ratio > 0:
                    hw_ack = ack_deg * steer_ratio
                    hw_req = total_steer * steer_ratio
                    plots.append(('Handwheel Angle', 'Steering wheel (deg)', [
                        ('Ackermann', hw_ack, '#555555', '--'),
                        ('Steering Angle', hw_req, '#4FC3F7', '-'),
                        ('Extra (US)', us * steer_ratio, '#BA68C8', '-.'),
                    ]))
                else:
                    plots.append(('Steer Correction', 'Front wheel angle (deg)', [
                        ('Ackermann', ack_deg, '#555555', '--'),
                        ('Steering Angle', total_steer, '#4FC3F7', '-'),
                        ('Extra (US)', us, '#BA68C8', '-.'),
                    ]))
            elif us is not None and np.any(us):
                plots.append(('Steer Correction', 'Extra steer (deg)', [
                    ('Extra (US)', us, '#BA68C8', '-'),
                ]))

        if 'steering_wheel_angle' in graphs:
            # Hand-wheel angle the driver must apply, plotted vs the sweep.
            # Constant-speed cornering with varying corner sharpness:
            #   v   = start_speed                                         [m/s]
            #   R   = v² / (a_y · g_earth)        (∞ at a_y = 0 → straight)
            #   ack = L / R                       (small-angle, road-wheel rad)
            #   SWA = (ack + understeer) · steer_ratio                  (deg)
            # At a_y = 0 the car is going straight, ack = 0 → SWA = 0,
            # which is what you'd expect physically (no input from the
            # driver when there's no lateral demand).  US correction is
            # optional — collapses onto Ackermann when zero.
            start_v_mph = float(sweep.get('start_speed_mph', 0.0))
            v_ms = start_v_mph / 2.23694
            if (wheelbase_m > 0 and v_ms > 1e-3 and steer_ratio > 0
                    and not is_longitudinal):
                # Geometric Ackermann at each lat-g (road-wheel deg)
                # ack_rad = L · a_y · g_earth / v²
                ack_rad = wheelbase_m * np.maximum(g_arr, 0.0) * 9.81 / (v_ms ** 2)
                ack_deg = np.degrees(ack_rad)
                us_arr = us if (us is not None) else np.zeros_like(g_arr)
                total_steer = ack_deg + us_arr            # road-wheel deg
                hw_req = total_steer * steer_ratio        # hand-wheel deg
                hw_ack = ack_deg * steer_ratio
                series = [
                    ('Ackermann',     hw_ack, '#555555', '--'),
                    ('Steering Angle', hw_req, '#4FC3F7', '-'),
                ]
                plots.append(('Steering Wheel Angle',
                              'Steering wheel (deg)', series))
                self._swa_plot_idx = len(plots) - 1
                self._swa_max_hw = None

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
            if title == 'Steering Wheel Angle' and self._swa_max_hw:
                # Shade the "beyond physical lock" band so you can see at
                # which lateral g the driver runs out of steering travel.
                mx = self._swa_max_hw
                y0, y1 = ax.get_ylim()
                if y1 > mx:
                    ax.axhspan(mx, max(y1, mx * 1.05),
                               facecolor='#E53935', alpha=0.10, zorder=0)

            # Grip-onset marker on steering-related plots.  Past this lateral
            # g the tires are saturated and the steady-state US gradient the
            # SWA / Steer-Correction curves are built on becomes unreliable
            # (e.g. a rear-saturating car shows the Steering Angle dropping
            # toward counter-steer — physically correct but past the
            # achievable operating point).
            if (self._g_grip_limit is not None and
                    title in ('Steering Wheel Angle', 'Steer Correction',
                              'Handwheel Angle', 'Understeer Gradient',
                              'Path Deviation')):
                ax.axvline(self._g_grip_limit, color='#FFC107',
                           lw=1.0, ls='--', alpha=0.6, zorder=3)
                # Shade the past-grip region so it's obvious it's not a
                # physically held operating point
                x_hi = g_arr[-1]
                if x_hi > self._g_grip_limit:
                    ax.axvspan(self._g_grip_limit, x_hi,
                               facecolor='#FFC107', alpha=0.06, zorder=0)
                # One compact label at the top of the plot
                y0, y1 = ax.get_ylim()
                ax.text(self._g_grip_limit, y1, f' grip: {self._g_grip_limit:.2f}g',
                        fontsize=7, color='#FFC107', va='top', ha='left',
                        alpha=0.8)

            ax.legend(fontsize=7, facecolor='#06060e', labelcolor='white',
                      framealpha=0.7, loc='best', handlelength=1.0, ncol=2)

            # No secondary speed axis on top — Speed is a separate plot
            # now (in the Graphs picker), so duplicating it as an axis
            # label up top is just clutter.  Remove the blue label.
            pass

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
#  GENERIC HOVER ANNOTATOR  (reusable for any matplotlib canvas)
# ==============================================================================

class HoverAnnotator:
    """Attach value-readout hover annotations to any matplotlib canvas.

    On mouse move, this reads every visible line plot in the axes under the
    cursor and pops up a small box showing the x-value plus each curve's
    y-value at the nearest sample.  No per-plot registration is required —
    the annotator scans ``fig.axes`` and ``ax.get_lines()`` at hover time,
    so it works automatically after every redraw.

    Works for:
        * Time-series plots (monotonic x) — snaps to nearest x-sample,
          reports every line's y at that x.
        * X–Y trajectory plots (non-monotonic x, e.g. path plots) — falls
          back to display-coordinate nearest-point.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self._ann = None
        self._bg = None
        canvas.mpl_connect('motion_notify_event', self._on_hover)
        canvas.mpl_connect('draw_event', self._on_draw)
        canvas.mpl_connect('figure_leave_event', lambda e: self._clear())

    # ── background caching for flicker-free overlay ─────────────────────
    def _on_draw(self, evt):
        try:
            self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        except Exception:
            self._bg = None

    def _blit(self):
        if self._bg is None:
            self.canvas.draw_idle()
            return
        try:
            self.canvas.restore_region(self._bg)
            if self._ann is not None:
                self._ann.axes.draw_artist(self._ann)
            self.canvas.blit(self.canvas.figure.bbox)
        except Exception:
            self._bg = None
            self.canvas.draw_idle()

    def _clear(self, redraw: bool = True):
        if self._ann is not None:
            try:
                self._ann.remove()
            except Exception:
                pass
            self._ann = None
            if redraw:
                self._blit()

    # ── hover callback ──────────────────────────────────────────────────
    def _on_hover(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self._clear()
            return

        # Accept every visible data line.  Matplotlib auto-assigns labels
        # like "_line0" / "_child3" to unlabeled plots, so filtering by
        # "starts with _" drops the yaw-rate / ay / roll traces entirely.
        # Instead, reject axhline/axvline (they have exactly 2 points and
        # use blended transforms) by requiring at least 3 data points.
        lines = [ln for ln in ax.get_lines()
                 if len(ln.get_xdata()) >= 3
                 and ln.get_linestyle() not in ('None', 'none')
                 and ln.get_visible()]
        if not lines:
            self._clear()
            return

        def _display_name(ln, ax, multi: bool) -> str:
            """Legend label if user-set, else fall back to y-label."""
            raw = ln.get_label() or ''
            if raw and not raw.startswith('_'):
                return raw
            ylbl = ax.get_ylabel() or 'y'
            # Strip "(units)" trailing piece so it reads cleanly.
            if '(' in ylbl:
                ylbl = ylbl.split('(')[0].strip()
            return ylbl if ylbl else 'y'

        multi = len(lines) > 1

        xd0 = np.asarray(lines[0].get_xdata(), float)
        x_monotonic = (np.all(np.diff(xd0) >= 0)
                       or np.all(np.diff(xd0) <= 0))

        if x_monotonic:
            # Snap to nearest x-sample; report every line's y at that x.
            idx = int(np.argmin(np.abs(xd0 - event.xdata)))
            x_ann = float(xd0[idx])
            rows = []
            xlabel = ax.get_xlabel() or 'x'
            rows.append(f'{xlabel}: {_fmt_num(x_ann)}')
            for ln in lines:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                if (len(xd) == len(xd0)
                        and len(xd) >= 3
                        and np.allclose(xd[:3], xd0[:3])):
                    i = idx
                else:
                    i = int(np.argmin(np.abs(xd - event.xdata)))
                if i >= len(yd):
                    continue
                yv = float(yd[i])
                if not np.isfinite(yv):
                    continue
                rows.append(f'{_display_name(ln, ax, multi)}: {_fmt_num(yv)}')
            if len(rows) < 2:
                self._clear()
                return
            x_at = x_ann
            y_at = event.ydata
        else:
            # Non-monotonic (e.g. X-Y trajectory): nearest point in
            # display coords across all lines.
            mouse_disp = np.array([event.x, event.y], float)
            best = None   # (line, idx, dist, x, y)
            for ln in lines:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                try:
                    pts = ax.transData.transform(
                        np.column_stack([xd, yd]))
                except Exception:
                    continue
                d = np.hypot(pts[:, 0] - mouse_disp[0],
                             pts[:, 1] - mouse_disp[1])
                i = int(np.argmin(d))
                if best is None or d[i] < best[2]:
                    best = (ln, i, d[i], float(xd[i]), float(yd[i]))
            if best is None or best[2] > 50:  # pixels
                self._clear()
                return
            ln, i, _, x_at, y_at = best
            rows = [
                f'{ax.get_xlabel() or "x"}: {_fmt_num(x_at)}',
                f'{ax.get_ylabel() or "y"}: {_fmt_num(y_at)}',
            ]
            name = _display_name(ln, ax, multi)
            raw = ln.get_label() or ''
            if raw and not raw.startswith('_'):
                rows.append(f'({name})')

        txt = '\n'.join(rows)
        self._clear(redraw=False)
        self._ann = ax.annotate(
            txt,
            xy=(x_at, y_at),
            xytext=(10, 10), textcoords='offset points',
            fontsize=7, color='#e0e0e0',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a',
                      ec='#444444', alpha=0.9),
            zorder=50,
        )
        self._blit()


def _fmt_num(v: float) -> str:
    """Compact human-readable float (3 sig figs, trailing zeros trimmed)."""
    if not np.isfinite(v):
        return 'nan'
    av = abs(v)
    if av == 0:
        return '0'
    if av >= 1000 or av < 0.01:
        return f'{v:.3g}'
    if av >= 100:
        return f'{v:.1f}'
    if av >= 10:
        return f'{v:.2f}'
    return f'{v:.3f}'


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
                 longitudinal_g: float = 0.0, aero_Fz: dict = None):
        super().__init__()
        self._solver = solver
        self._lat_g = lateral_g
        self._lon_g = longitudinal_g
        self._aero_Fz = aero_Fz

    def run(self):
        try:
            result = self._solver.solve(self._lat_g, self._lon_g,
                                        aero_Fz=self._aero_Fz)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class _DynamicsSweepWorker(QThread):
    """Runs lateral or longitudinal sweep off the main thread.

    When aero_Fz_per_g is provided, aero downforce scales with V^2
    (i.e. linearly with g at constant turn radius):
        aero_Fz(g) = {k: v * |g| for k, v in aero_Fz_per_g.items()}
    """
    finished = Signal(dict)   # sweep arrays
    failed   = Signal(str)

    def __init__(self, solver, g_min: float, g_max: float,
                 n_points: int, longitudinal_g: float = 0.0,
                 mode: str = 'lateral', lateral_g: float = 0.0,
                 aero_Fz_per_g: dict = None,
                 start_speed_mph: float = 0.0,
                 end_speed_mph: float = 200.0,
                 sweep_axis: str = 'g',
                 v_min_mph: float = 0.0,
                 v_max_mph: float = 60.0,
                 turn_radius_m: float = 10.0,
                 traj_direction: str = 'accel'):
        super().__init__()
        self._solver = solver
        self._g_min = g_min
        self._g_max = g_max
        self._n = n_points
        self._lon_g = longitudinal_g
        self._lat_g = lateral_g
        self._mode = mode
        self._aero_per_g = aero_Fz_per_g
        # Acceleration trajectory (longitudinal mode):  the X-axis is
        # speed, not g, and we trace the traction/power-limited envelope
        # from start_speed_mph up to end_speed_mph.  Lateral and
        # combined modes ignore these — they keep their existing g-sweep
        # semantics.
        self._start_speed_mph = float(start_speed_mph)
        self._end_speed_mph   = float(end_speed_mph)
        # Sweep axis: 'g' (sweep lat-g) or 'speed' (sweep speed at fixed R).
        self._sweep_axis    = str(sweep_axis or 'g')
        self._v_min_mph     = float(v_min_mph)
        self._v_max_mph     = float(v_max_mph)
        self._turn_radius_m = float(turn_radius_m)
        self._traj_direction = str(traj_direction or 'accel')

    def _aero_at_g(self, g_val: float) -> dict | None:
        if self._aero_per_g is None:
            return None
        g = abs(g_val)
        return {k: v * g for k, v in self._aero_per_g.items()}

    def run(self):
        try:
            # For V^2-scaled aero we must call solve() per-point ourselves
            # so each g gets its own scaled aero_Fz.
            if self._aero_per_g is not None:
                result = self._sweep_with_aero()
            elif self._mode == 'longitudinal':
                # Time-domain acceleration trajectory from start_speed_mph.
                # Speed grows monotonically over real seconds, achieved
                # longitudinal-g traces the traction-then-power-then-drag
                # envelope.  Bounded naturally by drag (CdA), no
                # hardcoded duration.  This mode ignores sweep_axis.
                result = self._solver.sweep_acceleration_trajectory(
                    start_speed_mph=self._start_speed_mph,
                    lateral_g=self._lat_g,
                    direction=self._traj_direction,
                    end_speed_mph=self._end_speed_mph,
                    target_lon_g=self._lon_g)
                result['start_speed_mph'] = self._start_speed_mph
                result['traj_direction']  = self._traj_direction
            elif self._sweep_axis == 'speed':
                # Sweep by speed (X = mph) at fixed turn radius.  Lat-g
                # is derived per-step from v² = a_y · g_e · R so the
                # operating points stay self-consistent.  Works for both
                # pure lateral (lon=0) and combined (lon != 0) — the
                # constant longitudinal-g rides along.
                result = self._solver.sweep_by_speed(
                    v_min_mph=self._v_min_mph,
                    v_max_mph=self._v_max_mph,
                    turn_radius_m=self._turn_radius_m,
                    n_points=self._n,
                    longitudinal_g=self._lon_g)
                result['start_speed_mph'] = self._start_speed_mph
            elif self._mode == 'combined':
                result = self._solver.sweep_combined(
                    lat_range=(self._g_min, self._g_max),
                    lon_g=self._lon_g,
                    n_points=self._n)
                result['start_speed_mph'] = self._start_speed_mph
            else:
                result = self._solver.sweep_lateral_g(
                    g_range=(self._g_min, self._g_max),
                    n_points=self._n,
                    longitudinal_g=self._lon_g)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

    def _sweep_with_aero(self) -> dict:
        """Manual sweep loop: each g-point gets V^2-scaled aero_Fz."""
        import numpy as _np
        from scipy.ndimage import uniform_filter1d as _uf
        from vahan.kinematics import KinematicMetrics

        if self._mode == 'longitudinal':
            g_arr = _np.linspace(self._g_min, self._g_max, self._n)
            x_key = 'longitudinal_g'
        else:
            g_arr = _np.linspace(self._g_min, self._g_max, self._n)
            x_key = 'lateral_g'

        keys = ['roll_angle_deg', 'pitch_angle_deg',
                'rc_height_front_mm', 'rc_height_rear_mm',
                'elastic_lt_front_N', 'elastic_lt_rear_N',
                'geometric_lt_front_N', 'geometric_lt_rear_N',
                'understeer_gradient_deg']
        corner_keys = ['Fz', 'travel', 'camber', 'utilization']

        out = {x_key: g_arr}
        for k in keys:
            out[k] = _np.zeros(self._n)
        for ck in corner_keys:
            for lbl in ['FL', 'FR', 'RL', 'RR']:
                out[f'{ck}_{lbl}'] = _np.zeros(self._n)

        self._solver._warm = {}
        for i, gv in enumerate(g_arr):
            if self._mode == 'longitudinal':
                lat_g, lon_g = self._lat_g, gv
            elif self._mode == 'combined':
                lat_g, lon_g = gv, self._lon_g
            else:
                lat_g, lon_g = gv, self._lon_g

            # V^2-scaled aero: scale by the g magnitude being swept
            aero = self._aero_at_g(lat_g if self._mode != 'longitudinal' else self._lat_g)
            r = self._solver.solve(lat_g, lon_g, aero_Fz=aero)

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

        # Smooth
        for k in ['understeer_gradient_deg']:
            if len(out[k]) >= 5:
                out[k] = _uf(out[k], size=5, mode='nearest')
        for lbl in ['FL', 'FR', 'RL', 'RR']:
            uk = f'utilization_{lbl}'
            if len(out[uk]) >= 3:
                out[uk] = _uf(out[uk], size=3, mode='nearest')
        return out


class _SensitivityWorker(QThread):
    """Runs dynamics sensitivity analysis off the main thread."""
    finished = Signal(dict)
    failed   = Signal(str)

    def __init__(self, sens: DynamicsSensitivity,
                 lateral_g: float, longitudinal_g: float,
                 turn_radius_m: float = None):
        super().__init__()
        self._sens = sens
        self._lat_g = lateral_g
        self._lon_g = longitudinal_g
        self._turn_radius_m = turn_radius_m

    def run(self):
        try:
            result = self._sens.analyze(self._lat_g, self._lon_g,
                                        turn_radius_m=self._turn_radius_m)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class _TransientSimWorker(QThread):
    """Runs TransientSolver.simulate() off the main thread."""
    finished = Signal(object)   # TransientResult
    failed   = Signal(str)

    def __init__(self, solver: TransientSolver, inputs: TransientInputs):
        super().__init__()
        self._solver = solver
        self._inputs = inputs

    def run(self):
        try:
            result = self._solver.simulate(self._inputs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class _ReportWorker(QThread):
    """Runs dynamics sweeps + docx generation off the main thread.

    Receives a partial data dict (kinematic results + loads already computed
    on the main thread) and fills in the dynamics sweeps using the user's
    current panel parameters before calling generate_report().
    """
    progress = Signal(str, int)   # (label, 0–100)
    finished = Signal(str)        # output_path on success
    failed   = Signal(str)        # error message on failure

    def __init__(self, ss_solver, data: dict, output_path: str,
                 sweep_params: dict = None):
        super().__init__()
        self._solver = ss_solver
        self._data   = data
        self._path   = output_path
        self._sp     = sweep_params or {}

    def run(self):
        try:
            from vahan.report_gen import generate_report

            sp = self._sp
            aero_per_g = sp.get('aero_Fz_per_g')  # None when aero is off

            # ── Cornering sweep (uses panel's g range + lon-g) ────────────
            self.progress.emit('Cornering sweep…', 10)
            g_min = sp.get('g_min', 0.0)
            g_max = sp.get('g_max', 1.5)
            n_pts = sp.get('n_points', 41)
            lon_g_corn = sp.get('lon_g_cornering', 0.0)

            if aero_per_g:
                # V²-scaled: at constant turn radius V² ∝ g, so downforce
                # scales linearly with g.  Must loop manually.
                import numpy as _np
                g_arr = _np.linspace(g_min, g_max, n_pts)
                keys = ['roll_angle_deg', 'pitch_angle_deg',
                        'rc_height_front_mm', 'rc_height_rear_mm',
                        'elastic_lt_front_N', 'elastic_lt_rear_N',
                        'geometric_lt_front_N', 'geometric_lt_rear_N',
                        'understeer_gradient_deg']
                corner_keys = ['Fz', 'travel', 'camber', 'utilization']
                dyn_corn = {'lateral_g': g_arr}
                for k in keys:
                    dyn_corn[k] = _np.zeros(n_pts)
                for ck in corner_keys:
                    for lbl in ['FL', 'FR', 'RL', 'RR']:
                        dyn_corn[f'{ck}_{lbl}'] = _np.zeros(n_pts)
                self._solver._warm = {}
                for i, lg in enumerate(g_arr):
                    aero_at_g = {k: v * abs(lg) for k, v in aero_per_g.items()}
                    r = self._solver.solve(lg, lon_g_corn, aero_Fz=aero_at_g)
                    dyn_corn['roll_angle_deg'][i] = r.roll_angle_deg
                    dyn_corn['pitch_angle_deg'][i] = r.pitch_angle_deg
                    dyn_corn['rc_height_front_mm'][i] = r.rc_height_front_m * 1000
                    dyn_corn['rc_height_rear_mm'][i] = r.rc_height_rear_m * 1000
                    dyn_corn['elastic_lt_front_N'][i] = r.elastic_lt_front_N
                    dyn_corn['elastic_lt_rear_N'][i] = r.elastic_lt_rear_N
                    dyn_corn['geometric_lt_front_N'][i] = r.geometric_lt_front_N
                    dyn_corn['geometric_lt_rear_N'][i] = r.geometric_lt_rear_N
                    dyn_corn['understeer_gradient_deg'][i] = r.understeer_gradient_deg
                    for lbl in ['FL', 'FR', 'RL', 'RR']:
                        dyn_corn[f'Fz_{lbl}'][i] = r.Fz.get(lbl, 0)
                        dyn_corn[f'travel_{lbl}'][i] = r.travel.get(lbl, 0)
                        dyn_corn[f'camber_{lbl}'][i] = r.camber.get(lbl, 0)
                        dyn_corn[f'utilization_{lbl}'][i] = r.utilization.get(lbl, 0)
            else:
                dyn_corn = self._solver.sweep_lateral_g(
                    g_range=(g_min, g_max),
                    n_points=n_pts,
                    longitudinal_g=lon_g_corn)
            self._data['dyn_cornering'] = dyn_corn

            # ── Acceleration trajectory ───────────────────────────────────
            self.progress.emit('Acceleration trajectory…', 28)
            start_mph = sp.get('start_speed_mph', 0.0)
            target_accel = sp.get('target_lon_g_accel', 1.5)
            dyn_accel = self._solver.sweep_acceleration_trajectory(
                start_speed_mph=start_mph,
                target_lon_g=abs(target_accel),
                direction='accel',
                end_speed_mph=0.0)
            self._data['dyn_accel'] = dyn_accel

            # ── Braking trajectory ────────────────────────────────────────
            self.progress.emit('Braking trajectory…', 46)
            brake_start = sp.get('brake_start_mph', 60.0)
            target_brake = sp.get('target_lon_g_brake', -1.5)
            dyn_brake = self._solver.sweep_acceleration_trajectory(
                start_speed_mph=brake_start,
                target_lon_g=-abs(target_brake),
                direction='brake',
                end_speed_mph=0.0)
            self._data['dyn_brake'] = dyn_brake

            # ── DOCX rendering ────────────────────────────────────────────
            self.progress.emit('Rendering report pages…', 55)

            def _prog_inner(msg, pct):
                self.progress.emit(msg, 55 + int(pct * 0.44))

            generate_report(self._path, self._data, progress_cb=_prog_inner)
            self.finished.emit(self._path)

        except Exception as exc:
            import traceback
            self.failed.emit(f'{exc}\n{traceback.format_exc()}')


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
                           'cg_x_mm': 0., 'cg_y_mm': 845., 'cg_z_mm': 280.,
                           'front_brake_bias_pct': 65.}
        self._steer     = {'rack_travel_per_rev_mm': 60.,
                           'total_rack_travel_mm': 100.}
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
        # (aero geom feeds solver only, no 3D visuals to push)

    # ==========================================================================
    #  BUILD UI
    # ==========================================================================

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('File')
        save_act = fm.addAction('Save Project…')
        save_act.setToolTip('Save all hardpoints (FL/FR/RL/RR), vehicle params, and settings to JSON')
        load_act = fm.addAction('Load Project…')
        save_act.triggered.connect(self._save_project)
        load_act.triggered.connect(self._load_project)
        fm.addSeparator()
        export_rpt_act = fm.addAction('Export Report…')
        export_rpt_act.setToolTip(
            'Generate a Vehicle Dynamics Report (.docx) with all graphs and '
            'auto-analysis — opens and edits cleanly in Google Docs')
        export_rpt_act.triggered.connect(self._export_report)

        vm = mb.addMenu('View')
        hp_act = vm.addAction('All Hardpoints…')
        hp_act.triggered.connect(self._show_all_hardpoints)

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Project (all hardpoints + vehicle params)',
            '', 'Vahan Project (*.vahan);;JSON (*.json)')
        if not path:
            return
        mp = self._motion_panel
        # version 2: every panel input is captured under "panels" so the
        # full state of the dynamics / transient / loads / aero pages
        # round-trips through save→load.  Older v1 files still load —
        # the load path falls back to defaults on missing blocks.
        data = {
            'version': 2,
            'front_hp':  {k: v.tolist() for k, v in self._front_hp.items()},
            'rear_hp':   {k: v.tolist() for k, v in self._rear_hp.items()},
            'front_arb': {k: v.tolist() for k, v in self._front_arb.items()},
            'rear_arb':  {k: v.tolist() for k, v in self._rear_arb.items()},
            'car':       self._car.copy(),
            'steer':     self._steer.copy(),
            'alignment': self._alignment.copy(),
            'motion': {
                'type':              mp.motion,
                'min':               mp.min_val,
                'max':               mp.max_val,
                'stroke_mm':         self._motion_panel.stroke_mm,
                'preload_front_mm':  self._motion_panel.preload_front_mm,
                'preload_rear_mm':   self._motion_panel.preload_rear_mm,
                'fully_extended_mm': self._motion_panel.fully_extended_mm,
            },
            'panels': {
                'dynamics': self._dynamics_panel.get_state(),
                'skidpad':  self._skidpad_panel.get_state(),
                'loads':    self._loads_panel.get_state(),
                'aero':     self._aero_panel.get_state(),
                'brake_calc': self._brake_calc_panel.get_state(),
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(
            f'Saved all hardpoints (FL/RL + ARB) + vehicle params + panel state: {path}', 5000)

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
                car_data.setdefault('cg_y_mm', 845.)
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

            # Restore panel state (v2+).  Old v1 files have no "panels"
            # block; in that case the four panels keep their defaults.
            # Each panel's set_state() is missing-key tolerant so a
            # partial dict (e.g. an older v2 that didn't have aero)
            # still loads cleanly.
            panels = data.get('panels', {})
            if isinstance(panels, dict):
                if 'dynamics' in panels:
                    self._dynamics_panel.set_state(panels['dynamics'])
                if 'skidpad' in panels:
                    self._skidpad_panel.set_state(panels['skidpad'])
                if 'loads' in panels:
                    self._loads_panel.set_state(panels['loads'])
                if 'aero' in panels:
                    self._aero_panel.set_state(panels['aero'])
                if 'brake_calc' in panels:
                    self._brake_calc_panel.set_state(panels['brake_calc'])

            self._rebuild_solvers()
            self._run_sweep()
            self._update_3d()
            self.statusBar().showMessage(f'Loaded: {path}', 4000)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    def _show_all_hardpoints(self):
        """Popup showing all 4 corners' hardpoints (FL input, FR mirrored, RL input, RR mirrored)."""
        fl = self._front_hp
        fr = _mirror_x(fl)
        rl = self._rear_hp
        rr = _mirror_x(rl)

        # Merge ARB points into each corner's dict (right side = X-mirrored)
        fl_full = {**fl, **self._front_arb}
        fr_full = {**fr, **_mirror_x(self._front_arb)}
        rl_full = {**rl, **self._rear_arb}
        rr_full = {**rr, **_mirror_x(self._rear_arb)}

        corners = [('FL', fl_full), ('FR', fr_full), ('RL', rl_full), ('RR', rr_full)]
        names = list(fl.keys()) + list(self._front_arb.keys())

        dlg = QDialog(self)
        dlg.setWindowTitle('All Hardpoints (mm)')
        dlg.setMinimumSize(920, 700)
        dlg.setStyleSheet('''
            QDialog { background: #000; color: #e0e0e0; }
            QLabel  { color: #e0e0e0; }
            QTableWidget { background: #0a0a0a; color: #e0e0e0;
                           gridline-color: #2a2a2a; border: none; font-size: 11px; }
            QHeaderView::section { background: #111; color: #ccc;
                                   border: 1px solid #2a2a2a; padding: 3px;
                                   font-weight: bold; font-size: 11px; }
        ''')
        lay = QVBoxLayout(dlg)

        note = QLabel('FL and RL are input values.  FR and RR are X-mirrored.  '
                       'Save Project exports all hardpoints + vehicle params to JSON.')
        note.setStyleSheet('color: #FFA726; font-size: 11px; padding: 4px;')
        note.setWordWrap(True)
        lay.addWidget(note)

        # 4 columns per corner (X, Y, Z) = 12 data cols + 1 name col = 13
        ncols = 1 + 4 * 3  # name + FL(x,y,z) + FR(x,y,z) + RL(x,y,z) + RR(x,y,z)
        tbl = QTableWidget(len(names), ncols)

        headers = ['Point']
        for label, _ in corners:
            headers += [f'{label} X', f'{label} Y', f'{label} Z']
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, ncols):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            tbl.setColumnWidth(c, 62)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        # Color the corner headers
        corner_colors = {'FL': '#4FC3F7', 'FR': '#81C784', 'RL': '#FFB74D', 'RR': '#CE93D8'}

        for ri, name in enumerate(names):
            it = QTableWidgetItem(name)
            it.setForeground(QColor('#cccccc'))
            f = it.font(); f.setBold(True); it.setFont(f)
            tbl.setItem(ri, 0, it)

            for ci, (label, hp_dict) in enumerate(corners):
                pt = hp_dict.get(name)
                if pt is None:
                    continue
                mm = pt * 1000.0
                color = corner_colors[label]
                for ax in range(3):
                    col = 1 + ci * 3 + ax
                    cell = QTableWidgetItem(f'{mm[ax]:.2f}')
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    cell.setForeground(QColor(color))
                    tbl.setItem(ri, col, cell)

        tbl.resizeRowsToContents()
        lay.addWidget(tbl)

        # ── buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        _btn_style = ('QPushButton { background: #333; color: white; padding: 6px 16px; '
                      'border-radius: 3px; } QPushButton:hover { background: #555; }')

        _btn_green = ('QPushButton { background: #2E7D32; color: white; padding: 6px 16px; '
                      'border-radius: 3px; font-weight: bold; } '
                      'QPushButton:hover { background: #388E3C; }')
        _btn_purple = ('QPushButton { background: #6A1B9A; color: white; padding: 6px 16px; '
                       'border-radius: 3px; font-weight: bold; } '
                       'QPushButton:hover { background: #8E24AA; }')

        copy_btn = QPushButton('Copy to Clipboard')
        copy_btn.setStyleSheet(_btn_green)
        def _copy():
            lines = ['All Hardpoints (mm)', '=' * 90]
            hdr = f'{"Point":22s}'
            for label, _ in corners:
                hdr += f'  {label + " X":>8s} {label + " Y":>8s} {label + " Z":>8s}'
            lines.append(hdr)
            lines.append('-' * 90)
            for name in names:
                row_txt = f'{name:22s}'
                for label, hp_dict in corners:
                    pt = hp_dict.get(name)
                    if pt is not None:
                        mm = pt * 1000.0
                        row_txt += f'  {mm[0]:8.2f} {mm[1]:8.2f} {mm[2]:8.2f}'
                    else:
                        row_txt += f'  {"—":>8s} {"—":>8s} {"—":>8s}'
                lines.append(row_txt)
            QApplication.clipboard().setText('\n'.join(lines))
            copy_btn.setText('Copied!')
        copy_btn.clicked.connect(_copy)
        btn_row.addWidget(copy_btn)

        # Copy CSV for FeatureScript paste
        onshape_btn = QPushButton('Copy for Onshape')
        onshape_btn.setStyleSheet(_btn_purple)
        onshape_btn.setToolTip('Copy as CSV for pasting into the Vahan Hardpoints FeatureScript')
        def _copy_onshape():
            csv_lines = []
            for name in names:
                vals = []
                for label, hp_dict in corners:
                    pt = hp_dict.get(name)
                    if pt is not None:
                        mm = pt * 1000.0
                        vals.extend([f'{mm[0]:.2f}', f'{mm[1]:.2f}', f'{mm[2]:.2f}'])
                    else:
                        vals.extend(['0', '0', '0'])
                csv_lines.append(f'{name},{",".join(vals)}')
            QApplication.clipboard().setText('|'.join(csv_lines))
            onshape_btn.setText('Copied!')
        onshape_btn.clicked.connect(_copy_onshape)
        btn_row.addWidget(onshape_btn)

        # Export JSON for Onshape upload
        json_btn = QPushButton('Export JSON')
        json_btn.setStyleSheet(_btn_style)
        json_btn.setToolTip('Save JSON file for Onshape tab import')
        def _export_json():
            path, _ = QFileDialog.getSaveFileName(
                dlg, 'Export Hardpoints for Onshape',
                'hardpoints.json', 'JSON (*.json)')
            if path:
                data = {}
                for label, hp_dict in corners:
                    data[label] = {}
                    for name in names:
                        pt = hp_dict.get(name)
                        if pt is not None:
                            mm = pt * 1000.0
                            data[label][name] = [round(mm[0], 2), round(mm[1], 2), round(mm[2], 2)]
                import json as _json
                with open(path, 'w') as f:
                    _json.dump(data, f, indent=2)
                json_btn.setText('Saved!')
        json_btn.clicked.connect(_export_json)
        btn_row.addWidget(json_btn)

        btn_row.addStretch()
        close_btn = QPushButton('Close')
        close_btn.setStyleSheet(_btn_style)
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.exec()

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
        self._skidpad_panel = SkidpadPanel()
        self._skidpad_follower = None   # SkidpadPathFollower from last run
        self._aero_panel = AeroPanel()
        self._loads_panel = LoadsPanel()
        self._brake_calc_panel = BrakeCalcPanel()
        ik_inner = QWidget()
        ik_layout = QVBoxLayout(ik_inner)
        ik_layout.setContentsMargins(0, 0, 0, 0)
        ik_layout.setSpacing(4)
        ik_layout.addWidget(self._ik_panel)
        ik_layout.addWidget(self._dynamics_panel)
        ik_layout.addWidget(self._dynamics_opt_panel)
        ik_layout.addWidget(self._skidpad_panel)
        ik_layout.addWidget(self._aero_panel)
        ik_layout.addWidget(self._loads_panel)
        ik_layout.addWidget(self._brake_calc_panel)
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
        self._skidpad_panel.simulate_requested.connect(self._on_skidpad_simulate)
        self._skidpad_panel.signals_changed.connect(self._on_skidpad_signals)
        self._aero_panel.solve_requested.connect(self._on_aero_solve)
        self._aero_panel.sweep_requested.connect(self._on_aero_sweep)
        self._dynamics_panel.apply_aero_toggled.connect(self._on_apply_aero_toggle)
        self._loads_panel.loads_requested.connect(self._on_compute_loads)
        self._brake_calc_panel.compute_requested.connect(self._on_compute_brakes)
        self._motion_panel.damper_params_changed.connect(self._on_damper_limits)
        self._motion_panel.apply_sag_requested.connect(self._on_apply_sag)
        # Push initial damper limits to IK panel + sag display.
        # Deferred so the hardpoints/solvers finish initialising first
        # (needed for live MR lookup via _query_static_mr).
        QTimer.singleShot(0, self._refresh_sag)

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
        Return (spring_min_m, spring_max_m) based on stroke and computed
        static sag.

        At design position (travel=0) the spring has length `spring_0`.
        At static the damper has compressed by `sag_mm` from full droop, so:
            full_droop_spring = spring_0 + sag_mm          (damper extends out)
            full_bump_spring  = spring_0 − (stroke − sag_mm)  (damper bottoms)

        Sag is computed from preload + spring rate + corner weight + MR
        via VehicleParams.static_sag() — no longer a user input.
        """
        try:
            st0 = solver.solve(0.)
            spring_0  = st0.spring_length
            stroke_m  = self._motion_panel.stroke_mm / 1000.

            # Determine which axle this corner belongs to from the solver's hardpoints
            # (front solvers share lca Y with FL; rear share with RL).  We
            # fall back to the front-axle sag if we can't tell.
            is_front = True
            try:
                for lbl in ('FL', 'FR'):
                    if self._solvers.get(lbl) is solver:
                        is_front = True; break
                else:
                    for lbl in ('RL', 'RR'):
                        if self._solvers.get(lbl) is solver:
                            is_front = False; break
            except Exception:
                pass

            # Pull the latest sag dict via the motion panel label text is fragile —
            # recompute directly so this works even before the first paint.
            dyn_params = self._dynamics_panel.get_params()
            if hasattr(self, '_car') and isinstance(self._car, dict):
                dyn_params.setdefault('wheelbase_m',
                                      self._car.get('wheelbase_mm', 1530) / 1000.)
                dyn_params.setdefault('front_track_m',
                                      self._car.get('track_f_mm', 1220) / 1000.)
                dyn_params.setdefault('rear_track_m',
                                      self._car.get('track_r_mm', 1200) / 1000.)
                dyn_params.setdefault('cg_to_front_axle_m',
                                      self._car.get('cg_y_mm', 765) / 1000.)
            veh = VehicleParams(**dyn_params)
            sag = veh.static_sag(
                preload_front_mm=self._motion_panel.preload_front_mm,
                preload_rear_mm=self._motion_panel.preload_rear_mm,
                stroke_mm=self._motion_panel.stroke_mm,
                mr_front=self._query_static_mr('front'),
                mr_rear=self._query_static_mr('rear'),
            )
            sag_m = (sag['sag_shock_front_mm'] if is_front
                     else sag['sag_shock_rear_mm']) / 1000.
            droop_len = spring_0 + sag_m
            bump_len  = spring_0 - (stroke_m - sag_m)
            return bump_len, droop_len
        except Exception:
            return 0., 1.

    def _probe_static_ackermann(self, ref_steer_wheel_deg: float = 25.0) -> float:
        """
        Compute a representative Ackermann % by probing FL and FR at a
        reference steering-wheel angle.  Ackermann is a *geometry* property
        of the steering linkage — independent of heave/roll/pitch — so this
        gives a meaningful live readout even when the motion panel is not
        in steer mode (current rack = 0 would otherwise collapse to NaN).

        Returns NaN if the solver fails or the geometry is degenerate.
        """
        try:
            rt_m = _rack_travel_from_angle(ref_steer_wheel_deg, self._steer)
            corners = self._all_corner_hp()
            toes = {}
            for lbl in ('FL', 'FR'):
                hp_d    = corners[lbl]
                steered = self._steered_hp(hp_d, rt_m, True)
                d       = hp_d['tie_rod_outer'] - hp_d['tie_rod_inner']
                tierod_len_sq = float(d @ d)
                solver = SuspensionConstraints(
                    _hp_obj(steered),
                    tierod_len_sq=tierod_len_sq,
                    pushrod_body='uca',
                )
                st = solver.solve(0.)
                m  = KinematicMetrics(st, 'left' if lbl == 'FL' else 'right')
                toes[lbl] = float(m.toe)

            wb = self._car.get('wheelbase_mm', 1537.) / 1000.
            ft = self._car.get('track_f_mm', 1222.) / 1000.
            return _ackermann_from_pair(toes['FL'], toes['FR'], wb, ft)
        except Exception:
            return float('nan')

    def _build_steering_geometry(self, veh: VehicleParams
                                 ) -> Optional[SteeringGeometry]:
        """
        Build a ``SteeringGeometry`` by probing the front kinematics at a
        grid of rack positions.

        Returns ``None`` if the probe fails completely — callers should
        treat a ``None`` result as "no rack saturation known, fall back
        to ``veh.max_steer_angle_deg``".

        The probe re-uses the same pattern as ``_probe_static_ackermann``:
        apply ``_steered_hp`` to shift the inboard tie-rod end by
        ``rack_m`` in +X, build a fresh ``SuspensionConstraints`` with
        the original tie-rod length, solve at ``travel = 0``, and read
        the toe angle out of ``KinematicMetrics``.
        """
        try:
            corners = self._all_corner_hp()
        except Exception:
            return None

        rack_cfg = self._steer or {}
        rack_per_rev = float(rack_cfg.get('rack_travel_per_rev_mm',
                                          veh.rack_travel_per_rev_mm))
        total_rack   = float(rack_cfg.get('total_rack_travel_mm',
                                          veh.total_rack_travel_mm))

        def _probe(rack_m: float, side: str) -> float:
            """road-wheel angle at given rack position, side='FL'|'FR'."""
            try:
                hp_d = corners[side]
                steered = self._steered_hp(hp_d, rack_m, True)
                d = hp_d['tie_rod_outer'] - hp_d['tie_rod_inner']
                tierod_len_sq = float(d @ d)
                solver = SuspensionConstraints(
                    _hp_obj(steered),
                    tierod_len_sq=tierod_len_sq,
                    pushrod_body='uca',
                )
                st = solver.solve(0.)
                m  = KinematicMetrics(st, 'left' if side == 'FL' else 'right')
                # KinematicMetrics.toe is in radians; sign convention
                # there matches +rack → +toe for the rack-driven side.
                return float(m.toe)
            except Exception:
                return float('nan')

        try:
            return SteeringGeometry.from_probe(
                front_solver_factory=_probe,
                front_hp_fl=corners.get('FL', {}),
                front_hp_fr=corners.get('FR', {}),
                rack_travel_per_rev_mm=rack_per_rev,
                total_rack_travel_mm=total_rack,
            )
        except Exception:
            return None

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
            # Solvers are always built with sag_offset_m=0 — no hidden
            # geometric shift.  The "Apply Sag to Hardpoints" button on
            # the MotionPanel is the only path that changes geometry
            # because of damper params, and it does so by rewriting the
            # actual hardpoints (not by setting an offset).

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

    def _compute_arb_geometry_from_kinematics(self, axle: str = 'F') -> dict | None:
        """
        Derive ARB arm length, half-length and motion ratio from the kinematic model.

        These three numbers are uniquely determined by the geometry once the
        ARB hardpoints (`arb_pivot`, `arb_arm_end`, `arb_drop_top`) and the
        rocker chain are in place — there is no need for the user to type
        them.  Only the bar diameter (cross-section) and material properties
        (G, E) remain as inputs.

        Parameters
        ----------
        axle : 'F' or 'R'

        Returns
        -------
        dict | None
            ``{'arm_length_mm', 'half_length_mm', 'mr'}`` or ``None`` if the
            kinematic data is not yet available.

        Notes
        -----
        - **arm_length** = ‖arb_arm_end − arb_pivot‖ at design.
        - **half_length** = |arb_pivot.x|, i.e. half of the lateral pivot-to-
          pivot distance (the active twisting span of a symmetric bar).
        - **MR** = wheel_travel / arm_tip_travel, dimensionless, by central
          difference: solve corner kinematics at travel = ±1 mm, walk the
          rocker rotation to find the new world position of `arb_drop_top`,
          then run the bell-crank solver to get the bar twist θ; arm tip
          travel = arm_length·θ for small θ.  This matches the workbook
          (B46 / C46) convention so MR > 1 means wheel moves more than arm
          tip.
        """
        label = 'FL' if axle == 'F' else 'RL'
        solver = self._solvers.get(label)
        arb_hp = self._front_arb if axle == 'F' else self._rear_arb
        if solver is None or not arb_hp:
            return None

        try:
            pivot    = arb_hp['arb_pivot']
            arm_end  = arb_hp['arb_arm_end']
            drop_top = arb_hp['arb_drop_top']
        except (KeyError, TypeError):
            return None

        # Geometric values — straight from hardpoints, no perturbation.
        arm_len_m  = float(np.linalg.norm(arm_end - pivot))
        half_len_m = float(abs(pivot[0]))

        # MR — central-difference perturbation through the rocker → bell crank
        # chain.  ±1 mm wheel travel keeps us deep in the linear regime where
        # arm_tip_travel ≈ arm_length · bar_twist_angle, so MR is independent
        # of perturbation magnitude.
        dt = 0.001
        try:
            st_p = solver.solve(+dt)
            st_m = solver.solve(-dt)
            rp = solver.hp.rocker_pivot
            ax_pt = getattr(solver.hp, 'rocker_axis_pt',
                            rp + np.array([0., 0.0254, 0.]))
            r_axis = _norm(ax_pt - rp)
            arm_dt = drop_top - rp
            dt_w_p = rp + _rodrigues(arm_dt, r_axis, st_p.rocker_angle)
            dt_w_m = rp + _rodrigues(arm_dt, r_axis, st_m.rocker_angle)
            ang_p, _, _ = self._solve_arb_bellcrank(dt_w_p, arb_hp)
            ang_m, _, _ = self._solve_arb_bellcrank(dt_w_m, arb_hp)
            d_ang = float(ang_p - ang_m)            # rad over 2·dt of travel
            arm_tip_disp = abs(arm_len_m * d_ang)   # arc length at arm tip
            if arm_tip_disp > 1e-9:
                mr = (2.0 * dt) / arm_tip_disp
            else:
                return None
        except Exception:
            return None

        if not np.isfinite(mr) or mr <= 0.0:
            return None

        return {
            'arm_length_mm':  arm_len_m  * 1000.0,
            'half_length_mm': half_len_m * 1000.0,
            'mr':             float(mr),
        }

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

                # ── Ackermann %: post-process from FL+FR toe curves ──────────
                # _all_metrics leaves ackermann as NaN because the post-solve
                # hook (compute_ackermann_post) needs the full sweep.  Here we
                # already have FL and FR steer angles directly — no mirror
                # symmetry assumption needed — so compute per-step.
                # NB: res_fl['toe'] has the static-toe alignment offset added;
                # subtract it off so we feed raw kinematic steer angles into
                # the Ackermann math (a constant toe-in bias on both wheels
                # would otherwise ruin the |inner|−|outer| relationship near
                # zero steer).
                wb       = self._car.get('wheelbase_mm', 1537.) / 1000.
                ft       = self._car.get('track_f_mm',   1222.) / 1000.
                toe_off  = self._alignment.get('front_toe_deg', 0.)
                ack = np.full(n, float('nan'))
                for i in range(n):
                    fl_raw = res_fl['toe'][i] - toe_off
                    fr_raw = res_fr['toe'][i] - toe_off
                    ack[i] = _ackermann_from_pair(fl_raw, fr_raw, wb, ft)
                res_fl['ackermann'] = ack
                res_fr['ackermann'] = ack.copy()

                # ── Geometric turn radius: bicycle-model R from FL/FR toe ─────
                # Pure kinematics — the radius the car traces assuming zero
                # tire slip, computed from the two front-wheel steer angles.
                # Subtract static toe-offset so an aligned straight-ahead
                # (both wheels parallel to car) reads as infinite radius.
                from vahan.metrics_catalog import compute_turn_radius_post
                fl_raw_arr = res_fl['toe'] - toe_off
                fr_raw_arr = res_fr['toe'] - toe_off
                tr = compute_turn_radius_post(fl_raw_arr, fr_raw_arr,
                                              wheelbase_m=wb)
                res_fl['turn_radius'] = tr
                res_fr['turn_radius'] = tr.copy()

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

            # ── Roll axis inclination (vehicle-level) ────────────────────────
            # Roll axis goes from front RC to rear RC.  Inclination angle is
            # the tilt in the side view (YZ plane, X = 0):
            #     incl = atan2(RC_R - RC_F, wheelbase)        (rad → deg)
            # Positive incl = roll axis rises from front to rear, which is
            # the typical setup for high-rake / RR-bias cars (couples body
            # roll into a small pitch-up under cornering).  The metric is the
            # same for all four corners — it's a vehicle property, not a
            # corner property — so we copy the result into every corner's
            # array so the existing per-corner plot machinery just works.
            wb_m = self._car.get('wheelbase_mm', 1530.0) / 1000.0
            fl = self._sweep_results.get('FL', {})
            rl = self._sweep_results.get('RL', {})
            rc_f = fl.get('rc_height')   # mm, axle-level (post-processed above)
            rc_r = rl.get('rc_height')
            if (rc_f is not None and rc_r is not None
                    and len(rc_f) == len(rc_r) and wb_m > 1e-6):
                # Convert mm → m before atan2; result back to degrees.
                rise_m = (np.asarray(rc_r) - np.asarray(rc_f)) / 1000.0
                incl_deg = np.degrees(np.arctan2(rise_m, wb_m))
                for lbl in ('FL', 'FR', 'RL', 'RR'):
                    if lbl in self._sweep_results:
                        self._sweep_results[lbl]['roll_axis_incl'] = incl_deg.copy()

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
            state_prev = None           # previous SolvedState for IC finite difference

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
                    state_prev = st       # keep state_prev current for IC continuity
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
                                    state_prev=state_prev,
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
                state_prev = st   # for the IC finite difference next step

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
                # Two-point solve for MR + kinematic IC: solve at t - δ first.
                # state_prev is also used by _ic_y/_ic_z to compute the
                # rigid-body-finite-difference instant centre, which
                # avoids the asymptotic spikes the static-projection
                # method produces when the YZ-arm projections happen
                # to be parallel.
                side = 'left' if label in ('FL', 'RL') else 'right'
                _dt = 0.001  # 1mm perturbation
                t_prev = float(t) - _dt
                spring_prev = travel_prev = None
                st_prev = None
                try:
                    st_prev = solver.solve(t_prev)
                    spring_prev = float(np.sqrt(
                        (st_prev.rocker_spring_pt[0] - st_prev.spring_chassis_pt[0])**2 +
                        (st_prev.rocker_spring_pt[1] - st_prev.spring_chassis_pt[1])**2 +
                        (st_prev.rocker_spring_pt[2] - st_prev.spring_chassis_pt[2])**2))
                    travel_prev = t_prev
                except Exception:
                    st_prev = None
                corner_vals = _all_metrics(st, side,
                    spring_prev=spring_prev, travel_prev=travel_prev,
                    state_prev=st_prev,
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
            cg_y = self._car.get('cg_y_mm', 845.) / 1000.
            cg_z = self._car.get('cg_z_mm', 280.) / 1000.
            self.view3d.update_cg((cg_x, cg_y, cg_z))

            # ── Ackermann %: compute from current FL+FR steer angles ─────────
            # The per-step fn in metrics_catalog leaves this as NaN (it's
            # normally populated only by the post-processing sweep).  We have
            # both corners solved at the same rack position here, so we can
            # compute a live value directly — but the toe values in
            # corner_vals have static-toe alignment added, which corrupts the
            # Ackermann math (adds a constant offset that kills the
            # |inner|−|outer| relationship at small steer angles).  Subtract
            # the alignment offset so we feed the raw kinematic steer angles.
            # At/near zero steer the pair collapses (both wheels at 0°),
            # so we fall back to a probe at a reference steer angle —
            # Ackermann is a geometry property that should still show a value
            # in heave/roll/pitch modes.  Rear wheels are unsteered, so
            # Ackermann stays NaN for RL/RR (shown as "—" in the panel).
            fl_vals = all_corner_values.get('FL', {})
            fr_vals = all_corner_values.get('FR', {})
            if fl_vals and fr_vals:
                wb      = self._car.get('wheelbase_mm', 1537.) / 1000.
                ft      = self._car.get('track_f_mm',   1222.) / 1000.
                toe_off = self._alignment.get('front_toe_deg', 0.)
                fl_toe_raw = fl_vals.get('toe', float('nan')) - toe_off
                fr_toe_raw = fr_vals.get('toe', float('nan')) - toe_off
                ack = _ackermann_from_pair(fl_toe_raw, fr_toe_raw, wb, ft)
                if np.isnan(ack):
                    # Live state at/near zero steer → probe the geometry
                    ack = self._probe_static_ackermann()
                fl_vals['ackermann'] = ack
                fr_vals['ackermann'] = ack

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
            half_mm  = total_mm / 2.0
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
                    # Absolute max hand-wheel angle (one-way, deg) — used by
                    # the Steering Wheel Angle plot as the physical lock line.
                    self._dynamics_panel._cached_max_hw_deg = hw_deg
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
        """
        Compute static sag from stroke + preload + vehicle/spring/MR, then
        forward per-axle sag to the IK panel, values panel, and motion
        panel display.

        Sag is now an OUTPUT, not an input.  The dynamics panel supplies
        mass, weight distribution, spring rate, and MR; this handler
        combines them with the preload + stroke from MotionPanel to
        compute where the damper sits at rest.

        *** Changing damper params NEVER moves the geometry. ***  The
        hardpoints (and the 3D view) stay exactly where the user drew
        them.  If `fully_extended_mm` > 0, the handler also computes
        how much the CAD damper is already compressed and the shift
        needed to reach physics-consistent static sag — but these
        numbers are shown as a diagnostic only.  To actually commit
        that shift into the hardpoints the user must click the
        "Apply Sag to Hardpoints" button on the MotionPanel, which
        emits apply_sag_requested → _on_apply_sag.
        """
        stroke       = params.get('stroke_mm',        55.0)
        preload_f    = params.get('preload_front_mm',  0.0)
        preload_r    = params.get('preload_rear_mm',   0.0)
        L_full       = float(params.get('fully_extended_mm', 0.0))

        # Build a VehicleParams from the dynamics panel to get spring/MR/mass.
        sag_info = None
        try:
            dyn_params = self._dynamics_panel.get_params()
            # Add wheelbase + track + CG from the car dict so mass split is right.
            if hasattr(self, '_car') and isinstance(self._car, dict):
                dyn_params.setdefault('wheelbase_m',
                                      self._car.get('wheelbase_mm', 1530) / 1000.0)
                dyn_params.setdefault('front_track_m',
                                      self._car.get('track_f_mm', 1220) / 1000.0)
                dyn_params.setdefault('rear_track_m',
                                      self._car.get('track_r_mm', 1200) / 1000.0)
                dyn_params.setdefault('cg_height_m',
                                      self._car.get('cg_z_mm', 280) / 1000.0)
                dyn_params.setdefault('cg_to_front_axle_m',
                                      self._car.get('cg_y_mm', 765) / 1000.0)

            veh = VehicleParams(**dyn_params)

            # Prefer live kinematic MR at static (travel = 0) if the solvers
            # are loaded — gives the actual geometric MR, not a user number.
            mr_f = self._query_static_mr('front')
            mr_r = self._query_static_mr('rear')

            sag_info = veh.static_sag(
                preload_front_mm=preload_f,
                preload_rear_mm=preload_r,
                stroke_mm=stroke,
                mr_front=mr_f,
                mr_rear=mr_r,
            )
        except Exception:
            # Dynamics panel not ready yet — fall back to zero sag so
            # the UI still functions.
            import traceback; traceback.print_exc()
            sag_info = {
                'sag_shock_front_mm': 0.0, 'sag_shock_rear_mm': 0.0,
                'sag_front_pct': 0.0,      'sag_rear_pct': 0.0,
            }

        sag_f = sag_info['sag_shock_front_mm']
        sag_r = sag_info['sag_shock_rear_mm']

        # Per-axle forwarding.
        # IKPanel has a single axle selector — pick the matching sag/MR.
        try:
            ik_axle = 'front' if self._ik_panel._axle.currentIndex() == 0 else 'rear'
        except Exception:
            ik_axle = 'front'
        ik_sag = sag_f if ik_axle == 'front' else sag_r
        ik_mr  = sag_info.get('mr_front_used' if ik_axle == 'front'
                              else 'mr_rear_used', 1.0) or 1.0
        self._ik_panel.set_damper_limits(stroke, ik_sag, ik_mr)

        # ValuesPanel uses per-axle sag for per-corner bump/droop.
        self._values_panel.update_damper_params(stroke, sag_f, sag_r)

        # Motion panel read-only sag display happens below, after we enrich
        # sag_info with the CAD-compression / shift diagnostics.

        # ── compute shift DIAGNOSTIC only — DO NOT touch geometry ─────────
        # Per axle, we work out how far the kinematic "display travel = 0"
        # reference would need to shift to land on physics-consistent
        # static ride height.  These numbers are *only* shown in the
        # MotionPanel sag readout; they are cached in
        # `self._pending_sag_shift_m` so the "Apply Sag to Hardpoints"
        # button can commit them without having to recompute.
        #
        # The solver-level `sag_offset_m` is NEVER auto-written from here
        # — that path was the source of "the pushrod moves when I edit
        # damper stuff" behaviour and has been removed.  Two cases:
        #
        #   L_full == 0  (disabled, default)
        #     No diagnostic.  Shift is 0; nothing would happen on apply.
        #
        #   L_full >  0  (user entered damper fully-extended length)
        #     Measure L_cad = damper length at current hardpoints and
        #     work out how much of the stroke is already used in CAD:
        #       comp_cad  = L_full − L_cad             (mm, shock)
        #       comp_need = sag_shock (from physics)   (mm, shock)
        #       shift     = comp_need − comp_cad       (mm, shock)
        #     Convert to wheel travel via MR.  Positive shift = the car
        #     needs to sit lower at static than it does in CAD.
        sag_shock_f = float(sag_info.get('sag_shock_front_mm', 0.0))
        sag_shock_r = float(sag_info.get('sag_shock_rear_mm',  0.0))
        mr_f_used   = float(sag_info.get('mr_front_used', 1.0) or 1.0)
        mr_r_used   = float(sag_info.get('mr_rear_used',  1.0) or 1.0)

        pending_shift_m = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        cad_comp_f   = None
        cad_comp_r   = None
        shift_shock_f = 0.0
        shift_shock_r = 0.0
        over_ext_f = over_ext_r = over_comp_f = over_comp_r = False

        if L_full > 0.0 and hasattr(self, '_solvers') and self._solvers:
            def _cad_damper_len(label):
                """L_cad in mm: damper length at current hardpoints (travel=0)."""
                s = self._solvers.get(label)
                if s is None:
                    return None
                # Temporarily clear offset for a deterministic CAD measurement.
                saved = getattr(s, 'sag_offset_m', 0.0)
                s.sag_offset_m = 0.0
                try:
                    st = s.solve(0.0)
                    return float(np.linalg.norm(
                        st.rocker_spring_pt - st.spring_chassis_pt)) * 1000.0
                finally:
                    s.sag_offset_m = saved

            L_cad_f = _cad_damper_len('FL')
            L_cad_r = _cad_damper_len('RL')

            def _shift_for(L_cad, sag_shock, mr, stroke_mm):
                """Returns (wheel_offset_m, comp_cad_mm, shift_shock_mm,
                            over_extended, over_compressed)."""
                if L_cad is None:
                    return 0.0, None, 0.0, False, False
                comp_cad = L_full - L_cad                  # mm shock
                over_ext = comp_cad < -0.5                 # CAD longer than L_full
                over_comp = comp_cad > stroke_mm + 0.5     # CAD beyond full bump
                shift_shock = sag_shock - comp_cad         # mm shock
                mr_safe = mr if mr and mr > 0.05 else 1.0
                wheel_m = (shift_shock / mr_safe) / 1000.0
                return wheel_m, comp_cad, shift_shock, over_ext, over_comp

            off_f, cad_comp_f, shift_shock_f, over_ext_f, over_comp_f = \
                _shift_for(L_cad_f, sag_shock_f, mr_f_used, stroke)
            off_r, cad_comp_r, shift_shock_r, over_ext_r, over_comp_r = \
                _shift_for(L_cad_r, sag_shock_r, mr_r_used, stroke)
            pending_shift_m = {'FL': off_f, 'FR': off_f, 'RL': off_r, 'RR': off_r}

        # Cache the prospective shift for the "Apply to HPs" button.  We
        # deliberately do NOT push it onto any solver or run any sweep —
        # the 3D view stays put, and the app stays responsive.
        self._pending_sag_shift_m = pending_shift_m

        # Enrich the sag_info dict with CAD-damper diagnostics so the motion
        # panel can show how much of the stroke is already used at CAD and
        # how far the geometry would shift if the user hit "Apply".
        try:
            sag_info = dict(sag_info)
            sag_info['cad_compression_front_mm'] = cad_comp_f
            sag_info['cad_compression_rear_mm']  = cad_comp_r
            sag_info['shift_shock_front_mm']     = shift_shock_f
            sag_info['shift_shock_rear_mm']      = shift_shock_r
            sag_info['cad_over_extended_front']  = over_ext_f
            sag_info['cad_over_extended_rear']   = over_ext_r
            sag_info['cad_over_compressed_front'] = over_comp_f
            sag_info['cad_over_compressed_rear']  = over_comp_r
            self._motion_panel.update_sag_display(sag_info)
        except Exception:
            pass

    def _on_apply_sag(self):
        """
        Commit the currently-pending sag shift into the actual hardpoints.

        Background.  With `fully_extended_mm` set, `_on_damper_limits`
        computes per-axle shifts `_pending_sag_shift_m` that describe how
        far the wheel (and everything attached to it) would need to move
        in the chassis frame to put the car at its physics-consistent
        static ride height.  Those shifts are NOT auto-applied — the
        user's CAD hardpoints stay where they drew them until this
        method runs.

        Operation.  For each axle (front = FL, rear = RL master copy):
          1. Solve the corner at travel = shift_m (wheel-space metres).
             The solver preserves every link length (including the
             pushrod), so the resulting geometry is kinematically
             consistent with the CAD.
          2. Copy the solved moving points (uca_outer, lca_outer,
             tie_rod_outer, wheel_center, pushrod_outer, pushrod_inner,
             rocker_spring_pt) back into `_front_hp` / `_rear_hp`.
             Chassis-side points are untouched — they're rigidly
             attached to the frame.
          3. Rebuild the solvers from the new hardpoints.  Now travel=0
             IS the physics-consistent static position.
          4. Refresh the hardpoint panels, replot, redraw 3D, and
             re-run the sag diagnostic (which should now show ~0
             shift remaining).
        """
        if not hasattr(self, '_solvers') or not self._solvers:
            self.statusBar().showMessage('Apply sag: solvers not ready', 4000)
            return
        shifts = getattr(self, '_pending_sag_shift_m', None) or {}
        shift_f = float(shifts.get('FL', 0.0))
        shift_r = float(shifts.get('RL', 0.0))
        if abs(shift_f) < 1e-6 and abs(shift_r) < 1e-6:
            self.statusBar().showMessage(
                'Apply sag: no shift to apply — set "Fully ext." on the '
                'Motion panel to compute one', 5000)
            return

        # Keys that the solver actually updates (moving outboard points).
        MOVING_KEYS = ('uca_outer', 'lca_outer', 'tie_rod_outer',
                       'wheel_center', 'pushrod_outer', 'pushrod_inner',
                       'rocker_spring_pt')

        def _commit(label, axle_hp, shift_m):
            """Solve at `shift_m` and write the moving points back into axle_hp."""
            if abs(shift_m) < 1e-9:
                return 0
            solver = self._solvers.get(label)
            if solver is None:
                return 0
            saved_off = float(getattr(solver, 'sag_offset_m', 0.0))
            solver.sag_offset_m = 0.0
            try:
                state = solver.solve(shift_m)
            finally:
                solver.sag_offset_m = saved_off
            mp = state.all_moving_points()
            # SolvedState uses 'tr_outer'; the HP dict uses 'tie_rod_outer'.
            translation = {'tr_outer': 'tie_rod_outer'}
            touched = 0
            for k in ('uca_outer', 'lca_outer', 'tr_outer', 'wheel_center',
                      'pushrod_outer', 'pushrod_inner', 'rocker_spring_pt'):
                hp_key = translation.get(k, k)
                if hp_key in axle_hp and k in mp:
                    axle_hp[hp_key] = mp[k].copy()
                    touched += 1
            return touched

        try:
            nf = _commit('FL', self._front_hp, shift_f)
            nr = _commit('RL', self._rear_hp,  shift_r)
        except Exception as e:
            self.statusBar().showMessage(f'Apply sag failed: {e}', 6000)
            return

        # Rebuild solvers from the updated hardpoints — travel=0 is now
        # the physics-consistent static ride height, no hidden offsets.
        self._pending_sag_shift_m = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        try:
            self._rebuild_solvers()
            # Push updated hardpoint values into the hardpoint editor panels
            # so the user sees the new numbers.
            if hasattr(self, '_front_hp_panel'):
                self._front_hp_panel.refresh(self._front_hp, self._front_arb)
            if hasattr(self, '_rear_hp_panel'):
                self._rear_hp_panel.refresh(self._rear_hp, self._rear_arb)
            self._run_sweep()
            self._update_3d()
            # Re-run the sag diagnostic so the shift numbers refresh to ~0.
            self._refresh_sag()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Apply sag rebuild: {e}', 6000)
            return

        self.statusBar().showMessage(
            f'Sag applied — shift F {shift_f*1000:+.1f} mm / '
            f'R {shift_r*1000:+.1f} mm committed '
            f'(F: {nf} pts, R: {nr} pts)', 6000)

    def _refresh_sag(self):
        """
        Recompute static sag from the current MotionPanel + DynamicsPanel
        state and push the result to all consumers (motion display, IK,
        values panel).  Call this whenever spring rate, mass, or
        hardpoints (MR) change.
        """
        try:
            self._on_damper_limits({
                'stroke_mm':         self._motion_panel.stroke_mm,
                'preload_front_mm':  self._motion_panel.preload_front_mm,
                'preload_rear_mm':   self._motion_panel.preload_rear_mm,
                'fully_extended_mm': self._motion_panel.fully_extended_mm,
            })
        except Exception:
            pass

    def _query_static_mr(self, axle: str) -> float:
        """
        Return the geometric motion ratio (d_spring / d_wheel) at static
        for the given axle, using a finite difference on the corner solver.
        Falls back to None if solvers aren't available, so callers can use
        their dataclass default MR.

        axle : 'front' or 'rear'
        """
        try:
            label = 'FL' if axle == 'front' else 'RL'
            solver = self._solvers.get(label) if hasattr(self, '_solvers') else None
            if solver is None:
                return None
            # Always measure MR at the CAD reference (internal travel = 0)
            # to get a deterministic value independent of the current
            # sag_offset_m.  Otherwise the MR drifts with the offset and
            # the sag iteration wouldn't converge in one pass.
            saved_offset = float(getattr(solver, 'sag_offset_m', 0.0))
            solver.sag_offset_m = 0.0
            try:
                s0 = solver.solve(0.0)
                s1 = solver.solve(0.001)  # 1 mm bump at CAD ref
            finally:
                solver.sag_offset_m = saved_offset
            import numpy as np
            L0 = float(np.linalg.norm(s0.rocker_spring_pt - s0.spring_chassis_pt))
            L1 = float(np.linalg.norm(s1.rocker_spring_pt - s1.spring_chassis_pt))
            mr = (L0 - L1) / 0.001  # positive number (spring shortens under bump)
            if mr <= 0.05 or mr > 3.0:
                return None
            return mr
        except Exception:
            return None

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
        - ARB arm length, half-length and MR are pulled from the kinematic
          model (hardpoints + bell-crank solver) and pushed to the panel
          before reading parameters, so the panel only owns D / G / E.
        """
        # ── Push kinematically-derived ARB geometry into the panel ───────
        # Done BEFORE get_params() so the panel's wheel-rate calculation
        # uses fresh arm length / half-length / MR values straight from the
        # kinematic model, not stale spinbox numbers.
        arb_F = self._compute_arb_geometry_from_kinematics('F')
        arb_R = self._compute_arb_geometry_from_kinematics('R')
        if arb_F is not None and arb_R is not None:
            self._dynamics_panel.set_derived_arb_geometry(arb_F, arb_R)

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
        # Spring/MR/mass may have changed — refresh static sag readout.
        self._refresh_sag()
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
            aero_fz = self._get_active_aero_Fz(at_g=spec['lateral_g'])
            msg = 'Solving...'
            if aero_fz:
                msg += f' (aero: {sum(aero_fz.values()):.0f} N)'
            self._dynamics_panel.set_status(msg)

            worker = _DynamicsSolveWorker(
                ss, spec['lateral_g'], spec.get('longitudinal_g', 0.0),
                aero_Fz=aero_fz)
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
            aero_per_g = self._get_aero_Fz_per_g()
            msg = f'Sweeping ({mode})...'
            if aero_per_g:
                msg += ' + aero (V\u00b2)'
            self._dynamics_panel.set_status(msg)

            worker = _DynamicsSweepWorker(
                ss, spec['g_min'], spec['g_max'],
                spec.get('n_points', 41),
                longitudinal_g=spec.get('longitudinal_g', 0.0),
                mode=mode,
                lateral_g=spec.get('lateral_g', 0.0),
                aero_Fz_per_g=aero_per_g,
                start_speed_mph=spec.get('start_speed_mph', 0.0),
                end_speed_mph=spec.get('end_speed_mph', 200.0),
                sweep_axis=spec.get('sweep_axis', 'g'),
                v_min_mph=spec.get('v_min_mph', 0.0),
                v_max_mph=spec.get('v_max_mph', 60.0),
                turn_radius_m=spec.get('turn_radius_m', 10.0),
                traj_direction=spec.get('traj_direction', 'accel'))
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
        max_hw = getattr(self._dynamics_panel, '_cached_max_hw_deg', 0.0)
        hp_w = self._dynamics_panel._power_hp.value() * 745.7
        mass = self._dynamics_panel._total_mass.value()
        self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                 turn_radius_m=turn_r, wheelbase_m=wb,
                                 steer_ratio=sr, max_hw_deg=max_hw,
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
            sr = getattr(self._dynamics_panel, '_cached_steer_ratio', 0.0)
            max_hw = getattr(self._dynamics_panel, '_cached_max_hw_deg', 0.0)
            hp_w = self._dynamics_panel._power_hp.value() * 745.7
            mass = self._dynamics_panel._total_mass.value()
            self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                     turn_radius_m=turn_r, wheelbase_m=wb,
                                     steer_ratio=sr, max_hw_deg=max_hw,
                                     power_W=hp_w, mass_kg=mass)

    def _on_dyn_corners_sel(self, corners: list):
        """Re-plot dynamics with new corner selection."""
        sweep = getattr(self, '_dyn_sweep_data', None)
        if sweep is not None:
            graphs = self._dynamics_panel.get_selected_graphs()
            turn_r = self._dynamics_panel._turn_radius.value()
            wb = self._car.get('wheelbase_mm', 1530) / 1000
            sr = getattr(self._dynamics_panel, '_cached_steer_ratio', 0.0)
            max_hw = getattr(self._dynamics_panel, '_cached_max_hw_deg', 0.0)
            hp_w = self._dynamics_panel._power_hp.value() * 745.7
            mass = self._dynamics_panel._total_mass.value()
            self.curves.plot_dynamics(sweep, graphs=graphs, corners=corners,
                                     turn_radius_m=turn_r, wheelbase_m=wb,
                                     steer_ratio=sr, max_hw_deg=max_hw,
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
                sens, spec['lateral_g'], spec['longitudinal_g'],
                turn_radius_m=spec.get('turn_radius_m'))
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
    #  BRAKE CALCULATOR
    # ==========================================================================

    def _on_compute_brakes(self):
        """Compute brake pressures, lockup limits, and rotor temps."""
        try:
            from vahan.loads import compute_brake_system, compute_brake_thermal

            solver = self._build_dynamics_solver()
            veh = solver._veh

            # Read lat/lon g from the brake panel's own spinners
            lat_g = self._brake_calc_panel._lat_g.value()
            lon_g = self._brake_calc_panel._lon_g.value()

            # Solve steady-state to get Fz + camber distribution
            result = solver.solve(lat_g, lon_g)

            # Get brake params from loads panel (caliper geometry)
            bp_f = self._loads_panel.get_brake_params_front()
            bp_r = self._loads_panel.get_brake_params_rear()

            # Get system params — tire radius from VehicleParams
            system = self._brake_calc_panel.get_system_params(
                tire_radius_m=veh.tire_radius_m)

            # Tire model: TTC data if loaded, else LinearTireModel fallback
            tire = self._tire_model
            if tire is None:
                from vahan.tire_model import LinearTireModel
                tire = LinearTireModel()

            brakes = compute_brake_system(
                Fz=result.Fz,
                brake_params_f=bp_f,
                brake_params_r=bp_r,
                system=system,
                tire_model=tire,
                cambers=result.camber,
            )

            # Rotor thermal — single braking event
            th = self._brake_calc_panel.get_thermal_params()
            thermal = compute_brake_thermal(
                vehicle_mass_kg=veh.total_mass_kg,
                bias_pct_front=system.bias_pct_front,
                speed_start_mph=th['speed_start_mph'],
                speed_end_mph=th['speed_end_mph'],
                rotor_mass_f_kg=th['rotor_mass_f_kg'],
                rotor_mass_r_kg=th['rotor_mass_r_kg'],
                rotor_cp=th['rotor_cp'],
                ambient_C=th['ambient_C'],
            )

            self._brake_calc_panel.show_results(
                brakes, Fz=result.Fz, lat_g=lat_g, lon_g=lon_g,
                thermal=thermal)
            self.statusBar().showMessage(
                f'Brake calc done at {lat_g:.1f}g lat, {lon_g:.1f}g lon', 4000)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._brake_calc_panel._status.setText(f'Error: {e}')

    # ==========================================================================
    #  AERO DOWNFORCE
    # ==========================================================================

    _last_aero_result = None  # most recent AeroResult (stores deficit per corner + g_ref)
    _aero_active = False      # True when "Apply Aero" is toggled on

    def _get_aero_Fz_per_g(self) -> dict | None:
        """Per-corner aero Fz normalised to 1g (V²-scaled).

        At constant turn radius R,  V² = g · g_earth · R,  so downforce
        F = ½·ρ·V²·CL·A  scales linearly with g:
            Fz_corner(g) = Fz_per_g[corner] · g

        Two sources, controlled by the panel's "Aero source" combobox:

        ``solved`` — uses the deficit from the Aero Load Targets inverse
            solver (the original behaviour).  Best for "what aero do I
            *need* to hit a target utilization?"

        ``custom`` — uses user-supplied F_ref + V_ref + CoP from
            DynamicsPanel.get_custom_aero_params().  Best for validating
            CFD: "my CFD says I produce X N at Y km/h with CoP at Z%
            rear — what does that do to handling?"

        Returns dict with per-corner Fz at 1g, or None if aero is OFF
        or the active source has no usable data.
        """
        if not self._aero_active:
            return None

        source = 'solved'
        if hasattr(self._dynamics_panel, 'get_aero_source'):
            source = self._dynamics_panel.get_aero_source()

        if source == 'custom':
            return self._custom_aero_Fz_per_g()

        # Default: solved-deficit path (legacy behaviour).
        r = self._last_aero_result
        if r is None:
            return None
        g_ref = r.lateral_g
        if g_ref < 0.01:
            return None
        # Use axle-level need (symmetric: max of left/right per axle)
        fn = r.front_axle_need_N
        rn = r.rear_axle_need_N
        if fn + rn < 0.1:
            return None
        return {
            'FL': fn / g_ref, 'FR': fn / g_ref,
            'RL': rn / g_ref, 'RR': rn / g_ref,
        }

    def _custom_aero_Fz_per_g(self) -> dict | None:
        """Per-corner aero Fz at 1g from the DynamicsPanel's user-typed
        CFD numbers.  Returns None if the inputs are degenerate (zero
        downforce, zero ref-speed, etc.).

        Conversion chain:
            CL·A = 2 · F_ref / (ρ · V_ref²)
            F(V) = ½ · ρ · V² · CL·A
            V² at 1g, radius R = 1 · g_earth · R
        Then split front/rear by CoP%, halved per side.

        We pull the turn radius R from the Dynamics panel (same field
        the velocity-axis sweep plot uses), so a user who explores at
        different cornering radii naturally sees the aero load scale.
        """
        cfg = self._dynamics_panel.get_custom_aero_params()
        F_ref     = float(cfg['F_ref_N'])
        V_ref_kph = float(cfg['V_ref_kph'])
        rear_pct  = float(cfg['cop_rear_pct'])
        rho       = float(cfg['air_density'])

        if F_ref <= 0.0 or V_ref_kph <= 0.0 or rho <= 0.0:
            return None

        R = float(self._dynamics_panel._turn_radius.value())
        if R <= 0.0:
            return None

        V_ref_ms = V_ref_kph / 3.6
        # CL·A = 2·F_ref / (ρ·V_ref²)
        CLA = 2.0 * F_ref / (rho * V_ref_ms * V_ref_ms)
        # At g = 1: V² = g_earth · R  →  F = 0.5 · ρ · V² · CL·A
        g_earth = 9.80665
        F_at_1g = 0.5 * rho * (g_earth * R) * CLA
        # Equivalent closed form (drops ρ): F_at_1g = F_ref · g_earth·R / V_ref²
        # — kept the longer form above for clarity.

        rear_frac = max(0.0, min(1.0, rear_pct / 100.0))
        front_frac = 1.0 - rear_frac
        # Symmetric L/R split per axle
        F_front_per = F_at_1g * front_frac / 2.0
        F_rear_per  = F_at_1g * rear_frac  / 2.0
        return {
            'FL': F_front_per, 'FR': F_front_per,
            'RL': F_rear_per,  'RR': F_rear_per,
        }

    def _get_active_aero_Fz(self, at_g: float = None) -> dict | None:
        """Return per-corner aero Fz at a specific g (V^2-scaled), or None."""
        per_g = self._get_aero_Fz_per_g()
        if per_g is None:
            return None
        if at_g is None:
            at_g = self._dynamics_panel._lat_g.value()
        g = abs(at_g)
        return {k: v * g for k, v in per_g.items()}

    def _on_apply_aero_toggle(self, checked: bool):
        """Re-fired whenever the Apply Aero toggle flips OR the source
        combobox changes (DynamicsPanel re-emits to force a refresh).
        Reads the active source and updates the in-panel readout to
        match.  When custom mode is selected with no usable inputs, the
        label still shows OFF so the user notices."""
        self._aero_active = checked
        if not checked:
            self._dynamics_panel.update_aero_label(0)
            return

        per_g = self._get_aero_Fz_per_g()
        if per_g is None:
            self._dynamics_panel.update_aero_label(0)
            return
        # Sum the per-corner Fz at 1g and report it as the "applied"
        # load — same convention as the previous solved-only readout.
        total_at_1g = sum(per_g.values())
        self._dynamics_panel.update_aero_label(total_at_1g)

    # ── Skidpad / Transient dynamics ─────────────────────────────────────

    def _on_skidpad_simulate(self, params: dict):
        """Build a TransientSolver from current GUI state and run a sim."""
        import traceback
        try:
            self._skidpad_panel.set_solving(True)
            self._skidpad_panel.set_status('Simulating...')

            # ── Build VehicleParams the same way the dynamics panel does ─
            dyn_params = self._dynamics_panel.get_params()
            car = self._car
            dyn_params['front_track_m'] = car['track_f_mm'] / 1000
            dyn_params['rear_track_m'] = car['track_r_mm'] / 1000
            dyn_params['wheelbase_m'] = car['wheelbase_mm'] / 1000
            dyn_params['cg_height_m'] = car['cg_z_mm'] / 1000
            dyn_params['cg_to_front_axle_m'] = car.get('cg_y_mm', 1100) / 1000
            if 'front_brake_bias_pct' in car:
                dyn_params['front_brake_bias'] = car['front_brake_bias_pct'] / 100
            # Steering-rack geometry — so VehicleParams can supply it to
            # the steering-geometry probe below AND so saving/loading the
            # project reflects the current rack.
            steer_cfg = self._steer or {}
            for k in ('rack_travel_per_rev_mm', 'total_rack_travel_mm'):
                if k in steer_cfg:
                    dyn_params[k] = steer_cfg[k]
            # Motion ratios from kinematics at design position
            dt = 0.001
            for label, key in [('FL', 'motion_ratio_front'), ('RL', 'motion_ratio_rear')]:
                solver = self._solvers.get(label)
                if solver:
                    try:
                        s_plus = solver.solve(+dt)
                        s_minus = solver.solve(-dt)
                        mr = abs(s_plus.spring_length - s_minus.spring_length) / (2 * dt)
                        if 0.1 < mr < 3.0:
                            dyn_params[key] = mr
                    except Exception:
                        pass
            for label in ('FL', 'RL'):
                solver = self._solvers.get(label)
                if solver:
                    try:
                        state = solver.solve(0.0)
                        dyn_params['unsprung_cg_height_m'] = float(state.wheel_center[2])
                        break
                    except Exception:
                        pass
            veh = VehicleParams(**dyn_params)

            # Tire model fallback to linear if none loaded
            tire = self._tire_model
            if tire is None:
                from vahan.tire_model import LinearTireModel
                tire = LinearTireModel()

            # ── Auto-compute inertias + Ackermann from car geometry ───────
            # Yaw inertia Izz: the "bicycle limit" is m·a·b (all mass at
            # the axles, zero gyradius beyond the wheelbase).  The factor
            # lives on VehicleParams so the user can override it — don't
            # bury it here.
            m_total = veh.total_mass_kg
            a = veh.cg_to_front_axle_m
            b = veh.cg_to_rear_axle_m
            auto_Izz = veh.yaw_inertia_factor * m_total * a * b
            # Roll inertia Ixx about the roll axis: Ixx = m_s · k_roll².
            # Gyradius fraction is a VehicleParams field so it's adjustable.
            track_avg = 0.5 * (veh.front_track_m + veh.rear_track_m)
            k_roll = veh.roll_gyradius_track_frac * track_avg
            auto_Ixx = veh.sprung_mass_kg * (k_roll ** 2)
            # Ackermann %: probe the current front-suspension geometry.
            # Falls back to 0 % (parallel steer) only if the probe fails.
            auto_ack = self._probe_static_ackermann()
            if np.isnan(auto_ack):
                auto_ack = 0.0

            # ── Steering geometry from rack + kinematics ────────────────
            # Builds a rack ↔ road-wheel mapping by actually probing the
            # front suspension at a range of rack positions.  This is the
            # piece that makes ``rack_travel_per_rev_mm`` visible in the
            # simulation output — change the rack ratio and the
            # steering-wheel angle the driver needs changes with it.
            steering_geom = self._build_steering_geometry(veh)

            # ── Roll damping derived from real damper specs ────────────
            # The panel exposes four bump/rebound damper coefficients
            # (N·s/m at the shock).  Convert to chassis-roll damping:
            #   c_phi_axle = (c_bump + c_rebound) · MR² · t² / 4
            #   c_phi      = c_phi_F + c_phi_R
            # Derivation:
            #   In pure body roll at φ̇, the outer wheel moves up at
            #   v_w = (t/2)·φ̇ (BUMP) while the inner wheel moves down at
            #   the same speed (REBOUND).  Each damper applies a force
            #   F_w = c_d·MR²·v_w at the wheel (MR is the code's
            #   shock/wheel ratio: F_w = F_d·MR with F_d = c_d·v_d and
            #   v_d = v_w·MR).  Summing the wheel forces × half-track
            #   over both axle wheels gives the formula above.
            c_F = (float(params['damper_F_bump_Nspm'])
                   + float(params['damper_F_rebound_Nspm']))
            c_R = (float(params['damper_R_bump_Nspm'])
                   + float(params['damper_R_rebound_Nspm']))
            mr_f, mr_r = veh.motion_ratio_front, veh.motion_ratio_rear
            t_f,  t_r  = veh.front_track_m,     veh.rear_track_m
            c_phi = (c_F * mr_f * mr_f * t_f * t_f / 4.0
                     + c_R * mr_r * mr_r * t_r * t_r / 4.0)

            tparams = TransientParams(
                sprung_roll_inertia=auto_Ixx,
                yaw_inertia=auto_Izz,
                roll_damping_Nms_rad=c_phi,
                ackermann_pct=auto_ack,
                steering_tau_s=params['steer_tau_s'],
            )

            solver = TransientSolver(
                veh, tire,
                corner_solvers=self._solvers,
                params=tparams,
                steering_geometry=steering_geom,
                shock_stroke_mm=getattr(self._motion_panel, 'stroke_mm', 50.0),
            )

            # ── Build steering profile ─────────────────────────────────────
            test = params['test_type']
            direction = params['direction']
            sign = +1.0 if direction.startswith('l') else -1.0
            auto_peak_steer_deg = float('nan')
            auto_sim_duration_s = float('nan')
            auto_derived_speed_ms = float('nan')

            # ── Resolve solve-mode (target speed vs target lateral g) ──
            # On a fixed-radius path, v and a_y are linked by v² = a_y·g·R.
            # If the user picked "Target lateral g", derive v from their
            # requested lat-g and the relevant radius; then overwrite
            # params['target_speed_ms'] so every downstream consumer
            # (path follower, TransientInputs, logs) uses the same value.
            # Only skidpad tests admit this mode — open tests (step/ramp/
            # sine) have no fixed radius, so the panel forces them back
            # to target_speed mode in _on_test_changed().
            solve_mode = params.get('solve_mode', 'target_speed')
            if solve_mode == 'target_lat_g' and test in ('skidpad', 'skidpad_full'):
                a_y_g = max(float(params.get('target_lat_g', 0.0)), 1e-6)
                R_eff = (9.125 if test == 'skidpad_full'
                         else float(params.get('skidpad_radius_m', 9.125)))
                v_derived = float(np.sqrt(a_y_g * 9.80665 * R_eff))
                params = {**params, 'target_speed_ms': v_derived}
                auto_derived_speed_ms = v_derived

            if test == 'skidpad':
                steering = SteeringProfile.skidpad(
                    radius_m=params['skidpad_radius_m'],
                    wheelbase_m=veh.wheelbase_m,
                    t_entry=0.5,
                    ramp_duration=params['ramp_duration_s'],
                    direction=direction,
                )
            elif test == 'skidpad_full':
                # Full FSAE figure-8: entry → 2 laps on first circle →
                # crossover → 2 laps on opposite circle → exit.  Radius is
                # FIXED at 9.125 m (FSAE regulation — path centreline
                # between inner cone ring 15.25 m dia and outer 21.25 m
                # dia).  Peak steer and sim duration are DERIVED; the
                # user only controls speed.
                #
                # Uses CLOSED-LOOP path following (pure pursuit) instead of
                # a pre-defined steering profile.  An open-loop sign flip
                # at the crossover cannot close the figure-8 — the car
                # physically cannot reverse its yaw rate instantly, so the
                # second circle's centre ends up offset several metres
                # forward from the first.  The path follower tracks the
                # ideal figure-8 polyline, closing that gap and putting
                # both circle centres on a line perpendicular to entry.
                R_skidpad = 9.125
                first_dir = 'right' if direction.startswith('r') else 'left'
                path_follower = SkidpadPathFollower(
                    radius_m=R_skidpad,
                    wheelbase_m=veh.wheelbase_m,
                    speed_ms=params['target_speed_ms'],
                    first_direction=first_dir,
                    n_laps_per_side=2,
                    t_entry_s=1.0,
                    exit_straight_m=8.0,
                    # Physical steering saturation from the rack-derived
                    # geometry — no more hardcoded 30°.
                    max_steer_rad=(steering_geom.max_road_wheel_rad
                                   if steering_geom is not None else None),
                )
                steering = None  # closed-loop; no open-loop profile
                # Override sim duration (user can't see the field in this
                # mode — it's hidden by _on_test_changed).
                params = {**params,
                          'sim_duration_s': float(path_follower.total_time_s + 1.0)}
                auto_peak_steer_deg = np.degrees(veh.wheelbase_m / R_skidpad)
                auto_sim_duration_s = params['sim_duration_s']
                # Stash for later overlay on the path plot.
                self._skidpad_follower = path_follower
            elif test == 'step':
                steering = SteeringProfile.step(
                    t_step=0.5,
                    steer_rad=sign * np.radians(abs(params['peak_steer_deg'])),
                )
            elif test == 'ramp':
                steering = SteeringProfile.ramp(
                    t_start=0.5,
                    t_end=0.5 + params['ramp_duration_s'],
                    steer_rad=sign * np.radians(abs(params['peak_steer_deg'])),
                )
            elif test == 'sine':
                # In sine mode the panel puts frequency into ramp_duration
                steering = SteeringProfile.sine(
                    amplitude_rad=sign * np.radians(abs(params['peak_steer_deg'])),
                    frequency_hz=params['ramp_duration_s'],
                    t_start=0.5,
                )
            else:
                steering = SteeringProfile.constant(0.0)

            if test == 'skidpad_full':
                inputs = TransientInputs(
                    v_x_target_ms=params['target_speed_ms'],
                    steering_controller=path_follower,
                    duration_s=params['sim_duration_s'],
                    dt_s=params['dt_s'],
                )
            else:
                inputs = TransientInputs(
                    v_x_target_ms=params['target_speed_ms'],
                    steering=steering,
                    duration_s=params['sim_duration_s'],
                    dt_s=params['dt_s'],
                )
                # Clear any leftover path follower from a previous run.
                self._skidpad_follower = None

            # Echo the auto-computed values back to the panel so the user
            # sees what the solver is using (Izz, Ixx, Ackermann %, and
            # the derived sim duration / peak steer for skidpad_full).
            # If lat-g mode derived the speed, pass that through so the
            # user can reconcile it with the Target speed field.
            self._skidpad_panel.set_auto_info(
                yaw_Izz=auto_Izz,
                sprung_Ixx=auto_Ixx,
                ackermann_pct=auto_ack,
                sim_duration_s=auto_sim_duration_s,
                peak_steer_deg=auto_peak_steer_deg,
                derived_speed_ms=auto_derived_speed_ms,
                derived_roll_damping=c_phi,
            )

            # ── Run in worker thread ───────────────────────────────────────
            worker = _TransientSimWorker(solver, inputs)
            worker.finished.connect(self._on_skidpad_done)
            worker.failed.connect(self._on_skidpad_failed)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            self._skidpad_worker = worker
            worker.start()

        except Exception as e:
            traceback.print_exc()
            self._skidpad_panel.set_solving(False)
            self._skidpad_panel.set_status(f'Error: {e}')

    def _on_skidpad_done(self, result: TransientResult):
        self._skidpad_panel.set_solving(False)
        self._skidpad_panel.set_status(
            f'Done. {len(result.t)} steps, '
            f'{result.t[-1]:.2f} s sim time.')
        self._skidpad_panel.show_result(result)
        self._last_transient_result = result
        # Plot using currently selected signals
        signals = self._skidpad_panel.get_selected_signals()
        if signals:
            self._plot_transient(result, signals)

    def _on_skidpad_failed(self, msg: str):
        self._skidpad_panel.set_solving(False)
        self._skidpad_panel.set_status(f'Error: {msg}')
        self.statusBar().showMessage(f'Transient sim error: {msg}', 6000)

    def _on_skidpad_signals(self, signals: list):
        """Re-plot existing transient result with new signal selection."""
        r = getattr(self, '_last_transient_result', None)
        if r is not None and signals:
            self._plot_transient(r, signals)

    def _plot_transient(self, result: TransientResult, signals: list):
        """
        Show (or refresh) the persistent transient-results dialog.

        The dialog stays open and has its own checkable signal list so the
        user can toggle which plots appear without closing the popup.  On
        subsequent calls (new simulation done, or panel list changed) it
        just updates the stored result / selection and redraws.
        """
        if not signals:
            return
        self._transient_result = result

        # First call — build dialog.  Subsequent calls reuse it.
        if getattr(self, '_transient_dialog', None) is None:
            self._build_transient_dialog(signals)
        else:
            # Sync the in-dialog toggle list with the requested selection so
            # the user sees the signals they just chose from the panel, then
            # re-render with the current data.
            self._sync_transient_sig_list(signals)
            self._render_transient_canvas()

        # Non-modal: user can still interact with the main window and the
        # side-panel signal list while the popup is open.
        self._transient_dialog.show()
        self._transient_dialog.raise_()
        self._transient_dialog.activateWindow()

    def _build_transient_dialog(self, initial_signals: list):
        """Construct the persistent transient-results dialog once."""
        # Pull the signal menu from the panel so labels stay in sync.
        try:
            from gui.panels import _TRANSIENT_SIGNALS as SIGS
        except Exception:
            SIGS = [
                ('yaw_rate',   'Yaw rate (deg/s)'),
                ('ay',         'Lateral g'),
                ('roll',       'Roll angle (deg)'),
                ('beta',       'Body slip (deg)'),
                ('velocity',   'Velocity (m/s)'),
                ('slip_angle', 'Tire slip angles'),
                ('Fz',         'Per-corner Fz'),
                ('path',       'Trajectory (X-Y)'),
                ('steer',      'Steering input'),
            ]

        dlg = QDialog(self)
        dlg.setWindowTitle('Skidpad / Transient Results')
        dlg.resize(1200, 720)
        dlg.setStyleSheet('QDialog { background: #000; color: #e0e0e0; }')
        # Non-modal — QDialog defaults to modal via exec(), we use show().
        dlg.setModal(False)

        root = QHBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left: signal toggle list ────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)
        hdr = QLabel('Signals')
        hdr.setStyleSheet('color: #FFB74D; font-weight: bold; font-size: 13px;')
        left.addWidget(hdr)

        sig_list = QListWidget()
        sig_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        sig_list.setFixedWidth(220)
        sig_list.setStyleSheet(
            'QListWidget { background: #0a0a0a; color: #e0e0e0; '
            'border: 1px solid #222; font-size: 13px; }'
            'QListWidget::item { padding: 4px; }'
            'QListWidget::item:selected { background: #333; color: white; }'
        )
        for key, label in SIGS:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            sig_list.addItem(item)
        left.addWidget(sig_list, stretch=1)

        hint = QLabel('Ctrl/Shift-click to multi-select')
        hint.setStyleSheet('color: #666; font-size: 10px;')
        left.addWidget(hint)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dlg.hide)
        close_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 20px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #2474a6; }'
        )
        left.addWidget(close_btn)

        root.addLayout(left)

        # ── Right: matplotlib canvas ────────────────────────────────────
        fig = Figure(facecolor='#000000')
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(canvas, stretch=1)

        # Stash widgets on self so _render_transient_canvas / sync can reach them.
        self._transient_dialog = dlg
        self._transient_sig_list = sig_list
        self._transient_fig = fig
        self._transient_canvas = canvas

        # Hover value readouts — works for every signal including path (X-Y).
        self._transient_hover = HoverAnnotator(canvas)

        # Selecting signals inside the dialog re-renders.
        sig_list.itemSelectionChanged.connect(self._render_transient_canvas)

        # Closing the dialog (X button) just hides it; next _plot_transient
        # call will show it again.  Clearing the reference would force a
        # rebuild, losing the selection — not what we want.

        # Initial selection.
        self._sync_transient_sig_list(initial_signals)
        self._render_transient_canvas()

    def _sync_transient_sig_list(self, signals: list):
        """Check the list items whose keys are in `signals`, uncheck others."""
        lst = getattr(self, '_transient_sig_list', None)
        if lst is None:
            return
        want = set(signals or [])
        blocked = lst.blockSignals(True)
        for i in range(lst.count()):
            it = lst.item(i)
            key = it.data(Qt.ItemDataRole.UserRole)
            it.setSelected(key in want)
        lst.blockSignals(blocked)

    def _current_transient_signals(self) -> list:
        lst = getattr(self, '_transient_sig_list', None)
        if lst is None:
            return []
        out = []
        for i in range(lst.count()):
            it = lst.item(i)
            if it.isSelected():
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def _render_transient_canvas(self):
        """Redraw the figure using the current result + current signal selection."""
        result = getattr(self, '_transient_result', None)
        fig    = getattr(self, '_transient_fig', None)
        canvas = getattr(self, '_transient_canvas', None)
        if result is None or fig is None or canvas is None:
            return

        signals = self._current_transient_signals()
        fig.clear()

        if not signals:
            # Placeholder when nothing selected.
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor('#000000')
            for s in ('bottom', 'top', 'left', 'right'):
                ax.spines[s].set_color('#333')
            ax.tick_params(colors='#666', labelsize=8)
            ax.text(0.5, 0.5, 'Select one or more signals on the left',
                    ha='center', va='center', color='#888', fontsize=12,
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            canvas.draw()
            return

        n = len(signals)
        cols = 2 if n >= 2 else 1
        rows = (n + cols - 1) // cols
        for i, sig in enumerate(signals):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor('#000000')
            for s in ('bottom', 'top', 'left', 'right'):
                ax.spines[s].set_color('#333')
            ax.tick_params(colors='#e0e0e0', labelsize=9)
            ax.xaxis.label.set_color('#e0e0e0')
            ax.yaxis.label.set_color('#e0e0e0')
            ax.title.set_color('#FFB74D')
            ax.grid(True, color='#222', linestyle='-', linewidth=0.5)
            self._plot_signal(ax, result, sig)

        fig.tight_layout()
        canvas.draw()

    def _plot_signal(self, ax, result: TransientResult, sig: str):
        """Draw a single signal onto ax."""
        t = result.t
        corner_colors = {'FL': '#ffd600', 'FR': '#ef5350',
                         'RL': '#4fc3f7', 'RR': '#ffffff'}
        if sig == 'yaw_rate':
            ax.plot(t, np.degrees(result.yaw_rate), color='#ffd600')
            ax.set_title('Yaw rate')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('deg/s')
        elif sig == 'ay':
            ax.plot(t, result.ay / 9.81, color='#ef5350')
            ax.set_title('Lateral g')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('g')
        elif sig == 'roll':
            ax.plot(t, np.degrees(result.roll), color='#4fc3f7')
            ax.set_title('Roll angle')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('deg')
        elif sig == 'beta':
            ax.plot(t, np.degrees(result.beta), color='#ffffff')
            ax.set_title('Body slip (beta)')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('deg')
        elif sig == 'velocity':
            # Forward speed v_x; show total speed |v| as a dashed overlay so
            # you can see when body slip starts contributing (|v| > v_x).
            MPH_PER_MS = 2.23693629   # exact: 1 m/s = 2.2369... mph
            v_total = np.sqrt(result.v_x**2 + result.v_y**2)
            ax.plot(t, result.v_x * MPH_PER_MS, color='#FFB74D', linewidth=1.4,
                    label='v_x (forward)')
            if np.any(np.abs(result.v_y) > 0.01):
                ax.plot(t, v_total * MPH_PER_MS, color='#4fc3f7', linewidth=1.0,
                        linestyle='--', label='|v| (total)')
                ax.legend(fontsize=8, facecolor='#000',
                          edgecolor='#333', labelcolor='#e0e0e0')
            ax.set_title('Velocity')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('mph')
        elif sig == 'slip_angle':
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                ax.plot(t, np.degrees(result.slip_angle[lbl]),
                        color=corner_colors[lbl], label=lbl, linewidth=1)
            ax.legend(fontsize=8, facecolor='#000', edgecolor='#333', labelcolor='#e0e0e0')
            ax.set_title('Tire slip angles')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('deg')
        elif sig == 'Fz':
            for lbl in ('FL', 'FR', 'RL', 'RR'):
                ax.plot(t, result.Fz[lbl], color=corner_colors[lbl],
                        label=lbl, linewidth=1)
            ax.legend(fontsize=8, facecolor='#000', edgecolor='#333', labelcolor='#e0e0e0')
            ax.set_title('Per-corner Fz')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('N')
        elif sig == 'path':
            # Colour the trajectory by forward speed using a LineCollection
            # so the driver can see where the car sheds / gains velocity.
            from matplotlib.collections import LineCollection
            X, Y = result.X, result.Y
            v = result.v_x

            # ── Ideal FSAE skidpad overlay (if this was a skidpad_full run) ──
            follower = getattr(self, '_skidpad_follower', None)
            if follower is not None:
                ix, iy = follower.ideal_path()
                ax.plot(ix, iy, color='#555555', linewidth=1.0,
                        linestyle='--', alpha=0.7,
                        label='Ideal path', zorder=1)
                # Circle centres (the two imaginary points around which
                # the car laps — these should sit on a perpendicular to
                # the entry line in a proper FSAE figure-8).
                (cx1, cy1), (cx2, cy2) = follower.circle_centres()
                ax.plot([cx1, cx2], [cy1, cy2], color='#777777',
                        linewidth=0.8, linestyle=':',
                        marker='+', markersize=12, zorder=1,
                        label='Centre line')
                # Inner cone ring (the drivable boundary): inner_R =
                # R - 1.5 m (track half-width), outer_R = R + 1.5 m.
                inner_R = follower.R - 1.5
                outer_R = follower.R + 1.5
                th = np.linspace(0, 2*np.pi, 100)
                for (cx, cy) in [(cx1, cy1), (cx2, cy2)]:
                    ax.plot(cx + inner_R*np.cos(th), cy + inner_R*np.sin(th),
                            color='#B71C1C', linewidth=0.6,
                            alpha=0.5, zorder=1)
                    ax.plot(cx + outer_R*np.cos(th), cy + outer_R*np.sin(th),
                            color='#B71C1C', linewidth=0.6,
                            alpha=0.5, zorder=1)

            pts = np.column_stack([X, Y]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            MPH_PER_MS = 2.23693629
            v_mid_mph = 0.5 * (v[:-1] + v[1:]) * MPH_PER_MS
            if np.ptp(v_mid_mph) > 1e-3:
                lc = LineCollection(
                    segs, cmap='plasma', linewidth=2.0,
                    array=v_mid_mph,
                    norm=plt_Normalize(v_mid_mph.min(), v_mid_mph.max()),
                    zorder=3,
                )
                ax.add_collection(lc)
                cb = ax.figure.colorbar(lc, ax=ax, pad=0.02, shrink=0.85)
                cb.set_label('v_x (mph)', color='#e0e0e0', fontsize=8)
                cb.ax.yaxis.set_tick_params(color='#888', labelsize=7)
                for l in cb.ax.get_yticklabels():
                    l.set_color('#e0e0e0')
            else:
                # Constant speed — plain line
                ax.plot(X, Y, color='#FFB74D', linewidth=1.8, zorder=3,
                        label=f'v={v.mean() * MPH_PER_MS:.1f} mph')
            # Invisible plot for auto-scaling (LineCollection doesn't
            # trigger axis scaling on its own).
            ax.plot(X, Y, color='none')
            ax.plot(X[0], Y[0], 'o', color='#4fc3f7',
                    label='Start', markersize=6, zorder=4)
            ax.plot(X[-1], Y[-1], 's', color='#ef5350',
                    label='End', markersize=6, zorder=4)
            ax.legend(fontsize=8, facecolor='#000',
                      edgecolor='#333', labelcolor='#e0e0e0')
            ax.set_title('Trajectory (coloured by v_x)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_aspect('equal', adjustable='datalim')
        elif sig == 'steer':
            # Primary trace is the driver's steering-wheel angle (the
            # thing that physically changes when rack mm/rev changes).
            # Road-wheel angle is shown as a secondary dashed trace so
            # you can still see the tire side of the linkage.
            has_wheel = (result.steer_wheel_deg.size
                         and np.any(np.abs(result.steer_wheel_deg) > 1e-6))
            if has_wheel:
                ax.plot(t, result.steer_wheel_deg, color='#FFB74D', linewidth=1.4,
                        label='Steering wheel (driver)')
                ax.plot(t, np.degrees(result.steer_actual),
                        color='#4fc3f7', linestyle='--', linewidth=1.1,
                        label='Road wheel (actual)')
                ax.plot(t, np.degrees(result.steer),
                        color='#666666', linestyle=':', linewidth=1.0,
                        label='Road wheel (commanded)')
                ax.set_title('Steering input')
                ax.set_xlabel('Time (s)'); ax.set_ylabel('deg')
            else:
                # Legacy fallback when no steering geometry is available —
                # road-wheel angle only.
                ax.plot(t, np.degrees(result.steer), color='#888',
                        label='Command', linewidth=1)
                ax.plot(t, np.degrees(result.steer_actual), color='#FFB74D',
                        label='Actual', linewidth=1.2)
                ax.set_title('Road-wheel steer')
                ax.set_xlabel('Time (s)'); ax.set_ylabel('deg')
            ax.legend(fontsize=8, facecolor='#000', edgecolor='#333',
                      labelcolor='#e0e0e0')

    def _on_aero_solve(self, params: dict):
        try:
            self._aero_panel._status.setText('Solving...')
            ss = self._build_dynamics_solver()
            aero = AeroDownforceSolver(ss)
            result = aero.solve(
                params['lateral_g'], params['longitudinal_g'],
                params['target_util'],
            )
            self._last_aero_result = result
            self._aero_panel.show_result(result)
            if self._aero_active:
                total = result.front_axle_need_N + result.rear_axle_need_N
                self._dynamics_panel.update_aero_label(total)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._aero_panel._status.setText(f'Error: {e}')

    def _on_aero_sweep(self, params: dict):
        try:
            self._aero_panel._status.setText('Sweeping...')
            ss = self._build_dynamics_solver()
            aero = AeroDownforceSolver(ss)
            turn_r = self._dynamics_panel._turn_radius.value()

            g_range = np.linspace(0.1, params['lateral_g'], 21)
            sweep = aero.sweep(
                g_range, params['longitudinal_g'], params['target_util'])

            # ── Plot in dynamics figure ──
            _styles = {
                'FL': (CORNER_PLOT_COLORS['FL'], '-'),
                'FR': (CORNER_PLOT_COLORS['FR'], '--'),
                'RL': (CORNER_PLOT_COLORS['RL'], '-.'),
                'RR': (CORNER_PLOT_COLORS['RR'], ':'),
            }
            _leg_kw = dict(fontsize=7, facecolor='#06060e',
                           labelcolor='white', framealpha=0.7,
                           loc='best', handlelength=1.5, ncol=2)

            fig = self.curves.fig
            fig.clear()
            gs = sweep['lateral_g']

            show_speed = turn_r > 0
            top_margin = 0.86 if show_speed else 0.92
            fig.subplots_adjust(
                hspace=0.55, wspace=0.40,
                left=0.09, right=0.97, top=top_margin, bottom=0.12)

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

            # ── Subplot 2: axle-level needs + total ──
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(gs, sweep['front_need'], color='#FFD600', linewidth=2.2,
                     linestyle='-', marker='v', markersize=4,
                     markevery=3, label='Front axle')
            ax2.plot(gs, sweep['rear_need'], color='#E53935', linewidth=2.2,
                     linestyle='-', marker='^', markersize=4,
                     markevery=3, label='Rear axle')
            ax2.plot(gs, sweep['total'], color='#FFFFFF', linewidth=2.0,
                     linestyle=':', marker='o', markersize=2,
                     markevery=3, label='Total', alpha=0.7)
            ax2.set_xlabel('Lateral g')
            ax2.set_ylabel('Downforce needed (N)')
            ax2.set_title(f'Aero targets (util\u2264{params["target_util"]:.0%})',
                          color='white', fontsize=10)
            ax2.legend(**{**_leg_kw, 'ncol': 1, 'loc': 'upper left'})
            ax2.grid(True, alpha=0.2)

            # Annotate rear bias at final g
            bias_final = sweep['rear_bias_pct'][-1]
            total_final = sweep['total'][-1]
            if total_final > 0:
                ax2.annotate(
                    f'Rear bias: {bias_final:.0f}%',
                    xy=(gs[-1], total_final),
                    xytext=(-60, 12), textcoords='offset points',
                    color='#aaa', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#666'))

            # Style + velocity secondary x-axis (same as dynamics plots)
            for ax in [ax1, ax2]:
                ax.set_facecolor('#000000')
                ax.tick_params(colors='#888')
                ax.xaxis.label.set_color('#aaa')
                ax.yaxis.label.set_color('#aaa')

                if show_speed:
                    try:
                        R = turn_r
                        def _g_to_mph(g, R=R):
                            return np.sqrt(np.maximum(g, 0) * 9.81 * R) * 2.23694
                        def _mph_to_g(mph, R=R):
                            v = mph / 2.23694
                            return v**2 / (9.81 * R) if R > 0 else 0.0
                        secax = ax.secondary_xaxis('top',
                                                   functions=(_g_to_mph, _mph_to_g))
                        secax.set_xlabel('Speed (mph)', color='#4FC3F7',
                                        fontsize=7, labelpad=2)
                        secax.tick_params(colors='#4FC3F7', labelsize=7)
                    except Exception:
                        pass

            fig.tight_layout(rect=[0, 0, 1, top_margin + 0.04])

            # Re-populate CurvesCanvas hover registry so hovering over the
            # aero plots shows per-curve value readouts.
            self.curves._all_axes = [ax1, ax2]
            self.curves._vlines = []
            self.curves._plot_data = []
            for _ax in (ax1, ax2):
                vl = _ax.axvline(x=float('nan'), color='#ffffff', lw=0.8,
                                 ls='--', alpha=0.5, zorder=10)
                self.curves._vlines.append(vl)
                series = []
                for line in _ax.get_lines():
                    lbl = line.get_label()
                    if lbl.startswith('_') or lbl == '':
                        continue
                    series.append((line.get_xdata(), line.get_ydata(),
                                   lbl, line.get_color()))
                self.curves._plot_data.append((_ax, series))

            self.curves.draw()
            self._aero_panel._status.setText(
                f'Sweep done: 0.1\u2013{params["lateral_g"]:.1f}g, {len(g_range)} pts')
        except Exception as e:
            import traceback; traceback.print_exc()
            self._aero_panel._status.setText(f'Error: {e}')

    # ==========================================================================
    #  REPORT EXPORT
    # ==========================================================================

    def _export_report(self):
        """Collect all data from current Vahan state and generate a VD Report.

        Kinematic sweeps (heave / roll) run on the main thread — fast pure
        math.  Dynamics sweeps and DOCX rendering run in _ReportWorker so the
        UI stays responsive.
        """
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import QBuffer, QIODevice

        # ── File save dialog ───────────────────────────────────────────────
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Vehicle Dynamics Report',
            'VD_Report.docx', 'Word Document (*.docx)')
        if not path:
            return

        # ── 3D view screenshot (Qt grab — main thread only) ───────────────
        view3d_png = None
        try:
            px = self.view3d.native.grab()
            if not px.isNull():
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                px.save(buf, 'PNG')
                view3d_png = bytes(buf.data())
        except Exception:
            pass  # screenshot is optional

        # ── Kinematic sweeps (fast, main thread) ──────────────────────────
        # Rebuild solvers at zero steer so heave / roll sweeps are at the
        # design position (same as what _run_sweep does before heave / roll).
        self._rebuild_solvers(0.)

        n = 81
        # Heave range: use the panel's current range when in heave mode;
        # fall back to ±50 mm otherwise so the report always has data.
        if self._motion_panel.motion == 'heave':
            lo_mm = self._motion_panel.min_val
            hi_mm = self._motion_panel.max_val
        else:
            lo_mm, hi_mm = -50., 50.

        t_heave  = np.linspace(lo_mm / 1000., hi_mm / 1000., n)
        heave_x  = t_heave * 1000.  # mm

        _flip_x = np.array([-1., 1., 1.])
        _aln    = self._alignment

        def _arb(lbl):
            src = self._front_arb if lbl in ('FL', 'FR') else self._rear_arb
            return ({k: v * _flip_x for k, v in src.items()}
                    if lbl in ('FR', 'RR') else src)

        def _c_off(lbl):
            return (_aln['front_camber_deg'] if lbl in ('FL', 'FR')
                    else _aln['rear_camber_deg'])

        def _t_off(lbl):
            return (_aln['front_toe_deg'] if lbl in ('FL', 'FR')
                    else _aln['rear_toe_deg'])

        heave_results = {}
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            if lbl in self._solvers:
                heave_results[lbl] = self._do_sweep(
                    self._solvers[lbl], t_heave,
                    'left' if lbl in ('FL', 'RL') else 'right',
                    arb_hp=_arb(lbl),
                    camber_off=_c_off(lbl), toe_off=_t_off(lbl),
                    is_front=lbl in ('FL', 'FR'),
                )

        # Roll sweep — ±3 ° about the longitudinal axis.
        # Use the same track-halfwidth convention as _run_sweep (front WC X).
        roll_degs = np.linspace(-3., 3., n)
        th = self._front_hp['wheel_center'][0]
        t_l = np.sin(np.radians(roll_degs)) * th
        t_r = -t_l
        roll_results = {}
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            if lbl in self._solvers:
                t = t_l if lbl in ('FL', 'RL') else t_r
                roll_results[lbl] = self._do_sweep(
                    self._solvers[lbl], t,
                    'left' if lbl in ('FL', 'RL') else 'right',
                    arb_hp=_arb(lbl),
                    camber_off=_c_off(lbl), toe_off=_t_off(lbl),
                    is_front=lbl in ('FL', 'FR'),
                )

        # ── Dynamics solver (touches panel UI labels — main thread) ────────
        try:
            ss_solver = self._build_dynamics_solver()
        except Exception as exc:
            QMessageBox.warning(self, 'Export Report',
                                f'Could not build dynamics solver:\n{exc}')
            return

        veh_params = self._dynamics_panel.get_params()

        # ── Read current panel state for sweep params ─────────────────────
        test_mode = (self._dynamics_panel._test_mode.currentData()
                     if hasattr(self._dynamics_panel, '_test_mode') else 'cornering')
        lat_g  = self._dynamics_panel._lat_g.value()
        lon_g  = self._dynamics_panel._lon_g.value()
        g_min  = self._dynamics_panel._g_min.value()
        g_max  = self._dynamics_panel._g_max.value()
        start_speed = self._dynamics_panel._start_speed.value()

        # Aero: include if user has it toggled on
        aero_per_g = self._get_aero_Fz_per_g() if self._aero_active else None

        # Build sweep_params dict that mirrors the user's current config.
        # Cornering uses the panel's g range + its lon-g;
        # Straights uses the panel's start speed + lon-g.
        if test_mode == 'straights':
            target_accel = lon_g if lon_g > 0 else 1.5
            target_brake = lon_g if lon_g < 0 else -1.5
            brake_start  = start_speed if start_speed > 5 else 60.0
        else:
            target_accel = 1.5
            target_brake = -1.5
            brake_start  = 60.0

        sweep_params = {
            'g_min':               g_min,
            'g_max':               g_max,
            'lon_g_cornering':     lon_g if test_mode == 'cornering' else 0.0,
            'n_points':            41,
            'start_speed_mph':     start_speed,
            'target_lon_g_accel':  target_accel,
            'target_lon_g_brake':  target_brake,
            'brake_start_mph':     brake_start,
            'aero_Fz_per_g':       aero_per_g,
        }

        # ── Component loads (computed at the panel's current g point) ─────
        loads_data = None
        try:
            from vahan.loads import compute_all_corners
            result = ss_solver.solve(lat_g, lon_g,
                                     aero_Fz=self._get_active_aero_Fz(at_g=lat_g)
                                     if self._aero_active else None)
            bp_f = self._loads_panel.get_brake_params_front()
            bp_r = self._loads_panel.get_brake_params_rear()
            up   = self._loads_panel.get_upright_params()
            veh  = ss_solver._veh
            loads_data = {
                'lat_g': lat_g,
                'lon_g': lon_g,
                'corners': compute_all_corners(
                    self._solvers, result,
                    brake_params_f=bp_f, brake_params_r=bp_r,
                    upright_params_f=up, upright_params_r=up,
                    wheel_radius_m=veh.tire_radius_m,
                    motion_ratio_f=veh.motion_ratio_front,
                    motion_ratio_r=veh.motion_ratio_rear,
                ),
            }
        except Exception:
            pass  # loads section optional — skip if solver fails

        # ── Assemble data dict ─────────────────────────────────────────────
        data = {
            'car_params':    self._car.copy(),
            'veh_params':    veh_params,
            'heave_x_mm':    heave_x,
            'heave_results': heave_results,
            'roll_x_deg':    roll_degs,
            'roll_results':  roll_results,
            'dyn_cornering': {},   # filled by _ReportWorker
            'dyn_accel':     {},
            'dyn_brake':     {},
            'view3d_png':    view3d_png,
            'loads':         loads_data,
        }

        # ── Progress dialog ────────────────────────────────────────────────
        prog = QProgressDialog('Preparing…', None, 0, 100, self)
        prog.setWindowTitle('Vahan — Export VD Report')
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(5)

        # ── Worker ────────────────────────────────────────────────────────
        worker = _ReportWorker(ss_solver, data, path,
                               sweep_params=sweep_params)
        worker.progress.connect(
            lambda msg, pct: (prog.setLabelText(msg), prog.setValue(pct)))
        worker.finished.connect(lambda p: self._on_report_done(p, prog))
        worker.failed.connect(lambda e: self._on_report_failed(e, prog))
        self._report_worker = worker   # keep alive (prevent GC)
        worker.start()

    def _on_report_done(self, path: str, prog):
        prog.close()
        self.statusBar().showMessage(f'Report saved: {path}', 8000)
        reply = QMessageBox.question(
            self, 'Report Ready',
            f'Report saved to:\n{path}\n\nOpen now?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            import os
            try:
                os.startfile(path)           # Windows — opens with default app
            except AttributeError:
                import subprocess
                subprocess.Popen(['open', path])   # macOS

    def _on_report_failed(self, msg: str, prog):
        prog.close()
        # Truncate very long tracebacks in the dialog.
        short = msg[:900] + ('…' if len(msg) > 900 else '')
        QMessageBox.critical(self, 'Report Error',
                             f'Report generation failed:\n\n{short}')
        self.statusBar().showMessage('Report export failed.', 6000)

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
