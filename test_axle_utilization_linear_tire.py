"""axle_utilization and its consumers must accept the parametric LinearTireModel.

vahan/dynamics.py SteadyStateSolver.axle_utilization read ``tire.fz_range``
unguarded while the two sibling sites in solve() use
``getattr(tire, 'fz_range', (0.0,))``.  With no tyre file loaded the
Aero panel's Solve, corner_speed.per_corner_utilization and
cg_tolerance.car_metrics therefore raised AttributeError — and
corner_speed.per_corner_limit_g turned the raised solves into a silent
``limit_g`` equal to its lower bracket (0.3 g).  Failing-then-passing check
(2026-10-05 design-review audit).  Run: python test_axle_utilization_linear_tire.py
"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen'); os.environ['VAHAN_MCP'] = '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from gui.main_window import MainWindow
from vahan.dynamics import AeroDownforceSolver
from vahan.tire_model import LinearTireModel
from vahan import corner_speed as CS

w = MainWindow()
ss = w._build_dynamics_solver()
tire = ss._tire_for('FL')
assert isinstance(tire, LinearTireModel), type(tire)
res = ss.solve(1.0, 0.0)
util = ss.axle_utilization(res)                 # AttributeError before the fix
assert set(util) == {'F', 'R'} and all(np.isfinite(v) and 0.0 < v < 2.0 for v in util.values()), util
aero = AeroDownforceSolver(ss).solve(1.5, 0.0, target_util=0.9)
assert aero is not None
pcu = CS.per_corner_utilization(ss, 1.0)
if pcu is not None:
    assert all(np.isfinite(c['utilization']) for c in pcu['corners'].values()), pcu
lim = CS.per_corner_limit_g(ss)
assert lim['n_failed_solves'] == 0 and np.isfinite(lim['limit_g']) and lim['limit_g'] > 0.31, lim
print(f'per-corner grip limit {lim["limit_g"]:.3f} g with 0 failed solves (was 0.300 g from 24 raised solves)')
print(f'axle utilisation with the parametric tyre: F {util["F"]:.3f} / R {util["R"]:.3f}; '
      f'aero solve and per-corner utilisation run   pass')
