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
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QThread, QEvent, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QImage, QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize as plt_Normalize

from vahan import DoubleWishboneHardpoints
from vahan.solver import (SuspensionConstraints, SolvedState,
                          _norm, _rodrigues, _build_frame)
from vahan.kinematics import KinematicMetrics, _intersect_2d
from vahan.metrics_catalog import (CATALOG, CATALOG_MAP, DEFAULT_Y_KEYS,
                                    compute_ackermann_post)

from gui.view3d import View3D, HP_NAMES
from gui.panels import (
    MotionPanel, CarParamsPanel, HardpointPanel,
    ValuesPanel, GraphPickerPanel, SteeringPanel, AlignmentPanel,
    CollapsibleSection, InverseKinematicsPanel, DynamicsPanel, DynamicsOptPanel,
    LoadsPanel, AeroPanel, SkidpadPanel, BrakeCalcPanel,
    VehicleConstantsPanel, AnalysisPlotsPanel, DirectEditPanel,
    FrameInterferencePanel,
)
from gui.plot_dialog import PlotDialog
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


# ── Direct-damper variants (used when AxleTopology.damper_actuation == DIRECT) ──
# Same wishbone/tie-rod block as the standard sets, but no pushrod/rocker —
# instead a damper between damper_chassis_pt (top, chassis-fixed) and
# damper_outer_pt (bottom, attached to the chosen damper_body — usually LCA).
DEFAULT_FRONT_HP_DIRECT = {
    'uca_front':         np.array([ 0.26353, -0.12700,  0.26353]),
    'uca_rear':          np.array([ 0.23243,  0.12700,  0.24877]),
    'uca_outer':         np.array([ 0.48260,  0.00912,  0.28598]),
    'lca_front':         np.array([ 0.21590, -0.11748,  0.12065]),
    'lca_rear':          np.array([ 0.21590,  0.12342,  0.12700]),
    'lca_outer':         np.array([ 0.53340, -0.00318,  0.11913]),
    'tie_rod_inner':     np.array([ 0.21908, -0.06985,  0.15199]),
    'tie_rod_outer':     np.array([ 0.54293, -0.07303,  0.17145]),
    'wheel_center':      np.array([ 0.55880,  0.00000,  0.20320]),
    # Direct damper: top mount up on the chassis rail (Y centred), bottom mount
    # ~70% along the LCA toward the BJ (representative; user can edit).
    'damper_chassis_pt': np.array([ 0.18000,  0.00000,  0.45000]),
    'damper_outer_pt':   np.array([ 0.45000, -0.00318,  0.13000]),
}
DEFAULT_REAR_HP_DIRECT = {
    'uca_front':         np.array([ 0.27940,  1.44780,  0.26975]),
    'uca_rear':          np.array([ 0.24778,  1.65895,  0.27148]),
    'uca_outer':         np.array([ 0.48895,  1.54940,  0.28075]),
    'lca_front':         np.array([ 0.28158,  1.44958,  0.14356]),
    'lca_rear':          np.array([ 0.22860,  1.65895,  0.12700]),
    'lca_outer':         np.array([ 0.53340,  1.53670,  0.11913]),
    'tie_rod_inner':     np.array([ 0.28158,  1.44958,  0.14356]),
    'tie_rod_outer':     np.array([ 0.53340,  1.46086,  0.12631]),
    'wheel_center':      np.array([ 0.55880,  1.53670,  0.20320]),
    'damper_chassis_pt': np.array([ 0.18000,  1.54623,  0.45000]),
    'damper_outer_pt':   np.array([ 0.45000,  1.54623,  0.13000]),
}


# ── Pullrod variants ─────────────────────────────────────────────────────
# Kinematically identical to pushrod (the solver treats them the same — the
# only physical difference is rod direction and rocker location).  These
# defaults move the rocker LOW (down by the chassis floor) instead of high,
# and route the rod from a low point on the LCA/upright DOWN to the rocker.
# All other constraints — wishbone, tie rod, wheel — are the same as the
# pushrod default set.
DEFAULT_FRONT_HP_PULLROD = {
    'uca_front':         np.array([ 0.26353, -0.12700,  0.26353]),
    'uca_rear':          np.array([ 0.23243,  0.12700,  0.24877]),
    'uca_outer':         np.array([ 0.48260,  0.00912,  0.28598]),
    'lca_front':         np.array([ 0.21590, -0.11748,  0.12065]),
    'lca_rear':          np.array([ 0.21590,  0.12342,  0.12700]),
    'lca_outer':         np.array([ 0.53340, -0.00318,  0.11913]),
    'tie_rod_inner':     np.array([ 0.21908, -0.06985,  0.15199]),
    'tie_rod_outer':     np.array([ 0.54293, -0.07303,  0.17145]),
    'wheel_center':      np.array([ 0.55880,  0.00000,  0.20320]),
    # Pullrod: from a high point on the UPRIGHT (above UCA outer) DOWN to a
    # chassis-mounted rocker near the chassis floor (low rocker pivot).
    'pushrod_outer':     np.array([ 0.50000,  0.00000,  0.30500]),  # high, on upright — Y on rocker plane (was 0.005, 5mm off)
    'pushrod_inner':     np.array([ 0.20000,  0.00000,  0.08000]),  # low, on rocker
    'rocker_pivot':      np.array([ 0.17000,  0.00000,  0.10000]),  # low pivot
    'rocker_spring_pt':  np.array([ 0.14000,  0.00000,  0.16000]),  # spring goes up-back
    'spring_chassis_pt': np.array([ 0.02000,  0.00000,  0.22000]),  # chassis top
    'rocker_axis_pt':    np.array([ 0.17000,  0.02540,  0.10000]),  # axis along Y
}
DEFAULT_REAR_HP_PULLROD = {
    'uca_front':         np.array([ 0.27940,  1.44780,  0.26975]),
    'uca_rear':          np.array([ 0.24778,  1.65895,  0.27148]),
    'uca_outer':         np.array([ 0.48895,  1.54940,  0.28075]),
    'lca_front':         np.array([ 0.28158,  1.44958,  0.14356]),
    'lca_rear':          np.array([ 0.22860,  1.65895,  0.12700]),
    'lca_outer':         np.array([ 0.53340,  1.53670,  0.11913]),
    'tie_rod_inner':     np.array([ 0.28158,  1.44958,  0.14356]),
    'tie_rod_outer':     np.array([ 0.53340,  1.46086,  0.12631]),
    'wheel_center':      np.array([ 0.55880,  1.53670,  0.20320]),
    'pushrod_outer':     np.array([ 0.50000,  1.54623,  0.30500]),  # high on upright — Y on rocker plane (was 1.540, 6mm off)
    'pushrod_inner':     np.array([ 0.20000,  1.54623,  0.08000]),  # low rocker
    'rocker_pivot':      np.array([ 0.17000,  1.54623,  0.10000]),
    'rocker_spring_pt':  np.array([ 0.14000,  1.54623,  0.16000]),
    'spring_chassis_pt': np.array([ 0.02000,  1.54623,  0.22000]),
    'rocker_axis_pt':    np.array([ 0.17000,  1.57163,  0.10000]),
}

# ── T-bar corner variants (used when arb_type == TBAR) ──────────────────
# Built from the user's reference data: rocker_pivot is ~10 mm REARWARD of
# the wheel centre, pushrod_inner is FORWARD of the pivot and LOW, and
# rocker_spring_pt is INBOARD and HIGHER.  Plane normal (the rocker plane)
# computed from those three points is roughly (-0.44 X, -0.87 Y, -0.22 Z)
# = mostly -Y (FORWARD, toward the nosecone) with some -X (inboard) and
# slight -Z (down).
#
# So the rocker plane is TILTED toward the driver — damper extends from
# rocker_spring_pt FORWARD along the plane (mostly -Y, slight +Z) toward
# a chassis attach point near the nosecone.  Drop link extends FORWARD
# along the plane too, to the T-bar lever that sits IN FRONT of the axle.
#
# Vahan +Y = REARWARD (front axle Y=0, rear axle Y≈1.537).
DEFAULT_FRONT_HP_TBAR = {
    'uca_front':         np.array([ 0.26353, -0.12700,  0.26353]),
    'uca_rear':          np.array([ 0.23243,  0.12700,  0.24877]),
    'uca_outer':         np.array([ 0.48260,  0.00912,  0.28598]),
    'lca_front':         np.array([ 0.21590, -0.11748,  0.12065]),
    'lca_rear':          np.array([ 0.21590,  0.12342,  0.12700]),
    'lca_outer':         np.array([ 0.53340, -0.00318,  0.11913]),
    'tie_rod_inner':     np.array([ 0.21908, -0.06985,  0.15199]),
    'tie_rod_outer':     np.array([ 0.54293, -0.07303,  0.17145]),
    'wheel_center':      np.array([ 0.55880,  0.00000,  0.20320]),
    # Rocker low and inboard, mounted at chassis side rail.  Pivot ~10 mm
    # REARWARD of the wheel centre, pushrod_inner FORWARD of pivot + LOW.
    'pushrod_outer':     np.array([ 0.38500, -0.06400,  0.30500]),  # high on UCA/upright
    'pushrod_inner':     np.array([ 0.20000,  0.00000,  0.08000]),  # low, forward
    'rocker_pivot':      np.array([ 0.17000,  0.01000,  0.10000]),  # low chassis pickup
    'rocker_spring_pt':  np.array([ 0.14000,  0.01000,  0.16000]),  # inboard + UP
    # Damper from rocker_spring_pt to a chassis attach.  IMPORTANT: with
    # this rocker layout, rocker_spring_pt swings INBOARD + REARWARD +
    # DOWN on bump (verified empirically), so the chassis attach must be
    # in THAT direction for the damper to COMPRESS.  Earlier defaults
    # put chassis_pt FORWARD (-Y), which made the damper EXTEND on bump
    # -- backwards.  If the user wants the damper to point forward
    # toward the nosecone (typical T-bar packaging), the rocker arm
    # GEOMETRY needs redesign (swap which side of pivot the spring arm
    # sits on), not just the chassis attach position.
    'spring_chassis_pt': np.array([ 0.05000,  0.10000,  0.10000]),
    # rocker_axis_pt snapped to be NORMAL to the rocker plane defined by
    # (rocker_pivot, rocker_spring_pt, pushrod_inner).  The previous
    # value [0.17, 0.04, 0.1] was just pivot + (0, 0.03, 0) which assumed
    # an XZ rocker plane -- but the T-bar rocker plane is tilted, so
    # the pin axis was 29 deg off-normal, leading to the wrong sense of
    # rocker rotation on bump (spring length INCREASED on bump instead
    # of decreased).  Snap normal direction = +Y-ish (matches old intent).
    'rocker_axis_pt':    np.array([ 0.18091,  0.03182,  0.10546]),
    # Rocker-side drop attach.  PROJECTED onto the bellcrank plane
    # (rocker_pivot, pushrod_inner, rocker_spring_pt) so the rocker is a
    # true planar plate -- the ARB drop-arm pickup is coplanar with the
    # pushrod + shock arms (was 45.8 mm out of plane).  The T-bar lever
    # tip (tbar_arm_end) sits ~41 mm away along the drop link.
    'rocker_tbar_drop_pt': np.array([ 0.18500, -0.02000,  0.19000]),
}
DEFAULT_REAR_HP_TBAR = {
    'uca_front':         np.array([ 0.27940,  1.44780,  0.26975]),
    'uca_rear':          np.array([ 0.24778,  1.65895,  0.27148]),
    'uca_outer':         np.array([ 0.48895,  1.54940,  0.28075]),
    'lca_front':         np.array([ 0.28158,  1.44958,  0.14356]),
    'lca_rear':          np.array([ 0.22860,  1.65895,  0.12700]),
    'lca_outer':         np.array([ 0.53340,  1.53670,  0.11913]),
    'tie_rod_inner':     np.array([ 0.28158,  1.44958,  0.14356]),
    'tie_rod_outer':     np.array([ 0.53340,  1.46086,  0.12631]),
    'wheel_center':      np.array([ 0.55880,  1.53670,  0.20320]),
    'pushrod_outer':     np.array([ 0.38500,  1.60000,  0.30500]),
    'pushrod_inner':     np.array([ 0.20000,  1.53623,  0.08000]),
    'rocker_pivot':      np.array([ 0.17000,  1.54623,  0.10000]),
    'rocker_spring_pt':  np.array([ 0.14000,  1.54623,  0.16000]),
    # Rear damper: chassis attach placed inboard-rearward-low of
    # rocker_spring_pt, same reasoning as DEFAULT_FRONT_HP_TBAR -- the
    # rocker arm geometry swings spring_pt in that direction on bump
    # so chassis_pt MUST lie in that direction for compression.
    'spring_chassis_pt': np.array([ 0.05000,  1.63623,  0.10000]),
    # Snapped to plane normal (see DEFAULT_FRONT_HP_TBAR comment).
    'rocker_axis_pt':    np.array([ 0.18091,  1.56805,  0.10546]),
    # Coplanar drop attach, congruent with the FRONT (same rocker-relative
    # offset).  Sits IN FRONT of the rear axle (Y < 1.546) per design.
    'rocker_tbar_drop_pt': np.array([ 0.18500,  1.51623,  0.19000]),
}

# ── ARB / T-bar / Heave / Decoupled per-axle hardware ──────────────────
# Bellcrank ARB (Vahan baseline) is DEFAULT_FRONT_ARB / DEFAULT_REAR_ARB
# defined above — keep those.  New variants below:

# CONTROL-ARM ARB: bar bolts DIRECTLY to LCA (NO drop link).
# 2 stored HPs per axle (mirrored to opposite side).  The "arm" is just
# the segment from arb_pivot (chassis bushing) to arb_lca_attach (bushing
# through the LCA).
DEFAULT_FRONT_ARB_CONTROL_ARM = {
    'arb_pivot':      np.array([ 0.18000, -0.04000,  0.14000]),
    'arb_lca_attach': np.array([ 0.34500, -0.04000,  0.13000]),
}
DEFAULT_REAR_ARB_CONTROL_ARM = {
    'arb_pivot':      np.array([ 0.18000,  1.50623,  0.14000]),
    'arb_lca_attach': np.array([ 0.34500,  1.50623,  0.13000]),
}

# T-BAR — sits IN FRONT OF the axle (FRONT: Y < 0; REAR: Y < rear axle Y).
# Levers reach back from the bar's top node to each rocker's drop arm.
# Torsion bar drops perpendicular to the (tilted) rocker plane.
#
# Geometry built from the user's reference geometry (T-bar Y in front of
# axle by ~60 mm, bar height matches the rocker drop arm height ≈ 0.18 m).
#
#   tbar_base_chassis  — chassis floor pivot.  Pivot axis is parallel to
#                        the steering rack (X direction, lateral).
#   tbar_top_node      — central node where the two levers meet, at the
#                        same Y as the rocker drop arm and same Z (so the
#                        levers + drop links lie IN the tilted rocker plane).
#   tbar_arm_end       — LEFT lever tip.  Pure +X from tbar_top_node so the
#                        lever is perpendicular to the (vertical) torsion
#                        axis; arm length 185 mm.  Mirrored for right.
#   tbar_drop_top      — top of the LEFT drop link.  Equals the rocker's
#                        rocker_tbar_drop_pt (same physical point, stored in
#                        both dicts).  Offset ~41 mm from tbar_arm_end so the
#                        drop link is a REAL, visible rod -- NOT the old
#                        degenerate zero-length link.  The drop direction is
#                        97% aligned with the lever-tip sweep, so the rocker
#                        actually drives the lever (non-singular).  The
#                        rocker's drop arm carries this point as the
#                        suspension moves (tracked live via
#                        _arb_drop_top_world).
DEFAULT_FRONT_TBAR = {
    'tbar_base_chassis': np.array([ 0.00000, -0.06000,  0.05000]),  # chassis floor, IN FRONT of axle
    'tbar_top_node':     np.array([ 0.00000, -0.06000,  0.18000]),  # torsion-bar top, ahead of axle
    'tbar_arm_end':      np.array([ 0.18500, -0.06000,  0.18000]),  # lever tip (pure +X, 185 mm)
    'tbar_drop_top':     np.array([ 0.18500, -0.02000,  0.19000]),  # == rocker_tbar_drop_pt; 41 mm drop link
}
DEFAULT_REAR_TBAR = {
    # Rear T-bar sits IN FRONT of the rear axle (between axle and driver).
    # Congruent with DEFAULT_FRONT_TBAR (same rocker-relative geometry).
    'tbar_base_chassis': np.array([ 0.00000,  1.47623,  0.05000]),
    'tbar_top_node':     np.array([ 0.00000,  1.47623,  0.18000]),
    'tbar_arm_end':      np.array([ 0.18500,  1.47623,  0.18000]),  # lever tip (pure +X, 185 mm)
    'tbar_drop_top':     np.array([ 0.18500,  1.51623,  0.19000]),  # == rocker_tbar_drop_pt; 41 mm drop link
}

# HEAVE 3RD ELEMENT (used WITH T-bar topology): the T-bar's base sits on
# a FLOATING bracket hinged to chassis along a LATERAL axis.  Heave
# coilover restrains the bracket's pivot motion.  4 stored HPs per axle.
DEFAULT_FRONT_HEAVE = {
    # Bracket hinge axis sits at the T-bar's base (same XY centreline, same
    # Z floor) — parallel to steering rack (X direction).
    'heave_bracket_hinge_l':   np.array([-0.06000, -0.06000,  0.05000]),
    'heave_bracket_hinge_r':   np.array([ 0.06000, -0.06000,  0.05000]),
    # 3rd-element coilover sits IN THE ROCKER PLANE (along Y at chassis
    # top) — bracket pivot in heave mode translates the bracket end along Y.
    # FRONT axle: coilover runs FORWARD (toward nosecone).
    # 3rd (heave) spring — reference position (the point the user grabs); the
    # solver + dynamics read THESE so the graph responds to what's visible.
    'heave_spring_tbar_pt':    np.array([ 0.00000, -0.02700,  0.50300]),
    'heave_spring_chassis_pt': np.array([ 0.00000,  0.23000,  0.54000]),
    # ── reference-derived heave/T-bar central geometry (scaled to our 1.22 m
    # track; mated to our wishbones via pushrod_outer).  Drives the kinematic
    # graph + dynamics through vahan/heave_tbar.HeaveTBarRockerSolver
    # (wheel->pushrod->rocker->drop-link->T-bar heave-pivot->3rd spring).
    'htb_rocker_pivot':   np.array([ 0.11090,  0.30100,  0.52900]),
    'htb_rocker_axis':    np.array([ 0.00000,  0.31200,  0.30600]),
    'htb_pushrod_inner':  np.array([ 0.08850,  0.25900,  0.54300]),
    'htb_drop_foot':      np.array([ 0.05540,  0.27600,  0.56800]),
    'htb_arm_tip':        np.array([ 0.05790, -0.06700,  0.50300]),
    'htb_tbar_pivot':     np.array([ 0.00000, -0.02900,  0.31600]),
    'htb_tbar_axis':      np.array([ 0.08850, -0.02900,  0.31600]),
    'htb_tbar_junc':      np.array([ 0.00000, -0.06700,  0.50300]),
    # Corner damper (the reference "coilover") on the bellcrank's FAR end
    # (Y behind the pushrod) -> chassis.  rocker end rotates with the bellcrank.
    'htb_coil_rocker':    np.array([ 0.04714,  0.32100,  0.57500]),
    'htb_coil_chassis':   np.array([ 0.05459,  0.60100,  0.59500]),
}
DEFAULT_REAR_HEAVE = {
    'heave_bracket_hinge_l':   np.array([-0.06000,  1.47000,  0.05000]),
    'heave_bracket_hinge_r':   np.array([ 0.06000,  1.47000,  0.05000]),
    # REAR axle: coilover runs REARWARD (toward gearbox bay).
    'heave_spring_tbar_pt':    np.array([ 0.00000,  1.51000,  0.50300]),
    'heave_spring_chassis_pt': np.array([ 0.00000,  1.76700,  0.54000]),
    # reference central, Y-shifted to the rear axle (front Y + 1.537 m).
    'htb_rocker_pivot':   np.array([ 0.11090,  1.83800,  0.52900]),
    'htb_rocker_axis':    np.array([ 0.00000,  1.84900,  0.30600]),
    'htb_pushrod_inner':  np.array([ 0.08850,  1.79600,  0.54300]),
    'htb_drop_foot':      np.array([ 0.05540,  1.81300,  0.56800]),
    'htb_arm_tip':        np.array([ 0.05790,  1.47000,  0.50300]),
    'htb_tbar_pivot':     np.array([ 0.00000,  1.50800,  0.31600]),
    'htb_tbar_axis':      np.array([ 0.08850,  1.50800,  0.31600]),
    'htb_tbar_junc':      np.array([ 0.00000,  1.47000,  0.50300]),
    # Corner damper (Y-shifted to the rear axle: front Y + 1.537 m).
    'htb_coil_rocker':    np.array([ 0.04714,  1.85800,  0.57500]),
    'htb_coil_chassis':   np.array([ 0.05459,  2.13800,  0.59500]),
}

# DECOUPLED CORNER (no rocker, no corner spring — pushrod feeds straight
# into the per-axle TWIN-BELLCRANK block defined below).
#
# Standard wishbone + tie rod + upright + wheel center; the only extra
# is pushrod_outer (the upright-side attach for the pushrod going to its
# OWN side's bellcrank).  Pushrod_inner / rocker / damper attaches all
# live in DEFAULT_FRONT_DECOUPLED (per-axle, NOT mirrored — see asymmetry
# note there).
DEFAULT_FRONT_HP_DECOUPLED = {
    'uca_front':     np.array([ 0.26353, -0.12700,  0.26353]),
    'uca_rear':      np.array([ 0.23243,  0.12700,  0.24877]),
    'uca_outer':     np.array([ 0.48260,  0.00912,  0.28598]),
    'lca_front':     np.array([ 0.21590, -0.11748,  0.12065]),
    'lca_rear':      np.array([ 0.21590,  0.12342,  0.12700]),
    'lca_outer':     np.array([ 0.53340, -0.00318,  0.11913]),
    'tie_rod_inner': np.array([ 0.21908, -0.06985,  0.15199]),
    'tie_rod_outer': np.array([ 0.54293, -0.07303,  0.17145]),
    'wheel_center':  np.array([ 0.55880,  0.00000,  0.20320]),
    'pushrod_outer': np.array([ 0.44000,  0.00000,  0.30000]),
}
DEFAULT_REAR_HP_DECOUPLED = {
    'uca_front':     np.array([ 0.27940,  1.44780,  0.26975]),
    'uca_rear':      np.array([ 0.24778,  1.65895,  0.27148]),
    'uca_outer':     np.array([ 0.48895,  1.54940,  0.28075]),
    'lca_front':     np.array([ 0.28158,  1.44958,  0.14356]),
    'lca_rear':      np.array([ 0.22860,  1.65895,  0.12700]),
    'lca_outer':     np.array([ 0.53340,  1.53670,  0.11913]),
    'tie_rod_inner': np.array([ 0.28158,  1.44958,  0.14356]),
    'tie_rod_outer': np.array([ 0.53340,  1.46086,  0.12631]),
    'wheel_center':  np.array([ 0.55880,  1.53670,  0.20320]),
    'pushrod_outer': np.array([ 0.44000,  1.54000,  0.30000]),
}

# DECOUPLED — twin bellcrank + cross-car heave damper + cross-car roll
# damper, geometry derived from the reference layout "Push-Pull Heave and
# Roll Dampers" reference layout.
#
# source-frame convention (verified by inspection of the user's reference
# project): X = longitudinal (+X forward), Y = lateral (+Y LEFT),
# Z = vertical (+Z up).  Units = inches.  Origin = front axle plane.
#
# Vahan convention:  X = lateral (+X = LEFT outboard = FL/RL),
#                    Y = longitudinal (+Y = rearward, front axle at Y=0),
#                    Z = vertical (+Z up).  Units = metres.
#
# Conversion used here:
#   Vahan_X = +Src_Y * 0.0254
#   Vahan_Y = -Src_X * 0.0254   (source +X fwd → Vahan -Y fwd)
#   Vahan_Z = +Src_Z * 0.0254
#
# Mechanical concept (NOT a monoshock cradle — that was the previous
# wrong model):
#   * TWO bellcranks, one per side.  Each pivots about its OWN
#     longitudinal axis (parallel to the chassis fore-aft direction).
#     Each is fed by its OWN pushrod.
#   * NO corner springs.
#   * HEAVE coilover: cross-car damper between the two rockers, both
#     attaches at the SAME Z (above each pivot).  In heave both rockers
#     rotate in mirrored directions → the two attaches move toward each
#     other laterally → damper compresses.  In roll one rocker rotates
#     opposite the other → the two attaches translate together → damper
#     length unchanged.
#   * ROLL coilover: cross-car damper between the two rockers, but with
#     ASYMMETRIC Z: LEFT attach BELOW the LEFT pivot, RIGHT attach ABOVE
#     the RIGHT pivot.  In heave both rocker rotations move the LEFT end
#     up while the RIGHT end goes down by ~equal amounts → damper rotates
#     but doesn't change length.  In roll both ends move the same Z
#     direction → damper length changes a lot.
#
# That asymmetric Z is what decouples the modes.  Verified mathematically
# in the user's reference project (heave-mode length change ~3.94×θ, roll
# damper change in heave is second-order in θ; roll damper length change
# ~4.72×θ in roll, heave damper change in roll is second-order).
#
# Stored hardpoints (per axle, 10 total — EXPLICIT left/right because of
# the asymmetric roll damper Z):
#   rocker_pivot_left/right       chassis-fixed pivot point
#   rocker_axis_pt_left/right     2nd point defining pivot axis (along ~Y)
#   pushrod_inner_left/right      rocker-side pushrod attach
#   heave_damper_left/right       rocker attach for cross-car heave damper
#                                 (both at SAME Z — symmetric)
#   roll_damper_left/right        rocker attach for cross-car roll damper
#                                 (DIFFERENT Z — LEFT below pivot, RIGHT
#                                 above pivot — that's the decoupling)
DEFAULT_FRONT_DECOUPLED = {
    # LEFT rocker (FL).  OK Y=+5.276, Z=19.685, X=1.181→5.118 (pivot axis
    # along +X, in Vahan = -Y, i.e. forward).
    'rocker_pivot_left':       np.array([ 0.13401, -0.03000,  0.50000]),
    'rocker_axis_pt_left':     np.array([ 0.13401, -0.13000,  0.50000]),
    'pushrod_inner_left':      np.array([ 0.15001, -0.02000,  0.56000]),
    # 12-DOF IK optimum (vahan.ik_decoupled, full_3d=True).  All three
    # axes of each damper attach are varied within a 150 mm envelope
    # around the rocker pivot.  Final cross-coupling: heave_in_roll
    # leak ~3.3e-5 mm, roll_in_heave leak ~8.5e-5 mm over ±10 mm wheel
    # travel.  Practically perfect decoupling.
    'heave_damper_left':       np.array([ 0.12150179, -0.10000000,  0.60415480]),
    'roll_damper_left':        np.array([ 0.10000429,  0.04001000,  0.44144344]),

    # RIGHT rocker (FR).  Mirror in X for the corner block; damper
    # attaches are 12-DOF IK optimised (asymmetric roll-damper Z is what
    # creates the decoupling -- LEFT below pivot, RIGHT above).
    'rocker_pivot_right':      np.array([-0.13401, -0.03000,  0.50000]),
    'rocker_axis_pt_right':    np.array([-0.13401, -0.13000,  0.50000]),
    'pushrod_inner_right':     np.array([-0.15001, -0.02000,  0.56000]),
    'heave_damper_right':      np.array([-0.12143666, -0.10000000,  0.60415820]),
    'roll_damper_right':       np.array([-0.10000205,  0.04001000,  0.55854264]),
}
DEFAULT_REAR_DECOUPLED = {
    # Same layout as front, shifted to rear axle (Y += 1.5367 m).
    # Z values optimised by vahan.ik_decoupled (see DEFAULT_FRONT_DECOUPLED
    # comment).  Front and rear cradles share the same Z geometry because
    # they share the same corner kinematics (mirrored corner HPs).
    'rocker_pivot_left':       np.array([ 0.13401,  1.50670,  0.50000]),
    'rocker_axis_pt_left':     np.array([ 0.13401,  1.40670,  0.50000]),
    'pushrod_inner_left':      np.array([ 0.15001,  1.51670,  0.56000]),
    # 12-DOF IK optimum -- same values as front (front + rear cradles
    # share corner kinematics, so the optimised attach geometry is
    # identical apart from the Y shift).
    'heave_damper_left':       np.array([ 0.12150179,  1.43670000,  0.60415480]),
    'roll_damper_left':        np.array([ 0.10000429,  1.57670000,  0.44144344]),

    'rocker_pivot_right':      np.array([-0.13401,  1.50670,  0.50000]),
    'rocker_axis_pt_right':    np.array([-0.13401,  1.40670,  0.50000]),
    'pushrod_inner_right':     np.array([-0.15001,  1.51670,  0.56000]),
    'heave_damper_right':      np.array([-0.12143666,  1.43670000,  0.60415820]),
    'roll_damper_right':       np.array([-0.10000205,  1.57670000,  0.55854264]),
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
    """Build a DoubleWishboneHardpoints from a plain dict (metres).

    Filters the input to ONLY keys the dataclass accepts.  Some topologies
    add extra corner HPs that the kinematic solver doesn't know about —
    e.g. ``rocker_tbar_drop_pt`` lives in the T-bar corner default dict so
    we can render the drop link to the T-bar lever, but the corner
    kinematic solver (wishbones + pushrod + rocker) has no use for it.
    Without this filter, calling _hp_obj on a T-bar corner dict raised
    a TypeError that was silently swallowed by _rebuild_solvers' try/except,
    leaving the previous solver instance in place — so T-bar / HEAVE_TBAR /
    DECOUPLED were silently running on STANDARD-topology kinematics.
    """
    import dataclasses
    accepted = {f.name for f in dataclasses.fields(DoubleWishboneHardpoints)}
    filtered = {k: np.array(v, float) for k, v in d.items() if k in accepted}
    return DoubleWishboneHardpoints(**filtered)


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
        # Right-click context menu for export / copy
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    # ── Export / copy support ────────────────────────────────────────────
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_dark = QAction('Copy graph (as displayed)', self)
        copy_dark.triggered.connect(self._copy_as_displayed)
        menu.addAction(copy_dark)

        copy_light = QAction('Copy graph (light theme, for documents)', self)
        copy_light.triggered.connect(self._copy_light_theme)
        menu.addAction(copy_light)

        menu.addSeparator()

        save_dark = QAction('Save graph as image (as displayed)…', self)
        save_dark.triggered.connect(lambda: self._save_image(light=False))
        menu.addAction(save_dark)

        save_light = QAction('Save graph as image (light theme)…', self)
        save_light.triggered.connect(lambda: self._save_image(light=True))
        menu.addAction(save_light)

        menu.exec(event.globalPos())

    def _fig_to_bytes(self, light=False, fmt='png', dpi=200):
        """Render current figure to PNG bytes. Optionally swap to light theme."""
        import io
        buf = io.BytesIO()
        if light:
            self._with_light_theme(lambda: self.fig.savefig(
                buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white'))
        else:
            self.fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()

    def _with_light_theme(self, fn):
        """Temporarily swap dark theme colors to light, call fn, swap back."""
        original = {}
        original['fig_facecolor'] = self.fig.get_facecolor()
        self.fig.set_facecolor('white')
        for ax in self.fig.axes:
            original.setdefault('axes', []).append({
                'facecolor': ax.get_facecolor(),
                'spine_colors': {k: s.get_edgecolor() for k, s in ax.spines.items()},
                'tick_color': ax.xaxis.get_ticklabels()[0].get_color()
                              if ax.xaxis.get_ticklabels() else '#777',
                'xlabel_color': ax.xaxis.label.get_color(),
                'ylabel_color': ax.yaxis.label.get_color(),
            })
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_edgecolor('#666')
            ax.tick_params(colors='#333')
            ax.xaxis.label.set_color('#222')
            ax.yaxis.label.set_color('#222')
            ax.grid(True, color='#DDD', lw=0.6)
        # Suptitle
        original['suptitle_color'] = self.fig._suptitle.get_color() if self.fig._suptitle else None
        if self.fig._suptitle:
            self.fig._suptitle.set_color('#111')
        try:
            fn()
        finally:
            # Restore
            self.fig.set_facecolor(original['fig_facecolor'])
            for ax, ax_orig in zip(self.fig.axes, original.get('axes', [])):
                ax.set_facecolor(ax_orig['facecolor'])
                for k, s in ax.spines.items():
                    s.set_edgecolor(ax_orig['spine_colors'].get(k, '#222'))
                ax.tick_params(colors='#777')
                ax.xaxis.label.set_color('#888')
                ax.yaxis.label.set_color('#888')
                ax.grid(True, color='#1a1a1a', lw=0.5)
            if self.fig._suptitle and original['suptitle_color']:
                self.fig._suptitle.set_color(original['suptitle_color'])
            self.draw_idle()

    def _copy_as_displayed(self):
        data = self._fig_to_bytes(light=False)
        img = QImage.fromData(data, 'PNG')
        QApplication.clipboard().setImage(img)

    def _copy_light_theme(self):
        data = self._fig_to_bytes(light=True)
        img = QImage.fromData(data, 'PNG')
        QApplication.clipboard().setImage(img)

    def _save_image(self, light=False):
        default_name = 'graph_light.png' if light else 'graph.png'
        path, sel_filter = QFileDialog.getSaveFileName(
            self, 'Save graph as image', default_name,
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:
            return
        fmt = 'png'
        if path.lower().endswith('.pdf'):
            fmt = 'pdf'
        elif path.lower().endswith('.svg'):
            fmt = 'svg'
        data = self._fig_to_bytes(light=light, fmt=fmt, dpi=300)
        with open(path, 'wb') as f:
            f.write(data)

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

        # Colour-blind-safe series palettes (user confuses purple/blue and
        # red/green/orange): stick to yellow / red / white / blue + lightness.
        if 'lt' in graphs:
            plots.append(('Load Transfer', 'LT (N)', [
                ('Elastic F', sweep['elastic_lt_front_N'], '#64B5F6', '-'),
                ('Elastic R', sweep['elastic_lt_rear_N'], '#0D47A1', '--'),
                ('Geo F', sweep['geometric_lt_front_N'], '#E53935', '-.'),
                ('Geo R', sweep['geometric_lt_rear_N'], '#FFFFFF', ':'),
            ]))

        if 'rc' in graphs:
            plots.append(('Roll Centre Height', 'RC (mm)', [
                ('Front', sweep['rc_height_front_mm'], '#42A5F5', '-'),
                ('Rear', sweep['rc_height_rear_mm'], '#FFD600', '--'),
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
                        ('Extra (US)', us * steer_ratio, '#FFD600', '-.'),
                    ]))
                else:
                    plots.append(('Steer Correction', 'Front wheel angle (deg)', [
                        ('Ackermann', ack_deg, '#555555', '--'),
                        ('Steering Angle', total_steer, '#4FC3F7', '-'),
                        ('Extra (US)', us, '#FFD600', '-.'),
                    ]))
            elif us is not None and np.any(us):
                plots.append(('Steer Correction', 'Extra steer (deg)', [
                    ('Extra (US)', us, '#FFD600', '-'),
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

    # ── Tire / grip characterization ─────────────────────────────────────
    def plot_tire_grip(self, tire, result=None, camber_deg: float = 0.0,
                       fc_fz_levels=None, fc_3d: bool = False):
        """Tire characterization + friction-circle view (2×2):
          • Lateral force Fy vs slip angle (a family at several Fz)
          • Cornering stiffness Cα vs normal load
          • Aligning moment Mz vs slip angle (family at several Fz)
          • Friction circle — each corner's (Fx, Fy) operating point
            normalised by its grip capacity μ·Fz against the unit circle.

        `tire` is a TireModel or LinearTireModel.  `result` (optional
        SteadyStateResult) supplies the friction-circle operating point.
        Mz is zero for the linear fallback — annotated so it's clear TTC
        data is needed for aligning moment."""
        self._hover_ann = None
        self.fig.clf()
        self._all_axes = []
        self._vlines = []
        self._plot_data = []

        # Load levels for the SA-family curves + the Fz sweep range
        try:
            fz_lo, fz_hi = tire.fz_range
            fz_lo = max(float(fz_lo), 50.0); fz_hi = max(float(fz_hi), fz_lo + 100.0)
        except Exception:
            fz_lo, fz_hi = 250.0, 1300.0
        fz_levels = np.linspace(fz_lo, fz_hi, 4)
        # Slip-angle axis (rising side of the curve)
        try:
            sa_lo, sa_hi = tire.sa_range
            sa = np.linspace(0.0, max(float(sa_hi), 6.0), 70)
        except Exception:
            sa = np.linspace(0.0, 12.0, 70)
        # Colourblind-safe load ramp (blue → white → yellow → red = light→heavy)
        LOAD_COLS = ['#4FC3F7', '#FFFFFF', '#FFD600', '#FF5252']

        def _style(ax, xlabel, ylabel, title):
            ax.set_facecolor('#080808')
            ax.tick_params(colors='#777777', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#222222')
            ax.set_xlabel(xlabel, color='#888888', fontsize=8, labelpad=2)
            ax.set_ylabel(ylabel, color='#888888', fontsize=8, labelpad=2)
            ax.set_title(title, color='#cccccc', fontsize=9)
            ax.grid(True, color='#1a1a1a', lw=0.5)

        def _legend(ax, **kw):
            ax.legend(fontsize=7, facecolor='#06060e', labelcolor='white',
                      framealpha=0.7, handlelength=1.0, **kw)

        # (1) Lateral force vs slip angle
        ax1 = self.fig.add_subplot(2, 2, 1)
        _style(ax1, 'Slip angle (deg)', 'Lateral force |Fy| (N)',
               'Lateral Force vs Slip Angle')
        for fz, col in zip(fz_levels, LOAD_COLS):
            fy = np.array([abs(float(tire.Fy(float(s), float(fz), camber_deg)))
                           for s in sa])
            ax1.plot(sa, fy, color=col, lw=1.8, label=f'{fz:.0f} N')
        _legend(ax1, loc='lower right', title='Fz', title_fontsize=7)

        # (2) Cornering stiffness vs load
        ax2 = self.fig.add_subplot(2, 2, 2)
        _style(ax2, 'Normal load Fz (N)', 'Cornering stiffness Cα (N/deg)',
               'Cornering Stiffness vs Normal Load')
        fz_sweep = np.linspace(max(50.0, fz_lo * 0.4), fz_hi * 1.05, 70)
        ca = np.array([abs(float(tire.cornering_stiffness(float(f), camber_deg)))
                       for f in fz_sweep])
        ax2.plot(fz_sweep, ca, color='#4FC3F7', lw=1.8)

        # (3) Aligning moment vs slip angle
        ax3 = self.fig.add_subplot(2, 2, 3)
        _style(ax3, 'Slip angle (deg)', 'Aligning moment Mz (Nm)',
               'Aligning Moment vs Slip Angle')
        mz_any = False
        for fz, col in zip(fz_levels, LOAD_COLS):
            mz = np.array([float(tire.Mz(float(s), float(fz), camber_deg))
                           for s in sa])
            if np.any(np.abs(mz) > 1e-6):
                mz_any = True
            ax3.plot(sa, mz, color=col, lw=1.8, label=f'{fz:.0f} N')
        if mz_any:
            _legend(ax3, loc='best', title='Fz', title_fontsize=7)
        else:
            ax3.text(0.5, 0.5, 'No aligning-moment data\n'
                     '(linear model — load TTC data for Mz)',
                     transform=ax3.transAxes, ha='center', va='center',
                     color='#777777', fontsize=8)

        # (4) Friction circle in FORCE (N): one circle per vertical load.
        # Radius = the tire's peak lateral force at that load (friction-circle
        # assumption mu_x = mu_y, same as the solver).  Loads are either the
        # user's list (fc_fz_levels) or the automatic fz_levels above.
        # fc_3d stacks the circles over the tire's full load range instead.
        circle_fz = list(fc_fz_levels) if fc_fz_levels else [float(f) for f in fz_levels]
        th = np.linspace(0, 2 * np.pi, 120)

        def _peak_force(fzv):
            try:
                return abs(float(tire.peak_Fy(max(float(fzv), 10.0))))
            except Exception:
                return abs(float(tire.peak_mu(max(float(fzv), 10.0),
                                              camber_deg))) * max(float(fzv), 10.0)

        if fc_3d:
            ax4 = self.fig.add_subplot(2, 2, 4, projection='3d')
            ax4.set_facecolor('#080808')
            # MATLAB-style shaded surface: fine mesh, colour = grip force
            # radius at that load ('hot' colormap — black->red->yellow->white,
            # colourblind-safe), light-source shading for depth.
            from matplotlib.colors import Normalize
            import matplotlib.cm as _cm
            th_f = np.linspace(0, 2 * np.pi, 121)
            fz_span = np.linspace(max(fz_lo * 0.4, 50.0), fz_hi * 1.05, 90)
            R = np.array([_peak_force(f) for f in fz_span])
            TH, FZg = np.meshgrid(th_f, fz_span)
            Rg = np.repeat(R[:, None], len(th_f), axis=1)
            Xs, Ys = Rg * np.cos(TH), Rg * np.sin(TH)
            norm = Normalize(vmin=float(R.min()), vmax=float(R.max()))
            cmap = _cm.get_cmap('hot')
            face = cmap(norm(Rg))
            # light from the upper-left: modulate brightness by the surface
            # normal (radial for a cone) against the light azimuth
            bright = 0.55 + 0.45 * np.clip(np.cos(TH - np.radians(135)), 0, 1)
            face[..., :3] *= bright[..., None]
            surf = ax4.plot_surface(Xs, Ys, FZg, facecolors=face,
                                    rstride=1, cstride=1, linewidth=0,
                                    antialiased=True, shade=False)
            m = _cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(Rg)
            cb = self.fig.colorbar(m, ax=ax4, shrink=0.55, pad=0.10)
            cb.set_label('grip force μ(Fz)·Fz  (N)', color='#888888', fontsize=7)
            cb.ax.tick_params(colors='#777777', labelsize=6)
            for fzv, col in zip(circle_fz, ['#FFFFFF', '#4FC3F7', '#B0BEC5']):
                r = _peak_force(fzv)
                ax4.plot(r * np.cos(th), r * np.sin(th), fzv, color=col,
                         lw=1.8, label=f'{fzv:.0f} N')
            if result is not None:
                CC = dict(CORNER_PLOT_COLORS)
                for c in ('FL', 'FR', 'RL', 'RR'):
                    try:
                        fz = float(getattr(result, 'Fz', {}).get(c, 0.0))
                        fy = float(getattr(result, 'Fy', {}).get(c, 0.0))
                        fx = float(getattr(result, 'Fx', {}).get(c, 0.0))
                        if fz > 1.0:
                            ax4.scatter([fy], [fx], [fz], s=45,
                                        color=CC.get(c, '#ffffff'),
                                        edgecolor='black', lw=0.5)
                    except Exception:
                        continue
            ax4.set_xlabel('Fy lat (N)', color='#888888', fontsize=7)
            ax4.set_ylabel('Fx long (N)', color='#888888', fontsize=7)
            ax4.set_zlabel('Fz load (N)', color='#888888', fontsize=7)
            ax4.set_title('Friction Circle vs Vertical Load (3D)',
                          color='#cccccc', fontsize=9)
            ax4.tick_params(colors='#777777', labelsize=6)
            _legend(ax4, loc='upper left', bbox_to_anchor=(-0.05, 1.0),
                    title='Fz', title_fontsize=7)
        else:
            ax4 = self.fig.add_subplot(2, 2, 4)
            _style(ax4, 'Longitudinal force Fx (N)', 'Lateral force Fy (N)',
                   'Friction Circle vs Vertical Load')
            r_max = 1.0
            for fzv, col in zip(circle_fz, LOAD_COLS):
                r = _peak_force(fzv)
                r_max = max(r_max, r)
                ax4.plot(r * np.cos(th), r * np.sin(th), color=col, lw=1.8,
                         label=f'Fz {fzv:.0f} N → {r:.0f} N')
            ax4.axhline(0, color='#222222', lw=0.6)
            ax4.axvline(0, color='#222222', lw=0.6)
            plotted = False
            if result is not None:
                CC = dict(CORNER_PLOT_COLORS)
                for c in ('FL', 'FR', 'RL', 'RR'):
                    try:
                        fz = float(getattr(result, 'Fz', {}).get(c, 0.0))
                        fy = float(getattr(result, 'Fy', {}).get(c, 0.0))
                        fx = float(getattr(result, 'Fx', {}).get(c, 0.0))
                        if fz <= 1.0:
                            continue
                        ax4.scatter(fx, fy, s=45, color=CC.get(c, '#ffffff'),
                                    edgecolor='black', lw=0.5, zorder=5,
                                    label=c)
                        plotted = True
                    except Exception:
                        continue
            lim = r_max * 1.15
            ax4.set_xlim(-lim, lim); ax4.set_ylim(-lim, lim)
            ax4.set_aspect('equal', adjustable='box')
            _legend(ax4, loc='upper right')
            if not plotted:
                ax4.text(0, -lim * 1.1, 'solve dynamics for operating points',
                         ha='center', va='top', color='#777777', fontsize=7)

        self.fig.subplots_adjust(hspace=0.55, wspace=0.38,
                                 left=0.10, right=0.97, top=0.90, bottom=0.11)
        tname = getattr(tire, 'tire_id', 'tire')
        self.fig.suptitle(f'Tire / Grip Characterization  —  {tname}',
                          color='#cccccc', fontsize=10, y=0.98)
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
        # Per-axle extra hardware that only some topologies use.  Empty
        # dicts when not applicable (e.g. heave is only populated for
        # HEAVE_TBAR spring config; decoupled only for DECOUPLED).
        self._front_heave: dict = {}
        self._rear_heave:  dict = {}
        self._front_decoupled: dict = {}
        self._rear_decoupled:  dict = {}
        # Suspension topology — defaults to standard pushrod+bellcrank-ARB on
        # both axles (the only layout the rest of the codebase currently
        # supports out of the box).  Replaced by the wizard or by load.
        from vahan.topology import SuspensionTopology
        self._topology = SuspensionTopology.standard()
        self._car       = {'axle_spacing_mm': 1537., 'wheelbase_mm': 1537.,
                           'track_f_mm': 1222., 'track_r_mm': 1200.,
                           'wheel_offset_f_mm': 25., 'wheel_offset_r_mm': 25.,
                           'tire_outer_dia_mm': 406., 'tire_rim_dia_mm': 330.,
                           'tire_width_mm': 200., 'show_ground': True,
                           'cg_x_mm': 0., 'cg_y_mm': 845., 'cg_z_mm': 280.,
                           'front_brake_bias_pct': 65.,
                           # Rack length drives tie_rod_inner X.  Set by the
                           # startup wizard; defaults to the legacy hard-coded
                           # value (2 × 219.08 mm) so existing files behave
                           # identically when rack_length_mm isn't present.
                           'rack_length_mm': 438.16,
                           # Spring/damper outside diameter (mm).  Editable
                           # in the wizard and Car panel; used by view3d to
                           # render the spring as a visible cylinder/thick
                           # line.  Typical Eibach 1.88" ID coil: OD ≈ 63 mm
                           # incl. coils; OZ Öhlins TTX36 damper body ≈ 50 mm.
                           'spring_od_mm': 63.0,
                           'damper_od_mm': 50.0,
                           # Track-change behaviour: False (default) shifts
                           # only outboard pickups + wheel; True also shifts
                           # inboard chassis pickups so arms keep length.
                           'track_pushes_inboard': False}
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
        self._spring_travel_cache: dict = {}  # label → (t_lo, t_hi) spring-stroke travel range
        self._show_rc        = True
        self._show_roll_axis = True
        self._show_cg        = True
        self._show_pitch_axis = True

        # ── Direct-edit state ─────────────────────────────────────────────
        # `_edit_mode` is a toggle from the 3D overlay panel.  When ON, the
        # view3d widget receives WASD/QE keystrokes and emits hardpoint
        # deltas via `_on_hp_move`.  `_edit_history` is a bounded LIFO of
        # the previous hp-value (for Ctrl+Z).
        self._edit_mode          = False
        self._edit_increment_mm  = 1.0
        self._edit_history: list = []
        self._redo_stack: list = []     # undone edits, re-applied by Ctrl+Y
        # Baseline geometry snapshot for the ghost overlay + Δ readout
        # ('Set baseline' in Direct Edit).  None until the user sets one.
        self._baseline_geo: dict | None = None
        # Constrained-nudge mode: 'free' | 'link' | 'plane'
        self._edit_constraint = 'free'
        self._3d_pending     = False     # deferred 3D update flag
        self._tire_model     = None      # front-axle TireModel / LinearTireModel
        self._tire_model_rear = None     # rear tire (None = same as front; set
                                         # for a SPLIT front/rear compound setup)
        from vahan.differential import Differential
        self._diff = Differential()      # Drexler FSAE LSD (default option 1)
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
        change_topo_act = fm.addAction('Change Topology…')
        change_topo_act.setToolTip(
            'Re-open the per-axle topology selector (damper actuation, ARB '
            'type, spring config).  Replaces topology-specific hardpoint '
            'blocks with the defaults for the new topology; corner HPs, '
            'vehicle dimensions, and panel state are preserved.')
        change_topo_act.triggered.connect(self._change_topology_dialog)
        fm.addSeparator()
        export_rpt_act = fm.addAction('Export Report…')
        export_rpt_act.setToolTip(
            'Generate a Vehicle Dynamics Report (.docx) with all graphs and '
            'auto-analysis — opens and edits cleanly in Google Docs')
        export_rpt_act.triggered.connect(self._export_report)
        export_csv_act = fm.addAction('Export Sweep Data (CSV)…')
        export_csv_act.setToolTip(
            'Write the current kinematic sweep (every metric, all 4 corners) '
            'and the last dynamics sweep to CSV files for Excel/Matlab')
        export_csv_act.triggered.connect(self._export_sweep_csv)

        # ── Page switcher: Suspension ↔ Laptime ───────────────────────────
        pm = mb.addMenu('Page')
        susp_act = pm.addAction('Suspension')
        susp_act.setShortcut('Ctrl+1')
        susp_act.triggered.connect(lambda: self._switch_page(0))
        lap_act = pm.addAction('Laptime')
        lap_act.setShortcut('Ctrl+2')
        lap_act.triggered.connect(lambda: self._switch_page(1))
        city_act = pm.addAction('Design City')
        city_act.setShortcut('Ctrl+3')
        city_act.triggered.connect(lambda: self._switch_page(2))
        loads_act = pm.addAction('Loads')
        loads_act.setShortcut('Ctrl+4')
        loads_act.triggered.connect(lambda: self._switch_page(3))

        vm = mb.addMenu('View')
        hp_act = vm.addAction('All Hardpoints…')
        hp_act.triggered.connect(self._show_all_hardpoints)

    def _change_topology_dialog(self):
        """Open a small dialog with just the per-axle topology pickers,
        seeded with the currently-loaded topology.  On Accept, calls
        ``set_topology`` with the new choice — that replaces the
        topology-conditional hardpoint blocks (ARB / heave / decoupled)
        with the new defaults, but PRESERVES self._car / self._steer /
        wheelbase + track / spring OD / direct-edit history (all the
        non-topology state).
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        )
        from gui.startup_dialog import _AxleTopologyEditor
        from vahan.topology import SuspensionTopology

        dlg = QDialog(self)
        dlg.setWindowTitle('Change Suspension Topology')
        dlg.setMinimumSize(620, 520)
        dlg.setStyleSheet("""
            QDialog { background:#0e0e10; }
            QLabel  { color:#e8e8ec; }
            QPushButton { background:#1a1a1a; color:#e8e8e8; border:1px solid #2a2a2a;
                          padding:8px 16px; border-radius:4px; font-size:12px; }
            QPushButton:hover    { background:#2a2a2a; }
            QPushButton:default,
            QPushButton#primary  { background:#FFD600; color:#0a0a0a; border:0;
                                   font-weight:bold; }
            QPushButton:default:hover,
            QPushButton#primary:hover { background:#FFEB3B; }
        """)
        layout = QVBoxLayout(dlg); layout.setSpacing(12); layout.setContentsMargins(16, 16, 16, 16)
        info = QLabel(
            'Pick a new topology per axle.  Corner-HP coordinates and '
            'vehicle dimensions are kept; topology-specific hardware '
            '(ARB block / heave bracket / decoupled bellcranks) is '
            'replaced with the new defaults so geometry is consistent.\n'
            'After applying, you can nudge any hardpoint in Direct Edit.')
        info.setStyleSheet('color:#8a8a92; font-size:11px;')
        info.setWordWrap(True)
        layout.addWidget(info)

        cur = self._topology if hasattr(self, '_topology') else SuspensionTopology.standard()
        editor_front = _AxleTopologyEditor('FRONT AXLE', cur.front)
        editor_rear  = _AxleTopologyEditor('REAR AXLE',  cur.rear)
        layout.addWidget(editor_front)
        layout.addWidget(editor_rear)
        layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_cancel = QPushButton('Cancel'); btn_cancel.clicked.connect(dlg.reject)
        btn_apply  = QPushButton('Apply');  btn_apply.setObjectName('primary')
        btn_apply.setDefault(True); btn_apply.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_cancel); btn_row.addStretch(1); btn_row.addWidget(btn_apply)
        layout.addLayout(btn_row)

        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        new_topo = SuspensionTopology(
            front=editor_front.axle_topology(),
            rear=editor_rear.axle_topology(),
        )
        v = new_topo.validate()
        if v['front'] or v['rear']:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Invalid topology',
                                'Front: ' + '; '.join(v['front']) +
                                '\nRear: ' + '; '.join(v['rear']))
            return
        # Apply — no dimensions arg so wizard-time scaling/shifts don't
        # re-run.  Corner HPs are repopulated from per-topology defaults;
        # if the user wants them preserved against this change, that's a
        # bigger feature (default-vs-edited tracking) — for now they
        # accept that switching topology resets corner HPs.
        self.set_topology(new_topo)
        self.statusBar().showMessage(
            f'Topology changed to: {new_topo.describe()}', 5000)

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
            'version': 3,
            'front_hp':  {k: v.tolist() for k, v in self._front_hp.items()},
            'rear_hp':   {k: v.tolist() for k, v in self._rear_hp.items()},
            'front_arb': {k: v.tolist() for k, v in self._front_arb.items()},
            'rear_arb':  {k: v.tolist() for k, v in self._rear_arb.items()},
            # Topology-conditional per-axle extras — empty {} when unused
            'front_heave':     {k: v.tolist() for k, v in self._front_heave.items()},
            'rear_heave':      {k: v.tolist() for k, v in self._rear_heave.items()},
            'front_decoupled': {k: v.tolist() for k, v in self._front_decoupled.items()},
            'rear_decoupled':  {k: v.tolist() for k, v in self._rear_decoupled.items()},
            'car':       self._car.copy(),
            'steer':     self._steer.copy(),
            'alignment': self._alignment.copy(),
            'topology':  self._topology.to_dict(),
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
            self._load_project_from_path(path)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    def set_topology(self, topology, dimensions: dict | None = None):
        """Apply a new SuspensionTopology to this MainWindow.

        Replaces stored hardpoints with the appropriate default set for the
        chosen topology, then rebuilds the solvers and refreshes the 3D view.

        ``dimensions`` (optional) is the dict from the startup wizard's
        Vehicle-Dimensions editor:
            wheelbase_mm, track_f_mm, track_r_mm,
            rack_length_mm, total_rack_travel_mm
        When supplied, these values are written into ``self._car`` /
        ``self._steer`` AND the default ``tie_rod_inner`` X is overridden
        to ``±rack_length_mm/2`` on the front corner so the rack endpoint
        never has to be entered twice (poka-yoke).

        Implementation status (this session):
            DIRECT damper          — full (separate hardpoint set, solver path)
            PUSHROD                — full (Vahan baseline)
            PULLROD                — full (kinematics identical to pushrod;
                                      uses a low-rocker hardpoint default)
            BELLCRANK ARB          — full (Vahan baseline)
            CONTROL_ARM ARB        — full (drop link to LCA, body-frame
                                      tracked; MR via _arb_drop_top_world).
                                      Adjust `arb_drop_top` to a sensible
                                      point on the LCA span (X ≈ half-track).
            T-bar ARB              — full at the kinematic level.  Treats the
                                      bar as a centre-pivoted torsion element
                                      with the same drop-link-to-rocker
                                      constraint as a bellcrank U-bar.
                                      Roll stiffness uses the bellcrank
                                      lever/arm math; if your T-bar has a
                                      non-lateral torsion axis, model it
                                      explicitly in the bar-stiffness input
                                      rather than via _solve_arb_bellcrank.
            CORNER springs         — full (Vahan baseline)

        Designed but NOT implemented (next session):
            HEAVE_TBAR (3rd element on T-bar pivot mount)
                Physical model: the T-bar pivot bracket is mounted to a
                vertical-sliding bracket on the chassis; a 3rd-element
                coilover sits between that bracket and the chassis.
                In HEAVE both drop links push the bracket up equally, so
                the bracket translates (heave spring compresses).  In
                ROLL the drop links twist the bar without translating the
                bracket, so the heave spring sees zero load.
                Solver work needed: extend ARB DOFs from 1 (rotation θ) to
                2 (θ + bracket Z translation z_b), with two constraints
                (drop-link lengths from each side).  Heave force from drop
                links is reacted by K_heave_spring at the bracket.

            DECOUPLED (twin rockers per axle: flat heave + angled roll)
                Each pushrod drives its corner rocker as today.  Each corner
                rocker has TWO output points: one feeds a centre 'heave
                rocker' (flat, sums the two motions = heave), one feeds a
                centre 'roll rocker' (angled, differences them = roll).
                Two separate springs.  Corner springs are removed.
                Solver work needed: replace the existing single-rocker
                1-D Newton-Raphson with a coupled solve over four rockers
                (two corner + two centre), six unknowns total, six length
                constraints (4 pushrod-to-corner + 2 corner-to-centre).
                Dynamics also needs to know that heave and roll modes see
                different K_spring values.
        """
        from vahan.topology import (SuspensionTopology, DamperActuation,
                                    ARBType, SpringConfig)
        self._topology = (topology if topology is not None
                          else SuspensionTopology.standard())

        # ── Corner hardpoint defaults per damper actuation + spring config ─
        # Corner-rocker keys are MEANINGLESS on a central-mechanism axle
        # (heave-T-bar / plain T-bar via the central bellcrank): the rocker +
        # coil live on the central block, so leaving these in the corner dict
        # would render phantom chassis markers for hardware that isn't there.
        _CORNER_ROCKER_KEYS = ('pushrod_inner', 'rocker_pivot', 'rocker_axis_pt',
                               'rocker_spring_pt', 'spring_chassis_pt')

        def _corner_hp_for(axle_top, set_pushrod, set_pullrod,
                           set_direct, set_tbar, set_decoupled):
            """Pick the right corner-HP default set for this axle."""
            if axle_top.spring_config == SpringConfig.DECOUPLED:
                return {k: v.copy() for k, v in set_decoupled.items()}
            if axle_top.damper_actuation == DamperActuation.DIRECT:
                return {k: v.copy() for k, v in set_direct.items()}
            if axle_top.damper_actuation == DamperActuation.PULLROD:
                base = set_pullrod
            elif axle_top.arb_type == ARBType.TBAR:
                base = set_tbar
            else:
                base = set_pushrod
            # Central-mechanism axle (pushrod/pullrod feeding the central
            # T-bar bellcranks): strip the per-corner rocker chain.
            if (axle_top.spring_config == SpringConfig.HEAVE_TBAR
                    or axle_top.arb_type == ARBType.TBAR):
                return {k: v.copy() for k, v in base.items()
                        if k not in _CORNER_ROCKER_KEYS}
            return {k: v.copy() for k, v in base.items()}

        self._front_hp = _corner_hp_for(
            self._topology.front,
            DEFAULT_FRONT_HP, DEFAULT_FRONT_HP_PULLROD,
            DEFAULT_FRONT_HP_DIRECT, DEFAULT_FRONT_HP_TBAR,
            DEFAULT_FRONT_HP_DECOUPLED)
        self._rear_hp = _corner_hp_for(
            self._topology.rear,
            DEFAULT_REAR_HP, DEFAULT_REAR_HP_PULLROD,
            DEFAULT_REAR_HP_DIRECT, DEFAULT_REAR_HP_TBAR,
            DEFAULT_REAR_HP_DECOUPLED)

        # NOTE: wizard dimensions are applied LATER, after every per-axle
        # hardware dict (ARB / T-bar / heave / decoupled cradle) has been
        # populated.  Earlier we applied them here — but then _shift_axle_y
        # ran against empty hardware dicts and the topology code below
        # overwrote them with un-shifted hard-coded defaults, leaving the
        # rear ARB / heave bracket / cradle visibly stuck at Y=1.5 m
        # while the rest of the rear suspension moved with the wheelbase.

        # ── Per-axle hardware: ARB / T-bar / Heave / Decoupled cradle ────
        # These live in self._front_arb / self._rear_arb (legacy name —
        # actually holds any anti-roll device hardpoints).  For T-bar +
        # heave we also store the heave-element HPs in a separate dict.
        # Decoupled topology stores the central cradle HPs in
        # self._front_decoupled / self._rear_decoupled.
        def _uses_central_tbar(axle_top):
            """ONE MECHANISM (user-confirmed): "t bar is just heave t bar
            without 3rd element."  A pushrod/pullrod CORNER-spring axle with a
            T-bar ARB uses the SAME central bellcrank -> drop-link -> T-bar
            block as HEAVE_TBAR, just without the 3rd spring.  DIRECT axles
            have no pushrod to feed the central bellcrank, and DECOUPLED axles
            already have their own central cradle — both keep their existing
            models."""
            return (axle_top.arb_type == ARBType.TBAR
                    and axle_top.spring_config == SpringConfig.CORNER
                    and axle_top.damper_actuation in (DamperActuation.PUSHROD,
                                                      DamperActuation.PULLROD))

        def _arb_hp_for(axle_top, set_bellcrank, set_control_arm, set_tbar):
            if axle_top.arb_type == ARBType.NONE:
                return {}
            if axle_top.arb_type == ARBType.CONTROL_ARM:
                return {k: v.copy() for k, v in set_control_arm.items()}
            if axle_top.arb_type == ARBType.TBAR:
                if _uses_central_tbar(axle_top):
                    return {}   # modelled by the central T-bar dict below
                return {k: v.copy() for k, v in set_tbar.items()}
            return {k: v.copy() for k, v in set_bellcrank.items()}

        self._front_arb = _arb_hp_for(
            self._topology.front,
            DEFAULT_FRONT_ARB, DEFAULT_FRONT_ARB_CONTROL_ARM, DEFAULT_FRONT_TBAR)
        self._rear_arb = _arb_hp_for(
            self._topology.rear,
            DEFAULT_REAR_ARB, DEFAULT_REAR_ARB_CONTROL_ARM, DEFAULT_REAR_TBAR)

        # Central T-bar mechanism dict (the htb_ block).  Populated for:
        #   * HEAVE_TBAR axles            -> full set incl. 3rd spring
        #   * plain T-bar ARB axles       -> same set MINUS the 3rd-spring +
        #     (pushrod/pullrod + CORNER)     bracket keys (no 3rd element)
        _THIRD_SPRING_KEYS = ('heave_spring_tbar_pt', 'heave_spring_chassis_pt',
                              'heave_bracket_hinge_l', 'heave_bracket_hinge_r')

        def _central_tbar_for(axle_top, defaults):
            if axle_top.spring_config == SpringConfig.HEAVE_TBAR:
                return {k: v.copy() for k, v in defaults.items()}
            if _uses_central_tbar(axle_top):
                return {k: v.copy() for k, v in defaults.items()
                        if k not in _THIRD_SPRING_KEYS}
            return {}

        self._front_heave = _central_tbar_for(self._topology.front,
                                              DEFAULT_FRONT_HEAVE)
        self._rear_heave = _central_tbar_for(self._topology.rear,
                                             DEFAULT_REAR_HEAVE)

        # Decoupled central cradle (only meaningful with DECOUPLED spring config)
        self._front_decoupled = (
            {k: v.copy() for k, v in DEFAULT_FRONT_DECOUPLED.items()}
            if self._topology.front.spring_config == SpringConfig.DECOUPLED
            else {}
        )
        self._rear_decoupled = (
            {k: v.copy() for k, v in DEFAULT_REAR_DECOUPLED.items()}
            if self._topology.rear.spring_config == SpringConfig.DECOUPLED
            else {}
        )

        # ── Apply vehicle dimensions NOW, after every hardware dict is
        # populated.  If the caller supplied `dimensions` (e.g. wizard at
        # startup), use those.  Otherwise — for example when the user
        # changes topology via File -> Change Topology AFTER the wizard
        # has already run — fall back to the current self._car / self._steer
        # values so the track / wheelbase / rack-length the user set in
        # the wizard aren't silently reverted to legacy defaults by the
        # per-topology hardpoint repopulation above.
        effective_dims = dimensions
        if effective_dims is None and hasattr(self, '_car'):
            effective_dims = {
                'wheelbase_mm':   float(self._car.get('wheelbase_mm', 1536.7)),
                'track_f_mm':     float(self._car.get('track_f_mm',   1117.6)),
                'track_r_mm':     float(self._car.get('track_r_mm',   1117.6)),
                'rack_length_mm': float(self._car.get('rack_length_mm', 438.16)),
                'total_rack_travel_mm':
                    float(self._steer.get('total_rack_travel_mm', 100.0)),
                'spring_od_mm':   float(self._car.get('spring_od_mm', 63.0)),
                'damper_od_mm':   float(self._car.get('damper_od_mm', 50.0)),
            }
        if effective_dims:
            self._apply_wizard_dimensions(effective_dims)

        # Guarantee the actuation chain is coplanar for EVERY topology — done
        # LAST, after dimension scaling (which moves outboard X and would
        # otherwise knock pushrod_outer off a tilted rocker plane, as on the
        # T-bar).  Projects the stray pushrod / spring / damper points back
        # onto the rocker-plate plane so the mechanism is planar.
        self._enforce_actuation_coplanar(self._front_hp)
        self._enforce_actuation_coplanar(self._rear_hp)

        # ── Twin-rocker DECOUPLED solver — built only when needed ────────
        # 2-DOF (theta_L, theta_R) Newton-Raphson solver that takes both
        # wheels' pushrod_outer positions and returns the two rocker
        # angles + cross-car heave and roll damper compression deltas.
        # Stored on the window so dynamics / plotting code can re-solve
        # at any wheel position without rebuilding it from scratch.
        # NOTE: these cached handles are a convenience snapshot only.  Every
        # live consumer (3D view, kinematic graph, dynamics MR, load path) now
        # rebuilds via self._decoupled_solver() so an edit is never masked by a
        # stale cache.  We still populate them here for any incidental reference.
        self._front_cradle_solver = self._decoupled_solver(True)
        self._rear_cradle_solver  = self._decoupled_solver(False)

        # ── Warn on topologies whose kinematics aren't fully wired yet ───
        warnings = []
        for axle_name, axle_top in (('Front', self._topology.front),
                                    ('Rear',  self._topology.rear)):
            if axle_top.spring_config == SpringConfig.HEAVE_TBAR:
                warnings.append(f'{axle_name} heave-spring 3rd element: '
                                'kinematic stub only — solver does not yet '
                                'model the T-bar pivot translation that '
                                'compresses the 3rd-element spring.')
            if axle_top.spring_config == SpringConfig.DECOUPLED:
                warnings.append(f'{axle_name} decoupled twin-rocker: '
                                'placeholder only — kinematic + dynamics '
                                'solver work pending.')

        # ── Refresh panels + solvers + scene ────────────────────────────
        try:
            self._front_hp_panel.refresh(self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
            self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)
            # Refresh the direct-edit panel's hp list — include the per-axle
            # hardware so the user can nudge those points too (T-bar, heave
            # bracket, decoupled cradle, ARB etc.).
            self._refresh_hp_names()
            # Filter the plane-tilt pivot dropdown to this topology's pivots
            # (DIRECT -> damper endpoints, pushrod -> rocker/pushrod, etc.).
            self._refresh_plane_pivots()
        except Exception:
            pass
        # Topology-gate the dynamics panel: show/hide/relabel the spring + bar
        # inputs so the panel presents exactly the active topology's elements
        # (single-model GUI — not the same fields for every car).
        try:
            self._dynamics_panel.apply_topology(self._topology)
        except Exception:
            pass
        # Topology-filter the kinematic graph picker likewise: each flavour
        # lists the metrics of ITS mechanism (decoupled heave/roll springs,
        # T-bar twist + corner coil, legacy ARB chain only on standard cars).
        try:
            self._graph_panel.set_topology_flavors(self._axle_flavors())
        except Exception:
            pass
        # ...and the Live Values table (it lists CATALOG rows too).
        try:
            self._values_panel.set_topology_flavors(self._axle_flavors())
        except Exception:
            pass

        # Freeze cache must NOT leak across topologies: a state cached under the
        # previous topology (e.g. a pushrod corner) is geometrically wrong for
        # the new one, and the spring-limit freeze would resurrect it if the new
        # topology's first solve is out of bounds.  Clear so each topology starts
        # its freeze history fresh.
        self._last_valid_st.clear()
        self._rebuild_solvers()
        self._update_3d()

        msg = f'Topology set: {self._topology.describe()}'
        if warnings:
            msg += '  |  partial: ' + '; '.join(warnings)
        self.statusBar().showMessage(msg, 8000)

    # ── Poka-yoke: disclaimers on results that may be unreliable ────────
    def _check_dynamics_disclaimers(self, veh, steady_result=None) -> list[str]:
        """Return a list of human-readable warning strings about the
        current dynamics setup.  Empty if everything looks healthy.

        Called by the dynamics readout to surface conditions where the
        solver's assumptions break or where the user clearly hasn't
        configured topology-specific inputs.

        Categories of warning:
          * UNSET topology-specific inputs (DECOUPLED w/o heave/roll
            rates, HEAVE_TBAR w/o 3rd-element rate).
          * UNPHYSICAL results (inner wheel Fz < 0, K_eff < 0).
          * MMM yaw-equilibrium NOT iterated -- flag near-limit.
          * Lateral g exceeds typical tyre grip envelope (~1.8 g for FSAE).
        """
        warns: list[str] = []
        topo = getattr(self, '_topology', None)
        if topo is None:
            return warns

        from vahan.topology import SpringConfig

        # 1. DECOUPLED with zero damper rates
        for axle_name, axle, mode_key in (
            ('front', topo.front, 'topology_mode_front'),
            ('rear',  topo.rear,  'topology_mode_rear'),
        ):
            if axle.spring_config == SpringConfig.DECOUPLED:
                heave_rate = getattr(veh, f'decoupled_heave_rate_{axle_name}_Npm', 0.0)
                roll_rate  = getattr(veh, f'decoupled_roll_rate_{axle_name}_Npm', 0.0)
                if heave_rate <= 0 or roll_rate <= 0:
                    warns.append(
                        f'[DECOUPLED {axle_name.upper()}] heave_rate={heave_rate:.0f}, '
                        f'roll_rate={roll_rate:.0f} N/m -- set both [DECOUPLED] '
                        f'fields in Dynamics panel, otherwise wheel/roll rates '
                        f'fall back to corner-spring formula (wrong for this topology)')
            if axle.spring_config == SpringConfig.HEAVE_TBAR:
                rate_3rd = getattr(veh, f'heave_3rd_rate_{axle_name}_Npm', 0.0)
                if rate_3rd <= 0:
                    warns.append(
                        f'[HEAVE_TBAR {axle_name.upper()}] 3rd-elem rate=0 -- '
                        f'topology behaves as STANDARD until you set the '
                        f'[HEAVE_TBAR] 3rd-elem rate in Dynamics panel')

        # 2. UNPHYSICAL results from the latest steady-state solve
        if steady_result is not None:
            try:
                Fz = getattr(steady_result, 'Fz', None)
                if isinstance(Fz, dict):
                    min_fz = min(Fz.values())
                    if min_fz < 0:
                        warns.append(
                            f'INNER WHEEL LIFT: min Fz = {min_fz:.0f} N (negative) -- '
                            f'lateral g exceeds 2-wheel grip envelope; results '
                            f'beyond this g are not trustworthy')
            except Exception:
                pass

        # 3. K_eff < 0 in roll computation (gravity overturns spring restoring)
        try:
            G_MS2 = 9.81
            K_total = veh.roll_stiffness_total_Npm_rad
            # h_arm ~ sprung_cg_height_m - mean RC.  Use design RC = 0.05 as proxy.
            h_arm_est = veh.sprung_cg_height_m - 0.05
            K_grav = veh.sprung_mass_kg * G_MS2 * h_arm_est
            if K_grav > K_total:
                warns.append(
                    f'ROLL INSTABILITY: gravity overturning moment '
                    f'({K_grav:.0f} N*m/rad) exceeds total roll stiffness '
                    f'({K_total:.0f} N*m/rad) -- static roll diverges; '
                    f'stiffen springs/ARB or lower CG')
        except Exception:
            pass

        # 4. Lateral g requested above typical tyre envelope
        if steady_result is not None:
            ay = abs(getattr(steady_result, 'lateral_g', 0.0))
            if ay > 1.8:
                warns.append(
                    f'HIGH LATERAL G ({ay:.2f} g): above typical FSAE tyre '
                    f'envelope (~1.5-1.8 g).  Per-tyre friction-circle clamp '
                    f'still bounds the result, but yaw-equilibrium iteration '
                    f'(MMM-style trim) is NOT performed -- absolute limit may '
                    f'be lower if setup is unbalanced')

        return warns

    # ── Topology → dynamics-params injection ────────────────────────────
    def _apply_topology_to_dyn_params(self, dyn_params: dict) -> dict:
        """Inject topology-specific dynamics fields into a dyn_params dict.

        Standard topology: no-op (the corner-spring×MR² wheel-rate
        formula is the default).

        HEAVE_TBAR: marks the axle so the heave 3rd element is added to
        ride stiffness (corner spring + ½·K_3rd·MR_3rd²) and NOT to roll
        stiffness.  Currently we use ``spring_rate_*_Npm`` as the corner
        rate and leave ``heave_3rd_rate_*`` at zero unless the user has
        injected it manually — once the heave bracket kinematic solver
        lands we'll auto-compute MR_3rd from the bracket geometry too.

        DECOUPLED: replaces the corner-spring formula entirely.  Pulls
        geometric ratios MR_heave and MR_roll from the twin-rocker
        cradle solver (perturbs each wheel by 1 mm and measures the
        damper compression delta), and reuses the existing
        spring_rate_*_Npm field as the HEAVE damper rate and
        arb_rate_*_Npm as the ROLL damper rate.  That's a pragmatic
        re-purposing of existing UI controls — the user can read
        "front spring rate" as "front heave damper rate" when the
        front axle is decoupled, and similarly "front ARB rate" →
        "front roll damper rate at the wheel".
        """
        from vahan.topology import SpringConfig, ARBType
        topo = getattr(self, '_topology', None)
        if topo is None:
            return dyn_params

        out = dict(dyn_params)
        import numpy as np

        def _per_axle(is_front: bool):
            axle = topo.front if is_front else topo.rear
            mode = 'standard'
            if axle.spring_config == SpringConfig.HEAVE_TBAR:
                mode = 'heave_tbar'
            elif axle.spring_config == SpringConfig.DECOUPLED:
                mode = 'decoupled'
            suffix = 'front' if is_front else 'rear'
            out[f'topology_mode_{suffix}'] = mode

            # DECOUPLED — compute MRs from the cradle solver and reuse
            # spring_rate / arb_rate fields as heave / roll damper rates.
            if mode == 'decoupled':
                # ONE MODEL: drive the cradle from the REAL pushrod_outer motion
                # the corner (wishbone) solve produces — the SAME input the
                # kinematic graph uses — not a 1:1 vertical perturbation that
                # ignored the wishbone->pushrod ratio (was ~2x off).  Build the
                # cradle FRESH from current geometry so edits propagate.
                corner  = self._solvers.get('FL' if is_front else 'RL')
                slv     = self._decoupled_solver(is_front)
                if slv is None or corner is None:
                    return
                def _mir(p):
                    q = np.asarray(p, float).copy(); q[0] *= -1.0; return q
                try:
                    h = 3e-3
                    po_p = np.asarray(corner.solve(+h).pushrod_outer, float)
                    po_m = np.asarray(corner.solve(-h).pushrod_outer, float)
                    # heave: both wheels +h (symmetric); roll: opposite
                    MR_heave = abs((slv.solve(po_p, _mir(po_p)).heave_delta
                                    - slv.solve(po_m, _mir(po_m)).heave_delta) / (2*h))
                    MR_roll  = abs((slv.solve(po_p, _mir(po_m)).roll_delta
                                    - slv.solve(po_m, _mir(po_p)).roll_delta) / (2*h))
                except Exception:
                    return
                # DECOUPLED rates come ONLY from the dedicated heave/roll
                # spring-rate fields (the panel shows exactly these for a
                # decoupled axle and seeds non-zero defaults on topology
                # switch).  No silent fallback to the corner spring / ARB
                # fields — that would be a second hidden model.  A zero rate
                # is surfaced as a status-bar warning instead.
                k_heave_field = float(out.get(f'decoupled_heave_rate_{suffix}_Npm', 0.0))
                k_roll_field  = float(out.get(f'decoupled_roll_rate_{suffix}_Npm', 0.0))
                if k_heave_field <= 0.0 or k_roll_field <= 0.0:
                    try:
                        self.statusBar().showMessage(
                            f'DECOUPLED {suffix}: heave/roll spring rate is 0 — '
                            'set it in the Dynamics panel (axle has no corner '
                            'spring to fall back on)', 8000)
                    except Exception:
                        pass
                # MR is ALWAYS from the kinematic solver -- never reused
                # from a redundant field; the cradle geometry is the only
                # source of truth for the geometric ratio.
                out[f'decoupled_heave_rate_{suffix}_Npm'] = k_heave_field
                out[f'decoupled_heave_MR_{suffix}']       = float(abs(MR_heave))
                out[f'decoupled_roll_rate_{suffix}_Npm']  = k_roll_field
                out[f'decoupled_roll_MR_{suffix}']        = float(abs(MR_roll))

            # HEAVE_TBAR -- compute the 3rd-element MR geometrically by
            # asking the HeaveTBarSolver "how far does the heave spring
            # compress when the bracket rotates by 1 deg?", combined
            # with an estimate of "how many degrees of bracket rotation
            # per mm of symmetric wheel travel" derived from the T-bar
            # arm geometry.
            #
            # The product is:
            #   MR_3rd  =  d(spring_comp) / d(bracket_theta)
            #            *  d(bracket_theta) / d(wheel_z_symmetric)
            #
            # The first factor is solver.mr_curve()[1] - solver.mr_curve()[0].
            # The second factor we approximate from the geometry: the
            # average T-bar arm length (drop_top distance from pivot) and
            # the corner rocker MR.  For ride-rate purposes this is a
            # first-order estimate; users can override by typing the MR
            # directly into the dynamics panel once that UI lands.
            if mode == 'heave_tbar':
                # ONE MODEL: the heave 3rd-spring MR comes from the SAME T-bar
                # solver the kinematic graph uses (HeaveTBarRockerSolver,
                # vahan/heave_tbar.py), evaluated at design ride from the corner
                # solve's pushrod_outer — so the dynamics number is exactly what
                # the graph plots.  No separate/convoluted formula.
                heave_dict = self._front_heave if is_front else self._rear_heave
                hp_dict    = self._front_hp    if is_front else self._rear_hp
                corner     = self._solvers.get('FL' if is_front else 'RL')
                po_design  = hp_dict.get('pushrod_outer') if hp_dict else None
                mr3 = 0.0
                if (heave_dict and corner is not None and po_design is not None
                        and all(k in heave_dict for k in self._HTB_KEYMAP)):
                    try:
                        from vahan.heave_tbar import HeaveTBarRockerSolver
                        geo = {self._HTB_KEYMAP[k]: np.asarray(heave_dict[k], float)
                               for k in self._HTB_KEYMAP}
                        s = HeaveTBarRockerSolver(geo, po_design)
                        h = 0.003
                        Lp = s.heave(np.asarray(corner.solve(+h).pushrod_outer, float)).heave_spring_length
                        Lm = s.heave(np.asarray(corner.solve(-h).pushrod_outer, float)).heave_spring_length
                        mr3 = abs((Lp - Lm) / (2.0 * h))
                    except Exception:
                        mr3 = 0.0
                out[f'heave_3rd_MR_{suffix}'] = float(mr3)
                # Single-model guard: a HEAVE_TBAR axle whose 3rd-element rate
                # is still 0 silently degrades to corner-coils-only dynamics.
                # Surface it (the panel seeds a default, so this only fires on
                # old projects / hand-zeroed fields).
                if float(out.get(f'heave_3rd_rate_{suffix}_Npm', 0.0)) <= 0.0:
                    try:
                        self.statusBar().showMessage(
                            f'HEAVE_TBAR {suffix}: 3rd-element spring rate is 0 '
                            '— heave stiffness is corner coils only; set the '
                            '3rd (heave) spring rate in the Dynamics panel', 8000)
                    except Exception:
                        pass

        _per_axle(is_front=True)
        _per_axle(is_front=False)

        # ── RCVD item 3: geometric IR-rate correction ──────────────────
        # K_wheel = Fs * (dIR/d_delta) + K_s * IR^2.  Compute Fs (static
        # spring force = corner load / MR_static) and dIR/d_delta (MR
        # slope vs wheel travel) from the kinematic sweep results.  These
        # populate mr_slope_*_per_m and static_spring_force_*_N which the
        # VehicleParams._wheel_rate_for_axle method consumes.
        try:
            sweep = getattr(self, '_sweep_results', {}) or {}
            for label, suffix, fl_key, n_unsprung_key in (
                ('FL', 'front', 'unsprung_mass_front_kg', 'unsprung_mass_front_kg'),
                ('RL', 'rear',  'unsprung_mass_rear_kg',  'unsprung_mass_rear_kg'),
            ):
                fl = sweep.get(label, {})
                mr_arr = np.asarray(fl.get('motion_ratio', []), dtype=float)
                wc_z = np.asarray(fl.get('wc_z', []), dtype=float)
                finite = np.isfinite(mr_arr) & np.isfinite(wc_z)
                if finite.sum() < 5:
                    continue
                # mr_slope = derivative at the middle of the sweep
                idx = np.where(finite)[0]
                # Take a few points around the centre for a robust slope.
                # Index INTO the finite list (idx) so lo/hi are guaranteed
                # finite even when the sweep has NaN points (e.g. travel
                # extremes that fail to solve) — otherwise mid±5 can land on
                # a NaN sample and poison mr_slope → roll_angle → the whole
                # dynamics solve returns NaN.
                pos = len(idx) // 2
                lo = int(idx[max(0, pos - 5)])
                hi = int(idx[min(len(idx) - 1, pos + 5)])
                if hi - lo < 2:
                    continue
                # mr_slope in 1/m: dMR per metre of wheel travel.
                # wc_z is in mm in the sweep -- convert to m.
                dz_m = (wc_z[hi] - wc_z[lo]) / 1000.0
                if abs(dz_m) < 1e-9:
                    continue
                mr_slope = (mr_arr[hi] - mr_arr[lo]) / dz_m
                # Static spring force per corner: F_corner / MR_at_design
                m_total = (out.get('sprung_mass_kg', 0.0)
                           + out.get('unsprung_mass_front_kg', 0.0)
                           + out.get('unsprung_mass_rear_kg', 0.0))
                if m_total < 1.0:
                    continue
                wf_frac = (1.0 - out.get('cg_to_front_axle_m', 0.845)
                                  / out.get('wheelbase_m', 1.530))
                if suffix == 'front':
                    corner_load_N = m_total * 9.81 * wf_frac / 2.0
                else:
                    corner_load_N = m_total * 9.81 * (1 - wf_frac) / 2.0
                mr_static = float(mr_arr[mid]) if abs(mr_arr[mid]) > 1e-4 else 1.0
                Fs = corner_load_N / mr_static
                out[f'mr_slope_{suffix}_per_m'] = float(mr_slope)
                out[f'static_spring_force_{suffix}_N'] = float(Fs)
        except Exception:
            # Silent failure -- the RCVD correction is small enough that
            # missing it is far better than crashing the dyn-params build.
            pass

        return out

    # ── Wizard → coordinate generation ──────────────────────────────────
    def _apply_wizard_dimensions(self, dims: dict) -> None:
        """Write wizard vehicle-dimension values into the live state.

        Inputs (mm):
            wheelbase_mm, track_f_mm, track_r_mm,
            rack_length_mm, total_rack_travel_mm

        Effects:
          * ``self._car``  ← wheelbase_mm, track_f_mm, track_r_mm,
                            axle_spacing_mm (mirrored to wheelbase),
                            rack_length_mm
          * ``self._steer`` ← total_rack_travel_mm
          * ``self._front_hp['tie_rod_inner'][0]``  ← +rack_length_mm/2/1000
            (FL side is +X outboard).
          * Front corner X coords (wheel_center, *_outer, tie_rod_outer)
            scaled so the half-track matches user input.  Inner pickups
            (uca_front/rear, lca_front/rear, tie_rod_inner) are NOT scaled
            — those are chassis pickups whose X position is independent
            of track.  tie_rod_inner is set explicitly from rack length.
          * Rear corner X coords: scaled the same way using track_r_mm.
          * Rear-axle Y shift: every rear-axle dict (corner HPs, ARB /
            T-bar, heave bracket, decoupled cradle) translates by the
            same Δy so the entire rear suspension subassembly moves to
            the new wheelbase.  Done via ``_shift_axle_y`` so the same
            code path handles wizard-time shifts AND CarPanel live
            edits to axle_spacing_mm.
        """
        # 1. Write into the persistent panel state dicts
        wb_mm   = float(dims.get('wheelbase_mm',   self._car.get('wheelbase_mm',   1537.)))
        tf_mm   = float(dims.get('track_f_mm',     self._car.get('track_f_mm',     1222.)))
        tr_mm   = float(dims.get('track_r_mm',     self._car.get('track_r_mm',     1200.)))
        rk_len  = float(dims.get('rack_length_mm', self._car.get('rack_length_mm', 438.16)))
        rk_trav = float(dims.get('total_rack_travel_mm',
                                  self._steer.get('total_rack_travel_mm', 100.)))

        self._car['wheelbase_mm']    = wb_mm
        self._car['axle_spacing_mm'] = wb_mm     # keep mirrors consistent
        self._car['track_f_mm']      = tf_mm
        self._car['track_r_mm']      = tr_mm
        self._car['rack_length_mm']  = rk_len
        self._steer['total_rack_travel_mm'] = rk_trav
        # Spring / damper OD (mm) — drive view3d render width
        if 'spring_od_mm' in dims:
            self._car['spring_od_mm'] = float(dims['spring_od_mm'])
        if 'damper_od_mm' in dims:
            self._car['damper_od_mm'] = float(dims['damper_od_mm'])

        # 2. Default track from the hard-coded HPs (wheel_center.X * 2).
        # We use this as the "before" reference for the scaling factor.
        try:
            default_track_f_mm = 2.0 * abs(DEFAULT_FRONT_HP['wheel_center'][0]) * 1000.0
            default_track_r_mm = 2.0 * abs(DEFAULT_REAR_HP['wheel_center'][0]) * 1000.0
        except Exception:
            default_track_f_mm = 1117.6  # 2 × 0.55880
            default_track_r_mm = 1117.6
        sx_f = tf_mm / default_track_f_mm if default_track_f_mm else 1.0
        sx_r = tr_mm / default_track_r_mm if default_track_r_mm else 1.0

        # 3. Default rear-axle Y reference: take it from the existing
        # wheel_center Y on the rear default (this is what "wheelbase = 1537"
        # encodes in metres).
        try:
            default_wb_m = float(DEFAULT_REAR_HP['wheel_center'][1])
        except Exception:
            default_wb_m = 1.537
        dy_m = (wb_mm / 1000.0) - default_wb_m

        # 4. Which corner-HP keys live on the OUTBOARD ring (scaled by track)
        # vs the chassis side (kept as-is, except tie_rod_inner which is
        # driven explicitly from rack length below)?
        OUTBOARD_KEYS = {
            'uca_outer', 'lca_outer', 'tie_rod_outer', 'wheel_center',
            'pushrod_outer', 'damper_outer_pt',
        }

        def _scale_corner_x(hp_dict: dict, sx: float):
            """Scale outboard X by sx in-place (no Y touch)."""
            for k, v in hp_dict.items():
                if k in OUTBOARD_KEYS:
                    arr = np.asarray(v, float).copy()
                    arr[0] *= sx
                    hp_dict[k] = arr

        # 5. Front: scale only X (front axle Y is the origin — Y stays 0)
        _scale_corner_x(self._front_hp, sx_f)
        # 6. Rear: scale X here, then shift Y for the WHOLE AXLE (corner +
        # ARB + heave + decoupled) via _shift_axle_y so every piece of
        # hardware moves together.  Old code only touched _rear_hp, leaving
        # the ARB / heave bracket / cradle stuck at the original hard-coded
        # Y — that's the "rear ARB hard-fixed when changing wheelbase" bug.
        _scale_corner_x(self._rear_hp, sx_r)
        self._shift_axle_y(is_front=False, dy_m=dy_m)

        # 7. Drive tie_rod_inner X from rack length (front only; rear has
        # no rack — its tie_rod_inner is a chassis toe-link pickup that
        # isn't related to a rack).  FL = +X outboard, so the LEFT rack
        # end is at +rack_length/2 in our coord convention.
        if 'tie_rod_inner' in self._front_hp:
            tri = self._front_hp['tie_rod_inner'].copy()
            tri[0] = (rk_len / 2.0) / 1000.0
            self._front_hp['tie_rod_inner'] = tri

        # 8. Push wheelbase/track values into the CarPanel widget if it
        # already exists, so the UI shows what the wizard captured.
        try:
            if hasattr(self, '_car_panel'):
                self._car_panel.set_params(self._car)
        except Exception:
            pass

    def _load_project_from_path(self, path: str):
        """Programmatic load (also used by the startup wizard).  Raises on error."""
        with open(path) as f:
            data = json.load(f)

        def _arr(d, key):
            return {k: np.array(v, float) for k, v in d[key].items()}

        self._front_hp  = _arr(data, 'front_hp')
        self._rear_hp   = _arr(data, 'rear_hp')
        self._front_arb = _arr(data, 'front_arb')
        self._rear_arb  = _arr(data, 'rear_arb')

        # Topology-conditional extras — missing-key safe (old v2 files have
        # no entries; they default to empty dicts, which set_topology will
        # repopulate from defaults if the loaded topology needs them).
        def _arr_safe(d, key):
            block = d.get(key, {})
            return {k: np.array(v, float) for k, v in block.items()}
        self._front_heave     = _arr_safe(data, 'front_heave')
        self._rear_heave      = _arr_safe(data, 'rear_heave')
        self._front_decoupled = _arr_safe(data, 'front_decoupled')
        self._rear_decoupled  = _arr_safe(data, 'rear_decoupled')

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
        # Backward compat: pre-wizard files have no rack_length_mm.
        # Fall back to the legacy hard-coded 2 x 219.08 mm = 438.16 mm.
        car_data.setdefault('rack_length_mm', 438.16)
        # Spring/damper OD added later; default to typical FSAE coil + damper.
        car_data.setdefault('spring_od_mm', 63.0)
        car_data.setdefault('damper_od_mm', 50.0)
        # Track-change behaviour (added with the inboard-pickup option).
        car_data.setdefault('track_pushes_inboard', False)
        # Rear driveshaft / differential packaging (added 2026-07; rear-only).
        # Diff default on the rear axle line, mid-height, centreline; tripod/
        # shaft dims are placeholders until real numbers land.
        car_data.setdefault('diff_long_mm', car_data.get('wheelbase_mm', 1537.))
        car_data.setdefault('diff_vert_mm', 150.)
        car_data.setdefault('diff_lateral_offset_mm', 0.)
        car_data.setdefault('diff_housing_width_mm', 292.1)   # inboard pivot spacing
        car_data.setdefault('tripod_od_mm', 90.)
        car_data.setdefault('driveshaft_dia_mm', 25.4)
        car_data.setdefault('rotor_dia_mm', 240.0)
        car_data.setdefault('show_driveshaft', True)
        car_data.setdefault('show_brakes', True)
        car_data.setdefault('show_shock_thickness', True)
        self._car.update(car_data)
        self._steer.update(data.get('steer', {}))
        self._alignment.update(data.get('alignment', {}))

        # Topology — present in files written by this version onwards;
        # absent in older files → fall back to the Vahan standard layout.
        from vahan.topology import SuspensionTopology
        topology_dict = data.get('topology')
        self._topology = SuspensionTopology.from_dict(topology_dict or {})

        self._front_hp_panel.refresh(self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
        self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)
        self._car_panel.set_params(self._car)
        try:
            self.view3d.sync_view_controls(
                view_mode=self._car.get('view_mode', 'normal'),
                perspective=self._car_panel._chk_perspective.isChecked(),
                floor=self._car.get('show_ground', True),
                thickness=self._car.get('show_shock_thickness', True))
        except Exception:
            pass

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

        # Re-apply the topology gating that set_topology() normally does —
        # loading sets self._topology DIRECTLY (so the loaded hardpoints are
        # kept, not re-defaulted), which previously left the dynamics panel,
        # graph picker and values table shaped like the PREVIOUS car, and the
        # 3D freeze cache holding the previous topology's geometry.
        # NOTE: apply_topology runs AFTER set_state so the gating reflects the
        # LOADED topology, but seeds only zero-valued rate fields (loaded
        # non-zero rates are untouched).
        try:
            self._dynamics_panel.apply_topology(self._topology)
        except Exception:
            pass
        try:
            self._graph_panel.set_topology_flavors(self._axle_flavors())
        except Exception:
            pass
        try:
            self._values_panel.set_topology_flavors(self._axle_flavors())
        except Exception:
            pass
        self._last_valid_st.clear()

        self._refresh_hp_names()   # every hardpoint (incl. ARB) editable
        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()
        self.statusBar().showMessage(
            f'Loaded: {path}  |  topology: {self._topology.describe()}', 5000)

    def _export_sweep_csv(self):
        """File → Export Sweep Data (CSV): write the current kinematic sweep
        (every metric × 4 corners vs the sweep axis) and, if one was run, the
        last dynamics sweep — for Excel / MATLAB post-processing."""
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Sweep Data (CSV)', '', 'CSV (*.csv)')
        if not path:
            return
        base = path[:-4] if path.lower().endswith('.csv') else path
        wrote = []
        try:
            res = getattr(self, '_sweep_results', None)
            x = np.asarray(getattr(self, '_x_arr', []), float)
            if res and len(x) > 2:
                corners = [c for c in ('FL', 'FR', 'RL', 'RR') if c in res]
                keys = sorted({
                    k for c in corners for k, v in res[c].items()
                    if isinstance(v, np.ndarray) and len(v) == len(x)
                    and not k.startswith('_')})
                kin_path = base + '_kinematics.csv'
                with open(kin_path, 'w', newline='') as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow([self._x_label]
                                  + [f'{k}_{c}' for k in keys for c in corners])
                    for i in range(len(x)):
                        row = [f'{x[i]:.6g}']
                        for k in keys:
                            for c in corners:
                                v = res[c].get(k)
                                ok = (isinstance(v, np.ndarray)
                                      and len(v) == len(x)
                                      and np.isfinite(v[i]))
                                row.append(f'{float(v[i]):.6g}' if ok else '')
                        wcsv.writerow(row)
                wrote.append(kin_path)

            dyn = getattr(self, '_dyn_sweep_data', None)
            if dyn:
                xkey = ('time_s' if 'time_s' in dyn
                        else 'lateral_g' if 'lateral_g' in dyn
                        else 'longitudinal_g')
                xd = np.asarray(dyn.get(xkey, []), float)
                if len(xd) > 2:
                    dkeys = sorted(
                        k for k, v in dyn.items()
                        if isinstance(v, np.ndarray) and len(v) == len(xd)
                        and k != xkey)
                    dyn_path = base + '_dynamics.csv'
                    with open(dyn_path, 'w', newline='') as f:
                        wcsv = csv.writer(f)
                        wcsv.writerow([xkey] + dkeys)
                        for i in range(len(xd)):
                            row = [f'{xd[i]:.6g}']
                            for k in dkeys:
                                vi = float(dyn[k][i])
                                row.append(f'{vi:.6g}' if np.isfinite(vi) else '')
                            wcsv.writerow(row)
                    wrote.append(dyn_path)
        except Exception as e:
            QMessageBox.critical(self, 'Export CSV', f'Export failed: {e}')
            return
        if wrote:
            self.statusBar().showMessage(
                'Exported: ' + '  |  '.join(wrote), 8000)
        else:
            QMessageBox.information(
                self, 'Export CSV',
                'Nothing to export yet — run a kinematic sweep (it runs '
                'automatically) or a dynamics Sweep first.')

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

    def _switch_page(self, idx: int):
        """Page menu: 0 = Suspension, 1 = Laptime, 2 = Design City.  Pages 1
        and 2 are built lazily on first use."""
        if idx >= 1 and self._pages.count() < 2:
            try:
                from gui.laptime_page import LaptimePage
                self._laptime_page = LaptimePage(self)
                self._pages.addWidget(self._laptime_page)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.statusBar().showMessage(f'Laptime page failed: {e}', 8000)
                return
        if idx >= 2 and self._pages.count() < 3:
            try:
                from gui.city_page import CityPage
                self._city_page = CityPage(self)
                self._pages.addWidget(self._city_page)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.statusBar().showMessage(f'Design City page failed: {e}', 8000)
                return
        if idx >= 3 and self._pages.count() < 4:
            try:
                from gui.loads_page import LoadsPage
                self._loads_page = LoadsPage(self)
                self._pages.addWidget(self._loads_page)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.statusBar().showMessage(f'Loads page failed: {e}', 8000)
                return
        if idx < self._pages.count():
            self._pages.setCurrentIndex(idx)
            self.statusBar().showMessage(
                ('Suspension', 'Laptime', 'Design City', 'Loads')[idx] + ' page', 2000)

    def _build_ui(self):
        self._build_menu()
        # ── PAGES: 0 = Suspension (everything below), 1 = Laptime ─────────
        # The whole legacy layout becomes page 0 of a stacked widget; the
        # Laptime page (track + QSS sim of the CURRENT car) is page 1.
        from PyQt6.QtWidgets import QStackedWidget
        self._pages = QStackedWidget()
        self.setCentralWidget(self._pages)
        central = QWidget()
        self._pages.addWidget(central)          # page 0 — Suspension
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 3D + curves
        self.view3d = View3D()
        self.view3d.set_on_pick(self._on_pick)
        self.view3d.set_on_move(self._on_hp_move)
        self.view3d.set_on_constraint(self._on_constraint_mode)
        self.view3d.set_on_increment(self._on_edit_increment_changed)

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
        self._front_hp_panel = HardpointPanel('Front Hardpoints', self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
        self._rear_hp_panel  = HardpointPanel('Rear Hardpoints',  self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)
        self._values_panel   = ValuesPanel()

        # 3D overlay toggles (collapsible)
        self._overlay_box = CollapsibleSection('3D Overlays', header_color='#cccccc')
        self._chk_rc   = QCheckBox('Roll Centres')
        self._chk_ra   = QCheckBox('Roll Axis')
        self._chk_pa   = QCheckBox('Pitch Axis')
        self._chk_cg   = QCheckBox('Centre of Gravity')
        self._chk_rc.setChecked(True)
        self._chk_ra.setChecked(True)
        self._chk_pa.setChecked(True)
        self._chk_cg.setChecked(True)
        self._overlay_box.add_widget(self._chk_rc)
        self._overlay_box.add_widget(self._chk_ra)
        self._overlay_box.add_widget(self._chk_pa)
        self._overlay_box.add_widget(self._chk_cg)
        self._chk_rc.toggled.connect(lambda v: (
            setattr(self, '_show_rc', v),         self._update_3d()))
        self._chk_ra.toggled.connect(lambda v: (
            setattr(self, '_show_roll_axis', v),  self._update_3d()))
        self._chk_pa.toggled.connect(lambda v: (
            setattr(self, '_show_pitch_axis', v), self._update_3d()))
        self._chk_cg.toggled.connect(lambda v: (
            setattr(self, '_show_cg', v),         self._update_3d()))

        from PyQt6.QtGui import QKeySequence

        # Ctrl+Y → redo the last undone hardpoint edit
        self._redo_action = QAction('Redo hardpoint edit', self)
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)  # Ctrl+Y
        self._redo_action.triggered.connect(self._redo_edit)
        self.addAction(self._redo_action)

        # Ctrl+Z → undo last hardpoint edit (only meaningful when in edit mode)
        self._undo_action = QAction('Undo Edit', self)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)  # Ctrl+Z
        self._undo_action.triggered.connect(self._undo_edit)
        self.addAction(self._undo_action)

        # Direct-edit panel (dedicated section in the sidebar)
        self._direct_edit_panel = DirectEditPanel()
        self._direct_edit_panel.enabled_changed.connect(self._on_toggle_edit_mode)
        self._direct_edit_panel.hp_selected.connect(self._on_panel_hp_selected)
        self._direct_edit_panel.step_changed.connect(self._on_panel_step_changed)
        self._direct_edit_panel.mirror_axle_changed.connect(
            self._on_panel_mirror_changed)
        self._direct_edit_panel.apply_clicked.connect(self._on_edit_apply)
        self._direct_edit_panel.discard_clicked.connect(self._on_edit_discard)
        self._direct_edit_panel.plane_tilt_requested.connect(self._on_plane_tilt)
        self._direct_edit_panel.plane_axle_changed.connect(self._refresh_plane_pivots)
        self._direct_edit_panel.group_move_requested.connect(self._on_group_move)
        self._direct_edit_panel.position_typed.connect(self._on_position_typed)
        self._direct_edit_panel.constraint_mode_changed.connect(
            self._on_constraint_mode)
        self._direct_edit_panel.baseline_set_requested.connect(self._set_baseline)
        self._direct_edit_panel.ghost_toggled.connect(self._on_ghost_toggled)
        self._direct_edit_panel.snap_axis_to_normal_requested.connect(
            self._on_snap_axis_to_normal)
        self._direct_edit_panel.snap_actuation_to_plane_requested.connect(
            self._on_snap_actuation_to_plane)
        # Frame / interference check — its OWN always-visible panel (placed high
        # in the left sidebar) so the thickness toggle is never buried.
        self._frame_panel = FrameInterferencePanel()
        self._frame_panel.frame_changed.connect(self._update_3d)
        self._direct_edit_panel.rack_length_changed.connect(self._on_rack_length)
        self._direct_edit_panel.shock_length_changed.connect(self._on_shock_length)
        self._mirror_to_other_axle = False

        # Left sidebar (existing controls)
        sidebar_inner = QWidget()
        sv = QVBoxLayout(sidebar_inner)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(4)
        for w in [self._motion_panel, self._steer_panel, self._alignment_panel,
                  self._car_panel, self._frame_panel, self._overlay_box,
                  self._graph_panel, self._front_hp_panel, self._rear_hp_panel,
                  self._values_panel, self._direct_edit_panel]:
            sv.addWidget(w)
        self._refresh_hp_names()   # list every hardpoint (incl. ARB) at startup

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
        self._vehicle_constants_panel = VehicleConstantsPanel()
        self._analysis_plots_panel = AnalysisPlotsPanel()
        ik_inner = QWidget()
        ik_layout = QVBoxLayout(ik_inner)
        ik_layout.setContentsMargins(0, 0, 0, 0)
        ik_layout.setSpacing(4)
        ik_layout.addWidget(self._ik_panel)
        ik_layout.addWidget(self._dynamics_panel)
        ik_layout.addWidget(self._vehicle_constants_panel)
        ik_layout.addWidget(self._dynamics_opt_panel)
        ik_layout.addWidget(self._skidpad_panel)
        ik_layout.addWidget(self._aero_panel)
        ik_layout.addWidget(self._loads_panel)
        ik_layout.addWidget(self._brake_calc_panel)
        ik_layout.addWidget(self._analysis_plots_panel)
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
        self._car_panel.perspective_changed.connect(self.view3d.set_perspective)
        self.view3d.set_on_view_controls(self._on_view_controls_changed)
        self._front_hp_panel.hp_changed.connect(
            lambda d, cat: self._on_hp(d, 'front', cat))
        self._rear_hp_panel.hp_changed.connect(
            lambda d, cat: self._on_hp(d, 'rear', cat))
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
        self._dynamics_panel.tire_plots_requested.connect(self._on_tire_plots)
        # Live refresh of vehicle constants when ANY dynamics input changes
        # (spring rate, ARB geometry, mass etc.) — debounced 200 ms so a
        # spinbox spam during typing doesn't rebuild the solver per keystroke.
        self._constants_refresh_timer = QTimer()
        self._constants_refresh_timer.setSingleShot(True)
        self._constants_refresh_timer.setInterval(200)
        self._constants_refresh_timer.timeout.connect(self._refresh_vehicle_constants)
        self._dynamics_panel.params_changed.connect(
            lambda _p: self._constants_refresh_timer.start())
        # Live refresh of the kinematic GRAPHS when a hardpoint is edited (WASD
        # nudge / group-move).  Same ONE-MODEL principle as the 3D view: moving
        # a hardpoint must move every graph it drives, not just the 3D.  The 3D
        # is updated synchronously in the edit handler (fast); this debounced
        # 150 ms timer re-sweeps the graphs so a held key or a multi-point
        # group-move coalesces into a single sweep instead of one per keystroke.
        self._edit_sweep_timer = QTimer()
        self._edit_sweep_timer.setSingleShot(True)
        self._edit_sweep_timer.setInterval(150)
        self._edit_sweep_timer.timeout.connect(self._run_sweep)
        self._dynamics_panel.graph_selection_changed.connect(self._on_dyn_graph_sel)
        self._dynamics_panel.corners_changed.connect(self._on_dyn_corners_sel)
        self._dynamics_opt_panel.analyze_requested.connect(self._on_sensitivity_analyze)
        self._skidpad_panel.simulate_requested.connect(self._on_skidpad_simulate)
        self._skidpad_panel.signals_changed.connect(self._on_skidpad_signals)
        self._aero_panel.solve_requested.connect(self._on_aero_solve)
        self._aero_panel.sweep_requested.connect(self._on_aero_sweep)
        self._dynamics_panel.apply_aero_toggled.connect(self._on_apply_aero_toggle)
        self._loads_panel.loads_requested.connect(self._on_compute_loads)
        self._loads_panel.wheel_package_requested.connect(self._open_wheel_package)
        self._brake_calc_panel.compute_requested.connect(self._on_compute_brakes)
        # Analysis & validation plots
        self._analysis_plots_panel.brake_capacity_requested.connect(self._on_plot_brake_capacity)
        self._analysis_plots_panel.ride_freq_requested.connect(self._on_plot_ride_freq)
        self._analysis_plots_panel.rc_vs_roll_requested.connect(self._on_plot_rc_vs_roll)
        self._analysis_plots_panel.steering_torque_requested.connect(self._on_plot_steering_torque)
        self._analysis_plots_panel.ackermann_demand_requested.connect(self._on_plot_ackermann_demand)
        self._analysis_plots_panel.ackermann_fzfy_requested.connect(self._on_plot_ackermann_fzfy)
        self._analysis_plots_panel.rack_zero_ackermann_requested.connect(self._on_rack_zero_ackermann)
        self._analysis_plots_panel.mmd_requested.connect(self._on_plot_mmd)
        self._analysis_plots_panel.wheel_rate_linearity_requested.connect(self._on_plot_wheel_rate_linearity)
        self._analysis_plots_panel.llt_requested.connect(self._on_plot_llt)
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

    def _spring_travel_range(self, solver, label) -> tuple[float, float]:
        """Travel range [t_lo, t_hi] (metres) over which this corner's spring
        stays within its stroke [s_min, s_max].  Cached per label (stable
        geometry; the cache is cleared on _rebuild_solvers).

        Used to CLAMP the input travel: an over-stroke pose then holds AT the
        stroke limit (deterministic, and identical for the mirror corner under a
        symmetric input) instead of freezing at a per-corner cached SolvedState
        that can DESYNC between left/right (roll caches a bump pose on one side
        and a droop pose on the other → the two sides render at different ride
        heights → the intermittent asymmetric ARB / rocker / pushrod bug)."""
        cache = self._spring_travel_cache
        if label in cache:
            return cache[label]
        try:
            s_min, s_max = self._spring_limits(solver)
        except Exception:
            cache[label] = (-1.0, 1.0); return cache[label]

        def spring(t):
            try:
                return float(solver.solve(float(t)).spring_length)
            except Exception:
                return float('nan')

        # spring_length DECREASES with bump travel; bisect each stroke boundary.
        def bisect(target, lo, hi):
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                fm = spring(mid)
                if not np.isfinite(fm):
                    hi = mid; continue
                if fm > target:      # spring too long → need more bump → larger t
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        t_hi = bisect(s_min, 0.0, 0.12)    # bump (compression) limit
        t_lo = bisect(s_max, -0.12, 0.0)   # droop (extension) limit
        if not np.isfinite(t_hi):
            t_hi = 0.12
        if not np.isfinite(t_lo):
            t_lo = -0.12
        cache[label] = (float(t_lo), float(t_hi))
        return cache[label]

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
            dyn_params = self._apply_topology_to_dyn_params(dyn_params)

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

        Damper actuation (pushrod / pullrod / direct) and damper body are
        read from `self._topology` per axle.  Backward compatible: if
        topology was never set, the standard pushrod default applies.
        """
        # geometry changed → the per-corner spring-stroke travel range is stale
        if hasattr(self, '_spring_travel_cache'):
            self._spring_travel_cache.clear()
        try:
            corners = self._all_corner_hp()
            rt = _rack_travel_from_angle(steer_angle_deg, self._steer)
            from vahan.topology import (DamperActuation, DamperMount,
                                        SpringConfig, ARBType)
            for label, hp_d in corners.items():
                is_front = label in ('FL', 'FR')
                steered  = self._steered_hp(hp_d, rt, is_front)

                # Always use the DESIGN tie-rod length (before any rack travel).
                # Moving tie_rod_inner with the rack must not change the rod length.
                d = hp_d['tie_rod_outer'] - hp_d['tie_rod_inner']
                design_tierod_len_sq = float(d @ d)

                # Topology-driven actuation + body
                axle_top = (self._topology.front if is_front
                            else self._topology.rear)
                actuation_str = axle_top.damper_actuation.value  # 'pushrod' etc.

                # Standard pushrod convention: front UCA-mounted, rear LCA-mounted.
                # If the user explicitly chose a mount in the wizard, use that
                # for both pushrod_body (when pushrod/pullrod) and damper_body
                # (when direct).
                mount_str = axle_top.damper_mount.value  # 'uca'/'lca'/'upright'

                # DECOUPLED corners have no per-corner rocker.  Both
                # pushrods feed a central twin-bellcrank handled by
                # vahan/monoshock.py.  Use the dedicated 'cradle_link'
                # actuation mode -- it tracks pushrod_outer through the
                # UCA/LCA rigid body (so the cradle solver gets the
                # LIVE pushrod_outer world position) but DOES NOT
                # invoke any rocker solve.  No dummy hardpoints.
                # Rocker-related fields on SolvedState return NaN for
                # cradle_link corners; anything that reads them outside
                # of the cradle solver will see the NaN and surface the
                # bug immediately.
                solver_hp = dict(steered)
                solver_actuation = actuation_str
                # External-mechanism cars: the pushrod feeds a SEPARATE central
                # block, NOT a per-corner rocker+spring.
                #   * DECOUPLED   -> twin-bellcrank cradle (monoshock.py)
                #   * HEAVE_TBAR  -> central bellcranks + ONE T-bar doing heave
                #     (pivot -> 3rd spring) AND roll (twist), via heave_tbar.py
                #   * plain T-bar ARB (pushrod/pullrod) -> the SAME central
                #     mechanism without the 3rd spring (user: "t bar is just
                #     heave t bar without 3rd element"); the corner coilover
                #     lives on the central bellcrank.
                # cradle_link makes the corner just track the LIVE pushrod_outer
                # and report NaN rocker/spring fields (no phantom corner spring).
                _central_tbar = (
                    axle_top.arb_type == ARBType.TBAR
                    and axle_top.spring_config == SpringConfig.CORNER
                    and axle_top.damper_actuation in (DamperActuation.PUSHROD,
                                                      DamperActuation.PULLROD))
                if (axle_top.spring_config in (SpringConfig.DECOUPLED,
                                               SpringConfig.HEAVE_TBAR)
                        or _central_tbar):
                    solver_actuation = 'cradle_link'

                self._solvers[label] = SuspensionConstraints(
                    _hp_obj(solver_hp),
                    tierod_len_sq=design_tierod_len_sq,
                    pushrod_body=mount_str,
                    damper_actuation=solver_actuation,
                    damper_body=mount_str,
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
            # Don't just swallow — print the traceback to stderr so
            # broken solver-init bugs (e.g. an extra HP key the dataclass
            # rejects) don't silently leave stale solvers in place.
            import traceback as _tb
            _tb.print_exc()
            self.statusBar().showMessage(f'Solver init: {e}', 6000)

    def _arb_drop_top_world(self, label: str, state) -> np.ndarray | None:
        """World position of the ARB drop-link top, accounting for topology.

        BELLCRANK ARB: the drop-link top is on the rocker, so its position
        comes from the rocker rotation (state.rocker_angle around the rocker's
        axis).  This is the legacy Vahan path.

        CONTROL_ARM ARB: the drop-link top is on the LCA at the design
        hardpoint `arb_drop_top`.  As the LCA rotates through suspension
        travel, that point sweeps in space — we track it by expressing the
        design point in the LCA's body frame and re-applying the frame at
        the solved LCA pose.

        TBAR ARB: same kinematic path as BELLCRANK (rocker-mounted drop
        link); the difference is in how stiffness is computed downstream,
        not in where the point is in space.

        Returns world-space (X, Y, Z) in metres, or None if data is missing.
        """
        from vahan.topology import ARBType
        is_front = label in ('FL', 'FR')
        axle_top = self._topology.front if is_front else self._topology.rear
        arb_hp = self._front_arb if is_front else self._rear_arb
        if not arb_hp:
            return None
        solver = self._solvers.get(label)
        if solver is None:
            return None
        hp = solver.hp

        # Pick the design-position key for the drop-link top per topology
        is_right = label in ('FR', 'RR')
        flip = np.array([-1., 1., 1.]) if is_right else np.array([1., 1., 1.])
        if axle_top.arb_type == ARBType.CONTROL_ARM:
            # Drop-top = arb_lca_attach (no separate drop link; lever bolts
            # directly to LCA).  Track it through the LCA body frame.
            if 'arb_lca_attach' not in arb_hp:
                return None
            attach_design = arb_hp['arb_lca_attach'] * flip
            F0 = _build_frame(hp.lca_front, hp.lca_rear, hp.lca_outer)
            local = F0.T @ (attach_design - hp.lca_front)
            F_now = _build_frame(hp.lca_front, hp.lca_rear, state.lca_outer)
            return F_now @ local + hp.lca_front

        if axle_top.arb_type == ARBType.TBAR:
            # Drop-top = tbar_drop_top, mounted on rocker_tbar_drop_pt.
            # Same kinematic flow as bellcrank: rocker rotates the drop
            # top about the rocker's axis.
            if 'tbar_drop_top' not in arb_hp:
                return None
            dt_design = arb_hp['tbar_drop_top'] * flip
        else:
            # BELLCRANK ARB
            if 'arb_drop_top' not in arb_hp:
                return None
            dt_design = arb_hp['arb_drop_top'] * flip

        # Both BELLCRANK and TBAR: drop-top moves with the rocker
        if hp.rocker_pivot is None:
            return None
        rp = hp.rocker_pivot
        ax_pt = getattr(hp, 'rocker_axis_pt', None)
        if ax_pt is None:
            ax_pt = rp + np.array([0., 0.0254, 0.])
        r_axis = _norm(ax_pt - rp)
        arm_dt = dt_design - rp
        return rp + _rodrigues(arm_dt, r_axis, state.rocker_angle)

    @staticmethod
    def _solve_arb_bellcrank(arb_drop_top_world: np.ndarray,
                              arb_hp: dict) -> tuple:
        """
        Solve for the ARB / T-bar lever-arm rotation angle given the current
        world position of the drop-link rocker attachment point.

        Generalised to handle three ARB topologies, dispatched by inspecting
        the keys in ``arb_hp``:

        * Bellcrank U-bar ARB — keys: arb_pivot / arb_arm_end / arb_drop_top.
          Torsion bar axis is LATERAL (+X in our world frame).  Arm vector
          = arb_arm_end − arb_pivot.

        * T-bar — keys: tbar_base_chassis / tbar_top_node / tbar_arm_end
          / tbar_drop_top.  Torsion bar axis is whatever the user defined
          (tbar_top_node − tbar_base_chassis, normalised) — typically near
          vertical for a centre-pivoted T-bar with longitudinal levers,
          but the solver handles arbitrary directions via Rodrigues.  Arm
          vector = tbar_arm_end − tbar_top_node.

        * (Control-arm ARB doesn't go through this solver — it has no
          rocker-rotation DOF, so the corresponding code path skips it.)

        Returns (lever_angle_rad, arm_end_world, drop_link_travel_m).
        """
        # Dispatch on which key set is present.  T-bar takes precedence
        # since both could in principle coexist in a malformed dict.
        if 'tbar_top_node' in arb_hp and 'tbar_base_chassis' in arb_hp:
            base = np.asarray(arb_hp['tbar_base_chassis'], float)
            pv   = np.asarray(arb_hp['tbar_top_node'],    float)
            ae0  = np.asarray(arb_hp['tbar_arm_end'],     float)
            dt0  = np.asarray(arb_hp['tbar_drop_top'],    float)
            # Torsion axis from base to top node (chassis-fixed line about
            # which the lever arms rotate).  Normalised.  T-bar typically
            # has this near vertical (+Z) but the solver doesn't care.
            ax_v = pv - base
            n_ax = float(np.linalg.norm(ax_v))
            bc_axis = ax_v / n_ax if n_ax > 1e-9 else np.array([0., 0., 1.])
        else:
            pv   = np.asarray(arb_hp['arb_pivot'],     float)
            ae0  = np.asarray(arb_hp['arb_arm_end'],   float)
            dt0  = np.asarray(arb_hp['arb_drop_top'],  float)
            # Bellcrank U-bar torsion axis: lateral (+X).  This was the
            # original hard-coded direction; kept for back-compat with
            # existing bellcrank ARBs.
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
        is_front = (axle == 'F')
        # ── HEAVE_TBAR: ONE T-bar — derive arm/half/MR from the htb TWIST ────
        # The roll device is the single heave T-bar (user-confirmed), so the
        # roll RATE must come from the SAME bar the roll graph uses (the htb
        # twist), not the corner-rocker ARB path (NaN under cradle_link).  Only
        # the geometry is derived here; the bar diameter + material stay user
        # inputs (dynamics panel), so no car-build property is assumed.
        #   arm_length A  = |arm_tip - tbar_junc|   (lever from twist axis)
        #   half_length L = |tbar_junc - tbar_pivot| (torsion-active shaft)
        #   MR = wheel_travel / arm_tip_travel = 1 / (A * d(twist)/d(travel))
        heave = self._front_heave if is_front else self._rear_heave
        if heave:
            hs = self._heave_tbar_solver(is_front)
            corner = self._solvers.get('FL' if is_front else 'RL')
            if hs is None or corner is None:
                return None
            try:
                junc  = np.asarray(hs._g['tbar_junc'], float)
                tip   = np.asarray(hs._tip0, float)
                pivot = np.asarray(hs._tbar_pivot, float)
                arm_len_m  = float(np.linalg.norm(tip - junc))
                half_len_m = float(np.linalg.norm(junc - pivot))
                dt = 0.001
                tw_p = hs.twist(np.asarray(corner.solve(+dt).pushrod_outer, float))
                tw_m = hs.twist(np.asarray(corner.solve(-dt).pushrod_outer, float))
                tip_disp = arm_len_m * abs(tw_p - tw_m)
                if arm_len_m < 1e-4 or half_len_m < 1e-4 or tip_disp < 1e-9:
                    return None
                mr = (2.0 * dt) / tip_disp
                if not np.isfinite(mr) or mr <= 0.0:
                    return None
                return {'arm_length_mm': arm_len_m * 1000.0,
                        'half_length_mm': half_len_m * 1000.0, 'mr': mr}
            except Exception:
                return None

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

        # MR — central-difference perturbation through the drop-link chain.
        # ±1 mm wheel travel keeps us in the linear regime where
        # arm_tip_travel ≈ arm_length · bar_twist_angle, so MR is independent
        # of perturbation magnitude.
        dt = 0.001
        try:
            st_p = solver.solve(+dt)
            st_m = solver.solve(-dt)
            dt_w_p = self._arb_drop_top_world(label, st_p)
            dt_w_m = self._arb_drop_top_world(label, st_m)
            if dt_w_p is None or dt_w_m is None:
                return None
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

    def _decoupled_solver(self, is_front: bool):
        """ONE MODEL: build a FRESH TwinRockerDecoupledSolver from the CURRENT
        decoupled geometry for this axle (or None if the axle isn't decoupled).

        Single source of truth for the cradle kinematics.  The solver's
        constructor FREEZES a snapshot of every hardpoint (it ``.copy()``s
        each one), so a cached solver goes stale the instant the user nudges
        a cradle point.  Building fresh here -- and routing the 3D view, the
        kinematic graph, the dynamics MR, and the load path all through this
        one method -- guarantees an edit to ANY cradle hardpoint flows into
        every consumer, and that they can never drift apart."""
        deco = self._front_decoupled if is_front else self._rear_decoupled
        hp   = self._front_hp if is_front else self._rear_hp
        if not deco or hp is None:
            return None
        po = hp.get('pushrod_outer')
        if po is None:
            return None
        po = np.asarray(po, float)
        po_r = po.copy(); po_r[0] *= -1.0          # X-mirror for the partner wheel
        try:
            from vahan.monoshock import TwinRockerDecoupledSolver
            return TwinRockerDecoupledSolver(deco, po, po_r)
        except Exception:
            return None

    def _axle_flavors(self) -> set:
        """The set of suspension flavours present on the current car —
        'standard' / 'decoupled' / 'heave_tbar' / 'tbar' — used to filter the
        topology-specific inputs + graphs (single-model GUI)."""
        from vahan.topology import DamperActuation, ARBType, SpringConfig
        fl = set()
        topo = getattr(self, '_topology', None)
        if topo is None:
            return {'standard'}
        for a in (topo.front, topo.rear):
            if a.spring_config == SpringConfig.DECOUPLED:
                fl.add('decoupled')
            elif a.spring_config == SpringConfig.HEAVE_TBAR:
                fl.add('heave_tbar')
            elif (a.arb_type == ARBType.TBAR
                    and a.damper_actuation in (DamperActuation.PUSHROD,
                                               DamperActuation.PULLROD)):
                fl.add('tbar')
            else:
                fl.add('standard')
        return fl or {'standard'}

    def _inject_live_topology_values(self, vals: dict, travels: dict) -> None:
        """Fill the topology-element metrics (heave/roll spring len+MR, 3rd
        spring len+MR, T-bar twist, corner coil len) at the CURRENT pose into
        the per-corner live-values dicts.  Same solvers as graphs/3D/dynamics
        (single-model) — MRs by ±1 mm central difference about the pose."""
        dt = 0.001

        def _mir(p):
            q = np.asarray(p, float).copy(); q[0] *= -1.0; return q

        for is_front, (lL, lR) in ((True, ('FL', 'FR')), (False, ('RL', 'RR'))):
            cL = self._solvers.get(lL)
            cR = self._solvers.get(lR)
            if cL is None or cR is None:
                continue
            tL = float(travels.get(lL, 0.0)); tR = float(travels.get(lR, 0.0))

            deco = self._decoupled_solver(is_front)
            if deco is not None:
                try:
                    def _lens(dl, dr):
                        st = deco.solve(
                            np.asarray(cL.solve(tL + dl).pushrod_outer, float),
                            np.asarray(cR.solve(tR + dr).pushrod_outer, float))
                        return st.heave_length, st.roll_length
                    h0, r0 = _lens(0.0, 0.0)
                    hp_, rp_ = _lens(+dt, +dt)      # symmetric -> heave MR
                    hm_, rm_ = _lens(-dt, -dt)
                    ha_, ra_ = _lens(+dt, -dt)      # antisymmetric -> roll MR
                    hb_, rb_ = _lens(-dt, +dt)
                    out = {
                        'heave_spring_len': h0 * 1000.0,
                        'roll_spring_len': r0 * 1000.0,
                        'heave_spring_mr': abs(hp_ - hm_) / (2 * dt),
                        'roll_spring_mr': abs(ra_ - rb_) / (2 * dt),
                    }
                    for lbl in (lL, lR):
                        vals.get(lbl, {}).update(out)
                except Exception:
                    pass
                continue

            hts = self._heave_tbar_solver(is_front)
            if hts is not None:
                try:
                    poL0 = np.asarray(cL.solve(tL).pushrod_outer, float)
                    poLp = np.asarray(cL.solve(tL + dt).pushrod_outer, float)
                    poLm = np.asarray(cL.solve(tL - dt).pushrod_outer, float)
                    # right corner follows ITS OWN travel, mirrored into the
                    # left frame for the (left-side) solver
                    poR0 = _mir(np.asarray(cR.solve(tR).pushrod_outer, float))
                    # twist = antisymmetric half-difference of the two sides'
                    # shaft angles (exactly 0 in pure heave — see sweep inject)
                    out_ax = {'tbar_twist': float(np.degrees(
                        0.5 * (hts.twist(poL0) - hts.twist(poR0))))}
                    if hts.has_third_spring:
                        L0 = hts.heave(poL0).heave_spring_length
                        Lp = hts.heave(poLp).heave_spring_length
                        Lm = hts.heave(poLm).heave_spring_length
                        out_ax['third_spring_len'] = L0 * 1000.0
                        out_ax['third_spring_mr'] = abs(Lp - Lm) / (2 * dt)
                    cl_L = hts.coil_length(poL0)
                    cl_R = hts.coil_length(poR0)
                    for lbl, cl in ((lL, cl_L), (lR, cl_R)):
                        d = vals.get(lbl, {})
                        d.update(out_ax)
                        if np.isfinite(cl):
                            d['coil_len'] = cl * 1000.0
                except Exception:
                    pass

    def _heave_tbar_solver(self, is_front: bool):
        """ONE MODEL: build a FRESH HeaveTBarRockerSolver from the CURRENT
        heave-T-bar geometry for this axle (or None if not a heave-T-bar axle).

        Same rationale as _decoupled_solver: the solver freezes a snapshot of
        the (skewed) rocker axis + bar axes + lengths, so a cached one goes
        stale on edit.  Routing the 3D view, the kinematic graph, the dynamics
        MR, and the load path all through this single builder keeps them one
        model and live to edits."""
        heave = self._front_heave if is_front else self._rear_heave
        hp    = self._front_hp if is_front else self._rear_hp
        if not heave or hp is None:
            return None
        po = hp.get('pushrod_outer')
        if po is None:
            return None
        geo = {self._HTB_KEYMAP[k]: v for k, v in heave.items()
               if k in self._HTB_KEYMAP}
        # Mechanism keys the solver always needs.  The 3rd-spring keys are
        # OPTIONAL: present -> HEAVE_TBAR; absent -> plain T-bar ARB (the same
        # mechanism without the 3rd element).
        need = ('rocker_pivot_left', 'rocker_axis_pt_left', 'pushrod_inner_left',
                'tbar_drop_foot_left', 'tbar_arm_tip_left', 'tbar_pivot',
                'tbar_axis', 'tbar_junc')
        if not all(k in geo for k in need):
            return None
        # OPTIONAL corner damper (the reference coilover) — render-only, kept
        # OUT of the keymap so it isn't required for the MR solve / old files.
        if heave.get('htb_coil_rocker') is not None:
            geo['coil_rocker_left'] = np.asarray(heave['htb_coil_rocker'], float)
        if heave.get('htb_coil_chassis') is not None:
            geo['coil_chassis'] = np.asarray(heave['htb_coil_chassis'], float)
        try:
            from vahan.heave_tbar import HeaveTBarRockerSolver
            return HeaveTBarRockerSolver(geo, np.asarray(po, float))
        except Exception:
            return None

    def _inject_decoupled_mr(self, motion: str):
        """ONE MODEL: replace the dead per-corner motion_ratio on DECOUPLED
        axles with the REAL cross-car spring MR, computed from the same solved
        model the 3D view uses: corner wishbone solve -> pushrod_outer ->
        TwinRockerDecoupledSolver -> heave/roll spring length.

        In heave/pitch the HEAVE spring is the active one (both wheels of the
        axle move together); in roll the ROLL spring is active (opposite).  The
        result is the spring whose rate the dynamics uses, so the graph, the
        dynamics, and the 3D are finally the same model.  No-op on axles that
        aren't decoupled (no cradle solver)."""
        if motion not in ('heave', 'roll', 'pitch'):
            return

        from vahan.monoshock import TwinRockerDecoupledSolver

        def _mir(p):
            q = np.asarray(p, float).copy(); q[0] *= -1.0; return q

        for is_front, lbl, other in ((True, 'FL', 'FR'), (False, 'RL', 'RR')):
            deco = self._front_decoupled if is_front else self._rear_decoupled
            hp = self._front_hp if is_front else self._rear_hp
            corner = self._solvers.get(lbl)
            res = self._sweep_results.get(lbl)
            po_design = hp.get('pushrod_outer') if hp else None
            if not deco or corner is None or res is None or po_design is None:
                continue
            # Build the cradle FRESH from current geometry every sweep — so a
            # nudge to any cradle point flows straight into the graph (the
            # cached solver from set_topology would be stale = disconnected).
            cradle = self._decoupled_solver(is_front)
            if cradle is None:
                continue
            t_in = res.get('_travel_in')
            if t_in is None or len(t_in) < 3:
                continue
            n = len(t_in)
            cache = {}
            def _po(t):
                k = round(float(t), 9)
                if k not in cache:
                    try:
                        cache[k] = np.asarray(corner.solve(float(t)).pushrod_outer, float)
                    except Exception:
                        cache[k] = None
                return cache[k]

            spring = np.full(n, np.nan)
            heave_len = np.full(n, np.nan)
            roll_len = np.full(n, np.nan)
            for i, t in enumerate(t_in):
                pL = _po(t)
                if pL is None or not np.all(np.isfinite(pL)):
                    continue
                try:
                    # one cradle solve per sweep point gives BOTH coilover
                    # lengths along the actual sweep trajectory — the basis
                    # for the distinct Heave-spring / Roll-spring graphs.
                    pR = _po(-t) if motion == 'roll' else pL
                    if pR is None:
                        continue
                    st = cradle.solve(pL, _mir(pR))
                    heave_len[i] = st.heave_length
                    roll_len[i] = st.roll_length
                    spring[i] = (st.roll_length if motion == 'roll'
                                 else st.heave_length)
                except Exception:
                    continue

            def _mr_curve(arr):
                m = np.full(n, np.nan)
                for i in range(1, n - 1):
                    dt = float(t_in[i + 1] - t_in[i - 1])
                    if (abs(dt) > 1e-9 and np.isfinite(arr[i + 1])
                            and np.isfinite(arr[i - 1])):
                        m[i] = abs((arr[i + 1] - arr[i - 1]) / dt)
                if n >= 2:
                    m[0], m[-1] = m[1], m[-2]
                return m

            mr = _mr_curve(spring)
            res['motion_ratio'] = mr
            res['spring_len'] = spring * 1000.0
            # Topology metrics: BOTH coilovers, distinctly labelled in the
            # catalog ('Heave-spring …' / 'Roll-spring …').  In a heave sweep
            # the roll spring barely moves (MR ~ 0) and vice versa — that IS
            # the decoupling, made visible.
            extra = {
                'heave_spring_len': heave_len * 1000.0,
                'roll_spring_len': roll_len * 1000.0,
                'heave_spring_mr': _mr_curve(heave_len),
                'roll_spring_mr': _mr_curve(roll_len),
            }
            res.update(extra)
            o = self._sweep_results.get(other)
            if o is not None:
                o['motion_ratio'] = mr.copy()
                o['spring_len'] = (spring * 1000.0).copy()
                for k, v in extra.items():
                    o[k] = v.copy()

    _HTB_KEYMAP = {
        'htb_rocker_pivot': 'rocker_pivot_left', 'htb_rocker_axis': 'rocker_axis_pt_left',
        'htb_pushrod_inner': 'pushrod_inner_left', 'htb_drop_foot': 'tbar_drop_foot_left',
        'htb_arm_tip': 'tbar_arm_tip_left', 'htb_tbar_pivot': 'tbar_pivot',
        'htb_tbar_axis': 'tbar_axis', 'htb_tbar_junc': 'tbar_junc',
        # The 3rd-spring endpoints keep their visible names so a nudge to the
        # point the user grabs flows into the MR.
        'heave_spring_tbar_pt': 'heave_spring_tbar_pt',
        'heave_spring_chassis_pt': 'heave_spring_chassis_pt',
    }

    def _inject_heave_tbar_mr(self, motion: str):
        """ONE MODEL: HEAVE_TBAR axles get their real 3rd-spring motion ratio
        from vahan/heave_tbar.HeaveTBarRockerSolver, driven by the SAME
        pushrod_outer the corner solve produces (wheel -> pushrod -> rocker ->
        drop-link -> T-bar heave-pivot -> 3rd spring).  Replaces the corner-
        coilover MR the graph used to show.  heave/pitch -> 3rd-spring MR;
        roll -> T-bar twist per metre.  No-op unless the htb_ central geometry
        is present (HEAVE_TBAR only)."""
        if motion not in ('heave', 'roll', 'pitch'):
            return
        try:
            from vahan.heave_tbar import HeaveTBarRockerSolver
        except Exception:
            return
        for is_front, lbl, other in ((True, 'FL', 'FR'), (False, 'RL', 'RR')):
            heave = self._front_heave if is_front else self._rear_heave
            corner = self._solvers.get(lbl)
            res = self._sweep_results.get(lbl)
            hp = self._front_hp if is_front else self._rear_hp
            if not heave or corner is None or res is None or not hp:
                continue
            po_design = hp.get('pushrod_outer')
            if po_design is None:
                continue
            t_in = res.get('_travel_in')
            if t_in is None or len(t_in) < 3:
                continue
            solver = self._heave_tbar_solver(is_front)
            if solver is None:
                continue
            n = len(t_in)
            spring = np.full(n, np.nan)
            third_len = np.full(n, np.nan)
            twist_deg = np.full(n, np.nan)
            coil_len = np.full(n, np.nan)
            # ACTIVE spring per car flavour (one mechanism, two flavours):
            #   * HEAVE_TBAR (has 3rd spring): heave/pitch -> 3rd-spring
            #     length; roll -> T-bar twist.
            #   * plain T-bar ARB (no 3rd spring): the ride spring IS the
            #     corner coilover on the central bellcrank -> its length is
            #     the spring curve for EVERY motion (each side follows its
            #     own rocker).
            # The Topology metrics (3rd-spring length/MR, T-bar twist, corner
            # coil length) are filled alongside for the graph picker.
            #
            # T-bar twist is the ANTISYMMETRIC mode: half the difference of
            # the two sides' shaft angles.  A single-sided twist() call would
            # wrongly report twist in pure heave (the solver can't tell the
            # heave pivot from twist with one foot) — verified: 15.7 deg of
            # phantom twist at +20 mm heave.  Using both sides it is exactly
            # 0 in heave and the true twist in roll.
            corner_o = self._solvers.get(other)
            for i, t in enumerate(t_in):
                try:
                    po = np.asarray(corner.solve(float(t)).pushrod_outer, float)
                    if not np.all(np.isfinite(po)):
                        continue
                    tw_L = solver.twist(po)
                    tw_R = tw_L                      # symmetric fallback
                    if corner_o is not None:
                        t_o = -float(t) if motion == 'roll' else float(t)
                        po_o = np.asarray(
                            corner_o.solve(t_o).pushrod_outer, float)
                        po_o[0] *= -1.0              # mirror into LEFT frame
                        tw_R = solver.twist(po_o)
                    twist_deg[i] = float(np.degrees(0.5 * (tw_L - tw_R)))
                    coil_len[i] = solver.coil_length(po)
                    if solver.has_third_spring:
                        third_len[i] = solver.heave(po).heave_spring_length
                        spring[i] = (solver.twist(po) if motion == 'roll'
                                     else third_len[i])
                    else:
                        spring[i] = coil_len[i]
                except Exception:
                    continue
            def _mr_of(arr, ts):
                m = np.full(len(arr), np.nan)
                for i in range(1, len(arr) - 1):
                    dt = float(ts[i + 1] - ts[i - 1])
                    if (abs(dt) > 1e-9 and np.isfinite(arr[i + 1])
                            and np.isfinite(arr[i - 1])):
                        m[i] = abs((arr[i + 1] - arr[i - 1]) / dt)
                if len(arr) >= 2:
                    m[0], m[-1] = m[1], m[-2]
                return m

            res['motion_ratio'] = _mr_of(spring, t_in)
            res['spring_len'] = spring * 1000.0
            # Topology metrics: shared central items (3rd spring, twist) +
            # this corner's coil.
            res['third_spring_len'] = third_len * 1000.0
            res['third_spring_mr'] = _mr_of(third_len, t_in)
            res['tbar_twist'] = twist_deg
            res['coil_len'] = coil_len * 1000.0
            o = self._sweep_results.get(other)
            if o is not None:
                # central shared elements: same curve on both corners
                o['third_spring_len'] = res['third_spring_len'].copy()
                o['third_spring_mr'] = res['third_spring_mr'].copy()
                o['tbar_twist'] = twist_deg.copy()
                # per-corner coilover: the partner corner follows ITS OWN
                # rocker (opposite travel in roll) — compute, don't copy.
                o_t = o.get('_travel_in')
                corner_o = self._solvers.get(other)
                sp_o = None
                if o_t is not None and corner_o is not None:
                    sp_o = np.full(len(o_t), np.nan)
                    for i, t in enumerate(o_t):
                        try:
                            po_o = np.asarray(
                                corner_o.solve(float(t)).pushrod_outer, float)
                            po_o[0] *= -1.0    # mirror into the LEFT frame
                            sp_o[i] = solver.coil_length(po_o)
                        except Exception:
                            continue
                    o['coil_len'] = sp_o * 1000.0
                if solver.has_third_spring:
                    # active spring is the shared central element: copy exact.
                    o['motion_ratio'] = res['motion_ratio'].copy()
                    o['spring_len'] = (spring * 1000.0).copy()
                elif sp_o is not None:
                    # active spring is the per-corner coil: per-corner curve.
                    o['motion_ratio'] = _mr_of(sp_o, o_t)
                    o['spring_len'] = sp_o * 1000.0

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
                    label=lbl,
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
                axle_rc_x = np.full(n_steps, float('nan'))
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
                        axle_rc[i] = rc[1] * 1000.    # Z in mm
                        axle_rc_x[i] = rc[0] * 1000.  # LATERAL in mm
                for lbl in (left_lbl, right_lbl):
                    if lbl in self._sweep_results:
                        self._sweep_results[lbl]['rc_height'] = axle_rc.copy()
                        # RC LATERAL migration — the industry red-flag plot a
                        # height-only RC graph hides (an RC darting sideways
                        # in roll = geometric load-transfer asymmetry).
                        self._sweep_results[lbl]['rc_lateral'] = axle_rc_x.copy()

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

            # ONE MODEL: decoupled axles get their real cross-car heave/roll
            # spring MR from the cradle solver (was dead NaN before).
            self._inject_decoupled_mr(motion)
            # ONE MODEL: heave-T-bar axles get their real 3rd-spring MR from
            # the T-bar solver (was the wrong corner-coilover MR before).
            self._inject_heave_tbar_mr(motion)

            # ── Surface the ACTUAL solvable range ─────────────────────────
            # The constraint solver fails silently past the geometry's real
            # articulation limit (ball-joint/linkage lock) and the graphs just
            # blank — tell the user what range actually solved instead of
            # letting a ±50 mm request quietly deliver ±32 mm.
            try:
                fl = self._sweep_results.get('FL', {})
                ref = np.asarray(fl.get('camber', []), float)
                if len(ref) == len(x_arr) and len(ref) > 2:
                    fin = np.where(np.isfinite(ref))[0]
                    if len(fin) and (fin[0] > 0 or fin[-1] < len(ref) - 1):
                        unit = '°' if motion in ('roll', 'steer') else ' mm'
                        self.statusBar().showMessage(
                            f'Kinematic limit: solver holds only '
                            f'{x_arr[fin[0]]:+.1f}..{x_arr[fin[-1]]:+.1f}{unit} '
                            f'of the requested {x_arr[0]:+.1f}..'
                            f'{x_arr[-1]:+.1f}{unit} — geometry locks beyond '
                            f'(graphs blank there)', 10000)
            except Exception:
                pass

            self._replot()
        except Exception as e:
            self.statusBar().showMessage(f'Sweep: {e}', 6000)
            import traceback; traceback.print_exc()

    # Keys whose valid values depend on the spring/rocker being within limits.
    # All other metrics (camber, toe, RC height, etc.) come from the main
    # Newton solver and are independent of whether the rocker/spring is OOB.
    _SPRING_KEYS = frozenset({'motion_ratio', 'spring_len', 'rocker_angle'})

    def _do_sweep(self, solver, travels, side, arb_hp=None,
                  camber_off=0., toe_off=0., is_front=True, label=None):
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
        out['_travel_in'] = travels_arr   # input travel (m) — for cross-car cradle/tbar MR

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

                # DECOUPLED corner solver returns NaN spring_length (no
                # per-corner spring -- cradle handles the real damper).
                # Skip the stroke-trim and record ALL metrics for those.
                # For other topologies, stroke-trim out-of-range points.
                if np.isnan(st.spring_length):
                    spring_ok = True
                    # don't record spring_lens[i] (stays NaN, correctly)
                else:
                    spring_ok = spring_min <= st.spring_length <= spring_max
                    if spring_ok:
                        spring_lens[i] = st.spring_length

                # Outside stroke limits: leave ALL metrics as NaN (trim the curve).
                # Warm-start variables still update so the solver stays on-track.
                if not spring_ok:
                    spring_prev = travel_prev = None
                    state_prev = st       # keep state_prev current for IC continuity
                    continue

                # ── ARB bell-crank (topology-aware drop-top path) ─────────
                arb_kwargs = {}
                if arb_hp is not None and label is not None:
                    try:
                        dt_w = self._arb_drop_top_world(label, st)
                        if dt_w is not None:
                            ang, _, dl_t = self._solve_arb_bellcrank(dt_w, arb_hp)
                            arb_kwargs = {
                                'arb_angle':       ang,
                                'arb_drop_travel': dl_t,
                                'arb_mr': min(abs(np.degrees(ang) / (float(t) * 1000)), 5.0)
                                          if abs(float(t)) > 1e-9 else float('nan'),
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

    def _assemble_corners_draw(self, travels, rt_m):
        """Build the per-corner draw dicts + metrics from the ONE
        solved model (self._solvers).  Shared by the Suspension 3-D view
        (_update_3d) and the embedded Loads-page 3-D view (build_load_view)
        so both read a single geometry -- never a second/hardcoded model.
        Returns (corners_draw, all_corner_values)."""
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
            # DECOUPLED (cradle_link) corners have NO corner spring, so
            # st.spring_length is intentionally NaN.  Running the freeze on
            # them would ALWAYS fail the bounds test and substitute a STALE
            # state left over from a previously-selected pushrod topology
            # (finite pushrod_inner) — which renders a PHANTOM corner pushrod
            # line 217 mm off the real cradle pushrod.  Skip the freeze; the
            # fresh state's NaN pushrod_inner correctly suppresses the corner
            # pushrod line, and the real pushrod is drawn from the cradle pose.
            if getattr(solver, '_damper_actuation', None) == 'cradle_link':
                pass   # keep fresh decoupled state (pushrod_inner = NaN)
            else:
                # Spring-stroke limit: instead of freezing at a per-corner cached
                # state (which desyncs between mirror sides after a roll/pitch and
                # yields the intermittent asymmetric render), CLAMP the input
                # travel to the stroke range and re-solve.  A symmetric input then
                # clamps identically on both sides -> the corners stay mirror
                # images; and the corner always renders (no cache-miss skip).
                try:
                    s_min, s_max = self._spring_limits(solver)
                    if not (s_min <= st.spring_length <= s_max):
                        t_lo, t_hi = self._spring_travel_range(solver, label)
                        tc = min(max(float(t), t_lo), t_hi)
                        if abs(tc - float(t)) > 1e-9:
                            st = solver.solve(tc)   # hold AT the stroke limit
                except Exception:
                    pass

            hp_d = hp_dicts[label]

            # ── steering visual: show tie_rod_inner at steered position ──
            is_front = label in ('FL', 'FR')
            steered_hp_d = self._steered_hp(hp_d, rt_m, is_front)
            pts = _state_to_pts(st, steered_hp_d)

            # rocker_pivot is chassis-fixed.  For direct-damper topologies
            # there is no rocker — fall back to damper_chassis_pt so the
            # downstream rendering code never KeyErrors.
            if 'rocker_pivot' in hp_d:
                pts['rocker_pivot'] = hp_d['rocker_pivot']
            elif 'damper_chassis_pt' in hp_d:
                pts['rocker_pivot'] = hp_d['damper_chassis_pt']

            # ── arb_drop_top: route via the topology-aware helper ───────
            # Bellcrank: drop top is on the rocker (rotates with state.rocker_angle)
            # Control-arm: drop top is on the LCA (sweeps with LCA pose)
            # T-bar: same kinematic path as bellcrank
            arb_hp = self._front_arb if is_front else self._rear_arb
            try:
                dt_world = self._arb_drop_top_world(label, st)
                if dt_world is not None:
                    pts['arb_drop_top'] = dt_world
                    arb_hp_vis = (arb_hp if label not in ('FR', 'RR')
                                  else {k: v * flip_x for k, v in arb_hp.items()})
                    _, ae_world, _ = self._solve_arb_bellcrank(
                        dt_world, arb_hp_vis)
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
                'pts': pts, 'spin_axis': spin_vis, 'label': label,
                # Only the keys actually present in this corner's hardpoint
                # dict are real, editable points.  _state_to_pts injects
                # derived render points (direct-damper reports pushrod_inner
                # == damper attach; decoupled feeds a shared cradle) — the
                # renderer uses those to DRAW geometry but must NOT turn
                # them into pickable markers (clicking a phantom that isn't
                # in the edit list selects nothing).
                'editable': set(hp_d.keys())})

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
        return corners_draw, all_corner_values

    def build_load_view(self, view3d, corner_label, lat_g, lon_g, vec_mode='resultant'):
        """Drive a (possibly SEPARATE) View3D into Load mode from the ONE solved
        model — used by the embedded Loads-page 3-D view so inputs -> table ->
        picture all read one model.  `corner_label` None => all four corners
        (exit isolation).  Mirrors the Load-mode setup inside _update_3d but
        targets the passed view3d instead of self.view3d, and holds a static
        design-ride pose (the load "case" is the g-vector, not a kinematic sweep).
        """
        if not self._solvers:
            return
        cp = self._car
        try:
            view3d.set_tire_params(
                outer_r=cp['tire_outer_dia_mm'] / 2000.,
                rim_r=cp['tire_rim_dia_mm'] / 2000.,
                half_w=cp['tire_width_mm'] / 2000.,
            )
        except Exception:
            pass
        travels = {lbl: 0.0 for lbl in ('FL', 'FR', 'RL', 'RR')}
        corners_draw, _ = self._assemble_corners_draw(travels, 0.0)
        if not corners_draw:
            return
        try:
            _thick = cp.get('show_shock_thickness', True)
            view3d.set_spring_dims(
                cp.get('spring_od_mm', 63.0) if _thick else 2.0,
                cp.get('damper_od_mm', 50.0) if _thick else 2.0)
            view3d.set_thickness(_thick)
        except Exception:
            pass
        try:
            view3d.set_view_mode('load')
            view3d.sync_view_controls(view_mode='load')
            view3d.set_isolate_corner(corner_label)     # None => all corners
            view3d.set_brakes(cp.get('show_brakes', True))
            view3d.set_brake_dims(
                cp.get('rotor_dia_mm', 240.0),
                # keep the drawn caliper on the SAME bolt line and angle the load
                # model uses, or the picture contradicts the arrows
                mount_height_mm=getattr(self._loads_panel.get_brake_params_front(),
                                        'caliper_mount_height_mm', 27.9),
                caliper_angle_deg=getattr(self._loads_panel.get_upright_params(),
                                          'caliper_angle_deg', 45.0),
                vertical_mounts=getattr(self._loads_panel.get_upright_params(),
                                        'caliper_vertical_mounts', True),
                rotor_thickness_mm=getattr(self._loads_panel.get_brake_params_front(),
                                           'rotor_thickness_mm', 6.35))
        except Exception:
            pass
        # Draw the ARB structure too (the user wants it on the Loads page, not
        # just its force arrows).  view3d keeps the ARB visible in Load mode even
        # when a corner is isolated.
        try:
            _arb = self._assemble_arb_segs(corners_draw)
        except Exception:
            _arb = []
        view3d.update_scene(corners_draw, _arb)
        # rear driveshaft / diff package (RWD) — same build as _update_3d
        try:
            import types as _types
            from vahan.driveshaft import package as _ds_package
            _rear_states = {
                c['label']: _types.SimpleNamespace(
                    wheel_center=c['pts']['wheel_center'],
                    spin_axis=c['spin_axis'])
                for c in corners_draw if c['label'] in ('RL', 'RR')
                and 'wheel_center' in c['pts']}
            _pkg = (_ds_package(cp, _rear_states)
                    if len(_rear_states) == 2 else None)
            _only = corner_label if corner_label in ('RL', 'RR') else None
            _show_ds = (cp.get('show_driveshaft', True)
                        and (corner_label is None or corner_label in ('RL', 'RR')))
            view3d.set_driveshaft_package(_pkg, show=_show_ds, only=_only)
        except Exception:
            pass
        # force-vector arrows (feed hover) — from the SAME solved model
        try:
            from gui.wheel_package import load_arrows
            view3d.set_load_vectors(
                load_arrows(self, lat_g, lon_g, mode=vec_mode, only_corner=corner_label))
        except Exception:
            view3d.set_load_vectors([])
        # Frame the camera on a corner CHANGE only (don't fight manual orbit):
        # zoom onto the isolated wheel, or the whole car for 'all corners'.
        try:
            if getattr(view3d, '_load_framed_corner', '__unset__') != corner_label:
                if corner_label:
                    _cd = next((c for c in corners_draw
                                if c['label'] == corner_label
                                and 'wheel_center' in c['pts']), None)
                    if _cd is not None:
                        wc = np.asarray(_cd['pts']['wheel_center'], float)
                        # Frame the WHOLE corner (wheel + rocker/ARB), not just the
                        # wheel, so the rocker / spring / ARB load arrows are in view.
                        rp = np.asarray(_cd['pts'].get('rocker_pivot', wc), float)
                        ctr = 0.5 * (wc + rp)
                        view3d.set_camera_center((float(ctr[0]), float(ctr[1]), float(ctr[2])))
                        view3d._cam.scale_factor = 1.05
                else:
                    wb_half = cp['axle_spacing_mm'] / 2000.
                    view3d.set_camera_center((0., wb_half, 0.2))
                    view3d._cam.scale_factor = 2.6
                view3d._canvas.update()
                view3d._load_framed_corner = corner_label
        except Exception:
            pass

    def _assemble_arb_segs(self, corners_draw):
        """ARB linkage segments (torsion bar + levers + drop links) from the
        live corners_draw.  Shared by _update_3d and build_load_view so the ARB
        renders on the Loads page too (not just its force arrows)."""
        flip_x = np.array([-1., 1., 1.])
        arb_segs = []
        # The arb_hp dict has different key sets per topology:
        #   bellcrank   → 'arb_pivot' + 'arb_arm_end' + 'arb_drop_top'
        #   control-arm → 'arb_pivot' + 'arb_lca_attach'
        #   t-bar       → 'tbar_base_chassis' + 'tbar_top_node'
        #                 + 'tbar_arm_end' + 'tbar_drop_top'
        #   none        → empty dict
        # The rendering branches accordingly and skips silently on
        # missing keys (topology mid-implementation).
        for axle_l, axle_r, arb_hp in [
            ('FL', 'FR', self._front_arb),
            ('RL', 'RR', self._rear_arb),
        ]:
            if not arb_hp:
                continue   # NONE topology
            # HEAVE_TBAR: the roll device IS the ONE heave T-bar, drawn by
            # the htb_ linkage section below — do NOT also draw the legacy
            # separate ARB T-bar (it would be a second, redundant bar, and
            # its drop link reads the now-NaN cradle_link corner rocker).
            # The heave dict is populated ONLY for HEAVE_TBAR axles.
            if (self._front_heave if axle_l == 'FL' else self._rear_heave):
                continue
            if 'arb_pivot' in arb_hp and 'arb_arm_end' in arb_hp:
                # ── Bellcrank ARB rendering ──────────────────────────
                pv_l = arb_hp['arb_pivot'].copy()
                pv_r = pv_l * flip_x
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
            elif 'arb_pivot' in arb_hp and 'arb_lca_attach' in arb_hp:
                # ── Control-arm ARB rendering (NO drop link) ─────────
                pv_l = arb_hp['arb_pivot'].copy()
                pv_r = pv_l * flip_x
                lc_l = arb_hp['arb_lca_attach'].copy()
                lc_r = lc_l * flip_x
                arb_segs += [
                    (pv_l, pv_r),    # torsion section
                    (pv_l, lc_l),    # left arm = lever (no drop link)
                    (pv_r, lc_r),    # right arm
                ]
            elif 'tbar_base_chassis' in arb_hp:
                # ── T-bar rendering (torsion bar + LIVE levers + LIVE
                #    drop links) ───────────────────────────────────────
                # The T-bar DOES have a drop link from each lever tip to
                # the corner ROCKER (bellcrank).  Render it from the LIVE
                # solved points so both the lever and its drop link track
                # the solver under travel — exactly as the bellcrank-ARB
                # branch above does:
                #   * lever tip  = arb_arm_end_world  (solved lever pose,
                #                  rotated about the torsion-bar axis)
                #   * rocker end = arb_drop_top       (rocker_tbar_drop_pt
                #                  carried by the rotating rocker)
                # The default geometry separates tbar_arm_end (lever tip)
                # from tbar_drop_top (== rocker_tbar_drop_pt) by a real
                # ~41 mm drop-link rod, so the link renders as a visible
                # segment and actuates the lever.  The wiring below is
                # correct for whatever separation the geometry specifies.
                base = arb_hp['tbar_base_chassis'].copy()
                top  = arb_hp['tbar_top_node'].copy()
                ae_l_design = arb_hp['tbar_arm_end'].copy()
                ae_r_design = ae_l_design * flip_x
                arb_segs += [(base, top)]   # torsion bar (chassis-fixed)
                for c in corners_draw:
                    ae_w = c['pts'].get('arb_arm_end_world')
                    dt   = c['pts'].get('arb_drop_top')
                    if c['label'] == axle_l:
                        ae = ae_w if ae_w is not None else ae_l_design
                        arb_segs += [(top, ae)]          # left lever (live)
                        if dt is not None:
                            arb_segs += [(ae, dt)]       # left drop link (live)
                    if c['label'] == axle_r:
                        ae = ae_w if ae_w is not None else ae_r_design
                        arb_segs += [(top, ae)]          # right lever (live)
                        if dt is not None:
                            arb_segs += [(ae, dt)]       # right drop link (live)
        return arb_segs

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

            corners_draw, all_corner_values = self._assemble_corners_draw(travels, rt_m)
            flip_x = np.array([-1., 1., 1.])

            # ── ARB visual ────────────────────────────────────────────────────
            # Topology: arb_drop_top (on rocker, moving)
            #           → arb_arm_end (blade tip, rotates about torsion-bar axis)
            #           → arb_pivot (fixed)
            #           arb_pivot_L → arb_pivot_R  (torsion bar, lateral)
            # arb_arm_end_world stored in pts by the bell-crank solve above.
            arb_segs = self._assemble_arb_segs(corners_draw)

            # ── HEAVE-T-BAR: the ONE T-bar mechanism, posed LIVE ───────────
            # The pushrod feeds the htb_ rocker (skewed plane); the rocker drives
            # a drop link into the SINGLE T-bar, which pivots (heave -> 3rd
            # spring) and twists (roll).  Drawn from HeaveTBarRockerSolver.pose()
            # at the LIVE corner pushrod_outer -- the SAME model the graph +
            # dynamics use.  The solver models the LEFT side; RIGHT is the
            # X-mirror.  The 3rd spring is the central SHOCK cylinder below
            # (posed at the live bar attach).  No corner spring, no separate ARB
            # T-bar -- it is ONE bar (user-confirmed).
            # solver.pose_axle(po_L, po_R) solves the WHOLE bar at once: ONE
            # shared junction, pivoting in heave and twisting in roll — so there
            # is no phantom second bar and pure roll does NOT actuate the heave
            # spring.  The lever is the single line left_tip → junction →
            # right_tip (bisected at the torsion-bar top), NOT a line down to the
            # chassis pivot.  No pivot-axis indicator lines.
            # The whole mechanism (coloured linkage + bellcrank plates + corner
            # dampers) is rendered by view3d.update_heave_tbar() from the
            # pose_axle dict per axle — NOT dumped into the single-colour ARB
            # line.  The 3rd-spring shock cylinder is posed at the live bar attach.
            htb_shock_front = None
            htb_shock_rear = None
            htb_pose_front = None
            htb_pose_rear = None
            for is_front, axle_corners in ((True, ('FL', 'FR')),
                                           (False, ('RL', 'RR'))):
                heave_hp = self._front_heave if is_front else self._rear_heave
                if not heave_hp:
                    continue
                solver = self._heave_tbar_solver(is_front)
                if solver is None:
                    continue
                po_L = po_R = None
                for c in corners_draw:
                    if c['label'] == axle_corners[0]:
                        po_L = c['pts'].get('pushrod_outer')
                    elif c['label'] == axle_corners[1]:
                        po_R = c['pts'].get('pushrod_outer')
                if po_L is None or po_R is None:
                    continue
                try:
                    p = solver.pose_axle(np.asarray(po_L, float),
                                         np.asarray(po_R, float))
                except Exception:
                    continue
                # 3rd-spring shock only exists on the HEAVE_TBAR flavour; the
                # plain T-bar ARB is the same mechanism without it.
                shock = None
                if 'arb_attach' in p and 'chassis' in p:
                    shock = {'heave_spring_tbar_pt': np.asarray(p['arb_attach'], float),
                             'heave_spring_chassis_pt': np.asarray(p['chassis'], float)}
                if is_front:
                    htb_shock_front, htb_pose_front = shock, p
                else:
                    htb_shock_rear, htb_pose_rear = shock, p
            try:
                self.view3d.update_heave_tbar(htb_pose_front, htb_pose_rear)
            except Exception:
                pass

            # ── Decoupled (reference twin-bellcrank) ────────────────────
            # Mesh path (bellcrank plates + heave damper + roll damper
            # cylinders) handled in view3d.update_decoupled_cradles.
            # Here we add only the line segments that have no mesh form:
            # the pushrod-to-rocker connections + each rocker's pivot
            # axis indicator (LEFT and RIGHT separately).
            #
            # CRITICAL: pushrod_inner_{left,right} and the damper-attach
            # points are RIGIDLY ATTACHED TO THE BELLCRANK -- they rotate
            # with the bellcrank under load.  Earlier this code used the
            # STATIC design positions from self._{front,rear}_decoupled
            # for pin_l / pin_r, which made the pushrod visually stretch
            # by ~100 mm under +25 mm bump (the pushrod_outer moved with
            # the upright but pin_l stayed at the design pose).  Re-solve
            # the cradle here with the LIVE pushrod_outer from each
            # corner and use the solved bellcrank pose for all
            # cradle-resident points.
            decoupled_state_front = None
            decoupled_state_rear  = None
            for dec_hp, axle_corners, solver_attr, state_target in (
                (self._front_decoupled, ('FL', 'FR'),
                 '_front_cradle_solver', 'front'),
                (self._rear_decoupled,  ('RL', 'RR'),
                 '_rear_cradle_solver',  'rear'),
            ):
                if not dec_hp:
                    continue
                try:
                    # Pull LIVE pushrod_outer for each side from
                    # corners_draw (the solved state).
                    po_L = None; po_R = None
                    for c in corners_draw:
                        if c['label'] == axle_corners[0]:
                            po_L = c['pts'].get('pushrod_outer',
                                                c['pts']['wheel_center'])
                        elif c['label'] == axle_corners[1]:
                            po_R = c['pts'].get('pushrod_outer',
                                                c['pts']['wheel_center'])
                    # FRESH solver from current geometry (cached = stale on edit)
                    cradle_solver = self._decoupled_solver(state_target == 'front')
                    pin_l = dec_hp['pushrod_inner_left']
                    pin_r = dec_hp['pushrod_inner_right']
                    heave_l = dec_hp.get('heave_damper_left')
                    heave_r = dec_hp.get('heave_damper_right')
                    roll_l  = dec_hp.get('roll_damper_left')
                    roll_r  = dec_hp.get('roll_damper_right')
                    if cradle_solver is not None and po_L is not None and po_R is not None:
                        st = cradle_solver.solve(po_L, po_R)
                        # The solver returns the LIVE positions of all
                        # bellcrank-resident points -- use them.
                        pin_l   = st.pushrod_inner_L
                        pin_r   = st.pushrod_inner_R
                        heave_l = st.heave_L
                        heave_r = st.heave_R
                        roll_l  = st.roll_L
                        roll_r  = st.roll_R
                        # Stash for the view3d mesh update below
                        live = {
                            'rocker_pivot_left':    dec_hp['rocker_pivot_left'],
                            'rocker_pivot_right':   dec_hp['rocker_pivot_right'],
                            'rocker_axis_pt_left':  dec_hp['rocker_axis_pt_left'],
                            'rocker_axis_pt_right': dec_hp['rocker_axis_pt_right'],
                            'pushrod_inner_left':   pin_l,
                            'pushrod_inner_right':  pin_r,
                            'heave_damper_left':    heave_l,
                            'heave_damper_right':   heave_r,
                            'roll_damper_left':     roll_l,
                            'roll_damper_right':    roll_r,
                        }
                        if state_target == 'front':
                            decoupled_state_front = live
                        else:
                            decoupled_state_rear = live
                    # Pivot axis indicators (one per side, chassis-fixed)
                    arb_segs += [(dec_hp['rocker_pivot_left'],
                                  dec_hp['rocker_axis_pt_left']),
                                 (dec_hp['rocker_pivot_right'],
                                  dec_hp['rocker_axis_pt_right'])]
                    # Pushrods: rendered from live pushrod_outer to the
                    # LIVE pushrod_inner positions (now rigid-rod length).
                    if po_L is not None:
                        arb_segs.append((po_L, pin_l))
                    if po_R is not None:
                        arb_segs.append((po_R, pin_r))
                except KeyError:
                    pass

            # Push cradle dicts to view3d (mesh-based plate + spring
            # cylinders).  Use the cradle-solver POSE-UPDATED positions
            # if available, otherwise fall back to the static design dict.
            try:
                self.view3d.update_decoupled_cradles(
                    decoupled_state_front if decoupled_state_front
                    else (self._front_decoupled or None),
                    decoupled_state_rear  if decoupled_state_rear
                    else (self._rear_decoupled  or None))
            except Exception:
                pass

            # Heave 3rd element rendered as a SHOCK cylinder (not a line),
            # posed at the LIVE bar-side attach (htb_shock_*) so it compresses
            # with travel; falls back to the static design dict if the solver
            # pose wasn't available.
            try:
                self.view3d.update_heave_tbar_shock(
                    htb_shock_front if htb_shock_front else (self._front_heave or None),
                    htb_shock_rear  if htb_shock_rear  else (self._rear_heave  or None))
            except Exception:
                pass

            # ── Rack visual ───────────────────────────────────────────────────
            hp_world = self._all_corner_hp()
            rack_l = self._steered_hp(hp_world['FL'], rt_m, True)['tie_rod_inner']
            rack_r = self._steered_hp(hp_world['FR'], rt_m, True)['tie_rod_inner']
            self.view3d.update_rack(rack_l, rack_r)

            self.view3d.toggle_ground(self._car.get('show_ground', True))
            # Push current spring / damper OD (mm) into the renderer so the
            # cylinder mesh next built in update_scene uses the latest value.
            # When the shock-thickness view toggle is OFF, draw them as thin
            # (~2 mm) cylinders so they read as lines and declutter the view.
            try:
                _thick = self._car.get('show_shock_thickness', True)
                self.view3d.set_spring_dims(
                    self._car.get('spring_od_mm', 63.0) if _thick else 2.0,
                    self._car.get('damper_od_mm', 50.0) if _thick else 2.0)
                self.view3d.set_thickness(_thick)   # members / ARB / driveshaft too
            except Exception:
                pass
            # View mode + corner isolation MUST be set before update_scene so
            # the desaturation and single-corner hiding apply on this frame.
            try:
                self.view3d.set_view_mode(self._car.get('view_mode', 'normal'))
                self.view3d.set_isolate_corner(self._car.get('wheel_pkg_corner'))
                self.view3d.set_brakes(self._car.get('show_brakes', True))
                self.view3d.set_brake_dims(
                self._car.get('rotor_dia_mm', 240.0),
                # keep the drawn caliper on the SAME bolt line and angle the load
                # model uses, or the picture contradicts the arrows
                mount_height_mm=getattr(self._loads_panel.get_brake_params_front(),
                                        'caliper_mount_height_mm', 27.9),
                caliper_angle_deg=getattr(self._loads_panel.get_upright_params(),
                                          'caliper_angle_deg', 45.0),
                vertical_mounts=getattr(self._loads_panel.get_upright_params(),
                                        'caliper_vertical_mounts', True),
                rotor_thickness_mm=getattr(self._loads_panel.get_brake_params_front(),
                                           'rotor_thickness_mm', 6.35))
            except Exception:
                pass
            self.view3d.update_scene(corners_draw, arb_segs)

            # ── Rear driveshaft / differential package (rear-only, RWD) ──────
            # Build from the LIVE solved rear wheel_center + spin_axis carried
            # in corners_draw (ONE MODEL), so the shafts move with the uprights.
            _pkg = None
            try:
                import types as _types
                from vahan.driveshaft import package as _ds_package
                _rear_states = {
                    c['label']: _types.SimpleNamespace(
                        wheel_center=c['pts']['wheel_center'],
                        spin_axis=c['spin_axis'])
                    for c in corners_draw if c['label'] in ('RL', 'RR')
                    and 'wheel_center' in c['pts']}
                _pkg = (_ds_package(self._car, _rear_states)
                        if len(_rear_states) == 2 else None)
                _iso = self._car.get('wheel_pkg_corner')
                _only = _iso if _iso in ('RL', 'RR') else None
                _show_ds = (self._car.get('show_driveshaft', True)
                            and (_iso is None or _iso in ('RL', 'RR')))
                self.view3d.set_driveshaft_package(_pkg, show=_show_ds, only=_only)
            except Exception:
                pass

            # ── View mode (normal / load / interference) + clash highlight ───
            try:
                from vahan.interference import clashes as _clashfn
                _mode = self._car.get('view_mode', 'normal')
                _clash_segs = []
                if _mode == 'interference' and _pkg is not None:
                    _TR = 0.008
                    _dr = 0.5 * float(self._car.get('driveshaft_dia_mm', 25.4)) / 1000.0
                    for c in corners_draw:
                        if c['label'] not in ('RL', 'RR'):
                            continue
                        pp = c['pts']
                        if not all(k in pp for k in ('uca_front', 'lca_front',
                                                     'pushrod_inner', 'tie_rod_inner')):
                            continue
                        def _P(k):
                            return np.asarray(pp[k], float)
                        mem = [
                            {'name': 'upper arm front', 'a': _P('uca_front'), 'b': _P('uca_outer'), 'r': _TR},
                            {'name': 'upper arm rear',  'a': _P('uca_rear'),  'b': _P('uca_outer'), 'r': _TR},
                            {'name': 'lower arm front', 'a': _P('lca_front'), 'b': _P('lca_outer'), 'r': _TR},
                            {'name': 'lower arm rear',  'a': _P('lca_rear'),  'b': _P('lca_outer'), 'r': _TR},
                            {'name': 'tie / toe rod',   'a': _P('tie_rod_inner'),  'b': _P('tie_rod_outer'), 'r': _TR},
                            {'name': 'pushrod',         'a': _P('pushrod_outer'), 'b': _P('pushrod_inner'), 'r': _TR},
                        ]
                        _seg = _pkg.get(c['label'])
                        if _seg is not None:
                            mem.append({'name': 'driveshaft', 'a': np.asarray(_seg['inner'], float),
                                        'b': np.asarray(_seg['outer'], float), 'r': _dr})
                        _byn = {m['name']: (m['a'], m['b']) for m in mem}
                        for cl in _clashfn(mem):
                            _clash_segs.append(_byn[cl['a']])
                            _clash_segs.append(_byn[cl['b']])
                self.view3d.set_clashes(_clash_segs)
            except Exception:
                pass

            # ── Load mode: force-vector arrows at the load points ────────────
            try:
                if self._car.get('view_mode', 'normal') == 'load':
                    from gui.wheel_package import load_arrows
                    lat_g = self._dynamics_panel._lat_g.value()
                    lon_g = self._dynamics_panel._lon_g.value()
                    vm = self._car.get('load_vec_mode', 'resultant')
                    only = self._car.get('wheel_pkg_corner') or None
                    self.view3d.set_load_vectors(
                        load_arrows(self, lat_g, lon_g, mode=vm, only_corner=only))
                else:
                    self.view3d.set_load_vectors([])
            except Exception:
                self.view3d.set_load_vectors([])
            self._frame_overlay(corners_draw)
            self._update_dimension_readouts()

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
            self.view3d.set_cg_visible(self._show_cg)

            # ── Pitch axis ────────────────────────────────────────────────
            # The pitch axis is the LATERAL (X-direction) line about which
            # the sprung mass pitches under longitudinal acceleration.
            # We use the canonical Milliken definition: a lateral line
            # at the CG's longitudinal position, at the height of the
            # roll axis interpolated to the CG.
            #
            #   pitch axis Y  =  Y_CG
            #   pitch axis Z  =  Z_RC_front + (Z_RC_rear − Z_RC_front)
            #                       · (Y_CG − Y_front_RC) / (Y_rear_RC − Y_front_RC)
            #
            # Pitch moment about this axis = m_s · a_x · (h_cg − h_pitch_axis),
            # the exact symmetric counterpart of the roll-moment formula.
            # This is what the dynamics solver implicitly uses when computing
            # pitch angle, longitudinal load transfer, and anti-dive effects.
            pitch_center = None
            if (front_rc is not None and rear_rc is not None):
                cg_y = self._car.get('cg_y_mm', 845.) / 1000.
                y_f, z_f = float(front_rc[1]), float(front_rc[2])
                y_r, z_r = float(rear_rc[1]),  float(rear_rc[2])
                dy = y_r - y_f
                if abs(dy) > 1e-6:
                    t = (cg_y - y_f) / dy   # fraction along roll axis
                    z_pa = z_f + (z_r - z_f) * t
                else:
                    z_pa = (z_f + z_r) / 2.0
                pitch_center = np.array([0.0, cg_y, float(z_pa)], float)

            # Half-width: use average front/rear track so the line spans the car
            track_hw = (self._car.get('track_f_mm', 1222.) +
                        self._car.get('track_r_mm', 1200.)) / 4000.
            if self._show_pitch_axis and pitch_center is not None:
                self.view3d.update_pitch_axis(pitch_center, half_width=track_hw)
            self.view3d.set_pitch_axis_visible(
                self._show_pitch_axis and pitch_center is not None)

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

            # ── LIVE topology-element values at the CURRENT pose ───────────
            # Single-model: the Live Values table's topology rows (heave/roll
            # spring, 3rd spring, T-bar twist, corner coil) track the slider,
            # computed from the same solvers that drive the 3D + graphs.
            try:
                self._inject_live_topology_values(all_corner_values, travels)
            except Exception:
                pass

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
        # Shifts every rear-axle dict (corner + ARB + heave + decoupled)
        # so the whole subassembly moves together.
        das = (params.get('axle_spacing_mm', old.get('axle_spacing_mm', 1537.))
               - old.get('axle_spacing_mm', old.get('wheelbase_mm', 1537.))) / 1000.
        self._shift_axle_y(is_front=False, dy_m=das)

        # ── wheelbase delta → dynamics only, NO hardpoint shift ───────────
        # Wheelbase = contact-patch distance, used for load transfer,
        # Ackermann, understeer gradient, etc.  Does not move geometry.

        # ── track delta → shift outboard pickups + wheel_center in X ──────
        # "Outboard" = the upright pickup points and wheel centre.
        # By default INBOARD chassis mounts stay fixed (bolted to frame),
        # so control arms get longer/steeper as the user widens track.
        # If the user ticked "Track change also pushes inboard pickups"
        # in the CarPanel, ALSO shift the inboard pickups by the same Δx
        # so the whole suspension subassembly translates laterally and
        # the control arms keep their length.
        _OUTBOARD = {'uca_outer', 'lca_outer', 'tie_rod_outer',
                     'wheel_center', 'pushrod_outer'}
        _INBOARD  = {'uca_front', 'uca_rear', 'lca_front', 'lca_rear',
                     'tie_rod_inner'}
        push_inboard = bool(params.get('track_pushes_inboard', False))
        shift_keys = _OUTBOARD | _INBOARD if push_inboard else _OUTBOARD

        dt_f = (params['track_f_mm'] - old['track_f_mm']) / 2000.  # half-track Δ (m)
        if abs(dt_f) > 1e-9:
            dx = np.array([dt_f, 0., 0.])
            for k in shift_keys:
                if k in self._front_hp:
                    self._front_hp[k] = self._front_hp[k] + dx
        dt_r = (params['track_r_mm'] - old['track_r_mm']) / 2000.
        if abs(dt_r) > 1e-9:
            dx = np.array([dt_r, 0., 0.])
            for k in shift_keys:
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
        self._front_hp_panel.refresh(self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
        self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)

        # ── single source of truth: tire radius ──────────────────────────
        # The tire size lives ONCE, in CarParams (tire_outer_dia_mm) — it
        # drives the 3D tire, the kinematic contact patch AND the dynamics/
        # brakes/loads radius.  Previously the DynamicsPanel had its own
        # 'Tire radius' the user could set to a DIFFERENT value (silent
        # over-constraint).  Keep that spinbox as a synced display.
        try:
            r_mm = float(params.get('tire_outer_dia_mm', 406.4)) / 2.0
            sb = self._dynamics_panel._tire_radius
            if abs(sb.value() - r_mm) > 1e-6:
                sb.blockSignals(True); sb.setValue(r_mm); sb.blockSignals(False)
        except Exception:
            pass

        self._rebuild_solvers()
        self._run_sweep()
        self._update_3d()
        self._update_min_turn_radius()

        # Update dynamics readout (weight distribution, etc.)
        self._dynamics_panel._on_driving_changed()

    def _on_hp(self, hp_dict: dict, axle: str, category: str = 'corner'):
        """Receive a table-driven hardpoint edit and route it to the right
        stored dict on this MainWindow.

        Args:
            hp_dict: new {key: 3-vector} for the edited category — overrides
                     the existing dict completely for that category.
            axle:    'front' or 'rear'.
            category: 'corner' | 'arb' | 'heave' | 'decoupled'.  Tells
                     us which stored dict to overwrite.  Default 'corner'
                     for legacy callers that don't supply a category.
        """
        new_dict = {k: np.asarray(v, float).copy() for k, v in hp_dict.items()}
        if axle == 'front':
            target_map = {
                'corner':    '_front_hp',
                'arb':       '_front_arb',
                'heave':     '_front_heave',
                'decoupled': '_front_decoupled',
            }
        else:
            target_map = {
                'corner':    '_rear_hp',
                'arb':       '_rear_arb',
                'heave':     '_rear_heave',
                'decoupled': '_rear_decoupled',
            }
        attr = target_map.get(category, target_map['corner'])
        setattr(self, attr, new_dict)

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
        # Mirror the selection into the Direct Edit panel (search ALL
        # category dicts — ARB/heave/cradle points are pickable too).
        try:
            self._direct_edit_panel.set_selection(name, corner)
            _, _, d = self._find_hp_dict(name, corner)
            if d is not None and name in d:
                self._direct_edit_panel.set_position(
                    np.asarray(d[name], float) * 1000.0)
            self._update_baseline_delta()
        except Exception:
            pass
        if self._edit_mode:
            self._show_edit_status(name, corner)

    # ── Direct-edit mode ─────────────────────────────────────────────────────
    #
    # When `_edit_mode` is on, the 3D viewer captures keyboard focus and the
    # WASD/QE keys nudge the selected hardpoint by `_edit_increment_mm`.
    # We always edit the stored FL/RL data — FR/RR are mirror views of the
    # same underlying point.  World-frame delta is applied directly to the
    # stored value, and the mirror is regenerated on the next render pass.

    def _on_toggle_edit_mode(self, enabled: bool):
        self._edit_mode = bool(enabled)
        self.view3d.set_edit_mode(self._edit_mode)
        app = QApplication.instance()
        if enabled:
            app.installEventFilter(self)
            self.statusBar().showMessage(
                'EDIT MODE: pick a hardpoint from the panel, then WASD/QE moves.  '
                '1-6 = step.  Ctrl+Z = undo.', 0)
            self.view3d.native.setFocus()
        else:
            app.removeEventFilter(self)
            self.statusBar().clearMessage()

    # ── Global key capture for edit-mode keystrokes ──────────────────────
    # Stored as plain ints because PyQt6 strict-enum mode can make
    # `event.key() == Qt.Key.Key_W` fail.  event.key() always returns int.
    _EDIT_KEYS = frozenset({
        int(Qt.Key.Key_W), int(Qt.Key.Key_A), int(Qt.Key.Key_S), int(Qt.Key.Key_D),
        int(Qt.Key.Key_Q), int(Qt.Key.Key_E),
        int(Qt.Key.Key_1), int(Qt.Key.Key_2), int(Qt.Key.Key_3),
        int(Qt.Key.Key_4), int(Qt.Key.Key_5), int(Qt.Key.Key_6),
    })
    def eventFilter(self, obj, event):
        if not self._edit_mode or event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(obj, event)
        # Don't hijack keys while the user is typing in a spinbox / text field
        from PyQt6.QtWidgets import (QAbstractSpinBox, QLineEdit,
                                      QPlainTextEdit, QTextEdit)
        fw = QApplication.focusWidget()
        if isinstance(fw, (QAbstractSpinBox, QLineEdit,
                           QPlainTextEdit, QTextEdit)):
            return super().eventFilter(obj, event)
        key_int = int(event.key())
        if key_int not in self._EDIT_KEYS:
            return super().eventFilter(obj, event)
        try:
            self.view3d._qt_keypress(event)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'EDIT error: {e}', 4000)
        return True   # consumed

    def _on_edit_increment_changed(self, mm: float):
        """Fired when 1-6 keys change the step inside the 3D viewer."""
        self._edit_increment_mm = float(mm)
        self._direct_edit_panel.set_step(mm)
        name, corner = self.view3d.get_selection()
        sel_str = f'{corner}.{name}' if name and corner else '(no selection)'
        self.statusBar().showMessage(
            f'EDIT  |  step Δ = {mm:.2f} mm  |  selected: {sel_str}', 0)

    def _on_panel_hp_selected(self, hp_name: str, corner: str):
        """User picked a hardpoint from the edit-panel list."""
        self.view3d.set_selection(hp_name, corner)
        is_front = corner in ('FL', 'FR')
        # Search ALL category dicts (corner / ARB / heave / decoupled) — the
        # panel list is the union of them, so an ARB or central-T-bar point
        # must show its position too (was: corner dict only → blank readout).
        _, _, d = self._find_hp_dict(hp_name, corner)
        if d is not None and hp_name in d:
            self._direct_edit_panel.set_position(
                np.asarray(d[hp_name], float) * 1000.0)
        else:
            self._direct_edit_panel.set_position(None)
        self._update_baseline_delta()
        # Also highlight in the hardpoint table panel for visual sync
        try:
            (self._front_hp_panel if is_front else self._rear_hp_panel) \
                .highlight_row(hp_name)
        except Exception:
            pass

    def _on_panel_step_changed(self, mm: float):
        """User clicked a step button in the edit panel."""
        self._edit_increment_mm = float(mm)
        self.view3d.set_increment_mm(mm)
        self.statusBar().showMessage(f'EDIT  |  step Δ = {mm:.2f} mm', 2000)

    def _on_panel_mirror_changed(self, enabled: bool):
        self._mirror_to_other_axle = bool(enabled)
        self.statusBar().showMessage(
            f'Mirror F↔R: {"ON" if enabled else "OFF"}', 2000)

    def _on_edit_apply(self):
        """Commit pending edits — clears undo history, the geometry becomes baseline."""
        n = len(self._edit_history)
        if n == 0:
            self.statusBar().showMessage('Nothing to apply', 2000)
            return
        self._edit_history.clear()
        self._direct_edit_panel.set_edit_count(0)
        self.statusBar().showMessage(
            f'Applied — {n} edit step{"s" if n != 1 else ""} committed. '
            f'Current geometry is now the baseline.', 4000)

    def _on_edit_discard(self):
        """Revert every pending edit (loops the undo stack until empty)."""
        if not self._edit_history:
            self.statusBar().showMessage('Nothing to discard', 2000)
            return
        n = len(self._edit_history)
        while self._edit_history:
            self._undo_edit()
        self._direct_edit_panel.set_edit_count(0)
        self.statusBar().showMessage(
            f'Discarded {n} edit step{"s" if n != 1 else ""} — back to baseline.', 4000)

    def _hp_dicts_for_axle(self, is_front: bool) -> list[tuple[str, dict]]:
        """Return [(category, hp_dict), ...] for all per-axle hardpoint dicts
        on the chosen axle.  Used by direct edit so any hardpoint — corner,
        ARB/T-bar, heave bracket, decoupled cradle — is editable."""
        if is_front:
            return [
                ('corner',    self._front_hp),
                ('arb',       self._front_arb),
                ('heave',     self._front_heave),
                ('decoupled', self._front_decoupled),
            ]
        return [
            ('corner',    self._rear_hp),
            ('arb',       self._rear_arb),
            ('heave',     self._rear_heave),
            ('decoupled', self._rear_decoupled),
        ]

    def _refresh_hp_names(self):
        """Direct-edit hardpoint list = union of EVERY hardpoint dict on both
        axles (corner + ARB + heave + decoupled) so EVERY point is adjustable.
        Called on startup, config load, and topology change."""
        names = []
        for is_front in (True, False):
            for _cat, d in self._hp_dicts_for_axle(is_front):
                names += list(d.keys())
        self._direct_edit_panel.set_hp_names(list(dict.fromkeys(names)))

    def _find_hp_dict(self, hp_name: str, corner: str):
        """Locate which axle dict actually contains `hp_name`.
        Returns (is_front, category, dict) or (None, None, None) if not found.
        Searches corner→arb→heave→decoupled on the matching axle first, then
        falls back to the other axle so e.g. tbar_arm_end can be edited
        whichever corner the user picked.
        """
        is_front_pref = corner in ('FL', 'FR')
        for is_front_try in (is_front_pref, not is_front_pref):
            for cat, d in self._hp_dicts_for_axle(is_front_try):
                if hp_name in d:
                    return is_front_try, cat, d
        return None, None, None

    def _sync_track_from_hardpoints(self):
        """Geometry is the single source of truth for track width: keep the
        cached track_f_mm / track_r_mm in lock-step with the live wheel-centre
        hardpoints so a WASD / Direct-Edit nudge of a wheel flows straight into
        the dynamics (LLTD, load transfer), loads, and Ackermann — every
        consumer pulls track from self._car at compute time.

        Track = 2 x |wheel_center.X|  (only FL/RL are stored; FR/RR are
        X-mirrors).  A missing / non-finite wheel centre leaves the dialed-in
        value untouched.

        NOTE: this updates the cached value the solvers read, NOT the Car-panel
        spin-box.  The panel field stays the *dialed* dimension; the dimensions
        -apply path (which rescales the whole wheel ring) is still where you set
        track by number.  Touching the spin-box here would feed that path's
        non-idempotent rescale and compound the track on the next apply.
        """
        try:
            wc_f = self._front_hp.get('wheel_center')
            if wc_f is not None and np.all(np.isfinite(wc_f)) and abs(wc_f[0]) > 1e-6:
                self._car['track_f_mm'] = 2.0 * abs(float(wc_f[0])) * 1000.0
            wc_r = self._rear_hp.get('wheel_center')
            if wc_r is not None and np.all(np.isfinite(wc_r)) and abs(wc_r[0]) > 1e-6:
                self._car['track_r_mm'] = 2.0 * abs(float(wc_r[0])) * 1000.0
        except Exception:
            pass

    # ── Designability helpers: typed positions, constraints, baseline ─────
    # Slide-along-link partners: in 'Link' constraint mode, WASD movement is
    # projected onto the line from the point to its partner (the member it
    # physically belongs to), so the point slides along its own link.
    _LINK_PARTNER = {
        'pushrod_outer': 'pushrod_inner', 'pushrod_inner': 'pushrod_outer',
        'tie_rod_outer': 'tie_rod_inner', 'tie_rod_inner': 'tie_rod_outer',
        'rocker_spring_pt': 'spring_chassis_pt',
        'spring_chassis_pt': 'rocker_spring_pt',
        'damper_outer_pt': 'damper_chassis_pt',
        'damper_chassis_pt': 'damper_outer_pt',
        'htb_coil_rocker': 'htb_coil_chassis',
        'htb_coil_chassis': 'htb_coil_rocker',
        'htb_drop_foot': 'htb_arm_tip', 'htb_arm_tip': 'htb_drop_foot',
        'htb_pushrod_inner': 'pushrod_outer',
        'heave_spring_tbar_pt': 'heave_spring_chassis_pt',
        'heave_spring_chassis_pt': 'heave_spring_tbar_pt',
        'arb_drop_top': 'arb_arm_end', 'arb_arm_end': 'arb_drop_top',
        'uca_outer': 'uca_front', 'lca_outer': 'lca_front',
        'pushrod_inner_left': 'pushrod_outer',
        'heave_damper_left': 'heave_damper_right',
        'heave_damper_right': 'heave_damper_left',
        'roll_damper_left': 'roll_damper_right',
        'roll_damper_right': 'roll_damper_left',
    }

    def _on_constraint_mode(self, mode: str):
        self._edit_constraint = mode
        self._direct_edit_panel.set_constraint_mode(mode)
        self.statusBar().showMessage(
            {'free': 'Nudge: FREE (world axes)',
             'link': 'Nudge: LINK-constrained — point slides along its member',
             'plane': 'Nudge: PLANE-constrained — point stays in the rocker/'
                      'bellcrank plane'}.get(mode, mode), 4000)

    def _edit_plane_normal(self, is_front: bool):
        """Unit normal of the actuation plane for constraint mode 'plane':
        the corner rocker plane when it exists, else the central T-bar
        bellcrank plane, else the decoupled left-bellcrank plane."""
        hp = self._front_hp if is_front else self._rear_hp
        heave = self._front_heave if is_front else self._rear_heave
        deco = self._front_decoupled if is_front else self._rear_decoupled
        for d, ks in ((hp, ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')),
                      (heave, ('htb_rocker_pivot', 'htb_pushrod_inner',
                               'htb_drop_foot')),
                      (deco, ('rocker_pivot_left', 'pushrod_inner_left',
                              'heave_damper_left'))):
            if d and all(k in d for k in ks):
                p0 = np.asarray(d[ks[0]], float)
                n = np.cross(np.asarray(d[ks[1]], float) - p0,
                             np.asarray(d[ks[2]], float) - p0)
                nn = float(np.linalg.norm(n))
                if nn > 1e-12:
                    return n / nn
        return None

    def _constrain_delta(self, hp_name: str, corner: str,
                         delta: np.ndarray) -> np.ndarray:
        """Project a WASD nudge delta per the active constraint mode."""
        if self._edit_constraint == 'link':
            partner = self._LINK_PARTNER.get(hp_name)
            if partner:
                _, _, d_self = self._find_hp_dict(hp_name, corner)
                _, _, d_part = self._find_hp_dict(partner, corner)
                if (d_self is not None and d_part is not None
                        and partner in d_part):
                    u = (np.asarray(d_part[partner], float)
                         - np.asarray(d_self[hp_name], float))
                    nu = float(np.linalg.norm(u))
                    if nu > 1e-9:
                        u /= nu
                        return float(np.dot(delta, u)) * u
            self.statusBar().showMessage(
                f'LINK constraint: no partner member for "{hp_name}" — '
                'moved free', 3000)
        elif self._edit_constraint == 'plane':
            is_front = corner in ('FL', 'FR')
            n = self._edit_plane_normal(is_front)
            if n is not None:
                return delta - float(np.dot(delta, n)) * n
            self.statusBar().showMessage(
                'PLANE constraint: no actuation plane on this axle — '
                'moved free', 3000)
        return delta

    def _on_position_typed(self, x_mm: float, y_mm: float, z_mm: float):
        """Direct Edit XYZ spinbox committed — move the selected point to the
        ABSOLUTE coordinate, through the normal delta path (undo / guards /
        live graphs all apply).  Constraint modes do NOT apply to typed
        coordinates: typing is explicit."""
        name, corner = self.view3d.get_selection()
        if not name:
            name = (self._direct_edit_panel._hp_list.currentItem().text()
                    if self._direct_edit_panel._hp_list.currentItem() else None)
            corner = self._direct_edit_panel.current_corner()
        if not name:
            return
        _, _, d = self._find_hp_dict(name, corner)
        if d is None:
            return
        target = np.array([x_mm, y_mm, z_mm], float) / 1000.0
        delta = target - np.asarray(d[name], float)
        if float(np.linalg.norm(delta)) < 1e-9:
            return
        self._on_hp_move(name, corner, delta, apply_constraint=False)

    def _set_baseline(self):
        """Snapshot the current geometry: grey ghost in the 3D + Δ readout."""
        snap = {}
        for is_front in (True, False):
            for cat in ('corner', 'arb', 'heave', 'decoupled'):
                d = self._axle_dict_by_category(is_front, cat)
                if d:
                    for k, v in d.items():
                        snap[(is_front, k)] = np.array(v, copy=True)
        self._baseline_geo = snap
        # Freeze the CURRENT rendered linkage as the ghost geometry.
        try:
            self.view3d.set_ghost(self.view3d.capture_ghost())
        except Exception:
            pass
        self._update_baseline_delta()
        self.statusBar().showMessage(
            f'Baseline set: {len(snap)} points snapshotted — grey ghost shows '
            'it; Δ readout measures against it', 5000)

    def _on_ghost_toggled(self, visible: bool):
        try:
            self.view3d.set_ghost_visible(visible)
        except Exception:
            pass

    def _update_baseline_delta(self):
        """Refresh the Direct Edit Δ-from-baseline readout for the selection."""
        try:
            name, corner = self.view3d.get_selection()
            if not name or self._baseline_geo is None:
                self._direct_edit_panel.set_baseline_delta(None)
                return
            is_front = (corner or 'FL') in ('FL', 'FR')
            base = self._baseline_geo.get((is_front, name))
            _, _, d = self._find_hp_dict(name, corner or 'FL')
            if base is None or d is None or name not in d:
                self._direct_edit_panel.set_baseline_delta(None)
                return
            delta_mm = (np.asarray(d[name], float) - base) * 1000.0
            self._direct_edit_panel.set_baseline_delta(tuple(delta_mm))
        except Exception:
            self._direct_edit_panel.set_baseline_delta(None)

    def _on_hp_move(self, hp_name: str, corner: str, delta_xyz: np.ndarray,
                    apply_constraint: bool = True):
        """Apply a world-frame delta to ANY stored hardpoint (corner / ARB /
        T-bar / heave / decoupled cradle), regardless of which dict it lives in.

        If `_mirror_to_other_axle` is ON, the same delta also applies to the
        matching hardpoint name on the opposite axle (in the same category).

        Poka-yoke: after the move, the damper's spring length is checked
        against its physical bounds (set on MotionPanel).  Out-of-range
        edits are rolled back with a status-bar warning.
        """
        delta_arr = np.asarray(delta_xyz, float)
        # Constrained nudge (Link / Plane modes) — WASD deltas only; typed
        # absolute coordinates bypass this (apply_constraint=False).
        if apply_constraint and self._edit_constraint != 'free':
            delta_arr = self._constrain_delta(hp_name, corner, delta_arr)
            if float(np.linalg.norm(delta_arr)) < 1e-12:
                return   # delta entirely outside the allowed direction

        # Locate the primary target dict (whichever axle + category owns
        # this hp_name, with corner-axle preference).
        is_front, cat, primary = self._find_hp_dict(hp_name, corner)
        if primary is None:
            self.statusBar().showMessage(
                f'EDIT: hardpoint "{hp_name}" not found in any axle dict', 3000)
            return

        targets: list[tuple[bool, str, dict]] = [(is_front, cat, primary)]
        if self._mirror_to_other_axle:
            for c, d in self._hp_dicts_for_axle(not is_front):
                if c == cat and hp_name in d:
                    targets.append((not is_front, c, d))
                    break

        # Snapshot for undo BEFORE mutating
        snapshot = [
            {'is_front': f, 'category': c, 'hp_name': hp_name,
             'prev': np.array(d[hp_name], copy=True)}
            for f, c, d in targets
        ]

        # Snapshot pre-edit spring lengths so the damper-bounds guard can
        # distinguish a NEW violation (block) from a pre-existing out-of-range
        # default (allow — the user must be able to nudge the point to tune
        # the geometry back into range).  Solvers still reflect the OLD
        # geometry at this point, before the delta is applied.
        bounds_baseline = {}
        if cat == 'corner':
            for _f, _c, _d in targets:
                bounds_baseline[_f] = self._spring_length_for_axle(_f)

        # Apply the delta
        for _f, _c, d in targets:
            d[hp_name] = np.array(d[hp_name]) + delta_arr

        # Rebuild solvers + check damper bounds (only meaningful for corner
        # edits — ARB / heave / cradle don't change the corner kinematics)
        self._rebuild_solvers()
        block_reason = ''
        if cat == 'corner':
            block_reason = self._check_damper_bounds_after_edit(
                [(f, d) for f, _c, d in targets], baseline=bounds_baseline)
        if block_reason:
            for entry in snapshot:
                hd = self._axle_dict_by_category(entry['is_front'], entry['category'])
                if hd is not None:
                    hd[entry['hp_name']] = entry['prev']
            self._rebuild_solvers()
            self._update_3d()
            self.statusBar().showMessage(f'BLOCKED — {block_reason}', 5000)
            return

        # Commit to undo history
        self._edit_history.append(snapshot)
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)

        # Geometry changed — resync track width so a wheel nudge reaches the
        # dynamics / loads / Ackermann (LLTD etc.), not just the 3D view.
        self._sync_track_from_hardpoints()

        # Refresh hardpoint table panels (corner + ARB columns)
        for f, _c, _d in targets:
            try:
                panel = self._front_hp_panel if f else self._rear_hp_panel
                hpd   = self._front_hp       if f else self._rear_hp
                arb   = self._front_arb      if f else self._rear_arb
                panel.refresh(hpd, arb)
                panel.highlight_row(hp_name)
            except Exception:
                pass

        # Refresh 3D + live metrics
        self._update_3d()
        # ONE MODEL: the edited hardpoint drives the kinematic graphs too —
        # re-sweep them (debounced) so they track the nudge, not just the 3D.
        self._edit_sweep_timer.start()

        # Update Direct Edit panel display
        new_pos = primary[hp_name] * 1000.0
        self._direct_edit_panel.set_position(tuple(new_pos))
        self._direct_edit_panel.set_edit_count(len(self._edit_history))
        self._update_baseline_delta()

        mirror_tag = '  (mirrored F↔R)' if self._mirror_to_other_axle else ''
        side = 'F' if is_front else 'R'
        self.statusBar().showMessage(
            f'EDIT  |  {side}.{cat}.{hp_name} → '
            f'({new_pos[0]:+8.2f}, {new_pos[1]:+8.2f}, {new_pos[2]:+8.2f}) mm  '
            f'|  step Δ = {self._edit_increment_mm:.2f} mm{mirror_tag}', 0)

    def _axle_dict_by_category(self, is_front: bool, category: str):
        """Helper for undo — return the dict for (axle, category)."""
        if is_front:
            return {'corner': self._front_hp, 'arb': self._front_arb,
                    'heave': self._front_heave,
                    'decoupled': self._front_decoupled}.get(category)
        return {'corner': self._rear_hp, 'arb': self._rear_arb,
                'heave': self._rear_heave,
                'decoupled': self._rear_decoupled}.get(category)

    def _shift_axle_y(self, is_front: bool, dy_m: float) -> None:
        """Translate every hardpoint on the given axle by dy_m metres in Y.

        This includes ALL four per-axle dicts: corner HPs, ARB (or T-bar)
        hardware, heave 3rd-element bracket, and decoupled cradle.  Any of
        them may be empty depending on the active topology — we skip
        empty dicts silently.

        Used by:
          * the startup wizard's axle-spacing input (shifts the rear axle
            so the entire rear suspension subassembly moves to match the
            user's chosen wheelbase / axle spacing)
          * the CarPanel's live axle_spacing_mm spinbox (same shift on
            every change, so editing the value after the wizard runs has
            the same visible effect)

        Previous bug: only `_rear_hp` (and in `_on_car` also `_rear_arb`)
        were shifted — heave bracket and decoupled cradle stayed at the
        original hard-coded Y, leaving them visibly disconnected from the
        wheel/rocker block in the 3D view.
        """
        if abs(dy_m) < 1e-12:
            return
        dy = np.array([0., dy_m, 0.])
        if is_front:
            dicts = (self._front_hp, self._front_arb,
                     self._front_heave, self._front_decoupled)
        else:
            dicts = (self._rear_hp, self._rear_arb,
                     self._rear_heave, self._rear_decoupled)
        for d in dicts:
            for k in list(d):
                d[k] = np.asarray(d[k], float) + dy

    # ── Plane-tilt handler ──────────────────────────────────────────────
    # Hardpoints that live on the actuation plane and therefore move
    # together when the plane is rotated.  Categorised by which dict they
    # live in so we can pull them from the right axle bucket.
    _PLANE_KEYS_CORNER = (
        'pushrod_outer', 'pushrod_inner',
        'rocker_pivot', 'rocker_spring_pt', 'rocker_axis_pt',
        'spring_chassis_pt',
        # Decoupled twin rocker — also planar
        'roll_rocker_pivot', 'roll_rocker_spring_pt',
        'roll_rocker_axis_pt', 'roll_spring_chassis_pt',
        # Direct-damper rendering uses these as the endpoints
        'damper_chassis_pt', 'damper_outer_pt',
    )
    _PLANE_KEYS_ARB = (
        'arb_drop_top', 'arb_arm_end',
        # T-bar uses tbar_* names — also planar (they sit on the
        # drop-link / lever path that shares the rocker plane)
        'tbar_arm_end', 'tbar_drop_top',
    )

    # Curated pivot candidates for the plane-tilt control.  The dropdown is
    # filtered to those that actually exist on the chosen axle (poka-yoke):
    # pushrod / pullrod cars expose the rocker + pushrod points; a DIRECT axle
    # exposes its damper endpoints (so the damper line can still be tilted —
    # the rotation handler already moves damper_* points); a decoupled axle
    # exposes pushrod_outer.
    _PLANE_PIVOT_CANDIDATES = (
        'rocker_pivot', 'rocker_spring_pt', 'pushrod_inner', 'pushrod_outer',
        'spring_chassis_pt', 'damper_chassis_pt', 'damper_outer_pt',
    )

    def _refresh_plane_pivots(self, axle: str | None = None):
        """Repopulate the plane-tilt pivot dropdown with the pivots that exist
        on the chosen axle's topology, so DIRECT / decoupled cars get valid
        pivots and the user can never pick one the axle lacks.  Called on
        topology change and whenever the plane-axle selector changes."""
        try:
            if axle is None:
                axle = self._direct_edit_panel.current_plane_axle()
            is_front = str(axle).lower().startswith('f')
            corner_dict = self._front_hp if is_front else self._rear_hp
            valid = [p for p in self._PLANE_PIVOT_CANDIDATES if p in corner_dict]
            self._direct_edit_panel.set_plane_pivots(valid)
        except Exception:
            pass

    def _on_plane_tilt(self, axle: str, axis: str, deg: float, pivot_key: str):
        """Rotate every plane-resident hardpoint by `deg` about a line
        through `pivot_key` parallel to the chosen world axis (X/Y/Z).

        Recorded as ONE undo step so Ctrl+Z reverses the whole rotation.
        """
        is_front = (axle.lower().startswith('f'))
        # Find the pivot point in any of the per-axle dicts
        pivot = None
        for _cat, d in self._hp_dicts_for_axle(is_front):
            if pivot_key in d:
                pivot = np.asarray(d[pivot_key], float).copy()
                break
        if pivot is None:
            self.statusBar().showMessage(
                f'PLANE TILT — pivot hardpoint "{pivot_key}" not in '
                f'{"front" if is_front else "rear"} axle', 4000)
            return

        # Axis unit vector — canonical (world X/Y/Z) or derived from the
        # pushrod (chassis-end → wheel-end) for the "tilt about pushrod"
        # case.  Rotating about the pushrod with pivot ON the pushrod
        # line gives the design-intent move: pushrod itself stays put,
        # rocker / damper / drop-link swing around as one rigid plane.
        axis = axis.upper()
        axis_vec = None
        if axis == 'PUSHROD':
            # Need both pushrod endpoints on this axle
            corner_d = self._front_hp if is_front else self._rear_hp
            if ('pushrod_inner' in corner_d and 'pushrod_outer' in corner_d):
                v = (np.asarray(corner_d['pushrod_inner'], float)
                     - np.asarray(corner_d['pushrod_outer'], float))
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    axis_vec = v / n
            if axis_vec is None:
                self.statusBar().showMessage(
                    'PLANE TILT — no pushrod on this axle (DIRECT damper '
                    'topology); pick a world axis instead', 4000)
                return
        else:
            axis_vec = {'X': np.array([1., 0., 0.]),
                        'Y': np.array([0., 1., 0.]),
                        'Z': np.array([0., 0., 1.])}.get(axis)
        if axis_vec is None:
            self.statusBar().showMessage(
                f'PLANE TILT — bad axis "{axis}" (expected PUSHROD or X/Y/Z)',
                3000)
            return

        # Rotation matrix — Rodrigues formula works for any unit axis.
        # R = I + sin(θ)·K + (1−cos(θ))·K², where K is the skew-symmetric
        # cross-product matrix of the unit axis.
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        kx, ky, kz = axis_vec
        K = np.array([[ 0, -kz,  ky],
                      [ kz,  0, -kx],
                      [-ky, kx,   0]], float)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)

        # Collect targets across CORNER + ARB on the chosen axle
        plane_targets: list[tuple[str, dict, str]] = []   # (category, dict, key)
        corner_dict = self._front_hp if is_front else self._rear_hp
        arb_dict    = self._front_arb if is_front else self._rear_arb
        heave_dict  = self._front_heave if is_front else self._rear_heave
        deco_dict   = self._front_decoupled if is_front else self._rear_decoupled
        for k in self._PLANE_KEYS_CORNER:
            if k in corner_dict and k != pivot_key:
                plane_targets.append(('corner', corner_dict, k))
        for k in self._PLANE_KEYS_ARB:
            if k in arb_dict and k != pivot_key:
                plane_targets.append(('arb', arb_dict, k))
        # Heave bracket + decoupled cradle attach via the plane too —
        # rotate them with it.
        for k in list(heave_dict.keys()):
            if k != pivot_key:
                plane_targets.append(('heave', heave_dict, k))
        for k in list(deco_dict.keys()):
            if k != pivot_key:
                plane_targets.append(('decoupled', deco_dict, k))

        if not plane_targets:
            self.statusBar().showMessage(
                'PLANE TILT — no plane hardpoints found on this axle', 3000)
            return

        # Snapshot for undo — ONE bundled step
        snapshot = [
            {'is_front': is_front, 'category': cat,
             'hp_name': k, 'prev': np.array(d[k], copy=True)}
            for cat, d, k in plane_targets
        ]

        # Pre-edit damper length, so the bounds guard only blocks a tilt that
        # WORSENS an out-of-range condition (a pre-existing out-of-range
        # default must remain tiltable so the user can correct it).
        bounds_baseline = {is_front: self._spring_length_for_axle(is_front)}

        # Apply rotation: p' = pivot + R @ (p - pivot)
        for cat, d, k in plane_targets:
            p = np.asarray(d[k], float)
            d[k] = pivot + R @ (p - pivot)

        # Validate (corner edits → damper-bounds check)
        self._rebuild_solvers()
        block_reason = self._check_damper_bounds_after_edit(
            [(is_front, corner_dict)], baseline=bounds_baseline)
        if block_reason:
            for entry in snapshot:
                hd = self._axle_dict_by_category(entry['is_front'],
                                                  entry['category'])
                if hd is not None:
                    hd[entry['hp_name']] = entry['prev']
            self._rebuild_solvers()
            self._update_3d()
            self.statusBar().showMessage(f'BLOCKED — {block_reason}', 5000)
            return

        # Commit to undo history (ONE entry covering all rotated points)
        self._edit_history.append(snapshot)
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)

        # Refresh panels
        try:
            panel = self._front_hp_panel if is_front else self._rear_hp_panel
            panel.refresh(corner_dict, arb_dict)
        except Exception:
            pass
        self._update_3d()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))

        side = 'F' if is_front else 'R'
        self.statusBar().showMessage(
            f'PLANE TILT  |  {side} axle | {len(plane_targets)} pts rotated '
            f'{deg:+.2f}° about {axis} through {pivot_key}', 0)

    def _on_snap_actuation_to_plane(self, axle: str):
        """Project the whole actuation chain onto the rocker plate plane.

        Moves pushrod_outer / spring_chassis_pt / damper ends / rocker pin
        (via _enforce_actuation_coplanar) AND the ARB drop-link top
        (arb_drop_top) onto the plane, so the bellcrank stays a planar
        mechanism after manual nudges.  Recorded as ONE undo step.
        """
        is_front = axle.lower().startswith('f')
        hp = self._front_hp if is_front else self._rear_hp
        arb = self._front_arb if is_front else self._rear_arb
        need = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')
        if not all(k in hp and hp[k] is not None
                   and np.all(np.isfinite(hp[k])) for k in need):
            self.statusBar().showMessage(
                f'SNAP ACTUATION — {axle} axle has no rocker plane '
                '(direct / decoupled topology?)', 3000)
            return
        # Plane normal from the plate (pivot, pushrod_inner, rocker_spring_pt).
        p0 = np.asarray(hp['rocker_pivot'], float)
        n = np.cross(np.asarray(hp['pushrod_inner'], float) - p0,
                     np.asarray(hp['rocker_spring_pt'], float) - p0)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            self.statusBar().showMessage(
                'SNAP ACTUATION — plate plane degenerate (points colinear)', 4000)
            return
        n /= nn
        # Snapshot every point the snap will move, for one-shot undo.
        snap = []
        for k in ('pushrod_outer', 'spring_chassis_pt', 'damper_chassis_pt',
                  'damper_outer_pt', 'rocker_axis_pt'):
            if k in hp and hp[k] is not None and np.all(np.isfinite(hp[k])):
                snap.append({'is_front': is_front, 'category': 'corner',
                             'hp_name': k, 'prev': np.asarray(hp[k], float).copy()})
        if 'arb_drop_top' in arb and np.all(np.isfinite(arb['arb_drop_top'])):
            snap.append({'is_front': is_front, 'category': 'arb',
                         'hp_name': 'arb_drop_top',
                         'prev': np.asarray(arb['arb_drop_top'], float).copy()})
        # Apply: corner chain via the shared enforcer, ARB drop link by hand.
        self._enforce_actuation_coplanar(hp)
        if 'arb_drop_top' in arb and np.all(np.isfinite(arb['arb_drop_top'])):
            P = np.asarray(arb['arb_drop_top'], float)
            arb['arb_drop_top'] = P - float(np.dot(P - p0, n)) * n
        # Record undo + refresh.
        self._edit_history.append(snap)
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)
        try:
            panel = self._front_hp_panel if is_front else self._rear_hp_panel
            panel.refresh(hp, arb,
                          self._front_heave if is_front else self._rear_heave,
                          self._front_decoupled if is_front else self._rear_decoupled)
        except Exception:
            pass
        self._rebuild_solvers()
        self._update_3d()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))
        self.statusBar().showMessage(
            f'Snapped {len(snap)} actuation point(s) onto the {axle} rocker '
            'plane.', 4000)

    @staticmethod
    def _seg_seg_dist(p1, q1, p2, q2):
        """Minimum distance between two 3-D segments p1-q1 and p2-q2 (metres)."""
        p1, q1, p2, q2 = (np.asarray(x, float) for x in (p1, q1, p2, q2))
        d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
        a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
        if a < 1e-12 and e < 1e-12:
            return float(np.linalg.norm(r))
        if a < 1e-12:
            s = 0.0; t = float(np.clip(f / e, 0, 1))
        else:
            c = float(d1 @ r)
            if e < 1e-12:
                t = 0.0; s = float(np.clip(-c / a, 0, 1))
            else:
                b = float(d1 @ d2); den = a * e - b * b
                s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-9 else 0.0
                t = (b * s + f) / e
                if t < 0:
                    t = 0.0; s = float(np.clip(-c / a, 0, 1))
                elif t > 1:
                    t = 1.0; s = float(np.clip((b - c) / a, 0, 1))
        return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))

    def _frame_overlay(self, corners_draw):
        """Draw control arms / pushrod / tie-rod / rocker / ARB at real
        thickness, flag interferences (red), and show the rocker-pivot bearing
        clearance cylinder (yellow).  Gated by the Direct Edit 'frame' toggle.
        Parts are modelled as tubes; interference = segment-segment gap < sum of
        radii between two different parts on the same corner that don't share a
        joint.  The rocker plane bisects its tube extrudes (radius = thickness/2)."""
        panel = self._frame_panel
        try:
            if not panel.frame_enabled():
                self.view3d.set_frame_overlay([], [], [], False)
                return
            dims = panel.frame_dims()
        except Exception:
            self.view3d.set_frame_overlay([], [], [], False)
            return
        r_ctrl = dims.get('ctrl_arm_od', 19.0) / 2000.0
        r_arb  = dims.get('arb_od', 14.0) / 2000.0
        r_rock = dims.get('rocker_th', 8.0) / 2000.0
        r_push = dims.get('pushrod_od', 16.0) / 2000.0
        r_bear = dims.get('bearing_od', 38.1) / 2000.0
        bear_len = dims.get('bearing_len', 25.4) / 1000.0
        part_cyls = []; bearing_cyls = []; segs = []
        for c in corners_draw:
            pts = c['pts']; lbl = c['label']

            def add(part, a, b, r):
                if (a in pts and b in pts and np.all(np.isfinite(pts[a]))
                        and np.all(np.isfinite(pts[b]))):
                    p0 = np.asarray(pts[a], float); p1 = np.asarray(pts[b], float)
                    part_cyls.append((p0, p1, r)); segs.append((lbl, part, p0, p1, r))

            add('UCA', 'uca_front', 'uca_outer', r_ctrl)
            add('UCA', 'uca_rear', 'uca_outer', r_ctrl)
            add('LCA', 'lca_front', 'lca_outer', r_ctrl)
            add('LCA', 'lca_rear', 'lca_outer', r_ctrl)
            add('pushrod', 'pushrod_inner', 'pushrod_outer', r_push)
            add('tierod', 'tie_rod_inner', 'tie_rod_outer', r_push)
            add('rocker', 'rocker_pivot', 'pushrod_inner', r_rock)
            add('rocker', 'rocker_pivot', 'rocker_spring_pt', r_rock)
            add('rocker', 'rocker_pivot', 'arb_drop_top', r_rock)
            dt = pts.get('arb_drop_top'); ae = pts.get('arb_arm_end_world')
            if dt is not None and ae is not None and np.all(np.isfinite(dt)) and np.all(np.isfinite(ae)):
                p0 = np.asarray(dt, float); p1 = np.asarray(ae, float)
                part_cyls.append((p0, p1, r_arb)); segs.append((lbl, 'ARB', p0, p1, r_arb))
            pv = pts.get('rocker_pivot'); axp = pts.get('rocker_axis_pt')
            if pv is not None and axp is not None and np.all(np.isfinite(pv)) and np.all(np.isfinite(axp)):
                pv = np.asarray(pv, float); d = np.asarray(axp, float) - pv
                n = float(np.linalg.norm(d))
                if n > 1e-9:
                    u = d / n
                    bearing_cyls.append((pv - 0.5 * bear_len * u, pv + 0.5 * bear_len * u, r_bear))
        # interference: group segments into PARTS; skip part-pairs that share a
        # joint (they bolt together there, so touching is expected — not a clash).
        # Report the worst overlap per part-pair on the same corner.
        parts = {}; ends = {}
        for lbl, part, a, b, r in segs:
            k = (lbl, part)
            parts.setdefault(k, []).append((a, b, r))
            s = ends.setdefault(k, set())
            s.add(tuple(np.round(a, 4))); s.add(tuple(np.round(b, 4)))
        clash_cyls = []; reports = []
        keys = list(parts)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if k1[0] != k2[0] or (ends[k1] & ends[k2]):
                    continue
                worst = None
                for a1, b1, r1 in parts[k1]:
                    for a2, b2, r2 in parts[k2]:
                        gap = self._seg_seg_dist(a1, b1, a2, b2) - (r1 + r2)
                        if gap < 0 and (worst is None or gap < worst[0]):
                            worst = (gap, a1, b1, r1, a2, b2, r2)
                if worst:
                    g, a1, b1, r1, a2, b2, r2 = worst
                    clash_cyls += [(a1, b1, r1), (a2, b2, r2)]
                    reports.append(f"{k1[0]} {k1[1]}↔{k2[1]}: overlap {(-g)*1000:.1f}mm")
        self.view3d.set_frame_overlay(part_cyls, clash_cyls, bearing_cyls, True)
        try:
            panel.set_frame_readout('No interference' if not reports
                                    else '  |  '.join(reports[:6]))
        except Exception:
            pass

    def _shock_ends(self, hp):
        """(fixed_end_key, moving_chassis_key) for the damper, or (None, None)."""
        if 'rocker_spring_pt' in hp and 'spring_chassis_pt' in hp:
            return 'rocker_spring_pt', 'spring_chassis_pt'
        if 'damper_outer_pt' in hp and 'damper_chassis_pt' in hp:
            return 'damper_outer_pt', 'damper_chassis_pt'
        return None, None

    def _update_dimension_readouts(self):
        """Populate the panel's rack length + front/rear shock length boxes
        from the current geometry."""
        rack = None
        if 'tie_rod_inner' in self._front_hp:
            rack = abs(float(self._front_hp['tie_rod_inner'][0])) * 2000.0

        def shock(hp):
            fk, mk = self._shock_ends(hp)
            if fk is None:
                return None
            return float(np.linalg.norm(np.asarray(hp[mk], float)
                                        - np.asarray(hp[fk], float))) * 1000.0
        try:
            self._direct_edit_panel.set_dimensions(
                rack, shock(self._front_hp), shock(self._rear_hp))
        except Exception:
            pass

    def _commit_dim_edit(self, is_front, category, name, prev):
        """Record a one-point length edit for undo + refresh panels/solver/3D."""
        self._edit_history.append([{'is_front': is_front, 'category': category,
                                    'hp_name': name, 'prev': prev}])
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)
        try:
            panel = self._front_hp_panel if is_front else self._rear_hp_panel
            panel.refresh(self._front_hp if is_front else self._rear_hp,
                          self._front_arb if is_front else self._rear_arb,
                          self._front_heave if is_front else self._rear_heave,
                          self._front_decoupled if is_front else self._rear_decoupled)
        except Exception:
            pass
        self._rebuild_solvers()
        self._update_3d()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))

    def _on_rack_length(self, length_mm: float):
        """Front rack length (tip-to-tip inner tie-rod pickups): set
        tie_rod_inner X to +/- length/2; Y and Z (position) untouched."""
        hp = self._front_hp
        if 'tie_rod_inner' not in hp:
            return
        cur = np.asarray(hp['tie_rod_inner'], float)
        if abs(abs(cur[0]) * 2000.0 - length_mm) < 0.05:
            return
        new = cur.copy()
        new[0] = (length_mm / 2000.0) * (1.0 if cur[0] >= 0 else -1.0)
        hp['tie_rod_inner'] = new
        self._commit_dim_edit(True, 'corner', 'tie_rod_inner', cur)
        self.statusBar().showMessage(
            f'Front rack length set to {length_mm:.1f} mm '
            f'(tie_rod_inner X = {new[0]*1000:.1f} mm).', 4000)

    def _on_shock_length(self, axle: str, length_mm: float):
        """Damper mount-to-mount length: move the chassis end along the damper
        axis; the rocker/spring end stays put.  Position is unchanged."""
        is_front = axle.lower().startswith('f')
        hp = self._front_hp if is_front else self._rear_hp
        fk, mk = self._shock_ends(hp)
        if fk is None:
            self.statusBar().showMessage(
                f'SHOCK LENGTH — {axle} axle has no recognised damper ends', 3000)
            return
        fixed = np.asarray(hp[fk], float); mov = np.asarray(hp[mk], float)
        d = mov - fixed; n = float(np.linalg.norm(d))
        if n < 1e-9 or abs(n * 1000.0 - length_mm) < 0.05:
            return
        hp[mk] = fixed + (length_mm / 1000.0) * (d / n)
        self._commit_dim_edit(is_front, 'corner', mk, mov)
        self.statusBar().showMessage(
            f'{axle.capitalize()} shock length set to {length_mm:.1f} mm '
            f'({mk} moved along the damper axis).', 4000)

    def _on_snap_axis_to_normal(self, axle: str):
        """Recompute rocker_axis_pt = rocker_pivot + L · n_hat where n_hat
        is the unit normal to the rocker plane (defined by rocker_pivot,
        rocker_spring_pt, pushrod_inner) and L is the current pin length
        |rocker_axis_pt − rocker_pivot|.

        Use this after manually editing rocker_axis_pt if the pin
        direction drifted off the plane normal — physical rockers pivot
        about a pin that MUST be perpendicular to the plate.

        The edit is recorded as ONE undo step (only the rocker_axis_pt
        moves; the plane points stay put).  If the axle doesn't have a
        rocker (e.g. DIRECT damper), or any of the three reference
        points is missing, the operation is a no-op with a status msg.
        """
        is_front = axle.lower().startswith('f')
        hp = self._front_hp if is_front else self._rear_hp
        needed = ('rocker_pivot', 'rocker_axis_pt',
                  'rocker_spring_pt', 'pushrod_inner')
        missing = [k for k in needed if k not in hp]
        if missing:
            self.statusBar().showMessage(
                f'SNAP AXIS — {axle} axle missing: {", ".join(missing)} '
                f'(no rocker on this topology?)', 3000)
            return

        pivot = np.asarray(hp['rocker_pivot'],     float)
        spr   = np.asarray(hp['rocker_spring_pt'], float)
        pin   = np.asarray(hp['pushrod_inner'],    float)
        # Two in-plane vectors from the pivot to the other two plane points.
        v1 = spr - pivot
        v2 = pin - pivot
        n = np.cross(v1, v2)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-9:
            self.statusBar().showMessage(
                'SNAP AXIS — rocker_spring_pt, pushrod_inner are colinear '
                'with rocker_pivot; plane is degenerate', 4000)
            return
        n_hat = n / n_norm

        # Pin length = current |axis_pt − pivot|
        old_axis_pt = np.asarray(hp['rocker_axis_pt'], float).copy()
        L = float(np.linalg.norm(old_axis_pt - pivot))
        if L < 1e-9:
            # Fall back to a 25 mm pin if the axis pt was coincident
            L = 0.025
        # Choose the sign that keeps the new axis on the same side as the
        # old one (don't suddenly flip the pin direction).
        new_axis_pt_pos = pivot + L * n_hat
        new_axis_pt_neg = pivot - L * n_hat
        if (np.linalg.norm(new_axis_pt_pos - old_axis_pt)
            < np.linalg.norm(new_axis_pt_neg - old_axis_pt)):
            new_axis_pt = new_axis_pt_pos
        else:
            new_axis_pt = new_axis_pt_neg

        # Snapshot for one-shot undo
        snapshot = [{'is_front': is_front, 'category': 'corner',
                     'hp_name': 'rocker_axis_pt',
                     'prev': old_axis_pt}]
        hp['rocker_axis_pt'] = new_axis_pt
        self._edit_history.append(snapshot)
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)

        # Refresh + status
        try:
            panel = self._front_hp_panel if is_front else self._rear_hp_panel
            panel.refresh(hp, (self._front_arb if is_front
                               else self._rear_arb),
                          (self._front_heave if is_front
                           else self._rear_heave),
                          (self._front_decoupled if is_front
                           else self._rear_decoupled))
        except Exception:
            pass
        self._rebuild_solvers()
        self._update_3d()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))

        # Report angular deviation that was corrected (how off-normal was it?)
        old_dir = (old_axis_pt - pivot)
        old_dir = old_dir / (np.linalg.norm(old_dir) + 1e-12)
        dot = abs(float(np.dot(old_dir, n_hat)))
        ang_off_deg = float(np.degrees(np.arccos(min(1.0, dot))))
        self.statusBar().showMessage(
            f'SNAP AXIS  |  {axle} | corrected {ang_off_deg:.2f}° off-normal '
            f'(pin length = {L*1000:.1f} mm)', 4000)

    # ── Group moves (Direct-Edit panel) ──────────────────────────────────
    # Translate a NAMED set of hardpoints together, on one axle, as ONE undo
    # step.  Geometry stays the single source of truth: every move funnels
    # through the same rebuild + track-resync path the WASD nudge uses, so the
    # dynamics / loads follow.
    #   'spring': inboard spring/actuation hardware — rocker, spring-chassis,
    #             pushrod-inner, (direct) damper-chassis, and the whole
    #             heave / decoupled-cradle dicts.  EXCLUDES outboard
    #             pushrod_outer / damper_outer (they ride the wishbone).  Z.
    #   'arb'   : the ARB body — every ARB point EXCEPT the drop-link top
    #             (stays bolted to the rocker/upright).  Z.
    #   'arms'  : inboard control-arm pickups (uca/lca front+rear) + the REAR
    #             toe-link inboard point.  EXCLUDES the FRONT tie_rod_inner
    #             (steering-rack end).  X (in/out).
    _GROUP_SPRING_CORNER = frozenset({
        'rocker_pivot', 'rocker_spring_pt', 'pushrod_inner', 'spring_chassis_pt',
        'rocker_axis_pt', 'damper_chassis_pt', 'rocker_tbar_drop_pt',
    })
    _GROUP_ARB_EXCLUDE = frozenset({'arb_drop_top', 'tbar_drop_top'})
    _GROUP_ARMS_CORNER = frozenset({'uca_front', 'uca_rear', 'lca_front', 'lca_rear'})

    def _group_targets(self, group: str, is_front: bool):
        """Resolve a named group-move into [(category, dict, key), ...]."""
        corner = self._front_hp if is_front else self._rear_hp
        arb    = self._front_arb if is_front else self._rear_arb
        heave  = self._front_heave if is_front else self._rear_heave
        deco   = self._front_decoupled if is_front else self._rear_decoupled
        out = []
        if group == 'spring':
            out += [('corner', corner, k) for k in corner
                    if k in self._GROUP_SPRING_CORNER]
            out += [('heave', heave, k) for k in heave]        # 3rd-element spring setup
            out += [('decoupled', deco, k) for k in deco]      # twin-rocker cradle
        elif group == 'arb':
            out += [('arb', arb, k) for k in arb
                    if k not in self._GROUP_ARB_EXCLUDE]
        elif group == 'arms':
            out += [('corner', corner, k) for k in corner
                    if k in self._GROUP_ARMS_CORNER]
            # toe-link inboard point: REAR only — front tie_rod_inner is the rack
            if not is_front and 'tie_rod_inner' in corner:
                out.append(('corner', corner, 'tie_rod_inner'))
        return out

    def _move_group(self, group: str, axle: str, delta_xyz):
        """Translate a named group of hardpoints by a world delta (m) on one
        axle, as ONE undo step — same poka-yoke + rebuild + track-resync path
        as the WASD nudge."""
        is_front = str(axle).lower().startswith('f')
        delta = np.asarray(delta_xyz, float)
        targets = self._group_targets(group, is_front)
        if not targets:
            self.statusBar().showMessage(
                f'GROUP MOVE — no "{group}" points on the {axle} axle', 3000)
            return
        corner_dict = self._front_hp if is_front else self._rear_hp
        snapshot = [{'is_front': is_front, 'category': c, 'hp_name': k,
                     'prev': np.array(d[k], copy=True)} for c, d, k in targets]
        baseline = {is_front: self._spring_length_for_axle(is_front)}
        for c, d, k in targets:
            d[k] = np.asarray(d[k], float) + delta
        self._rebuild_solvers()
        # Moving the spring setup changes pushrod length → re-check damper range.
        block = ''
        if group == 'spring':
            block = self._check_damper_bounds_after_edit(
                [(is_front, corner_dict)], baseline=baseline)
        if block:
            for e in snapshot:
                hd = self._axle_dict_by_category(e['is_front'], e['category'])
                if hd is not None:
                    hd[e['hp_name']] = e['prev']
            self._rebuild_solvers()
            self._update_3d()
            self.statusBar().showMessage(f'BLOCKED — {block}', 5000)
            return
        self._edit_history.append(snapshot)
        self._redo_stack.clear()
        if len(self._edit_history) > 200:
            self._edit_history.pop(0)
        self._sync_track_from_hardpoints()
        try:
            panel = self._front_hp_panel if is_front else self._rear_hp_panel
            panel.refresh(corner_dict,
                          self._front_arb if is_front else self._rear_arb,
                          self._front_heave if is_front else self._rear_heave,
                          self._front_decoupled if is_front else self._rear_decoupled)
        except Exception:
            pass
        self._update_3d()
        # ONE MODEL: group move drives the kinematic graphs too (debounced).
        self._edit_sweep_timer.start()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))
        self.statusBar().showMessage(
            f'GROUP MOVE  |  {axle} {group}: {len(targets)} pts by '
            f'{np.round(delta * 1000, 2)} mm', 3000)

    def _on_group_move(self, group: str, axle: str, axis: int, mm: float):
        """Signal handler: build the world delta and dispatch to _move_group."""
        delta = np.zeros(3, float)
        if 0 <= int(axis) <= 2:
            delta[int(axis)] = float(mm) / 1000.0
        self._move_group(group, axle, delta)

    def _enforce_actuation_coplanar(self, hp: dict) -> dict:
        """Force a corner's actuation chain to be COPLANAR (in-place).

        A bellcrank / pushrod / pullrod rocker is a planar mechanism by
        definition: the plate plane is fixed by rocker_pivot, pushrod_inner and
        rocker_spring_pt, and the pushrod + spring must lie IN that plane.
        Hand-authored per-topology defaults drifted off it (measured up to
        198 mm), so here we PROJECT pushrod_outer / spring_chassis_pt / direct-
        damper ends onto the plate plane and snap rocker_axis_pt to the plane
        normal.  No-op for cars without a rocker (direct-corner, decoupled).
        """
        need = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')
        if not all(k in hp and hp[k] is not None
                   and np.all(np.isfinite(hp[k])) for k in need):
            return hp
        p0 = np.asarray(hp['rocker_pivot'], float)
        a = np.asarray(hp['pushrod_inner'], float) - p0
        b = np.asarray(hp['rocker_spring_pt'], float) - p0
        n = np.cross(a, b)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            return hp                      # degenerate plate — leave as-is
        n = n / nn
        for k in ('pushrod_outer', 'spring_chassis_pt',
                  'damper_chassis_pt', 'damper_outer_pt'):
            if k in hp and hp[k] is not None and np.all(np.isfinite(hp[k])):
                P = np.asarray(hp[k], float)
                hp[k] = P - float(np.dot(P - p0, n)) * n
        # rocker pin must be perpendicular to the plate
        if hp.get('rocker_axis_pt') is not None:
            ax = np.asarray(hp['rocker_axis_pt'], float)
            L = float(np.linalg.norm(ax - p0)) or 0.0254
            sign = 1.0 if np.dot(ax - p0, n) >= 0 else -1.0
            hp['rocker_axis_pt'] = p0 + sign * L * n
        return hp

    def _spring_length_for_axle(self, is_front: bool):
        """Spring (damper) length at design ride (t=0) for one axle, in metres.

        Returns None if no solver exists or the solve fails.  Used to
        snapshot the PRE-edit damper length so the bounds guard can tell a
        newly-introduced violation from a pre-existing one.
        """
        label = 'FL' if is_front else 'RL'
        solver = self._solvers.get(label)
        if solver is None:
            return None
        try:
            return float(solver.solve(0.0).spring_length)
        except Exception:
            return None

    def _check_damper_bounds_after_edit(self, targets, baseline=None) -> str:
        """After a hardpoint edit, verify the damper still fits its physical range.

        Returns an empty string if the move is OK, or a human-readable
        blocking reason.

        Poka-yoke philosophy: the guard only blocks an edit that *introduces
        or worsens* an out-of-range condition.  It must NEVER block an edit
        merely because the geometry was ALREADY out of range — e.g. a
        pullrod / direct / T-bar default whose damper free-length the user
        hasn't tuned yet.  Punishing a pre-existing condition would make the
        point un-editable (the very thing the user needs to fix it).  So:

          * in-bounds after the edit            -> always allow
          * out-of-bounds, but no worse than    -> allow (lets the user nudge
            it was before the edit                 the point back toward range)
          * out-of-bounds and worse than before -> BLOCK

        `baseline` maps is_front -> pre-edit spring length (m); points not in
        it are treated as having been in range (v_before = 0), so a fresh
        in->out edit is correctly blocked even without a baseline.

        Skips entirely when MotionPanel has no fully-extended damper length
        configured (treated as "user hasn't told us the physical limits yet").
        """
        try:
            full_ext_mm = float(self._motion_panel.fully_extended_mm)
            stroke_mm   = float(self._motion_panel.stroke_mm)
        except Exception:
            return ''
        if full_ext_mm <= 0 or stroke_mm <= 0:
            return ''   # damper bounds not configured — silently allow
        full_ext_m = full_ext_mm / 1000.0
        stroke_m   = stroke_mm / 1000.0
        min_L = full_ext_m - stroke_m
        baseline = baseline or {}
        TOL = 1e-6   # 1 micron — ignore solver numerical noise

        def _violation(L):
            """Signed-magnitude (m) by which L sits outside [min_L, full_ext_m]."""
            if L is None:
                return 0.0
            if L < min_L:
                return min_L - L
            if L > full_ext_m:
                return L - full_ext_m
            return 0.0

        for tgt_is_front, _ in targets:
            label = 'FL' if tgt_is_front else 'RL'
            solver = self._solvers.get(label)
            if solver is None:
                continue
            try:
                st = solver.solve(0.0)
                L = float(st.spring_length)
            except Exception:
                return (f'{"Front" if tgt_is_front else "Rear"} kinematics '
                        f'failed to converge — geometry invalid')
            v_after = _violation(L)
            if v_after <= TOL:
                continue   # in bounds after the edit — always OK

            # Out of range AFTER the edit.  Only block if the edit made it
            # WORSE than it already was (pre-existing out-of-range geometry
            # stays editable so the user can tune it back in).
            v_before = _violation(baseline.get(tgt_is_front))
            if v_after <= v_before + TOL:
                continue   # no worse than before — allow

            axle = 'Front' if tgt_is_front else 'Rear'
            worse = 'further ' if v_before > TOL else ''
            if L < min_L:
                return (f'{axle} damper bottoming {worse}out of range: '
                        f'spring = {L*1000:.1f} mm < min {min_L*1000:.1f} mm '
                        f'(free {full_ext_mm:.0f} − stroke {stroke_mm:.0f})')
            return (f'{axle} damper over-extending {worse}out of range: '
                    f'spring = {L*1000:.1f} mm > free {full_ext_mm:.0f} mm')
        return ''

    def _show_edit_status(self, hp_name: str, corner: str):
        is_front = corner in ('FL', 'FR')
        hp_dict = self._front_hp if is_front else self._rear_hp
        if hp_name in hp_dict:
            x, y, z = hp_dict[hp_name] * 1000.0
            self.statusBar().showMessage(
                f'EDIT  |  {corner}.{hp_name} → ({x:+8.2f}, {y:+8.2f}, {z:+8.2f}) mm  '
                f'|  step Δ = {self._edit_increment_mm:.2f} mm', 0)

    def _undo_edit(self):
        if not self._edit_history:
            self.statusBar().showMessage('Edit history empty', 2000)
            return
        snapshot = self._edit_history.pop()
        # snapshot is a list of {is_front, category, hp_name, prev}
        # (older snapshots predating the category field default to 'corner').
        # Capture the CURRENT values first so the undo can be REDOne (Ctrl+Y).
        redo_snap = []
        for entry in snapshot:
            cat = entry.get('category', 'corner')
            hp_dict = self._axle_dict_by_category(entry['is_front'], cat)
            if hp_dict is not None and entry['hp_name'] in hp_dict:
                redo_snap.append({'is_front': entry['is_front'], 'category': cat,
                                  'hp_name': entry['hp_name'],
                                  'prev': np.array(hp_dict[entry['hp_name']],
                                                   copy=True)})
        if redo_snap:
            self._redo_stack.append(redo_snap)
        for entry in snapshot:
            cat = entry.get('category', 'corner')
            hp_dict = self._axle_dict_by_category(entry['is_front'], cat)
            if hp_dict is not None:
                hp_dict[entry['hp_name']] = entry['prev']
        # Undo can restore a wheel centre — resync track so the dynamics /
        # loads follow the restored geometry too.
        self._sync_track_from_hardpoints()
        # Refresh hp table panels (front + rear if both touched)
        for is_front in {e['is_front'] for e in snapshot}:
            try:
                panel = (self._front_hp_panel if is_front
                         else self._rear_hp_panel)
                arb   = self._front_arb if is_front else self._rear_arb
                hpd   = self._front_hp  if is_front else self._rear_hp
                panel.refresh(hpd, arb)
            except Exception:
                pass
        self._rebuild_solvers()
        self._update_3d()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))
        self.statusBar().showMessage(
            f'Undo: {snapshot[0]["hp_name"]} restored  '
            f'({len(self._edit_history)} more in history; Ctrl+Y = redo)', 3000)

    def _redo_edit(self):
        """Ctrl+Y — re-apply the most recently undone edit.  The redo stack is
        cleared by any fresh edit (standard branch-invalidating redo)."""
        if not self._redo_stack:
            self.statusBar().showMessage('Nothing to redo', 2000)
            return
        redo_snap = self._redo_stack.pop()
        # Push the CURRENT values back onto the undo history first.
        undo_snap = []
        for entry in redo_snap:
            hp_dict = self._axle_dict_by_category(entry['is_front'],
                                                  entry['category'])
            if hp_dict is not None and entry['hp_name'] in hp_dict:
                undo_snap.append({'is_front': entry['is_front'],
                                  'category': entry['category'],
                                  'hp_name': entry['hp_name'],
                                  'prev': np.array(hp_dict[entry['hp_name']],
                                                   copy=True)})
        if undo_snap:
            self._edit_history.append(undo_snap)
        for entry in redo_snap:
            hp_dict = self._axle_dict_by_category(entry['is_front'],
                                                  entry['category'])
            if hp_dict is not None:
                hp_dict[entry['hp_name']] = entry['prev']
        self._sync_track_from_hardpoints()
        for is_front in {e['is_front'] for e in redo_snap}:
            try:
                panel = (self._front_hp_panel if is_front
                         else self._rear_hp_panel)
                arb = self._front_arb if is_front else self._rear_arb
                hpd = self._front_hp if is_front else self._rear_hp
                panel.refresh(hpd, arb)
            except Exception:
                pass
        self._rebuild_solvers()
        self._update_3d()
        self._edit_sweep_timer.start()
        self._direct_edit_panel.set_edit_count(len(self._edit_history))
        self.statusBar().showMessage(
            f'Redo: {redo_snap[0]["hp_name"]}  '
            f'({len(self._redo_stack)} more to redo)', 3000)

    # ── Inverse Kinematics ───────────────────────────────────────────────────

    _ik_thread: _IKWorker | None = None
    _ik_explore_thread: _IKExploreWorker | None = None

    def _build_ik_solver(self, spec: dict, bound_mm: float) -> InverseSolver:
        """Create a configured InverseSolver from a UI spec dict."""
        from vahan.optimizer import _evaluate_sweep

        axle = spec['axle']
        hp = dict(self._front_hp if axle == 'front' else self._rear_hp)
        side = 'left'
        # Mirror the live solver build (_rebuild_solvers): pushrod_body is the
        # user-configured damper mount, NOT a hardcoded front=uca/rear=lca guess.
        # Keeps the IK forward eval's motion_ratio / arb_mr target curves
        # faithful to the real corner solver for mismatched mounts (e.g. an
        # LCA- or upright-mounted front damper).  Geometry metrics are
        # mount-independent, so this only sharpens the rocker-driven targets.
        axle_top = self._topology.front if axle == 'front' else self._topology.rear
        pushrod_body = axle_top.damper_mount.value  # 'uca' / 'lca' / 'upright'

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

        # Primary target curve: lo (at min travel/droop) -> hi (at max/bump),
        # with a selectable shape so a NONLINEAR (e.g. progressive/exponential)
        # motion ratio can be targeted — a rising-rate MR that stiffens the
        # wheel rate under compression to resist aero heave and hold ride
        # height.  Linear (default) reproduces the old behaviour.
        from vahan.optimizer import shaped_target
        target_lo = spec.get('target_lo', spec.get('target', 0.0))
        target_hi = spec.get('target_hi', target_lo)
        target_ramp = shaped_target(
            float(target_lo), float(target_hi), n_pts,
            shape=spec.get('target_shape', 'linear'),
            curvature=float(spec.get('target_curvature', 2.0)))

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
                # Mirror the live solver build: pushrod_body = configured mount.
                axle_top = (self._topology.front if axle == 'front'
                            else self._topology.rear)
                pushrod_body = axle_top.damper_mount.value  # 'uca'/'lca'/'upright'
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
                from vahan.optimizer import _evaluate_sweep, shaped_target
                target_lo = spec.get('target_lo', 0.0)
                target_hi = spec.get('target_hi', target_lo)
                target_ramp = shaped_target(
                    float(target_lo), float(target_hi), n_pts,
                    shape=spec.get('target_shape', 'linear'),
                    curvature=float(spec.get('target_curvature', 2.0)))

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

            dyn_params = self._apply_topology_to_dyn_params(dyn_params)


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
                self._front_hp_panel.refresh(self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
            if hasattr(self, '_rear_hp_panel'):
                self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)
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
            self._front_hp_panel.refresh(self._front_hp, self._front_arb, self._front_heave, self._front_decoupled)
        else:
            self._rear_hp = sus_hp
            if arb_hp:
                self._rear_arb = arb_hp
            self._rear_hp_panel.refresh(self._rear_hp,  self._rear_arb,  self._rear_heave,  self._rear_decoupled)

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

    def _refresh_arb_geometry_into_panel(self) -> None:
        """Push kinematically-derived ARB geometry (arm/half/MR) into the
        dynamics panel so its ARB wheel-rate calculation reflects the model
        as it is NOW.  Every path that consumes get_params() for anything
        ARB-dependent (steady dynamics, transient/skidpad, loads, report)
        must call this first — ONE MODEL."""
        arb_F = self._compute_arb_geometry_from_kinematics('F')
        arb_R = self._compute_arb_geometry_from_kinematics('R')
        if arb_F is not None and arb_R is not None:
            self._dynamics_panel.set_derived_arb_geometry(arb_F, arb_R)

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
        self._refresh_arb_geometry_into_panel()

        dyn_params = self._dynamics_panel.get_params()
        car = self._car

        # Geometry from Car Params panel
        dyn_params['front_track_m'] = car['track_f_mm'] / 1000
        dyn_params['rear_track_m'] = car['track_r_mm'] / 1000
        dyn_params['wheelbase_m'] = car['wheelbase_mm'] / 1000
        dyn_params['cg_height_m'] = car['cg_z_mm'] / 1000
        # Tire radius: single source = CarParams tire OD (the panel spinbox is
        # a synced display; this override is what the solver actually uses).
        dyn_params['tire_radius_m'] = float(
            car.get('tire_outer_dia_mm', 406.4)) / 2000.0
        dyn_params['cg_to_front_axle_m'] = car.get('cg_y_mm', 1100) / 1000
        if 'front_brake_bias_pct' in car:
            dyn_params['front_brake_bias'] = car['front_brake_bias_pct'] / 100

        # Motion ratio from kinematic model: MR = d(spring_length)/d(travel)
        # Computed via central difference at design position (±1mm).
        # Central-T-bar axles (plain T-bar ARB / HEAVE_TBAR): the corner is
        # cradle_link (spring_length = NaN) and the corner COIL lives on the
        # central bellcrank — its MR comes from the heave-T-bar solver's
        # coil_length instead of the corner solve.
        dt = 0.001  # 1mm perturbation
        for label, param_key, is_front in [('FL', 'motion_ratio_front', True),
                                           ('RL', 'motion_ratio_rear', False)]:
            solver = self._solvers.get(label)
            if not solver:
                continue
            hts = self._heave_tbar_solver(is_front)
            try:
                if hts is not None and hts.coil_length(
                        np.asarray(solver.solve(0.0).pushrod_outer, float)) > 0:
                    po_p = np.asarray(solver.solve(+dt).pushrod_outer, float)
                    po_m = np.asarray(solver.solve(-dt).pushrod_outer, float)
                    mr = abs(hts.coil_length(po_p) - hts.coil_length(po_m)) / (2 * dt)
                else:
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

        dyn_params = self._apply_topology_to_dyn_params(dyn_params)


        # Differential params (corner entry/exit balance via locking %).
        dyn_params['diff_kind'] = self._diff.kind
        dyn_params['diff_power_ramp_deg'] = self._diff.power_ramp_deg
        dyn_params['diff_coast_ramp_deg'] = self._diff.coast_ramp_deg
        dyn_params['diff_preload_Nm'] = self._diff.preload_Nm
        dyn_params['engine_braking_Nm'] = getattr(self, '_engine_braking_Nm', 12.0)

        veh = VehicleParams(**dyn_params)
        # Design roll-centre heights (same kinematic model) so the constants
        # readout can show the ROLL GRADIENT (deg/g) — the standard design-
        # judge sanity number — without running a full dynamic solve here.
        rc_f = rc_r = None
        try:
            for lbl, attr in (('FL', 'f'), ('RL', 'r')):
                s = self._solvers.get(lbl)
                if s is not None:
                    m = KinematicMetrics(s.solve(0.0),
                                         'left')
                    if attr == 'f':
                        rc_f = float(m.roll_center_height)
                    else:
                        rc_r = float(m.roll_center_height)
        except Exception:
            pass
        # Update computed constants display
        self._dynamics_panel.update_constants(veh, rc_front_m=rc_f,
                                              rc_rear_m=rc_r)
        self._vehicle_constants_panel.update(veh)
        # Spring/MR/mass may have changed — refresh static sag readout.
        self._refresh_sag()
        tire = self._tire_model
        if tire is None:
            from vahan.tire_model import LinearTireModel
            tire = LinearTireModel()
        # Rear tire = front unless a split front/rear setup is active.
        tire_rear = self._tire_model_rear
        return SteadyStateSolver(veh, self._solvers, tire,
                                 tire_model_rear=tire_rear)

    def _refresh_vehicle_constants(self):
        """Rebuild VehicleParams and push to the constants popup.

        Wired to dynamics_panel.params_changed via a 200 ms debounce so
        every spinbox tick doesn't fire a rebuild. Catches all exceptions
        — a half-initialised panel must not break the GUI.
        """
        try:
            # Build the full solver so the panel sees identical numbers
            # to what Solve / Sweep would compute (same MR, same ARB
            # geometry pulled from kinematics).
            self._build_dynamics_solver()
        except Exception:
            pass

    def _try_autoload_tire(self):
        """Auto-load a tire model from whatever data the user has dropped in
        tire_data/ (gitignored — no tire data ships with the repo).  Loads the
        first .mat/.csv found; if the folder is empty (e.g. a fresh clone), the
        dynamics solver falls back to the parametric LinearTireModel.  No
        specific tire file is named here, so the public source implies no
        particular dataset."""
        import os, glob
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'tire_data')
        files = sorted(glob.glob(os.path.join(base, '*.mat')) +
                       glob.glob(os.path.join(base, '*.csv')))
        if files:
            self._on_tire_file(files[0])

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

    def _set_rear_tire(self, path):
        """Set the REAR-axle tire for a split front/rear compound setup.
        path=None -> rear tire is the same as the front (the usual case)."""
        if path is None:
            self._tire_model_rear = None
            return None
        from vahan.tire_model import TireModel
        self._tire_model_rear = TireModel.from_file(path)
        return self._tire_model_rear

    def _on_tire_plots(self):
        """Render the tire / grip characterization plots (Fy vs slip angle,
        cornering stiffness vs load, aligning moment vs slip angle, friction
        circle) from the active tire model + the current lat/lon g operating
        point.  Uses the loaded TTC model if present, else the linear fallback
        the dynamics solver installs."""
        try:
            ss = self._build_dynamics_solver()
            tire = getattr(ss, '_tire', None) or self._tire_model
            if tire is None:
                from vahan.tire_model import LinearTireModel
                tire = LinearTireModel()
            lat = float(self._dynamics_panel._lat_g.value())
            lon = float(self._dynamics_panel._lon_g.value())
            result = None
            try:
                result = ss.solve(lat, lon)
            except Exception:
                result = None
            cam = float(self._alignment.get('front_camber_deg', 0.0))
            fc = self._dynamics_panel.tire_plot_inputs()
            self.curves.plot_tire_grip(tire, result, camber_deg=cam,
                                       fc_fz_levels=fc['fz_levels'],
                                       fc_3d=fc['fc_3d'])
            self.statusBar().showMessage(
                f'Tire / grip plots  —  model: {getattr(tire, "tire_id", "tire")}', 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Tire plot error: {e}', 5000)

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

        # ── Poka-yoke: surface disclaimers so the user is informed
        # when results may be unreliable.  Shown in status bar (long
        # timeout if there are any warnings, since they need reading).
        try:
            veh = self._build_dynamics_solver()._veh
            warns = self._check_dynamics_disclaimers(veh, steady_result=result)
        except Exception:
            warns = []
        if warns:
            # Surface to status bar; one line per warn (truncated)
            warn_msg = '  |  '.join('⚠ ' + w.split(':')[0] for w in warns[:3])
            self.statusBar().showMessage(
                f'Dynamics: {result.roll_angle_deg:.3f} deg @ {result.lateral_g:.2f}g  |  {warn_msg}',
                15000)
            # Also write full text to the dynamics panel status line
            try:
                self._dynamics_panel.set_status(
                    'WARNINGS: ' + ' || '.join(warns))
            except Exception:
                pass
        else:
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
        # Total mass is DERIVED (sprung + unsprung F + unsprung R) — the old
        # standalone _total_mass spinbox was removed; only _total_mass_lbl shows
        # the sum.  Recompute it the same way the panel does.
        mass = (self._dynamics_panel._sprung_mass.value()
                + self._dynamics_panel._us_front.value()
                + self._dynamics_panel._us_rear.value())
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
            mass = (self._dynamics_panel._sprung_mass.value()
                    + self._dynamics_panel._us_front.value()
                    + self._dynamics_panel._us_rear.value())
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
            mass = (self._dynamics_panel._sprung_mass.value()
                    + self._dynamics_panel._us_front.value()
                    + self._dynamics_panel._us_rear.value())
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

    def _on_view_controls_changed(self, d):
        """Navcube view-controls box (mode / perspective / floor) drives the view
        and syncs the Car Parameters panel controls."""
        try:
            self._car['view_mode'] = d.get('view_mode', 'normal')
            self._car['show_ground'] = bool(d.get('floor', True))
            if 'thickness' in d:
                self._car['show_shock_thickness'] = bool(d['thickness'])
            try:
                cb = self._car_panel._view_mode_combo
                cb.blockSignals(True); cb.setCurrentText(self._car['view_mode'].capitalize())
                cb.blockSignals(False)
                self._car_panel._show_ground.blockSignals(True)
                self._car_panel._show_ground.setChecked(self._car['show_ground'])
                self._car_panel._show_ground.blockSignals(False)
                self._car_panel._chk_perspective.blockSignals(True)
                self._car_panel._chk_perspective.setChecked(bool(d.get('perspective', True)))
                self._car_panel._chk_perspective.blockSignals(False)
                if 'thickness' in d and hasattr(self._car_panel, '_show_shock_thick'):
                    self._car_panel._show_shock_thick.blockSignals(True)
                    self._car_panel._show_shock_thick.setChecked(self._car['show_shock_thickness'])
                    self._car_panel._show_shock_thick.blockSignals(False)
            except Exception:
                pass
            self.view3d.set_perspective(bool(d.get('perspective', True)))
            self._update_3d()
        except Exception:
            pass

    def _open_wheel_package(self):
        """Open the Seward-style Wheel Package Load Analysis (upright + arms)."""
        try:
            from gui import wheel_package
            old = getattr(self, '_wpkg_dialog', None)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            self._wpkg_dialog = wheel_package.build_dialog(self)
            self._wpkg_dialog.show()
            self._wpkg_dialog.raise_()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f'Wheel package error: {e}', 5000)

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

            # DECOUPLED corners keep their pushrod inner end on the central
            # twin-rocker cradle; hand the live cradle solvers to the load path
            # so it can recover the true pushrod_inner instead of seeing NaN.
            # FRESH cradle solvers (the cached ones go stale on any edit) so the
            # load path sees the SAME decoupled model as the 3D view / graph.
            cradle_solvers = {
                'front': self._decoupled_solver(True),
                'rear':  self._decoupled_solver(False),
            }
            # HEAVE_TBAR corners are likewise cradle_link — recover their
            # pushrod_inner from the heave-T-bar rocker (FRESH, never cached).
            heave_tbar_solvers = {
                'front': self._heave_tbar_solver(True),
                'rear':  self._heave_tbar_solver(False),
            }

            loads = compute_all_corners(
                self._solvers, result,
                brake_params_f=bp_f, brake_params_r=bp_r,
                upright_params_f=up, upright_params_r=up,
                wheel_radius_m=wheel_r,
                motion_ratio_f=veh.motion_ratio_front,
                motion_ratio_r=veh.motion_ratio_rear,
                cradle_solvers=cradle_solvers,
                heave_tbar_solvers=heave_tbar_solvers,
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
    #  ANALYSIS & VALIDATION PLOTS  (separate from main graph viewport)
    # ==========================================================================

    def _show_plot(self, fig, title):
        """Open fig in a PlotDialog popup."""
        dlg = PlotDialog(self, fig, title=title)
        # Value-readout on hover, same machinery as the kinematics canvas.
        dlg.hover = HoverAnnotator(dlg.canvas)
        dlg.show()
        # Keep a reference so Qt doesn't gc it
        if not hasattr(self, '_open_plot_dialogs'):
            self._open_plot_dialogs = []
        self._open_plot_dialogs.append(dlg)

    def _on_plot_brake_capacity(self):
        try:
            from vahan.analysis_plots import plot_brake_capacity
            solver = self._build_dynamics_solver()
            veh = solver._veh
            inp = self._analysis_plots_panel.brake_inputs()
            fig = plot_brake_capacity(
                total_mass_kg=veh.total_mass_kg,
                weight_dist_front=veh.cg_to_rear_axle_m / veh.wheelbase_m if hasattr(veh, 'cg_to_rear_axle_m')
                                  else (veh.cg_to_front_axle_m / veh.wheelbase_m if hasattr(veh, 'cg_to_front_axle_m') else 0.45),
                cg_height_m=veh.cg_height_m,
                wheelbase_m=veh.wheelbase_m,
                tire_radius_m=veh.tire_radius_m,
                brake_bias_front=veh.front_brake_bias,
                tire_model=self._tire_model,     # None if no TTC loaded
                **inp,
            )
            self._show_plot(fig, 'Brake-Torque Capacity vs Deceleration')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Brake plot error: {e}', 5000)

    def _on_plot_wheel_rate_linearity(self):
        try:
            from vahan.analysis_plots import plot_wheel_rate_linearity
            solver = self._build_dynamics_solver()
            veh = solver._veh
            if not getattr(self, '_solvers', None):
                self.statusBar().showMessage(
                    'Wheel-rate plot needs the kinematic solvers (load a car first)',
                    5000)
                return
            inp = self._analysis_plots_panel.wheel_rate_linearity_inputs()
            fig = plot_wheel_rate_linearity(
                solvers=self._solvers,
                k_spring_front_Npm=veh.spring_rate_front_Npm,
                k_spring_rear_Npm=veh.spring_rate_rear_Npm,
                **inp,
            )
            self._show_plot(fig, 'Wheel-Rate Linearity Across Travel')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Wheel-rate plot error: {e}', 5000)

    def _on_plot_llt(self):
        try:
            from vahan.analysis_plots import plot_lateral_load_transfer
            solver = self._build_dynamics_solver()
            fig = plot_lateral_load_transfer(solver)
            self._show_plot(fig, 'Lateral Load Transfer Distribution')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'LLT plot error: {e}', 5000)

    def _on_plot_ride_freq(self):
        try:
            from vahan.analysis_plots import plot_ride_freq_bump
            solver = self._build_dynamics_solver()
            veh = solver._veh
            mr_f = veh.motion_ratio_front
            mr_r = veh.motion_ratio_rear
            k_F = veh.spring_rate_front_Npm * mr_f ** 2
            k_R = veh.spring_rate_rear_Npm * mr_r ** 2
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            inp = self._analysis_plots_panel.ride_freq_inputs()
            fig = plot_ride_freq_bump(
                sprung_total_kg=veh.sprung_mass_kg,
                weight_dist_front=wf,
                wheelbase_m=veh.wheelbase_m,
                k_wheel_front_Npm=k_F,
                k_wheel_rear_Npm=k_R,
                **inp,
            )
            self._show_plot(fig, 'Ride Frequency — Pitch over a Bump')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Ride-freq plot error: {e}', 5000)

    def _on_plot_rc_vs_roll(self):
        try:
            from vahan.analysis_plots import plot_rc_vs_body_roll
            solver = self._build_dynamics_solver()
            inp = self._analysis_plots_panel.rc_vs_roll_inputs()
            fig = plot_rc_vs_body_roll(solver, **inp)
            self._show_plot(fig, 'RC Height vs Body Roll')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'RC plot error: {e}', 5000)

    def _on_plot_steering_torque(self):
        try:
            from vahan.analysis_plots import plot_steering_torque
            solver = self._build_dynamics_solver()
            veh = solver._veh
            if self._tire_model is None:
                self.statusBar().showMessage('Steering-torque plot needs TTC data loaded', 5000)
                return
            # Scrub radius + KPI from FL kinematic at design ride
            from vahan.kinematics import KinematicMetrics
            fl_solver = self._solvers.get('FL')
            if fl_solver is None:
                self.statusBar().showMessage('FL solver not ready', 5000); return
            state = fl_solver.solve(0.0)
            km = KinematicMetrics(state, side='left')
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            inp = self._analysis_plots_panel.steering_torque_inputs()
            fig = plot_steering_torque(
                tire_model=self._tire_model, vehicle_params=veh,
                scrub_radius_m=km.scrub_radius, kpi_deg=km.kpi,
                wheelbase_m=veh.wheelbase_m, weight_dist_front=wf,
                **inp,
            )
            self._show_plot(fig, 'Steering Torque vs Steer Angle')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Steering torque plot error: {e}', 5000)

    def _on_plot_ackermann_loads(self):
        try:
            from vahan.analysis_plots import plot_ackermann_loads
            if self._tire_model is None:
                self.statusBar().showMessage('Ackermann plot needs TTC data loaded', 5000)
                return
            solver = self._build_dynamics_solver()
            veh = solver._veh
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            inp = self._analysis_plots_panel.ackermann_inputs()
            fig = plot_ackermann_loads(
                tire_model=self._tire_model,
                total_mass_kg=veh.total_mass_kg,
                weight_dist_front=wf,
                wheelbase_m=veh.wheelbase_m,
                cg_height_m=veh.cg_height_m,
                track_front_m=veh.front_track_m if hasattr(veh, 'front_track_m')
                              else veh.track_front_m,
                **inp,
            )
            self._show_plot(fig, 'Ackermann Inner/Outer Fy Reasoning')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Ackermann plot error: {e}', 5000)

    def _on_plot_ackermann_demand(self):
        try:
            from vahan.analysis_plots import plot_ackermann_demand
            if self._tire_model is None:
                self.statusBar().showMessage('Ackermann demand plot needs TTC data loaded', 5000)
                return
            solver = self._build_dynamics_solver()
            veh = solver._veh
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            inp = self._analysis_plots_panel.ackermann_demand_inputs()
            fig = plot_ackermann_demand(
                tire_model=self._tire_model,
                total_mass_kg=veh.total_mass_kg,
                weight_dist_front=wf,
                wheelbase_m=veh.wheelbase_m,
                cg_height_m=veh.cg_height_m,
                track_front_m=veh.front_track_m if hasattr(veh, 'front_track_m')
                              else veh.track_front_m,
                **inp,
            )
            self._show_plot(fig, 'Ackermann Slip-Angle Budget')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Ackermann demand plot error: {e}', 5000)

    def _on_plot_ackermann_fzfy(self):
        try:
            from vahan.analysis_plots import plot_ackermann_fz_fy
            if self._tire_model is None:
                self.statusBar().showMessage('Fz-Fy map needs TTC data loaded', 5000)
                return
            solver = self._build_dynamics_solver()
            veh = solver._veh
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            inp = self._analysis_plots_panel.ackermann_demand_inputs()
            fig = plot_ackermann_fz_fy(
                tire_model=self._tire_model,
                total_mass_kg=veh.total_mass_kg,
                weight_dist_front=wf,
                wheelbase_m=veh.wheelbase_m,
                cg_height_m=veh.cg_height_m,
                track_front_m=veh.front_track_m if hasattr(veh, 'front_track_m')
                              else veh.track_front_m,
                **inp,
            )
            self._show_plot(fig, 'Fz–Fy Operating Map')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Fz-Fy map error: {e}', 5000)

    def _on_rack_zero_ackermann(self):
        """Sweep tie_rod_inner Y (fore-aft) to find the rack position that
        produces 0% Ackermann and report the delta from current in mm and inches."""
        try:
            from PyQt6.QtWidgets import QMessageBox
            corners = self._all_corner_hp()
            hp_fl   = corners['FL']
            hp_fr   = corners['FR']
            wb = self._car.get('wheelbase_mm', 1537.) / 1000.
            ft = self._car.get('track_f_mm',   1222.) / 1000.
            ref_rack = _rack_travel_from_angle(25.0, self._steer or {})

            def _ack_at_y_offset(dy_m: float) -> float:
                """Compute Ackermann % with tie_rod_inner Y shifted by dy_m."""
                toes = {}
                for lbl, hp_base in (('FL', hp_fl), ('FR', hp_fr)):
                    hp = {k: v.copy() for k, v in hp_base.items()}
                    hp['tie_rod_inner'] = hp_base['tie_rod_inner'] + np.array([0., dy_m, 0.])
                    steered = self._steered_hp(hp, ref_rack, True)
                    d = hp_base['tie_rod_outer'] - hp_base['tie_rod_inner']
                    tierod_len_sq = float(d @ d)
                    solver = SuspensionConstraints(
                        _hp_obj(steered),
                        tierod_len_sq=tierod_len_sq,
                        pushrod_body='uca',
                    )
                    st = solver.solve(0.)
                    m  = KinematicMetrics(st, 'left' if lbl == 'FL' else 'right')
                    toes[lbl] = float(m.toe)
                return _ackermann_from_pair(toes['FL'], toes['FR'], wb, ft)

            current_ack = _ack_at_y_offset(0.0)

            # Binary-search for 0% Ackermann over ±300 mm fore-aft range
            lo, hi = -0.300, 0.300   # metres
            ack_lo = _ack_at_y_offset(lo)
            ack_hi = _ack_at_y_offset(hi)

            # Check which direction 0% lives
            if ack_lo * ack_hi > 0:
                QMessageBox.warning(self, 'Rack Position',
                    f'Current Ackermann: {current_ack:.1f}%\n'
                    f'0% Ackermann not found within ±300 mm fore-aft range.\n'
                    f'(ack at −300 mm: {ack_lo:.1f}%, at +300 mm: {ack_hi:.1f}%)')
                return

            for _ in range(40):        # bisect to <0.01 mm precision
                mid = (lo + hi) / 2.0
                if _ack_at_y_offset(mid) * ack_lo < 0:
                    hi = mid
                else:
                    lo = mid

            dy_mm  = mid * 1000.0
            dy_in  = dy_mm / 25.4
            direction = 'rearward' if dy_mm < 0 else 'forward'
            QMessageBox.information(self, 'Rack Position for 0% Ackermann',
                f'Current Ackermann:  {current_ack:.1f}%\n'
                f'Target:             0%  (parallel steering)\n\n'
                f'Move rack {direction} by:\n'
                f'  {abs(dy_mm):.1f} mm\n'
                f'  {abs(dy_in):.2f} in\n\n'
                f'(tie_rod_inner Y shift = {dy_mm:+.1f} mm)')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'Rack 0% search error: {e}', 5000)

    def _on_plot_mmd(self):
        try:
            from vahan.analysis_plots import plot_mmd
            if self._tire_model is None:
                self.statusBar().showMessage('MMD plot needs TTC data loaded', 5000)
                return
            solver = self._build_dynamics_solver()
            veh = solver._veh
            wf = (veh.cg_to_rear_axle_m / veh.wheelbase_m) if hasattr(veh, 'cg_to_rear_axle_m') else 0.45
            tf = veh.front_track_m if hasattr(veh, 'front_track_m') else veh.track_front_m
            tr = veh.rear_track_m  if hasattr(veh, 'rear_track_m')  else veh.track_rear_m

            inp = self._analysis_plots_panel.mmd_inputs()

            # ── Convert lever arm Δ → ARB wheel-rate Δ ──────────────────────
            # K_arb ∝ 1 / L²  (K_t fixed, arm changes leverage)
            # K_new = K_base * (L_base / (L_base + ΔL))²
            # delta_arb = K_new − K_base = K_base * [(L_base / L_new)² − 1]
            dL_f_m = inp.pop('delta_arm_front_mm', 0.0) / 1000.0
            dL_r_m = inp.pop('delta_arm_rear_mm',  0.0) / 1000.0

            # Base lever arm from kinematic hardpoints (arb_pivot → arb_arm_end)
            def _arm_len(arb_hp):
                try:
                    return float(np.linalg.norm(
                        arb_hp['arb_arm_end'] - arb_hp['arb_pivot']))
                except (KeyError, TypeError):
                    return 0.0

            L_f_base = _arm_len(self._front_arb)
            L_r_base = _arm_len(self._rear_arb)

            def _darb(K_base, L_base, dL):
                if L_base < 1e-4 or K_base <= 0:
                    return 0.0
                L_new = max(L_base + dL, 1e-4)
                return K_base * ((L_base / L_new) ** 2 - 1.0)

            inp['delta_arb_front_Npm'] = _darb(veh.arb_rate_front_Npm, L_f_base, dL_f_m)
            inp['delta_arb_rear_Npm']  = _darb(veh.arb_rate_rear_Npm,  L_r_base, dL_r_m)

            fig = plot_mmd(
                tire_model=self._tire_model,
                total_mass_kg=veh.total_mass_kg,
                weight_dist_front=wf,
                wheelbase_m=veh.wheelbase_m,
                cg_height_m=veh.cg_height_m,
                track_front_m=tf,
                track_rear_m=tr,
                roll_stiffness_front_Npm_rad=veh.roll_stiffness_front_Npm_rad,
                roll_stiffness_rear_Npm_rad=veh.roll_stiffness_rear_Npm_rad,
                **inp,
            )
            self._show_plot(fig, 'Milliken Moment Diagram')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f'MMD plot error: {e}', 5000)

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
            # Refresh the kinematically-derived ARB geometry FIRST — without
            # this, get_params() computes the ARB wheel rates from whatever
            # arm/half/MR was pushed last (stale after hardpoint edits), and
            # the transient sim runs on a different bar than the model has.
            self._refresh_arb_geometry_into_panel()
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
            dyn_params = self._apply_topology_to_dyn_params(dyn_params)

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
        # Enrich with the SOLVED vehicle's derived values + the topology, so
        # the report reflects the actual model (single-model): the raw panel
        # dict lacks total_mass_kg / peak_mu / wheel rates / topology-applied
        # MRs, which the report's parameter table reads.
        try:
            _veh_solved = self._build_dynamics_solver()._veh
            veh_params['total_mass_kg'] = float(_veh_solved.total_mass_kg)
            veh_params['wheel_rate_front_Npm'] = float(_veh_solved.wheel_rate_front_Npm)
            veh_params['wheel_rate_rear_Npm'] = float(_veh_solved.wheel_rate_rear_Npm)
            veh_params['motion_ratio_front'] = float(_veh_solved.motion_ratio_front)
            veh_params['motion_ratio_rear'] = float(_veh_solved.motion_ratio_rear)
            for _k in ('topology_mode_front', 'topology_mode_rear',
                       'decoupled_heave_rate_front_Npm', 'decoupled_roll_rate_front_Npm',
                       'decoupled_heave_rate_rear_Npm', 'decoupled_roll_rate_rear_Npm',
                       'decoupled_heave_MR_front', 'decoupled_roll_MR_front',
                       'decoupled_heave_MR_rear', 'decoupled_roll_MR_rear',
                       'heave_3rd_rate_front_Npm', 'heave_3rd_rate_rear_Npm',
                       'heave_3rd_MR_front', 'heave_3rd_MR_rear'):
                if hasattr(_veh_solved, _k):
                    veh_params[_k] = getattr(_veh_solved, _k)
            try:
                veh_params['peak_mu'] = float(
                    self._tire_model.peak_mu(700.0, 0.0)) if self._tire_model else 1.5
            except Exception:
                pass
        except Exception:
            pass
        # Topology description (per axle) for the report header.
        try:
            veh_params['topology_desc'] = self._topology.describe()
        except Exception:
            pass

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

    # ── Startup wizard: open existing file OR pick a new topology ────────
    from gui.startup_dialog import StartupDialog
    from vahan.topology import SuspensionTopology
    wizard = StartupDialog()
    if wizard.exec() != wizard.DialogCode.Accepted:
        # User cancelled — exit cleanly
        sys.exit(0)
    result = wizard.result_payload()
    # Backward compat: old 2-tuple, new 3-tuple
    if len(result) == 3:
        mode, payload, dims = result
    else:
        mode, payload = result
        dims = None

    win = MainWindow()
    if mode == 'open':
        # Load the file the user picked.  _load_project_from_path is the
        # programmatic version of the File→Load menu action.
        try:
            win._load_project_from_path(payload)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(win, 'Load failed', str(e))
    elif mode == 'new':
        # Apply the chosen topology + dimensions — drives hardpoint set,
        # solver, 3D view, and rack/track/wheelbase coordinates.
        win.set_topology(payload, dimensions=dims)

    win.show()
    sys.exit(app.exec())
