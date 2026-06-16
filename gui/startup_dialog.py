"""
gui/startup_dialog.py — first-screen dialog shown at app launch.

Two options:
    1. Open an existing .vahan file (loads ALL inputs including topology).
    2. Start a new design — choose suspension topology per axle.

The dialog returns one of:
    ('open', path_to_file)
    ('new',  SuspensionTopology)
    (None,   None)              # user cancelled / closed the dialog
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QFileDialog, QGroupBox, QComboBox,
    QStackedWidget, QWidget, QFrame, QSpacerItem, QSizePolicy,
    QDoubleSpinBox, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from vahan.topology import (
    SuspensionTopology, AxleTopology,
    DamperActuation, DamperMount, ARBType, SpringConfig,
)


# Default values DERIVED from the legacy hard-coded hardpoint geometry
# (DEFAULT_FRONT_HP / DEFAULT_REAR_HP in gui/main_window.py).  These are the
# numbers that drop straight onto those coordinates with zero scaling, so a
# user who hits Continue without changing anything gets the legacy state
# byte-for-byte.  Computed:
#   wheelbase_mm  = DEFAULT_REAR_HP['wheel_center'].Y  * 1000  =  1536.7 mm
#   track_f_mm    = 2 × DEFAULT_FRONT_HP['wheel_center'].X * 1000  =  1117.6 mm
#   track_r_mm    = 2 × DEFAULT_REAR_HP ['wheel_center'].X * 1000  =  1117.6 mm
#   rack_length   = 2 × DEFAULT_FRONT_HP['tie_rod_inner'].X * 1000 =   438.16 mm
# Note: the older self._car panel default was 1222 mm front track, which
# silently disagreed with the geometry-side wheel_center.X = 0.5588 m
# (= 1117.6 mm).  The wizard now exposes the geometry as the single
# source of truth (poka-yoke).
_DEFAULT_DIMS = {
    'wheelbase_mm':         1536.7,
    'track_f_mm':           1117.6,
    'track_r_mm':           1117.6,
    'rack_length_mm':        438.16,
    'total_rack_travel_mm':  100.0,
    # Spring / damper outside diameter (mm).  Used by view3d to render
    # the spring stack at a realistic visible width and for clash checks
    # against the rocker plate / chassis tube routing.
    'spring_od_mm':           63.0,   # Eibach 1.88" ID coil, OD ≈ 63 mm
    'damper_od_mm':           50.0,   # Öhlins TTX36 / Penske 7800 ≈ 50 mm
}


# ─────────────────────────────────────────────────────────────────────────────
#  Vehicle dimensions editor — drives the corner-HP coordinate generation
#  so we don't have to over-constrain track/wheelbase/rack in multiple places.
# ─────────────────────────────────────────────────────────────────────────────

class _VehicleDimensionsEditor(QGroupBox):
    """Wheelbase, track widths, rack length + travel.

    Outputs from this widget are written into ``self._car`` / ``self._steer``
    on the MainWindow, AND used at topology-build time to set
    ``tie_rod_inner`` X = ±rack_length/2 so the user never has to enter that
    coordinate twice.  This is the poka-yoke contract: one input → one
    source of truth.
    """

    def __init__(self, parent=None):
        super().__init__('VEHICLE DIMENSIONS', parent)
        self.setStyleSheet("""
            QGroupBox { color:#FFD600; font-weight:bold; font-size:12px;
                        border:1px solid #2a2a2a; border-radius:6px;
                        margin-top:14px; padding-top:8px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 6px; }
            QLabel  { color:#cccccc; font-size:11.5px; }
            QDoubleSpinBox { background:#1a1a1a; color:#e8e8e8;
                             border:1px solid #2a2a2a; padding:3px 6px;
                             font-size:11.5px; min-width: 110px; }
        """)

        grid = QGridLayout(self); grid.setSpacing(6)

        def _spin(default, suffix=' mm', mn=10.0, mx=5000.0, step=10.0,
                  decimals=2):
            sb = QDoubleSpinBox()
            sb.setRange(mn, mx)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setSuffix(suffix)
            sb.setValue(default)
            return sb

        row = 0
        grid.addWidget(QLabel('Wheelbase:'), row, 0)
        self._wb = _spin(_DEFAULT_DIMS['wheelbase_mm'], mx=4000.0)
        grid.addWidget(self._wb, row, 1)
        grid.addWidget(QLabel('(front axle Y = 0, rear at +Y)'), row, 2)
        row += 1

        grid.addWidget(QLabel('Front track:'), row, 0)
        self._tf = _spin(_DEFAULT_DIMS['track_f_mm'], mx=3000.0)
        grid.addWidget(self._tf, row, 1)
        grid.addWidget(QLabel('(wheel-center to wheel-center, mm)'), row, 2)
        row += 1

        grid.addWidget(QLabel('Rear track:'), row, 0)
        self._tr = _spin(_DEFAULT_DIMS['track_r_mm'], mx=3000.0)
        grid.addWidget(self._tr, row, 1)
        grid.addWidget(QLabel('(wheel-center to wheel-center, mm)'), row, 2)
        row += 1

        grid.addWidget(QLabel('Rack length:'), row, 0)
        self._rk_len = _spin(_DEFAULT_DIMS['rack_length_mm'],
                             mx=2000.0, step=5.0)
        grid.addWidget(self._rk_len, row, 1)
        grid.addWidget(
            QLabel('→ tie_rod_inner X = ±rack_length/2  (no double-entry)'),
            row, 2)
        row += 1

        grid.addWidget(QLabel('Rack total travel:'), row, 0)
        self._rk_travel = _spin(_DEFAULT_DIMS['total_rack_travel_mm'],
                                mx=400.0, step=5.0)
        grid.addWidget(self._rk_travel, row, 1)
        grid.addWidget(QLabel('(full bump-to-bump rack stroke)'), row, 2)
        row += 1

        # Visual separator inside the group
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#2a2a2a; background:#2a2a2a; max-height:1px;')
        grid.addWidget(sep, row, 0, 1, 3); row += 1

        grid.addWidget(QLabel('Spring OD:'), row, 0)
        self._spring_od = _spin(_DEFAULT_DIMS['spring_od_mm'],
                                mn=5.0, mx=300.0, step=1.0, decimals=1)
        grid.addWidget(self._spring_od, row, 1)
        grid.addWidget(QLabel('(coil outside diameter — drives 3D render width)'),
                       row, 2)
        row += 1

        grid.addWidget(QLabel('Damper OD:'), row, 0)
        self._damper_od = _spin(_DEFAULT_DIMS['damper_od_mm'],
                                mn=5.0, mx=300.0, step=1.0, decimals=1)
        grid.addWidget(self._damper_od, row, 1)
        grid.addWidget(QLabel('(damper body OD — used when actuation = DIRECT)'),
                       row, 2)
        row += 1

        # Live consistency warning if rack length > front track
        self._warn_lbl = QLabel('')
        self._warn_lbl.setStyleSheet('color:#EF5350; font-size:10.5px;')
        self._warn_lbl.setWordWrap(True)
        grid.addWidget(self._warn_lbl, row, 0, 1, 3)

        for sb in (self._wb, self._tf, self._tr, self._rk_len, self._rk_travel,
                   self._spring_od, self._damper_od):
            sb.valueChanged.connect(self._refresh_validity)
        self._refresh_validity()

    def _refresh_validity(self):
        msgs = []
        if self._rk_len.value() >= self._tf.value():
            msgs.append('Rack length ≥ front track — tie-rod inner ends would '
                        'be at/outside the wheel; almost certainly wrong.')
        if self._rk_travel.value() > self._rk_len.value():
            msgs.append('Rack travel > rack length — impossible.')
        self._warn_lbl.setText('  '.join('⚠ ' + m for m in msgs))

    def dimensions(self) -> dict:
        """Return all collected dimensions as a plain dict."""
        return {
            'wheelbase_mm':         float(self._wb.value()),
            'track_f_mm':           float(self._tf.value()),
            'track_r_mm':           float(self._tr.value()),
            'rack_length_mm':       float(self._rk_len.value()),
            'total_rack_travel_mm': float(self._rk_travel.value()),
            'spring_od_mm':         float(self._spring_od.value()),
            'damper_od_mm':         float(self._damper_od.value()),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-axle topology editor (used twice — once for front, once for rear)
# ─────────────────────────────────────────────────────────────────────────────

class _AxleTopologyEditor(QGroupBox):
    """Compact editor for one axle's topology choices."""

    def __init__(self, title: str, defaults: AxleTopology, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox { color:#FFD600; font-weight:bold; font-size:12px;
                        border:1px solid #2a2a2a; border-radius:6px;
                        margin-top:14px; padding-top:8px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 6px; }
            QLabel  { color:#cccccc; font-size:11.5px; }
            QComboBox { background:#1a1a1a; color:#e8e8e8; border:1px solid #2a2a2a;
                        padding:4px 6px; font-size:11.5px; min-width: 140px; }
            QComboBox QAbstractItemView { background:#1a1a1a; color:#e8e8e8;
                        selection-background-color:#3a3a3a; }
        """)
        self._defaults = defaults

        grid = QGridLayout(self); grid.setSpacing(6)

        # Damper actuation
        grid.addWidget(QLabel('Damper actuation:'), 0, 0)
        self._cmb_actuation = QComboBox()
        for choice in DamperActuation:
            self._cmb_actuation.addItem(self._actuation_label(choice), choice)
        self._cmb_actuation.setCurrentText(self._actuation_label(defaults.damper_actuation))
        self._cmb_actuation.currentIndexChanged.connect(self._refresh_visibility)
        grid.addWidget(self._cmb_actuation, 0, 1)

        # Damper mount
        self._lbl_mount = QLabel('Mount point:')
        grid.addWidget(self._lbl_mount, 1, 0)
        self._cmb_mount = QComboBox()
        for choice in DamperMount:
            self._cmb_mount.addItem(self._mount_label(choice), choice)
        self._cmb_mount.setCurrentText(self._mount_label(defaults.damper_mount))
        grid.addWidget(self._cmb_mount, 1, 1)

        # ARB type
        grid.addWidget(QLabel('Anti-roll device:'), 2, 0)
        self._cmb_arb = QComboBox()
        for choice in ARBType:
            self._cmb_arb.addItem(self._arb_label(choice), choice)
        self._cmb_arb.setCurrentText(self._arb_label(defaults.arb_type))
        self._cmb_arb.currentIndexChanged.connect(self._refresh_visibility)
        grid.addWidget(self._cmb_arb, 2, 1)

        # Spring config
        grid.addWidget(QLabel('Spring configuration:'), 3, 0)
        self._cmb_spring = QComboBox()
        for choice in SpringConfig:
            self._cmb_spring.addItem(self._spring_label(choice), choice)
        self._cmb_spring.setCurrentText(self._spring_label(defaults.spring_config))
        self._cmb_spring.currentIndexChanged.connect(self._refresh_visibility)
        grid.addWidget(self._cmb_spring, 3, 1)

        # Live validation message
        self._validation_lbl = QLabel('')
        self._validation_lbl.setStyleSheet('color:#EF5350; font-size:10.5px;')
        self._validation_lbl.setWordWrap(True)
        grid.addWidget(self._validation_lbl, 4, 0, 1, 2)

        self._refresh_visibility()

    # ── Display labels ─────────────────────────────────────────────────
    @staticmethod
    def _actuation_label(c: DamperActuation) -> str:
        return {
            DamperActuation.DIRECT:  'Direct (no rocker)',
            DamperActuation.PUSHROD: 'Pushrod',
            DamperActuation.PULLROD: 'Pullrod',
        }[c]

    @staticmethod
    def _mount_label(c: DamperMount) -> str:
        return {
            DamperMount.UCA:     'Upper control arm',
            DamperMount.LCA:     'Lower control arm',
            DamperMount.UPRIGHT: 'Upright',
        }[c]

    @staticmethod
    def _arb_label(c: ARBType) -> str:
        return {
            ARBType.BELLCRANK:   'Bellcrank ARB (U-bar via rockers)',
            ARBType.CONTROL_ARM: 'Control-arm ARB (drop link to LCA)',
            ARBType.TBAR:        'T-bar (centre-pivoted torsion bar)',
            ARBType.NONE:        'None',
        }[c]

    @staticmethod
    def _spring_label(c: SpringConfig) -> str:
        return {
            SpringConfig.CORNER:     'Corner springs only',
            SpringConfig.HEAVE_TBAR: 'Corner + heave 3rd element on T-bar',
            SpringConfig.DECOUPLED:  'Decoupled (twin rockers: heave + roll)',
        }[c]

    # ── Visibility / validity logic ─────────────────────────────────────
    def _refresh_visibility(self):
        actuation = self._cmb_actuation.currentData()
        # Mount only meaningful when not direct
        is_direct = (actuation == DamperActuation.DIRECT)
        self._lbl_mount.setVisible(not is_direct or True)  # always show — Direct can also mount on a-arm/upright
        self._cmb_mount.setVisible(True)

        # Validate live
        warnings = self._build_axle().validate()
        if warnings:
            self._validation_lbl.setText('⚠ ' + '\n⚠ '.join(warnings))
        else:
            self._validation_lbl.setText('')

    def _build_axle(self) -> AxleTopology:
        return AxleTopology(
            damper_actuation=self._cmb_actuation.currentData(),
            damper_mount=self._cmb_mount.currentData(),
            arb_type=self._cmb_arb.currentData(),
            spring_config=self._cmb_spring.currentData(),
        )

    def axle_topology(self) -> AxleTopology:
        return self._build_axle()


# ─────────────────────────────────────────────────────────────────────────────
#  Main wizard dialog
# ─────────────────────────────────────────────────────────────────────────────

class StartupDialog(QDialog):
    """First-launch dialog: open existing file OR build a new topology."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Vahan — start')
        self.setMinimumSize(720, 700)
        self.setStyleSheet("""
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
            QRadioButton { color:#e8e8e8; font-size:13px; padding:4px; }
        """)

        # Result fields
        self._mode: str | None = None
        self._file_path: str | None = None
        self._topology: SuspensionTopology | None = None
        self._dimensions: dict | None = None      # populated only for 'new'

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self); layout.setSpacing(12); layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel('Vahan — Suspension Design')
        f = QFont(); f.setPointSize(16); f.setBold(True)
        title.setFont(f)
        title.setStyleSheet('color:#FFD600;')
        layout.addWidget(title)

        sub = QLabel('Start from an existing design or define a new suspension topology.')
        sub.setStyleSheet('color:#8a8a92; font-size:11.5px;')
        layout.addWidget(sub)

        # Top-level mode radios
        mode_box = QGroupBox()
        mode_box.setStyleSheet('QGroupBox{border:0;}')
        mode_layout = QVBoxLayout(mode_box); mode_layout.setSpacing(4)
        self._mode_group = QButtonGroup(self)
        self._rb_open = QRadioButton('Open an existing project (.vahan / .json)')
        self._rb_new  = QRadioButton('Start a new design — choose topology')
        self._rb_new.setChecked(True)
        self._mode_group.addButton(self._rb_open, 0)
        self._mode_group.addButton(self._rb_new,  1)
        self._rb_open.toggled.connect(self._on_mode_changed)
        self._rb_new.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._rb_open)
        mode_layout.addWidget(self._rb_new)
        layout.addWidget(mode_box)

        # Stacked: index 0 = open, 1 = new topology
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # Page 0 — open
        page_open = QWidget()
        po = QVBoxLayout(page_open); po.setSpacing(10)
        po.addWidget(QLabel('Pick a Vahan project file.  All inputs '
                            '(hardpoints, ARB, vehicle params, topology) '
                            'will load from the file.'))
        self._open_path_lbl = QLabel('— no file selected —')
        self._open_path_lbl.setStyleSheet('color:#8a8a92; font-family:Consolas; '
                                          'background:#1a1a1a; padding:8px; '
                                          'border:1px solid #2a2a2a; border-radius:4px;')
        po.addWidget(self._open_path_lbl)
        btn_browse = QPushButton('Browse…')
        btn_browse.clicked.connect(self._on_browse)
        po.addWidget(btn_browse, alignment=Qt.AlignmentFlag.AlignLeft)
        po.addStretch(1)
        self._stack.addWidget(page_open)

        # Page 1 — new topology
        page_new = QWidget()
        pn = QVBoxLayout(page_new); pn.setSpacing(8)
        pn.addWidget(QLabel('Configure each axle.  These choices set the '
                            'hardpoint set, kinematic solver, and 3D viewer '
                            'rendering — they can be changed later from File → '
                            'New Project.'))
        # Vehicle dimensions (drive default coordinates — no double-entry)
        self._editor_dims  = _VehicleDimensionsEditor()
        defaults = SuspensionTopology.standard()
        self._editor_front = _AxleTopologyEditor('FRONT AXLE',
                                                 defaults.front)
        self._editor_rear  = _AxleTopologyEditor('REAR AXLE',
                                                 defaults.rear)

        # Scrollable: there's a lot on this page now (dims + 2 axles + warnings)
        inner = QWidget()
        il = QVBoxLayout(inner); il.setSpacing(8); il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(self._editor_dims)
        il.addWidget(self._editor_front)
        il.addWidget(self._editor_rear)
        il.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border:0; background:transparent; }')
        pn.addWidget(scroll, 1)
        self._stack.addWidget(page_new)
        self._stack.setCurrentIndex(1)

        # Bottom buttons
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#2a2a2a; background:#2a2a2a; max-height:1px;')
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        self._btn_go = QPushButton('Continue')
        self._btn_go.setObjectName('primary')
        self._btn_go.setDefault(True)
        self._btn_go.clicked.connect(self._on_continue)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_go)
        layout.addLayout(btn_row)

    def _on_mode_changed(self):
        if self._rb_open.isChecked():
            self._stack.setCurrentIndex(0)
        else:
            self._stack.setCurrentIndex(1)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Vahan project', '',
            'Vahan project (*.vahan *.json);;All files (*)')
        if path:
            self._open_path_lbl.setText(path)
            self._file_path = path

    def _on_continue(self):
        if self._rb_open.isChecked():
            if not self._file_path:
                self._open_path_lbl.setText(
                    '⚠ pick a file first  — click Browse…')
                self._open_path_lbl.setStyleSheet(
                    'color:#EF5350; font-family:Consolas; background:#1a1a1a; '
                    'padding:8px; border:1px solid #EF5350; border-radius:4px;')
                return
            self._mode = 'open'
            self.accept()
        else:
            top = SuspensionTopology(
                front=self._editor_front.axle_topology(),
                rear=self._editor_rear.axle_topology(),
            )
            v = top.validate()
            if v['front'] or v['rear']:
                # Block until configuration is consistent
                return
            self._mode = 'new'
            self._topology = top
            self._dimensions = self._editor_dims.dimensions()
            self.accept()

    # ── Public result accessor ──────────────────────────────────────────
    def result_payload(self):
        """Return one of:
            ('open', path,     None)            — load existing project
            ('new',  topology, dimensions_dict) — build from a fresh topology
            (None,   None,     None)            — user cancelled
        """
        if self._mode == 'open':
            return ('open', self._file_path, None)
        if self._mode == 'new':
            return ('new', self._topology, self._dimensions)
        return (None, None, None)
