"""
gui/view3d.py — GPU-accelerated 3D suspension view (VisPy / OpenGL)

Axis convention: X=lateral(outboard), Y=longitudinal(fwd), Z=up

Mouse controls (Onshape style via direct Qt events):
    Right-drag              → Orbit (rotate)
    Middle-drag             → Pan
    Ctrl + Right-drag       → Pan
    Scroll wheel            → Zoom
"""

import numpy as np
from vispy import scene, app as vispy_app

vispy_app.use_app('pyqt6')

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF, QTransform


# ── NavCube widget ────────────────────────────────────────────────────────────

class NavCube(QWidget):
    """
    Onshape-style solid orientation cube with perspective-warped face labels.

    Convention: X = lateral / width, Y = longitudinal / length, Z = up.
    VisPy TurntableCamera: az=0 → camera at -Y looking at +Y (front of car).
    """
    view_requested = pyqtSignal(float, float)

    _VERTS = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],   # 0-3 bottom
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],   # 4-7 top
    ], dtype=float)

    # (fill_vi, outward_normal, label, snap_az, snap_el,
    #  text_vi = [TL, TR, BR, BL] when viewed head-on)
    _FACES = [
        ([0,1,2,3], [ 0, 0,-1], 'Bottom', None, -90, [0,1,2,3]),
        ([4,5,6,7], [ 0, 0, 1], 'Top',    None,  90, [7,6,5,4]),
        ([0,1,5,4], [ 0,-1, 0], 'Front',    0,   0,  [4,5,1,0]),
        ([2,3,7,6], [ 0, 1, 0], 'Back',   180,   0,  [6,7,3,2]),
        ([0,3,7,4], [-1, 0, 0], 'Left',   270,   0,  [7,4,0,3]),
        ([1,2,6,5], [ 1, 0, 0], 'Right',   90,   0,  [5,6,2,1]),
    ]

    # Fixed directional light for consistent face shading
    _LIGHT = np.array([0.35, -0.25, 0.85])
    _LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

    def __init__(self, parent=None, size=110):
        super().__init__(parent)
        self._sz = size
        self.setFixedSize(size, size)
        self._az = -60.
        self._el = 20.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_orientation(self, az, el):
        if az != self._az or el != self._el:
            self._az, self._el = az, el
            self.update()

    def _cam_dir(self):
        """Camera position direction — matches VisPy TurntableCamera."""
        az, el = np.radians(self._az), np.radians(self._el)
        return np.array([np.sin(az)*np.cos(el),
                         -np.cos(az)*np.cos(el),
                         np.sin(el)])

    def _project(self, verts):
        """Orthographic projection matching VisPy TurntableCamera."""
        az, el = np.radians(self._az), np.radians(self._el)
        ca, sa = np.cos(az), np.sin(az)
        ce, se = np.cos(el), np.sin(el)
        # Camera right = (ca, sa, 0), camera up = (-sa*se, ca*se, ce)
        R = np.array([[ca, sa, 0],
                      [-sa * se, ca * se, ce]])
        pts = (R @ np.asarray(verts).T).T
        s = self._sz * 0.30
        c = self._sz / 2
        pts = pts * s + c
        pts[:, 1] = self._sz - pts[:, 1]   # flip Y for screen coords
        return pts

    def paintEvent(self, _event):
        pts = self._project(self._VERTS)
        cam = self._cam_dir()

        # Only front-facing faces (dot > 0), sorted back-to-front
        draws = []
        for vi, nrm, label, snap_az, snap_el, text_vi in self._FACES:
            n = np.asarray(nrm, float)
            dot = float(np.dot(n, cam))
            if dot <= 0.0:
                continue
            depth = float(np.mean(self._VERTS[vi] @ cam))
            shade = max(float(np.dot(n, self._LIGHT)), 0.08)
            draws.append((depth, vi, label, dot, text_vi, shade))
        draws.sort(key=lambda x: x[0])

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        edge_pen = QPen(QColor(25, 28, 38), 2.0)

        # ── 1. Solid opaque faces ───────────────────────────────────────
        for _, vi, label, dot, text_vi, shade in draws:
            poly = QPolygonF([QPointF(float(pts[i, 0]), float(pts[i, 1]))
                              for i in vi])
            g = int(52 + 68 * shade)
            col = QColor(g - 2, g, g + 6)
            p.setPen(edge_pen)
            p.setBrush(col)
            p.drawPolygon(poly)

        # ── 2. Perspective-warped text on each visible face ─────────────
        for _, vi, label, dot, text_vi, shade in draws:
            if dot < 0.15:
                continue
            tw, th = 100., 44.
            src = QPolygonF([QPointF(0, 0), QPointF(tw, 0),
                             QPointF(tw, th), QPointF(0, th)])
            dst = QPolygonF([QPointF(float(pts[i, 0]), float(pts[i, 1]))
                             for i in text_vi])
            xform = QTransform()
            ok = QTransform.quadToQuad(src, dst, xform)
            if ok:
                p.save()
                p.setTransform(xform, True)
                alpha = int(min(255, 120 + 135 * dot))
                p.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
                p.setPen(QColor(195, 200, 212, alpha))
                p.drawText(0, 0, int(tw), int(th),
                           int(Qt.AlignmentFlag.AlignCenter), label)
                p.restore()

        # ── 3. XYZ axes from front-left-bottom corner (vertex 0) ───────
        ax_pts_3d = np.array([
            [-1, -1, -1],      # origin (vertex 0)
            [ 1.25, -1, -1],   # X tip
            [-1, 1.25, -1],    # Y tip
            [-1, -1, 1.25],    # Z tip
            [ 1.5, -1, -1],    # X label
            [-1, 1.5, -1],     # Y label
            [-1, -1, 1.5],     # Z label
        ])
        ap = self._project(ax_pts_3d)
        axes = [('X', QColor(220, 75, 75), 1),
                ('Y', QColor(75, 190, 75), 2),
                ('Z', QColor(75, 140, 230), 3)]
        for lbl, col, ti in axes:
            ox, oy = float(ap[0, 0]), float(ap[0, 1])
            ex, ey = float(ap[ti, 0]), float(ap[ti, 1])
            p.setPen(QPen(col, 2.0))
            p.drawLine(QPointF(ox, oy), QPointF(ex, ey))
            # arrowhead
            dx, dy = ex - ox, ey - oy
            ln = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / ln, dy / ln
            p.drawLine(QPointF(ex, ey),
                       QPointF(ex - ux * 5 + uy * 3, ey - uy * 5 - ux * 3))
            p.drawLine(QPointF(ex, ey),
                       QPointF(ex - ux * 5 - uy * 3, ey - uy * 5 + ux * 3))
        # axis labels
        p.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        for lbl, col, ti in axes:
            li = ti + 3   # label point index (4, 5, 6)
            p.setPen(col)
            p.drawText(int(ap[li, 0]) - 6, int(ap[li, 1]) - 6, 12, 12,
                       int(Qt.AlignmentFlag.AlignCenter), lbl)
        p.end()

    def mousePressEvent(self, event):
        pts = self._project(self._VERTS)
        mx, my = event.position().x(), event.position().y()
        cam = self._cam_dir()

        best, best_d = None, -1e9
        for vi, nrm, label, snap_az, snap_el, _tv in self._FACES:
            if np.dot(np.asarray(nrm, float), cam) <= 0.0:
                continue
            if self._pt_in_poly(mx, my,
                    [(float(pts[i, 0]), float(pts[i, 1])) for i in vi]):
                d = float(np.mean(self._VERTS[vi] @ cam))
                if d > best_d:
                    best_d = d
                    best = (snap_az, snap_el)
        if best:
            az, el = best
            if az is None:
                az = self._az
            self.view_requested.emit(float(az), float(el))

    @staticmethod
    def _pt_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


# ── geometry helpers ──────────────────────────────────────────────────────────

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp_frame(axis):
    s = _norm(np.asarray(axis, float))
    ref = np.array([0., 1., 0.]) if abs(s[1]) < 0.9 else np.array([1., 0., 0.])
    u = _norm(np.cross(s, ref))
    v = np.cross(s, u)
    return u, v


# ── mesh builders ─────────────────────────────────────────────────────────────

def _torus_section_mesh(center, axis, r_outer, half_width, n=48):
    """
    Tyre tread: open cylinder (no end caps) — pure tread band.
    Returns (vertices, faces).
    """
    u, v = _perp_frame(axis)
    w = _norm(np.asarray(axis, float))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r_outer * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    r0 = (center - half_width * w) + ring
    r1 = (center + half_width * w) + ring
    verts = np.vstack([r0, r1]).astype(np.float32)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    return verts, np.array(faces, np.uint32)


def _annulus_mesh(center, axis, r_inner, r_outer, n=48):
    """Sidewall annulus."""
    u, v = _perp_frame(axis)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = center + r_inner * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    outer = center + r_outer * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    verts = np.vstack([inner, outer]).astype(np.float32)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    return verts, np.array(faces, np.uint32)


def _merge(meshes):
    all_v, all_f, off = [], [], 0
    for v, f in meshes:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v), np.vstack(all_f)






def build_tire_mesh(center, spin_axis, outer_r, rim_r, half_w, n=48):
    """Tyre = tread cylinder + two sidewall annuli. NO rim — rim is interior."""
    c = np.asarray(center, float)
    s = _norm(np.asarray(spin_axis, float))
    parts = [_torus_section_mesh(c, s, outer_r, half_w, n)]
    for sign in (-1, 1):
        cap = c + sign * half_w * s
        parts.append(_annulus_mesh(cap, s, rim_r, outer_r, n))
    return _merge(parts)


def build_cylinder_between(p0, p1, radius, n=20):
    """Closed cylinder (with end caps) spanning p0 → p1 at given radius.

    Used for spring / damper rendering: pass the spring's two endpoints
    in world coords and the user-set OD/2 as radius.  Returns (verts, faces)
    suitable for a scene.Mesh.  Cap faces are included so the cylinder
    appears solid from any angle.

    Degenerate input (p0 ≈ p1) returns a zero-area placeholder so the
    caller can blindly assign it without checking.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    L  = float(np.linalg.norm(p1 - p0))
    if L < 1e-9 or radius <= 0.0:
        # Degenerate — caller still needs valid vertices/faces
        return (np.zeros((3, 3), np.float32),
                np.array([[0, 1, 2]], np.uint32))
    axis = (p1 - p0) / L
    center = 0.5 * (p0 + p1)
    half_w = 0.5 * L
    # Side cylinder (open) — same construction as the tyre tread band
    side_v, side_f = _torus_section_mesh(center, axis, radius, half_w, n)
    # Two end caps via _annulus_mesh with r_inner=0
    cap0_v, cap0_f = _annulus_mesh(center - half_w * axis, axis, 0.0, radius, n)
    cap1_v, cap1_f = _annulus_mesh(center + half_w * axis, axis, 0.0, radius, n)
    return _merge([(side_v, side_f), (cap0_v, cap0_f), (cap1_v, cap1_f)])


def build_box(center, u_ax, u_rad, u_tan, half_ax, half_rad, half_tan):
    """Oriented box (8 verts, 12 tris) about `center` along three unit axes.
    Used for the brake caliper block straddling the rotor."""
    c = np.asarray(center, float)
    u_ax = _norm(np.asarray(u_ax, float))
    u_rad = _norm(np.asarray(u_rad, float))
    u_tan = _norm(np.asarray(u_tan, float))
    verts = []
    for sa in (-1, 1):
        for sr in (-1, 1):
            for st in (-1, 1):
                verts.append(c + sa * half_ax * u_ax
                             + sr * half_rad * u_rad + st * half_tan * u_tan)
    verts = np.array(verts, np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.uint32)
    return verts, faces


def build_prism(poly, half_t):
    """Extrude a coplanar polygon (list of >=3 3-D verts) by +/-half_t along its
    own normal into a solid PLATE (two faces + side walls).  Used for the
    bellcrank/rocker, which is a flat machined PLATE, not a tube."""
    poly = np.asarray(poly, float)
    n = len(poly)
    if n < 3:
        return (np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32))
    nz = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    ln = np.linalg.norm(nz)
    nz = nz / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
    front = poly + nz * half_t
    back = poly - nz * half_t
    verts = np.vstack([front, back]).astype(np.float32)   # 0..n-1 front, n..2n-1 back
    faces = []
    for i in range(1, n - 1):                 # front face fan
        faces.append([0, i, i + 1])
    for i in range(1, n - 1):                 # back face fan (reversed)
        faces.append([n, n + i + 1, n + i])
    for i in range(n):                        # side walls
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
    return verts, np.array(faces, np.uint32)


def _tube_segments(segs, radius, n=10):
    """Merge a list of (p0, p1, rgba) into one (verts, faces, vertex_colors)
    mesh of solid cylinders — used to give the suspension members real 3-D
    thickness in ONE draw call."""
    allv, allf, allc = [], [], []
    off = 0
    for p0, p1, col in segs:
        v, f = build_cylinder_between(p0, p1, radius, n=n)
        allv.append(v)
        allf.append(f + off)
        allc.append(np.tile(np.asarray(col, np.float32)[:4], (len(v), 1)))
        off += len(v)
    if not allv:
        return (np.zeros((3, 3), np.float32),
                np.array([[0, 1, 2]], np.uint32),
                np.ones((3, 4), np.float32))
    return (np.concatenate(allv), np.concatenate(allf).astype(np.uint32),
            np.concatenate(allc))


# ── link topology ─────────────────────────────────────────────────────────────
# (key_a, key_b, RGBA)
# Note: rocker_spring_pt → spring_chassis_pt is now rendered as a SOLID
# CYLINDER (see _spring_meshes in update_scene) using the user's spring/
# damper OD.  The skinny grey line was redundant and would clip through
# the cylinder, so it's removed from this list.
LINKS = [
    ('uca_front',        'uca_outer',         (0.20, 0.55, 0.90, 1.0)),
    ('uca_rear',         'uca_outer',         (0.20, 0.55, 0.90, 1.0)),
    ('lca_front',        'lca_outer',         (0.90, 0.25, 0.25, 1.0)),
    ('lca_rear',         'lca_outer',         (0.90, 0.25, 0.25, 1.0)),
    ('tie_rod_inner',    'tie_rod_outer',     (0.25, 0.80, 0.40, 1.0)),
    ('pushrod_outer',    'pushrod_inner',     (0.95, 0.55, 0.10, 1.0)),
]

CHASSIS_PTS = frozenset({
    'uca_front', 'uca_rear', 'lca_front', 'lca_rear',
    'tie_rod_inner', 'rocker_pivot', 'spring_chassis_pt',
})

HP_NAMES = [
    'uca_front', 'uca_rear',  'uca_outer',
    'lca_front', 'lca_rear',  'lca_outer',
    'tie_rod_inner', 'tie_rod_outer', 'wheel_center',
    'pushrod_outer', 'pushrod_inner',
    'rocker_pivot',  'rocker_spring_pt', 'spring_chassis_pt',
]

_C_CHASSIS = (0.35, 0.65, 1.00, 1.0)
_C_MOVING  = (1.00, 0.35, 0.35, 1.0)
_C_SEL     = (1.00, 0.92, 0.23, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW3D
# ══════════════════════════════════════════════════════════════════════════════

class View3D:
    """
    GPU-accelerated 3-D suspension view.
    Camera controlled directly via Qt mouse events (reliable on Win/PyQt6).
    """

    def __init__(self):
        self._canvas = scene.SceneCanvas(
            title='Vahan 3D',
            bgcolor=(0.0, 0.0, 0.0, 1.0),
            keys=None,
            show=False,
        )
        self._view = self._canvas.central_widget.add_view()
        self._persp_fov = 40.0   # remembered perspective FOV for the toggle
        self._cam  = scene.TurntableCamera(
            fov=self._persp_fov, elevation=20, azimuth=-60,
            scale_factor=2.5, center=(0, 0, 0.2),
            interactive=False,   # we handle mouse manually
        )
        self._view.camera = self._cam

        # ── visuals ───────────────────────────────────────────────────────
        self._links_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=np.ones((2, 4), np.float32),
            connect='segments', width=2.5, antialias=True,
            parent=self._view.scene,
        )
        self._arb_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(0.90, 0.80, 0.10, 1.0),
            connect='segments', width=2.5,
            parent=self._view.scene,
        )
        # Suspension members (control arms / pushrod / tie rods) drawn as SOLID
        # TUBES (real thickness) merged into one mesh.  ~16 mm OD tube.
        self._member_mesh = scene.Mesh(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.array([[0, 1, 2]], np.uint32),
            vertex_colors=np.ones((3, 4), np.float32),
            parent=self._view.scene)
        self._member_r_full = 0.008     # 16 mm OD member tubes (thickness ON)
        self._member_r_thin = 0.0006    # ~1 mm wisps (thickness OFF → reads as lines)
        self._member_r = self._member_r_full
        self._thick_on  = True          # navcube "Thickness" toggle state
        self._rocker_half_t = 0.003     # bellcrank = 6 mm flat PLATE (extruded, not tubed)
        # ARB drawn as a SOLID TUBE too (was a thin line).
        self._arb_mesh = scene.Mesh(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.array([[0, 1, 2]], np.uint32),
            vertex_colors=np.ones((3, 4), np.float32),
            parent=self._view.scene)
        # Clash highlight (Interference mode): fat warning-red segments over any
        # members that interfere.  Empty until set_clashes() is called.
        self._clash_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(0.95, 0.15, 0.15, 1.0),
            connect='segments', width=7.0, antialias=True,
            parent=self._view.scene,
        )
        # View mode: 'normal' | 'load' (desaturate links) | 'interference'
        # (desaturate + show all thicknesses + highlight clashes).
        self._view_mode = 'normal'
        # Load-mode force vectors: coloured arrows at the load points (shaft +
        # 3-D arrowhead barbs).  Populated by set_load_vectors().
        self._loadvec_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=np.ones((2, 4), np.float32),
            connect='segments', width=3.0, antialias=True,
            parent=self._view.scene)
        self._loadvec_count = 0
        self._loadvec_snap = []          # [(tip_world, label)] for Load-mode hover
        # Corner isolation (wheel-package view): draw only this corner, or all.
        self._isolate = None
        self._markers = scene.Markers(parent=self._view.scene)
        self._markers.set_data(
            pos=np.zeros((1, 3), np.float32),
            face_color=np.ones((1, 4), np.float32),
            size=9, edge_width=0,
        )

        # Per-corner meshes (4 corners max)
        self._tire_meshes    = [self._new_mesh((0.18, 0.18, 0.18, 0.6)) for _ in range(4)]
        self._upright_meshes = [self._new_mesh((0.50, 0.40, 0.30, 0.30)) for _ in range(4)]
        self._rocker_meshes  = [self._new_mesh((0.85, 0.70, 0.12, 1.0)) for _ in range(4)]
        # Spring / damper cylinder per corner — radius set via set_spring_dims
        self._spring_meshes  = [self._new_mesh((0.75, 0.75, 0.78, 0.70)) for _ in range(4)]
        # Outside diameters in METRES.  Defaults match the wizard defaults
        # so a fresh View3D renders correctly even before the host calls
        # set_spring_dims().
        self._spring_od_m = 0.063
        self._damper_od_m = 0.050

        # ── Rear driveshaft / differential package ───────────────────────
        # Diff body (mid-grey), two tripods (yellow), two half-shafts (light
        # grey).  Colourblind-safe (grey/white/yellow — no red/green pairing).
        # Populated by set_driveshaft_package(); empty until then.
        self._diff_mesh = self._new_mesh((0.55, 0.55, 0.58, 0.90))
        self._driveshaft_meshes = [self._new_mesh((0.82, 0.82, 0.85, 0.92))
                                   for _ in range(2)]   # RL, RR
        self._tripod_meshes = [self._new_mesh((0.95, 0.85, 0.25, 0.95))
                               for _ in range(2)]       # RL, RR
        # Brake rotors (grey disc coaxial with the wheel) + calipers (red block
        # straddling the rotor at the top).  One per corner.
        self._rotor_meshes = [self._new_mesh((0.32, 0.32, 0.35, 0.85)) for _ in range(4)]
        self._caliper_meshes = [self._new_mesh((0.62, 0.64, 0.68, 0.95)) for _ in range(4)]
        self._show_brakes = True
        # Brake dims (metres).  Rotor thickness fixed at the Wilwood GP200's
        # 0.25 in rotor width; rotor radius is a user input (<= 11 in max dia).
        self._rotor_r = 0.120
        self._rotor_hw = 0.5 * 0.25 * 0.0254        # 0.25 in / 2 = 3.175 mm
        # (mesh, base_rgba) for every SOLID car part — greyed in Load /
        # Interference mode so only the force vectors keep colour.
        self._car_meshes = (
            [(m, (0.18, 0.18, 0.18, 0.6)) for m in self._tire_meshes]
            + [(m, (0.50, 0.40, 0.30, 0.30)) for m in self._upright_meshes]
            + [(m, (0.85, 0.70, 0.12, 1.0)) for m in self._rocker_meshes]
            + [(m, (0.75, 0.75, 0.78, 0.70)) for m in self._spring_meshes]
            + [(self._diff_mesh, (0.55, 0.55, 0.58, 0.90))]
            + [(m, (0.82, 0.82, 0.85, 0.92)) for m in self._driveshaft_meshes]
            + [(m, (0.95, 0.85, 0.25, 0.95)) for m in self._tripod_meshes]
            + [(m, (0.32, 0.32, 0.35, 0.85)) for m in self._rotor_meshes]
            + [(m, (0.62, 0.64, 0.68, 0.95)) for m in self._caliper_meshes])

        # ── Decoupled (twin-bellcrank) visuals ───────────────────────────
        # Per axle (front + rear) we render:
        #   * LEFT bellcrank plate    →  self._cradle_meshes
        #   * RIGHT bellcrank plate   →  self._cradle_lat_spring_meshes
        #   * HEAVE coilover cylinder →  self._cradle_obl_spring_meshes
        #     (cross-car, between heave_damper_left and heave_damper_right)
        #   * ROLL coilover cylinder  →  self._cradle_roll_damper_meshes
        #     (cross-car, diagonal, asymmetric Z so heave doesn't compress it)
        # Slot names kept from the old (wrong) cradle model so we don't
        # have to rewire downstream code — they're just repurposed.
        # Drawn only when the topology supplies the cradle hardpoints,
        # otherwise set to an empty placeholder triangle.
        self._cradle_meshes  = [self._new_mesh((0.85, 0.70, 0.12, 0.75))
                                for _ in range(2)]   # LEFT bellcrank (purple)
        self._cradle_lat_spring_meshes = [
            self._new_mesh((0.20, 0.55, 0.90, 0.75)) for _ in range(2)]
        # RIGHT bellcrank (blue) — distinct colour so the two bellcranks
        # are visually distinguishable in the 3D view.
        self._cradle_obl_spring_meshes = [
            self._new_mesh((0.95, 0.55, 0.10, 0.85)) for _ in range(2)]
        # HEAVE coilover (orange-ish)
        self._cradle_roll_damper_meshes = [
            self._new_mesh((0.95, 0.20, 0.20, 0.85)) for _ in range(2)]
        # ROLL coilover (red — visually distinct from heave damper)

        # ── HEAVE_TBAR third-element shock ───────────────────────────────
        # The heave 3rd element is a SHOCK (damper cylinder), NOT a bare
        # line.  One per axle (front / rear), drawn between the heave
        # spring's bracket attach and its chassis attach.  Orange-ish to
        # read as a damper, matching the decoupled heave coilover.
        self._heave_tbar_shock_meshes = [
            self._new_mesh((0.95, 0.55, 0.10, 0.85)) for _ in range(2)]

        # ── HEAVE_TBAR linkage (per-segment COLOURED) + bellcrank plates +
        #    corner-damper cylinders.  The whole heave-T-bar mechanism gets its
        #    own coloured render path (not the single-colour ARB line) so the
        #    pushrod, lever, torsion bar, drop links + corner dampers are all
        #    distinguishable.  Colours kept to the colour-blind-safe set
        #    (white / yellow / red / blue + grey).
        self._htb_link_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=np.ones((2, 4), np.float32),
            connect='segments', width=2.6, antialias=True,
            parent=self._view.scene,
        )
        # 4 bellcrank plates (FL, FR, RL, RR) — filled translucent gold.
        self._htb_bell_meshes = [
            self._new_mesh((0.78, 0.64, 0.12, 0.55)) for _ in range(4)]
        # 4 corner-damper cylinders (FL, FR, RL, RR) — orange, read as dampers.
        self._htb_coil_meshes = [
            self._new_mesh((0.95, 0.55, 0.10, 0.85)) for _ in range(4)]

        # ── Frame / interference overlay (control arms, ARB, rocker, pushrod
        # drawn at their real thickness) + rocker-pivot bearing clearance.
        # All hidden until the user enables the frame in the Direct Editor.
        self._frame_mesh   = self._new_mesh((0.78, 0.78, 0.80, 0.28))  # parts (grey)
        self._clash_mesh   = self._new_mesh((0.90, 0.10, 0.10, 0.60))  # interferences (red)
        self._bearing_mesh = self._new_mesh((0.92, 0.85, 0.20, 0.45))  # bearing clearance (yellow)
        for _m in (self._frame_mesh, self._clash_mesh, self._bearing_mesh):
            _m.visible = False

        # ── Baseline GHOST overlay ────────────────────────────────────────
        # Grey copy of the linkage frozen at 'Set baseline' time, so the
        # designer can always see where they started while nudging points.
        self._ghost_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(0.55, 0.55, 0.58, 0.38),
            connect='segments', width=1.6, antialias=True,
            parent=self._view.scene,
        )
        self._ghost_vis.visible = False
        # last-uploaded linkage line arrays (for ghost capture)
        self._last_link_pos = None
        self._last_arb_pos = None
        self._last_htb_pos = None

        self._ground = self._make_ground_grid()
        self._ground.visible = True

        # Rack visual (front axle only) — updated via update_rack()
        self._rack_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(0.70, 0.70, 0.20, 1.0),
            connect='segments', width=4.0,
            parent=self._view.scene,
        )
        # Rack end markers
        self._rack_markers = scene.Markers(parent=self._view.scene)
        self._rack_markers.set_data(
            pos=np.zeros((2, 3), np.float32),
            face_color=(0.90, 0.85, 0.20, 1.0),
            size=10, edge_width=0,
        )

        # ── Roll-centre + roll-axis overlays ──────────────────────────────
        # Two large translucent white markers (front RC, rear RC)
        self._rc_markers = scene.Markers(parent=self._view.scene)
        self._rc_markers.set_data(
            pos=np.zeros((2, 3), np.float32),
            face_color=(1.0, 1.0, 1.0, 0.45),
            size=20, edge_width=0,
        )
        self._rc_markers.visible = True

        # Line connecting front RC → rear RC (roll axis)
        self._roll_axis_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(1.0, 1.0, 1.0, 0.25),
            connect='segments', width=2.0,
            parent=self._view.scene,
        )
        self._roll_axis_vis.visible = True

        # ── CG sphere (translucent white) ────────────────────────────────
        self._cg_marker = scene.Markers(parent=self._view.scene)
        self._cg_marker.set_data(
            pos=np.zeros((1, 3), np.float32),
            face_color=(1.0, 1.0, 1.0, 0.35),
            size=22, edge_width=0,
        )
        self._cg_marker.visible = False

        # ── Pitch-axis overlays (front/rear SVIC + pitch axis line) ──────
        self._svic_markers = scene.Markers(parent=self._view.scene)
        self._svic_markers.set_data(
            pos=np.zeros((2, 3), np.float32),
            face_color=(1.0, 0.85, 0.0, 0.55),
            size=18, edge_width=0,
        )
        self._svic_markers.visible = False

        self._pitch_axis_vis = scene.Line(
            pos=np.zeros((2, 3), np.float32),
            color=(1.0, 0.85, 0.0, 0.30),
            connect='segments', width=2.0,
            parent=self._view.scene,
        )
        self._pitch_axis_vis.visible = False

        # ── camera state ──────────────────────────────────────────────────
        self._mouse_last = None
        self._mouse_btn  = None

        # ── hook Qt events directly on the native widget ──────────────────
        n = self._canvas.native
        n.setMouseTracking(True)
        n.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        n.mousePressEvent   = self._qt_press
        n.mouseMoveEvent    = self._qt_move
        n.mouseReleaseEvent = self._qt_release
        n.wheelEvent        = self._qt_wheel
        n.keyPressEvent     = self._qt_keypress

        # ── NavCube overlay ───────────────────────────────────────────────
        self._navcube = NavCube(parent=n, size=110)
        self._navcube.view_requested.connect(self._snap_camera)
        self._navcube.set_orientation(self._cam.azimuth, self._cam.elevation)

        # ── View-controls overlay UNDER the navcube: mode dropdown +
        #    perspective + floor toggles (the box the user asked for).
        from PyQt6.QtWidgets import QComboBox, QCheckBox, QWidget, QVBoxLayout
        self._viewctrl = QWidget(parent=n)
        self._viewctrl.setStyleSheet(
            'QWidget{background:rgba(18,18,22,190);border-radius:7px;}'
            'QComboBox{color:#e6e6e6;background:#2a2a30;border:1px solid #444;'
            'border-radius:3px;padding:2px 4px;font:11px "Segoe UI";}'
            'QCheckBox{color:#dcdce2;font:11px "Segoe UI";spacing:5px;}')
        _vl = QVBoxLayout(self._viewctrl)
        _vl.setContentsMargins(7, 5, 7, 5); _vl.setSpacing(3)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(['Normal', 'Load', 'Interference'])
        self._mode_combo.setToolTip(
            'Normal · Load (force vectors, desaturated) · Interference '
            '(highlight clashing parts)')
        self._persp_chk = QCheckBox('Perspective'); self._persp_chk.setChecked(True)
        self._floor_chk = QCheckBox('Floor'); self._floor_chk.setChecked(True)
        self._thick_chk = QCheckBox('Thickness'); self._thick_chk.setChecked(True)
        self._thick_chk.setToolTip('Draw members / ARB / pushrods / driveshaft / '
                                   'springs as solid tubes (off = thin lines)')
        _vl.addWidget(self._mode_combo)
        _vl.addWidget(self._persp_chk)
        _vl.addWidget(self._floor_chk)
        _vl.addWidget(self._thick_chk)
        self._viewctrl.adjustSize()
        self._on_view_controls = None
        self._mode_combo.currentTextChanged.connect(self._emit_view_controls)
        self._persp_chk.toggled.connect(self._emit_view_controls)
        self._floor_chk.toggled.connect(self._emit_view_controls)
        self._thick_chk.toggled.connect(self._emit_view_controls)

        # ── Watermark (top-right credit) ──────────────────────────────────
        self._watermark = QLabel('Made by Yu @ Cougar Racing', parent=n)
        self._watermark.setStyleSheet(
            'color: rgba(205, 210, 220, 165); background: transparent;'
            'font: 600 11px "Segoe UI";')
        self._watermark.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._watermark.adjustSize()
        self._watermark.raise_()

        # Load-mode hover tooltip (shows the load at the arrow under the cursor)
        self._load_tooltip = QLabel('', parent=n)
        self._load_tooltip.setStyleSheet(
            'color:#f4f4f6;background:rgba(18,18,22,225);border:1px solid #555;'
            'border-radius:4px;padding:3px 7px;font:12px "Segoe UI";')
        self._load_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._load_tooltip.hide()

        # position in top-right; will be repositioned on resize
        self._orig_resize = n.resizeEvent
        n.resizeEvent = self._qt_resize

        # ── picking state ─────────────────────────────────────────────────
        self._hp_snap:    list[tuple] = []   # [(name, pos3d), ...]
        self._selected:   str | None  = None
        self._selected_corner: str | None = None
        self._on_pick_cb  = None

        # ── direct-edit state ─────────────────────────────────────────────
        # WASD/QE moves the selected hardpoint by `_increment_mm`; keys 1–6
        # rebind the increment.  Movement is in WORLD frame and applied to
        # the FL/RL stored data (FR/RR are mirrored on render).
        self._edit_mode      = False
        self._increment_mm   = 1.0
        self._on_move_cb     = None     # cb(hp_name, corner, delta_xyz_m)
        self._on_increment_cb = None    # cb(increment_mm)

        # ── tire params ───────────────────────────────────────────────────
        self._tire_outer_r = 0.203
        self._tire_rim_r   = 0.165
        self._tire_half_w  = 0.100

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def native(self):
        return self._canvas.native

    def set_tire_params(self, outer_r, rim_r, half_w):
        self._tire_outer_r = outer_r
        self._tire_rim_r   = rim_r
        self._tire_half_w  = half_w

    def set_selected(self, name):
        self._selected = name

    def set_on_pick(self, cb):
        self._on_pick_cb = cb

    # ── Direct-edit mode API ──────────────────────────────────────────────
    def set_edit_mode(self, enabled: bool):
        """Enter/leave keyboard-driven hardpoint editing."""
        self._edit_mode = bool(enabled)
        if self._edit_mode:
            self._canvas.native.setFocus()
        self._canvas.update()

    def is_edit_mode(self) -> bool:
        return self._edit_mode

    def set_on_move(self, cb):
        """cb(hp_name, corner_label, delta_xyz_m) — fired on WASD/QE in edit mode."""
        self._on_move_cb = cb

    def set_on_increment(self, cb):
        """cb(increment_mm) — fired when keys 1-6 change the step."""
        self._on_increment_cb = cb

    def get_increment_mm(self) -> float:
        return self._increment_mm

    def get_selection(self):
        """Return (hp_name, corner) of the currently picked point, or (None, None)."""
        return self._selected, self._selected_corner

    def set_selection(self, hp_name: str | None, corner: str | None):
        """Programmatically set the picked hardpoint (used by the edit panel)."""
        self._selected = hp_name
        self._selected_corner = corner
        if self._edit_mode:
            self._canvas.native.setFocus()
        self._canvas.update()

    def set_increment_mm(self, mm: float):
        """Programmatically set the keyboard nudge step."""
        self._increment_mm = float(mm)

    def toggle_ground(self, visible: bool):
        self._ground.visible = visible

    def set_camera_center(self, center):
        """Set the orbit pivot point (world-space 3-tuple)."""
        self._cam.center = tuple(center)
        self._canvas.update()

    def update_rc(self, front_xyz, rear_xyz):
        """
        Update roll-centre sphere positions and roll axis.
        front_xyz / rear_xyz: (x, y, z) metres, or None to hide.
        """
        pts = [p for p in (front_xyz, rear_xyz) if p is not None]
        if pts:
            self._rc_markers.set_data(
                pos=np.array(pts, np.float32),
                face_color=(1.0, 1.0, 1.0, 0.45),
                size=20, edge_width=0,
            )
        if front_xyz is not None and rear_xyz is not None:
            self._roll_axis_vis.set_data(
                pos=np.array([front_xyz, rear_xyz], np.float32),
                color=(1.0, 1.0, 1.0, 0.25),
            )
        self._canvas.update()

    def update_cg(self, xyz):
        """Update CG sphere position. xyz: (x, y, z) in metres."""
        if xyz is not None:
            self._cg_marker.set_data(
                pos=np.array([xyz], np.float32),
                face_color=(1.0, 1.0, 1.0, 0.35),
                size=22, edge_width=0,
            )
            self._cg_marker.visible = True
        else:
            self._cg_marker.visible = False
        self._canvas.update()

    def set_cg_visible(self, visible: bool):
        self._cg_marker.visible = visible
        self._canvas.update()

    def set_rc_visible(self, visible: bool):
        self._rc_markers.visible = visible
        self._canvas.update()

    def set_roll_axis_visible(self, visible: bool):
        self._roll_axis_vis.visible = visible
        self._canvas.update()

    def update_pitch_axis(self, pitch_center_xyz, half_width: float = 1.0):
        """
        Draw the pitch axis as a LATERAL line (along X) through the pitch center.

        The pitch center is the side-view intersection of the front and rear
        suspension force lines (contact patch → SVIC), analogous to how the
        roll center is the front-view intersection of the arm IC lines.
        The pitch axis runs perpendicular to the roll axis (which is longitudinal).

        pitch_center_xyz : (x, y, z) metres
        half_width       : half-length of the displayed line (m), default = track half-width
        """
        if pitch_center_xyz is not None:
            pc = np.asarray(pitch_center_xyz, float)
            p_left  = np.array([pc[0] - half_width, pc[1], pc[2]], np.float32)
            p_right = np.array([pc[0] + half_width, pc[1], pc[2]], np.float32)
            self._svic_markers.set_data(
                pos=np.array([pc], np.float32),
                face_color=(1.0, 0.85, 0.0, 0.60),
                size=20, edge_width=0,
            )
            self._pitch_axis_vis.set_data(
                pos=np.array([p_left, p_right], np.float32),
                color=(1.0, 0.85, 0.0, 0.35),
            )
        self._canvas.update()

    def set_pitch_axis_visible(self, visible: bool):
        self._svic_markers.visible = visible
        self._pitch_axis_vis.visible = visible
        self._canvas.update()

    def update_rack(self, rack_left: np.ndarray, rack_right: np.ndarray):
        """
        Draw the steering rack as a thick line between left and right rack ends.
        rack_left / rack_right are world-space 3D positions of the rack tube ends
        (= the tie_rod_inner points on each front corner).
        """
        pos = np.array([rack_left, rack_right], np.float32)
        self._rack_vis.set_data(pos=pos, color=(0.70, 0.70, 0.20, 1.0))
        self._rack_markers.set_data(
            pos=pos,
            face_color=(0.90, 0.85, 0.20, 1.0),
            size=10, edge_width=0,
        )

    def update_scene(self, corners: list[dict], arb_segs=None):
        """
        corners: list of dicts with keys 'pts', 'spin_axis', 'label'
        arb_segs: list of (p0, p1) world-space segments for ARB
        """
        link_pos = []
        link_col = []
        member_segs = []          # (p0, p1, rgba) → solid member tubes
        mk_pos   = []
        mk_col   = []
        self._hp_snap = []

        for ci, corner in enumerate(corners):
            pts  = corner['pts']
            spin = corner['spin_axis']
            wc   = pts['wheel_center']

            # corner isolation (wheel-package view): hide the other corners
            if self._isolate and corner.get('label') != self._isolate:
                _z = np.zeros((3, 3), np.float32); _t = np.array([[0, 1, 2]], np.uint32)
                self._tire_meshes[ci].set_data(vertices=_z, faces=_t)
                self._upright_meshes[ci].set_data(vertices=_z, faces=_t)
                self._rocker_meshes[ci].set_data(vertices=_z, faces=_t)
                self._spring_meshes[ci].set_data(vertices=_z, faces=_t)
                self._rotor_meshes[ci].set_data(vertices=_z, faces=_t)
                self._caliper_meshes[ci].set_data(vertices=_z, faces=_t)
                continue

            # control-arm / pushrod / tie-rod members: drawn as SOLID TUBES
            # (real 3-D thickness), collected here and merged into one mesh
            # below.  (The old thin-line rendering had no thickness — a thin
            # pushrod looked like a fat bar when you zoomed in.)
            for pa, pb, col in LINKS:
                if (pa in pts and pb in pts
                        and np.all(np.isfinite(pts[pa]))
                        and np.all(np.isfinite(pts[pb]))):
                    member_segs.append((pts[pa], pts[pb], col))

            # upright polygon (UCA BJ, LCA BJ, tie rod outer)
            uv = np.array([pts['uca_outer'], pts['lca_outer'],
                           pts['tie_rod_outer']], np.float32)
            self._upright_meshes[ci].set_data(
                vertices=uv, faces=np.array([[0,1,2]], np.uint32))
            for pa2, pb2 in [('uca_outer','lca_outer'),
                              ('lca_outer','tie_rod_outer'),
                              ('tie_rod_outer','uca_outer')]:
                link_pos += [pts[pa2], pts[pb2]]
                link_col += [(0.65, 0.50, 0.38, 1.0)] * 2

            # rocker polygon: pivot at centre, 3 arms (pushrod, spring, ARB drop)
            # order by angle so the fan covers the whole rocker plate.
            #
            # Direct-damper topologies report rocker_pivot == pushrod_inner
            # (the rocker collapses to a point because there's no rocker).
            # Detect that and skip the rocker mesh — the spring line still
            # draws between rocker_spring_pt and spring_chassis_pt, which for
            # direct damper are the damper's two endpoints (i.e. the damper).
            rk4 = ('rocker_pivot', 'arb_drop_top', 'pushrod_inner', 'rocker_spring_pt')
            rk3 = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')

            def _have(keys):
                # Key present AND every coordinate finite.  DECOUPLED corners
                # report NaN pushrod_inner / rocker_spring_pt (the pushrod
                # feeds the shared twin-bellcrank cradle, NOT a per-corner
                # rocker), so this NaN guard keeps a stale bellcrank from a
                # previously-selected topology from being re-drawn.
                return all(k in pts and np.all(np.isfinite(pts[k]))
                           for k in keys)

            _rocker_collapsed = (_have(('rocker_pivot', 'pushrod_inner'))
                                 and float(np.linalg.norm(
                                     pts['rocker_pivot'] - pts['pushrod_inner'])) < 1e-4)
            # A genuine rocker PLATE has the pushrod attach and the spring
            # attach at DISTINCT points (non-zero spread).  A DIRECT damper
            # re-keys BOTH pushrod_inner and rocker_spring_pt to the same
            # damper-outer point, so the "plate" is degenerate.  Pulling
            # arb_drop_top into the 4-point fan there paints a large phantom
            # triangle -- a rocker that physically isn't present (e.g. DIRECT
            # damper + control-arm ARB shares the arb_drop_top vertex).
            # Require real spread before drawing the quad; otherwise fall
            # through to the degenerate triangle (rk3), which still renders
            # the damper as a line via its edges.
            _rocker_has_spread = (
                _have(('pushrod_inner', 'rocker_spring_pt'))
                and float(np.linalg.norm(
                    pts['pushrod_inner'] - pts['rocker_spring_pt'])) > 1e-4)
            if _rocker_collapsed:
                # Hide the rocker mesh entirely; clear it to a zero triangle.
                self._rocker_meshes[ci].set_data(
                    vertices=np.zeros((3, 3), np.float32),
                    faces=np.array([[0, 1, 2]], np.uint32))
            elif _have(rk4) and _rocker_has_spread:
                # bellcrank = flat PLATE: extrude the quad by the plate thickness
                # (a machined plate, not a tube frame).  Thin the plate with the
                # thickness toggle so it reads as an outline when OFF.
                _ht = self._rocker_half_t if self._thick_on else 0.0004
                rv, rf = build_prism([pts[k] for k in rk4], _ht)
                self._rocker_meshes[ci].set_data(vertices=rv, faces=rf)
            elif _have(rk3):
                _ht = self._rocker_half_t if self._thick_on else 0.0004
                rv, rf = build_prism([pts[k] for k in rk3], _ht)
                self._rocker_meshes[ci].set_data(vertices=rv, faces=rf)
            else:
                # No valid per-corner rocker (DECOUPLED — pushrod goes to the
                # shared cradle — or missing/NaN rocker points).  Clear the
                # mesh so a bellcrank drawn for a prior topology disappears
                # instead of lingering on screen.
                self._rocker_meshes[ci].set_data(
                    vertices=np.zeros((3, 3), np.float32),
                    faces=np.array([[0, 1, 2]], np.uint32))

            # tire (tread + sidewalls only, translucent)
            tv, tf = build_tire_mesh(
                wc, spin,
                self._tire_outer_r, self._tire_rim_r, self._tire_half_w)
            self._tire_meshes[ci].set_data(vertices=tv, faces=tf)

            # brake rotor (disc coaxial with the wheel) + caliper block at top
            if ci < len(self._rotor_meshes):
                _z = np.zeros((3, 3), np.float32); _t = np.array([[0, 1, 2]], np.uint32)
                if self._show_brakes:
                    s = _norm(np.asarray(spin, float))
                    rr = self._rotor_r
                    wcv = np.asarray(wc, float)
                    # rotor: disc, thickness = GP200 0.25 in rotor width
                    rv, rf = build_cylinder_between(
                        wcv - s * self._rotor_hw, wcv + s * self._rotor_hw, rr, n=32)
                    self._rotor_meshes[ci].set_data(vertices=rv, faces=rf)
                    # caliper: GP200 body ~97.8 mm long x 27.9 mm mount height x
                    # ~40 mm axial, straddling the rotor edge on the SIDE
                    # (fore-aft / trailing, not the top).
                    rad = np.array([0, 1.0, 0]) - np.dot([0, 1.0, 0], s) * s
                    rad = rad / max(np.linalg.norm(rad), 1e-9)   # rearward in wheel plane
                    tan = np.cross(s, rad)
                    cal_c = wcv + rad * (rr - 0.006)
                    cv, cf = build_box(cal_c, s, rad, tan,
                                       half_ax=0.020, half_rad=0.0140, half_tan=0.0489)
                    self._caliper_meshes[ci].set_data(vertices=cv, faces=cf)
                else:
                    self._rotor_meshes[ci].set_data(vertices=_z, faces=_t)
                    self._caliper_meshes[ci].set_data(vertices=_z, faces=_t)

            # markers
            #
            # `editable` is the set of keys that are REAL hardpoints for this
            # corner (== the keys of the corner's per-axle hp dict).  It is set
            # by the renderer in main_window._assemble_corners.  _state_to_pts
            # unconditionally injects derived render points (e.g. a direct
            # damper reports pushrod_inner == damper attach; a decoupled corner
            # is fed by a shared cradle) — those points are needed to DRAW the
            # geometry but must NOT become pickable markers, because clicking a
            # phantom that isn't in the edit list selects nothing (a ghost
            # marker).  Gate marker creation on `editable` so only genuine,
            # movable hardpoints get a pickable dot + a _hp_snap entry.
            editable = corner.get('editable')   # None for legacy callers
            for name in HP_NAMES:
                if name not in pts:
                    continue
                if editable is not None and name not in editable:
                    continue   # derived / visualisation-only point, not a hardpoint
                p = pts[name]
                if not np.all(np.isfinite(p)):
                    continue   # DECOUPLED: pushrod_inner / rocker_spring_pt NaN
                mk_pos.append(p)
                if name == self._selected:
                    c = _C_SEL
                elif name in CHASSIS_PTS:
                    c = _C_CHASSIS
                else:
                    c = _C_MOVING
                mk_col.append(c)
                self._hp_snap.append((name, p.copy(), corner['label']))

        # thickness OFF → members render as crisp THIN LINES.  A sub-millimetre
        # tube renders as a dotted/broken mesh, so route the member segments into
        # the Line visual instead; the tube mesh below is then built from an empty
        # list (cleared).
        if not self._thick_on:
            for _p0, _p1, _c in member_segs:
                link_pos += [_p0, _p1]
                link_col += [_c, _c]
            member_segs = []

        # upload the thin lines (upright + rocker plate edges only now)
        if link_pos:
            self._last_link_pos = np.array(link_pos, np.float32)
            lc = np.array(link_col, np.float32)
            if self._view_mode in ('load', 'interference') and len(lc):
                # desaturate the wireframe so overlaid force vectors / clash
                # highlights read clearly on top of a pale skeleton.
                grey = np.array([0.78, 0.78, 0.80, 1.0], np.float32)
                lc = lc * 0.30 + grey * 0.70
                lc[:, 3] = 0.55
            self._links_vis.set_data(pos=self._last_link_pos, color=lc)
        else:
            self._links_vis.set_data(pos=np.zeros((2, 3), np.float32))

        # ── member TUBES (control arms / pushrod / tie rods) — solid 3-D ─────
        mv, mf, mc = _tube_segments(member_segs, self._member_r, n=10)
        if self._view_mode in ('load', 'interference') and len(mc):
            grey = np.array([0.78, 0.78, 0.80, 1.0], np.float32)
            mc = mc * 0.30 + grey * 0.70
            mc[:, 3] = 0.55
        self._member_mesh.set_data(vertices=mv, faces=mf, vertex_colors=mc)

        # ── desaturate every solid car part in Load / Interference mode so
        #    ONLY the force vectors / clash highlights keep colour ──────────
        _des = self._view_mode in ('load', 'interference')
        for _m, _base in self._car_meshes:
            try:
                _m.color = (0.5, 0.5, 0.52, 0.30) if _des else _base
            except Exception:
                pass

        # upload markers
        if mk_pos:
            self._markers.set_data(
                pos=np.array(mk_pos, np.float32),
                face_color=np.array(mk_col, np.float32),
                size=9, edge_width=0)

        # ARB — SOLID TUBE (thickness ON) or a crisp THIN LINE (thickness OFF, so
        # it doesn't render as a dotted sub-mm tube); hidden when a corner is
        # isolated, desaturated in Load / Interference mode.
        self._arb_vis.set_data(pos=np.zeros((2, 3), np.float32))
        if arb_segs and not self._isolate:
            ap = np.array([p for seg in arb_segs for p in seg], np.float32)
            self._last_arb_pos = ap
            _acol = ((0.55, 0.55, 0.57, 0.9)
                     if self._view_mode in ('load', 'interference')
                     else (0.90, 0.80, 0.10, 1.0))
            if self._thick_on:
                _asegs = [(seg[0], seg[1], _acol) for seg in arb_segs]
                av, af, ac = _tube_segments(_asegs, self._member_r, n=10)
                self._arb_mesh.set_data(vertices=av, faces=af, vertex_colors=ac)
            else:
                self._arb_mesh.set_data(vertices=np.zeros((3, 3), np.float32),
                                        faces=np.array([[0, 1, 2]], np.uint32))
                self._arb_vis.set_data(pos=ap, color=_acol)
        else:
            self._last_arb_pos = None
            self._arb_mesh.set_data(vertices=np.zeros((3, 3), np.float32),
                                    faces=np.array([[0, 1, 2]], np.uint32))

        # ── Spring / damper cylinders (per corner) ───────────────────────
        # Draw a translucent cylinder of the user-set OD between
        # rocker_spring_pt and spring_chassis_pt.  Direct-damper topologies
        # re-key those points to the damper endpoints upstream, so this
        # one path handles both spring-on-rocker and direct-damper rendering.
        for ci, corner in enumerate(corners):
            if ci >= len(self._spring_meshes):
                break
            pts = corner['pts']
            if self._isolate and corner.get('label') != self._isolate:
                self._spring_meshes[ci].set_data(
                    vertices=np.zeros((3, 3), np.float32),
                    faces=np.array([[0, 1, 2]], np.uint32))
                continue
            if ('rocker_spring_pt' in pts) and ('spring_chassis_pt' in pts) \
                    and np.all(np.isfinite(pts['rocker_spring_pt'])) \
                    and np.all(np.isfinite(pts['spring_chassis_pt'])):
                # Pick OD: damper for collapsed-rocker (DIRECT) case,
                # spring otherwise.  We use rocker_pivot == pushrod_inner
                # as the marker for "rocker collapsed → this segment is
                # the damper alone" (same heuristic used for the rocker
                # mesh).
                if ('rocker_pivot' in pts and 'pushrod_inner' in pts
                        and float(np.linalg.norm(
                            pts['rocker_pivot'] - pts['pushrod_inner'])) < 1e-4):
                    od_m = self._damper_od_m
                else:
                    od_m = self._spring_od_m
                v, f = build_cylinder_between(
                    pts['rocker_spring_pt'], pts['spring_chassis_pt'],
                    radius=0.5 * od_m, n=20)
                self._spring_meshes[ci].set_data(vertices=v, faces=f)
            else:
                # No spring on this corner (e.g. DECOUPLED corner has none)
                self._spring_meshes[ci].set_data(
                    vertices=np.zeros((3, 3), np.float32),
                    faces=np.array([[0, 1, 2]], np.uint32))

        self._canvas.update()

    # ── Decoupled twin-bellcrank rendering ──────────────────────────────
    def update_decoupled_cradles(self, front_dict: dict | None,
                                 rear_dict:  dict | None) -> None:
        """Render the reference twin-bellcrank decoupled layout per axle.

        Each non-empty dict must carry the 10 keys (L + R) for the two
        bellcranks + the two cross-car coilovers:
            rocker_pivot_left/right, rocker_axis_pt_left/right,
            pushrod_inner_left/right,
            heave_damper_left/right,   (cross-car HEAVE coilover ends)
            roll_damper_left/right     (cross-car ROLL coilover ends —
                                        asymmetric Z: LEFT below pivot,
                                        RIGHT above pivot)

        Rendering:
          * Two bellcrank plates per axle (LEFT + RIGHT), each as a
            triangle-fan quadrilateral whose vertices are the 4
            rocker-resident points (pivot, pushrod_inner, heave_attach,
            roll_attach).  We reuse self._cradle_meshes[slot] for the
            LEFT bellcrank and self._cradle_lat_spring_meshes[slot] for
            the RIGHT bellcrank (was: lateral spring mesh, now repurposed).
          * Heave coilover: cross-car cylinder between heave_damper_left
            and heave_damper_right, using the DAMPER OD.
          * Roll coilover: cross-car cylinder between roll_damper_left
            and roll_damper_right, using the SPRING OD (slightly thicker
            so the user can tell them apart at a glance — the colour
            also differs: lateral mesh = orange-ish, oblique mesh = blue).
          * For backward-compat with the old cradle-mesh attribute names,
            we keep ``self._cradle_meshes`` (one quad per axle) and
            reuse the two spring mesh slots for: ``_cradle_lat_spring_meshes``
            → ROLL coilover cylinder, ``_cradle_obl_spring_meshes`` →
            HEAVE coilover cylinder + RIGHT bellcrank.  Total visual
            slots per axle: 1 LEFT bellcrank + 1 RIGHT bellcrank + 2
            damper cylinders, plus we add a small line/cylinder for
            each rocker's pivot axis indicator.
        """
        empty_v = np.zeros((3, 3), np.float32)
        empty_f = np.array([[0, 1, 2]], np.uint32)

        for slot, d in enumerate((front_dict, rear_dict)):
            # Note: with the redesign we now need MORE mesh slots than the
            # old 3 per axle.  Reuse what's already wired and clear unused:
            left_bell_mesh   = self._cradle_meshes[slot]
            right_bell_mesh  = self._cradle_lat_spring_meshes[slot]
            heave_coil_mesh  = self._cradle_obl_spring_meshes[slot]
            # roll damper cylinder needs its own slot — see __init__

            # Clear if this axle isn't decoupled
            if not d or 'rocker_pivot_left' not in d:
                left_bell_mesh.set_data(vertices=empty_v, faces=empty_f)
                right_bell_mesh.set_data(vertices=empty_v, faces=empty_f)
                heave_coil_mesh.set_data(vertices=empty_v, faces=empty_f)
                if hasattr(self, '_cradle_roll_damper_meshes'):
                    self._cradle_roll_damper_meshes[slot].set_data(
                        vertices=empty_v, faces=empty_f)
                continue

            try:
                pivL = np.asarray(d['rocker_pivot_left'],    float)
                pivR = np.asarray(d['rocker_pivot_right'],   float)
                pinL = np.asarray(d['pushrod_inner_left'],   float)
                pinR = np.asarray(d['pushrod_inner_right'],  float)
                heaveL = np.asarray(d['heave_damper_left'],  float)
                heaveR = np.asarray(d['heave_damper_right'], float)
                rollL  = np.asarray(d['roll_damper_left'],   float)
                rollR  = np.asarray(d['roll_damper_right'],  float)
            except KeyError:
                left_bell_mesh.set_data(vertices=empty_v, faces=empty_f)
                right_bell_mesh.set_data(vertices=empty_v, faces=empty_f)
                heave_coil_mesh.set_data(vertices=empty_v, faces=empty_f)
                if hasattr(self, '_cradle_roll_damper_meshes'):
                    self._cradle_roll_damper_meshes[slot].set_data(
                        vertices=empty_v, faces=empty_f)
                continue

            # Bellcrank quadrilateral as triangle-fan from the centre.
            # SVD-projected angle ordering keeps the fan non-self-crossing
            # for arbitrary 3-D quad orientations.
            def _bell_quad(pivot, pin, heave_pt, roll_pt):
                centre = 0.25 * (pivot + pin + heave_pt + roll_pt)
                pts4 = np.array([pivot, pin, heave_pt, roll_pt], np.float32)
                rel = pts4 - centre
                try:
                    _, _, vh = np.linalg.svd(rel, full_matrices=False)
                    u_loc = rel @ vh[0]
                    v_loc = rel @ vh[1]
                    order = np.argsort(np.arctan2(v_loc, u_loc))
                    pts4 = pts4[order]
                except np.linalg.LinAlgError:
                    pass
                verts = np.vstack([centre[None, :].astype(np.float32), pts4])
                faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                                 np.uint32)
                return verts, faces

            vL, fL = _bell_quad(pivL, pinL, heaveL, rollL)
            vR, fR = _bell_quad(pivR, pinR, heaveR, rollR)
            left_bell_mesh.set_data(vertices=vL, faces=fL)
            right_bell_mesh.set_data(vertices=vR, faces=fR)

            # Heave coilover cylinder (cross-car) — use DAMPER OD so it
            # looks distinct from the spring; renders in the obl_spring
            # slot which has a blue tint.
            v, f = build_cylinder_between(
                heaveL, heaveR, radius=0.5 * self._damper_od_m, n=20)
            heave_coil_mesh.set_data(vertices=v, faces=f)

            # Roll coilover cylinder (cross-car, diagonal) — separate
            # mesh slot so it renders with its own colour.
            if hasattr(self, '_cradle_roll_damper_meshes'):
                v, f = build_cylinder_between(
                    rollL, rollR, radius=0.5 * self._spring_od_m, n=20)
                self._cradle_roll_damper_meshes[slot].set_data(
                    vertices=v, faces=f)

    # ── HEAVE_TBAR third-element shock rendering ─────────────────────────
    def update_heave_tbar_shock(self, front_heave: dict | None,
                                rear_heave:  dict | None) -> None:
        """Render the HEAVE_TBAR third element as a SHOCK (damper cylinder).

        The heave 3rd element is a damper/coilover, NOT a suspension link —
        per the user's design intent it must read as a shock cylinder in the
        3D view, not a thin line.  Draw a cylinder of the damper OD between
        the bracket-side spring attach (``heave_spring_tbar_pt``) and the
        chassis-side attach (``heave_spring_chassis_pt``) for each axle that
        carries a heave element; clear the slot otherwise.

        The endpoints are the design-ride pose (the same pose at which every
        other element in update_scene is drawn), so this is faithful to the
        solved geometry without inventing any motion.
        """
        empty_v = np.zeros((3, 3), np.float32)
        empty_f = np.array([[0, 1, 2]], np.uint32)
        for slot, d in enumerate((front_heave, rear_heave)):
            mesh = self._heave_tbar_shock_meshes[slot]
            if not d or 'heave_spring_tbar_pt' not in d \
                    or 'heave_spring_chassis_pt' not in d:
                mesh.set_data(vertices=empty_v, faces=empty_f)
                continue
            p0 = np.asarray(d['heave_spring_tbar_pt'],    float)
            p1 = np.asarray(d['heave_spring_chassis_pt'], float)
            if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                mesh.set_data(vertices=empty_v, faces=empty_f)
                continue
            v, f = build_cylinder_between(p0, p1,
                                          radius=0.5 * self._damper_od_m, n=20)
            mesh.set_data(vertices=v, faces=f)

    # ── HEAVE_TBAR full mechanism (coloured linkage + plates + dampers) ──
    @staticmethod
    def _plate_mesh(pts):
        """Triangle-fan plate from N (>=3) coplanar-ish points, ordered around
        their centroid (SVD projection) so the fan never self-crosses."""
        pts = [np.asarray(p, float) for p in pts
               if p is not None and np.all(np.isfinite(p))]
        if len(pts) < 3:
            return (np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32))
        P = np.array(pts, float)
        c = P.mean(axis=0)
        rel = P - c
        try:
            _, _, vh = np.linalg.svd(rel, full_matrices=False)
            order = np.argsort(np.arctan2(rel @ vh[1], rel @ vh[0]))
            P = P[order]
        except np.linalg.LinAlgError:
            pass
        verts = np.vstack([c[None, :], P]).astype(np.float32)
        n = len(P)
        faces = np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], np.uint32)
        return verts, faces

    def update_heave_tbar(self, front_pose: dict | None,
                          rear_pose: dict | None) -> None:
        """Render the heave-T-bar mechanism (both axles) from the solver's
        ``pose_axle()`` output: ONE T-bar (grey torsion bar + gold lever across),
        per-side gold bellcrank PLATES, WHITE pushrods, blue drop links, and the
        two orange corner-damper cylinders.  Each element is colour-coded so the
        pushrod is distinct from the rest (no longer one flat ARB colour)."""
        WHITE = (0.92, 0.92, 0.96, 1.0); GOLD = (0.85, 0.70, 0.12, 1.0)
        GREY = (0.55, 0.55, 0.60, 1.0); BLUE = (0.30, 0.55, 0.95, 1.0)
        LINE_SPEC = [
            ('pushrod_outer_L', 'pushrod_inner_L', WHITE),
            ('pushrod_outer_R', 'pushrod_inner_R', WHITE),
            ('tbar_pivot', 'junc', GREY),
            ('left_tip', 'junc', GOLD), ('junc', 'right_tip', GOLD),
            ('drop_foot_L', 'left_tip', BLUE), ('drop_foot_R', 'right_tip', BLUE),
            ('rocker_pivot_L', 'pushrod_inner_L', GOLD),
            ('rocker_pivot_L', 'drop_foot_L', GOLD),
            ('rocker_pivot_L', 'coil_rocker_L', GOLD),
            ('rocker_pivot_R', 'pushrod_inner_R', GOLD),
            ('rocker_pivot_R', 'drop_foot_R', GOLD),
            ('rocker_pivot_R', 'coil_rocker_R', GOLD),
        ]
        empty_v = np.zeros((3, 3), np.float32)
        empty_f = np.array([[0, 1, 2]], np.uint32)
        pos, col = [], []
        for slot, p in enumerate((front_pose, rear_pose)):
            bL, bR = 2 * slot, 2 * slot + 1
            if not p:
                for mi in (bL, bR):
                    self._htb_bell_meshes[mi].set_data(vertices=empty_v, faces=empty_f)
                    self._htb_coil_meshes[mi].set_data(vertices=empty_v, faces=empty_f)
                continue
            for a, b, c in LINE_SPEC:
                pa, pb = p.get(a), p.get(b)
                if (pa is not None and pb is not None
                        and np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
                    pos += [pa, pb]; col += [c, c]
            # bellcrank plates (L + R)
            for mi, keys in ((bL, ('rocker_pivot_L', 'pushrod_inner_L', 'drop_foot_L', 'coil_rocker_L')),
                             (bR, ('rocker_pivot_R', 'pushrod_inner_R', 'drop_foot_R', 'coil_rocker_R'))):
                v, f = self._plate_mesh([p.get(k) for k in keys])
                self._htb_bell_meshes[mi].set_data(vertices=v, faces=f)
            # corner-damper cylinders (L + R)
            for mi, (rk, ch) in ((bL, ('coil_rocker_L', 'coil_chassis_L')),
                                 (bR, ('coil_rocker_R', 'coil_chassis_R'))):
                rkp, chp = p.get(rk), p.get(ch)
                if (rkp is not None and chp is not None
                        and np.all(np.isfinite(rkp)) and np.all(np.isfinite(chp))):
                    v, f = build_cylinder_between(rkp, chp, radius=0.5 * self._damper_od_m, n=18)
                    self._htb_coil_meshes[mi].set_data(vertices=v, faces=f)
                else:
                    self._htb_coil_meshes[mi].set_data(vertices=empty_v, faces=empty_f)
        if pos:
            self._last_htb_pos = np.array(pos, np.float32)
            self._htb_link_vis.set_data(pos=self._last_htb_pos,
                                        color=np.array(col, np.float32))
        else:
            self._last_htb_pos = None
            self._htb_link_vis.set_data(pos=np.zeros((2, 3), np.float32),
                                        color=np.ones((2, 4), np.float32))
        self._canvas.update()

    # ── Baseline ghost API ───────────────────────────────────────────────
    def capture_ghost(self):
        """Concatenate the currently-rendered linkage line segments into one
        array (links + ARB + heave-T-bar) — frozen by set_ghost()."""
        segs = [p for p in (self._last_link_pos, self._last_arb_pos,
                            self._last_htb_pos)
                if p is not None and len(p) >= 2]
        return np.vstack(segs).copy() if segs else None

    def set_ghost(self, pos) -> None:
        """Freeze `pos` (segment-pair array) as the grey baseline ghost."""
        if pos is None or len(pos) < 2:
            self._ghost_vis.visible = False
        else:
            self._ghost_vis.set_data(pos=np.asarray(pos, np.float32),
                                     color=(0.55, 0.55, 0.58, 0.38))
            self._ghost_vis.visible = True
        self._canvas.update()

    def set_ghost_visible(self, visible: bool) -> None:
        self._ghost_vis.visible = bool(visible)
        self._canvas.update()

    # ── Frame / interference / bearing overlay ───────────────────────────
    def set_perspective(self, on: bool):
        """Toggle projection: True = perspective (eye view), False =
        orthographic / parallel (CAD view, no foreshortening)."""
        try:
            self._cam.fov = self._persp_fov if on else 0.0
            self._canvas.update()
        except Exception:
            pass

    def _combine_cyls(self, cyls, n=12):
        """Merge a list of (p0, p1, radius_m) into one (verts, faces) mesh."""
        VV = []; FF = []; off = 0
        for p0, p1, r in cyls:
            v, f = build_cylinder_between(np.asarray(p0, float),
                                          np.asarray(p1, float), float(r), n=n)
            VV.append(v); FF.append(np.asarray(f) + off); off += len(v)
        if not VV:
            return None, None
        return (np.concatenate(VV).astype(np.float32),
                np.concatenate(FF).astype(np.uint32))

    def set_frame_overlay(self, part_cyls, clash_cyls, bearing_cyls, visible):
        """Show structural parts at thickness (part_cyls), flagged
        interferences (clash_cyls, red) and rocker-pivot bearing clearance
        (bearing_cyls, yellow).  Each list holds (p0, p1, radius_m) tubes."""
        for mesh, cyls in ((self._frame_mesh, part_cyls),
                           (self._clash_mesh, clash_cyls),
                           (self._bearing_mesh, bearing_cyls)):
            v, f = self._combine_cyls(cyls) if (visible and cyls) else (None, None)
            if v is not None:
                mesh.set_data(vertices=v, faces=f); mesh.visible = True
            else:
                mesh.visible = False

    # ── Public dimension setter ─────────────────────────────────────────
    def set_spring_dims(self, spring_od_mm: float, damper_od_mm: float) -> None:
        """Update the spring / damper outside diameter (mm).

        Called by the host whenever the user-input OD changes (wizard, car
        panel, file load).  Next ``update_scene`` call rebuilds the spring
        cylinders at the new size.
        """
        try:
            self._spring_od_m = max(0.001, float(spring_od_mm) / 1000.0)
            self._damper_od_m = max(0.001, float(damper_od_mm) / 1000.0)
        except (TypeError, ValueError):
            pass

    def set_view_mode(self, mode: str) -> None:
        """Set the 3D view mode: 'normal', 'load' or 'interference'.

        The next update_scene() re-colours accordingly (load/interference
        desaturate the wireframe links so overlaid vectors / clash highlights
        stand out).  Interference mode is also where set_clashes() draws.
        """
        self._view_mode = mode if mode in ('normal', 'load', 'interference') else 'normal'

    def set_clashes(self, segments) -> None:
        """Highlight interfering members: segments = list of (p0, p1) world
        points (metres).  Empty / None clears the highlight."""
        try:
            self._clash_count = len(segments) if segments else 0
            if segments:
                pos = np.array([p for seg in segments for p in seg], np.float32)
                self._clash_vis.set_data(pos=pos, color=(0.95, 0.15, 0.15, 1.0))
            else:
                self._clash_vis.set_data(pos=np.zeros((2, 3), np.float32))
        except Exception:
            pass

    def set_on_view_controls(self, cb) -> None:
        """Register cb(dict) fired when the navcube view-controls change
        (keys: view_mode, perspective, floor)."""
        self._on_view_controls = cb

    def _emit_view_controls(self, *_):
        if self._on_view_controls:
            self._on_view_controls({
                'view_mode': self._mode_combo.currentText().lower(),
                'perspective': self._persp_chk.isChecked(),
                'floor': self._floor_chk.isChecked(),
                'thickness': self._thick_chk.isChecked()})

    def sync_view_controls(self, view_mode=None, perspective=None, floor=None,
                           thickness=None):
        """Set the overlay controls to match external state without re-firing."""
        for w, v, setter in ((self._mode_combo, view_mode, 'combo'),
                             (self._persp_chk, perspective, 'chk'),
                             (self._floor_chk, floor, 'chk'),
                             (self._thick_chk, thickness, 'chk')):
            if v is None:
                continue
            w.blockSignals(True)
            if setter == 'combo':
                w.setCurrentText(str(v).capitalize())
            else:
                w.setChecked(bool(v))
            w.blockSignals(False)

    def set_thickness(self, on) -> None:
        """Toggle solid-tube thickness for members / ARB / pushrods / driveshaft.
        OFF → member+ARB tubes shrink to ~1 mm wisps (read as lines); the
        driveshaft/tripod thin in set_driveshaft_package; springs thin via
        set_spring_dims (driven by the same show_shock_thickness flag).
        Caller must re-run update_scene / set_driveshaft_package to apply."""
        self._thick_on = bool(on)
        self._member_r = self._member_r_full if self._thick_on else self._member_r_thin

    def set_brakes(self, show: bool) -> None:
        """Show/hide the brake rotors + calipers in the 3-D view."""
        self._show_brakes = bool(show)

    def set_brake_dims(self, rotor_dia_mm) -> None:
        """Rotor diameter (mm); rotor thickness stays the GP200's 0.25 in."""
        try:
            self._rotor_r = max(0.02, float(rotor_dia_mm) / 2.0 / 1000.0)
        except (TypeError, ValueError):
            pass

    def set_isolate_corner(self, lbl) -> None:
        """Draw only corner ``lbl`` (e.g. 'RL'); None shows all four.  Used by
        the wheel-package view to isolate one corner."""
        self._isolate = lbl if lbl in ('FL', 'FR', 'RL', 'RR') else None

    def set_load_vectors(self, arrows) -> None:
        """Draw force-vector arrows in Load mode.

        arrows: list of (p, tip, rgba) world-point pairs + colour.  Each is a
        shaft plus a 4-barb 3-D arrowhead so the direction reads from any angle.
        Empty / None clears them.
        """
        try:
            self._loadvec_count = len(arrows) if arrows else 0
            self._loadvec_snap = []          # [(tip_world, label)] for hover
            if not arrows:
                self._loadvec_vis.set_data(pos=np.zeros((2, 3), np.float32))
                return
            segs = []; cols = []
            for a in arrows:
                p = np.asarray(a[0], float); tip = np.asarray(a[1], float); c = a[2]
                lab = a[3] if len(a) > 3 else ''
                self._loadvec_snap.append((tip.copy(), lab))
                segs += [p, tip]; cols += [c, c]
                d = tip - p; L = float(np.linalg.norm(d))
                if L < 1e-6:
                    continue
                u = d / L
                perp = np.cross(u, [0, 0, 1.0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(u, [0, 1.0, 0])
                perp = perp / np.linalg.norm(perp)
                perp2 = np.cross(u, perp)
                hl = min(0.02, 0.32 * L)
                for pp in (perp, -perp, perp2, -perp2):
                    barb = tip - u * hl + pp * hl * 0.55
                    segs += [tip, barb]; cols += [c, c]
            self._loadvec_vis.set_data(pos=np.array(segs, np.float32),
                                       color=np.array(cols, np.float32))
        except Exception:
            pass

    def set_driveshaft_package(self, pkg, show: bool = True, only=None) -> None:
        """Draw the rear diff body, tripods and half-shafts as thick cylinders.

        pkg: the dict from vahan.driveshaft.package(car, rear_states) (or None).
        Each half-shaft's OUTER end is the LIVE solved wheel_center carried in
        pkg, so the shafts move with the uprights every travel step (ONE MODEL).
        show=False (or pkg None) clears them.  only='RL'/'RR' draws just that
        corner's shaft + tripod and hides the (shared) diff body — for the
        isolated wheel-package view.
        """
        def _clear(m):
            m.set_data(vertices=np.zeros((3, 3), np.float32),
                       faces=np.array([[0, 1, 2]], np.uint32))
        if not pkg or not show:
            _clear(self._diff_mesh)
            for m in self._driveshaft_meshes + self._tripod_meshes:
                _clear(m)
            return
        c = np.asarray(pkg['diff_center'], float)
        span = float(pkg['housing_width_mm']) / 1000.0   # inboard pivot spacing
        shaft_r = 0.5 * float(pkg['shaft_dia_mm']) / 1000.0
        tri_r = 0.5 * float(pkg['tripod_od_mm']) / 1000.0
        body_w = 0.55 * span                  # diff BODY is shorter than the span
        diff_r = max(0.14 * span, 0.03)       # diff body radial size (visual)
        if not self._thick_on:                # thickness toggle OFF → thin wisps
            shaft_r = tri_r = 0.0006
            diff_r = max(0.10 * diff_r, 0.004)
        if only:                              # isolating one corner → no diff body
            _clear(self._diff_mesh)
        else:
            dv, df = build_cylinder_between(c + np.array([-body_w / 2, 0, 0]),
                                            c + np.array([body_w / 2, 0, 0]), diff_r, n=20)
            self._diff_mesh.set_data(vertices=dv, faces=df)
        for i, corner in enumerate(('RL', 'RR')):
            seg = pkg.get(corner)
            if not seg or (only and corner != only):
                _clear(self._driveshaft_meshes[i]); _clear(self._tripod_meshes[i])
                continue
            inner = np.asarray(seg['inner'], float)
            outer = np.asarray(seg['outer'], float)
            axis = np.asarray(seg['axis'], float)
            sv, sf = build_cylinder_between(inner, outer, shaft_r, n=16)
            self._driveshaft_meshes[i].set_data(vertices=sv, faces=sf)
            # tripod = stubby cylinder at the inner (diff) end, along the shaft
            t = float(pkg['tripod_od_mm']) / 1000.0
            tv, tf = build_cylinder_between(inner - axis * 0.5 * t,
                                            inner + axis * 0.5 * t, tri_r, n=16)
            self._tripod_meshes[i].set_data(vertices=tv, faces=tf)

    # ── internal ─────────────────────────────────────────────────────────────

    def _new_mesh(self, color):
        return scene.Mesh(
            vertices=np.zeros((3,3), np.float32),
            faces=np.array([[0,1,2]], np.uint32),
            color=color,
            parent=self._view.scene,
        )

    def _make_ground_grid(self):
        """Ground grid spanning ±8 m laterally, -2..12 m longitudinally."""
        segs = []
        for x in np.linspace(-8.0, 8.0, 33):    # 0.5 m spacing
            segs += [[x, -2.0, 0], [x, 12.0, 0]]
        for y in np.linspace(-2.0, 12.0, 57):   # 0.25 m spacing
            segs += [[-8.0, y, 0], [8.0, y, 0]]
        pos = np.array(segs, np.float32)
        col = np.full((len(pos), 4), [0.10, 0.10, 0.10, 1.0], np.float32)
        return scene.Line(pos=pos, color=col, connect='segments',
                          width=1.0, parent=self._view.scene)

    # ── Qt camera events ──────────────────────────────────────────────────────

    def _qt_press(self, event):
        try:
            self._mouse_last = event.pos()
            self._mouse_btn  = event.button()
            if event.button() == Qt.MouseButton.LeftButton:
                self._try_pick(event.pos())
        except Exception:
            pass

    def _hover_load(self, qpos):
        """Show the load label of the force-arrow nearest the cursor (Load mode)."""
        try:
            # scene -> canvas pixels THROUGH the camera.  NOTE: the ViewBox's own
            # get_transform('scene','canvas') sits ABOVE the camera and returns the
            # raw world metres (identity, w=1) — every tip then lands at pixel ~0,0
            # (top-left corner) so the tooltip only lit there.  node_transform from
            # the scene node to the canvas scene walks through the camera projection.
            tr = self._view.scene.node_transform(self._canvas.scene)
            best_lab, best_d2 = None, 22 ** 2
            for tip, lab in self._loadvec_snap:
                m = tr.map([float(tip[0]), float(tip[1]), float(tip[2]), 1.0])
                if m[3] <= 0:
                    continue
                sx, sy = m[0] / m[3], m[1] / m[3]
                d2 = (qpos.x() - sx) ** 2 + (qpos.y() - sy) ** 2
                if d2 < best_d2:
                    best_d2, best_lab = d2, lab
            if best_lab:
                self._load_tooltip.setText(best_lab); self._load_tooltip.adjustSize()
                self._load_tooltip.move(qpos.x() + 14, qpos.y() + 10)
                self._load_tooltip.show(); self._load_tooltip.raise_()
            else:
                self._load_tooltip.hide()
        except Exception:
            self._load_tooltip.hide()

    def _qt_move(self, event):
        try:
            # Load-mode hover: read the force at the arrow under the cursor.
            if (event.buttons() == Qt.MouseButton.NoButton
                    and self._view_mode == 'load' and self._loadvec_snap):
                self._hover_load(event.pos())
            elif self._load_tooltip.isVisible():
                self._load_tooltip.hide()
            if self._mouse_last is None:
                return
            cur = event.pos()
            dx  = cur.x() - self._mouse_last.x()
            dy  = cur.y() - self._mouse_last.y()
            self._mouse_last = cur

            buttons = event.buttons()
            mods    = event.modifiers()
            right   = bool(buttons & Qt.MouseButton.RightButton)
            middle  = bool(buttons & Qt.MouseButton.MiddleButton)
            ctrl    = bool(mods   & Qt.KeyboardModifier.ControlModifier)

            cam = self._cam
            if right and not ctrl:
                # Direct property assignment — works with interactive=False
                cam.azimuth   = (cam.azimuth   - dx * 0.45) % 360.0
                cam.elevation = float(np.clip(cam.elevation + dy * 0.45, -89.9, 89.9))
            elif middle or (right and ctrl):
                # cam.pan() silently fails with interactive=False.
                # Shift cam.center directly using world-space camera vectors.
                #
                # Camera position = center + scale * [sin(az)*cos(el), -cos(az)*cos(el), sin(el)]
                # View direction (toward center) = [-sin(az)*cos(el), cos(az)*cos(el), -sin(el)]
                # Camera RIGHT (cross(view_dir, world_up)) = [cos(az), sin(az), 0]
                az    = np.radians(cam.azimuth)
                sf    = cam.scale_factor * 0.003
                right_w = np.array([np.cos(az), np.sin(az), 0.0])   # screen-right in world
                c     = np.array(cam.center, float)
                c    -= right_w * (dx * sf)   # drag right → center shifts left → scene pans right
                c[2] += dy * sf               # drag down  → scene pans down
                cam.center = tuple(c)

            self._canvas.update()
            self._navcube.set_orientation(cam.azimuth, cam.elevation)
        except Exception:
            pass

    def _qt_wheel(self, event):
        try:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 0.88 ** (delta / 120.0)
            sf = max(0.05, self._cam.scale_factor * factor)
            self._cam.scale_factor = sf
            self._canvas.update()
        except Exception:
            pass

    def _qt_release(self, event):
        try:
            self._mouse_last = None
            self._mouse_btn  = None
        except Exception:
            pass

    # ── Keyboard (direct-edit mode only) ──────────────────────────────────
    # World-frame conventions: +X = lateral outboard for FL/RL (= driver's
    # left), +Y = longitudinal forward, +Z = up.
    #   W / S  →  +Y / −Y  (forward / back)
    #   A / D  →  +X / −X  (outboard / inboard, for FL/RL stored points)
    #   Q / E  →  +Z / −Z  (up / down)
    #   1-6    →  set step: 0.1 / 0.5 / 1 / 2 / 5 / 10 mm
    # Keys stored as plain ints — see eventFilter note in main_window.py.
    _INCREMENT_TABLE_MM = {
        int(Qt.Key.Key_1): 0.1, int(Qt.Key.Key_2): 0.5, int(Qt.Key.Key_3): 1.0,
        int(Qt.Key.Key_4): 2.0, int(Qt.Key.Key_5): 5.0, int(Qt.Key.Key_6): 10.0,
    }
    _MOVE_TABLE = {
        int(Qt.Key.Key_W): (1, +1.0), int(Qt.Key.Key_S): (1, -1.0),
        int(Qt.Key.Key_A): (0, +1.0), int(Qt.Key.Key_D): (0, -1.0),
        int(Qt.Key.Key_Q): (2, +1.0), int(Qt.Key.Key_E): (2, -1.0),
    }
    _CONSTRAINT_KEYS = {
        int(Qt.Key.Key_F): 'free',
        int(Qt.Key.Key_L): 'link',
        int(Qt.Key.Key_P): 'plane',
    }

    def set_on_constraint(self, cb):
        """cb(mode) — fired when F/L/P toggles the nudge constraint."""
        self._on_constraint_cb = cb

    def _cycle_selection(self, step: int):
        """Tab / Shift+Tab: walk the pickable points of the CURRENT corner
        (falls back to all corners when nothing is selected yet)."""
        if not self._hp_snap:
            return
        corner = self._selected_corner or 'FL'
        names = [n for n, _p, c in self._hp_snap if c == corner]
        if not names:
            names = [n for n, _p, c in self._hp_snap]
            corner = self._hp_snap[0][2]
        if not names:
            return
        try:
            i = names.index(self._selected)
        except ValueError:
            i = -step   # so the first Tab lands on names[0]
        nxt = names[(i + step) % len(names)]
        self._selected = nxt
        self._selected_corner = corner
        if self._on_pick_cb:
            self._on_pick_cb(nxt, corner)
        self._canvas.update()

    def _qt_keypress(self, event):
        if not self._edit_mode:
            return
        try:
            key = int(event.key())
            # Tab / Shift+Tab (or N / B — Qt's focus chain can eat Tab) —
            # cycle through the corner's points
            if key in (int(Qt.Key.Key_Tab), int(Qt.Key.Key_Backtab),
                       int(Qt.Key.Key_N), int(Qt.Key.Key_B)):
                back = (key in (int(Qt.Key.Key_Backtab), int(Qt.Key.Key_B))
                        or bool(event.modifiers()
                                & Qt.KeyboardModifier.ShiftModifier))
                self._cycle_selection(-1 if back else +1)
                return
            # F / L / P — nudge constraint mode
            if key in self._CONSTRAINT_KEYS:
                cb = getattr(self, '_on_constraint_cb', None)
                if cb:
                    cb(self._CONSTRAINT_KEYS[key])
                return
            if key in self._INCREMENT_TABLE_MM:
                self._increment_mm = self._INCREMENT_TABLE_MM[key]
                if self._on_increment_cb:
                    self._on_increment_cb(self._increment_mm)
                return
            if (self._selected is None or self._selected_corner is None
                    or self._on_move_cb is None or key not in self._MOVE_TABLE):
                return
            axis, sign = self._MOVE_TABLE[key]
            delta = np.zeros(3, dtype=float)
            delta[axis] = sign * self._increment_mm / 1000.0
            self._on_move_cb(self._selected, self._selected_corner, delta)
        except Exception:
            pass

    def _qt_resize(self, event):
        """Reposition the watermark + NavCube to the top-right corner on resize."""
        if self._orig_resize:
            self._orig_resize(event)
        w = event.size().width()
        self._watermark.adjustSize()
        self._watermark.move(w - self._watermark.width() - 8, 6)
        # NavCube sits just below the watermark so neither overlaps.
        self._navcube.move(w - self._navcube.width() - 4,
                           6 + self._watermark.height() + 4)
        # View-controls box sits just BELOW the navcube.
        if hasattr(self, '_viewctrl'):
            self._viewctrl.adjustSize()
            self._viewctrl.move(
                w - self._viewctrl.width() - 4,
                6 + self._watermark.height() + 4 + self._navcube.height() + 6)
            self._viewctrl.raise_()

    def _snap_camera(self, az, el):
        """Snap the camera to the requested orientation."""
        self._cam.azimuth = az % 360.0
        self._cam.elevation = float(np.clip(el, -89.9, 89.9))
        self._canvas.update()
        self._navcube.set_orientation(self._cam.azimuth, self._cam.elevation)

    # ── picking ───────────────────────────────────────────────────────────────

    def _try_pick(self, qpos):
        """
        Project each hardpoint through the camera to find the closest one
        to the click in screen space. Uses VisPy's camera transform.
        Callback receives (hp_name, corner_label) e.g. ('uca_front', 'FL').
        """
        if not self._hp_snap or not self._on_pick_cb:
            return
        try:
            # Same camera-aware transform as _hover_load (the plain ViewBox
            # get_transform excludes the camera → picking only worked at pixel 0,0).
            tr = self._view.scene.node_transform(self._canvas.scene)
            best_name, best_corner, best_d2 = None, None, 30**2
            for name, pos3d, corner_label in self._hp_snap:
                p4 = np.array([pos3d[0], pos3d[1], pos3d[2], 1.0], float)
                mapped = tr.map(p4)
                if mapped[3] <= 0:
                    continue
                sx = mapped[0] / mapped[3]
                sy = mapped[1] / mapped[3]
                d2 = (qpos.x() - sx)**2 + (qpos.y() - sy)**2
                if d2 < best_d2:
                    best_d2, best_name, best_corner = d2, name, corner_label
            if best_name:
                self._selected = best_name
                self._selected_corner = best_corner
                self._on_pick_cb(best_name, best_corner)
        except Exception:
            pass
