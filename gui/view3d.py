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
from PyQt6.QtWidgets import QWidget
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


# ── aero package mesh builders ───────────────────────────────────────────────

def _plate_strip(xs, y_le, y_te, z_le, z_te):
    """
    Thin plate from a set of spanwise stations.
    xs: (N,) span coords.  y_le/y_te/z_le/z_te: (N,) arrays.
    Returns (verts, faces).
    """
    n = len(xs)
    verts = []
    for i in range(n):
        verts.append([xs[i], y_le[i], z_le[i]])   # leading edge
        verts.append([xs[i], y_te[i], z_te[i]])   # trailing edge
    faces = []
    for i in range(n - 1):
        v0 = i * 2
        faces.append([v0, v0 + 1, v0 + 2])
        faces.append([v0 + 1, v0 + 3, v0 + 2])
    return np.array(verts, np.float32), np.array(faces, np.uint32)


def _rect(corners_4):
    """Two triangles from 4 corner vertices (flat quad)."""
    v = np.array(corners_4, np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    return v, f


def build_front_wing_mesh(y_center, z, span, chord):
    """
    Stylised front wing: reverse-swept main element, flap, endplates.

    y_center : longitudinal centre of the wing (m)
    z        : height above ground (m)
    span     : total wingspan (m)
    chord    : main element chord (m)
    """
    hs = span / 2.0
    ns = 24
    xs = np.linspace(-hs, hs, ns)
    t  = xs / hs                              # -1 … 1

    # ── main element: reverse sweep + slight dihedral ──
    sweep  = chord * 0.38 * (1.0 - t ** 2)   # centre sticks forward
    dihed  = 0.018 * abs(t) ** 1.4            # tips rise slightly
    y_le   = np.full(ns, y_center) - chord * 0.55 - sweep
    y_te   = np.full(ns, y_center) + chord * 0.45
    z_le   = np.full(ns, z) + dihed + chord * 0.02
    z_te   = np.full(ns, z) + dihed - chord * 0.01

    parts = [_plate_strip(xs, y_le, y_te, z_le, z_te)]

    # ── flap element: shorter span, more AoA ──
    fl_c = chord * 0.28
    fl_hs = hs * 0.80
    nf = 18
    fxs = np.linspace(-fl_hs, fl_hs, nf)
    ft  = fxs / fl_hs
    fl_y0 = y_center + chord * 0.50          # gap behind main TE
    aoa = np.radians(14)
    f_yle = np.full(nf, fl_y0)
    f_yte = np.full(nf, fl_y0 + fl_c * np.cos(aoa))
    f_zle = np.full(nf, z + chord * 0.06) + 0.008 * abs(ft) ** 1.2
    f_zte = f_zle - fl_c * np.sin(aoa)
    parts.append(_plate_strip(fxs, f_yle, f_yte, f_zle, f_zte))

    # ── endplates: curved leading edge, taller toward rear ──
    ep_h = chord * 0.28
    for sign in (-1.0, 1.0):
        x = sign * hs
        idx = 0 if sign < 0 else ns - 1
        yl = float(y_le[idx]) - chord * 0.06
        yr = float(y_te[idx]) + fl_c + chord * 0.04
        zm = z + float(dihed[idx])
        # 6-point profile: a tapered shape, not a rectangle
        parts.append(_rect([
            [x, yl,               zm - ep_h * 0.35],
            [x, yr,               zm - ep_h * 0.20],
            [x, yr + chord * 0.03, zm + ep_h * 0.65],
            [x, yl + chord * 0.08, zm + ep_h * 0.55],
        ]))

    return _merge(parts)


def build_rear_wing_mesh(y_center, z, span, chord):
    """
    Stylised rear wing: main element, flap, endplates, swan-neck supports.

    y_center : longitudinal centre (m)
    z        : height of main element lower surface (m)
    span     : total wingspan (m)
    chord    : main element chord (m)
    """
    hs = span / 2.0
    ns = 20
    xs = np.linspace(-hs, hs, ns)
    t  = xs / hs

    # ── main element: slight forward bow in centre ──
    bow    = chord * 0.08 * (1.0 - t ** 2)
    y_le   = np.full(ns, y_center) - chord * 0.5 - bow
    y_te   = np.full(ns, y_center) + chord * 0.5
    aoa    = np.radians(10)
    z_le   = np.full(ns, z) + chord * np.sin(aoa)
    z_te   = np.full(ns, z)
    parts  = [_plate_strip(xs, y_le, y_te, z_le, z_te)]

    # ── flap: above main, steeper AoA ──
    fl_c = chord * 0.35
    fl_aoa = np.radians(22)
    gap = chord * 0.04
    fl_yle = y_te + gap
    fl_yte = fl_yle + fl_c * np.cos(fl_aoa)
    fl_zle = z_le + chord * 0.03
    fl_zte = fl_zle - fl_c * np.sin(fl_aoa)
    parts.append(_plate_strip(xs, fl_yle, fl_yte, fl_zle, fl_zte))

    # ── endplates: connect main + flap, swept trailing edge ──
    ep_overhang = chord * 0.12
    for sign in (-1.0, 1.0):
        x = sign * hs
        idx = 0 if sign < 0 else ns - 1
        yl = float(y_le[idx]) - ep_overhang
        yr = float(fl_yte[idx]) + ep_overhang
        zbot = float(z_te[idx]) - chord * 0.15
        ztop = float(fl_zle[idx]) + chord * 0.08
        parts.append(_rect([
            [x, yl, zbot],
            [x, yr, zbot + chord * 0.05],
            [x, yr - chord * 0.06, ztop],
            [x, yl + chord * 0.04, ztop + chord * 0.03],
        ]))

    # ── swan-neck supports: thin pillars from below ──
    pillar_w = 0.008
    for sign in (-1.0, 1.0):
        x = sign * hs * 0.38
        ym = y_center
        parts.append(_rect([
            [x - pillar_w, ym, z - chord * 0.8],
            [x + pillar_w, ym, z - chord * 0.8],
            [x + pillar_w, ym, z],
            [x - pillar_w, ym, z],
        ]))

    return _merge(parts)


def build_diffuser_mesh(y_start, y_end, z_entry, z_exit, width):
    """
    Stylised diffuser: venturi ramp with side fences and internal strake.

    y_start : longitudinal entry position (m)
    y_end   : longitudinal exit position (m)
    z_entry : ground clearance at entry (m)
    z_exit  : exit lip height (m)
    width   : total width (m)
    """
    hw = width / 2.0
    ny = 20
    ys = np.linspace(y_start, y_end, ny)
    t  = np.linspace(0, 1, ny)

    # ── floor ramp: gentle curve from entry to exit (venturi) ──
    # S-curve: stays flat in first 40%, then kicks up
    u = np.clip((t - 0.4) / 0.6, 0.0, 1.0)
    s = u ** 1.6
    zs = z_entry + (z_exit - z_entry) * s

    # Width tapers slightly at entry, flares at exit
    hw_arr = hw * (0.85 + 0.15 * t)

    parts = []

    # ── floor surface (two halves for internal strake) ──
    for side_sign in (-1.0, 1.0):
        xs_inner = np.zeros(ny)                     # centreline
        xs_outer = side_sign * hw_arr
        parts.append(_plate_strip(
            xs_outer, ys, ys, zs, zs))              # outer edge
        # span from centre to outer edge at each Y station
        n_cw = 6
        for ci in range(n_cw):
            frac0 = ci / n_cw
            frac1 = (ci + 1) / n_cw
            x0 = side_sign * hw_arr * frac0
            x1 = side_sign * hw_arr * frac1
            parts.append(_plate_strip(x0, ys, ys, zs, zs))

    # Simpler approach: just a mesh grid for the floor
    parts = []
    n_cw = 10
    x_all = np.linspace(-hw, hw, n_cw)
    for ci in range(n_cw - 1):
        x0_arr = np.full(ny, x_all[ci]) * (0.85 + 0.15 * t)
        x1_arr = np.full(ny, x_all[ci + 1]) * (0.85 + 0.15 * t)
        # Floor strip between x0 and x1
        verts = []
        for j in range(ny):
            verts.append([x0_arr[j], ys[j], zs[j]])
            verts.append([x1_arr[j], ys[j], zs[j]])
        faces = []
        for j in range(ny - 1):
            v0 = j * 2
            faces.append([v0, v0 + 1, v0 + 2])
            faces.append([v0 + 1, v0 + 3, v0 + 2])
        parts.append((np.array(verts, np.float32),
                       np.array(faces, np.uint32)))

    # ── side fences ──
    fence_h = (z_exit - z_entry) * 0.7 + 0.025
    for sign in (-1.0, 1.0):
        xf = sign * hw_arr
        parts.append(_rect([
            [float(xf[0]),  y_start, z_entry],
            [float(xf[-1]), y_end,   z_exit],
            [float(xf[-1]), y_end,   z_exit + fence_h],
            [float(xf[0]),  y_start, z_entry + fence_h * 0.4],
        ]))

    # ── centre strake ──
    parts.append(_rect([
        [0, y_start, z_entry],
        [0, y_end,   z_exit],
        [0, y_end,   z_exit + fence_h * 0.6],
        [0, y_start, z_entry + fence_h * 0.25],
    ]))

    # ── exit lip (kick-up at the back) ──
    lip = z_exit * 0.15
    parts.append(_rect([
        [-hw_arr[-1], y_end, z_exit],
        [ hw_arr[-1], y_end, z_exit],
        [ hw_arr[-1] * 0.95, y_end + 0.015, z_exit + lip],
        [-hw_arr[-1] * 0.95, y_end + 0.015, z_exit + lip],
    ]))

    return _merge(parts)


def build_tire_mesh(center, spin_axis, outer_r, rim_r, half_w, n=48):
    """Tyre = tread cylinder + two sidewall annuli. NO rim — rim is interior."""
    c = np.asarray(center, float)
    s = _norm(np.asarray(spin_axis, float))
    parts = [_torus_section_mesh(c, s, outer_r, half_w, n)]
    for sign in (-1, 1):
        cap = c + sign * half_w * s
        parts.append(_annulus_mesh(cap, s, rim_r, outer_r, n))
    return _merge(parts)


# ── link topology ─────────────────────────────────────────────────────────────
# (key_a, key_b, RGBA)
LINKS = [
    ('uca_front',        'uca_outer',         (0.20, 0.55, 0.90, 1.0)),
    ('uca_rear',         'uca_outer',         (0.20, 0.55, 0.90, 1.0)),
    ('lca_front',        'lca_outer',         (0.90, 0.25, 0.25, 1.0)),
    ('lca_rear',         'lca_outer',         (0.90, 0.25, 0.25, 1.0)),
    ('tie_rod_inner',    'tie_rod_outer',     (0.25, 0.80, 0.40, 1.0)),
    ('pushrod_outer',    'pushrod_inner',     (0.95, 0.55, 0.10, 1.0)),
    ('rocker_spring_pt', 'spring_chassis_pt', (0.50, 0.50, 0.50, 1.0)),
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
        self._cam  = scene.TurntableCamera(
            fov=40, elevation=20, azimuth=-60,
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
        self._markers = scene.Markers(parent=self._view.scene)
        self._markers.set_data(
            pos=np.zeros((1, 3), np.float32),
            face_color=np.ones((1, 4), np.float32),
            size=9, edge_width=0,
        )

        # Per-corner meshes (4 corners max)
        self._tire_meshes    = [self._new_mesh((0.18, 0.18, 0.18, 0.6)) for _ in range(4)]
        self._upright_meshes = [self._new_mesh((0.50, 0.40, 0.30, 0.30)) for _ in range(4)]
        self._rocker_meshes  = [self._new_mesh((0.55, 0.20, 0.70, 0.75)) for _ in range(4)]

        self._ground = self._make_ground_grid()
        self._ground.visible = True

        # ── Aero package meshes (translucent, hidden until configured) ───
        self._fw_mesh   = self._new_mesh((1.0, 0.84, 0.0, 0.28))   # yellow
        self._rw_mesh   = self._new_mesh((0.26, 0.65, 0.96, 0.28)) # blue
        self._diff_mesh = self._new_mesh((0.90, 0.22, 0.21, 0.22)) # red
        self._fw_mesh.visible   = False
        self._rw_mesh.visible   = False
        self._diff_mesh.visible = False

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

        # ── camera state ──────────────────────────────────────────────────
        self._mouse_last = None
        self._mouse_btn  = None

        # ── hook Qt events directly on the native widget ──────────────────
        n = self._canvas.native
        n.setMouseTracking(True)
        n.mousePressEvent   = self._qt_press
        n.mouseMoveEvent    = self._qt_move
        n.mouseReleaseEvent = self._qt_release
        n.wheelEvent        = self._qt_wheel

        # ── NavCube overlay ───────────────────────────────────────────────
        self._navcube = NavCube(parent=n, size=110)
        self._navcube.view_requested.connect(self._snap_camera)
        self._navcube.set_orientation(self._cam.azimuth, self._cam.elevation)
        # position in top-right; will be repositioned on resize
        self._orig_resize = n.resizeEvent
        n.resizeEvent = self._qt_resize

        # ── picking state ─────────────────────────────────────────────────
        self._hp_snap:    list[tuple] = []   # [(name, pos3d), ...]
        self._selected:   str | None  = None
        self._on_pick_cb  = None

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

    def update_aero(self, params: dict):
        """
        Update aero-package 3D overlays.

        params dict keys (all lengths in metres):
            'fw_visible', 'fw_y', 'fw_z', 'fw_span', 'fw_chord',
            'rw_visible', 'rw_y', 'rw_z', 'rw_span', 'rw_chord',
            'diff_visible', 'diff_y_start', 'diff_y_end',
            'diff_z_entry', 'diff_z_exit', 'diff_width',
        """
        # ── front wing ──
        if params.get('fw_visible', False):
            v, f = build_front_wing_mesh(
                params['fw_y'], params['fw_z'],
                params['fw_span'], params['fw_chord'])
            self._fw_mesh.set_data(vertices=v, faces=f)
            self._fw_mesh.visible = True
        else:
            self._fw_mesh.visible = False

        # ── rear wing ──
        if params.get('rw_visible', False):
            v, f = build_rear_wing_mesh(
                params['rw_y'], params['rw_z'],
                params['rw_span'], params['rw_chord'])
            self._rw_mesh.set_data(vertices=v, faces=f)
            self._rw_mesh.visible = True
        else:
            self._rw_mesh.visible = False

        # ── diffuser ──
        if params.get('diff_visible', False):
            v, f = build_diffuser_mesh(
                params['diff_y_start'], params['diff_y_end'],
                params['diff_z_entry'], params['diff_z_exit'],
                params['diff_width'])
            self._diff_mesh.set_data(vertices=v, faces=f)
            self._diff_mesh.visible = True
        else:
            self._diff_mesh.visible = False

        self._canvas.update()

    def update_scene(self, corners: list[dict], arb_segs=None):
        """
        corners: list of dicts with keys 'pts', 'spin_axis', 'label'
        arb_segs: list of (p0, p1) world-space segments for ARB
        """
        link_pos = []
        link_col = []
        mk_pos   = []
        mk_col   = []
        self._hp_snap = []

        for ci, corner in enumerate(corners):
            pts  = corner['pts']
            spin = corner['spin_axis']
            wc   = pts['wheel_center']

            # links
            for pa, pb, col in LINKS:
                if pa in pts and pb in pts:
                    link_pos += [pts[pa], pts[pb]]
                    link_col += [col, col]

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
            # order by angle so the fan covers the whole rocker plate
            rk4 = ('rocker_pivot', 'arb_drop_top', 'pushrod_inner', 'rocker_spring_pt')
            rk3 = ('rocker_pivot', 'pushrod_inner', 'rocker_spring_pt')
            if all(k in pts for k in rk4):
                rv = np.array([pts[k] for k in rk4], np.float32)
                # fan from pivot: tri0=(0,1,2), tri1=(0,2,3)
                self._rocker_meshes[ci].set_data(
                    vertices=rv, faces=np.array([[0,1,2],[0,2,3]], np.uint32))
                for pa2, pb2 in [(rk4[0],rk4[1]),(rk4[1],rk4[2]),
                                 (rk4[2],rk4[3]),(rk4[3],rk4[0])]:
                    link_pos += [pts[pa2], pts[pb2]]
                    link_col += [(0.65, 0.25, 0.80, 1.0)] * 2
            elif all(k in pts for k in rk3):
                rv = np.array([pts[k] for k in rk3], np.float32)
                self._rocker_meshes[ci].set_data(
                    vertices=rv, faces=np.array([[0,1,2]], np.uint32))
                for pa2, pb2 in [(rk3[0],rk3[1]),(rk3[1],rk3[2]),(rk3[2],rk3[0])]:
                    link_pos += [pts[pa2], pts[pb2]]
                    link_col += [(0.65, 0.25, 0.80, 1.0)] * 2

            # tire (tread + sidewalls only, translucent)
            tv, tf = build_tire_mesh(
                wc, spin,
                self._tire_outer_r, self._tire_rim_r, self._tire_half_w)
            self._tire_meshes[ci].set_data(vertices=tv, faces=tf)

            # markers
            for name in HP_NAMES:
                if name not in pts:
                    continue
                p = pts[name]
                mk_pos.append(p)
                if name == self._selected:
                    c = _C_SEL
                elif name in CHASSIS_PTS:
                    c = _C_CHASSIS
                else:
                    c = _C_MOVING
                mk_col.append(c)
                self._hp_snap.append((name, p.copy(), corner['label']))

        # upload lines
        if link_pos:
            self._links_vis.set_data(
                pos=np.array(link_pos, np.float32),
                color=np.array(link_col, np.float32))

        # upload markers
        if mk_pos:
            self._markers.set_data(
                pos=np.array(mk_pos, np.float32),
                face_color=np.array(mk_col, np.float32),
                size=9, edge_width=0)

        # ARB
        if arb_segs:
            ap = np.array([p for seg in arb_segs for p in seg], np.float32)
            self._arb_vis.set_data(pos=ap, color=(0.90, 0.80, 0.10, 1.0))
        else:
            self._arb_vis.set_data(pos=np.zeros((2,3), np.float32))

        self._canvas.update()

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

    def _qt_move(self, event):
        try:
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

    def _qt_resize(self, event):
        """Reposition NavCube to top-right corner on resize."""
        if self._orig_resize:
            self._orig_resize(event)
        w = event.size().width()
        self._navcube.move(w - self._navcube.width() - 4, 4)

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
            tr = self._view.get_transform('scene', 'canvas')
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
                self._on_pick_cb(best_name, best_corner)
        except Exception:
            pass
