"""Render the live 3D view of a config to PNGs so the geometry can be LOOKED at.

    py capture_views.py <config.vahan> [out_dir] [tag]

Renders the same scene the app draws (no second model): full car isometric,
front corner, rear corner, and the rear corner in interference mode. Must run
with QT_QPA_PLATFORM unset — the offscreen platform has no OpenGL context.
"""
import os
import sys

os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ['VAHAN_MCP'] = '0'
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
from gui.main_window import MainWindow


def shoot(win, path, center_mm, distance_mm, az, el, mode='normal'):
    v = win.view3d
    v.set_view_mode(mode)
    cam = v._cam
    cam.center = tuple(np.asarray(center_mm, float) / 1000.0)
    try:
        cam.scale_factor = distance_mm / 1000.0
    except Exception:
        cam.distance = distance_mm / 1000.0
    cam.azimuth, cam.elevation = az, el
    v._canvas.update()
    app.processEvents()
    img = v._canvas.render()
    Image.fromarray(img[..., :3]).save(path)
    print('wrote', path)


def main():
    cfg = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'figs')
    tag = sys.argv[3] if len(sys.argv) > 3 else 'current'
    os.makedirs(out, exist_ok=True)

    win = MainWindow()
    win.resize(1500, 1000)
    win.show()
    win._load_project_from_path(cfg)
    win._rebuild_solvers(0.)
    app.processEvents()

    fhp, rhp = win._front_hp, win._rear_hp
    fc = np.asarray(fhp['wheel_center'], float) * 1000.0
    rc = np.asarray(rhp['wheel_center'], float) * 1000.0

    shoot(win, os.path.join(out, f'gui_{tag}_car.png'),
          [0, (fc[1] + rc[1]) / 2, 250], 2600, -60, 20)
    shoot(win, os.path.join(out, f'gui_{tag}_rear_iso.png'), rc, 900, -55, 18)
    shoot(win, os.path.join(out, f'gui_{tag}_rear_front_view.png'), rc, 800, 0, 5)
    shoot(win, os.path.join(out, f'gui_{tag}_rear_interference.png'), rc, 900, -55, 18,
          mode='interference')
    shoot(win, os.path.join(out, f'gui_{tag}_front_iso.png'), fc, 900, -125, 18)
    return 0


if __name__ == '__main__':
    sys.exit(main())
