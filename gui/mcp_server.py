"""MCP server embedded in Vahan — operate the RUNNING app from Claude directly.

Starts a streamable-HTTP MCP server on 127.0.0.1:8765 in a daemon thread when
the MainWindow constructs (set VAHAN_MCP=0 to disable; VAHAN_MCP_PORT to move).
Every tool call is marshalled onto the Qt GUI thread through a queue drained by
a QTimer — the solvers and view are not thread-safe, so tools run exactly like
user actions do.  Register once per machine:

    claude mcp add --transport http vahan http://127.0.0.1:8765/mcp

Tools: status, load_config, get_hardpoints, set_hardpoint, axle_metrics,
relocate (curve-preserving single-point search), hoopline (damper-mount-on-
hoop-line solutions), apply_hoopline, save_experiment, screenshot.
ONE MODEL: every tool reads/drives the live MainWindow; no physics here.
"""
from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
import traceback

_CALLS: "queue.Queue" = queue.Queue()
_WIN = None


def _gui(fn, timeout=900.0):
    """Run fn() on the GUI thread; block the MCP thread until done."""
    fut = concurrent.futures.Future()
    _CALLS.put((fn, fut))
    return fut.result(timeout=timeout)


def _drain():
    while True:
        try:
            fn, fut = _CALLS.get_nowait()
        except queue.Empty:
            return
        try:
            fut.set_result(fn())
        except Exception as e:                       # noqa: BLE001
            fut.set_exception(e)



def present_relocate_ghosts_impl(win, axle: str, point_label: str,
                                 solutions: list, target_mm=None,
                                 applied_idx=None, title_note: str = ''):
    """Ghost relocation solutions into the 3D scene WITHOUT touching the
    user's page, camera or focus.  Pinned results (every solution at the same
    point) draw the chain ONCE in white and colour only what differs — the
    ARB re-hang options.  Returns the text summary (caller decides where to
    show it: Packaging-page status line, MCP reply)."""
    import numpy as np
    from gui.view3d import build_sphere, build_cylinder_between
    COLS = [(0.95, 0.75, 0.15, 0.55),   # amber
            (0.35, 0.55, 0.95, 0.55),   # blue
            (0.20, 0.85, 0.75, 0.55),   # teal
            (0.85, 0.45, 0.90, 0.55),   # magenta
            (0.55, 0.90, 0.25, 0.55),   # green
            (0.92, 0.60, 0.30, 0.55)]   # orange
    CNAMES = ['AMBER', 'BLUE', 'TEAL', 'MAGENTA', 'GREEN', 'ORANGE']
    RED = (0.95, 0.15, 0.15, 0.85)
    WHITE = (0.92, 0.92, 0.92, 0.40)
    win._imported_parts = [q for q in win._imported_parts
                           if not str(q.get('name', '')).startswith('SOLUTION')]

    def add_part(nm, v, fc, col):
        win._imported_parts.append({
            'name': nm, 'source': 'marker', 'offset': [0, 0, 0],
            'flip': False, 'verts': np.asarray(v, float) * 1000.0,
            'faces': np.asarray(fc, int), '_col': col,
            'info': {'path': nm, 'n_faces': len(fc), 'size_mm': [0, 0, 0]}})

    sols = [x for x in solutions if x.get('bundle')]
    pts = [np.asarray(x['bundle']['hp']['spring_chassis_pt'], float)
           for x in sols if 'spring_chassis_pt' in x['bundle']['hp']]
    pinned = (len(pts) == len(sols) and len(pts) >= 1
              and float(np.ptp(np.asarray(pts), axis=0).max()) < 0.002
              and (target_mm is None or float(np.linalg.norm(
                  np.asarray(pts[0]) - np.asarray(target_mm, float)
                  / 1000.0)) < 0.003))

    if target_mm is not None:
        T = np.asarray(target_mm, float) / 1000.0
        for side in (1.0, -1.0):
            sv, sf = build_sphere(T * np.array([side, 1.0, 1.0]), 0.009,
                                  n_lat=10, n_lon=14)
            add_part('SOLUTION target (the pinned point)', sv, sf, RED)

    lines = []
    for i, sol in enumerate(sols):
        bh = {k: np.asarray(v, float) for k, v in sol['bundle']['hp'].items()}
        ba = {k: np.asarray(v, float) for k, v in sol['bundle']['arb'].items()}
        is_applied = (applied_idx is not None and i == applied_idx)
        col = RED if is_applied else COLS[i % len(COLS)]
        nm = 'SOLUTION #%d%s' % (i + 1, ' APPLIED' if is_applied else '')
        pv = ba['arb_pivot']
        bar_b = pv.copy(); bar_b[0] = -pv[0]
        chain = [(bh['rocker_spring_pt'], bh['spring_chassis_pt'], 0.014),
                 (bh['rocker_pivot'], bh['rocker_spring_pt'], 0.004),
                 (bh['rocker_pivot'], bh['pushrod_inner'], 0.004)]
        arbm = [(ba['arb_drop_top'], ba['arb_arm_end'], 0.0035),
                (ba['arb_arm_end'], ba['arb_pivot'], 0.0045),
                (pv, bar_b, 0.008)]
        draw_chain = (not pinned) or i == 0
        chain_col = WHITE if pinned else col
        for side in (1.0, -1.0):
            M = np.array([side, 1.0, 1.0])
            if draw_chain:
                for a, b, rad in chain:
                    va, fb = build_cylinder_between(
                        np.asarray(a, float) * M, np.asarray(b, float) * M,
                        rad, n=10)
                    add_part(nm, va, fb, chain_col)
            for a, b, rad in arbm:
                va, fb = build_cylinder_between(
                    np.asarray(a, float) * M, np.asarray(b, float) * M,
                    rad, n=10)
                add_part(nm, va, fb, col)
        ri = sol.get('resolved', {})
        pos = sol.get('pos_mm', (bh['spring_chassis_pt'] * 1000).round(1))
        if pinned:
            lines.append('option %d %s: ARB re-hung %g deg around the rocker '
                         '(branch %d), %.0f mm clear of the coilover, worst '
                         'parameter at %.0f%% of tolerance'
                         % (i + 1,
                            'RED (applied)' if is_applied else CNAMES[i % 6],
                            ri.get('arb_pose_deg', 0),
                            ri.get('arb_refit_branch', 0),
                            ri.get('arb_drop_clearance_mm', 0),
                            sol.get('worst_tol_frac', 0.0) * 100))
        else:
            lines.append('#%d %s  (%.0f, %.0f, %.0f) mm | %.0f mm from '
                         'target | worst parameter %.0f%% of tolerance'
                         % (i + 1,
                            'RED (applied)' if is_applied else CNAMES[i % 6],
                            pos[0], pos[1], pos[2],
                            sol.get('dist_from_target_mm', float('nan')),
                            sol.get('worst_tol_frac', 0.0) * 100))

    payload = []
    for i, q in enumerate(win._imported_parts):
        col = q.get('_col') or win._IMPORTED_PART_COLOURS[
            i % len(win._IMPORTED_PART_COLOURS)]
        payload.append((q['verts'], q['faces'], col))
    win.view3d.set_imported_parts(payload)
    win.view3d._canvas.update()
    head = ('POINT PINNED at (%.1f, %.1f, %.1f) mm — RED sphere. '
            'White ghost = the one resolved chain; each colour = one ARB '
            'option.' % tuple(np.asarray(target_mm, float))
            if (pinned and target_mm is not None) else
            'RED sphere = requested target; each colour = one full solution.')
    summary = head + '\n' + '\n'.join(lines) + \
        (('\n' + title_note) if title_note else '')
    return summary



def start_mcp(win, port: int = None):
    """Attach the MCP server to a MainWindow. Silent no-op on any failure —
    the app must never die because the connector could not start."""
    global _WIN
    if os.environ.get('VAHAN_MCP', '1') == '0':
        return None
    try:
        from mcp.server.fastmcp import FastMCP
        from PyQt6.QtCore import QTimer
    except Exception:
        return None
    _WIN = win
    port = int(port or os.environ.get('VAHAN_MCP_PORT', 8765))

    timer = QTimer(win)
    timer.timeout.connect(_drain)
    timer.start(50)
    win._mcp_timer = timer                            # keep alive

    mcp = FastMCP('vahan')
    try:
        mcp.settings.host = '127.0.0.1'
        mcp.settings.port = port
    except Exception:
        pass

    # ── tools ────────────────────────────────────────────────────────────────
    @mcp.tool()
    def status() -> dict:
        """App status: loaded point counts, window title, MCP health."""
        def f():
            return {'title': _WIN.windowTitle(),
                    'front_points': len(_WIN._front_hp),
                    'rear_points': len(_WIN._rear_hp),
                    'front_arb': len(_WIN._front_arb or {}),
                    'rear_arb': len(_WIN._rear_arb or {})}
        return _gui(f)

    @mcp.tool()
    def load_config(path: str) -> str:
        """Load a .vahan project into the live app (path relative to repo or
        absolute)."""
        def f():
            _WIN._load_project_from_path(path)
            _WIN._update_3d()
            return f'loaded {path}'
        return _gui(f)

    @mcp.tool()
    def get_hardpoints(axle: str) -> dict:
        """All hardpoints + ARB points of 'front' or 'rear', mm (left side)."""
        def f():
            from vahan import packaging as PK
            b = PK.get_bundle(_WIN, axle)
            return {'hp': {k: [round(float(x) * 1000, 3) for x in v]
                           for k, v in b['hp'].items()},
                    'arb': {k: [round(float(x) * 1000, 3) for x in v]
                            for k, v in b['arb'].items()}}
        return _gui(f)

    @mcp.tool()
    def set_hardpoint(axle: str, dict_name: str, key: str,
                      x_mm: float, y_mm: float, z_mm: float) -> str:
        """Move one point (hp|arb) of one axle to (x,y,z) mm; solvers + 3D
        view refresh."""
        def f():
            import numpy as np
            d = (_WIN._front_hp if axle == 'front' else _WIN._rear_hp) \
                if dict_name == 'hp' else \
                (_WIN._front_arb if axle == 'front' else _WIN._rear_arb)
            d[key] = np.array([x_mm, y_mm, z_mm]) / 1000.0
            _WIN._rebuild_solvers()
            _WIN._update_3d()
            return f'{axle} {dict_name}.{key} -> ({x_mm}, {y_mm}, {z_mm}) mm'
        return _gui(f)

    @mcp.tool()
    def axle_metrics(axle: str) -> dict:
        """Wheel kinematics (statics + camber/toe curves + bump steer) and
        MR/ARB rates for one axle, from the live solvers."""
        def f():
            from vahan import packaging as PK
            m = PK._axle_wheel_metrics(_WIN, axle)
            m['rates'] = PK._rates(_WIN)
            return m
        return _gui(f)

    @mcp.tool()
    def relocate(axle: str, dict_name: str, key: str, radius_mm: float = 120,
                 samples: int = 150, similarity_pct: float = 20,
                 n_show: int = 10) -> dict:
        """Curve-preserving relocation search for one hardpoint (the Relocate
        IK): diverse feasible alternative positions, clash-checked."""
        def f():
            from vahan.relocate import relocate_search
            return relocate_search(_WIN, axle, dict_name, key,
                                   radius_mm=radius_mm, n_samples=int(samples),
                                   min_sep_frac=similarity_pct / 100.0,
                                   n_show=int(n_show))
        return _gui(f)

    @mcp.tool()
    def hoopline(n_show: int = 8, similarity_pct: float = 20,
                 z_lo_mm: float = 380, z_hi_mm: float = 700,
                 steps: int = 22) -> dict:
        """Front damper chassis mount constrained ONTO the front-hoop line
        (aft inboard pickups extended): diverse feasible mount heights, each
        with MR/ARB re-tuned to baseline and the full oracle passed."""
        def f():
            from vahan.relocate import hoopline_search
            r = hoopline_search(_WIN, n_show=int(n_show),
                                sim_frac=similarity_pct / 100.0,
                                z_lo_mm=z_lo_mm, z_hi_mm=z_hi_mm,
                                n_steps=int(steps))
            r.pop('all_candidates', None)
            return r
        return _gui(f)

    @mcp.tool()
    def apply_hoopline(z_mm: float) -> dict:
        """APPLY the hoop-line construction at one mount height to the live
        model (rotate chain, mount on line, re-tune MR + ARB). Save or revert
        afterwards as needed."""
        def f():
            from vahan.relocate import hoopline_apply
            return hoopline_apply(_WIN, z_mm)
        return _gui(f)

    @mcp.tool()
    def save_experiment(name: str) -> str:
        """Save the live model to configs/experiments/<name>.vahan (never
        configs/ root)."""
        def f():
            os.makedirs('configs/experiments', exist_ok=True)
            p = os.path.join('configs', 'experiments', name + '.vahan')
            _WIN._save_project_to_path(p)
            return 'saved ' + p
        return _gui(f)

    @mcp.tool()
    def screenshot(path: str = '', azimuth: float = None,
                   elevation: float = None) -> str:
        """Render the live 3D view to a PNG (optionally set camera first)."""
        def f():
            from PIL import Image
            cam = _WIN.view3d._cam
            if azimuth is not None:
                cam.azimuth = azimuth
            if elevation is not None:
                cam.elevation = elevation
            _WIN.view3d._canvas.update()
            img = _WIN.view3d._canvas.render()
            p = path or os.path.join(os.getcwd(), 'mcp_view.png')
            Image.fromarray(img[..., :3]).save(p)
            return p
        return _gui(f)

    @mcp.tool()
    def present_hoopline(zs_mm: list, applied_z_mm: float = None) -> str:
        """PRESENT hoop-line solutions unmissably in the 3D view: draws the
        hoop line (white tube) + one sphere per solution mount (amber; the
        applied one red), frames the camera on them, raises the window and
        pops a non-modal summary box."""
        def f():
            import numpy as np
            from PyQt6.QtWidgets import QMessageBox
            from gui.view3d import build_sphere, build_cylinder_between
            hp = _WIN._front_hp
            A = np.asarray(hp['lca_rear'], float)
            B = np.asarray(hp['uca_rear'], float)
            d = (B - A) / np.linalg.norm(B - A)
            # drop stale solution markers, keep real imported parts
            _WIN._imported_parts = [p for p in _WIN._imported_parts
                                    if not str(p.get('name', ''))
                                    .startswith('SOLUTION')]
            top = A + d * ((0.72 - A[2]) / d[2])
            cv, cf = build_cylinder_between(A, top, 0.006, n=12)
            _WIN._imported_parts.append({
                'name': 'SOLUTION hoop line', 'source': 'marker',
                'offset': [0, 0, 0], 'flip': False,
                'verts': np.asarray(cv, float) * 1000.0,
                'faces': np.asarray(cf, int),
                'info': {'path': 'SOLUTION hoop line', 'n_faces': len(cf),
                         'size_mm': [0, 0, 0]}})
            lines = []
            for z in zs_mm:
                z_m = float(z) / 1000.0
                T = A + d * ((z_m - A[2]) / d[2])
                is_applied = (applied_z_mm is not None
                              and abs(float(z) - float(applied_z_mm)) < 2.0)
                r = 0.018 if is_applied else 0.014
                sv, sf = build_sphere(T, r, n_lat=10, n_lon=14)
                nm = ('SOLUTION APPLIED z=%d' if is_applied
                      else 'SOLUTION z=%d') % round(float(z))
                _WIN._imported_parts.append({
                    'name': nm, 'source': 'marker', 'offset': [0, 0, 0],
                    'flip': False,
                    'verts': np.asarray(sv, float) * 1000.0,
                    'faces': np.asarray(sf, int),
                    'info': {'path': nm, 'n_faces': len(sf),
                             'size_mm': [0, 0, 0]}})
                # mirror side too — the eye expects symmetry
                Tm = T * np.array([-1.0, 1.0, 1.0])
                sv2, sf2 = build_sphere(Tm, r, n_lat=10, n_lon=14)
                _WIN._imported_parts.append({
                    'name': nm + ' (right)', 'source': 'marker',
                    'offset': [0, 0, 0], 'flip': False,
                    'verts': np.asarray(sv2, float) * 1000.0,
                    'faces': np.asarray(sf2, int),
                    'info': {'path': nm, 'n_faces': len(sf2),
                             'size_mm': [0, 0, 0]}})
                lines.append('z = %d mm  ->  mount (%.0f, %.0f, %.0f) mm%s'
                             % (round(float(z)), T[0] * 1000, T[1] * 1000,
                                T[2] * 1000,
                                '   <= APPLIED NOW (red)' if is_applied else ''))
            # colour override: applied = red, others = amber, line = white
            payload = []
            for i, p in enumerate(_WIN._imported_parts):
                nm = str(p.get('name', ''))
                if nm.startswith('SOLUTION APPLIED'):
                    col = (0.95, 0.15, 0.15, 0.98)
                elif nm.startswith('SOLUTION hoop line'):
                    col = (0.95, 0.95, 0.95, 0.85)
                elif nm.startswith('SOLUTION'):
                    col = (0.95, 0.75, 0.15, 0.95)
                else:
                    col = _WIN._IMPORTED_PART_COLOURS[
                        i % len(_WIN._IMPORTED_PART_COLOURS)]
                payload.append((p['verts'], p['faces'], col))
            _WIN.view3d.set_imported_parts(payload)
            _WIN._switch_page(0)
            cam = _WIN.view3d._cam
            mid = A + d * ((0.60 - A[2]) / d[2])
            cam.center = tuple(mid)
            cam.azimuth, cam.elevation, cam.distance = -135, 12, 0.9
            _WIN.view3d._canvas.update()
            _WIN.showNormal(); _WIN.raise_(); _WIN.activateWindow()
            box = QMessageBox(_WIN)
            box.setWindowTitle('HOOP-LINE SOLUTIONS — look at the 3D view')
            box.setText(
                'Damper chassis mount ON the roll-hoop line.\n'
                'AMBER spheres = feasible mounts, RED = applied now, '
                'WHITE tube = the hoop line.\n\n' + '\n'.join(lines) +
                '\n\nEvery solution: MR and ARB re-tuned to baseline, all '
                'curves + clash sweep passed.\nClean configs: '
                'configs/experiments/hoopline_zNNN.vahan')
            box.setModal(False)
            box.show()
            _WIN._mcp_solution_box = box       # keep alive
            return 'presented %d solutions in the 3D view' % len(zs_mm)
        return _gui(f)

    @mcp.tool()
    def present_hoopline_ghosts(zs_mm: list, applied_z_mm: float = None) -> str:
        """PACKAGING view: draw each hoop-line solution's FULL actuation
        hardware (coilover, rocker levers, pushrod tip, ARB drop link + blade)
        as a translucent GHOST assembly in its own colour, both sides, plus
        the hoop line — so the eye judges packaging directly.  The applied
        solution stays the solid model; ghosts overlay the alternatives."""
        def f():
            import numpy as np
            from PyQt6.QtWidgets import QMessageBox
            from gui.view3d import build_sphere, build_cylinder_between
            from vahan.relocate import hoopline_bundle
            # solution colours (in-app palette; blue allowed in-app)
            COLS = [(0.95, 0.75, 0.15, 0.45),   # amber
                    (0.35, 0.55, 0.95, 0.45),   # blue
                    (0.92, 0.92, 0.92, 0.40),   # white
                    (0.20, 0.85, 0.75, 0.45)]   # teal
            RED = (0.95, 0.15, 0.15, 0.75)
            hp = _WIN._front_hp
            A = np.asarray(hp['lca_rear'], float)
            B = np.asarray(hp['uca_rear'], float)
            dline = (B - A) / np.linalg.norm(B - A)
            _WIN._imported_parts = [p for p in _WIN._imported_parts
                                    if not str(p.get('name', ''))
                                    .startswith('SOLUTION')]

            def add_part(nm, v, fc, col):
                _WIN._imported_parts.append({
                    'name': nm, 'source': 'marker', 'offset': [0, 0, 0],
                    'flip': False, 'verts': np.asarray(v, float) * 1000.0,
                    'faces': np.asarray(fc, int), '_col': col,
                    'info': {'path': nm, 'n_faces': len(fc),
                             'size_mm': [0, 0, 0]}})

            top = A + dline * ((0.72 - A[2]) / dline[2])
            cv, cf = build_cylinder_between(A, top, 0.005, n=10)
            add_part('SOLUTION hoop line', cv, cf, (0.95, 0.95, 0.95, 0.85))

            lines = []
            for i, z in enumerate(zs_mm):
                r = hoopline_bundle(_WIN, float(z))
                if not r.get('ok'):
                    lines.append('z=%d mm: INFEASIBLE (%s)'
                                 % (round(float(z)), r.get('fails')))
                    continue
                bh = r['bundle']['hp']; ba = r['bundle']['arb']
                is_applied = (applied_z_mm is not None
                              and abs(float(z) - float(applied_z_mm)) < 2.0)
                col = RED if is_applied else COLS[i % len(COLS)]
                nm = 'SOLUTION z=%d%s' % (round(float(z)),
                                          ' APPLIED' if is_applied else '')
                members = [
                    (bh['rocker_spring_pt'], bh['spring_chassis_pt'], 0.016),
                    (bh['rocker_pivot'], bh['rocker_spring_pt'], 0.004),
                    (bh['rocker_pivot'], bh['pushrod_inner'], 0.004),
                    (ba['arb_drop_top'], ba['arb_arm_end'], 0.0035),
                    (ba['arb_arm_end'], ba['arb_pivot'], 0.0045),
                ]
                for side in (1.0, -1.0):
                    M = np.array([side, 1.0, 1.0])
                    for a, b, rad in members:
                        va, fb = build_cylinder_between(
                            np.asarray(a, float) * M,
                            np.asarray(b, float) * M, rad, n=10)
                        add_part(nm, va, fb, col)
                    sv, sf = build_sphere(
                        np.asarray(bh['spring_chassis_pt'], float) * M,
                        0.012, n_lat=8, n_lon=10)
                    add_part(nm, sv, sf, col)
                cname = ('RED (applied)' if is_applied else
                         ['AMBER', 'BLUE', 'WHITE', 'TEAL'][i % 4])
                lines.append('z=%d mm  ->  %s ghost, damper %.0f mm'
                             % (round(float(z)), cname,
                                r['damper_static_mm']))

            payload = []
            for i, p in enumerate(_WIN._imported_parts):
                col = p.get('_col') or _WIN._IMPORTED_PART_COLOURS[
                    i % len(_WIN._IMPORTED_PART_COLOURS)]
                payload.append((p['verts'], p['faces'], col))
            _WIN.view3d.set_imported_parts(payload)
            _WIN._switch_page(0)
            cam = _WIN.view3d._cam
            mid = A + dline * ((0.58 - A[2]) / dline[2])
            cam.center = tuple(mid)
            cam.azimuth, cam.elevation, cam.distance = -140, 14, 1.15
            _WIN.view3d._canvas.update()
            _WIN.showNormal(); _WIN.raise_(); _WIN.activateWindow()
            box = QMessageBox(_WIN)
            box.setWindowTitle('PACKAGING — hoop-line solution ghosts')
            box.setText('Each colour = ONE complete solution assembly '
                        '(coilover + rocker levers + ARB link/blade), both '
                        'sides, overlaid for packaging judgement:\n\n'
                        + '\n'.join(lines) +
                        '\n\nOrbit/zoom freely — ghosts are part of the '
                        'scene. Load configs/experiments/hoopline_zNNN.vahan '
                        'to make any of them the live model.')
            box.setModal(False)
            box.show()
            _WIN._mcp_solution_box = box
            return 'ghosted %d solutions' % len(zs_mm)
        return _gui(f)

    @mcp.tool()
    def present_relocate_ghosts(axle: str, point_label: str,
                                result_json_path: str,
                                applied_idx: int = None) -> str:
        """PACKAGING view for ANY relocation/generative search result: load the
        saved result JSON (from vahan.relocate generative_search/relocate_
        search) and ghost every solution assembly in the 3D view, with a RED
        sphere at the requested target."""
        def f():
            import json as _json
            r = _json.load(open(result_json_path))
            return present_relocate_ghosts_impl(
                _WIN, axle, point_label, r.get('solutions', []),
                target_mm=r.get('target_mm'), applied_idx=applied_idx)
        return _gui(f)

    @mcp.tool()
    def set_note(text: str) -> str:
        """Write a note into the Packaging page's Relocate tab status line —
        visible to the user in the open app."""
        def f():
            try:
                _WIN._switch_page(6)
                _WIN._packaging_page._rl_status.setText(text)
                return 'note set (Packaging page, Relocate tab)'
            except Exception as e:
                return f'could not set note: {e}'
        return _gui(f)

    def _serve():
        try:
            mcp.run(transport='streamable-http')
        except Exception:
            traceback.print_exc()

    th = threading.Thread(target=_serve, daemon=True, name='vahan-mcp')
    th.start()
    win._mcp_thread = th
    try:
        win.statusBar().showMessage(
            f'MCP connector live on 127.0.0.1:{port} (claude mcp add '
            f'--transport http vahan http://127.0.0.1:{port}/mcp)', 8000)
    except Exception:
        pass
    return mcp
