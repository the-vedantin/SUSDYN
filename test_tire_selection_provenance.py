"""Focused regression for project-selected TTC data provenance.

Run from the repository root:
  python test_tire_selection_provenance.py

It requires the local, gitignored TTC files and makes only temporary project
copies.  It deliberately does not run the full geometric regression net.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VAHAN_MCP', '0')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


ROOT = Path(__file__).resolve().parent
V117 = ROOT / 'configs' / '2027_v117_(restore_v114_packaging).vahan'


def _latest_run6_config() -> Path:
    candidates = list((ROOT / 'configs').glob('2027_v*_*.vahan'))
    candidates = [p for p in candidates if 'run6' in p.name.lower()]
    if not candidates:
        raise RuntimeError('no run6 project config found')
    return max(candidates, key=lambda p: int(re.search(r'_v(\d+)', p.name).group(1)))


def _state(window: MainWindow) -> dict:
    tire = window._tire_model
    return {
        'file': window._car.get('tire_file'),
        'window': window._car.get('tire_speed_window_kph'),
        'warmup': window._car.get('tire_warmup_samples'),
        'model_window': getattr(tire, 'speed_window_kph', None),
        'pressure_psi': getattr(tire, 'pressure_psi', None),
        'samples': getattr(tire, 'samples_selected', None),
    }


def _expect_blocked(window: MainWindow, fragment: str) -> None:
    assert window._tire_model is None
    assert fragment in str(getattr(window, '_tire_selection_error', ''))
    try:
        window._build_dynamics_solver()
    except RuntimeError as exc:
        assert 'Tire selection blocked' in str(exc)
    else:
        raise AssertionError('dynamics solver built despite refused tyre selection')


def main() -> None:
    if not V117.exists() or not (ROOT / 'tire_data' / 'B2356run6.mat').exists():
        print('SKIP: local v117 config or TTC run6 file unavailable')
        return

    app = QApplication.instance() or QApplication([])
    run6 = _latest_run6_config()
    windows: list[MainWindow] = []
    try:
        first = MainWindow(); windows.append(first)
        first._load_project_from_path(str(run6))
        before = _state(first)
        assert before['file'] == 'B2356run6.mat'
        assert before['window'] == [38.234, 42.234]
        assert before['warmup'] == 1500
        assert before['model_window'] == (38.234, 42.234)
        assert before['samples'] and before['samples'] > 0
        report = first._tire_model.report_dict()
        assert report['selection']['speed_window_kph'] == [38.234, 42.234]
        assert report['selection']['warmup_samples_excluded'] == 1500
        assert report['coverage']['negative_ia_is_inferred_by_mirror_extension']
        assert report['coverage']['intermediate_ia_is_interpolated_between_rows']

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            saved = td_path / 'run6_roundtrip.vahan'
            first._save_project_to_path(str(saved))
            reread = MainWindow(); windows.append(reread)
            reread._load_project_from_path(str(saved))
            after = _state(reread)
            assert after == before, (before, after)

            # Loading v117 into the same window must clear raw6 selection.
            reread._load_project_from_path(str(V117))
            restored = _state(reread)
            assert restored['file'] == 'B2356raw8.mat'
            assert restored['window'] is None and restored['warmup'] is None
            assert restored['model_window'] is None

            missing = json.loads(V117.read_text(encoding='utf-8'))
            missing['car']['tire_file'] = 'does_not_exist.mat'
            missing_path = td_path / 'missing_named_tire.vahan'
            missing_path.write_text(json.dumps(missing), encoding='utf-8')
            missing_win = MainWindow(); windows.append(missing_win)
            missing_win._load_project_from_path(str(missing_path))
            _expect_blocked(missing_win, 'missing from tire_data')

            # The same contract applies to a checkout without TTC files: a
            # declared project tyre must not silently enable the linear model.
            missing_win._car['tire_file'] = 'B2356raw8.mat'
            with patch('glob.glob', return_value=[]):
                missing_win._try_autoload_tire()
            _expect_blocked(missing_win, 'contains no usable tyre files')

            malformed = json.loads(run6.read_text(encoding='utf-8'))
            malformed['car']['tire_speed_window_kph'] = [38.234]
            malformed_path = td_path / 'malformed_speed_window.vahan'
            malformed_path.write_text(json.dumps(malformed), encoding='utf-8')
            malformed_win = MainWindow(); windows.append(malformed_win)
            malformed_win._load_project_from_path(str(malformed_path))
            _expect_blocked(malformed_win, 'tire_speed_window_kph must be a [low, high]')

        print('PASS: raw6 selection save/load; v117 restoration; missing, empty-data, and malformed declarations fail closed')
    finally:
        for window in windows:
            window.close()


if __name__ == '__main__':
    main()
