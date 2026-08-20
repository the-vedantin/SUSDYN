"""Import STEP solids (differential, engine, …) into the 3-D view for clearance.

Self-contained QDialog.  The heavy lifting (tessellation + coordinate transform)
is in vahan.step_import; this is only the UI around it.  Placement is explicit:
the SolidWorks/Onshape toggle sets the axis convention, and the offset spinboxes
let the user drop the part where it actually sits on the car — nothing hidden.

The canonical list of imported parts lives on the MainWindow (self._main.
_imported_parts) so it is saved into the .vahan project; this dialog reads and
edits that list, then asks the window to refresh the 3-D view.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QListWidget,
    QFileDialog, QMessageBox, QGroupBox,
)

from vahan.step_import import (load_step_mesh, export_moved_step,
                               transform_report, HAVE_STEP, HAVE_OCC)

# Human label -> vahan.step_import source key
_SOURCE_ITEMS = [("SolidWorks (Y-up)", "solidworks"),
                 ("Onshape (Z-up)", "onshape")]


class StepImportDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import STEP — differential / engine clearance")
        self._main = main                 # MainWindow: owns _imported_parts
        self.resize(560, 470)

        root = QVBoxLayout(self)

        if not HAVE_STEP:
            root.addWidget(QLabel(
                "STEP import needs the 'cascadio' and 'trimesh' packages.\n"
                "Install them with:\n\n    pip install cascadio trimesh\n\n"
                "then reopen this dialog."))
            btn = QPushButton("Close"); btn.clicked.connect(self.reject)
            root.addWidget(btn)
            return

        # ── file + source ────────────────────────────────────────────────
        box = QGroupBox("File & coordinate source")
        g = QGridLayout(box)
        self._path = QLineEdit(); self._path.setPlaceholderText("choose a .STEP file…")
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        g.addWidget(QLabel("STEP file:"), 0, 0)
        g.addWidget(self._path, 0, 1)
        g.addWidget(browse, 0, 2)

        self._source = QComboBox()
        for label, _key in _SOURCE_ITEMS:
            self._source.addItem(label)
        self._source.setToolTip(
            "Which CAD tool exported this file — they disagree on which axis is "
            "up.  SolidWorks is Y-up; Onshape (and the Suspension Points "
            "FeatureScript) is Z-up like Vahan.")
        g.addWidget(QLabel("Exported from:"), 1, 0)
        g.addWidget(self._source, 1, 1, 1, 2)
        root.addWidget(box)

        # ── placement ────────────────────────────────────────────────────
        pbox = QGroupBox("Placement (Vahan frame: X lateral, Y forward, Z up — mm)")
        pg = QGridLayout(pbox)
        self._off = []
        for i, ax in enumerate(("X offset", "Y offset", "Z offset")):
            sp = QDoubleSpinBox()
            sp.setRange(-10000.0, 10000.0); sp.setDecimals(3)
            sp.setSingleStep(1.0); sp.setSuffix(" mm")
            pg.addWidget(QLabel(ax + ":"), 0, i * 2)
            pg.addWidget(sp, 0, i * 2 + 1)
            self._off.append(sp)
        self._flip = QCheckBox("Flip front/back (Y)")
        self._flip.setToolTip("Negate the longitudinal axis if the part faces the "
                              "wrong way after import.")
        pg.addWidget(self._flip, 1, 0, 1, 6)
        root.addWidget(pbox)

        # ── loaded parts list ────────────────────────────────────────────
        root.addWidget(QLabel("Loaded parts (saved with the project):"))
        self._list = QListWidget()
        root.addWidget(self._list, 1)
        self._info = QLabel(""); self._info.setWordWrap(True)
        root.addWidget(self._info)

        # ── buttons ──────────────────────────────────────────────────────
        row = QHBoxLayout()
        add = QPushButton("Add part"); add.clicked.connect(self._add)
        rm = QPushButton("Remove selected"); rm.clicked.connect(self._remove)
        clr = QPushButton("Clear all"); clr.clicked.connect(self._clear)
        exp = QPushButton("Export moved part…"); exp.clicked.connect(self._export)
        exp.setToolTip("Write a real .STEP of the selected part at its Vahan "
                       "position (via the OpenCASCADE kernel) plus a text file "
                       "of the move, for the powertrain team.")
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        for b in (add, rm, clr, exp):
            row.addWidget(b)
        row.addStretch(1); row.addWidget(close)
        root.addLayout(row)

        self.sync_from_state()

    # ── helpers ──────────────────────────────────────────────────────────
    def _parts(self):
        return getattr(self._main, "_imported_parts", [])

    def sync_from_state(self):
        """Rebuild the list widget from the window's part list (call after a
        project load, or when re-opening the dialog)."""
        if not HAVE_STEP:
            return
        self._list.clear()
        for p in self._parts():
            s = p["info"]["size_mm"] if "info" in p else [0, 0, 0]
            self._list.addItem(
                f'{p.get("name","part")}  —  {s[0]:.0f}×{s[1]:.0f}×{s[2]:.0f} mm '
                f'({p.get("source","onshape")})')

    def _refresh(self):
        self.sync_from_state()
        self._main._push_imported_parts_to_view()

    # ── actions ──────────────────────────────────────────────────────────
    def _browse(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Choose STEP file", "",
            "STEP files (*.step *.stp *.STEP *.STP);;All files (*)")
        if fn:
            self._path.setText(fn)

    def _add(self):
        path = self._path.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Choose a STEP file first.")
            return
        source = _SOURCE_ITEMS[self._source.currentIndex()][1]
        offset = tuple(sp.value() for sp in self._off)
        flip = self._flip.isChecked()
        try:
            verts, faces, info = load_step_mesh(
                path, source=source, offset_mm=offset, flip_forward=flip)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))
            return
        self._parts().append({"name": info["path"], "source": source,
                              "offset": list(offset), "flip": flip,
                              "src_path": path,          # for STEP re-export
                              "verts": verts, "faces": faces, "info": info})
        self._refresh()
        sz = info["size_mm"]
        self._info.setText(
            f"Loaded {info['path']}: {info['n_faces']} triangles, "
            f"size {sz[0]:.1f} × {sz[1]:.1f} × {sz[2]:.1f} mm ({source}).  "
            f"Adjust the offsets and re-add if placement is off.  Save the "
            f"project to keep it.")

    def _remove(self):
        r = self._list.currentRow()
        parts = self._parts()
        if 0 <= r < len(parts):
            del parts[r]
            self._refresh()

    def _clear(self):
        self._parts().clear()
        self._refresh()

    def _export(self):
        parts = self._parts()
        r = self._list.currentRow()
        if not (0 <= r < len(parts)):
            QMessageBox.warning(self, "No selection",
                                "Select a loaded part to export.")
            return
        p = parts[r]
        if not HAVE_OCC:
            QMessageBox.critical(
                self, "STEP export unavailable",
                "Writing a moved STEP needs the OpenCASCADE kernel.\n"
                "Install it with:\n\n    pip install cadquery-ocp")
            return
        src = p.get("src_path")
        if not src:
            QMessageBox.critical(
                self, "Original STEP unknown",
                "This part has no stored source path (it may have come from a "
                "saved project). Re-import the STEP, position it, then export.")
            return
        base = os.path.splitext(p.get("name", "part"))[0] + "_placed.step"
        out, _ = QFileDialog.getSaveFileName(
            self, "Export moved STEP", base, "STEP files (*.step *.stp)")
        if not out:
            return
        try:
            export_moved_step(src, out, source=p.get("source", "onshape"),
                              offset_mm=p.get("offset", (0., 0., 0.)),
                              flip_forward=p.get("flip", False))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        info = p.get("info", {})
        txt = transform_report(
            p.get("name", "part"), p.get("source", "onshape"),
            p.get("offset", (0., 0., 0.)),
            info.get("bounds_min_mm", [0, 0, 0]),
            info.get("bounds_max_mm", [0, 0, 0]))
        txt_path = os.path.splitext(out)[0] + "_placement.txt"
        try:
            with open(txt_path, "w") as f:
                f.write(txt)
        except OSError:
            txt_path = "(could not write)"
        self._info.setText(f"Exported STEP → {out}\nPlacement notes → {txt_path}")
        QMessageBox.information(
            self, "Exported",
            f"Wrote:\n  {out}\n  {txt_path}\n\nThe STEP is positioned exactly "
            f"as shown in Vahan.")
