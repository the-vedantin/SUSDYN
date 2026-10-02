"""Import + manage STEP solids (differential, engine, ...) in the 3-D view.

Self-contained QDialog.  The heavy lifting (tessellation + coordinate transform)
is in vahan.step_import; this is only the UI around it.  Placement is explicit:
the SolidWorks/Onshape toggle sets the axis convention, and the offset spinboxes
put the part where it sits on the car - nothing hidden.

Management (2026-09-21): every imported part is a row with a show/hide tick;
selecting a row loads its name, offsets, flip, colour and opacity into the
editors and "Apply to selected" re-places it in the view (a saved project keeps
its mesh, so the move is done on the stored vertices; a part with a known source
file is re-tessellated from the STEP).  Remove / clear / export as before.

The canonical list of imported parts lives on the MainWindow (self._main.
_imported_parts) so it is saved into the .vahan project; this dialog reads and
edits that list, then asks the window to refresh the 3-D view.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QColorDialog,
)

from vahan.step_import import (load_step_mesh, export_moved_step,
                               transform_report, HAVE_STEP, HAVE_OCC)

# Human label -> vahan.step_import source key
_SOURCE_ITEMS = [("SolidWorks (Y-up)", "solidworks"),
                 ("Onshape (Z-up)", "onshape")]


class StepImportDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STEP parts - import and manage")
        self._main = main                 # MainWindow: owns _imported_parts
        self._syncing = False
        self.resize(640, 560)

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
        box = QGroupBox("Import a new part")
        g = QGridLayout(box)
        self._path = QLineEdit(); self._path.setPlaceholderText("choose a .STEP file...")
        browse = QPushButton("Browse..."); browse.clicked.connect(self._browse)
        g.addWidget(QLabel("STEP file:"), 0, 0)
        g.addWidget(self._path, 0, 1)
        g.addWidget(browse, 0, 2)
        self._source = QComboBox()
        for label, _key in _SOURCE_ITEMS:
            self._source.addItem(label)
        self._source.setToolTip(
            "Which CAD tool exported this file - they disagree on which axis is "
            "up.  SolidWorks is Y-up; Onshape (and the Suspension Points "
            "FeatureScript) is Z-up like Vahan.")
        g.addWidget(QLabel("Exported from:"), 1, 0)
        g.addWidget(self._source, 1, 1)
        add = QPushButton("Add part"); add.clicked.connect(self._add)
        g.addWidget(add, 1, 2)
        root.addWidget(box)

        # ── loaded parts list (tick = shown) ─────────────────────────────
        root.addWidget(QLabel("Loaded parts (tick = shown in the 3-D view; saved with the project):"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._load_selected)
        self._list.itemChanged.connect(self._on_tick)
        root.addWidget(self._list, 1)

        # ── placement / look of the SELECTED part ────────────────────────
        pbox = QGroupBox("Selected part (Vahan frame: X lateral, Y rearward, Z up - mm)")
        pg = QGridLayout(pbox)
        pg.addWidget(QLabel("Name:"), 0, 0)
        self._name = QLineEdit(); pg.addWidget(self._name, 0, 1, 1, 5)
        self._off = []
        for i, ax in enumerate(("X offset", "Y offset", "Z offset")):
            sp = QDoubleSpinBox()
            sp.setRange(-10000.0, 10000.0); sp.setDecimals(3)
            sp.setSingleStep(1.0); sp.setSuffix(" mm")
            pg.addWidget(QLabel(ax + ":"), 1, i * 2)
            pg.addWidget(sp, 1, i * 2 + 1)
            self._off.append(sp)
        self._flip = QCheckBox("Flip front/back (Y)")
        self._flip.setToolTip("Negate the longitudinal axis if the part faces the "
                              "wrong way after import.")
        pg.addWidget(self._flip, 2, 0, 1, 2)
        pg.addWidget(QLabel("Opacity:"), 2, 2)
        self._opacity = QDoubleSpinBox(); self._opacity.setRange(0.05, 1.0); self._opacity.setSingleStep(0.05)
        self._opacity.setDecimals(2); self._opacity.setValue(0.55)
        pg.addWidget(self._opacity, 2, 3)
        self._colour_btn = QPushButton("Colour..."); self._colour_btn.clicked.connect(self._pick_colour)
        pg.addWidget(self._colour_btn, 2, 4)
        self._rgba = None
        apply = QPushButton("Apply to selected"); apply.clicked.connect(self._apply_selected)
        apply.setToolTip("Re-place / rename / recolour the selected part with the values above")
        pg.addWidget(apply, 3, 0, 1, 2)
        show_all = QPushButton("Show all"); show_all.clicked.connect(lambda: self._set_all_visible(True))
        hide_all = QPushButton("Hide all"); hide_all.clicked.connect(lambda: self._set_all_visible(False))
        pg.addWidget(show_all, 3, 2); pg.addWidget(hide_all, 3, 3)
        root.addWidget(pbox)

        self._info = QLabel(""); self._info.setWordWrap(True)
        root.addWidget(self._info)

        # ── buttons ──────────────────────────────────────────────────────
        row = QHBoxLayout()
        rm = QPushButton("Remove selected"); rm.clicked.connect(self._remove)
        clr = QPushButton("Clear all"); clr.clicked.connect(self._clear)
        exp = QPushButton("Export moved part..."); exp.clicked.connect(self._export)
        exp.setToolTip("Write a real .STEP of the selected part at its Vahan "
                       "position (via the OpenCASCADE kernel) plus a text file "
                       "of the move, for the powertrain team.")
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        for b in (rm, clr, exp):
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
        self._syncing = True
        row = self._list.currentRow()
        self._list.clear()
        for p in self._parts():
            s = p["info"]["size_mm"] if "info" in p else [0, 0, 0]
            it = QListWidgetItem(
                f'{p.get("name", "part")}  -  {s[0]:.0f} x {s[1]:.0f} x {s[2]:.0f} mm '
                f'({p.get("source", "onshape")}; offset {", ".join(f"{v:.1f}" for v in p.get("offset", (0, 0, 0)))} mm'
                f'{"; flipped" if p.get("flip") else ""})')
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if p.get("visible", True) else Qt.CheckState.Unchecked)
            self._list.addItem(it)
        self._syncing = False
        if 0 <= row < self._list.count():
            self._list.setCurrentRow(row)
        elif self._list.count():
            self._list.setCurrentRow(0)

    def _refresh(self):
        self.sync_from_state()
        self._main._push_imported_parts_to_view()

    def _load_selected(self, row):
        parts = self._parts()
        if not (0 <= row < len(parts)):
            return
        p = parts[row]
        self._name.setText(str(p.get("name", "part")))
        for sp, v in zip(self._off, p.get("offset", (0., 0., 0.))):
            sp.setValue(float(v))
        self._flip.setChecked(bool(p.get("flip", False)))
        self._opacity.setValue(float(p.get("opacity", 0.55)))
        self._rgba = list(p["rgba"]) if p.get("rgba") else None
        self._colour_btn.setStyleSheet(
            f"background: rgb({int(self._rgba[0]*255)},{int(self._rgba[1]*255)},{int(self._rgba[2]*255)})"
            if self._rgba else "")

    def _on_tick(self, it):
        if self._syncing:
            return
        row = self._list.row(it)
        parts = self._parts()
        if 0 <= row < len(parts):
            parts[row]["visible"] = it.checkState() == Qt.CheckState.Checked
            self._main._push_imported_parts_to_view()

    def _set_all_visible(self, on):
        for p in self._parts():
            p["visible"] = bool(on)
        self._refresh()

    def _pick_colour(self):
        start = QColor.fromRgbF(*self._rgba[:3]) if self._rgba else QColor(140, 145, 153)
        c = QColorDialog.getColor(start, self, "Part colour")
        if c.isValid():
            self._rgba = [c.redF(), c.greenF(), c.blueF(), float(self._opacity.value())]
            self._colour_btn.setStyleSheet(f"background: rgb({c.red()},{c.green()},{c.blue()})")

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
                              "visible": True, "opacity": float(self._opacity.value()),
                              "rgba": list(self._rgba) if self._rgba else None,
                              "verts": verts, "faces": faces, "info": info})
        self._refresh()
        self._list.setCurrentRow(self._list.count() - 1)
        sz = info["size_mm"]
        self._info.setText(
            f"Loaded {info['path']}: {info['n_faces']} triangles, "
            f"size {sz[0]:.1f} x {sz[1]:.1f} x {sz[2]:.1f} mm ({source}).  "
            f"Move it with the offsets + Apply to selected.  Save the project to keep it.")

    def _apply_selected(self):
        row = self._list.currentRow()
        parts = self._parts()
        if not (0 <= row < len(parts)):
            QMessageBox.warning(self, "No selection", "Select a loaded part first.")
            return
        p = parts[row]
        new_off = np.array([sp.value() for sp in self._off], float)
        new_flip = self._flip.isChecked()
        src = p.get("src_path")
        if src and os.path.isfile(src):
            try:
                verts, faces, info = load_step_mesh(src, source=p.get("source", "onshape"),
                                                    offset_mm=tuple(new_off), flip_forward=new_flip)
                p["verts"], p["faces"], p["info"] = verts, faces, info
            except Exception as e:
                QMessageBox.critical(self, "Re-import failed", str(e))
                return
        else:
            # Saved project: undo the stored placement on the stored mesh, redo the new one.
            v = np.asarray(p["verts"], float) - np.asarray(p.get("offset", (0., 0., 0.)), float)
            if bool(p.get("flip", False)) != new_flip:
                v[:, 1] = -v[:, 1]
            p["verts"] = v + new_off
            if "info" in p:
                p["info"]["bounds_min_mm"] = [float(x) for x in p["verts"].min(axis=0)]
                p["info"]["bounds_max_mm"] = [float(x) for x in p["verts"].max(axis=0)]
        p["offset"] = [float(x) for x in new_off]
        p["flip"] = bool(new_flip)
        p["name"] = self._name.text().strip() or p.get("name", "part")
        p["opacity"] = float(self._opacity.value())
        if self._rgba:
            p["rgba"] = list(self._rgba[:3]) + [p["opacity"]]
        self._refresh()
        self._list.setCurrentRow(row)
        self._info.setText(f"Applied to '{p['name']}': offset {p['offset']} mm, flip {p['flip']}, opacity {p['opacity']:.2f}.  "
                           "Save the project to keep it.")

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
        self._info.setText(f"Exported STEP -> {out}\nPlacement notes -> {txt_path}")
        QMessageBox.information(
            self, "Exported",
            f"Wrote:\n  {out}\n  {txt_path}\n\nThe STEP is positioned exactly "
            f"as shown in Vahan.")
