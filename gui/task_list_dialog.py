"""Help -> Task list: docs/GUI_TASKS.md shown as a checklist.

The markdown file is the single source (Claude appends promised features there);
ticking a box rewrites that line's `[ ]` / `[x]` in place, nothing else is touched.
"""
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                             QPushButton, QLabel, QLineEdit)

TASK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'GUI_TASKS.md')


class TaskListDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Task list (docs/GUI_TASKS.md)')
        self.resize(760, 480)
        root = QVBoxLayout(self)
        hint = QLabel('Tick = done. Add a task below; Claude adds promised features here too.')
        hint.setWordWrap(True)
        root.addWidget(hint)
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._list, 1)
        row = QHBoxLayout()
        self._new = QLineEdit(); self._new.setPlaceholderText('new task...')
        add = QPushButton('Add'); add.clicked.connect(self._add)
        reload_btn = QPushButton('Reload file'); reload_btn.clicked.connect(self.reload)
        close = QPushButton('Close'); close.clicked.connect(self.accept)
        row.addWidget(self._new, 1); row.addWidget(add); row.addWidget(reload_btn); row.addWidget(close)
        root.addLayout(row)
        self._lines = []

    def reload(self):
        self._list.blockSignals(True)
        self._list.clear()
        try:
            with open(TASK_FILE, encoding='utf-8') as f:
                self._lines = f.read().splitlines()
        except OSError:
            self._lines = ['# Vahan GUI task list', '']
        for i, line in enumerate(self._lines):
            s = line.strip()
            if s.startswith('- [ ]') or s.startswith('- [x]') or s.startswith('- [X]'):
                it = QListWidgetItem(s[5:].strip())
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if s[3] in 'xX' else Qt.CheckState.Unchecked)
                it.setData(Qt.ItemDataRole.UserRole, i)
                self._list.addItem(it)
        self._list.blockSignals(False)

    def _write(self):
        with open(TASK_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._lines) + '\n')

    def _on_item_changed(self, it):
        i = it.data(Qt.ItemDataRole.UserRole)
        if i is None or not (0 <= i < len(self._lines)):
            return
        done = it.checkState() == Qt.CheckState.Checked
        line = self._lines[i]
        k = line.find('- [')
        if k >= 0:
            self._lines[i] = line[:k + 3] + ('x' if done else ' ') + line[k + 4:]
            self._write()

    def _add(self):
        text = self._new.text().strip()
        if not text:
            return
        self._lines.append(f'- [ ] {text}')
        self._write(); self._new.clear(); self.reload()
