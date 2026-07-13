"""Subtitle Translator v4 — PySide6 GUI."""

import io, json, os, sys, time, traceback
from pathlib import Path
from enum import Enum

from PySide6.QtCore import (
    Qt, QThread, QSettings, Signal, QObject, QTimer,
)
from PySide6.QtGui import QFont, QAction, QIcon, QTextCursor, QTextCharFormat, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QPlainTextEdit, QTextEdit, QFrame,
    QFileDialog, QDialog, QDialogButtonBox, QGridLayout,
    QMessageBox, QCheckBox, QSizePolicy, QStyle,
    QStyledItemDelegate, QStyleOptionViewItem,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from translator import (
    __version__, Config, _auto_device, find_srt_files, output_path_for,
    translate_fast, translate_polish,
    load_nllb, load_glossary, load_german_fixes, load_titles,
    generate_qa_report,
    _checkpoint_path, _remove_checkpoint,
)


# ---------------------------------------------------------------------------
# Elide delegate — shows "..." for text wider than column
# ---------------------------------------------------------------------------

class ElideDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = painter.fontMetrics().elidedText(
            opt.text, Qt.ElideRight, opt.rect.width()
        )
        QApplication.style().drawControl(QStyle.CE_ItemViewItem, opt, painter)


# ---------------------------------------------------------------------------
# Stdout capture — feeds all print() calls into the GUI log
# ---------------------------------------------------------------------------

class LogSignal(QObject):
    written = Signal(str)


class LogCapture(io.StringIO):
    """Replace sys.stdout so engine print() calls appear in the GUI log."""

    def __init__(self, signal: LogSignal):
        super().__init__()
        self._signal = signal

    def write(self, s: str):
        if s.strip():
            self._signal.written.emit(s.rstrip())
        super().write(s)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Drag-and-drop file zone
# ---------------------------------------------------------------------------

class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setFrameShape(QFrame.StyledPanel)
        self._label = QLabel(
            "Drop .srt files here  or  click Browse below",
            self,
            alignment=Qt.AlignCenter,
        )
        self._label.setStyleSheet("color: #888; font-size: 12px; border: none;")
        lay = QVBoxLayout(self)
        lay.addWidget(self._label)
        self._normal_style = (
            "DropZone { border: 2px dashed #555; border-radius: 6px; "
            "background: #1e1e1e; }"
        )
        self._drag_style = (
            "DropZone { border: 2px dashed #4a9eff; border-radius: 6px; "
            "background: #1a2a40; }"
        )
        self.setStyleSheet(self._normal_style)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._drag_style)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._normal_style)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._normal_style)
        files = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() == ".srt":
                files.append(p)
        if files:
            self.files_dropped.emit(files)


# ---------------------------------------------------------------------------
# Status enum for queue rows
# ---------------------------------------------------------------------------

class FileStatus(Enum):
    QUEUED = 0
    RUNNING = 1
    DONE = 2
    SKIPPED = 3
    ERROR = 4


# ---------------------------------------------------------------------------
# Translation worker (runs in QThread)
# ---------------------------------------------------------------------------

class TranslationWorker(QObject):
    progress = Signal(int, int)         # done, total
    file_progress = Signal(int, int)    # file_index, file_count
    log = Signal(str)
    current_file = Signal(str)          # filename
    current_en = Signal(str)            # current EN subtitle
    current_de = Signal(str)            # current DE subtitle
    step_changed = Signal(str)          # current pipeline step
    speed_eta = Signal(float, float)    # l/s, eta_seconds
    file_done = Signal(str, dict)       # filename, stats dict
    all_done = Signal()
    error_happened = Signal(str)

    def __init__(self, files, cfg, output_dir, mode, polish_model, ollama_host, resume):
        super().__init__()
        self.files = files
        self.cfg = cfg
        self.output_dir = output_dir
        self.mode = mode
        self.polish_model = polish_model
        self.ollama_host = ollama_host
        self.resume = resume
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total_files = len(self.files)
        try:
            for idx, fpath in enumerate(self.files):
                if self._cancelled:
                    break

                self.file_progress.emit(idx + 1, total_files)
                self.current_file.emit(fpath.name)
                self.log.emit(f"[{idx + 1}/{total_files}] {fpath.name}")

                # Determine output path
                if self.output_dir:
                    out = Path(self.output_dir) / output_path_for(fpath).name
                else:
                    out = output_path_for(fpath)

                # Skip if exists and not forced and no checkpoint
                has_checkpoint = _checkpoint_path(out).exists()
                if out.exists() and not self.cfg.force and not has_checkpoint:
                    self.log.emit(f"  [SKIP] {fpath.name} (already exists)")
                    self.file_done.emit(fpath.name, {"skipped": True})
                    continue

                # Build a per-file Config
                file_cfg = Config(
                    src_lang=self.cfg.src_lang,
                    tgt_lang=self.cfg.tgt_lang,
                    device=self.cfg.device,
                    batch_size=self.cfg.batch_size,
                    ollama_model=self.polish_model or self.cfg.ollama_model,
                    ollama_host=getattr(self, 'ollama_host', self.cfg.ollama_host),
                    input_dir=str(fpath.parent),
                    force=self.cfg.force,
                    resume=self.resume or has_checkpoint,
                    mode="fast",
                )

                # Pre-read SRT for live preview
                import pysrt
                _eng_subs = pysrt.open(str(fpath), encoding="utf-8")
                _eng_texts = [s.text for s in _eng_subs]
                _ger_path = output_path_for(fpath)
                _ger_texts_pct = {}

                # --- Fast pass ---
                self.step_changed.emit("Translating (NLLB)")

                def timed_progress(done, total):
                    if self._cancelled:
                        raise KeyboardInterrupt()
                    elapsed = time.time() - getattr(self, '_t0', time.time())
                    self._batch_elapsed = elapsed
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    self.progress.emit(done, total)
                    self.speed_eta.emit(rate, eta)
                    # Emit current EN line for live preview
                    if done > 0 and done - 1 < len(_eng_texts):
                        self.current_en.emit(_eng_texts[done - 1])
                    # Try to read latest DE from output file (checkpoint/batch)
                    if _ger_path.exists() and (_ger_texts_pct.get(done) is None):
                        try:
                            _ger_subs = pysrt.open(str(_ger_path), encoding="utf-8")
                            for gi, gs in enumerate(_ger_subs):
                                _ger_texts_pct[gi + 1] = gs.text
                        except Exception:
                            pass
                    if done in _ger_texts_pct:
                        self.current_de.emit(_ger_texts_pct[done])

                self._t0 = time.time()

                try:
                    self.step_changed.emit("Protecting placeholders")

                    success = translate_fast(fpath, file_cfg,
                                             progress_callback=timed_progress,
                                             output_path=out)
                except KeyboardInterrupt:
                    self.log.emit("  Cancelled by user")
                    self.all_done.emit()
                    return
                except Exception as e:
                    self.error_happened.emit(f"Error translating {fpath.name}: {e}\n{traceback.format_exc()}")
                    self.file_done.emit(fpath.name, {"error": str(e)})
                    continue

                if not success:
                    self.file_done.emit(fpath.name, {"error": "translate_fast returned False"})
                    continue

                # --- Polish pass (if mode includes polish) ---
                if self.mode in ("polish", "full"):
                    self.step_changed.emit("Polishing (Ollama)")
                    try:
                        translate_polish(fpath, file_cfg, nllb_path=out)
                    except Exception as e:
                        self.log.emit(f"  [WARN] Polish error (non-fatal): {e}")

                # --- Read QA report ---
                stats = {"skipped": False}
                qa_path = out.with_suffix(".qa_report.json")
                if qa_path.exists():
                    try:
                        with open(qa_path, encoding="utf-8") as f:
                            qa_data = json.load(f)
                        total_score = qa_data.get("total_score", 0)
                        suspicious = sum(
                            1 for s in qa_data.get("scores", {}).values()
                            if isinstance(s, (int, float)) and s >= 5
                        )
                        stats["qa_score"] = total_score
                        stats["suspicious"] = suspicious
                    except Exception:
                        pass

                # Count output lines
                try:
                    import pysrt
                    out_subs = pysrt.open(str(out), encoding="utf-8")
                    stats["lines"] = len(out_subs)
                except Exception:
                    pass

                self.file_done.emit(fpath.name, stats)
                self.log.emit(f"  DONE: {out.name}")

            # All files done
            self.step_changed.emit("Complete")
            self.all_done.emit()

        except Exception as e:
            self.error_happened.emit(f"Worker error: {e}\n{traceback.format_exc()}")
            self.all_done.emit()


# ---------------------------------------------------------------------------
# Summary dialog
# ---------------------------------------------------------------------------

class SummaryDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Translation Complete")
        self.setMinimumWidth(560)
        self.setModal(True)
        self.results = results

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("Translation Complete")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        # Per-file stats
        for fname, s in results:
            if s.get("skipped") or s.get("error"):
                continue

            name_label = QLabel(Path(fname).stem)
            name_font = QFont()
            name_font.setPointSize(11)
            name_font.setBold(True)
            name_label.setFont(name_font)
            layout.addWidget(name_label)

            grid = QGridLayout()
            grid.setSpacing(6)
            row = 0

            def add_row(label, value):
                nonlocal row
                lbl = QLabel(label)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                val = QLabel(str(value))
                val_font = QFont()
                val_font.setBold(True)
                val.setFont(val_font)
                grid.addWidget(lbl, row, 0)
                grid.addWidget(val, row, 1)
                row += 1

            add_row("Runtime", s.get("runtime", "—"))
            qa = s.get("qa_score", "—")
            add_row("QA Score", qa)
            add_row("Suspicious Lines", s.get("suspicious", 0))
            add_row("Ollama Corrections", s.get("ollama_corrections", "—"))
            add_row("TM Hit Rate", s.get("tm_hit_rate", "—"))

            # Buttons for this file
            btn_row = QHBoxLayout()
            open_sub_btn = QPushButton("Open Subtitle")
            open_sub_btn.clicked.connect(lambda checked, fn=fname: self._open_srt(fn))
            open_folder_btn = QPushButton("Open Folder")
            open_folder_btn.clicked.connect(lambda checked, fn=fname: self._open_folder(fn))
            export_qa_btn = QPushButton("Export QA Report")
            export_qa_btn.clicked.connect(lambda checked, fn=fname: self._export_qa(fn))
            btn_row.addWidget(open_sub_btn)
            btn_row.addWidget(open_folder_btn)
            btn_row.addWidget(export_qa_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            # Separator between files
            if len(results) > 1:
                sep2 = QFrame()
                sep2.setFrameShape(QFrame.HLine)
                sep2.setStyleSheet("color: #333;")
                layout.addWidget(sep2)

        layout.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)

        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #ddd; }
            QLabel { color: #ddd; font-size: 13px; }
            QPushButton { padding: 6px 16px; font-size: 12px; }
            QPushButton:hover { border-color: #4a9eff; }
        """)

    def _open_srt(self, fname):
        for p in self.parent().file_queue:
            if p.name == fname:
                out = output_path_for(p)
                if out.exists():
                    os.startfile(str(out))
                break

    def _open_folder(self, fname):
        for p in self.parent().file_queue:
            if p.name == fname:
                os.startfile(str(p.parent))
                break

    def _export_qa(self, fname):
        """Find and export the QA report JSON for a finished file."""
        for p in self.parent().file_queue:
            if p.name == fname:
                qa_path = p.with_suffix(".qa_report.json")
                out_path = Path(self.parent().out_path.text().strip()) if self.parent().out_path.text().strip() else p.parent
                qa_path = out_path / qa_path.name if out_path != p.parent else qa_path
                if not qa_path.exists():
                    QMessageBox.information(self, "No QA Report",
                                            f"No QA report found for {fname}.\n\n"
                                            "Reports are generated after translation completes.")
                    return
                dest, _ = QFileDialog.getSaveFileName(
                    self, "Export QA Report", qa_path.stem + ".txt",
                    "Text files (*.txt);;JSON files (*.json);;All files (*)"
                )
                if not dest:
                    return
                try:
                    import json
                    with open(qa_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if dest.endswith(".json"):
                        import shutil
                        shutil.copy2(str(qa_path), dest)
                    else:
                        lines = []
                        lines.append(f"QA Report: {fname}")
                        lines.append("=" * 50)
                        lines.append(f"Total Score: {data.get('total_score', 'N/A')}")
                        lines.append(f"Suspicious Lines: {sum(1 for s in data.get('scores', {}).values() if isinstance(s, (int, float)) and s >= 5)}")
                        lines.append("")
                        for lid, score in sorted(data.get("scores", {}).items()):
                            reasons = data.get("reasons", {}).get(lid, [])
                            if reasons:
                                lines.append(f"  Line {lid}: score={score} — {', '.join(reasons)}")
                        lines.append("")
                        for lid, details in data.get("details", {}).items():
                            if details:
                                lines.append(f"  Line {lid}: {details}")
                        with open(dest, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines))
                    self.parent().log(f"QA report exported to {dest}")
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"Could not export report:\n{e}")
                break


# ===================================================================
# Style Sheets
# ===================================================================

SECTION_HEADER = """
    QGroupBox {
        border: 1px solid #444; border-radius: 6px; margin-top: 16px;
        padding: 18px 12px 12px; font-weight: bold; font-size: 12px; color: #bbb;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 12px; padding: 0 6px;
    }
"""
SECTION_HEADER_LIGHT = """
    QGroupBox {
        border: 1px solid #ccc; border-radius: 6px; margin-top: 16px;
        padding: 18px 12px 12px; font-weight: bold; font-size: 12px; color: #555;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 12px; padding: 0 6px;
    }
"""

DARK_STYLE = f"""
QMainWindow, QDialog {{ background: #1e1e1e; }}
QWidget {{ color: #ddd; font-size: 12px; }}
QGroupBox {{
    border: 1px solid #444; border-radius: 6px; margin-top: 16px;
    padding: 18px 12px 12px; font-weight: bold; font-size: 12px; color: #bbb;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
}}
QPushButton {{
    background: #2d2d2d; border: 1px solid #555; border-radius: 4px;
    padding: 6px 16px; color: #ddd; min-height: 26px;
}}
QPushButton:hover {{ background: #3a3a3a; border-color: #4a9eff; }}
QPushButton:pressed {{ background: #1a1a1a; }}
QPushButton:disabled {{ color: #555; background: #222; border-color: #333; }}
QPushButton#startBtn {{
    background: #1a6b3c; border-color: #2a8b4c; font-weight: bold;
    font-size: 14px; padding: 10px 40px; min-height: 32px; border-radius: 6px;
}}
QPushButton#startBtn:hover {{ background: #1f7d46; }}
QPushButton#startBtn:disabled {{ background: #1a3b2c; border-color: #2a4b3c; color: #555; }}
QPushButton#cancelBtn {{
    background: #6b1a1a; border-color: #8b2a2a; font-weight: bold;
    font-size: 14px; padding: 10px 40px; min-height: 32px; border-radius: 6px;
}}
QPushButton#cancelBtn:hover {{ background: #7d1f1f; }}
QPushButton#cancelBtn:disabled {{ background: #3b1a1a; border-color: #4b2a2a; color: #555; }}
QLineEdit {{
    background: #252525; border: 1px solid #444; border-radius: 3px;
    padding: 5px 8px; color: #ddd; min-height: 22px;
}}
QLineEdit:focus {{ border-color: #4a9eff; }}
QComboBox {{
    background: #252525; border: 1px solid #444; border-radius: 3px;
    padding: 4px 8px; color: #ddd; min-height: 24px;
}}
QComboBox:hover {{ border-color: #4a9eff; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: #2d2d2d; color: #ddd; selection-background-color: #1a3a5a;
}}
QSpinBox {{
    background: #252525; border: 1px solid #444; border-radius: 3px;
    padding: 4px 8px; color: #ddd; min-height: 24px;
}}
QSpinBox:focus {{ border-color: #4a9eff; }}
QProgressBar {{
    border: 1px solid #444; border-radius: 4px; text-align: center;
    color: #ddd; background: #252525; min-height: 10px;
    font-size: 11px;
}}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #1a6b3c, stop:1 #2a9b5c); border-radius: 3px; }}
QTableWidget {{
    background: #1a1a1a; border: 1px solid #333; border-radius: 3px;
    gridline-color: #333; color: #ddd;
}}
QTableWidget::item {{ padding: 3px 6px; }}
QHeaderView::section {{
    background: #252525; color: #aaa; border: 1px solid #333;
    padding: 4px 8px; font-weight: bold;
}}
QPlainTextEdit, QTextEdit {{
    background: #121212; border: 1px solid #333; border-radius: 3px;
    color: #ddd; font-family: 'Consolas', 'Courier New', monospace;
    padding: 4px;
}}
QRadioButton {{ spacing: 4px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 2px solid #555; background: #2d2d2d;
}}
QRadioButton::indicator:checked {{
    background: #2a8b4c; border-color: #2a8b4c;
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 2px solid #555; background: #2d2d2d;
}}
QCheckBox::indicator:checked {{
    background: #2a8b4c; border-color: #2a8b4c;
}}
QLabel#subPreviewEn {{
    color: #7af; font-size: 12px; padding: 6px 8px;
    border: 1px solid #333; border-radius: 3px; background: #181818;
}}
QLabel#subPreviewDe {{
    color: #7f7; font-size: 12px; padding: 6px 8px;
    border: 1px solid #333; border-radius: 3px; background: #181818;
}}
QLabel#curFileLabel {{
    font-size: 14px; font-weight: bold; color: #4a9eff; padding: 4px 0;
}}
QLabel#stepLabel {{
    color: #aaa; font-size: 12px; padding: 2px 0;
}}
QLabel#statusCuda {{ color: #4a9eff; }}
QLabel#statusTm {{ color: #51cf66; }}
QLabel#statusQa {{ color: #ffd93d; }}
QLabel#statusOllama {{ color: #da77f2; }}
QLabel#statusElapsed {{ color: #aaa; }}
QMenuBar {{ background: #1a1a1a; color: #ddd; border-bottom: 1px solid #333; }}
QMenuBar::item:selected {{ background: #2d2d2d; }}
QMenu {{ background: #1e1e1e; color: #ddd; border: 1px solid #444; }}
QMenu::item:selected {{ background: #2a3a5a; }}
QSplitter::handle {{ background: #333; width: 2px; }}
"""

LIGHT_STYLE = f"""
QMainWindow, QDialog {{ background: #f5f5f5; }}
QWidget {{ color: #222; font-size: 12px; }}
QGroupBox {{
    border: 1px solid #ccc; border-radius: 6px; margin-top: 16px;
    padding: 18px 12px 12px; font-weight: bold; font-size: 12px; color: #555;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
}}
QPushButton {{
    background: #e8e8e8; border: 1px solid #aaa; border-radius: 4px;
    padding: 6px 16px; color: #222; min-height: 26px;
}}
QPushButton:hover {{ background: #ddd; border-color: #4a9eff; }}
QPushButton:pressed {{ background: #ccc; }}
QPushButton:disabled {{ color: #999; background: #eee; border-color: #ddd; }}
QPushButton#startBtn {{
    background: #1a8b3c; border-color: #2a9b4c; font-weight: bold;
    font-size: 14px; padding: 10px 40px; min-height: 32px; border-radius: 6px; color: #fff;
}}
QPushButton#startBtn:hover {{ background: #1f9d46; }}
QPushButton#startBtn:disabled {{ background: #aaddbb; border-color: #99ccaa; color: #999; }}
QPushButton#cancelBtn {{
    background: #cc3333; border-color: #dd4444; font-weight: bold;
    font-size: 14px; padding: 10px 40px; min-height: 32px; border-radius: 6px; color: #fff;
}}
QPushButton#cancelBtn:hover {{ background: #dd4444; }}
QPushButton#cancelBtn:disabled {{ background: #ddbbbb; border-color: #ccaabb; color: #999; }}
QLineEdit {{
    background: #fff; border: 1px solid #ccc; border-radius: 3px;
    padding: 5px 8px; color: #222; min-height: 22px;
}}
QLineEdit:focus {{ border-color: #4a9eff; }}
QComboBox {{
    background: #fff; border: 1px solid #ccc; border-radius: 3px;
    padding: 4px 8px; color: #222; min-height: 24px;
}}
QComboBox:hover {{ border-color: #4a9eff; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: #fff; color: #222; selection-background-color: #cce5ff;
}}
QSpinBox {{
    background: #fff; border: 1px solid #ccc; border-radius: 3px;
    padding: 4px 8px; color: #222; min-height: 24px;
}}
QSpinBox:focus {{ border-color: #4a9eff; }}
QProgressBar {{
    border: 1px solid #ccc; border-radius: 4px; text-align: center;
    color: #222; background: #fff; min-height: 10px; font-size: 11px;
}}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #1a8b3c, stop:1 #2abb5c); border-radius: 3px; }}
QTableWidget {{
    background: #fff; border: 1px solid #ccc; border-radius: 3px;
    gridline-color: #ddd; color: #222;
}}
QTableWidget::item {{ padding: 3px 6px; }}
QHeaderView::section {{
    background: #e8e8e8; color: #555; border: 1px solid #ccc;
    padding: 4px 8px; font-weight: bold;
}}
QPlainTextEdit, QTextEdit {{
    background: #fafafa; border: 1px solid #ccc; border-radius: 3px;
    color: #222; font-family: 'Consolas', 'Courier New', monospace;
    padding: 4px;
}}
QRadioButton {{ spacing: 4px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 2px solid #aaa; background: #fff;
}}
QRadioButton::indicator:checked {{
    background: #2a8b4c; border-color: #2a8b4c;
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 2px solid #aaa; background: #fff;
}}
QCheckBox::indicator:checked {{
    background: #2a8b4c; border-color: #2a8b4c;
}}
QLabel#subPreviewEn {{
    color: #2366ad; font-size: 12px; padding: 6px 8px;
    border: 1px solid #ddd; border-radius: 3px; background: #f0f4ff;
}}
QLabel#subPreviewDe {{
    color: #1a8b3c; font-size: 12px; padding: 6px 8px;
    border: 1px solid #ddd; border-radius: 3px; background: #f0fff4;
}}
QLabel#curFileLabel {{
    font-size: 14px; font-weight: bold; color: #2366ad; padding: 4px 0;
}}
QLabel#stepLabel {{
    color: #666; font-size: 12px; padding: 2px 0;
}}
QMenuBar {{ background: #e8e8e8; color: #222; border-bottom: 1px solid #ccc; }}
QMenuBar::item:selected {{ background: #ddd; }}
QMenu {{ background: #fff; color: #222; border: 1px solid #ccc; }}
QMenu::item:selected {{ background: #cce5ff; }}
QSplitter::handle {{ background: #ccc; width: 2px; }}
"""


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # GPU
        gpu_grp = QGroupBox("GPU")
        gpu_lay = QGridLayout(gpu_grp)
        gpu_lay.setSpacing(8)
        gpu_lay.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        gpu_lay.addWidget(self.device_combo, 0, 1)
        gpu_lay.addWidget(QLabel("Batch size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(64)
        self.batch_spin.setSuffix(" lines")
        gpu_lay.addWidget(self.batch_spin, 1, 1)
        layout.addWidget(gpu_grp)

        # Ollama
        ollama_grp = QGroupBox("Ollama")
        ollama_lay = QGridLayout(ollama_grp)
        ollama_lay.setSpacing(8)
        ollama_lay.addWidget(QLabel("Model:"), 0, 0)
        self.ollama_model = QLineEdit()
        self.ollama_model.setPlaceholderText("subtitle-translator")
        ollama_lay.addWidget(self.ollama_model, 0, 1)
        ollama_lay.addWidget(QLabel("Host:"), 1, 0)
        self.ollama_host = QLineEdit()
        self.ollama_host.setPlaceholderText("http://127.0.0.1:11434")
        ollama_lay.addWidget(self.ollama_host, 1, 1)
        layout.addWidget(ollama_grp)

        # Resume
        self.resume_cb = QCheckBox("Resume checkpoints")
        self.resume_cb.setToolTip("Resume interrupted translations from saved checkpoint")
        layout.addWidget(self.resume_cb)

        layout.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Copy values from parent
        if parent and hasattr(parent, 'device_combo'):
            self.device_combo.setCurrentIndex(parent.device_combo.currentIndex())
            self.batch_spin.setValue(parent.batch_spin.value())
            self.resume_cb.setChecked(parent.resume_cb.isChecked())
            self.ollama_model.setText(parent.polish_model)
            self.ollama_host.setText(parent.ollama_host)


# ===================================================================
# Main Window
# ===================================================================

class TranslatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Subtitle Translator v{__version__}")
        self.setMinimumSize(960, 680)
        self.resize(1100, 760)

        self.file_queue: list[Path] = []
        self.results: list[tuple[str, dict]] = []
        self.worker: TranslationWorker | None = None
        self.thread: QThread | None = None
        self._running = False
        self._dark_mode = True
        self._log_auto_scroll = True

        self.settings = QSettings("SubtitleTranslator", "v4")
        self.polish_model = self.settings.value("ollama/model", "subtitle-translator")
        self.ollama_host = self.settings.value("ollama/host", "http://127.0.0.1:11434")
        self.log_signal = LogSignal()
        self.log_signal.written.connect(self._on_log)

        self._setup_ui()
        # Status bar must be set up before restoring settings
        self._setup_status_bar()
        self._restore_settings()
        self.setStyleSheet(DARK_STYLE)

    # ------------------------------------------------------------------
    # Status Bar
    # ------------------------------------------------------------------

    def _setup_status_bar(self):
        self.status_bar = self.statusBar()

        def make_label(obj_name, text):
            lbl = QLabel(text)
            lbl.setObjectName(obj_name)
            lbl.setContentsMargins(8, 0, 8, 0)
            return lbl

        self.stat_cuda = make_label("statusCuda", "CUDA ✓")
        self.stat_gpu = make_label("statusGpu", "GPU --")
        self.stat_tm = make_label("statusTm", "TM: 0%")
        self.stat_qa = make_label("statusQa", "QA: 0")
        self.stat_ollama = make_label("statusOllama", "Ollama: —")
        self.stat_elapsed = make_label("statusElapsed", "Elapsed: 00:00")

        sep = QLabel("│")
        sep.setStyleSheet("color: #444;")
        sep0 = QLabel("│")
        sep0.setStyleSheet("color: #444;")
        sep2 = QLabel("│")
        sep2.setStyleSheet("color: #444;")
        sep3 = QLabel("│")
        sep3.setStyleSheet("color: #444;")
        sep4 = QLabel("│")
        sep4.setStyleSheet("color: #444;")

        self.status_bar.addPermanentWidget(self.stat_cuda)
        self.status_bar.addPermanentWidget(sep0)
        self.status_bar.addPermanentWidget(self.stat_gpu)
        self.status_bar.addPermanentWidget(sep)
        self.status_bar.addPermanentWidget(self.stat_tm)
        self.status_bar.addPermanentWidget(sep2)
        self.status_bar.addPermanentWidget(self.stat_qa)
        self.status_bar.addPermanentWidget(sep3)
        self.status_bar.addPermanentWidget(self.stat_ollama)
        self.status_bar.addPermanentWidget(sep4)
        self.status_bar.addPermanentWidget(self.stat_elapsed)

        self.status_bar.showMessage("Ready")

    def _update_status_elapsed(self):
        if self._running and hasattr(self, '_start_time'):
            elapsed = time.time() - self._start_time
            m, s = divmod(int(elapsed), 60)
            self.stat_elapsed.setText(f"Elapsed: {m:02d}:{s:02d}")
        # GPU / VRAM — periodic poll
        gpu_text = "GPU --"
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                used_mb = (total_b - free_b) / 1024 / 1024
                total_mb = total_b / 1024 / 1024
                gpu_text = f"GPU {used_mb:.0f} / {total_mb:.0f} MB"
        except Exception:
            pass
        self.stat_gpu.setText(gpu_text)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        menubar = self.menuBar()
        view_menu = menubar.addMenu("View")
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)
        self.dark_mode_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.dark_mode_action)

        tools_menu = menubar.addMenu("Tools")
        settings_action = QAction("Settings…", self)
        settings_action.triggered.connect(self._open_settings)
        tools_menu.addAction(settings_action)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setSpacing(6)
        outer.setContentsMargins(8, 8, 8, 8)

        # === Input + Output row ===
        io_row = QHBoxLayout()
        io_row.setSpacing(8)

        # Input
        input_grp = QGroupBox("Input Files")
        input_lay = QVBoxLayout(input_grp)
        input_lay.setSpacing(6)
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        input_lay.addWidget(self.drop_zone)

        browse_row = QHBoxLayout()
        browse_row.setSpacing(6)
        self.browse_btn = QPushButton("Browse Files…")
        self.browse_btn.setToolTip("Select individual .srt files")
        self.browse_btn.clicked.connect(self._on_browse_files)
        self.browse_folder_btn = QPushButton("Browse Folder…")
        self.browse_folder_btn.setToolTip("Scan a folder for all .srt files")
        self.browse_folder_btn.clicked.connect(self._on_browse_folder)
        self.clear_btn = QPushButton("Clear Queue")
        self.clear_btn.setToolTip("Remove all files from the queue")
        self.clear_btn.clicked.connect(self._on_clear_queue)
        browse_row.addWidget(self.browse_btn)
        browse_row.addWidget(self.browse_folder_btn)
        browse_row.addWidget(self.clear_btn)
        browse_row.addStretch()
        input_lay.addLayout(browse_row)
        io_row.addWidget(input_grp, 3)

        # Output
        out_grp = QGroupBox("Output Folder")
        out_lay = QHBoxLayout(out_grp)
        out_lay.setSpacing(6)
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("Same as input (default)")
        self.out_path.setReadOnly(True)
        out_lay.addWidget(self.out_path)
        self.out_btn = QPushButton("Change…")
        self.out_btn.setToolTip("Choose a different output folder")
        self.out_btn.clicked.connect(self._on_change_output)
        out_lay.addWidget(self.out_btn)
        io_row.addWidget(out_grp, 1)

        outer.addLayout(io_row)

        # === File Queue Table ===
        queue_grp = QGroupBox("File Queue")
        queue_lay = QVBoxLayout(queue_grp)
        queue_lay.setSpacing(4)
        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["File", "Size", "Progress", "Status"])
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.queue_table.setColumnWidth(1, 70)
        self.queue_table.setColumnWidth(3, 120)
        self.queue_table.setItemDelegateForColumn(0, ElideDelegate(self.queue_table))
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.verticalHeader().setDefaultSectionSize(26)
        queue_lay.addWidget(self.queue_table)
        outer.addWidget(queue_grp, 3)

        # === Settings row ===
        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)

        # --- Mode ---
        mode_grp = QGroupBox("Mode")
        mode_lay = QHBoxLayout(mode_grp)
        mode_lay.setSpacing(8)

        self.mode_group = QButtonGroup(self)

        def make_mode_radio(text, desc, button_id, recommended=False):
            vlay = QVBoxLayout()
            vlay.setSpacing(0)
            rb = QRadioButton(text)
            rb.setToolTip(desc)
            if recommended:
                rb.setStyleSheet("QRadioButton { color: #4a9eff; font-weight: bold; }")
            self.mode_group.addButton(rb, button_id)
            dl = QLabel(desc)
            dl.setStyleSheet("color: #888; font-size: 10px; padding-left: 20px;")
            if recommended:
                badge = QLabel("★ Recommended")
                badge.setStyleSheet("color: #4a9eff; font-size: 9px; padding-left: 20px; font-weight: bold;")
                vlay.addWidget(rb)
                vlay.addWidget(badge)
            else:
                vlay.addWidget(rb)
            vlay.addWidget(dl)
            return rb, vlay

        mode_widgets = QWidget()
        mode_w = QHBoxLayout(mode_widgets)
        mode_w.setSpacing(12)
        mode_w.setContentsMargins(0, 0, 0, 0)

        rb_fast, lay_fast = make_mode_radio("Fast", "Fastest translation", 0)
        self.mode_fast = rb_fast
        mode_w.addLayout(lay_fast)

        rb_full, lay_full = make_mode_radio("Full", "Best quality", 2, recommended=True)
        self.mode_full = rb_full
        mode_w.addLayout(lay_full)

        rb_polish, lay_polish = make_mode_radio("Polish", "Improve existing translation", 1)
        self.mode_polish = rb_polish
        mode_w.addLayout(lay_polish)

        rb_reg, lay_reg = make_mode_radio("Regression", "Developer testing", 3)
        self.mode_regression = rb_reg
        mode_w.addLayout(lay_reg)

        mode_w.addStretch()
        mode_lay.addWidget(mode_widgets)
        settings_row.addWidget(mode_grp, 3)

        # --- GPU ---
        gpu_grp = QGroupBox("GPU")
        gpu_lay = QGridLayout(gpu_grp)
        gpu_lay.setSpacing(6)
        gpu_lay.setContentsMargins(8, 16, 8, 8)

        gpu_lay.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("GPU (CUDA) or CPU")
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setMinimumWidth(80)
        gpu_lay.addWidget(self.device_combo, 0, 1)

        gpu_lay.addWidget(QLabel("Batch:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(64)
        self.batch_spin.setSuffix(" lines")
        self.batch_spin.setToolTip("Lines per NLLB batch — larger = faster, uses more VRAM")
        self.batch_spin.setMinimumWidth(90)
        gpu_lay.addWidget(self.batch_spin, 0, 3)

        gpu_lay.addWidget(QLabel("Beams:"), 0, 4)
        self.beams_spin = QSpinBox()
        self.beams_spin.setRange(1, 8)
        self.beams_spin.setValue(4)
        self.beams_spin.setToolTip("NLLB beam search width — higher = better quality but slower")
        self.beams_spin.setMinimumWidth(60)
        gpu_lay.addWidget(self.beams_spin, 0, 5)

        self.resume_cb = QCheckBox("Resume checkpoints")
        self.resume_cb.setToolTip("Resume interrupted translations from saved checkpoint")
        gpu_lay.addWidget(self.resume_cb, 1, 0, 1, 6)

        gpu_lay.setColumnStretch(6, 1)
        settings_row.addWidget(gpu_grp, 2)
        settings_row.addStretch()

        outer.addLayout(settings_row)

        # === Progress Section ===
        prog_grp = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_grp)
        prog_lay.setSpacing(4)
        prog_lay.setContentsMargins(8, 12, 8, 8)

        # File + Stage row
        top_prog = QHBoxLayout()
        self.cur_file_label = QLabel("No file selected")
        self.cur_file_label.setObjectName("curFileLabel")
        top_prog.addWidget(self.cur_file_label)
        top_prog.addStretch()
        self.steps_label = QLabel("Waiting…")
        self.steps_label.setObjectName("stepLabel")
        top_prog.addWidget(self.steps_label)
        prog_lay.addLayout(top_prog)

        # Progress bar (full width, percentage in bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(22)
        self.progress_bar.setFormat("%p%")
        prog_lay.addWidget(self.progress_bar)

        # Stats row — Lines left, Speed + ETA right
        stats_row = QHBoxLayout()
        self.lines_label = QLabel("Lines: 0 / 0")
        self.lines_label.setStyleSheet("min-width: 120px;")
        stats_row.addWidget(self.lines_label)
        stats_row.addStretch()
        self.speed_label = QLabel("Speed: --")
        self.speed_label.setStyleSheet("min-width: 100px;")
        self.eta_label = QLabel("ETA: --")
        self.eta_label.setStyleSheet("min-width: 80px;")
        stats_row.addWidget(self.speed_label)
        stats_row.addWidget(self.eta_label)
        prog_lay.addLayout(stats_row)

        # Live subtitle preview — fixed heights, no jumping
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.NoFrame)
        preview_lay = QVBoxLayout(preview_frame)
        preview_lay.setSpacing(2)
        preview_lay.setContentsMargins(0, 0, 0, 0)

        # EN header + content
        en_header = QLabel("  English")
        en_header.setStyleSheet("font-size: 10px; font-weight: bold; color: #666; padding: 2px 0;")
        preview_lay.addWidget(en_header)
        en_sep = QFrame()
        en_sep.setFrameShape(QFrame.HLine)
        en_sep.setStyleSheet("color: #333;")
        preview_lay.addWidget(en_sep)

        self.sub_en_view = QTextEdit()
        self.sub_en_view.setReadOnly(True)
        self.sub_en_view.setObjectName("subPreviewEn")
        self.sub_en_view.setMinimumHeight(58)
        self.sub_en_view.setMaximumHeight(90)
        self.sub_en_view.setPlaceholderText("Captain!\nRetreat immediately!")
        preview_lay.addWidget(self.sub_en_view)

        # DE header + content
        de_header = QLabel("  German")
        de_header.setStyleSheet("font-size: 10px; font-weight: bold; color: #666; padding: 2px 0;")
        preview_lay.addWidget(de_header)
        de_sep = QFrame()
        de_sep.setFrameShape(QFrame.HLine)
        de_sep.setStyleSheet("color: #333;")
        preview_lay.addWidget(de_sep)

        self.sub_de_view = QTextEdit()
        self.sub_de_view.setReadOnly(True)
        self.sub_de_view.setObjectName("subPreviewDe")
        self.sub_de_view.setMinimumHeight(58)
        self.sub_de_view.setMaximumHeight(90)
        self.sub_de_view.setPlaceholderText("Hauptmann!\nSofort zurückziehen!")
        preview_lay.addWidget(self.sub_de_view)

        prog_lay.addWidget(preview_frame)

        outer.addWidget(prog_grp, 1)

        # === Log Section ===
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setSpacing(4)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMinimumHeight(110)
        self.log_view.setMouseTracking(True)
        self.log_view.cursorPositionChanged.connect(self._on_log_cursor_moved)
        log_font = QFont("Consolas", 10)
        log_font.setStyleHint(QFont.Monospace)
        self.log_view.setFont(log_font)
        self.log_view.installEventFilter(self)
        # Detect scroll bar to pause/resume auto-scroll
        self.log_view.verticalScrollBar().valueChanged.connect(self._on_log_scrolled)
        log_lay.addWidget(self.log_view)

        log_btn_row = QHBoxLayout()
        log_btn_row.setSpacing(6)
        self.save_log_btn = QPushButton("Save Log…")
        self.save_log_btn.setToolTip("Save log contents to a text file")
        self.save_log_btn.clicked.connect(self._on_save_log)
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setToolTip("Clear the log display")
        self.clear_log_btn.clicked.connect(self.log_view.clear)
        log_btn_row.addWidget(self.save_log_btn)
        log_btn_row.addWidget(self.clear_log_btn)
        log_btn_row.addStretch()
        log_lay.addLayout(log_btn_row)

        outer.addWidget(log_grp, 0)

        # === Bottom Buttons ===
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("START TRANSLATION")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setToolTip("Start translating queued files")
        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.setToolTip("Stop the current translation")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        self.open_out_btn = QPushButton("Open Output Folder")
        self.open_out_btn.setToolTip("Open the output folder in file explorer")
        self.open_out_btn.clicked.connect(self._on_open_output)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_out_btn)
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Slots: file handling
    # ------------------------------------------------------------------

    def _on_files_dropped(self, files: list[Path]):
        for f in files:
            if f not in self.file_queue:
                self.file_queue.append(f)
        self._update_queue_table()

    def _on_browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select SRT files", "",
            "SRT files (*.srt);;All files (*)"
        )
        for fp in files:
            p = Path(fp)
            if p not in self.file_queue:
                self.file_queue.append(p)
        self._update_queue_table()

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with SRT files")
        if folder:
            files = find_srt_files(folder)
            for f in files:
                if f not in self.file_queue:
                    self.file_queue.append(f)
            self._update_queue_table()
            self.log(f"Added {len(files)} files from {folder}")

    def _on_clear_queue(self):
        self.file_queue.clear()
        self._update_queue_table()
        self.results.clear()

    def _update_queue_table(self):
        self.queue_table.setRowCount(len(self.file_queue))
        for i, p in enumerate(self.file_queue):
            name_item = QTableWidgetItem(p.name)
            name_item.setToolTip(str(p))
            self.queue_table.setItem(i, 0, name_item)

            size = p.stat().st_size if p.exists() else 0
            size_str = f"{size:,} bytes" if size < 1024 else f"{size // 1024} KB"
            self.queue_table.setItem(i, 1, QTableWidgetItem(size_str))

            self._set_row_status(i, FileStatus.QUEUED)

    def _set_row_status(self, row: int, status: FileStatus, pct: int = 0):
        """Set the progress bar + status icon for a queue row."""
        if status == FileStatus.QUEUED:
            self._set_waiting_row(row, "○", "Waiting")
        elif status == FileStatus.RUNNING:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(pct)
            bar.setTextVisible(True)
            bar.setFormat(f"{pct}%")
            self.queue_table.setCellWidget(row, 2, bar)
            item = QTableWidgetItem("▶ Translating")
            item.setForeground(QColor("#4a9eff"))
            self.queue_table.setItem(row, 3, item)
        elif status == FileStatus.DONE:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(100)
            bar.setTextVisible(False)
            self.queue_table.setCellWidget(row, 2, bar)
            item = QTableWidgetItem("✓ Finished")
            item.setForeground(QColor("#51cf66"))
            self.queue_table.setItem(row, 3, item)
        elif status == FileStatus.SKIPPED:
            self._set_waiting_row(row, "⏭", "Skipped")
        elif status == FileStatus.ERROR:
            self._set_waiting_row(row, "✖", "Failed")
            item = self.queue_table.item(row, 3)
            if item:
                item.setForeground(QColor("#ff6b6b"))

    def _set_waiting_row(self, row, icon, text):
        self.queue_table.removeCellWidget(row, 2)
        prog_item = QTableWidgetItem(f"{icon}  {text}")
        prog_item.setTextAlignment(Qt.AlignCenter)
        self.queue_table.setItem(row, 2, prog_item)
        item = QTableWidgetItem(f"{icon} {text}")
        self.queue_table.setItem(row, 3, item)

    def _on_change_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.out_path.setText(folder)

    def _on_open_output(self):
        folder = self.out_path.text().strip()
        if not folder or not Path(folder).exists():
            if self.file_queue:
                folder = str(self.file_queue[0].parent)
            else:
                folder = os.getcwd()
        os.startfile(folder)

    # ------------------------------------------------------------------
    # Slots: start / cancel
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._running:
            return

        if not self.file_queue:
            QMessageBox.information(self, "No Files", "Add .srt files to the queue first.")
            return

        # Determine mode
        mode_id = self.mode_group.checkedId()
        mode_map = {0: "fast", 1: "polish", 2: "full", 3: "regression"}
        mode = mode_map.get(mode_id, "fast")

        # Auto-detect Ollama for Full/Polish mode (informational only — does not change mode)
        if mode in ("full", "polish"):
            try:
                import requests
                tags_url = f"{self.ollama_host}/api/tags"
                r = requests.get(tags_url, timeout=5)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                else:
                    models = []
                if self.polish_model not in models:
                    if models:
                        preferred = ["qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b",
                                     "mistral:7b", "llama3.1:8b", "gemma2:9b"]
                        chosen = next((m for p in preferred if p in models), models[0])
                        self.log(f"  [Ollama] Using '{chosen}'")
                        self.polish_model = chosen
                        self.settings.setValue("ollama/model", chosen)
                    else:
                        self.log("  [Ollama] No models installed — install one: ollama pull qwen2.5:7b")
                self.stat_ollama.setText(f"Ollama: {self.polish_model.split(':')[0]}")
            except requests.ConnectionError:
                self.log("  [Ollama] Not running — Full mode will skip polish (start with: ollama serve)")
                self.stat_ollama.setText("Ollama: ✗")
            except Exception as e:
                self.log(f"  [Ollama] Error: {e}")

        # Build config
        device = self.device_combo.currentText()
        batch_size = self.batch_spin.value()
        num_beams = self.beams_spin.value()
        output_dir = self.out_path.text().strip() or None
        resume = self.resume_cb.isChecked()

        cfg = Config(
            device=_auto_device(device),
            batch_size=batch_size,
            num_beams=num_beams,
            force=True,
        )

        # Setup worker
        self._running = True
        self._start_time = time.time()
        self.results.clear()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.lines_label.setText("Lines: 0 / 0")
        self.eta_label.setText("ETA: --")
        self.speed_label.setText("Speed: --")
        self.steps_label.setText("Starting…")
        self.sub_en_view.clear()
        self.sub_de_view.clear()
        self.cur_file_label.setText("Preparing…")

        # Update status bar
        self.stat_elapsed.setText("Elapsed: 00:00")
        self.status_bar.showMessage("Translating…")

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.browse_folder_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.out_btn.setEnabled(False)
        self.mode_fast.setEnabled(False)
        self.mode_polish.setEnabled(False)
        self.mode_full.setEnabled(False)
        self.mode_regression.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.batch_spin.setEnabled(False)
        self.beams_spin.setEnabled(False)
        self.resume_cb.setEnabled(False)

        # Elapsed timer
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_status_elapsed)
        self._elapsed_timer.start(1000)

        # Redirect stdout
        self._old_stdout = sys.stdout
        self.log_capture = LogCapture(self.log_signal)
        sys.stdout = self.log_capture

        self.thread = QThread()
        self.worker = TranslationWorker(
            files=list(self.file_queue),
            cfg=cfg,
            output_dir=output_dir,
            mode=mode,
            polish_model=self.polish_model,
            ollama_host=self.ollama_host,
            resume=resume,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.log.connect(self._on_log)
        self.worker.current_file.connect(self._on_current_file)
        self.worker.current_en.connect(self._on_current_en)
        self.worker.current_de.connect(self._on_current_de)
        self.worker.step_changed.connect(self._on_step_changed)
        self.worker.speed_eta.connect(self._on_speed_eta)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.error_happened.connect(self._on_worker_error)

        self.thread.finished.connect(self._on_thread_finished)

        self.thread.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
            self.log("Cancelling… (will stop after current batch)")
        self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Slots: worker signals
    # ------------------------------------------------------------------

    def _on_progress(self, done: int, total: int):
        pct = int(done / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lines_label.setText(f"Lines: {done} / {total}")
        # Update running row progress bar
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 3)
            if item and item.text() == "▶ Running":
                self.queue_table.removeCellWidget(row, 2)
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(pct)
                bar.setTextVisible(True)
                bar.setFormat(f"{pct}%")
                self.queue_table.setCellWidget(row, 2, bar)
                break

    def _on_file_progress(self, idx: int, total: int):
        # Mark previous file done
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 3)
            if item and "▶" in item.text():
                self._set_row_status(row, FileStatus.DONE)
        # Mark current as running
        if idx <= self.queue_table.rowCount():
            self._set_row_status(idx - 1, FileStatus.RUNNING, 0)
        self._file_idx = idx
        self._file_total = total

    def _on_log(self, msg: str):
        fmt = QTextCharFormat()
        msg_lower = msg.lower()
        # Order matters: check most specific first
        if "[error]" in msg_lower or msg.startswith("Error "):
            fmt.setForeground(QColor("#ff6b6b"))
            fmt.setFontWeight(QFont.Bold)
        elif "[warn]" in msg_lower or "warning" in msg_lower:
            fmt.setForeground(QColor("#ffd93d"))
        elif "[skip]" in msg_lower:
            fmt.setForeground(QColor("#6bcbff"))
        elif "[resume]" in msg_lower:
            fmt.setForeground(QColor("#6bcbff"))
        elif "done" in msg_lower:
            fmt.setForeground(QColor("#51cf66"))
        elif "summary" in msg_lower:
            fmt.setForeground(QColor("#da77f2"))
            fmt.setFontWeight(QFont.Bold)
        else:
            # Default blue for INFO
            fmt.setForeground(QColor("#6bcbff"))

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(msg + "\n", fmt)

        if self._log_auto_scroll:
            self.log_view.setTextCursor(cursor)
            self.log_view.ensureCursorVisible()

    def _on_log_scrolled(self, value):
        sb = self.log_view.verticalScrollBar()
        self._log_auto_scroll = (value >= sb.maximum() - 10)

    def _on_current_file(self, fname: str):
        ep = ""
        if hasattr(self, '_file_idx') and hasattr(self, '_file_total'):
            ep = f"Episode {self._file_idx} of {self._file_total}\n"
        self.cur_file_label.setText(f"{ep}{fname}")

    def _on_current_en(self, text: str):
        self.sub_en_view.setPlainText(text)

    def _on_current_de(self, text: str):
        self.sub_de_view.setPlainText(text)

    def _on_step_changed(self, step: str):
        self.steps_label.setText(step)
        if "ollama" in step.lower():
            self.stat_ollama.setText("Ollama: Active")
        elif "NLLB" in step:
            self.stat_ollama.setText("Ollama: —")
        elif "Complete" in step:
            self.stat_ollama.setText("Ollama: —")

    def _on_speed_eta(self, speed: float, eta: float):
        if speed > 0:
            self.speed_label.setText(f"Speed: {speed:.0f} l/s")
        else:
            self.speed_label.setText("Speed: --")
        if eta > 0:
            m, s = divmod(int(eta), 60)
            self.eta_label.setText(f"ETA: {m:02d}:{s:02d}")
        else:
            self.eta_label.setText("ETA: --")

    def _on_file_done(self, fname: str, stats: dict):
        self.results.append((fname, stats))
        # Update queue table row
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 0)
            if item and item.text() == fname:
                if stats.get("error"):
                    self._set_row_status(row, FileStatus.ERROR)
                elif stats.get("skipped"):
                    self._set_row_status(row, FileStatus.SKIPPED)
                else:
                    self._set_row_status(row, FileStatus.DONE)
                    parts = []
                    if "qa_score" in stats:
                        parts.append(f"QA: {stats['qa_score']}")
                    if "lines" in stats:
                        parts.append(f"{stats['lines']} lines")
                    if "suspicious" in stats and stats["suspicious"]:
                        parts.append(f"{stats['suspicious']} susp.")
                    self.queue_table.item(row, 3).setText(" | ".join(parts) if parts else "Finished")
                break

    def _on_all_done(self):
        if hasattr(self, '_old_stdout') and self._old_stdout:
            sys.stdout = self._old_stdout

        if hasattr(self, '_elapsed_timer'):
            self._elapsed_timer.stop()

        self.log("All files processed.")
        self.steps_label.setText("Complete")

        if self.results:
            dlg = SummaryDialog(self.results, self)
            dlg.exec()

        self._finish()

    def _on_worker_error(self, msg: str):
        self.log(f"[ERROR] {msg}")

    def _on_thread_finished(self):
        self._finish()

    def _finish(self):
        self._running = False
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.browse_folder_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.out_btn.setEnabled(True)
        self.mode_fast.setEnabled(True)
        self.mode_polish.setEnabled(True)
        self.mode_full.setEnabled(True)
        self.mode_regression.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.batch_spin.setEnabled(True)
        self.beams_spin.setEnabled(True)
        self.resume_cb.setEnabled(True)
        if hasattr(self, '_old_stdout') and self._old_stdout:
            sys.stdout = self._old_stdout

        if hasattr(self, '_elapsed_timer'):
            self._elapsed_timer.stop()

        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None

        self.status_bar.showMessage("Ready")

    def log(self, msg: str):
        self._on_log(msg)

    # ------------------------------------------------------------------
    # Log save
    # ------------------------------------------------------------------

    def _toggle_theme(self, checked: bool):
        self._dark_mode = checked
        self.setStyleSheet(DARK_STYLE if checked else LIGHT_STYLE)

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.device_combo.setCurrentIndex(dlg.device_combo.currentIndex())
            self.batch_spin.setValue(dlg.batch_spin.value())
            self.resume_cb.setChecked(dlg.resume_cb.isChecked())
            self.polish_model = dlg.ollama_model.text().strip() or "subtitle-translator"
            self.ollama_host = dlg.ollama_host.text().strip() or "http://127.0.0.1:11434"
            self.settings.setValue("ollama/model", self.polish_model)
            self.settings.setValue("ollama/host", self.ollama_host)

    def eventFilter(self, obj, event):
        if obj is self.log_view and event.type() == event.Type.MouseButtonDblClick:
            cursor = self.log_view.cursorForPosition(event.pos())
            cursor.select(QTextCursor.LineUnderCursor)
            line = cursor.selectedText().strip()
            if line:
                QMessageBox.information(self, "Log Entry", line, QMessageBox.Ok)
            return True
        return super().eventFilter(obj, event)

    def _on_log_cursor_moved(self):
        pass

    def _on_save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "translation_log.txt",
            "Text files (*.txt);;All files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_view.toPlainText())
                self.log(f"Log saved to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save log:\n{e}")

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _restore_settings(self):
        self.restoreGeometry(self.settings.value("window/geometry", b""))
        self.restoreState(self.settings.value("window/state", b""))
        dev = self.settings.value("settings/device", "cuda")
        idx = self.device_combo.findText(dev)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.batch_spin.setValue(int(self.settings.value("settings/batch_size", 64)))
        self.beams_spin.setValue(int(self.settings.value("settings/num_beams", 4)))
        out = self.settings.value("settings/output_dir", "")
        if out:
            self.out_path.setText(out)
        mode = self.settings.value("settings/mode", "fast")
        mode_map = {"fast": 0, "polish": 1, "full": 2, "regression": 3}
        if mode in mode_map:
            btn = self.mode_group.button(mode_map[mode])
            if btn:
                btn.setChecked(True)
        resume = self.settings.value("settings/resume", "false")
        self.resume_cb.setChecked(resume.lower() == "true")
        dark = self.settings.value("settings/dark_mode", "true")
        self._dark_mode = dark.lower() != "false"
        self.dark_mode_action.setChecked(self._dark_mode)

        # Restore file queue from last session
        queue_str = self.settings.value("queue/files", "")
        if queue_str:
            for p in queue_str.split("|"):
                path = Path(p)
                if path.exists():
                    self.file_queue.append(path)
            self._update_queue_table()

        # Update CUDA status
        if torch_is_available():
            self.stat_cuda.setText("CUDA ✓")
        else:
            self.stat_cuda.setText("CPU")

    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("settings/device", self.device_combo.currentText())
        self.settings.setValue("settings/batch_size", self.batch_spin.value())
        self.settings.setValue("settings/num_beams", self.beams_spin.value())
        self.settings.setValue("settings/output_dir", self.out_path.text())
        mode_map = {0: "fast", 1: "polish", 2: "full", 3: "regression"}
        mode = mode_map.get(self.mode_group.checkedId(), "fast")
        self.settings.setValue("settings/mode", mode)
        self.settings.setValue("settings/resume", "true" if self.resume_cb.isChecked() else "false")
        self.settings.setValue("queue/files", "|".join(str(p) for p in self.file_queue))

        # Restore stdout if still captured
        if hasattr(self, '_old_stdout') and self._old_stdout:
            sys.stdout = self._old_stdout

        self.settings.setValue("settings/dark_mode", "true" if self._dark_mode else "false")

        # Clean up thread
        if self._running and self.worker:
            self.worker.cancel()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)

        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helper: torch availability check
# ---------------------------------------------------------------------------

def torch_is_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TranslatorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
