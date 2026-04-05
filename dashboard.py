"""
dashboard.py
------------
AI-Powered HCA — Staff & Patient Desktop Dashboard
Built with PyQt6. Run with: python dashboard.py
"""

import sys
import os
import subprocess
import datetime
import sqlite3

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QStackedWidget,
    QFrame, QScrollArea, QSizePolicy, QHeaderView, QDialog,
    QLineEdit, QFormLayout, QMessageBox, QGraphicsDropShadowEffect,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QSize, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QPainter, QBrush,
    QLinearGradient, QIcon, QPen, QFontDatabase
)

# ── Database path ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "hca.db")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
TASK_SCRIPT = os.path.join(os.path.dirname(__file__), "task.py")

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":           "#0A0E1A",   # deep navy background
    "panel":        "#111827",   # card background
    "panel2":       "#1A2235",   # elevated card
    "border":       "#1E2D45",   # subtle borders
    "accent":       "#00C2FF",   # cyan accent
    "accent2":      "#0077B6",   # deeper blue
    "green":        "#00E096",   # success / correct
    "amber":        "#FFB800",   # warning
    "red":          "#FF4757",   # danger / emergency
    "text":         "#E8F0FE",   # primary text
    "text2":        "#7B8FA6",   # secondary text
    "text3":        "#3D5066",   # muted text
    "white":        "#FFFFFF",
}


def db_query(sql, params=()):
    """Run a SELECT query and return list of dicts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Query error: {e}")
        return []


def shadow(widget, radius=20, offset=4, color="#000000", alpha=120):
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(radius)
    fx.setOffset(offset, offset)
    c = QColor(color)
    c.setAlpha(alpha)
    fx.setColor(c)
    widget.setGraphicsEffect(fx)
    return fx


# ── Reusable card widget ───────────────────────────────────────────────────────
class Card(QFrame):
    def __init__(self, parent=None, elevated=False):
        super().__init__(parent)
        bg = C["panel2"] if elevated else C["panel"]
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 1px solid {C['border']};
                border-radius: 12px;
            }}
        """)
        shadow(self, radius=16, alpha=80)


# ── Stat card ──────────────────────────────────────────────────────────────────
class StatCard(Card):
    def __init__(self, title, value, subtitle="", accent=C["accent"], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(4)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px; font-weight:600; "
                          f"letter-spacing:1px; background:transparent; border:none;")
        top.addWidget(lbl)
        top.addStretch()

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{accent}; font-size:8px; background:transparent; border:none;")
        top.addWidget(dot)
        lay.addLayout(top)

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setStyleSheet(
            f"color:{C['white']}; font-size:32px; font-weight:700; "
            f"background:transparent; border:none;"
        )
        lay.addWidget(self.val_lbl)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color:{C['text3']}; font-size:10px; "
                              f"background:transparent; border:none;")
            lay.addWidget(sub)

    def set_value(self, v):
        self.val_lbl.setText(str(v))


# ── Section header ─────────────────────────────────────────────────────────────
def section_header(text):
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color:{C['text2']}; font-size:10px; font-weight:700; "
        f"letter-spacing:2px; padding:0 4px;"
    )
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"color:{C['border']}; border:none; "
                       f"border-top:1px solid {C['border']};")
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 8, 0, 4)
    lay.setSpacing(4)
    lay.addWidget(lbl)
    lay.addWidget(line)
    return w


# ── Emergency banner ───────────────────────────────────────────────────────────
class EmergencyBanner(QFrame):
    resolved = pyqtSignal(int)  # emits emergency_id when resolved

    def __init__(self, emergency_id, patient_name, timestamp, trigger_phrase="", parent=None):
        super().__init__(parent)
        self._emergency_id = emergency_id
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #3D0A0A, stop:1 #1A0505);
                border: 1px solid {C['red']};
                border-left: 4px solid {C['red']};
                border-radius: 8px;
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)

        icon = QLabel("🚨")
        icon.setStyleSheet("font-size:20px; background:transparent; border:none;")
        lay.addWidget(icon)

        info = QVBoxLayout()
        name_lbl = QLabel(f"EMERGENCY — {patient_name.upper()}")
        name_lbl.setStyleSheet(
            f"color:{C['red']}; font-size:13px; font-weight:700; "
            f"background:transparent; border:none;"
        )
        time_lbl = QLabel(f"Triggered at {timestamp}"
                          + (f"  ·  \"{trigger_phrase}\"" if trigger_phrase else ""))
        time_lbl.setStyleSheet(
            f"color:{C['text2']}; font-size:10px; background:transparent; border:none;"
        )
        info.addWidget(name_lbl)
        info.addWidget(time_lbl)
        lay.addLayout(info)
        lay.addStretch()

        resolve_btn = QPushButton("✔  Mark Resolved")
        resolve_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['red']}; color: white;
                border: none; border-radius: 6px;
                padding: 6px 16px; font-size:11px; font-weight:600;
            }}
            QPushButton:hover {{ background: #cc2233; }}
            QPushButton:pressed {{ background: #991122; }}
        """)
        resolve_btn.clicked.connect(self._resolve)
        lay.addWidget(resolve_btn)
        shadow(self, radius=12, color=C["red"], alpha=60)

    def _resolve(self):
        """Mark this emergency as resolved in the DB and remove the banner."""
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "UPDATE emergencies SET resolved=1, resolved_at=? WHERE id=?",
                (now, self._emergency_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not resolve emergency:\n{e}")
            return
        self.resolved.emit(self._emergency_id)
        self.setVisible(False)
        self.deleteLater()


# ── Patients table ─────────────────────────────────────────────────────────────
class PatientsTable(QTableWidget):
    patient_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ["Name", "Ward / Room", "Sessions", "Last Score", "Last Seen"]
        )
        self._style()
        self.cellDoubleClicked.connect(self._on_click)
        self._data = []

    def _style(self):
        self.setStyleSheet(f"""
            QTableWidget {{
                background: transparent;
                border: none;
                color: {C['text']};
                font-size: 13px;
                gridline-color: {C['border']};
                selection-background-color: {C['accent2']};
            }}
            QHeaderView::section {{
                background: {C['panel2']};
                color: {C['text2']};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 8px;
                border: none;
                border-bottom: 1px solid {C['border']};
            }}
            QTableWidget::item {{
                padding: 10px 8px;
                border-bottom: 1px solid {C['border']};
            }}
            QTableWidget::item:selected {{
                background: {C['accent2']};
                color: white;
            }}
        """)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def load(self):
        patients = db_query("SELECT * FROM patients ORDER BY updated_at DESC")
        self._data = []
        self.setRowCount(0)

        for p in patients:
            sessions = db_query(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY started_at DESC",
                (p["id"],)
            )
            last = sessions[0] if sessions else None
            score_str = (f"{last['total_score']}/{last['max_score']}"
                         if last else "—")
            last_seen = last["started_at"][:10] if last else "—"

            row = self.rowCount()
            self.insertRow(row)
            self.setRowHeight(row, 48)

            items = [
                p.get("name", ""),
                p.get("address") or "—",
                str(len(sessions)),
                score_str,
                last_seen,
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(C["text"]))
                if col == 0:
                    item.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
                self.setItem(row, col, item)

            self._data.append(p)

    def _on_click(self, row, _):
        if row < len(self._data):
            self.patient_selected.emit(self._data[row])


# ── Sessions table ─────────────────────────────────────────────────────────────
class SessionsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(
            ["Date", "Task", "Score", "%", "Duration", "Report"]
        )
        self._style()

    def _style(self):
        self.setStyleSheet(f"""
            QTableWidget {{
                background: transparent; border: none;
                color: {C['text']}; font-size: 12px;
                gridline-color: {C['border']};
            }}
            QHeaderView::section {{
                background: {C['panel2']}; color: {C['text2']};
                font-size: 10px; font-weight: 700;
                letter-spacing: 1px; padding: 6px 8px;
                border: none; border-bottom: 1px solid {C['border']};
            }}
            QTableWidget::item {{
                padding: 8px; border-bottom: 1px solid {C['border']};
            }}
        """)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def load_for_patient(self, patient_id):
        sessions = db_query(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY started_at DESC",
            (patient_id,)
        )
        self.setRowCount(0)
        for s in sessions:
            row = self.rowCount()
            self.insertRow(row)
            self.setRowHeight(row, 40)

            score = s.get("total_score", 0)
            maxs  = s.get("max_score", 1) or 1
            pct   = int(score / maxs * 100)

            # Duration
            try:
                fmt = "%Y-%m-%d %H:%M:%S"
                start = datetime.datetime.strptime(s["started_at"], fmt)
                end   = datetime.datetime.strptime(s["ended_at"], fmt)
                dur   = int((end - start).total_seconds())
                dur_str = f"{dur//60}m {dur%60}s"
            except Exception:
                dur_str = "—"

            color = C["green"] if pct >= 75 else (C["amber"] if pct >= 50 else C["red"])

            vals = [
                s.get("started_at", "")[:16],
                s.get("task_name", ""),
                f"{score}/{maxs}",
                f"{pct}%",
                dur_str,
                "📄 View",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 2 or col == 3:
                    item.setForeground(QColor(color))
                    item.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                else:
                    item.setForeground(QColor(C["text"]))
                self.setItem(row, col, item)


# ── Score bar widget ───────────────────────────────────────────────────────────
class MiniScoreBar(QWidget):
    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions[-10:]  # last 10
        self.setMinimumHeight(80)
        self.setStyleSheet("background:transparent;")

    def paintEvent(self, event):
        if not self.sessions:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        n = len(self.sessions)
        bar_w = min(32, (w - (n + 1) * 6) // n)
        gap   = (w - n * bar_w) // (n + 1)

        for i, s in enumerate(self.sessions):
            score = s.get("total_score", 0)
            maxs  = s.get("max_score", 1) or 1
            pct   = score / maxs
            bar_h = int((h - 24) * pct) + 4

            x = gap + i * (bar_w + gap)
            y = h - bar_h - 20

            color = (QColor(C["green"]) if pct >= 0.75
                     else QColor(C["amber"]) if pct >= 0.5
                     else QColor(C["red"]))

            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x, y, bar_w, bar_h, 4, 4)

            # Label
            p.setPen(QPen(QColor(C["text2"])))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(x, h - 4, f"{int(pct*100)}%")

        p.end()


# ── Patient profile dialog ─────────────────────────────────────────────────────
class ProfileDialog(QDialog):
    def __init__(self, patient: dict, parent=None):
        super().__init__(parent)
        self.patient = patient
        self.setWindowTitle(f"Profile — {patient['name']}")
        self.setFixedSize(480, 540)
        self.setStyleSheet(f"""
            QDialog {{ background: {C['panel']}; }}
            QLabel {{ color: {C['text']}; font-size: 12px;
                      background: transparent; border: none; }}
            QLineEdit {{
                background: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 7px 10px; font-size: 12px;
            }}
            QLineEdit:focus {{ border: 1px solid {C['accent']}; }}
            QFormLayout QLabel {{ color: {C['text2']}; font-size: 11px; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 20)
        lay.setSpacing(12)

        title = QLabel(f"Personal Profile — {patient['name']}")
        title.setStyleSheet(
            f"color:{C['white']}; font-size:16px; font-weight:700; "
            f"background:transparent; border:none;"
        )
        lay.addWidget(title)

        sub = QLabel(
            "These details are used by Jennet to ask personalised memory questions "
            "after the orientation task. Leave blank to skip a question."
        )
        sub.setStyleSheet(
            f"color:{C['text2']}; font-size:11px; background:transparent; border:none;"
        )
        sub.setWordWrap(True)
        lay.addWidget(sub)

        # Load existing profile from DB
        existing = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM patient_profiles WHERE patient_id=?",
                (patient["id"],)
            ).fetchone()
            if row:
                existing = dict(row)
            conn.close()
        except Exception:
            pass

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def field(placeholder, key):
            w = QLineEdit()
            w.setPlaceholderText(placeholder)
            w.setText(existing.get(key) or "")
            return w

        self.f_hometown  = field("e.g. Liverpool",         "hometown")
        self.f_spouse    = field("e.g. Margaret",          "spouse_name")
        self.f_children  = field("e.g. John, Sarah",       "children_names")
        self.f_job       = field("e.g. School Teacher",    "occupation")
        self.f_hobbies   = field("e.g. Gardening, Chess",  "hobbies")
        self.f_food      = field("e.g. Fish and Chips",    "favourite_food")
        self.f_sport     = field("e.g. Football",          "favourite_sport")
        self.f_show      = field("e.g. Coronation Street", "favourite_show")

        form.addRow("Hometown:",         self.f_hometown)
        form.addRow("Spouse / Partner:", self.f_spouse)
        form.addRow("Children:",         self.f_children)
        form.addRow("Occupation:",       self.f_job)
        form.addRow("Hobbies:",          self.f_hobbies)
        form.addRow("Favourite Food:",   self.f_food)
        form.addRow("Favourite Sport:",  self.f_sport)
        form.addRow("Favourite Show:",   self.f_show)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(f"""
            QPushButton {{
                background: {C['panel2']}; color: {C['text2']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 8px 20px; font-size: 12px;
            }}
            QPushButton:hover {{ color: {C['white']}; }}
        """)
        cancel.clicked.connect(self.reject)

        save_btn = QPushButton("💾  Save Profile")
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['accent']}; color: {C['bg']};
                border: none; border-radius: 6px;
                padding: 8px 20px; font-size: 12px; font-weight: 700;
            }}
            QPushButton:hover {{ background: #33D1FF; }}
        """)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

    def _save(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vals = (
            self.f_hometown.text().strip()  or None,
            self.f_spouse.text().strip()    or None,
            self.f_children.text().strip()  or None,
            self.f_job.text().strip()       or None,
            self.f_hobbies.text().strip()   or None,
            self.f_food.text().strip()      or None,
            self.f_sport.text().strip()     or None,
            self.f_show.text().strip()      or None,
            now,
        )
        try:
            conn = sqlite3.connect(DB_PATH)
            existing = conn.execute(
                "SELECT id FROM patient_profiles WHERE patient_id=?",
                (self.patient["id"],)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE patient_profiles SET
                        hometown=?, spouse_name=?, children_names=?,
                        occupation=?, hobbies=?, favourite_food=?,
                        favourite_sport=?, favourite_show=?, updated_at=?
                    WHERE patient_id=?
                """, vals + (self.patient["id"],))
            else:
                conn.execute("""
                    INSERT INTO patient_profiles
                    (hometown, spouse_name, children_names, occupation,
                     hobbies, favourite_food, favourite_sport,
                     favourite_show, updated_at, patient_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, vals + (self.patient["id"],))
            conn.commit()
            conn.close()
            QMessageBox.information(
                self, "Saved", f"Profile saved for {self.patient['name']}."
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save profile:\n{e}")


# ── Patient detail panel ───────────────────────────────────────────────────────
class PatientDetailPanel(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C['bg']};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        # Back button
        back_btn = QPushButton("← Back to Patients")
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C['accent']};
                border: none; font-size: 13px; font-weight: 600;
                padding: 4px 0; text-align: left;
            }}
            QPushButton:hover {{ color: {C['white']}; }}
        """)
        back_btn.clicked.connect(self.back_requested.emit)
        lay.addWidget(back_btn)

        # Patient info card
        self.info_card = Card()
        info_lay = QHBoxLayout(self.info_card)
        info_lay.setContentsMargins(24, 20, 24, 20)

        left = QVBoxLayout()
        self.name_lbl = QLabel("—")
        self.name_lbl.setStyleSheet(
            f"color:{C['white']}; font-size:24px; font-weight:700; "
            f"background:transparent; border:none;"
        )
        self.sub_lbl = QLabel("—")
        self.sub_lbl.setStyleSheet(
            f"color:{C['text2']}; font-size:12px; background:transparent; border:none;"
        )
        left.addWidget(self.name_lbl)
        left.addWidget(self.sub_lbl)
        info_lay.addLayout(left)
        info_lay.addStretch()

        # Edit profile button
        self._current_patient = None
        edit_btn = QPushButton("✎  Edit Profile")
        edit_btn.setFixedHeight(36)
        edit_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['panel2']}; color: {C['accent']};
                border: 1px solid {C['accent']}; border-radius: 8px;
                padding: 0 16px; font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{
                background: {C['accent']}; color: {C['bg']};
            }}
        """)
        edit_btn.clicked.connect(self._open_profile_editor)
        info_lay.addWidget(edit_btn)

        # Stat cards row
        self.stat_sessions  = StatCard("SESSIONS", "—", accent=C["accent"])
        self.stat_avg_score = StatCard("AVG SCORE", "—", accent=C["green"])
        self.stat_emergency = StatCard("EMERGENCIES", "—", accent=C["red"])
        for s in [self.stat_sessions, self.stat_avg_score, self.stat_emergency]:
            s.setFixedWidth(140)
            info_lay.addWidget(s)

        lay.addWidget(self.info_card)

        # Score trend
        lay.addWidget(section_header("Score Trend"))
        self.score_bar_card = Card()
        self.score_bar_layout = QVBoxLayout(self.score_bar_card)
        self.score_bar_layout.setContentsMargins(16, 12, 16, 12)
        self.score_bar_widget = None
        lay.addWidget(self.score_bar_card)

        # Sessions
        lay.addWidget(section_header("Session History"))
        self.sessions_table = SessionsTable()
        lay.addWidget(self.sessions_table)

    def load_patient(self, patient: dict):
        self._current_patient = patient
        self.name_lbl.setText(patient.get("name", "—"))
        self.sub_lbl.setText(
            f"Ward: {patient.get('address') or 'Not recorded'}  ·  "
            f"Registered: {patient.get('created_at', '')[:10]}"
        )

        sessions = db_query(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY started_at DESC",
            (patient["id"],)
        )
        emergencies = db_query(
            "SELECT * FROM emergencies WHERE patient_id=?",
            (patient["id"],)
        )

        self.stat_sessions.set_value(len(sessions))
        self.stat_emergency.set_value(len(emergencies))

        if sessions:
            scores = [
                s["total_score"] / (s["max_score"] or 1) * 100
                for s in sessions if s.get("max_score")
            ]
            avg = int(sum(scores) / len(scores)) if scores else 0
            self.stat_avg_score.set_value(f"{avg}%")
        else:
            self.stat_avg_score.set_value("—")

        # Score bar
        if self.score_bar_widget:
            self.score_bar_layout.removeWidget(self.score_bar_widget)
            self.score_bar_widget.deleteLater()
        self.score_bar_widget = MiniScoreBar(list(reversed(sessions)))
        self.score_bar_layout.addWidget(self.score_bar_widget)

        self.sessions_table.load_for_patient(patient["id"])

    def _open_profile_editor(self):
        if self._current_patient:
            dlg = ProfileDialog(self._current_patient, self)
            dlg.exec()


# ── Start session dialog ───────────────────────────────────────────────────────
class StartSessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start New Session")
        self.setFixedSize(400, 220)
        self.setStyleSheet(f"""
            QDialog {{ background: {C['panel']}; }}
            QLabel {{ color: {C['text']}; font-size: 13px; background:transparent; }}
            QLineEdit {{
                background: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 8px 12px; font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {C['accent']}; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        title = QLabel("Start Session")
        title.setStyleSheet(
            f"color:{C['white']}; font-size:18px; font-weight:700; background:transparent;"
        )
        lay.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)
        self.name_input    = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Rami")
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("e.g. Ward 3, Room 5")
        form.addRow("Patient Name:", self.name_input)
        form.addRow("Location:", self.location_input)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(f"""
            QPushButton {{
                background: {C['panel2']}; color: {C['text2']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 8px 20px; font-size: 12px;
            }}
            QPushButton:hover {{ color: {C['white']}; }}
        """)
        cancel.clicked.connect(self.reject)

        start = QPushButton("▶  Start")
        start.setStyleSheet(f"""
            QPushButton {{
                background: {C['accent']}; color: {C['bg']};
                border: none; border-radius: 6px;
                padding: 8px 20px; font-size: 12px; font-weight: 700;
            }}
            QPushButton:hover {{ background: #33D1FF; }}
        """)
        start.clicked.connect(self.accept)
        btn_row.addWidget(cancel)
        btn_row.addStretch()
        btn_row.addWidget(start)
        lay.addLayout(btn_row)

    def get_values(self):
        return self.name_input.text().strip(), self.location_input.text().strip()


# ── Sidebar navigation ─────────────────────────────────────────────────────────
class SidebarBtn(QPushButton):
    def __init__(self, icon_text, label, parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_text}  {label}")
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C['text2']};
                border: none;
                border-radius: 8px;
                text-align: left;
                padding: 0 12px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {C['panel2']};
                color: {C['text']};
            }}
            QPushButton:checked {{
                background: {C['accent2']};
                color: {C['white']};
                font-weight: 700;
            }}
        """)


# ── Live session indicator ─────────────────────────────────────────────────────
class LiveIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(8)
        self.setStyleSheet(f"""
            background: #0D2A1A;
            border: 1px solid {C['green']};
            border-radius: 8px;
        """)

        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color:{C['green']}; font-size:10px; background:transparent; border:none;")
        self.lbl = QLabel("No active session")
        self.lbl.setStyleSheet(f"color:{C['green']}; font-size:11px; font-weight:600; background:transparent; border:none;")
        lay.addWidget(self.dot)
        lay.addWidget(self.lbl)
        lay.addStretch()

        # Blink timer
        self._visible = True
        self._timer = QTimer()
        self._timer.timeout.connect(self._blink)
        self._timer.start(800)

    def _blink(self):
        self._visible = not self._visible
        self.dot.setStyleSheet(
            f"color:{'transparent' if not self._visible else C['green']}; "
            f"font-size:10px; background:transparent; border:none;"
        )

    def set_active(self, patient_name):
        self.lbl.setText(f"Live: {patient_name}")

    def set_idle(self):
        self.lbl.setText("No active session")


# ── Dashboard page ─────────────────────────────────────────────────────────────
class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C['bg']};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.stat_patients  = StatCard("TOTAL PATIENTS",  "—", "registered", C["accent"])
        self.stat_sessions  = StatCard("TOTAL SESSIONS",  "—", "all time",   C["green"])
        self.stat_today     = StatCard("SESSIONS TODAY",  "—", "today",      C["amber"])
        self.stat_emergency = StatCard("EMERGENCIES",     "—", "unresolved", C["red"])
        for s in [self.stat_patients, self.stat_sessions,
                  self.stat_today, self.stat_emergency]:
            stats_row.addWidget(s)
        lay.addLayout(stats_row)

        # Emergencies section
        lay.addWidget(section_header("Active Emergencies"))
        self.emergency_scroll = QScrollArea()
        self.emergency_scroll.setWidgetResizable(True)
        self.emergency_scroll.setMaximumHeight(180)
        self.emergency_scroll.setStyleSheet(
            f"QScrollArea {{ background:transparent; border:none; }}"
        )
        self.emergency_container = QWidget()
        self.emergency_container.setStyleSheet("background:transparent;")
        self.emergency_layout = QVBoxLayout(self.emergency_container)
        self.emergency_layout.setContentsMargins(0, 0, 0, 0)
        self.emergency_layout.setSpacing(8)
        self.emergency_scroll.setWidget(self.emergency_container)
        lay.addWidget(self.emergency_scroll)

        # Recent sessions
        lay.addWidget(section_header("Recent Sessions"))
        self.recent_table = SessionsTable()
        lay.addWidget(self.recent_table)

        self.refresh()

    def _on_emergency_resolved(self, emergency_id):
        """Refresh stats and banner list after an emergency is resolved."""
        self.refresh()

    def refresh(self):
        today = datetime.date.today().isoformat()

        n_patients  = db_query("SELECT COUNT(*) as n FROM patients")[0]["n"]
        n_sessions  = db_query("SELECT COUNT(*) as n FROM sessions")[0]["n"]
        n_today     = db_query(
            "SELECT COUNT(*) as n FROM sessions WHERE started_at LIKE ?",
            (f"{today}%",)
        )[0]["n"]
        n_emergency = db_query(
            "SELECT COUNT(*) as n FROM emergencies WHERE resolved=0"
        )[0]["n"]

        self.stat_patients.set_value(n_patients)
        self.stat_sessions.set_value(n_sessions)
        self.stat_today.set_value(n_today)
        self.stat_emergency.set_value(n_emergency)

        # Emergencies
        for i in reversed(range(self.emergency_layout.count())):
            self.emergency_layout.itemAt(i).widget().deleteLater()

        emergencies = db_query(
            "SELECT * FROM emergencies WHERE resolved=0 ORDER BY timestamp DESC"
        )
        if emergencies:
            for em in emergencies:
                banner = EmergencyBanner(
                    emergency_id=em["id"],
                    patient_name=em["patient_name"],
                    timestamp=em["timestamp"],
                    trigger_phrase=em.get("trigger_phrase", ""),
                )
                banner.resolved.connect(self._on_emergency_resolved)
                self.emergency_layout.addWidget(banner)
        else:
            lbl = QLabel("  ✓  No active emergencies")
            lbl.setStyleSheet(
                f"color:{C['green']}; font-size:12px; padding:8px; "
                f"background: #0D2A1A; border:1px solid {C['green']}; "
                f"border-radius:6px;"
            )
            self.emergency_layout.addWidget(lbl)

        # Recent sessions
        recent = db_query("""
            SELECT s.*, p.name AS patient_name FROM sessions s
            JOIN patients p ON p.id = s.patient_id
            ORDER BY s.started_at DESC LIMIT 10
        """)
        self.recent_table.setRowCount(0)
        for s in recent:
            row = self.recent_table.rowCount()
            self.recent_table.insertRow(row)
            self.recent_table.setRowHeight(row, 40)

            score = s.get("total_score", 0)
            maxs  = s.get("max_score", 1) or 1
            pct   = int(score / maxs * 100)
            color = C["green"] if pct >= 75 else (C["amber"] if pct >= 50 else C["red"])

            vals = [
                s.get("patient_name", ""),
                s.get("started_at", "")[:16],
                s.get("task_name", ""),
                f"{score}/{maxs}",
                f"{pct}%",
                "📄",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 3 or col == 4:
                    item.setForeground(QColor(color))
                    item.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                else:
                    item.setForeground(QColor(C["text"]))
                self.recent_table.setItem(row, col, item)


# ── Reports page ───────────────────────────────────────────────────────────────
class ReportsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C['bg']};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        lay.addWidget(section_header("Generated Reports"))

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Filename", "Patient", "Date", "Open"])
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: transparent; border: none;
                color: {C['text']}; font-size: 13px;
                gridline-color: {C['border']};
            }}
            QHeaderView::section {{
                background: {C['panel2']}; color: {C['text2']};
                font-size: 10px; font-weight: 700;
                letter-spacing: 1px; padding: 8px;
                border: none; border-bottom: 1px solid {C['border']};
            }}
            QTableWidget::item {{
                padding: 10px 8px; border-bottom: 1px solid {C['border']};
            }}
        """)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._open_report)
        lay.addWidget(self.table)

        self._files = []
        self.refresh()

    def refresh(self):
        self.table.setRowCount(0)
        self._files = []
        if not os.path.exists(REPORTS_DIR):
            return

        files = sorted(
            [f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")],
            reverse=True
        )
        for f in files:
            parts = f.replace("HCA_Report_", "").replace(".pdf", "").split("_")
            patient = parts[0] if parts else "—"
            date    = "_".join(parts[1:3]) if len(parts) >= 3 else "—"

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setRowHeight(row, 44)

            for col, val in enumerate([f, patient, date, "📄 Open"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(C["accent"] if col == 3 else C["text"]))
                self.table.setItem(row, col, item)
            self._files.append(os.path.join(REPORTS_DIR, f))

    def _open_report(self, row, _):
        if row < len(self._files):
            os.startfile(self._files[row])


# ── Main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jennet — AI Healthcare Assistant Dashboard")
        self.setMinimumSize(1200, 760)
        self._live_process = None

        self._setup_ui()

        # Auto-refresh every 15 seconds
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._auto_refresh)
        self._refresh_timer.start(15000)

    def _setup_ui(self):
        self.setStyleSheet(f"QMainWindow {{ background: {C['bg']}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(f"""
            background: {C['panel']};
            border-right: 1px solid {C['border']};
        """)
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(12, 24, 12, 24)
        sb_lay.setSpacing(4)

        # Logo
        logo = QLabel("✦ JENNET")
        logo.setStyleSheet(
            f"color:{C['accent']}; font-size:18px; font-weight:900; "
            f"letter-spacing:3px; padding:0 8px 16px 8px; background:transparent;"
        )
        sb_lay.addWidget(logo)

        sub = QLabel("HCA Dashboard")
        sub.setStyleSheet(
            f"color:{C['text3']}; font-size:10px; letter-spacing:1px; "
            f"padding:0 8px 20px 8px; background:transparent;"
        )
        sb_lay.addWidget(sub)

        # Nav buttons
        self.nav_dashboard = SidebarBtn("⬡", "Dashboard")
        self.nav_patients  = SidebarBtn("⊙", "Patients")
        self.nav_reports   = SidebarBtn("◫", "Reports")
        self.nav_dashboard.setChecked(True)

        self.nav_dashboard.clicked.connect(lambda: self._nav(0))
        self.nav_patients.clicked.connect(lambda:  self._nav(1))
        self.nav_reports.clicked.connect(lambda:   self._nav(2))

        for btn in [self.nav_dashboard, self.nav_patients, self.nav_reports]:
            sb_lay.addWidget(btn)

        sb_lay.addStretch()

        # Live indicator
        self.live_indicator = LiveIndicator()
        sb_lay.addWidget(self.live_indicator)

        sb_lay.addSpacing(12)

        # Start session button
        start_btn = QPushButton("▶  Start Session")
        start_btn.setFixedHeight(42)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C['accent']}, stop:1 {C['accent2']});
                color: {C['bg']};
                border: none; border-radius: 8px;
                font-size: 13px; font-weight: 700;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #33D1FF, stop:1 {C['accent']});
            }}
        """)
        start_btn.clicked.connect(self._start_session)
        shadow(start_btn, radius=16, color=C["accent"], alpha=80)
        sb_lay.addWidget(start_btn)

        root.addWidget(sidebar)

        # ── Main content ───────────────────────────────────────────────
        content_wrapper = QWidget()
        content_wrapper.setStyleSheet(f"background:{C['bg']};")
        content_lay = QVBoxLayout(content_wrapper)
        content_lay.setContentsMargins(28, 24, 28, 24)
        content_lay.setSpacing(0)

        # Header bar
        header = QHBoxLayout()
        self.page_title = QLabel("Dashboard")
        self.page_title.setStyleSheet(
            f"color:{C['white']}; font-size:22px; font-weight:700; background:transparent;"
        )
        header.addWidget(self.page_title)
        header.addStretch()

        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(
            f"color:{C['text2']}; font-size:12px; background:transparent;"
        )
        header.addWidget(self.clock_lbl)

        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.setFixedHeight(32)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['panel2']}; color: {C['text2']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 0 14px; font-size: 11px;
            }}
            QPushButton:hover {{ color: {C['white']}; border-color: {C['accent']}; }}
        """)
        refresh_btn.clicked.connect(self._manual_refresh)
        header.addWidget(refresh_btn)
        content_lay.addLayout(header)
        content_lay.addSpacing(20)

        # Stacked pages
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:transparent;")

        # Page 0: Dashboard
        self.dashboard_page = DashboardPage()
        self.stack.addWidget(self.dashboard_page)

        # Page 1: Patients (stacked: list + detail)
        self.patients_stack = QStackedWidget()
        self.patients_stack.setStyleSheet("background:transparent;")

        patients_list_page = QWidget()
        patients_list_page.setStyleSheet("background:transparent;")
        pl_lay = QVBoxLayout(patients_list_page)
        pl_lay.setContentsMargins(0, 0, 0, 0)
        pl_lay.setSpacing(12)
        pl_lay.addWidget(section_header("All Patients"))
        self.patients_table = PatientsTable()
        self.patients_table.patient_selected.connect(self._show_patient_detail)
        pl_lay.addWidget(self.patients_table)
        self.patients_stack.addWidget(patients_list_page)

        self.patient_detail = PatientDetailPanel()
        self.patient_detail.back_requested.connect(
            lambda: self.patients_stack.setCurrentIndex(0)
        )
        self.patients_stack.addWidget(self.patient_detail)
        self.stack.addWidget(self.patients_stack)

        # Page 2: Reports
        self.reports_page = ReportsPage()
        self.stack.addWidget(self.reports_page)

        content_lay.addWidget(self.stack)
        root.addWidget(content_wrapper)

        # Clock timer
        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

        # Initial data load
        self.patients_table.load()

    def _nav(self, index):
        self.stack.setCurrentIndex(index)
        titles = ["Dashboard", "Patients", "Reports"]
        self.page_title.setText(titles[index])
        for btn, i in [(self.nav_dashboard, 0),
                       (self.nav_patients,  1),
                       (self.nav_reports,   2)]:
            btn.setChecked(i == index)
        if index == 2:
            self.reports_page.refresh()

    def _show_patient_detail(self, patient: dict):
        self.patient_detail.load_patient(patient)
        self.patients_stack.setCurrentIndex(1)

    def _start_session(self):
        dlg = StartSessionDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, location = dlg.get_values()
            if not name:
                QMessageBox.warning(self, "Missing Name", "Please enter a patient name.")
                return

            python = sys.executable
            cmd = [python, TASK_SCRIPT]
            try:
                self._live_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                input_data = f"{name}\n{location}\n"
                self._live_process.stdin.write(input_data)
                self._live_process.stdin.flush()
                self.live_indicator.set_active(name)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not start session:\n{e}")

    def _update_clock(self):
        now = datetime.datetime.now().strftime("%A, %d %B %Y  ·  %H:%M:%S")
        self.clock_lbl.setText(now)

    def _auto_refresh(self):
        current = self.stack.currentIndex()
        if current == 0:
            self.dashboard_page.refresh()
        elif current == 1:
            self.patients_table.load()

    def _manual_refresh(self):
        self.dashboard_page.refresh()
        self.patients_table.load()
        self.reports_page.refresh()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Jennet HCA Dashboard")

    # Global font
    app.setFont(QFont("Segoe UI", 11))

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(C["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Base,            QColor(C["panel"]))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(C["panel2"]))
    palette.setColor(QPalette.ColorRole.Text,            QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(C["accent2"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(C["white"]))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
