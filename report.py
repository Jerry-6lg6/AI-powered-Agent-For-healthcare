"""
report.py
---------
Clinical PDF report generator for the AI-Powered HCA system.

Usage:
    python report.py                        # Report for most recent session
    python report.py --patient "Rami"       # All sessions for a patient
    python report.py --session 8            # Specific session by ID
    python report.py --patient "Rami" --session 8  # Specific session for patient

Output:
    reports/HCA_Report_<PatientName>_<Date>.pdf
"""

import argparse
import os
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus import Flowable

from database.db import DatabaseManager


# ── Colour palette ─────────────────────────────────────────────────────────────
NHS_BLUE       = colors.HexColor("#005EB8")   # NHS brand blue
NHS_DARK       = colors.HexColor("#003087")   # Dark navy
NHS_LIGHT_BLUE = colors.HexColor("#41B6E6")   # Light accent
SECTION_BG     = colors.HexColor("#F0F4F8")   # Light grey section headers
WHITE          = colors.white
TEXT_DARK      = colors.HexColor("#212B32")   # Dark text
PASS_GREEN     = colors.HexColor("#007F3B")   # Correct answer
FAIL_RED       = colors.HexColor("#DA291C")   # Incorrect answer
WARN_AMBER     = colors.HexColor("#FFB81C")   # Warning / partial
EMERGENCY_RED  = colors.HexColor("#8C1515")   # Emergency events


# ── Custom Flowable: coloured banner ──────────────────────────────────────────
class Banner(Flowable):
    """A solid colour banner with white text — used for the report header."""

    def __init__(self, width, height, bg_color, text, font_size=22):
        super().__init__()
        self.banner_width  = width
        self.banner_height = height
        self.bg_color      = bg_color
        self.text          = text
        self.font_size     = font_size

    def wrap(self, available_width, available_height):
        return self.banner_width, self.banner_height

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg_color)
        c.rect(0, 0, self.banner_width, self.banner_height, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawCentredString(self.banner_width / 2, self.banner_height / 2 - 8, self.text)


# ── Style helpers ──────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()

    def add(name, **kwargs):
        base.add(ParagraphStyle(name=name, **kwargs))

    add("HCATitle",
        fontSize=26, leading=32, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER)

    add("HCASubtitle",
        fontSize=11, leading=14, textColor=colors.HexColor("#BDD7EE"),
        fontName="Helvetica", alignment=TA_CENTER)

    add("SectionHeader",
        fontSize=12, leading=16, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
        backColor=NHS_BLUE, borderPadding=(6, 8, 6, 8))

    add("FieldLabel",
        fontSize=9, leading=12, textColor=colors.HexColor("#6B7280"),
        fontName="Helvetica-Bold")

    add("FieldValue",
        fontSize=10, leading=13, textColor=TEXT_DARK,
        fontName="Helvetica")

    add("BodyText2",
        fontSize=10, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica")

    add("SmallGrey",
        fontSize=8, leading=11, textColor=colors.HexColor("#6B7280"),
        fontName="Helvetica")

    add("ScoreHuge",
        fontSize=36, leading=42, textColor=NHS_BLUE,
        fontName="Helvetica-Bold", alignment=TA_CENTER)

    add("ScoreLabel",
        fontSize=10, leading=13, textColor=colors.HexColor("#6B7280"),
        fontName="Helvetica", alignment=TA_CENTER)

    add("FooterText",
        fontSize=8, leading=10, textColor=colors.HexColor("#9CA3AF"),
        fontName="Helvetica", alignment=TA_CENTER)

    add("EmergencyText",
        fontSize=10, leading=14, textColor=EMERGENCY_RED,
        fontName="Helvetica-Bold")

    return base


def _section_header(text, styles):
    """Render a full-width blue section header bar."""
    page_w = A4[0] - 40*mm
    return [
        Spacer(1, 6*mm),
        Banner(page_w, 20, NHS_BLUE, text, font_size=11),
        Spacer(1, 3*mm),
    ]


def _two_col_table(pairs, styles, col_widths=None):
    """
    Render a list of (label, value) pairs as a clean two-column table.
    pairs: list of (label_str, value_str)
    """
    page_w = A4[0] - 40*mm
    col_widths = col_widths or [50*mm, page_w - 50*mm]
    data = [
        [
            Paragraph(label, styles["FieldLabel"]),
            Paragraph(str(value), styles["FieldValue"])
        ]
        for label, value in pairs
    ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, SECTION_BG]),
    ]))
    return t


def _score_pill(score, max_score):
    """Return colour based on score percentage."""
    if max_score == 0:
        return WARN_AMBER
    pct = score / max_score
    if pct >= 0.75:
        return PASS_GREEN
    elif pct >= 0.5:
        return WARN_AMBER
    return FAIL_RED


# ── Duration helper ────────────────────────────────────────────────────────────
def _duration(started_at: str, ended_at: str) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        start = datetime.datetime.strptime(started_at, fmt)
        end   = datetime.datetime.strptime(ended_at,   fmt)
        secs  = int((end - start).total_seconds())
        return f"{secs // 60}m {secs % 60}s"
    except Exception:
        return "N/A"


# ── Main report builder ────────────────────────────────────────────────────────
def generate_report(session_data: dict, db: DatabaseManager,
                    output_dir: str = "reports") -> str:
    """
    Build a clinical PDF for one session.
    Returns the path to the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    styles = _styles()

    session    = session_data["session"]
    responses  = session_data["responses"]
    patient_name = session["patient_name"]
    session_id   = session["id"]

    # ── Fetch emergency events for this session ────────────────────────
    emergencies = db.get_all_unresolved_emergencies()  # we'll filter by session
    # Also get resolved ones via a direct query approach using patient history
    all_emergencies = []
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emergencies WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,)
            ).fetchall()
            all_emergencies = [dict(r) for r in rows]
    except Exception:
        all_emergencies = []

    # ── Fetch previous sessions for trend ─────────────────────────────
    history = db.get_patient_history(patient_name)

    # ── Fetch previous session responses for comparison ────────────────
    patient_id = session.get("patient_id")
    prev_responses = {}
    if patient_id:
        try:
            prev_responses = db.get_last_session_responses(
                patient_id=patient_id,
                exclude_session_id=session_id
            )
        except Exception:
            prev_responses = {}

    # ── File path ──────────────────────────────────────────────────────
    date_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    filepath  = os.path.join(output_dir, f"HCA_Report_{safe_name}_{date_str}.pdf")

    # ── Document setup ─────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm,  bottomMargin=20*mm,
        title=f"HCA Clinical Report — {patient_name}",
        author="AI-Powered HCA System (Jennet)",
        subject="Cognitive Orientation Assessment"
    )

    page_w = A4[0] - 40*mm
    story  = []

    # ══════════════════════════════════════════════════════════════════
    # HEADER BANNER
    # ══════════════════════════════════════════════════════════════════
    story.append(Banner(page_w, 72, NHS_DARK,
                        "AI-Powered Healthcare Assistant", font_size=20))
    story.append(Spacer(1, 1*mm))

    subtitle_bar_data = [[
        Paragraph("COGNITIVE ORIENTATION ASSESSMENT REPORT", styles["SmallGrey"]),
        Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
                  styles["SmallGrey"])
    ]]
    subtitle_bar = Table(subtitle_bar_data, colWidths=[page_w * 0.6, page_w * 0.4])
    subtitle_bar.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), SECTION_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("ALIGN",        (1, 0), (1, 0),   "RIGHT"),
    ]))
    story.append(subtitle_bar)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — PATIENT INFORMATION
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("1. Patient Information", styles)

    # Fetch full patient record
    patient = db.get_patient_by_name(patient_name) or {}

    patient_pairs = [
        ("Patient Name",    patient_name),
        ("Date of Birth",   patient.get("date_of_birth") or "Not recorded"),
        ("Address / Ward",  patient.get("address")       or "Not recorded"),
        ("Patient ID",      f"#{patient.get('id', 'N/A')}"),
        ("First Registered",patient.get("created_at",    "N/A")),
        ("Total Sessions",  len(history) if history else 1),
    ]
    story.append(_two_col_table(patient_pairs, styles))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — SESSION OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("2. Session Overview", styles)

    duration = _duration(session.get("started_at", ""),
                         session.get("ended_at", ""))
    session_pairs = [
        ("Session ID",      f"#{session_id}"),
        ("Task Name",       session.get("task_name", "N/A")),
        ("Date",            session.get("started_at", "N/A")[:10]),
        ("Start Time",      session.get("started_at", "N/A")[11:19]),
        ("End Time",        session.get("ended_at",   "N/A")[11:19]),
        ("Duration",        duration),
        ("Completed",       "Yes" if session.get("completed") else "No"),
        ("Emergency Events",str(len(all_emergencies))),
    ]
    story.append(_two_col_table(session_pairs, styles))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 — SCORE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("3. Score Summary", styles)

    total_score = session.get("total_score", 0)
    max_score   = session.get("max_score",   0)
    pct         = int((total_score / max_score * 100) if max_score else 0)
    pill_color  = _score_pill(total_score, max_score)

    # Score display as a 3-column summary card
    score_data = [[
        Paragraph(f"{total_score}/{max_score}", styles["ScoreHuge"]),
        Paragraph(f"{pct}%", styles["ScoreHuge"]),
        Paragraph(
            "PASS" if pct >= 75 else ("PARTIAL" if pct >= 50 else "NEEDS SUPPORT"),
            ParagraphStyle("ScoreStatus", fontSize=18, leading=24,
                           fontName="Helvetica-Bold", alignment=TA_CENTER,
                           textColor=pill_color)
        ),
    ], [
        Paragraph("Total Score", styles["ScoreLabel"]),
        Paragraph("Percentage", styles["ScoreLabel"]),
        Paragraph("Outcome", styles["ScoreLabel"]),
    ]]
    score_table = Table(score_data, colWidths=[page_w/3]*3)
    score_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), SECTION_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("LINEAFTER",     (0, 0), (1, -1),  0.5, colors.HexColor("#D1D5DB")),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 3*mm))

    # Clinical interpretation
    if pct >= 75:
        interp = ("The patient demonstrated <b>good orientation awareness</b> during "
                  "this session. Responses were largely accurate and within expected range.")
    elif pct >= 50:
        interp = ("The patient showed <b>partial orientation awareness</b>. Some responses "
                  "were incorrect, which may indicate mild cognitive difficulty. "
                  "Continued monitoring is advised.")
    else:
        interp = ("The patient demonstrated <b>significant difficulty with orientation</b> "
                  "during this session. Clinical review and further assessment is recommended.")

    story.append(Paragraph(interp, styles["BodyText2"]))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4 — QUESTION & RESPONSE DETAIL
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("4. Question and Response Detail", styles)

    if responses:
        # Table header
        col_w = [7*mm, 35*mm, 26*mm, 32*mm, 15*mm, 15*mm, 15*mm, 25*mm]
        header_row = [
            Paragraph("#",                  styles["FieldLabel"]),
            Paragraph("Question",           styles["FieldLabel"]),
            Paragraph("Expected Answer",    styles["FieldLabel"]),
            Paragraph("Patient's Answer",   styles["FieldLabel"]),
            Paragraph("Result",             styles["FieldLabel"]),
            Paragraph("Confidence",         styles["FieldLabel"]),
            Paragraph("Think Time",         styles["FieldLabel"]),
            Paragraph("Last Session",       styles["FieldLabel"]),
        ]
        table_data = [header_row]

        # Track thinking time flags for clinical notes
        slow_think_questions = []

        for i, r in enumerate(responses, 1):
            correct      = r.get("is_correct", 0)
            is_emergency = r.get("is_emergency", 0)
            is_silence   = r.get("is_silence", 0)
            is_dont_know = r.get("is_dont_know", 0)
            q_name       = r.get("question_name", "")

            # ── Result cell ───────────────────────────────────────────
            if is_emergency:
                result_text  = "EMERGENCY"
                result_color = EMERGENCY_RED
            elif correct:
                result_text  = "CORRECT"
                result_color = PASS_GREEN
            elif is_silence:
                result_text  = "SILENCE"
                result_color = WARN_AMBER
            elif is_dont_know:
                result_text  = "UNSURE"
                result_color = WARN_AMBER
            else:
                result_text  = "INCORRECT"
                result_color = FAIL_RED

            result_style = ParagraphStyle(
                f"result_{i}", fontSize=8, fontName="Helvetica-Bold",
                textColor=result_color, alignment=TA_CENTER
            )

            # ── Confidence cell ───────────────────────────────────────
            conf_val = r.get("confidence", 0.0) or 0.0
            conf_pct = f"{conf_val * 100:.0f}%"
            if conf_val >= 0.85:
                conf_color = "#007F3B"
            elif conf_val >= 0.55:
                conf_color = "#FFB81C"
            else:
                conf_color = "#DA291C"
            conf_style = ParagraphStyle(
                f"conf_{i}", fontSize=8, fontName="Helvetica-Bold",
                textColor=colors.HexColor(conf_color), alignment=TA_CENTER
            )

            # ── Thinking time cell ────────────────────────────────────
            think_t = r.get("thinking_time", 0) or 0
            speak_t = r.get("speaking_time", 0) or 0
            if think_t > 10:
                think_color  = "#DA291C"
                think_label  = f"⚠ {think_t:.1f}s"
                slow_think_questions.append(q_name)
            elif think_t > 5:
                think_color  = "#FFB81C"
                think_label  = f"{think_t:.1f}s"
            else:
                think_color  = "#007F3B"
                think_label  = f"{think_t:.1f}s"
            think_style = ParagraphStyle(
                f"think_{i}", fontSize=8, fontName="Helvetica-Bold",
                textColor=colors.HexColor(think_color), alignment=TA_CENTER
            )

            # ── Last session comparison cell ──────────────────────────
            prev = prev_responses.get(q_name)
            if prev is None:
                last_text  = "—"
                last_color = "#9CA3AF"
                last_flag  = ""
            else:
                prev_ans     = prev.get("patient_answer", "") or "—"
                prev_correct = prev.get("is_correct", 0)
                # Decline detection: was correct last time, wrong now
                if prev_correct and not correct:
                    last_flag  = " ▼"
                    last_color = "#DA291C"
                elif not prev_correct and correct:
                    last_flag  = " ▲"
                    last_color = "#007F3B"
                else:
                    last_flag  = ""
                    last_color = "#6B7280"
                last_text = prev_ans[:20] + ("…" if len(prev_ans) > 20 else "")
                last_text += last_flag

            last_style = ParagraphStyle(
                f"last_{i}", fontSize=7,
                textColor=colors.HexColor(last_color), alignment=TA_LEFT
            )

            speak_note = f"<font size='7' color='#9CA3AF'>Speak: {speak_t:.1f}s</font>"

            table_data.append([
                Paragraph(str(i),                            styles["SmallGrey"]),
                Paragraph(r.get("question_text", ""),        styles["BodyText2"]),
                Paragraph(r.get("expected_answer", ""),      styles["BodyText2"]),
                Paragraph(
                    r.get("patient_answer", "") + f"<br/>{speak_note}",
                    styles["BodyText2"]
                ),
                Paragraph(result_text,   result_style),
                Paragraph(conf_pct,      conf_style),
                Paragraph(think_label,   think_style),
                Paragraph(last_text,     last_style),
            ])

        resp_table = Table(table_data, colWidths=col_w, repeatRows=1)
        resp_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND",    (0, 0), (-1, 0),  NHS_LIGHT_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            # Body
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, SECTION_BG]),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
            ("ALIGN",         (4, 0), (4, -1),  "CENTER"),
            ("ALIGN",         (5, 0), (5, -1),  "CENTER"),
            ("ALIGN",         (6, 0), (6, -1),  "CENTER"),
        ]))
        story.append(resp_table)

        # ── Last session legend ────────────────────────────────────────
        if prev_responses:
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph(
                "<font color='#DA291C'>▼ Decline</font> — correct last session, incorrect this session  &nbsp;&nbsp;"
                "<font color='#007F3B'>▲ Improvement</font> — incorrect last session, correct this session  &nbsp;&nbsp;"
                "<font color='#DA291C'>⚠</font> — thinking time &gt;10s (possible processing difficulty)",
                ParagraphStyle("legend", fontSize=7, textColor=colors.HexColor("#6B7280"),
                               parent=styles["BodyText2"])
            ))
    else:
        story.append(Paragraph("No responses were recorded for this session.",
                               styles["BodyText2"]))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5 — EMERGENCY EVENTS
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("5. Emergency Events", styles)

    if all_emergencies:
        em_col_w = [12*mm, 55*mm, 60*mm, 25*mm]
        em_header = [
            Paragraph("#",              styles["FieldLabel"]),
            Paragraph("Timestamp",      styles["FieldLabel"]),
            Paragraph("Trigger Phrase", styles["FieldLabel"]),
            Paragraph("Resolved",       styles["FieldLabel"]),
        ]
        em_data = [em_header]
        for i, em in enumerate(all_emergencies, 1):
            resolved = "Yes" if em.get("resolved") else "No"
            resolved_style = ParagraphStyle(
                f"emres_{i}", fontSize=9, fontName="Helvetica-Bold",
                textColor=PASS_GREEN if em.get("resolved") else FAIL_RED
            )
            em_data.append([
                Paragraph(str(i),                          styles["SmallGrey"]),
                Paragraph(em.get("timestamp", ""),         styles["BodyText2"]),
                Paragraph(em.get("trigger_phrase", ""),    styles["EmergencyText"]),
                Paragraph(resolved,                        resolved_style),
            ])
        em_table = Table(em_data, colWidths=em_col_w)
        em_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#FEE2E2")),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, colors.HexColor("#FFF5F5")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#FECACA")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(em_table)
    else:
        story.append(Paragraph(
            "No emergency events were recorded during this session.",
            styles["BodyText2"]
        ))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 6 — SESSION HISTORY TREND
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("6. Session History & Score Trend", styles)

    if history:
        hist_col_w = [10*mm, 45*mm, 45*mm, 30*mm, 30*mm]
        hist_header = [
            Paragraph("#",          styles["FieldLabel"]),
            Paragraph("Date",       styles["FieldLabel"]),
            Paragraph("Task",       styles["FieldLabel"]),
            Paragraph("Score",      styles["FieldLabel"]),
            Paragraph("Duration",   styles["FieldLabel"]),
        ]
        hist_data = [hist_header]

        for i, s in enumerate(reversed(history[-10:]), 1):  # last 10 sessions
            s_score = s.get("total_score", 0)
            s_max   = s.get("max_score", 0)
            s_pct   = int(s_score / s_max * 100) if s_max else 0
            is_curr = s.get("id") == session_id

            score_col = ParagraphStyle(
                f"hist_{i}", fontSize=9, fontName="Helvetica-Bold",
                textColor=_score_pill(s_score, s_max)
            )
            hist_data.append([
                Paragraph(str(i) + (" *" if is_curr else ""), styles["SmallGrey"]),
                Paragraph(s.get("started_at", "")[:16],        styles["BodyText2"]),
                Paragraph(s.get("task_name", ""),              styles["BodyText2"]),
                Paragraph(f"{s_score}/{s_max} ({s_pct}%)",     score_col),
                Paragraph(_duration(s.get("started_at", ""),
                                    s.get("ended_at") or ""), styles["BodyText2"]),
            ])

        hist_table = Table(hist_data, colWidths=hist_col_w)
        hist_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  NHS_LIGHT_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, SECTION_BG]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(hist_table)
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("* Current session", styles["SmallGrey"]))
    else:
        story.append(Paragraph("No previous session history available.",
                               styles["BodyText2"]))

    # ══════════════════════════════════════════════════════════════════
    # SECTION 7 — CLINICAL NOTES & RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════
    story += _section_header("7. Clinical Notes & Recommendations", styles)

    # Auto-generate clinical notes based on session data
    notes = []

    if pct == 100:
        notes.append("Patient answered all orientation questions correctly.")
    elif pct >= 75:
        notes.append("Patient demonstrated good overall orientation. Minor errors noted.")
    elif pct >= 50:
        notes.append("Patient showed partial orientation awareness. Monitoring recommended.")
    else:
        notes.append("Patient struggled significantly with orientation tasks. Clinical review advised.")

    if all_emergencies:
        notes.append(f"ALERT: {len(all_emergencies)} emergency event(s) were triggered "
                     f"during this session. Staff were notified.")

    if responses:
        avg_think = sum((r.get("thinking_time") or 0) for r in responses) / len(responses)
        if avg_think > 10:
            notes.append(f"Patient showed extended thinking time (avg {avg_think:.1f}s), "
                         "which may indicate difficulty processing questions.")
        elif avg_think < 2:
            notes.append(f"Patient responded quickly (avg {avg_think:.1f}s thinking time).")

    silence_count = sum(1 for r in responses if r.get("is_silence"))
    if silence_count > 0:
        notes.append(f"Patient was silent {silence_count} time(s). "
                     "Consider checking for hearing or comprehension difficulties.")

    dont_know_count = sum(1 for r in responses if r.get("is_dont_know"))
    if dont_know_count > 0:
        notes.append(f"Patient expressed uncertainty {dont_know_count} time(s).")

    # ── Thinking time analysis ─────────────────────────────────────────
    if responses:
        slow_qs = [r.get("question_name", "") for r in responses
                   if (r.get("thinking_time") or 0) > 10]
        if slow_qs:
            notes.append(
                f"Prolonged processing time (>10s) detected on "
                f"{len(slow_qs)} question(s): {', '.join(slow_qs)}. "
                f"This may indicate difficulty with specific orientation domains."
            )

    # ── Session comparison / decline detection ─────────────────────────
    if prev_responses:
        declines = []
        improvements = []
        for r in responses:
            q_name = r.get("question_name", "")
            prev   = prev_responses.get(q_name)
            if prev:
                was_correct = prev.get("is_correct", 0)
                now_correct = r.get("is_correct", 0)
                if was_correct and not now_correct:
                    declines.append(q_name)
                elif not was_correct and now_correct:
                    improvements.append(q_name)
        if declines:
            notes.append(
                f"Decline detected: patient answered {', '.join(declines)} correctly "
                f"last session but not this session. Clinical review recommended."
            )
        if improvements:
            notes.append(
                f"Improvement noted: patient now correctly answered "
                f"{', '.join(improvements)} which were incorrect last session."
            )

    # Score trend from history
    if len(history) >= 2:
        recent_scores = [
            (s.get("total_score", 0) / s.get("max_score", 1) * 100)
            for s in history[:3] if s.get("max_score")
        ]
        if len(recent_scores) >= 2:
            trend = recent_scores[0] - recent_scores[-1]
            if trend > 10:
                notes.append("Score trend: Improving compared to recent sessions.")
            elif trend < -10:
                notes.append("Score trend: Declining compared to recent sessions. "
                             "Further clinical assessment recommended.")
            else:
                notes.append("Score trend: Stable performance across recent sessions.")

    # Add recommendation for staff
    notes.append("This report was auto-generated by the AI-Powered HCA System (Jennet) "
                 "and should be reviewed by a qualified clinician.")

    for note in notes:
        story.append(Paragraph(f"• {note}", styles["BodyText2"]))
        story.append(Spacer(1, 2*mm))

    # ══════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width=page_w, thickness=0.5,
                            color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 3*mm))

    footer_data = [[
        Paragraph("AI-Powered Healthcare Assistant — Jennet",
                  styles["FooterText"]),
        Paragraph(f"Session #{session_id} | Patient: {patient_name}",
                  styles["FooterText"]),
        Paragraph("CONFIDENTIAL — Clinical Use Only",
                  ParagraphStyle("ConfidentialFooter", fontSize=8,
                                 fontName="Helvetica-Bold",
                                 textColor=FAIL_RED, alignment=TA_RIGHT)),
    ]]
    footer_table = Table(footer_data, colWidths=[page_w/3]*3)
    footer_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
    ]))
    story.append(footer_table)

    # ── Build PDF ──────────────────────────────────────────────────────
    doc.build(story)
    print(f"[REPORT] Clinical report saved: {filepath}")
    return filepath


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a clinical PDF report for an HCA session."
    )
    parser.add_argument("--patient", type=str, default=None,
                        help="Patient name to report on")
    parser.add_argument("--session", type=int, default=None,
                        help="Specific session ID to report on")
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for PDF (default: reports/)")
    args = parser.parse_args()

    db = DatabaseManager()

    # Resolve which session to report on
    if args.session:
        session_data = db.get_session_summary(args.session)
        if not session_data:
            print(f"[ERROR] Session #{args.session} not found.")
            return

    elif args.patient:
        history = db.get_patient_history(args.patient)
        if not history:
            print(f"[ERROR] No sessions found for patient '{args.patient}'.")
            return
        # Most recent session
        session_data = db.get_session_summary(history[0]["id"])

    else:
        # Most recent session of any patient
        recent = db.get_recent_sessions(limit=1)
        if not recent:
            print("[ERROR] No sessions found in the database.")
            return
        session_data = db.get_session_summary(recent[0]["id"])

    generate_report(session_data, db, output_dir=args.output)


if __name__ == "__main__":
    main()
