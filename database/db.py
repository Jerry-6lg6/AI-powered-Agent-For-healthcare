"""
db.py
-----
DatabaseManager: handles all interactions with the SQLite database.
Provides clean methods for task.py to call without any SQL knowledge needed.
"""

import sqlite3
import datetime
import os
from typing import Optional, List, Dict, Any

from database.models import ALL_TABLES


DB_PATH = os.path.join(os.path.dirname(__file__), "hca.db")


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # rows behave like dicts
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _now(self) -> str:
        return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

    def _init_db(self):
        """Create all tables if they do not already exist."""
        with self._connect() as conn:
            for statement in ALL_TABLES:
                conn.execute(statement)
        print(f"[DB] Database ready at: {self.db_path}")

    # ------------------------------------------------------------------
    # Patient methods
    # ------------------------------------------------------------------

    def get_or_create_patient(self, name: str,
                               date_of_birth: str = None,
                               address: str = None) -> Dict[str, Any]:
        """
        Look up a patient by name.
        If found  → return their record.
        If not    → create a new record and return it.
        """
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()

            if row:
                print(f"[DB] Returning patient: {row['name']} (id={row['id']})")
                return dict(row)

            # New patient
            conn.execute(
                """INSERT INTO patients (name, date_of_birth, address, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, date_of_birth, address, now, now)
            )
            row = conn.execute(
                "SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
            print(f"[DB] New patient created: {row['name']} (id={row['id']})")
            return dict(row)

    def get_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return patient record or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Return all patients."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY name ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Session methods
    # ------------------------------------------------------------------

    def start_session(self, patient_id: int, task_name: str) -> int:
        """
        Open a new session for a patient.
        Returns the new session id.
        """
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO sessions (patient_id, task_name, started_at)
                   VALUES (?, ?, ?)""",
                (patient_id, task_name, now)
            )
            session_id = cursor.lastrowid
            print(f"[DB] Session started (id={session_id}) for patient_id={patient_id}")
            return session_id

    def end_session(self, session_id: int,
                    total_score: int = 0,
                    max_score: int = 0,
                    completed: bool = True):
        """Close a session and record the final score."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions
                   SET ended_at=?, total_score=?, max_score=?, completed=?
                   WHERE id=?""",
                (now, total_score, max_score, int(completed), session_id)
            )
            print(f"[DB] Session {session_id} ended. Score: {total_score}/{max_score}")

    def get_sessions_for_patient(self, patient_id: int) -> List[Dict[str, Any]]:
        """Return all sessions for a patient, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE patient_id=?
                   ORDER BY started_at DESC""",
                (patient_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Response methods
    # ------------------------------------------------------------------

    def save_response(self,
                      session_id: int,
                      question_name: str,
                      question_text: str,
                      expected_answer: str,
                      patient_answer: str,
                      is_correct: bool,
                      thinking_time: float = 0.0,
                      speaking_time: float = 0.0,
                      state: dict = None,
                      confidence: float = 0.0):
        """
        Save a single question/answer record to the responses table.
        `state` is the dict returned by clf.state_match().
        `confidence` is the similarity score (0.0–1.0) from the classifier.
        """
        state = state or {}
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO responses (
                       session_id, question_name, question_text,
                       expected_answer, patient_answer, is_correct,
                       thinking_time, speaking_time,
                       is_dont_know, is_stop, is_emergency,
                       is_free_talk, is_silence, confidence, timestamp
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, question_name, question_text,
                    expected_answer, patient_answer, int(is_correct),
                    thinking_time, speaking_time,
                    int(state.get("dont_know", 0)),
                    int(state.get("stop", 0)),
                    int(state.get("emergency", 0)),
                    int(state.get("free_talk", 0)),
                    int(state.get("silence", 0)),
                    round(float(confidence), 4),
                    now,
                )
            )

    def get_responses_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Return all responses for a given session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM responses WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_last_session_responses(self, patient_id: int,
                                   exclude_session_id: int) -> Dict[str, Any]:
        """
        Fetch responses from the most recent completed session for this patient,
        excluding the current session. Returns a dict keyed by question_name
        so the report can look up previous answers by question.

        Used for session-comparison column in the PDF report.
        """
        with self._connect() as conn:
            # Find the most recent session before the current one
            prev = conn.execute(
                """SELECT id FROM sessions
                   WHERE patient_id = ?
                     AND id != ?
                     AND completed = 1
                   ORDER BY started_at DESC
                   LIMIT 1""",
                (patient_id, exclude_session_id)
            ).fetchone()

            if not prev:
                return {}

            rows = conn.execute(
                "SELECT * FROM responses WHERE session_id=? ORDER BY timestamp ASC",
                (prev["id"],)
            ).fetchall()

            # Key by question_name for fast lookup
            return {r["question_name"]: dict(r) for r in rows}

    # ------------------------------------------------------------------
    # Diagnostic / memory query methods
    # ------------------------------------------------------------------

    def get_patient_history(self, name: str) -> List[Dict[str, Any]]:
        """
        Return all sessions + scores for a patient by name.
        Useful for tracking progress over time.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.id, s.task_name, s.started_at, s.ended_at,
                          s.total_score, s.max_score, s.completed
                   FROM sessions s
                   JOIN patients p ON p.id = s.patient_id
                   WHERE LOWER(p.name) = LOWER(?)
                   ORDER BY s.started_at DESC""",
                (name,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_weak_areas(self, name: str) -> List[Dict[str, Any]]:
        """
        Return questions the patient most frequently gets wrong.
        Ordered by error count descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.question_name,
                          r.question_text,
                          COUNT(*) AS attempts,
                          SUM(CASE WHEN r.is_correct=0 THEN 1 ELSE 0 END) AS errors,
                          ROUND(AVG(r.thinking_time), 2) AS avg_thinking_time
                   FROM responses r
                   JOIN sessions s ON s.id = r.session_id
                   JOIN patients p ON p.id = s.patient_id
                   WHERE LOWER(p.name) = LOWER(?)
                   GROUP BY r.question_name
                   ORDER BY errors DESC""",
                (name,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_thinking_time_trend(self, name: str) -> List[Dict[str, Any]]:
        """
        Return average thinking time per session for a patient.
        Can be plotted to show cognitive response trends.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.started_at,
                          ROUND(AVG(r.thinking_time), 2) AS avg_thinking_time,
                          ROUND(AVG(r.speaking_time), 2) AS avg_speaking_time
                   FROM responses r
                   JOIN sessions s ON s.id = r.session_id
                   JOIN patients p ON p.id = s.patient_id
                   WHERE LOWER(p.name) = LOWER(?)
                   GROUP BY s.id
                   ORDER BY s.started_at ASC""",
                (name,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent sessions across all patients."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT p.name AS patient_name, s.task_name,
                          s.started_at, s.total_score, s.max_score, s.completed
                   FROM sessions s
                   JOIN patients p ON p.id = s.patient_id
                   ORDER BY s.started_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        """
        Return a full summary of one session:
        patient name, all responses, score, duration.
        """
        with self._connect() as conn:
            session = conn.execute(
                """SELECT s.*, p.name AS patient_name
                   FROM sessions s
                   JOIN patients p ON p.id = s.patient_id
                   WHERE s.id=?""",
                (session_id,)
            ).fetchone()

            if not session:
                return {}

            responses = conn.execute(
                "SELECT * FROM responses WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,)
            ).fetchall()

            return {
                "session": dict(session),
                "responses": [dict(r) for r in responses]
            }

    # ------------------------------------------------------------------
    # Emergency methods
    # ------------------------------------------------------------------

    def save_emergency(self, patient_id: int, session_id: int,
                       patient_name: str, trigger_phrase: str) -> int:
        """Save an emergency event. Returns the new emergency id."""
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO emergencies
                   (patient_id, session_id, patient_name, trigger_phrase, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (patient_id, session_id, patient_name, trigger_phrase, now)
            )
            emergency_id = cursor.lastrowid
            print(f"[DB] Emergency saved (id={emergency_id}) for {patient_name}")
            return emergency_id

    def resolve_emergency(self, emergency_id: int):
        """Mark an emergency as resolved."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE emergencies SET resolved=1, resolved_at=? WHERE id=?",
                (now, emergency_id)
            )
            print(f"[DB] Emergency {emergency_id} marked as resolved.")

    def get_emergencies_for_patient(self, name: str):
        """Return all emergency events for a patient by name."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.* FROM emergencies e
                   JOIN patients p ON p.id = e.patient_id
                   WHERE LOWER(p.name) = LOWER(?)
                   ORDER BY e.timestamp DESC""",
                (name,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_unresolved_emergencies(self):
        """Return all emergencies that have not been resolved."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emergencies WHERE resolved=0 ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Patient profile methods
    # ------------------------------------------------------------------

    def save_patient_profile(self, patient_id: int, hometown: str = None,
                              spouse_name: str = None, children_names: str = None,
                              occupation: str = None, hobbies: str = None,
                              favourite_food: str = None, favourite_sport: str = None,
                              favourite_show: str = None):
        """Create or update a patient's personal profile."""
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM patient_profiles WHERE patient_id=?",
                (patient_id,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE patient_profiles SET
                        hometown=?, spouse_name=?, children_names=?,
                        occupation=?, hobbies=?, favourite_food=?,
                        favourite_sport=?, favourite_show=?, updated_at=?
                    WHERE patient_id=?
                """, (hometown, spouse_name, children_names, occupation,
                      hobbies, favourite_food, favourite_sport,
                      favourite_show, now, patient_id))
            else:
                conn.execute("""
                    INSERT INTO patient_profiles
                    (patient_id, hometown, spouse_name, children_names,
                     occupation, hobbies, favourite_food, favourite_sport,
                     favourite_show, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (patient_id, hometown, spouse_name, children_names,
                      occupation, hobbies, favourite_food, favourite_sport,
                      favourite_show, now))
            print(f"[DB] Profile saved for patient_id={patient_id}")

    def get_patient_profile(self, patient_id: int):
        """Return a patient's personal profile as a dict, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patient_profiles WHERE patient_id=?",
                (patient_id,)
            ).fetchone()
            return dict(row) if row else None
