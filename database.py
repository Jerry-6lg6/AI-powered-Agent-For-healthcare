import sqlite3
import json
from datetime import datetime


# ============================================================
# SECTION 1: DATABASE MANAGER
# Handles the actual connection to the SQLite database file.
# Every other class uses this to talk to the database.
# ============================================================

class DatabaseManager:

    def __init__(self, db_path="patients.db"):
        # db_path is the name of the file where all data is saved.
        # SQLite creates this file automatically if it doesn't exist.
        self.db_path = db_path
        self.connection = None

    def connect(self):
        # Opens the connection to the database file.
        # Call this once at the start of each session.
        self.connection = sqlite3.connect(self.db_path)

        # This makes rows come back as dictionaries (key: value)
        # instead of plain tuples, which is much easier to work with.
        self.connection.row_factory = sqlite3.Row

        print(f"Connected to database: {self.db_path}")

    def disconnect(self):
        # Closes the connection cleanly.
        # Call this at the end of each session.
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def initialise_tables(self):
        # Creates all four tables if they don't already exist.
        # Safe to call every time — IF NOT EXISTS prevents duplication.

        # --- Patients table ---
        # Stores basic static info about each patient.
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id   INTEGER PRIMARY KEY,
                name         TEXT,
                address      TEXT,
                date_added   TEXT
            )
        """)

        # --- Sessions table ---
        # Stores one row per completed session.
        # The full session detail is stored as a JSON string.
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id       TEXT PRIMARY KEY,
                patient_id       INTEGER,
                task_name        TEXT,
                date             TEXT,
                time_of_day      TEXT,
                final_state      TEXT,
                total_score      INTEGER,
                session_duration REAL,
                full_session_json TEXT
            )
        """)

        # --- Profiles table ---
        # Stores one row per patient — the living adaptive summary.
        # Updated after every session.
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                patient_id               INTEGER PRIMARY KEY,
                avg_thinking_time        REAL,
                avg_speaking_time        REAL,
                confidence_indicator     REAL,
                preferred_speed          REAL,
                bd_retry_success_rate    REAL,
                special_state_frequency  REAL,
                recovery_rate            REAL,
                fatigue_threshold        INTEGER,
                error_rate_trend         TEXT,
                best_time_of_day         TEXT,
                total_sessions           INTEGER,
                last_updated             TEXT
            )
        """)

        # --- Special events table ---
        # Stores real-time events that happen mid-session.
        # e.g. emergency, rest request, speed change, score==2
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS special_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER,
                session_id   TEXT,
                event_type   TEXT,
                timestamp    TEXT,
                metadata     TEXT
            )
        """)

        # Save all the table creations
        self.connection.commit()
        print("All tables initialised.")

    def execute_query(self, query, params=()):
        # A single safe place to run all SQL queries.
        # All other methods call this instead of writing raw SQL themselves.
        # params uses ? placeholders to prevent SQL injection.
        try:
            cursor = self.connection.execute(query, params)
            self.connection.commit()
            return cursor
        except Exception as e:
            print(f"Database error: {e}")
            return None


# ============================================================
# SECTION 2: PATIENT TABLE
# Stores and retrieves basic patient information.
# ============================================================

class PatientTable:

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_patient(self, patient):
        # Adds a new patient row to the patients table.
        # Called once when a patient is first registered.
        self.db.execute_query(
            "INSERT INTO patients VALUES (?, ?, ?, ?)",
            (patient.id,
             patient.name,
             patient._address,
             str(datetime.now().date()))
        )
        print(f"Patient {patient.name} added to database.")

    def get_patient(self, patient_id):
        # Retrieves one patient's details by their ID.
        # Returns a dictionary, or None if not found.
        cursor = self.db.execute_query(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        )
        row = cursor.fetchone()

        if row:
            # Convert the row into a plain dictionary
            return dict(row)
        return None

    def get_all_patients(self):
        # Returns a list of all patients in the database.
        cursor = self.db.execute_query("SELECT * FROM patients")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_patient(self, patient_id, updated_fields):
        # Updates specific fields for a patient.
        # updated_fields is a dictionary e.g. {"address": "new address"}
        for field, value in updated_fields.items():
            self.db.execute_query(
                f"UPDATE patients SET {field} = ? WHERE patient_id = ?",
                (value, patient_id)
            )
        print(f"Patient {patient_id} updated.")


# ============================================================
# SECTION 3: SESSION TABLE
# Stores one complete session record per row after each session ends.
# ============================================================

class SessionTable:

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_session(self, session_record):
        # Saves a completed session to the database.
        # The full nested session detail is stored as a JSON string.
        self.db.execute_query(
            """INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_record["session_id"],
                session_record["patient_id"],
                session_record["task_name"],
                session_record["date"],
                session_record["time_of_day"],
                session_record["final_state"],
                session_record["total_score"],
                session_record["session_duration"],

                # Convert the full nested dictionary to a JSON string for storage
                json.dumps(session_record)
            )
        )
        print(f"Session {session_record['session_id']} saved.")

    def get_sessions_by_patient(self, patient_id):
        # Returns all sessions for a patient, oldest first.
        cursor = self.db.execute_query(
            "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date ASC",
            (patient_id,)
        )
        rows = cursor.fetchall()

        # Parse the JSON string back into a dictionary for each session
        sessions = []
        for row in rows:
            row_dict = dict(row)
            row_dict["full_session_json"] = json.loads(row_dict["full_session_json"])
            sessions.append(row_dict)
        return sessions

    def get_last_n_sessions(self, patient_id, n=5):
        # Returns only the most recent N sessions for a patient.
        cursor = self.db.execute_query(
            "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date DESC LIMIT ?",
            (patient_id, n)
        )
        rows = cursor.fetchall()
        sessions = []
        for row in rows:
            row_dict = dict(row)
            row_dict["full_session_json"] = json.loads(row_dict["full_session_json"])
            sessions.append(row_dict)
        return sessions

    def get_session_by_id(self, session_id):
        # Returns one specific session by its unique ID.
        cursor = self.db.execute_query(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            row_dict = dict(row)
            row_dict["full_session_json"] = json.loads(row_dict["full_session_json"])
            return row_dict
        return None


# ============================================================
# SECTION 4: PROFILE TABLE
# Stores the living adaptive summary for each patient.
# One row per patient — updated after every session.
# ============================================================

class ProfileTable:

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_profile(self, patient_id):
        # Creates a blank default profile for a new patient.
        # All values start at safe neutral defaults.
        self.db.execute_query(
            """INSERT INTO profiles VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                3.0,        # avg_thinking_time        — default 3 seconds
                2.0,        # avg_speaking_time         — default 2 seconds
                0.5,        # confidence_indicator      — neutral 0.5
                1.0,        # preferred_speed           — start at normal speed
                0.5,        # bd_retry_success_rate     — neutral 50%
                0.0,        # special_state_frequency   — none yet
                0.5,        # recovery_rate             — neutral 50%
                5,          # fatigue_threshold         — default 5 questions
                "stable",   # error_rate_trend          — stable to start
                "morning",  # best_time_of_day          — default morning
                0,          # total_sessions            — none yet
                str(datetime.now())  # last_updated
            )
        )
        print(f"Profile created for patient {patient_id}.")

    def get_profile(self, patient_id):
        # Retrieves the current profile for a patient.
        # Returns a dictionary of all profile fields.
        cursor = self.db.execute_query(
            "SELECT * FROM profiles WHERE patient_id = ?",
            (patient_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def update_profile(self, patient_id, updated_profile):
        # Overwrites the profile with freshly calculated values.
        # Called by ProfileManager at the end of every session.
        self.db.execute_query(
            """UPDATE profiles SET
               avg_thinking_time       = ?,
               avg_speaking_time       = ?,
               confidence_indicator    = ?,
               preferred_speed         = ?,
               bd_retry_success_rate   = ?,
               special_state_frequency = ?,
               recovery_rate           = ?,
               fatigue_threshold       = ?,
               error_rate_trend        = ?,
               best_time_of_day        = ?,
               total_sessions          = ?,
               last_updated            = ?
               WHERE patient_id        = ?""",
            (
                updated_profile["avg_thinking_time"],
                updated_profile["avg_speaking_time"],
                updated_profile["confidence_indicator"],
                updated_profile["preferred_speed"],
                updated_profile["bd_retry_success_rate"],
                updated_profile["special_state_frequency"],
                updated_profile["recovery_rate"],
                updated_profile["fatigue_threshold"],
                updated_profile["error_rate_trend"],
                updated_profile["best_time_of_day"],
                updated_profile["total_sessions"],
                str(datetime.now()),
                patient_id
            )
        )
        print(f"Profile updated for patient {patient_id}.")


# ============================================================
# SECTION 5: SPECIAL EVENTS TABLE
# Records real-time mid-session events immediately as they happen.
# ============================================================

class SpecialEventTable:

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def log_event(self, patient_id, session_id, event_type, metadata):
        # Writes a single event row immediately when it occurs.
        # metadata is a dictionary — stored as a JSON string.
        self.db.execute_query(
            "INSERT INTO special_events (patient_id, session_id, event_type, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                patient_id,
                session_id,
                event_type,
                str(datetime.now()),
                json.dumps(metadata)   # convert metadata dict to string
            )
        )

    def get_events_by_session(self, session_id):
        # Returns all events that occurred in a given session.
        cursor = self.db.execute_query(
            "SELECT * FROM special_events WHERE session_id = ?",
            (session_id,)
        )
        rows = cursor.fetchall()
        events = []
        for row in rows:
            row_dict = dict(row)
            row_dict["metadata"] = json.loads(row_dict["metadata"])
            events.append(row_dict)
        return events

    def get_events_by_type(self, patient_id, event_type):
        # Returns all events of a specific type for a patient
        # across all sessions. Useful for spotting patterns.
        cursor = self.db.execute_query(
            "SELECT * FROM special_events WHERE patient_id = ? AND event_type = ?",
            (patient_id, event_type)
        )
        rows = cursor.fetchall()
        events = []
        for row in rows:
            row_dict = dict(row)
            row_dict["metadata"] = json.loads(row_dict["metadata"])
            events.append(row_dict)
        return events


# ============================================================
# SECTION 6: PROFILE MANAGER
# Analyses all past session data and recalculates the patient
# profile after each session ends. This is the learning layer.
# ============================================================

class ProfileManager:

    def __init__(self, session_table: SessionTable,
                 profile_table: ProfileTable,
                 event_table: SpecialEventTable):
        self.sessions = session_table
        self.profiles = profile_table
        self.events = event_table

    def recalculate_profile(self, patient_id):
        # Master method — called once at session end.
        # Pulls all sessions, runs all analysis, saves updated profile.

        all_sessions = self.sessions.get_sessions_by_patient(patient_id)

        # Need at least one session to calculate anything
        if len(all_sessions) == 0:
            return

        # Run all individual analysis methods
        avg_thinking    = self._calculate_avg_thinking_time(all_sessions)
        avg_speaking    = self._calculate_avg_speaking_time(all_sessions)
        confidence      = self._calculate_confidence_indicator(all_sessions)
        speed           = self._calculate_preferred_speed(all_sessions)
        bd_retry_rate   = self._calculate_bd_retry_success_rate(all_sessions)
        special_freq    = self._calculate_special_state_frequency(all_sessions)
        recovery        = self._calculate_recovery_rate(all_sessions)
        fatigue         = self._calculate_fatigue_threshold(all_sessions)
        trend           = self._calculate_error_rate_trend(all_sessions)
        best_time       = self._detect_time_of_day_trend(all_sessions)

        # Bundle everything into one dictionary
        updated_profile = {
            "avg_thinking_time"       : avg_thinking,
            "avg_speaking_time"       : avg_speaking,
            "confidence_indicator"    : confidence,
            "preferred_speed"         : speed,
            "bd_retry_success_rate"   : bd_retry_rate,
            "special_state_frequency" : special_freq,
            "recovery_rate"           : recovery,
            "fatigue_threshold"       : fatigue,
            "error_rate_trend"        : trend,
            "best_time_of_day"        : best_time,
            "total_sessions"          : len(all_sessions)
        }

        # Save the updated profile back to the database
        self.profiles.update_profile(patient_id, updated_profile)

    # ── Helper: collect all turns across all sessions ──────────────────
    def _get_all_turns(self, sessions):
        # Walks through every session → every question → every turn
        # and returns them all as a flat list.
        turns = []
        for session in sessions:
            full = session["full_session_json"]
            for question in full.get("questions", []):
                for turn in question.get("turns", []):
                    turns.append(turn)
        return turns

    # ── Individual analysis methods ─────────────────────────────────────

    def _calculate_avg_thinking_time(self, sessions):
        # Average time before the patient started speaking across all turns.
        turns = self._get_all_turns(sessions)
        times = [t["thinking_time"] for t in turns if "thinking_time" in t]
        if len(times) == 0:
            return 3.0  # return safe default if no data
        return sum(times) / len(times)

    def _calculate_avg_speaking_time(self, sessions):
        # Average duration of patient speech across all turns.
        turns = self._get_all_turns(sessions)
        times = [t["speaking_time"] for t in turns if "speaking_time" in t]
        if len(times) == 0:
            return 2.0
        return sum(times) / len(times)

    def _calculate_confidence_indicator(self, sessions):
        # Derives a confidence score per turn from three signals:
        #   short thinking time  → more confident
        #   long speaking time   → more confident
        #   correct answer       → more confident
        # Returns a score between 0.0 (low) and 1.0 (high).
        turns = self._get_all_turns(sessions)
        if len(turns) == 0:
            return 0.5

        scores = []
        for turn in turns:
            thinking = turn.get("thinking_time", 3.0)
            speaking = turn.get("speaking_time", 2.0)
            correct  = turn.get("correct", 0)

            # Short thinking = higher score (cap at 10s)
            thinking_score = max(0, 1 - (thinking / 10))

            # Long speaking = higher score (cap at 5s)
            speaking_score = min(1, speaking / 5)

            # Correct answer = 1, incorrect = 0
            correct_score = correct

            # Average the three signals equally
            turn_confidence = (thinking_score + speaking_score + correct_score) / 3
            scores.append(turn_confidence)

        return sum(scores) / len(scores)

    def _calculate_preferred_speed(self, sessions):
        # Compares average correct rate at normal speed vs slow speed.
        # Returns whichever speed had the better correct rate.

        normal_scores = []
        slow_scores   = []

        for session in sessions:
            full = session["full_session_json"]

            # Check what speed was used in this session
            # If speed changes happened, use the final speed recorded
            speed_changes = full.get("speed_changes", [])
            used_slow = any(sc["new_speed"] < 1.0 for sc in speed_changes)

            score = full.get("total_score", 0)

            if used_slow:
                slow_scores.append(score)
            else:
                normal_scores.append(score)

        # Compare averages — default to normal speed if no data
        avg_normal = sum(normal_scores) / len(normal_scores) if normal_scores else 0
        avg_slow   = sum(slow_scores)   / len(slow_scores)   if slow_scores   else 0

        if avg_slow > avg_normal:
            return 0.85     # SPEED_SLOW
        return 1.0          # SPEED

    def _calculate_bd_retry_success_rate(self, sessions):
        # Of all the times a breakdown retry was used,
        # how often did it lead to a correct answer?
        turns = self._get_all_turns(sessions)
        retry_turns = [t for t in turns if t.get("bd_retry_used") == True]

        if len(retry_turns) == 0:
            return 0.5  # neutral default

        successful = [t for t in retry_turns if t.get("bd_retry_successful") == True]
        return len(successful) / len(retry_turns)

    def _calculate_special_state_frequency(self, sessions):
        # How often does score==2 (special state) trigger per session on average?
        turns = self._get_all_turns(sessions)
        if len(sessions) == 0:
            return 0.0

        special_count = sum(1 for t in turns if t.get("special_state_triggered") == True)
        return special_count / len(sessions)

    def _calculate_recovery_rate(self, sessions):
        # Of all questions that escalated to breakdown,
        # how many were eventually answered correctly?
        breakdown_questions = []
        for session in sessions:
            full = session["full_session_json"]
            for question in full.get("questions", []):
                if question.get("escalated_to_breakdown") == True:
                    breakdown_questions.append(question)

        if len(breakdown_questions) == 0:
            return 0.5

        recovered = [q for q in breakdown_questions if q.get("final_result") == True]
        return len(recovered) / len(breakdown_questions)

    def _calculate_fatigue_threshold(self, sessions):
        # Looks at which question index rest breaks tend to occur at.
        # Returns the average question number before fatigue sets in.
        rest_points = []
        for session in sessions:
            full = session["full_session_json"]
            speed_changes = full.get("speed_changes", [])
            for sc in speed_changes:
                rest_points.append(sc.get("question_index", 5))

        if len(rest_points) == 0:
            return 5    # default: assume fatigue after 5 questions

        return int(sum(rest_points) / len(rest_points))

    def _calculate_error_rate_trend(self, sessions):
        # Compares error rates from the first half of sessions
        # to the second half to determine direction.
        if len(sessions) < 2:
            return "stable"

        scores = [s["total_score"] for s in sessions]
        mid = len(scores) // 2

        first_half_avg  = sum(scores[:mid]) / mid
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid)

        if second_half_avg > first_half_avg:
            return "improving"
        elif second_half_avg < first_half_avg:
            return "declining"
        return "stable"

    def _detect_time_of_day_trend(self, sessions):
        # Groups sessions by morning vs afternoon
        # and returns whichever has the higher average score.
        morning_scores   = []
        afternoon_scores = []

        for session in sessions:
            score     = session["total_score"]
            time_of_day = session["time_of_day"]

            if time_of_day == "morning":
                morning_scores.append(score)
            else:
                afternoon_scores.append(score)

        avg_morning   = sum(morning_scores)   / len(morning_scores)   if morning_scores   else 0
        avg_afternoon = sum(afternoon_scores) / len(afternoon_scores) if afternoon_scores else 0

        if avg_morning >= avg_afternoon:
            return "morning"
        return "afternoon"


# ============================================================
# SECTION 7: ADAPTATION ENGINE
# Reads the patient profile and decides how the agent
# should behave in the next session.
# ============================================================

class AdaptationEngine:

    def __init__(self, profile_table: ProfileTable):
        self.profiles = profile_table

    def generate_config(self, patient_id):
        # Master method — called at session start.
        # Reads the profile and returns a filled AdaptationConfig.

        profile = self.profiles.get_profile(patient_id)

        # If no profile exists yet, return safe defaults
        if profile is None:
            return AdaptationConfig()

        config = AdaptationConfig()
        config.recommended_speed          = self._determine_speed(profile)
        config.preferred_wait_time        = self._determine_wait_time(profile)
        config.breakdown_sensitivity      = self._determine_breakdown_sensitivity(profile)
        config.max_questions_before_rest  = self._determine_rest_frequency(profile)
        config.bd_retry_budget            = self._determine_bd_retry_budget(profile)
        config.confidence_threshold       = self._determine_confidence_threshold(profile)

        return config

    def _determine_speed(self, profile):
        # Use the speed the patient historically performs best at.
        return profile["preferred_speed"]

    def _determine_wait_time(self, profile):
        # If the patient has a high avg thinking time,
        # give them more wait time before the session starts.
        avg_thinking = profile["avg_thinking_time"]
        if avg_thinking > 5.0:
            return 2    # 2 minutes wait time
        return 1        # 1 minute default

    def _determine_breakdown_sensitivity(self, profile):
        # If recovery rate via breakdown is high,
        # escalate to breakdown sooner (lower threshold).
        recovery_rate = profile["recovery_rate"]
        if recovery_rate > 0.7:
            return 1    # escalate after 1 failed attempt
        return 2        # escalate after 2 failed attempts

    def _determine_rest_frequency(self, profile):
        # Use the fatigue threshold from the profile.
        # If the patient tires quickly, reduce questions before rest.
        return profile["fatigue_threshold"]

    def _determine_bd_retry_budget(self, profile):
        # If retries frequently help this patient, keep the budget at 1.
        # If retries rarely help, set to 0 (skip straight to next question).
        bd_retry_rate = profile["bd_retry_success_rate"]
        if bd_retry_rate >= 0.5:
            return 1    # retries are worth it
        return 0        # retries don't help this patient

    def _determine_confidence_threshold(self, profile):
        # How long to wait for a response before gently prompting.
        # Based on the patient's average thinking time.
        return profile["avg_thinking_time"] * 1.5   # 50% buffer above their average


class AdaptationConfig:
    # Simple container for all the values the agent
    # needs at session start. Defaults are safe neutral values.

    def __init__(self):
        self.recommended_speed          = 1.0   # normal speed
        self.preferred_wait_time        = 1     # 1 minute
        self.breakdown_sensitivity      = 2     # escalate after 2 failures
        self.max_questions_before_rest  = 5     # rest after 5 questions
        self.bd_retry_budget            = 1     # one retry per breakdown question
        self.confidence_threshold       = 4.5   # prompt after 4.5 seconds


# ============================================================
# SECTION 8: EVENT LOGGER
# Fires small immediate writes to the database mid-session.
# Each method matches an event type in SpecialEventTable.
# ============================================================

class EventLogger:

    def __init__(self, db_manager: DatabaseManager,
                 event_table: SpecialEventTable):
        self.db = db_manager
        self.events = event_table

    def flag_emergency(self, patient_id, session_id):
        # Called when emergency == 1 in _handle_state()
        self.events.log_event(patient_id, session_id,
                              "emergency",
                              {"timestamp": str(datetime.now())})

    def flag_rest_request(self, patient_id, session_id, rest_count):
        # Called when stop == 1 in _handle_state()
        self.events.log_event(patient_id, session_id,
                              "rest_requested",
                              {"rest_count": rest_count})

    def flag_speed_change(self, patient_id, session_id,
                          old_speed, new_speed, reason):
        # Called when the agent slows down in encourage()
        self.events.log_event(patient_id, session_id,
                              "speed_changed",
                              {"old_speed": old_speed,
                               "new_speed": new_speed,
                               "reason"   : reason})

    def flag_special_state(self, patient_id, session_id, question_id):
        # Called when score == 2 is returned by the classifier
        self.events.log_event(patient_id, session_id,
                              "special_state",
                              {"question_id": question_id,
                               "timestamp"  : str(datetime.now())})

    def flag_bd_retry_used(self, patient_id, session_id,
                           question_id, result_after_retry):
        # Called when bd_retry_used is consumed in INTERACT_BREAKDOWN
        self.events.log_event(patient_id, session_id,
                              "bd_retry_used",
                              {"question_id"        : question_id,
                               "result_after_retry" : result_after_retry})


# ============================================================
# SECTION 9: PATIENT DATABASE (FACADE)
# The only class that task.py needs to import.
# Wraps everything above into three simple calls:
#   load_session_config()  ← session start
#   save_session()         ← session end
#   flag_event()           ← mid session
# ============================================================

class PatientDatabase:

    def __init__(self, db_path="patients.db"):
        # Create and connect all the internal components
        self.db_manager     = DatabaseManager(db_path)
        self.patients       = PatientTable(self.db_manager)
        self.sessions       = SessionTable(self.db_manager)
        self.profiles       = ProfileTable(self.db_manager)
        self.special_events = SpecialEventTable(self.db_manager)
        self.profile_manager = ProfileManager(
            self.sessions,
            self.profiles,
            self.special_events
        )
        self.adaptation_engine = AdaptationEngine(self.profiles)
        self.event_logger = EventLogger(self.db_manager, self.special_events)

    def initialise(self):
        # Call this once at the very start before doing anything else.
        # Opens the connection and creates tables if needed.
        self.db_manager.connect()
        self.db_manager.initialise_tables()

    def register_patient(self, patient):
        # Adds a new patient to the database and creates their profile.
        # Call this the first time a patient uses the system.
        self.patients.add_patient(patient)
        self.profiles.create_profile(patient.id)

    def load_session_config(self, patient_id):
        # Called at the START of perform_task().
        # Returns an AdaptationConfig the agent uses to configure itself.
        return self.adaptation_engine.generate_config(patient_id)

    def save_session(self, patient_id, session_record):
        # Called at the END of perform_task().
        # Saves the full session then recalculates the patient profile
        # so the next session benefits from what just happened.
        self.sessions.save_session(session_record)
        self.profile_manager.recalculate_profile(patient_id)
        print(f"Session saved and profile updated for patient {patient_id}.")

    def flag_event(self, patient_id, session_id, event_type, metadata):
        # Called MID-SESSION for urgent real-time events.
        # Routes to the correct EventLogger method.

        if event_type == "emergency":
            self.event_logger.flag_emergency(patient_id, session_id)

        elif event_type == "rest_requested":
            self.event_logger.flag_rest_request(
                patient_id, session_id,
                metadata.get("rest_count", 0))

        elif event_type == "speed_changed":
            self.event_logger.flag_speed_change(
                patient_id, session_id,
                metadata.get("old_speed"),
                metadata.get("new_speed"),
                metadata.get("reason"))

        elif event_type == "special_state":
            self.event_logger.flag_special_state(
                patient_id, session_id,
                metadata.get("question_id"))

        elif event_type == "bd_retry_used":
            self.event_logger.flag_bd_retry_used(
                patient_id, session_id,
                metadata.get("question_id"),
                metadata.get("result_after_retry"))

    def close(self):
        # Call this when the program finishes to close the connection cleanly.
        self.db_manager.disconnect()


# ============================================================
# QUICK TEST — runs only when you execute this file directly
# Use this to verify the database is working before connecting
# it to task.py
# ============================================================

if __name__ == "__main__":

    # --- Set up ---
    db = PatientDatabase("test_patients.db")
    db.initialise()

    # --- Simulate adding a patient ---
    class MockPatient:
        def __init__(self):
            self.id       = 1
            self.name     = "Jerry"
            self._address = "Lennon Studio"

    patient = MockPatient()
    db.register_patient(patient)

    # --- Check the profile was created ---
    profile = db.profiles.get_profile(1)
    print("Profile after registration:", profile)

    # --- Load config (should return defaults) ---
    config = db.load_session_config(1)
    print("Config speed:", config.recommended_speed)
    print("Config wait time:", config.preferred_wait_time)

    # --- Simulate flagging a mid-session event ---
    db.flag_event(1, "session_001", "speed_changed",
                  {"old_speed": 1.0, "new_speed": 0.85, "reason": "dont_know"})

    # --- Simulate saving a session ---
    mock_session = {
        "session_id"       : "session_001",
        "patient_id"       : 1,
        "task_name"        : "Orientation_task",
        "date"             : str(datetime.now().date()),
        "time_of_day"      : "morning",
        "final_state"      : "FINISH",
        "total_score"      : 1,
        "session_duration" : 120.5,
        "rest_breaks_taken": 0,
        "speed_changes"    : [{"question_index": 0, "new_speed": 0.85, "reason": "dont_know"}],
        "reset_required"   : False,
        "questions"        : [
            {
                "question_id"            : 0,
                "question_text"          : "What is today's date?",
                "expected_answer"        : "Thursday March 6th 2026",
                "escalated_to_breakdown" : False,
                "final_result"           : True,
                "which_sub_q_failed"     : [],
                "turns"                  : [
                    {
                        "thinking_time"           : 2.5,
                        "speaking_time"           : 3.1,
                        "classifier_score"        : 1,
                        "special_state_triggered" : False,
                        "correct"                 : 1,
                        "action_taken"            : "correct",
                        "bd_retry_used"           : False,
                        "bd_retry_successful"     : False
                    }
                ]
            }
        ]
    }

    db.save_session(1, mock_session)

    # --- Check profile was updated ---
    updated_profile = db.profiles.get_profile(1)
    print("Profile after session:", updated_profile)

    # --- Close ---
    db.close()
