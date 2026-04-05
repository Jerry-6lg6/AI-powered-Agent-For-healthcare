"""
models.py
---------
SQL schema definitions for the HCA patient memory system.
All tables are created here and referenced by db.py.
"""

CREATE_PATIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    date_of_birth TEXT  DEFAULT NULL,
    address     TEXT    DEFAULT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    task_name   TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT    DEFAULT NULL,
    total_score INTEGER DEFAULT 0,
    max_score   INTEGER DEFAULT 0,
    completed   INTEGER DEFAULT 0,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
"""

CREATE_RESPONSES_TABLE = """
CREATE TABLE IF NOT EXISTS responses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    question_name    TEXT    NOT NULL,
    question_text    TEXT    NOT NULL,
    expected_answer  TEXT    NOT NULL,
    patient_answer   TEXT    NOT NULL,
    is_correct       INTEGER NOT NULL,
    thinking_time    REAL    DEFAULT 0.0,
    speaking_time    REAL    DEFAULT 0.0,
    is_dont_know     INTEGER DEFAULT 0,
    is_stop          INTEGER DEFAULT 0,
    is_emergency     INTEGER DEFAULT 0,
    is_free_talk     INTEGER DEFAULT 0,
    is_silence       INTEGER DEFAULT 0,
    confidence       REAL    DEFAULT 0.0,
    timestamp        TEXT    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""

CREATE_EMERGENCIES_TABLE = """
CREATE TABLE IF NOT EXISTS emergencies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    session_id      INTEGER NOT NULL,
    patient_name    TEXT    NOT NULL,
    trigger_phrase  TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    resolved        INTEGER DEFAULT 0,
    resolved_at     TEXT    DEFAULT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""

CREATE_PATIENT_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS patient_profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL UNIQUE,
    hometown        TEXT    DEFAULT NULL,
    spouse_name     TEXT    DEFAULT NULL,
    children_names  TEXT    DEFAULT NULL,
    occupation      TEXT    DEFAULT NULL,
    hobbies         TEXT    DEFAULT NULL,
    favourite_food  TEXT    DEFAULT NULL,
    favourite_sport TEXT    DEFAULT NULL,
    favourite_show  TEXT    DEFAULT NULL,
    updated_at      TEXT    NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
"""

ALL_TABLES = [
    CREATE_PATIENTS_TABLE,
    CREATE_SESSIONS_TABLE,
    CREATE_RESPONSES_TABLE,
    CREATE_EMERGENCIES_TABLE,
    CREATE_PATIENT_PROFILES_TABLE,
]
