"""
analytics.py — SQLite logging layer for the Maternal Breastfeeding Support tool.

Records every conversation turn with:
  - session metadata (id, timestamp, route taken)
  - semantic scores (pain, latch, supply, stress, urgency)
  - retrieval quality (top score returned by FAISS)
  - message metadata (length, turn number, baby age known)

Used by dashboard.py to power the Streamlit analytics dashboard.

Usage:
    from analytics import log_turn, init_db
    init_db()                    # call once at app startup
    log_turn(session_id, ...)    # call in /submit route after routing
"""

import sqlite3
import uuid
from datetime import datetime
import os

DB_PATH = os.environ.get("ANALYTICS_DB", "analytics.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist, safely handling duplicates."""
    conn = get_conn()
    cursor = conn.cursor()
    
    # Create sessions table safely
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            started_at      TEXT NOT NULL,
            ended_at        TEXT,
            total_turns     INTEGER DEFAULT 0,
            had_urgent      INTEGER DEFAULT 0,
            had_support     INTEGER DEFAULT 0,
            routes_taken    TEXT
        );
    """)
    
    # Create turns table safely
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            turn_number     INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            route           TEXT NOT NULL,
            detail_level    TEXT,
            conv_state      TEXT,
            score_pain      REAL,
            score_latch     REAL,
            score_supply    REAL,
            score_stress    REAL,
            score_urgency   REAL,
            flag_pain       INTEGER,
            flag_latch      INTEGER,
            flag_supply     INTEGER,
            flag_stress     INTEGER,
            flag_urgency    INTEGER,
            retrieval_hit   INTEGER,
            retrieval_top_score REAL,
            user_msg_len    INTEGER,
            baby_age_known  INTEGER,
            is_closing      INTEGER,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );
    """)
    
    conn.commit()
    conn.close()

def ensure_session(session_id: str):
    """Create a session row if it doesn't exist yet."""
    conn = get_conn()
    exists = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO sessions (session_id, started_at, routes_taken) VALUES (?, ?, ?)",
            (session_id, datetime.utcnow().isoformat(), "[]")
        )
        conn.commit()
    conn.close()


def log_turn(
    session_id: str,
    turn_number: int,
    route: str,
    scores: dict,           # raw cosine similarity scores (floats)
    flags: dict,            # triggered flags (0/1)
    detail_level: str,
    conv_state: str,
    retrieval_chunks: list, # list of retrieved text chunks
    retrieval_scores: list, # list of FAISS scores (floats), can be empty
    user_msg_len: int,
    baby_age_known: bool,
    is_closing: bool,
):
    """
    Log a single conversation turn to the database.

    Call this from app.py's /submit route after routing and retrieval,
    before generating the LLM response.
    """
    ensure_session(session_id)

    ts = datetime.utcnow().isoformat()
    retrieval_hit = 1 if retrieval_chunks else 0
    retrieval_top = max(retrieval_scores) if retrieval_scores else None

    conn = get_conn()

    conn.execute("""
        INSERT INTO turns (
            session_id, turn_number, timestamp,
            route, detail_level, conv_state,
            score_pain, score_latch, score_supply, score_stress, score_urgency,
            flag_pain, flag_latch, flag_supply, flag_stress, flag_urgency,
            retrieval_hit, retrieval_top_score,
            user_msg_len, baby_age_known, is_closing
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?
        )
    """, (
        session_id, turn_number, ts,
        route, detail_level, conv_state,
        scores.get("pain"), scores.get("latch"),
        scores.get("supply"), scores.get("stress"), scores.get("urgency"),
        flags.get("pain", 0), flags.get("latch", 0),
        flags.get("supply", 0), flags.get("stress", 0), flags.get("urgency", 0),
        retrieval_hit, retrieval_top,
        user_msg_len, int(baby_age_known), int(is_closing),
    ))

    # Update session summary
    import json
    session = conn.execute(
        "SELECT routes_taken, had_urgent, had_support FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()

    routes = json.loads(session["routes_taken"])
    routes.append(route)

    had_urgent = session["had_urgent"] or int("URGENT" in route)
    had_support = session["had_support"] or int(route == "SUPPORT")

    conn.execute("""
        UPDATE sessions SET
            total_turns = ?,
            ended_at = ?,
            had_urgent = ?,
            had_support = ?,
            routes_taken = ?
        WHERE session_id = ?
    """, (turn_number, ts, had_urgent, had_support, json.dumps(routes), session_id))

    conn.commit()
    conn.close()


def seed_demo_data():
    """
    Populate the database with realistic demo data so the dashboard
    looks meaningful even before real users interact with the app.

    Run once: python analytics.py
    """
    import json
    import random
    from datetime import timedelta

    init_db()

    routes = ["CLINICAL", "SUPPORT", "URGENT_INFANT", "URGENT_MATERNAL", "QUESTION_FIRST", "CLOSING", "REASSURE"]
    route_weights = [0.35, 0.20, 0.04, 0.04, 0.18, 0.10, 0.09]

    detail_levels = ["DETAILED", "VAGUE", "NEUTRAL"]
    conv_states = ["NEW", "ACTIVE_BREASTFEEDING_THREAD", "OTHER"]

    now = datetime.utcnow()

    for i in range(120):  
        session_id = str(uuid.uuid4())
        session_start = now - timedelta(days=random.randint(0, 30),
                                        hours=random.randint(0, 23),
                                        minutes=random.randint(0, 59))
        num_turns = random.randint(1, 8)

        conn = get_conn()
        conn.execute(
            "INSERT INTO sessions (session_id, started_at, routes_taken) VALUES (?, ?, ?)",
            (session_id, session_start.isoformat(), "[]")
        )
        conn.commit()
        conn.close()

        session_routes = []
        had_urgent = 0
        had_support = 0

        for turn in range(1, num_turns + 1):
            route = random.choices(routes, weights=route_weights)[0]
            session_routes.append(route)
            if "URGENT" in route: had_urgent = 1
            if route == "SUPPORT": had_support = 1

            ts = (session_start + timedelta(minutes=turn * random.randint(1, 5))).isoformat()

            pain_score = random.uniform(0.15, 0.65)
            latch_score = random.uniform(0.10, 0.55)
            supply_score = random.uniform(0.10, 0.50)
            stress_score = random.uniform(0.10, 0.55)
            urgency_score = random.uniform(0.10, 0.60)

            conn = get_conn()
            conn.execute("""
                INSERT INTO turns (
                    session_id, turn_number, timestamp,
                    route, detail_level, conv_state,
                    score_pain, score_latch, score_supply, score_stress, score_urgency,
                    flag_pain, flag_latch, flag_supply, flag_stress, flag_urgency,
                    retrieval_hit, retrieval_top_score,
                    user_msg_len, baby_age_known, is_closing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, turn, ts,
                route,
                random.choice(detail_levels),
                random.choice(conv_states),
                pain_score, latch_score, supply_score, stress_score, urgency_score,
                int(pain_score >= 0.35), int(latch_score >= 0.35),
                int(supply_score >= 0.35), int(stress_score >= 0.25), int(urgency_score >= 0.40),
                random.randint(0, 1), random.uniform(0.3, 0.85),
                random.randint(20, 350),
                random.randint(0, 1), int(route == "CLOSING"),
            ))
            conn.execute("""
                UPDATE sessions SET
                    total_turns = ?, ended_at = ?,
                    had_urgent = ?, had_support = ?,
                    routes_taken = ?
                WHERE session_id = ?
            """, (turn, ts, had_urgent, had_support, json.dumps(session_routes), session_id))
            conn.commit()
            conn.close()

    print(f"✅ Seeded 120 demo sessions into {DB_PATH}")


if __name__ == "__main__":
    seed_demo_data()
