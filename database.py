import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "rfp_evaluation.db"

DEFAULT_CRITERIA = [
    (1, "Technical Capability", "Architecture, integrations, scalability, and technical fit.", 30.0, 10.0, 1),
    (2, "Implementation Plan", "Timeline, milestones, staffing, dependencies, and risk plan.", 20.0, 10.0, 1),
    (3, "Commercial Value", "Pricing clarity, total cost, assumptions, and value for money.", 20.0, 10.0, 1),
    (4, "Security & Compliance", "Security controls, certifications, privacy, and auditability.", 20.0, 10.0, 1),
    (5, "Support & Experience", "Support model, similar projects, references, and service levels.", 10.0, 10.0, 1),
]

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_criteria (
                criterion_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                weight REAL NOT NULL,
                max_score REAL NOT NULL DEFAULT 10,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfp_runs (
                rfp_run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS supplier_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rfp_run_id TEXT NOT NULL,
                supplier_name TEXT NOT NULL,
                source_filename TEXT,
                submission_date TEXT NOT NULL,
                experience_rating REAL NOT NULL,
                absolute_score REAL NOT NULL,
                ppi REAL NOT NULL,
                final_rank INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs(rfp_run_id)
            )
        """)
        count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if count == 0:
            conn.executemany("""
                INSERT INTO evaluation_criteria
                (criterion_id, name, description, weight, max_score, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, DEFAULT_CRITERIA)

def get_active_criteria():
    with connect() as conn:
        rows = conn.execute("""
            SELECT criterion_id, name, description, weight, max_score, is_active
            FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id
        """).fetchall()
    return [dict(r) for r in rows]

def save_run(results, criteria):
    run_id = "RFP-" + uuid.uuid4().hex[:10].upper()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute("INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
                     (run_id, now, "completed"))
        for result in results:
            conn.execute("""
                INSERT INTO supplier_results
                (rfp_run_id, supplier_name, source_filename, submission_date,
                 experience_rating, absolute_score, ppi, final_rank, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, result["supplier_name"], result.get("source_filename", ""),
                  result["submission_date"], result["experience_rating"],
                  result["absolute_score"], result["ppi"], result["final_rank"],
                  json.dumps(result, ensure_ascii=False)))
    return run_id

def get_run_results(run_id):
    with connect() as conn:
        rows = conn.execute("SELECT result_json FROM supplier_results WHERE rfp_run_id=? ORDER BY final_rank", (run_id,)).fetchall()
    return [json.loads(row["result_json"]) for row in rows]
