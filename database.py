"""
SQLite persistence for criteria, runs and results.

Design goals:

* **A run is reproducible from the database alone.** Criteria are mutable, so
  storing only a foreign key to `evaluation_criteria` would mean that editing a
  weight silently rewrites the meaning of every historical ranking. Each run
  therefore snapshots the exact criteria it used into `run_criteria`.
* **A run records what produced it.** Provider, model and temperature are stored
  on the run, because "which model scored this?" is the first question anyone
  asks about an AI-assisted procurement decision.
* **Queryable columns plus a full snapshot.** Ranking metrics live in real
  columns so runs can be sorted and compared in SQL; the complete per-supplier
  result, including evidence and the tool trace, is kept as a JSON document for
  audit.

The schema is created and migrated in place, so an existing `rfp_evaluation.db`
from an earlier version is upgraded rather than discarded.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Default on-disk location, overridable with RFP_DB_PATH (used by the test suite).
DEFAULT_DB_PATH = Path(__file__).parent / "rfp_evaluation.db"
_DB_PATH = Path(os.getenv("RFP_DB_PATH", str(DEFAULT_DB_PATH)))

#: Kept as a module attribute for backwards compatibility with earlier imports.
DB_PATH = _DB_PATH

DEFAULT_CRITERIA: Sequence[tuple] = (
    (1, "Technical Capability", "Architecture, integrations, scalability, and technical fit.", 30.0, 10.0, 1),
    (2, "Implementation Plan", "Timeline, milestones, staffing, dependencies, and risk plan.", 20.0, 10.0, 1),
    (3, "Commercial Value", "Pricing clarity, total cost, assumptions, and value for money.", 20.0, 10.0, 1),
    (4, "Security & Compliance", "Security controls, certifications, privacy, and auditability.", 20.0, 10.0, 1),
    (5, "Support & Experience", "Support model, similar projects, references, and service levels.", 10.0, 10.0, 1),
)


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------


def get_db_path() -> Path:
    return _DB_PATH


def set_db_path(path) -> None:
    """Point the module at a different database file. Used by tests for isolation."""
    global _DB_PATH, DB_PATH
    _DB_PATH = Path(path)
    DB_PATH = _DB_PATH


def connect() -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enforced."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    # SQLite disables foreign key enforcement by default; the FKs in this schema
    # are only meaningful if it is switched on per connection.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session():
    """
    A transactional connection that is always closed.

    ``with sqlite3.connect(...) as conn`` commits the transaction but leaves the
    connection open, which leaks handles and, on Windows, keeps a lock on the
    database file until the object is garbage collected. Every access in this
    module goes through here so a connection is committed on success, rolled
    back on failure, and closed either way.
    """
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _ensure_columns(conn: sqlite3.Connection, table: str, definitions: Dict[str, str]) -> None:
    """Add any missing columns so older database files migrate forward in place."""
    existing = set(_columns(conn, table))
    if not existing:
        return
    for column, definition in definitions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def init_db(seed: bool = True) -> None:
    """Create or migrate every table, and seed the default criteria once."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with session() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_criteria (
                criterion_id INTEGER PRIMARY KEY,
                name         TEXT NOT NULL,
                description  TEXT NOT NULL DEFAULT '',
                weight       REAL NOT NULL,
                max_score    REAL NOT NULL DEFAULT 10,
                is_active    INTEGER NOT NULL DEFAULT 1,
                updated_at   TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfp_runs (
                rfp_run_id            TEXT PRIMARY KEY,
                created_at            TEXT NOT NULL,
                status                TEXT NOT NULL,
                provider              TEXT,
                model                 TEXT,
                temperature           REAL,
                supplier_count        INTEGER NOT NULL DEFAULT 0,
                criteria_count        INTEGER NOT NULL DEFAULT 0,
                criteria_weight_total REAL,
                winner                TEXT,
                notes                 TEXT,
                run_json              TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_criteria (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                rfp_run_id   TEXT NOT NULL,
                criterion_id INTEGER NOT NULL,
                name         TEXT NOT NULL,
                description  TEXT NOT NULL DEFAULT '',
                weight       REAL NOT NULL,
                max_score    REAL NOT NULL,
                UNIQUE (rfp_run_id, criterion_id),
                FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs (rfp_run_id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS supplier_results (
                result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                rfp_run_id        TEXT NOT NULL,
                supplier_name     TEXT NOT NULL,
                source_filename   TEXT,
                submission_date   TEXT NOT NULL,
                experience_rating REAL NOT NULL,
                absolute_score    REAL NOT NULL,
                ppi               REAL NOT NULL,
                final_rank        INTEGER NOT NULL,
                evidence_checked  INTEGER NOT NULL DEFAULT 0,
                evidence_verified INTEGER NOT NULL DEFAULT 0,
                grounding_rate    REAL,
                warning_count     INTEGER NOT NULL DEFAULT 0,
                evaluation_failed INTEGER NOT NULL DEFAULT 0,
                result_json       TEXT NOT NULL,
                FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs (rfp_run_id) ON DELETE CASCADE
            )
        """)

        # Forward-migrate databases created by earlier versions.
        _ensure_columns(conn, "evaluation_criteria", {"updated_at": "TEXT"})
        _ensure_columns(conn, "rfp_runs", {
            "provider": "TEXT", "model": "TEXT", "temperature": "REAL",
            "supplier_count": "INTEGER NOT NULL DEFAULT 0",
            "criteria_count": "INTEGER NOT NULL DEFAULT 0",
            "criteria_weight_total": "REAL", "winner": "TEXT",
            "notes": "TEXT", "run_json": "TEXT",
        })
        _ensure_columns(conn, "supplier_results", {
            "evidence_checked": "INTEGER NOT NULL DEFAULT 0",
            "evidence_verified": "INTEGER NOT NULL DEFAULT 0",
            "grounding_rate": "REAL",
            "warning_count": "INTEGER NOT NULL DEFAULT 0",
            "evaluation_failed": "INTEGER NOT NULL DEFAULT 0",
        })

        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON supplier_results (rfp_run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_rank ON supplier_results (rfp_run_id, final_rank)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_criteria_run ON run_criteria (rfp_run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON rfp_runs (created_at DESC)")

        if seed and conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0] == 0:
            conn.executemany(
                """INSERT INTO evaluation_criteria
                   (criterion_id, name, description, weight, max_score, is_active, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [row + (_utc_now(),) for row in DEFAULT_CRITERIA],
            )


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def get_active_criteria() -> List[Dict[str, Any]]:
    with session() as conn:
        rows = conn.execute("""
            SELECT criterion_id, name, description, weight, max_score, is_active
            FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id
        """).fetchall()
    return [dict(row) for row in rows]


def get_all_criteria() -> List[Dict[str, Any]]:
    with session() as conn:
        rows = conn.execute("""
            SELECT criterion_id, name, description, weight, max_score, is_active, updated_at
            FROM evaluation_criteria ORDER BY criterion_id
        """).fetchall()
    return [dict(row) for row in rows]


def total_active_weight() -> float:
    return sum(float(c["weight"]) for c in get_active_criteria())


def upsert_criterion(
    criterion_id: Optional[int],
    name: str,
    description: str,
    weight: float,
    max_score: float = 10.0,
    is_active: bool = True,
) -> int:
    """Insert or update one criterion. Returns its id."""
    name = str(name).strip()
    if not name:
        raise ValueError("A criterion needs a name.")
    weight = float(weight)
    max_score = float(max_score)
    if weight < 0:
        raise ValueError("Weight cannot be negative.")
    if max_score <= 0:
        raise ValueError("Max score must be greater than zero.")

    with session() as conn:
        if criterion_id is None:
            row = conn.execute("SELECT COALESCE(MAX(criterion_id), 0) + 1 AS next FROM evaluation_criteria").fetchone()
            criterion_id = int(row["next"])
        conn.execute("""
            INSERT INTO evaluation_criteria
                (criterion_id, name, description, weight, max_score, is_active, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (criterion_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                weight = excluded.weight,
                max_score = excluded.max_score,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
        """, (int(criterion_id), name, str(description or ""), weight, max_score,
              1 if is_active else 0, _utc_now()))
    return int(criterion_id)


def replace_criteria(criteria: Iterable[Dict[str, Any]]) -> int:
    """Replace the entire criteria table in one transaction. Returns the row count."""
    rows = list(criteria)
    with session() as conn:
        conn.execute("DELETE FROM evaluation_criteria")
        conn.executemany("""
            INSERT INTO evaluation_criteria
                (criterion_id, name, description, weight, max_score, is_active, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [(
            int(row["criterion_id"]),
            str(row["name"]).strip(),
            str(row.get("description", "") or ""),
            float(row["weight"]),
            float(row.get("max_score", 10.0)),
            1 if row.get("is_active", True) else 0,
            _utc_now(),
        ) for row in rows])
    return len(rows)


def set_criterion_active(criterion_id: int, is_active: bool) -> None:
    with session() as conn:
        conn.execute(
            "UPDATE evaluation_criteria SET is_active = ?, updated_at = ? WHERE criterion_id = ?",
            (1 if is_active else 0, _utc_now(), int(criterion_id)),
        )


def delete_criterion(criterion_id: int) -> None:
    with session() as conn:
        conn.execute("DELETE FROM evaluation_criteria WHERE criterion_id = ?", (int(criterion_id),))


def reset_criteria_to_defaults() -> int:
    """Restore the shipped default criteria. Historical runs keep their own snapshots."""
    with session() as conn:
        conn.execute("DELETE FROM evaluation_criteria")
        conn.executemany(
            """INSERT INTO evaluation_criteria
               (criterion_id, name, description, weight, max_score, is_active, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [row + (_utc_now(),) for row in DEFAULT_CRITERIA],
        )
    return len(DEFAULT_CRITERIA)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    return "RFP-" + uuid.uuid4().hex[:10].upper()


def save_run(
    results: Sequence[Dict[str, Any]],
    criteria: Sequence[Dict[str, Any]],
    run_meta: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Persist a complete run: metadata, the criteria snapshot and every result.

    Written in a single transaction so a run is never half-stored. The criteria
    snapshot is what makes the run reproducible after the active criteria change.
    """
    meta = dict(run_meta or {})
    run_id = run_id or new_run_id()
    results = list(results)
    criteria = list(criteria)
    winner = next((r["supplier_name"] for r in results if r.get("final_rank") == 1), None)

    with session() as conn:
        conn.execute("""
            INSERT INTO rfp_runs
                (rfp_run_id, created_at, status, provider, model, temperature,
                 supplier_count, criteria_count, criteria_weight_total, winner, notes, run_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            meta.get("created_at") or _utc_now(),
            meta.get("status", "completed"),
            meta.get("provider"),
            meta.get("model"),
            float(meta["temperature"]) if meta.get("temperature") is not None else None,
            len(results),
            len(criteria),
            sum(float(c["weight"]) for c in criteria) if criteria else None,
            winner,
            meta.get("notes"),
            json.dumps(meta, ensure_ascii=False, default=str),
        ))

        conn.executemany("""
            INSERT INTO run_criteria
                (rfp_run_id, criterion_id, name, description, weight, max_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [(
            run_id,
            int(c["criterion_id"]),
            c["name"],
            str(c.get("description", "") or ""),
            float(c["weight"]),
            float(c["max_score"]),
        ) for c in criteria])

        conn.executemany("""
            INSERT INTO supplier_results
                (rfp_run_id, supplier_name, source_filename, submission_date, experience_rating,
                 absolute_score, ppi, final_rank, evidence_checked, evidence_verified,
                 grounding_rate, warning_count, evaluation_failed, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [_result_row(run_id, result) for result in results])

    return run_id


def _result_row(run_id: str, result: Dict[str, Any]) -> tuple:
    grounding = result.get("evidence_summary") or {}
    return (
        run_id,
        result["supplier_name"],
        result.get("source_filename", ""),
        result["submission_date"],
        float(result["experience_rating"]),
        float(result["absolute_score"]),
        float(result["ppi"]),
        int(result["final_rank"]),
        int(grounding.get("checked") or 0),
        int(grounding.get("verified") or 0),
        float(grounding["grounding_rate"]) if grounding.get("grounding_rate") is not None else None,
        len(result.get("warnings") or []),
        1 if result.get("evaluation_failed") else 0,
        json.dumps(result, ensure_ascii=False, default=str),
    )


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Most recent runs first, for the history view."""
    with session() as conn:
        rows = conn.execute("""
            SELECT rfp_run_id, created_at, status, provider, model, temperature,
                   supplier_count, criteria_count, criteria_weight_total, winner
            FROM rfp_runs ORDER BY created_at DESC, rowid DESC LIMIT ?
        """, (int(limit),)).fetchall()
    return [dict(row) for row in rows]


def get_run_results(run_id: str) -> List[Dict[str, Any]]:
    """Every stored supplier result for a run, in rank order."""
    with session() as conn:
        rows = conn.execute("""
            SELECT result_json FROM supplier_results
            WHERE rfp_run_id = ? ORDER BY final_rank
        """, (run_id,)).fetchall()
    return [json.loads(row["result_json"]) for row in rows]


def get_run_criteria(run_id: str) -> List[Dict[str, Any]]:
    """The criteria snapshot captured when the run executed."""
    with session() as conn:
        rows = conn.execute("""
            SELECT criterion_id, name, description, weight, max_score
            FROM run_criteria WHERE rfp_run_id = ? ORDER BY criterion_id
        """, (run_id,)).fetchall()
    return [dict(row) for row in rows]


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Reload a complete run: metadata, criteria snapshot and results.

    This is the reproducibility guarantee in practice — everything needed to
    re-explain a historical ranking comes back out of the database, with no
    dependence on the criteria currently active.
    """
    with session() as conn:
        row = conn.execute("SELECT * FROM rfp_runs WHERE rfp_run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None

    run = dict(row)
    try:
        run["meta"] = json.loads(run.get("run_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        run["meta"] = {}
    run.pop("run_json", None)

    return {
        "run_id": run_id,
        "run": run,
        "criteria": get_run_criteria(run_id),
        "results": get_run_results(run_id),
    }


def delete_run(run_id: str) -> None:
    """
    Delete a run together with its criteria snapshot and results.

    Children are removed explicitly rather than relying solely on ON DELETE
    CASCADE, so the delete also behaves correctly on a database migrated from an
    earlier version whose foreign keys were declared without a cascade clause.
    """
    with session() as conn:
        conn.execute("DELETE FROM supplier_results WHERE rfp_run_id = ?", (run_id,))
        conn.execute("DELETE FROM run_criteria WHERE rfp_run_id = ?", (run_id,))
        conn.execute("DELETE FROM rfp_runs WHERE rfp_run_id = ?", (run_id,))


if __name__ == "__main__":
    init_db()
    active = get_active_criteria()
    print(f"Database ready at {get_db_path()}")
    print(f"{len(active)} active criteria totalling {sum(float(c['weight']) for c in active):g}% weight")
