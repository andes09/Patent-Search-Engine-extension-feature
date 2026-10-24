"""SQLite schema and queue operations for the ingestion POC.

SQLite stands in for Postgres.  Every connection uses WAL mode so concurrent
workers wait (busy_timeout) rather than error.  The claim strategy is chosen
once at startup based on the SQLite version:
  - 3.35+  : single atomic  UPDATE … RETURNING
  - older  : BEGIN IMMEDIATE + SELECT + UPDATE  (serialised via the write lock)
"""
import random
import sqlite3
import time
from pathlib import Path

# ── State machine ────────────────────────────────────────────────────────────
# ready states (claimable)
READY_STATES: tuple[str, ...] = ("pending", "parsed", "embedded")
# in-progress states
INPROGRESS_STATES: tuple[str, ...] = ("parsing", "embedding", "indexing")
# in-progress → previous ready state (used on failure / lease recovery)
PREV_READY: dict[str, str] = {
    "parsing": "pending",
    "embedding": "parsed",
    "indexing": "embedded",
}
# ready → in-progress (used on claim)
CLAIM_STATE: dict[str, str] = {
    "pending": "parsing",
    "parsed": "embedding",
    "embedded": "indexing",
}
# in-progress → next state on success
NEXT_STATE: dict[str, str] = {
    "parsing": "parsed",
    "embedding": "embedded",
    "indexing": "indexed",
}

_use_returning: bool | None = None


def _get_use_returning() -> bool:
    global _use_returning
    if _use_returning is None:
        ver = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
        _use_returning = ver >= (3, 35)
    return _use_returning


def sqlite_claim_method() -> str:
    return "UPDATE...RETURNING" if _get_use_returning() else "BEGIN IMMEDIATE"


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode and a 5-second busy timeout."""
    conn = sqlite3.connect(
        str(db_path), timeout=10, check_same_thread=False, isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    # executescript issues an implicit COMMIT first, then runs DDL
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id              INTEGER PRIMARY KEY,
            doc_number      TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            state           TEXT NOT NULL DEFAULT 'pending',
            attempts        INTEGER NOT NULL DEFAULT 0,
            max_attempts    INTEGER NOT NULL DEFAULT 4,
            next_attempt_at REAL    NOT NULL DEFAULT 0,
            last_error      TEXT,
            worker_id       TEXT,
            claimed_at      REAL,
            updated_at      REAL    NOT NULL DEFAULT 0,
            UNIQUE(doc_number, content_hash)
        );
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY,
            job_id      INTEGER NOT NULL,
            stage       TEXT    NOT NULL,
            worker_id   TEXT,
            outcome     TEXT    NOT NULL,
            started_at  REAL    NOT NULL,
            finished_at REAL    NOT NULL,
            error       TEXT
        );
    """)


def claim_job(
    conn: sqlite3.Connection, worker_id: str, now: float
) -> sqlite3.Row | None:
    """Atomically claim one ready job whose next_attempt_at <= now.

    Returns the updated row (state already set to the in-progress value),
    or None if no job is available.
    """
    ph = ",".join("?" * len(READY_STATES))
    if _get_use_returning():
        # Single statement: subquery + update + return are one atomic unit.
        return conn.execute(
            f"""
            UPDATE jobs SET
                state = CASE state
                    WHEN 'pending'  THEN 'parsing'
                    WHEN 'parsed'   THEN 'embedding'
                    WHEN 'embedded' THEN 'indexing'
                END,
                worker_id = ?, claimed_at = ?, updated_at = ?
            WHERE id = (
                SELECT id FROM jobs
                WHERE state IN ({ph}) AND next_attempt_at <= ?
                ORDER BY next_attempt_at LIMIT 1
            )
            RETURNING *
            """,
            (worker_id, now, now, *READY_STATES, now),
        ).fetchone()

    # Older SQLite: BEGIN IMMEDIATE serialises concurrent workers via a
    # write lock; no other writer can sneak in between SELECT and UPDATE.
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            f"SELECT * FROM jobs WHERE state IN ({ph}) AND next_attempt_at <= ?"
            " ORDER BY next_attempt_at LIMIT 1",
            (*READY_STATES, now),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        conn.execute(
            "UPDATE jobs SET state=?, worker_id=?, claimed_at=?, updated_at=?"
            " WHERE id=?",
            (CLAIM_STATE[row["state"]], worker_id, now, now, row["id"]),
        )
        conn.execute("COMMIT")
        return conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def complete_job(
    conn: sqlite3.Connection,
    job_id: int,
    in_progress: str,
    worker_id: str,
    started_at: float,
) -> None:
    """Advance job to its next state and append a success event."""
    now = time.time()
    conn.execute("BEGIN")
    conn.execute(
        "UPDATE jobs SET state=?, attempts=0, last_error=NULL, updated_at=?"
        " WHERE id=?",
        (NEXT_STATE[in_progress], now, job_id),
    )
    conn.execute(
        "INSERT INTO events (job_id, stage, worker_id, outcome, started_at, finished_at)"
        " VALUES (?,?,?,?,?,?)",
        (job_id, in_progress, worker_id, "success", started_at, now),
    )
    conn.execute("COMMIT")


def fail_job(
    conn: sqlite3.Connection,
    job_id: int,
    in_progress: str,
    error: str,
    worker_id: str,
    started_at: float,
    base: float = 0.1,
) -> None:
    """Increment attempts; move to dead or back to ready with exponential backoff."""
    now = time.time()
    row = conn.execute(
        "SELECT attempts, max_attempts FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    attempts = row["attempts"] + 1
    if attempts >= row["max_attempts"]:
        new_state, next_at = "dead", 0.0
    else:
        new_state = PREV_READY[in_progress]
        # base * 2^attempts gives exponential growth; jitter spreads retry storms
        next_at = now + base * (2 ** attempts) + random.uniform(0, base)
    conn.execute("BEGIN")
    conn.execute(
        "UPDATE jobs SET state=?, attempts=?, last_error=?,"
        " next_attempt_at=?, updated_at=? WHERE id=?",
        (new_state, attempts, error, next_at, now, job_id),
    )
    conn.execute(
        "INSERT INTO events"
        " (job_id, stage, worker_id, outcome, started_at, finished_at, error)"
        " VALUES (?,?,?,?,?,?,?)",
        (job_id, in_progress, worker_id, "failure", started_at, now, error),
    )
    conn.execute("COMMIT")


def recover_leases(conn: sqlite3.Connection, lease_secs: float = 3.0) -> int:
    """Return jobs stuck in-progress longer than lease_secs to their ready state.

    Counts as one attempt (the crashed worker consumed a slot).
    Returns the number of jobs recovered.
    """
    now = time.time()
    ph = ",".join("?" * len(INPROGRESS_STATES))
    rows = conn.execute(
        f"SELECT id, state FROM jobs WHERE state IN ({ph}) AND claimed_at <= ?",
        (*INPROGRESS_STATES, now - lease_secs),
    ).fetchall()
    if not rows:
        return 0
    conn.execute("BEGIN")
    for r in rows:
        conn.execute(
            "UPDATE jobs SET state=?, attempts=attempts+1, worker_id=NULL,"
            " claimed_at=NULL, next_attempt_at=?, updated_at=? WHERE id=?",
            (PREV_READY[r["state"]], now, now, r["id"]),
        )
        # Record in the audit log so the status report can count recoveries
        conn.execute(
            "INSERT INTO events"
            " (job_id, stage, worker_id, outcome, started_at, finished_at)"
            " VALUES (?,?,'supervisor','lease_recovered',?,?)",
            (r["id"], r["state"], now - lease_secs, now),
        )
    conn.execute("COMMIT")
    return len(rows)


def all_done(conn: sqlite3.Connection) -> bool:
    """True when every job is either indexed or dead."""
    n = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE state NOT IN ('indexed','dead')"
    ).fetchone()[0]
    return n == 0
