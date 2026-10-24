"""Tests for poc/ ingestion job queue.

Every test that touches the database uses a temp path so tests are isolated.
Integration tests that need real multiprocessing spawn subprocesses with
--db pointing at the temp path, so nothing touches poc/queue.db.
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

from poc.db import (
    INPROGRESS_STATES,
    PREV_READY,
    all_done,
    claim_job,
    complete_job,
    connect,
    create_schema,
    fail_job,
    recover_leases,
)

ROOT = Path(__file__).parent.parent


# ── Fixtures & helpers ────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh database for unit tests."""
    conn = connect(tmp_path / "test.db")
    create_schema(conn)
    return conn


def _insert(conn, doc_number: str, content_hash: str = "h0",
            state: str = "pending") -> int:
    """Insert one job and return its id. isolation_level=None → autocommit."""
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO jobs"
        " (doc_number, content_hash, state, next_attempt_at, updated_at)"
        " VALUES (?,?,?,?,?)",
        (doc_number, content_hash, state, now, now),
    )
    return conn.execute(
        "SELECT id FROM jobs WHERE doc_number=? AND content_hash=?",
        (doc_number, content_hash),
    ).fetchone()["id"]


def _make_claimable(conn, job_id: int) -> None:
    """Set next_attempt_at to the past so the job is immediately claimable."""
    conn.execute(
        "UPDATE jobs SET next_attempt_at=? WHERE id=?",
        (time.time() - 1, job_id),
    )


# ── Seeding ───────────────────────────────────────────────────────────────────

def test_seed_idempotent(db):
    """INSERT OR IGNORE: same (doc_number, content_hash) never creates a second row."""
    _insert(db, "001", "aaa")
    _insert(db, "001", "aaa")   # duplicate → ignored
    count = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE doc_number='001'"
    ).fetchone()[0]
    assert count == 1


def test_mutate_creates_new_job(db):
    """A changed content_hash produces a new row, leaving the original intact."""
    _insert(db, "001", "hash_original")
    _insert(db, "001", "hash_mutated")   # different hash → new row
    count = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE doc_number='001'"
    ).fetchone()[0]
    assert count == 2


# ── State transitions ─────────────────────────────────────────────────────────

def test_claim_advances_state(db):
    """Claiming a pending job moves it to parsing and sets worker_id."""
    _insert(db, "doc1")
    job = claim_job(db, "w0", time.time())
    assert job is not None
    assert job["state"] == "parsing"
    assert job["worker_id"] == "w0"


def test_full_pipeline_indexed(db):
    """A job succeeds through all three stages and ends up indexed."""
    _insert(db, "doc1")
    for stage in ("parsing", "embedding", "indexing"):
        job = claim_job(db, "w0", time.time())
        assert job is not None, f"expected job for stage {stage}"
        assert job["state"] == stage
        complete_job(db, job["id"], stage, "w0", time.time())
    state = db.execute("SELECT state FROM jobs").fetchone()[0]
    assert state == "indexed"


def test_failure_returns_to_ready(db):
    """A single failure returns the job to its ready state with attempts=1."""
    jid = _insert(db, "doc1")
    t = time.time()
    job = claim_job(db, "w0", t)
    assert job is not None
    fail_job(db, jid, "parsing", "boom", "w0", t)
    row = db.execute("SELECT state, attempts FROM jobs WHERE id=?", (jid,)).fetchone()
    assert row["state"] == "pending"
    assert row["attempts"] == 1


def test_backoff_increases_with_attempts(db):
    """Each successive failure produces a larger next_attempt_at delay."""
    jid = _insert(db, "doc1")
    delays = []
    for _ in range(3):
        _make_claimable(db, jid)
        t = time.time()
        job = claim_job(db, "w0", t)
        assert job is not None
        fail_job(db, jid, "parsing", "err", "w0", t)
        row = db.execute("SELECT next_attempt_at FROM jobs WHERE id=?", (jid,)).fetchone()
        delays.append(row["next_attempt_at"] - t)
    # Exponential backoff: each delay must exceed the previous
    assert delays[1] > delays[0], f"delays did not grow: {delays}"
    assert delays[2] > delays[1], f"delays did not grow: {delays}"


def test_dead_after_max_attempts(db):
    """A job that keeps failing reaches dead state after max_attempts and stores last_error."""
    jid = _insert(db, "doc1")
    for _ in range(4):   # default max_attempts = 4
        _make_claimable(db, jid)
        t = time.time()
        job = claim_job(db, "w0", t)
        assert job is not None, "expected a claimable job"
        fail_job(db, jid, "parsing", "always fails", "w0", t)
    row = db.execute("SELECT state, last_error FROM jobs WHERE id=?", (jid,)).fetchone()
    assert row["state"] == "dead"
    assert row["last_error"] == "always fails"


# ── No double-claiming ────────────────────────────────────────────────────────

def test_no_double_claim(db):
    """Once a job is in-progress, a second claim attempt gets None (or a different job)."""
    _insert(db, "doc1")
    job1 = claim_job(db, "w0", time.time())
    assert job1 is not None
    # The only job is now in-progress; no other ready job exists
    job2 = claim_job(db, "w1", time.time())
    assert job2 is None, "two workers claimed the same job"


def test_no_overlapping_events(db):
    """In events, no two attempts on the same (job, stage) overlap in time."""
    jid = _insert(db, "doc1")
    # Attempt 1: fail
    _make_claimable(db, jid)
    t1 = time.time()
    job = claim_job(db, "w0", t1)
    assert job is not None
    fail_job(db, jid, "parsing", "err", "w0", t1)
    # Attempt 2: succeed
    _make_claimable(db, jid)
    t2 = time.time()
    job = claim_job(db, "w0", t2)
    assert job is not None
    complete_job(db, jid, "parsing", "w0", t2)
    events = db.execute(
        "SELECT started_at, finished_at FROM events"
        " WHERE job_id=? AND stage='parsing' ORDER BY started_at",
        (jid,),
    ).fetchall()
    assert len(events) == 2
    # finished_at of attempt 1 must not exceed started_at of attempt 2
    assert events[0]["finished_at"] <= events[1]["started_at"], (
        f"events overlap: attempt1 ended {events[0]['finished_at']:.6f},"
        f" attempt2 started {events[1]['started_at']:.6f}"
    )


# ── Lease recovery ────────────────────────────────────────────────────────────

def test_lease_recovery(db):
    """A job stuck in-progress past the lease is returned to its ready state."""
    jid = _insert(db, "doc1")
    # Simulate a crashed worker that took the job to "embedding"
    old_claimed = time.time() - 10   # 10s ago, well past the 3s lease
    db.execute(
        "UPDATE jobs SET state='embedding', worker_id='dead', claimed_at=? WHERE id=?",
        (old_claimed, jid),
    )
    recovered = recover_leases(db, lease_secs=3.0)
    assert recovered == 1
    row = db.execute("SELECT state, attempts FROM jobs WHERE id=?", (jid,)).fetchone()
    assert row["state"] == PREV_READY["embedding"]   # "parsed"
    assert row["attempts"] == 1
    # The event log must record the recovery
    ev = db.execute(
        "SELECT outcome FROM events WHERE job_id=? AND outcome='lease_recovered'",
        (jid,),
    ).fetchone()
    assert ev is not None
    # After recovery the job can be claimed and completed
    _make_claimable(db, jid)
    job = claim_job(db, "w0", time.time())
    assert job is not None
    assert job["state"] == "embedding"


def test_lease_does_not_fire_before_expiry(db):
    """Jobs claimed recently must not be recovered."""
    jid = _insert(db, "doc1")
    db.execute(
        "UPDATE jobs SET state='parsing', worker_id='w0', claimed_at=? WHERE id=?",
        (time.time(), jid),   # just now
    )
    recovered = recover_leases(db, lease_secs=3.0)
    assert recovered == 0


# ── Poison docs ───────────────────────────────────────────────────────────────

def test_poison_dead_with_error(db):
    """A poison doc always fails at embedding; after max_attempts it is dead."""
    jid = _insert(db, "poison_doc")
    # Advance through parsing successfully
    job = claim_job(db, "w0", time.time())
    assert job["state"] == "parsing"
    complete_job(db, jid, "parsing", "w0", time.time())
    # Poison: always fails at embedding (4 attempts → dead)
    for _ in range(4):
        _make_claimable(db, jid)
        job = claim_job(db, "w0", time.time())
        assert job is not None and job["state"] == "embedding"
        fail_job(db, jid, "embedding", "poison: permanent failure", "w0", time.time())
    row = db.execute("SELECT state, last_error FROM jobs WHERE id=?", (jid,)).fetchone()
    assert row["state"] == "dead"
    assert row["last_error"] is not None


# ── Integration: full pipeline via subprocess ─────────────────────────────────

def _run_queue(db_path: Path, *args: str, timeout: int = 120) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "poc.queue", "--db", str(db_path)] + list(args),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(
            f"poc.queue {' '.join(args)} failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def test_run_all_done(tmp_path):
    """After run, every job is indexed or dead; nothing stuck in-progress."""
    db_path = tmp_path / "run.db"
    _run_queue(db_path, "init")

    # Insert 30 lightweight synthetic jobs directly — faster than seeding 640 patents
    conn = connect(db_path)
    now = time.time()
    for i in range(30):
        conn.execute(
            "INSERT INTO jobs (doc_number, content_hash, state, next_attempt_at, updated_at)"
            " VALUES (?,?,'pending',?,?)",
            (f"doc{i:04d}", f"hash{i}", now, now),
        )

    # No failures so the run is fast and deterministic
    _run_queue(db_path, "run", "--workers", "2",
               "--fail-rate", "0", "--poison-rate", "0", "--crash-rate", "0")

    conn2 = connect(db_path)
    stuck = conn2.execute(
        "SELECT state, COUNT(*) n FROM jobs"
        " WHERE state NOT IN ('indexed','dead') GROUP BY state"
    ).fetchall()
    assert not stuck, f"jobs still in-progress: {[(r['state'], r['n']) for r in stuck]}"

    indexed = conn2.execute(
        "SELECT COUNT(*) FROM jobs WHERE state='indexed'"
    ).fetchone()[0]
    assert indexed == 30


def test_run_leaves_nothing_inprogress(tmp_path):
    """With failures enabled, everything still ends as indexed or dead — never stuck."""
    db_path = tmp_path / "run2.db"
    _run_queue(db_path, "init")

    conn = connect(db_path)
    now = time.time()
    for i in range(20):
        conn.execute(
            "INSERT INTO jobs (doc_number, content_hash, state, next_attempt_at, updated_at)"
            " VALUES (?,?,'pending',?,?)",
            (f"doc{i:04d}", f"hash{i}", now, now),
        )

    _run_queue(db_path, "run", "--workers", "2",
               "--fail-rate", "0.3", "--poison-rate", "0.1",
               "--crash-rate", "0", "--seed", "99",
               timeout=180)

    conn2 = connect(db_path)
    inprog = conn2.execute(
        f"SELECT COUNT(*) FROM jobs WHERE state IN"
        f" {tuple(INPROGRESS_STATES)!r}"
    ).fetchone()[0]
    assert inprog == 0, f"{inprog} jobs left in-progress"
