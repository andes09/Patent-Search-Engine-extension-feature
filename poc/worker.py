"""Worker process logic — top-level functions are required on Windows.

Windows uses the "spawn" start method: new processes import this module fresh.
Top-level (non-nested) functions can be pickled by name, which is how the
multiprocessing module transfers the target callable to the child process.
"""
import os
import random
import time
from pathlib import Path

from poc.db import (
    all_done,
    claim_job,
    complete_job,
    connect,
    fail_job,
    recover_leases,
)


def worker_loop(
    db_path: str,
    fail_rate: float,
    poison_set: frozenset[str],
    crash_rate: float,
    rng_seed: int,
    worker_id: str,
) -> None:
    """Claim and process jobs until the queue is empty.

    Each worker opens its own SQLite connection — connections must never be
    shared across processes.
    """
    rng = random.Random(rng_seed)
    conn = connect(Path(db_path))

    while True:
        # Check for jobs whose worker crashed before completing
        recover_leases(conn)

        now = time.time()
        job = claim_job(conn, worker_id, now)
        if job is None:
            if all_done(conn):
                break          # nothing left; clean exit
            time.sleep(0.05)   # brief wait before retrying (backoff in effect)
            continue

        job_id = job["id"]
        doc_number = job["doc_number"]
        in_progress = job["state"]   # e.g. "parsing", "embedding", "indexing"
        started_at = time.time()

        # Crash simulation: process exits abruptly mid-job.
        # The supervisor detects exit code != 0 and restarts.
        # Lease recovery will reclaim the abandoned job after the lease expires.
        if rng.random() < crash_rate:
            os._exit(1)

        # Simulate real work (parse / embed / index): 5–30 ms
        time.sleep(rng.uniform(0.005, 0.030))

        # Poison docs always fail at the embedding stage regardless of attempts.
        # They exhaust max_attempts and move to dead with the error preserved.
        is_poison = (doc_number in poison_set) and (in_progress == "embedding")
        if is_poison:
            fail_job(
                conn, job_id, in_progress,
                "poison: permanent failure", worker_id, started_at,
            )
        elif rng.random() < fail_rate:
            fail_job(
                conn, job_id, in_progress,
                "transient error", worker_id, started_at,
            )
        else:
            complete_job(conn, job_id, in_progress, worker_id, started_at)
