"""python -m poc.queue [--db PATH] <init|seed|run|status|retry-dead>

All process-spawning code lives under  if __name__ == "__main__":  so that
Windows' "spawn" start method does not re-execute it when importing the module
in a worker process.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
import time
from pathlib import Path

from poc.db import (
    PREV_READY,
    all_done,
    connect,
    create_schema,
    sqlite_claim_method,
)

_DEFAULT_DB = Path(__file__).parent / "queue.db"
_DATA_DIR = Path(__file__).parent.parent / "data" / "data" / "patent_data_small"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    db = Path(args.db)
    if db.exists():
        db.unlink()
    conn = connect(db)
    create_schema(conn)
    print(f"SQLite {sqlite3.sqlite_version}: using {sqlite_claim_method()}")
    print(f"Initialized {db}")


def _load_patents(mutate_n: int = 0, rng_seed: int = 42) -> list[dict]:
    """Read every patent JSON file; optionally alter N entries to new hashes."""
    import random
    patents: list[dict] = []
    for f in sorted(_DATA_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            patents.extend(json.load(fh))
    if mutate_n > 0:
        rng = random.Random(rng_seed)
        for i in rng.sample(range(len(patents)), min(mutate_n, len(patents))):
            # Adding _mutated changes the JSON, so sha1 differs → new job row
            patents[i] = dict(patents[i], _mutated=True)
    return patents


def cmd_seed(args: argparse.Namespace) -> None:
    patents = _load_patents(mutate_n=args.mutate)
    conn = connect(Path(args.db))
    now = time.time()
    conn.execute("BEGIN")
    new_rows = 0
    for p in patents:
        content_hash = hashlib.sha1(
            json.dumps(p, sort_keys=True).encode()
        ).hexdigest()
        cur = conn.execute(
            "INSERT OR IGNORE INTO jobs"
            " (doc_number, content_hash, state, next_attempt_at, updated_at)"
            " VALUES (?,?,'pending',?,?)",
            (p["doc_number"], content_hash, now, now),
        )
        new_rows += cur.rowcount
    conn.execute("COMMIT")
    print(
        f"Seeded {len(patents)} patents: {new_rows} new job(s), "
        f"{len(patents) - new_rows} unchanged (INSERT OR IGNORE)"
    )


def _print_status(conn: sqlite3.Connection, elapsed: float | None = None) -> None:
    rows = conn.execute(
        "SELECT state, COUNT(*) n FROM jobs GROUP BY state ORDER BY state"
    ).fetchall()
    total = sum(r["n"] for r in rows)
    print(f"\n{'State':<12} {'Count':>6}  {'%':>5}")
    print("-" * 28)
    for r in rows:
        print(f"  {r['state']:<10} {r['n']:>6}  {100 * r['n'] / max(total, 1):>4.1f}%")

    oldest = conn.execute(
        "SELECT MIN(updated_at) FROM jobs WHERE state='pending'"
    ).fetchone()[0]
    if oldest:
        print(f"Oldest pending job: {time.time() - oldest:.1f}s ago")

    if elapsed and elapsed > 0:
        done = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state IN ('indexed','dead')"
        ).fetchone()[0]
        print(f"Throughput: {done / elapsed:.1f} jobs/sec")

    retries = conn.execute(
        "SELECT COUNT(*) FROM events WHERE outcome='failure'"
    ).fetchone()[0]
    leases = conn.execute(
        "SELECT COUNT(*) FROM events WHERE outcome='lease_recovered'"
    ).fetchone()[0]
    print(f"Retries: {retries}   Lease recoveries: {leases}")

    dead = conn.execute(
        "SELECT j.doc_number, j.last_error,"
        "  (SELECT stage FROM events WHERE job_id=j.id AND outcome='failure'"
        "   ORDER BY finished_at DESC LIMIT 1) last_stage"
        " FROM jobs j WHERE j.state='dead'"
    ).fetchall()
    if dead:
        print(f"\nDead-letter queue ({len(dead)} job(s)):")
        for d in dead[:10]:
            print(
                f"  {d['doc_number']}  stage={d['last_stage']}"
                f"  error={d['last_error']}"
            )
        if len(dead) > 10:
            print(f"  ... and {len(dead) - 10} more")


def cmd_status(args: argparse.Namespace) -> None:
    conn = connect(Path(args.db))
    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                _print_status(conn)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        _print_status(conn)


def cmd_retry_dead(args: argparse.Namespace) -> None:
    conn = connect(Path(args.db))
    now = time.time()
    rows = conn.execute(
        "SELECT j.id,"
        "  COALESCE("
        "    (SELECT stage FROM events WHERE job_id=j.id"
        "     ORDER BY finished_at DESC LIMIT 1),"
        "    'parsing'"
        "  ) last_stage"
        " FROM jobs j WHERE j.state='dead'"
    ).fetchall()
    conn.execute("BEGIN")
    for r in rows:
        ready = PREV_READY.get(r["last_stage"], "pending")
        conn.execute(
            "UPDATE jobs SET state=?, attempts=0, last_error=NULL,"
            " next_attempt_at=?, updated_at=? WHERE id=?",
            (ready, now, now, r["id"]),
        )
    conn.execute("COMMIT")
    print(f"Reset {len(rows)} dead job(s) to ready state")


def cmd_run(args: argparse.Namespace) -> None:
    import random
    from poc.worker import worker_loop  # import here so workers don't re-run main

    db_path = str(Path(args.db).resolve())
    conn = connect(Path(db_path))
    print(f"SQLite {sqlite3.sqlite_version}: using {sqlite_claim_method()}")

    # Build the poison set once from a seeded RNG so runs are reproducible
    all_docs = [r[0] for r in conn.execute("SELECT doc_number FROM jobs")]
    n_poison = int(len(all_docs) * args.poison_rate)
    poison_set = frozenset(
        random.Random(args.seed).sample(all_docs, min(n_poison, len(all_docs)))
    )
    print(
        f"Workers: {args.workers}  fail_rate: {args.fail_rate}"
        f"  poison: {len(poison_set)}  crash_rate: {args.crash_rate}"
    )

    def spawn(idx: int) -> multiprocessing.Process:
        p = multiprocessing.Process(
            target=worker_loop,
            args=(db_path, args.fail_rate, poison_set,
                  args.crash_rate, args.seed + idx, f"w{idx}"),
            daemon=True,
        )
        p.start()
        return p

    processes: dict[int, multiprocessing.Process] = {}
    next_id = 0
    for i in range(args.workers):
        processes[i] = spawn(i)
        next_id = i + 1

    start = time.perf_counter()
    while True:
        if all_done(conn):
            break
        # Detect crashed workers (exit code != 0) and restart them
        for idx, p in list(processes.items()):
            if not p.is_alive() and p.exitcode not in (0, None):
                print(f"Worker w{idx} crashed (exit {p.exitcode}), restarting...")
                processes[next_id] = spawn(next_id)
                del processes[idx]
                next_id += 1
        time.sleep(0.2)

    for p in processes.values():
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    elapsed = time.perf_counter() - start
    print(f"\nFinished in {elapsed:.1f}s")
    _print_status(conn, elapsed=elapsed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="poc.queue")
    parser.add_argument(
        "--db", default=str(_DEFAULT_DB), metavar="PATH",
        help="SQLite database file (default: poc/queue.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    sp = sub.add_parser("seed")
    sp.add_argument("--mutate", type=int, default=0, metavar="N",
                    help="alter N patents before hashing to create new jobs")

    rp = sub.add_parser("run")
    rp.add_argument("--workers", type=int, default=4)
    rp.add_argument("--fail-rate", type=float, default=0.2, dest="fail_rate",
                    help="probability of a transient error per job-stage")
    rp.add_argument("--poison-rate", type=float, default=0.02, dest="poison_rate",
                    help="fraction of docs that always fail at embedding")
    rp.add_argument("--crash-rate", type=float, default=0.01, dest="crash_rate",
                    help="probability a worker crashes mid-job")
    rp.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducible runs")

    stp = sub.add_parser("status")
    stp.add_argument("--watch", action="store_true",
                     help="refresh every second until Ctrl-C")

    sub.add_parser("retry-dead")

    args = parser.parse_args()
    dispatch = {
        "init": cmd_init,
        "seed": cmd_seed,
        "run": cmd_run,
        "status": cmd_status,
        "retry-dead": cmd_retry_dead,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
