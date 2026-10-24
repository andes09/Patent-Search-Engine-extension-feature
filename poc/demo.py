"""python -m poc.demo

One-command end-to-end demonstration for the video walkthrough.
Runs init → seed → seed (idempotency) → run → status → mutate + run → status.
Target: finishes in under 60 seconds.
"""
import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
_DB = str(Path(__file__).parent / "queue.db")
_BASE_CMD = [sys.executable, "-m", "poc.queue", "--db", _DB]


def run(*args: str) -> None:
    """Run a poc.queue sub-command, printing it first for visibility."""
    display = " ".join(args)
    print(f"\n$ python -m poc.queue {display}")
    print("-" * 50)
    sys.stdout.flush()
    subprocess.run(_BASE_CMD + list(args), check=True, cwd=_ROOT)


def main() -> None:
    print("=" * 50)
    print("ThinkStruct Part 2 POC - Ingestion Queue Demo")
    print("=" * 50)
    sys.stdout.flush()

    print("\n--- 1. Initialise database and seed patents ---")
    run("init")
    run("seed")

    print("\n--- 2. Seed again (should add 0 new rows) -----")
    run("seed")

    print("\n--- 3. Run workers with default failure rates -")
    run("run", "--workers", "4")

    print("\n--- 4. Final status report --------------------")
    run("status")

    print("\n--- 5. Mutate 5 patents; only those 5 reprocess")
    run("seed", "--mutate", "5")
    run("run", "--workers", "4")
    run("status")

    print("\n" + "=" * 50)
    print("Demo complete.")


if __name__ == "__main__":
    main()
