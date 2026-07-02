"""
Cloud Run Job entrypoint — resolves station config, retrains if the
location changed, runs one prediction/validation cycle, and exits.

Cloud Scheduler triggers this every 30 minutes.
"""
import sys
from main import run_once

if __name__ == "__main__":
    try:
        run_once()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
