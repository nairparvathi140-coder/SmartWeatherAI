"""
Cloud Run Job entrypoint — runs one prediction cycle and exits.

Cloud Scheduler triggers this every N minutes.
"""
import os
import sys
from main import main

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
