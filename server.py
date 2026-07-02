"""
Cloud Run SERVICE entrypoint.

Cloud Run expects a container that listens on $PORT and answers health
checks. Our real work is the continuous prediction/retrain loop, so we:
  1. Run that loop (main.main_loop) in a background daemon thread.
  2. Serve a tiny HTTP endpoint on $PORT for Cloud Run's health probe,
     also exposing last-cycle status for quick debugging.

Run with min-instances=1 so the loop is always alive.
"""
import os
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import main

# Shared, thread-safe-enough status snapshot for the health endpoint.
STATE = {
    "started_at": time.time(),
    "loop_alive": False,
    "last_error": None,
}


def _run_loop():
    STATE["loop_alive"] = True
    try:
        main.main_loop()
    except Exception as e:  # loop should never exit; record if it does
        STATE["loop_alive"] = False
        STATE["last_error"] = str(e)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "service": "smart-weather-ai",
            "loop_alive": STATE["loop_alive"],
            "uptime_sec": round(time.time() - STATE["started_at"]),
            "last_error": STATE["last_error"],
        }).encode()
        self.send_response(200 if STATE["loop_alive"] else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence per-request logging noise


if __name__ == "__main__":
    threading.Thread(target=_run_loop, daemon=True).start()

    port = int(os.environ.get("PORT", "8080"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Health server listening on :{port}; pipeline loop running in background.")
    httpd.serve_forever()
