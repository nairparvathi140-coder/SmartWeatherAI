"""
Sample recorder — the "excel" engine.

Real-time readings are sampled every ~10 min into a dated CSV (the excel),
then aggregated over rolling windows:
  - 20-min window  -> testing / validation (actual-vs-predicted)
  - 60-min window  -> hourly trend

Each excel row carries an explicit timestamp + date + time. The CSVs live
under data/buffers/. NOTE for Cloud Run: that disk is ephemeral (wiped on
restart), so the CSVs are a LOCAL convenience — the durable, timestamped copy
of every aggregate is the nested RTDB record written by the pipeline. On
Cloud Run we'd either mount a GCS bucket for the CSVs or rely solely on RTDB.
"""
import os
import csv
from datetime import datetime, timedelta

BUFFER_DIR = os.path.join("data", "buffers")

FIELDS = [
    "temperature", "humidity", "wind_speed", "wind_direction",
    "rainfall", "pressure", "irradiance",
]

# In-memory rolling window of recent samples (pruned to ~2h).
_samples = []  # list of dicts: {"dt": datetime, <field>: float, ...}


def record_sample(sensor):
    """Append the current reading to the in-memory buffer and today's excel."""
    now = datetime.now()
    entry = {"dt": now}
    for k in FIELDS:
        try:
            entry[k] = float(sensor.get(k, 0))
        except (TypeError, ValueError):
            entry[k] = 0.0
    _samples.append(entry)

    # Prune anything older than 2 hours so memory stays bounded.
    cutoff = now - timedelta(hours=2)
    while _samples and _samples[0]["dt"] < cutoff:
        _samples.pop(0)

    _append_csv(f"samples_{now:%Y-%m-%d}.csv", now, entry)
    return entry


def aggregate(window_minutes):
    """Average each field over the samples within the trailing window.
    Returns (reading_dict, sample_count) or (None, 0) if no samples."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)
    window = [s for s in _samples if s["dt"] >= cutoff]
    if not window:
        return None, 0
    agg = {
        k: round(sum(s[k] for s in window) / len(window), 4)
        for k in FIELDS
    }
    return agg, len(window)


def write_result(kind, actual, predicted):
    """Record an aggregated actual-vs-predicted row to a results excel.
    kind: 'twentymin' or 'hourly'."""
    now = datetime.now()
    path = f"{kind}_results_{now:%Y-%m-%d}.csv"
    header = ["timestamp", "date", "time"] \
        + [f"actual_{k}" for k in FIELDS] \
        + [f"pred_{k}" for k in FIELDS]
    row = {
        "timestamp": now.isoformat(),
        "date": f"{now:%Y-%m-%d}",
        "time": f"{now:%H:%M}",
    }
    for k in FIELDS:
        row[f"actual_{k}"] = actual.get(k, "")
        row[f"pred_{k}"] = (predicted or {}).get(k, "")
    _append_csv_row(path, header, row)


def _append_csv(filename, now, entry):
    header = ["timestamp", "date", "time"] + FIELDS
    row = {
        "timestamp": now.isoformat(),
        "date": f"{now:%Y-%m-%d}",
        "time": f"{now:%H:%M:%S}",
        **{k: entry[k] for k in FIELDS},
    }
    _append_csv_row(filename, header, row)


def _append_csv_row(filename, header, row):
    os.makedirs(BUFFER_DIR, exist_ok=True)
    path = os.path.join(BUFFER_DIR, filename)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
