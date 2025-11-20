# core/logger.py
import json, datetime, os
LOGFILE = os.path.join(os.getcwd(), "x_tutor_log.jsonl")

def log_query(entry: dict):
    entry = dict(entry)
    entry.setdefault("ts", datetime.datetime.now().isoformat())
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass
