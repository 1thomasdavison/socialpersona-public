"""Durable reservations for explicitly budgeted, non-retrying API experiments."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Failed/ambiguous requests retain their full reservation across restarts."""

    def __init__(self, path: Path, cap: float):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Budget must be positive and finite")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, request_key TEXT, "
                       "model TEXT, reserved REAL, actual REAL, status TEXT, usage TEXT, created REAL)")
            old = db.execute("SELECT value FROM settings WHERE key='cap'").fetchone()
            if old and float(old[0]) != cap:
                raise ValueError("Existing budget cap cannot be changed by resuming a run")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('cap', ?)", (str(cap),))
        self.cap = cap

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, key: str, model: str, amount: float) -> str:
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Reservation must be positive and finite")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM calls").fetchone()[0]
            if used + amount > self.cap:
                raise BudgetExceeded(f"Budget gate: committed={used:.3f}, next={amount:.3f}, cap={self.cap:.2f}")
            call_id = uuid.uuid4().hex
            db.execute("INSERT INTO calls VALUES (?,?,?,?,NULL,'reserved',NULL,?)",
                       (call_id, key, model, amount, time.time()))
        return call_id

    def settle(self, call_id: str, usage: dict, input_rate: float, output_rate: float):
        inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(inp, int) or not isinstance(out, int) or inp < 0 or out < 0:
            self.fail(call_id)
            raise ValueError("Missing valid usage; full reservation retained")
        actual = (inp * input_rate + out * output_rate) / 1000
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,status FROM calls WHERE id=?", (call_id,)).fetchone()
            if row is None or row[1] != 'reserved':
                raise ValueError("Call is missing or already settled")
            db.execute("UPDATE calls SET actual=?,status='settled',usage=? WHERE id=?",
                       (actual, json.dumps(usage), call_id))
        if actual > row[0] + 1e-6:
            raise RuntimeError("Actual usage exceeded conservative reservation; stop for pricing review")

    def fail(self, call_id: str):
        with self.connect() as db:
            db.execute("UPDATE calls SET status='uncertain' WHERE id=? AND status='reserved'", (call_id,))

    def snapshot(self) -> dict:
        with self.connect() as db:
            count, paid, committed = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(actual),0),COALESCE(SUM(COALESCE(actual,reserved)),0) FROM calls"
            ).fetchone()
            uncertain = db.execute("SELECT COUNT(*) FROM calls WHERE actual IS NULL").fetchone()[0]
        return dict(cap_ca=self.cap, requests=count, metered_ca=round(paid, 6),
                    committed_ca=round(committed, 6), uncertain_requests=uncertain)
