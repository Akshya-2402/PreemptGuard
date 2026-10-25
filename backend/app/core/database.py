import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterator

from app.core.config import DATABASE_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS managed_processes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    workload_type TEXT NOT NULL,
    pid INTEGER,
    status TEXT NOT NULL,
    policy TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    restart_count INTEGER NOT NULL DEFAULT 0,
    last_restart_at REAL,
    consecutive_high_cpu INTEGER NOT NULL DEFAULT 0,
    next_recheck_at REAL,
    retry_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    process_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    cpu_percent REAL,
    memory_mb REAL,
    memory_percent REAL,
    memory_growth_mb_s REAL,
    process_state TEXT NOT NULL,
    responsive INTEGER NOT NULL,
    heartbeat_age_s REAL,
    uptime_s REAL,
    recent_failure_count INTEGER NOT NULL DEFAULT 0,
    recent_restart_count INTEGER NOT NULL DEFAULT 0,
    operation_status TEXT,
    FOREIGN KEY(process_id) REFERENCES managed_processes(id)
);
CREATE INDEX IF NOT EXISTS idx_measurements_process_time
    ON measurements(process_id, timestamp);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    process_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    risk_score REAL NOT NULL,
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    confidence_kind TEXT NOT NULL DEFAULT 'history_coverage_not_probability',
    horizon_s INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    features_json TEXT NOT NULL,
    contributors_json TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    FOREIGN KEY(process_id) REFERENCES managed_processes(id)
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    process_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    risk_score REAL NOT NULL,
    confidence REAL NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    outcome TEXT,
    FOREIGN KEY(process_id) REFERENCES managed_processes(id)
);
CREATE TABLE IF NOT EXISTS recovery_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    process_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    attempt_number INTEGER NOT NULL,
    old_pid INTEGER,
    new_pid INTEGER,
    status TEXT NOT NULL,
    duration_s REAL,
    reason TEXT NOT NULL,
    verification_period_s REAL,
    FOREIGN KEY(process_id) REFERENCES managed_processes(id)
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    configuration_json TEXT NOT NULL,
    status TEXT NOT NULL,
    results_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitor_overhead (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cpu_percent REAL,
    memory_mb REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_monitor_overhead_time ON monitor_overhead(timestamp);
"""


class Database:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(decisions)")
            }
            if "estimated_cost" not in columns:
                connection.execute(
                    "ALTER TABLE decisions ADD COLUMN estimated_cost REAL NOT NULL DEFAULT 0.0"
                )
            measurement_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(measurements)")
            }
            if "recent_failure_count" not in measurement_columns:
                connection.execute(
                    "ALTER TABLE measurements ADD COLUMN recent_failure_count INTEGER NOT NULL DEFAULT 0"
                )
            if "recent_restart_count" not in measurement_columns:
                connection.execute(
                    "ALTER TABLE measurements ADD COLUMN recent_restart_count INTEGER NOT NULL DEFAULT 0"
                )
            if "operation_status" not in measurement_columns:
                connection.execute("ALTER TABLE measurements ADD COLUMN operation_status TEXT")
            prediction_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
            }
            if "confidence_kind" not in prediction_columns:
                connection.execute(
                    "ALTER TABLE predictions ADD COLUMN confidence_kind TEXT NOT NULL "
                    "DEFAULT 'history_coverage_not_probability'"
                )
            process_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(managed_processes)")
            }
            if "retry_count" not in process_columns:
                connection.execute(
                    "ALTER TABLE managed_processes ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
                )
            decision_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(decisions)")
            }
            if "evidence_json" not in decision_columns:
                connection.execute(
                    "ALTER TABLE decisions ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '{}'"
                )
            recovery_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(recovery_attempts)")
            }
            if "verification_period_s" not in recovery_columns:
                connection.execute(
                    "ALTER TABLE recovery_attempts ADD COLUMN verification_period_s REAL"
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, sql: str, parameters: Iterable[Any] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, tuple(parameters))
            return int(cursor.lastrowid or 0)

    def query(self, sql: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
            return [dict(row) for row in rows]

    def store_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)


database = Database()
