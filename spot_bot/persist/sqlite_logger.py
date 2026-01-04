from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


class SQLiteLogger:
    """Lightweight persistence layer for Spot Bot runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                timestamp TEXT PRIMARY KEY,
                rv REAL,
                C REAL,
                psi REAL,
                C_int REAL,
                S REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                timestamp TEXT PRIMARY KEY,
                risk_state TEXT,
                risk_budget REAL,
                reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS intents (
                timestamp TEXT PRIMARY KEY,
                desired_exposure REAL,
                reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                timestamp TEXT,
                action TEXT,
                qty REAL,
                price REAL,
                fee REAL,
                order_id TEXT,
                status TEXT,
                PRIMARY KEY (timestamp, order_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                timestamp TEXT PRIMARY KEY,
                equity_usdt REAL,
                btc REAL,
                usdt REAL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _insert(self, table: str, columns: Sequence[str], values: Sequence[Any]) -> None:
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        self.conn.execute(sql, tuple(values))
        self.conn.commit()

    def log_bars(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self._insert(
                "bars",
                ["timestamp", "open", "high", "low", "close", "volume"],
                [
                    str(row.get("timestamp")),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                ],
            )

    def log_features(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self._insert(
                "features",
                ["timestamp", "rv", "C", "psi", "C_int", "S"],
                [
                    str(row.get("timestamp")),
                    row.get("rv"),
                    row.get("C"),
                    row.get("psi"),
                    row.get("C_int"),
                    row.get("S"),
                ],
            )

    def log_decision(self, timestamp: Any, risk_state: str, risk_budget: float, reason: str) -> None:
        self._insert(
            "decisions",
            ["timestamp", "risk_state", "risk_budget", "reason"],
            [str(timestamp), risk_state, float(risk_budget), reason],
        )

    def log_intent(self, timestamp: Any, desired_exposure: float, reason: str) -> None:
        self._insert(
            "intents",
            ["timestamp", "desired_exposure", "reason"],
            [str(timestamp), float(desired_exposure), reason],
        )

    def log_execution(
        self,
        timestamp: Any,
        action: str,
        qty: float,
        price: float,
        fee: float,
        order_id: Optional[str] = None,
        status: str = "filled",
    ) -> None:
        self._insert(
            "executions",
            ["timestamp", "action", "qty", "price", "fee", "order_id", "status"],
            [str(timestamp), action, float(qty), float(price), float(fee), order_id or "", status],
        )

    def log_equity(self, timestamp: Any, equity_usdt: float, btc: float, usdt: float) -> None:
        self._insert(
            "equity",
            ["timestamp", "equity_usdt", "btc", "usdt"],
            [str(timestamp), float(equity_usdt), float(btc), float(usdt)],
        )

    def latest_bar_timestamp(self) -> Optional[str]:
        cur = self.conn.execute("SELECT timestamp FROM bars ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None

    def latest_equity(self) -> Optional[Dict[str, float]]:
        cur = self.conn.execute(
            "SELECT timestamp, equity_usdt, btc, usdt FROM equity ORDER BY timestamp DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"timestamp": row[0], "equity_usdt": row[1], "btc": row[2], "usdt": row[3]}


__all__ = ["SQLiteLogger"]
