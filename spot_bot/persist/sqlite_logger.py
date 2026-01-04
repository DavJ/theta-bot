from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd


def _to_epoch_ms(ts: Any) -> int:
    """Normalize timestamps to integer milliseconds."""
    if ts is None:
        raise ValueError("Timestamp is required")
    if isinstance(ts, (int, float)):
        return int(ts)
    return int(pd.to_datetime(ts, utc=True).value // 1_000_000)


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


class SQLiteLogger:
    """Lightweight persistence layer for Spot Bot runs with SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.init()

    def init(self) -> None:
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                ts INTEGER PRIMARY KEY,
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
                ts INTEGER PRIMARY KEY,
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
                ts INTEGER PRIMARY KEY,
                risk_state TEXT,
                risk_budget REAL,
                reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS intents (
                ts INTEGER PRIMARY KEY,
                desired_exposure REAL,
                reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER,
                mode TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                fee REAL,
                order_id TEXT,
                status TEXT,
                meta TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                ts INTEGER PRIMARY KEY,
                equity_usdt REAL,
                btc REAL,
                usdt REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _insert_or_replace(self, table: str, columns: Sequence[str], values: Sequence[Any]) -> None:
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        self.conn.execute(sql, tuple(values))
        self.conn.commit()

    def upsert_bar(self, ts: Any, open: float, high: float, low: float, close: float, volume: float) -> None:  # noqa: A002
        self._insert_or_replace(
            "bars",
            ["ts", "open", "high", "low", "close", "volume"],
            [_to_epoch_ms(ts), _to_float(open), _to_float(high), _to_float(low), _to_float(close), _to_float(volume)],
        )

    def upsert_features(
        self, ts: Any, rv: float | None, C: float | None, psi: float | None, C_int: float | None, S: float | None  # noqa: N803
    ) -> None:
        self._insert_or_replace(
            "features",
            ["ts", "rv", "C", "psi", "C_int", "S"],
            [_to_epoch_ms(ts), _to_float(rv), _to_float(C), _to_float(psi), _to_float(C_int), _to_float(S)],
        )

    def upsert_decision(self, ts: Any, risk_state: str, risk_budget: float, reason: str) -> None:
        self._insert_or_replace(
            "decisions",
            ["ts", "risk_state", "risk_budget", "reason"],
            [_to_epoch_ms(ts), risk_state, float(risk_budget), reason],
        )

    def upsert_intent(self, ts: Any, desired_exposure: float, reason: str) -> None:
        self._insert_or_replace(
            "intents",
            ["ts", "desired_exposure", "reason"],
            [_to_epoch_ms(ts), float(desired_exposure), reason],
        )

    def insert_execution(
        self,
        ts: Any,
        mode: str,
        side: str,
        qty: float,
        price: float,
        fee: float,
        order_id: str = "",
        status: str = "filled",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(meta or {})
        self.conn.execute(
            """
            INSERT INTO executions (ts, mode, side, qty, price, fee, order_id, status, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (_to_epoch_ms(ts), mode, side, float(qty), float(price), float(fee), order_id, status, meta_json),
        )
        self.conn.commit()

    def upsert_equity(self, ts: Any, equity_usdt: float, btc: float, usdt: float) -> None:
        self._insert_or_replace(
            "equity",
            ["ts", "equity_usdt", "btc", "usdt"],
            [_to_epoch_ms(ts), float(equity_usdt), float(btc), float(usdt)],
        )

    def get_last_ts(self) -> Optional[int]:
        cur = self.conn.execute("SELECT ts FROM bars ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else None

    def get_latest_equity(self) -> Optional[Dict[str, float]]:
        cur = self.conn.execute("SELECT ts, equity_usdt, btc, usdt FROM equity ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        return {"ts": int(row[0]), "equity_usdt": float(row[1]), "btc": float(row[2]), "usdt": float(row[3])}

    def list_executions(self, limit: int = 50) -> list[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT ts, mode, side, qty, price, fee, order_id, status, meta FROM executions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for ts, mode, side, qty, price, fee, order_id, status, meta_raw in cur.fetchall():
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {"raw_meta": meta_raw}
            rows.append(
                {
                    "ts": int(ts),
                    "mode": mode,
                    "side": side,
                    "qty": _to_float(qty),
                    "price": _to_float(price),
                    "fee": _to_float(fee),
                    "order_id": order_id,
                    "status": status,
                    "meta": meta,
                }
            )
        return rows

    def set_kv(self, key: str, value: Any) -> None:
        self._insert_or_replace("kv", ["key", "value"], [key, json.dumps(value)])

    def get_kv(self, key: str) -> Any:
        cur = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

    # Backward-compatible helpers -------------------------------------------------
    def log_bars(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            ts = row.get("ts", row.get("timestamp"))
            self.upsert_bar(
                ts=ts,
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row.get("close"),
                volume=row.get("volume"),
            )

    def log_features(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            ts = row.get("ts", row.get("timestamp"))
            self.upsert_features(ts, row.get("rv"), row.get("C"), row.get("psi"), row.get("C_int"), row.get("S"))

    def log_decision(self, timestamp: Any, risk_state: str, risk_budget: float, reason: str) -> None:
        self.upsert_decision(timestamp, risk_state, risk_budget, reason)

    def log_intent(self, timestamp: Any, desired_exposure: float, reason: str) -> None:
        self.upsert_intent(timestamp, desired_exposure, reason)

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
        self.insert_execution(
            ts=timestamp,
            mode="legacy",
            side=action,
            qty=qty,
            price=price,
            fee=fee,
            order_id=order_id or "",
            status=status,
            meta=None,
        )

    def log_equity(self, timestamp: Any, equity_usdt: float, btc: float, usdt: float) -> None:
        self.upsert_equity(timestamp, equity_usdt, btc, usdt)

    def latest_bar_timestamp(self) -> Optional[str]:
        last = self.get_last_ts()
        return str(last) if last is not None else None

    def latest_equity(self) -> Optional[Dict[str, float]]:
        return self.get_latest_equity()


__all__ = ["SQLiteLogger"]
