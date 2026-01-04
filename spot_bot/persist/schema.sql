CREATE TABLE IF NOT EXISTS bars (
    ts INTEGER PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);

CREATE TABLE IF NOT EXISTS features (
    ts INTEGER PRIMARY KEY,
    rv REAL,
    C REAL,
    psi REAL,
    C_int REAL,
    S REAL
);

CREATE TABLE IF NOT EXISTS decisions (
    ts INTEGER PRIMARY KEY,
    risk_state TEXT,
    risk_budget REAL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS intents (
    ts INTEGER PRIMARY KEY,
    desired_exposure REAL,
    reason TEXT
);

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
);

CREATE TABLE IF NOT EXISTS equity (
    ts INTEGER PRIMARY KEY,
    equity_usdt REAL,
    btc REAL,
    usdt REAL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
