from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from theta_bot_averaging.backtest import run_backtest
from theta_bot_averaging.data import build_targets, load_dataset
from theta_bot_averaging.features import build_features
from theta_bot_averaging.models import BaselineModel
from theta_bot_averaging.validation.purged_split import PurgedTimeSeriesSplit


@dataclass
class WalkforwardConfig:
    data_path: str
    horizon: int = 1
    threshold_bps: float = 10.0
    model_type: str = "logit"
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    spread_bps: float = 0.5
    n_splits: int = 5
    purge: int = 0
    embargo: int = 0
    output_dir: str = "runs"


def _load_config(path: str) -> WalkforwardConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return WalkforwardConfig(**raw)


def run_walkforward(config_path: str) -> Dict:
    cfg = _load_config(config_path)
    df = load_dataset(cfg.data_path)
    df_targets = build_targets(df, horizon=cfg.horizon, threshold_bps=cfg.threshold_bps)
    features = build_features(df_targets)
    data = pd.concat([features, df_targets[["label", "future_return"]]], axis=1).dropna()
    features = data[features.columns]
    targets = data["label"]
    future_returns = data["future_return"]

    splitter = PurgedTimeSeriesSplit(
        n_splits=cfg.n_splits, purge=cfg.purge, embargo=cfg.embargo
    )
    all_metrics = []
    predictions = []

    thr = cfg.threshold_bps / 10_000.0

    for train_idx, test_idx in splitter.split(features.index):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_train, y_train = features.iloc[train_idx], targets.iloc[train_idx]
        X_test = features.iloc[test_idx]
        model = BaselineModel(
            mode=cfg.model_type,
            positive_threshold=thr,
            negative_threshold=-thr,
        )
        model.fit(X_train, y_train, future_return=future_returns.loc[X_train.index])
        preds = model.predict(X_test)
        fold_df = pd.DataFrame(
            {
                "predicted_return": preds.predicted_return,
                "signal": preds.signal,
                "future_return": future_returns.iloc[test_idx],
            }
        )
        if fold_df["predicted_return"].isna().any():
            raise ValueError("Missing predicted_return. Run inference first or fix pipeline.")
        backtest_res = run_backtest(
            fold_df,
            position=fold_df["signal"],
            future_return_col="future_return",
            fee_rate=cfg.fee_rate,
            slippage_bps=cfg.slippage_bps,
            spread_bps=cfg.spread_bps,
        )
        metrics = backtest_res.metrics
        metrics["fold_start"] = str(fold_df.index.min())
        metrics["fold_end"] = str(fold_df.index.max())
        all_metrics.append(metrics)
        predictions.append(fold_df)

    aggregated = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_path).stem
    out_dir = Path(cfg.output_dir) / timestamp / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.concat(predictions).sort_index()
    preds_df.to_parquet(out_dir / "predictions.parquet")

    backtest_res = run_backtest(
        preds_df,
        position=preds_df["signal"],
        future_return_col="future_return",
        fee_rate=cfg.fee_rate,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
        output_dir=out_dir,
    )

    metrics_payload = {
        "fold_metrics": all_metrics,
        "aggregate": aggregated,
        "backtest": backtest_res.metrics,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return {"metrics": aggregated, "output_dir": str(out_dir)}
