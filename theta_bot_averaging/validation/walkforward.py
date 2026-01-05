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
from theta_bot_averaging.features import build_features, build_dual_stream_inputs
from theta_bot_averaging.models import BaselineModel, DualStreamModel
from theta_bot_averaging.validation.purged_split import PurgedTimeSeriesSplit
from theta_bot_averaging.utils import SignalMode


@dataclass
class WalkforwardConfig:
    data_path: str
    horizon: int = 1
    threshold_bps: float = 10.0
    model_type: str = "logit"
    signal_mode: SignalMode = "threshold"
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    spread_bps: float = 0.5
    n_splits: int = 5
    purge: int = 0
    embargo: int = 0
    output_dir: str = "runs"
    # Dual-stream specific parameters
    theta_window: int = 48
    theta_q: float = 0.9
    theta_terms: int = 8
    mellin_k: int = 16
    mellin_alpha: float = 0.5
    mellin_omega_max: float = 1.0
    torch_epochs: int = 50
    torch_batch_size: int = 32
    torch_lr: float = 1e-3


def _load_config(path: str) -> WalkforwardConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return WalkforwardConfig(**raw)


def run_walkforward(config_path: str) -> Dict:
    cfg = _load_config(config_path)
    df = load_dataset(cfg.data_path)
    df_targets = build_targets(df, horizon=cfg.horizon, threshold_bps=cfg.threshold_bps)

    thr = cfg.threshold_bps / 10_000.0

    # Check if dual_stream model
    if cfg.model_type == "dual_stream":
        # Build dual-stream inputs
        index, X_theta, X_mellin = build_dual_stream_inputs(
            df=df_targets,
            window=cfg.theta_window,
            q=cfg.theta_q,
            n_terms=cfg.theta_terms,
            mellin_k=cfg.mellin_k,
            alpha=cfg.mellin_alpha,
            omega_max=cfg.mellin_omega_max,
        )

        # Align targets and future_return to the valid index
        targets = df_targets.loc[index, "label"]
        future_returns = df_targets.loc[index, "future_return"]

        # Convert to numpy arrays for model
        y_array = targets.values
        future_return_array = future_returns.values

        splitter = PurgedTimeSeriesSplit(
            n_splits=cfg.n_splits, purge=cfg.purge, embargo=cfg.embargo
        )
        all_metrics = []
        predictions = []

        for train_idx, test_idx in splitter.split(index):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            # Split arrays
            X_theta_train, X_theta_test = X_theta[train_idx], X_theta[test_idx]
            X_mellin_train, X_mellin_test = X_mellin[train_idx], X_mellin[test_idx]
            y_train = y_array[train_idx]
            future_return_train = future_return_array[train_idx]
            test_index = index[test_idx]

            # Create and train model
            model = DualStreamModel(
                positive_threshold=thr,
                negative_threshold=-thr,
                signal_mode=cfg.signal_mode,
                epochs=cfg.torch_epochs,
                batch_size=cfg.torch_batch_size,
                lr=cfg.torch_lr,
            )
            model.fit(X_theta_train, X_mellin_train, y_train, future_return=future_return_train)

            # Predict
            preds = model.predict(X_theta_test, X_mellin_test, test_index)

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

    else:
        # Original baseline path
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

        for train_idx, test_idx in splitter.split(features.index):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            X_train, y_train = features.iloc[train_idx], targets.iloc[train_idx]
            X_test = features.iloc[test_idx]
            model = BaselineModel(
                mode=cfg.model_type,
                positive_threshold=thr,
                negative_threshold=-thr,
                signal_mode=cfg.signal_mode,
            )
            model.fit(X_train, y_train, future_return=future_returns.iloc[train_idx])
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
    try:
        preds_df.to_parquet(out_dir / "predictions.parquet")
    except ImportError:
        preds_df.to_csv(out_dir / "predictions.csv")

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
