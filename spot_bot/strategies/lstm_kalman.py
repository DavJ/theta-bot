"""
LSTM + Kalman hybrid strategy for spot crypto trading.

Combines:
- Local linear trend Kalman filter for level/trend state estimation
- Numpy-based LSTM cell for sequential pattern recognition
- Biquaternion-inspired features (C, psi, S) from the unified feature pipeline
  (aligned with github.com/DavJ/unified-biquaternion-theory)
- Risk/price edge gating via realized volatility and risk_budget
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from spot_bot.utils.normalization import clip01

from .base import Intent, Strategy


# Minimum variance guard (same as sibling strategies)
MIN_VARIANCE = 1e-8

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    x = np.clip(x, -60.0, 60.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _sigmoid_to_signed(z: float | np.ndarray) -> float | np.ndarray:
    """
    Map a z-score to a signed signal in (-1, 1) via logistic squashing.

    Equivalent to  2 * sigmoid(-z) - 1:  negative z → positive signal (buy bias).
    """
    x = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x)) * 2.0 - 1.0


def _get_feature_col(
    df: pd.DataFrame, col: str, default: float
) -> pd.Series:
    """Return a numeric Series for *col* from df, filling missing values with *default*."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe_normalize(series: pd.Series, window: int = 48) -> pd.Series:
    """Rolling z-score normalization, NaN-safe."""
    mu = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std(ddof=0).fillna(0.0)
    sigma = sigma.where(sigma > MIN_VARIANCE, MIN_VARIANCE)
    return ((series - mu) / sigma).fillna(0.0)


# ──────────────────────────────────────────────────────────
# Numpy LSTM cell
# ──────────────────────────────────────────────────────────

class _NumpyLSTMCell:
    """
    Single-layer LSTM cell in pure numpy.

    Weights are Xavier-initialized with forget-gate bias = 1 (standard trick
    that encourages gradient flow at the start).  Output is a scalar via a
    learned linear projection.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        hs = hidden_size

        scale_i = np.sqrt(1.0 / input_size)
        scale_h = np.sqrt(1.0 / hidden_size)
        scale_o = np.sqrt(1.0 / hidden_size)

        # [input, forget, cell, output] gate weights – shape (4*hs, input_size)
        self.Wi = rng.uniform(-scale_i, scale_i, (4 * hs, input_size))
        # Recurrent weights – shape (4*hs, hs)
        self.Wh = rng.uniform(-scale_h, scale_h, (4 * hs, hs))
        # Biases – forget gate bias set to 1 for gradient stability
        self.b = np.zeros(4 * hs)
        self.b[hs : 2 * hs] = 1.0  # forget gate bias

        # Output projection: (hs,) → scalar
        self.w_out = rng.uniform(-scale_o, scale_o, hs)
        self.b_out = 0.0

        self.hidden_size = hs
        self.input_size = input_size

    def zero_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.hidden_size), np.zeros(self.hidden_size)

    def step(
        self, x: np.ndarray, h: np.ndarray, c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One LSTM step. Returns (h_new, c_new)."""
        hs = self.hidden_size
        gates = self.Wi @ x + self.Wh @ h + self.b
        i_g = _sigmoid(gates[:hs])
        f_g = _sigmoid(gates[hs : 2 * hs])
        g_g = np.tanh(gates[2 * hs : 3 * hs])
        o_g = _sigmoid(gates[3 * hs :])
        c_new = f_g * c + i_g * g_g
        c_new = np.clip(c_new, -10.0, 10.0)
        h_new = o_g * np.tanh(c_new)
        return h_new, c_new

    def project(self, h: np.ndarray) -> float:
        """Linear output projection → scalar in (-∞, +∞)."""
        return float(np.dot(self.w_out, h) + self.b_out)


# ──────────────────────────────────────────────────────────
# Kalman helpers (shared with KalmanStrategy but self-contained)
# ──────────────────────────────────────────────────────────

def _run_local_linear_kalman(
    prices: pd.Series,
    q_level: float,
    q_trend: float,
    r: float,
) -> Tuple[float, float, float]:
    """
    Local linear trend Kalman filter.

    Returns (level_est, trend_est, innovation_var) after processing all prices.
    """
    x = np.array([float(prices.iloc[0]), 0.0])
    P = np.eye(2, dtype=float)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_trend]])
    H = np.array([1.0, 0.0])
    innov_var = float(r)

    for price in prices:
        y = float(price)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innov = y - H @ x_pred
        innov_var = float(H @ P_pred @ H.T + r)
        if not np.isfinite(innov_var) or innov_var <= 0.0:
            innov_var = MIN_VARIANCE
        K = P_pred @ H / innov_var
        x = x_pred + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

    return float(x[0]), float(x[1]), innov_var


# ──────────────────────────────────────────────────────────
# Strategy params
# ──────────────────────────────────────────────────────────

@dataclass
class LSTMKalmanParams:
    # Kalman filter noise params
    q_level: float = 1e-4
    q_trend: float = 1e-6
    r: float = 1e-3
    # Kalman z-score scale (exposure sensitivity)
    k_kalman: float = 1.5
    # LSTM architecture
    hidden_size: int = 32
    # Number of bars used as LSTM context window (0 = full history)
    lstm_lookback: int = 128
    # Random seed for LSTM weight initialization
    lstm_seed: int = 42
    # Mix weight: final = kalman_weight * kalman_signal + (1-kalman_weight) * lstm_signal
    kalman_weight: float = 0.6
    # Max exposure (long/flat: 0..1)
    emax: float = 1.0
    # Min bars before emitting a non-zero signal
    min_bars: int = 20
    # Normalization window for input features
    norm_window: int = 48
    # Risk budget feature column (if present in features_df)
    risk_budget_col: str = "risk_budget"


# ──────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────

# Input feature names fed to the LSTM (7 inputs)
_FEATURE_NAMES = ["kalman_z", "log_ret", "rv", "C", "psi_sin", "psi_cos", "S"]
_INPUT_SIZE = len(_FEATURE_NAMES)


class LSTMKalmanStrategy(Strategy):
    """
    Hybrid LSTM + Kalman strategy for spot crypto.

    Signal pipeline:
    1. Kalman filter on close prices → level, trend, z-score
    2. LSTM processes sequential features:
         [kalman_z, log_return, rv, C, psi_sin, psi_cos, S]
       where C, psi, S are biquaternion-inspired regime features
       (aligned with DavJ/unified-biquaternion-theory).
    3. Final exposure = kalman_weight * kalman_signal
                      + (1 - kalman_weight) * lstm_signal
    4. Scaled by risk_budget (if available in features_df).

    All dependencies are numpy-only; no PyTorch required.
    """

    def __init__(
        self,
        q_level: float = LSTMKalmanParams.q_level,
        q_trend: float = LSTMKalmanParams.q_trend,
        r: float = LSTMKalmanParams.r,
        k_kalman: float = LSTMKalmanParams.k_kalman,
        hidden_size: int = LSTMKalmanParams.hidden_size,
        lstm_lookback: int = LSTMKalmanParams.lstm_lookback,
        lstm_seed: int = LSTMKalmanParams.lstm_seed,
        kalman_weight: float = LSTMKalmanParams.kalman_weight,
        emax: float = LSTMKalmanParams.emax,
        min_bars: int = LSTMKalmanParams.min_bars,
        norm_window: int = LSTMKalmanParams.norm_window,
    ) -> None:
        self.params = LSTMKalmanParams(
            q_level=float(q_level),
            q_trend=float(q_trend),
            r=float(r),
            k_kalman=float(k_kalman),
            hidden_size=int(hidden_size),
            lstm_lookback=int(lstm_lookback),
            lstm_seed=int(lstm_seed),
            kalman_weight=float(kalman_weight),
            emax=float(emax),
            min_bars=int(min_bars),
            norm_window=int(norm_window),
        )
        self._lstm = _NumpyLSTMCell(
            input_size=_INPUT_SIZE,
            hidden_size=int(hidden_size),
            seed=int(lstm_seed),
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_close(self, df: pd.DataFrame) -> pd.Series:
        for col in ("close", "Close", "price"):
            if col in df.columns:
                return df[col].astype(float)
        raise ValueError("features_df must contain a 'close' or 'price' column.")

    def _build_lstm_inputs(
        self, df: pd.DataFrame, kalman_z_series: pd.Series
    ) -> pd.DataFrame:
        """
        Build the 7-column input DataFrame for the LSTM from features_df.

        Columns: kalman_z, log_ret, rv, C, psi_sin, psi_cos, S.

        Biquaternion-inspired features (C, psi, S) are sourced from the
        feature pipeline which implements the unified-biquaternion-theory
        scale-phase decomposition.  psi is mapped to (sin, cos) to preserve
        circular geometry.
        """
        p = self.params
        close = self._extract_close(df)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)

        # Realized volatility (prefer pre-computed column)
        if "rv" in df.columns:
            rv_raw = pd.to_numeric(df["rv"], errors="coerce").fillna(0.0)
        else:
            rv_raw = log_ret.pow(2).rolling(24, min_periods=1).sum().pow(0.5)

        # Biquaternion-theory regime features (C: concentration, psi: phase, S: composite)
        C = _get_feature_col(df, "C", default=0.5)
        psi = _get_feature_col(df, "psi", default=0.0)
        S = _get_feature_col(df, "S", default=0.5)

        # Normalize inputs (rolling z-score)
        kalman_z_norm = _safe_normalize(kalman_z_series, window=p.norm_window)
        log_ret_norm = _safe_normalize(log_ret, window=p.norm_window)
        rv_norm = _safe_normalize(rv_raw, window=p.norm_window)
        C_norm = _safe_normalize(C, window=p.norm_window)
        S_norm = _safe_normalize(S, window=p.norm_window)

        # psi is circular → encode as (sin, cos)
        psi_sin = np.sin(2.0 * np.pi * psi)
        psi_cos = np.cos(2.0 * np.pi * psi)

        feat = pd.DataFrame(
            {
                "kalman_z": kalman_z_norm,
                "log_ret": log_ret_norm,
                "rv": rv_norm,
                "C": C_norm,
                "psi_sin": psi_sin.fillna(0.0),
                "psi_cos": psi_cos.fillna(1.0),
                "S": S_norm,
            },
            index=df.index,
        )
        return feat.fillna(0.0)

    # ------------------------------------------------------------------
    # Kalman z-series over full history
    # ------------------------------------------------------------------

    def _kalman_z_series(self, prices: pd.Series) -> pd.Series:
        """
        Compute a rolling Kalman z-score for each bar.

        We run the filter incrementally and collect z_{t} = (price_t - level_t) / sqrt(innov_var_t).
        """
        p = self.params
        x = np.array([float(prices.iloc[0]), 0.0])
        P = np.eye(2, dtype=float)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[p.q_level, 0.0], [0.0, p.q_trend]])
        H = np.array([1.0, 0.0])
        innov_var = float(p.r)

        zvals = []
        for price in prices:
            y = float(price)
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            innov = y - H @ x_pred
            innov_var = float(H @ P_pred @ H.T + p.r)
            if not np.isfinite(innov_var) or innov_var <= 0.0:
                innov_var = MIN_VARIANCE
            K = P_pred @ H / innov_var
            x = x_pred + K * innov
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
            level_t = float(x[0])
            z_t = (y - level_t) / float(np.sqrt(innov_var))
            zvals.append(z_t)

        return pd.Series(zvals, index=prices.index, dtype=float)

    # ------------------------------------------------------------------
    # LSTM forward pass over a sequence
    # ------------------------------------------------------------------

    def _lstm_signal_series(self, feat_df: pd.DataFrame) -> pd.Series:
        """
        Run the LSTM forward pass over all rows of feat_df.

        Returns a Series of raw LSTM outputs (one per bar).
        Applies tanh squashing so output ∈ (-1, 1).
        """
        h, c = self._lstm.zero_state()
        signals = []
        for _, row in feat_df.iterrows():
            x = row.values.astype(float)
            h, c = self._lstm.step(x, h, c)
            raw = self._lstm.project(h)
            signals.append(float(np.tanh(raw)))
        return pd.Series(signals, index=feat_df.index, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        if features_df is None or features_df.empty:
            return Intent(desired_exposure=0.0, reason="No features available", diagnostics={})

        close = self._extract_close(features_df).dropna()
        if close.empty or len(close) < self.params.min_bars:
            return Intent(desired_exposure=0.0, reason="Insufficient history", diagnostics={})

        p = self.params

        # ── 1. Kalman z-score series ──────────────────────────────────
        kalman_z_series = self._kalman_z_series(close)
        final_z = float(kalman_z_series.iloc[-1])

        # Kalman-only exposure: logistic squash of -k*z (lower price → buy)
        x_kal = float(np.clip(p.k_kalman * final_z, -60.0, 60.0))
        kalman_signal = float(_sigmoid_to_signed(x_kal))

        # ── 2. LSTM features ──────────────────────────────────────────
        df_feats = features_df.loc[close.index]
        lookback = p.lstm_lookback if p.lstm_lookback > 0 else len(close)
        if len(close) > lookback:
            df_feats = df_feats.iloc[-lookback:]
            kz_window = kalman_z_series.iloc[-lookback:]
        else:
            kz_window = kalman_z_series

        lstm_inputs = self._build_lstm_inputs(df_feats, kz_window)
        lstm_out_series = self._lstm_signal_series(lstm_inputs)
        lstm_signal = float(lstm_out_series.iloc[-1])  # ∈ (-1, 1)

        # ── 3. Combine ───────────────────────────────────────────────
        w = float(np.clip(p.kalman_weight, 0.0, 1.0))
        combined = w * kalman_signal + (1.0 - w) * lstm_signal  # ∈ (-1, 1)

        # ── 4. Risk/price edge gating ─────────────────────────────────
        last_row = features_df.iloc[-1]
        risk_budget = 1.0
        if isinstance(last_row, pd.Series) and p.risk_budget_col in last_row.index:
            risk_budget = float(last_row[p.risk_budget_col])
        risk_budget = float(np.clip(risk_budget, 0.0, 1.0))

        # ── 5. Final exposure (long/flat → clip to [0, emax]) ─────────
        raw_exposure = combined * p.emax * risk_budget
        desired_exposure = clip01(raw_exposure)

        # Kalman state for diagnostics
        level_est, trend_est, innov_var = _run_local_linear_kalman(
            close, p.q_level, p.q_trend, p.r
        )

        diagnostics = {
            "level": level_est,
            "trend": trend_est,
            "innovation_var": innov_var,
            "kalman_z": final_z,
            "kalman_signal": kalman_signal,
            "lstm_signal": lstm_signal,
            "combined": combined,
            "risk_budget": risk_budget,
            "kalman_weight": w,
        }
        reason = "LSTM+Kalman long bias" if desired_exposure > 0 else "No signal"
        return Intent(desired_exposure=desired_exposure, reason=reason, diagnostics=diagnostics)

    def generate_series(
        self,
        features_df: pd.DataFrame,
        risk_budgets: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Vectorised exposure series for backtesting.

        Runs the full Kalman + LSTM pipeline over the entire features_df and
        returns a per-bar exposure series.
        """
        if features_df is None or features_df.empty:
            return pd.Series(dtype=float)

        close = self._extract_close(features_df).dropna()
        if close.empty:
            return pd.Series(0.0, index=features_df.index, dtype=float)

        p = self.params

        # ── Kalman z-score series ─────────────────────────────────────
        kalman_z_series = self._kalman_z_series(close)

        # ── LSTM feature matrix ───────────────────────────────────────
        df_feats = features_df.loc[close.index]
        lstm_inputs = self._build_lstm_inputs(df_feats, kalman_z_series)
        lstm_out_series = self._lstm_signal_series(lstm_inputs)

        # ── Kalman signal per bar ─────────────────────────────────────
        x_kal = (p.k_kalman * kalman_z_series).clip(-60.0, 60.0)
        kalman_signal_series = _sigmoid_to_signed(x_kal)

        # ── Combine ───────────────────────────────────────────────────
        w = float(np.clip(p.kalman_weight, 0.0, 1.0))
        combined_series = w * kalman_signal_series + (1.0 - w) * lstm_out_series

        # ── Risk budget ───────────────────────────────────────────────
        if risk_budgets is not None:
            rb = risk_budgets.reindex(close.index).fillna(1.0).clip(0.0, 1.0)
        elif p.risk_budget_col in features_df.columns:
            rb = (
                pd.to_numeric(features_df[p.risk_budget_col], errors="coerce")
                .reindex(close.index)
                .fillna(1.0)
                .clip(0.0, 1.0)
            )
        else:
            rb = pd.Series(1.0, index=close.index)

        raw_exposure = combined_series * p.emax * rb
        exposure = raw_exposure.clip(0.0, 1.0)

        # Pad with 0 for insufficient history prefix
        result = pd.Series(0.0, index=features_df.index, dtype=float)
        valid_mask = pd.Series(False, index=features_df.index)
        valid_mask.loc[close.index] = True
        cumcount = valid_mask.cumsum()
        result.loc[valid_mask & (cumcount >= p.min_bars)] = exposure[cumcount[close.index] >= p.min_bars].values
        return result
