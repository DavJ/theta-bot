# Logtime (Mellin-Approx) Cepstrum Research Note

## 0) Executive summary
- Scale-invariant (logtime/Mellin-approx) cepstral phase ψ reduces redundancy with log-phase concentration C and improves ensemble S out-of-sample versus linear cepstrum.
- Linear cepstrum ψ is stable but adds limited orthogonal information; logtime ψ adds incremental regime signal.
- Targets are volatility proxies (y_vol) and absolute returns (y_absret); no direct price-direction prediction.
- Practical use is as a risk/regime layer (position sizing, exposure gating), not standalone alpha.
- Embargoed OOS tests on BTC/USDT 1h show S(logtime) AUC 0.618 > C AUC 0.615, while S(linear) AUC 0.598 < C AUC 0.615.
- Longer ψ windows (e.g., 512) degrade C_int stability; overly broad/top-k cepstral aggregation can hurt when band limits are tight.

## 1) Problem framing
- Goal: explore scale-aware, phase-based representations for volatility regimes on BTC/USDT 1h (research sandbox).
- Signals aim at next-horizon volatility proxies (y_vol) and absolute returns (y_absret), not price direction.
- Evaluate whether internal cepstral phase ψ adds regime information beyond scale concentration C.

## 2) Definitions and notation
Let \(p_t\) be close price and
\[
r_t = \log\frac{p_t}{p_{t-1}}.
\]
Example realized-volatility proxy over window \(w\):
\[
RV_t = \sqrt{\sum_{i=0}^{w-1} r_{t-i}^2 } .
\]

Phase mapping (scale removal):
\[
\operatorname{frac}(u) = u - \lfloor u \rfloor \in [0,1), \quad
\phi_t = 2\pi \, \operatorname{frac}\!\left( \log_b(x_t) \right),
\]
where \(x_t\) is a volatility feature (e.g., \(RV_t\)), base \(b>1\).

Circular embedding:
\[
z_t = e^{i\phi_t} = \cos\phi_t + i\sin\phi_t.
\]

Rolling concentration over window \(W\):
\[
R_t = \left| \frac{1}{W} \sum_{k=0}^{W-1} z_{t-k} \right|, \qquad C_t := R_t.
\]
Interpretation: high \(C_t\) = aligned phases (low dispersion); low \(C_t\) = dispersed regimes.

## 3) Cepstral internal phase ψ (linear-time)
For each rolling window of length \(N\):
1. Windowed series \(s[n],\ n=0..N-1\) (e.g., normalized RV segment).
2. FFT: \(S[k] = \text{FFT}(s)[k]\).
3. Log magnitude: \(L[k] = \log(|S[k]| + \varepsilon)\) with \(\varepsilon = 10^{-12}\) (EPS_LOG in implementation).
4. Cepstrum: \(c[n] = \text{IFFT}(L)[n]\).
5. Dominant quefrency in band \(n \in [n_{\min}, n_{\max}]\), where \(n_{\min}\) is set by `cepstrum_min_bin` (default 2) and \(n_{\max} = \lfloor \text{max\_frac} \cdot N \rfloor\):
\[
n^* = \arg\max_{n_{\min} \le n \le n_{\max}} |c[n]|.
\]
6. Internal phase:
\[
\psi_t = \operatorname{frac}\!\left( \frac{\arg(c[n^*])}{2\pi} \right) \in [0,1),
\]
where \(\arg(\cdot)\) is taken in \((-\pi,\pi]\); dividing by \(2\pi\) and taking the fractional part yields a toroidal phase in \([0,1)\).

## 4) Logtime (Mellin-approx) cepstrum
Key modification: logarithmic time warping before cepstrum to emphasize scale invariance.
1. Log-spaced indices for window \(N\) (as implemented):
\[
i_k = \left\lfloor \exp\!\big(\ell_k\big) \right\rfloor - 1,\quad \ell_k \in \text{linspace}(\log 1.0, \log N, N),
\]
clipped to \([0, N-1]\) and deduplicated via `np.unique` before interpolation.
2. Logtime sample: \(s_{\text{log}}[k] = s[i_k]\).
3. Resample \(s_{\text{log}}\) back to length \(N\) via linear interpolation to obtain \(\hat{s}[n]\).
4. Apply the same cepstrum steps on \(\hat{s}[n]\) to extract \(\psi_{\text{logtime},t}\).
Intuition: approximates Mellin/scale-invariant analysis using FFT on log-time.

## 5) Torus internal concentration \(C_{\text{int}}\) and ensemble \(S\)
Embed scale phase and internal phase:
\[
u_t = e^{i\phi_t}, \qquad v_t = e^{i 2\pi \psi_t}.
\]
Rolling torus concentration (window \(W\)):
\[
C_{\text{int},t} = \frac{1}{2} \left\| \frac{1}{W} \sum_{k=0}^{W-1} \begin{bmatrix}
\cos\phi_{t-k} \\ \sin\phi_{t-k} \\ \cos(2\pi\psi_{t-k}) \\ \sin(2\pi\psi_{t-k})
\end{bmatrix} \right\|,
\]
where the norm of a fully aligned 4D unit vector reaches 2, so the factor \(1/2\) scales the resultant to \([0,1]\) as implemented.

Ensemble score (non-directional):
\[
S_t = \frac{1}{2}\Big(\operatorname{rank}(C_t) + \operatorname{rank}(C_{\text{int},t})\Big).
\]

## 6) Evaluation protocol
- Data: BTC/USDT 1h from Binance (ccxt), OHLCV pagination, default limit_total=6000.
- Split: chronological train/test, split ratio 0.7 with embargo = max(horizon, target_window) (e.g., 24 bars).
- Targets: \(y_{\text{vol}}\) (forward volatility proxy), \(y_{\text{absret}}\) (forward absolute return).
- Metrics:
  - IC (Spearman/Pearson as implemented) vs \(y_{\text{vol}}\), \(y_{\text{absret}}\).
  - AUC for classifying high/low \(y_{\text{vol}}\) (thresholding per implementation).
  - Bucket ratio: top quantile vs bottom quantile of target conditioned on signal.
- Price direction is not modeled; signals are regime/risk only.

## 7) Empirical results (embargoed train/test)

**Table 1 — Linear vs Logtime Cepstrum (ψ-mode=cepstrum, N=256, min_bin=4, max_frac=0.2, embargoed split)**

| Domain   | Set   | C AUC | C_int AUC | S AUC |
|:--------:|:-----:|:-----:|:---------:|:-----:|
| Linear   | TRAIN | 0.658 | 0.603     | 0.647 |
| Linear   | TEST  | 0.615 | 0.578     | 0.598 |
| Logtime  | TRAIN | 0.658 | 0.613     | 0.630 |
| Logtime  | TEST  | 0.615 | 0.604     | 0.618 |

Window sensitivity notes:
- Worse (linear, OOS): \(S_{\text{AUC}} = 0.598 < C_{\text{AUC}} = 0.615\).
- Better (logtime, OOS): \(S_{\text{AUC}} = 0.618 > C_{\text{AUC}} = 0.615\).
- Very long \(\psi_{\text{window}}=512\) degrades \(C_{\text{int}}\) stability.
- Top-k cepstral averaging can worsen results, especially when the quefrency band is already restricted (min_bin/max_frac).

## 8) Key conclusion
The cepstral internal phase does not predict price direction; it captures volatility regime structure. Logtime/Mellin-approx cepstrum yields a \(\psi\) that is less redundant with \(C\) and improves embargoed OOS ensemble performance. The signal is best used as a risk/regime component (e.g., position sizing, exposure gating), not as standalone trading alpha.

## 9) Reproducibility (commands)
- Linear cepstrum embargo run:
```
python btc_log_phase_sweep.py \
  --symbol BTC/USDT --timeframe 1h --limit-total 6000 \
  --psi-mode cepstrum --psi-window 256 \
  --cepstrum-domain linear --cepstrum-min-bin 4 --cepstrum-max-frac 0.2 \
  --split 0.7 --embargo 24 --conc-window 256 --rv-window 24
```
- Logtime cepstrum embargo run:
```
python btc_log_phase_sweep.py \
  --symbol BTC/USDT --timeframe 1h --limit-total 6000 \
  --psi-mode cepstrum --psi-window 256 \
  --cepstrum-domain logtime --cepstrum-min-bin 4 --cepstrum-max-frac 0.2 \
  --split 0.7 --embargo 24 --conc-window 256 --rv-window 24
```
- Optional baseline sweep (no split, default domain linear):
```
python btc_log_phase_sweep.py --symbol BTC/USDT --timeframe 1h --limit-total 6000
```

## 10) Next steps
- Residualize \(\psi\) against \(C\) to test orthogonality.
- Test dissipative sources (drv, vol-of-vol).
- Validate across timeframes/assets.
- Add transaction-cost/execution modeling after stability is confirmed.
