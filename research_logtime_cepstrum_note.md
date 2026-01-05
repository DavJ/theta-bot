# Scale-Invariant Cepstral Phase for Volatility Regime Assessment (BTC/USDT 1h)

**Abstract.** This note examines whether cepstral internal phase features yield non-directional information about volatility regimes in BTC/USDT 1h data. A linear-time cepstrum and a logtime (Mellin-approximate) cepstrum extract toroidal phases that are aggregated into torus concentration \(C_{\text{int}}\) and a rank-based ensemble \(S\) with log-phase concentration \(C\). Embargoed out-of-sample tests show that the logtime cepstral phase reduces redundancy with \(C\) and improves ensemble AUC (0.618 vs 0.615 baseline), whereas the linear cepstral phase does not. Evaluation targets volatility proxies and absolute returns only; price-direction prediction is intentionally excluded. The study is exploratory and restricted to a BTC/USDT 1h sandbox, so universality is not claimed. Results indicate applicability as a risk/regime layer rather than standalone alpha.

## 0) Executive summary
**Findings**
- Logtime (Mellin-approx) cepstral phase \(\psi\) reduces redundancy with \(C\) and raises embargoed out-of-sample ensemble \(S\) AUC to 0.618 versus \(C\) at 0.615, while linear \(\psi\) lowers \(S\) AUC to 0.598.
- Linear-time cepstral \(\psi\) is stable yet contributes limited orthogonal signal; logtime \(\psi\) adds incremental regime information.
- Targets are volatility proxies (\(y_{\text{vol}}\)) and absolute returns (\(y_{\text{absret}}\)); no direct price-direction modeling is attempted.
- Longer \(\psi\) windows (e.g., 512) destabilize \(C_{\text{int}}\); broad or top-k cepstral aggregation harms results when quefrency bands are already tight.

**Implications**
- The signals are suited to risk/regime use (position sizing, exposure gating) and are not proposed as standalone directional alpha.
- Evidence is specific to BTC/USDT 1h under embargoed splits and should not be generalized without further testing.

## 1) Problem framing
- Scope: exploratory, non-directional assessment of scale-aware, phase-based representations for volatility regimes.
- Sandbox: BTC/USDT 1h (Binance) is used solely as a controlled setting, without asserting cross-asset or cross-horizon universality.
- Objective: test whether internal cepstral phase \(\psi\) adds volatility-regime information beyond log-phase concentration \(C\) for targets \(y_{\text{vol}}\) and \(y_{\text{absret}}\).

## 2) Definitions and notation
Let \(p_t\) denote close price and define log return
\[
r_t = \log\frac{p_t}{p_{t-1}}.
\]
A realized-volatility proxy over window \(w\) is
\[
RV_t = \sqrt{\sum_{i=0}^{w-1} r_{t-i}^2 } .
\]

Phase mapping (scale removal):
\[
\operatorname{frac}(u) = u - \lfloor u \rfloor \in [0,1), \quad
\phi_t = 2\pi \, \operatorname{frac}\!\left( \log_b(x_t) \right),
\]
where \(x_t\) is a volatility feature (e.g., \(RV_t\)) and \(b>1\).

Circular embedding:
\[
z_t = e^{i\phi_t} = \cos\phi_t + i\sin\phi_t.
\]

Rolling log-phase concentration over window \(W\):
\[
C_t = \left| \frac{1}{W} \sum_{k=0}^{W-1} z_{t-k} \right|.
\]

## 3) Cepstral internal phase \(\psi\) (linear-time cepstrum)
For each rolling window of length \(N\):
1. Windowed series \(s[n],\ n=0..N-1\) (normalized volatility segment).
2. FFT: \(S[k] = \text{FFT}(s)[k]\).
3. Log magnitude: \(L[k] = \log(|S[k]| + \varepsilon)\) with \(\varepsilon = 10^{-12}\).
4. Cepstrum: \(c[n] = \text{IFFT}(L)[n]\).
5. Dominant quefrency in band \(n \in [n_{\min}, n_{\max}]\), with \(n_{\min} =\) `cepstrum_min_bin` and \(n_{\max} = \lfloor \text{max\_frac} \cdot N \rfloor\):
\[
n^* = \arg\max_{n_{\min} \le n \le n_{\max}} |c[n]|.
\]
6. Internal cepstral phase (toroidal):
\[
\psi_t = \operatorname{frac}\!\left( \frac{\arg(c[n^*])}{2\pi} \right) \in [0,1),
\]
where \(\arg(\cdot)\) is taken in \((-\pi,\pi]\); using the argument preserves the orientation of the dominant quefrency component while mapping it to a periodic phase.

### Real vs complex cepstrum phase
- **Real cepstrum degeneracy:** When only \(\log |S[k]|\) is inverted, \(c[n]\) is (numerically) real for real-valued inputs. Its phase collapses to \(\{0, \pi\}\), so \(\psi\) often takes only two toroidal values.
- **Complex cepstrum fix:** Form the complex log spectrum with unwrapped phase, \(L[k] = \log(|S[k]| + \varepsilon) + i\,\operatorname{unwrap}(\angle S[k])\), then \(c[n] = \text{IFFT}(L[k])\). The dominant coefficient \(c[n^*]\) now carries a meaningful internal angle; \(\psi = \operatorname{frac}(\angle c[n^*]/2\pi)\) varies across \([0,1)\).

## 4) Logtime (Mellin-approx) cepstrum
Modification: logarithmic time warping before the cepstrum to approximate scale invariance (approximate Mellin behavior, not a full Mellin transform).
1. Log-spaced indices for window \(N\):
\[
i_k = \left\lfloor \exp\!\big(\ell_k\big) \right\rfloor - 1,\quad \ell_k \in \text{linspace}(\log 1.0, \log N, N),
\]
clipped to \([0, N-1]\) and deduplicated via `np.unique` before interpolation.
2. Logtime sample: \(s_{\text{log}}[k] = s[i_k]\).
3. Resample \(s_{\text{log}}\) to length \(N\) via linear interpolation to obtain \(\hat{s}[n]\).
4. Apply the linear-time cepstrum steps to \(\hat{s}[n]\) to extract \(\psi_{\text{logtime},t}\).
The logtime variant yields a cepstral phase that is more scale-invariant and less redundant with \(C\).

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
where the norm of a fully aligned 4D unit vector reaches 2; the factor \(1/2\) scales the resultant to \([0,1]\) as implemented.

Rank-based, non-directional ensemble:
\[
S_t = \frac{1}{2}\Big(\operatorname{rank}(C_t) + \operatorname{rank}(C_{\text{int},t})\Big).
\]
Ranks are computed cross-sectionally, so \(S_t\) aggregates concentrations without directional exposure.

## 6) Evaluation protocol (methods)
- Data: BTC/USDT 1h (Binance via ccxt), OHLCV pagination with `limit_total=6000`.
- Split: chronological train/test with ratio 0.7 and an embargo equal to \(\max(\text{horizon}, \text{target\_window})\) (e.g., 24 bars) applied around the split.
- Targets: forward volatility proxy \(y_{\text{vol}}\) and forward absolute return \(y_{\text{absret}}\); price-direction targets are intentionally excluded.
- Metrics: Spearman/Pearson IC versus \(y_{\text{vol}}\) and \(y_{\text{absret}}\); AUC for high/low \(y_{\text{vol}}\) classification; bucket ratio (top vs bottom quantile of target conditioned on signal).
- Signals: non-directional regime/risk scores only; no directional forecasting is attempted.

## 7) Empirical results (embargoed train/test)

**Table 1 — Linear vs Logtime Cepstrum (ψ-mode=cepstrum, N=256, min_bin=4, max_frac=0.2, embargoed split)**

| Domain   | Set   | C AUC | C_int AUC | S AUC |
|:--------:|:-----:|:-----:|:---------:|:-----:|
| Linear   | TRAIN | 0.658 | 0.603     | 0.647 |
| Linear   | TEST  | 0.615 | 0.578     | 0.598 |
| Logtime  | TRAIN | 0.658 | 0.613     | 0.630 |
| Logtime  | TEST  | 0.615 | 0.604     | 0.618 |

Comparative statements under embargoed out-of-sample testing:
- Linear domain: \(S_{\text{AUC}} = 0.598\) is lower than \(C_{\text{AUC}} = 0.615\), indicating no ensemble gain.
- Logtime domain: \(S_{\text{AUC}} = 0.618\) exceeds \(C_{\text{AUC}} = 0.615\), indicating a modest ensemble gain.
- Extended \(\psi\) windows (e.g., 512) reduce \(C_{\text{int}}\) stability.
- Top-k cepstral averaging deteriorates performance when quefrency bands are already constrained by \(n_{\min}\) and \(n_{\max}\).

## 8) Conclusions
Demonstrated: logtime cepstral internal phase yields a less redundant signal with \(C\) and improves embargoed out-of-sample ensemble AUC for volatility-regime targets in BTC/USDT 1h. Linear cepstral phase does not improve ensemble performance under the same protocol.

Not demonstrated: directional alpha or universal cross-asset validity. The evidence is limited to a non-directional, volatility-focused sandbox.

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
- Residualize \(\psi\) against \(C\) to quantify orthogonality and incremental information.
- Probe dissipative sources (e.g., realized variance of returns, volatility-of-volatility) within the same embargoed protocol.
- Validate across additional timeframes and assets with identical splits before any deployment consideration.
- Incorporate transaction-cost and execution constraints only after stability is confirmed across assets/timeframes.
