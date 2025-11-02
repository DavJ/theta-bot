# 🧩 Copilot Implementation Brief – theta_predictor_v9

## 🎯 Goal
Enhance the current `theta_predictor.py` implementation to:
1. Reintroduce **biquaternionic time**,
2. Add a **Fokker–Planck drift term**,
3. Support **market regime detection** (PCA-based),
4. Maintain strict **walk-forward validation**,
5. Stay consistent with **UBT theoretical framework**.

---

## ⚙️ Technical Instructions

### 1️⃣ Biquaternionic Time Support
Define the biquaternionic time variable:
```python
tau = t + j * psi
```
where `j` is an imaginary quaternionic unit. Represent each theta coefficient as:
\[ \Theta_k = a_k + i b_k + j c_k + k d_k \]
Use complex projection for regression:
```python
def project_to_complex(theta_biquat):
    return theta_biquat[..., 0] + 1j * theta_biquat[..., 1]
```

### 2️⃣ Fokker–Planck Drift Term
Add drift term capturing macro bias:
\[ A_t = \beta_0 + \beta_1 \tanh(EMA_{16}(r_t)) \]
Example:
```python
ema_drift = prices.diff().ewm(span=16).mean()
drift_term = beta0 + beta1 * np.tanh(ema_drift)
pred = theta_pred + drift_term
```
Train `beta0`, `beta1` within each walk-forward window.

### 3️⃣ PCA Regime Detection
Perform PCA on theta features:
```python
pca = PCA(n_components=2)
theta_pca = pca.fit_transform(theta_features)
```
Cluster or segment PCA space to detect trend/mean-reverting regimes and toggle drift adaptively.

### 4️⃣ CLI Parameters
```
--enable-biquaternion
--enable-drift
--enable-pca-regimes
--ridge-lambda
--drift-beta0
--drift-beta1
```

### 5️⃣ Walk-Forward Validation
Ensure both ridge and drift components are refitted in each WF step using only past data.

### 6️⃣ Visualization
Generate additional plots:
- `theta_output/theta_drift_overlay.png`
- `theta_output/theta_regime_clusters.png`
- `theta_output/theta_biquat_projection.png`

---

## 📊 Expected Results
- Correlation: 0.10–0.20
- Hit rate: 52–56%
- Sharpe improvement > +0.3 vs v8
- Drift term reflects sentiment bias
- Full UBT theoretical consistency (τ = t + jψ)

---

## 📚 Documentation Update
Add section to README.md:
> ### Theta Predictor v9 – Biquaternion Drift Model
> Introduces a biquaternionic time base, Fokker–Planck drift coupling,  
> and adaptive regime detection via PCA. Restores the weak but real predictive edge  
> observed in earlier biquaternionic implementations while maintaining strict walk-forward causality.
