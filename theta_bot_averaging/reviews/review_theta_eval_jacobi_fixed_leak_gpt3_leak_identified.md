def evaluate_symbol_csv(symbol_path, interval, window, horizon,
minP, maxP, nP,
sigma, lam,
pred_ensemble, max_by,
out_dir):
df = pd.read_csv(symbol_path)
df.columns = [c.lower() for c in df.columns]


closes = df['close'].values
times = df['time'].values


rows = []


# Původní verze: celý dataset je použit pro každý trénovací krok
for entry_idx in range(window + maxP, len(closes) - horizon):
last_price = closes[entry_idx]
future_price = closes[entry_idx + horizon]


x_now, contrib_per_P = compute_theta_features(closes, entry_idx, minP, maxP, nP, window, sigma)


X = []
Y = []
for idx in range(window + maxP, len(closes) - horizon):
x_i, _ = compute_theta_features(closes, idx, minP, maxP, nP, window, sigma)
X.append(x_i)
delta = closes[idx + horizon] - closes[idx]
Y.append(delta)


X = np.array(X)
Y = np.array(Y)


beta = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ Y


if pred_ensemble == 'avg':
pred_delta = float(x_now @ beta)
else:
k = int(np.argmax(contrib_per_P))
pred_delta = float(x_now[2 * k:2 * k + 2] @ beta[2 * k:2 * k + 2])


true_delta = future_price - last_price


pred_dir = int(np.sign(pred_delta))
true_dir = int(np.sign(true_delta))
correct_pred_val = 1 if pred_dir == true_dir else 0


rows.append({
'time': str(times[entry_idx]),
'entry_idx': int(entry_idx),
'compare_idx': int(entry_idx + horizon),
'last_price': float(last_price),
'pred_price': float(last_price + pred_delta),
'future_price': float(future_price),
'pred_delta': float(pred_delta),
'true_delta': float(true_delta),
'correct_pred': int(correct_pred_val)
})


# Problém: X, Y jsou trénovány z celého datasetu včetně budoucnosti, tj. i z dat po entry_idx
# -> Model trénovaný na budoucích informacích = leakage


df_out = pd.DataFrame(rows)
out_path = get_out_path(symbol_path, out_dir)
df_out.to_csv(out_path, index=False)


return compute_summary(df_out)
