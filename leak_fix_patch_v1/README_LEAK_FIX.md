# Leak-Fix Patch (v1) — theta_eval_hbatch_biquat_max.py

This patch removes **look‑ahead leakage** from the rolling trainer in
`theta_bot_averaging/theta_eval_hbatch_biquat_max.py` by ensuring the training
window never uses targets that extend **past** the current `entry_idx`.

## What’s the issue?

In some versions of the script, the training upper bound was:
```python
hi = compare_idx        # ❌ leads to leakage
Xw = X_all[lo:hi, :]
yw = closes[lo+horizon : hi+horizon] - closes[lo:hi]
```
For the last training sample `i = hi-1 = compare_idx-1 = entry_idx`, the target
touches `close[entry_idx + horizon]` (a future price) → **data leak**.

## The fix (conceptually)

Train only up to `hi_tr = compare_idx - horizon` so the latest target ends at
`entry_idx`:
```python
hi_tr = compare_idx - horizon
Xw = X_all[lo:hi_tr, :]
yw = closes[lo+horizon : hi_tr+horizon] - closes[lo:hi_tr]
assert (hi_tr + horizon) <= (entry_idx + 1)   # leak guard
```

## How to apply

### Option A — One‑shot auto‑patcher (recommended)

```bash
# from repo root
python leak_fix_patch_v1/scripts/apply_leak_fix.py   theta_bot_averaging/theta_eval_hbatch_biquat_max.py
```

- The patcher is **idempotent**: if the file is already fixed (contains `hi_tr`),
  it won’t modify again.
- A backup of the original file is written next to it with suffix `.bak`.

### Option B — Manual edit (if the patcher can’t find the exact pattern)

In your file, find the block that builds the rolling training window and
replace:

- `hi = compare_idx` → `hi_tr = compare_idx - horizon`
- All slices `[...]lo:hi...]` used for **training** become `[...]lo:hi_tr...]`
- Targets `closes[lo+horizon : hi+horizon]` become `closes[lo+horizon : hi_tr+horizon]`
- Add a guard right after computing `hi_tr`:
  ```python
  if hi_tr <= lo:
      continue
  assert (hi_tr + horizon) <= (entry_idx + 1), "Leak guard failed"
  ```

### Quick sanity checks (optional)

Run your existing OOS suite (great you already have it):
```bash
python robustness_suite_v3_oos.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT   --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 16   --sigma 0.8 --lambda 1e-3 --limit 2000   --pred-ensemble avg --max-by transform   --oos-split 0.7   --out robustness_report_v3_oos.csv
```

You should continue seeing no‑leak flags, with OOS `corr` ≈ 0.50–0.60 and hit‑rate
≈ 0.62–0.66 (numbers will vary slightly by environment).

---

### Notes

- The patcher targets the common pattern used in your repo (based on the file
  you shared). If your local variant differs substantially, use **Option B**.
- Keep the assert — it’s a cheap runtime tripwire against future regressions.
