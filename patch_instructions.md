# CSV QoL + Inverse-Transform Patch (zip)

This patch adds:
1. **CSV QoL flags** to `theta_eval_hbatch_biquat.py`  
   - `--csv-time-col` (default: `time`)  
   - `--csv-close-col` (default: `close`)  
   Use these when `--symbols` points to a CSV file.
2. **CSV loader** inside `fetch_ohlcv()` that reads arbitrary CSV paths with the chosen column names.
3. **Inverse-transform block** mapping normalized delta back to **price** before metrics (so MAE returns look sane).

## How to apply

```bash
unzip theta_csv_qol_patch.zip -d theta_csv_qol_patch
cd theta_csv_qol_patch

# Run from the repo root where theta_eval_hbatch_biquat.py lives:
python apply_patch.py
```

- The patcher creates a timestamped backup in `./backup/` before changing anything.
- If it cannot find a safe place for the inverse-transform, it will still add the CSV flags/loader and will print a 6‑line snippet for manual paste.

## Usage (after patch)

```bash
python theta_eval_hbatch_biquat.py   --symbols /absolute/path/to/your.csv   --csv-time-col timestamp   --csv-close-col close_price   --interval 1h --window 256 --horizon 4   --minP 24 --maxP 480 --nP 12   --sigma 0.8 --lambda 1e-3 --limit 2000   --phase biquat   --out hbatch_from_csv.csv
```

> If your model outputs a **normalized delta**, the inverse-transform is correct as provided.  
> If it outputs a **normalized level (z-price)**, replace the snippet by:
>
> ```python
> price_hat = mu + sigma * float(y_hat_norm)
> pred_price = float(price_hat)
> ```
>
> and remove the `last_price + ...` line.
