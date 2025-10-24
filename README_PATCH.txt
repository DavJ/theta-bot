theta-bot patch: add per-row analytics columns
==================================================

This patch updates `theta_eval_hbatch_biquat_max.py` so that each
generated `eval_h_*.csv` contains the following additional columns:

  - pred_dir        : sign of predicted delta (-1, 0, +1)
  - true_dir        : sign of realized delta (-1, 0, +1)
  - correct_pred    : 1 if pred_dir == true_dir and pred_dir != 0 else 0
  - hold_ret        : buy&hold return over the same horizon

These fields are required by the OOS suite and downstream analytics.
The logic uses only same-bar info (last_price) and the future bar at
horizon (future_price) that is already being used to compute true_delta,
so there is no data leakage.

How to apply:
  1) Make a backup of your current file:
     cp theta_bot_averaging/theta_eval_hbatch_biquat_max.py             theta_bot_averaging/theta_eval_hbatch_biquat_max.py.bak

  2) Copy the patched file from this zip into your repo, e.g.:
     unzip patch_eval_columns.zip
     ./apply_patch.sh

  3) Re-run your evaluator and OOS suite.

