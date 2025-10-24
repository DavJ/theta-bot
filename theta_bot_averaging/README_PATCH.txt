This patch updates theta_eval_hbatch_biquat_max.py to:
- Prevent lookahead leakage (train until hi_horizon = hi - horizon; predict at hi-1).
- Align target y with the training slice.
- Add evaluation CSV columns: pred_dir, true_dir, correct_pred.

Apply:
  cp patch_theta_eval_full/theta_eval_hbatch_biquat_max.py theta_bot_averaging/theta_eval_hbatch_biquat_max.py
