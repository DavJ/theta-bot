python theta_bot_averaging/theta_eval_hbatch_jacobi_fixed_leak.py --symbols "$PWD/prices/BTCUSDT_1h.csv,$PWD/prices/ETHUSDT_1h.csv"   --interval 1h --window 512 --horizon 8 --baseP 36.0   --N-even 6   --N-odd 6   --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --out theta_trading_addon/results/hbatch_biquat_summary.csv

=== Running /Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv ===


Uloženo CSV: eval_h_BTCUSDT_1H.csv
Error saving JSON summary: Object of type int64 is not JSON serializable

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---
hit_rate_pred:   0.753653
hit_rate_hold:   0.482255
corr_pred_true:  0.759012
mae_price (delta): 824.840972
mae_return:      0.007255
count:          479


=== Running /Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv ===


Uloženo CSV: eval_h_ETHUSDT_1H.csv
Error saving JSON summary: Object of type int64 is not JSON serializable

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---
hit_rate_pred:   0.757829
hit_rate_hold:   0.496868
corr_pred_true:  0.765551
mae_price (delta): 49.006330
mae_return:      0.011999
count:          479


Uloženo: theta_trading_addon/results/hbatch_biquat_summary.csv
                                                     symbol target_type  window  horizon     baseP    sigma  N_even  N_odd  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true  mae_price  mae_return  count
/Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6       0.753653       0.482255   0.271399        0.759012 824.840972    0.007255    479
/Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6       0.757829       0.496868   0.260960        0.765551  49.006330    0.011999    479


##################################################################################################################################################################################################################

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (fixing-leakage)$ python theta_bot_averaging/theta_eval_hbatch_jacobi_fixed_leak_slipping_window.py --symbols "$PWD/prices/BTCUSDT_1h.csv,$PWD/prices/ETHUSDT_1h.csv"   --interval 1h --window 512 --horizon 8 --baseP 36.0   --N-even 6   --N-odd 6   --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --out theta_trading_addon/results/hbatch_biquat_summary.csv

=== Running /Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H.csv
Uloženo summary: sum_h_BTCUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected & Robust) ---
hit_rate_pred:   0.564583
hit_rate_hold:   0.481250
corr_pred_true:  0.207473
mae_price (delta): 1194.496246
mae_return:      0.010457
count:          480

Sanity Shuffle Corr: 0.0804
Sanity Lag-1 Corr:   0.1853

=== Running /Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H.csv
Uloženo summary: sum_h_ETHUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected & Robust) ---
hit_rate_pred:   0.552083
hit_rate_hold:   0.497917
corr_pred_true:  0.142315
mae_price (delta): 71.608268
mae_return:      0.017429
count:          480

Sanity Shuffle Corr: 0.0175
Sanity Lag-1 Corr:   0.1358

Uloženo (pouze úspěšné běhy): theta_trading_addon/results/hbatch_biquat_summary.csv
                                                     symbol target_type  window  horizon     baseP    sigma  N_even  N_odd   lambda  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_price  mae_return  count  corr_shuffle  corr_lag1
/Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.564583       0.481250   0.083333        0.207473 1194.496246    0.010457    480      0.080364   0.185273
/Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.552083       0.497917   0.054167        0.142315   71.608268    0.017429    480      0.017496   0.135815

##################################################################################################################################################################################################################

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (fixing-leakage)$ python theta_bot_averaging/theta_eval_hbatch_jacobi_fixed_leak_walk_forward.py --symbols "$PWD/prices/BTCUSDT_1h.csv,$PWD/prices/ETHUSDT_1h.csv"   --interval 1h --window 512 --horizon 8 --baseP 36.0   --N-even 6   --N-odd 6   --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --out theta_trading_addon/results/hbatch_biquat_summary.csv

=== Running /Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv ===

Starting walk-forward from index 512 to 991

Uloženo CSV: eval_h_BTCUSDT_1H.csv
Sanity Check - Shuffle Corr: 0.0804
Sanity Check - Lag-1 Corr:   0.1853
Uloženo summary: sum_h_BTCUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---
hit_rate_pred:   0.564583
hit_rate_hold:   0.481250
corr_pred_true:  0.207473
mae_price (delta): 1194.496247
mae_return:      0.010457
count:          480

Sanity Shuffle Corr: 0.0804
Sanity Lag-1 Corr:   0.1853

=== Running /Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv ===

Starting walk-forward from index 512 to 991

Uloženo CSV: eval_h_ETHUSDT_1H.csv
Sanity Check - Shuffle Corr: 0.0175
Sanity Check - Lag-1 Corr:   0.1358
Uloženo summary: sum_h_ETHUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---
hit_rate_pred:   0.552083
hit_rate_hold:   0.497917
corr_pred_true:  0.142315
mae_price (delta): 71.608268
mae_return:      0.017429
count:          480

Sanity Shuffle Corr: 0.0175
Sanity Lag-1 Corr:   0.1358

Uloženo: theta_trading_addon/results/hbatch_biquat_summary.csv
                                                     symbol target_type  window  horizon     baseP    sigma  N_even  N_odd   lambda  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_price  mae_return  count  corr_shuffle  corr_lag1
/Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.564583       0.481250   0.083333        0.207473 1194.496247    0.010457    480      0.080364   0.185273
/Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.552083       0.497917   0.054167        0.142315   71.608268    0.017429    480      0.017496   0.135815


