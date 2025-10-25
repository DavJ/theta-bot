NOTE: these scripts seems to be non-quaternionic

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


###################################################(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (fixing-leakage)$ python theta_bot_averaging/theta_eval_hbatch_jacobi_fixed_leak2.py --symbols "$PWD/prices/BTCUSDT_1h.csv,$PWD/prices/ETHUSDT_1h.csv"   --interval 1h --window 512 --horizon 8 --baseP 36.0   --N-even 6   --N-odd 6   --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --out theta_trading_addon/results/hbatch_biquat_summary.csv

=== Running /Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H.csv
Uloženo summary: sum_h_BTCUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.558333
hit_rate_hold:   0.481250
corr_pred_true:  0.121301
mae_price (delta): 1212.182063
mae_return:      0.010611
count:          480

Sanity Shuffle Corr: 0.0699
Sanity Lag-1 Corr:   0.1103

=== Running /Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H.csv
Uloženo summary: sum_h_ETHUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.539583
hit_rate_hold:   0.497917
corr_pred_true:  0.057726
mae_price (delta): 72.735287
mae_return:      0.017704
count:          480

Sanity Shuffle Corr: 0.0108
Sanity Lag-1 Corr:   0.0611

Uloženo (pouze úspěšné běhy): theta_trading_addon/results/hbatch_biquat_summary.csv
                                                     symbol target_type  window  horizon     baseP    sigma  N_even  N_odd   lambda  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_price  mae_return  count  corr_shuffle  corr_lag1
/Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.558333       0.481250   0.077083        0.121301 1212.182063    0.010611    480      0.069919   0.110344
/Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.539583       0.497917   0.041667        0.057726   72.735287    0.017704    480      0.010845   0.061068
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (fixing-leakage)$ 




(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (fixing-leakage)$ python theta_bot_averaging/theta_eval_hbatch_jacobi_fixed_leak2.py --symbols "$PWD/prices/BTCUSDT_1h.csv,$PWD/prices/ETHUSDT_1h.csv"   --interval 1h --window 512 --horizon 8 --baseP 36.0   --N-even 6   --N-odd 6   --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --out theta_trading_addon/results/hbatch_biquat_summary.csv

=== Running /Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H.csv
Uloženo summary: sum_h_BTCUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.558333
hit_rate_hold:   0.481250
corr_pred_true:  0.121301
mae_price (delta): 1212.182063
mae_return:      0.010611
count:          480

Sanity Shuffle Corr: 0.0699
Sanity Lag-1 Corr:   0.1103

=== Running /Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H.csv
Uloženo summary: sum_h_ETHUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.539583
hit_rate_hold:   0.497917
corr_pred_true:  0.057726
mae_price (delta): 72.735287
mae_return:      0.017704
count:          480

Sanity Shuffle Corr: 0.0108
Sanity Lag-1 Corr:   0.0611

Uloženo (pouze úspěšné běhy): theta_trading_addon/results/hbatch_biquat_summary.csv
                                                     symbol target_type  window  horizon     baseP    sigma  N_even  N_odd   lambda  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_price  mae_return  count  corr_shuffle  corr_lag1
/Users/davidjaros/workspace/theta-bot/prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.558333       0.481250   0.077083        0.121301 1212.182063    0.010611    480      0.069919   0.110344
/Users/davidjaros/workspace/theta-bot/prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.539583       0.497917   0.041667        0.057726   72.735287    0.017704    480      0.010845   0.061068

########################################################################################################################

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot/theta_bot_averaging (make-theta-biquat)$ python theta_biquat_predict_diag.py \
>   --symbols "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv" \
>   --csv-time-col time \
>   --csv-close-col close \
>   --window 512 \
>   --horizon 8 \
>   --baseP 36.0 \
>   --N-even 6 \
>   --N-odd 6 \
>   --sigma 0.8 \
>   --lambda 1e-3 \
>   --target-type delta \
>   --ema-alpha 0.0 \
>   --out results_diag/summary_diag.csv


=== Running ../prices/BTCUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.121
[DBG] corr(price,price) = 0.963
[DBG] hit rate        = 0.558
[DBG] anti-hit rate   = 0.442
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 6.419e+04
-------------------------

Uloženo summary: sum_h_BTCUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.558333
hit_rate_hold:   0.481250
corr_pred_true:  0.121301
mae_price (delta): 1212.182063
mae_return:      0.010611
count:          480


=== Running ../prices/ETHUSDT_1h.csv ===

Starting walk-forward from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.058
[DBG] corr(price,price) = 0.937
[DBG] hit rate        = 0.540
[DBG] anti-hit rate   = 0.460
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 110
-------------------------

Uloženo summary: sum_h_ETHUSDT_1H.json

--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---
hit_rate_pred:   0.539583
hit_rate_hold:   0.497917
corr_pred_true:  0.057726
mae_price (delta): 72.735287
mae_return:      0.017704
count:          480


Uloženo (pouze úspěšné běhy): results_diag/summary_diag.csv
                  symbol target_type  window  horizon     baseP    sigma  N_even  N_odd   lambda  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_price  mae_return  count  corr_shuffle  corr_lag1
../prices/BTCUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.558333       0.481250   0.077083        0.121301 1212.182063    0.010611    480      0.069919   0.110344
../prices/ETHUSDT_1h.csv       delta     512        8 36.000000 0.800000       6      6 0.001000       0.539583       0.497917   0.041667        0.057726   72.735287    0.017704    480      0.010845   0.061068


########################################################################################################################

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot/theta_bot_averaging (make-theta-biquat)$ python theta_eval_gpt_ridge.py \
>   --symbols "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv" \
>   --csv-time-col time \
>   --csv-close-col close \
>   --window 512 \
>   --horizon 8 \
>   --q 0.5 \
>   --lambda 1e-3 \
>   --ema-alpha 0.0 \
>   --out results_gpt_ridge/summary_gptRidge.csv

=== Running ../prices/BTCUSDT_1h.csv ===

Generating theta basis with q=0.5 for 1000 points...
Starting Walk-Forward Ridge from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H_gptRidge.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.135
[DBG] corr(price,price) = 0.536
[DBG] hit rate        = 0.519
[DBG] anti-hit rate   = 0.481
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 1.758e+06
-------------------------

Uloženo summary: sum_h_BTCUSDT_1H_gptRidge.json

--- HSTRATEGY vs HOLD (GPT Logic with Ridge) ---
hit_rate_pred:   0.518750
hit_rate_hold:   0.481250
corr_pred_true (Δ):  0.135421
mae_delta:      11034.403487
mae_return:      0.095168
count:          480

Diag corr_price:  0.535596
Diag anti_hit:    0.481250
Diag zero_rate:   0.000000

=== Running ../prices/ETHUSDT_1h.csv ===

Generating theta basis with q=0.5 for 1000 points...
Starting Walk-Forward Ridge from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H_gptRidge.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.178
[DBG] corr(price,price) = 0.683
[DBG] hit rate        = 0.535
[DBG] anti-hit rate   = 0.465
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 4551
-------------------------

Uloženo summary: sum_h_ETHUSDT_1H_gptRidge.json

--- HSTRATEGY vs HOLD (GPT Logic with Ridge) ---
hit_rate_pred:   0.535417
hit_rate_hold:   0.497917
corr_pred_true (Δ):  0.178374
mae_delta:      403.433442
mae_return:      0.096065
count:          480

Diag corr_price:  0.682871
Diag anti_hit:    0.464583
Diag zero_rate:   0.000000

Uloženo (pouze úspěšné běhy): results_gpt_ridge/summary_gptRidge.csv
                  symbol  window  horizon  q_param   lambda  ema_alpha  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true    mae_delta  mae_return  count  corr_price  anti_hit_rate  zero_rate  mean_delta_prod
../prices/BTCUSDT_1h.csv     512        8 0.500000 0.001000   0.000000       0.518750       0.481250   0.037500        0.135421 11034.403487    0.095168    480    0.535596       0.481250   0.000000   1757659.659701
../prices/ETHUSDT_1h.csv     512        8 0.500000 0.001000   0.000000       0.535417       0.497917   0.037500        0.178374   403.433442    0.096065    480    0.682871       0.464583   0.000000      4551.082325


########################################################################################################################
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot/theta_bot_averaging (make-theta-biquat)$ python theta_eval_gpt_ridge_delta.py   --symbols "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv"   --csv-time-col time   --csv-close-col close   --window 512   --horizon 8   --q 0.5   --lambda 1e-3   --ema-alpha 0.0   --out results_gpt_ridge_delta/summary_gptRidgeDelta.csv

=== Running ../prices/BTCUSDT_1h.csv ===

Generating theta basis with q=0.5 for 1000 points...
Starting Walk-Forward Ridge (Delta Target) from index t0=512 up to 991

Uloženo CSV: eval_h_BTCUSDT_1H_gptRidgeDelta.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.213
[DBG] corr(price,price) = 0.965
[DBG] hit rate        = 0.552
[DBG] anti-hit rate   = 0.448
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 1.533e+05
-------------------------

Uloženo summary: sum_h_BTCUSDT_1H_gptRidgeDelta.json

--- HSTRATEGY vs HOLD (GPT Logic + Ridge on Delta) ---
hit_rate_pred:   0.552083
hit_rate_hold:   0.481250
corr_pred_true (Δ):  0.213204
mae_delta:      1222.172974
mae_return:      0.010682
count:          480

Diag corr_price:  0.964816
Diag anti_hit:    0.447917
Diag zero_rate:   0.000000

=== Running ../prices/ETHUSDT_1h.csv ===

Generating theta basis with q=0.5 for 1000 points...
Starting Walk-Forward Ridge (Delta Target) from index t0=512 up to 991

Uloženo CSV: eval_h_ETHUSDT_1H_gptRidgeDelta.csv

--- DEBUG DIAGNOSTICS ---
[DBG] corr(Δ,Δ)         = 0.121
[DBG] corr(price,price) = 0.940
[DBG] hit rate        = 0.508
[DBG] anti-hit rate   = 0.492
[DBG] zero rate       = 0.000
[DBG] sum check (h+a+z) = 1.000
[DBG] mean(Δ_pred*Δ_true) = 305.4
-------------------------

Uloženo summary: sum_h_ETHUSDT_1H_gptRidgeDelta.json

--- HSTRATEGY vs HOLD (GPT Logic + Ridge on Delta) ---
hit_rate_pred:   0.508333
hit_rate_hold:   0.497917
corr_pred_true (Δ):  0.121484
mae_delta:      72.366542
mae_return:      0.017603
count:          480

Diag corr_price:  0.939541
Diag anti_hit:    0.491667
Diag zero_rate:   0.000000

Uloženo (pouze úspěšné běhy): results_gpt_ridge_delta/summary_gptRidgeDelta.csv
                  symbol  window  horizon  q_param   lambda  ema_alpha target_type  hit_rate_pred  hit_rate_hold  delta_hit  corr_pred_true   mae_delta  mae_return  count  corr_price  anti_hit_rate  zero_rate  mean_delta_prod
../prices/BTCUSDT_1h.csv     512        8 0.500000 0.001000   0.000000       delta       0.552083       0.481250   0.070833        0.213204 1222.172974    0.010682    480    0.964816       0.447917   0.000000    153251.958991
../prices/ETHUSDT_1h.csv     512        8 0.500000 0.001000   0.000000       delta       0.508333       0.497917   0.010417        0.121484   72.366542    0.017603    480    0.939541       0.491667   0.000000       305.415913


