1) NOTE: THESE SCRIPTS SHOULD BE QUATERNIONIC BUT ARE NOT YET (IT'S ONLY FIRST STEP STILL 2D) 

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py   --csv prices/BTCUSDT_1h.csv   --price-col close   --horizon 1 --window 512   --q 0.6 --n-terms 24 --n-freq 8   --lambda 1.0   --outdir theta_bot_averaging/results_biquat_btc
Saved predictions to: theta_bot_averaging/results_biquat_btc/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc/summary_biquat_ridge.csv
Metrics: {'n_samples': 487, 'corr_pred_true': -0.07072828913520914, 'hit_rate': np.float64(0.4702258726899384), 'mae': 389.88522884923054, 'rmse': 550.199776180829, 'window': 512, 'horizon': 1, 'q': 0.6, 'n_terms': 24, 'n_freq': 8, 'lambda': 1.0, 'phase_scale': 1.0}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py   --csv prices/BTCUSDT_1h.csv   --price-col close   --horizon 8 --window 512   --q 0.6 --n-terms 24 --n-freq 8   --lambda 1.0   --outdir theta_bot_averaging/results_biquat_btc
Saved predictions to: theta_bot_averaging/results_biquat_btc/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc/summary_biquat_ridge.csv
Metrics: {'n_samples': 480, 'corr_pred_true': -0.08201186768451846, 'hit_rate': np.float64(0.51875), 'mae': 1212.6492259203346, 'rmse': 1740.9441796036285, 'window': 512, 'horizon': 8, 'q': 0.6, 'n_terms': 24, 'n_freq': 8, 'lambda': 1.0, 'phase_scale': 1.0}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py   --csv prices/BTCUSDT_1h.csv   --price-col close   --horizon 4 --window 512   --q 0.6 --n-terms 24 --n-freq 8   --lambda 1.0   --outdir theta_bot_averaging/results_biquat_btc
Saved predictions to: theta_bot_averaging/results_biquat_btc/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc/summary_biquat_ridge.csv
Metrics: {'n_samples': 484, 'corr_pred_true': -0.07592052572680738, 'hit_rate': np.float64(0.49173553719008267), 'mae': 836.6642114848751, 'rmse': 1185.5322130746463, 'window': 512, 'horizon': 4, 'q': 0.6, 'n_terms': 24, 'n_freq': 8, 'lambda': 1.0, 'phase_scale': 1.0}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py   --csv prices/BTCUSDT_1h.csv   --price-col close   --horizon 4 --window 512   --q 0.6 --n-terms 24 --n-freq 8   --lambda 1.3   --outdir theta_bot_averaging/results_biquat_btc
Saved predictions to: theta_bot_averaging/results_biquat_btc/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc/summary_biquat_ridge.csv
Metrics: {'n_samples': 484, 'corr_pred_true': -0.07587445999510027, 'hit_rate': np.float64(0.49173553719008267), 'mae': 836.6560461899597, 'rmse': 1185.5184143356953, 'window': 512, 'horizon': 4, 'q': 0.6, 'n_terms': 24, 'n_freq': 8, 'lambda': 1.3, 'phase_scale': 1.0}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py \
>   --csv prices/BTCUSDT_1h.csv --price-col close \
>   --horizon 1 --window 256 \
>   --q 0.65 --n-terms 18 --n-freq 6 \
>   --lambda 0.3 --phase-scale 0.8 \
>   --outdir theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08
Saved predictions to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/summary_biquat_ridge.csv
Metrics: {'n_samples': 743, 'corr_pred_true': 0.01127894239233773, 'hit_rate': np.float64(0.5087483176312247), 'mae': 337.9867863492727, 'rmse': 493.39728521063097, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 18, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 0.8}
>
> (venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py \
>   --csv prices/BTCUSDT_1h.csv --price-col close \
>   --horizon 1 --window 256 \
>   --q 0.65 --n-terms 18 --n-freq 6 \
>   --lambda 0.3 --phase-scale 0.8 \
>   --outdir theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08
Saved predictions to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/summary_biquat_ridge.csv
Metrics: {'n_samples': 743, 'corr_pred_true': 0.01127894239233773, 'hit_rate': np.float64(0.5087483176312247), 'mae': 337.9867863492727, 'rmse': 493.39728521063097, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 18, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 0.8}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py \
>   --csv prices/BTCUSDT_1h.csv --price-col close \
>   --horizon 1 --window 256 \
>   --q 0.65 --n-terms 18 --n-freq 6 \
>   --lambda 0.3 --phase-scale 0.8 \
>   --outdir theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08
Saved predictions to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc_win256_q065_nt18_nf6_l03_ps08/summary_biquat_ridge.csv
Metrics: {'n_samples': 743, 'corr_pred_true': 0.01127894239233773, 'hit_rate': np.float64(0.5087483176312247), 'mae': 337.9867863492727, 'rmse': 493.39728521063097, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 18, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 0.8}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py \
>   --csv prices/BTCUSDT_1h.csv --price-col close \
>   --horizon 1 --window 256 \
>   --q 0.75 --n-terms 16 --n-freq 4 \
>   --lambda 3.0 --phase-scale 1.2 \
>   --outdir theta_bot_averaging/results_biquat_btc_win256_q075_nt16_nf4_l3_ps12
Saved predictions to: theta_bot_averaging/results_biquat_btc_win256_q075_nt16_nf4_l3_ps12/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc_win256_q075_nt16_nf4_l3_ps12/summary_biquat_ridge.csv
Metrics: {'n_samples': 743, 'corr_pred_true': 0.009198020880114724, 'hit_rate': np.float64(0.48048452220726784), 'mae': 340.81404728780103, 'rmse': 494.0786514621186, 'window': 256, 'horizon': 1, 'q': 0.75, 'n_terms': 16, 'n_freq': 4, 'lambda': 3.0, 'phase_scale': 1.2}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_biquaternion_ridge.py \
>   --csv prices/BTCUSDT_1h.csv --price-col close \
>   --horizon 1 --window 320 \
>   --q 0.5 --n-terms 20 --n-freq 6 \
>   --lambda 1.0 --phase-scale 1.0 \
>   --outdir theta_bot_averaging/results_biquat_btc_win320_q05_nt20_nf6_l1
Saved predictions to: theta_bot_averaging/results_biquat_btc_win320_q05_nt20_nf6_l1/predictions.csv
Saved summary to: theta_bot_averaging/results_biquat_btc_win320_q05_nt20_nf6_l1/summary_biquat_ridge.csv
Metrics: {'n_samples': 679, 'corr_pred_true': -0.025925546374920172, 'hit_rate': np.float64(0.5036818851251841), 'mae': 349.33920116782434, 'rmse': 502.66313976732465, 'window': 320, 'horizon': 1, 'q': 0.5, 'n_terms': 20, 'n_freq': 6, 'lambda': 1.0, 'phase_scale': 1.0}
>
########################################################################################################################
QUATERNIONIC SCRIPTS:

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_quaternion_ridge.py   --csv prices/BTCUSDT_1h.csv --price-col close   --horizon 1 --window 256   --q 0.65 --n-terms 18 --n-freq 6   --lambda 0.3 --phase-scale 0.8   --outdir theta_bot_averaging/results_quat_btc
[summary] {'n_samples': 743, 'corr_pred_true': -0.04447953911936541, 'hit_rate': 0.4952893674293405, 'mae': 357.66279400856183, 'rmse': 513.4907727018872, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 18, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 0.8}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_quaternion_ridge.py   --csv prices/BTCUSDT_1h.csv --price-col close   --horizon 8 --window 512   --q 0.65 --n-terms 18 --n-freq 6   --lambda 0.3 --phase-scale 0.8   --outdir theta_bot_averaging/results_quat_btc
[summary] {'n_samples': 480, 'corr_pred_true': -0.09633669843174208, 'hit_rate': 0.49375, 'mae': 1234.2870847735403, 'rmse': 1774.0427009418556, 'window': 512, 'horizon': 8, 'q': 0.65, 'n_terms': 18, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 0.8}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ 

(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_quaternion_ridge.py   --csv prices/BTCUSDT_1h.csv --price-col close   --horizon 1 --window 256   --q 0.65 --n-terms 16 --n-freq 6   --lambda 0.3 --phase-scale 1.0   --outdir theta_bot_averaging/results_quat_btc_fourier
[summary] {'n_samples': 743, 'corr_pred_true': -0.0663654192584863, 'hit_rate': 0.506056527590848, 'mae': 357.97942612861095, 'rmse': 516.1536547312347, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 16, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 1.0}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ 


(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ python theta_bot_averaging/theta_eval_quaternion_ridge_v2.py   --csv prices/BTCUSDT_1h.csv --price-col close   --horizon 1 --window 256   --q 0.65 --n-terms 16 --n-freq 6   --lambda 0.3 --phase-scale 1.0   --ema-alpha 0.02 --block-norm   --outdir theta_bot_averaging/results_quat_btc_v2
[summary_v2] {'n_samples': 743, 'corr_pred_true': 0.018874909752857674, 'hit_rate': 0.506056527590848, 'anti_hit_rate': 0.4939434724091521, 'zero_rate': 0.0, 'corr_price': 0.7767795234650198, 'mae': 384.11939630668377, 'rmse': 548.6343109413705, 'window': 256, 'horizon': 1, 'q': 0.65, 'n_terms': 16, 'n_freq': 6, 'lambda': 0.3, 'phase_scale': 1.0, 'ema_alpha': 0.02, 'block_norm': True}
(venv) davidjaros@Davids-Mac-mini:~/workspace/theta-bot (make-theta-biquat)$ 
