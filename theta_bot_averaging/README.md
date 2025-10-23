# theta_bot_averaging (stable baseline)

Zmražená baseline s evaluátorem `theta_eval_hbatch_biquat_max.py` (kauzální regrese Basis→Δ) a OOS skriptem `robustness_suite_v3_oos.py`.
Spouštěcí skripty jsou v `scripts/`. Parametry baseline v `configs/params_stable.json`.

## Quick start
```bash
cd theta_bot_averaging
bash scripts/bomba2.sh
```
Výstupy najdeš v `results/`.
