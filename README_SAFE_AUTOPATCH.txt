Použití autopatcheru (bezpečný, idempotentní):
1) Rozbal ZIP v kořeni repozitáře (kde je složka tests_backtest/cli).
2) Spusť:   python scripts/patch_run_theta_benchmark_safe.py
3) Ověř:    grep -n "theta_fft_hybrid" tests_backtest/cli/run_theta_benchmark.py

Patcher vytvoří zálohu: tests_backtest/cli/run_theta_benchmark.py.bak
