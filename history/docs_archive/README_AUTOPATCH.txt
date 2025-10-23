Použití autopatcheru (macOS):
1) Rozbal ZIP v kořeni repozitáře.
2) Spusť:   bash scripts/patch_run_theta_benchmark_v3.sh
3) Ověř:    grep -n "theta_fft_hybrid" tests_backtest/cli/run_theta_benchmark.py

Pokud autopatcher nenajde očekávané kotvy (jiné rozložení kódu), napiš – pošlu plný soubor.
