PATCH: Theta FFT 1D/2D/3D (rychlé varianty) + auto-patch CLI

Soubory:
- tests_backtest/common/transforms_theta_fast.py
    - theta_fft_fast(series, K=..., tau_im=...)
    - theta_fft_hybrid(series, K=..., tau_re=..., tau_im=...)
    - theta_fft_dynamic(series, K=..., tau_re0=..., tau_im0=..., beta_re=..., beta_im=...)
- scripts/patch_cli_mac.sh  (automatické doplnění importů, argumentů a větví do run_theta_benchmark.py)

Použití:
1) Rozbal do kořene repa: unzip ~/Downloads/theta-fft-1D2D3D-v2.zip
2) Spusť auto patch (macOS):
   bash scripts/patch_cli_mac.sh

Pak můžeš hned spustit např. 2D hybrid:
python -m tests_backtest.cli.run_theta_benchmark \  --symbol BTCUSDT --interval 5m --limit 20000 \  --variants theta_fft_fast theta_fft_hybrid theta_fft_dynamic \  --models ckalman \  --theta-K 48 --theta-tau 0.12 --theta-tau-re 0.03 --theta-beta-re 0.02 --theta-beta-im 0.01 \  --upper-grid "0.504,0.508,0.512" \  --lower-grid "0.496,0.492,0.488" \  --fee-side 0.00036 \  --outdir reports_theta_fft_all
