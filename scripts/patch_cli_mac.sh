#!/usr/bin/env bash
set -euo pipefail

CLI="tests_backtest/cli/run_theta_benchmark.py"

if [[ ! -f "$CLI" ]]; then
  echo "Nenalezen $CLI"; exit 1
fi

# 1) Importy (pokud už nejsou)
if ! grep -q "transforms_theta_fast" "$CLI"; then
  echo "[patch] Přidávám import transforms_theta_fast"
  /usr/bin/sed -i '' '1,40 s|^from tests_backtest\.common\.transforms_theta.*$|&\
from tests_backtest.common.transforms_theta_fast import (theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic)|' "$CLI" || true
fi

# 2) Argumenty (přidej, pokud nejsou)
if ! grep -q -- "--theta-tau-re" "$CLI"; then
  echo "[patch] Přidávám argparse param --theta-tau-re"
  /usr/bin/sed -i '' 's|ap\.add_argument('--theta\-tau',[^)]*)|&\
ap.add_argument('--theta-tau-re', type=float, default=0.03)\
ap.add_argument('--theta-beta-re', type=float, default=0.0)\
ap.add_argument('--theta-beta-im', type=float, default=0.0)|' "$CLI" || true
fi

# 3) Přidej větve do build_dataset(...) – vložíme za větev fft_complex, pokud existuje
if grep -n "elif variant=='fft_complex'" "$CLI" >/dev/null; then
  echo "[patch] Vkládám větve pro theta_fft_* za fft_complex"
  /usr/bin/sed -i '' "s|elif variant=='fft_complex'.*|&\\
    elif variant=='theta_fft_fast':\\
        z, _ = theta_fft_fast(seg, K=theta_K, tau_im=theta_tau)\\
        feats_c.append(z); f = np.zeros(1)\\
    elif variant=='theta_fft_hybrid':\\
        z, _ = theta_fft_hybrid(seg, K=theta_K, tau_re=args.theta_tau_re, tau_im=theta_tau)\\
        feats_c.append(z); f = np.zeros(1)\\
    elif variant=='theta_fft_dynamic':\\
        z, _ = theta_fft_dynamic(seg, K=theta_K, tau_re0=args.theta_tau_re, tau_im0=theta_tau, beta_re=args.theta_beta_re, beta_im=args.theta_beta_im)\\
        feats_c.append(z); f = np.zeros(1)|" "$CLI" || true
else
  echo "[warn] Nenalezena větev fft_complex, pokusím se vložit do build_dataset"
  /usr/bin/sed -i '' "s|def build_dataset(.*):|&\\
    # vloženo patcherem: theta FFT varianty\\
    #   theta_fft_fast: 1D (tau_im)\\
    #   theta_fft_hybrid: 2D (tau_re + tau_im)\\
    #   theta_fft_dynamic: 3D (tau_re(t), tau_im(t))\\
    #   -> výstup jde do feats_c (komplexní) pro ckalman\\
|" "$CLI" || true
fi

echo "[patch] Hotovo. Zkontroluj git diff a spusť benchmark."
