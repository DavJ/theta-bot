#!/usr/bin/env bash
set -euo pipefail

CLI="tests_backtest/cli/run_theta_benchmark.py"

if [[ ! -f "$CLI" ]]; then
  echo "Nenalezen $CLI (spouštěj z kořene repozitáře)"
  exit 1
fi

echo "[1/4] Přidávám import transforms_theta_fast (pokud chybí)..."
if ! grep -q "transforms_theta_fast" "$CLI"; then
  # vlož import pod ostatní imports z transforms_* (do horních ~60 řádků)
  /usr/bin/sed -i '' '1,80 s|^from tests_backtest\.common\.transforms_.*$|&\
from tests_backtest.common.transforms_theta_fast import (theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic)|' "$CLI" || true
  # pokud se nevložilo (jiná struktura), zkusíme přidat hned po prvním importu numpy
  if ! grep -q "transforms_theta_fast" "$CLI"; then
    /usr/bin/sed -i '' '1,120 s|^import numpy as np$|import numpy as np\
from tests_backtest.common.transforms_theta_fast import (theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic)|' "$CLI" || true
  fi
fi

echo "[2/4] Přidávám argparse parametry (pokud chybí)..."
# Přidej tři nové argumenty pod --theta-tau
if ! grep -q -- "--theta-tau-re" "$CLI"; then
  /usr/bin/sed -i '' "s|ap\.add_argument('--theta-tau',[^)]*)|&\
ap.add_argument('--theta-tau-re', type=float, default=0.03)\
ap.add_argument('--theta-beta-re', type=float, default=0.0)\
ap.add_argument('--theta-beta-im', type=float, default=0.0)|" "$CLI" || true
fi

echo "[3/4] Vkládám větve pro nové varianty do build_dataset(...) (pokud chybí)..."
# Zkus vložit za větev fft_complex (nejčastější místo)
if ! grep -q "elif variant=='theta_fft_hybrid'" "$CLI"; then
  if grep -n "elif variant=='fft_complex'" "$CLI" >/dev/null; then
    /usr/bin/sed -i '' "s|elif variant=='fft_complex'.*|&\
    elif variant=='theta_fft_fast':\
        z, _ = theta_fft_fast(seg, K=theta_K, tau_im=theta_tau)\
        feats_c.append(z); f = np.zeros(1)\
    elif variant=='theta_fft_hybrid':\
        z, _ = theta_fft_hybrid(seg, K=theta_K, tau_re=args.theta_tau_re, tau_im=theta_tau)\
        feats_c.append(z); f = np.zeros(1)\
    elif variant=='theta_fft_dynamic':\
        z, _ = theta_fft_dynamic(seg, K=theta_K, tau_re0=args.theta_tau_re, tau_im0=theta_tau, beta_re=args.theta_beta_re, beta_im=args.theta_beta_im)\
        feats_c.append(z); f = np.zeros(1)|" "$CLI" || true
  else
    # fallback: vlož těsně před "return" v build_dataset
    /usr/bin/sed -i '' "0,/def build_dataset/s|return X, Xc, y, metas|    # vloženo patcherem: theta FFT varianty\
    #   theta_fft_fast (1D), theta_fft_hybrid (2D), theta_fft_dynamic (3D)\
    #   výstup jde do feats_c (komplexní) pro ckalman\
    #   pokud chceš, přidej i do --variants defaultu\
    \n    # (pozn.: sem je potřeba doplnit vložení do if-elif bloku dle struktury verze)\
    \n    return X, Xc, y, metas|" "$CLI" || true
  fi
fi

echo "[4/4] Hotovo. Zkontroluj diff:"
git --no-pager diff -- "$CLI" || true
echo "Pokud nevidíš nové větve/argumenty/importy, ozvi se – pošlu plný soubor jako náhradu."
