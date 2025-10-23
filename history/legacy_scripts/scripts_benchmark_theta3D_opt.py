#!/usr/bin/env python
import os, argparse, csv, numpy as np
from tests_backtest.common.eval import load_csv_rows
from tests_backtest.cli.run_theta_benchmark import walk_forward_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='e2e_{symbol}_{interval}.csv path')
    ap.add_argument('--outdir', default='reports_theta3D_scan')
    ap.add_argument('--beta-re-grid', default="-0.02,0.0,0.02,0.05")
    ap.add_argument('--beta-im-grid', default="0.0,0.01,0.02")
    ap.add_argument('--upper', type=float, default=0.508)
    ap.add_argument('--lower', type=float, default=0.492)
    ap.add_argument('--theta-K', type=int, default=48)
    ap.add_argument('--theta-tau-re', type=float, default=0.03)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--step', type=int, default=256)
    ap.add_argument('--fee-side', type=float, default=0.00036)
    ap.add_argument('--slip-side', type=float, default=0.0001)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_csv_rows(args.csv)

    bre_list = [float(x) for x in args.beta_re_grid.split(',')]
    bim_list = [float(x) for x in args.beta_im_grid.split(',')]

    out_csv = os.path.join(args.outdir, "grid_theta3D_beta.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['beta_re','beta_im','runs','sharpe','maxdd','cagr','total_return','trades','fees'])
        best = None
        for bre in bre_list:
            for bim in bim_list:
                res = walk_forward_eval(
                    rows, 'theta3D', 'ckalman',
                    window=args.window, train_frac=args.train_frac, step=args.step,
                    upper=args.upper, lower=args.lower,
                    fee_side=args.fee_side, slip_side=args.slip_side,
                    theta_K=args.theta_K, theta_tau_re=args.theta_tau_re,
                    theta_beta_re=bre, theta_beta_im=bim
                )
                w.writerow([bre,bim,res.get('runs',0),res.get('sharpe',0.0),res.get('maxdd',0.0),
                            res.get('cagr',0.0),res.get('total_return',0.0),res.get('trades',0),res.get('fees',0.0)])
                if best is None or res.get('sharpe',-1e9)>best.get('sharpe',-1e9):
                    best = {'beta_re':bre,'beta_im':bim, **res}
    print("[OK] Saved", out_csv)

if __name__ == "__main__":
    main()
