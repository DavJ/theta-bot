import os, csv, json, argparse, numpy as np
from tests_backtest.data.binance_download import download_to_csv
from tests_backtest.common.transforms import raw_features, fft_features, wavelet_features, theta_features
from tests_backtest.common.models import train_logreg, predict_logreg, Kalman1D, torch_lstm_available, fit_lstm_and_predict
from tests_backtest.common.eval import load_csv_rows, simulate_trading

def build_dataset(rows, window, variant):
    closes=[float(r['c']) for r in rows]
    feats=[]; labels=[]; times=[]; highs=[]; lows=[]; closes_meta=[]
    for i in range(window, len(rows)-1):
        seg=closes[i-window:i]
        if variant=='raw': f=raw_features(seg)
        elif variant=='fft': f=fft_features(seg, top_n=16)
        elif variant=='wavelet': f=wavelet_features(seg, levels=3)
        elif variant=='theta': f=theta_features(seg, K=16, tau_im=0.25, gram=False)
        elif variant=='theta_gs': f=theta_features(seg, K=16, tau_im=0.25, gram=True)
        else: raise ValueError('unknown variant')
        feats.append(f); labels.append(1 if closes[i+1]>closes[i] else 0)
        times.append(int(rows[i]['t'])); highs.append(float(rows[i]['h'])); lows.append(float(rows[i]['l'])); closes_meta.append(float(rows[i]['c']))
    X=np.asarray(feats); y=np.asarray(labels,dtype=int); meta={'times':np.array(times),'highs':np.array(highs),'lows':np.array(lows),'closes':np.array(closes_meta)}
    return X,y,meta

def walk_forward_eval(rows, variant, model, window=256, train_frac=0.7, step=256,
                      upper=0.55, lower=0.45, atr_period=14, atr_thresh=0.0,
                      fee_side=0.00056, slip_side=0.0001, weight=1.0):
    X_all,y_all,meta=build_dataset(rows,window,variant)
    times,highs,lows,closes=meta['times'],meta['highs'],meta['lows'],meta['closes']
    res=[]; start=0
    while start + step*2 < len(X_all):
        end=min(len(X_all), start+step*2)
        Xb=X_all[start:end]; yb=y_all[start:end]
        ntr=int(len(Xb)*train_frac)
        if ntr<30 or (len(Xb)-ntr)<30: break
        if model=='logreg':
            w,b=train_logreg(Xb[:ntr], yb[:ntr]); probs=predict_logreg(Xb[ntr:], w, b)
        elif model=='kalman':
            seg_cl=closes[start:start+len(Xb)]
            rets=np.diff(np.log(seg_cl[:ntr+1])); probs=np.ones(len(Xb)-ntr)*0.5
            if len(rets)>3:
                kf=Kalman1D(q=1e-5, r=1e-3); p_full=kf.fit_predict_proba(rets); probs=np.ones(len(Xb)-ntr)*p_full[-1]
        elif model=='lstm':
            probs=fit_lstm_and_predict(Xb[:ntr], yb[:ntr], Xb[ntr:], epochs=8)
        else: raise ValueError('unknown model')
        offset=start+ntr
        clos=closes[offset:offset+len(probs)]; highs_b=highs[offset:offset+len(probs)]; lows_b=lows[offset:offset+len(probs)]; times_b=times[offset:offset+len(probs)]
        sim=simulate_trading(probs, clos, highs_b, lows_b, times_b, upper, lower, atr_period, atr_thresh, fee_side, slip_side, weight)
        res.append(sim); start += step
    return res

def aggregate(results):
    if not results: return {}
    import numpy as np
    s=np.array([r['sharpe'] for r in results]); dd=np.array([r['maxdd'] for r in results]); c=np.array([r['cagr'] for r in results]); tr=np.array([r['total_return'] for r in results]); n=np.array([r['trades'] for r in results]); f=np.array([r['fees'] for r in results])
    return {'runs':len(results),'sharpe_mean':float(s.mean()),'maxdd_mean':float(dd.mean()),'cagr_mean':float(c.mean()),'total_return_mean':float(tr.mean()),'trades_sum':int(n.sum()),'fees_sum':float(f.sum())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--interval', default='5m')
    ap.add_argument('--limit', type=int, default=10000)
    ap.add_argument('--start', type=int, default=None); ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--variants', nargs='+', default=['raw','fft','wavelet','theta','theta_gs'])
    ap.add_argument('--models', nargs='+', default=['logreg','kalman','lstm'])
    ap.add_argument('--window', type=int, default=256); ap.add_argument('--train-frac', type=float, default=0.7); ap.add_argument('--step', type=int, default=256)
    ap.add_argument('--upper', type=float, default=0.55); ap.add_argument('--lower', type=float, default=0.45)
    ap.add_argument('--atr-period', type=int, default=14); ap.add_argument('--atr-thresh', type=float, default=0.0015)
    ap.add_argument('--fee-side', type=float, default=0.00056); ap.add_argument('--slip-side', type=float, default=0.0001); ap.add_argument('--weight', type=float, default=1.0)
    ap.add_argument('--outdir', default='reports_theta_benchmark')
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path=os.path.join(args.outdir, f"e2e_{args.symbol}_{args.interval}.csv")
    print("[download] ->", csv_path)
    download_to_csv(args.symbol, args.interval, csv_path, limit=args.limit, start_ms=args.start, end_ms=args.end)
    rows=load_csv_rows(csv_path)

    if 'lstm' in args.models:
        if not torch_lstm_available():
            print("[warn] torch není k dispozici, LSTM se přeskočí"); args.models=[m for m in args.models if m!='lstm']

    summary=[]
    for variant in args.variants:
        for model in args.models:
            print(f"[run] {variant} + {model}")
            res=walk_forward_eval(rows, variant, model, window=args.window, train_frac=args.train_frac, step=args.step,
                                  upper=args.upper, lower=args.lower, atr_period=args.atr_period, atr_thresh=args.atr_thresh,
                                  fee_side=args.fee_side, slip_side=args.slip_side, weight=args.weight)
            agg=aggregate(res); summary.append({'variant':variant,'model':model, **agg})
            eq=[]; 
            for r in res: eq.extend(r['equity'])
            with open(os.path.join(args.outdir, f"equity_{variant}_{model}.csv"), 'w', newline='') as f:
                w=csv.writer(f); w.writerow(['timestamp','equity'])
                for t,v in eq: w.writerow([t,v])

    with open(os.path.join(args.outdir, "summary.csv"), 'w', newline='') as f:
        w=csv.writer(f); w.writerow(['variant','model','runs','sharpe_mean','maxdd_mean','cagr_mean','total_return_mean','trades_sum','fees_sum'])
        for r in summary: w.writerow([r['variant'], r['model'], r.get('runs',0), r.get('sharpe_mean',0), r.get('maxdd_mean',0), r.get('cagr_mean',0), r.get('total_return_mean',0), r.get('trades_sum',0), r.get('fees_sum',0)])

    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write(f"# Theta Benchmark Summary ({args.symbol} {args.interval})\n\n")
        f.write("| variant | model | runs | sharpe | maxdd | cagr | total_return | trades | fees |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            f.write(f"| {r['variant']} | {r['model']} | {r.get('runs',0)} | {r.get('sharpe_mean',0):.2f} | {r.get('maxdd_mean',0):.2f} | {r.get('cagr_mean',0):.2f} | {r.get('total_return_mean',0):.2f} | {r.get('trades_sum',0)} | {r.get('fees_sum',0):.2f} |\n")
    print("Hotovo ->", args.outdir)

if __name__=='__main__':
    main()
