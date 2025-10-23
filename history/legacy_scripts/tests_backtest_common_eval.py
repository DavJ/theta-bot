import csv, numpy as np, math
def load_csv_rows(path):
    rows=[]; 
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({'t':int(row['timestamp']),'o':float(row['open']),'h':float(row['high']),
                         'l':float(row['low']),'c':float(row['close']),'v':float(row['volume'])})
    return rows
def atr(high,low,close,period=14):
    if len(close)<period+1: return np.array([])
    trs=[]; 
    for i in range(1,len(close)):
        trs.append(max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    trs=np.array(trs,dtype=float)
    atr_vals=np.zeros_like(trs); atr_vals[period-1]=trs[:period].mean(); alpha=1.0/period
    for i in range(period,len(trs)): atr_vals[i]=(atr_vals[i-1]*(1-alpha)) + alpha*trs[i]
    out=np.concatenate([np.full(1,np.nan), atr_vals]); return out
def hyst(prob, upper, lower, prev):
    if prob>=upper: return 1
    if prob<=lower: return -1
    return prev
def simulate_trading(probs, closes, highs, lows, times,
                     upper=0.55, lower=0.45, atr_period=14, atr_thresh=0.0,
                     fee_side=0.00056, slip_side=0.0001, weight=1.0):
    equity=1000.0; pos=0; qty=0.0; trades=0; fees_total=0.0
    atr_arr=atr(highs,lows,closes,period=atr_period); eq_series=[]
    for i in range(len(probs)):
        price=closes[i]; allowed=True
        if not np.isnan(atr_arr[i] if i<len(atr_arr) else np.nan):
            if atr_thresh>0.0 and (atr_arr[i]/max(price,1e-12))<atr_thresh: allowed=False
        new_pos=hyst(probs[i],upper,lower,pos) if allowed else pos
        if new_pos!=pos:
            if pos!=0:
                side=-pos; fill=price*(1-slip_side) if side<0 else price*(1+slip_side)
                delta=-qty*fill; fee=abs(delta)*(2*fee_side); equity+=delta - fee; fees_total+=fee; qty=0.0; trades+=1
            if new_pos!=0 and allowed:
                target=equity*weight*new_pos; fill=price*(1+slip_side) if new_pos>0 else price*(1-slip_side)
                qty=target/fill; fee=abs(target)*(2*fee_side); equity-=fee; fees_total+=fee; trades+=1
            pos=new_pos
        eq = equity + qty*(price - closes[i]); eq_series.append((times[i], eq))
    if pos!=0:
        price=closes[-1]; side=-pos; fill=price*(1-slip_side) if side<0 else price*(1+slip_side)
        delta=-qty*fill; fee=abs(delta)*(2*fee_side); equity+=delta - fee; fees_total+=fee; qty=0.0; trades+=1
    if not eq_series: return {'equity': [], 'trades': trades, 'fees': fees_total, 'sharpe':0.0, 'maxdd':0.0, 'cagr':0.0, 'total_return':0.0}
    eq_vals=np.array([v for _,v in eq_series]); rets=np.diff(eq_vals)/eq_vals[:-1]
    sharpe=(np.mean(rets)/(np.std(rets)+1e-12))*math.sqrt(365*24*12)
    peak=eq_vals[0]; dd=0.0
    for v in eq_vals: peak=max(peak,v); dd=max(dd,(peak-v)/peak if peak>0 else 0.0)
    total=eq_vals[-1]/eq_vals[0] - 1.0; months=max(1.0, len(eq_series)/(12*24*6)); cagr=(1+total)**(12.0/months) - 1.0
    return {'equity': eq_series, 'trades': trades, 'fees': fees_total, 'sharpe':sharpe, 'maxdd':dd, 'cagr':cagr, 'total_return':total}
