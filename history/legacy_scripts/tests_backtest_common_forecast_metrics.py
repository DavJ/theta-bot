import numpy as np

def weighted_mse(y_true_seq, y_pred_seq, alpha=0.9):
    # Exponential weights toward near future: w[h] = alpha**h
    H = len(y_true_seq)
    if H == 0:
        return np.nan
    w = np.array([alpha**h for h in range(H)], dtype=float)
    err2 = (np.asarray(y_true_seq) - np.asarray(y_pred_seq))**2
    return float((w @ err2) / (w.sum() + 1e-12))

def direction_accuracy(y_true_seq, y_pred_seq):
    yt = np.asarray(y_true_seq); yp = np.asarray(y_pred_seq)
    dy_true = np.sign(np.diff(np.r_[yt[0], yt]))
    dy_pred = np.sign(np.diff(np.r_[yp[0], yp]))
    return float((dy_true == dy_pred).mean())

def mae(y_true_seq, y_pred_seq, alpha=1.0):
    H = len(y_true_seq)
    if H == 0: return np.nan
    w = np.array([alpha**h for h in range(H)], dtype=float)
    err = np.abs(np.asarray(y_true_seq) - np.asarray(y_pred_seq))
    return float((w @ err) / (w.sum() + 1e-12))

def mape(y_true_seq, y_pred_seq, alpha=1.0):
    H = len(y_true_seq)
    if H == 0: return np.nan
    w = np.array([alpha**h for h in range(H)], dtype=float)
    yt = np.asarray(y_true_seq)
    denom = np.maximum(1e-8, np.abs(yt))
    pe = np.abs((yt - np.asarray(y_pred_seq)) / denom)
    return float((w @ pe) / (w.sum() + 1e-12))
