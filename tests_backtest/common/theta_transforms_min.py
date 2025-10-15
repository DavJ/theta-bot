import numpy as np

def _ensure1d(x):
    return np.asarray(x, dtype=float).ravel()

def _ema(x, span):
    alpha = 2/(span+1.0)
    y = np.empty_like(x)
    s = x[0]
    for i,xi in enumerate(x):
        s = alpha*xi + (1-alpha)*s
        y[i] = s
    return y

def raw_features(seg):
    seg = _ensure1d(seg)
    x = seg
    ret = np.diff(x)/x[:-1]
    if len(ret) < 1:
        return np.array([0,0,0,0], dtype=float)
    ema_fast = _ema(x, span=max(2, len(x)//16))[-1]
    ema_slow = _ema(x, span=max(4, len(x)//4))[-1]
    return np.array([x[-1], ema_fast, ema_slow, ret[-1]], dtype=float)

def fft_reim(seg, topn=16):
    x = _ensure1d(seg)
    z = np.fft.rfft(x)
    idx = np.argsort(-np.abs(z))[1:topn+1]
    z_top = z[idx]
    return np.r_[z_top.real, z_top.imag]

def theta1D(seg, K=32, tau=0.12):
    x = _ensure1d(seg)
    n = len(x)
    t = np.linspace(-0.5, 0.5, n) * n
    g = np.exp(-0.5*(t/(tau*n + 1e-12))**2)
    xw = x * g
    Z = np.fft.rfft(xw)
    idx = np.argsort(-np.abs(Z))[1:K+1]
    Zk = Z[idx]
    return np.r_[Zk.real, Zk.imag]

def theta2D(seg, K=32, tau=0.12, tau_re=0.02):
    x = _ensure1d(seg)
    n = len(x)
    t = np.linspace(-0.5, 0.5, n) * n
    g = np.exp(-0.5*(t/(tau*n + 1e-12))**2)
    chirp = np.exp(1j * np.pi * tau_re * (t/n)**2)
    xw = x * g * chirp
    Z = np.fft.rfft(xw)
    idx = np.argsort(-np.abs(Z))[1:K+1]
    Zk = Z[idx]
    return np.r_[Zk.real, Zk.imag]

def theta3D(seg, K=32, tau=0.12, tau_re=0.02, psi=0.0):
    x = _ensure1d(seg)
    n = len(x)
    t = np.linspace(-0.5, 0.5, n) * n
    g = np.exp(-0.5*(t/(tau*n + 1e-12))**2)
    chirp = np.exp(1j * np.pi * tau_re * (t/n)**2)
    drift = np.exp(1j * 2*np.pi * psi * (t/n))
    xw = x * g * chirp * drift
    Z = np.fft.rfft(xw)
    idx = np.argsort(-np.abs(Z))[1:K+1]
    Zk = Z[idx]
    return np.r_[Zk.real, Zk.imag]
