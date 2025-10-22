# -*- coding: utf-8 -*-
"""
theta/biquat_phase.py
---------------------
Bikvaternionová (3‑osá) fáze pro Theta bázi + generování ortonormální báze
z nelineárních funkcí fáze. Modul je self‑contained (numpy).

API (stabilní):
    build_biquat_basis(prices, tvec, *, sigma=0.8, max_harm=3, ema_spans=(16,32,64),
                       ridge=1e-6, return_names=False)
        -> (X, names) pokud return_names=True, jinak jen X

- prices: 1D array (N,) – uzavírací cena v okně
- tvec:   1D array (N,) – čas v "barech" (např. 0..N-1) nebo v sekundách
- sigma:  škálování "rychlosti" fáze (větší = strmější fáze)
- max_harm: kolik harmonických (1..max_harm) z každé osy + křížové členy
- ema_spans: trojice spanů EMA pro latentní osy ψ_x, ψ_y, ψ_z
- ridge:  malé Tikhonovovo λ při QR re‑ortogonalizaci (stabilita)
- return_names: pokud True, vrátí i seznam názvů sloupců báze
"""
from __future__ import annotations
import numpy as np

def _ema(x: np.ndarray, span: float) -> np.ndarray:
    if span <= 1:
        return x.astype(float, copy=True)
    alpha = 2.0 / (span + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        sd = eps
    return (x - mu) / sd

def _grad_log_prices(prices: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(prices, 1e-12, None))
    ret = np.diff(lp, prepend=lp[0])
    return ret

def _phase_components(prices: np.ndarray, tvec: np.ndarray,
                      sigma: float, ema_spans=(16,32,64)):
    r = _grad_log_prices(prices)
    r1 = _ema(r, ema_spans[0])
    r2 = _ema(r, ema_spans[1])
    r3 = _ema(r, ema_spans[2])
    u1 = _zscore(r1)
    u2 = _zscore(r2)
    u3 = _zscore(r3)
    th1 = sigma * np.cumsum(u1)
    th2 = sigma * np.cumsum(u2)
    th3 = sigma * np.cumsum(u3)
    return th1, th2, th3

def _feature_block_from_phase(theta: np.ndarray, max_harm: int):
    feats = []
    for k in range(1, max_harm+1):
        feats.append(np.cos(k * theta))
        feats.append(np.sin(k * theta))
    return feats

def _cross_blocks(th1: np.ndarray, th2: np.ndarray, max_harm: int):
    feats = []
    for k in range(1, max_harm+1):
        feats.append(np.cos(k * (th1 + th2)))
        feats.append(np.sin(k * (th1 + th2)))
        feats.append(np.cos(k * (th1 - th2)))
        feats.append(np.sin(k * (th1 - th2)))
    return feats

def _stable_orthonormalize(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    col_norm = np.linalg.norm(Xc, axis=0)
    keep = col_norm > 1e-10
    Xc = Xc[:, keep]
    if Xc.size == 0:
        return np.zeros((X.shape[0], 0))
    # Gram-Schmidt QR
    Q = np.zeros_like(Xc)
    for j in range(Xc.shape[1]):
        v = Xc[:, j].copy()
        for k in range(j):
            r = np.dot(Q[:, k], v)
            v = v - r * Q[:, k]
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            v = np.zeros_like(v)
            norm = 1.0
        Q[:, j] = v / norm
    # ridge-like rescale
    return Q / np.sqrt(1.0 + ridge)

def build_biquat_basis(prices: np.ndarray,
                       tvec: np.ndarray,
                       *, sigma: float = 0.8,
                       max_harm: int = 3,
                       ema_spans=(16,32,64),
                       ridge: float = 1e-6,
                       return_names: bool = False):
    prices = np.asarray(prices, dtype=float)
    tvec = np.asarray(tvec, dtype=float)
    N = prices.shape[0]
    if N != tvec.shape[0]:
        raise ValueError("prices and tvec must have the same length")

    th1, th2, th3 = _phase_components(prices, tvec, sigma=sigma, ema_spans=ema_spans)

    blocks = []
    names = []

    for i, th in enumerate([th1, th2, th3], start=1):
        fb = _feature_block_from_phase(th, max_harm)
        blocks.extend(fb)
        for k in range(1, max_harm+1):
            names += [f"cos{k}_th{i}", f"sin{k}_th{i}"]

    pairs = [(th1, th2), (th1, th3), (th2, th3)]
    pair_names = [("th1","th2"), ("th1","th3"), ("th2","th3")]
    for (tha, thb), (na, nb) in zip(pairs, pair_names):
        cb = _cross_blocks(tha, thb, max_harm)
        blocks.extend(cb)
        for k in range(1, max_harm+1):
            names += [f"cos{k}_{na}+{nb}", f"sin{k}_{na}+{nb}", f"cos{k}_{na}-{nb}", f"sin{k}_{na}-{nb}"]

    Xraw = np.vstack(blocks).T
    X = _stable_orthonormalize(Xraw, ridge=ridge)

    if return_names:
        M = X.shape[1]
        names_out = names[:M] if len(names) >= M else [f"feat_{i}" for i in range(M)]
        return X, names_out
    return X
