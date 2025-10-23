
import numpy as np

def _ema_weights(n, alpha):
    if alpha <= 0 or alpha >= 1:
        return np.ones(n, dtype=float)
    w = np.power(1.0 - alpha, np.arange(n-1, -1, -1, dtype=float))
    return w / w.sum()

def build_theta_components(t_idx, z, sigma=0.8, N_even=6, N_odd=6,
                           include_theta0=True, include_theta1=True,
                           include_theta2=True, include_theta3=True):
    """
    Construct *theoretical* Jacobi-theta components (finite q-series truncation)
    on a time grid.

    Parameters
    ----------
    t_idx : ndarray shape (W,)
        Windowed time indices (monotone increasing integers).
    z : ndarray shape (W,)
        Phase argument z(t).
    sigma : float
        Imaginary part in tau = i*sigma. Controls q = exp(-pi*sigma).
    N_even, N_odd : int
        Truncation levels for even (2n) and odd (2n+1) harmonics.
    include_theta0..theta3 : bool
        Select which Jacobi families to include.

    Returns
    -------
    B : ndarray shape (W, D)
        Column-stacked components (unnormalized).
    names : list[str]
        Names of each column for debugging.
    col_weights : ndarray shape (D,)
        Natural q-weights for columns (for weighted orthogonalization / ridge).
    """
    W = len(t_idx)
    z = np.asarray(z, dtype=float)
    q = np.exp(-np.pi * float(sigma))
    cols = []
    names = []
    weights = []

    # theta3: 1 + 2 sum_{n>=1} q^{n^2} cos(2 n z)
    if include_theta3:
        # constant term (optional; keep to capture mean)
        const = np.ones(W, dtype=float)
        cols.append(const)
        names.append("theta3_const")
        weights.append(1.0)
        for n in range(1, N_even+1):
            wq = 2.0 * (q ** (n*n))
            c = np.cos(2.0 * n * z)
            cols.append(c)
            names.append(f"theta3_cos_2n_n{n}")
            weights.append(wq)

    # theta0 == theta4: 1 + 2 sum_{n>=1} (-1)^n q^{n^2} cos(2 n z)
    if include_theta0:
        const0 = np.ones(W, dtype=float)
        cols.append(const0)
        names.append("theta0_const")
        weights.append(1.0)
        for n in range(1, N_even+1):
            wq = 2.0 * (q ** (n*n))
            c = ((-1.0)**n) * np.cos(2.0 * n * z)
            cols.append(c)
            names.append(f"theta0_cos_2n_n{n}")
            weights.append(wq)

    # theta2: 2 sum_{m>=0} q^{(m+1/2)^2} cos((2m+1) z)
    if include_theta2:
        for m in range(0, N_odd+1):
            power = (m + 0.5)**2
            wq = 2.0 * (q ** power)
            c = np.cos((2*m + 1.0) * z)
            cols.append(c)
            names.append(f"theta2_cos_odd_m{m}")
            weights.append(wq)

    # theta1: 2 sum_{m>=0} (-1)^m q^{(m+1/2)^2} sin((2m+1) z)
    if include_theta1:
        for m in range(0, N_odd+1):
            power = (m + 0.5)**2
            wq = 2.0 * (q ** power)
            s = ((-1.0)**m) * np.sin((2*m + 1.0) * z)
            cols.append(s)
            names.append(f"theta1_sin_odd_m{m}")
            weights.append(wq)

    B = np.column_stack(cols) if cols else np.zeros((W,0), dtype=float)
    return B, names, np.asarray(weights, dtype=float)


def weighted_qr(B, row_weights=None, col_weights=None, ridge=0.0):
    """
    Weighted column-QR: apply row and column weights, QR, then map back.

    Inner product: <u,v> = u^T W_row v, columns scaled by sqrt(W_col).

    Returns
    -------
    Q : ndarray (W, D_eff)
        Orthonormal columns in the weighted sense.
    R : ndarray (D_eff, D_eff)
    col_scale : ndarray (D,)
        Applied column sqrt-weights (for debugging).
    """
    B = np.asarray(B, dtype=float)
    W, D = B.shape
    if D == 0:
        return np.zeros_like(B), np.zeros((0,0)), np.ones((0,), dtype=float)

    if row_weights is None:
        row_weights = np.ones(W, dtype=float)
    if col_weights is None:
        col_weights = np.ones(D, dtype=float)

    rw = np.sqrt(np.asarray(row_weights, dtype=float)).reshape(-1,1)  # (W,1)
    cw = np.sqrt(np.asarray(col_weights, dtype=float)).reshape(1,-1)  # (1,D)

    BW = (B * rw) * cw  # row & col scaling
    # economic QR
    Qw, Rw = np.linalg.qr(BW, mode='reduced')  # (W, D_eff), (D_eff, D_eff)
    # Map Qw back to unweighted Q: solve (Qw = (B*rw*cw)*R^{-1}) => Q = Qw / rw
    # But we want orthonormal w.r.t. weighted inner product: return Q = Qw / rw
    Q = Qw / rw
    R = Rw / cw.squeeze()  # adjust R for column scaling
    return Q, R, cw.squeeze()


def ridge_project(Q, x, row_weights=None, ridge=1e-3):
    """
    If Q has (approx) orthonormal columns under weighted inner product,
    we can project by normal equations in that metric:
        beta = (Q^T W Q + λI)^{-1} Q^T W x
    """
    Wn = Q.shape[0]
    if row_weights is None:
        Wrow = np.ones(Wn, dtype=float)
    else:
        Wrow = np.asarray(row_weights, dtype=float)
    # Form A = Q^T W Q, b = Q^T W x
    WQ = Q * Wrow.reshape(-1,1)
    A = Q.T @ WQ
    b = Q.T @ (Wrow * x)
    if ridge > 0:
        A = A + ridge * np.eye(A.shape[0])
    beta = np.linalg.solve(A, b)
    return beta


def forecast_theta_biquat_basis(close, H=4, window=256, sigma=0.8,
                                N_even=6, N_odd=6,
                                omega=None,  # if None, auto-estimate
                                ema_alpha=0.0, ridge=1e-3,
                                include_theta0=True, include_theta1=True,
                                include_theta2=True, include_theta3=True):
    """
    Rolling forecast using orthonormalized finite theta basis.
    Returns arrays of predictions aligned to close[window:len(close)-H].

    Parameters mirror the theory spec:
      - close: price series
      - H: forecast horizon (bars)
      - window: past window length
      - sigma, N_even, N_odd: theta basis settings
      - omega: base frequency; if None, estimate from autocorr peak
      - ema_alpha: row-weighting for recent samples (0 => uniform)
      - ridge: ridge regularization in projection
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    out_len = n - window - H + 1
    if out_len <= 0:
        return np.array([]), np.array([])

    # detrend: last value hold-out style
    x = close.copy()

    # estimate omega if needed (very crude: from argmax autocorr in [12..96])
    if omega is None:
        max_lag = min(96, window//2)
        lags = np.arange(12, max_lag+1, dtype=int)
        acs = []
        xc = x - np.mean(x[:window])
        for L in lags:
            acs.append(np.corrcoef(xc[:window-L], xc[L:window])[0,1])
        Lbest = lags[int(np.nanargmax(acs))] if len(lags)>0 else 24
        omega = 2.0*np.pi / max(4, int(Lbest))

    preds = np.zeros(out_len, dtype=float)
    trues = np.zeros(out_len, dtype=float)

    for i in range(out_len):
        s = i
        e = i + window
        f = e + H - 1
        win = x[s:e]
        target = x[f]

        t_idx = np.arange(window, dtype=float)
        z = omega * t_idx

        # build basis
        B, names, col_w = build_theta_components(
            t_idx, z, sigma=sigma, N_even=N_even, N_odd=N_odd,
            include_theta0=include_theta0, include_theta1=include_theta1,
            include_theta2=include_theta2, include_theta3=include_theta3
        )

        # row weights (EMA)
        rw = _ema_weights(window, ema_alpha)

        # weighted QR -> orthonormal columns
        Q, R, _ = weighted_qr(B, row_weights=rw, col_weights=col_w)

        if Q.shape[1] == 0:
            preds[i] = win[-1]  # fallback: naive
        else:
            beta = ridge_project(Q, win, row_weights=rw, ridge=ridge)
            # forecast at t0+H-1: reuse same basis but shifted in time
            t_idx_f = np.arange(window + H - 1, dtype=float)[-window:]
            zf = omega * t_idx_f
            Bf, _, _ = build_theta_components(
                t_idx_f, zf, sigma=sigma, N_even=N_even, N_odd=N_odd,
                include_theta0=include_theta0, include_theta1=include_theta1,
                include_theta2=include_theta2, include_theta3=include_theta3
            )
            # map to Q-space: project Bf onto Q columns via least squares
            # (since Q spans Col(BW), we can compute yhat = (Bf) @ gamma,
            #  and approximate gamma via solving Q*gamma ≈ Bf[:,j] for each j.
            #  Simpler: compute coefficients of Bf in Q basis: C = Q^T W Bf
            Wrow = rw
            C = (Q.T * Wrow) @ Bf
            # predicted value is sum_j beta_j * (Q-basis component at forecast time)
            # we need the forecast-time row of Q-basis. Construct q_fore from Bf:
            # the last row of Bf corresponds to forecast time t = e+H-1
            b_fore = Bf[-1, :]  # (D,)
            q_fore = C @ np.linalg.pinv((Q.T * Wrow) @ Q)  # map B->Q basis
            # q_fore is ill-defined; simpler: compute forecast via linear model in B-space.
            # Compute beta_B solving (B^T W B + λI) β = B^T W x, then y_fore = b_fore^T β.
            W = np.diag(Wrow)
            A = B.T @ W @ B + ridge * np.eye(B.shape[1])
            rhs = B.T @ W @ win
            beta_B = np.linalg.solve(A, rhs)
            preds[i] = float(b_fore @ beta_B)

        trues[i] = target

    return preds, trues, dict(omega=omega, sigma=sigma, N_even=N_even, N_odd=N_odd)
