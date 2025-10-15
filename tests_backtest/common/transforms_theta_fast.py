import numpy as np

__all__ = ["theta_fft_fast", "theta_fft_hybrid", "theta_fft_dynamic"]

def _as_f64(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return np.ascontiguousarray(x / std)

def _gauss_window(n, tau_re):
    # tau_re > 0; interpretujeme jako "časovou" šířku -> sigma = tau_re * N
    N = n.size
    sigma = max(1e-8, float(tau_re)) * N
    return np.exp(-(n - (N-1)/2.0)**2 / (2.0 * sigma * sigma))

def theta_fft_fast(x, K=32):
    """
    1D: čisté rFFT na reálném signálu (bez okna a fáze).
    Vrací prvních B koeficientů (B = min(K, len(rfft(x)))).
    """
    x = _as_f64(x)
    Z = np.fft.rfft(x)
    B = int(min(K, Z.shape[-1]))
    return Z[:B], {"mode": "rfft", "len": x.size, "B": B}

def theta_fft_hybrid(x, K=32, tau_re=0.03):
    """
    2D: reálné gaussovské okno před rFFT (žádná fáze).
    """
    x = _as_f64(x)
    N = x.size
    n = np.arange(N, dtype=np.float64)
    w = _gauss_window(n, float(tau_re))
    xw = x * w  # stále reálné
    Z = np.fft.rfft(xw)
    B = int(min(K, Z.shape[-1]))
    return Z[:B], {"mode": "rfft", "len": N, "B": B, "tau_re": float(tau_re)}

def theta_fft_dynamic(x, K=32, tau_re=0.03, phi_scale=0.002):
    """
    3D: stejné reálné gauss okno + navíc fázová složka (imaginary time).
    - fázi modelujeme jako kvadratickou v čase: phi(n) = phi_scale * (n - center)^2
    - komplexní okno => plná FFT
    """
    x = _as_f64(x)
    N = x.size
    n = np.arange(N, dtype=np.float64)
    w_re = _gauss_window(n, float(tau_re))
    center = (N - 1) / 2.0
    phase = float(phi_scale) * (n - center) ** 2
    w = w_re * np.exp(1j * phase)  # komplexní okno
    xw = x * w
    Z = np.fft.fft(xw)
    B = int(min(K, Z.shape[-1]))
    return Z[:B], {"mode": "fft", "len": N, "B": B, "tau_re": float(tau_re), "phi_scale": float(phi_scale)}
