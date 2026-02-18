"""
theta_fit_tau.py
================
Scientific envelope-fitting utilities for forensic fingerprint analysis.

Provides:
  - gauss_envelope      : Gaussian envelope model
  - theta3_envelope     : Jacobi theta-3 envelope model
  - fit_gauss_envelope  : least-squares fit of a Gaussian envelope
  - fit_theta3_envelope : least-squares fit of a theta-3 envelope
  - plot_fit_result     : diagnostic plot (matplotlib imported lazily here)

matplotlib is intentionally NOT imported at module level so that importing
any of the fitting helpers does NOT trigger the pyparsing DeprecationWarning
that is promoted to an error by pytest/CI.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Envelope models
# ---------------------------------------------------------------------------

def gauss_envelope(
    t: np.ndarray,
    A: float,
    mu: float,
    sigma: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Gaussian envelope: A * exp(-0.5 * ((t - mu) / sigma)**2) + offset."""
    return A * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + offset


def theta3_envelope(
    t: np.ndarray,
    A: float,
    mu: float,
    T: float,
    q: float,
    offset: float = 0.0,
    n_terms: int = 20,
) -> np.ndarray:
    """Jacobi theta-3 envelope.

    theta_3(z, q) = 1 + 2 * sum_{n=1}^{N} q^{n^2} * cos(2 * n * z)

    The envelope is evaluated as::

        A * theta_3(pi * (t - mu) / T, q) + offset

    Parameters
    ----------
    t       : sample positions
    A       : amplitude scale
    mu      : centre (shift)
    T       : period
    q       : nome  (0 < q < 1)
    offset  : vertical offset
    n_terms : number of summation terms (default 20)
    """
    q = np.clip(q, 1e-9, 1.0 - 1e-9)
    z = np.pi * (t - mu) / T
    result = np.ones_like(t, dtype=float)
    q_power = float(q)  # q^1 at n=1; incremented as q^(n^2) = q^((n-1)^2) * q^(2n-1)
    for n in range(1, n_terms + 1):
        result = result + 2.0 * q_power * np.cos(2.0 * n * z)
        q_power *= q ** (2 * n + 1)  # advance: q^((n+1)^2) = q^(n^2) * q^(2n+1)
    return A * result + offset


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _estimate_gauss_p0(t: np.ndarray, data: np.ndarray):
    """Rough initial-parameter estimate for Gaussian fit."""
    idx_peak = int(np.argmax(np.abs(data)))
    A0 = float(data[idx_peak])
    mu0 = float(t[idx_peak])
    sigma0 = float((t[-1] - t[0]) / 6.0) or 1.0
    offset0 = float(np.median(data))
    return [A0, mu0, sigma0, offset0]


def fit_gauss_envelope(
    t: np.ndarray,
    data: np.ndarray,
    p0=None,
    maxfev: int = 10_000,
) -> dict:
    """Fit a Gaussian envelope to *data* sampled at positions *t*.

    Parameters
    ----------
    t      : 1-D array of sample positions (e.g. time or frequency)
    data   : 1-D array of observed values
    p0     : optional initial parameters [A, mu, sigma, offset]
    maxfev : maximum function evaluations passed to ``scipy.optimize.curve_fit``

    Returns
    -------
    dict with keys:
        ``params``  – fitted [A, mu, sigma, offset]
        ``cov``     – covariance matrix
        ``fitted``  – model evaluated at *t*
        ``residual``– data - fitted
    """
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)

    if p0 is None:
        p0 = _estimate_gauss_p0(t, data)

    def _model(t_, A, mu, sigma, offset):
        return gauss_envelope(t_, A, mu, sigma, offset)

    params, cov = curve_fit(_model, t, data, p0=p0, maxfev=maxfev)
    fitted = gauss_envelope(t, *params)
    return {
        "params": params,
        "cov": cov,
        "fitted": fitted,
        "residual": data - fitted,
    }


def _estimate_theta3_p0(t: np.ndarray, data: np.ndarray):
    """Rough initial-parameter estimate for theta-3 fit."""
    idx_peak = int(np.argmax(np.abs(data)))
    A0 = float(data[idx_peak]) / 3.0
    mu0 = float(t[idx_peak])
    T0 = float((t[-1] - t[0]) / 2.0) or 1.0
    q0 = 0.5
    offset0 = float(np.median(data))
    return [A0, mu0, T0, q0, offset0]


def fit_theta3_envelope(
    t: np.ndarray,
    data: np.ndarray,
    p0=None,
    n_terms: int = 20,
    maxfev: int = 10_000,
) -> dict:
    """Fit a Jacobi theta-3 envelope to *data* sampled at positions *t*.

    Parameters
    ----------
    t       : 1-D array of sample positions
    data    : 1-D array of observed values
    p0      : optional initial parameters [A, mu, T, q, offset]
    n_terms : terms used in the theta-3 series
    maxfev  : maximum function evaluations

    Returns
    -------
    dict with keys:
        ``params``  – fitted [A, mu, T, q, offset]
        ``cov``     – covariance matrix
        ``fitted``  – model evaluated at *t*
        ``residual``– data - fitted
    """
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)

    if p0 is None:
        p0 = _estimate_theta3_p0(t, data)

    def _model(t_, A, mu, T, q, offset):
        return theta3_envelope(t_, A, mu, T, q, offset, n_terms=n_terms)

    bounds = (
        [-np.inf, -np.inf, 1e-6, 1e-9, -np.inf],
        [np.inf, np.inf, np.inf, 1.0 - 1e-9, np.inf],
    )
    params, cov = curve_fit(
        _model, t, data, p0=p0, bounds=bounds, maxfev=maxfev
    )
    fitted = theta3_envelope(t, *params, n_terms=n_terms)
    return {
        "params": params,
        "cov": cov,
        "fitted": fitted,
        "residual": data - fitted,
    }


# ---------------------------------------------------------------------------
# Peak detection helper
# ---------------------------------------------------------------------------

def detect_tau_peaks(
    t: np.ndarray,
    data: np.ndarray,
    height_fraction: float = 0.3,
    min_distance_fraction: float = 0.05,
) -> np.ndarray:
    """Return indices of prominent peaks in *data*.

    Parameters
    ----------
    t, data                : arrays of equal length
    height_fraction        : peaks must exceed this fraction of max(|data|)
    min_distance_fraction  : minimum spacing between peaks as fraction of span

    Returns
    -------
    Sorted array of peak indices.
    """
    data = np.asarray(data, dtype=float)
    height = height_fraction * float(np.max(np.abs(data)))
    distance = max(1, int(min_distance_fraction * len(data)))
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def residual_rms(result: dict) -> float:
    """Root-mean-square of the fit residual."""
    return float(np.sqrt(np.mean(result["residual"] ** 2)))


def residual_max_abs(result: dict) -> float:
    """Maximum absolute residual."""
    return float(np.max(np.abs(result["residual"])))


# ---------------------------------------------------------------------------
# Plotting (matplotlib imported lazily to avoid side-effects at import time)
# ---------------------------------------------------------------------------

def plot_fit_result(
    t: np.ndarray,
    data: np.ndarray,
    result: dict,
    title: str = "Envelope fit",
    output_path: str | None = None,
) -> None:
    """Plot original data alongside the fitted envelope.

    matplotlib is imported *inside* this function so that importing
    ``theta_fit_tau`` for its fitting utilities does not pull in matplotlib
    (and therefore does not trigger the pyparsing DeprecationWarning).

    Parameters
    ----------
    t           : sample positions
    data        : observed data
    result      : dict returned by fit_gauss_envelope / fit_theta3_envelope
    title       : figure title
    output_path : if given, save the figure to this path instead of showing it
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_top = axes[0]
    ax_top.plot(t, data, label="data", color="steelblue", linewidth=1.0)
    ax_top.plot(
        t,
        result["fitted"],
        label="fit",
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
    )
    ax_top.set_ylabel("amplitude")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")

    ax_bot = axes[1]
    ax_bot.plot(
        t,
        result["residual"],
        label="residual",
        color="firebrick",
        linewidth=1.0,
    )
    ax_bot.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    ax_bot.set_xlabel("t")
    ax_bot.set_ylabel("residual")
    ax_bot.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=120)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry-point (not imported by the library users)
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a Gaussian or theta-3 envelope to a CSV time-series."
    )
    parser.add_argument("csv", help="Path to input CSV with columns t,value")
    parser.add_argument(
        "--model",
        choices=["gauss", "theta3"],
        default="gauss",
        help="Envelope model (default: gauss)",
    )
    parser.add_argument("--output", default=None, help="Path for output PNG")
    args = parser.parse_args()

    import csv

    rows = []
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((float(row["t"]), float(row["value"])))

    t_arr = np.array([r[0] for r in rows])
    v_arr = np.array([r[1] for r in rows])

    if args.model == "gauss":
        result = fit_gauss_envelope(t_arr, v_arr)
    else:
        result = fit_theta3_envelope(t_arr, v_arr)

    print(f"Fitted params : {result['params']}")
    print(f"Residual RMS  : {residual_rms(result):.6f}")

    if args.output:
        plot_fit_result(t_arr, v_arr, result, output_path=args.output)
        print(f"Plot saved to : {args.output}")


if __name__ == "__main__":
    _main()
