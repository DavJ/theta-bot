import numpy as np

def theta3(z: complex, tau: complex, n_max: int = 50) -> complex:
    """
    Jacobi Theta 3 function approximation.
    """
    return sum(
        np.exp(np.pi * 1j * n**2 * tau + 2 * np.pi * 1j * n * z)
        for n in range(-n_max, n_max + 1)
    )

def theta_transform(x: np.ndarray, tau: complex) -> complex:
    """
    Generalized Theta Transform (ZΘT) of time series x[n].
    """
    N = len(x)
    return sum(x[n] * theta3(n, tau) for n in range(N))
