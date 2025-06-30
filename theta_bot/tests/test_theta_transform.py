import numpy as np
from src.theta_transform import theta_transform

def test_theta_transform():
    x = np.array([1, 2, 3, 4])
    tau = 0.5 + 0.1j
    result = theta_transform(x, tau)
    assert isinstance(result, complex)
