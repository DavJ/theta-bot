from src.theta_transform import theta_transform
import numpy as np

if __name__ == "__main__":
    # Sample input signal
    x = np.array([1, 0.5, -0.3, 0.2, 0.1])
    tau = 0.5 + 0.2j

    # Apply theta transform
    transformed = theta_transform(x, tau)
    print(f"Theta Transform result for tau={tau}: {transformed}")
