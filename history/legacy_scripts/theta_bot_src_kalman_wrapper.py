from pykalman import KalmanFilter
import numpy as np

def kalman_predict(x: np.ndarray):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, state_covs = kf.em(x).smooth(x)
    return state_means, state_covs
