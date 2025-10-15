import numpy as np

class ComplexKalmanDiag:
    """
    2D (Re,Im) per-bin Kalman; učení na train deltách, pak ROLLING předpověď po testu.
    Vrací p_t v [0,1] pro každý bar testu (časově proměnné).
    Citlivější defaulty (q, r, scale, eps_bias), aby se p odlepilo od 0.5.
    """
    def __init__(self, q=1e-3, r=5e-4, scale=40.0, eps_bias=1e-2):
        self.q = float(q); self.r = float(r)
        self.scale = float(scale); self.eps_bias = float(eps_bias)

    def _filter_train_last(self, obs):
        # obs: [T,2] (Re,Im) delty
        x = np.zeros(2); P = np.eye(2)
        F = np.eye(2); H = np.eye(2)
        Q = self.q * np.eye(2); R = self.r * np.eye(2)
        for t in range(obs.shape[0]):
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update
            y = obs[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P
        return x, P  # poslední stav po TRAIN

    def _roll_test_probs(self, x0, P0, obs_test, w):
        # sekvenčně zpracuj test delty; mapuj Re(drift) na p_t přes sigmoid
        F = np.eye(2); H = np.eye(2)
        Q = self.q * np.eye(2); R = self.r * np.eye(2)
        x = x0.copy(); P = P0.copy()
        probs = []
        for t in range(obs_test.shape[0]):
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update
            y = obs_test[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P
            # skóre z reálné složky driftu; promítneme váhu binu
            s = float(w * x[0])
            if abs(s) < 1e-12:
                s = self.eps_bias
            probs.append(1.0 / (1.0 + np.exp(-self.scale * s)))
        return np.array(probs, dtype=float)

    def fit_predict_proba(self, Z_train, Z_test):
        # Z_*: [T, B] complex; pracujeme nad deltami
        if Z_train is None or Z_test is None or Z_train.size == 0 or Z_test.size == 0:
            return np.ones(Z_test.shape[0]) * 0.5

        B = Z_train.shape[1]
        dZ_tr = np.diff(Z_train, axis=0)
        dZ_te = np.diff(Z_test,  axis=0)
        if dZ_tr.shape[0] < 2 or dZ_te.shape[0] < 1:
            return np.ones(Z_test.shape[0]) * 0.5

        # váhy podle průměrné magnitudy (per-bin)
        weights = np.mean(np.abs(Z_train), axis=0) + 1e-6
        weights = weights / (weights.sum() + 1e-12)

        # Per-bin trénink posledního stavu + rolling test; agregace přes váhy
        probs_bins = []
        for b in range(B):
            obs_tr = np.stack([dZ_tr[:, b].real, dZ_tr[:, b].imag], axis=1)
            x_last, P_last = self._filter_train_last(obs_tr)

            obs_te = np.stack([dZ_te[:, b].real, dZ_te[:, b].imag], axis=1)
            probs_b = self._roll_test_probs(x_last, P_last, obs_te, w=weights[b])
            probs_bins.append(probs_b)

        probs_bins = np.stack(probs_bins, axis=1)  # [T_test-1, B]
        probs = (probs_bins * weights[None, :]).sum(axis=1)  # vážený průměr napříč biny

        # zarovnej délku na počet test barů (přišel jeden krok kvůli diff)
        if len(probs) < Z_test.shape[0]:
            probs = np.concatenate([[probs[0]], probs], axis=0)
        return probs
