import numpy as np

class ThetaBasis:
    def __init__(self, K=16, tau_im=0.25, ridge=1e-3, gram=True):
        self.K, self.tau_im, self.ridge, self.gram = K, tau_im, ridge, gram
        self.phases = None

    def _theta3(self, t_over_T, q):
        n = np.arange(-20, 21)
        return np.sum(q**(n**2) * np.cos(2*np.pi*np.outer(t_over_T, n)), axis=1)

    def design_matrix(self, T):
        if self.phases is None:
            self.phases = np.linspace(0, 1, self.K, endpoint=False)
        q = np.exp(-np.pi * self.tau_im)
        t_over_T = np.linspace(0, 1, T, endpoint=False)
        mats = [ self._theta3((t_over_T + phi) % 1.0, q) for phi in self.phases ]
        Phi = np.stack(mats, axis=1)
        Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-8)
        if self.gram:
            Phi = self._gram(Phi)
        return Phi

    def _gram(self, M):
        Q = np.zeros_like(M)
        for i in range(M.shape[1]):
            v = M[:, i].copy()
            for j in range(i):
                v -= np.dot(Q[:, j], M[:, i]) * Q[:, j]
            Q[:, i] = v / (np.linalg.norm(v) + 1e-12)
        return Q

    def fit(self, s):
        T = len(s)
        Phi = self.design_matrix(T)
        A = Phi.T @ Phi + self.ridge * np.eye(self.K)
        b = Phi.T @ s
        a = np.linalg.solve(A, b)
        recon = Phi @ a
        resid = s - recon
        return resid

class SimpleKalman:
    # minimal scalar kalman on residual
    def __init__(self, q=1e-4, r=1e-3):
        self.q, self.r = q, r
        self.x = 0.0
        self.p = 1.0

    def step(self, z):
        # predict
        x_pred = self.x
        p_pred = self.p + self.q
        # update
        k = p_pred / (p_pred + self.r)
        innov = z - x_pred
        self.x = x_pred + k * innov
        self.p = (1.0 - k) * p_pred
        return self.x, innov

class ThetaKalman:
    def __init__(self, K=16, tau_im=0.25):
        self.theta = ThetaBasis(K=K, tau_im=tau_im)
        self.kalman = SimpleKalman()
        self._last_len = 0
        self._innovs = []

    def process(self, closes):
        if len(closes) < 128:
            return None
        s = np.log(np.asarray(closes))
        resid = self.theta.fit(s)
        # Use last residual as observation
        z = float(resid[-1])
        x, innov = self.kalman.step(z)
        self._innovs.append(innov)
        return {"innov": self._innovs[-256:]}
