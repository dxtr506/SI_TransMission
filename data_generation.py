import numpy as np


def random_covariance(p, rho, rng):
    A = (rng.random((p, p)) < rho) * rho
    S = A.T @ A + np.eye(p)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)

# gen data theo 5.1 của paper

# Beta^(0)_j = 0.3 * 1 {j <= s}, s = 10, p = 500   -> j = 1..10, 10 toa do
# X^(0) ~ N(0, I)
# delta^(k)_j ~ N(0, (h / 50)^2); 1 <= j <= 50     -> j = 1..50, 50 toa do
# X^(k) ~ N(0, Sigma^(k))

def generate_data(h, rho, K = 4, p = 500, s = 10, nt = 150, ns = 200, seed = 0):
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:s] = 0.3

    X0 = rng.standard_normal((nt, p))
    y0 = X0 @ beta + rng.standard_normal(nt)

    X_list, y_list, w_list = [], [], []

    for _ in range(K):
        Sigma = random_covariance(p, rho, rng)
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        Xk = rng.standard_normal((ns, p)) @ L.T
        delta = np.zeros(p)

        # numpy nhận vào độ lệch chuẩn -> (h/50)
        delta[:50] = rng.normal(0.0, h / 50, 50)

        # beta(k) = beta(0) + delta
        wk = beta + delta
        X_list.append(Xk)
        y_list.append(Xk @ wk + rng.standard_normal(ns))
        w_list.append(wk)

    return {"X0": X0, "y0": y0, "X": X_list, "y": y_list, "beta": beta, "w": w_list}
