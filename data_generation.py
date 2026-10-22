import numpy as np


def generate_design(rng, p, nt, ns, K):
    # X0 (nt x p) and X_k (ns x p), k = 1..K, with iid N(0, 1) entries
    X0 = rng.standard_normal((nt, p))
    X_list = [rng.standard_normal((ns, p)) for _ in range(K)]
    return X0, X_list


def generate_contrasts(rng, p, K, H):
    # c_k: iid N(0, (H/50)^2) on the first 50 coordinates of every source, 0 elsewhere (He et al., Sec. 5.1)
    contrasts = []
    for _ in range(K):
        c = np.zeros(p)
        c[:50] = rng.normal(0.0, H / 50, 50)
        contrasts.append(c)
    return contrasts


NOISES = ["normal", "laplace", "skew", "t10", "t20"]


def draw_noise(rng, size, noise="normal"):
    # iid errors with mean 0 and variance 1; only "normal" matches the Gaussian model of the inference
    if noise == "normal":
        return rng.standard_normal(size)
    if noise == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    if noise == "skew":
        # skew normal with shape a = 10: d|u| + sqrt(1 - d^2) v, u, v iid N(0, 1), d = a / sqrt(1 + a^2)
        d = 10.0 / np.sqrt(101.0)
        x = d * np.abs(rng.standard_normal(size)) + np.sqrt(1.0 - d * d) * rng.standard_normal(size)
        return (x - d * np.sqrt(2.0 / np.pi)) / np.sqrt(1.0 - 2.0 * d * d / np.pi)
    df = {"t10": 10, "t20": 20}[noise]
    return rng.standard_t(df, size) * np.sqrt((df - 2.0) / df)


def generate_response(rng, X0, X_list, sigma, beta0=None, contrasts=None, noise="normal"):
    # y0 = X0 beta0 + sigma eps0, y_k = X_k (beta0 + c_k) + sigma eps_k, eps iid with mean 0 and variance 1
    # (N(0, 1) by default, see draw_noise); beta0 = 0 and c_k = 0 by default (global null)
    p = X0.shape[1]
    if beta0 is None:
        beta0 = np.zeros(p)
    if contrasts is None:
        contrasts = [np.zeros(p) for _ in X_list]
    y0 = X0 @ beta0 + sigma * draw_noise(rng, X0.shape[0], noise)
    y_list = [Xk @ (beta0 + c) + sigma * draw_noise(rng, Xk.shape[0], noise)
              for Xk, c in zip(X_list, contrasts)]
    return y0, y_list
