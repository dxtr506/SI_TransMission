import numpy as np


def generate_design(rng, p, nt, ns, K):
    # X0 (nt x p) and X_k (ns x p), k = 1..K, with iid N(0, 1) entries
    X0 = rng.standard_normal((nt, p))
    X_list = [rng.standard_normal((ns, p)) for _ in range(K)]
    return X0, X_list


def generate_contrasts(rng, p, K, H):
    # c_k: iid N(0, (H/50)^2) on the first 50 coordinates, 0 elsewhere
    contrasts = []
    for _ in range(K):
        c = np.zeros(p)
        c[:50] = rng.normal(0.0, H / 50, 50)
        contrasts.append(c)
    return contrasts


def generate_response(rng, X0, X_list, sigma, beta0=None, contrasts=None):
    # y0 = X0 beta0 + sigma eps0, y_k = X_k (beta0 + c_k) + sigma eps_k, eps iid N(0, 1)
    # beta0 = 0 and c_k = 0 by default (global null)
    p = X0.shape[1]
    if beta0 is None:
        beta0 = np.zeros(p)
    if contrasts is None:
        contrasts = [np.zeros(p) for _ in X_list]
    y0 = X0 @ beta0 + sigma * rng.standard_normal(X0.shape[0])
    y_list = [Xk @ (beta0 + c) + sigma * rng.standard_normal(Xk.shape[0])
              for Xk, c in zip(X_list, contrasts)]
    return y0, y_list
