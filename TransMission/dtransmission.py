import numpy as np
from sklearn.linear_model import Lasso
from TransMission.transmission import build_tf_design, weighted_lasso


def compute_debiased_beta_simple(Xk, yk):
    #SCAD remark 10
    ns, p = Xk.shape
    
    alpha_scad = 0.5 * np.sqrt(np.log(p) / ns) # giá trị cố định theo cthuc
    model = Lasso(alpha=alpha_scad, fit_intercept=False, tol=1e-12, max_iter=100000)
    model.fit(Xk, yk)
    beta_hat = model.coef_

    # tham số scad
    a_scad, gamma_scad = 3.7, alpha_scad
    t = np.abs(beta_hat) # |B_j^| của từng hệ số

    # mức phạt
    # |β̂_j| ≤ λ : w_j = λ
    #  λ < |β̂_j| ≤ aλ: w_j = (aλ − |β̂_j|)/(a − 1)
    # |β̂_j| > aλ: w_j = 0
    w = np.where(t <= gamma_scad, gamma_scad, np.where(t <= a_scad * gamma_scad, (a_scad * gamma_scad - t) / (a_scad - 1), 0.0))
    w = np.where(w < 1e-8, 1e-8, w)

    model = Lasso(alpha=1.0, fit_intercept=False, tol=1e-12, max_iter=100000)
    model.fit(Xk / w[None, :], yk)
    beta_tilde = model.coef_ / w
    return beta_tilde


# Không cần viết solver riêng cho ||beta_tilde - beta||^2
# (1/2N)‖y₀ − X₀β⁽⁰⁾‖² + Σ_k (n_S/2N)‖β̃⁽ᵏ⁾ − β⁽ᵏ⁾‖² + phạt ℓ1 
# -> (1/2N)‖y₀ − X₀β⁽⁰⁾‖² + Σ_k (1/2N)‖ỹ_k − X̃_kβ⁽ᵏ⁾‖² + phạt ℓ1
def build_pseudo_source_data(beta_tilde, ns, p):
    K = len(beta_tilde)
    X_pseudo = [np.sqrt(ns[k]) * np.eye(p) for k in range(K)]
    y_pseudo = [np.sqrt(ns[k]) * beta_tilde[k] for k in range(K)]
    return X_pseudo, y_pseudo


def weighted_lasso_D(X, y, alphas, weights, N):
    n_actual = X.shape[0]
    scale = N / n_actual
    alphas_adj = [a * scale for a in alphas]
    return weighted_lasso(X, y, alphas_adj, weights)


def penalty_weights_D(ns_list, nt, p):
    K = len(ns_list)
    N = sum(ns_list) + nt
    a = np.empty((K + 1) * p)
    for k in range(K):
        a[k * p:(k + 1) * p] = ns_list[k] / np.sqrt(N * nt)
    a[K * p:] = 1.0
    return a


def dtransmission_fixed(X0, y0, beta_tilde_list, ns_list, lambda_0, lambda_T):
    X0 = np.asarray(X0, dtype=float); y0 = np.asarray(y0, dtype=float)
    nt, p = X0.shape
    K = len(beta_tilde_list)
    N = nt + sum(ns_list)

    beta_T = weighted_lasso(X0, y0, [lambda_T], np.ones(p))[0]
    s_hat = int(np.count_nonzero(beta_T))
    B = float(np.abs(beta_T).sum() + s_hat * lambda_T)

    X_pseudo, y_pseudo = build_pseudo_source_data(beta_tilde_list, ns_list, p)
    X_tf, y_tf = build_tf_design(X0, y0, X_pseudo, y_pseudo)
    weights = penalty_weights_D(ns_list, nt, p)
    theta = weighted_lasso_D(X_tf, y_tf, [lambda_0], weights, N)[0]
    beta_candidate = theta[K * p:].copy()

    G = float(np.abs(X0.T @ (y0 - X0 @ beta_candidate) / nt).max())
    R = float(np.abs(beta_candidate).sum())
    feasible = bool(G <= lambda_T and R <= B)
    coef = beta_candidate if feasible else beta_T
    branch = "transmission" if feasible else "fallback"

    return {"coef": coef, "branch": branch,
            "feature_selection": np.flatnonzero(np.abs(coef) != 0),
            "lambda_0": lambda_0, "lambda_T": lambda_T,
            "beta_T": beta_T, "theta": theta, "beta_candidate": beta_candidate,
            "s_hat": s_hat, "G": G, "R": R, "B": B, "feasible": feasible}
