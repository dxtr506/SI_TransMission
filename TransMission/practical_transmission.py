import numpy as np

from TransMission.transmission import (build_tf_design, penalty_weights, weighted_lasso,)


# Grids, folds

def make_grids(X0, X_list, M_0=40, M_T=40, lambda_min=0.15, lambda_max=1.5):

    lambda_0 = np.linspace(lambda_min, lambda_max, M_0)
    lambda_T = np.linspace(lambda_min, lambda_max, M_T)
    
    return lambda_0, lambda_T
     
def make_folds(nt, V=3):
    return np.array_split(np.arange(nt), V)


# Algorithm

def target_loss(X0, y0, idx, beta):
    r = y0[idx] - X0[idx] @ beta
    return r @ r / (2 * len(idx))


def practical_transmission_CV(X0, y0, X_list, y_list, folds=None, V=3):
    
    X0 = np.asarray(X0, float)
    y0 = np.asarray(y0, float)
    nt, p = X0.shape

    N = sum(np.shape(Xk)[0] for Xk in X_list) + nt
    K = len(X_list)

    lamb0_grid, lambt_grid = make_grids(X0, X_list)
  
    if folds is None:
        folds = make_folds(nt, V)
    w = np.array([len(Iv) / nt for Iv in folds])
    ones_p = np.ones(p)

    # Step 1: constraint bounds B(lambT) and the single-task fallback
    bh0_T = weighted_lasso(X0, y0, lambt_grid, ones_p)

    B = np.array([np.abs(b).sum() + int(np.count_nonzero(b)) * lamT
                  for b, lamT in zip(bh0_T, lambt_grid)])

    Lcv_T = np.zeros(len(lambt_grid))
    for v, Iv in enumerate(folds):
        tr = np.setdiff1d(np.arange(nt), Iv)
        for i, b in enumerate(weighted_lasso(X0[tr], y0[tr], lambt_grid, ones_p)):
            Lcv_T[i] += w[v] * target_loss(X0, y0, Iv, b)
    beta_T_cv = bh0_T[int(np.argmin(Lcv_T))]

    # Step 2: CV and refit for the unconstrained weighted lasso 
    X_tf, y_tf = build_tf_design(X0, y0, X_list, y_list)
    a = penalty_weights(X0, X_list)
    tgt_rows = np.arange(N - nt, N)          # target rows inside the TF design
    tgt_cols = slice(K * p, (K + 1) * p)     # theta^(0) block

    Lcv = np.zeros(len(lamb0_grid))
    for v, Iv in enumerate(folds):
        keep = np.setdiff1d(np.arange(N), tgt_rows[Iv])
        for i, theta in enumerate(
                weighted_lasso(X_tf[keep], y_tf[keep], lamb0_grid, a)):
            Lcv[i] += w[v] * target_loss(X0, y0, Iv, theta[tgt_cols])

    beta_full = [theta[tgt_cols]
                 for theta in weighted_lasso(X_tf, y_tf, lamb0_grid, a)]
    G = np.array([np.abs(X0.T @ (y0 - X0 @ b) / nt).max() for b in beta_full])
    R = np.array([np.abs(b).sum() for b in beta_full])

    # Step 3: feasibility post-check 
    feasible = ((G[:, None] <= lambt_grid[None, :])
                & (R[:, None] <= B[None, :])).any(axis=1)

    # Step 4: select, or fall back 
    if feasible.any():
        idx = np.flatnonzero(feasible)
        best = idx[int(np.argmin(Lcv[idx]))]
        coef, branch = beta_full[best], "transmission"
    else:
        best = None
        coef, branch = beta_T_cv, "fallback"

    return {
        "coef": coef,
        "branch": branch,
        "feature_selection": np.flatnonzero(np.abs(coef) != 0),
        "lam0_hat": None if best is None else lamb0_grid[best],
        "lam0_grid": lamb0_grid, "lamT_grid": lambt_grid,
        "Lcv": Lcv, "Lcv_T": Lcv_T, "G": G, "R": R, "B": B,
        "feasible": feasible, "beta_T_cv": beta_T_cv,
        "beta_path": beta_full,
    }
