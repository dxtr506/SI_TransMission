import numpy as np
from sklearn.linear_model import Lasso

# Solver

def weighted_lasso(X, y, alphas, weights):
    # argmin 1/(2n)||y - Xw||^2 + alpha*sum_j a_j|w_j|.
    # sklearn argmin 1/(2n)||y - X~v||^2 + alpha * |v|

    # v = a * w, vì aj > 0 -> |w| = |v| / a
    # X~ = X / a
    Xs = np.asfortranarray(X / weights, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    model = Lasso(fit_intercept=False, tol=1e-14, max_iter=200000, warm_start=True)
    coefs = []
    for alpha in alphas:
        model.alpha = alpha
        model.fit(Xs, y, check_input=False)

        # w = v / a
        coef = model.coef_ / weights

        check_kkt(X, y, coef, alpha, weights)
        coefs.append(coef)
    return coefs


# check nghiệm tối ưu thỏa kkt hay chưa
def check_kkt(X, y, coef, alpha, weights):
    if not np.isfinite(coef).all():
        raise RuntimeError("weighted lasso returned non-finite coefficients")
    grad = -X.T @ (y - X @ coef) / X.shape[0]
    if not np.isfinite(grad).all():
        raise RuntimeError("weighted lasso has a non-finite KKT gradient")
    active = np.abs(coef) != 0
    viol = 0.0

    # Beta != 0 -> subgradient = sign(Beta)
    # # -> kkt: grad(Beta) + alpha * w * sign(beta) = 0   
    if active.any():
        viol = np.abs(grad[active] + alpha * weights[active] * np.sign(coef[active])).max()

    # Beta = 0 -> subgradient -> grad|beta| = [-1, 1]
    # kkt: z in [-1, 1] : grad(L) + alpha * w * z = 0 
    # <-> |grad| <= alpha * weights

    if (~active).any():
        viol = max(viol, np.maximum(0.0, np.abs(grad[~active]) - alpha * weights[~active]).max())
    scale = max(1.0, alpha * weights.max())
    if viol > 1e-7 * scale:
        raise RuntimeError(f"weighted lasso did not converge: KKT violation {viol} ")


# Design, penalty weights, grids, folds
# build theta, X, y
def build_tf_design(X0, y0, X_list, y_list):
    K = len(X_list)
    p = X0.shape[1]

    # số hàng X
    n_rows = sum(Xk.shape[0] for Xk in X_list) + X0.shape[0]

    X = np.zeros((n_rows, (K + 1) * p))
    r = 0

    # k = 0, Xk = X1 .... k = K - 1, Xk = XK
    for k, Xk in enumerate(X_list):
        nk = Xk.shape[0]
        #  [r:r + nk] lấy các hàng của source hiện tại
        X[r:r + nk, k * p:(k + 1) * p] = Xk
        X[r:r + nk, K * p:] = Xk
        r += nk
    X[r:, K * p:] = X0
    y = np.concatenate(list(y_list) + [y0])
    return X, y

# a = [ak, a0]; a_0 = 1 cho target block, a_k = n_k / sqrt(N n_t)
def penalty_weights(X0, X_list):
    K = len(X_list)
    p = X0.shape[1]
    nt = X0.shape[0]
    N = sum(Xk.shape[0] for Xk in X_list) + nt
    a = np.empty((K + 1) * p)
    for k, Xk in enumerate(X_list):
        a[k * p:(k + 1) * p] = Xk.shape[0] / np.sqrt(N * nt) # ak
    a[K * p:] = 1.0 # a0
    return a

def target_noise_score_quantile(X0, sigma, quantile, n_draws, batch_size, rng):
    nt = X0.shape[0]
    scores = np.empty(n_draws)

    start = 0
    while start < n_draws:
        stop = min(start + batch_size, n_draws)
        noise = sigma * rng.standard_normal((nt, stop - start))
        scores[start:stop] = np.max(np.abs(X0.T @ noise), axis=0) / nt
        start = stop

    return float(np.quantile(scores, quantile))


def transmission_noise_score_quantile(
        X0, X_list, sigma, quantile, n_draws, batch_size, rng):
    nt, p = X0.shape
    N = nt + sum(Xk.shape[0] for Xk in X_list)
    source_weights = [Xk.shape[0] / np.sqrt(N * nt) for Xk in X_list]
    scores = np.empty(n_draws)

    start = 0
    while start < n_draws:
        stop = min(start + batch_size, n_draws)
        width = stop - start

        target_noise = sigma * rng.standard_normal((nt, width))
        common_score = X0.T @ target_noise
        batch_scores = np.zeros(width)

        for Xk, ak in zip(X_list, source_weights):
            source_noise = sigma * rng.standard_normal((Xk.shape[0], width))
            source_score = Xk.T @ source_noise

            # Scores for delta^(k), whose weighted-Lasso penalty is lambda_0 * a_k.
            batch_scores = np.maximum(
                batch_scores,
                np.max(np.abs(source_score), axis=0) / (N * ak),
            )

            # The beta^(0) block appears in every source and target linear predictor.
            common_score += source_score

        batch_scores = np.maximum(
            batch_scores,
            np.max(np.abs(common_score), axis=0) / N,
        )
        scores[start:stop] = batch_scores
        start = stop

    return float(np.quantile(scores, quantile))


def noise_score_references(
        X0, X_list, sigma=1.0, quantile=0.95,
        n_draws=2000, batch_size=100, seed=0):
    X0 = np.asarray(X0, dtype=float)
    X_list = [np.asarray(Xk, dtype=float) for Xk in X_list]

    target_seed, transmission_seed = np.random.SeedSequence(seed).spawn(2)
    lambda_T_ref = target_noise_score_quantile(
        X0, sigma, quantile, n_draws, batch_size,
        np.random.default_rng(target_seed),
    )
    lambda_0_ref = transmission_noise_score_quantile(
        X0, X_list, sigma, quantile, n_draws, batch_size,
        np.random.default_rng(transmission_seed),
    )

    return lambda_0_ref, lambda_T_ref


def make_grids(
        X0, X_list, sigma=1.0, quantile=0.95,
        M_0=60, M_T=40, c_min=0.1, c_max=4.0,
        n_draws=2000, batch_size=100, seed=0):

    lambda_0_ref, lambda_T_ref = noise_score_references(
        X0, X_list, sigma=sigma, quantile=quantile,
        n_draws=n_draws, batch_size=batch_size, seed=seed,
    )

    multipliers_0 = np.geomspace(c_min, c_max, M_0)
    multipliers_T = np.geomspace(c_min, c_max, M_T)
    lambda_0 = lambda_0_ref * multipliers_0
    lambda_T = lambda_T_ref * multipliers_T

    # Descending order preserves efficient warm starts in weighted_lasso.
    return lambda_0[::-1].copy(), lambda_T[::-1].copy()


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



def practical_transmission_non_CV(X0, y0, X_list, y_list, lambda_0, lambda_T):

    X0 = np.asarray(X0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    nt, p = X0.shape

    # Step 1: target fit, verification bound and fallback estimate.
    beta_T = weighted_lasso(X0, y0, [lambda_T], np.ones(p))[0]
    s_hat = int(np.count_nonzero(beta_T))
    B = float(np.abs(beta_T).sum() + s_hat * lambda_T)

    # Step 2: a single transfer candidate using all samples.
    X_tf, y_tf = build_tf_design(X0, y0, X_list, y_list)
    weights = penalty_weights(X0, X_list)
    theta = weighted_lasso(X_tf, y_tf, [lambda_0], weights)[0]
    beta_candidate = theta[len(X_list) * p:].copy()

    # Steps 3 and 4: verify the candidate, or fall back to the target fit.
    G = float(np.abs(X0.T @ (y0 - X0 @ beta_candidate) / nt).max())
    R = float(np.abs(beta_candidate).sum())
    feasible = bool(G <= lambda_T and R <= B)
    coef = beta_candidate if feasible else beta_T
    branch = "transmission" if feasible else "fallback"

    return {
        "coef": coef,
        "branch": branch,
        "feature_selection": np.flatnonzero(np.abs(coef) != 0),
        "lambda_0": lambda_0, "lambda_T": lambda_T,
        "beta_T": beta_T, "theta": theta, "beta_candidate": beta_candidate,
        "s_hat": s_hat, "G": G, "R": R, "B": B,
        "feasible": feasible,
    }

