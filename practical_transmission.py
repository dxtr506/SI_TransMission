import numpy as np
from skglm import WeightedLasso


# Solver

def weighted_lasso(X, y, alphas, weights):
    # argmin 1/(2n)||y - Xw||^2 + alpha*sum_j a_j|w_j|.
    model = WeightedLasso(weights=weights, tol=1e-10, fit_intercept=False)
    coefs = []
    for alpha in alphas:
        model.alpha = alpha
        model.fit(X, y)
        coef = model.coef_.copy()

        # skglm dùng KKT làm điều kiện dừng, nhưng khi hết max_iter mà chưa đạt
        # tol thì nó im lặng trả về nghiệm chưa hội tụ (anderson_cd.py không có
        # ConvergenceWarning). G, R, s_hat đều so trực tiếp với ngưỡng nên phải
        # tự kiểm, để lỗi nổ ra thay vì âm thầm làm sai hậu kiểm.
        check_kkt(X, y, coef, alpha, weights)
        coefs.append(coef)
    return coefs


# check nghiệm tối ưu thỏa kkt hay chưa
def check_kkt(X, y, coef, alpha, weights):
    grad = -X.T @ (y - X @ coef) / X.shape[0]
    active = np.abs(coef) > 1e-9
    viol = 0.0

    # Beta != 0 -> subgradient = sign(Beta)
    # # -> kkt: grad(Beta) + alpha * w * sign(beta) = 0   
    if active.any():
        viol = np.abs(grad[active] + alpha * weights[active] * np.sign(coef[active])).max()

    # Beta = 0 -> subgradient -> grad|beta| = [-1, 1]
    # kkt: z in [-1, 1] : grad(L) + alpha * w * z = 0 
    # <-> |grad| <= alpha * w <-> |grad| - alpha * z <= 0

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

# choice para in A.3 (theo collary 5)
# khi h = 0 thì vế số hạng đầu của lambda_0 mất,
def make_grids(p, N, nt, M_0=100, M_T=20):
    # tạo grid lambda giảm dần 
    lambda_0 = np.sqrt(np.log(p) / N) * np.geomspace(1e-1, 10 ** 1.5, M_0)
    lambda_t = np.sqrt(np.log(p) / nt) * np.geomspace(1e-1, 1e1, M_T)
    return lambda_0[::-1].copy(), lambda_t[::-1].copy()


def make_folds(nt, V=3):
    return np.array_split(np.arange(nt), V)


# Algorithm

def target_loss(X0, y0, idx, beta):
    r = y0[idx] - X0[idx] @ beta
    return r @ r / (2 * len(idx))


def practical_transmission(X0, y0, X_list, y_list, folds=None, V=3):
    
    X0 = np.asarray(X0, float)
    y0 = np.asarray(y0, float)
    nt, p = X0.shape

    N = sum(np.shape(Xk)[0] for Xk in X_list) + nt
    K = len(X_list)

    lamb0_grid, lambt_grid = make_grids(p, N, nt)
  
    if folds is None:
        folds = make_folds(nt, V)
    w = np.array([len(Iv) / nt for Iv in folds])
    ones_p = np.ones(p)

    # Step 1: constraint bounds B(lambT) and the single-task fallback
    bh0_T = weighted_lasso(X0, y0, lambt_grid, ones_p)

    # s_hat: cùng ngưỡng 1e-9 với check_kkt
    B = np.array([np.abs(b).sum() + int((np.abs(b) > 1e-9).sum()) * lamT
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
        "feature_selection": np.flatnonzero(np.abs(coef) > 1e-9),
        "lam0_hat": None if best is None else lamb0_grid[best],
        "lam0_grid": lamb0_grid, "lamT_grid": lambt_grid,
        "Lcv": Lcv, "Lcv_T": Lcv_T, "G": G, "R": R, "B": B,
        "feasible": feasible, "beta_T_cv": beta_T_cv,
        "beta_path": beta_full,
    }

    