import numpy as np
from skglm import Lasso

# Solver

def weighted_lasso(X, y, alphas, weights):
    # argmin 1/(2n)||y - Xw||^2 + alpha*sum_j a_j|w_j|.
    # skglm argmin 1/(2n)||y - X~v||^2 + alpha * |v|

    # v = a * w, vì aj > 0 -> |w| = |v| / a
    # X~ = X / a
    Xs = np.asfortranarray(X / weights, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    # ws_strategy="fixpoint", no warm start: other settings sometimes stop with a wrong active set.
    # tol=1e-11: for large y skglm cannot reach 1e-12; max_iter / max_epochs (skglm defaults) bound
    # the time of a fit that does not reach tol, whose solution check_kkt still checks.
    model = Lasso(fit_intercept=False, tol=1e-11, max_iter=50, max_epochs=50000, ws_strategy="fixpoint")
    coefs = []
    for alpha in alphas:
        model.alpha = alpha
        model.fit(Xs, y)

        # w = v / a
        coef = model.coef_ / weights

        check_kkt(X, y, coef, alpha, weights)
        coefs.append(coef)
    return coefs


# Số fit đã giải và số fit vi phạm KKT trong tiến trình này.
kkt_count = {"fits": 0, "violations": 0}


# check nghiệm tối ưu thỏa kkt hay chưa (chỉ đếm, không dừng chương trình)
def check_kkt(X, y, coef, alpha, weights):
    kkt_count["fits"] += 1
    if not np.isfinite(coef).all():
        kkt_count["violations"] += 1
        return
    grad = -X.T @ (y - X @ coef) / X.shape[0]
    if not np.isfinite(grad).all():
        kkt_count["violations"] += 1
        return
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
    # skglm sometimes reports convergence while its solution violates KKT;
    # such fits are counted here.
    kkt_count["violations"] += int(viol > 1e-6 * scale)


# Design, penalty weights
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
