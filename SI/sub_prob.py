import numpy as np

# Conditioning events of practical TransMission along Y(z) = a + b z, as functions of z.
#   1. Lasso states: each fit keeps its active set and signs on an interval of z.
#   2. Verification: J[i, t] = {z : lambda_0[i] passes both checks with lambda_T[t]}.
#   3. CV choice: CV losses are quadratic in z; their argmin changes only where two cross.


# 1. Lasso states

def solve_interval(psi, gamma):
    lu, ru = -np.inf, np.inf

    for i in range(len(psi)):
        if psi[i] == 0:
            if gamma[i] < 0:
                return [np.inf, -np.inf]
        elif psi[i] > 0:
            val = gamma[i] / psi[i]
            if val < ru:
                ru = val
        else:
            val = gamma[i] / psi[i]
            if val > lu:
                lu = val

    if lu > ru:
        return [np.inf, -np.inf]

    return [lu, ru]

# KKT for target Lasso 
def compute_Zu(XA, XAc, a, b, A, sA, Ac, lambda_T, nt):
    # Solves min 1/(2 nt) ||a + bz - X beta||^2 + lambda_T ||beta||_1
    # on (A, sA). Returns the interval and beta_T(z) = c_T + d_T z.

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
   
    p = len(A) + len(Ac)

    c, d = np.zeros(p), np.zeros(p)
    psi, gamma = [], []

    # Active :
    # beta_A(z) = c[A] + d[A] z, where
    # c[A] = (XA.T XA)^(-1) (XA.T a - nt lambda_T sA)
    # d[A] = (XA.T XA)^(-1) XA.T b
    if len(A) > 0:
        gram = XA.T @ XA
        c[A] = np.linalg.solve(gram, XA.T @ a - nt * lambda_T * sA)
        d[A] = np.linalg.solve(gram, XA.T @ b)

        # sign(beta_A(z)) = sA
        # <=> sA * (c[A] + d[A] z) >= 0
        # <=> (-sA * d[A]) z <= sA * c[A]
        psi.append(-sA * d[A])
        gamma.append(sA * c[A])

    # Inactive :
    # y(z) - XA beta_A(z) = e0 + e1 z, where
    # e0 = a - XA c[A] and e1 = b - XA d[A]
    if len(Ac) > 0:
        e0 = a - XA @ c[A]
        e1 = b - XA @ d[A]
        scale = nt * lambda_T

        # q_Ac(z) = XAc.T (y(z) - XA beta_A(z)) / (nt lambda_T)
        #         = q0 + q1 z
        q0 = XAc.T @ e0 / scale
        q1 = XAc.T @ e1 / scale

        # |q_Ac(z)| <= 1
        # <=> q1 z <= 1 - q0 and -q1 z <= 1 + q0
        psi += [q1, -q1]
        gamma += [np.ones(Ac.size) - q0, np.ones(Ac.size) + q0]

    l, r = solve_interval(np.concatenate(psi), np.concatenate(gamma))
    return l, r, c, d

# KKT for weighted lasso 
def compute_Zv(XA, XAc, a, b, A, sA, Ac, w, lambda_0, N):
    # Solves min 1/(2N) ||a + bz - X theta||^2
    #        + lambda_0 sum_j w_j |theta_j|
    # on (A, sA). Returns the interval and theta(z) = c + dz.

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    p = len(A) + len(Ac)

    c, d = np.zeros(p), np.zeros(p)
    psi, gamma = [], []

    # Active:
    # theta_A(z) = c[A] + d[A] z, where
    # c[A] = (XA.T XA)^(-1) (XA.T a - N lambda_0 (w[A] * sA))
    # d[A] = (XA.T XA)^(-1) XA.T b
    if len(A) > 0:
        gram = XA.T @ XA
        c[A] = np.linalg.solve(
            gram,
            XA.T @ a - N * lambda_0 * (w[A] * sA),
        )
        d[A] = np.linalg.solve(gram, XA.T @ b)

        # sign(theta_A(z)) = sA
        # <=> sA * (c[A] + d[A] z) >= 0
        # <=> (-sA * d[A]) z <= sA * c[A]
        psi.append(-sA * d[A])
        gamma.append(sA * c[A])

    # Inactive:
    # y(z) - XA theta_A(z) = e0 + e1 z, where
    # e0 = a - XA c[A] and e1 = b - XA d[A]
    if len(Ac) > 0:
        e0 = a - XA @ c[A]
        e1 = b - XA @ d[A]
        scale = N * lambda_0 * w[Ac]

        # q_Ac(z) = XAc.T (y(z) - XA theta_A(z))
        #           / (N lambda_0 w[Ac])
        #         = q0 + q1 z
        q0 = XAc.T @ e0 / scale
        q1 = XAc.T @ e1 / scale

        # |q_Ac(z)| <= 1
        # <=> q1 z <= 1 - q0 and -q1 z <= 1 + q0
        psi += [q1, -q1]
        gamma += [np.ones(Ac.size) - q0, np.ones(Ac.size) + q0]

    l, r = solve_interval(np.concatenate(psi), np.concatenate(gamma))
    return l, r, c, d


# 2. Verification

def compute_B(s, c, d, lambda_T):
    # Target fits t: B_t(z) = ||beta_T,t(z)||_1 + |A_t| lambda_T[t] = B0 + B1 z.
    return (s * c).sum(1) + (s != 0).sum(1) * lambda_T, (s * d).sum(1)


def compute_R(s, c, d):
    # Transfer candidates i: R_i(z) = ||beta_i(z)||_1 = R0 + R1 z.
    return (s * c).sum(1), (s * d).sum(1)


def compute_ZG(X0, a0, b0, c, d, lambda_T):
    # Transfer candidates i: beta_i(z) = c_i + d_i z gives
    # g_i(z) = X0.T (y0(z) - X0 beta_i(z)) / nt = G0 + G1 z.
    # [GL, GU][i, t] = {z : ||g_i(z)||_inf <= lambda_T[t]}.
    nt = X0.shape[0]
    G0 = (a0 - c @ X0.T) @ X0 / nt
    G1 = (b0 - d @ X0.T) @ X0 / nt

    # -lambda_T[t] <= G0 + G1 z <= lambda_T[t] on every coordinate
    lambda_T3 = lambda_T[None, :, None]   # shape (1, M_T, 1) for broadcasting
    g0, g1 = G0[:, None, :], G1[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        up, dn = (lambda_T3 - g0) / g1, (-lambda_T3 - g0) / g1
    upper = np.where(g1 > 0, up, np.where(g1 < 0, dn, np.inf)).min(axis=2)
    lower = np.where(g1 > 0, dn, np.where(g1 < 0, up, -np.inf)).max(axis=2)
    never = ((g1 == 0) & (np.abs(g0) > lambda_T3)).any(axis=2)
    lower[never], upper[never] = np.inf, -np.inf
    return lower, upper


def compute_ZR(R0, R1, B0, B1):
    # [RL, RU][i, t] = {z : R_i(z) <= B_t(z)}
    #                = {z : (R1_i - B1_t) z <= B0_t - R0_i}.
    coef = R1[:, None] - B1[None, :]
    rhs = B0[None, :] - R0[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rhs / coef
    lower = np.where(coef < 0, ratio, -np.inf)
    upper = np.where(coef > 0, ratio, np.inf)
    never = (coef == 0) & (rhs < 0)
    lower[never], upper[never] = np.inf, -np.inf
    return lower, upper


def compute_J(GL, GU, RL, RU):
    # J[i, t] = [GL, GU][i, t] ∩ [RL, RU][i, t].
    lower, upper = np.maximum(GL, RL), np.minimum(GU, RU)
    empty = lower > upper
    lower[empty], upper[empty] = np.inf, -np.inf
    return lower, upper


# 3. CV choice

def compute_cv_loss(X0v, a0v, b0v, c, d):
    # Fold fits: ||y0[I_v](z) - X0[I_v] beta_{-v}(z)||^2 as (z^2, z, 1) coefficients.
    E0 = a0v - c @ X0v.T
    E1 = b0v - d @ X0v.T
    return np.column_stack([(E1 * E1).sum(1), 2.0 * (E0 * E1).sum(1), (E0 * E0).sum(1)])


def compute_crossings(Q, k):
    # Roots of L_k(z) - L_j(z) = 0 for every j (nan where there are none).
    A2, A1, A0 = (Q[k] - Q).T
    disc = A1 * A1 - 4.0 * A2 * A0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -0.5 * (A1 + np.copysign(np.sqrt(disc), A1))
        roots = np.column_stack([q / A2, A0 / q])
    roots[disc < 0] = np.nan
    roots[k] = np.nan
    return roots
