import numpy as np


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
def compute_Zu(XA, XAc, a, b, A, sA, Ac, lambda_T, nT):
    # Solves min 1/(2 nT) ||a + bz - X beta||^2 + lambda_T ||beta||_1
    # on (A, sA). Returns the interval and beta_T(z) = c_T + d_T z.

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
   
    p = len(A) + len(Ac)

    c, d = np.zeros(p), np.zeros(p)
    psi, gamma = [], []

    # Active :
    # beta_A(z) = c[A] + d[A] z, where
    # c[A] = (XA.T XA)^(-1) (XA.T a - nT lambda_T sA)
    # d[A] = (XA.T XA)^(-1) XA.T b
    if len(A) > 0:
        gram = XA.T @ XA
        c[A] = np.linalg.solve(gram, XA.T @ a - nT * lambda_T * sA)
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
        scale = nT * lambda_T

        # q_Ac(z) = XAc.T (y(z) - XA beta_A(z)) / (nT lambda_T)
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


def compute_ZG(X0, a, b, cC, dC, lambda_T, nT):
    # beta_C(z) = cC + dC z
    # g(z) = X0.T (a + bz - X0 beta_C(z)) / nT
    #      = g0 + g1 z
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    cC = np.asarray(cC, dtype=float).ravel()
    dC = np.asarray(dC, dtype=float).ravel()

    p = X0.shape[1]

    g0 = X0.T @ (a - X0 @ cC) / nT
    g1 = X0.T @ (b - X0 @ dC) / nT

    # ||g(z)||_inf <= lambda_T
    # <=> g1 z <= lambda_T 1_p - g0
    # and -g1 z <= lambda_T 1_p + g0
    psi = np.concatenate([g1, -g1])
    gamma = np.concatenate([lambda_T * np.ones(p) - g0, lambda_T * np.ones(p) + g0])

    l, r = solve_interval(psi, gamma)
    return l, r


def compute_ZR(cT, dT, A, sA, cC, dC, C, sC, lambda_T):
    # On Z_v, ||beta_C(z)||_1 = R0 + R1 z, where
    # R0 = sC.T cC[C] and R1 = sC.T dC[C].
    R0 = float(sC @ cC[C]) if len(C) > 0 else 0.0
    R1 = float(sC @ dC[C]) if len(C) > 0 else 0.0

    # On Z_u, B(z) = ||beta_T(z)||_1 + |A| lambda_T
    #                  = B0 + B1 z, where
    # B0 = sA.T cT[A] + |A| lambda_T and B1 = sA.T dT[A].
    B0 = float(sA @ cT[A]) if len(A) > 0 else 0.0
    B1 = float(sA @ dT[A]) if len(A) > 0 else 0.0
    B0 += len(A) * lambda_T

    # R(z) <= B(z)
    # <=> (R1 - B1) z <= B0 - R0
    psi = [R1 - B1]
    gamma = [B0 - R0]

    l, r = solve_interval(psi, gamma)
    return l, r
