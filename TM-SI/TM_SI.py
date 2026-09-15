from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from practical_transmission import (
    build_tf_design,
    penalty_weights,
    practical_transmission_non_CV,
    weighted_lasso,
)
from sub_prob import compute_ZG, compute_ZR, compute_Zu, compute_Zv
from utils import (
    calculate_a_b,
    calculate_tn_p_value,
    construct_Sigma,
    construct_test_statistic,
    merge_intervals,
)


def divide_and_conquer(X0, X, a, b, M, w, lambda_0, lambda_T, z_min, z_max):
    # Return {z in [z_min, z_max] : M(Y(z)) = M} as a list of intervals.
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    nT, p = X0.shape
    N = X.shape[0]
    target_start = X.shape[1] - p

    intervals = []
    z = float(z_min)

    while z < z_max:
        # Target Lasso state; beta_T(z) = cT + dT z 

        # build y = a + bz
        Y0z = (a + b * z)[-nT:]
        beta_T = weighted_lasso(X0, Y0z, [lambda_T], np.ones(p))[0]

        # active, inactive, signs
        A = np.flatnonzero(beta_T != 0)
        Ac = np.setdiff1d(np.arange(p), A)
        sA = np.sign(beta_T[A])

        lu, ru, cT, dT = compute_Zu(X0[:, A], X0[:, Ac], a[-nT:], b[-nT:], 
                                    A, sA, Ac, lambda_T, nT)
        right_u = min(ru, z_max)
 

        z_v = z
        while z_v < right_u:
            # Transfer weighted Lasso state; theta(z) = c + d z 
            
            Yz = a + b * z_v
            theta = weighted_lasso(X, Yz, [lambda_0], w)[0]
            L = np.flatnonzero(theta != 0)
            Lc = np.setdiff1d(np.arange(theta.size), L)
            sL = np.sign(theta[L])

            lv, rv, c, d = compute_Zv(
                X[:, L], X[:, Lc], a, b, L, sL, Lc, w, lambda_0, N
            )

        
            Z_uv = (max(lu, lv, z_min), min(ru, rv, z_max))

            cC, dC = c[target_start:], d[target_start:]
            keep = L >= target_start
            C, sC = L[keep] - target_start, sL[keep]

            ZG = compute_ZG(X0, a[-nT:], b[-nT:], cC, dC, lambda_T, nT)
            ZR = compute_ZR(cT, dT, A, sA, cC, dC, C, sC, lambda_T)

            # Transmission where both checks pass, fallback on the rest.
            l_tr = max(Z_uv[0], ZG[0], ZR[0])
            r_tr = min(Z_uv[1], ZG[1], ZR[1])
            transmission = [(l_tr, r_tr)] if l_tr <= r_tr else []
            fallback = (
                [(l, r) for l, r in [(Z_uv[0], l_tr), (r_tr, Z_uv[1])] if l < r]
                if transmission
                else [Z_uv]
            )

            # Both branches can carry M, so the two tests are independent.
            if np.array_equal(C, M):
                intervals += transmission
            if np.array_equal(A, M):
                intervals += fallback

            z_v = max(Z_uv[1], z_v) + 1e-6

        z = max(right_u, z) + 1e-6

    return merge_intervals(intervals)


def fixed_tuning_TM_SI(X0, y0, X_list, y_list, lambda_0, lambda_T, Sigma_0, Sigma_K):
    # Test every feature of the observed model.
    observed = practical_transmission_non_CV(
        X0, y0, X_list, y_list, lambda_0, lambda_T
    )
    M = observed["feature_selection"]

    X, Y = build_tf_design(X0, y0, X_list, y_list)
    w = penalty_weights(X0, X_list)
    Sigma = construct_Sigma(Sigma_0, Sigma_K)
    results = []

    for j in M:
        etaj, etajTY = construct_test_statistic(j, X0, Y, M)
        a, b = calculate_a_b(etaj, Y, Sigma)

        sigma_eta = np.sqrt(etaj @ (Sigma @ etaj))

        intervals = divide_and_conquer(X0, X, a, b, M, w,
                    lambda_0, lambda_T, -20.0 * sigma_eta, 20.0 * sigma_eta,)

        
        results.append({
            "feature": int(j),
            "test_statistic": etajTY,
            "intervals": intervals,
            "p_value": calculate_tn_p_value(intervals, etajTY, etaj, Sigma),
        })

    return {
        "selected_model": M,
        "branch": observed["branch"],
        "results": results,
    }


def fixed_tuning_TM_SI_randj(X0, y0, X_list, y_list, lambda_0, lambda_T, Sigma_0, Sigma_K, rng=None):
    # Test one feature drawn uniformly from the observed model.
    observed = practical_transmission_non_CV(X0, y0, X_list, y_list, lambda_0, lambda_T)
    M = observed["feature_selection"]

    if len(M) == 0:
        results = []
    else:
        X, Y = build_tf_design(X0, y0, X_list, y_list)
        w = penalty_weights(X0, X_list)
        Sigma = construct_Sigma(Sigma_0, Sigma_K)

        rng = np.random.default_rng(rng)
        j = M[rng.integers(len(M))]

        etaj, etajTY = construct_test_statistic(j, X0, Y, M)
        a, b = calculate_a_b(etaj, Y, Sigma)

        sigma_eta = np.sqrt(etaj @ (Sigma @ etaj))

        intervals = divide_and_conquer(X0, X, a, b, M, w,
                    lambda_0, lambda_T, -20.0 * sigma_eta, 20.0 * sigma_eta,)


        results = [{
            "feature": int(j),
            "test_statistic": etajTY,
            "intervals": intervals,
            "p_value": calculate_tn_p_value(intervals, etajTY, etaj, Sigma),
        }]

    return {
        "selected_model": M,
        "branch": observed["branch"],
        "results": results,
    }
