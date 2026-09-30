import numpy as np
from TransMission.transmission import build_tf_design
from TransMission.dtransmission import build_pseudo_source_data, dtransmission_fixed, penalty_weights_D
from SI.TM_SI import divide_and_conquer
from SI.utils import calculate_a_b, calculate_p_value, construct_test_statistic



def setup(X0, y0, beta_tilde_list, ns_list, lambda_0):
    nt, p = X0.shape
    X_pseudo, y_pseudo = build_pseudo_source_data(beta_tilde_list, ns_list, p)
    X_D, Y_D = build_tf_design(X0, y0, X_pseudo, y_pseudo)
    w = penalty_weights_D(ns_list, nt, p)
    N = nt + sum(ns_list)
    lambda_0_eff = lambda_0 * N / X_D.shape[0]
    return X_D, Y_D, w, lambda_0_eff


def fixed_tuning_DTM_SI(X0, y0, beta_tilde_list, ns_list, lambda_0, lambda_T, Sigma_0):
    observed = dtransmission_fixed(X0, y0, beta_tilde_list, ns_list, lambda_0, lambda_T)
    M = observed["feature_selection"]
    nt = X0.shape[0]
    X_D, Y_D, w, lambda_0_eff = setup(X0, y0, beta_tilde_list, ns_list, lambda_0)
    fixed = Y_D[:-nt]   # pseudo-source responses stay fixed; the target block moves
    results = []

    for j in M:
        # Test statistic and line on the target block only.
        etaj, etajTY = construct_test_statistic(j, X0, y0, M)
        a0, b0 = calculate_a_b(etaj, y0, Sigma_0)
        a = np.concatenate([fixed, a0])
        b = np.concatenate([np.zeros(fixed.size), b0])

        sigma_eta = np.sqrt(etaj @ (Sigma_0 @ etaj))
        intervals = divide_and_conquer(X0, X_D, a, b, M, w, lambda_0_eff, lambda_T, -20.0 * sigma_eta, 20.0 * sigma_eta)

        results.append({"feature": int(j), "test_statistic": etajTY, "intervals": intervals, "p_value": calculate_p_value(intervals, etajTY, etaj, Sigma_0)})

    return {
        "selected_model": M,
        "branch": observed["branch"],
        "results": results,}


def fixed_tuning_DTM_SI_randj(X0, y0, beta_tilde_list, ns_list, lambda_0, lambda_T, Sigma_0, rng=None, candidates=None):
    observed = dtransmission_fixed(X0, y0, beta_tilde_list, ns_list, lambda_0, lambda_T)
    M = observed["feature_selection"]
    pool = M if candidates is None else np.intersect1d(M, candidates)

    if len(pool) == 0:
        return {"selected_model": M, "branch": observed["branch"], "results": []}
    
    results = []

    nt = X0.shape[0]
    X_D, Y_D, w, lambda_0_eff = setup(X0, y0, beta_tilde_list, ns_list, lambda_0)
    fixed = Y_D[:-nt]   # pseudo-source responses stay fixed; the target block moves

    rng = np.random.default_rng(rng)
    j = pool[rng.integers(len(pool))]

    # Test statistic and line on the target block only.
    etaj, etajTY = construct_test_statistic(j, X0, y0, M)
    a0, b0 = calculate_a_b(etaj, y0, Sigma_0)
    a = np.concatenate([fixed, a0])
    b = np.concatenate([np.zeros(fixed.size), b0])

    sigma_eta = np.sqrt(etaj @ (Sigma_0 @ etaj))
    intervals = divide_and_conquer(X0, X_D, a, b, M, w, lambda_0_eff, lambda_T, -20.0 * sigma_eta, 20.0 * sigma_eta)

    results = [{"feature": int(j), "test_statistic": etajTY, "intervals": intervals, "p_value": calculate_p_value(intervals, etajTY, etaj, Sigma_0),}]

    return {
        "selected_model": M,
        "branch": observed["branch"],
        "results": results,}
