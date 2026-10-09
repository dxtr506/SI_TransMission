import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from TransMission.transmission import build_tf_design, kkt_count, penalty_weights, weighted_lasso
from TransMission.practical_transmission import make_folds, make_grids, practical_transmission_CV
from SI.sub_prob import (compute_B, compute_crossings, compute_cv_loss, compute_J, compute_R, compute_ZG,
                         compute_ZR, compute_Zu, compute_Zv)
from SI.utils import calculate_a_b, calculate_p_value, construct_Sigma, construct_test_statistic, merge_intervals

# Conditioning event {z : M(Y(z)) = M_obs}, decided box by box (sub_prob.py holds the formulas):
#   1. Lasso states   refit, compute_Z        compute_Zu, compute_Zv
#   2. Verification   update_cache, solve_box compute_B, compute_R, compute_ZG, compute_ZR, compute_J
#   3. CV choice      update_cache            compute_cv_loss, compute_crossings
#   4. Output model   partition               feasible set from J, CV argmin, model == M_obs


def construct_setup(X0, y0, X_list, y_list, V=3):
    nt, p = X0.shape
    X, Y = build_tf_design(X0, y0, X_list, y_list)
    n = X.shape[0]
    lambda_0, lambda_T = make_grids()
    w = penalty_weights(X0, X_list)
    folds = make_folds(nt, V)

    target = np.arange(n - nt, n)
    held_out = [(None, np.array([], dtype=int))] + list(enumerate(folds))
    families = []
    for v, Iv in held_out:
        tr = np.setdiff1d(np.arange(nt), Iv)
        families.append({"target": True, "fold": v, "rows": target[tr], "X": X0[tr],
                         "grid": lambda_T, "w": np.ones(p), "block": slice(0, p)})
    for v, Iv in held_out:
        keep = np.setdiff1d(np.arange(n), target[Iv])
        families.append({"target": False, "fold": v, "rows": keep, "X": X[keep],
                         "grid": lambda_0, "w": w, "block": slice(X.shape[1] - p, None)})

    return {"X0": X0, "Y": Y, "folds": folds, "families": families, "lambda_0": lambda_0, "lambda_T": lambda_T}


 
def compute_Z(family, beta, lambda_value, af, bf):
    # KKT interval of one fit; signs and affine path of its target block.
    Xf = family["X"]
    A = np.flatnonzero(beta != 0)
    sA = np.sign(beta[A])
    Ac = np.setdiff1d(np.arange(beta.size), A)
    if family["target"]:
        l, r, c, d = compute_Zu(Xf[:, A], Xf[:, Ac], af, bf, A, sA, Ac, lambda_value, Xf.shape[0])
    else:
        l, r, c, d = compute_Zv(Xf[:, A], Xf[:, Ac], af, bf, A, sA, Ac, family["w"], lambda_value, Xf.shape[0])
    blk = family["block"]
    return l, r, np.sign(beta[blk]), c[blk], d[blk]


def refit(family, state, idx, a, b, probe):
    # Re-solve the fits `idx` of one family at the probe and store their
    # right endpoint, target-block signs and affine path.
    af, bf, grid = a[family["rows"]], b[family["rows"]], family["grid"]

    for i, beta in zip(idx, weighted_lasso(family["X"], af + bf * probe, grid[idx], family["w"])):
        _, hi, s, c, d = compute_Z(family, beta, grid[i], af, bf)
        state["hi"][i] = max(hi, probe)
        state["s"][i], state["c"][i], state["d"][i] = s, c, d


def update_cache(S, cache, family, state, idx, a, b):
    # Recompute only the parts of events 2 and 3 that depend on the fits idx of this family.
    X0 = S["X0"]
    nt = X0.shape[0]
    s, c, d = state["s"][idx], state["c"][idx], state["d"][idx]
    if family["fold"] is None and family["target"]:
        # 2. verification bound B_t of the target Lasso
        cache["B0"][idx], cache["B1"][idx] = compute_B(s, c, d, S["lambda_T"][idx])
    elif family["fold"] is None:
        # 2. verification checks G and R of the transfer candidates
        cache["GL"][idx], cache["GU"][idx] = compute_ZG(X0, a[-nt:], b[-nt:], c, d, S["lambda_T"])
        cache["R0"][idx], cache["R1"][idx] = compute_R(s, c, d)
    else:
        # 3. CV loss L_CV = sum_v part_v / (2 nt) and its crossings with the other lambdas
        key, v = ("T" if family["target"] else "0"), family["fold"]
        Iv = S["folds"][v]
        cache["parts_" + key][v][idx] = compute_cv_loss(X0[Iv], a[-nt:][Iv], b[-nt:][Iv], c, d)
        Q, roots = cache["Q_" + key], cache["roots_" + key]
        Q[idx] = cache["parts_" + key][:, idx].sum(0) / (2.0 * nt)
        for k in idx:
            roots[k] = roots[:, k] = compute_crossings(Q, k)


def partition(lo, hi, J_lo, J_hi, cache, s_T, s_0, M):
    # Cut [lo, hi] where a verification check switches or two CV losses cross,
    # then decide the output on each piece exactly as the pipeline does.
    Q_T, Q_0 = cache["Q_T"], cache["Q_0"]
    cuts = np.concatenate([J_lo.ravel(), J_hi.ravel(),
                           cache["roots_0"].ravel(), cache["roots_T"].ravel()])
    edges = np.concatenate([[lo], np.unique(cuts[(cuts > lo) & (cuts < hi)]), [hi]])

    kept = []
    for left, right in zip(edges[:-1], edges[1:]):
        z = 0.5 * (left + right)
        powers = np.array([z * z, z, 1.0])
        feasible = np.flatnonzero(((J_lo <= z) & (z <= J_hi)).any(axis=1))
        if feasible.size:
            # transmission: first argmin of L_CV over the feasible lambda_0
            model = np.flatnonzero(s_0[feasible[np.argmin(Q_0[feasible] @ powers)]])
        else:
            # fallback: target Lasso at the first argmin of L_CV,T
            model = np.flatnonzero(s_T[np.argmin(Q_T @ powers)])
        if np.array_equal(model, M):
            kept.append((left, right))
    return kept


def solve_box(cache, lo, hi, s_T, s_0, M):
    RL, RU = compute_ZR(cache["R0"], cache["R1"], cache["B0"], cache["B1"])
    J_lo, J_hi = compute_J(cache["GL"], cache["GU"], RL, RU)
    return partition(lo, hi, J_lo, J_hi, cache, s_T, s_0, M)


def init_states_cache(S):
    # states: target full, target folds, transfer full, transfer folds.
    # cache: per-fit quantities, refreshed by update_cache whenever a fit changes state.
    p, V = S["X0"].shape[1], len(S["folds"])
    M0, MT = S["lambda_0"].size, S["lambda_T"].size
    states = [{"hi": np.full(family["grid"].size, -np.inf), "s": np.zeros((family["grid"].size, p)),
               "c": np.zeros((family["grid"].size, p)), "d": np.zeros((family["grid"].size, p))}
              for family in S["families"]]
    cache = {
        "B0": np.zeros(MT), "B1": np.zeros(MT),
        "GL": np.zeros((M0, MT)), "GU": np.zeros((M0, MT)),
        "R0": np.zeros(M0), "R1": np.zeros(M0),
        "parts_T": np.zeros((V, MT, 3)), "parts_0": np.zeros((V, M0, 3)),
        "Q_T": np.zeros((MT, 3)), "Q_0": np.zeros((M0, 3)),
        "roots_T": np.full((MT, MT, 2), np.nan), "roots_0": np.full((M0, M0, 2), np.nan),
    }
    return states, cache


def over_conditioning_interval(S, a, b, z_obs):
    # Interval around z_obs on which every fit keeps its observed active set and signs,
    # every (lambda_0, lambda_T) pair keeps its observed feasibility, and the selected
    # lambda stays the same. Returns the interval and the KKT counts of the fits solved here.
    before = dict(kkt_count)
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    states, cache = init_states_cache(S)

    # 1. Lasso states: intersection of the KKT intervals of all fits at z_obs
    lo, hi = -np.inf, np.inf
    for family, st in zip(S["families"], states):
        af, bf, grid = a[family["rows"]], b[family["rows"]], family["grid"]
        for i, beta in enumerate(weighted_lasso(family["X"], af + bf * z_obs, grid, family["w"])):
            l, r, st["s"][i], st["c"][i], st["d"][i] = compute_Z(family, beta, grid[i], af, bf)
            lo, hi = max(lo, l), min(hi, r)
        update_cache(S, cache, family, st, np.arange(grid.size), a, b)

    # 2. Verification: every pair keeps its observed feasibility
    RL, RU = compute_ZR(cache["R0"], cache["R1"], cache["B0"], cache["B1"])
    J_lo, J_hi = compute_J(cache["GL"], cache["GU"], RL, RU)
    feasible = (J_lo <= z_obs) & (z_obs <= J_hi)
    lo = max(lo, J_lo[feasible].max(initial=-np.inf), J_hi[~feasible & (J_hi < z_obs)].max(initial=-np.inf))
    hi = min(hi, J_hi[feasible].min(initial=np.inf), J_lo[~feasible & (J_lo > z_obs)].min(initial=np.inf))

    # 3. CV choice: the selected lambda keeps the smallest CV loss among its candidates
    rows = np.flatnonzero(feasible.any(axis=1))
    if rows.size:
        Q, roots, candidates = cache["Q_0"], cache["roots_0"], rows
    else:
        Q, roots, candidates = cache["Q_T"], cache["roots_T"], np.arange(S["lambda_T"].size)
    chosen = candidates[np.argmin(Q[candidates] @ np.array([z_obs * z_obs, z_obs, 1.0]))]
    crossings = roots[chosen][candidates].ravel()
    crossings = crossings[np.isfinite(crossings)]
    lo = max(lo, crossings[crossings < z_obs].max(initial=-np.inf))
    hi = min(hi, crossings[crossings > z_obs].min(initial=np.inf))

    return (lo, hi), {k: kkt_count[k] - before[k] for k in kkt_count}


def divide_and_conquer_practical(S, a, b, M, z_min, z_max):
    # Return {z in [z_min, z_max] : M(Y(z)) = M} for the CV-tuned pipeline,
    # and the KKT counts (fits, violations) of the Lasso fits solved here.
    before = dict(kkt_count)
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    M = np.sort(np.asarray(M, dtype=int))
    V = len(S["folds"])
    states, cache = init_states_cache(S)

    intervals = []
    z = float(z_min)
    while z < z_max:
        probe = z + min(0.5 * (z_max - z), 1e-5 * max(1.0, abs(z)))
        for family, st in zip(S["families"], states):
            idx = np.flatnonzero(st["hi"] <= z)
            if idx.size:
                refit(family, st, idx, a, b, probe)
                update_cache(S, cache, family, st, idx, a, b)

        right = min(min(st["hi"].min() for st in states), z_max)
        intervals += solve_box(cache, z, right, states[0]["s"], states[V + 1]["s"], M)
        z = right

    return merge_intervals(intervals), {k: kkt_count[k] - before[k] for k in kkt_count}


def parallel_divide_and_conquer_practical(S, a, b, M, z_min, z_max, n_jobs, n_segments=None):
    # Split [z_min, z_max] into n_segments equal segments (default n_jobs, as in
    # Algorithm 3 of PPL-SI) and traverse them with n_jobs processes.  With more
    # segments than processes, a process that finishes a segment takes the next
    # one, so no process sits idle waiting for the heaviest segment.  Every
    # segment is traversed exactly, so the union is exact.
    if n_segments is None:
        n_segments = n_jobs
    if n_segments == 1:
        return divide_and_conquer_practical(S, a, b, M, z_min, z_max)
    edges = np.linspace(z_min, z_max, n_segments + 1)
    parts = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
        delayed(divide_and_conquer_practical)(S, a, b, M, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
    )
    kkt = {k: sum(counts[k] for _, counts in parts) for k in kkt_count}
    return merge_intervals([iv for part, _ in parts for iv in part]), kkt



def practical_TM_SI(X0, y0, X_list, y_list, Sigma_0, Sigma_K, V=3, n_jobs=1, n_segments=None):
    # Test every feature of the observed model.
    before = dict(kkt_count)
    observed = practical_transmission_CV(X0, y0, X_list, y_list, V=V)
    kkt_observed = {k: kkt_count[k] - before[k] for k in kkt_count}
    M = observed["feature_selection"]
    S = construct_setup(X0, y0, X_list, y_list, V)
    Sigma = construct_Sigma(Sigma_0, Sigma_K)
    results = []

    for j in M:
        etaj, etajTY = construct_test_statistic(j, S["X0"], S["Y"], M)
        a, b = calculate_a_b(etaj, S["Y"], Sigma)

        sigma_eta = np.sqrt(etaj @ (Sigma @ etaj))
        intervals, kkt_si = parallel_divide_and_conquer_practical(S, a, b, M, -20.0 * sigma_eta, 20.0 * sigma_eta,
                                                                  n_jobs, n_segments)

        results.append({"feature": int(j), "test_statistic": etajTY, "intervals": intervals, "p_value": calculate_p_value(intervals, etajTY, etaj, Sigma), "kkt_si": kkt_si})

    return {"selected_model": M, "branch": observed["branch"], "results": results, "kkt_observed": kkt_observed}


def practical_TM_SI_randj(X0, y0, X_list, y_list, Sigma_0, Sigma_K, V=3,
                          rng=None, candidates=None, n_jobs=1, n_segments=None):
    # Test one feature drawn uniformly from the observed model (or from its
    # intersection with `candidates`, e.g. the true support for TPR).
    # The same feature also gets the over-conditioning p-value (everything the pipeline
    # decides is conditioned on, so the region is one interval) and the naive p-value.
    before = dict(kkt_count)
    observed = practical_transmission_CV(X0, y0, X_list, y_list, V=V)
    kkt_observed = {k: kkt_count[k] - before[k] for k in kkt_count}
    M = observed["feature_selection"]
    pool = M if candidates is None else np.intersect1d(M, candidates)

    if len(pool) == 0:
        return {"selected_model": M, "branch": observed["branch"], "results": [], "kkt_observed": kkt_observed}

    S = construct_setup(X0, y0, X_list, y_list, V)
    Sigma = construct_Sigma(Sigma_0, Sigma_K)
    rng = np.random.default_rng(rng)
    j = pool[rng.integers(len(pool))]

    etaj, etajTY = construct_test_statistic(j, S["X0"], S["Y"], M)
    a, b = calculate_a_b(etaj, S["Y"], Sigma)

    sigma_eta = np.sqrt(etaj @ (Sigma @ etaj))
    z_min, z_max = -20.0 * sigma_eta, 20.0 * sigma_eta
    intervals, kkt_si = parallel_divide_and_conquer_practical(S, a, b, M, z_min, z_max, n_jobs, n_segments)

    (lo, hi), kkt_oc = over_conditioning_interval(S, a, b, etajTY)
    interval_oc = (max(lo, z_min), min(hi, z_max))

    results = [{"feature": int(j), "test_statistic": etajTY, "intervals": intervals,
                "p_value": calculate_p_value(intervals, etajTY, etaj, Sigma), "kkt_si": kkt_si,
                "interval_oc": interval_oc, "p_value_oc": calculate_p_value([interval_oc], etajTY, etaj, Sigma),
                "kkt_oc": kkt_oc, "p_value_naive": float(2.0 * norm.sf(abs(etajTY) / sigma_eta))}]

    return {"selected_model": M, "branch": observed["branch"], "results": results, "kkt_observed": kkt_observed}


def data_split_randj(X0, y0, X_list, y_list, Sigma_0, V=3, rng=None, candidates=None):
    # Data splitting: run the pipeline on the first half of the target with all sources,
    # then test one selected feature on the second half by least squares (z-test, known Sigma_0).
    nt = X0.shape[0]
    first, second = np.arange(nt // 2), np.arange(nt // 2, nt)
    observed = practical_transmission_CV(X0[first], y0[first], X_list, y_list, V=V)
    M = observed["feature_selection"]
    pool = M if candidates is None else np.intersect1d(M, candidates)
    out = {"selected_model": M, "branch": observed["branch"], "status": "tested", "feature": None, "p_value": None}
    if len(pool) == 0:
        out["status"] = "empty"
        return out
    if M.size >= second.size:
        # least squares on the second half is not identifiable
        out["status"] = "too_large"
        return out

    rng = np.random.default_rng(rng)
    j = pool[rng.integers(len(pool))]
    eta, z = construct_test_statistic(j, X0[second], y0[second], M)
    sd = np.sqrt(eta @ (Sigma_0[np.ix_(second, second)] @ eta))
    out["feature"], out["p_value"] = int(j), float(2.0 * norm.sf(abs(z) / sd))
    return out
