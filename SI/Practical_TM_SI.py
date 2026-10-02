import numpy as np
from joblib import Parallel, delayed

from TransMission.transmission import build_tf_design, kkt_count, penalty_weights, weighted_lasso
from TransMission.practical_transmission import make_folds, make_grids, practical_transmission_CV
from SI.sub_prob import compute_Zu, compute_Zv
from SI.utils import calculate_a_b, calculate_p_value, construct_Sigma, construct_test_statistic, merge_intervals


def construct_setup(X0, y0, X_list, y_list, V=3):
    nt, p = X0.shape
    X, Y = build_tf_design(X0, y0, X_list, y_list)
    n = X.shape[0]
    lambda_0, lambda_T = make_grids(X0, X_list)
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


def refit(family, state, idx, a, b, z, probe):
    # Re-solve the fits `idx` of one family at the probe and store their
    # right endpoint, target-block signs and affine path.
    af, bf, grid = a[family["rows"]], b[family["rows"]], family["grid"]

    for i, beta in zip(idx, weighted_lasso(family["X"], af + bf * probe, grid[idx], family["w"])):
        lo, hi, s, c, d = compute_Z(family, beta, grid[i], af, bf)
        state["hi"][i] = max(hi, probe)
        state["s"][i], state["c"][i], state["d"][i] = s, c, d


def compute_B(S, cache, state, idx):
    # Target full fits: B_t(z) = ||beta_T,t(z)||_1 + |A_t| lambda_T[t] = B0 + B1 z.
    s = state["s"][idx]
    cache["B0"][idx] = (s * state["c"][idx]).sum(1) + (s != 0).sum(1) * S["lambda_T"][idx]
    cache["B1"][idx] = (s * state["d"][idx]).sum(1)


def compute_ZG_grid(S, cache, state, idx, a, b):
    # Transfer full fits: beta_i(z) = c_i + d_i z gives g_i(z) = G0 + G1 z and
    # ||beta_i(z)||_1 = R0 + R1 z.  [GL, GU][i, t] = {z : ||g_i(z)||_inf <= lambda_T[t]}
    # holds on the whole piece of fit i, so it is computed once per piece.
    X0, lambda_T = S["X0"], S["lambda_T"]
    nt = X0.shape[0]
    s, c, d = state["s"][idx], state["c"][idx], state["d"][idx]
    G0 = (a[-nt:] - c @ X0.T) @ X0 / nt
    G1 = (b[-nt:] - d @ X0.T) @ X0 / nt

    # -lambda_T[t] <= G0 + G1 z <= lambda_T[t] on every coordinate
    lambda_T3 = lambda_T[None, :, None]   # shape (1, M_T, 1) for broadcasting
    g0, g1 = G0[:, None, :], G1[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        up, dn = (lambda_T3 - g0) / g1, (-lambda_T3 - g0) / g1
    upper = np.where(g1 > 0, up, np.where(g1 < 0, dn, np.inf)).min(axis=2)
    lower = np.where(g1 > 0, dn, np.where(g1 < 0, up, -np.inf)).max(axis=2)
    never = ((g1 == 0) & (np.abs(g0) > lambda_T3)).any(axis=2)
    lower[never], upper[never] = np.inf, -np.inf

    cache["GL"][idx], cache["GU"][idx] = lower, upper
    cache["R0"][idx], cache["R1"][idx] = (s * c).sum(1), (s * d).sum(1)


def compute_cv_loss(S, part, state, idx, a, b, v):
    # Fold fits: ||y0[I_v](z) - X0[I_v] beta_{-v}(z)||^2 as (z^2, z, 1) coefficients.
    X0 = S["X0"]
    nt = X0.shape[0]
    Iv = S["folds"][v]
    E0 = a[-nt:][Iv] - state["c"][idx] @ X0[Iv].T
    E1 = b[-nt:][Iv] - state["d"][idx] @ X0[Iv].T
    part[idx] = np.column_stack([(E1 * E1).sum(1), 2.0 * (E0 * E1).sum(1), (E0 * E0).sum(1)])


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


def update_cv_loss(S, cache, key, idx):
    # L_CV = sum_v part_v / (2 nt); refresh rows idx and their crossings with all others.
    nt = S["X0"].shape[0]
    Q, roots = cache["Q_" + key], cache["roots_" + key]
    Q[idx] = cache["parts_" + key][:, idx].sum(0) / (2.0 * nt)
    for k in idx:
        roots[k] = roots[:, k] = compute_crossings(Q, k)


def update_cache(S, cache, family, state, idx, a, b):
    # Recompute only what depends on the fits idx of this family.
    if family["fold"] is None:
        if family["target"]:
            compute_B(S, cache, state, idx)
        else:
            compute_ZG_grid(S, cache, state, idx, a, b)
    else:
        key = "T" if family["target"] else "0"
        compute_cv_loss(S, cache["parts_" + key][family["fold"]], state, idx, a, b, family["fold"])
        update_cv_loss(S, cache, key, idx)


def compute_J(cache):
    # J[i, t] = [GL, GU][i, t] ∩ {z : R0_i + R1_i z <= B0_t + B1_t z},
    # one interval per (lambda_0[i], lambda_T[t]).
    coef = cache["R1"][:, None] - cache["B1"][None, :]
    rhs = cache["B0"][None, :] - cache["R0"][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rhs / coef
    upper = np.minimum(cache["GU"], np.where(coef > 0, ratio, np.inf))
    lower = np.maximum(cache["GL"], np.where(coef < 0, ratio, -np.inf))
    empty = ((coef == 0) & (rhs < 0)) | (lower > upper)
    lower[empty], upper[empty] = np.inf, -np.inf
    return lower, upper


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
    J_lo, J_hi = compute_J(cache)
    return partition(lo, hi, J_lo, J_hi, cache, s_T, s_0, M)



def divide_and_conquer_practical(S, a, b, M, z_min, z_max):
    # Return {z in [z_min, z_max] : M(Y(z)) = M} for the CV-tuned pipeline,
    # and the KKT counts (fits, violations) of the Lasso fits solved here.
    before = dict(kkt_count)
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    M = np.sort(np.asarray(M, dtype=int))

    p, V = S["X0"].shape[1], len(S["folds"])
    M0, MT = S["lambda_0"].size, S["lambda_T"].size
    # states: target full, target folds, transfer full, transfer folds.
    states = [{"hi": np.full(family["grid"].size, -np.inf), "s": np.zeros((family["grid"].size, p)),
               "c": np.zeros((family["grid"].size, p)), "d": np.zeros((family["grid"].size, p))}
              for family in S["families"]]
    # Per-fit quantities, refreshed by update_cache whenever a fit changes state.
    cache = {
        "B0": np.zeros(MT), "B1": np.zeros(MT),
        "GL": np.zeros((M0, MT)), "GU": np.zeros((M0, MT)),
        "R0": np.zeros(M0), "R1": np.zeros(M0),
        "parts_T": np.zeros((V, MT, 3)), "parts_0": np.zeros((V, M0, 3)),
        "Q_T": np.zeros((MT, 3)), "Q_0": np.zeros((M0, 3)),
        "roots_T": np.full((MT, MT, 2), np.nan), "roots_0": np.full((M0, M0, 2), np.nan),
    }

    intervals = []
    z = float(z_min)
    while z < z_max:
        probe = z + min(0.5 * (z_max - z), 1e-5 * max(1.0, abs(z)))
        for family, st in zip(S["families"], states):
            idx = np.flatnonzero(st["hi"] <= z)
            if idx.size:
                refit(family, st, idx, a, b, z, probe)
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
    intervals, kkt_si = parallel_divide_and_conquer_practical(S, a, b, M, -20.0 * sigma_eta, 20.0 * sigma_eta,
                                                              n_jobs, n_segments)

    results = [{"feature": int(j), "test_statistic": etajTY, "intervals": intervals, "p_value": calculate_p_value(intervals, etajTY, etaj, Sigma), "kkt_si": kkt_si}]

    return {"selected_model": M, "branch": observed["branch"], "results": results, "kkt_observed": kkt_observed}
