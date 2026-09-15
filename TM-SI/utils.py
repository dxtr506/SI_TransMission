import numpy as np
from scipy.linalg import block_diag
from mpmath import mp
mp.dps = 500


def construct_Sigma(Sigma_0, Sigma_K):
    Sigma = block_diag(*Sigma_K, Sigma_0)
    return Sigma

def construct_test_statistic(j, X0, Y, M):
    X0 = np.asarray(X0, dtype=float)
    Y = np.ravel(Y).astype(float)
    M = np.asarray(M, dtype=int)

    X0_M = X0[:, M]

    # index j in selected model M
    position_j = np.flatnonzero(M == int(j))

    e_j = np.zeros(M.size)
    e_j[position_j[0]] = 1.0

    # target
    etaj_target = X0_M @ np.linalg.solve(X0_M.T @ X0_M, e_j)

    etaj = np.zeros(Y.size)

    # source (=0) + target
    etaj[-X0.shape[0]:] = etaj_target
    etajTY = float(etaj @ Y)
    return etaj, etajTY


def calculate_a_b(etaj, Y, Sigma):  
    etaj = np.asarray(etaj, dtype=float).reshape(-1, 1)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma, dtype=float)

    # eta.T Sigma eta
    e1 = (etaj.T @ Sigma @ etaj).item()
    # Sigma eta / (eta.T Sigma eta)
    b = (Sigma @ etaj) / e1

    # (I - b eta.T)Y
    # e2 = np.eye(len(Y)) - b @ etaj.T
    # a = e2 @ Y

    a = Y - b * (etaj.T @ Y).item()

    return a.ravel(), b.ravel()
    


def merge_intervals(intervals, tol=1e-10):
    valid_intervals = []
    for left, right in intervals:
        left, right = float(left), float(right)
        if np.isnan(left) or np.isnan(right):
            raise ValueError("Interval endpoints must not be NaN")
        if left > right:
            continue
        valid_intervals.append((left, right))

    # sort theo left endpoint
    intervals = sorted(valid_intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] - merged[-1][1] > tol:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    return merged

def pivot(intervals, etajTy, etaj, Sigma, tn_mu=0):
    intervals = merge_intervals(intervals)

    etaj = etaj.ravel()
    stdev = np.sqrt(etaj @ (Sigma @ etaj))

    numerator = mp.mpf('0')
    denominator = mp.mpf('0')

    for (left, right) in intervals:
        cdf_left = mp.ncdf((left - tn_mu)/ stdev)
        cdf_right = mp.ncdf((right - tn_mu)/ stdev)
        piece = cdf_right - cdf_left
        denominator += piece

        if etajTy >= right:
            numerator += piece
        elif left <= etajTy < right:
            numerator += mp.ncdf((etajTy - tn_mu)/ stdev) - cdf_left

    if denominator <= 0:
        raise ValueError("The truncation region has zero Gaussian probability")
    return float(numerator/ denominator)


def calculate_tn_p_value(intervals, etajTy, etaj, Sigma, tn_mu = 0):
    cdf = pivot(intervals, etajTy, etaj, Sigma, tn_mu)
    return float(2.0 * min(cdf, 1.0 - cdf))
