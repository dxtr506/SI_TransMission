import argparse
import json
import os
import time

import numpy as np
from scipy import stats

from data_generation import generate_contrasts, generate_design, generate_response
from SI.Practical_TM_SI import data_split_randj, practical_TM_SI_randj
from TransMission.practical_transmission import make_grids

# p-value field of each method in a repetition record, for each run type
METHODS = {"si": {"proposed": "p_value", "over_conditioning": "p_value_oc", "naive": "p_value_naive"},
           "split": {"data_split": "p_value"}}


def summarize(reps, mode, method, alpha):
    summary = {
        "n_reps": len(reps),
        "n_not_tested": sum(not r["tested"] for r in reps),
        "n_invalid": sum(r["invalid"] for r in reps),
        "mean_model_size": float(np.mean([r["model_size"] for r in reps])) if reps else None,
    }
    if method == "split":
        summary["status"] = {s: sum(r["status"] == s for r in reps) for s in ("tested", "empty", "too_large")}
    for name, key in METHODS[method].items():
        p_values = np.array([r[key] for r in reps if r[key] is not None and not r["invalid"]])
        s = {"n_tests": int(p_values.size)}
        if p_values.size:
            rate = float(np.mean(p_values <= alpha))
            s["rejection_rate"] = rate
            s["rejection_rate_se"] = float(np.sqrt(rate * (1 - rate) / p_values.size))
            if mode == "fpr":
                statistic, p_ks = stats.kstest(p_values, "uniform")
                s["ks_statistic"], s["ks_p_value"] = float(statistic), float(p_ks)
        summary[name] = s
    return summary


def save(path, config, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=1)
    os.replace(tmp, path)


def run_si(args, X0, X_list, y0, y_list, Sigma_0, Sigma_K, rng, candidates, i, kkt, entry):
    # Proposed, over-conditioning and naive p-values of one repetition (same feature).
    result = practical_TM_SI_randj(X0, y0, X_list, y_list, Sigma_0, Sigma_K,
                                   V=args.V, rng=rng, candidates=candidates, n_jobs=args.n_jobs)
    kkt["observed_fits"] += result["kkt_observed"]["fits"]
    kkt["observed_violations"] += result["kkt_observed"]["violations"]
    rep = {"rep": i, "model_size": len(result["selected_model"]), "branch": result["branch"],
           "tested": bool(result["results"]), "feature": None, "p_value": None, "invalid": False,
           "p_value_oc": None, "interval_oc": None, "p_value_naive": None}
    if result["results"]:
        r = result["results"][0]
        p_value = float(r["p_value"])
        rep["feature"], rep["p_value"] = r["feature"], p_value
        rep["invalid"] = not (np.isfinite(p_value) and 0.0 <= p_value <= 1.0)
        rep["p_value_oc"], rep["interval_oc"] = r["p_value_oc"], list(r["interval_oc"])
        rep["p_value_naive"] = r["p_value_naive"]
        kkt["si_fits"] += r["kkt_si"]["fits"]
        kkt["si_violations"] += r["kkt_si"]["violations"]
        kkt["oc_fits"] += r["kkt_oc"]["fits"]
        kkt["oc_violations"] += r["kkt_oc"]["violations"]
        if r["kkt_si"]["violations"]:
            entry["kkt_si_violation_reps"].append(i)
    return rep


def run_split(args, X0, X_list, y0, y_list, Sigma_0, rng, candidates, i):
    # Data splitting p-value of one repetition.
    split = data_split_randj(X0, y0, X_list, y_list, Sigma_0, V=args.V, rng=rng, candidates=candidates)
    return {"rep": i, "model_size": len(split["selected_model"]), "branch": split["branch"],
            "status": split["status"], "tested": split["status"] == "tested",
            "feature": split["feature"], "p_value": split["p_value"], "invalid": False}


def run_one(args, nt, config, results):
    rng = np.random.default_rng(args.seed + nt)
    X0, X_list = generate_design(rng, args.p, nt, args.ns, args.K)
    beta0 = np.zeros(args.p)
    contrasts, candidates = None, None
    if args.mode == "tpr":
        contrasts = generate_contrasts(rng, args.p, args.K, args.H)
        candidates = np.arange(args.S)
        beta0[candidates] = args.delta
    Sigma_0 = args.sigma ** 2 * np.eye(nt)
    Sigma_K = [args.sigma ** 2 * np.eye(args.ns) for _ in range(args.K)]

    kkt = {"observed_fits": 0, "observed_violations": 0, "si_fits": 0, "si_violations": 0,
           "oc_fits": 0, "oc_violations": 0}
    entry = {"reps": [], "kkt": kkt, "kkt_si_violation_reps": []}
    results[str(nt)] = entry
    start = time.perf_counter()

    for i in range(1, args.n_rep + 1):
        y0, y_list = generate_response(rng, X0, X_list, args.sigma, beta0, contrasts)
        t = time.perf_counter()
        if args.method == "si":
            rep = run_si(args, X0, X_list, y0, y_list, Sigma_0, Sigma_K, rng, candidates, i, kkt, entry)
        else:
            rep = run_split(args, X0, X_list, y0, y_list, Sigma_0, rng, candidates, i)
        rep["seconds"] = time.perf_counter() - t
        entry["reps"].append(rep)
        n_tests = sum(r["p_value"] is not None and not r["invalid"] for r in entry["reps"])
        done = args.n_tests is not None and n_tests >= args.n_tests

        if i % args.save_every == 0 or i == args.n_rep or done:
            entry["summary"] = summarize(entry["reps"], args.mode, args.method, args.alpha)
            entry["seconds_total"] = time.perf_counter() - start
            save(args.out, config, results)
            s = entry["summary"]
            rates = " ".join(f"{m}={s[m]['n_tests']}:{s[m].get('rejection_rate')}" for m in METHODS[args.method])
            print(f"[{args.mode}/{args.method}] n_T={nt} {i}/{args.n_rep} not_tested={s['n_not_tested']} "
                  f"tests:rate {rates} time={entry['seconds_total']:.0f}s "
                  f"KKT observed {kkt['observed_violations']}/{kkt['observed_fits']} "
                  f"SI {kkt['si_violations']}/{kkt['si_fits']} OC {kkt['oc_violations']}/{kkt['oc_fits']}",
                  flush=True)
        if done:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fpr", "tpr"], required=True)
    parser.add_argument("--method", choices=["si", "split"], default="si",
                        help="si: proposed, over-conditioning and naive; split: data splitting")
    parser.add_argument("--nts", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--n_rep", type=int, default=500, help="maximum number of repetitions per n_T")
    parser.add_argument("--n_tests", type=int, default=None,
                        help="stop an n_T once the method (proposed, or data splitting) has this many p-values")
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--ns", type=int, default=200)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--V", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--S", type=int, default=16, help="tpr: target support size")
    parser.add_argument("--H", type=float, default=10.0, help="tpr: source contrast size")
    parser.add_argument("--delta", type=float, default=0.3, help="tpr: signal strength")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=185)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lambda_0, lambda_T = make_grids()
    config = {**vars(args), "lambda_0_grid": lambda_0.tolist(), "lambda_T_grid": lambda_T.tolist()}
    results = {}
    for nt in args.nts:
        run_one(args, nt, config, results)


if __name__ == "__main__":
    main()
