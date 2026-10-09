@echo off
rem FPR (pivot under the global null) for n_T = 50, 100, 150, 200.
rem 1. Data splitting: results\fpr_split_results.json (cheap, runs first).
rem 2. Proposed, over-conditioning and naive (same feature in each repetition): results\fpr_results.json.
rem Each n_T stops once the method has N_TESTS p-values, or after N_REP (SI) / N_REP_SPLIT repetitions.
rem Results are saved every SAVE_EVERY repetitions.
rem If python is not on PATH, set PY to its full path, e.g. C:\Ananconda\python.exe
setlocal
set "PY=python"
set "N_JOBS=16"
set "N_TESTS=200"
set "N_REP=3000"
set "N_REP_SPLIT=20000"
set "SAVE_EVERY=10"

cd /d "%~dp0"
"%PY%" run_experiment.py --mode fpr --method split --nts 50 100 150 200 --n_tests %N_TESTS% --n_rep %N_REP_SPLIT% --save_every 100 --out results\fpr_split_results.json
"%PY%" run_experiment.py --mode fpr --method si --nts 50 100 150 200 --n_tests %N_TESTS% --n_rep %N_REP% --n_jobs %N_JOBS% --save_every %SAVE_EVERY% --out results\fpr_results.json
pause
