@echo off
rem TPR (power) for n_T = 50, 100, 150, 200.
rem Target support: first S coordinates with signal DELTA (He et al.: S=16, DELTA=0.3).
rem Source contrasts: N(0, (H/50)^2) on 50 random coordinates per source (He et al.: h = 5,...,25).
rem 1. Data splitting: results\tpr_split_results.json (cheap, runs first).
rem 2. Proposed, over-conditioning and naive (same feature in each repetition): results\tpr_results.json.
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
set "S=16"
set "H=10"
set "DELTA=0.3"

cd /d "%~dp0"
"%PY%" run_experiment.py --mode tpr --method split --nts 50 100 150 200 --n_tests %N_TESTS% --n_rep %N_REP_SPLIT% --save_every 100 --S %S% --H %H% --delta %DELTA% --out results\tpr_split_results.json
"%PY%" run_experiment.py --mode tpr --method si --nts 50 100 150 200 --n_tests %N_TESTS% --n_rep %N_REP% --n_jobs %N_JOBS% --save_every %SAVE_EVERY% --S %S% --H %H% --delta %DELTA% --out results\tpr_results.json
pause
