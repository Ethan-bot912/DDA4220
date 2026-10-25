@echo off
REM ===== R5: FULL 20-epoch four-event B1+B2 with STACKED speedup (C config) =====
REM Config C = workers=4 + crop cache (measured fastest on this 4-core box, 1.78x).
REM Published git baseline for comparison = workers=0, no cache, same seed42.
REM Published: post 15,705.3 s / best_ep 5 / F1 0.601235705274549
REM            early 15,628.7 s / best_ep 9 / F1 0.6022778097528944
REM Cache verified bit-identical (ep1/ep2 loss+F1 diff = 0.0 vs uncached).

set ROOT=D:\dda4220-project\DDA4220-main
cd /d %ROOT%

set PY=%ROOT%\.venv\Scripts\python.exe
set MAN=%ROOT%\dataset\pilot_tier3_4events\manifest_classification.csv
set CACHE=%ROOT%\cache\pilot_tier3_4events
set OUT=%ROOT%\output\resnet18
set LOG=%ROOT%\reports\compare_seed42\full20\log

if not exist "%LOG%" mkdir "%LOG%"

set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

echo ===== R5 C full20 seed42 START %DATE% %TIME% ===== > "%LOG%\run_r5_20ep.log"

echo [1/2] B1 post w4+cache 20ep >> "%LOG%\run_r5_20ep.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --cache-dir "%CACHE%" ^
  --output "%OUT%\post_seed42_CACHEW4_20ep.json" ^
  --project-root "%ROOT%" ^
  --mode post --backbone resnet18 --epochs 20 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 4 ^
  --seed 42 --device cpu --benchmark-steps 20 --benchmark-warmup 5 ^
  >> "%LOG%\post_seed42_CACHEW4_20ep.log" 2>&1
echo [1/2] B1 done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\run_r5_20ep.log"

echo [2/2] B2 early w4+cache 20ep >> "%LOG%\run_r5_20ep.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --cache-dir "%CACHE%" ^
  --output "%OUT%\early_seed42_CACHEW4_20ep.json" ^
  --project-root "%ROOT%" ^
  --mode early --backbone resnet18 --epochs 20 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 4 ^
  --seed 42 --device cpu --benchmark-steps 20 --benchmark-warmup 5 ^
  >> "%LOG%\early_seed42_CACHEW4_20ep.log" 2>&1
echo [2/2] B2 done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\run_r5_20ep.log"

echo ===== END %DATE% %TIME% ===== >> "%LOG%\run_r5_20ep.log"
