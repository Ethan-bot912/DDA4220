@echo off
REM ===== seed42 SAME-BOX A/B/C baseline probe (2 epochs each) =====
REM Purpose: give every strategy the SAME start point on THIS machine (i7-1165G7, 4C/8T).
REM All three use identical training args (seed 42, resnet18, batch 32, lr 1e-3,
REM crop 128, context 16, manifest pilot_tier3_4events). Only the I/O path differs.
REM
REM   A = workers 0, no cache   <- teammate's 810.8 s baseline config
REM   B = workers 4, no cache   <- teammate's speedup (his method)
REM   C = workers 4, cache      <- my method (cache) on his worker setting
REM
REM NOTE: epochs are capped at 2 via --max-train? NO -- that would change data.
REM Instead all three run 2 full epochs on the full 17,552-row train split.
REM Epoch 1-2 have the largest LR (no decay effect yet), so this is slightly
REM conservative but identically so for all three arms.

set ROOT=D:\dda4220-project\DDA4220-main
cd /d %ROOT%

set PY=%ROOT%\.venv\Scripts\python.exe
set MAN=%ROOT%\dataset\pilot_tier3_4events\manifest_classification.csv
set CACHE=%ROOT%\cache\pilot_tier3_4events
set R=%ROOT%\reports\compare_seed42\probe
set LOG=%R%\log

if not exist "%LOG%" mkdir "%LOG%"

set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

echo ===== SEED42 PROBE START %DATE% %TIME% ===== > "%LOG%\probe.log"

echo [A] workers0 nocache >> "%LOG%\probe.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --output "%R%\A_post_w0_nocache.json" ^
  --project-root "%ROOT%" ^
  --mode post --backbone resnet18 --epochs 2 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 0 ^
  --seed 42 --device cpu --benchmark-steps 2 --benchmark-warmup 1 ^
  >> "%LOG%\A_post_w0_nocache.log" 2>&1
echo [A] done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\probe.log"

echo [B] workers4 nocache >> "%LOG%\probe.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --output "%R%\B_post_w4_nocache.json" ^
  --project-root "%ROOT%" ^
  --mode post --backbone resnet18 --epochs 2 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 4 ^
  --seed 42 --device cpu --benchmark-steps 2 --benchmark-warmup 1 ^
  >> "%LOG%\B_post_w4_nocache.log" 2>&1
echo [B] done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\probe.log"

echo [C] workers4 cache >> "%LOG%\probe.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --cache-dir "%CACHE%" ^
  --output "%R%\C_post_w4_cache.json" ^
  --project-root "%ROOT%" ^
  --mode post --backbone resnet18 --epochs 2 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 4 ^
  --seed 42 --device cpu --benchmark-steps 2 --benchmark-warmup 1 ^
  >> "%LOG%\C_post_w4_cache.log" 2>&1
echo [C] done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\probe.log"

echo [D] workers8 cache >> "%LOG%\probe.log"
%PY% -m stage_b_damage.train_baseline ^
  --manifest "%MAN%" ^
  --cache-dir "%CACHE%" ^
  --output "%R%\D_post_w8_cache.json" ^
  --project-root "%ROOT%" ^
  --mode post --backbone resnet18 --epochs 2 --batch-size 32 ^
  --lr 0.001 --crop-size 128 --context 16 --workers 8 ^
  --seed 42 --device cpu --benchmark-steps 2 --benchmark-warmup 1 ^
  >> "%LOG%\D_post_w8_cache.log" 2>&1
echo [D] done exit=%ERRORLEVEL% at %TIME% >> "%LOG%\probe.log"

echo ===== END %DATE% %TIME% ===== >> "%LOG%\probe.log"
