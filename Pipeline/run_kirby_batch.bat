@echo off
echo === Kirby batch run: k_v1 (10M steps) ===
call conda run -n kirby_env python -m kirby_pipeline.train --variant k_v1 --config configs/kirby/k_v1_10m.yaml
if errorlevel 1 goto end
echo === Kirby batch run: k_v2 (10M steps) ===
call conda run -n kirby_env python -m kirby_pipeline.train --variant k_v2 --config configs/kirby/k_v2_10m.yaml
if errorlevel 1 goto end
echo === Kirby batch run: k_v3 (10M steps) ===
call conda run -n kirby_env python -m kirby_pipeline.train --variant k_v3 --config configs/kirby/k_v3_10m.yaml
:end
echo === Batch run finished ===
