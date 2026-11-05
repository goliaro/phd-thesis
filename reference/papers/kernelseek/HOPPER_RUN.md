# Hopper benchmark run (fa3_hopper.cu vs FA3 / cuDNN / FlashInfer)

Goal: one full run of the Hopper suite with the cuDNN and FlashInfer columns enabled, and the
result files it writes.

**Hardware:** one idle H200 or H100 (sm90). More GPUs are used automatically if visible.
**Software:** CUDA toolkit on `PATH` (`nvcc`; ≥ 12.3 needed, 13 lets `setup.sh` run unmodified), Python 3.10+, a fresh venv.

```bash
git clone <flash-cuda repo url> && cd flash-cuda
git submodule update --init reference/cutlass reference/flash-attention
python -m venv .venv && source .venv/bin/activate
./setup.sh --with-vllm          # torch + vLLM (FA3 baseline) + cuDNN frontend + FlashInfer
python -m bench.eval --target hopper --set all 2>&1 | tee hopper_all.log
```

Takes roughly 30–60 min. `--set all` is 182 scenario definitions; about 105 apply on Hopper, the rest
are reported as unsupported, which is expected. If `setup.sh` fails on the Blackwell-only FA4/CuteDSL
step (CUDA 12 box), skip it: `pip install torch vllm nvidia-cudnn-frontend flashinfer-python matplotlib`
and run the last command.

**Check** the end of the log for lines like

```
correct: 105/105 ...
perf over 105 configs:  geomean=1.1xx
cuDNN over NN configs: ...
FlashInfer over NN configs: ...
```

If either of the last two says "disabled" or "not importable", that package did not install; fix it
and rerun (the run is deterministic, nothing else is needed).

**Send back**

- `hopper_all.log`
- `docs/last_results_hopper.json`
- `docs/results_hopper_*.csv`
- `docs/perf_fa3_hopper*.png`
- `nvidia-smi -L` output and `python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"`

Thank you!
