# Table 2 — Multiscale Gaussian

The [frozen protocol](frozen_protocol.txt) lists the settings, four starting positions and ten paired seed blocks. Each chain has 1,000 warm-up and 10,000 retained draws. The method summary and block results are near the top.

| Manuscript method | Simulation script | Original result report |
|---|---|---|
| HMC | [HMC.py](HMC.py) | [HMC_results.txt](HMC_results.txt) |
| Bounded-scale THC | [THC_bounded.py](THC_bounded.py) | [THC_bounded_results.txt](THC_bounded_results.txt) |

## Reproduce the reports

From this folder, use Python 3.12.4, NumPy 2.2.2, Numba 0.65.1 and llvmlite 0.47.0, the versions recorded in the reports. The original jobs used four chain workers, four Numba threads and one thread per BLAS library. On a POSIX shell, set the thread counts as follows.

```sh
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export PYTHONHASHSEED=0
```

Run the following commands from this folder. The first command checks the software with separate test seeds. The remaining commands run the frozen comparisons and write new reports and checkpoints to `rerun/`. Each method can also run on its own.

```sh
python HMC.py --preflight --workers 4
python HMC.py --run --workers 4 --output-dir rerun
python THC_bounded.py --run --workers 4 --output-dir rerun
```

Choose a fresh output folder for a new run. To continue an interrupted run, add `--resume` to its original command using the same script, software versions and output directory. The scripts preserve existing reports.
