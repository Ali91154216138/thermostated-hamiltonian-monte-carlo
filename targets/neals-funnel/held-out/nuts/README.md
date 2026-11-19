# Table 19 — Neal's funnel

The [frozen protocol](frozen_protocol.txt) lists the settings, four starting positions and ten paired seed blocks. Each chain has 1,000 warm-up and 10,000 retained draws. Each method report identifies its script and protocol by SHA-256. The method summary and block results are near the top.

| Manuscript method | Simulation script | Original result report |
|---|---|---|
| Original slice-NUTS | [NUTS.py](NUTS.py) | [NUTS_results.txt](NUTS_results.txt) |
| Scalar bounded-scale THC | [Scalar_THC.py](Scalar_THC.py) | [Scalar_THC_results.txt](Scalar_THC_results.txt) |
| Periodic THC | [Periodic_THC.py](Periodic_THC.py) | [Periodic_THC_results.txt](Periodic_THC_results.txt) |
| Conditional THC | [Conditional_THC.py](Conditional_THC.py) | [Conditional_THC_results.txt](Conditional_THC_results.txt) |

The methods in this folder use the same held-out blocks. The other comparator in the manuscript uses a separate set of held-out runs.

## Reproduce the reports

From this folder, use Python 3.12.4, NumPy 2.2.2, Numba 0.65.1 and llvmlite 0.47.0, the versions recorded in the reports. The original jobs used four chain workers, four Numba threads and one thread per BLAS library. On a POSIX shell, set the thread counts as follows.

```sh
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export PYTHONHASHSEED=0
```

Run the following commands from this folder. The first command checks the software with separate test seeds. The remaining commands run the frozen comparisons and write new reports and checkpoints to `rerun/`. Each method can also run on its own.

```sh
python NUTS.py --preflight --workers 4
python NUTS.py --run --workers 4 --output-dir rerun
python Scalar_THC.py --run --workers 4 --output-dir rerun
python Periodic_THC.py --run --workers 4 --output-dir rerun
python Conditional_THC.py --run --workers 4 --output-dir rerun
```

Choose a fresh output folder for a new run. To continue an interrupted run, add `--resume` to its original command using the same script, software versions and output directory. The scripts preserve existing reports.
