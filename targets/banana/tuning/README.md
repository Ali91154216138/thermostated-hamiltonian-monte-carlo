# Banana tuning

Use this page for the tuning tables. Each method links to its Python program, recorded settings and winning result. A shared tuning result is stored once. The columns below count the configurations with a defined score and all configurations evaluated.

| Table(s) | Method | Defined / tested | Winning score |
|---|---|---:|---:|
| 7 | [HMC](#method-hmc) | 98 / 143 | 0.004123623805 |
| 7 | [Bounded-scale THC](#method-thc_bounded) | 49,518 / 81,081 | 0.00498903126 |
| 7 | [Conditional-shear THC-v4](#method-thc_v4) | 31,259 / 45,278 | 0.04117917867 |
| 9, 11 | [Periodic THC-v2](#method-thc_v2) | 39,824 / 41,533 | 0.00550555239 |
| 9 | [S-QHMC](#method-s_qhmc) | 14,409 / 16,731 | 0.005327753493 |
| 11 | [Original slice-NUTS](#method-nuts) | 35 / 35 | 0.002595566072 |

[Winning settings and scores in JSON](winners.json) · [Full data archive](../../../data/README.md)

The recorded campaigns used ten tuning blocks, four chains per block, and 1,000 warm-up plus 10,000 retained samples per chain. Scores summarize worst-observable efficiency across the ten blocks. The settings in the manifests give the complete grid and seed schedule. The JSON includes ancillary settings from the frozen evaluation protocol.

The paths labelled “archive report” below are members of `THC_tuning_Banana_20261007.zip`. From the repository root, the following command extracts and checks the tuning winners from that ZIP without running a sampler. It uses the Python 3.11+ standard library.

```sh
python tuning-code/extract_winners.py /path/to/THC_tuning_Banana_20261007.zip
```

For numerical reruns, use a separate output directory and the production Python environment described in the archived run records. Commands below assume Linux and are run at the repository root. The original Slurm files include cluster-specific project paths.

<a id="method-hmc"></a>

### HMC

**Table 7** · [Sampler](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_hmc_EXPANDED_20260925.py) · [Settings and seed record](records/HMC/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `L=80`, `epsilon=0.04`.

[Original final report](records/HMC/HMC_Banana_expanded_tuning_20260925.txt). Full report in archive: `Banana/HMC/results/HMC_Banana_expanded_tuning_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/launcher/checkpoint_runner.py --method hmc --source tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_hmc_EXPANDED_20260925.py --run-dir runs/banana/HMC
```

<a id="method-thc_bounded"></a>

### Bounded-scale THC

**Table 7** · [Sampler](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_bounded_EXPANDED_20260925.py) · [Settings and seed record](records/THC_bounded/FROZEN.txt) · [Run launcher](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `lambda_value=0.5596157879354227`, `a=4.0`, `Q=0.5`, `epsilon=0.05`, `L=60`.

[Original final report](records/THC_bounded/FINAL_TUNING_REPORT.txt). Full report in archive: `Banana/Bounded_scale_THC/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/THC_bounded/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 62585` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py banana THC_bounded --part 0 --configuration 62585
```

<a id="method-thc_v4"></a>

### Conditional-shear THC-v4

**Table 7** · [Sampler](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_v4_EXPANDED_20260925.py) · [Settings and seed record](records/THC_v4/FROZEN.txt) · [Run launcher](../../../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `kappa=0.5`, `tau=0.5`, `Q=0.1`, `a=8.0`, `epsilon=0.1`, `L=20`.

[Original final report](records/THC_v4/FINAL_TUNING_REPORT.txt). Full report in archive: `Banana/Conditional_shear_THC_v4/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/THC_v4/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 120673` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py banana THC_v4 --part 0 --configuration 120673
```

<a id="method-thc_v2"></a>

### Periodic THC-v2

**Tables 9, 11** · [Sampler](../../../tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/code/banana_periodic_EXPANDED_20260925.py) · [Settings and seed record](records/THC_v2/FROZEN.txt) · [Run launcher](../../../tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `a=0.5`, `Q=4.0`, `epsilon=0.05`, `L=60`, `alpha=0.025`, `phi=0.0`.

[Original final report](records/THC_v2/FINAL_TUNING_REPORT.txt). Full report in archive: `Banana/Periodic_THC_v2/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/THC_v2/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 6100` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py banana THC_v2 --part 0 --configuration 6100
```

<a id="method-s_qhmc"></a>

### S-QHMC

**Table 9** · [Sampler](../../../tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/code/banana_sqhmc_EXPANDED_20260925.py) · [Settings and seed record](records/S_QHMC/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `L=30`, `epsilon=0.12`, `mu_m=0.5`, `sigma_m=0.0025`.

[Winning entry from the original final report](records/S_QHMC/selected-representative.txt). Full report in archive: `Banana/S_QHMC/results/S_QHMC_Banana_expanded_tuning_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py --method sqhmc --source tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/code/banana_sqhmc_EXPANDED_20260925.py --run-dir runs/banana/S_QHMC
```

<a id="method-nuts"></a>

### Original slice-NUTS

**Table 11** · [Sampler](../../../tuning-code/grex_banana_nuts_expanded_20260925/NUTS/code/banana_nuts_tuner_EXPANDED_20260925.py) · [Settings and seed record](records/NUTS/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_banana_nuts_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `target_acceptance=0.96`, `maximum_tree_depth=12`.

[Original final report](records/NUTS/NUTS_Banana_expanded_tuning_20260925.txt). Full report in archive: `Banana/NUTS/results/NUTS_Banana_expanded_tuning_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_banana_nuts_expanded_20260925/launcher/checkpoint_runner.py --source tuning-code/grex_banana_nuts_expanded_20260925/NUTS/code/banana_nuts_tuner_EXPANDED_20260925.py --run-dir runs/banana/NUTS
```

