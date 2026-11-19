# Neal’s funnel tuning

Use this page for the tuning tables. Each method links to its Python program, recorded settings and winning result. A shared tuning result is stored once. The columns below count the configurations with a defined score and all configurations evaluated.

| Table(s) | Method | Defined / tested | Winning score |
|---|---|---:|---:|
| 13 | [HMC](#method-hmc) | 218 / 252 | 0.0001853868984 |
| 13, 16, 18 | [Scalar bounded-scale THC](#method-scalar_thc) | 73,179 / 86,980 | 0.0003502922187 |
| 13, 16, 18 | [Conditional THC](#method-conditional_thc) | 40,432 / 59,976 | 0.02238403088 |
| 16, 18 | [Periodic THC](#method-periodic_thc) | 87,042 / 105,792 | 0.0004242728052 |
| 16 | [S-QHMC](#method-s_qhmc) | 18,465 / 31,752 | 0.0003215621733 |
| 18 | [Original slice-NUTS](#method-nuts) | 31 / 35 | 0.0006754152349 |

[Winning settings and scores in JSON](winners.json) · [Full data archive](../../../data/README.md)

The recorded campaigns used ten tuning blocks, four chains per block, and 1,000 warm-up plus 10,000 retained samples per chain. Scores summarize worst-observable efficiency across the ten blocks. The settings in the manifests give the complete grid and seed schedule. The JSON includes ancillary settings from the frozen evaluation protocol.

The paths labelled “archive report” below are members of `THC_tuning_Neals_funnel_20261007.zip`. From the repository root, the following command extracts and checks the tuning winners from that ZIP without running a sampler. It uses the Python 3.11+ standard library.

```sh
python tuning-code/extract_winners.py /path/to/THC_tuning_Neals_funnel_20261007.zip
```

For numerical reruns, use a separate output directory and the production Python environment described in the archived run records. Commands below assume Linux and are run at the repository root. The original Slurm files include cluster-specific project paths.

<a id="method-hmc"></a>

### HMC

**Table 13** · [Sampler](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_hmc_EXPANDED_20260925.py) · [Settings and seed record](records/HMC/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `epsilon=0.15`, `L=30`.

[Original final report](records/HMC/HMC_Neals_funnel_expanded_tuning.txt). Full report in archive: `Neals_funnel/HMC/results/HMC_Neals_funnel_expanded_tuning.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/launcher/checkpoint_runner.py --method hmc --source tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_hmc_EXPANDED_20260925.py --run-dir runs/neals-funnel/HMC
```

<a id="method-scalar_thc"></a>

### Scalar bounded-scale THC

**Tables 13, 16, 18** · [Sampler](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_scalar_EXPANDED_20260925.py) · [Settings and seed record](records/Scalar_THC/FROZEN.txt) · [Run launcher](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `lambda_value=0.004987541511038968`, `a=0.0625`, `Q=0.003125`, `epsilon=0.18`, `L=10`.

[Original final report](records/Scalar_THC/FINAL_TUNING_REPORT.txt). Full report in archive: `Neals_funnel/Scalar_bounded_scale_THC/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/Scalar_THC/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 197` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py neals-funnel Scalar_THC --part 5 --configuration 197
```

<a id="method-conditional_thc"></a>

### Conditional THC

**Tables 13, 16, 18** · [Sampler](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py) · [Settings and seed record](records/Conditional_THC/part_000/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `tau=0.375`, `Q=1.0`, `epsilon=0.3`, `L=30`.

[Original final report](records/Conditional_THC/Conditional_THC_Neals_funnel_expanded_tuning.txt). Full report in archive: `Neals_funnel/Conditional_THC/results/Conditional_THC_Neals_funnel_expanded_tuning.txt`.

The command below runs the complete search.

```sh
for part in 0 1 2 3; do
  python tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/launcher/checkpoint_runner.py --method conditional --source tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py --run-dir runs/neals-funnel/Conditional_THC --part "$part" --parts 4
done
```

<a id="method-periodic_thc"></a>

### Periodic THC

**Tables 16, 18** · [Sampler](../../../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_periodic_EXPANDED_20260925.py) · [Settings and seed record](records/Periodic_THC/FROZEN.txt) · [Run launcher](../../../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `alpha=0.75`, `phi=1.5707963267948966`, `a=0.25`, `Q=128.0`, `epsilon=0.2`, `L=5`.

[Original final report](records/Periodic_THC/FINAL_TUNING_REPORT.txt). Full report in archive: `Neals_funnel/Periodic_THC/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/Periodic_THC/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 976710` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py neals-funnel Periodic_THC --part 6 --configuration 976710
```

<a id="method-s_qhmc"></a>

### S-QHMC

**Table 16** · [Sampler](../../../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_sqhmc_EXPANDED_20260925.py) · [Settings and seed record](records/S_QHMC/part_000/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `mu_m=-0.5`, `sigma_m=0.25`, `epsilon=0.12`, `L=10`.

[Original final report](records/S_QHMC/S_QHMC_Neals_funnel_expanded_tuning.txt). Full report in archive: `Neals_funnel/S_QHMC/results/S_QHMC_Neals_funnel_expanded_tuning.txt`.

The command below runs the complete search.

```sh
for part in 0 1 2 3; do
  python tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/launcher/checkpoint_runner.py --method sqhmc --source tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_sqhmc_EXPANDED_20260925.py --run-dir runs/neals-funnel/S_QHMC --part "$part" --parts 4
done
```

<a id="method-nuts"></a>

### Original slice-NUTS

**Table 18** · [Sampler](../../../tuning-code/grex_funnel_nuts_expanded_20260925/NUTS/code/neals_funnel_nuts_tuner_EXPANDED_20260925.py) · [Settings and seed record](records/NUTS/run_manifest.txt) · [Run launcher](../../../tuning-code/grex_funnel_nuts_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `target_acceptance=0.45`, `maximum_tree_depth=12`.

[Original final report](records/NUTS/NUTS_Neals_funnel_expanded_tuning.txt). Full report in archive: `Neals_funnel/NUTS/results/NUTS_Neals_funnel_expanded_tuning.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_funnel_nuts_expanded_20260925/launcher/checkpoint_runner.py --source tuning-code/grex_funnel_nuts_expanded_20260925/NUTS/code/neals_funnel_nuts_tuner_EXPANDED_20260925.py --run-dir runs/neals-funnel/NUTS
```

