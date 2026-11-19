# Multiscale Gaussian tuning

Use this page for the tuning tables. Each method links to its original Python program, recorded settings and winning result. A shared tuning result is stored once. The columns below count the configurations with a defined score and all configurations evaluated.

| Table(s) | Method | Defined / tested | Winning score |
|---|---|---:|---:|
| 1 | [HMC](#method-hmc) | 273 / 324 | 0.008791180312 |
| 1 | [Bounded-scale THC](#method-thc_bounded) | 17,977 / 19,522 | 0.01012904825 |
| 3, 5 | [Refreshed-mixture THC-v3](#method-thc_v3) | 9,936 / 11,411 | 0.01234963634 |
| 3 | [S-QHMC](#method-s_qhmc) | 23,441 / 37,908 | 0.01081297399 |
| 5 | [Original slice-NUTS](#method-nuts) | 35 / 35 | 0.008799874131 |

[Winning settings and scores in JSON](winners.json) · [Full data archive](../../../data/README.md)

The recorded campaigns used ten tuning blocks, four chains per block, and 1,000 warm-up plus 10,000 retained samples per chain. Scores summarize worst-observable efficiency across the ten blocks. The settings in the manifests give the complete grid and seed schedule. The JSON includes ancillary settings from the frozen evaluation protocol.

The paths labelled “archive report” below are members of `THC_tuning_Multiscale_Gaussian_20261007.zip`. From the repository root, the following command extracts and checks the tuning winners from that ZIP without running a sampler. It uses the Python 3.11+ standard library.

```sh
python tuning-code/extract_winners.py /path/to/THC_tuning_Multiscale_Gaussian_20261007.zip
```

For numerical reruns, use a separate output directory and the production Python environment described in the archived run records. Commands below assume Linux and are run at the repository root. The original Slurm files include cluster-specific project paths.

<a id="method-hmc"></a>

### HMC

**Table 1** · [Exact sampler](../../../tuning-code/grex_gaussian_expanded_20260925/HMC/code/multiscale_gaussian_hmc_tuner_EXPANDED_20260925_GREX_RUN.py) · [Settings and seed record](records/HMC/run_manifest.txt) · [Original launcher](../../../tuning-code/grex_gaussian_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `epsilon=0.055`, `L=15`.

[Original final report](records/HMC/HMC_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt). Full report in archive: `Multiscale_Gaussian/HMC/results/HMC_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_gaussian_expanded_20260925/launcher/checkpoint_runner.py --method hmc --source tuning-code/grex_gaussian_expanded_20260925/HMC/code/multiscale_gaussian_hmc_tuner_EXPANDED_20260925_GREX_RUN.py --run-dir runs/multiscale-gaussian/HMC
```

<a id="method-thc_bounded"></a>

### Bounded-scale THC

**Table 1** · [Exact sampler](../../../tuning-code/grex_gaussian_expanded_20260925/THC/code/multiscale_gaussian_thc_tuner_EXPANDED_20260925_GREX_RUN.py) · [Settings and seed record](records/THC_bounded/FROZEN.txt) · [Original launcher](../../../tuning-code/grex_gaussian_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `epsilon=0.04`, `L=25`, `lambda_value=0.22314355131420976`, `a=0.5`, `Q=0.25`.

[Original final report](records/THC_bounded/FINAL_TUNING_REPORT.txt). Full report in archive: `Multiscale_Gaussian/Bounded_scale_THC/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/THC_bounded/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 30303` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py multiscale-gaussian THC_bounded --part 0 --configuration 30303
```

<a id="method-thc_v3"></a>

### Refreshed-mixture THC-v3

**Tables 3, 5** · [Exact sampler](../../../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py) · [Settings and seed record](records/THC_v3/FROZEN.txt) · [Original launcher](../../../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `epsilon=0.06`, `L=30`, `delta=0.03`, `Q=150.0`, `inner_r=8`.

[Original final report](records/THC_v3/FINAL_TUNING_REPORT.txt). Full report in archive: `Multiscale_Gaussian/Refreshed_mixture_THC_v3/results/FINAL_TUNING_REPORT.txt`.

The original campaign used the [saved search wrapper](../../../tuning-code/grex_thc_budgeted_20260927/budgeted_runner.py). The [selection lists](records/THC_v3/selection) contain the exact configuration indices assigned to each worker. The command below reruns the winning configuration with its ten original seed blocks. Omit `--configuration 118080` to replay that entire worker part. Repeat for the other numbered parts to cover the full tested set. Add `--check-only` to verify the code, settings and worker list without sampling. To keep numerical output, redirect it to a new file (one JSON record per configuration).

```sh
python tuning-code/replay_selection.py multiscale-gaussian THC_v3 --part 3 --configuration 118080
```

<a id="method-s_qhmc"></a>

### S-QHMC

**Table 3** · [Exact sampler](../../../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/S_QHMC/code/multiscale_gaussian_s_qhmc_tuner_EXPANDED_20260925.py) · [Settings and seed record](records/S_QHMC/run_manifest.txt) · [Original launcher](../../../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `L=20`, `epsilon=0.05`, `mu_m=0.0`, `sigma_m=0.1`.

[Winning entry from the original final report](records/S_QHMC/selected-representative.txt). Full report in archive: `Multiscale_Gaussian/S_QHMC/results/S_QHMC_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/launcher/checkpoint_runner.py --method sqhmc --source tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/S_QHMC/code/multiscale_gaussian_s_qhmc_tuner_EXPANDED_20260925.py --run-dir runs/multiscale-gaussian/S_QHMC
```

<a id="method-nuts"></a>

### Original slice-NUTS

**Table 5** · [Exact sampler](../../../tuning-code/grex_gaussian_nuts_expanded_20260925/NUTS/code/multiscale_gaussian_nuts_tuner_EXPANDED_20260925.py) · [Settings and seed record](records/NUTS/run_manifest.txt) · [Original launcher](../../../tuning-code/grex_gaussian_nuts_expanded_20260925/launcher/checkpoint_runner.py)

Winning settings: `target_acceptance=0.425`, `maximum_tree_depth=12`.

[Original final report](records/NUTS/NUTS_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt). Full report in archive: `Multiscale_Gaussian/NUTS/results/NUTS_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt`.

The command below runs the complete search.

```sh
python tuning-code/grex_gaussian_nuts_expanded_20260925/launcher/checkpoint_runner.py --source tuning-code/grex_gaussian_nuts_expanded_20260925/NUTS/code/multiscale_gaussian_nuts_tuner_EXPANDED_20260925.py --run-dir runs/multiscale-gaussian/NUTS
```

