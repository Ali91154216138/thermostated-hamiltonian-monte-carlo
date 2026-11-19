# Table 1. Multiscale Gaussian — tuning

HMC and THC

| Method | Defined/tested | Frozen settings | Tuning score |
| --- | --- | --- | --- |
| HMC | 273/324 | epsilon=0.055, L=15, T=0.825 | 8.7912e-3 |
| THC | 17977/19522 | lambda=log(1.25), a=0.5, Q=0.25, epsilon=0.04, L=25, T=1 | 1.0129e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

## Code, settings and results

- HMC · [Python script](../tuning-code/grex_gaussian_expanded_20260925/HMC/code/multiscale_gaussian_hmc_tuner_EXPANDED_20260925_GREX_RUN.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/HMC/run_manifest.txt) · [Original tuning report](../targets/multiscale-gaussian/tuning/records/HMC/HMC_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-hmc)
- THC · [Python script](../tuning-code/grex_gaussian_expanded_20260925/THC/code/multiscale_gaussian_thc_tuner_EXPANDED_20260925_GREX_RUN.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/THC_bounded/FROZEN.txt) · [Original tuning report](../targets/multiscale-gaussian/tuning/records/THC_bounded/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-thc_bounded)

[Machine-readable winners and scores](../targets/multiscale-gaussian/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 1
```

[All tables](../README.md)
