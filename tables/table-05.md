# Table 5. Multiscale Gaussian — tuning

Original slice-NUTS and Refreshed-mixture THC-v3

| Method | Frozen settings | Tuning score |
| --- | --- | --- |
| Original slice-NUTS | target_acceptance=0.425, maximum_tree_depth=12 | 8.79987e-3 |
| Refreshed-mixture THC-v3 | delta=0.03, Q=150, epsilon=0.06, L=30, inner_r=8 | 1.23496e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

The THC-v3 tuning winner is shared with Table 3.

## Code, settings and results

- Original slice-NUTS · [Python script](../tuning-code/grex_gaussian_nuts_expanded_20260925/NUTS/code/multiscale_gaussian_nuts_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_gaussian_nuts_expanded_20260925/NUTS/code/multiscale_gaussian_nuts_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/NUTS/run_manifest.txt) · [Original tuning report](../targets/multiscale-gaussian/tuning/records/NUTS/NUTS_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-nuts)
- Refreshed-mixture THC-v3 · [Python script](../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/THC_v3/FROZEN.txt) · [Original tuning report](../targets/multiscale-gaussian/tuning/records/THC_v3/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-thc_v3)

[Machine-readable winners and scores](../targets/multiscale-gaussian/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 5
```

[All tables](../README.md)
