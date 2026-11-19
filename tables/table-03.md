# Table 3. Multiscale Gaussian — tuning

S-QHMC and Refreshed-mixture THC-v3

| Method | Frozen settings | Tuning score |
| --- | --- | --- |
| S-QHMC | mu_m=0, sigma_m=0.1, epsilon=0.05, L=20 | 1.08130e-2 |
| Refreshed-mixture THC-v3 | delta=0.03, Q=150, epsilon=0.06, L=30 | 1.23496e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

The same THC-v3 tuning search supplies Table 5.

## Code, settings and results

- S-QHMC · [Python script](../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/S_QHMC/code/multiscale_gaussian_s_qhmc_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/S_QHMC/code/multiscale_gaussian_s_qhmc_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/S_QHMC/run_manifest.txt) · [Original report excerpt (winning setting and blocks)](../targets/multiscale-gaussian/tuning/records/S_QHMC/selected-representative.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-s_qhmc)
- Refreshed-mixture THC-v3 · [Python script](../tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_gaussian_mixture_sqhmc_expanded_20260925/THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/multiscale-gaussian/tuning/records/THC_v3/FROZEN.txt) · [Original tuning report](../targets/multiscale-gaussian/tuning/records/THC_v3/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/multiscale-gaussian/tuning/README.md#method-thc_v3)

[Machine-readable winners and scores](../targets/multiscale-gaussian/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 3
```

[All tables](../README.md)
