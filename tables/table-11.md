# Table 11. Banana — tuning

Original slice-NUTS and Periodic THC-v2

| Method | Frozen settings | Tuning score |
| --- | --- | --- |
| Original slice-NUTS | target_acceptance=0.96, maximum_tree_depth=12 | 2.59557e-3 |
| Periodic THC-v2 | alpha=0.025, phi=0, a=0.5, Q=4, epsilon=0.05, L=60 | 5.50555e-3 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

Periodic THC-v2 shares its tuning winner with Table 9.

## Code, settings and results

- Original slice-NUTS · [Python script](../tuning-code/grex_banana_nuts_expanded_20260925/NUTS/code/banana_nuts_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_banana_nuts_expanded_20260925/NUTS/code/banana_nuts_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/banana/tuning/records/NUTS/run_manifest.txt) · [Original tuning report](../targets/banana/tuning/records/NUTS/NUTS_Banana_expanded_tuning_20260925.txt) · [Run instructions and full archive location](../targets/banana/tuning/README.md#method-nuts)
- Periodic THC-v2 · [Python script](../tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/code/banana_periodic_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_banana_periodic_sqhmc_expanded_20260925/code/banana_periodic_EXPANDED_20260925.py) · [Original search settings](../targets/banana/tuning/records/THC_v2/FROZEN.txt) · [Original tuning report](../targets/banana/tuning/records/THC_v2/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/banana/tuning/README.md#method-thc_v2)

[Machine-readable winners and scores](../targets/banana/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 11
```

[All tables](../README.md)
