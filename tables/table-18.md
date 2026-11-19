# Table 18. Neal’s funnel — tuning

Original slice-NUTS, Scalar THC, Periodic THC, Conditional THC

| Method | Frozen settings | Tuning score |
| --- | --- | --- |
| Original slice-NUTS | target_acceptance=0.45, maximum_tree_depth=12 | 6.75415e-4 |
| Scalar THC | lambda=log(1.005), a=0.0625, Q=0.003125, epsilon=0.18, L=10 | 3.50292e-4 |
| Periodic THC | alpha=0.75, phi=pi/2, a=0.25, Q=128, epsilon=0.20, L=5 | 4.24273e-4 |
| Conditional THC | tau=0.375, Q=1, epsilon=0.30, L=30 | 2.23840e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

All three THC tuning winners are shared with Table 16.

## Code, settings and results

- Original slice-NUTS · [Python script](../tuning-code/grex_funnel_nuts_expanded_20260925/NUTS/code/neals_funnel_nuts_tuner_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_funnel_nuts_expanded_20260925/NUTS/code/neals_funnel_nuts_tuner_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/NUTS/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/NUTS/NUTS_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-nuts)
- Scalar THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_scalar_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_scalar_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Scalar_THC/FROZEN.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Scalar_THC/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-scalar_thc)
- Periodic THC · [Python script](../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_periodic_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_periodic_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Periodic_THC/FROZEN.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Periodic_THC/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-periodic_thc)
- Conditional THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Conditional_THC/part_000/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Conditional_THC/Conditional_THC_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-conditional_thc)

[Machine-readable winners and scores](../targets/neals-funnel/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 18
```

[All tables](../README.md)
