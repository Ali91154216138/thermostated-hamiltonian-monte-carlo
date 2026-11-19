# Table 16. Neal’s funnel — tuning

S-QHMC, Scalar THC, Periodic THC, Conditional THC

| Method | Frozen settings | Tuning score |
| --- | --- | --- |
| S-QHMC | mu_m=-0.5, sigma_m=0.25, epsilon=0.12, L=10 | 3.21562e-4 |
| Scalar THC | lambda=log(1.005), a=0.0625, Q=0.003125, epsilon=0.18, L=10 | 3.50292e-4 |
| Periodic THC | alpha=0.75, phi=pi/2, a=0.25, Q=128, epsilon=0.20, L=5 | 4.24273e-4 |
| Conditional THC | tau=0.375, Q=1, epsilon=0.30, L=30 | 2.23840e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

The scalar and conditional THC tuning campaigns also enter Table 13. All three THC tuning campaigns shown here are shared with Table 18.

## Code, settings and results

- S-QHMC · [Python script](../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_sqhmc_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/S_QHMC/part_000/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/S_QHMC/S_QHMC_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-s_qhmc)
- Scalar THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_scalar_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Scalar_THC/FROZEN.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Scalar_THC/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-scalar_thc)
- Periodic THC · [Python script](../tuning-code/grex_funnel_periodic_sqhmc_expanded_20260925/code/funnel_periodic_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Periodic_THC/FROZEN.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Periodic_THC/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-periodic_thc)
- Conditional THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Conditional_THC/part_000/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Conditional_THC/Conditional_THC_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-conditional_thc)

[Machine-readable winners and scores](../targets/neals-funnel/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 16
```

[All tables](../README.md)
