# Table 13. Neal’s funnel — tuning

HMC, Scalar THC, Conditional THC

| Method | Defined/tested | Frozen settings | Tuning score |
| --- | --- | --- | --- |
| HMC | 218/252 | epsilon=0.15, L=30, T=4.5 | 1.85387e-4 |
| Scalar THC | 73179/86980 | lambda=log(1.005), a=0.0625, Q=0.003125, epsilon=0.18, L=10, T=1.8 | 3.50292e-4 |
| Conditional THC | 40432/59976 | tau=0.375, Q=1, epsilon=0.30, L=30, T=9 | 2.23840e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

Tables 14 and 15 share the HMC run selected here.

## Code, settings and results

- HMC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_hmc_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/HMC/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/HMC/HMC_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-hmc)
- Scalar THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_scalar_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Scalar_THC/FROZEN.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Scalar_THC/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-scalar_thc)
- Conditional THC · [Python script](../tuning-code/grex_funnel_hmc_scalar_conditional_expanded_20260925/code/funnel_conditional_EXPANDED_20260925.py) · [Original search settings](../targets/neals-funnel/tuning/records/Conditional_THC/part_000/run_manifest.txt) · [Original tuning report](../targets/neals-funnel/tuning/records/Conditional_THC/Conditional_THC_Neals_funnel_expanded_tuning.txt) · [Run instructions and full archive location](../targets/neals-funnel/tuning/README.md#method-conditional_thc)

[Machine-readable winners and scores](../targets/neals-funnel/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 13
```

[All tables](../README.md)
