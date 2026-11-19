# Table 7. Banana — tuning

HMC, THC (bounded scale), THC-v4

| Method | Defined/tested | Frozen settings | Tuning score |
| --- | --- | --- | --- |
| HMC | 98/143 | epsilon=0.04, L=80, T=3.2 | 4.1236e-3 |
| THC (bounded scale) | 49518/81081 | lambda=log(1.75), a=4, Q=0.5, epsilon=0.05, L=60, T=3 | 4.9890e-3 |
| THC-v4 | 31259/45278 | kappa=0.5, tau=0.5, Q=0.1, a=8, epsilon=0.10, L=20, T=2 | 4.1179e-2 |

The tuning score is the geometric mean of the ten blockwise worst-observable efficiencies.

The THC-v4 search used the saved subset of its grid. The bounded-scale THC search completed its full grid.

## Code, settings and results

- HMC · [Python script](../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_hmc_EXPANDED_20260925.py) · [Original search settings](../targets/banana/tuning/records/HMC/run_manifest.txt) · [Original tuning report](../targets/banana/tuning/records/HMC/HMC_Banana_expanded_tuning_20260925.txt) · [Run instructions and full archive location](../targets/banana/tuning/README.md#method-hmc)
- THC (bounded scale) · [Python script](../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_bounded_EXPANDED_20260925.py) · [Original search settings](../targets/banana/tuning/records/THC_bounded/FROZEN.txt) · [Original tuning report](../targets/banana/tuning/records/THC_bounded/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/banana/tuning/README.md#method-thc_bounded)
- THC-v4 · [Python script](../tuning-code/grex_banana_hmc_bounded_v4_expanded_20260925/code/banana_v4_EXPANDED_20260925.py) · [Original search settings](../targets/banana/tuning/records/THC_v4/FROZEN.txt) · [Original tuning report](../targets/banana/tuning/records/THC_v4/FINAL_TUNING_REPORT.txt) · [Run instructions and full archive location](../targets/banana/tuning/README.md#method-thc_v4)

[Machine-readable winners and scores](../targets/banana/tuning/winners.json)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 7
```

[All tables](../README.md)
