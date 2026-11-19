# Table 14. Neal’s funnel — held-out comparison

HMC and Scalar THC

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMC | 30 | 0.9224 | 870.5144 | 3.8292e-5 | 1.8848e-4 | 1.05081 | 0.17402 |
| Scalar THC | 10 | 0.7382 | 600.4136 | 1.6655e-4 | 3.3638e-4 | 1.06094 | 0.22474 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

The HMC report and ten paired blocks are shared with Table 15.

## Code, settings and results

- HMC · [Python script](../targets/neals-funnel/held-out/hmc/HMC.py) · [Original block results](../targets/neals-funnel/held-out/hmc/HMC_results.txt)
- Scalar THC · [Python script](../targets/neals-funnel/held-out/hmc/Scalar_THC.py) · [Original block results](../targets/neals-funnel/held-out/hmc/Scalar_THC_results.txt)

[Frozen protocol and seeds](../targets/neals-funnel/held-out/hmc/frozen_protocol.txt) · [Run instructions](../targets/neals-funnel/held-out/hmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 14
```

[All tables](../README.md)
