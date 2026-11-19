# Table 17. Neal’s funnel — held-out comparison

S-QHMC, Scalar THC, Periodic THC, Conditional THC

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat |
| --- | --- | --- | --- | --- | --- | --- |
| S-QHMC | 10 | 0.8587 | 793.0731 | 1.2609e-4 | 2.4695e-4 | 1.06393 |
| Scalar THC | 10 | 0.7350 | 1289.4474 | 7.7553e-5 | 2.8578e-4 | 1.10691 |
| Periodic THC | 5 | 0.8164 | 2060.1995 | 9.7078e-5 | 3.7683e-4 | 1.24878 |
| Conditional THC | 30 | 0.8836 | 2.1870 | 1.5242e-2 | 2.2356e-2 | 1.00080 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

All three THC comparisons use the same S-QHMC run. The held-out THC runs for Table 19 use separate seed blocks.

## Code, settings and results

- S-QHMC · [Python script](../targets/neals-funnel/held-out/s-qhmc/S_QHMC.py) · [Original block results](../targets/neals-funnel/held-out/s-qhmc/S_QHMC_results.txt)
- Scalar THC · [Python script](../targets/neals-funnel/held-out/s-qhmc/Scalar_THC.py) · [Original block results](../targets/neals-funnel/held-out/s-qhmc/Scalar_THC_results.txt)
- Periodic THC · [Python script](../targets/neals-funnel/held-out/s-qhmc/Periodic_THC.py) · [Original block results](../targets/neals-funnel/held-out/s-qhmc/Periodic_THC_results.txt)
- Conditional THC · [Python script](../targets/neals-funnel/held-out/s-qhmc/Conditional_THC.py) · [Original block results](../targets/neals-funnel/held-out/s-qhmc/Conditional_THC_results.txt)

[Frozen protocol and seeds](../targets/neals-funnel/held-out/s-qhmc/frozen_protocol.txt) · [Run instructions](../targets/neals-funnel/held-out/s-qhmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 17
```

[All tables](../README.md)
