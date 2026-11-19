# Table 19. Neal’s funnel — held-out comparison

NUTS, Scalar THC, Periodic THC, Conditional THC

| Method | Gradients/draw | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Wins |
| --- | --- | --- | --- | --- | --- | --- |
| NUTS | 11.291 | 1089.379 | 9.4417e-5 | 5.5124e-4 | 1.17429 | — |
| Scalar THC | 10 | 897.811 | 1.1138e-4 | 2.7436e-4 | 1.04496 | 3/10 |
| Periodic THC | 5 | 2420.849 | 8.2616e-5 | 4.2885e-4 | 1.11279 | 3/10 |
| Conditional THC | 30 | 3.310 | 1.0070e-2 | 1.8132e-2 | 1.00056 | 10/10 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

Each THC row counts wins over NUTS on ten matched blocks. The dash marks the shared reference run. These are three pairwise comparisons.

## Code, settings and results

- NUTS · [Python script](../targets/neals-funnel/held-out/nuts/NUTS.py) · [Original block results](../targets/neals-funnel/held-out/nuts/NUTS_results.txt)
- Scalar THC · [Python script](../targets/neals-funnel/held-out/nuts/Scalar_THC.py) · [Original block results](../targets/neals-funnel/held-out/nuts/Scalar_THC_results.txt)
- Periodic THC · [Python script](../targets/neals-funnel/held-out/nuts/Periodic_THC.py) · [Original block results](../targets/neals-funnel/held-out/nuts/Periodic_THC_results.txt)
- Conditional THC · [Python script](../targets/neals-funnel/held-out/nuts/Conditional_THC.py) · [Original block results](../targets/neals-funnel/held-out/nuts/Conditional_THC_results.txt)

[Frozen protocol and seeds](../targets/neals-funnel/held-out/nuts/frozen_protocol.txt) · [Run instructions](../targets/neals-funnel/held-out/nuts/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 19
```

[All tables](../README.md)
