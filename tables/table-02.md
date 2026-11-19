# Table 2. Multiscale Gaussian — held-out comparison

HMC and THC

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMC | 15 | 0.7253 | 8.0920 | 8.2385e-3 | 8.6552e-3 | 1.00129 | 0.03548 |
| THC | 25 | 0.8746 | 4.3136 | 9.2731e-3 | 9.8609e-3 | 1.00082 | 0.02031 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

## Code, settings and results

- HMC · [Python script](../targets/multiscale-gaussian/held-out/hmc/HMC.py) · [Original block results](../targets/multiscale-gaussian/held-out/hmc/HMC_results.txt)
- THC · [Python script](../targets/multiscale-gaussian/held-out/hmc/THC_bounded.py) · [Original block results](../targets/multiscale-gaussian/held-out/hmc/THC_bounded_results.txt)

[Frozen protocol and seeds](../targets/multiscale-gaussian/held-out/hmc/frozen_protocol.txt) · [Run instructions](../targets/multiscale-gaussian/held-out/hmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 2
```

[All tables](../README.md)
