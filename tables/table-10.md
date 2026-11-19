# Table 10. Banana — held-out comparison

S-QHMC and Periodic THC-v2

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error | Nonfinite proposals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-QHMC | 29.9888 | 0.9453 | 170.1377 | 1.9639e-4 | 3.2420e-3 | 1.00340 | 0.05282 | 424 |
| Periodic THC-v2 | 59.9990 | 0.9745 | 34.2873 | 4.8616e-4 | 3.8163e-3 | 1.00036 | 0.04052 | 26 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

Nonfinite proposals counts events during retained sampling. All ten blocks had defined scores under the rule for this comparison. The THC-v2 report for Table 12 uses separate seed blocks.

## Code, settings and results

- S-QHMC · [Python script](../targets/banana/held-out/s-qhmc/S_QHMC.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/s-qhmc/S_QHMC.py) · [Original block results](../targets/banana/held-out/s-qhmc/S_QHMC_results.txt)
- Periodic THC-v2 · [Python script](../targets/banana/held-out/s-qhmc/THC_v2.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/s-qhmc/THC_v2.py) · [Original block results](../targets/banana/held-out/s-qhmc/THC_v2_results.txt)

[Frozen protocol and seeds](../targets/banana/held-out/s-qhmc/frozen_protocol.txt) · [Run instructions](../targets/banana/held-out/s-qhmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 10
```

[All tables](../README.md)
