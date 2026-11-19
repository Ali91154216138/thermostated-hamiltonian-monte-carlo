# Table 8. Banana — held-out comparison

HMC, THC (bounded scale), THC-v4

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMC | 80 | 0.9844 | 4.2125 | 2.9674e-3 | 4.2091e-3 | 1.00047 | 0.01627 |
| THC (bounded scale) | 59.9998 | 0.9698 | 15.8207 | -- | undefined | 1.00051 | 0.02508 |
| THC-v4 | 20 | 0.9518 | 1.3248 | 3.7741e-2 | 4.1774e-2 | 1.00019 | 0.01113 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

The bounded-scale THC run recorded nonfinite proposals. Under the rule fixed before sampling, four blocks have undefined efficiency and the ten-block score is undefined. The IAT and other diagnostics in that row describe the retained draws.

## Code, settings and results

- HMC · [Python script](../targets/banana/held-out/hmc/HMC.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/hmc/HMC.py) · [Original block results](../targets/banana/held-out/hmc/HMC_results.txt)
- THC (bounded scale) · [Python script](../targets/banana/held-out/hmc/THC_bounded.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/hmc/THC_bounded.py) · [Original block results](../targets/banana/held-out/hmc/THC_bounded_results.txt)
- THC-v4 · [Python script](../targets/banana/held-out/hmc/THC_v4.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/hmc/THC_v4.py) · [Original block results](../targets/banana/held-out/hmc/THC_v4_results.txt)

[Frozen protocol and seeds](../targets/banana/held-out/hmc/frozen_protocol.txt) · [Run instructions](../targets/banana/held-out/hmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 8
```

[All tables](../README.md)
