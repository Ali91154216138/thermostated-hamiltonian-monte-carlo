# Table 6. Multiscale Gaussian — held-out comparison

NUTS and Refreshed-mixture THC-v3

| Method | Gradients/draw | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Wins |
| --- | --- | --- | --- | --- | --- | --- |
| NUTS | 17.1471 | 8.1336 | 7.5500e-3 | 8.6381e-3 | 1.00175 | 0/10 |
| Refreshed-mixture THC-v3 | 30 | 2.8702 | 1.1614e-2 | 1.2182e-2 | 1.00047 | 10/10 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

Wins count the blocks in which a method had greater efficiency than the other method.

## Code, settings and results

- NUTS · [Python script](../targets/multiscale-gaussian/held-out/nuts/NUTS.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/multiscale-gaussian/held-out/nuts/NUTS.py) · [Original block results](../targets/multiscale-gaussian/held-out/nuts/NUTS_results.txt)
- Refreshed-mixture THC-v3 · [Python script](../targets/multiscale-gaussian/held-out/nuts/THC_v3.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/multiscale-gaussian/held-out/nuts/THC_v3.py) · [Original block results](../targets/multiscale-gaussian/held-out/nuts/THC_v3_results.txt)

[Frozen protocol and seeds](../targets/multiscale-gaussian/held-out/nuts/frozen_protocol.txt) · [Run instructions](../targets/multiscale-gaussian/held-out/nuts/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 6
```

[All tables](../README.md)
