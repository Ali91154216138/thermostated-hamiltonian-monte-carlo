# Table 12. Banana — held-out comparison

NUTS and Periodic THC-v2

| Method | Gradients/draw | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Wins |
| --- | --- | --- | --- | --- | --- | --- |
| NUTS | 18.5186 | 202.2931 | 2.5411e-4 | 1.8209e-3 | 1.00343 | 1/10 |
| Periodic THC-v2 | 59.9997 | 6.4174 | 2.5971e-3 | 4.9798e-3 | 1.00061 | 9/10 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

Wins count the blocks in which a method had greater efficiency than the other method. Complete periodic THC-v2 proposals used 60 gradients. Early termination reduces the actual mean.

## Code, settings and results

- NUTS · [Python script](../targets/banana/held-out/nuts/NUTS.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/nuts/NUTS.py) · [Original block results](../targets/banana/held-out/nuts/NUTS_results.txt)
- Periodic THC-v2 · [Python script](../targets/banana/held-out/nuts/THC_v2.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/banana/held-out/nuts/THC_v2.py) · [Original block results](../targets/banana/held-out/nuts/THC_v2_results.txt)

[Frozen protocol and seeds](../targets/banana/held-out/nuts/frozen_protocol.txt) · [Run instructions](../targets/banana/held-out/nuts/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 12
```

[All tables](../README.md)
