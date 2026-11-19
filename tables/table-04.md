# Table 4. Multiscale Gaussian — held-out comparison

S-QHMC and Refreshed-mixture THC-v3

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S-QHMC | 20 | 0.8071 | 4.9374 | 1.0127e-2 | 1.0698e-2 | 1.00085 | 0.02664 |
| Refreshed-mixture THC-v3 | 30 | 0.8338 | 2.8066 | 1.1877e-2 | 1.2257e-2 | 1.00048 | 0.02395 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

The THC-v3 held-out results here use the blocks paired with S-QHMC. Table 6 uses a separate run.

## Code, settings and results

- S-QHMC · [Python script](../targets/multiscale-gaussian/held-out/s-qhmc/S_QHMC.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/multiscale-gaussian/held-out/s-qhmc/S_QHMC.py) · [Original block results](../targets/multiscale-gaussian/held-out/s-qhmc/S_QHMC_results.txt)
- Refreshed-mixture THC-v3 · [Python script](../targets/multiscale-gaussian/held-out/s-qhmc/THC_v3.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/multiscale-gaussian/held-out/s-qhmc/THC_v3.py) · [Original block results](../targets/multiscale-gaussian/held-out/s-qhmc/THC_v3_results.txt)

[Frozen protocol and seeds](../targets/multiscale-gaussian/held-out/s-qhmc/frozen_protocol.txt) · [Run instructions](../targets/multiscale-gaussian/held-out/s-qhmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 4
```

[All tables](../README.md)
