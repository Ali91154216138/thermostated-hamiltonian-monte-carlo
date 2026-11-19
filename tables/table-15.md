# Table 15. Neal’s funnel — held-out comparison

HMC and Conditional THC

| Method | Gradients/draw | Acceptance | Maximum worst IAT | Minimum efficiency | Held-out score | Maximum R-hat | Maximum moment error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMC | 30 | 0.9224 | 870.5144 | 3.8292e-5 | 1.8848e-4 | 1.05081 | 0.17402 |
| Conditional THC | 30 | 0.8833 | 2.8766 | 1.1588e-2 | 1.9178e-2 | 1.00068 | 0.01452 |

The held-out score is the geometric mean of the ten blockwise worst-observable efficiencies. Gradients/draw counts physical target-gradient evaluations during retention.

The HMC report and ten paired blocks are shared with Table 14.

## Code, settings and results

- HMC · [Python script](../targets/neals-funnel/held-out/hmc/HMC.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/neals-funnel/held-out/hmc/HMC.py) · [Original block results](../targets/neals-funnel/held-out/hmc/HMC_results.txt)
- Conditional THC · [Python script](../targets/neals-funnel/held-out/hmc/Conditional_THC.py) · [Exact run file](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/blob/0c7e37a437f917256e3425f9c80ca248b4a82ae5/targets/neals-funnel/held-out/hmc/Conditional_THC.py) · [Original block results](../targets/neals-funnel/held-out/hmc/Conditional_THC_results.txt)

[Frozen protocol and seeds](../targets/neals-funnel/held-out/hmc/frozen_protocol.txt) · [Run instructions](../targets/neals-funnel/held-out/hmc/README.md)

From the repository root, check the displayed values with

```sh
python tables/verify_tables.py 15
```

[All tables](../README.md)
