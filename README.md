# Thermostated Hamiltonian Monte-Carlo

Alireza Sharifi and Leo T. Butler

Select a table to find its simulation code, settings and numerical results. Table numbers follow the November 19, 2026 manuscript.

| Target | Methods | Tuning | Held-out comparison |
|---|---|---|---|
| Multiscale Gaussian | HMC and bounded-scale THC | [Table 1](tables/table-01.md) | [Table 2](tables/table-02.md) |
| Multiscale Gaussian | S-QHMC and refreshed-mixture THC-v3 | [Table 3](tables/table-03.md) | [Table 4](tables/table-04.md) |
| Multiscale Gaussian | Slice-NUTS and refreshed-mixture THC-v3 | [Table 5](tables/table-05.md) | [Table 6](tables/table-06.md) |
| Banana | HMC, bounded-scale THC and conditional-shear THC-v4 | [Table 7](tables/table-07.md) | [Table 8](tables/table-08.md) |
| Banana | S-QHMC and periodic THC-v2 | [Table 9](tables/table-09.md) | [Table 10](tables/table-10.md) |
| Banana | Slice-NUTS and periodic THC-v2 | [Table 11](tables/table-11.md) | [Table 12](tables/table-12.md) |
| Neal’s funnel | HMC, scalar THC and Conditional THC | [Table 13](tables/table-13.md) | [Table 14](tables/table-14.md), [Table 15](tables/table-15.md) |
| Neal’s funnel | S-QHMC, scalar, periodic and Conditional THC | [Table 16](tables/table-16.md) | [Table 17](tables/table-17.md) |
| Neal’s funnel | Slice-NUTS, scalar, periodic and Conditional THC | [Table 18](tables/table-18.md) | [Table 19](tables/table-19.md) |

For example, [Table 10](tables/table-10.md) compares S-QHMC and periodic THC-v2 on the Banana target. Its [comparison folder](targets/banana/held-out/s-qhmc/README.md) links each script to its original result report. Its [frozen protocol](targets/banana/held-out/s-qhmc/frozen_protocol.txt) records the parameters, starts and seed blocks used for both methods.

Each table page offers two links for each method. **Python script** opens the shorter version for reading or rerunning the simulation. **Exact run file** opens the file used on GREX, including its original comments and documentation. The recorded results correspond to the [original source snapshot](https://github.com/Ali91154216138/thermostated-hamiltonian-monte-carlo/tree/0c7e37a437f917256e3425f9c80ca248b4a82ae5).

## Check a table

With Python 3.10 or later installed, run the following command from the repository root to check Table 10 against the saved method outputs.

```sh
python tables/verify_tables.py 10
```

Each table page gives the corresponding command and links to the original experiment. Omit the table number to check all 19 tables. The frozen protocol was recorded before the simulations. The result reports alongside it contain the completed runs.

## Reading the results

Tuning selected the parameters. Held-out runs then evaluated those settings on fresh seed blocks. Methods in the same comparison use paired blocks. Some tuning winners appear in more than one comparison, with a separate held-out report for each run. Tables 14 and 15 share the same HMC run.

The repository contains 17 tuning campaigns and 24 held-out method reports. The [data guide](data/README.md) identifies the three full tuning archives and their checksums. It also records their current availability.
