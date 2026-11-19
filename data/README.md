# Full tuning data

The three ZIP files contain the GREX tuning records for the simulations in the manuscript. Each archive covers one target. The tuning guides identify the file inside the ZIP used for each table. The 24 held-out method reports are included directly in the comparison folders in this repository.

The [three full tuning archives](https://doi.org/10.5281/zenodo.24380090) are published on Zenodo under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use the links below to download the archive for each target.

| Tuning guide | Archive | Size (MiB) | SHA-256 |
|---|---|---:|---|
| [Multiscale Gaussian](../targets/multiscale-gaussian/tuning/README.md) | [THC_tuning_Multiscale_Gaussian_20261007.zip](https://zenodo.org/records/24380090/files/THC_tuning_Multiscale_Gaussian_20261007.zip?download=1) | 40.5 | `29b01b64b62cd495e25e084ef5156d81dc0fdee289226b84809e757cd9942edf` |
| [Banana](../targets/banana/tuning/README.md) | [THC_tuning_Banana_20261007.zip](https://zenodo.org/records/24380090/files/THC_tuning_Banana_20261007.zip?download=1) | 113.8 | `19b1f13d676d72846e64837695ba91a8325736b779e6f6aeb1d11ce22445c128` |
| [Neal’s funnel](../targets/neals-funnel/tuning/README.md) | [THC_tuning_Neals_funnel_20261007.zip](https://zenodo.org/records/24380090/files/THC_tuning_Neals_funnel_20261007.zip?download=1) | 185.5 | `c14faeb6164bdeafcc802fbddc85be56eca3421d246b4abe0da3366b83791e44` |

Each ZIP expands to a directory named for the target, with separate folders for each method and a transfer manifest. The archive records include configuration-level outcomes, available block reports, and run settings. Individual simulation draws and restart databases are outside the scope of these archives.

To verify a downloaded archive, compare its SHA-256 checksum with the value above. Use `sha256sum filename.zip` on Linux, `shasum -a 256 filename.zip` on macOS, or `Get-FileHash filename.zip -Algorithm SHA256` in PowerShell.

The [source checksum record](source-checksums.json) identifies the version of each script used on GREX and the version with shorter documentation.
