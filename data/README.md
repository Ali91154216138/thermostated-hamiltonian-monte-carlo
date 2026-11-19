# Full tuning data

The three ZIP files contain the GREX tuning records for the simulations in the manuscript. Each archive covers one target. The tuning guides identify the file inside the ZIP used for each table. The 24 held-out method reports are included directly in the comparison folders in this repository.

**Availability during private preparation.** The authors have retained and verified these archives. The Zenodo record and DOI are pending. The following filenames identify the originals; they are not public download links.

| Tuning guide | Archive | Size (MiB) | SHA-256 |
|---|---|---:|---|
| [Multiscale Gaussian](../targets/multiscale-gaussian/tuning/README.md) | `THC_tuning_Multiscale_Gaussian_20261007.zip` | 40.5 | `29b01b64b62cd495e25e084ef5156d81dc0fdee289226b84809e757cd9942edf` |
| [Banana](../targets/banana/tuning/README.md) | `THC_tuning_Banana_20261007.zip` | 113.8 | `19b1f13d676d72846e64837695ba91a8325736b779e6f6aeb1d11ce22445c128` |
| [Neal’s funnel](../targets/neals-funnel/tuning/README.md) | `THC_tuning_Neals_funnel_20261007.zip` | 185.5 | `c14faeb6164bdeafcc802fbddc85be56eca3421d246b4abe0da3366b83791e44` |

Each ZIP expands to a directory named for the target, with separate folders for each method and a transfer manifest. The archive records include configuration-level outcomes, available block reports, and run settings. Individual simulation draws and restart databases are outside the scope of these archives.

To verify a downloaded archive, compare its SHA-256 checksum with the value above. Use `sha256sum filename.zip` on Linux, `shasum -a 256 filename.zip` on macOS, or `Get-FileHash filename.zip -Algorithm SHA256` in PowerShell.
