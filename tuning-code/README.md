# Tuning code

Choose the target for the instructions for each table.

- [Tables 1, 3 and 5 — Multiscale Gaussian](../targets/multiscale-gaussian/tuning/README.md)
- [Tables 7, 9 and 11 — Banana](../targets/banana/tuning/README.md)
- [Tables 13, 16 and 18 — Neal’s funnel](../targets/neals-funnel/tuning/README.md)

The `grex_*` directories contain the original method scripts, imported numerical modules, checkpoint runners and Slurm job scripts. The three target guides identify which versions produced each table. Each guide also links to the settings and results for that campaign. The original Slurm scripts refer to the authors’ GREX installation. For another machine, use the Python commands in the target guide.

## Read the archived results

The tables use the original GREX output. [extract_winners.py](extract_winners.py) reads the primary efficiency winner and counts directly from each report in an original target ZIP and checks the report checksum. It runs with the standard library in Python 3.11 or later. The [data guide](../data/README.md) identifies the three target archives. The target pages also provide copies of 15 complete reports and the original winning entries from the two largest reports. The full contents of those two reports are in their respective archives.

## Rerun a tuning experiment

The GREX environment used Linux, Python 3.12.4, NumPy 2.2.2, Numba 0.65.1 and llvmlite 0.47.0. Some version strings include the Compute Canada build suffix. Each run manifest records the versions for that method. For a run with the original checkpoint launcher, choose a fresh output directory.

Seven campaigns moved from an initial grid to a set of configurations fixed before the remaining runs. For Banana bounded-scale THC, that set was the whole grid. The original continuation worker read partial results from GREX checkpoint databases. The transferred data include the frozen configuration lists and completed reports. For each campaign, [replay_selection.py](replay_selection.py) can rerun any saved worker part from its seeds using the archived simulation, aggregation and diagnostic code. It checks the saved list and code against their original checksums. It recomputes all ten blocks for each requested configuration. Add `--check-only` to test the scientific inputs without sampling.

The archived outputs are the reference for the published numbers. As a numerical check, the Gaussian bounded-scale THC winning configuration was rerun across all ten blocks on Windows with Python 3.12.8, NumPy 2.4.6 and Numba 0.66.0. Its score was 0.01012989938. The original GREX report gives 0.01012904825, a difference of about 0.0084%. Table 1 uses the original GREX result.
