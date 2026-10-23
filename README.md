# S1 data and code

Numerical data and Python scripts for *A mass-uniform L1 method for nonlinear fractional networks with independent vanishing masses*, by Daniil Zatsepin.

The archive contains the solvers, reference calculations, table and figure scripts, and 199 NPZ files. The experiments cover linear and nonlinear networks, jointly vanishing masses, scalar mesh comparisons, a manufactured solution, and a CPE circuit with a finite RC comparison. For the legacy linear and C² tests, only summaries and diagnostics were retained.

## Check and reproduce

Run these commands from the repository root. The integrity check needs only Python:

```sh
python verify_archive.py
```

For the numerical checks and tables, use Python 3.12:

```sh
python -m pip install -r requirements-numerics.txt
python preparation/smoke_checks.py
python tables/extract_tables.py
```

For figures, use a separate Python 3.14 environment:

```sh
python -m pip install -r requirements-plot.txt
python preparation/plot_figures.py
```

[REPRODUCING.md](REPRODUCING.md) lists the full calculation commands, saved-data coverage, and the mapping from historical figure filenames to the four figures in the manuscript. Production and export scripts write output files, so use a separate working copy for reruns.
