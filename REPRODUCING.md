# S1 public numerical archive: reproduction notes

Run all commands from the repository root.

## Environments

The archived calculations used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0
and mpmath 1.3.0 on Windows. `longdouble` equals `float64` in that environment.
Create a numerical environment and install `requirements-numerics.txt`.
The separate artwork environment uses NumPy 2.5.4, Matplotlib 3.11.2 and
Pillow 13.0.0, recorded in `requirements-plot.txt`.

Examples, using the Python executable in the selected environment:

```sh
python -m pip install -r requirements-numerics.txt
```

`CHECKSUMS.json` records the size and SHA-256 of each distributed file.
`provenance/` distinguishes the original producer hashes from portable
source adaptations. Per-study metadata describe the saved calculations;
their producer hashes can differ from the current portable source files.

## Read saved results without time integration

```sh
python tables/extract_tables.py
python legacy/mass3/check_mass3.py
python experiments/all_masses_vanishing/audit/independent_audit.py
python experiments/scalar_mesh_comparison/audit_study.py
python experiments/nonlinear_reference/audit_study.py
python experiments/nonlinear_reference/independent_audit.py
```

The table extractor regenerates the 31 linear, 15 nonlinear refinement and
five manufactured L1 table rows from archived results. The legacy check
re-evaluates scalar matrix references at high precision, without time stepping.
The remaining audits inspect stored trajectories, exact-grid keys and
full-history residuals. They update their own reports, and the extractor
updates its CSV/check report. Use a working copy when preserving a release
checkout byte for byte matters.

The portable read-only numerical smoke command is
`python preparation/smoke_checks.py`. It checks seven small study families,
all 199 archived NPZ files for finite arrays, and the manufactured L1
fixture against a stored trajectory. It does not modify archived outputs.

The CPE audit also solves a separate finite RC ODE and performs Laplace
inversion. It is a reference check, rather than a saved-data-only operation:

```sh
python experiments/cpe_network/audit_study.py
```

## Regenerate the five figures from saved results

In the presentation environment:

```sh
python -m pip install -r requirements-plot.txt
python preparation/plot_figures.py
```

Outputs are `figures/Fig1.{png,eps}`, `Fig3`, `Fig4`, `Fig5`, and `Fig6`,
plus figure-generation provenance. These historical asset names are retained
by the public exporter. The image files are generated locally, rather than
distributed in the release. The submission package uses renamed
copies with identical image bytes:

| Printed figure | Public exporter output | Submission asset |
|---|---|---|
| Figure 1 | `figures/Fig3.{png,eps}` | `Fig1.{png,eps}` |
| Figure 2 | `figures/Fig5.{png,eps}` | `Fig2.{png,eps}` |
| Figure 3 | `figures/Fig4.{png,eps}` | `Fig3.{png,eps}` |
| Figure 4 | `figures/Fig6.{png,eps}` | `Fig4.{png,eps}` |

The exporter also produces the historical `Fig1` legacy plot, which is not
used in the current manuscript. The historical `Fig2` diagnostic is excluded
from the manuscript and the public exporter. Historical asset names are
independent of the current manuscript's figure numbering; there is no
separate figure supplement.

The `present_*.py` files are plotting definitions used by this exporter.
Their former article-text/table writers and local-virtual-environment
guards are excluded. Invoke the common exporter rather than these snippets.

## Fresh calculations

Fresh calculations write new numerical files. Run them in a separate working
copy, then compare with the archived tables. Finite refinements and ordinary
floating arithmetic do not provide interval-certified error bounds.

Legacy linear and C²-edge studies:

```sh
python legacy/mass3/mass3_l1_check.py --output legacy/mass3/repeat_mass3_l1_check.json
python legacy/mass3/extend_mass3.py
python legacy/mass3/check_mass3.py
python tables/extract_tables.py
```

This first script produces the historical levels, the extension adds the
refinements used in the article, and the checker recalculates the reference
diagnostics. The archived legacy records contain summary errors, residuals
and iterations. Full refined legacy trajectories were not retained.
In the original legacy output, `N = 4*m` is a mesh-construction budget and
`K` is the actual number of intervals. The extended records and extracted
tables use `B3` for that budget and `N_actual` for the interval count.

Manufactured L1 test, into a directory that does not already exist:

```sh
python legacy/manufactured/run_manufactured.py --out fresh-manufactured --levels 20 40
```

Omit `--levels` for 20, 40, 80, 160 and 320. The driver also runs a finest-grid
tolerance control. The public kernel includes only the L1 dependency closure,
and its five-node model fixture excludes unrelated block/PL solvers.
The historical PI2 comparison is excluded from this S1 publication archive.

Joint mass limits:

```sh
python experiments/all_masses_vanishing/run_experiment.py --smoke
python experiments/all_masses_vanishing/run_experiment.py
```

The smoke command runs four small trajectories for the smallest scale and
writes `smoke.*` files. The full command produces 48 trajectories.

Scalar matched meshes:

```sh
python experiments/scalar_mesh_comparison/run_study.py --pilot
python experiments/scalar_mesh_comparison/run_study.py
```

`--pilot` checks spectral references only. The full command produces
90 trajectories: two fractional orders, three masses, five grid levels,
and three mesh constructions at matched interval counts.

Nonlinear cubic system and independent Volterra reference:

```sh
python experiments/nonlinear_reference/run_study.py
python experiments/nonlinear_reference/compare_reference.py
python experiments/nonlinear_reference/validate_reference.py
python experiments/nonlinear_reference/cross_grade_check.py --recompute
python experiments/nonlinear_reference/audit_study.py
python experiments/nonlinear_reference/independent_audit.py
```

The L1 stage has 20 runs. The comparison generates 12 Volterra reference
levels if they are absent, and otherwise reuses and checks the existing
archives. The cross-grading command above recomputes four independent refinements.
`run_study.py --pilot` produces only the reduced L1 pilot and does not provide
the complete inputs required by the subsequent full comparison.

CPE/RC study:

```sh
python experiments/cpe_network/run_study.py
python experiments/cpe_network/audit_study.py
```

The parameter specification is
`experiments/cpe_network/PARAMETERS.json`. It gives the primary
citation, factual component values, scaling and new network choices. The
third-party paper, extracted text and page images are not redistributed.

## Saved-state scope

The repository contains 199 NPZ archives: 48 joint-mass runs, 90 scalar
comparisons, 20 nonlinear L1 runs, 12 Volterra reference levels, four
cross-grading trajectories, 18 CPE L1 runs, one CPE/RC comparison, and six
manufactured L1 trajectories including the tolerance control. Legacy
linear/C² refinements have summary data, as stated above.

All NPZ archives are byte-identical to the source dataset. Numeric JSON
leaves were checked to remain unchanged during path normalization. For
the manufactured study, the public metadata/tables explicitly select the
L1 records and omit the separate PI2 comparison.
