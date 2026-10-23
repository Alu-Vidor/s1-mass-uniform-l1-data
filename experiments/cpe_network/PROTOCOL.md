# CPE circuit benchmark

The source component values and the new circuit design are documented in
`PARAMETERS.json` in this directory
(`experiments/cpe_network/PARAMETERS.json` from repository root).
The published base CPE scale is 56 microF s^(-1/2), order one half.
The network graph, resistor links, shunts, scaling factors and current input
are our design choices. No measurements of this new network are supplied.

Each node has a CPE and a 1 kohm shunt to ground. Links 1--2 and 2--3 are
2 kohm. CPE admittances are scaled by (1,1/4,1/16), implemented in the
published RC circuit by C -> s C, R -> R/s. Zero voltage and zero prehistory
are followed by a constant (10 microA,0,0) current. G0=1mS, V0=10mV,
t0=T=0.050176s give exactly d=(1/4,1/16,1/64) on [0,1].

Production levels m=16,32,64,128,256,512 use the exact rational union of
three mass parents, the unit background parent, and a uniform parent.
For each resulting actual count N, comparison grids are j/N and (j/N)^3.
All 18 L1 solutions retain full history. Full residual vectors and exact
rational node numerators/denominators are stored with the state arrays.
There are no tolerance-based node merges.

The ideal fractional reference uses the symmetric generalized spectral
formula with erfcx. At first nodes and nodal/quarter/log-probe error maxima,
it is checked in 80 and 110 decimal digits. Direct inversion of the original
matrix Laplace transform, without diagonalization, supplies four independent
checks. Sampled reconstruction errors include all nodes, cell quarter-points
and 801 global logarithmic times in [1e-12,1]. The global probes were added
after the first run to check resolution inside a uniform grid's first cell;
all 18 solves were rerun with this final sampling protocol. These final
arrays and metrics, rather than the exploratory outputs, are archived.

The model comparison assembles all 33 states of the scaled published
finite RC realization: three node parallel capacitors and ten internal
series-branch capacitor states per node. A constant-input matrix exponential
uses a symmetric eigensystem in 60/90 decimal digits. Independent SciPy
Radau integration checks it. Ideal and finite RC voltages are compared at
zero and 1601 log times from 1e-10s to T. Their gap is a modelling discrepancy,
not an L1 error. The time window is finite, and the emulator's shunt resistor
creates a different eventual DC limit. Its theoretical frequency phase band
does not certify a switched transient. Neither sampled metric is a certified
continuous supremum.

Run from repository root with the numerical environment described in
`REPRODUCING.md`. Use a working copy because these commands write results:

```text
python experiments/cpe_network/run_study.py
python experiments/cpe_network/audit_study.py
```

In the separate plotting environment, run the common exporter from
repository root:

```text
python preparation/plot_figures.py
```

It reads saved result arrays and the plotting definitions in `present_study.py`
without repeating time integration. The CPE output is `figures/Fig6.{png,eps}`,
used as Figure S2 in the Online Resource and copied into the submission
package as `ESM_FigS2.{png,eps}`. The presentation module is not a standalone
command.
