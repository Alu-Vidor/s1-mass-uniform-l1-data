# Scalar fractional-order and matched-mesh supplement

Final production data comprise 90 solves. The standalone driver supplements the graph experiments with scalar validation at two orders other than one half.

## Problem and budgets

The problem is

$$
dD_C^\alpha u+u=1/5,\qquad u(0)=1,\qquad 0\le t\le1.
$$

The initial value and forcing are independent of the mass, and the initial value is unprepared. Orders are exactly `3/10` and `7/10`; masses are exactly `1`, `1/10000`, and `1/100000000`. Each of the six cases uses `m=16,32,64,128,256`.

For every case and level, the union uses the physical mass parent, the auxiliary unit parent, and the uniform `m`-parent. Two baselines use exactly its actual interval count `N`: a uniform grid, and the standard graded grid `t_j=(j/N)^((2-alpha)/alpha)`. The counts are `N=2m-1` for unit mass and `N=3m-2` for the two smaller masses. Each matched triple therefore has the same `N(N+1)/2` scalar history weights and the same number of scalar implicit steps. These counts do not equate arithmetic bit costs or reference costs. Saved per-run times include quadrature and high-precision checks; they are not a timing benchmark.

## Exact algebraic specification and certified rational rounding

Write `alpha=p/10`, where `p=3` or `7`. With rational `s=sqrt(eta)`, a parent satisfies

$$
(t_j^{(\eta)})^p=\left(\frac{s j}{m(1+s)-j}\right)^{20}.
$$

The uniform parent's `p`th powers are `(j/m)^p`. These rational powers decide exact equality and order. An integer `p`th root gives a rigorous dyadic enclosure of every target node with width `2^-512`; its two bounding inequalities are checked with integer arithmetic. A positive rational lower bound `gamma` on all successive exact-union gaps is obtained from these enclosures. This is stronger than the parent-gap condition in Appendix B of the manuscript.

Parent nodes are rounded to a common dyadic grid with step at most `gamma/32`, so nearest rounding has error at most `gamma/64`. The recorded enclosure upper bound is verified to be at most `gamma/16` using exact fractions. The uniform parent remains exact because `m` is a power of two and divides the dyadic denominator. Equal exact nodes receive the same rounded value; strict distinct nodes remain distinct. Applying the deterministic rounding before taking the rational union gives the saved grid. No tolerance merging is used. The standard graded baseline similarly has exact rational `p`th powers `(j/N)^(20-p)` and receives certified dyadic rounding; the uniform baseline is the exact rational grid `j/N`.

Every trajectory saves integer node numerators and a common integer denominator. Grid JSON files additionally save the exact algebraic keys, the rational gap bound, the rational rounding-error upper bound, and the integer precision. The independent audit verifies the rounding enclosure of every saved node by exact rational power inequalities. The largest recorded error-bound/gamma ratio is `0.015435768437241348`, below `1/16`.

## L1 evaluation and residual

The driver imports no old solver. All node and interval differences are formed as exact integer differences before conversion to binary64. For `a=t_n-t_(j-1)`, `b=t_n-t_j`, `h=a-b`, the L1 weight is

$$
b_{nj}=\frac{a^{1-\alpha}-b^{1-\alpha}}{\Gamma(2-\alpha)h}.
$$

When `r=h/a<1/2`, its numerator ratio is evaluated with `-expm1((1-alpha)*log1p(-r))/r`, multiplied by `a^(-alpha)`. For `r>=1/2`, `a` and `b` are converted from independent exact differences and the direct power difference is used. This prevents both subtraction of nearly equal powers in remote intervals and loss of a very small right distance through forming `1-r`.

Each scalar linear step is solved in correction form. The final residual is recomputed from the complete, actually stored increment history. Raw vectors, normalized residuals, and oldest weights are saved. The normalization is exactly `1+d*b_(n1)`. The maximum production normalized residual is `5.1133158999449935e-14`; the maximum raw residual is `1.489514067642972e-11`. These ordinary floating-point diagnostics are not interval residual enclosures.

## Independent reference

The exact solution is `1/5+(4/5)*E_alpha(-t^alpha/d)`. The production reference uses the positive spectral representation, independently of L1. Put `s=t/d^(1/alpha)` and `y=log(r)`. The reference integrand for the relaxation factor is

$$
e^{-\exp(\log s+y)}\frac{\sin(\pi\alpha)}{\pi}
\frac{e^{-|\alpha y|}}{1+2\cos(\pi\alpha)e^{-|\alpha y|}+e^{-2|\alpha y|}}.
$$

SciPy adaptive quadrature integrates over `[-50/alpha,50/alpha]`, split at zero and at the exponential transition, with absolute and relative targets `5e-14` per segment. The two omitted density tails, including the solution amplitude, are bounded by

$$
\frac{(8/5)\sin(\pi\alpha)}{\pi\alpha}
\frac{e^{-50}}{1-2e^{-50}}<3\times10^{-22}.
$$

The reported maximum quadrature error estimate is `7.668923625021217e-14`; the adaptive quadrature estimates are not rigorous enclosures. The inner exponential cutoff in binary64 lies below its representable range.

The driver checks the first positive node, the nodal error maximum and sampled reconstruction error maximum of each run using mpmath quadrature with 65 decimal digits of working precision and tail exponent 170. Repeated points are cached. There are 221 recorded checks, with maximum discrepancy `1.1102230246251565e-15` after conversion to binary64. These are finite checks, not a uniform enclosure or a claim of 65 correct digits.

The audit independently evaluates 12 points using the defining Mittag-Leffler series and six terminal points using de Hoog complex Laplace inversion at degree 100, with 90 decimal digits of working precision. Their maximum discrepancy from the saved spectral reference is `9.992007221626409e-16`.

Reference evaluation uses binary64 conversions of exact rational times. Positivity and unit total mass of the spectral density imply `t*|u'(t)|<=0.8/e`; the recorded relative time-conversion errors therefore bound the corresponding reference change. The largest such bound is `3.249755152100747e-17`.

## Errors and probes

`E_node` is the maximum over every saved node. `E_PL_sampled` is the maximum over every node, every cell quarter/mid/three-quarter point, 97 logarithmic points from `1e-8*rho` to `1e8*rho` clipped to `(0,1]`, and another 129 geometric points joining `rho` to the first positive node when `rho<t_1`. Here `rho=d^(1/alpha)`. The geometric bridge is needed when a baseline's first cell spans many decades beyond the layer. All probe times, numerical reconstructions and reference values are saved. These sampled maxima are lower bounds on continuous maximum errors; they are not certified suprema.

For every individual refinement series, the rate is `log(E_previous/E_current)/log(N_current/N_previous)`. The rates in the final six-row table use union nodal errors from `m=128` to `256`. The target slopes in main Figure 3 (historically named output `figures/Fig4.{png,eps}`) are prescribed `-(2-alpha)`, without fitting.

## Audit and reproducibility

`audit_study.py` imports no production solver and performs no reintegration. It checks all 90 trajectory lengths, all saved maxima, all exact rounding enclosures and all 30 matched-budget triples. It independently reconstructs the original full L1 weights by direct power differences in 90-digit arithmetic at 109 listed nodes across all 18 finest trajectories. The maximum normalized residual there is `5.1209177738205414e-14`; the largest discrepancy from the stored normalized residual is `2.8360364015557293e-16`. This remains a finite-node audit.

From repository root, in the numerical environment described in `REPRODUCING.md`. Use a working copy because these commands write results and audit reports:

```text
python experiments/scalar_mesh_comparison/run_study.py
python experiments/scalar_mesh_comparison/audit_study.py
```

The recorded final numerical run took about 48 seconds including reference checks. It used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0 on Windows 11. In the separate plotting environment, run the common exporter from repository root:

```text
python preparation/plot_figures.py
```

It reads the plotting definitions in `present_study.py`, which is not a standalone command. The scalar output is `figures/Fig4.{png,eps}`; its identical submission copy is named `Fig3.{png,eps}`.

`PROVENANCE.json` records the numerical driver and original source/data hashes. Its manuscript-fragment hash identifies a historical snapshot checked during the numerical run; that manuscript fragment is not a public solver dependency. `AUDIT.json`, `PRESENTATION_PROVENANCE.json` and `MANIFEST.json` retain historical checks and exports. The public release is checked by running `python verify_archive.py` from repository root; its `CHECKSUMS.json` is authoritative for the distributed file bytes.

An initial attempt stopped on a Windows checkpoint write error. Checkpoint writes were made atomic with retry; a subsequent run completed. Code review then improved the general weight branch and the reconstruction probes before the final 90-run production freeze. The interrupted-run log and provisional arrays are not part of the public release; the archived arrays contain the final complete run.
