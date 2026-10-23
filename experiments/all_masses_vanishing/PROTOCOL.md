# S1: simultaneous vanishing of all three physical masses

The standalone driver reads and SHA256-hashes the legacy source at `legacy/mass3/mass3_l1_check.py`; it does not modify or import it. The production command is run from repository root, in a working copy because it writes results:

```powershell
python ./experiments/all_masses_vanishing/run_experiment.py > ./experiments/all_masses_vanishing/run.log
```

`results.json` contains the full protocol, run summaries, versions, thread settings, the original solver and experiment-script hashes, NPZ hashes, spectral diagnostics and high-precision checks. `results.csv` contains the same run rows in tabular form. `run.log` records progress. Every run has its own full NPZ trajectory, rather than only a summary of errors.

## Fixed problem data

We solve

$$
M(\delta)D_C^{1/2}u(t)+Au(t)=p_0+t p_1,\qquad 0<t\le1,
$$
$$
A=\begin{pmatrix}6&-5&0\\-5&9.3&-3\\0&-3&3.9\end{pmatrix}
=\operatorname{diag}(1,1.3,0.9)+B\operatorname{diag}(5,3)B^T,
\quad B=\begin{pmatrix}1&0\\-1&1\\0&-1\end{pmatrix},
$$
$$
u(0)=(1,-0.4,0.2)^T,\quad
p_0=(0.4,-0.2,0.3)^T,\quad p_1=(0.2,0.1,-0.15)^T.
$$

The graph, reaction coefficients, initial state and affine load are independent of all masses. The load is bounded in C²[0,1], with p''=0. The initial state is unprepared: p₀−Au₀=(−7.6,9.12,−1.68)ᵀ. All data satisfy the S1 assumptions with P=1, minimum node loss c_*=0.9 and maximum node loss c^*=1.3. The nonzero slope also exercises the background-load contribution when all physical layers vanish. No manufactured mass-dependent load is used.

Two paths are tested:

* **Fixed ratios:** M(δ)=diag(δ,δ/4,δ/16).
* **Changing ratios:** M(δ)=diag(δ,δ²,δ³).

In both paths δ∈{1,10⁻²,10⁻⁴,10⁻⁶}. Every diagonal entry tends to zero. In the second path the largest-to-smallest mass ratio reaches 10¹², while the fastest physical time scale d₃² reaches 10⁻³⁶.

## Mesh, full-history scheme, stored residual

For each physical mass dᵢ and the auxiliary background scale η₀=1, let s=√η. All selected square roots are exact rational numbers. With m∈{16,32,64,128,256,512}, form the rational parent

$$
t_j^{(\eta)}=\left(\frac{s j}{m(1+s)-j}\right)^4,\qquad j=0,\ldots,m.
$$

Take the exact `Fraction` union of the four indexed scale parents and the uniform parent j/m. Delete exact equalities only, without tolerance merging. The number N of intervals is computed from the resulting union, so m≤N≤5m−4; coincidences at δ=1 reduce N. The background parent is included separately even when a physical mass equals one.

For the full L1 history at each node,

$$
b_{nj}=\frac{1}{\Gamma(3/2)\{\sqrt{t_n-t_{j-1}}+\sqrt{t_n-t_j}\}},
\quad M\sum_{j=1}^n b_{nj}(U^j-U^{j-1})+AU^n=p(t_n).
$$

Every time difference is first formed as an exact rational difference, then rounded once to float64; the rationalized half-order formula avoids subtracting nearly equal square roots. The linear solve is a 3×3 Cholesky correction solve. Every state increment is recomputed from actually stored nodal values. The history always starts at zero; no restart, compression or history window is used.

The implementation explicitly recomputes the complete stored-history residual

$$
\zeta_*^{\rm hist}=\max_{n,i}\frac{\left|d_i\sum_{j=1}^n b_{nj}(U_i^j-U_i^{j-1})+(AU^n)_i-p_i(t_n)\right|}{c_i+d_i b_{n1}}.
$$

The denominator uses the **oldest** weight bₙ₁. The NPZ stores both the unnormalized residual vector and each normalized component at every positive node. After recomputing the full residual, the script fails the run if its normalized maximum exceeds 10⁻¹². These are floating-arithmetic diagnostics for the evaluated weights; they are not interval-certified residual bounds for an exact real-arithmetic operator.

## Independent analytical reference for the affine load

The reference does not use the L1 approximation. Let

$$
H=A^{1/2}M^{-1}A^{1/2}=Q\operatorname{diag}(\lambda_k)Q^T,
\quad y=Q^TA^{1/2}u,
$$
$$
a=Q^TA^{-1/2}p_0,\quad b=Q^TA^{-1/2}p_1,
\quad y^0=Q^TA^{1/2}u^0.
$$

Each scalar mode solves D_C^{1/2}yₖ+λₖyₖ=λₖ(aₖ+bₖt). Its closed-form solution is

$$
y_k(t)=a_k+(y_k^0-a_k)E_{1/2}(-\lambda_k\sqrt t)
+b_k t\,g(\lambda_k\sqrt t),\qquad u=A^{-1/2}Qy,
$$
$$
E_{1/2}(-x)=e^{x^2}\operatorname{erfc}(x)=\operatorname{erfcx}(x),
\qquad
g(x)=1-\frac{2}{\sqrt\pi x}+\frac{1-\operatorname{erfcx}(x)}{x^2}.
$$

Indeed, the Laplace transform of the zero-initial-data affine response is λ/[s²(√s+λ)], so its inverse is t−2√t/(λ√π)+(1−E₁/₂(−λ√t))/λ²=t g(λ√t). This also follows from λt³/²E₁/₂,₅/₂(−λ√t). The constant-load and initial terms follow directly from the scalar resolvent. Thus the affine formula has the correct load, initial condition, and memory origin.

For x<0.5, cancellation in g is avoided with its convergent Taylor series

$$
g(x)=\sum_{k=1}^{\infty}\frac{(-1)^{k-1}x^k}{\Gamma(2+k/2)}.
$$

The float64 implementation uses 44 terms; at x≤0.5 the first omitted term is below 3×10⁻³⁷ and successive terms decrease. At t=0 the initial state is assigned explicitly. For x≥0.5, SciPy `erfcx` and the displayed formula are used. The eigensystem and transformations are calculated in 80 decimal digits with mpmath before rounding to float64, avoiding loss of the slow eigenvalue when the masses have highly separated ratios. The ideal decimal A is used in the high-precision construction; the solver uses its float64 representation. This representation difference is well below the measured discretization error and is included in the reference diagnostics.

For the largest m at both endpoint values δ=1 and δ=10⁻⁶ of each path, a separate 120-digit eigensystem and scalar evaluation check the float reference at zero, the first node, the terminal node, the maximum nodal-error location, both neighboring nodes of dᵢ² times {10⁻⁶,10⁻³,1,10³,10⁶}, and the macroscopic target t=1/2. The chosen targets are clipped at T. The 120-digit `erfcx` evaluation uses direct exp(x²)erfc(x) below x=20 and truncates its asymptotic expansion at working precision above that threshold. The checked nodes and high-precision values are preserved in `results.json`. This check is not a rigorous interval enclosure over every trajectory node.

## Error quantities and orders

The nodal error is E=maxₙ‖Uⁿ−u_ref(tₙ)‖∞. The reported scaled error is E N³/². Orders use the actual union cardinality,

$$
q_N=\frac{\log(E_{\rm previous}/E)}{\log(N/N_{\rm previous})}.
$$

At every knot and at the quarter, half and three-quarter points of every exact rational cell, the piecewise-linear numerical trajectory is compared with the analytical reference. The resulting `sampled_PL_error` is explicitly a **sampled** diagnostic, not a certified continuous supremum. The error of linearly interpolating the reference itself is sampled on the same points and reported separately. NPZ files preserve the interior probe times, reference states, and PL numerical states.

## Environment and limitations

The recorded numerical environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0. Install `requirements-numerics.txt` in a Python 3.12 environment as described in `REPRODUCING.md`; an author-local virtual environment is not required. The script sets OPENBLAS, OMP, MKL, NUMEXPR, VECLIB and BLIS thread counts to one before importing NumPy/SciPy. On the recorded Windows environment, `numpy.longdouble` has the same precision as float64. No claim of higher precision for the L1 solve is made.

To regenerate the figures from saved results, use the separate plotting environment and run `python preparation/plot_figures.py` from repository root. The common exporter reads the plotting definitions in `present_results.py`; that module is not a standalone command. This study produces the historically named output `figures/Fig3.{png,eps}`, which is main Figure 1 and is copied to `Fig1.{png,eps}` in the submission package.

Finite selected paths and masses illustrate the proven theorem and do not themselves establish uniform convergence over an unbounded parameter family. The reference and residual diagnostics are not interval certificates. No speedup or uniform bit-complexity claim follows from this experiment.
