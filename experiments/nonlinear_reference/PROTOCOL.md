# Nonlinear network with an independent reference

The study design was fixed before production runs. Pilot runs use only
levels 16 and 32 to check implementation; their output is not a result archive.

We solve the autonomous Caputo system on [0,1], with common order alpha=0.3
or 0.7. The incidence matrix is the chain 1--2--3, with columns (1,-1,0)
and (0,1,-1). Node losses are (1,1.3,0.9), and edge currents are
phi_e(z)=w_e(z+0.1 z^3), w=(5,3). The initial vector (1,-0.5,0.7) and
constant forcing (0.2,-0.1,0.3) are the same in every run and are not
chosen from a target solution. The masses are delta*(1,1/4,1/16), with
delta=1 and 1/100. All edge laws are globally monotone and C-infinity;
there is no additional nonlinear node reaction.

Primary L1 levels are m=16,32,64,128,256. The grid contains one parent for
each distinct physical mass, the background parent eta=1, and the uniform
m-parent. Algebraic node keys are ordered exactly, enclosed by integer
root inequalities, rounded onto a common dyadic grid with a verified
rounding-error bound below gamma/16, and unioned exactly. No close-node
tolerance is used. Full-history residuals are recomputed from the stored
increments and normalized using the oldest, rather than newest, weight.

The independent reference discretizes the Volterra integral equation
u=u0+I^alpha M^{-1}(p-F(u)), by implicit piecewise-linear product
integration of its right-hand side. This is a fractional Adams--Moulton
method, not a fine-grid L1 derivative formula. Its power mesh is independent
of the L1 union. The initial design specified K=2048,4096,8192, with
further refinement if discrepancies exceeded one per cent of the finest
L1 error. The final production levels are recorded below. Refinement is a
reference validation criterion, not an assumption of a certified error
bound. The complete reference implementation and validation outcomes are
included in this archive.

Pilot adjustment before reference production: the grade 2/alpha led to a
4096-versus-2048 discrepancy of 4.55e-4 for the more stiff alpha=0.3 case.
Grade 4/alpha reduced that discrepancy to 1.39e-5. Production therefore
uses grade 4/alpha and K=8192,16384,32768. Every L1 node from all five
levels, and all sampled reconstruction probes from the finest L1 level,
are evaluated on precisely the same set at all three reference levels.
Sampled reconstruction errors are reported at the finest L1 level only;
nodal errors and rates are reported at all five levels.

Primary reported quantities are the maximum nodal error relative to the
independent reference, and a sampled maximum error of the piecewise-linear
L1 trajectory. The latter uses all L1 nodes, three internal points per cell,
and fixed logarithmic probes about every mass scale d_i^(1/alpha).
Reference refinements are compared on these same probes. Rates use the
actual L1 interval count N. We do not call a finite sampled maximum the
continuous supremum or an interval certificate. Nonlinear solver residuals,
reference discrepancies, full trajectories, exact grid metadata, code and
environment versions are retained.

No universal advantage or physical calibration is claimed by this
dimensionless experiment. The study tests the previously missing combination
of nonlinear coupling, unprepared data, non-half orders and independently
computed solution values.

## Reproduction

Run the commands in the nonlinear-study section of `REPRODUCING.md` from
repository root, in a separate working copy when recomputing results.
They distinguish the L1 solves, Volterra references, refinement checks
and audits. The `--pilot` L1 run is insufficient for the full reference
comparison and does not replace the production archive.

For figures, run `python preparation/plot_figures.py` from repository
root in the separate plotting environment. It reads saved results and
the plotting definitions in `present_study.py`; the presentation module
is not a standalone command. The historical output is
`figures/Fig5.{png,eps}`, used as main Figure 2 and copied to
`Fig2.{png,eps}` in the submission package.
