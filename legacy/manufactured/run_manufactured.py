"""S1 five-node manufactured L1 experiment.
L1 calculations and checks are extracted from the historical driver."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
for _thread_variable in THREAD_VARIABLES:
    os.environ[_thread_variable] = '1'
import numpy as np
import scipy
import graph_full
from serial_pl import finite_block as f
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'results' / 'manufactured_nonlinear'
DEFAULT_LEVELS = (20, 40, 80, 160, 320)
METHODS = ('L1',)

class ManufacturedCase:
    """A callable-load fixture for existing low-level full-system entries.

    F and its Jacobian are the production five-node tanh model.  force(t)
    depends only on the prescribed exact solution and independent edge sums;
    it never calls the production F or a computed numerical trajectory.
    """

    def __init__(self):
        self.base = f.Model()
        self.alpha = 0.5
        self.T = 1.0
        self.linear = False
        self.n = 5
        self.beta = 0.6
        self.masses = self.base.mass.copy()
        self.u0 = self.base.u0.copy()
        self.losses = self.base.losses.copy()
        self.scale = self.base.scale.copy()
        self.c_min = float(np.min(self.losses))
        self.direction = np.array([0.8, -0.6, 0.45, 0.7, -0.5])
        self.edges = ((3, 0), (0, 1), (1, 2), (2, 4), (0, 2), (1, 4))
        self.conductances = (1.0, 1.0, 1.0, 1.0, 0.7, 0.5)
        self.caputo_factor = math.gamma(4.0) / math.gamma(4.0 - self.alpha)
        expected = ((self.masses, [0.02, 0.002, 0.0002, 1.0, 1.0]), (self.u0, [0.8, -0.5, 0.6, 0.1, 0.2]), (self.losses, [0.5, 0.5, 0.5, 1.5, 1.5]), (self.scale, np.ones(5)))
        if any((not np.array_equal(a, b) for a, b in expected)) or self.base.beta != self.beta:
            raise RuntimeError('The production base model changed; review the prescribed case')

    def exact(self, times):
        t = np.asarray(times, dtype=float)
        return self.u0 + t[..., None] ** 3 * self.direction

    def exact_caputo(self, times):
        t = np.asarray(times, dtype=float)
        return self.caputo_factor * t[..., None] ** (3.0 - self.alpha) * self.direction

    def independent_F(self, state):
        """Componentwise constitutive law, without base.F/A0/inc products."""
        u = np.asarray(state, dtype=float)
        if u.shape != (5,) or not np.all(np.isfinite(u)):
            raise ValueError('A finite five-component state is required')
        out = self.losses * u
        for (a, b), g in zip(self.edges, self.conductances):
            z = float(u[a] - u[b])
            flux = g * (z + self.beta * math.tanh(z))
            out[a] += flux
            out[b] -= flux
        return out

    def force(self, time):
        t = float(time)
        if not math.isfinite(t) or not 0.0 <= t <= self.T:
            raise ValueError('Manufactured load time must be in [0,T]')
        return self.masses * self.exact_caputo(t) + self.independent_F(self.exact(t))

    def F(self, state):
        return self.base.F(state)

    def jacobian(self, state):
        return self.base.J(state)

    def metadata(self):
        return dict(alpha=self.alpha, T=self.T, beta=self.beta, masses=self.masses.tolist(), a=self.u0.tolist(), b=self.direction.tolist(), losses=self.losses.tolist(), scale=self.scale.tolist(), edges=[list(e) for e in self.edges], conductances=list(self.conductances), exact_solution='u*(t)=a+b*t^3', exact_Caputo='Gamma(4)/Gamma(4-alpha)*b*t^(3-alpha)', forcing='M*D_C^alpha u* + componentwise edge sum at u*', prepared_initial_data=True, public_entry_scope='Callable low-level full kernels; Model is unmodified')

def _require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def _half_l1_weights(times, n):
    """Independent stable half-order L1 weights, oldest to newest."""
    distances = times[n] - times[:n + 1]
    return 2.0 / (math.sqrt(math.pi) * (np.sqrt(distances[:-1]) + np.sqrt(distances[1:])))

def _weighted_sum(weights, values):
    """Independent scalar summation for diagnostics, not a stepping kernel."""
    return np.array([math.fsum((float(w) * float(x) for w, x in zip(weights, column))) for column in np.asarray(values).T])

def linear_interpolant_error(case, times, values):
    """Exact extrema candidates of PL states minus the prescribed cubic.

    On a cell the error derivative is slope-3*b_i*t^2.  Thus endpoints and
    the admissible nonnegative square root exhaust the extrema on [0,1].
    Numerical evaluation of these finitely many candidates is not certified.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.shape != (len(t), 5) or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError('An increasing five-state trajectory from zero is required')
    _require(np.all(np.isfinite(y)), 'Nonfinite interpolant values')
    error = float(np.max(np.abs(y - case.exact(t))))
    for n, h in enumerate(np.diff(t)):
        slope = (y[n + 1] - y[n]) / h
        for i, b in enumerate(case.direction):
            if b == 0.0:
                continue
            ratio = float(slope[i] / (3.0 * b))
            if ratio >= 0.0:
                point = math.sqrt(ratio)
                if t[n] < point < t[n + 1]:
                    state = y[n, i] + slope[i] * (point - t[n])
                    target = case.u0[i] + b * point ** 3
                    error = max(error, abs(float(state - target)))
    return error

def l1_diagnostics(case, times, values, force, exact):
    """Full substitution and stored-history residuals in G17 units.

    The global comparison budget uses losses+mass*w_oldest, not the current
    diagonal weight.  Production F is deliberately used in the exact-state
    substitution so independent-load assembly differences are included.
    """
    shape = values.shape
    exact_derivative = case.exact_caputo(times)
    exact_discrete = np.zeros(shape)
    pure_defect = np.zeros(shape)
    substitution = np.zeros(shape)
    residual = np.zeros(shape)
    denominator = np.empty(shape)
    denominator[0] = case.losses
    for n in range(1, len(times)):
        w = _half_l1_weights(times, n)
        exact_discrete[n] = _weighted_sum(w, np.diff(exact[:n + 1], axis=0))
        pure_defect[n] = case.masses * (exact_discrete[n] - exact_derivative[n])
        substitution[n] = case.masses * exact_discrete[n] + case.F(exact[n]) - force[n]
        derivative = _weighted_sum(w, np.diff(values[:n + 1], axis=0))
        residual[n] = case.masses * derivative + case.F(values[n]) - force[n]
        denominator[n] = case.losses + case.masses * w[0]
    arrays = dict(exact_caputo=exact_derivative, exact_L1_derivative=exact_discrete, pure_derivative_force_defect=pure_defect, exact_substitution_raw=substitution, exact_substitution_normalized=substitution / denominator, actual_equation_residual_raw=residual, actual_equation_residual_normalized=residual / denominator, residual_denominator=denominator)
    record = dict(pure_derivative_force_defect_max=float(np.max(np.abs(pure_defect))), exact_substitution_raw_max=float(np.max(np.abs(substitution))), G17_consistency_budget=float(np.max(np.abs(substitution / denominator))), independent_actual_state_residual=float(np.max(np.abs(residual / denominator))), normalization='losses+mass*oldest_L1_weight', consistency_units='state after G17 normalization; raw defect has force units', exact_arithmetic_global_comparison_applicable=True, consistency_evaluation_is_interval_certified=False)
    return (record, arrays)

def solve_level(case, method, n, tol=1e-12):
    """Call one existing solver and return diagnostics plus saved arrays."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('At least two uniform cells are required')
    if isinstance(tol, bool) or not math.isfinite(float(tol)) or tol <= 0.0:
        raise ValueError('A finite positive tolerance is required')
    times = np.linspace(0.0, case.T, n + 1)
    exact = case.exact(times)
    force = np.array([case.force(t) for t in times])
    sol = graph_full.l1_solve(case, times, tol=tol)
    values = sol['values']
    sources = force - np.array([case.F(u) for u in values])
    residuals = sol['residuals']
    actual = residuals['max_normalized_full_residual']
    consistency, arrays = l1_diagnostics(case, times, values, force, exact)
    solver_counters = dict(sol['stats'])
    solver_counters['arithmetic'] = 'float64; supplied uniform float grid'
    E_node = float(np.max(np.abs(values - exact)))
    E_linearinterp = linear_interpolant_error(case, times, values)
    _require(np.all(np.isfinite(values)) and np.all(np.isfinite(sources)), 'Nonfinite trajectory')
    _require(np.array_equal(values[0], case.u0), 'Initial state mismatch')
    _require(actual <= tol, f'{method}: actual algebraic tolerance was not met')
    _require(E_linearinterp >= E_node, 'Continuous interpolant maximum missed a node')
    record = dict(method=method, n=int(n), nodes=int(n + 1), algebraic_tol=float(tol), E_node=E_node, E_linearinterp=E_linearinterp, solver_actual_state_residual=float(actual), solver_residuals=residuals, solver_counters=solver_counters, **consistency)
    if method == 'L1':
        budget = record['G17_consistency_budget']
        algebraic = max(actual, record['independent_actual_state_residual'])
        scale = max(1.0, float(np.max(np.abs(values))), float(np.max(np.abs(force) / case.losses)))
        slack = 512.0 * np.finfo(float).eps * scale
        record.update(G17_error_estimate=budget + algebraic, G17_rounding_check_slack=slack, G17_comparison_check=bool(E_node <= budget + algebraic + slack), G17_estimate_is_interval_certified=False)
        _require(record['G17_comparison_check'], 'L1 error exceeded the computed G17 comparison budget')
    arrays.update(times=times, values=values, exact=exact, forcing=force, sources=sources, exact_error=values - exact)
    _require(all((np.all(np.isfinite(a)) for a in arrays.values())), 'Nonfinite saved diagnostic')
    return (record, arrays)

def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')

def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value) + '\n', encoding='utf-8')

def _save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _source_hashes():
    paths = (Path(__file__).resolve(), Path(graph_full.__file__).resolve(), Path(f.__file__).resolve())
    return {path.relative_to(HERE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

def _dense_extrema_check(case, arrays, E_linearinterp):
    t, values = (arrays['times'], arrays['values'])
    query = np.unique(np.concatenate([np.linspace(a, b, 2001) for a, b in zip(t[:-1], t[1:])]))
    sampled = np.column_stack([np.interp(query, t, values[:, i]) for i in range(5)])
    dense = float(np.max(np.abs(sampled - case.exact(query))))
    slack = 128.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    _require(dense <= E_linearinterp + slack, 'Dense check exceeded the continuous extrema result')
    _require(E_linearinterp - dense <= 1e-06 * E_linearinterp + slack, 'Dense check disagreed with the extrema result')
    return dict(analytic_candidates=E_linearinterp, independently_sampled=dense, query_nodes=len(query), checked_only_at_coarsest_level=True)

def run_benchmark(out=DEFAULT_OUT, levels=DEFAULT_LEVELS, tol=1e-12, control_tol=1e-13):
    """Write one fresh reproducible experiment, refusing existing output paths."""
    levels = tuple(levels)
    if len(levels) < 2 or any((isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 for n in levels)) or any((b != 2 * a for a, b in zip(levels, levels[1:]))):
        raise ValueError('At least two successively doubled integer levels are required')
    if not (0.0 < control_tol < tol and math.isfinite(tol) and math.isfinite(control_tol)):
        raise ValueError('Finite 0 < control_tol < tol required')
    out = Path(out).resolve()
    _require(not out.exists(), f'Refusing to overwrite an existing experiment: {out}')
    case = ManufacturedCase()
    source_hashes = _source_hashes()
    force_identity_error = max((float(np.max(np.abs(case.independent_F(case.exact(t)) - case.F(case.exact(t))))) for t in (0.0, 0.17, 1.0)))
    _require(force_identity_error < 2e-14, 'Independent edge assembly disagrees with production F')
    out.mkdir(parents=True)
    protocol = dict(case=case.metadata(), levels=list(levels), mesh='uniform np.linspace(0,T,n+1)', primary_tol=tol, finest_control_tol=control_tol, methods=dict(L1='graph_full.l1_solve callable fixture'), E_node='max over all five coordinates and all stored time nodes', E_linearinterp='continuous maximum of linear interpolation of stored states minus cubic', G17_budget='max exact full substitution/(losses+mass*oldest L1 weight)', arithmetic_certified=False, physical_error_certified=False, performance_experiment=False, source_sha256=source_hashes)
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__, thread_environment={name: os.environ[name] for name in THREAD_VARIABLES}, argv=sys.argv, independent_force_identity_error=force_identity_error)
    _save_json(out / 'PROTOCOL.json', protocol)
    _save_json(out / 'ENVIRONMENT.json', environment)
    _save_json(out / 'SOURCE_SHA256.json', source_hashes)
    rows = []
    saved = {}
    dense_checks = {}
    for method in METHODS:
        previous = None
        for n in levels:
            row, arrays = solve_level(case, method, n, tol)
            row['p_node'] = None if previous is None else math.log2(previous['E_node'] / row['E_node'])
            row['p_linearinterp'] = None if previous is None else math.log2(previous['E_linearinterp'] / row['E_linearinterp'])
            if previous is not None:
                _require(row['E_node'] < previous['E_node'], f'{method}: nodal error did not decrease')
            if n == levels[0]:
                dense_checks[method] = _dense_extrema_check(case, arrays, row['E_linearinterp'])
            name = f'{method.lower()}_n{n}.npz'
            np.savez_compressed(out / name, **arrays)
            row['trajectory_file'] = name
            rows.append(row)
            saved[method, n] = (row, arrays)
            previous = row
            print(f"{method:6s} n={n:4d} E_node={row['E_node']:.12e} p={row['p_node']}", flush=True)
    convergence = []
    accuracy = []
    for n in levels:
        l1, _ = saved['L1', n]
        convergence.append(dict(n=n, E_node_L1=l1['E_node'], p_L1=l1['p_node']))
    for row in rows:
        accuracy.append(dict(method=row['method'], n=row['n'], algebraic_tol=row['algebraic_tol'], E_node=row['E_node'], p_node=row['p_node'], E_linearinterp=row['E_linearinterp'], p_linearinterp=row['p_linearinterp'], solver_actual_state_residual=row['solver_actual_state_residual'], independent_actual_state_residual=row['independent_actual_state_residual'], pure_derivative_force_defect_max=row.get('pure_derivative_force_defect_max'), exact_substitution_raw_max=row['exact_substitution_raw_max'], G17_consistency_budget=row.get('G17_consistency_budget'), G17_error_estimate=row.get('G17_error_estimate'), G17_comparison_check=row.get('G17_comparison_check')))
    controls = []
    for method in METHODS:
        row, arrays = solve_level(case, method, levels[-1], control_tol)
        primary, primary_arrays = saved[method, levels[-1]]
        difference = float(np.max(np.abs(arrays['values'] - primary_arrays['values'])))
        _require(difference < 0.001 * primary['E_node'], f'{method}: finest tolerance materially affects discretization error')
        name = f'{method.lower()}_n{levels[-1]}_tol_control.npz'
        np.savez_compressed(out / name, **arrays)
        row.update(primary_tol=tol, state_difference_from_primary=difference, difference_over_primary_error=difference / primary['E_node'], trajectory_file=name)
        controls.append(row)
        print(f'{method:6s} finest tol control: max state difference={difference:.12e}', flush=True)
    _require(_source_hashes() == source_hashes, 'Source files changed during the experiment')
    _save_csv(out / 'CONVERGENCE.csv', convergence)
    _save_csv(out / 'ACCURACY.csv', accuracy)
    summary = dict(protocol=protocol, environment=environment, convergence=convergence, diagnostics=rows, finest_tolerance_controls=controls, continuous_extrema_checks=dense_checks, all_runtime_checks_passed=True)
    _save_json(out / 'SUMMARY.json', summary)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    _save_json(out / 'RESULT_SHA256.json', manifest)
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--levels', type=int, nargs='+', default=DEFAULT_LEVELS)
    parser.add_argument('--tol', type=float, default=1e-12)
    parser.add_argument('--control-tol', type=float, default=1e-13)
    args = parser.parse_args(argv)
    run_benchmark(args.out, args.levels, args.tol, args.control_tol)
    print(f'Saved: {args.out.resolve()}', flush=True)
if __name__ == '__main__':
    main()
