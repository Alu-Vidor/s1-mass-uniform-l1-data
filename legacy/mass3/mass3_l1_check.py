#!/usr/bin/env python3
"""MASS3 self-check: exact rational alpha=1/2 union; full-history L1.

Finite calculations are diagnostics, not proof or a separate review 06.
Parents are exact Fraction objects. No tolerance merging is performed.
An exact erfcx spectral formula provides the linear reference; evaluating
that formula uses scipy float64. Nonlinear checks compare nested refinements.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.special import erfcx


LD = np.longdouble
GAMMA_15 = LD(str(math.sqrt(math.pi) / 2))
C = np.array([1.0, 1.3, 0.9], dtype=LD)
B = np.array([[1, 0], [-1, 1], [0, -1]], dtype=LD)
U0 = np.array([1.0, -0.4, 0.2], dtype=LD)


def asld(x: Fraction) -> np.longdouble:
    return LD(str(x.numerator)) / LD(str(x.denominator))


def rational_mesh(s1: Fraction, s2: Fraction, m: int):
    """Here eta=s**2, alpha=1/2, T=1; q_j=(s*j/(m*(1+s)-j))**4."""
    parents = []
    for s in (s1, s2, Fraction(1)):
        parents.append([(s*j / (m*(1+s)-j))**4 for j in range(m+1)])
    parents.append([Fraction(j, m) for j in range(m+1)])
    nodes = sorted(set().union(*map(set, parents)))
    assert nodes[0] == 0 and nodes[-1] == 1
    assert all(a < b for a, b in zip(nodes, nodes[1:]))
    assert m <= len(nodes)-1 <= 4*m-3
    return nodes


def row_weights(nodes, n: int):
    # For alpha=1/2 rationalization avoids subtracting close square roots.
    # Every time distance is formed as an exact rational BEFORE rounding.
    # Fraction.__float__ rounds the exact ratio to float64 as a single operation,
    # hence preserves ordering even when two far-history distances round alike.
    # Separate longdouble numerator/denominator conversions need not do so.
    distances = np.array([LD(float(nodes[n]-nodes[j])) for j in range(n+1)], dtype=LD)
    roots = np.sqrt(distances)
    weights = 1 / (GAMMA_15 * (roots[:-1]+roots[1:]))
    assert np.all(weights > 0)
    assert np.all(np.diff(weights) >= 0)
    return weights


def spd_solve3(a, rhs):
    """Three-dimensional Cholesky in longdouble (numpy.linalg lacks float128)."""
    l00 = np.sqrt(a[0, 0])
    l10, l20 = a[1, 0]/l00, a[2, 0]/l00
    l11 = np.sqrt(a[1, 1]-l10*l10)
    l21 = (a[2, 1]-l20*l10)/l11
    l22 = np.sqrt(a[2, 2]-l20*l20-l21*l21)
    y0 = rhs[0]/l00
    y1 = (rhs[1]-l10*y0)/l11
    y2 = (rhs[2]-l20*y0-l21*y1)/l22
    x2 = y2/l22
    x1 = (y1-l21*x2)/l11
    x0 = (y0-l10*x1-l20*x2)/l00
    return np.array([x0, x1, x2], dtype=LD)


def phi(s, nonlinear):
    if nonlinear:
        return 5*s + LD('0.2')*s*np.abs(s)**LD('1.5')
    return np.array([5, 3], dtype=LD)*s


def phi_prime(s, nonlinear):
    if nonlinear:
        return 5 + LD('0.5')*np.abs(s)**LD('1.5')
    return np.array([5, 3], dtype=LD)


def force(u, nonlinear):
    return C*u + B@phi(B.T@u, nonlinear)


def jacobian(u, nonlinear):
    return np.diag(C) + (B*phi_prime(B.T@u, nonlinear))@B.T


def load(t, nonlinear):
    if nonlinear:
        return np.array([LD('.3')+LD('.2')*np.sin(t),
                         LD('.6')+LD('.15')*np.cos(2*t),
                         LD('.2')+LD('.1')*np.sin(3*t)], dtype=LD)
    return np.array([.4, -.2, .3], dtype=LD)


def solve(nodes, masses, nonlinear=False, tolerance=LD('1e-12')):
    k = len(nodes)-1
    u = np.zeros((k+1, 3), dtype=LD)
    u[0] = U0
    increments = np.zeros((k, 3), dtype=LD)
    max_residual = LD(0)
    max_residual_each = np.zeros(3, dtype=LD)
    max_absolute_residual = LD(0)
    newton_iterations = 0
    line_search_evaluations = 0
    max_step_iterations = 0
    history_terms = 0
    for n in range(1, k+1):
        weights = row_weights(nodes, n)
        history = weights[:-1] @ increments[:n-1]
        history_terms += n-1
        p = load(asld(nodes[n]), nonlinear)
        denominators = C+masses*weights[0]
        x = u[n-1].copy()
        for it in range(40):
            residual = masses*(weights[-1]*(x-u[n-1])+history)+force(x, nonlinear)-p
            normalized = np.max(np.abs(residual)/denominators)
            if normalized <= tolerance:
                break
            matrix = jacobian(x, nonlinear)+np.diag(masses*weights[-1])
            update = spd_solve3(matrix, -residual)
            step = LD(1)
            accepted = False
            for _ in range(32):
                candidate = x+step*update
                trial = masses*(weights[-1]*(candidate-u[n-1])+history)+force(candidate, nonlinear)-p
                trial_norm = np.max(np.abs(trial)/denominators)
                line_search_evaluations += 1
                if trial_norm <= (1-LD('1e-4')*step)*normalized or trial_norm <= tolerance:
                    x = candidate
                    accepted = True
                    break
                step *= LD('.5')
            if not accepted:
                raise RuntimeError(f'Line search failed: step {n}, residual {normalized}')
            newton_iterations += 1
        else:
            raise RuntimeError(f'Nonlinear iteration failed at step {n}')
        max_step_iterations = max(max_step_iterations, it)
        u[n] = x
        # Recompute increments from the ACTUAL stored nodal values.
        increments[n-1] = u[n]-u[n-1]
        full_residual = masses*(weights @ increments[:n])+force(u[n], nonlinear)-p
        history_terms += n
        ratios = np.abs(full_residual)/denominators
        max_residual_each = np.maximum(max_residual_each, ratios)
        max_residual = max(max_residual, np.max(ratios))
        max_absolute_residual = max(max_absolute_residual, np.max(np.abs(full_residual)))
    edge_values = u@B
    return u, {
        'max_normalized_full_residual': float(max_residual),
        'max_normalized_full_residual_by_component': [float(x) for x in max_residual_each],
        'max_absolute_full_residual': float(max_absolute_residual),
        'newton_updates': newton_iterations,
        'max_newton_updates_per_step': max_step_iterations,
        'line_search_residual_evaluations': line_search_evaluations,
        'history_scalar_weighted_vector_terms_including_residual_check': history_terms,
        'expected_history_terms_K_squared': k*k,
        'retained_history_start': 0,
        'max_solution_component': float(np.max(np.abs(u))),
        'edge_difference_ranges': [[float(v.min()), float(v.max())] for v in edge_values.T],
        'edge_sign_changes_between_nodes': [int(np.sum(v[1:]*v[:-1] < 0)) for v in edge_values.T],
    }


def exact_linear_reference(masses):
    a = np.asarray(jacobian(U0, False), dtype=float)
    eval_a, q_a = eigh(a)
    ahalf = (q_a*np.sqrt(eval_a))@q_a.T
    ainvhalf = (q_a/np.sqrt(eval_a))@q_a.T
    h = (ahalf/np.asarray(masses, float))@ahalf
    eigenvalues, q = eigh(h, driver='evr')
    equilibrium = np.linalg.solve(a, np.asarray(load(LD(0), False), float))
    modal = q.T@ahalf@(np.asarray(U0, float)-equilibrium)
    left = ainvhalf@q
    def ref(times):
        times = np.asarray(times, dtype=float)
        profiles = erfcx(np.sqrt(times)[:, None]*eigenvalues[None, :])
        result = equilibrium+(profiles*modal[None, :])@left.T
        result[times == 0] = np.asarray(U0, float)
        return result
    diagnostic = {
        'eigenvalues': eigenvalues.tolist(),
        'spectral_eigenpair_absolute_residual': float(np.max(np.abs(h@q-q*eigenvalues))),
        'reference_evaluation': 'scipy erfcx float64; closed alpha=1/2 formula, not interval arithmetic',
    }
    return ref, diagnostic


def linear_errors(nodes, values, reference):
    times = [float(t) for t in nodes]
    exact = reference(times)
    error = np.max(np.abs(values-exact), axis=0)
    interp_exact_error = LD(0)
    continuous_error = LD(0)
    for n in range(1, len(nodes)):
        probes = [(nodes[n-1]*(4-j)+nodes[n]*j)/4 for j in (1, 2, 3)]
        exact_probes = reference([float(t) for t in probes])
        for j in range(3):
            theta = LD(j+1)/4
            interp = (1-theta)*values[n-1]+theta*values[n]
            interp_exact = (1-theta)*exact[n-1]+theta*exact[n]
            continuous_error = max(continuous_error, np.max(np.abs(interp-exact_probes[j])))
            interp_exact_error = max(interp_exact_error, np.max(np.abs(interp_exact-exact_probes[j])))
    continuous_error = max(continuous_error, np.max(error))
    return {
        'nodal_error': float(np.max(error)),
        'nodal_error_by_component': [float(x) for x in error],
        'sampled_PL_error': float(continuous_error),
        'sampled_exact_interpolation_error': float(interp_exact_error),
        'continuous_sampling': 'Every knot plus 1/4,1/2,3/4 of every exact rational interval; not certified supremum',
    }


def nested_comparison(coarse_nodes, coarse, fine_nodes, fine):
    lookup = {t: i for i, t in enumerate(fine_nodes)}
    indices = [lookup[t] for t in coarse_nodes]
    nodal = np.max(np.abs(coarse-fine[indices]), axis=0)
    pl_error = np.zeros(3, dtype=LD)
    j = 0
    for n, t in enumerate(fine_nodes):
        while j+1 < len(coarse_nodes)-1 and coarse_nodes[j+1] < t:
            j += 1
        theta = asld((t-coarse_nodes[j])/(coarse_nodes[j+1]-coarse_nodes[j]))
        interpolated = (1-theta)*coarse[j]+theta*coarse[j+1]
        pl_error = np.maximum(pl_error, np.abs(interpolated-fine[n]))
    return {'nodal_difference': float(np.max(nodal)),
            'nodal_difference_by_component': [float(v) for v in nodal],
            'PL_difference': float(np.max(pl_error)),
            'PL_difference_by_component': [float(v) for v in pl_error],
            'meaning': 'Nested refinement differences, not errors against an exact nonlinear solution'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='mass3_l1_check.json')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    ms = [16, 32, 64] if args.quick else [16, 32, 64, 128]
    cases = {
        'equal_unit': ('1', '1'),
        'equal_fast': ('1/100', '1/100'),
        'near_fast': ('1/100', '1000001/100000000'),
        'separated': ('1/10000', '1/10'),
        'separated_reversed': ('1/10', '1/10000'),
    }
    report = {'purpose': 'MASS3 development self-check, not independent review 06',
              'alpha': .5, 'T': 1, 'new_budget': 'N_MASS3=4*m',
              'arithmetic': {'solution': str(np.dtype(LD)),
                             'longdouble_epsilon': float(np.finfo(LD).eps),
                             'grid_and_differences': 'Exact fractions; exact distances rounded once to float64 after subtraction; subsequent arithmetic longdouble',
                             'tolerance_merging': False},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'linear': [], 'nonlinear': [], 'limits': [
                  'Finite selected masses and alpha=1/2 do not prove uniformity over all parameters.',
                  'erfcx spectral references are evaluated in float64, without interval error bounds.',
                  'Residuals use the stated rounded L1 weights and floating arithmetic; they are not certified bounds for the exact mathematical operator.',
                  'Sampled continuous linear errors are not certified suprema.',
                  'Nonlinear refinement differences are not rigorous upper error bounds.',
                  'No performance speedup comparison is performed.']}
    for name, pair in cases.items():
        s1, s2 = map(Fraction, pair)
        masses = np.array([asld(s1*s1), asld(s2*s2), LD(1)], dtype=LD)
        reference, diagnostic = exact_linear_reference(masses)
        case = {'name': name, 'masses_exact': [str(s1*s1), str(s2*s2), '1'],
                'reference': diagnostic, 'runs': []}
        previous = None
        for m in ms:
            nodes = rational_mesh(s1, s2, m)
            values, info = solve(nodes, masses)
            errors = linear_errors(nodes, values, reference)
            entry = {'m': m, 'N': 4*m, 'K': len(nodes)-1,
                     'minimum_step': float(min(b-a for a, b in zip(nodes, nodes[1:]))),
                     **info, **errors}
            if previous is not None:
                entry['observed_nodal_order'] = math.log2(previous/entry['nodal_error'])
            previous = entry['nodal_error']
            case['runs'].append(entry)
        report['linear'].append(case)
        print('linear', name, 'last error', case['runs'][-1]['nodal_error'], flush=True)
    for name in ('equal_fast', 'separated', 'separated_reversed'):
        s1, s2 = map(Fraction, cases[name])
        masses = np.array([asld(s1*s1), asld(s2*s2), LD(1)], dtype=LD)
        case = {'name': name, 'masses_exact': [str(s1*s1), str(s2*s2), '1'],
                'phi': '5*s+0.2*s*abs(s)**1.5',
                'phi_second_derivative': '0.75*sign(s)*abs(s)**0.5; continuous, not Lipschitz at zero',
                'load': '[.3+.2*sin(t), .6+.15*cos(2*t), .2+.1*sin(3*t)]',
                'initial': [float(x) for x in U0], 'runs': [], 'comparisons': []}
        previous = None
        prior_difference = None
        for m in ms:
            nodes = rational_mesh(s1, s2, m)
            values, info = solve(nodes, masses, nonlinear=True)
            case['runs'].append({'m': m, 'N': 4*m, 'K': len(nodes)-1, **info})
            if previous is not None:
                old_nodes, old_values, old_m = previous
                comparison = nested_comparison(old_nodes, old_values, nodes, values)
                comparison.update({'coarse_m': old_m, 'fine_m': m})
                if prior_difference is not None:
                    comparison['observed_PL_self_order'] = math.log2(prior_difference/comparison['PL_difference'])
                prior_difference = comparison['PL_difference']
                case['comparisons'].append(comparison)
            previous = nodes, values, m
        report['nonlinear'].append(case)
        print('nonlinear', name, 'last difference', case['comparisons'][-1]['PL_difference'], flush=True)
    report['elapsed_seconds'] = time.monotonic()-start
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('written', args.output, 'seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
