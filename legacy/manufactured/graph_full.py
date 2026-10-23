"""S1 L1 dependency closure extracted from graph_full.py.

Function/class bodies are unchanged source extracts.
"""
from __future__ import annotations

from fractions import Fraction

import math

import sys

import time

import mpmath as mp

import numpy as np

from scipy.linalg import cho_factor, cho_solve, eigh

from scipy.special import erfc, erfcinv, gamma

def l1_weights(tn, left, right, alpha=0.5):
    """Stable L1 weights for intervals [left,right] ending at or before tn."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a, b = float(tn)-left, float(tn)-right
    if np.any(b < 0) or np.any(a <= b):
        raise ValueError('L1 weights require positive, ordered intervals')
    if alpha == 0.5:
        return 1.0/(gamma(1.5)*(np.sqrt(a)+np.sqrt(b)))
    h = right-left
    result = np.empty_like(a)
    recent = b == 0
    result[recent] = h[recent]**(-alpha)/gamma(2-alpha)
    old = ~recent
    z = h[old]/b[old]
    result[old] = (b[old]**(1-alpha)*np.expm1((1-alpha)*np.log1p(z))
                   /(h[old]*gamma(2-alpha)))
    return result

def _row_weights(mesh, n, alpha):
    ticks = getattr(mesh, 'integer_ticks', None)
    if ticks is not None and len(ticks) == len(mesh):
        den = mesh.integer_denominator
        distances = np.asarray([(ticks[n]-ticks[j])/den for j in range(n+1)])
        if alpha == 0.5:
            return 1.0/(gamma(1.5)*(np.sqrt(distances[:-1])+np.sqrt(distances[1:])))
        h = np.asarray([(ticks[j+1]-ticks[j])/den for j in range(n)])
        result = np.empty(n)
        b = distances[1:-1]
        result[:-1] = (b**(1-alpha)*np.expm1((1-alpha)*np.log1p(h[:-1]/b))
                       /(h[:-1]*gamma(2-alpha)))
        result[-1] = h[-1]**(-alpha)/gamma(2-alpha)
        return result
    exact = getattr(mesh, 'exact_times', None)
    if exact is None or len(exact) != len(mesh):
        return l1_weights(mesh[n], mesh[:n], mesh[1:n+1], alpha)
    # Matches the stable MASS3 idea, now for any graph and arbitrary parents.
    distances = np.asarray([float(exact[n]-exact[j]) for j in range(n+1)])
    if alpha == 0.5:
        return 1.0/(gamma(1.5)*(np.sqrt(distances[:-1])+np.sqrt(distances[1:])))
    h = np.asarray([float(exact[j+1]-exact[j]) for j in range(n)])
    result = np.empty(n)
    b = distances[1:-1]
    result[:-1] = (b**(1-alpha)*np.expm1((1-alpha)*np.log1p(h[:-1]/b))
                   /(h[:-1]*gamma(2-alpha)))
    result[-1] = h[-1]**(-alpha)/gamma(2-alpha)
    return result

def _losses(model):
    value = getattr(model, 'losses', None)
    return np.full(len(model.masses), model.c_min) if value is None else np.asarray(value)

def _stats(model, method, N):
    return {'method': method, 'n': int(N), 'dimension': len(model.masses),
            'history_weights': 0, 'history_scalar_madds': 0,
            'history_fraction_subtractions': 0, 'history_state_updates': 0,
            'history_integer_subtractions': 0,
            'force_evaluations': 0, 'F_evaluations': 0, 'jacobian_evaluations': 0,
            'nonlinear_iterations': 0, 'linear_solves': 0, 'matrix_factorizations': 0,
            'linear_flops_est': 0, 'line_search_evaluations': 0,
            'residual_checks': 0, 'output_nodes': N+1,
            'output_scalar_values': (N+1)*len(model.masses),
            'history_start': 0.0, 'arithmetic': 'float64; exact rational mesh differences',
            'machine_epsilon': float(np.finfo(float).eps)}

def _spd_solve(matrix, rhs, stats):
    n = len(rhs)
    factor = cho_factor(matrix, lower=True, check_finite=False)
    out = cho_solve(factor, rhs, check_finite=False)
    stats['matrix_factorizations'] += 1
    stats['linear_solves'] += 1
    stats['linear_flops_est'] += n**3/3+2*n*n
    return out

def _require_finite(value, name, shape=None):
    """Reject a failed floating computation before calling unchecked LAPACK."""
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        raise ArithmeticError(f'{name} must be finite with shape {shape}')
    return array

def _normalized_residual(residual, denominator):
    residual = _require_finite(residual, 'Discrete residual', denominator.shape)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = np.abs(residual) / denominator
    return _require_finite(normalized, 'Normalized discrete residual', denominator.shape)

def _nonlinear_step(model, previous, coefficient, history, load, denominator, tol, stats,
                    evaluate_residual=None):
    """Damped Newton; L1 supplies the same full-history residual as its postcheck.

    Other backends can retain their own operator residual by omitting the
    callback. No tolerance is silently enlarged to the floating-point floor.
    """
    mass = np.asarray(model.masses)
    x = _require_finite(previous, 'Newton starting state').copy()
    denominator = _require_finite(denominator, 'Residual denominator', x.shape)
    if np.any(denominator <= 0):
        raise ArithmeticError('Residual denominator must be positive')

    def residual_at(candidate):
        _require_finite(candidate, 'Newton state', x.shape)
        if evaluate_residual is None:
            residual = mass*(coefficient*(candidate-previous)+history)+model.F(candidate)-load
        else:
            residual = evaluate_residual(candidate)
        stats['F_evaluations'] += 1
        stats['residual_checks'] += 1
        return _require_finite(residual, 'Newton discrete residual', x.shape)

    for _ in range(60):
        r = residual_at(x)
        norm = float(np.max(_normalized_residual(r, denominator)))
        if norm <= tol:
            return x
        j = _require_finite(np.asarray(model.jacobian(x))+np.diag(mass*coefficient),
                            'Newton matrix', (len(x), len(x)))
        stats['jacobian_evaluations'] += 1
        update = _require_finite(_spd_solve(j, -r, stats), 'Newton update', x.shape)
        scale = 1.0
        for _ in range(32):
            trial = x+scale*update
            rr = residual_at(trial)
            stats['line_search_evaluations'] += 1
            if np.max(_normalized_residual(rr, denominator)) <= max(tol, (1-1e-4*scale)*norm):
                x = trial
                break
            scale *= 0.5
        else:
            raise ArithmeticError(f'Newton line search failed; normalized residual {norm:.6g}')
        stats['nonlinear_iterations'] += 1
    raise ArithmeticError('Newton did not reach its declared full discrete residual tolerance')

def _finish(result, model, start, mesh, workspace_bytes):
    stats = result['stats']
    stats.update(getattr(mesh, 'mesh_stats', {}))
    stats['solve_seconds'] = time.perf_counter()-start
    stats['seconds'] = stats['solve_seconds']+stats.get('mesh_seconds', 0.0)
    stats['output_storage_bytes'] = int(result['values'].nbytes+np.asarray(mesh).nbytes)
    stats['peak_array_storage_bytes_est'] = int(workspace_bytes+stats['output_storage_bytes'])
    stats['peak_storage_bytes_est'] = (stats['peak_array_storage_bytes_est']+
                                      stats.get('mesh_fraction_storage_bytes_est', 0)+
                                      stats.get('mesh_integer_storage_bytes_est', 0))
    stats['peak_storage_bytes_est'] = max(stats['peak_storage_bytes_est'],
                                         stats.get('mesh_preparation_peak_python_storage_bytes_est', 0))
    result['counters'] = stats
    return result

def l1_solve(model, times, tol=1e-11, solver='newton'):
    """Implicit L1; every returned step satisfies its measured full G17 check.

    This is a check in floating arithmetic, not an interval or physical-error
    certificate. An unattainable tolerance causes an explicit failure.
    """
    if solver != 'newton':
        raise ValueError('The available algebraic backend is residual-controlled Newton')
    if (isinstance(tol, (bool, np.bool_))
            or not isinstance(tol, (int, float, np.integer, np.floating))):
        raise ValueError('tol must be a positive finite real scalar')
    try:
        tol = float(tol)
    except (OverflowError, ValueError) as exc:
        raise ValueError('tol must be a positive finite real scalar') from exc
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be a positive finite real scalar')
    start = time.perf_counter()
    t = np.asarray(times, dtype=float)
    mass = np.asarray(model.masses, dtype=float)
    if (t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t))
            or t[0] != 0 or not np.all(np.diff(t) > 0)):
        raise ValueError('An ordered full-history mesh starting at zero is required')
    if mass.ndim != 1 or mass.size == 0 or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError('Positive finite masses are required')
    n, dim, alpha = len(t)-1, len(mass), float(model.alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('A finite order 0 < alpha < 1 is required')
    initial = np.asarray(model.u0, dtype=float)
    if initial.shape != (dim,) or not np.all(np.isfinite(initial)):
        raise ValueError('A finite initial state with one value per mass is required')
    losses = np.asarray(_losses(model), dtype=float)
    if losses.shape != (dim,) or not np.all(np.isfinite(losses)) or np.any(losses <= 0):
        raise ValueError('Positive finite losses with one value per mass are required')
    values = np.zeros((n+1, dim)); values[0] = initial
    increments = np.zeros((n, dim))
    stats = _stats(model, 'GRAPH-FULL L1', n)
    maximum, absolute = np.zeros(dim), 0.0
    linear = bool(getattr(model, 'linear', False))
    if linear:
        matrix = _require_finite(model.linear_matrix, 'Linear matrix', (dim, dim))
        offset = _require_finite(model.F(np.zeros(dim)), 'Affine offset', (dim,))
        stats['F_evaluations'] += 1
    for k in range(1, n+1):
        w = _require_finite(_row_weights(times, k, alpha), f'L1 step {k} weights', (k,))
        if np.any(w <= 0):
            raise ArithmeticError(f'L1 step {k}: history weights must be positive')
        history = _require_finite(w[:-1]@increments[:k-1], f'L1 step {k} history', (dim,))
        stats['history_weights'] += k
        stats['history_scalar_madds'] += dim*(k-1)
        if getattr(times, 'exact_times', None) is not None:
            stats['history_fraction_subtractions'] += k+1
        if getattr(times, 'integer_ticks', None) is not None:
            stats['history_integer_subtractions'] += k+1
        load = _require_finite(model.force(t[k]), f'L1 step {k} load', (dim,))
        stats['force_evaluations'] += 1
        denominator = _require_finite(losses+mass*w[0], f'L1 step {k} denominator', (dim,))
        if np.any(denominator <= 0):
            raise ArithmeticError(f'L1 step {k}: residual denominator must be positive')

        def full_residual(candidate):
            # Only the current increment is changed by a trial. All previously
            # accepted increments remain fixed. Newton and the postcheck thus
            # sum the same complete stored history, in the same order.
            increments[k-1] = candidate-values[k-1]
            _require_finite(increments[k-1], f'L1 step {k} increment', (dim,))
            stats['history_scalar_madds'] += dim*k
            return mass*(w@increments[:k])+model.F(candidate)-load

        if linear:
            rhs = _require_finite(load-offset+mass*(w[-1]*values[k-1]-history),
                                  f'L1 step {k} right hand side', (dim,))
            step_matrix = _require_finite(matrix+np.diag(mass*w[-1]),
                                          f'L1 step {k} matrix', (dim, dim))
            values[k] = _require_finite(_spd_solve(step_matrix, rhs, stats),
                                        f'L1 step {k} state', (dim,))
        else:
            try:
                values[k] = _nonlinear_step(model, values[k-1], w[-1], history,
                                            load, denominator, tol, stats,
                                            evaluate_residual=full_residual)
            except ArithmeticError as exc:
                raise ArithmeticError(f'L1 step {k} at t={t[k]:.17g}: {exc}') from exc
            _require_finite(values[k], f'L1 step {k} state', (dim,))
        # Recompute using all ACTUALLY STORED increments, including interval 1.
        residual = full_residual(values[k])
        stats['F_evaluations'] += 1; stats['residual_checks'] += 1
        normalized = _normalized_residual(residual, denominator)
        measured = float(np.max(normalized))
        if measured > tol:
            raise ArithmeticError(
                f'L1 step {k} at t={t[k]:.17g}: full G17 residual '
                f'{measured:.17g} exceeds requested tolerance {tol:.17g}')
        maximum = np.maximum(maximum, normalized)
        absolute = max(absolute, float(np.max(np.abs(residual))))
    residuals = {'max_normalized_full_residual': float(maximum.max()),
                 'max_normalized_full_residual_by_component': maximum.tolist(),
                 'max_absolute_full_residual': absolute,
                 'requested_algebraic_tolerance': tol,
                 'arithmetic_is_interval_certified': False}
    result = {'times': times, 'values': values, 'stats': stats, 'residuals': residuals}
    return _finish(result, model, start, times, increments.nbytes+8*(n+4*dim*dim+8*dim))
