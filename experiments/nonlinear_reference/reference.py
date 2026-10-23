"""Independent implicit product integration for a Caputo graph system.

This solver discretises the equivalent Volterra integral equation, integrating
the piecewise-linear interpolation of the *right-hand side*.  It does not use
Caputo L1 difference weights.  Convergence comparisons are numerical evidence,
not interval certificates.
"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.special import gamma


def pi_weights(alpha, t, left, right):
    """Positive integrals of the two linear basis functions / Gamma(alpha).

    Supports partial final intervals: ``right`` is the actual integrated end
    and all intervals use this end as their linear basis endpoint.  For tiny
    h/(t-left), a positive Taylor series avoids subtracting O(h) quantities.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    h = right-left
    a = t-left
    b = t-right
    if np.any(h <= 0) or np.any(b < 0):
        raise ValueError('invalid product-integration interval')
    r = h/a
    wl = np.empty_like(r)
    wr = np.empty_like(r)
    small = r < 0.025
    if np.any(small):
        rs = r[small]
        coeff = np.ones_like(rs)
        ls = np.full_like(rs, .5)
        rr = np.full_like(rs, .5)
        # c_k=(1-alpha)_k/k!. k=0 terms were included above.
        for k in range(1, 16):
            coeff *= (k-alpha)*rs/k
            ls += coeff/((k+1)*(k+2))
            rr += coeff/(k+2)
        scale = h[small]*a[small]**(alpha-1)/gamma(alpha)
        wl[small] = scale*ls
        wr[small] = scale*rr
    large = ~small
    if np.any(large):
        rl = r[large]
        bl = b[large]
        al = a[large]
        # Form b/a independently; h/a may round to one near the last interval.
        logratio = np.log(bl/al, where=bl > 0,
                          out=np.full_like(bl, -np.inf))
        one_a = -np.expm1(alpha*logratio)/alpha
        one_a1 = -np.expm1((alpha+1)*logratio)/(alpha+1)
        scale = al**alpha/gamma(alpha)/rl
        wr[large] = scale*(one_a-one_a1)
        wl[large] = scale*(one_a1-(bl/al)*one_a)
    if np.any(wl < 0) or np.any(wr < 0):
        raise ArithmeticError('negative integral weights')
    return wl, wr


def graph_force_jac(u, C, B, edge_weights, edge_cubic=0.1,
                    node_cubic=0.0):
    z = B.T@u
    ew = np.asarray(edge_weights)
    ec = np.asarray(edge_cubic)
    nc = np.asarray(node_cubic)
    force = C*u+nc*u**3+B@(ew*(z+ec*z**3))
    jac = np.diag(C+3*nc*u**2)+(B*(ew*(1+3*ec*z*z)))@B.T
    return force, jac


def solve_reference(alpha, masses, C, B, edge_weights, edge_cubic,
                    u0, p, K, grade=None, probes=None, node_cubic=0.0,
                    T=1.0, newton_tol=2e-14, max_newton=30):
    """Return independent integral-equation solution on a power-graded mesh.

    ``probes`` are evaluated through the continuous integral of interpolated
    forcing after the nodal solve; they do not alter the reference grid.
    ``p`` is a constant vector or a function of physical time.
    """
    start = time.perf_counter()
    masses = np.asarray(masses, float)
    C = np.asarray(C, float)
    B = np.asarray(B, float)
    u0 = np.asarray(u0, float)
    grade = float(2/alpha if grade is None else grade)
    nodes = T*(np.arange(K+1, dtype=float)/K)**grade
    U = np.empty((K+1, len(u0)), float)
    f = np.empty_like(U)
    U[0] = u0
    pf = p if callable(p) else lambda t: np.asarray(p, float)
    F0, _ = graph_force_jac(u0,C,B,edge_weights,edge_cubic,node_cubic)
    f[0] = pf(0)-F0
    maxit = 0
    worst_residual = 0.0
    for n in range(1, K+1):
        wl, wr = pi_weights(alpha,nodes[n],nodes[:n],nodes[1:n+1])
        integral = wl@f[:n]
        if n > 1:
            integral += wr[:-1]@f[1:n]
        w = wr[-1]
        pn = pf(nodes[n])
        rhs = masses*u0+integral+w*pn
        v = U[n-1].copy()
        for it in range(max_newton):
            F, J = graph_force_jac(v,C,B,edge_weights,edge_cubic,node_cubic)
            residual = masses*v+w*F-rhs
            update = np.linalg.solve(np.diag(masses)+w*J,residual)
            v -= update
            if np.max(np.abs(update)) <= newton_tol*(1+np.max(np.abs(v))):
                break
        else:
            raise RuntimeError(f'Newton failed at n={n}')
        F, _ = graph_force_jac(v,C,B,edge_weights,edge_cubic,node_cubic)
        residual = masses*v+w*F-rhs
        worst_residual = max(worst_residual,
                             float(np.max(np.abs(residual)/(masses+w*C))))
        maxit = max(maxit,it+1)
        U[n] = v
        f[n] = pn-F
    result = dict(alpha=float(alpha),masses=masses,C=C,B=B,
                  edge_weights=np.asarray(edge_weights),
                  edge_cubic=np.asarray(edge_cubic),
                  node_cubic=np.asarray(node_cubic),u0=u0,nodes=nodes,U=U,
                  forcing=f,K=int(K),grade=grade,T=float(T),
                  max_newton=maxit,max_normalized_algebraic_residual=worst_residual,
                  elapsed_seconds=time.perf_counter()-start)
    if probes is not None:
        result['probes'] = np.asarray(probes,float)
        result['probe_U'] = evaluate_reference(result,probes)
    return result


def evaluate_reference(result, probes):
    """Continuous Volterra reconstruction, integrating reference forcing.

    This is deliberately distinct from linearly interpolating the solution U.
    The reconstruction is exact for the forcing interpolation used by the
    product-integration method, up to floating-point rounding.
    """
    nodes, f = result['nodes'],result['forcing']
    masses, u0 = result['masses'],result['u0']
    alpha = result['alpha']
    probes = np.asarray(probes,float)
    out = np.empty((len(probes),len(u0)),float)
    for k,t in enumerate(probes):
        if t < 0 or t > nodes[-1]*(1+1e-15):
            raise ValueError('query outside reference interval')
        if t == 0:
            out[k] = u0
            continue
        n = min(int(np.searchsorted(nodes,t,side='left')),len(nodes)-1)
        if nodes[n] == t:
            out[k] = result['U'][n]
            continue
        # n>=1. The final right endpoint is t, with forcing interpolated from
        # the original reference cell [nodes[n-1], nodes[n]].
        ends = nodes[1:n+1].copy()
        ends[-1] = t
        wl,wr = pi_weights(alpha,t,nodes[:n],ends)
        integral = wl@f[:n]
        if n > 1:
            integral += wr[:-1]@f[1:n]
        fraction = (t-nodes[n-1])/(nodes[n]-nodes[n-1])
        ft = (1-fraction)*f[n-1]+fraction*f[n]
        integral += wr[-1]*ft
        out[k] = u0+integral/masses
    return out
