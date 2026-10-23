"""Quick checks of all S1 solver families and the saved array archive.

Run from any directory: python path/to/preparation/smoke_checks.py.
This entry reads saved files and runs small solves. It writes no files.
It does not repeat the complete production refinement studies.
"""
from __future__ import annotations

import ast
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import mpmath
import numpy as np
import scipy


REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
CHECKS = []
START = time.perf_counter()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(name, function):
    start = time.perf_counter()
    try:
        detail = function()
        entry = dict(name=name, status='PASS', seconds=time.perf_counter()-start,
                     details=detail)
    except Exception as exc:
        entry = dict(name=name, status='FAIL', seconds=time.perf_counter()-start,
                     error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    CHECKS.append(entry)
    print(json.dumps(entry, ensure_ascii=True), flush=True)


def syntax_and_archives():
    scripts = list(REPO.rglob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)
    counts = dict(archives=0, arrays=0, numeric_entries=0, string_arrays=0)
    for path in REPO.rglob('*.npz'):
        with np.load(path, allow_pickle=False) as arrays:
            counts['archives'] += 1
            for name in arrays.files:
                value = arrays[name]
                counts['arrays'] += 1
                if np.issubdtype(value.dtype, np.number):
                    assert np.all(np.isfinite(value)), (str(path),name)
                    counts['numeric_entries'] += value.size
                elif value.dtype.kind in 'SU':
                    counts['string_arrays'] += 1
                else:
                    raise AssertionError((str(path),name,str(value.dtype)))
    return dict(scripts_parsed=len(scripts), **counts)


def small_mass3():
    module = load('release_mass3', 'legacy/mass3/mass3_l1_check.py')
    rows = []
    for nonlinear in (False,True):
        errors=[]
        for m in (4,8):
            nodes = module.rational_mesh(F(1,100), F(1,10), m)
            masses = np.array([1e-4,.01,1], dtype=module.LD)
            states, info = module.solve(nodes,masses,nonlinear=nonlinear)
            assert np.all(np.isfinite(states))
            assert np.array_equal(states[0],module.U0)
            assert info['max_normalized_full_residual'] < 1e-11
            rows.append(dict(nonlinear=nonlinear,m=m,N=len(nodes)-1,residual=info['max_normalized_full_residual']))
    return dict(runs=rows)


def small_allmass():
    module = load('release_allmass', 'experiments/all_masses_vanishing/run_experiment.py')
    rows=[]
    for roots in ([F(1,100),F(1,200),F(1,400)], [F(1,10),F(1,100),F(1,1000)]):
        errors=[]
        masses = np.array([float(x*x) for x in roots])
        reference,_ = module.reference_factory([x*x for x in roots])
        for m in (4,8):
            nodes = module.make_mesh(roots,m)
            states,residual,normalized,updates = module.solve(nodes,masses)
            assert np.max(normalized)<1e-11
            exact=reference(list(map(float,nodes)))
            error=float(np.max(np.abs(states-exact))); errors.append(error)
            rows.append(dict(m=m,N=len(nodes)-1,residual=float(np.max(normalized)),error=error,updates=updates))
        assert errors[1]<errors[0]
    return dict(runs=rows)


def small_scalar():
    module = load('release_scalar','experiments/scalar_mesh_comparison/run_study.py')
    rows=[]
    for p in (3,7):
        for mass in (F(1),F(1,10000)):
            for m in (4,8):
                nums,den,info=module.grid(p,mass,m,'union')
                times=module.as_float(nums,den)
                states,residual,normalized,oldest=module.solve(nums,den,p/10,float(mass))
                exact,_=module.spectral_reference(times,p/10,float(mass))
                assert np.all(np.isfinite(states)) and np.max(normalized)<1e-11
                rows.append(dict(alpha=p/10,mass=str(mass),m=m,N=len(nums)-1,residual=float(np.max(normalized)),error=float(np.max(abs(states-exact)))))
    return dict(runs=rows)


def small_cubic_and_pi():
    module=load('release_cubic','experiments/nonlinear_reference/run_study.py')
    reference=load('release_pi','experiments/nonlinear_reference/reference.py')
    rows=[]
    for p in (3,7):
        delta=F(1)
        masses=np.array([float(delta*x) for x in module.RATIOS])
        nums,den,info=module.exact_grid(p,delta,4)
        sol=module.solve_l1(nums,den,p/10,masses)
        assert np.max(sol['normalized_residuals'])<1e-11
        pi=reference.solve_reference(p/10,masses,module.C,module.B,module.EDGE,module.EDGE_CUBIC,module.U0,module.LOAD,32,probes=[0.,.3,1.])
        assert np.all(np.isfinite(pi['U']))
        assert np.array_equal(pi['U'][0],module.U0)
        assert pi['max_normalized_algebraic_residual']<1e-11
        assert np.allclose(pi['probe_U'][[0,2]],pi['U'][[0,-1]],atol=1e-12,rtol=0)
        rows.append(dict(alpha=p/10,l1_N=len(nums)-1,l1_residual=float(np.max(sol['normalized_residuals'])),PI_K=32,PI_residual=pi['max_normalized_algebraic_residual']))
    return dict(runs=rows)


def small_cpe():
    module=load('release_cpe','experiments/cpe_network/run_study.py')
    rows=[]
    for m in (4,8):
        nodes=module.base.make_mesh([F(1,2),F(1,4),F(1,8)],m)
        states,residual,normalized=module.solve(nodes)
        exact=module.reference(list(map(float,nodes)))
        assert np.max(normalized)<1e-11
        rows.append(dict(m=m,N=len(nodes)-1,residual=float(np.max(normalized)),error=float(np.max(abs(states-exact)))))
    assert rows[1]['error']<rows[0]['error']
    rc,info=module.rc_reference([0.,module.T*.1,module.T],dps=45)
    assert np.all(np.isfinite(rc)) and np.array_equal(rc[0],np.zeros(3))
    return dict(runs=rows,RC_samples=3)


def manufactured_closure():
    folder=REPO/'legacy/manufactured'
    sys.path.insert(0,str(folder))
    module=load('release_manufactured','legacy/manufactured/run_manufactured.py')
    case=module.ManufacturedCase()
    record,arrays=module.solve_level(case,'L1',20)
    assert record['G17_comparison_check']
    saved=folder/'results/manufactured_nonlinear/l1_n20.npz'
    with np.load(saved,allow_pickle=False) as original:
        difference=float(np.max(abs(original['values']-arrays['values'])))
        assert difference<1e-13
    return dict(n=20,E_node=record['E_node'],residual=record['solver_actual_state_residual'],saved_state_max_difference=difference)


for name,function in [('syntax_and_all_archives',syntax_and_archives),('legacy_linear_and_C2_smoke',small_mass3),('joint_mass_limits_smoke',small_allmass),('scalar_alpha03_alpha07_smoke',small_scalar),('cubic_L1_and_independent_PI_smoke',small_cubic_and_pi),('CPE_and_RC_smoke',small_cpe),('manufactured_L1_and_archived_n20',manufactured_closure)]:
    check(name,function)

passed = all(x['status']=='PASS' for x in CHECKS)
print(json.dumps(dict(status='PASS' if passed else 'FAIL', check_groups=len(CHECKS), seconds=time.perf_counter()-START, scope='Quick floating-point checks, not a full production rerun or an interval certificate.'), ensure_ascii=True))
raise SystemExit(0 if passed else 1)
