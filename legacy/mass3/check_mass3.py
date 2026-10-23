"""Reproduction, mesh accounting, and selected high precision reference checks."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import mpmath as mp

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'mass3_l1_check.py'
SPEC = importlib.util.spec_from_file_location('mass3_original', SOURCE)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
historical = json.loads(SOURCE.with_suffix('.json').read_text(encoding='utf-8'))
repeat = json.loads((HERE/'repeat_mass3_l1_check.json').read_text(encoding='utf-8'))
extended = json.loads((HERE/'extended_mass3.json').read_text(encoding='utf-8'))
linear_repeat_difference = max(abs(a['nodal_error']-b['nodal_error'])
    for ca, cb in zip(historical['linear'], repeat['linear']) for a,b in zip(ca['runs'], cb['runs']))
nonlinear_repeat_difference = max(abs(a['PL_difference']-b['PL_difference'])
    for ca, cb in zip(historical['nonlinear'], repeat['nonlinear'])
    for a,b in zip(ca['comparisons'], cb['comparisons']))
mp.mp.dps = 80
B = mp.matrix([[1,0],[-1,1],[0,-1]])
A = mp.diag([mp.mpf('1'),mp.mpf('1.3'),mp.mpf('.9')])+B*mp.diag([5,3])*B.T
lam_a, q_a = mp.eigsy(A)
ah = q_a*mp.diag([mp.sqrt(v) for v in lam_a])*q_a.T
aih = q_a*mp.diag([1/mp.sqrt(v) for v in lam_a])*q_a.T
equilibrium = mp.lu_solve(A, mp.matrix(['.4','-.2','.3']))
u0 = mp.matrix(['1','-.4','.2'])
reference_checks = []
for case in extended['linear']:
    masses_mp = [mp.mpf(v.split('/')[0])/mp.mpf(v.split('/')[1]) if '/' in v else mp.mpf(v)
                 for v in case['masses_exact']]
    h = ah*mp.diag([1/v for v in masses_mp])*ah
    ev, q = mp.eigsy(h)
    modal = q.T*ah*(u0-equilibrium)
    left = aih*q
    masses = S.np.array([float(v) for v in masses_mp], dtype=S.LD)
    ref, _ = S.exact_linear_reference(masses)
    times = [0.0] + [10**(-24+24*j/80) for j in range(81)]
    low = ref(times)
    maxdiff = mp.mpf(0)
    for i,t in enumerate(times):
        if t == 0:
            exact = u0
        else:
            z = [mp.sqrt(mp.mpf(t))*v for v in ev]
            exact = equilibrium+left*mp.matrix([mp.exp(x*x)*mp.erfc(x)*modal[j] for j,x in enumerate(z)])
        maxdiff = max(maxdiff, max(abs(mp.mpf(float(low[i,j]))-exact[j]) for j in range(3)))
    reference_checks.append({'name':case['name'],'precision_decimal_digits':80,
                             'sample_count':len(times),'max_absolute_difference':float(maxdiff)})
mesh_checks = []
for case in extended['linear']:
    multiplicity = 2 if case['name']=='equal_unit' else 3 if case['name']=='equal_fast' else 4
    for r in case['runs']:
        assert r['N_actual'] == multiplicity*r['m']-(multiplicity-1)
    mesh_checks.append({'name':case['name'],'N_actual_formula':f'{multiplicity}*m-{multiplicity-1}',
                        'all_refinements_pass':True})
checks = {'historical_source_unchanged_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'full_original_script_repeated':True,
          'repeat_max_absolute_difference_of_linear_nodal_errors':linear_repeat_difference,
          'repeat_max_absolute_difference_of_nonlinear_PL_differences':nonlinear_repeat_difference,
          'mesh_checks':mesh_checks,'reference_checks':reference_checks,
          'linear_max_normalized_residual':max(r['max_normalized_full_residual'] for c in extended['linear'] for r in c['runs']),
          'nonlinear_max_normalized_residual':max(r['max_normalized_full_residual'] for c in extended['nonlinear'] for r in c['runs']),
          'linear_last_m_doubling_orders':{c['name']:c['runs'][-1]['observed_nodal_order'] for c in extended['linear']},
          'nonlinear_last_PL_self_orders':{c['name']:c['comparisons'][-1]['observed_PL_self_order'] for c in extended['nonlinear']},
          'all_error_sequences_strictly_decrease':all(all(a['nodal_error']>b['nodal_error'] for a,b in zip(c['runs'],c['runs'][1:])) for c in extended['linear']),
          'all_PL_difference_sequences_strictly_decrease':all(all(a['PL_difference']>b['PL_difference'] for a,b in zip(c['comparisons'],c['comparisons'][1:])) for c in extended['nonlinear']),
          'limits':['Selected masses and alpha=1/2 only; no numerical proof of parameter uniformity.',
                    'Nonlinear PL differences compare two numerical solutions; no true nonlinear error bound.',
                    '80 digit reference cross-check on 82 selected points is not a certified global bound.']}
(HERE/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
