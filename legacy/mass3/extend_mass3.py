#!/usr/bin/env python3
"""Selected further refinements of the unchanged historical MASS3 solver."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'mass3_l1_check.py'
SPEC = importlib.util.spec_from_file_location('mass3_original', SOURCE)
S = importlib.util.module_from_spec(SPEC)
sys.dont_write_bytecode = True
SPEC.loader.exec_module(S)


def main():
    started = time.monotonic()
    baseline = json.loads((HERE / 'repeat_mass3_l1_check.json').read_text(encoding='utf-8'))
    cases = {'equal_unit': ('1', '1'), 'equal_fast': ('1/100', '1/100'),
             'near_fast': ('1/100', '1000001/100000000'),
             'separated': ('1/10000', '1/10'), 'separated_reversed': ('1/10', '1/10000')}
    report = {'source': str(SOURCE.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'alpha': .5, 'T': 1, 'python': platform.python_version(),
              'arithmetic': baseline['arithmetic'], 'versions': baseline['versions'],
              'linear': [], 'nonlinear': [], 'limits': baseline['limits']}
    for name, pair in cases.items():
        s1, s2 = map(S.Fraction, pair)
        masses = S.np.array([S.asld(s1*s1), S.asld(s2*s2), S.LD(1)], dtype=S.LD)
        reference, diagnostic = S.exact_linear_reference(masses)
        prior_case = next(c for c in baseline['linear'] if c['name'] == name)
        runs = list(prior_case['runs'])
        for m in ([256, 512, 1024] if name == 'equal_fast' else [256, 512]):
            nodes = S.rational_mesh(s1, s2, m)
            values, info = S.solve(nodes, masses)
            errors = S.linear_errors(nodes, values, reference)
            runs.append({'m': m, 'B3': 4*m, 'N_actual': len(nodes)-1, **info, **errors,
                         'observed_nodal_order': math.log2(runs[-1]['nodal_error']/errors['nodal_error'])})
            print('linear', name, m, 'error', errors['nodal_error'], flush=True)
        for r in runs:
            r.setdefault('B3', r.get('N'))
            r.setdefault('N_actual', r.get('K'))
            r['scaled_nodal_error_N_actual_to_1_5'] = r['nodal_error'] * r['N_actual']**1.5
        report['linear'].append({'name': name, 'masses_exact': prior_case['masses_exact'],
                                 'reference': diagnostic, 'runs': runs})
    for name in ('equal_fast', 'separated', 'separated_reversed'):
        s1, s2 = map(S.Fraction, cases[name])
        masses = S.np.array([S.asld(s1*s1), S.asld(s2*s2), S.LD(1)], dtype=S.LD)
        prior_case = next(c for c in baseline['nonlinear'] if c['name'] == name)
        runs, comparisons = list(prior_case['runs']), list(prior_case['comparisons'])
        old_m = 128
        old_nodes = S.rational_mesh(s1, s2, old_m)
        old_values, _ = S.solve(old_nodes, masses, nonlinear=True)
        for m in (256, 512):
            nodes = S.rational_mesh(s1, s2, m)
            values, info = S.solve(nodes, masses, nonlinear=True)
            runs.append({'m': m, 'B3': 4*m, 'N_actual': len(nodes)-1, **info})
            comp = S.nested_comparison(old_nodes, old_values, nodes, values)
            comp.update({'coarse_m': old_m, 'fine_m': m,
                         'observed_PL_self_order': math.log2(comparisons[-1]['PL_difference']/comp['PL_difference'])})
            comparisons.append(comp)
            old_m, old_nodes, old_values = m, nodes, values
            print('nonlinear', name, m, 'PLdifference', comp['PL_difference'], flush=True)
        for r in runs:
            r.setdefault('B3', r.get('N'))
            r.setdefault('N_actual', r.get('K'))
        report['nonlinear'].append({**{k: v for k, v in prior_case.items() if k not in ('runs', 'comparisons')},
                                    'runs': runs, 'comparisons': comparisons})
    report['elapsed_seconds'] = time.monotonic() - started
    (HERE / 'extended_mass3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
