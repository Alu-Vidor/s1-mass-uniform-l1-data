"""Extract S1 tables from saved results; never import or run a solver.

Run from any directory with Python 3.12+. All paths are relative to this file.
Observed rates in N are recomputed from stored errors and actual interval counts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MASS_DIR = ROOT / 'legacy/mass3'
EXACT_DIR = ROOT / 'legacy/manufactured/results/manufactured_nonlinear'
CASE_ID = {'equal_unit': 'M1', 'equal_fast': 'M2', 'near_fast': 'M3',
           'separated': 'M4', 'separated_reversed': 'M5'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(name, rows):
    path = HERE / name
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    data = read_json(MASS_DIR / 'extended_mass3.json')
    checks = read_json(MASS_DIR / 'checks.json')
    summary = read_json(EXACT_DIR / 'SUMMARY.json')
    manufactured_sources = read_json(EXACT_DIR / 'SOURCE_SHA256.json')
    manufactured_results = read_json(EXACT_DIR / 'RESULT_SHA256.json')
    current_files = {item['path']: item for item in read_json(ROOT / 'CHECKSUMS.json')['files']}
    adaptations_path = ROOT / 'provenance/source_adaptations.json'
    adaptations = read_json(adaptations_path)
    def check_current(path):
        relative = path.relative_to(ROOT).as_posix()
        item = current_files[relative]
        assert path.stat().st_size == item['bytes'], relative
        assert sha256(path) == item['sha256'], relative
    for path in (Path(__file__), adaptations_path, EXACT_DIR / 'SOURCE_SHA256.json', EXACT_DIR / 'RESULT_SHA256.json'):
        check_current(path)
    for name, expected_hash in manufactured_sources.items():
        source = ROOT / 'legacy/manufactured' / name
        relative = source.relative_to(ROOT).as_posix()
        check_current(source)
        if sha256(source) != expected_hash:
            assert expected_hash == adaptations[relative]['archived_public_sha256'], relative
    for name, expected_hash in manufactured_results.items():
        assert sha256(EXACT_DIR / name) == expected_hash
    linear, nonlinear, manufactured = [], [], []
    for case in data['linear']:
        previous = None
        for run in case['runs']:
            n, m, error = run['N_actual'], run['m'], run['nodal_error']
            expected = {'M1': 2*m-1, 'M2': 3*m-2}.get(CASE_ID[case['name']], 4*m-3)
            assert n == expected
            row = dict(case=CASE_ID[case['name']], name=case['name'],
                       masses_exact=';'.join(case['masses_exact']), m=m, B3=run['B3'],
                       N_actual=n, E_node=error,
                       q_m=math.log2(previous['E_node']/error) if previous else '',
                       q_N=(math.log(previous['E_node']/error)/math.log(n/previous['N_actual']))
                           if previous else '',
                       E_N_to_1_5=error*n**1.5,
                       normalized_full_residual=run['max_normalized_full_residual'],
                       sampled_PL_error=run['sampled_PL_error'])
            if 'observed_nodal_order' in run:
                assert math.isclose(row['q_m'], run['observed_nodal_order'], abs_tol=1e-13)
            linear.append(row)
            previous = row
    for case in data['nonlinear']:
        runs = {r['m']: r for r in case['runs']}
        previous = None
        for comparison in case['comparisons']:
            coarse, fine = comparison['coarse_m'], comparison['fine_m']
            difference = comparison['PL_difference']
            row = dict(case=CASE_ID[case['name']], name=case['name'], coarse_m=coarse,
                       fine_m=fine, coarse_N_actual=runs[coarse]['N_actual'],
                       fine_N_actual=runs[fine]['N_actual'], D_PL=difference,
                       q_m_self=math.log2(previous['D_PL']/difference) if previous else '',
                       q_N_self=(math.log(previous['D_PL']/difference) /
                                 math.log(runs[coarse]['N_actual']/previous['coarse_N_actual']))
                                 if previous else '',
                       residual_coarse=runs[coarse]['max_normalized_full_residual'],
                       residual_fine=runs[fine]['max_normalized_full_residual'])
            if 'observed_PL_self_order' in comparison:
                assert math.isclose(row['q_m_self'], comparison['observed_PL_self_order'], abs_tol=1e-13)
            nonlinear.append(row)
            previous = row
    with (EXACT_DIR / 'ACCURACY.csv').open(encoding='utf-8', newline='') as stream:
        accuracy = list(csv.DictReader(stream))
    for row in accuracy:
        if row['method'] == 'L1':
            e = float(row['E_node'])
            budget = float(row['G17_error_estimate'])
            assert e <= budget
            manufactured.append(dict(N_actual=int(row['n']), h=1/int(row['n']),
                                     E_node=e, q_node=row['p_node'],
                                     E_PL=float(row['E_linearinterp']), q_PL=row['p_linearinterp'],
                                     consistency_budget=float(row['G17_consistency_budget']),
                                     normalized_full_residual=float(row['independent_actual_state_residual']),
                                     total_budget=budget, budget_to_error=budget/e))
    outputs = [write_csv('linear_mass3.csv', linear),
               write_csv('nonlinear_mass3.csv', nonlinear),
               write_csv('manufactured_L1.csv', manufactured)]
    source = ROOT / 'legacy/mass3/mass3_l1_check.py'
    assert sha256(source) == data['source_sha256'] == checks['historical_source_unchanged_sha256']
    assert len(linear) == 31 and len(nonlinear) == 15 and len(manufactured) == 5
    inputs = [MASS_DIR / 'extended_mass3.json', MASS_DIR / 'checks.json',
              EXACT_DIR / 'ACCURACY.csv', EXACT_DIR / 'CONVERGENCE.csv',
              EXACT_DIR / 'SUMMARY.json', source]
    manifest = dict(purpose='Extraction and arithmetic checks of saved data only; no numerical solver run',
                    row_counts=dict(linear=31, nonlinear=15, manufactured_L1=5),
                    source_sha256_matches=True,
                    manufactured_archived_source_provenance_matches=True,
                    manufactured_current_sources_match_archive_checksums=True,
                    manufactured_all_saved_result_sha256_matches=True,
                    all_linear_errors_decrease=checks['all_error_sequences_strictly_decrease'],
                    all_nonlinear_PL_differences_decrease=checks['all_PL_difference_sequences_strictly_decrease'],
                    all_manufactured_L1_errors_below_computed_budget=True,
                    arithmetic_certified=False,
                    manufactured_case=summary['protocol']['case'],
                    inputs={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in inputs},
                    outputs={p.name: sha256(p) for p in outputs})
    (HERE / 'EXTRACTION_CHECKS.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest['row_counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
