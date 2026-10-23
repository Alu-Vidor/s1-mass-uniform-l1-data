"""Sensitivity of the independent PI reference to a different grading.

The default rechecks archived cross-grade solutions.  Pass --recompute to
recompute all four PI trajectories. This is a grading sensitivity diagnostic,
not an additional independent method or an interval certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reference import solve_reference, evaluate_reference
from run_study import C, B, EDGE, EDGE_CUBIC, U0, LOAD, RATIOS

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    results = json.loads((HERE/'results.json').read_text())
    probes = np.unique(np.r_[0, np.geomspace(1e-30, 1, 601),
                             np.linspace(0, 1, 101)])
    checks = []
    for alpha in (.3, .7):
        for delta in (1, .01):
            masslabel = '1' if delta == 1 else '1_100'
            label = f'reference_a{round(alpha*10)}_delta_{masslabel}_K32768'
            meta = json.loads((HERE/(label+'.json')).read_text())
            data = np.load(HERE/(label+'.npz'))
            production = dict(meta)
            production.update({k:data[k] for k in ('nodes','U','forcing','masses')})
            filename = f'cross_grade_a{round(alpha*10)}_delta{str(delta).replace(".","_")}.npz'
            archive = HERE/filename
            if args.recompute or not archive.exists():
                masses = delta*np.array([float(r) for r in RATIOS])
                cross = solve_reference(alpha,masses,C,B,EDGE,EDGE_CUBIC,U0,
                                        LOAD,16384,grade=6/alpha,probes=probes)
                np.savez_compressed(archive,**{k:cross[k] for k in
                    ('nodes','U','forcing','probes','probe_U','masses')})
            cross = np.load(archive)
            diff = abs(evaluate_reference(production,cross['probes'])-cross['probe_U'])
            where = np.unravel_index(np.argmax(diff),diff.shape)
            finest = next(c for c in results['reference_checks']
                          if c['alpha'] == alpha and (c['delta']=='1') == (delta==1))
            ratio = float(diff.max()/finest['finest_L1_nodal_error'])
            assert ratio < .01
            checks.append(dict(alpha=alpha,delta=finest['delta'],
                cross_reference_K=16384,cross_grade=6/alpha,
                production_reference_K=32768,production_grade=4/alpha,
                probe_count=len(cross['probes']),max_difference=float(diff.max()),
                argmax_time=float(cross['probes'][where[0]]),
                difference_over_L1_finest_nodal_error=ratio,
                cross_archive=filename,
                cross_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    output = dict(status='PASS',
        nature='Alternative grading sensitivity diagnostic for the same independent PI method, not an additional independent method or interval certificate',
        checks=checks)
    (HERE/'CROSS_GRADE_CHECK.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    main()
