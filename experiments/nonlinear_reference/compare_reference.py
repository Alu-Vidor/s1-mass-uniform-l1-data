"""Run a separate Volterra product-integration reference and compare L1 output."""
from __future__ import annotations
import os
for _var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_var]='1'
import csv
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np
from reference import solve_reference,evaluate_reference
from run_study import B,C,EDGE,EDGE_CUBIC,U0,LOAD,RATIOS,HERE,helpers,sha

REFERENCE_LEVELS=(8192,16384,32768)

def run():
    started=time.time()
    l1=json.loads((HERE/'l1_results.json').read_text(encoding='utf-8'))
    rows=l1['rows']
    checks=[]
    for p in (3,7):
        for delta in ('1','1/100'):
            family=[r for r in rows if r['alpha_numerator']==p and r['delta']==delta]
            assert len(family)==5
            archives=[np.load(HERE/(r['case']+'.npz')) for r in family]
            # Every reported nodal error and the finest sampled reconstruction
            # error are checked on precisely the same probes at every K.
            probes=np.unique(np.concatenate([a['times'] for a in archives]+[archives[-1]['probes']]))
            masses=np.array([float(Fraction(delta)*r) for r in RATIOS])
            values=[]
            for K in REFERENCE_LEVELS:
                label=f'reference_a{p}_delta_{delta.replace("/","_")}_K{K}'
                target=HERE/(label+'.npz')
                if target.exists():
                    saved=np.load(target)
                    assert np.array_equal(saved['probes'],probes)
                    v=saved['probe_U']
                    meta=json.loads((HERE/(label+'.json')).read_text(encoding='utf-8'))
                else:
                    ref=solve_reference(p/10,masses,C,B,EDGE,EDGE_CUBIC,U0,LOAD,K,grade=4/(p/10))
                    print('solved',label,'seconds',ref['elapsed_seconds'],flush=True)
                    query_start=time.time()
                    v=evaluate_reference(ref,probes)
                    print('evaluated',label,'probes',len(probes),'seconds',time.time()-query_start,flush=True)
                    np.savez_compressed(target,nodes=ref['nodes'],U=ref['U'],forcing=ref['forcing'],
                                        probes=probes,probe_U=v,masses=masses)
                    meta={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                          for k,v in ref.items() if k not in ('nodes','U','forcing')}
                    meta['probe_count']=len(probes)
                    meta['array_sha256']=sha(target)
                    helpers.write_json(HERE/(label+'.json'),meta)
                values.append(v)
            differences=[float(np.max(abs(b-a))) for a,b in zip(values[:-1],values[1:])]
            reference=values[-1]
            prior=None
            for row,a in zip(family,archives):
                positions=np.searchsorted(probes,a['times'])
                assert np.array_equal(probes[positions],a['times'])
                errors=np.max(abs(a['U']-reference[positions]),axis=1)
                index=int(np.argmax(errors))
                row['nodal_error']=float(errors[index])
                row['nodal_error_time']=float(a['times'][index])
                row['reference_refinement_discrepancy']=differences[-1]
                row['reference_K']=REFERENCE_LEVELS[-1]
                if prior:
                    row['rate_actual_N']=float(np.log(prior['nodal_error']/row['nodal_error'])/np.log(row['N']/prior['N']))
                prior=row
            fine=archives[-1]
            positions=np.searchsorted(probes,fine['probes'])
            assert np.array_equal(probes[positions],fine['probes'])
            sampled_errors=np.max(abs(fine['reconstruction']-reference[positions]),axis=1)
            index=int(np.argmax(sampled_errors))
            family[-1]['sampled_PL_error']=float(sampled_errors[index])
            family[-1]['sampled_PL_error_time']=float(fine['probes'][index])
            item={'alpha':p/10,'delta':delta,'reference_levels':list(REFERENCE_LEVELS),
                  'grade':4/(p/10),'probe_count':len(probes),'successive_probe_discrepancies':differences,
                  'reference_increment_observed_order':float(np.log2(differences[0]/differences[1])),
                  'finest_L1_nodal_error':family[-1]['nodal_error'],
                  'finest_L1_sampled_error':family[-1]['sampled_PL_error'],
                  'reference_discrepancy_over_finest_L1_error':differences[-1]/family[-1]['nodal_error'],
                  'caution':'Refinement discrepancy is numerical evidence, not a certified reference error bound.'}
            checks.append(item)
            print(json.dumps(item),flush=True)
            helpers.write_json(HERE/'results_checkpoint.json',{'rows':rows,'reference_checks':checks})
    assert all(c['reference_discrepancy_over_finest_L1_error']<.01 for c in checks)
    output={'parameters':l1['parameters'],'rows':rows,'reference_checks':checks,
            'reference_method':'Implicit piecewise-linear product integration of the equivalent Volterra equation; continuous forcing-integral reconstruction.',
            'reference_code_sha256':sha(HERE/'reference.py'),
            'l1_code_sha256':sha(HERE/'run_study.py'),
            'environment':l1['environment'],'elapsed_seconds':time.time()-started}
    helpers.write_json(HERE/'results.json',output)
    columns=sorted(set().union(*[set(r) for r in rows]))
    with (HERE/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print('ALL REFERENCE CHECKS PASSED',time.time()-started,flush=True)

if __name__=='__main__':
    run()
