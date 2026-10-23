"""Independent finite audit of saved nonlinear L1 and PI data."""
from __future__ import annotations
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    result=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    checks=[]
    hp=[]
    for row in result['rows']:
        arr=np.load(HERE/(row['case']+'.npz'))
        grid=json.loads((HERE/(row['case']+'_grid.json')).read_text(encoding='utf-8'))
        nums=list(map(int,grid['numerators']));den=int(grid['denominator'])
        assert len(nums)==row['N']+1 and np.all(np.diff(arr['times'])>0)
        assert nums[0]==0 and nums[-1]==den and all(a<b for a,b in zip(nums[:-1],nums[1:]))
        assert np.array_equal(arr['increments'],np.diff(arr['U'],axis=0))
        assert abs(float(np.max(arr['normalized_residuals']))-row['max_normalized_residual'])<1e-25
        assert np.max(arr['normalized_residuals'])<1e-11
        info=grid['grid_info']
        gamma=Fraction(info['gamma_lower_exact']);rounding=Fraction(info['absolute_rounding_upper_exact'])
        assert 16*rounding<=gamma
        assert den%row['m']==0
        assert all(j*den//row['m'] in set(nums) for j in range(row['m']+1))
        checks.append(dict(case=row['case'],N=row['N'],grid_rounding_upper_over_gamma=float(rounding/gamma),
                           finite_arrays=bool(np.all(np.isfinite(arr['U']))),
                           normalized_residual_max=float(np.max(arr['normalized_residuals']))))
        if row['m']!=256:
            continue
        indices=sorted(set([1,row['N']//4,row['N']//2,row['N'],int(np.argmax(np.max(arr['normalized_residuals'],axis=1)))+1]))
        with mp.workdps(85):
            alpha=mp.mpf(row['alpha_numerator'])/10
            delta=Fraction(row['delta'])
            masses=[mp.mpf((delta*r).numerator)/(delta*r).denominator for r in (Fraction(1),Fraction(1,4),Fraction(1,16))]
            times=[mp.mpf(k)/den for k in nums]
            state=[[mp.mpf(float(x)) for x in v] for v in arr['U']]
            for n in indices:
                weights=[((times[n]-times[j])**(1-alpha)-(times[n]-times[j+1])**(1-alpha))
                         /((times[j+1]-times[j])*mp.gamma(2-alpha)) for j in range(n)]
                v=state[n];z=(v[0]-v[1],v[1]-v[2])
                currents=[w*(zz+mp.mpf('.1')*zz**3) for w,zz in zip((5,3),z)]
                force=[v[0]+currents[0],mp.mpf('1.3')*v[1]-currents[0]+currents[1],mp.mpf('.9')*v[2]-currents[1]]
                residual=[masses[i]*mp.fsum(weights[j]*(state[j+1][i]-state[j][i]) for j in range(n))+force[i]-load
                          for i,load in enumerate((mp.mpf('.2'),mp.mpf('-.1'),mp.mpf('.3')))]
                normalized=[abs(residual[i])/(c+masses[i]*weights[0]) for i,c in enumerate((mp.mpf(1),mp.mpf('1.3'),mp.mpf('.9')))]
                hp.append(dict(case=row['case'],n=n,normalized_residual=float(max(normalized)),
                               discrepancy_from_stored=float(max(abs(normalized[i]-mp.mpf(float(arr['normalized_residuals'][n-1,i]))) for i in range(3)))))
    for check in result['reference_checks']:
        assert check['reference_discrepancy_over_finest_L1_error']<.01
        assert check['successive_probe_discrepancies'][1]<check['successive_probe_discrepancies'][0]
    for meta_file in HERE.glob('reference_a*_K*.json'):
        meta=json.loads(meta_file.read_text(encoding='utf-8'))
        assert hashlib.sha256(meta_file.with_suffix('.npz').read_bytes()).hexdigest()==meta['array_sha256']
    audit={'status':'passed','trajectory_count':len(checks),'grid_and_saved_history_checks':checks,
           'high_precision_selected_node_residuals':hp,'high_precision_digits':85,
           'max_high_precision_normalized_residual':max(x['normalized_residual'] for x in hp),
           'max_discrepancy_from_stored_residual':max(x['discrepancy_from_stored'] for x in hp),
           'reference_levels_per_family':3,'reference_family_count':len(result['reference_checks']),
           'reference_validation_file':'REFERENCE_VALIDATION.json',
           'limitations':'Finite selected-node and refinement checks are not interval certificates of global residual or reference error.'}
    (HERE/'AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('grid_and_saved_history_checks','high_precision_selected_node_residuals')},indent=2))

if __name__=='__main__':
    main()
