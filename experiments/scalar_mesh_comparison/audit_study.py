"""Read-only audit of saved scalar trajectories, without importing the solver."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mp_fraction(f):
    return mp.mpf(f.numerator)/f.denominator

def ml_series(x,alpha):
    total = mp.mpf(1)
    for k in range(1,5000):
        term = (-x)**k/mp.gamma(1+alpha*k)
        total += term
        if abs(term) < mp.mpf('1e-78') and k > 20:
            return total,k
    raise RuntimeError('Series did not converge')

def main():
    begin = time.time()
    rows = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    checks = json.loads((HERE/'reference_checks.json').read_text(encoding='utf-8'))
    assert len(rows) == 90
    groups = {}
    for row in rows:
        groups.setdefault((row['alpha'],row['mass_exact'],row['m']),[]).append(row)
    assert len(groups) == 30
    for group in groups.values():
        assert {r['mesh'] for r in group} == {'union','uniform','graded'}
        assert len({r['N_actual'] for r in group}) == 1
        assert len({r['history_weight_evaluations'] for r in group}) == 1
    inventory,full_residual_checks,independent_reference_checks = [],[],[]
    rounding_ratio,conversion_bound = 0.,0.
    with mp.workdps(90):
        for row in rows:
            path = HERE/row['trajectory_file']
            with np.load(path,allow_pickle=False) as saved:
                nums = list(map(int,saved['node_numerators']))
                den = int(saved['node_denominator'])
                assert nums[0] == 0 and nums[-1] == den
                assert all(a < b for a,b in zip(nums[:-1],nums[1:]))
                assert row['N_actual'] == len(nums)-1
                node = float(np.max(np.abs(saved['U']-saved['reference'])))
                probe = float(np.max(np.abs(saved['probe_U']-saved['probe_reference'])))
                assert node == row['E_node'] and probe == row['E_PL_sampled']
                assert float(np.max(saved['normalized_residual'])) == row['normalized_full_residual_max']
                # Exact rational rounding inequality, not a floating tolerance.
                info = json.loads(path.with_name(path.stem+'_grid.json').read_text(encoding='utf-8'))
                if row['mesh'] != 'uniform':
                    gamma = Fraction(info['gamma_lower_exact'])
                    error = Fraction(info['absolute_rounding_upper_exact'])
                    assert 16*error <= gamma
                    rounding_ratio = max(rounding_ratio,float(error/gamma))
                    order = info['root_order']
                    enc = info['enclosure_bits']
                    # Verify every rational node against its exact algebraic target.
                    # |rounded-target| <= reported error follows from rational powers.
                    for k,key in zip(nums,info['algebraic_keys_t_power']):
                        key = Fraction(key)
                        t = Fraction(k,den)
                        assert max(Fraction(0),t-error)**order <= key <= (t+error)**order
                conversion_bound = max(conversion_bound,info['scalar_reference_time_conversion_bound'])
                inventory.append({'file':path.name,'sha256':sha(path),'N':len(nums)-1})
                if row['m'] != 256:
                    continue
                a = mp.mpf(str(row['alpha']))
                d = mp_fraction(Fraction(row['mass_exact']))
                nmax = len(nums)-1
                indices = sorted(set([1,2,nmax//4,nmax//2,nmax,
                    1+int(np.argmax(saved['normalized_residual'])),
                    int(np.argmax(np.abs(saved['U']-saved['reference'])))]))
                indices = [n for n in indices if n > 0]
                for n in indices:
                    tn = mp.mpf(nums[n])/den
                    history = mp.mpf(0)
                    oldest = None
                    for j in range(1,n+1):
                        left = mp.mpf(nums[n]-nums[j-1])/den
                        right = mp.mpf(nums[n]-nums[j])/den
                        h = mp.mpf(nums[j]-nums[j-1])/den
                        weight = (left**(1-a)-right**(1-a))/(mp.gamma(2-a)*h)
                        if oldest is None:
                            oldest = weight
                        history += weight*(mp.mpf(float(saved['U'][j]))-mp.mpf(float(saved['U'][j-1])))
                    residual = d*history+mp.mpf(float(saved['U'][n]))-mp.mpf('.2')
                    normalized = abs(residual)/(1+d*oldest)
                    full_residual_checks.append({'file':path.name,'n':n,
                        'normalized_residual_90dp':mp.nstr(normalized,45),
                        'stored_normalized_residual':float(saved['normalized_residual'][n-1]),
                        'discrepancy':abs(float(normalized)-float(saved['normalized_residual'][n-1]))})
                # Independent analytic representation on each finest union grid.
                if row['mesh'] == 'union':
                    for n in sorted(set([1,int(np.argmax(np.abs(saved['U']-saved['reference'])))])):
                        tt = mp.mpf(nums[n])/den
                        x = tt**a/d
                        if x < 5:
                            value,count = ml_series(x,a)
                            ref = mp.mpf('.2')+mp.mpf('.8')*value
                            independent_reference_checks.append({'file':path.name,'n':n,
                                'representation':'Mittag-Leffler defining series at 90 decimal digits',
                                'terms':count,'x':mp.nstr(x,25),'reference':mp.nstr(ref,55),
                                'saved_discrepancy':abs(float(ref)-float(saved['reference'][n]))})
                    # Complex Laplace inversion also tests the large-x endpoint.
                    tt = mp.mpf(nums[-1])/den
                    scaled = tt/d**(1/a)
                    ref = mp.mpf('.2')+mp.mpf('.8')*mp.invertlaplace(
                        lambda z:z**(a-1)/(z**a+1),scaled,method='dehoog',degree=100)
                    independent_reference_checks.append({'file':path.name,'n':nmax,
                        'representation':'de Hoog complex Laplace inversion, degree 100 at 90 decimal digits',
                        'x':mp.nstr(tt**a/d,25),'reference':mp.nstr(ref,55),
                        'saved_discrepancy':abs(float(ref)-float(saved['reference'][-1]))})
    result = {'trajectory_count':len(inventory),'matched_N_groups_checked':len(groups),
        'all_saved_error_maxima_reproduced':True,'all_rational_rounding_enclosures_checked':True,
        'maximum_rounding_error_bound_over_gamma':rounding_ratio,
        'maximum_scalar_reference_time_conversion_bound':conversion_bound,
        'production_high_precision_reference_check_count':sum(len(c['checked_nodes']) for c in checks),
        'production_high_precision_reference_max_discrepancy':max(v['discrepancy'] for c in checks for v in c['checked_nodes']),
        'independent_full_history_residual_check_count':len(full_residual_checks),
        'independent_full_history_residual_max':max(float(v['normalized_residual_90dp']) for v in full_residual_checks),
        'independent_full_history_residual_discrepancy_max':max(v['discrepancy'] for v in full_residual_checks),
        'independent_reference_check_count':len(independent_reference_checks),
        'independent_reference_max_discrepancy':max(v['saved_discrepancy'] for v in independent_reference_checks),
        'inventory':inventory,'full_residual_checks':full_residual_checks,
        'independent_reference_checks':independent_reference_checks,
        'results_json_sha256':sha(HERE/'results.json'),'audit_script_sha256':sha(__file__),
        'elapsed_seconds':time.time()-begin,
        'scope':'No reintegration. Exact rounding checks apply to all saved grids; 90-digit residual checks apply only to listed nodes of all 18 finest trajectories. Finite reference checks are not interval enclosures.'}
    assert result['independent_full_history_residual_max'] < 1e-11
    assert result['independent_reference_max_discrepancy'] < 1e-12
    (HERE/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))

if __name__ == '__main__':
    main()
