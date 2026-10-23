"""Independent scalar order/mesh supplement. No old solver is imported or changed.

Run from repo root: python <this file>
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad, IntegrationWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
U0, LOAD, AMPLITUDE = 1., .2, .8
LEVELS = (16,32,64,128,256)
ENCLOSURE_BITS = 512

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    """Atomic checkpoint, with retry for transient Windows sharing failures."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    for attempt in range(8):
        try:
            temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
            os.replace(temporary,path)
            return
        except OSError:
            if attempt == 7:
                raise
            time.sleep(.1*(attempt+1))

def introot(value, order):
    """Exact floor(value**(1/order)), certified by integer inequalities."""
    if value < 2:
        return value
    x = 1 << ((value.bit_length()+order-1)//order)
    while True:
        y = ((order-1)*x+value//x**(order-1))//order
        if y >= x:
            break
        x = y
    while (x+1)**order <= value:
        x += 1
    while x**order > value:
        x -= 1
    assert x**order <= value < (x+1)**order
    return x

def enclosed_root(key, order, bits=ENCLOSURE_BITS):
    k = introot((key.numerator << (bits*order))//key.denominator,order)
    # [k*2^-bits,(k+1)*2^-bits] is an exact rational enclosure.
    assert k**order*key.denominator <= key.numerator << (bits*order)
    assert (k+1)**order*key.denominator > key.numerator << (bits*order)
    return k

def rational_round(keys, order):
    """Certified rounding before exact union; spacing is never tolerance-merged."""
    keys = sorted(set(keys))
    lows = [enclosed_root(key,order) for key in keys]
    gap_units = min(b-a-1 for a,b in zip(lows[:-1],lows[1:]))
    if gap_units <= 0:
        raise RuntimeError('512-bit root enclosures do not separate nodes')
    # 2^-bits <= gamma/32, hence nearest-node error <= gamma/64.
    bits = ENCLOSURE_BITS-(gap_units.bit_length()-1)+5
    assert 0 < bits < ENCLOSURE_BITS
    den = 1 << bits
    numerators = []
    for key,low in zip(keys,lows):
        k = low >> (ENCLOSURE_BITS-bits)
        midpoint_numerator = 2*k+1
        if key.numerator*(2*den)**order >= midpoint_numerator**order*key.denominator:
            k += 1
        numerators.append(k)
    assert numerators[0] == 0 and numerators[-1] == den
    assert all(a < b for a,b in zip(numerators[:-1],numerators[1:]))
    # Exact upper bound on rounding from the enclosure endpoints.
    rounding_units = max(max(abs((k << (ENCLOSURE_BITS-bits))-l),
                             abs((k << (ENCLOSURE_BITS-bits))-(l+1)))
                         for k,l in zip(numerators,lows))
    assert 16*rounding_units <= gap_units
    info = {'root_order':order,'algebraic_keys_t_power':list(map(str,keys)),
            'enclosure_bits':ENCLOSURE_BITS,'rounding_denominator_bits':bits,
            'gamma_lower_exact':str(Fraction(gap_units,1 << ENCLOSURE_BITS)),
            'absolute_rounding_upper_exact':str(Fraction(rounding_units,1 << ENCLOSURE_BITS)),
            'rounding_upper_over_gamma':rounding_units/gap_units,
            'uniform_parent_kept_exact':True,
            'exact_node_count_before_rounding':len(keys),
            'exact_node_count_after_rounding':len(numerators)}
    return numerators,den,info

def grid(alpha_num, mass, m, mesh, target_n=None):
    if mesh == 'uniform':
        n = target_n
        return list(range(n+1)),n,{'construction':'exact j/N','uniform_parent_kept_exact':True}
    if mesh == 'graded':
        n = target_n
        keys = [Fraction(j,n)**(20-alpha_num) for j in range(n+1)]
        nums,den,info = rational_round(keys,alpha_num)
        info['grading_exponent_exact'] = str(Fraction(20-alpha_num,alpha_num))
        info['construction'] = 'rounded exact algebraic standard graded nodes'
        return nums,den,info
    sqrt_mass = Fraction(1,10**(int(-math.log10(float(mass)))//2))
    assert sqrt_mass*sqrt_mass == mass
    parents = [[(s*j/(m*(1+s)-j))**20 for j in range(m+1)]
               for s in (sqrt_mass,Fraction(1))]
    parents.append([Fraction(j,m)**alpha_num for j in range(m+1)])
    keys = sorted(set().union(*map(set,parents)))
    nums,den,info = rational_round(keys,alpha_num)
    # The m-parent is exact because m is a power of two and divides den.
    assert den % m == 0
    nums_set = set(nums)
    assert all(j*den//m in nums_set for j in range(m+1))
    info['construction'] = 'exact algebraic parent order, certified dyadic parent rounding, exact union'
    info['parent_occurrences'] = 3*(m+1)
    info['exact_duplicate_internal_occurrences_removed'] = 3*(m-1)+1-(len(nums)-1)
    return nums,den,info

def as_float(numerators,den):
    if den & (den-1) == 0:
        bits = den.bit_length()-1
        return np.array([math.ldexp(float(k),-bits) for k in numerators])
    return np.array([float(Fraction(k,den)) for k in numerators])

def weights(nums,den,n,alpha,steps):
    # Every subtraction is exact integer subtraction on the declared rational grid.
    a = as_float([nums[n]-v for v in nums[:n]],den)
    r = steps[:n]/a
    w = np.empty(n)
    far = r < .5
    w[far] = a[far]**(-alpha)*(-np.expm1((1-alpha)*np.log1p(-r[far])))/r[far]
    # For near intervals, form the right distance independently. Computing it
    # as 1-r can lose a tiny but significant (t_n-t_j)**(1-alpha) term.
    near_indices = np.flatnonzero(~far)
    b = as_float([nums[n]-nums[int(j)+1] for j in near_indices],den)
    w[~far] = (a[~far]**(1-alpha)-b**(1-alpha))/steps[:n][~far]
    w /= math.gamma(2-alpha)
    assert np.all(w > 0)
    # Float monotonicity may have several equal remote weights.
    assert np.min(np.diff(w),initial=0) >= -1e-12*np.max(w)
    return w

def solve(nums,den,alpha,mass):
    nmax = len(nums)-1
    u = np.empty(nmax+1); u[0] = U0
    increments = np.empty(nmax)
    residual = np.empty(nmax)
    normalized = np.empty(nmax)
    b_oldest = np.empty(nmax)
    steps = as_float([b-a for a,b in zip(nums[:-1],nums[1:])],den)
    for n in range(1,nmax+1):
        w = weights(nums,den,n,alpha,steps)
        history = float(w[:-1]@increments[:n-1])
        # Correction form; recomputation uses all actually stored increments.
        x = u[n-1]+(LOAD-u[n-1]-mass*history)/(1+mass*w[-1])
        for _ in range(2):
            r = mass*(history+w[-1]*(x-u[n-1]))+x-LOAD
            x -= r/(1+mass*w[-1])
        u[n] = x
        increments[n-1] = u[n]-u[n-1]
        residual[n-1] = mass*float(w@increments[:n])+u[n]-LOAD
        b_oldest[n-1] = w[0]
        normalized[n-1] = abs(residual[n-1])/(1+mass*w[0])
    assert np.max(normalized) < 1e-11
    return u,residual,normalized,b_oldest

def spectral_reference(times,alpha,mass):
    """Positive real spectral integral, not a discrete-time reference."""
    sine,cosine = math.sin(math.pi*alpha),math.cos(math.pi*alpha)
    limit = 50/alpha
    refs,estimates = [],[]
    for t in times:
        if t == 0:
            refs.append(U0); estimates.append(0.); continue
        log_s = math.log(float(t))-math.log(mass)/alpha
        def integrand(y):
            z = log_s+y
            if z > 6.62:
                return 0.
            v = math.exp(-abs(alpha*y))
            density = sine/math.pi*v/(1+2*cosine*v+v*v)
            return math.exp(-math.exp(z))*density
        cuts = sorted(set([-limit,limit]+[v for v in (0.,-log_s,6.62-log_s) if -limit < v < limit]))
        terms,errs = [],[]
        for left,right in zip(cuts[:-1],cuts[1:]):
            value,err = quad(integrand,left,right,epsabs=5e-14,epsrel=5e-14,limit=180)
            terms.append(value); errs.append(err)
        refs.append(LOAD+AMPLITUDE*math.fsum(terms))
        estimates.append(AMPLITUDE*math.fsum(errs))
    return np.array(refs),np.array(estimates)

def hp_spectral(t_exact,alpha_num,mass,decimal_digits=65):
    """Independent high-precision quadrature, with analytic finite-tail bounds."""
    with mp.workdps(decimal_digits):
        t = mp.mpf(t_exact.numerator)/t_exact.denominator
        if not t:
            return mp.mpf(1)
        a = mp.mpf(alpha_num)/10
        d = mp.mpf(mass.numerator)/mass.denominator
        log_s = mp.log(t)-mp.log(d)/a
        sine,cosine = mp.sin(mp.pi*a),mp.cos(mp.pi*a)
        limit = mp.mpf(170)/a
        def integrand(y):
            z = log_s+y
            if z > mp.mpf(6):
                # e^-exp(6)<7e-176, below the working precision.
                return mp.mpf(0)
            v = mp.exp(-abs(a*y))
            return mp.exp(-mp.exp(z))*sine/mp.pi*v/(1+2*cosine*v+v*v)
        # Short subintervals avoid missed peaks on a long integration interval.
        cuts = sorted(set([-limit,limit,mp.mpf(0),-log_s]+[
            mp.mpf(j) for j in range(int(mp.floor(-log_s))-20,int(mp.ceil(-log_s))+8,4)]))
        value = mp.fsum(mp.quad(integrand,[left,right]) for left,right in zip(cuts[:-1],cuts[1:]))
        return +mp.mpf('.2')+mp.mpf('.8')*value

def probes(times,alpha,mass):
    cells = times[:-1,None]+(times[1:]-times[:-1])[:,None]*np.array([.25,.5,.75])[None,:]
    rho = math.exp(math.log(mass)/alpha)
    layers = rho*np.logspace(-8,8,97)
    layers = layers[(layers > 0)&(layers <= 1)]
    # A nonadapted first cell can span many decades after the physical layer.
    # Its reconstruction maximum may lie well beyond a fixed rho-probe window.
    bridge = np.geomspace(rho,times[1],129) if rho < times[1] else np.empty(0)
    return np.unique(np.concatenate((times,cells.ravel(),layers,bridge,np.array([0.,1.]))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    started = time.time()
    if args.pilot:
        for p in (3,7):
            for x in (1e-8,.1,1.,10.,1e8):
                tt = x**(10/p)
                v,err = spectral_reference([tt],p/10,1.)
                hp = hp_spectral(Fraction(tt),p,Fraction(1),55)
                print(p,x,v[0],mp.nstr(hp,20),abs(v[0]-float(hp)),err[0],flush=True)
        print('pilot seconds',time.time()-started)
        return
    warnings.simplefilter('error',IntegrationWarning)
    rows,checks = [],[]
    protected = [Path(__file__).resolve(), ROOT/'experiments/all_masses_vanishing/run_experiment.py', ROOT/'experiments/all_masses_vanishing/results.json']
    input_hashes = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    cache = {}
    for alpha_num in (3,7):
        alpha = alpha_num/10
        for mass in (Fraction(1),Fraction(1,10**4),Fraction(1,10**8)):
            for m in LEVELS:
                union = grid(alpha_num,mass,m,'union')
                n_target = len(union[0])-1
                for mesh in ('union','uniform','graded'):
                    before = time.time()
                    nums,den,grid_info = union if mesh == 'union' else grid(alpha_num,mass,m,mesh,n_target)
                    times = as_float(nums,den)
                    assert np.all(np.diff(times) > 0)
                    u,residual,normalized,b_oldest = solve(nums,den,alpha,float(mass))
                    reference,quad_error = spectral_reference(times,alpha,float(mass))
                    node_error = np.abs(u-reference)
                    eval_times = probes(times,alpha,float(mass))
                    eval_ref,eval_quad_error = spectral_reference(eval_times,alpha,float(mass))
                    eval_u = np.interp(eval_times,times,u)
                    sampled_error = np.abs(eval_u-eval_ref)
                    idx = int(np.argmax(node_error))
                    idx_probe = int(np.argmax(sampled_error))
                    # Check the error maxima and first positive node in high precision.
                    check_times = [Fraction(nums[idx],den),Fraction(nums[1],den),Fraction(float(eval_times[idx_probe]))]
                    local_checks = []
                    for checked in sorted(set(check_times)):
                        key = (alpha_num,mass,checked)
                        if key not in cache:
                            cache[key] = hp_spectral(checked,alpha_num,mass)
                        hp = cache[key]
                        ordinary,_ = spectral_reference([float(checked)],alpha,float(mass))
                        local_checks.append({'time_exact':str(checked),
                            'reference_65_digits':mp.nstr(hp,60),
                            'float64_reference':float(ordinary[0]),
                            'discrepancy':abs(float(hp)-ordinary[0])})
                    check = {'alpha':alpha,'mass_exact':str(mass),'m':m,'mesh':mesh,
                             'decimal_digits':65,'checked_nodes':local_checks}
                    checks.append(check)
                    name = f'a{alpha_num}_d{mass.denominator}_m{m:04d}_{mesh}'
                    np.savez_compressed(HERE/(name+'.npz'),
                        t=times,U=u,reference=reference,quad_error_estimate=quad_error,
                        residual=residual,normalized_residual=normalized,b_oldest=b_oldest,
                        node_numerators=np.array(list(map(str,nums))),node_denominator=np.array(str(den)),
                        probe_t=eval_times,probe_U=eval_u,probe_reference=eval_ref,
                        probe_quad_error_estimate=eval_quad_error)
                    relative_time_conversion = max(abs(Fraction(float(t))-Fraction(k,den))/Fraction(k,den)
                                                   for t,k in zip(times[1:],nums[1:]))
                    grid_info['float64_time_max_relative_conversion_error'] = str(relative_time_conversion)
                    grid_info['scalar_reference_time_conversion_bound'] = AMPLITUDE/math.e*float(relative_time_conversion)/(1-float(relative_time_conversion))
                    write_json(HERE/(name+'_grid.json'),grid_info)
                    row = {'alpha':alpha,'mass':float(mass),'mass_exact':str(mass),'m':m,'mesh':mesh,
                           'N_actual':len(nums)-1,'history_weight_evaluations':len(nums)*(len(nums)-1)//2,
                           'E_node':float(np.max(node_error)),
                           'E_PL_sampled':float(np.max(sampled_error)),
                           'node_error_max_t':float(times[idx]),'sampled_error_max_t':float(eval_times[idx_probe]),
                           'normalized_full_residual_max':float(np.max(normalized)),
                           'raw_residual_max':float(np.max(np.abs(residual))),
                           'quad_error_estimate_max':float(max(np.max(quad_error),np.max(eval_quad_error))),
                           'hp_check_discrepancy_max':float(max(v['discrepancy'] for v in local_checks)),
                           'minimum_step':float(np.min(np.diff(times))),'probe_count':len(eval_times),
                           'seconds_including_reference_checks':time.time()-before,'trajectory_file':name+'.npz'}
                    previous = [r for r in rows if r['alpha']==alpha and r['mass']==float(mass) and r['mesh']==mesh]
                    for field in ('E_node','E_PL_sampled'):
                        row['q_N_'+field] = math.log(previous[-1][field]/row[field])/math.log(row['N_actual']/previous[-1]['N_actual']) if previous else None
                    rows.append(row)
                    print(json.dumps({k:row[k] for k in ('alpha','mass','m','mesh','N_actual','E_node','E_PL_sampled','normalized_full_residual_max','seconds_including_reference_checks')}),flush=True)
                    write_json(HERE/'results.json',rows)
                    write_json(HERE/'reference_checks.json',checks)
    with (HERE/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    assert input_hashes == {str(p.relative_to(ROOT)):sha(p) for p in protected}
    provenance = {'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'mpmath':mp.__version__,'platform':platform.platform(),'script_sha256':sha(__file__),
        'original_inputs_unchanged_sha256':input_hashes,'elapsed_seconds':time.time()-started,
        'command':' python '+str(Path(__file__).relative_to(ROOT)),
        'run_count':len(rows),'reference':'positive spectral integral on log spectral variable; SciPy adaptive quadrature; selected 65-decimal-digit mpmath quadrature checks',
        'reference_truncation_log_density_exponent':50,
        'hp_reference_truncation_log_density_exponent':170,
        'rho_probe_multipliers':'97 logspace points from 1e-8 to 1e8, clipped to (0,1], plus 129 geometric points from rho to firstnode when rho<firstnode, all nodes and all quarter/mid/three-quarter cell points',
        'residual':'recomputed complete stored increment history, normalized by 1+d*b_oldest',
        'scope':'scalar linear fixed-data unprepared relaxation; no nonlinear or general-graph claim',
        'files':{p.name:sha(p) for p in HERE.glob('*.npz')}}
    write_json(HERE/'PROVENANCE.json',provenance)
    print('COMPLETED',len(rows),'runs',time.time()-started,'seconds',flush=True)

if __name__ == '__main__':
    main()
