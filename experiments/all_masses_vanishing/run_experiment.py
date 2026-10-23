#!/usr/bin/env python3
"""Reproducible S1 all-masses-vanishing experiment; alpha=1/2, T=1.

Run from the repository root with requirements-numerics.txt installed.
Source solver is read/hashed only.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
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

import mpmath as mp
import numpy as np
import scipy
from scipy.special import erfcx, gamma
from contextlib import nullcontext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'legacy/mass3/mass3_l1_check.py'
C = np.array([1.,1.3,.9])
B = np.array([[1.,0.],[-1.,1.],[0.,-1.]])
A = np.diag(C)+(B*np.array([5.,3.]))@B.T
U0 = np.array([1.,-.4,.2])
P0 = np.array([.4,-.2,.3])
P1 = np.array([.2,.1,-.15])
GAMMA15 = math.sqrt(math.pi)/2
FULL_RESIDUAL_SUCCESS_THRESHOLD = 1e-12

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_mesh(sqrts, m):
    parents = [[(s*j/(m*(1+s)-j))**4 for j in range(m+1)]
               for s in (*sqrts,Fraction(1))]
    parents.append([Fraction(j,m) for j in range(m+1)])
    nodes = sorted(set().union(*map(set,parents)))
    assert nodes[0] == 0 and nodes[-1] == 1
    assert m <= len(nodes)-1 <= 5*m-4
    return nodes

def weights(nodes,n):
    # Subtraction is exact rational; float conversion follows subtraction.
    distances=np.array([float(nodes[n]-nodes[j]) for j in range(n+1)])
    roots=np.sqrt(distances)
    result=1/(GAMMA15*(roots[:-1]+roots[1:]))
    assert np.all(result>0) and np.all(np.diff(result)>=0)
    return result

def spd3(a,rhs):
    # Explicit Cholesky, matching the source algorithm's 3x3 correction solve.
    l00=np.sqrt(a[0,0]); l10=a[1,0]/l00; l20=a[2,0]/l00
    l11=np.sqrt(a[1,1]-l10*l10)
    l21=(a[2,1]-l20*l10)/l11
    l22=np.sqrt(a[2,2]-l20*l20-l21*l21)
    y0=rhs[0]/l00; y1=(rhs[1]-l10*y0)/l11
    y2=(rhs[2]-l20*y0-l21*y1)/l22
    x2=y2/l22; x1=(y1-l21*x2)/l11
    x0=(y0-l10*x1-l20*x2)/l00
    return np.array([x0,x1,x2])

def solve(nodes,masses):
    nmax=len(nodes)-1
    values=np.zeros((nmax+1,3)); values[0]=U0
    increments=np.zeros((nmax,3))
    residuals=np.zeros((nmax,3)); normalized=np.zeros((nmax,3))
    updates=0
    for n in range(1,nmax+1):
        w=weights(nodes,n)
        hist=w[:-1]@increments[:n-1]
        p=P0+float(nodes[n])*P1
        denom=C+masses*w[0]
        x=values[n-1].copy()
        matrix=A+np.diag(masses*w[-1])
        # Correction form avoids subtracting huge right-hand-side terms.
        for k in range(4):
            r=masses*(w[-1]*(x-values[n-1])+hist)+A@x-p
            if np.max(np.abs(r)/denom)<2e-14:
                break
            x+=spd3(matrix,-r)
            updates+=1
        values[n]=x
        increments[n-1]=values[n]-values[n-1]
        # Reevaluate full history from the actually stored states.
        residuals[n-1]=masses*(w@increments[:n])+A@values[n]-p
        normalized[n-1]=np.abs(residuals[n-1])/denom
        if np.max(normalized[n-1]) > FULL_RESIDUAL_SUCCESS_THRESHOLD:
            raise RuntimeError(f'Full stored-history residual exceeded success threshold at step {n}')
    return values,residuals,normalized,updates

def mp_matrices(masses,precision=80):
    with mp.workdps(precision):
        ma=mp.matrix([['6','-5','0'],['-5','9.3','-3'],['0','-3','3.9']])
        ea,qa=mp.eigsy(ma)
        ah=qa*mp.diag([mp.sqrt(v) for v in ea])*qa.T
        aih=qa*mp.diag([1/mp.sqrt(v) for v in ea])*qa.T
        masses_mp=[mp.mpf(d.numerator)/d.denominator for d in masses]
        h=ah*mp.diag([1/d for d in masses_mp])*ah
        eigen,q=mp.eigsy(h)
        left=aih*q
        u0=mp.matrix(['1','-.4','.2'])
        p0=mp.matrix(['.4','-.2','.3']); p1=mp.matrix(['.2','.1','-.15'])
        y0=q.T*ah*u0; a=q.T*aih*p0; b=q.T*aih*p1
        return eigen,left,y0,a,b

def affine_g(x):
    x=np.asarray(x)
    result=np.empty_like(x)
    small=x<.5
    v=x[small]
    acc=np.zeros_like(v)
    for k in range(1,45):
        acc+=((-1.)**(k-1)/gamma(2+k/2))*v**k
    result[small]=acc
    v=x[~small]
    result[~small]=1-2/(np.sqrt(np.pi)*v)+(1-erfcx(v))/v**2
    return result

def reference_factory(masses):
    mats=mp_matrices(masses,80)
    eigen,left,y0,a,b=mats
    ev=np.array(list(eigen),float)
    le=np.array(left.tolist(),float)
    y=np.array(list(y0),float); aa=np.array(list(a),float); bb=np.array(list(b),float)
    def reference(times):
        tt=np.atleast_1d(np.asarray(times,float))
        x=np.sqrt(tt[:,None])*ev[None,:]
        modal=aa[None,:]+(y-aa)[None,:]*erfcx(x)+tt[:,None]*bb[None,:]*affine_g(x)
        result=modal@le.T
        result[tt==0]=U0
        return result
    return reference,{'eigenvalues':[str(v) for v in eigen],
        'eigensystem_precision_decimal_digits':80,
        'formula_evaluation':'float64 erfcx; Taylor branch x<0.5 for affine response'}

def mp_erfcx(x):
    if x<20:
        return mp.exp(x*x)*mp.erfc(x)
    term=mp.mpf(1); total=term
    for k in range(1,1200):
        term*=-(2*k-1)/(2*x*x)
        total+=term
        if abs(term)<mp.eps*abs(total):
            return total/(mp.sqrt(mp.pi)*x)
    raise RuntimeError('asymptotic erfcx did not converge')

def mp_reference(masses,times,precision=120):
    with mp.workdps(precision):
        eigen,left,y0,a,b=mp_matrices(masses,precision)
        results=[]
        for t in times:
            tt=mp.mpf(t.numerator)/t.denominator
            if tt==0:
                results.append([mp.mpf('1'),mp.mpf('-.4'),mp.mpf('.2')]);continue
            modal=[]
            for j in range(3):
                x=eigen[j]*mp.sqrt(tt); ee=mp_erfcx(x)
                if x<mp.mpf('.5'):
                    gg=mp.fsum([(-1)**(k-1)*x**k/mp.gamma(2+mp.mpf(k)/2)
                                for k in range(1,180)])
                else:
                    gg=1-2/(mp.sqrt(mp.pi)*x)+(1-ee)/(x*x)
                modal.append(a[j]+(y0[j]-a[j])*ee+b[j]*tt*gg)
            results.append(list(left*mp.matrix(modal)))
        return results

def high_precision_check(masses,nodes,values,reference):
    times=np.array([float(t) for t in nodes])
    indices={0,1,len(nodes)-1,int(np.argmax(np.max(np.abs(values-reference(times)),axis=1)))}
    # Every physical layer rho_i=d_i**2, plus the macroscopic background.
    targets=[Fraction(1,2),Fraction(1)]
    for d in masses:
        for factor in (Fraction(1,10**6),Fraction(1,1000),Fraction(1),Fraction(1000),Fraction(10**6)):
            targets.append(min(Fraction(1),d*d*factor))
    for target in targets:
        k=int(np.searchsorted(times,float(target)))
        for j in (max(0,k-1),min(k,len(nodes)-1)):
            indices.add(j)
    indices=sorted(indices)
    chosen=[nodes[i] for i in indices]
    hi=mp_reference(masses,chosen,120)
    hi_float=np.array([[float(v) for v in row] for row in hi])
    double=reference([float(t) for t in chosen])
    err=np.abs(double-hi_float)
    return {'decimal_digits':120,'number_of_checked_nodes':len(chosen),
            'max_float_reference_discrepancy':float(np.max(err)),
            'selection':'0, first node, terminal, nodal error maximum, neighbors of d_i^2 times 1e-6,1e-3,1,1e3,1e6 and t=1/2,1',
            'nodes':[{'index':i,'t_exact':str(t),'t':float(t),
                      'reference_120dp':[mp.nstr(v,70) for v in row],
                      'float_reference':list(map(float,ref)),
                      'max_discrepancy':float(np.max(np.abs(ref-np.array([float(v) for v in row]))))}
                     for i,t,row,ref in zip(indices,chosen,hi,double)]}

def run_case(path,delta,sqrts,ms,high_check,smoke=False):
    masses_exact=[s*s for s in sqrts]
    masses=np.array([float(d) for d in masses_exact])
    reference,diagnostic=reference_factory(masses_exact)
    rows=[];checks=[]
    for m in ms:
        begin=time.monotonic()
        nodes=make_mesh(sqrts,m); actual_n=len(nodes)-1
        values,raw_residual,normalized,updates=solve(nodes,masses)
        times=np.array([float(t) for t in nodes]); exact=reference(times)
        errors=np.abs(values-exact); nodal=float(np.max(errors))
        probes=[(nodes[i-1]*(4-j)+nodes[i]*j)/4
                for i in range(1,len(nodes)) for j in (1,2,3)]
        probe_times=np.array([float(t) for t in probes])
        probe_ref=reference(probe_times)
        theta=np.tile(np.array([.25,.5,.75]),actual_n)[:,None]
        repeated_left=np.repeat(values[:-1],3,axis=0)
        repeated_right=np.repeat(values[1:],3,axis=0)
        pl=(1-theta)*repeated_left+theta*repeated_right
        pl_error=max(nodal,float(np.max(np.abs(pl-probe_ref))))
        exact_pl=(1-theta)*np.repeat(exact[:-1],3,axis=0)+theta*np.repeat(exact[1:],3,axis=0)
        tag=f'{path}_delta_{str(delta).replace("/","_")}_m{m:04}'
        if smoke: tag='smoke_'+tag
        out=HERE/(tag+'.npz')
        np.savez_compressed(out,times=times,
            node_numerators=np.array([str(t.numerator) for t in nodes]),
            node_denominators=np.array([str(t.denominator) for t in nodes]),
            states=values,reference=exact,absolute_errors=errors,
            full_history_residual=raw_residual,normalized_full_history_residual=normalized,
            probe_times=probe_times,probe_reference=probe_ref,probe_PL_states=pl,
            masses=masses,masses_exact=np.array(list(map(str,masses_exact))),
            delta_exact=str(delta),path=path,m=m,actual_N=actual_n,
            minimum_exact_step=str(min(b-a for a,b in zip(nodes,nodes[1:]))))
        row={'path':path,'delta':float(delta),'delta_exact':str(delta),
             'm':m,'actual_N':actual_n,'masses_exact':list(map(str,masses_exact)),
             'minimum_step':float(min(b-a for a,b in zip(nodes,nodes[1:]))),
             'exact_duplicate_removals':5*(m-1)+1-actual_n,
             'nodal_error':nodal,'nodal_error_by_component':np.max(errors,axis=0).tolist(),
             'sampled_PL_error':pl_error,
             'sampled_exact_interpolation_error':float(np.max(np.abs(exact_pl-probe_ref))),
             'scaled_nodal_error_N_1p5':nodal*actual_n**1.5,
             'scaled_sampled_PL_error_N_1p5':pl_error*actual_n**1.5,
             'max_normalized_full_history_residual':float(np.max(normalized)),
             'full_residual_success_threshold':FULL_RESIDUAL_SUCCESS_THRESHOLD,
             'max_absolute_full_history_residual':float(np.max(np.abs(raw_residual))),
             'correction_updates':updates,'history_terms_including_residual':actual_n**2,
             'source_npz':out.name,'source_npz_sha256':sha(out)}
        if rows:
            old=rows[-1]
            ratio=math.log(actual_n/old['actual_N'])
            row['observed_nodal_order_actual_N']=math.log(old['nodal_error']/nodal)/ratio
            row['observed_sampled_PL_order_actual_N']=math.log(old['sampled_PL_error']/pl_error)/ratio
        if high_check and m==max(ms):
            check=high_precision_check(masses_exact,nodes,values,reference)
            check.update({'path':path,'delta_exact':str(delta),'m':m})
            checks.append(check)
            row['reference_120dp_check_discrepancy']=check['max_float_reference_discrepancy']
        row['elapsed_seconds']=time.monotonic()-begin
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    return rows,checks,diagnostic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--ms',default='16,32,64,128,256,512')
    args=ap.parse_args()
    ms=[16,32] if args.smoke else [int(x) for x in args.ms.split(',')]
    deltas=[Fraction(1),Fraction(1,100),Fraction(1,10000),Fraction(1,10**6)]
    if args.smoke:deltas=[Fraction(1,10**6)]
    begin=time.monotonic();initial_sha=sha(SOURCE)
    allrows=[];allchecks=[];diagnostics=[]
    with nullcontext():
        for path in ('fixed_ratios','changing_ratios'):
            for delta in deltas:
                ds=Fraction(math.isqrt(delta.numerator),math.isqrt(delta.denominator))
                assert ds*ds==delta
                sqrts=([ds,ds/2,ds/4] if path=='fixed_ratios'
                       else [ds,delta,delta*ds])
                rows,checks,diag=run_case(path,delta,sqrts,ms,
                                        delta in (Fraction(1),Fraction(1,10**6)),smoke=args.smoke)
                allrows.extend(rows);allchecks.extend(checks)
                diagnostics.append({'path':path,'delta_exact':str(delta),**diag})
        pools='One-thread environment set before NumPy/SciPy import; threadpoolctl unavailable in unchanged environment'
    assert initial_sha==sha(SOURCE),'original solver changed during run'
    report={'purpose':'New S1 full-history L1 experiment: all physical masses vanish',
        'alpha':.5,'T':1,'C':C.tolist(),'B':B.tolist(),'A':A.tolist(),
        'initial':U0.tolist(),'load_intercept':P0.tolist(),'load_slope':P1.tolist(),
        'load':'p(t)=p0+t*p1; independent of masses; bounded C2 on [0,1]',
        'initial_mismatch_p0_minus_AU0':(P0-A@U0).tolist(),
        'fixed_ratios':'diag(delta,delta/4,delta/16)',
        'changing_ratios':'diag(delta,delta**2,delta**3)',
        'mesh':'Exact Fraction union of all three physical mass parents, eta0=1 parent, uniform parent; remove exact equalities only',
        'original_solver':str(SOURCE),'original_solver_sha256':initial_sha,
        'experiment_script':str(Path(__file__).resolve()),'experiment_script_sha256':sha(Path(__file__)),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'mpmath':mp.__version__,'platform':platform.platform(),'executable':sys.executable,
            'longdouble_epsilon':float(np.finfo(np.longdouble).eps),
            'longdouble_equals_float64':bool(np.finfo(np.longdouble).eps==np.finfo(np.float64).eps),
            'thread_environment':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')},
            'threadpools_during_run':pools},
        'residual_definition':'max_{n,i}|d_i sum_{j=1}^n b_nj(U_i^j-U_i^(j-1))+(AU^n)_i-p_i(t_n)|/(c_i+d_i*b_n1); exact rational time distances rounded once before sqrt; not interval certified',
        'sampled_PL_definition':'Every knot and theta=1/4,1/2,3/4 in every exact rational cell; sampled quantity, not certified continuous supremum',
        'limits':['Finite selected masses do not prove mass-uniformity.','No certified supremum is claimed for sampled PL error.','Float64 residuals and references are diagnostics, not interval certificates.','Runtime is descriptive, not a speedup or bit-complexity claim.'],
        'reference_diagnostics':diagnostics,'runs':allrows,'high_precision_checks':allchecks,
        'elapsed_seconds':time.monotonic()-begin,'command_arguments':sys.argv[1:]}
    stem='smoke' if args.smoke else 'results'
    (HERE/(stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    keys=list(dict.fromkeys(k for row in allrows for k in row))
    with (HERE/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in allrows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print(f'COMPLETE {stem}: {len(allrows)} runs, elapsed {report["elapsed_seconds"]:.3f}s',flush=True)

if __name__=='__main__':main()
