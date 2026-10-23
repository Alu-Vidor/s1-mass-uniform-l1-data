"""Nonlinear network L1 study with an independent Volterra PI reference.

No earlier data or solver files are changed. The grid/weight helper is read-only.
"""
from __future__ import annotations
import os
for _var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy

HERE=Path(__file__).resolve().parent
S1=HERE.parents[1]
_spec=importlib.util.spec_from_file_location('frozen_scalar_helpers',HERE.parent/'scalar_mesh_comparison/run_study.py')
helpers=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
B=np.array([[1.,0.],[-1.,1.],[0.,-1.]])
C=np.array([1.,1.3,.9])
EDGE=np.array([5.,3.])
EDGE_CUBIC=.1
U0=np.array([1.,-.5,.7])
LOAD=np.array([.2,-.1,.3])
RATIOS=(Fraction(1),Fraction(1,4),Fraction(1,16))
LEVELS=(16,32,64,128,256)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def F(u):
    z=B.T@u
    return C*u+B@(EDGE*(z+EDGE_CUBIC*z**3))

def jac(u):
    z=B.T@u
    return np.diag(C)+B@np.diag(EDGE*(1+3*EDGE_CUBIC*z*z))@B.T

def exact_grid(alpha_num, delta, m, mesh='union',target_n=None):
    if mesh!='union':
        return helpers.grid(alpha_num,Fraction(1),m,mesh,target_n)
    # All masses are rational squares; keys equal t**alpha_num exactly.
    scales=sorted(set([Fraction(1)]+[delta*r for r in RATIOS]))
    parents=[]
    for d in scales:
        sn=math.isqrt(d.numerator); sd=math.isqrt(d.denominator)
        assert sn*sn==d.numerator and sd*sd==d.denominator
        root=Fraction(sn,sd)
        parents.append([(root*j/(m*(1+root)-j))**20 for j in range(m+1)])
    parents.append([Fraction(j,m)**alpha_num for j in range(m+1)])
    keys=sorted(set().union(*map(set,parents)))
    nums,den,info=helpers.rational_round(keys,alpha_num)
    assert den%m==0
    node_set=set(nums)
    assert all(j*den//m in node_set for j in range(m+1))
    info['construction']='exact algebraic keys, certified dyadic rounding, exact union'
    info['parent_scales']=list(map(str,scales))
    info['uniform_parent_kept_exact']=True
    return nums,den,info

def solve_l1(nums,den,alpha,masses):
    nmax=len(nums)-1
    u=np.empty((nmax+1,3));u[0]=U0
    increments=np.empty((nmax,3))
    residuals=np.empty((nmax,3)); normalized=np.empty((nmax,3))
    newton_counts=np.empty(nmax,dtype=int)
    oldest=np.empty(nmax)
    steps=helpers.as_float([b-a for a,b in zip(nums[:-1],nums[1:])],den)
    for n in range(1,nmax+1):
        w=helpers.weights(nums,den,n,alpha,steps)
        history=w[:-1]@increments[:n-1]
        x=u[n-1].copy()
        normalizer=C+masses*w[0]
        for iteration in range(15):
            residual=masses*(history+w[-1]*(x-u[n-1]))+F(x)-LOAD
            if np.max(abs(residual)/normalizer)<=5e-13:
                break
            update=np.linalg.solve(jac(x)+np.diag(masses*w[-1]),-residual)
            current=float(np.max(abs(residual)/normalizer))
            length=1.
            for _ in range(20):
                candidate=x+length*update
                rnext=masses*(history+w[-1]*(candidate-u[n-1]))+F(candidate)-LOAD
                if np.max(abs(rnext)/normalizer)<current:
                    x=candidate
                    break
                length*=.5
            else:
                raise RuntimeError(('Newton line search failed',n,current))
        else:
            raise RuntimeError(('Newton failed',n))
        u[n]=x;increments[n-1]=u[n]-u[n-1]
        residuals[n-1]=masses*(w@increments[:n])+F(u[n])-LOAD
        normalized[n-1]=abs(residuals[n-1])/normalizer
        newton_counts[n-1]=iteration
        oldest[n-1]=w[0]
    assert np.max(normalized)<1e-11
    return dict(U=u,increments=increments,residuals=residuals,
                normalized_residuals=normalized,newton_updates=newton_counts,
                oldest_weights=oldest)

def make_probes(t,alpha,masses):
    cells=(t[:-1,None]+np.diff(t)[:,None]*np.array([.25,.5,.75])).ravel()
    layers=np.concatenate([float(d)**(1/alpha)*np.logspace(-6,6,121) for d in masses])
    return np.unique(np.concatenate((t,cells,layers[(layers>0)&(layers<1)])))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--delta',default=None)
    args=parser.parse_args()
    started=time.time()
    levels=(16,32) if args.pilot else LEVELS
    deltas=[Fraction(args.delta)] if args.delta else [Fraction(1),Fraction(1,100)]
    rows=[]
    for p in (3,7):
        for delta in deltas:
            masses=np.array([float(delta*r) for r in RATIOS])
            previous=None
            for m in levels:
                nums,den,info=exact_grid(p,delta,m)
                t=helpers.as_float(nums,den)
                sol=solve_l1(nums,den,p/10,masses)
                label=f'a{p}_delta_{delta.numerator}_{delta.denominator}_m{m}'
                probes=make_probes(t,p/10,masses)
                reconstruction=np.array([np.interp(probes,t,sol['U'][:,i]) for i in range(3)]).T
                if not args.pilot:
                    np.savez_compressed(HERE/(label+'.npz'),times=t,probes=probes,reconstruction=reconstruction,
                                        masses=masses,**sol)
                    helpers.write_json(HERE/(label+'_grid.json'),{'numerators':list(map(str,nums)),
                        'denominator':str(den),'grid_info':info})
                row=dict(case=label,alpha=p/10,alpha_numerator=p,delta=str(delta),
                         masses=masses.tolist(),m=m,N=len(t)-1,
                         max_normalized_residual=float(np.max(sol['normalized_residuals'])),
                         max_newton_updates=int(np.max(sol['newton_updates'])),
                         probe_count=len(probes),min_step=float(np.min(np.diff(t))))
                if previous:
                    pt,pu=previous
                    fine=np.array([np.interp(t,pt,pu[:,i]) for i in range(3)]).T
                    row['previous_PL_difference']=float(np.max(abs(fine-sol['U'])))
                previous=t,sol['U']
                rows.append(row)
                print(json.dumps(row),flush=True)
    if not args.pilot:
        helpers.write_json(HERE/'l1_results.json',{'parameters':{
            'T':1,'B':B.tolist(),'C':C.tolist(),'edge_weights':EDGE.tolist(),
            'edge_cubic':EDGE_CUBIC,'u0':U0.tolist(),'load':LOAD.tolist(),
            'initial_incompatibility':(LOAD-F(U0)).tolist(),
            'description':'F(v)=C v+B diag(edge_weights)[B^T v+0.1(B^T v)^3]',
            'mass_ratios':list(map(str,RATIOS))},'rows':rows,
            'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                           'platform':platform.platform()},'elapsed_seconds':time.time()-started})
    print('elapsed',time.time()-started,flush=True)

if __name__=='__main__':
    main()
