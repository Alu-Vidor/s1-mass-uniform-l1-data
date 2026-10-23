"""Published-component CPE circuit: L1 error and separate finite-RC model gap.

Run with requirements-numerics.txt installed.
No fitted or measured data for the new network are claimed.
All numerical trajectories, exact grid keys, and reference checks are retained.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from fractions import Fraction as F
from pathlib import Path
import hashlib, importlib.util, json, math, platform, time
import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.special import erfcx

HERE=Path(__file__).resolve().parent
DOSSIER=HERE.parents[1]
spec=importlib.util.spec_from_file_location('allmass',HERE.parent/'all_masses_vanishing/run_experiment.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
M=np.array([.25,.0625,.015625])
MX=[F(1,4),F(1,16),F(1,64)]
S=np.array([1.,.25,.0625])
A=np.array([[1.5,-.5,0],[-.5,2.,-.5],[0,-.5,1.5]])
P=np.array([1.,0,0]);U0=np.zeros(3)
T=0.050176; V0=.01;G0=.001

def reference(times):
    """Generalized spectral solution, symmetric mass transformation."""
    lam,q=eigh(A/np.sqrt(M[:,None]*M[None,:]))
    steady=np.linalg.solve(A,P)
    weights=q.T@(np.sqrt(M)*steady)
    tt=np.atleast_1d(times)
    # (1-erfcx) response avoids subtracting the whole steady vector.
    response=(1-erfcx(np.sqrt(tt[:,None])*lam)) * weights
    out=(response@q.T)/np.sqrt(M)
    out[tt==0]=0
    return out

def mp_reference(times,dps):
    with mp.workdps(dps):
        mm=[mp.mpf(1)/4,mp.mpf(1)/16,mp.mpf(1)/64]
        aa=mp.matrix([['1.5','-.5','0'],['-.5','2','-.5'],['0','-.5','1.5']])
        root=mp.diag([mp.sqrt(d) for d in mm]); ir=root**-1
        lam,q=mp.eigsy(ir*aa*ir); ss=aa**-1*mp.matrix([1,0,0]); w=q.T*root*ss
        out=[]
        for tt in times:
            t=mp.mpf(str(tt)); vv=[]
            for j in range(3):
                x=lam[j]*mp.sqrt(t)
                vv.append(w[j]*(1-base.mp_erfcx(x)))
            out.append([float(v) for v in ir*q*mp.matrix(vv)])
        return np.array(out)

def solve(nodes):
    nmax=len(nodes)-1;u=np.zeros((nmax+1,3));du=np.zeros((nmax,3));rall=np.zeros((nmax,3));rn=np.zeros_like(rall)
    for n in range(1,nmax+1):
        w=base.weights(nodes,n);hist=w[:-1]@du[:n-1]
        mat=A+np.diag(M*w[-1]);x=u[n-1].copy()
        for k in range(3):
            r=M*(w[-1]*(x-u[n-1])+hist)+A@x-P
            if np.max(abs(r)/(1+M*w[0]))<1e-14:break
            x+=np.linalg.solve(mat,-r)
        u[n]=x;du[n-1]=x-u[n-1]
        rall[n-1]=M*(w@du[:n])+A@x-P
        rn[n-1]=abs(rall[n-1])/(1+M*w[0])
    assert np.max(rn)<1e-12
    return u,rall,rn

def rc_matrices():
    rp=15000.;cp=.5e-9
    rb=np.array([9120,3650,1460,583,220,83,30,15,6,3.])
    cb=np.array([11e-6,4.4e-6,1.68e-6,680e-9,267e-9,100e-9,38e-9,11e-9,5.9e-9,3e-9])
    cap=np.zeros(33);k=np.zeros((33,33));k[:3,:3]=G0*A;cap[:3]=cp*S
    for i in range(3):
        k[i,i]+=S[i]/rp
        for j in range(10):
            h=3+10*i+j; g=S[i]/rb[j];cap[h]=S[i]*cb[j]
            k[i,i]+=g;k[h,h]+=g;k[i,h]-=g;k[h,i]-=g
    b=np.zeros(33);b[0]=1e-5
    return cap,k,b

def rc_reference(times,dps=60):
    """33-state constant-input response, high-precision symmetric eigensystem."""
    cap,k,b=rc_matrices()
    with mp.workdps(dps):
        cc=mp.diag([1/mp.sqrt(mp.mpf(str(v))) for v in cap])
        kk=mp.matrix([[mp.mpf(str(v)) for v in row] for row in k])
        bb=mp.matrix(list(map(lambda x:mp.mpf(str(x)),b)))
        lam,q=mp.eigsy(cc*kk*cc)
        w=q.T*cc*bb; left=cc*q
        le=np.array(left.tolist(),float);ev=np.array(list(lam),float);ww=np.array(list(w),float)
    tt=np.asarray(times)
    modal=(-np.expm1(-tt[:,None]*ev))*(ww/ev)
    return (modal@le.T)[:,:3],{'dps':dps,'min_eigenvalue_per_s':float(ev.min()),'max_eigenvalue_per_s':float(ev.max()),'matrix_dimension':33}

def main():
    rows=[];selected=[];start=time.monotonic()
    for m in (16,32,64,128,256,512):
        union=base.make_mesh([F(1,2),F(1,4),F(1,8)],m);n=len(union)-1
        for kind in ('union','uniform','graded'):
            nodes=union if kind=='union' else [F(j,n) if kind=='uniform' else F(j,n)**3 for j in range(n+1)]
            u,r,rn=solve(nodes);tt=np.array(list(map(float,nodes)));ref=reference(tt)
            probes=[nodes[j]+(nodes[j+1]-nodes[j])*F(k,4) for j in range(n) for k in (1,2,3)]
            pt=np.array(list(map(float,probes)));theta=np.tile([.25,.5,.75],n)[:,None]
            pl=np.repeat(u[:-1],3,axis=0)*(1-theta)+np.repeat(u[1:],3,axis=0)*theta
            pr=reference(pt)
            logt=np.geomspace(1e-12,1,801)
            logu=np.column_stack([np.interp(logt,tt,u[:,j]) for j in range(3)])
            logr=reference(logt)
            en=float(np.max(abs(u-ref)));es=max(en,float(np.max(abs(pl-pr))),float(np.max(abs(logu-logr))))
            np.savez_compressed(HERE/f'{kind}_m{m:04}.npz',times=tt,states=u,reference=ref,node_numerators=np.array([str(t.numerator) for t in nodes]),node_denominators=np.array([str(t.denominator) for t in nodes]),probe_times=pt,probe_states=pl,probe_reference=pr,log_probe_times=logt,log_probe_states=logu,log_probe_reference=logr,full_history_residual=r,normalized_residual=rn)
            row={'mesh':kind,'m':m,'N':n,'nodal_error':en,'sampled_PL_error':es,'error_mV':10*es,'max_normalized_residual':float(np.max(rn)),'min_step':float(min(b-a for a,b in zip(nodes,nodes[1:])))}
            rows.append(row)
            selected.extend([tt[1],tt[np.argmax(np.max(abs(u-ref),axis=1))],pt[np.argmax(np.max(abs(pl-pr),axis=1))],logt[np.argmax(np.max(abs(logu-logr),axis=1))]])
            print(json.dumps(row),flush=True)
    selected=sorted(set([0.,1.]+selected));h80=mp_reference(selected,80);h110=mp_reference(selected,110)
    check={'checked_times':selected,'max_float_vs_110dp':float(np.max(abs(reference(selected)-h110))),'max_80dp_vs_110dp':float(np.max(abs(h80-h110))),'reference_110dp_rounded_to_float':h110.tolist()}
    assert check['max_float_vs_110dp']<1e-11
    times_s=np.r_[0,np.geomspace(1e-10,T,1601)]
    ideal=V0*reference(times_s/T);rc60,info=rc_reference(times_s,60);rc90,info90=rc_reference(times_s,90)
    gap=abs(ideal-rc90);imax=np.unravel_index(np.argmax(gap),gap.shape)
    np.savez_compressed(HERE/'physical_model_comparison.npz',times_s=times_s,ideal_CPE_volts=ideal,finite_RC_volts=rc90,absolute_gap_volts=gap)
    model={'maximum_sampled_gap_V':float(gap[imax]),'argmax_time_s':float(times_s[imax[0]]),'argmax_node_1based':int(imax[1]+1),'max_gap_relative_to_input_scale':float(gap.max()/V0),'terminal_gap_V':float(gap[-1].max()),'RC_60_vs_90dp_V':float(np.max(abs(rc60-rc90))),'RC_reference':info90,'sampling':'zero plus1601logtimesfrom1e-10s toT;notglobalcontinuousbound','ideal_steady_volts':(V0*np.linalg.solve(A,P)).tolist(),'finite_RC_steady_volts':np.linalg.solve(rc_matrices()[1],rc_matrices()[2])[:3].tolist()}
    for kind in ('union','uniform','graded'):
        rr=[r for r in rows if r['mesh']==kind]
        for before,after in zip(rr,rr[1:]):after['nodal_rate']=math.log(before['nodal_error']/after['nodal_error'])/math.log(after['N']/before['N'])
    result={'protocol':'PublishedSotner2020baseRCemulator;new3nodepassivenetwork;zero-prehistorycurrentstep;nohardwaremeasurement','alpha':.5,'d_exact':list(map(str,MX)),'T_s':T,'V0_V':V0,'Q_scales':S.tolist(),'p':P.tolist(),'A':A.tolist(),'runs':rows,'reference_check':check,'physical_model_comparison':model,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__},'elapsed_seconds':time.monotonic()-start,'full_history':True,'sampled_PL_not_continuous_certificate':True}
    (HERE/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'reference':check['max_float_vs_110dp'],'model':model,'elapsed_s':result['elapsed_seconds']},indent=2))
    files=[p for p in HERE.iterdir() if p.is_file() and p.name!='MANIFEST.json']
    (HERE/'MANIFEST.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))

if __name__=='__main__':main()
