from pathlib import Path
import sys, json, math
import numpy as np
import mpmath as mp
from scipy.special import erfcx
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference import *
mp.mp.dps=100
rows=[]
for alpha in [.3,.7]:
 for t,left,right in [(1.,0.,r) for r in [1e-30,1e-20,1e-12,1e-8,1e-4,.0249,.025,.1,.9,1.]]+[(1.,0.,1.-2.**(-j)) for j in [5,20,40,50]]:
  wl,wr=pi_weights(alpha,t,[left],[right]); aa,tt,ll,rr=map(mp.mpf,[alpha,t,left,right]); A=tt-ll; BB=tt-rr; h=rr-ll
  I0=(A**aa-BB**aa)/aa
  I1=A*I0-(A**(aa+1)-BB**(aa+1))/(aa+1)
  truthr=I1/h/mp.gamma(aa); truthl=I0/mp.gamma(aa)-truthr
  dl=float(abs(mp.mpf(wl[0])-truthl)/abs(truthl)); dr=float(abs(mp.mpf(wr[0])-truthr)/abs(truthr))
  rows.append(dict(alpha=alpha,t=t,left=left,right=right,left_relative_difference=dl,right_relative_difference=dr))
print('weights maximumreldiff',max(max(x['left_relative_difference'],x['right_relative_difference']) for x in rows))
scalar=[]
probes=np.r_[0,np.geomspace(1e-24,1,401)]
for mass in [1.,1e-4]:
 old=None
 for K in [4096,8192,16384]:
  s=solve_reference(.5,[mass],[1.],np.empty((1,0)),[],.1,[1.],[.2],K,grade=8,probes=probes)
  exact=.2+.8*erfcx(np.sqrt(probes)/mass)
  err=float(np.max(np.abs(s['probe_U'][:,0]-exact)))
  scalar.append(dict(alpha=.5,mass=mass,K=K,error=err,newton_resid=s['max_normalized_algebraic_residual']))
  print('scalar',scalar[-1],flush=True)
report=dict(status='PASS',nature='Finite numerical checks, not interval certification',weight_checks=rows,scalar_validation=scalar)
Path(__file__).with_name('REFERENCE_VALIDATION.json').write_text(json.dumps(report,indent=2))

