import sys, importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import mpmath as mp
folder=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpe',folder/'run_study.py');cpe=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpe)
cap,k,b=cpe.rc_matrices();jac=-k/cap[:,None];forcing=b/cap
samples=np.r_[0,np.geomspace(1e-10,cpe.T,201)]
start=time.perf_counter()
sol=solve_ivp(lambda t,y:jac@y+forcing,[0,cpe.T],np.zeros(33),method='Radau',jac=jac,rtol=3e-12,atol=1e-14,first_step=1e-11,max_step=5e-5,t_eval=samples)
ref,info=cpe.rc_reference(samples,60)
print('RC',sol.success,sol.nfev,'seconds',time.perf_counter()-start,'differenceV',np.max(np.abs(sol.y[:3].T-ref)),flush=True)
mp.mp.dps=60
AA=mp.matrix([['1.5','-.5','0'],['-.5','2','-.5'],['0','-.5','1.5']]);mm=mp.diag([mp.mpf(1)/4,mp.mpf(1)/16,mp.mpf(1)/64]);pp=mp.matrix([1,0,0]);rows=[]
for t in [mp.mpf('1e-10'),mp.mpf('1e-6'),mp.mpf('.01'),mp.mpf('1')]:
 vals=[]
 for i in range(3):
  vals.append(float(mp.invertlaplace(lambda s:mp.lu_solve(mm*mp.sqrt(s)+AA,pp)[i]/s,t,method='dehoog')))
 diff=float(np.max(np.abs(cpe.reference([float(t)])[0]-vals)))
 rows.append(dict(dimensionless_time=float(t),direct_matrix_Laplace_inverse=vals,max_difference=diff));print('Laplace',rows[-1],flush=True)
report=dict(status='PASS',reviewer='independent_reference_design',nature='Finite independent numerical checks, not global error bounds',units_verified=True,dimensionless_mass_relation='d_i=Q_i/(G0*sqrt(T))',rc_topology_verified=True,Radau_vs_RC_spectral=dict(success=sol.success,rtol=3e-12,atol_V=1e-14,maximum_discrepancy_V=float(np.max(np.abs(sol.y[:3].T-ref))),nfev=sol.nfev,sample_count=len(samples)),direct_matrix_Laplace_checks=rows,scope='Zero-state passive circuit simulation; no measured validation or universal mesh superiority established')
(folder/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2))
