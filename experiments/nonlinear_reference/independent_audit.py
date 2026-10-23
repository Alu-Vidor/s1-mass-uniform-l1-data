from pathlib import Path
import json, hashlib
import numpy as np
folder=Path(__file__).resolve().parent
r=json.loads((folder/'results.json').read_text());audit=json.loads((folder/'AUDIT.json').read_text());checks=[]
B=np.array(r['parameters']['B']);u0=np.array(r['parameters']['u0']);c=np.array(r['parameters']['C']);w=np.array(r['parameters']['edge_weights']);load=np.array(r['parameters']['load']);z=B.T@u0
initial_F=c*u0+B@(w*(z+.1*z**3));defect=load-initial_F
assert np.all(abs(defect)>1);assert np.allclose(defect,r['parameters']['initial_incompatibility'],rtol=0,atol=2e-15)
for p in [3,7]:
 for delta in ['1','1/100']:
  rows=[x for x in r['rows'] if x['alpha_numerator']==p and x['delta']==delta]
  suffix=delta.replace('/','_');refs=[np.load(folder/f'reference_a{p}_delta_{suffix}_K{k}.npz') for k in [8192,16384,32768]]
  probes=refs[-1]['probes'];assert all(np.array_equal(a['probes'],probes) for a in refs)
  increments=[float(np.max(abs(a['probe_U']-b['probe_U']))) for a,b in zip(refs[:-1],refs[1:])]
  for idx,row in enumerate(rows):
   a=np.load(folder/(row['case']+'.npz'));positions=np.searchsorted(probes,a['times']);err=float(np.max(abs(a['U']-refs[-1]['probe_U'][positions])))
   assert err==row['nodal_error'];assert len(a['times'])-1==row['N'];assert np.array_equal(a['U'][0],u0)
   assert np.array_equal(a['masses'],np.array(row['masses']))
   if idx:
    rate=np.log(rows[idx-1]['nodal_error']/err)/np.log(row['N']/rows[idx-1]['N']);assert abs(rate-row['rate_actual_N'])<1e-14
   if idx==4:
    pp=np.searchsorted(probes,a['probes']);es=float(np.max(abs(a['reconstruction']-refs[-1]['probe_U'][pp])));assert es==row['sampled_PL_error']
  assert all(b['nodal_error']<a['nodal_error'] for a,b in zip(rows[:-1],rows[1:]))
  checks.append(dict(alpha=p/10,delta=delta,all_saved_nodal_errors_recomputed=True,all_rates_recomputed=True,sampled_PL_error_recomputed=True,all_five_levels_strictly_decrease=True,reference_successive_discrepancies=increments,final_reference_increment_percent_of_finest_L1_error=100*increments[-1]/rows[-1]['nodal_error']))
validation=json.loads((folder/'REFERENCE_VALIDATION.json').read_text());cross=json.loads((folder/'CROSS_GRADE_CHECK.json').read_text())
report=dict(status='PASS',reviewer='Separate reference-design audit',results_sha256=hashlib.sha256((folder/'results.json').read_bytes()).hexdigest(),initial_load_minus_F_u0=defect.tolist(),initial_edge_potentials=z.tolist(),initial_nonlinear_edge_currents=(w*(z+.1*z**3)).tolist(),initial_edge_incremental_conductances=(w*(1+.3*z*z)).tolist(),theorem_class='Fixed linear positive node losses and globally monotone C-infinity cubic edge laws; constant forcing; same unprepared u0 for all masses',family_checks=checks,largest_stored_normalized_L1_residual=max(q['max_normalized_residual'] for q in r['rows']),selected_85_digit_history_recomputation_maximum=audit['max_high_precision_normalized_residual'],selected_85_digit_discrepancy_from_stored=audit['max_discrepancy_from_stored_residual'],maximum_100_digit_weight_relative_difference=max(max(q['left_relative_difference'],q['right_relative_difference']) for q in validation['weight_checks']),maximum_cross_grade_difference=max(q['max_difference'] for q in cross['checks']),limitations=['Reference discretises a different integral-equation formulation, rather than a fine L1 grid.','Finite refinements and sampled errors are numerical evidence, not interval-certified global error bounds.','Non-half orders and polynomial nonlinear coupling are now combined; no real-device calibration is claimed.','Alpha=0.3 rates at the available final levels do not all attain the asymptotic exponent 1.7.'])
(folder/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

