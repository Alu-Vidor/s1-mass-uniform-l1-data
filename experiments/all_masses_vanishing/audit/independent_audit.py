#!/usr/bin/env python3
"""Read stored S1 trajectories; never integrate the differential system.

The only output is this audit directory's audit_result.json. Production
scripts, summaries and trajectories are read without modification.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_and_check():
    summary_file = EXP / "results.json"
    summary_sha = sha(summary_file)
    report = json.loads(summary_file.read_text(encoding="utf-8"))
    input_files = [summary_file, EXP / "results.csv",
                   EXP / "run_experiment.py", EXP.parents[1] / "legacy/mass3/mass3_l1_check.py"]
    recorded = json.loads((EXP.parents[1] / "provenance/original_sha256.json").read_text(encoding="utf-8"))
    assert report["experiment_script_sha256"] in (sha(input_files[2]), recorded["experiments/all_masses_vanishing/run_experiment.py"])
    assert sha(input_files[3]) == report["original_solver_sha256"]
    a = np.array([[6, -5, 0], [-5, 9.3, -3], [0, -3, 3.9]])
    c = np.array([1, 1.3, .9])
    p0 = np.array([.4, -.2, .3])
    p1 = np.array([.2, .1, -.15])
    groups = {}
    final_rows = []
    full_checks = []
    checked_files = []
    max_norm_diff = 0.
    max_normalization_diff = 0.
    max_order_diff = 0.
    for row in report["runs"]:
        source = EXP / row["source_npz"]
        source_sha = sha(source)
        assert source_sha == row["source_npz_sha256"]
        checked_files.append({"name": source.name, "sha256": source_sha})
        with np.load(source, allow_pickle=False) as data:
            nodes = [F(int(x), int(y)) for x, y in
                     zip(data["node_numerators"], data["node_denominators"])]
            m = row["m"]
            nmax = len(nodes) - 1
            assert nmax == row["actual_N"] == int(data["actual_N"])
            masses_exact = list(map(F, row["masses_exact"]))
            roots = [F(math.isqrt(d.numerator), math.isqrt(d.denominator))
                     for d in masses_exact]
            assert all(s * s == d for s, d in zip(roots, masses_exact))
            parents = [[(s * j / (m * (1 + s) - j)) ** 4
                        for j in range(m + 1)] for s in (*roots, F(1))]
            parents.append([F(j, m) for j in range(m + 1)])
            independent_mesh = sorted(set().union(*map(set, parents)))
            assert nodes == independent_mesh
            assert row["exact_duplicate_removals"] == 5 * (m - 1) + 1 - nmax
            assert row["minimum_step"] == float(min(y - x for x, y in zip(nodes, nodes[1:])))
            assert np.array_equal(data["times"], np.array(list(map(float, nodes))))
            assert list(map(str, masses_exact)) == list(data["masses_exact"])
            assert np.array_equal(data["masses"], np.array(list(map(float, masses_exact))))
            u = data["states"]
            reference = data["reference"]
            errors = np.abs(u - reference)
            assert np.array_equal(errors, data["absolute_errors"])
            assert row["nodal_error"] == float(np.max(errors))
            assert np.array_equal(np.max(errors, axis=0), row["nodal_error_by_component"])
            probes = [(nodes[k-1] * (4-j) + nodes[k] * j) / 4
                      for k in range(1, len(nodes)) for j in (1, 2, 3)]
            assert np.array_equal(data["probe_times"], np.array(list(map(float, probes))))
            theta = np.tile(np.array([.25, .5, .75]), nmax)[:, None]
            pl = (1-theta) * np.repeat(u[:-1], 3, axis=0) + theta * np.repeat(u[1:], 3, axis=0)
            assert np.array_equal(pl, data["probe_PL_states"])
            assert row["sampled_PL_error"] == max(float(np.max(errors)),
                float(np.max(np.abs(pl - data["probe_reference"]))))
            raw = data["full_history_residual"]
            normalized = data["normalized_full_history_residual"]
            assert row["max_absolute_full_history_residual"] == float(np.max(np.abs(raw)))
            masses = data["masses"]
            rt = np.sqrt(np.array(list(map(float, nodes[1:]))))
            rprev = np.sqrt(np.array([float(t-nodes[1]) for t in nodes[1:]]))
            bfirst = 2/math.sqrt(math.pi)/(rt+rprev)
            independent_normalized = np.abs(raw)/(c[None, :]+bfirst[:, None]*masses[None, :])
            max_normalization_diff = max(max_normalization_diff,
                float(np.max(np.abs(normalized-independent_normalized))))
            assert np.allclose(normalized, independent_normalized, rtol=8e-16, atol=1e-30)
            assert row["max_normalized_full_history_residual"] == float(np.max(normalized))
            assert float(np.max(normalized)) < 1e-12
            group = groups.setdefault((row["path"], row["delta_exact"]), [])
            if group:
                old = group[-1]
                q = math.log(old["nodal_error"]/row["nodal_error"])/math.log(nmax/old["actual_N"])
                max_order_diff = max(max_order_diff, abs(q-row["observed_nodal_order_actual_N"]))
                assert math.isclose(q, row["observed_nodal_order_actual_N"], rel_tol=2e-15)
                assert row["nodal_error"] < old["nodal_error"]
            group.append(row)
            if m == 512:
                indices = sorted(set([1, 2, nmax//4, nmax//2, nmax-1, nmax,
                    1+int(np.argmax(np.max(normalized, axis=1))),
                    int(np.argmax(np.max(errors, axis=1)))]))
                series_checks = []
                for k in indices:
                    distances = np.array([float(nodes[k]-nodes[j]) for j in range(k+1)])
                    roots_time = np.sqrt(distances)
                    weights = 2/math.sqrt(math.pi)/(roots_time[:-1]+roots_time[1:])
                    residual = masses*np.sum(weights[:, None]*np.diff(u[:k+1], axis=0), axis=0)
                    residual += a@u[k]-(p0+float(nodes[k])*p1)
                    norm = np.abs(residual)/(c+masses*weights[0])
                    diff = float(np.max(np.abs(norm-normalized[k-1])))
                    max_norm_diff = max(max_norm_diff, diff)
                    assert float(np.max(norm)) < 1e-12
                    check = {"path": row["path"], "delta_exact": row["delta_exact"],
                        "m": m, "time_index": k, "time_exact": str(nodes[k]),
                        "max_recomputed_normalized_residual": float(np.max(norm)),
                        "max_saved_normalized_residual": float(np.max(normalized[k-1])),
                        "max_componentwise_normalized_difference": diff}
                    series_checks.append(check)
                    full_checks.append(check)
                final_rows.append({"path": row["path"], "delta_exact": row["delta_exact"],
                    "actual_N": nmax, "nodal_error": row["nodal_error"],
                    "observed_nodal_order_actual_N": row["observed_nodal_order_actual_N"],
                    "max_saved_normalized_residual": float(np.max(normalized)),
                    "max_selected_recomputed_normalized_residual": max(
                        x["max_recomputed_normalized_residual"] for x in series_checks),
                    "residual_to_error_ratio": float(np.max(normalized))/row["nodal_error"]})
    assert len(report["runs"]) == len(checked_files) == 48
    assert len(groups) == 8 and len(final_rows) == 8 and len(full_checks) == 64
    return report, {"results_json_sha256": summary_sha,
        "input_files": [{"path": str(f), "sha256": sha(f)} for f in input_files],
        "checked_trajectory_files": checked_files,
        "all_48_trajectory_hashes_pass": True,
        "all_48_exact_meshes_and_actual_sizes_pass": True,
        "all_48_error_arrays_and_quarter_point_PL_values_pass": True,
        "all_48_oldest_weight_residual_normalizations_and_orders_pass": True,
        "max_residual_normalization_difference": max_normalization_diff,
        "max_order_recalculation_difference": max_order_diff,
        "full_history_selected_time_nodes": full_checks,
        "max_selected_full_history_normalized_residual": max(
            x["max_recomputed_normalized_residual"] for x in full_checks),
        "max_normalized_difference_from_saved_residual": max_norm_diff,
        "max_saved_normalized_residual_all_runs": max(
            x["max_normalized_full_history_residual"] for x in report["runs"]),
        "max_saved_absolute_residual_all_runs": max(
            x["max_absolute_full_history_residual"] for x in report["runs"]),
        "minimum_nodal_error_all_runs": min(x["nodal_error"] for x in report["runs"]),
        "final_m512_rows": final_rows}


def alternate_reference_checks(report):
    checks = []
    with mp.workdps(120):
        a = mp.matrix([["6", "-5", "0"], ["-5", "9.3", "-3"], ["0", "-3", "3.9"]])
        u0 = mp.matrix(["1", "-.4", ".2"])
        p0 = mp.matrix([".4", "-.2", ".3"])
        p1 = mp.matrix([".2", ".1", "-.15"])
        for row in report["runs"]:
            if row["m"] != 512:
                continue
            with np.load(EXP/row["source_npz"], allow_pickle=False) as data:
                nodes = [F(int(x), int(y)) for x, y in
                         zip(data["node_numerators"], data["node_denominators"])]
                masses = [mp.mpf(d.numerator)/d.denominator for d in map(F, row["masses_exact"])]
                mh = mp.diag([mp.sqrt(d) for d in masses])
                mih = mp.diag([1/mp.sqrt(d) for d in masses])
                eigenvalues, q = mp.eigsy(mih*a*mih)
                initial = q.T*mh*u0
                f0 = q.T*mih*p0
                f1 = q.T*mih*p1
                indices = sorted(set([1, len(nodes)//2, len(nodes)-1,
                    int(np.argmax(np.max(data["absolute_errors"], axis=1)))]))
                for n in indices:
                    tt = mp.mpf(nodes[n].numerator)/nodes[n].denominator
                    modal = []
                    for j in range(3):
                        x = eigenvalues[j]*mp.sqrt(tt)
                        if x < 10**6:
                            e = mp.exp(x*x)*mp.erfc(x)
                        else:
                            term = mp.mpf(1)
                            series = term
                            for k in range(1, 30):
                                term *= -(2*k-1)/(2*x*x)
                                series += term
                            e = series/(mp.sqrt(mp.pi)*x)
                        g = 1-2/(mp.sqrt(mp.pi)*x)+(1-e)/(x*x)
                        modal.append(f0[j]/eigenvalues[j] +
                            (initial[j]-f0[j]/eigenvalues[j])*e +
                            f1[j]/eigenvalues[j]*tt*g)
                    alternate = mih*q*mp.matrix(modal)
                    discrepancy = float(np.max(np.abs(
                        np.array([float(v) for v in alternate])-data["reference"][n])))
                    assert discrepancy < 2e-14
                    checks.append({"path": row["path"], "delta_exact": row["delta_exact"],
                        "m": row["m"], "time_index": n, "time_exact": str(nodes[n]),
                        "alternative_reference_120dp": [mp.nstr(v, 90) for v in alternate],
                        "saved_float_reference": data["reference"][n].tolist(),
                        "max_float_discrepancy": discrepancy})
    assert len(checks) == 32
    return {"decimal_digits": 120,
        "method": "Independent symmetric matrix M^(-1/2) A M^(-1/2), transformed scalar affine forcing, then M^(-1/2) Q reconstruction; production script is not imported.",
        "special_function_evaluation": "mp.exp(x*x)*mp.erfc(x) below x=1e6; 30-term erfcx asymptotic expansion above x=1e6.",
        "selection": "First node, middle-index node, terminal node and maximum nodal-error node in each of the eight m=512 trajectories.",
        "checked_nodes": checks,
        "max_float_discrepancy": max(x["max_float_discrepancy"] for x in checks)}


def main():
    report, result = read_and_check()
    result["alternate_reference_check"] = alternate_reference_checks(report)
    # Recheck the production snapshot after all reads before writing our report.
    assert result["results_json_sha256"] == sha(EXP/"results.json")
    result["audit_script_sha256"] = sha(Path(__file__).resolve())
    result["status"] = "PASS: no substantive discrepancy found"
    result["limitations"] = [
        "This script reads saved trajectories and performs no differential-equation solver run.",
        "Full-history residuals are independently recomputed at 64 selected time nodes; saved residual normalizations are checked at every node of all 48 trajectories.",
        "Alternative 120-digit reference checks cover 32 selected nodes, not every point or an interval enclosure.",
        "Floating-point differences, residuals and sampled PL errors are numerical diagnostics, not certified bounds.",
        "Selected finite masses do not prove mass-uniformity or an asymptotic convergence rate."]
    target = HERE/"audit_result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
        "results_json_sha256": result["results_json_sha256"],
        "trajectory_count": len(result["checked_trajectory_files"]),
        "selected_full_history_nodes": len(result["full_history_selected_time_nodes"]),
        "max_normalized_residual_difference": result["max_normalized_difference_from_saved_residual"],
        "alternate_120dp_reference_nodes": len(result["alternate_reference_check"]["checked_nodes"]),
        "max_alternate_reference_discrepancy": result["alternate_reference_check"]["max_float_discrepancy"]}, indent=2))


if __name__ == "__main__":
    main()
