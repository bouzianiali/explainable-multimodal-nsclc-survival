#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final paired comparison:
missingness-aware standard concatenation + ModDrop
vs proposed gated residual model.

No training, tuning, recalibration, CT inference, or GPU is used.
The script consumes the already exported train/test risk-mode CSV files,
applies identical patient-level missingness masks to both models, and performs:
  1) fixed-cohort missingness simulation (0/10/30/50/100%)
  2) paired patient bootstrap
  3) Benjamini-Hochberg correction across 15 scenario x metric tests
  4) full-vs-no-clinical dependence diagnostics

Expected aware CSV columns:
  PatientID, Survival.time, deadstatus.event,
  risk_full_clinical, risk_no_clinical

Expected proposed CSV columns:
  PatientID, Survival.time, deadstatus.event,
  risk_full OR risk_full_clinical,
  risk_no_clinical

Authoring context: LUNG1 TNM-corrected locked split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score


ID = "PatientID"
TIME = "Survival.time"
EVENT = "deadstatus.event"

AWARE = "Missingness-aware concatenation + ModDrop"
PROP = "Proposed gated residual"
MODELS = [AWARE, PROP]
FRACTIONS = [0.0, 0.10, 0.30, 0.50, 1.0]


def load_modes(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    base = [ID, TIME, EVENT]
    miss = [c for c in base if c not in df.columns]
    if miss:
        raise RuntimeError(f"{label}: missing required columns {miss}")

    full_candidates = ["risk_full_clinical", "risk_full", "risk_score"]
    no_candidates = ["risk_no_clinical", "risk_no_clin"]

    full_col = next((c for c in full_candidates if c in df.columns), None)
    no_col = next((c for c in no_candidates if c in df.columns), None)

    if full_col is None or no_col is None:
        raise RuntimeError(
            f"{label}: could not identify full/no-clinical risk columns. "
            f"Columns={list(df.columns)}"
        )

    out = df[base + [full_col, no_col]].copy()
    out = out.rename(columns={
        full_col: "risk_full",
        no_col: "risk_no_clinical",
    })
    out[ID] = out[ID].astype(str)
    out[TIME] = out[TIME].astype(float)
    out[EVENT] = out[EVENT].astype(int)
    out["risk_full"] = out["risk_full"].astype(float)
    out["risk_no_clinical"] = out["risk_no_clinical"].astype(float)

    if out[ID].duplicated().any():
        raise RuntimeError(f"{label}: duplicate PatientID detected")

    return out.sort_values(ID).reset_index(drop=True)


def check_same(a: pd.DataFrame, b: pd.DataFrame, la: str, lb: str):
    if len(a) != len(b):
        raise RuntimeError(f"{la}/{lb}: N mismatch {len(a)} vs {len(b)}")
    if not np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()):
        raise RuntimeError(f"{la}/{lb}: PatientID ordering/set mismatch")
    if not np.allclose(a[TIME], b[TIME], rtol=0, atol=1e-8):
        raise RuntimeError(f"{la}/{lb}: survival-time mismatch")
    if not np.array_equal(a[EVENT].to_numpy(), b[EVENT].to_numpy()):
        raise RuntimeError(f"{la}/{lb}: event mismatch")


def to_surv_array(time, event):
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.asarray(event, int).astype(bool)
    y["time"] = np.asarray(time, float)
    return y


def harrell_cindex(time, event, risk):
    return float(
        concordance_index(
            np.asarray(time, float),
            -np.asarray(risk, float),
            np.asarray(event, int),
        )
    )


def fit_breslow(time, event, risk):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)

    shift = float(np.max(risk))
    rel = np.exp(risk - shift)
    event_times = np.sort(np.unique(time[event == 1]))

    increments = []
    for et in event_times:
        d = int(np.sum((time == et) & (event == 1)))
        denom = float(np.sum(rel[time >= et]))
        increments.append(d / denom if denom > 0 else 0.0)

    return {
        "event_times": event_times,
        "cum_hazard": np.cumsum(np.asarray(increments, float)),
        "shift": shift,
    }


def survival_matrix(risk, times, baseline):
    risk = np.asarray(risk, float)
    times = np.asarray(times, float)
    et = baseline["event_times"]
    ch = baseline["cum_hazard"]
    shift = baseline["shift"]

    H0 = np.zeros(len(times), float)
    for j, t in enumerate(times):
        k = np.searchsorted(et, t, side="right") - 1
        if k >= 0:
            H0[j] = ch[k]

    rr = np.exp(risk - shift)
    return np.exp(-rr[:, None] * H0[None, :])


def evaluate_metrics(train_df, test_df, train_risk, test_risk, times):
    y_train = to_surv_array(train_df[TIME], train_df[EVENT])
    y_test = to_surv_array(test_df[TIME], test_df[EVENT])

    c = harrell_cindex(test_df[TIME], test_df[EVENT], test_risk)

    aucs, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        np.asarray(test_risk, float),
        np.asarray(times, float),
    )

    baseline = fit_breslow(
        train_df[TIME].to_numpy(float),
        train_df[EVENT].to_numpy(int),
        train_risk,
    )
    surv = survival_matrix(test_risk, times, baseline)

    _, brier = brier_score(
        y_train, y_test, surv, np.asarray(times, float)
    )
    ibs = float(
        integrated_brier_score(
            y_train, y_test, surv, np.asarray(times, float)
        )
    )

    return {
        "C_index": float(c),
        "Mean_dynamic_AUC": float(mean_auc),
        "Integrated_Brier_Score": float(ibs),
    }


def exact_missing_mask(n, frac, seed):
    n = int(n)
    if frac <= 0:
        return np.zeros(n, dtype=bool)
    if frac >= 1:
        return np.ones(n, dtype=bool)
    k = int(round(float(frac) * n))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=k, replace=False)
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def scenario_masks(n_train, n_test, frac, seed, rep):
    pct = int(round(100 * frac))
    if pct == 0:
        return np.zeros(n_train, bool), np.zeros(n_test, bool)
    if pct == 100:
        return np.ones(n_train, bool), np.ones(n_test, bool)

    # Deterministic, reproducible and identical across models.
    tr_seed = seed + 100000 + 1000 * pct + rep
    te_seed = seed + 200000 + 1000 * pct + rep
    return (
        exact_missing_mask(n_train, frac, tr_seed),
        exact_missing_mask(n_test, frac, te_seed),
    )


def compose(df, mask):
    return np.where(
        np.asarray(mask, bool),
        df["risk_no_clinical"].to_numpy(float),
        df["risk_full"].to_numpy(float),
    )


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, float)
    out[order] = q
    return out


def sign_p(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    p_lo = np.mean(vals <= 0)
    p_hi = np.mean(vals >= 0)
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def summarize_model_dependence(df, label):
    d = df["risk_full"].to_numpy(float) - df["risk_no_clinical"].to_numpy(float)
    sp = pd.Series(df["risk_full"]).corr(
        pd.Series(df["risk_no_clinical"]), method="spearman"
    )
    pr = pd.Series(df["risk_full"]).corr(
        pd.Series(df["risk_no_clinical"]), method="pearson"
    )
    return {
        "Model": label,
        "N": len(df),
        "Mean_signed_full_minus_no": float(np.mean(d)),
        "Mean_abs_full_minus_no": float(np.mean(np.abs(d))),
        "Max_abs_full_minus_no": float(np.max(np.abs(d))),
        "Pearson_full_vs_no": float(pr),
        "Spearman_full_vs_no": float(sp),
        "Pct_full_minus_no_positive": float(100 * np.mean(d > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-train", required=True)
    ap.add_argument("--aware-test", required=True)
    ap.add_argument("--proposed-train", required=True)
    ap.add_argument("--proposed-test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario-repeats", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--times", nargs="+", type=int, default=[365, 730, 1095])
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    aware_tr = load_modes(Path(args.aware_train), "aware train")
    aware_te = load_modes(Path(args.aware_test), "aware test")
    prop_tr = load_modes(Path(args.proposed_train), "proposed train")
    prop_te = load_modes(Path(args.proposed_test), "proposed test")

    check_same(aware_tr, prop_tr, "aware train", "proposed train")
    check_same(aware_te, prop_te, "aware test", "proposed test")

    print("QC patient/outcome alignment: PASS")
    print(f"Train N={len(aware_tr)} | events={int(aware_tr[EVENT].sum())}")
    print(f"Test  N={len(aware_te)} | events={int(aware_te[EVENT].sum())}")
    print("Held-out test: evaluation only; no training/tuning performed.")

    # -------------------------------------------------------------
    # Modality-dependence diagnostic
    # -------------------------------------------------------------
    dep = pd.DataFrame([
        summarize_model_dependence(aware_te, AWARE),
        summarize_model_dependence(prop_te, PROP),
    ])
    dep.to_csv(
        outdir / "MissingnessAware_vs_Proposed_ModalityDependence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Fixed cohort simulation
    # -------------------------------------------------------------
    model_data = {
        AWARE: (aware_tr, aware_te),
        PROP: (prop_tr, prop_te),
    }

    sim_rows = []
    for frac in FRACTIONS:
        pct = int(round(100 * frac))
        nrep = 1 if pct in (0, 100) else int(args.scenario_repeats)
        print(f"Simulation {pct}% | repeats={nrep}")

        for rep in range(nrep):
            tr_mask, te_mask = scenario_masks(
                len(aware_tr), len(aware_te), frac, args.seed, rep
            )

            for model in MODELS:
                tr_df, te_df = model_data[model]
                m = evaluate_metrics(
                    tr_df, te_df,
                    compose(tr_df, tr_mask),
                    compose(te_df, te_mask),
                    args.times,
                )
                sim_rows.append({
                    "Model": model,
                    "Missing_percent": pct,
                    "Repeat": rep,
                    **m,
                })

    sim = pd.DataFrame(sim_rows)
    sim.to_csv(
        outdir / "MissingnessAware_vs_Proposed_Simulation_Raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    srows = []
    for (model, pct), g in sim.groupby(["Model", "Missing_percent"]):
        row = {
            "Model": model,
            "Missing_percent": int(pct),
            "Repeats": len(g),
            "Interval_type": (
                "deterministic endpoint"
                if int(pct) in (0, 100)
                else "missingness-simulation percentile range"
            ),
        }
        for metric in ["C_index", "Mean_dynamic_AUC", "Integrated_Brier_Score"]:
            v = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = np.mean(v)
            row[f"{metric}_sd"] = np.std(v, ddof=1) if len(v) > 1 else 0.0
            row[f"{metric}_p2_5"] = np.percentile(v, 2.5)
            row[f"{metric}_p97_5"] = np.percentile(v, 97.5)
        srows.append(row)

    sim_summary = pd.DataFrame(srows).sort_values(
        ["Missing_percent", "Model"]
    ).reset_index(drop=True)
    sim_summary.to_csv(
        outdir / "MissingnessAware_vs_Proposed_Simulation_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Paired bootstrap
    # -------------------------------------------------------------
    rng = np.random.default_rng(args.seed + 900000)
    boot_diff_rows = []
    boot_model_rows = []

    n_test = len(aware_te)

    for frac in FRACTIONS:
        pct = int(round(100 * frac))
        print(f"Paired bootstrap {pct}% | requested={args.bootstrap}")

        store = {
            metric: []
            for metric in ["C_index", "Mean_dynamic_AUC", "Integrated_Brier_Score"]
        }
        model_store = {
            model: {metric: [] for metric in store}
            for model in MODELS
        }

        valid = 0
        attempts = 0
        max_attempts = int(args.bootstrap * 1.5) + 100

        while valid < args.bootstrap and attempts < max_attempts:
            attempts += 1

            # same missingness realization for both models
            if frac == 0:
                tr_mask = np.zeros(len(aware_tr), bool)
                te_mask = np.zeros(len(aware_te), bool)
            elif frac == 1:
                tr_mask = np.ones(len(aware_tr), bool)
                te_mask = np.ones(len(aware_te), bool)
            else:
                tr_mask = exact_missing_mask(
                    len(aware_tr), frac, int(rng.integers(0, 2**31 - 1))
                )
                te_mask = exact_missing_mask(
                    len(aware_te), frac, int(rng.integers(0, 2**31 - 1))
                )

            idx = rng.integers(0, n_test, size=n_test)

            aware_boot = aware_te.iloc[idx].reset_index(drop=True)
            prop_boot = prop_te.iloc[idx].reset_index(drop=True)

            if aware_boot[EVENT].nunique() < 2:
                continue

            metrics = {}
            failed = False

            for model, tr_df, te_full, te_boot in [
                (AWARE, aware_tr, aware_te, aware_boot),
                (PROP, prop_tr, prop_te, prop_boot),
            ]:
                try:
                    tr_risk = compose(tr_df, tr_mask)
                    te_all_risk = compose(te_full, te_mask)
                    te_risk = te_all_risk[idx]

                    metrics[model] = evaluate_metrics(
                        tr_df, te_boot, tr_risk, te_risk, args.times
                    )
                except Exception:
                    failed = True
                    break

            if failed:
                continue

            valid += 1
            for metric in store:
                a = metrics[AWARE][metric]
                p = metrics[PROP][metric]
                store[metric].append(p - a)
                model_store[AWARE][metric].append(a)
                model_store[PROP][metric].append(p)

        if valid == 0:
            raise RuntimeError(f"No valid bootstrap replicates for {pct}%")

        for model in MODELS:
            for metric in store:
                v = np.asarray(model_store[model][metric], float)
                boot_model_rows.append({
                    "Model": model,
                    "Missing_percent": pct,
                    "Metric": metric,
                    "N_valid_bootstrap": len(v),
                    "Mean": np.mean(v),
                    "CI95_lower": np.percentile(v, 2.5),
                    "CI95_upper": np.percentile(v, 97.5),
                })

        for metric, vals in store.items():
            v = np.asarray(vals, float)
            boot_diff_rows.append({
                "Missing_percent": pct,
                "Metric": metric,
                "N_valid_bootstrap": len(v),
                "Delta_mean_Proposed_minus_Aware": np.mean(v),
                "CI95_lower": np.percentile(v, 2.5),
                "CI95_upper": np.percentile(v, 97.5),
                "Bootstrap_sign_p_two_sided": sign_p(v),
            })

    boot_models = pd.DataFrame(boot_model_rows)
    boot_diffs = pd.DataFrame(boot_diff_rows)

    boot_diffs["BH_q_across_15_global_tests"] = bh_adjust(
        boot_diffs["Bootstrap_sign_p_two_sided"].to_numpy(float)
    )

    boot_models.to_csv(
        outdir / "MissingnessAware_vs_Proposed_Bootstrap_ModelMetrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot_diffs.to_csv(
        outdir / "MissingnessAware_vs_Proposed_PairedBootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Summary text
    # -------------------------------------------------------------
    lines = [
        "LUNG1 — FINAL PAIRED COMPARISON: missingness-aware ModDrop baseline vs proposed gated residual",
        "",
        f"Train n={len(aware_tr)} | validation was used previously for checkpoint selection only",
        f"Held-out test n={len(aware_te)} | events={int(aware_te[EVENT].sum())}",
        f"Fixed-cohort repeats at 10/30/50%: {args.scenario_repeats}",
        f"Paired bootstrap requested: {args.bootstrap}",
        f"Horizons: {' | '.join(map(str, args.times))} days",
        "No model was trained, tuned, recalibrated, or selected in this comparison.",
        "",
        "FIXED-COHORT SUMMARY",
    ]

    for pct in [0, 10, 30, 50, 100]:
        lines.append(f"  Missing {pct}%:")
        for model in MODELS:
            r = sim_summary[
                (sim_summary["Model"] == model)
                & (sim_summary["Missing_percent"] == pct)
            ].iloc[0]
            lines.append(
                f"    {model}: "
                f"C={r['C_index_mean']:.4f}, "
                f"mean AUC={r['Mean_dynamic_AUC_mean']:.4f}, "
                f"IBS={r['Integrated_Brier_Score_mean']:.4f}"
            )

    lines += [
        "",
        "PAIRED BOOTSTRAP",
        "  Delta = Proposed - Missingness-aware comparator.",
        "  Positive delta favors Proposed for C-index/AUC.",
        "  Negative delta favors Proposed for IBS.",
        "  BH q-values correct the 15 scenario x metric tests.",
    ]

    for pct in [0, 10, 30, 50, 100]:
        lines.append(f"  Missing {pct}%:")
        x = boot_diffs[boot_diffs["Missing_percent"] == pct]
        for metric in ["C_index", "Mean_dynamic_AUC", "Integrated_Brier_Score"]:
            r = x[x["Metric"] == metric].iloc[0]
            lines.append(
                f"    {metric}: "
                f"delta={r['Delta_mean_Proposed_minus_Aware']:+.4f}, "
                f"95% CI [{r['CI95_lower']:+.4f}, {r['CI95_upper']:+.4f}], "
                f"p={r['Bootstrap_sign_p_two_sided']:.4f}, "
                f"BH q={r['BH_q_across_15_global_tests']:.4f}"
            )

    lines += [
        "",
        "INTERPRETATION GUARDRAILS",
        "  - Do not claim superiority if the paired 95% CI includes zero.",
        "  - 10/30/50% simulation percentile ranges are not confidence intervals.",
        "  - Both models were trained with whole-clinical ModDrop; the comparator has standard concatenation, whereas the proposed model adds an explicit availability-gated residual and CT-teacher anchor.",
        "  - The held-out test is evaluation only.",
        "  - Full-vs-no-clinical risk differences are descriptive model-dependence diagnostics, not normalized modality importance.",
    ]

    summary = "\n".join(lines)
    (outdir / "MissingnessAware_vs_Proposed_Summary.txt").write_text(
        summary, encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)
    print("Files written to:", outdir)


if __name__ == "__main__":
    main()
