#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LUNG1 - FINAL paired statistical comparison of four corrected survival models
=============================================================

Models compared on the SAME locked test patients:
  1) Clinical-only Model A
  2) CT-only
  3) Naive CT + clinical intermediate fusion
  4) Robust V2 gated fusion + modality dropout (0% missing clinical data)

Primary comparison:
  Harrell C-index on the locked test set, with paired patient-level bootstrap
  differences and two-sided bootstrap sign p-values.

Secondary comparisons:
  - time-dependent AUC at 365, 730, 1095 days
  - Brier score at 365, 730, 1095 days
  - Integrated Brier Score (IBS)

Important:
  * This script does NOT tune or refit the neural models.
  * It merges predictions by PatientID and checks outcome consistency.
  * The same bootstrap sample indices are used for every model in each replicate.
  * Test remains a final evaluation set.

Expected folders
----------------
/content/LUNG1_CT_Only/
  clinical_baselines/model_A/
    model.pkl
    train_features.csv
    test_features.csv
  ct_only_fullcox_efron/
    train_risk_scores.csv
    test_risk_scores.csv
  multimodal_fullcox_efron/
    train_risk_scores.csv
    test_risk_scores.csv
  gated_moddrop_v2_fullcox_efron/
    train_risk_modes.csv
    test_risk_modes.csv

Run
---
python compare_survival_models_paired.py \
  --root "/content/LUNG1_CT_Only" \
  --out "/content/LUNG1_CT_Only/statistical_comparison" \
  --bootstrap 5000 \
  --seed 20260926
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lifelines.utils import concordance_index
from sksurv.metrics import (
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)

try:
    import joblib
except Exception:
    joblib = None


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

ID = "PatientID"
TIME = "Survival.time"
EVENT = "deadstatus.event"

MODEL_NAMES = [
    "Clinical-only",
    "CT-only",
    "Naive CT+Clinical",
    "Robust V2",
]


def cindex(time, event, risk):
    return float(
        concordance_index(
            np.asarray(time, dtype=float),
            -np.asarray(risk, dtype=float),
            event_observed=np.asarray(event, dtype=int),
        )
    )


def to_surv_array(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    y = np.empty(
        len(time),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    y["event"] = event.astype(bool)
    y["time"] = time
    return y


def load_pickle_any(path: Path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    with open(path, "rb") as f:
        return pickle.load(f)


def extract_model(obj):
    """
    Handles a directly saved Cox object or a simple dict wrapper.
    """
    if hasattr(obj, "predict_log_partial_hazard"):
        return obj
    if hasattr(obj, "predict_partial_hazard"):
        return obj
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        preferred = [
            "model",
            "cox_model",
            "cph",
            "estimator",
            "coxph",
        ]
        for k in preferred:
            if k in obj:
                candidate = obj[k]
                if (
                    hasattr(candidate, "predict_log_partial_hazard")
                    or hasattr(candidate, "predict_partial_hazard")
                    or hasattr(candidate, "predict")
                ):
                    return candidate

        for candidate in obj.values():
            if (
                hasattr(candidate, "predict_log_partial_hazard")
                or hasattr(candidate, "predict_partial_hazard")
                or hasattr(candidate, "predict")
            ):
                return candidate

    raise RuntimeError(
        "Could not locate a prediction-capable model in model.pkl"
    )


def predict_clinical_risk(model, X):
    """
    Produces a monotonic risk score suitable for C-index/AUC.
    Prefer log partial hazard when available.
    """
    if hasattr(model, "predict_log_partial_hazard"):
        pred = model.predict_log_partial_hazard(X)
        return np.asarray(pred, dtype=float).reshape(-1)

    if hasattr(model, "predict_partial_hazard"):
        pred = np.asarray(
            model.predict_partial_hazard(X),
            dtype=float,
        ).reshape(-1)
        pred = np.maximum(pred, np.finfo(float).tiny)
        return np.log(pred)

    if hasattr(model, "predict"):
        pred = model.predict(X)
        return np.asarray(pred, dtype=float).reshape(-1)

    raise RuntimeError("Unsupported clinical model prediction API.")


# ---------------------------------------------------------------------
# Read the four models
# ---------------------------------------------------------------------

def clinical_risk_from_model(model_dir: Path, subset: str):
    """
    Derive risk from the already-fitted clinical Cox Model A.
    """
    feature_path = model_dir / f"{subset}_features.csv"
    model_path = model_dir / "model.pkl"

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    df = pd.read_csv(feature_path)

    required = [ID, TIME, EVENT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{feature_path} is missing columns: {missing}"
        )

    feature_cols = [
        c for c in df.columns
        if c not in [ID, TIME, EVENT]
    ]

    obj = load_pickle_any(model_path)
    model = extract_model(obj)

    X = df[feature_cols].copy()
    risk = predict_clinical_risk(model, X)

    if len(risk) != len(df):
        raise RuntimeError(
            "Clinical prediction length does not match feature table."
        )

    return pd.DataFrame({
        ID: df[ID].astype(str),
        TIME: df[TIME].astype(float),
        EVENT: df[EVENT].astype(int),
        "risk_score": risk.astype(float),
    })


def standard_risk_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [ID, TIME, EVENT, "risk_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path} is missing columns: {missing}"
        )

    return df[required].copy()


def robust_v2_risk_file(path: Path, risk_col="risk_full"):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [ID, TIME, EVENT, risk_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path} is missing columns: {missing}"
        )

    out = df[[ID, TIME, EVENT, risk_col]].copy()
    out = out.rename(columns={risk_col: "risk_score"})
    return out


# ---------------------------------------------------------------------
# Alignment and QC
# ---------------------------------------------------------------------

def align_predictions(predictions: dict[str, pd.DataFrame]):
    """
    Inner-align by PatientID, then require identical patient sets and outcomes.
    """
    normalized = {}

    for name, df in predictions.items():
        x = df.copy()
        x[ID] = x[ID].astype(str)

        if x[ID].duplicated().any():
            dup = x.loc[x[ID].duplicated(), ID].tolist()[:10]
            raise RuntimeError(
                f"Duplicate PatientID in {name}: {dup}"
            )

        normalized[name] = x.sort_values(ID).reset_index(drop=True)

    ref_name = MODEL_NAMES[0]
    ref = normalized[ref_name]
    ref_ids = ref[ID].tolist()

    for name in MODEL_NAMES[1:]:
        cur = normalized[name]

        if cur[ID].tolist() != ref_ids:
            only_ref = sorted(
                set(ref[ID]) - set(cur[ID])
            )[:10]
            only_cur = sorted(
                set(cur[ID]) - set(ref[ID])
            )[:10]

            raise RuntimeError(
                f"Patient sets/order differ between {ref_name} and {name}. "
                f"Only ref={only_ref}; only {name}={only_cur}"
            )

        if not np.allclose(
            cur[TIME].to_numpy(float),
            ref[TIME].to_numpy(float),
            rtol=0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Survival times differ between {ref_name} and {name}."
            )

        if not np.array_equal(
            cur[EVENT].to_numpy(int),
            ref[EVENT].to_numpy(int),
        ):
            raise RuntimeError(
                f"Event indicators differ between {ref_name} and {name}."
            )

    combined = ref[[ID, TIME, EVENT]].copy()

    for name in MODEL_NAMES:
        combined[name] = normalized[name]["risk_score"].to_numpy(float)

    return combined


# ---------------------------------------------------------------------
# Cox calibration for Brier/IBS
# ---------------------------------------------------------------------

def fit_breslow(time, event, risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    shift = float(np.max(risk))
    relative = np.exp(risk - shift)

    event_times = np.sort(
        np.unique(time[event == 1])
    )

    increments = []

    for et in event_times:
        d = np.sum(
            (time == et)
            & (event == 1)
        )
        denom = np.sum(
            relative[time >= et]
        )

        increments.append(
            d / denom
            if denom > 0
            else 0.0
        )

    return {
        "event_times": event_times,
        "cum_hazard": np.cumsum(
            np.asarray(increments, dtype=float)
        ),
        "shift": shift,
    }


def survival_matrix(risk, times, baseline):
    h0 = []

    for t in times:
        pos = np.searchsorted(
            baseline["event_times"],
            t,
            side="right",
        ) - 1

        h0.append(
            baseline["cum_hazard"][pos]
            if pos >= 0
            else 0.0
        )

    h0 = np.asarray(h0, dtype=float)
    relative = np.exp(
        np.asarray(risk, dtype=float)
        - baseline["shift"]
    )

    return np.exp(
        -relative[:, None] * h0[None, :]
    )


# ---------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------

def valid_times(train_time, test_time, requested):
    lo = max(
        float(np.min(train_time)),
        float(np.min(test_time)),
    )
    hi = min(
        float(np.max(train_time)),
        float(np.max(test_time)),
    )

    out = np.asarray(
        [x for x in requested if lo < x < hi],
        dtype=float,
    )

    if len(out) < 2:
        raise RuntimeError(
            "Fewer than two valid evaluation times; IBS cannot be computed."
        )

    return out


def point_metrics(
    train_aligned,
    test_aligned,
    times,
):
    y_train = to_surv_array(
        train_aligned[TIME],
        train_aligned[EVENT],
    )
    y_test = to_surv_array(
        test_aligned[TIME],
        test_aligned[EVENT],
    )

    rows = []
    horizon_rows = []
    survival_cache = {}

    for model in MODEL_NAMES:
        train_risk = train_aligned[model].to_numpy(float)
        test_risk = test_aligned[model].to_numpy(float)

        c = cindex(
            test_aligned[TIME],
            test_aligned[EVENT],
            test_risk,
        )

        aucs, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            test_risk,
            times,
        )

        baseline = fit_breslow(
            train_aligned[TIME],
            train_aligned[EVENT],
            train_risk,
        )

        surv = survival_matrix(
            test_risk,
            times,
            baseline,
        )
        survival_cache[model] = surv

        _, bs = brier_score(
            y_train,
            y_test,
            surv,
            times,
        )

        ibs = float(
            integrated_brier_score(
                y_train,
                y_test,
                surv,
                times,
            )
        )

        rows.append({
            "Model": model,
            "N_test": len(test_aligned),
            "Events_test": int(test_aligned[EVENT].sum()),
            "C_index": c,
            "Mean_dynamic_AUC": float(mean_auc),
            "Integrated_Brier_Score": ibs,
        })

        for t, auc, b in zip(
            times,
            aucs,
            bs,
        ):
            horizon_rows.append({
                "Model": model,
                "time_days": int(t),
                "AUC_dynamic": float(auc),
                "Brier_score": float(b),
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(horizon_rows),
        survival_cache,
    )


# ---------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------

def two_sided_bootstrap_sign_p(deltas):
    """
    Descriptive two-sided bootstrap sign probability around zero.
    Add-one correction avoids p=0.
    """
    d = np.asarray(deltas, dtype=float)
    n = len(d)

    left = (np.sum(d <= 0) + 1) / (n + 1)
    right = (np.sum(d >= 0) + 1) / (n + 1)

    return float(
        min(
            1.0,
            2.0 * min(left, right),
        )
    )


def paired_bootstrap(
    train_aligned,
    test_aligned,
    times,
    survival_cache,
    n_boot,
    seed,
):
    rng = np.random.default_rng(seed)

    n = len(test_aligned)

    y_train = to_surv_array(
        train_aligned[TIME],
        train_aligned[EVENT],
    )

    # Store model metric distributions on identical resamples.
    dist = {
        m: {
            "C_index": [],
            "Mean_dynamic_AUC": [],
            "Integrated_Brier_Score": [],
        }
        for m in MODEL_NAMES
    }

    auc_by_time = {
        m: {int(t): [] for t in times}
        for m in MODEL_NAMES
    }
    brier_by_time = {
        m: {int(t): [] for t in times}
        for m in MODEL_NAMES
    }

    valid_reps = 0

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        tb = test_aligned.iloc[idx].reset_index(drop=True)

        # Require both event states to keep all metrics interpretable.
        if tb[EVENT].nunique() < 2:
            continue

        yb = to_surv_array(
            tb[TIME],
            tb[EVENT],
        )

        rep = {}
        failed = False

        for model in MODEL_NAMES:
            risk_b = tb[model].to_numpy(float)

            try:
                c = cindex(
                    tb[TIME],
                    tb[EVENT],
                    risk_b,
                )

                aucs, mean_auc = cumulative_dynamic_auc(
                    y_train,
                    yb,
                    risk_b,
                    times,
                )

                surv_b = survival_cache[model][idx, :]

                _, bs = brier_score(
                    y_train,
                    yb,
                    surv_b,
                    times,
                )

                ibs = float(
                    integrated_brier_score(
                        y_train,
                        yb,
                        surv_b,
                        times,
                    )
                )
            except Exception:
                failed = True
                break

            rep[model] = {
                "C_index": float(c),
                "Mean_dynamic_AUC": float(mean_auc),
                "Integrated_Brier_Score": float(ibs),
                "AUCs": np.asarray(aucs, dtype=float),
                "Briers": np.asarray(bs, dtype=float),
            }

        if failed:
            continue

        valid_reps += 1

        for model in MODEL_NAMES:
            dist[model]["C_index"].append(
                rep[model]["C_index"]
            )
            dist[model]["Mean_dynamic_AUC"].append(
                rep[model]["Mean_dynamic_AUC"]
            )
            dist[model]["Integrated_Brier_Score"].append(
                rep[model]["Integrated_Brier_Score"]
            )

            for j, t in enumerate(times):
                auc_by_time[model][int(t)].append(
                    rep[model]["AUCs"][j]
                )
                brier_by_time[model][int(t)].append(
                    rep[model]["Briers"][j]
                )

    if valid_reps < max(500, int(n_boot * 0.8)):
        raise RuntimeError(
            f"Only {valid_reps}/{n_boot} valid bootstrap replicates."
        )

    # Individual model C-index uncertainty.
    individual_rows = []

    for model in MODEL_NAMES:
        for metric in [
            "C_index",
            "Mean_dynamic_AUC",
            "Integrated_Brier_Score",
        ]:
            v = np.asarray(
                dist[model][metric],
                dtype=float,
            )

            individual_rows.append({
                "Model": model,
                "Metric": metric,
                "Bootstrap_replicates": len(v),
                "Bootstrap_mean": float(np.mean(v)),
                "CI95_lower": float(np.percentile(v, 2.5)),
                "CI95_upper": float(np.percentile(v, 97.5)),
            })

    # Pairwise differences.
    pair_rows = []

    for i in range(len(MODEL_NAMES)):
        for j in range(i + 1, len(MODEL_NAMES)):
            a = MODEL_NAMES[i]
            b = MODEL_NAMES[j]

            for metric in [
                "C_index",
                "Mean_dynamic_AUC",
                "Integrated_Brier_Score",
            ]:
                va = np.asarray(
                    dist[a][metric],
                    dtype=float,
                )
                vb = np.asarray(
                    dist[b][metric],
                    dtype=float,
                )

                delta = vb - va

                pair_rows.append({
                    "Model_A": a,
                    "Model_B": b,
                    "Metric": metric,
                    "Delta_B_minus_A_bootstrap_mean": float(np.mean(delta)),
                    "CI95_lower": float(np.percentile(delta, 2.5)),
                    "CI95_upper": float(np.percentile(delta, 97.5)),
                    "Bootstrap_sign_p_two_sided": two_sided_bootstrap_sign_p(delta),
                    "Bootstrap_replicates": len(delta),
                })

    # Per-horizon pairwise differences.
    horizon_pair_rows = []

    for t in times:
        tt = int(t)

        for i in range(len(MODEL_NAMES)):
            for j in range(i + 1, len(MODEL_NAMES)):
                a = MODEL_NAMES[i]
                b = MODEL_NAMES[j]

                for metric_name, store in [
                    ("AUC_dynamic", auc_by_time),
                    ("Brier_score", brier_by_time),
                ]:
                    va = np.asarray(
                        store[a][tt],
                        dtype=float,
                    )
                    vb = np.asarray(
                        store[b][tt],
                        dtype=float,
                    )
                    delta = vb - va

                    horizon_pair_rows.append({
                        "time_days": tt,
                        "Model_A": a,
                        "Model_B": b,
                        "Metric": metric_name,
                        "Delta_B_minus_A_bootstrap_mean": float(np.mean(delta)),
                        "CI95_lower": float(np.percentile(delta, 2.5)),
                        "CI95_upper": float(np.percentile(delta, 97.5)),
                        "Bootstrap_sign_p_two_sided": two_sided_bootstrap_sign_p(delta),
                        "Bootstrap_replicates": len(delta),
                    })

    return (
        pd.DataFrame(individual_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(horizon_pair_rows),
        valid_reps,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    point,
    pairwise,
    times,
    n_boot,
    valid_reps,
):
    lines = [
        "LUNG1 - FINAL paired statistical comparison of corrected survival models",
        "",
        f"Requested bootstrap replicates: {n_boot}",
        f"Valid paired bootstrap replicates: {valid_reps}",
        f"Evaluation horizons: {' | '.join(str(int(t)) for t in times)} days",
        "",
        "POINT ESTIMATES ON LOCKED TEST SET",
    ]

    for _, r in point.iterrows():
        lines.append(
            f"{r['Model']}: "
            f"C-index={r['C_index']:.4f}, "
            f"mean dynamic AUC={r['Mean_dynamic_AUC']:.4f}, "
            f"IBS={r['Integrated_Brier_Score']:.4f}"
        )

    lines += [
        "",
        "PRIMARY PAIRED C-INDEX COMPARISONS",
        "(Delta = Model B - Model A)",
    ]

    pc = pairwise[
        pairwise["Metric"] == "C_index"
    ].copy()

    for _, r in pc.iterrows():
        lines.append(
            f"{r['Model_B']} vs {r['Model_A']}: "
            f"Delta C={r['Delta_B_minus_A_bootstrap_mean']:+.4f}, "
            f"95% CI [{r['CI95_lower']:+.4f}, {r['CI95_upper']:+.4f}], "
            f"bootstrap sign p={r['Bootstrap_sign_p_two_sided']:.4f}"
        )

    lines += [
        "",
        "Interpretation rule:",
        "A paired difference CI excluding 0 supports a difference in that metric",
        "under this bootstrap analysis. If the CI includes 0, do not claim",
        "statistical superiority from this test set.",
        "",
        "For IBS and Brier score, LOWER is better. Therefore a negative",
        "Delta(B-A) favors Model B. For C-index and AUC, HIGHER is better.",
        "",
        "Methodological note:",
        "All models are compared on exactly the same locked test patients.",
        "The paired bootstrap resamples patients identically across models.",
        "No model is refit or tuned in this script.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
        help="Root experiment directory, e.g. /content/LUNG1_CT_Only",
    )
    ap.add_argument(
        "--out",
        required=True,
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260926,
    )
    ap.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=[365, 730, 1095],
    )

    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    clinical_dir = root / "clinical_baselines" / "model_A"
    ct_dir = root / "ct_only_fullcox_efron"
    naive_dir = root / "multimodal_fullcox_efron"
    v2_dir = root / "gated_moddrop_v2_fullcox_efron"

    print("Loading predictions...")

    train_preds = {
        "Clinical-only": clinical_risk_from_model(
            clinical_dir,
            "train",
        ),
        "CT-only": standard_risk_file(
            ct_dir / "train_risk_scores.csv"
        ),
        "Naive CT+Clinical": standard_risk_file(
            naive_dir / "train_risk_scores.csv"
        ),
        "Robust V2": robust_v2_risk_file(
            v2_dir / "train_risk_modes.csv",
            "risk_full",
        ),
    }

    test_preds = {
        "Clinical-only": clinical_risk_from_model(
            clinical_dir,
            "test",
        ),
        "CT-only": standard_risk_file(
            ct_dir / "test_risk_scores.csv"
        ),
        "Naive CT+Clinical": standard_risk_file(
            naive_dir / "test_risk_scores.csv"
        ),
        "Robust V2": robust_v2_risk_file(
            v2_dir / "test_risk_modes.csv",
            "risk_full",
        ),
    }

    train = align_predictions(train_preds)
    test = align_predictions(test_preds)

    print(
        f"Aligned train patients: {len(train)} | "
        f"events={int(train[EVENT].sum())}"
    )
    print(
        f"Aligned test patients: {len(test)} | "
        f"events={int(test[EVENT].sum())}"
    )

    times = valid_times(
        train[TIME].to_numpy(float),
        test[TIME].to_numpy(float),
        args.times,
    )

    point, horizon_point, survival_cache = point_metrics(
        train,
        test,
        times,
    )

    print("\nPoint estimates:")
    print(point.to_string(index=False))

    individual_boot, pairwise, horizon_pairwise, valid_reps = paired_bootstrap(
        train,
        test,
        times,
        survival_cache,
        args.bootstrap,
        args.seed,
    )

    point.to_csv(
        outdir / "Model_Test_Point_Estimates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_point.to_csv(
        outdir / "Model_Test_Per_Horizon_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    individual_boot.to_csv(
        outdir / "Model_Bootstrap_Intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        outdir / "Pairwise_Bootstrap_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_pairwise.to_csv(
        outdir / "Pairwise_Per_Horizon_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save aligned risk table as an audit trail.
    test.to_csv(
        outdir / "Aligned_Test_Risk_Scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        point,
        pairwise,
        times,
        args.bootstrap,
        valid_reps,
    )

    (outdir / "Statistical_Comparison_Summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)
    print("Files written to:", outdir)


if __name__ == "__main__":
    main()
