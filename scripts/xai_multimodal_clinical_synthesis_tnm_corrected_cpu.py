#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LUNG1 — TNM-CORRECTED CPU MULTIMODAL CLINICAL XAI SYNTHESIS
============================================================

Purpose
-------
Use ONLY already-computed corrected SHAP outputs from the locked proposed
gated residual model to quantify how:
  1) the learned gate behaves;
  2) the exact clinical residual correction behaves;
  3) SHAP magnitude relates to the exact clinical correction.

No CT forward pass, no Grad-CAM, no GPU, no retraining, no tuning.

Model identity
--------------
risk_final = risk_CT + availability * gate * clinical_residual

When clinical data are available (availability=1):
delta_clinical = risk_final - risk_no_clinical
               = gate * clinical_residual

IMPORTANT
---------
The gate is NOT a normalized modality-importance score.
Associations involving the gate are descriptive and partly architecture-linked.

Expected input directory
------------------------
SHAP_Clinical_Modality_Effect.csv
SHAP_PerPatient_Grouped.csv
SHAP_PerPatient_Transformed.csv
SHAP_Risk_Recomputation_QC.csv

Outputs
-------
Multimodal_Clinical_XAI_PerPatient.csv
Multimodal_Clinical_XAI_Descriptives.csv
Multimodal_Clinical_XAI_Correlations.csv
Multimodal_Clinical_XAI_DominantGroups.csv
Multimodal_Clinical_XAI_QC.csv
Multimodal_Clinical_XAI_Summary.txt
Figure_Gate_vs_ClinicalCorrection.png/.pdf
Figure_Gate_vs_SHAPMagnitude.png/.pdf
Figure_ClinicalCorrection_vs_SHAPMagnitude.png/.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ID = "PatientID"


def load_csv(path: Path, required=None):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {missing}")
    if ID in df.columns:
        df[ID] = df[ID].astype(str)
    return df


def ensure_unique(df: pd.DataFrame, label: str):
    if df[ID].duplicated().any():
        bad = df.loc[df[ID].duplicated(), ID].tolist()[:10]
        raise RuntimeError(f"Duplicate PatientID in {label}: {bad}")


def bootstrap_mean_ci(values, n_boot, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(x[idx])

    return (
        float(np.mean(x)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def bootstrap_spearman(x, y, n_boot, seed):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 4:
        return {
            "N": int(len(x)),
            "rho": np.nan,
            "p": np.nan,
            "CI95_lower": np.nan,
            "CI95_upper": np.nan,
            "valid_bootstrap": 0,
        }

    point = spearmanr(x, y)
    rho = float(point.statistic)
    p = float(point.pvalue)

    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rr = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rr):
            vals.append(float(rr))

    vals = np.asarray(vals, dtype=float)

    return {
        "N": int(n),
        "rho": rho,
        "p": p,
        "CI95_lower": float(np.percentile(vals, 2.5)),
        "CI95_upper": float(np.percentile(vals, 97.5)),
        "valid_bootstrap": int(len(vals)),
    }


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    tmp = np.empty(m)
    tmp[order] = adj
    q[valid] = tmp
    return q


def scatter(df, x, y, xlabel, ylabel, title, outbase: Path):
    sub = df[[x, y]].dropna().copy()

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(sub[x], sub[y], alpha=0.75)

    if len(sub) >= 2:
        coef = np.polyfit(
            sub[x].to_numpy(float),
            sub[y].to_numpy(float),
            deg=1,
        )
        xs = np.linspace(float(sub[x].min()), float(sub[x].max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linestyle="--", linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(str(outbase) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shap-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--qc-tol", type=float, default=1e-5)

    args = ap.parse_args()

    shap_dir = Path(args.shap_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load corrected SHAP products
    # --------------------------------------------------------------
    modality = load_csv(
        shap_dir / "SHAP_Clinical_Modality_Effect.csv",
        required=[
            ID,
            "risk_full",
            "risk_no_clinical",
            "delta_clinical",
            "abs_delta_clinical",
            "gate_full",
            "clinical_residual_full",
        ],
    )
    ensure_unique(modality, "modality effect")

    grouped = load_csv(
        shap_dir / "SHAP_PerPatient_Grouped.csv",
        required=[
            ID,
            "group",
            "group_shap_additive",
            "abs_group_shap_additive",
        ],
    )

    transformed = load_csv(
        shap_dir / "SHAP_PerPatient_Transformed.csv",
        required=[
            ID,
            "feature",
            "group",
            "feature_value",
            "shap_value",
            "abs_shap",
        ],
    )

    risk_qc = load_csv(
        shap_dir / "SHAP_Risk_Recomputation_QC.csv",
        required=[
            ID,
            "abs_diff_full",
            "abs_diff_no_clinical",
        ],
    )
    ensure_unique(risk_qc, "risk recomputation QC")

    # --------------------------------------------------------------
    # Patient-level SHAP descriptors
    # --------------------------------------------------------------
    total_grouped = (
        grouped.groupby(ID, as_index=False)["abs_group_shap_additive"]
        .sum()
        .rename(
            columns={
                "abs_group_shap_additive":
                    "total_abs_grouped_clinical_SHAP"
            }
        )
    )

    idx_group = grouped.groupby(ID)["abs_group_shap_additive"].idxmax()
    dominant_group = grouped.loc[
        idx_group,
        [ID, "group", "group_shap_additive", "abs_group_shap_additive"],
    ].copy()

    dominant_group = dominant_group.rename(
        columns={
            "group": "dominant_SHAP_group",
            "group_shap_additive": "dominant_SHAP_group_signed",
            "abs_group_shap_additive": "dominant_SHAP_group_abs",
        }
    )

    idx_feature = transformed.groupby(ID)["abs_shap"].idxmax()
    dominant_feature = transformed.loc[
        idx_feature,
        [ID, "feature", "group", "feature_value", "shap_value", "abs_shap"],
    ].copy()

    dominant_feature = dominant_feature.rename(
        columns={
            "feature": "dominant_SHAP_feature",
            "group": "dominant_SHAP_feature_group",
            "feature_value": "dominant_SHAP_feature_value",
            "shap_value": "dominant_SHAP_feature_signed",
            "abs_shap": "dominant_SHAP_feature_abs",
        }
    )

    fused = (
        modality
        .merge(total_grouped, on=ID, validate="one_to_one")
        .merge(dominant_group, on=ID, validate="one_to_one")
        .merge(dominant_feature, on=ID, validate="one_to_one")
        .merge(
            risk_qc[
                [ID, "abs_diff_full", "abs_diff_no_clinical"]
            ],
            on=ID,
            validate="one_to_one",
        )
    )

    fused["gate_times_residual"] = (
        fused["gate_full"] * fused["clinical_residual_full"]
    )

    fused["structural_error"] = (
        fused["delta_clinical"] - fused["gate_times_residual"]
    ).abs()

    fused["risk_difference_identity_error"] = (
        fused["delta_clinical"]
        - (fused["risk_full"] - fused["risk_no_clinical"])
    ).abs()

    fused.to_csv(
        outdir / "Multimodal_Clinical_XAI_PerPatient.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # QC
    # --------------------------------------------------------------
    qc_rows = []

    def add_qc(name, value, expected, passed):
        qc_rows.append({
            "check": name,
            "value": value,
            "expected": expected,
            "pass": bool(passed),
        })

    add_qc(
        "Patient count",
        len(fused),
        64,
        len(fused) == 64,
    )

    max_full_qc = float(fused["abs_diff_full"].max())
    max_none_qc = float(fused["abs_diff_no_clinical"].max())
    max_struct = float(fused["structural_error"].max())
    max_risk_identity = float(fused["risk_difference_identity_error"].max())

    add_qc(
        "max saved-vs-recomputed full risk error",
        max_full_qc,
        f"<= {args.qc_tol}",
        max_full_qc <= args.qc_tol,
    )

    add_qc(
        "max saved-vs-recomputed no-clinical risk error",
        max_none_qc,
        f"<= {args.qc_tol}",
        max_none_qc <= args.qc_tol,
    )

    add_qc(
        "max |delta - gate*residual|",
        max_struct,
        f"<= {args.qc_tol}",
        max_struct <= args.qc_tol,
    )

    add_qc(
        "max |delta - (risk_full-risk_no_clinical)|",
        max_risk_identity,
        f"<= {args.qc_tol}",
        max_risk_identity <= args.qc_tol,
    )

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(
        outdir / "Multimodal_Clinical_XAI_QC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not bool(qc_df["pass"].all()):
        print(qc_df.to_string(index=False))
        raise RuntimeError("Multimodal clinical XAI QC failed.")

    # --------------------------------------------------------------
    # Descriptives
    # --------------------------------------------------------------
    specs = [
        ("gate_full", "Gate value"),
        ("delta_clinical", "Signed clinical correction"),
        ("abs_delta_clinical", "Absolute clinical correction"),
        (
            "total_abs_grouped_clinical_SHAP",
            "Total |grouped clinical SHAP|",
        ),
    ]

    desc_rows = []

    for i, (col, label) in enumerate(specs):
        x = pd.to_numeric(fused[col], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]

        mean, lo, hi = bootstrap_mean_ci(
            x,
            args.bootstrap,
            args.seed + 100 + i,
        )

        desc_rows.append({
            "variable": col,
            "label": label,
            "N": len(x),
            "mean": mean,
            "median": float(np.median(x)),
            "sd": float(np.std(x, ddof=1)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean_CI95_lower": lo,
            "mean_CI95_upper": hi,
        })

    desc = pd.DataFrame(desc_rows)
    desc.to_csv(
        outdir / "Multimodal_Clinical_XAI_Descriptives.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Exploratory associations
    # --------------------------------------------------------------
    corr_specs = [
        (
            "gate_full",
            "abs_delta_clinical",
            "Gate value vs |clinical correction|",
        ),
        (
            "gate_full",
            "total_abs_grouped_clinical_SHAP",
            "Gate value vs total |grouped clinical SHAP|",
        ),
        (
            "abs_delta_clinical",
            "total_abs_grouped_clinical_SHAP",
            "|clinical correction| vs total |grouped clinical SHAP|",
        ),
    ]

    corr_rows = []

    for i, (xcol, ycol, label) in enumerate(corr_specs):
        res = bootstrap_spearman(
            fused[xcol],
            fused[ycol],
            args.bootstrap,
            args.seed + 1000 + i,
        )

        corr_rows.append({
            "analysis": label,
            "x": xcol,
            "y": ycol,
            **res,
        })

    corr = pd.DataFrame(corr_rows)
    corr["BH_q_across_3_tests"] = bh_fdr(corr["p"].to_numpy(float))

    corr.to_csv(
        outdir / "Multimodal_Clinical_XAI_Correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Dominant SHAP conceptual group
    # --------------------------------------------------------------
    dom = (
        fused["dominant_SHAP_group"]
        .value_counts(dropna=False)
        .rename_axis("dominant_SHAP_group")
        .reset_index(name="N")
    )

    dom["Percent"] = 100.0 * dom["N"] / len(fused)

    dom.to_csv(
        outdir / "Multimodal_Clinical_XAI_DominantGroups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------
    scatter(
        fused,
        "gate_full",
        "abs_delta_clinical",
        "Gate value",
        "|Exact clinical correction|",
        "Gate versus exact clinical correction",
        outdir / "Figure_Gate_vs_ClinicalCorrection",
    )

    scatter(
        fused,
        "gate_full",
        "total_abs_grouped_clinical_SHAP",
        "Gate value",
        "Total |grouped clinical SHAP|",
        "Gate versus clinical SHAP magnitude",
        outdir / "Figure_Gate_vs_SHAPMagnitude",
    )

    scatter(
        fused,
        "abs_delta_clinical",
        "total_abs_grouped_clinical_SHAP",
        "|Exact clinical correction|",
        "Total |grouped clinical SHAP|",
        "Clinical ablation versus SHAP magnitude",
        outdir / "Figure_ClinicalCorrection_vs_SHAPMagnitude",
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    gate = desc[desc["variable"] == "gate_full"].iloc[0]
    signed = desc[desc["variable"] == "delta_clinical"].iloc[0]
    absolute = desc[desc["variable"] == "abs_delta_clinical"].iloc[0]

    pct_up = 100.0 * np.mean(fused["delta_clinical"].to_numpy(float) > 0)
    pct_down = 100.0 * np.mean(fused["delta_clinical"].to_numpy(float) < 0)

    lines = [
        "LUNG1 — TNM-CORRECTED CPU MULTIMODAL CLINICAL XAI SYNTHESIS",
        "",
        f"Integrated locked test patients: {len(fused)}",
        f"Bootstrap replicates: {args.bootstrap}",
        "No model was retrained, tuned, or recalibrated.",
        "",
        "QC",
    ]

    for _, r in qc_df.iterrows():
        lines.append(
            f"  {r['check']}: value={r['value']} | "
            f"expected={r['expected']} | pass={r['pass']}"
        )

    lines += [
        "",
        "GATE AND EXACT CLINICAL CONTRIBUTION",
        (
            f"  Mean gate={gate['mean']:.4f}, "
            f"95% bootstrap CI [{gate['mean_CI95_lower']:.4f}, "
            f"{gate['mean_CI95_upper']:.4f}]"
        ),
        (
            f"  Mean signed clinical correction={signed['mean']:+.6f}, "
            f"95% CI [{signed['mean_CI95_lower']:+.6f}, "
            f"{signed['mean_CI95_upper']:+.6f}]"
        ),
        (
            f"  Mean |clinical correction|={absolute['mean']:.6f}, "
            f"95% CI [{absolute['mean_CI95_lower']:.6f}, "
            f"{absolute['mean_CI95_upper']:.6f}]"
        ),
        (
            f"  Clinical correction increases model risk in {pct_up:.1f}% "
            f"and decreases it in {pct_down:.1f}% of test patients."
        ),
        "",
        "EXPLORATORY ASSOCIATIONS",
        (
            "  These are descriptive Spearman associations. "
            "BH-FDR q-values are reported across the 3 analyses."
        ),
    ]

    for _, r in corr.iterrows():
        lines.append(
            f"  {r['analysis']}: rho={r['rho']:.4f}, "
            f"95% bootstrap CI [{r['CI95_lower']:.4f}, "
            f"{r['CI95_upper']:.4f}], "
            f"p={r['p']:.4g}, q={r['BH_q_across_3_tests']:.4g}, "
            f"N={int(r['N'])}"
        )

    lines += [
        "",
        "DOMINANT CLINICAL SHAP GROUPS",
    ]

    for _, r in dom.iterrows():
        lines.append(
            f"  {r['dominant_SHAP_group']}: "
            f"{int(r['N'])}/{len(fused)} ({r['Percent']:.1f}%)"
        )

    lines += [
        "",
        "INTERPRETATION GUARDRAILS",
        "  1) The gate is a learned residual-update coefficient, not a normalized modality-importance score.",
        "  2) Because delta_clinical = gate * clinical_residual by construction, gate-vs-correction association is partly architecture-linked.",
        "  3) SHAP values are background-relative post-hoc attributions and are not causal effects.",
        "  4) This analysis uses the locked held-out internal test set only for post-hoc explanation; it performs no model selection.",
        "  5) Grad-CAM localization and perturbation faithfulness are intentionally deferred to the imaging-XAI step.",
    ]

    summary = "\n".join(lines)

    (outdir / "Multimodal_Clinical_XAI_Summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(summary)
    print("=" * 88)
    print("\nOutputs written to:", outdir)


if __name__ == "__main__":
    main()
