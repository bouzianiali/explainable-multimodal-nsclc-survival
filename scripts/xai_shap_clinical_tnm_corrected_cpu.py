#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LUNG1 Proposed Gated Residual - TNM-CORRECTED Clinical SHAP / Expected Gradients
==========================================================

Purpose
-------
Explain the clinical contribution of the LOCKED TNM-corrected proposed gated residual model
on the LOCKED test set, conditional on each patient's fixed CT context.

Method
------
- SHAP GradientExplainer / Expected Gradients.
- For each test patient, CT features are computed once and then held fixed.
- The clinical branch is explained relative to a TRAINING clinical background.
- No model refitting, recalibration, or tuning is performed.
- Exact clinical-modality ablation is reported separately:
      delta_clinical = risk_full - risk_no_clinical
  because this is not the same quantity as a SHAP attribution.

Outputs
-------
- SHAP_PerPatient_Transformed.csv
- SHAP_PerPatient_Grouped.csv
- SHAP_Global_Feature_Importance.csv
- SHAP_Grouped_Clinical_Importance_ADDITIVE.csv
- SHAP_Grouped_Clinical_Importance_TOTALABS.csv
- SHAP_Clinical_Modality_Effect.csv
- SHAP_Feature_Stability.csv
- SHAP_Group_Stability.csv
- SHAP_Summary.txt
- SHAP_Grouped_Importance.png/pdf
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import shap
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model architecture - MUST match corrected Robust V2
# ---------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RobustFusionV2(nn.Module):
    def __init__(self, clinical_dim, dropout=0.30):
        super().__init__()

        self.ct_encoder = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            nn.AdaptiveAvgPool3d(1),
        )

        self.ct_risk_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.ct_feature_proj = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
        )

        self.gate = nn.Sequential(
            nn.Linear(64 + 64 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_ct_context(self, ct):
        fmap = self.ct_encoder(ct)
        flat = torch.flatten(fmap, 1)
        base_risk = self.ct_risk_head(fmap).squeeze(1)
        zct = self.ct_feature_proj(flat)
        return base_risk, zct

    def forward(self, ct, clinical, availability):
        base_risk, zct = self.encode_ct_context(ct)

        zcl = self.clinical_encoder(clinical * availability)

        gate = self.gate(
            torch.cat([zct, zcl, availability], dim=1)
        ).squeeze(1)

        residual = self.residual_head(
            torch.cat([zct, zcl], dim=1)
        ).squeeze(1)

        risk = (
            base_risk
            + availability.squeeze(1) * gate * residual
        )

        return risk, base_risk, gate, residual


class FixedCTClinicalWrapper(nn.Module):
    """
    SHAP wrapper for one patient:
    CT context is fixed; only clinical variables vary.
    """

    def __init__(self, model, base_risk, zct):
        super().__init__()
        self.clinical_encoder = model.clinical_encoder
        self.gate = model.gate
        self.residual_head = model.residual_head

        self.register_buffer(
            "base_risk_fixed",
            base_risk.detach().reshape(1),
        )
        self.register_buffer(
            "zct_fixed",
            zct.detach().reshape(1, -1),
        )

    def forward(self, clinical):
        n = clinical.shape[0]

        zct = self.zct_fixed.expand(n, -1)
        base = self.base_risk_fixed.expand(n)

        availability = torch.ones(
            (n, 1),
            dtype=clinical.dtype,
            device=clinical.device,
        )

        zcl = self.clinical_encoder(clinical)

        gate = self.gate(
            torch.cat([zct, zcl, availability], dim=1)
        ).squeeze(1)

        residual = self.residual_head(
            torch.cat([zct, zcl], dim=1)
        ).squeeze(1)

        risk = base + gate * residual

        return risk.unsqueeze(1)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_table(split_path: Path, clinical_dir: Path, subset: str):
    split = pd.read_csv(split_path)

    sub = split[
        split["Split"].astype(str).str.lower()
        == subset.lower()
    ].copy()

    feature_path = clinical_dir / f"{subset}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    clinical = pd.read_csv(feature_path)

    feature_cols = [
        c for c in clinical.columns
        if c not in [
            "PatientID",
            "Survival.time",
            "deadstatus.event",
        ]
    ]

    base = sub[
        ["PatientID", "Survival.time", "deadstatus.event"]
    ].copy()

    merged = base.merge(
        clinical[["PatientID"] + feature_cols],
        on="PatientID",
        how="left",
        validate="one_to_one",
    )

    if merged[feature_cols].isna().any().any():
        bad = merged.loc[
            merged[feature_cols].isna().any(axis=1),
            "PatientID",
        ].tolist()

        raise RuntimeError(
            f"Missing transformed clinical features for {bad[:10]}"
        )

    return merged.reset_index(drop=True), feature_cols


def load_ct(vol_dir: Path, patient_id: str):
    path = vol_dir / (
        f"{patient_id}_standardized_192x160x224.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=False)

    if "ct_hu_clipped" not in z.files:
        raise KeyError(
            f"{path}: missing ct_hu_clipped"
        )

    ct_hu = np.asarray(
        z["ct_hu_clipped"],
        dtype=np.float32,
    )

    # Must match training/XAI normalization.
    ct = np.clip(
        (ct_hu + 1000.0) / 1400.0,
        0.0,
        1.0,
    ).astype(np.float32)

    return ct


# ---------------------------------------------------------------------
# Clinical feature groups
# ---------------------------------------------------------------------

def feature_group(feature: str) -> str:
    f = feature.lower()

    if f == "age":
        return "Age"

    if f.startswith("gender_"):
        return "Gender"

    if f.startswith("histology_"):
        return "Histology"

    if (
        "clinical.m.stage" in f
        or f.startswith("m_stage")
        or f.startswith("m.")
    ):
        return "M stage"

    if (
        "clinical.n.stage" in f
        or f.startswith("n_stage")
        or f.startswith("n.")
    ):
        return "N stage"

    if (
        "clinical.t.stage" in f
        or f.startswith("t_stage")
        or f.startswith("t.")
    ):
        return "T stage"

    return f"Other:{feature}"


GROUP_ORDER = [
    "Age",
    "Gender",
    "Histology",
    "M stage",
    "N stage",
    "T stage",
]


def validate_tnm_corrected_features(feature_cols):
    """
    Hard safety check for the corrected LUNG1 clinical encoding.

    The raw sentinel-like values T=5, N=4 and M=3 must NOT survive as
    clinical stages. They were recoded as missing before train-only
    preprocessing. The transformed feature set must therefore expose
    explicit MISSING indicators rather than the old sentinel categories.
    """
    cols = list(feature_cols)

    forbidden = {
        "clinical.T.Stage_5",
        "Clinical.N.Stage_4",
        "Clinical.M.Stage_3",
    }
    bad = sorted(forbidden.intersection(cols))
    if bad:
        raise RuntimeError(
            "OLD / INVALID TNM transformed features detected: "
            + ", ".join(bad)
            + ". Use the TNM_CORRECTED clinical tables and checkpoint."
        )

    expected_missing = {
        "clinical.T.Stage_MISSING",
        "Clinical.N.Stage_MISSING",
        "Clinical.M.Stage_MISSING",
    }
    missing = sorted(expected_missing.difference(cols))
    if missing:
        raise RuntimeError(
            "Expected TNM-corrected MISSING indicators are absent: "
            + ", ".join(missing)
            + ". Refusing to continue."
        )

    if len(cols) != 21:
        raise RuntimeError(
            f"Expected 21 corrected Model-A transformed features, got {len(cols)}."
        )

    return True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_shap_output(values, n_features: int):
    if isinstance(values, list):
        if len(values) != 1:
            raise RuntimeError(
                f"Unexpected SHAP output list length: {len(values)}"
            )
        values = values[0]

    a = np.asarray(values)

    # Typical possibilities:
    # (1, features), (1, features, 1), (features,)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]

    if a.ndim == 2:
        if a.shape[0] != 1:
            raise RuntimeError(
                f"Expected one explained patient, got shape {a.shape}"
            )
        a = a[0]

    if a.ndim != 1 or len(a) != n_features:
        raise RuntimeError(
            f"Unexpected SHAP shape after normalization: {a.shape}"
        )

    return a.astype(float)


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


def ranking_stability(matrix_abs, labels, n_boot, seed):
    """
    matrix_abs: patients x features/groups, already absolute-valued
    Stability = Spearman correlation between the original global ranking and
    each patient-bootstrap global ranking.
    """
    arr = np.asarray(matrix_abs, dtype=float)

    original = np.mean(arr, axis=0)
    original_rank = pd.Series(
        original,
        index=labels,
    ).rank(
        method="average",
        ascending=False,
    )

    rng = np.random.default_rng(seed)
    n = arr.shape[0]

    rhos = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        imp = np.mean(arr[idx], axis=0)

        rank = pd.Series(
            imp,
            index=labels,
        ).rank(
            method="average",
            ascending=False,
        )

        rho = spearmanr(
            original_rank.to_numpy(),
            rank.to_numpy(),
        ).statistic

        if np.isfinite(rho):
            rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)

    return {
        "mean_spearman": float(np.mean(rhos)),
        "median_spearman": float(np.median(rhos)),
        "CI95_lower": float(np.percentile(rhos, 2.5)),
        "CI95_upper": float(np.percentile(rhos, 97.5)),
        "valid_bootstrap_replicates": int(len(rhos)),
    }


def bootstrap_spearman(x, y, n_boot, seed):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    point = float(
        spearmanr(x, y).statistic
    )

    rng = np.random.default_rng(seed)
    n = len(x)

    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho = spearmanr(
            x[idx],
            y[idx],
        ).statistic

        if np.isfinite(rho):
            vals.append(float(rho))

    vals = np.asarray(vals)

    return (
        point,
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clinical-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--background-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--nsamples",
        type=int,
        default=500,
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
        "--max-patients",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help=(
            "Default is CPU so this XAI step does not consume Colab GPU quota. "
            "Use cuda explicitly only when available."
        ),
    )

    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="PyTorch CPU threads used when --device cpu.",
    )

    ap.add_argument(
        "--risk-modes",
        default="",
        help=(
            "Optional corrected robust test_risk_modes.csv. If supplied, "
            "the script checks recomputed full/no-clinical risks against it."
        ),
    )

    ap.add_argument(
        "--risk-qc-tol",
        type=float,
        default=1e-5,
    )

    args = ap.parse_args()

    seed_all(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clinical_dir = Path(args.clinical_dir)

    train, feature_cols = load_table(
        Path(args.split),
        clinical_dir,
        "train",
    )

    test, test_feature_cols = load_table(
        Path(args.split),
        clinical_dir,
        "test",
    )

    if feature_cols != test_feature_cols:
        raise RuntimeError(
            "Train/test transformed clinical feature order differs."
        )

    validate_tnm_corrected_features(feature_cols)

    if args.max_patients > 0:
        test = test.iloc[
            :args.max_patients
        ].copy()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    ckpt_cols = ckpt.get(
        "clinical_feature_cols"
    )

    if (
        ckpt_cols is not None
        and list(ckpt_cols) != list(feature_cols)
    ):
        raise RuntimeError(
            "Clinical feature order does not match checkpoint."
        )

    dropout = float(
        ckpt.get(
            "args",
            {},
        ).get(
            "dropout",
            0.30,
        )
    )

    model = RobustFusionV2(
        len(feature_cols),
        dropout=dropout,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if device.type == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))

    model = model.to(device)
    model.eval()

    print("Test patients:", len(test))
    print("Transformed clinical features:", len(feature_cols))
    print("TNM-corrected feature audit: PASS")
    print("Device:", device)
    if device.type == "cpu":
        print("CPU-only XAI mode: no Colab GPU quota is required.")

    # --------------------------------------------------------------
    # Training clinical background
    # --------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    bg_n = min(
        args.background_size,
        len(train),
    )

    bg_idx = rng.choice(
        len(train),
        size=bg_n,
        replace=False,
    )

    bg_np = train.iloc[
        bg_idx
    ][
        feature_cols
    ].to_numpy(
        dtype=np.float32
    )

    bg_tensor = torch.from_numpy(
        bg_np
    ).to(device)

    print("Training background size:", bg_n)
    print("Expected-gradient samples per patient:", args.nsamples)

    groups = [
        feature_group(c)
        for c in feature_cols
    ]

    print("\nFEATURE GROUP MAPPING")
    for g in dict.fromkeys(groups):
        feats = [
            c for c, gg in zip(
                feature_cols,
                groups,
            )
            if gg == g
        ]
        print(f"  {g}: {', '.join(feats)}")

    shap_rows = []
    group_rows = []
    modality_rows = []

    reconstruction_residuals = []

    # --------------------------------------------------------------
    # Explain one locked test patient at a time
    # --------------------------------------------------------------
    # Two resumable caches are used:
    #   1) fixed CT context (base risk + projected CT representation)
    #   2) patient-specific Expected-Gradients result
    #
    # This makes the analysis safe to interrupt/restart on a CPU runtime.
    # --------------------------------------------------------------

    context_cache_dir = outdir / "CT_Context_Cache"
    context_cache_dir.mkdir(parents=True, exist_ok=True)

    shap_cache_dir = outdir / (
        f"SHAP_Patient_Cache_bg{bg_n}_n{args.nsamples}_seed{args.seed}"
    )
    shap_cache_dir.mkdir(parents=True, exist_ok=True)

    risk_qc_rows = []

    # Optional saved-risk QC table from the corrected final robust model.
    saved_modes = None
    if args.risk_modes:
        risk_path = Path(args.risk_modes)
        if not risk_path.exists():
            raise FileNotFoundError(risk_path)

        saved_modes = pd.read_csv(risk_path)
        needed = {
            "PatientID",
            "risk_full",
            "risk_no_clinical",
        }
        missing_cols = sorted(needed.difference(saved_modes.columns))
        if missing_cols:
            raise RuntimeError(
                f"{risk_path} missing columns: {missing_cols}"
            )

        saved_modes["PatientID"] = saved_modes["PatientID"].astype(str)
        saved_modes = saved_modes.set_index("PatientID")

    for patient_idx, row in test.iterrows():
        pid = str(row["PatientID"])

        print(
            f"[{patient_idx + 1}/{len(test)}] {pid}"
        )

        clinical_np = row[
            feature_cols
        ].to_numpy(
            dtype=np.float32
        )

        clinical = torch.from_numpy(
            clinical_np[None]
        ).to(device)

        # ----------------------------------------------------------
        # Fixed CT context: compute once, then cache to disk.
        # ----------------------------------------------------------
        context_path = context_cache_dir / f"{pid}_ct_context.npz"

        if context_path.exists():
            z = np.load(context_path, allow_pickle=False)
            base_np = np.asarray(z["base_risk"], dtype=np.float32).reshape(1)
            zct_np = np.asarray(z["zct"], dtype=np.float32).reshape(1, -1)

            base_context = torch.from_numpy(base_np).to(device)
            zct_context = torch.from_numpy(zct_np).to(device)

            context_source = "cache"
        else:
            ct_np = load_ct(
                Path(args.vol_dir),
                pid,
            )

            ct = torch.from_numpy(
                ct_np[None, None]
            ).to(device)

            with torch.no_grad():
                base_context, zct_context = model.encode_ct_context(
                    ct
                )

            np.savez_compressed(
                context_path,
                base_risk=base_context.detach().cpu().numpy(),
                zct=zct_context.detach().cpu().numpy(),
            )

            del ct
            context_source = "computed"

        wrapper = FixedCTClinicalWrapper(
            model,
            base_context,
            zct_context,
        ).to(device)
        wrapper.eval()

        patient_cache = shap_cache_dir / f"{pid}_shap.npz"

        if patient_cache.exists():
            z = np.load(patient_cache, allow_pickle=False)

            sv = np.asarray(z["shap_values"], dtype=float)
            bg_mean_pred = float(z["background_mean_prediction"])
            risk_full_float = float(z["risk_full"])
            risk_none_float = float(z["risk_no_clinical"])
            gate_full_float = float(z["gate_full"])
            residual_full_float = float(z["clinical_residual_full"])
            reconstruction_residual = float(z["reconstruction_residual"])

            shap_source = "cache"
        else:
            explainer = shap.GradientExplainer(
                wrapper,
                bg_tensor,
            )

            # Deterministic but patient-specific Expected-Gradients sampling.
            shap_values = explainer.shap_values(
                clinical,
                nsamples=args.nsamples,
                rseed=args.seed + patient_idx,
            )

            sv = normalize_shap_output(
                shap_values,
                len(feature_cols),
            )

            with torch.no_grad():
                bg_mean_pred = float(
                    wrapper(bg_tensor).mean().cpu().item()
                )

                risk_full_float = float(
                    wrapper(clinical).squeeze().cpu().item()
                )

                # Explicit no-clinical path of the proposed architecture:
                # availability=0 => final risk = CT base risk.
                risk_none_float = float(
                    base_context.reshape(-1)[0].cpu().item()
                )

                availability = torch.ones(
                    (1, 1),
                    dtype=clinical.dtype,
                    device=device,
                )

                zcl = model.clinical_encoder(clinical)
                gate_full = model.gate(
                    torch.cat(
                        [zct_context, zcl, availability],
                        dim=1,
                    )
                ).squeeze(1)

                residual_full = model.residual_head(
                    torch.cat(
                        [zct_context, zcl],
                        dim=1,
                    )
                ).squeeze(1)

                gate_full_float = float(
                    gate_full.cpu().item()
                )
                residual_full_float = float(
                    residual_full.cpu().item()
                )

            reconstructed = (
                bg_mean_pred
                + float(np.sum(sv))
            )

            reconstruction_residual = (
                risk_full_float
                - reconstructed
            )

            np.savez_compressed(
                patient_cache,
                shap_values=np.asarray(sv, dtype=np.float64),
                background_mean_prediction=np.asarray(bg_mean_pred),
                risk_full=np.asarray(risk_full_float),
                risk_no_clinical=np.asarray(risk_none_float),
                gate_full=np.asarray(gate_full_float),
                clinical_residual_full=np.asarray(residual_full_float),
                reconstruction_residual=np.asarray(reconstruction_residual),
            )

            del explainer
            shap_source = "computed"

        reconstruction_residuals.append(
            reconstruction_residual
        )

        # ----------------------------------------------------------
        # Optional exact risk QC against the already locked output.
        # ----------------------------------------------------------
        if saved_modes is not None:
            if pid not in saved_modes.index:
                raise RuntimeError(
                    f"{pid} absent from saved risk modes."
                )

            saved_full = float(
                saved_modes.loc[pid, "risk_full"]
            )
            saved_none = float(
                saved_modes.loc[pid, "risk_no_clinical"]
            )

            risk_qc_rows.append({
                "PatientID": pid,
                "risk_full_recomputed": risk_full_float,
                "risk_full_saved": saved_full,
                "abs_diff_full": abs(risk_full_float - saved_full),
                "risk_no_clinical_recomputed": risk_none_float,
                "risk_no_clinical_saved": saved_none,
                "abs_diff_no_clinical": abs(risk_none_float - saved_none),
                "context_source": context_source,
                "shap_source": shap_source,
            })

        # Transformed-feature SHAP rows
        for feature, group, value, shap_value in zip(
            feature_cols,
            groups,
            clinical_np,
            sv,
        ):
            shap_rows.append({
                "PatientID": pid,
                "Survival.time": float(
                    row["Survival.time"]
                ),
                "deadstatus.event": int(
                    row["deadstatus.event"]
                ),
                "feature": feature,
                "group": group,
                "feature_value": float(value),
                "shap_value": float(shap_value),
                "abs_shap": float(
                    abs(shap_value)
                ),
            })

        # Conceptual clinical group attributions
        unique_groups = list(
            dict.fromkeys(groups)
        )

        for group in unique_groups:
            idx = [
                i
                for i, g in enumerate(groups)
                if g == group
            ]

            additive = float(
                np.sum(
                    sv[idx]
                )
            )

            total_abs = float(
                np.sum(
                    np.abs(
                        sv[idx]
                    )
                )
            )

            group_rows.append({
                "PatientID": pid,
                "group": group,
                "group_shap_additive": additive,
                "abs_group_shap_additive": abs(additive),
                "group_total_abs_transformed_shap": total_abs,
            })

        delta = float(
            risk_full_float
            - risk_none_float
        )

        modality_rows.append({
            "PatientID": pid,
            "Survival.time": float(
                row["Survival.time"]
            ),
            "deadstatus.event": int(
                row["deadstatus.event"]
            ),
            "risk_full": risk_full_float,
            "risk_no_clinical": risk_none_float,
            "delta_clinical": delta,
            "abs_delta_clinical": abs(delta),
            "gate_full": gate_full_float,
            "clinical_residual_full": residual_full_float,
            "background_mean_prediction": bg_mean_pred,
            "shap_sum": float(
                np.sum(sv)
            ),
            "expected_gradients_reconstruction_residual":
                reconstruction_residual,
        })

        del (
            clinical,
            wrapper,
            base_context,
            zct_context,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Saved-risk QC must pass before accepting XAI results.
    # --------------------------------------------------------------
    if risk_qc_rows:
        risk_qc = pd.DataFrame(risk_qc_rows)
        risk_qc.to_csv(
            outdir / "SHAP_Risk_Recomputation_QC.csv",
            index=False,
            encoding="utf-8-sig",
        )

        max_full = float(
            risk_qc["abs_diff_full"].max()
        )
        max_none = float(
            risk_qc["abs_diff_no_clinical"].max()
        )

        print(
            "Saved-risk QC: "
            f"max |delta full|={max_full:.3e}; "
            f"max |delta no-clinical|={max_none:.3e}"
        )

        if (
            max_full > args.risk_qc_tol
            or max_none > args.risk_qc_tol
        ):
            raise RuntimeError(
                "Risk recomputation QC failed. "
                f"Tolerance={args.risk_qc_tol}; "
                f"full={max_full}; no-clinical={max_none}."
            )

    shap_long = pd.DataFrame(
        shap_rows
    )

    grouped = pd.DataFrame(
        group_rows
    )

    modality = pd.DataFrame(
        modality_rows
    )

    shap_long.to_csv(
        outdir / "SHAP_PerPatient_Transformed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped.to_csv(
        outdir / "SHAP_PerPatient_Grouped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    modality.to_csv(
        outdir / "SHAP_Clinical_Modality_Effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Global transformed feature importance + bootstrap CI
    # --------------------------------------------------------------

    feature_summary_rows = []

    for fi, feature in enumerate(feature_cols):
        sub = shap_long[
            shap_long["feature"]
            == feature
        ]

        mean_abs, lo, hi = bootstrap_mean_ci(
            sub["abs_shap"],
            args.bootstrap,
            args.seed + 1000 + fi,
        )

        feature_summary_rows.append({
            "feature": feature,
            "group": feature_group(feature),
            "N": len(sub),
            "mean_abs_shap": mean_abs,
            "CI95_lower": lo,
            "CI95_upper": hi,
        })

    feature_summary = pd.DataFrame(
        feature_summary_rows
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    feature_summary[
        "rank"
    ] = np.arange(
        1,
        len(feature_summary) + 1,
    )

    feature_summary.to_csv(
        outdir / "SHAP_Global_Feature_Importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Grouped importance
    # --------------------------------------------------------------

    group_add_rows = []
    group_abs_rows = []

    unique_groups = list(
        dict.fromkeys(groups)
    )

    for gi, group in enumerate(
        unique_groups
    ):
        sub = grouped[
            grouped["group"] == group
        ]

        mean_abs, lo, hi = bootstrap_mean_ci(
            sub["abs_group_shap_additive"],
            args.bootstrap,
            args.seed + 2000 + gi,
        )

        group_add_rows.append({
            "group": group,
            "N": len(sub),
            "mean_abs_group_shap": mean_abs,
            "CI95_lower": lo,
            "CI95_upper": hi,
        })

        mean_total, lo_total, hi_total = bootstrap_mean_ci(
            sub["group_total_abs_transformed_shap"],
            args.bootstrap,
            args.seed + 3000 + gi,
        )

        group_abs_rows.append({
            "group": group,
            "N": len(sub),
            "mean_total_abs_transformed_shap": mean_total,
            "CI95_lower": lo_total,
            "CI95_upper": hi_total,
        })

    group_add = pd.DataFrame(
        group_add_rows
    ).sort_values(
        "mean_abs_group_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    group_add["rank"] = np.arange(
        1,
        len(group_add) + 1,
    )

    group_totalabs = pd.DataFrame(
        group_abs_rows
    ).sort_values(
        "mean_total_abs_transformed_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    group_totalabs["rank"] = np.arange(
        1,
        len(group_totalabs) + 1,
    )

    group_add.to_csv(
        outdir / "SHAP_Grouped_Clinical_Importance_ADDITIVE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_totalabs.to_csv(
        outdir / "SHAP_Grouped_Clinical_Importance_TOTALABS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Rank stability under patient bootstrap
    # --------------------------------------------------------------

    feature_matrix = (
        shap_long.pivot(
            index="PatientID",
            columns="feature",
            values="abs_shap",
        )[
            feature_cols
        ]
    )

    feat_stab = ranking_stability(
        feature_matrix.to_numpy(),
        feature_cols,
        args.bootstrap,
        args.seed + 4000,
    )

    pd.DataFrame([
        {
            "level": "transformed_feature",
            **feat_stab,
        }
    ]).to_csv(
        outdir / "SHAP_Feature_Stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_matrix = (
        grouped.pivot(
            index="PatientID",
            columns="group",
            values="abs_group_shap_additive",
        )[
            unique_groups
        ]
    )

    group_stab = ranking_stability(
        group_matrix.to_numpy(),
        unique_groups,
        args.bootstrap,
        args.seed + 5000,
    )

    pd.DataFrame([
        {
            "level": "clinical_group_additive",
            **group_stab,
        }
    ]).to_csv(
        outdir / "SHAP_Group_Stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # SHAP / clinical modality ablation relationship
    # --------------------------------------------------------------

    patient_total_abs = (
        grouped.groupby(
            "PatientID",
            as_index=False,
        )[
            "abs_group_shap_additive"
        ].sum()
    ).rename(
        columns={
            "abs_group_shap_additive":
                "total_abs_grouped_clinical_shap"
        }
    )

    modality = modality.merge(
        patient_total_abs,
        on="PatientID",
        how="left",
        validate="one_to_one",
    )

    rho, rho_lo, rho_hi = bootstrap_spearman(
        modality["abs_delta_clinical"],
        modality["total_abs_grouped_clinical_shap"],
        args.bootstrap,
        args.seed + 6000,
    )

    modality.to_csv(
        outdir / "SHAP_Clinical_Modality_Effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.0, 4.8)
    )

    plot_df = group_add.sort_values(
        "mean_abs_group_shap",
        ascending=True,
    )

    ax.barh(
        plot_df["group"],
        plot_df["mean_abs_group_shap"],
    )

    ax.set_xlabel(
        "Mean |group SHAP|"
    )
    ax.set_ylabel(
        "Clinical feature group"
    )
    ax.set_title(
        "Clinical SHAP importance - proposed gated residual model"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "SHAP_Grouped_Importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "SHAP_Grouped_Importance.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------------

    mean_delta, delta_lo, delta_hi = bootstrap_mean_ci(
        modality["delta_clinical"],
        args.bootstrap,
        args.seed + 7000,
    )

    mean_abs_delta, abs_delta_lo, abs_delta_hi = bootstrap_mean_ci(
        modality["abs_delta_clinical"],
        args.bootstrap,
        args.seed + 8000,
    )

    inc_pct = 100.0 * float(
        np.mean(
            modality["delta_clinical"]
            > 0
        )
    )

    dec_pct = 100.0 * float(
        np.mean(
            modality["delta_clinical"]
            < 0
        )
    )

    abs_res = np.abs(
        np.asarray(
            reconstruction_residuals,
            dtype=float,
        )
    )

    lines = [
        "LUNG1 - TNM-CORRECTED quantitative clinical SHAP for proposed gated residual model",
        "",
        f"Test patients explained: {len(test)}",
        f"Transformed clinical features: {len(feature_cols)}",
        "TNM-corrected feature audit: PASS (T5/N4/M3 sentinel categories absent)",
        f"Training-background size: {bg_n}",
        f"Execution device: {device.type}",
        "SHAP method: GradientExplainer / Expected Gradients",
        f"Expected-gradient samples per patient: {args.nsamples}",
        f"Bootstrap replicates: {args.bootstrap}",
        "",
        "FEATURE GROUP MAPPING",
    ]

    for g in dict.fromkeys(groups):
        feats = [
            c for c, gg in zip(
                feature_cols,
                groups,
            )
            if gg == g
        ]
        lines.append(
            f"  {g}: {', '.join(feats)}"
        )

    lines += [
        "",
        "TOP TRANSFORMED FEATURES BY MEAN |SHAP|",
    ]

    for _, r in feature_summary.head(10).iterrows():
        lines.append(
            f"  #{int(r['rank'])} "
            f"{r['feature']} [{r['group']}]: "
            f"mean|SHAP|={r['mean_abs_shap']:.6f}, "
            f"95% CI [{r['CI95_lower']:.6f}, "
            f"{r['CI95_upper']:.6f}]"
        )

    lines += [
        "",
        "GROUPED CLINICAL IMPORTANCE - ADDITIVE GROUP SHAP",
    ]

    for _, r in group_add.iterrows():
        lines.append(
            f"  #{int(r['rank'])} {r['group']}: "
            f"mean|group SHAP|={r['mean_abs_group_shap']:.6f}, "
            f"95% CI [{r['CI95_lower']:.6f}, "
            f"{r['CI95_upper']:.6f}]"
        )

    lines += [
        "",
        "EXACT CLINICAL-MODALITY ABLATION EFFECT",
        "  delta_clinical = final multimodal risk - no-clinical CT-fallback risk",
        (
            f"  Mean delta = {mean_delta:+.6f} "
            f"(95% CI [{delta_lo:+.6f}, {delta_hi:+.6f}])"
        ),
        (
            f"  Mean |delta| = {mean_abs_delta:.6f} "
            f"(95% CI [{abs_delta_lo:.6f}, {abs_delta_hi:.6f}])"
        ),
        (
            f"  Clinical increases risk in {inc_pct:.1f}% "
            f"and decreases it in {dec_pct:.1f}% of test patients."
        ),
        "",
        "SHAP / MODALITY-EFFECT CONSISTENCY",
        (
            "  Spearman correlation between |clinical ablation effect| "
            "and total |grouped clinical SHAP| = "
            f"{rho:.4f} "
            f"(95% bootstrap CI [{rho_lo:.4f}, {rho_hi:.4f}])"
        ),
        (
            "  Mean absolute Expected-Gradients reconstruction residual = "
            f"{float(np.mean(abs_res)):.6f}"
        ),
        (
            "  Max absolute Expected-Gradients reconstruction residual = "
            f"{float(np.max(abs_res)):.6f}"
        ),
        "",
        "BOOTSTRAP RANK STABILITY",
        (
            "  Feature-ranking Spearman mean = "
            f"{feat_stab['mean_spearman']:.4f} "
            f"(95% interval "
            f"[{feat_stab['CI95_lower']:.4f}, "
            f"{feat_stab['CI95_upper']:.4f}])"
        ),
        (
            "  Group-ranking Spearman mean = "
            f"{group_stab['mean_spearman']:.4f} "
            f"(95% interval "
            f"[{group_stab['CI95_lower']:.4f}, "
            f"{group_stab['CI95_upper']:.4f}])"
        ),
        "",
        "INTERPRETATION NOTES",
        (
            "  SHAP values explain clinical-feature effects conditional on "
            "each patient's fixed CT context."
        ),
        (
            "  SHAP values are relative to the TRAINING clinical-background "
            "distribution; they are not the same quantity as the exact "
            "clinical-modality ablation effect."
        ),
        (
            "  Bootstrap rank stability measures cohort-level ranking "
            "stability under patient resampling; it is not a measure of "
            "causal importance."
        ),
        (
            "  This is post-hoc evaluation on the locked test set. "
            "No model is refit or tuned."
        ),
        (
            "  CT contexts and patient-level SHAP results are cached only to "
            "allow safe CPU interruption/resumption; caching does not change "
            "the model or explanation method."
        ),
    ]

    text = "\n".join(lines)

    (
        outdir / "SHAP_Summary.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 80
        + "\n"
        + text
        + "\n"
        + "=" * 80
    )

    print(
        "Files written to:",
        outdir,
    )


if __name__ == "__main__":
    main()
