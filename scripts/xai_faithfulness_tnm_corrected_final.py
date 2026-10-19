#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LUNG1 TNM-CORRECTED proposed gated residual model - corrected 3D XAI faithfulness by deletion/insertion
=====================================================================

This corrected version fixes two issues in the previous experiment:

1) The perturbation reference now EXACTLY matches the volume reached at
   100% deletion / 0% insertion:
       reference_ct = original_ct
       reference_ct[valid_domain] = gaussian_blurred_ct[valid_domain]

   Therefore:
       deletion at 100% == reference_ct
       insertion at 0%  == reference_ct
       insertion at 100% == original_ct

2) PRIMARY faithfulness metrics no longer divide by a potentially tiny
   risk gap.

PRIMARY metrics
---------------
Deletion:
    AUC of |risk_deleted(f) - risk_original|
    Higher = stronger/faster effect.

Insertion:
    AUC of |risk_inserted(f) - risk_original|
    Lower = faster recovery to the original prediction, hence stronger
    faithfulness.

SECONDARY normalized metrics
----------------------------
Let gap = |risk_reference - risk_original|.

Deletion normalized effect:
    |risk_deleted - risk_original| / gap

Insertion normalized recovery:
    1 - |risk_inserted - risk_original| / gap

These are computed only when gap >= --min-gap. Their endpoint identities
are audited explicitly.

ZERO-CAM handling
-----------------
If a Grad-CAM or Grad-CAM++ map is completely zero, that patient/method/mode
is marked invalid for saliency-based ranking. No artificial voxel ranking is
created. Random control remains valid.

Strategies
----------
- Grad-CAM
- Grad-CAM++
- Random block ranking, repeated N times

Fractions
---------
0%, 5%, 10%, 20%, 30%, 50%, 100%

Recommended smoke test
----------------------
python xai_faithfulness_corrected.py \
  --vol-dir "/content/LUNG1_CT_Only/volumes" \
  --split "/content/LUNG1_CT_Only/LUNG1_Split_Assignment.csv" \
  --clinical-dir "/content/LUNG1_CT_Only/clinical_baselines/model_A" \
  --checkpoint "/content/LUNG1_CT_Only/gated_moddrop_v2/best_model.pt" \
  --cam-dir "/content/LUNG1_CT_Only/xai_gradcam_vs_gradcampp/lowres_maps" \
  --out "/content/LUNG1_CT_Only/xai_faithfulness_corrected_smoke" \
  --modes full_clinical \
  --random-repeats 2 \
  --bootstrap 200 \
  --max-patients 3 \
  --seed 20260926

Recommended full run
--------------------
python xai_faithfulness_corrected.py \
  --vol-dir "/content/LUNG1_CT_Only/volumes" \
  --split "/content/LUNG1_CT_Only/LUNG1_Split_Assignment.csv" \
  --clinical-dir "/content/LUNG1_CT_Only/clinical_baselines/model_A" \
  --checkpoint "/content/LUNG1_CT_Only/gated_moddrop_v2/best_model.pt" \
  --cam-dir "/content/LUNG1_CT_Only/xai_gradcam_vs_gradcampp/lowres_maps" \
  --out "/content/LUNG1_CT_Only/xai_faithfulness_corrected" \
  --modes full_clinical \
  --random-repeats 5 \
  --bootstrap 5000 \
  --blur-sigma 5.0 \
  --min-gap 0.01 \
  --seed 20260926
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F

from lifelines.utils import concordance_index


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Locked Robust V2 architecture
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

    def forward(self, ct, clinical, availability):
        fmap = self.ct_encoder(ct)
        flat = torch.flatten(fmap, 1)

        base_risk = self.ct_risk_head(fmap).squeeze(1)
        zct = self.ct_feature_proj(flat)
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


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def validate_tnm_corrected_features(feature_cols):
    cols = list(feature_cols)

    forbidden = {
        "clinical.T.Stage_5",
        "clinical.T.Stage_5.0",
        "Clinical.N.Stage_4",
        "Clinical.N.Stage_4.0",
        "Clinical.M.Stage_3",
        "Clinical.M.Stage_3.0",
    }
    bad = sorted(forbidden.intersection(cols))
    if bad:
        raise RuntimeError(
            "INVALID pre-correction TNM categories detected: "
            + ", ".join(bad)
        )

    expected = {
        "clinical.T.Stage_MISSING",
        "Clinical.N.Stage_MISSING",
        "Clinical.M.Stage_MISSING",
    }
    missing = sorted(expected.difference(cols))
    if missing:
        raise RuntimeError(
            "Expected TNM-corrected MISSING indicators are absent: "
            + ", ".join(missing)
        )

    if len(cols) != 21:
        raise RuntimeError(
            f"Expected 21 corrected transformed clinical features, got {len(cols)}."
        )


def load_test_table(split_path: Path, clinical_dir: Path):
    split = pd.read_csv(split_path)
    test = split[
        split["Split"].astype(str).str.lower() == "test"
    ].copy()

    clinical = pd.read_csv(
        clinical_dir / "test_features.csv"
    )

    feature_cols = [
        c for c in clinical.columns
        if c not in [
            "PatientID",
            "Survival.time",
            "deadstatus.event",
        ]
    ]

    validate_tnm_corrected_features(feature_cols)

    base = test[
        ["PatientID", "Survival.time", "deadstatus.event"]
    ].copy()

    df = base.merge(
        clinical[["PatientID"] + feature_cols],
        on="PatientID",
        how="left",
        validate="one_to_one",
    )

    if df[feature_cols].isna().any().any():
        raise RuntimeError(
            "Missing transformed clinical features."
        )

    return df.reset_index(drop=True), feature_cols


def load_ct(vol_dir: Path, patient_id: str):
    path = vol_dir / f"{patient_id}_standardized_192x160x224.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=False)

    if "ct_hu_clipped" not in z.files:
        raise KeyError(f"{path}: missing ct_hu_clipped")

    return np.asarray(
        z["ct_hu_clipped"],
        dtype=np.float32,
    )


def load_lowres_cams(
    cam_dir: Path,
    patient_id: str,
    mode: str,
):
    path = cam_dir / f"{patient_id}_{mode}_cams_lowres.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=False)

    return {
        "Grad-CAM": np.asarray(
            z["gradcam"],
            dtype=np.float32,
        ),
        "Grad-CAM++": np.asarray(
            z["gradcampp"],
            dtype=np.float32,
        ),
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_risk(
    model,
    ct_np,
    clinical_np,
    availability_value,
    device,
):
    ct = torch.from_numpy(
        ct_np[None, None].astype(np.float32)
    ).to(device)

    clinical = torch.from_numpy(
        clinical_np[None].astype(np.float32)
    ).to(device)

    availability = torch.tensor(
        [[availability_value]],
        dtype=torch.float32,
        device=device,
    )

    risk, base, gate, residual = model(
        ct,
        clinical,
        availability,
    )

    return {
        "risk": float(risk.cpu().item()),
        "base_risk": float(base.cpu().item()),
        "gate": float(gate.cpu().item()),
        "residual": float(residual.cpu().item()),
    }


# ---------------------------------------------------------------------
# CAM-cell perturbation helpers
# ---------------------------------------------------------------------

def valid_cam_cells(
    valid_domain_full,
    cam_shape,
    device,
):
    x = torch.from_numpy(
        valid_domain_full.astype(np.float32)[None, None]
    ).to(device)

    pooled = F.adaptive_max_pool3d(
        x,
        output_size=cam_shape,
    )

    return (
        pooled[0, 0].cpu().numpy()
        > 0.5
    )


def lowres_selection_to_full_mask(
    selected_lowres,
    full_shape,
    valid_domain_full,
    device,
):
    x = torch.from_numpy(
        selected_lowres.astype(np.float32)[None, None]
    ).to(device)

    up = F.interpolate(
        x,
        size=full_shape,
        mode="nearest",
    )

    mask = (
        up[0, 0].cpu().numpy()
        > 0.5
    )

    return mask & valid_domain_full


def map_is_zero(cam_low):
    cam_low = np.asarray(
        cam_low,
        dtype=float,
    )

    return bool(
        np.max(np.abs(cam_low))
        <= 1e-12
    )


def ranking_from_cam(
    cam_low,
    valid_low,
):
    idx = np.flatnonzero(
        valid_low.reshape(-1)
    )

    values = cam_low.reshape(-1)[idx]

    order = idx[
        np.argsort(
            values,
            kind="stable",
        )[::-1]
    ]

    return order


def random_ranking(
    valid_low,
    rng,
):
    idx = np.flatnonzero(
        valid_low.reshape(-1)
    ).copy()

    rng.shuffle(idx)

    return idx


def selection_from_ranking(
    ranking,
    cam_shape,
    fraction,
):
    n = len(ranking)

    if fraction <= 0:
        k = 0
    elif fraction >= 1:
        k = n
    else:
        k = max(
            1,
            int(np.ceil(fraction * n)),
        )

    selected = np.zeros(
        int(np.prod(cam_shape)),
        dtype=bool,
    )

    if k > 0:
        selected[ranking[:k]] = True

    return selected.reshape(cam_shape), k


# ---------------------------------------------------------------------
# Curve utilities
# ---------------------------------------------------------------------

def auc_trapz(
    fractions,
    values,
):
    x = np.asarray(
        fractions,
        dtype=float,
    )
    y = np.asarray(
        values,
        dtype=float,
    )

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if ok.sum() < 2:
        return np.nan

    return float(
        np.trapezoid(
            y[ok],
            x[ok],
        )
    )


def cindex_from_frame(df):
    return float(
        concordance_index(
            df["Survival.time"].to_numpy(float),
            -df["risk_perturbed"].to_numpy(float),
            event_observed=df[
                "deadstatus.event"
            ].to_numpy(int),
        )
    )


def paired_bootstrap_delta(
    a,
    b,
    n_boot,
    seed,
):
    """
    Returns delta = B - A.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]

    if len(a) == 0:
        return {
            "N": 0,
            "mean_A": np.nan,
            "mean_B": np.nan,
            "delta_B_minus_A": np.nan,
            "CI95_lower": np.nan,
            "CI95_upper": np.nan,
            "bootstrap_sign_p_two_sided": np.nan,
        }

    rng = np.random.default_rng(seed)
    n = len(a)

    deltas = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )
        deltas[i] = np.mean(
            b[idx] - a[idx]
        )

    left = (
        np.sum(deltas <= 0) + 1
    ) / (
        n_boot + 1
    )

    right = (
        np.sum(deltas >= 0) + 1
    ) / (
        n_boot + 1
    )

    p = min(
        1.0,
        2.0 * min(left, right),
    )

    return {
        "N": n,
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
        "delta_B_minus_A": float(
            np.mean(b - a)
        ),
        "CI95_lower": float(
            np.percentile(deltas, 2.5)
        ),
        "CI95_upper": float(
            np.percentile(deltas, 97.5)
        ),
        "bootstrap_sign_p_two_sided": float(p),
    }


# ---------------------------------------------------------------------
# Perturbation experiment
# ---------------------------------------------------------------------

def evaluate_ranking(
    model,
    original_ct,
    reference_ct,
    clinical_np,
    availability_value,
    ranking,
    cam_shape,
    valid_domain,
    fractions,
    device,
):
    deletion_rows = []
    insertion_rows = []

    for fraction in fractions:
        low_mask, n_selected = selection_from_ranking(
            ranking,
            cam_shape,
            fraction,
        )

        full_mask = lowres_selection_to_full_mask(
            low_mask,
            original_ct.shape,
            valid_domain,
            device,
        )

        # Deletion: original -> reference in selected regions
        deletion_ct = original_ct.copy()
        deletion_ct[full_mask] = reference_ct[full_mask]

        # Insertion: reference -> original in selected regions
        insertion_ct = reference_ct.copy()
        insertion_ct[full_mask] = original_ct[full_mask]

        risk_del = predict_risk(
            model,
            deletion_ct,
            clinical_np,
            availability_value,
            device,
        )["risk"]

        risk_ins = predict_risk(
            model,
            insertion_ct,
            clinical_np,
            availability_value,
            device,
        )["risk"]

        common = {
            "fraction": fraction,
            "selected_cam_cells": n_selected,
            "perturbed_voxels": int(
                full_mask.sum()
            ),
        }

        deletion_rows.append({
            **common,
            "risk_perturbed": risk_del,
        })

        insertion_rows.append({
            **common,
            "risk_perturbed": risk_ins,
        })

    return deletion_rows, insertion_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clinical-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cam-dir", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "full_clinical",
            "no_clinical",
        ],
        default=["full_clinical"],
    )

    ap.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
            1.0,
        ],
    )

    ap.add_argument(
        "--random-repeats",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )

    ap.add_argument(
        "--blur-sigma",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--min-gap",
        type=float,
        default=0.01,
        help=(
            "Threshold used ONLY for secondary normalized metrics. "
            "Primary raw AUC metrics do not depend on this threshold."
        ),
    )

    ap.add_argument(
        "--endpoint-tol",
        type=float,
        default=1e-5,
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
        default="auto",
        help="Execution device. Use cuda for the final 3D perturbation run.",
    )

    args = ap.parse_args()

    fractions = sorted(
        set(float(x) for x in args.fractions)
    )

    if fractions[0] != 0.0:
        fractions = [0.0] + fractions

    if fractions[-1] != 1.0:
        fractions = fractions + [1.0]

    if any(
        x < 0 or x > 1
        for x in fractions
    ):
        raise ValueError(
            "Fractions must be between 0 and 1."
        )

    seed_all(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table, feature_cols = load_test_table(
        Path(args.split),
        Path(args.clinical_dir),
    )

    if args.max_patients > 0:
        table = table.iloc[
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
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    print("Patients:", len(table))
    print("TNM-corrected clinical feature audit: PASS")
    print("Modes:", args.modes)
    print("Fractions:", fractions)
    print("Random repeats:", args.random_repeats)
    print("Blur sigma:", args.blur_sigma)
    print("Normalized min gap:", args.min_gap)
    print("Device:", device)

    curve_rows = []
    auc_rows = []
    baseline_rows = []
    zero_map_rows = []
    endpoint_rows = []

    for patient_idx, row in table.iterrows():
        pid = str(row["PatientID"])

        print(
            f"[{patient_idx + 1}/{len(table)}] {pid}"
        )

        ct_hu = load_ct(
            Path(args.vol_dir),
            pid,
        )

        original_ct = np.clip(
            (ct_hu + 1000.0) / 1400.0,
            0.0,
            1.0,
        ).astype(np.float32)

        gaussian_ct = gaussian_filter(
            original_ct,
            sigma=args.blur_sigma,
            mode="nearest",
        ).astype(np.float32)

        # Perturb only non-padding / non-exact-air domain.
        valid_domain = ct_hu > -999.0

        if valid_domain.sum() == 0:
            raise RuntimeError(
                f"{pid}: empty perturbation domain"
            )

        # Corrected reference: EXACT endpoint of 100% deletion / 0% insertion.
        reference_ct = original_ct.copy()
        reference_ct[
            valid_domain
        ] = gaussian_ct[
            valid_domain
        ]

        clinical_np = row[
            feature_cols
        ].to_numpy(
            dtype=np.float32
        )

        for mode in args.modes:
            availability_value = (
                1.0
                if mode == "full_clinical"
                else 0.0
            )

            cams = load_lowres_cams(
                Path(args.cam_dir),
                pid,
                mode,
            )

            cam_shape = cams[
                "Grad-CAM"
            ].shape

            if cams[
                "Grad-CAM++"
            ].shape != cam_shape:
                raise RuntimeError(
                    f"{pid}/{mode}: CAM shape mismatch"
                )

            valid_low = valid_cam_cells(
                valid_domain,
                cam_shape,
                device,
            )

            original_pred = predict_risk(
                model,
                original_ct,
                clinical_np,
                availability_value,
                device,
            )

            reference_pred = predict_risk(
                model,
                reference_ct,
                clinical_np,
                availability_value,
                device,
            )

            original_risk = original_pred[
                "risk"
            ]
            reference_risk = reference_pred[
                "risk"
            ]

            gap = abs(
                reference_risk
                - original_risk
            )

            baseline_rows.append({
                "PatientID": pid,
                "mode": mode,
                "Survival.time": float(
                    row["Survival.time"]
                ),
                "deadstatus.event": int(
                    row["deadstatus.event"]
                ),
                "original_risk": original_risk,
                "reference_risk": reference_risk,
                "absolute_reference_gap": gap,
                "normalized_metrics_eligible": int(
                    gap >= args.min_gap
                ),
                "valid_full_voxels": int(
                    valid_domain.sum()
                ),
                "valid_cam_cells": int(
                    valid_low.sum()
                ),
            })

            # ----------------------------------------------------------
            # Saliency methods
            # ----------------------------------------------------------
            for method in [
                "Grad-CAM",
                "Grad-CAM++",
            ]:
                zero_map = map_is_zero(
                    cams[method]
                )

                zero_map_rows.append({
                    "PatientID": pid,
                    "mode": mode,
                    "method": method,
                    "zero_map": int(zero_map),
                })

                if zero_map:
                    auc_rows.append({
                        "PatientID": pid,
                        "mode": mode,
                        "strategy": method,
                        "random_repeat": -1,
                        "valid_saliency_map": 0,
                        "absolute_reference_gap": gap,
                        "normalized_metrics_eligible": int(
                            gap >= args.min_gap
                        ),
                        "deletion_abs_change_auc": np.nan,
                        "insertion_abs_error_auc": np.nan,
                        "deletion_normalized_auc": np.nan,
                        "insertion_recovery_auc": np.nan,
                    })
                    continue

                ranking = ranking_from_cam(
                    cams[method],
                    valid_low,
                )

                deletion, insertion = evaluate_ranking(
                    model=model,
                    original_ct=original_ct,
                    reference_ct=reference_ct,
                    clinical_np=clinical_np,
                    availability_value=availability_value,
                    ranking=ranking,
                    cam_shape=cam_shape,
                    valid_domain=valid_domain,
                    fractions=fractions,
                    device=device,
                )

                del_abs = []
                ins_abs = []
                del_norm = []
                ins_rec = []

                for perturbation, records in [
                    ("deletion", deletion),
                    ("insertion", insertion),
                ]:
                    for rec in records:
                        risk_p = rec[
                            "risk_perturbed"
                        ]

                        abs_error = abs(
                            risk_p - original_risk
                        )

                        if gap >= args.min_gap:
                            if perturbation == "deletion":
                                normalized = (
                                    abs_error / gap
                                )
                            else:
                                normalized = (
                                    1.0
                                    - abs_error / gap
                                )
                        else:
                            normalized = np.nan

                        curve_rows.append({
                            "PatientID": pid,
                            "mode": mode,
                            "strategy": method,
                            "random_repeat": -1,
                            "valid_saliency_map": 1,
                            "perturbation": perturbation,
                            "fraction": rec["fraction"],
                            "selected_cam_cells": rec[
                                "selected_cam_cells"
                            ],
                            "perturbed_voxels": rec[
                                "perturbed_voxels"
                            ],
                            "Survival.time": float(
                                row["Survival.time"]
                            ),
                            "deadstatus.event": int(
                                row["deadstatus.event"]
                            ),
                            "original_risk": original_risk,
                            "reference_risk": reference_risk,
                            "risk_perturbed": risk_p,
                            "absolute_error_from_original":
                                abs_error,
                            "normalized_secondary_metric":
                                normalized,
                        })

                        if perturbation == "deletion":
                            del_abs.append(abs_error)
                            del_norm.append(normalized)
                        else:
                            ins_abs.append(abs_error)
                            ins_rec.append(normalized)

                # Endpoint audit.
                del_0 = deletion[0][
                    "risk_perturbed"
                ]
                del_100 = deletion[-1][
                    "risk_perturbed"
                ]
                ins_0 = insertion[0][
                    "risk_perturbed"
                ]
                ins_100 = insertion[-1][
                    "risk_perturbed"
                ]

                endpoint_rows.append({
                    "PatientID": pid,
                    "mode": mode,
                    "strategy": method,
                    "deletion_0_minus_original":
                        del_0 - original_risk,
                    "deletion_100_minus_reference":
                        del_100 - reference_risk,
                    "insertion_0_minus_reference":
                        ins_0 - reference_risk,
                    "insertion_100_minus_original":
                        ins_100 - original_risk,
                })

                auc_rows.append({
                    "PatientID": pid,
                    "mode": mode,
                    "strategy": method,
                    "random_repeat": -1,
                    "valid_saliency_map": 1,
                    "absolute_reference_gap": gap,
                    "normalized_metrics_eligible": int(
                        gap >= args.min_gap
                    ),
                    "deletion_abs_change_auc": auc_trapz(
                        fractions,
                        del_abs,
                    ),
                    "insertion_abs_error_auc": auc_trapz(
                        fractions,
                        ins_abs,
                    ),
                    "deletion_normalized_auc": auc_trapz(
                        fractions,
                        del_norm,
                    ),
                    "insertion_recovery_auc": auc_trapz(
                        fractions,
                        ins_rec,
                    ),
                })

            # ----------------------------------------------------------
            # Random controls
            # ----------------------------------------------------------
            for rr in range(
                args.random_repeats
            ):
                rng = np.random.default_rng(
                    args.seed
                    + 100000 * patient_idx
                    + 1000 * rr
                    + (
                        0
                        if mode == "full_clinical"
                        else 500
                    )
                )

                ranking = random_ranking(
                    valid_low,
                    rng,
                )

                deletion, insertion = evaluate_ranking(
                    model=model,
                    original_ct=original_ct,
                    reference_ct=reference_ct,
                    clinical_np=clinical_np,
                    availability_value=availability_value,
                    ranking=ranking,
                    cam_shape=cam_shape,
                    valid_domain=valid_domain,
                    fractions=fractions,
                    device=device,
                )

                del_abs = []
                ins_abs = []
                del_norm = []
                ins_rec = []

                for perturbation, records in [
                    ("deletion", deletion),
                    ("insertion", insertion),
                ]:
                    for rec in records:
                        risk_p = rec[
                            "risk_perturbed"
                        ]
                        abs_error = abs(
                            risk_p - original_risk
                        )

                        if gap >= args.min_gap:
                            if perturbation == "deletion":
                                normalized = (
                                    abs_error / gap
                                )
                            else:
                                normalized = (
                                    1.0
                                    - abs_error / gap
                                )
                        else:
                            normalized = np.nan

                        curve_rows.append({
                            "PatientID": pid,
                            "mode": mode,
                            "strategy": "Random",
                            "random_repeat": rr,
                            "valid_saliency_map": 1,
                            "perturbation": perturbation,
                            "fraction": rec["fraction"],
                            "selected_cam_cells": rec[
                                "selected_cam_cells"
                            ],
                            "perturbed_voxels": rec[
                                "perturbed_voxels"
                            ],
                            "Survival.time": float(
                                row["Survival.time"]
                            ),
                            "deadstatus.event": int(
                                row["deadstatus.event"]
                            ),
                            "original_risk": original_risk,
                            "reference_risk": reference_risk,
                            "risk_perturbed": risk_p,
                            "absolute_error_from_original":
                                abs_error,
                            "normalized_secondary_metric":
                                normalized,
                        })

                        if perturbation == "deletion":
                            del_abs.append(abs_error)
                            del_norm.append(normalized)
                        else:
                            ins_abs.append(abs_error)
                            ins_rec.append(normalized)

                endpoint_rows.append({
                    "PatientID": pid,
                    "mode": mode,
                    "strategy": f"Random_{rr}",
                    "deletion_0_minus_original":
                        deletion[0]["risk_perturbed"]
                        - original_risk,
                    "deletion_100_minus_reference":
                        deletion[-1]["risk_perturbed"]
                        - reference_risk,
                    "insertion_0_minus_reference":
                        insertion[0]["risk_perturbed"]
                        - reference_risk,
                    "insertion_100_minus_original":
                        insertion[-1]["risk_perturbed"]
                        - original_risk,
                })

                auc_rows.append({
                    "PatientID": pid,
                    "mode": mode,
                    "strategy": "Random",
                    "random_repeat": rr,
                    "valid_saliency_map": 1,
                    "absolute_reference_gap": gap,
                    "normalized_metrics_eligible": int(
                        gap >= args.min_gap
                    ),
                    "deletion_abs_change_auc": auc_trapz(
                        fractions,
                        del_abs,
                    ),
                    "insertion_abs_error_auc": auc_trapz(
                        fractions,
                        ins_abs,
                    ),
                    "deletion_normalized_auc": auc_trapz(
                        fractions,
                        del_norm,
                    ),
                    "insertion_recovery_auc": auc_trapz(
                        fractions,
                        ins_rec,
                    ),
                })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    curves = pd.DataFrame(curve_rows)
    aucs = pd.DataFrame(auc_rows)
    baselines = pd.DataFrame(baseline_rows)
    zero_maps = pd.DataFrame(zero_map_rows)
    endpoints = pd.DataFrame(endpoint_rows)

    curves.to_csv(
        outdir / "Faithfulness_Corrected_Curves_PerPatient.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aucs.to_csv(
        outdir / "Faithfulness_Corrected_AUC_PerPatient.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baselines.to_csv(
        outdir / "Faithfulness_Corrected_BaselineGap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    zero_maps.to_csv(
        outdir / "Faithfulness_Corrected_ZeroMaps.csv",
        index=False,
        encoding="utf-8-sig",
    )

    endpoints.to_csv(
        outdir / "Faithfulness_Corrected_EndpointAudit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Endpoint QC
    # -----------------------------------------------------------------
    endpoint_cols = [
        "deletion_0_minus_original",
        "deletion_100_minus_reference",
        "insertion_0_minus_reference",
        "insertion_100_minus_original",
    ]

    endpoint_max = {
        c: float(
            np.nanmax(
                np.abs(
                    endpoints[c].to_numpy(float)
                )
            )
        )
        for c in endpoint_cols
    }

    endpoint_pass = all(
        v <= args.endpoint_tol
        for v in endpoint_max.values()
    )

    # -----------------------------------------------------------------
    # Random mean per patient
    # -----------------------------------------------------------------
    random_mean = (
        aucs[
            aucs["strategy"] == "Random"
        ]
        .groupby(
            ["PatientID", "mode"],
            as_index=False,
        )
        .agg(
            deletion_abs_change_auc=(
                "deletion_abs_change_auc",
                "mean",
            ),
            insertion_abs_error_auc=(
                "insertion_abs_error_auc",
                "mean",
            ),
            deletion_normalized_auc=(
                "deletion_normalized_auc",
                "mean",
            ),
            insertion_recovery_auc=(
                "insertion_recovery_auc",
                "mean",
            ),
        )
    )

    random_mean[
        "strategy"
    ] = "RandomMean"

    deterministic = aucs[
        aucs["strategy"].isin(
            ["Grad-CAM", "Grad-CAM++"]
        )
    ][
        [
            "PatientID",
            "mode",
            "strategy",
            "deletion_abs_change_auc",
            "insertion_abs_error_auc",
            "deletion_normalized_auc",
            "insertion_recovery_auc",
        ]
    ].copy()

    comparison_table = pd.concat(
        [
            deterministic,
            random_mean[
                [
                    "PatientID",
                    "mode",
                    "strategy",
                    "deletion_abs_change_auc",
                    "insertion_abs_error_auc",
                    "deletion_normalized_auc",
                    "insertion_recovery_auc",
                ]
            ],
        ],
        ignore_index=True,
    )

    comparison_table.to_csv(
        outdir / "Faithfulness_Corrected_AUC_ComparisonTable.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Paired bootstrap
    # -----------------------------------------------------------------
    comparisons = [
        ("Grad-CAM", "Grad-CAM++"),
        ("RandomMean", "Grad-CAM"),
        ("RandomMean", "Grad-CAM++"),
    ]

    metrics = [
        (
            "deletion_abs_change_auc",
            True,
            "PRIMARY",
        ),
        (
            "insertion_abs_error_auc",
            False,
            "PRIMARY",
        ),
        (
            "deletion_normalized_auc",
            True,
            "SECONDARY",
        ),
        (
            "insertion_recovery_auc",
            True,
            "SECONDARY",
        ),
    ]

    pair_rows = []

    for mode_idx, mode in enumerate(
        args.modes
    ):
        sub = comparison_table[
            comparison_table["mode"] == mode
        ]

        for metric_idx, (
            metric,
            higher_is_better,
            analysis_role,
        ) in enumerate(metrics):
            for comp_idx, (
                model_a,
                model_b,
            ) in enumerate(comparisons):
                a = sub[
                    sub["strategy"] == model_a
                ][
                    ["PatientID", metric]
                ].rename(
                    columns={
                        metric: "value_A"
                    }
                )

                b = sub[
                    sub["strategy"] == model_b
                ][
                    ["PatientID", metric]
                ].rename(
                    columns={
                        metric: "value_B"
                    }
                )

                merged = a.merge(
                    b,
                    on="PatientID",
                    how="inner",
                    validate="one_to_one",
                )

                res = paired_bootstrap_delta(
                    merged["value_A"],
                    merged["value_B"],
                    args.bootstrap,
                    args.seed
                    + 10000 * mode_idx
                    + 1000 * metric_idx
                    + comp_idx,
                )

                pair_rows.append({
                    "mode": mode,
                    "analysis_role": analysis_role,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    **res,
                })

    paired = pd.DataFrame(pair_rows)

    paired.to_csv(
        outdir / "Faithfulness_Corrected_PairedBootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Mean curve summary
    # -----------------------------------------------------------------
    curve_summary = (
        curves.groupby(
            [
                "mode",
                "strategy",
                "perturbation",
                "fraction",
            ],
            as_index=False,
        )
        .agg(
            N=(
                "absolute_error_from_original",
                lambda x: int(
                    np.isfinite(
                        np.asarray(
                            x,
                            dtype=float,
                        )
                    ).sum()
                ),
            ),
            mean_absolute_error_from_original=(
                "absolute_error_from_original",
                "mean",
            ),
            sd_absolute_error_from_original=(
                "absolute_error_from_original",
                "std",
            ),
            mean_secondary_normalized_metric=(
                "normalized_secondary_metric",
                "mean",
            ),
        )
    )

    curve_summary.to_csv(
        outdir / "Faithfulness_Corrected_Curve_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Secondary cohort C-index curves
    # -----------------------------------------------------------------
    cindex_rows = []

    for mode in args.modes:
        for perturbation in [
            "deletion",
            "insertion",
        ]:
            for strategy in [
                "Grad-CAM",
                "Grad-CAM++",
            ]:
                ss = curves[
                    (curves["mode"] == mode)
                    & (
                        curves["perturbation"]
                        == perturbation
                    )
                    & (
                        curves["strategy"]
                        == strategy
                    )
                ]

                for fraction in fractions:
                    ff = ss[
                        ss["fraction"]
                        == fraction
                    ]

                    if len(ff) < 2:
                        c = np.nan
                    else:
                        c = cindex_from_frame(ff)

                    cindex_rows.append({
                        "mode": mode,
                        "perturbation": perturbation,
                        "strategy": strategy,
                        "fraction": fraction,
                        "N": len(ff),
                        "C_index": c,
                    })

            rs = curves[
                (curves["mode"] == mode)
                & (
                    curves["perturbation"]
                    == perturbation
                )
                & (
                    curves["strategy"]
                    == "Random"
                )
            ]

            for fraction in fractions:
                ff = rs[
                    rs["fraction"]
                    == fraction
                ]

                ff = (
                    ff.groupby(
                        [
                            "PatientID",
                            "Survival.time",
                            "deadstatus.event",
                        ],
                        as_index=False,
                    )
                    .agg(
                        risk_perturbed=(
                            "risk_perturbed",
                            "mean",
                        )
                    )
                )

                cindex_rows.append({
                    "mode": mode,
                    "perturbation": perturbation,
                    "strategy": "RandomMean",
                    "fraction": fraction,
                    "N": len(ff),
                    "C_index": (
                        cindex_from_frame(ff)
                        if len(ff) >= 2
                        else np.nan
                    ),
                })

    cindex_curves = pd.DataFrame(
        cindex_rows
    )

    cindex_curves.to_csv(
        outdir / "Faithfulness_Corrected_CIndex_Curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------
    normalized_eligible = int(
        baselines[
            "normalized_metrics_eligible"
        ].sum()
    )

    zero_summary = (
        zero_maps.groupby(
            ["mode", "method"],
            as_index=False,
        )["zero_map"]
        .sum()
    )

    lines = [
        "LUNG1 TNM-CORRECTED proposed gated residual model - CORRECTED deletion/insertion XAI faithfulness",
        "",
        f"Patients evaluated: {len(table)}",
        f"Modes: {', '.join(args.modes)}",
        f"Blur sigma: {args.blur_sigma}",
        (
            "Fractions: "
            + ", ".join(
                f"{int(x*100)}%"
                for x in fractions
            )
        ),
        f"Random repeats: {args.random_repeats}",
        f"Paired bootstrap replicates: {args.bootstrap}",
        "",
        "REFERENCE / ENDPOINT QC",
        (
            "reference_ct = original_ct with Gaussian blur ONLY inside "
            "the actual perturbation domain"
        ),
        f"Endpoint tolerance: {args.endpoint_tol}",
        f"Endpoint QC pass: {endpoint_pass}",
    ]

    for c, v in endpoint_max.items():
        lines.append(
            f"  max |{c}| = {v:.8f}"
        )

    lines += [
        "",
        "ZERO-CAM QC",
    ]

    for _, r in zero_summary.iterrows():
        lines.append(
            f"  {r['mode']} | {r['method']}: "
            f"{int(r['zero_map'])} zero maps"
        )

    lines += [
        "",
        "NORMALIZED SECONDARY METRICS",
        (
            f"Eligible gap threshold = {args.min_gap}; "
            f"eligible mode-patient rows = "
            f"{normalized_eligible}/{len(baselines)}"
        ),
        "",
        "PRIMARY PAIRED AUC COMPARISONS",
        "Deletion absolute-change AUC: HIGHER is better.",
        "Insertion absolute-error AUC: LOWER is better.",
        "Delta = Model B - Model A.",
    ]

    primary = paired[
        paired["analysis_role"]
        == "PRIMARY"
    ]

    for _, r in primary.iterrows():
        direction = (
            "higher better"
            if bool(
                r["higher_is_better"]
            )
            else "lower better"
        )

        lines.append(
            f"{r['mode']} | {r['metric']} ({direction}) | "
            f"{r['Model_B']} vs {r['Model_A']}: "
            f"A={r['mean_A']:.6f}, "
            f"B={r['mean_B']:.6f}, "
            f"Delta={r['delta_B_minus_A']:+.6f}, "
            f"95% CI "
            f"[{r['CI95_lower']:+.6f}, "
            f"{r['CI95_upper']:+.6f}], "
            f"bootstrap sign p="
            f"{r['bootstrap_sign_p_two_sided']:.4f}, "
            f"N={int(r['N'])}"
        )

    lines += [
        "",
        "INTERPRETATION",
        (
            "For deletion, a saliency method supports faithfulness when its "
            "absolute-change AUC is higher than RandomMean and the paired "
            "95% CI for (method - random) excludes 0."
        ),
        (
            "For insertion, a saliency method supports faithfulness when its "
            "absolute-error AUC is lower than RandomMean and the paired "
            "95% CI for (method - random) excludes 0 in the negative direction."
        ),
        (
            "Grad-CAM vs Grad-CAM++ should be interpreted the same way; "
            "do not declare a difference when the paired CI includes 0."
        ),
        (
            "Normalized metrics are secondary only. C-index perturbation "
            "curves are also secondary."
        ),
    ]

    text = "\n".join(lines)

    (
        outdir / "Faithfulness_Corrected_Summary.txt"
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
