#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LUNG1 — FINAL TNM-CORRECTED 3D Grad-CAM / Grad-CAM++ localization
=================================================================

Locked proposed gated residual survival model.
No training, no tuning, no recalibration.

Modes
-----
1) full_clinical : availability = 1
2) no_clinical   : availability = 0

Target
------
Final survival-risk score.

Reference
---------
Tumor mask is used ONLY for post-hoc XAI evaluation, never as model input.

Localization metrics
--------------------
- tumor energy fraction
- enrichment versus uniform tumor-volume fraction
- pointing hit
- saliency/tumor centroid distance (mm; standardized spacing = 1 mm)
- same-volume Dice
- same-volume IoU

Robustness / reproducibility
----------------------------
- hard TNM-corrected feature audit
- optional risk recomputation QC against locked test_risk_modes.csv
- patient/mode cache, so an interrupted Colab run can resume
- explicit --device cpu/cuda/auto

Outputs
-------
GradCAM_vs_GradCAMpp_PerPatient.csv
GradCAM_vs_GradCAMpp_MapSimilarity.csv
GradCAM_vs_GradCAMpp_Summary.csv
GradCAM_vs_GradCAMpp_PairedBootstrap.csv
GradCAM_Risk_Recomputation_QC.csv
GradCAM_vs_GradCAMpp_Summary.txt
lowres_maps/*.npz
patient_cache/*.npz
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


ID = "PatientID"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, 2, 1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RobustFusion(nn.Module):
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
            nn.Linear(129, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, ct, clinical, availability):
        fmap = self.ct_encoder(ct)
        flat = torch.flatten(fmap, 1)

        base = self.ct_risk_head(fmap).squeeze(1)
        zct = self.ct_feature_proj(flat)
        zcl = self.clinical_encoder(clinical * availability)

        gate = self.gate(
            torch.cat([zct, zcl, availability], dim=1)
        ).squeeze(1)

        residual = self.residual_head(
            torch.cat([zct, zcl], dim=1)
        ).squeeze(1)

        risk = base + availability.squeeze(1) * gate * residual
        return risk, base, gate, residual


def validate_tnm_corrected_features(cols):
    cols = list(cols)

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
            "TNM-corrected MISSING indicators absent: "
            + ", ".join(missing)
        )

    if len(cols) != 21:
        raise RuntimeError(
            f"Expected 21 corrected transformed features, got {len(cols)}."
        )


def load_test_table(split_path, clinical_dir):
    sp = pd.read_csv(split_path)
    sp = sp[sp["Split"].astype(str).str.lower() == "test"].copy()

    cf = pd.read_csv(Path(clinical_dir) / "test_features.csv")

    feats = [
        c for c in cf.columns
        if c not in [ID, "Survival.time", "deadstatus.event"]
    ]

    validate_tnm_corrected_features(feats)

    base = sp[[ID, "Survival.time", "deadstatus.event"]].copy()

    df = base.merge(
        cf[[ID] + feats],
        on=ID,
        how="left",
        validate="one_to_one",
    )

    if df[feats].isna().any().any():
        raise RuntimeError("Missing corrected transformed clinical features.")

    return df.reset_index(drop=True), feats


def load_patient(vol_dir, pid):
    fp = Path(vol_dir) / f"{pid}_standardized_192x160x224.npz"
    if not fp.exists():
        raise FileNotFoundError(fp)

    z = np.load(fp, allow_pickle=False)

    if "ct_hu_clipped" not in z.files:
        raise RuntimeError(f"{fp}: ct_hu_clipped missing")
    if "tumor_mask" not in z.files:
        raise RuntimeError(f"{fp}: tumor_mask missing")

    ct = np.asarray(z["ct_hu_clipped"], dtype=np.float32)
    mask = np.asarray(z["tumor_mask"]) > 0

    if ct.shape != mask.shape:
        raise RuntimeError(f"{pid}: CT/mask shape mismatch.")
    if int(mask.sum()) == 0:
        raise RuntimeError(f"{pid}: empty tumor mask.")

    return ct, mask


def normalize_cam(x):
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("Non-finite CAM.")

    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - lo) / (hi - lo)).astype(np.float32)


class DualCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad

    def close(self):
        self.handle.remove()

    def generate(self, ct, clinical, availability):
        self.model.zero_grad(set_to_none=True)

        risk, base, gate, residual = self.model(
            ct, clinical, availability
        )

        risk.sum().backward()

        A = self.activations
        G = self.gradients

        if A is None or G is None:
            raise RuntimeError("CAM hooks failed.")

        # Grad-CAM
        weights = G.mean((2, 3, 4), keepdim=True)
        gc = F.relu((weights * A).sum(1, keepdim=True))

        # Grad-CAM++
        g2 = G.pow(2)
        g3 = G.pow(3)
        asum = A.sum((2, 3, 4), keepdim=True)
        denom = 2.0 * g2 + asum * g3
        alpha = g2 / (denom + 1e-7)
        alpha = torch.where(
            torch.isfinite(alpha),
            alpha,
            torch.zeros_like(alpha),
        )
        weights_pp = (alpha * F.relu(G)).sum(
            (2, 3, 4),
            keepdim=True,
        )
        gcpp = F.relu((weights_pp * A).sum(1, keepdim=True))

        full_size = ct.shape[2:]

        gc_full = F.interpolate(
            gc,
            size=full_size,
            mode="trilinear",
            align_corners=False,
        )
        pp_full = F.interpolate(
            gcpp,
            size=full_size,
            mode="trilinear",
            align_corners=False,
        )

        return {
            "gc_low": normalize_cam(
                gc[0, 0].detach().float().cpu().numpy()
            ),
            "gc_full": normalize_cam(
                gc_full[0, 0].detach().float().cpu().numpy()
            ),
            "pp_low": normalize_cam(
                gcpp[0, 0].detach().float().cpu().numpy()
            ),
            "pp_full": normalize_cam(
                pp_full[0, 0].detach().float().cpu().numpy()
            ),
            "risk": float(risk.detach().cpu().item()),
            "base": float(base.detach().cpu().item()),
            "gate": float(gate.detach().cpu().item()),
            "residual": float(residual.detach().cpu().item()),
        }


def weighted_centroid(cam):
    total = float(cam.sum())
    if total <= 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    zp = cam.sum((1, 2))
    yp = cam.sum((0, 2))
    xp = cam.sum((0, 1))

    return np.asarray([
        np.dot(np.arange(cam.shape[0]), zp) / total,
        np.dot(np.arange(cam.shape[1]), yp) / total,
        np.dot(np.arange(cam.shape[2]), xp) / total,
    ], dtype=float)


def mask_centroid(mask):
    m = mask.astype(np.float64)
    total = float(m.sum())

    zp = m.sum((1, 2))
    yp = m.sum((0, 2))
    xp = m.sum((0, 1))

    return np.asarray([
        np.dot(np.arange(mask.shape[0]), zp) / total,
        np.dot(np.arange(mask.shape[1]), yp) / total,
        np.dot(np.arange(mask.shape[2]), xp) / total,
    ], dtype=float)


def localization_metrics(cam, mask):
    cam = np.asarray(cam, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    tumor_vox = int(mask.sum())
    cam_sum = float(cam.sum())
    zero = int(cam_sum <= 1e-12)

    energy = (
        float(cam[mask].sum() / cam_sum)
        if cam_sum > 1e-12
        else 0.0
    )

    tumor_fraction = float(tumor_vox / mask.size)
    enrichment = (
        float(energy / tumor_fraction)
        if tumor_fraction > 0
        else np.nan
    )

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    pointing = int(bool(mask[peak]))

    cc = weighted_centroid(cam)
    mc = mask_centroid(mask)

    distance = (
        float(np.linalg.norm(cc - mc))
        if np.all(np.isfinite(cc))
        else np.nan
    )

    flat_cam = cam.ravel()
    flat_mask = mask.ravel()
    k = tumor_vox

    if k >= len(flat_cam):
        selected = np.ones(len(flat_cam), dtype=bool)
    else:
        top = np.argpartition(flat_cam, -k)[-k:]
        selected = np.zeros(len(flat_cam), dtype=bool)
        selected[top] = True

    inter = int(np.logical_and(selected, flat_mask).sum())
    selected_n = int(selected.sum())

    dice = (
        2.0 * inter / (selected_n + tumor_vox)
        if (selected_n + tumor_vox) > 0
        else np.nan
    )

    union = selected_n + tumor_vox - inter
    iou = inter / union if union > 0 else np.nan

    return {
        "tumor_energy_fraction": energy,
        "energy_enrichment_vs_uniform": enrichment,
        "pointing_hit": pointing,
        "centroid_distance_mm": distance,
        "same_volume_dice": float(dice),
        "same_volume_iou": float(iou),
        "zero_map": zero,
    }


def cosine(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)

    if den <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / den)


def paired_bootstrap(a, b, n_boot, seed):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]

    rng = np.random.default_rng(seed)
    n = len(a)

    deltas = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(b[idx] - a[idx])

    left = (np.sum(deltas <= 0) + 1) / (n_boot + 1)
    right = (np.sum(deltas >= 0) + 1) / (n_boot + 1)

    return {
        "N": n,
        "gradcam_mean": float(np.mean(a)),
        "gradcampp_mean": float(np.mean(b)),
        "delta_pp_minus_gc": float(np.mean(b - a)),
        "CI95_lower": float(np.percentile(deltas, 2.5)),
        "CI95_upper": float(np.percentile(deltas, 97.5)),
        "sign_p_two_sided": float(min(1.0, 2.0 * min(left, right))),
    }


def load_saved_risk_modes(path):
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    required = {
        ID,
        "risk_full",
        "risk_no_clinical",
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            f"{p} missing columns: {missing}"
        )

    df[ID] = df[ID].astype(str)
    return df.set_index(ID)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clinical-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--risk-modes", default="")
    ap.add_argument("--out", required=True)

    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--max-patients", type=int, default=0)

    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    ap.add_argument(
        "--risk-qc-tol",
        type=float,
        default=1e-5,
    )

    args = ap.parse_args()

    seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    maps = out / "lowres_maps"
    maps.mkdir(parents=True, exist_ok=True)

    cache = out / "patient_cache"
    cache.mkdir(parents=True, exist_ok=True)

    df, feats = load_test_table(
        args.split,
        args.clinical_dir,
    )

    if args.max_patients > 0:
        df = df.iloc[:args.max_patients].copy()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    ckpt_cols = ckpt.get("clinical_feature_cols")
    if ckpt_cols is not None and list(ckpt_cols) != list(feats):
        raise RuntimeError(
            "Clinical feature order differs from corrected checkpoint."
        )

    dropout = float(
        ckpt.get("args", {}).get("dropout", 0.30)
    )

    model = RobustFusion(len(feats), dropout)
    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    saved = load_saved_risk_modes(args.risk_modes)

    print("Patients:", len(df))
    print("Clinical transformed features:", len(feats))
    print("TNM-corrected feature audit: PASS")
    print("Device:", device)

    engine = DualCAM(
        model,
        model.ct_encoder[4].net[3],
    )

    rows = []
    sims = []
    qc_rows = []

    try:
        for i, row in df.iterrows():
            pid = str(row[ID])
            print(f"[{i+1}/{len(df)}] {pid}")

            ct_hu, mask = load_patient(
                args.vol_dir,
                pid,
            )

            ct = np.clip(
                (ct_hu + 1000.0) / 1400.0,
                0.0,
                1.0,
            ).astype(np.float32)

            clinical_np = row[feats].to_numpy(
                dtype=np.float32
            )

            ctt = torch.from_numpy(
                ct[None, None]
            ).to(device)

            clt = torch.from_numpy(
                clinical_np[None]
            ).to(device)

            patient_mode_results = {}

            for mode, availability_value in [
                ("full_clinical", 1.0),
                ("no_clinical", 0.0),
            ]:
                cache_path = cache / f"{pid}_{mode}.npz"

                if cache_path.exists():
                    z = np.load(cache_path, allow_pickle=False)

                    result = {
                        "gc_low": np.asarray(z["gc_low"], np.float32),
                        "pp_low": np.asarray(z["pp_low"], np.float32),
                        "risk": float(z["risk"]),
                        "base": float(z["base"]),
                        "gate": float(z["gate"]),
                        "residual": float(z["residual"]),
                        "gc_metrics": {
                            k: float(z["gc_" + k])
                            for k in [
                                "tumor_energy_fraction",
                                "energy_enrichment_vs_uniform",
                                "pointing_hit",
                                "centroid_distance_mm",
                                "same_volume_dice",
                                "same_volume_iou",
                                "zero_map",
                            ]
                        },
                        "pp_metrics": {
                            k: float(z["pp_" + k])
                            for k in [
                                "tumor_energy_fraction",
                                "energy_enrichment_vs_uniform",
                                "pointing_hit",
                                "centroid_distance_mm",
                                "same_volume_dice",
                                "same_volume_iou",
                                "zero_map",
                            ]
                        },
                    }

                    source = "cache"

                else:
                    availability = torch.tensor(
                        [[availability_value]],
                        dtype=torch.float32,
                        device=device,
                    )

                    r = engine.generate(
                        ctt,
                        clt,
                        availability,
                    )

                    mg = localization_metrics(
                        r["gc_full"],
                        mask,
                    )

                    mp = localization_metrics(
                        r["pp_full"],
                        mask,
                    )

                    result = {
                        "gc_low": r["gc_low"],
                        "pp_low": r["pp_low"],
                        "risk": r["risk"],
                        "base": r["base"],
                        "gate": r["gate"],
                        "residual": r["residual"],
                        "gc_metrics": mg,
                        "pp_metrics": mp,
                    }

                    save_dict = {
                        "gc_low": r["gc_low"],
                        "pp_low": r["pp_low"],
                        "risk": np.asarray(r["risk"]),
                        "base": np.asarray(r["base"]),
                        "gate": np.asarray(r["gate"]),
                        "residual": np.asarray(r["residual"]),
                    }

                    for k, v in mg.items():
                        save_dict["gc_" + k] = np.asarray(v)

                    for k, v in mp.items():
                        save_dict["pp_" + k] = np.asarray(v)

                    np.savez_compressed(
                        cache_path,
                        **save_dict,
                    )

                    source = "computed"

                # Save low-resolution CAMs in format expected by faithfulness.
                np.savez_compressed(
                    maps / f"{pid}_{mode}_cams_lowres.npz",
                    gradcam=result["gc_low"],
                    gradcampp=result["pp_low"],
                )

                for method, metrics in [
                    ("Grad-CAM", result["gc_metrics"]),
                    ("Grad-CAM++", result["pp_metrics"]),
                ]:
                    clean_metrics = {
                        k: (
                            int(v)
                            if k in ["pointing_hit", "zero_map"]
                            else float(v)
                        )
                        for k, v in metrics.items()
                    }

                    rows.append({
                        ID: pid,
                        "mode": mode,
                        "method": method,
                        "Survival.time": float(row["Survival.time"]),
                        "deadstatus.event": int(row["deadstatus.event"]),
                        "risk_final": result["risk"],
                        "risk_ct_base": result["base"],
                        "gate_value": result["gate"],
                        "clinical_residual": result["residual"],
                        "cache_source": source,
                        **clean_metrics,
                    })

                patient_mode_results[mode] = result

                if saved is not None:
                    if pid not in saved.index:
                        raise RuntimeError(
                            f"{pid} absent from saved test risk modes."
                        )

                    expected = (
                        float(saved.loc[pid, "risk_full"])
                        if mode == "full_clinical"
                        else float(saved.loc[pid, "risk_no_clinical"])
                    )

                    qc_rows.append({
                        ID: pid,
                        "mode": mode,
                        "risk_recomputed": result["risk"],
                        "risk_saved": expected,
                        "abs_diff": abs(result["risk"] - expected),
                    })

            # Similarity between full and no-clinical maps.
            full = patient_mode_results["full_clinical"]
            none = patient_mode_results["no_clinical"]

            sims.append({
                ID: pid,
                "GradCAM_full_vs_no_cosine":
                    cosine(full["gc_low"], none["gc_low"]),
                "GradCAMpp_full_vs_no_cosine":
                    cosine(full["pp_low"], none["pp_low"]),
                "GradCAM_vs_GradCAMpp_full_cosine":
                    cosine(full["gc_low"], full["pp_low"]),
                "GradCAM_vs_GradCAMpp_no_cosine":
                    cosine(none["gc_low"], none["pp_low"]),
            })

            del ctt, clt
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        engine.close()

    per_patient = pd.DataFrame(rows)
    similarity = pd.DataFrame(sims)

    per_patient.to_csv(
        out / "GradCAM_vs_GradCAMpp_PerPatient.csv",
        index=False,
        encoding="utf-8-sig",
    )

    similarity.to_csv(
        out / "GradCAM_vs_GradCAMpp_MapSimilarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if qc_rows:
        qc = pd.DataFrame(qc_rows)
        qc.to_csv(
            out / "GradCAM_Risk_Recomputation_QC.csv",
            index=False,
            encoding="utf-8-sig",
        )

        max_diff = float(qc["abs_diff"].max())
        print(f"Risk QC max |delta| = {max_diff:.3e}")

        if max_diff > args.risk_qc_tol:
            raise RuntimeError(
                f"Risk QC failed: {max_diff} > {args.risk_qc_tol}"
            )

    metrics = [
        "tumor_energy_fraction",
        "energy_enrichment_vs_uniform",
        "pointing_hit",
        "centroid_distance_mm",
        "same_volume_dice",
        "same_volume_iou",
    ]

    higher = {
        "tumor_energy_fraction": True,
        "energy_enrichment_vs_uniform": True,
        "pointing_hit": True,
        "centroid_distance_mm": False,
        "same_volume_dice": True,
        "same_volume_iou": True,
    }

    summary_rows = []
    paired_rows = []

    for mode in ["full_clinical", "no_clinical"]:
        for method in ["Grad-CAM", "Grad-CAM++"]:
            g = per_patient[
                (per_patient["mode"] == mode)
                & (per_patient["method"] == method)
            ]

            for metric in metrics:
                v = pd.to_numeric(
                    g[metric],
                    errors="coerce",
                ).to_numpy(float)
                v = v[np.isfinite(v)]

                summary_rows.append({
                    "mode": mode,
                    "method": method,
                    "metric": metric,
                    "N": len(v),
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "sd": (
                        float(np.std(v, ddof=1))
                        if len(v) > 1
                        else 0.0
                    ),
                })

            summary_rows.append({
                "mode": mode,
                "method": method,
                "metric": "zero_map_count",
                "N": len(g),
                "mean": float(g["zero_map"].sum()),
                "median": np.nan,
                "sd": np.nan,
            })

        gc = per_patient[
            (per_patient["mode"] == mode)
            & (per_patient["method"] == "Grad-CAM")
        ].sort_values(ID)

        pp = per_patient[
            (per_patient["mode"] == mode)
            & (per_patient["method"] == "Grad-CAM++")
        ].sort_values(ID)

        if gc[ID].tolist() != pp[ID].tolist():
            raise RuntimeError("Grad-CAM/Grad-CAM++ patient alignment failed.")

        for j, metric in enumerate(metrics):
            res = paired_bootstrap(
                gc[metric],
                pp[metric],
                args.bootstrap,
                args.seed + 1000 * (1 if mode == "no_clinical" else 0) + j,
            )

            paired_rows.append({
                "mode": mode,
                "metric": metric,
                "higher_is_better": higher[metric],
                **res,
            })

    summary_df = pd.DataFrame(summary_rows)
    paired_df = pd.DataFrame(paired_rows)

    summary_df.to_csv(
        out / "GradCAM_vs_GradCAMpp_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        out / "GradCAM_vs_GradCAMpp_PairedBootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "LUNG1 — FINAL TNM-CORRECTED Grad-CAM / Grad-CAM++",
        "",
        f"Patients evaluated: {len(df)}",
        f"Device: {device}",
        "TNM-corrected clinical feature audit: PASS",
        "Target: final survival-risk score",
        "Target layer: last Conv3D of final CT encoder block",
        f"Paired bootstrap replicates: {args.bootstrap}",
        "",
        "PAIRED LOCALIZATION COMPARISON",
        "Delta = Grad-CAM++ minus Grad-CAM",
    ]

    for _, r in paired_df.iterrows():
        direction = (
            "higher better"
            if bool(r["higher_is_better"])
            else "lower better"
        )

        lines.append(
            f"  {r['mode']} | {r['metric']} ({direction}): "
            f"GC={r['gradcam_mean']:.4f}, "
            f"GC++={r['gradcampp_mean']:.4f}, "
            f"delta={r['delta_pp_minus_gc']:+.4f}, "
            f"95% CI [{r['CI95_lower']:+.4f}, {r['CI95_upper']:+.4f}], "
            f"p={r['sign_p_two_sided']:.4f}"
        )

    lines += [
        "",
        "FULL vs NO-CLINICAL SALIENCY STABILITY",
        (
            f"  Grad-CAM cosine mean="
            f"{similarity['GradCAM_full_vs_no_cosine'].mean():.4f}"
        ),
        (
            f"  Grad-CAM++ cosine mean="
            f"{similarity['GradCAMpp_full_vs_no_cosine'].mean():.4f}"
        ),
        "",
        "INTERPRETATION",
        "  Localization overlap is supportive evidence, not proof of faithfulness.",
        "  The survival model may use peri-tumoral or extra-tumoral context.",
        "  The next step is deletion/insertion perturbation faithfulness.",
        "  No model is retrained or tuned in this analysis.",
    ]

    summary_text = "\n".join(lines)

    (out / "GradCAM_vs_GradCAMpp_Summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(summary_text)
    print("=" * 88)
    print("\nOutputs:", out)


if __name__ == "__main__":
    main()
