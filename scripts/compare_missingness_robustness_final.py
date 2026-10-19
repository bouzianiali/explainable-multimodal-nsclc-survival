#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LUNG1 — FINAL comparative robustness analysis to missing clinical modality
=========================================================================

Compares, without retraining:
  1) Standard CT + clinical concatenation fusion
  2) Proposed gated residual fusion

Missing-clinical-modality scenarios:
  0%, 10%, 30%, 50%, 100%

IMPORTANT INTERPRETATION
------------------------
Standard fusion:
    A completely absent clinical modality is represented by an all-zero
    transformed clinical vector. This is a simple zero-vector fallback stress
    test. The standard fusion model was not trained with whole-modality dropout.

Proposed gated residual model:
    Uses the checkpoint's explicit no-clinical path (availability=0), already
    exported as `risk_no_clinical`; the complete-data path is `risk_full`.

No model is refit, retrained, or selected in this script.

UNCERTAINTY
-----------
1) Fixed-cohort Monte Carlo missingness simulation
   - 0% and 100% are deterministic.
   - 10%, 30%, 50% are repeated `--scenario-repeats` times.
   - The 2.5–97.5 percentiles across repeats are *missingness-simulation
     percentile ranges*, NOT confidence intervals.

2) Paired patient-level bootstrap
   - Same patient resample and same missingness mask for both models.
   - For 10/30/50%, missingness is re-simulated inside each bootstrap replicate.
   - Hence those intervals combine test-sample and missingness-assignment
     uncertainty.
   - Paired differences are defined as Proposed - Standard.
   - Benjamini-Hochberg q-values are reported across the 15 global
     scenario × metric comparisons.

METRICS
-------
- Harrell C-index (higher better)
- Mean cumulative/dynamic AUC at 365/730/1095 d (higher better)
- Integrated Brier Score, IBS (lower better)
- Per-horizon dynamic AUC and Brier score

EXPECTED INPUTS
---------------
clinical-dir/
    train_features.csv
    test_features.csv

standard-dir/
    best_model.pt
    train_risk_scores.csv
    test_risk_scores.csv

robust-dir/
    train_risk_modes.csv
    test_risk_modes.csv
    with columns: risk_full, risk_no_clinical

vol-dir/
    LUNG1-xxx_standardized_192x160x224.npz

OUTPUTS
-------
Risk_Mode_QC.csv
Standard_ZeroClinical_Train_Risks.csv
Standard_ZeroClinical_Test_Risks.csv

Missingness_Raw_Repeats.csv
Missingness_Summary.csv
Missingness_Paired_Differences_Simulation.csv
Missingness_Per_Horizon_Raw.csv
Missingness_Per_Horizon_Summary.csv

Paired_Bootstrap_Model_Metrics.csv
Paired_Bootstrap_Model_Differences.csv
Paired_Bootstrap_Per_Horizon_Differences.csv

Figure_Missingness_Cindex.png/.pdf
Figure_Missingness_MeanAUC.png/.pdf
Figure_Missingness_IBS.png/.pdf

Missingness_Robustness_Summary.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score


ID = "PatientID"
TIME = "Survival.time"
EVENT = "deadstatus.event"

STANDARD = "Standard fusion"
ROBUST = "Proposed gated residual"
MODELS = [STANDARD, ROBUST]
FRACTIONS = [0.0, 0.10, 0.30, 0.50, 1.0]


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =====================================================================
# Exact architecture of corrected STANDARD fusion model
# =====================================================================

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, 1)


class StandardFusionModel(nn.Module):
    def __init__(self, clinical_dim: int, dropout: float = 0.30):
        super().__init__()
        self.ct_encoder = CTEncoder()

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(96 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_full_and_zero(self, ct, clinical):
        # Reuse the same CT encoding for both clinical states.
        z_ct = self.ct_encoder(ct)

        z_full = self.clinical_encoder(clinical)
        risk_full = self.fusion(torch.cat([z_ct, z_full], dim=1)).squeeze(1)

        zero_clinical = torch.zeros_like(clinical)
        z_zero = self.clinical_encoder(zero_clinical)
        risk_zero = self.fusion(torch.cat([z_ct, z_zero], dim=1)).squeeze(1)

        return risk_full, risk_zero


# =====================================================================
# Clinical features + CT inference
# =====================================================================

def load_feature_table(clinical_dir: Path, subset: str):
    path = clinical_dir / f"{subset}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = [ID, TIME, EVENT]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"{path} missing required columns: {miss}")

    feature_cols = [c for c in df.columns if c not in required]
    if not feature_cols:
        raise RuntimeError(f"No clinical transformed features in {path}")
    if df[ID].duplicated().any():
        raise RuntimeError(f"Duplicate PatientID in {path}")

    return df, feature_cols


class CTClinicalDataset(Dataset):
    def __init__(self, feature_df, feature_cols, vol_dir):
        self.df = feature_df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.vol_dir = Path(vol_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[ID])

        fp = self.vol_dir / f"{pid}_standardized_192x160x224.npz"
        if not fp.exists():
            raise FileNotFoundError(fp)

        z = np.load(fp, allow_pickle=False)
        if "ct_hu_clipped" not in z.files:
            raise RuntimeError(f"`ct_hu_clipped` missing from {fp}")

        ct = np.asarray(z["ct_hu_clipped"], dtype=np.float32)

        # Exact normalization used in corrected standard fusion training.
        ct = np.clip((ct + 1000.0) / 1400.0, 0.0, 1.0)

        clinical = row[self.feature_cols].to_numpy(dtype=np.float32)

        return {
            "ct": torch.from_numpy(ct[None]).float(),
            "clinical": torch.from_numpy(clinical).float(),
            "patient_id": pid,
            "time": torch.tensor(float(row[TIME]), dtype=torch.float32),
            "event": torch.tensor(int(row[EVENT]), dtype=torch.int64),
        }


def load_standard_checkpoint(checkpoint_path: Path, clinical_dim: int, feature_cols):
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Standard checkpoint has no `model_state_dict`.")

    ckpt_cols = ckpt.get("clinical_feature_cols", None)
    if ckpt_cols is not None and list(ckpt_cols) != list(feature_cols):
        raise RuntimeError(
            "Clinical feature columns in checkpoint differ from corrected feature table."
        )

    dropout = 0.30
    if isinstance(ckpt.get("args", None), dict):
        dropout = float(ckpt["args"].get("dropout", dropout))

    model = StandardFusionModel(clinical_dim=clinical_dim, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt


@torch.no_grad()
def infer_standard_full_and_zero(
    model,
    feature_df,
    feature_cols,
    vol_dir,
    device,
    batch_size,
    num_workers,
):
    ds = CTClinicalDataset(feature_df, feature_cols, vol_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    rows = []

    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)

        risk_full, risk_zero = model.forward_full_and_zero(ct, clinical)

        for pid, t, e, rf, rz in zip(
            batch["patient_id"],
            batch["time"].numpy(),
            batch["event"].numpy(),
            risk_full.detach().cpu().numpy(),
            risk_zero.detach().cpu().numpy(),
        ):
            rows.append({
                ID: str(pid),
                TIME: float(t),
                EVENT: int(e),
                "risk_full_recomputed": float(rf),
                "risk_no_clinical": float(rz),
            })

    return pd.DataFrame(rows)


# =====================================================================
# Saved risks + QC
# =====================================================================

def load_standard_saved(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [ID, TIME, EVENT, "risk_score"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"{path} missing columns: {miss}")
    return df[required].copy()


def load_robust_modes(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [ID, TIME, EVENT, "risk_full", "risk_no_clinical"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"{path} missing columns: {miss}")
    return df[required].copy()


def normalize_and_sort(df):
    x = df.copy()
    x[ID] = x[ID].astype(str)
    x[TIME] = x[TIME].astype(float)
    x[EVENT] = x[EVENT].astype(int)

    if x[ID].duplicated().any():
        dups = x.loc[x[ID].duplicated(), ID].tolist()[:10]
        raise RuntimeError(f"Duplicate PatientID: {dups}")

    return x.sort_values(ID).reset_index(drop=True)


def check_same_patients_outcomes(a, b, label_a, label_b):
    a = normalize_and_sort(a)
    b = normalize_and_sort(b)

    if a[ID].tolist() != b[ID].tolist():
        raise RuntimeError(f"Patient sets/order differ: {label_a} vs {label_b}")
    if not np.allclose(a[TIME], b[TIME], rtol=0, atol=1e-8):
        raise RuntimeError(f"Survival times differ: {label_a} vs {label_b}")
    if not np.array_equal(a[EVENT].to_numpy(), b[EVENT].to_numpy()):
        raise RuntimeError(f"Event indicators differ: {label_a} vs {label_b}")

    return a, b


def build_mode_table(
    standard_saved,
    standard_recomputed,
    robust_modes,
    subset,
    qc_tol,
):
    s = normalize_and_sort(standard_saved)
    z = normalize_and_sort(standard_recomputed)
    r = normalize_and_sort(robust_modes)

    s, z = check_same_patients_outcomes(s, z, "standard saved", "standard recomputed")
    s, r = check_same_patients_outcomes(s, r, "standard saved", "robust")

    max_abs_diff = float(
        np.max(
            np.abs(
                s["risk_score"].to_numpy(float)
                - z["risk_full_recomputed"].to_numpy(float)
            )
        )
    )

    if max_abs_diff > qc_tol:
        raise RuntimeError(
            f"{subset}: recomputed standard full-clinical risks differ from saved "
            f"risks by max |Δ|={max_abs_diff:.6g}, above tolerance={qc_tol}. "
            "Checkpoint / architecture / input mismatch suspected."
        )

    out = s[[ID, TIME, EVENT]].copy()
    out["standard_risk_full"] = s["risk_score"].to_numpy(float)
    out["standard_risk_no_clinical"] = z["risk_no_clinical"].to_numpy(float)
    out["robust_risk_full"] = r["risk_full"].to_numpy(float)
    out["robust_risk_no_clinical"] = r["risk_no_clinical"].to_numpy(float)

    qc = {
        "Subset": subset,
        "N": len(out),
        "Events": int(out[EVENT].sum()),
        "Standard_full_recompute_max_abs_diff": max_abs_diff,
        "QC_tolerance": float(qc_tol),
        "QC_pass": bool(max_abs_diff <= qc_tol),
    }
    return out, qc


# =====================================================================
# Survival metrics
# =====================================================================

def harrell_cindex(time, event, risk):
    return float(
        concordance_index(
            np.asarray(time, dtype=float),
            -np.asarray(risk, dtype=float),
            event_observed=np.asarray(event, dtype=int),
        )
    )


def to_surv_array(time, event):
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.asarray(event, dtype=int).astype(bool)
    y["time"] = np.asarray(time, dtype=float)
    return y


def fit_breslow(time, event, risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

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
        "cum_hazard": np.cumsum(np.asarray(increments, dtype=float)),
        "shift": shift,
    }


def survival_matrix(risk, times, baseline):
    h0 = []
    for t in times:
        pos = np.searchsorted(baseline["event_times"], t, side="right") - 1
        h0.append(baseline["cum_hazard"][pos] if pos >= 0 else 0.0)

    rel = np.exp(np.asarray(risk, float) - baseline["shift"])
    return np.exp(-rel[:, None] * np.asarray(h0, float)[None, :])


def valid_times(train_time, test_time, requested):
    train_time = np.asarray(train_time, float)
    test_time = np.asarray(test_time, float)

    lo = max(float(np.min(train_time)), float(np.min(test_time)))
    hi = min(float(np.max(train_time)), float(np.max(test_time)))

    times = np.asarray(
        [float(t) for t in requested if lo < float(t) < hi],
        dtype=float,
    )

    if len(times) < 2:
        raise RuntimeError("Need at least two valid time horizons for IBS.")
    return times


def compute_metrics(train_base, test_base, train_risk, test_risk, times):
    train_time = train_base[TIME].to_numpy(float)
    train_event = train_base[EVENT].to_numpy(int)
    test_time = test_base[TIME].to_numpy(float)
    test_event = test_base[EVENT].to_numpy(int)

    y_train = to_surv_array(train_time, train_event)
    y_test = to_surv_array(test_time, test_event)

    c = harrell_cindex(test_time, test_event, test_risk)

    aucs, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        np.asarray(test_risk, float),
        times,
    )

    baseline = fit_breslow(train_time, train_event, train_risk)
    surv = survival_matrix(test_risk, times, baseline)

    _, brier = brier_score(y_train, y_test, surv, times)

    ibs = float(
        integrated_brier_score(
            y_train,
            y_test,
            surv,
            times,
        )
    )

    global_metrics = {
        "C_index": float(c),
        "Mean_dynamic_AUC": float(mean_auc),
        "Integrated_Brier_Score": float(ibs),
    }

    horizon = pd.DataFrame({
        "time_days": np.asarray(times, int),
        "AUC_dynamic": np.asarray(aucs, float),
        "Brier_score": np.asarray(brier, float),
    })

    return global_metrics, horizon


# =====================================================================
# Missingness masks
# =====================================================================

def exact_missing_mask(n, fraction, rng):
    n = int(n)
    n_missing = int(round(float(fraction) * n))
    mask = np.zeros(n, dtype=bool)

    if n_missing > 0:
        idx = rng.choice(n, size=n_missing, replace=False)
        mask[idx] = True

    return mask


def compose_risk(mode_table, model, missing_mask):
    if model == STANDARD:
        full = mode_table["standard_risk_full"].to_numpy(float)
        missing = mode_table["standard_risk_no_clinical"].to_numpy(float)
    elif model == ROBUST:
        full = mode_table["robust_risk_full"].to_numpy(float)
        missing = mode_table["robust_risk_no_clinical"].to_numpy(float)
    else:
        raise ValueError(model)

    return np.where(np.asarray(missing_mask, bool), missing, full)


def scenario_masks(n_train, n_test, fraction, seed, repeat):
    # Deterministic endpoints.
    if fraction == 0.0:
        return (
            np.zeros(n_train, dtype=bool),
            np.zeros(n_test, dtype=bool),
        )
    if fraction == 1.0:
        return (
            np.ones(n_train, dtype=bool),
            np.ones(n_test, dtype=bool),
        )

    tag = int(round(fraction * 1000))
    rng_train = np.random.default_rng(seed + 100_000 + tag * 1000 + repeat * 2)
    rng_test = np.random.default_rng(seed + 200_000 + tag * 1000 + repeat * 2 + 1)

    return (
        exact_missing_mask(n_train, fraction, rng_train),
        exact_missing_mask(n_test, fraction, rng_test),
    )


# =====================================================================
# Fixed-cohort Monte Carlo stress test
# =====================================================================

def run_fixed_cohort_simulation(
    train_modes,
    test_modes,
    times,
    repeats,
    seed,
):
    global_rows = []
    horizon_rows = []
    diff_rows = []

    for frac in FRACTIONS:
        pct = int(round(frac * 100))
        nrep = 1 if frac in (0.0, 1.0) else int(repeats)

        print(f"Fixed-cohort missingness {pct}% | repeats={nrep}")

        for rep in range(nrep):
            train_mask, test_mask = scenario_masks(
                len(train_modes),
                len(test_modes),
                frac,
                seed,
                rep,
            )

            metrics = {}
            per_horizon = {}

            for model in MODELS:
                tr = compose_risk(train_modes, model, train_mask)
                te = compose_risk(test_modes, model, test_mask)

                gm, hm = compute_metrics(
                    train_modes,
                    test_modes,
                    tr,
                    te,
                    times,
                )
                metrics[model] = gm
                per_horizon[model] = hm

                global_rows.append({
                    "Model": model,
                    "Missing_percent": pct,
                    "Repeat": rep,
                    "N_train_missing": int(train_mask.sum()),
                    "N_test_missing": int(test_mask.sum()),
                    **gm,
                })

                for _, rr in hm.iterrows():
                    horizon_rows.append({
                        "Model": model,
                        "Missing_percent": pct,
                        "Repeat": rep,
                        "time_days": int(rr["time_days"]),
                        "AUC_dynamic": float(rr["AUC_dynamic"]),
                        "Brier_score": float(rr["Brier_score"]),
                    })

            for metric in [
                "C_index",
                "Mean_dynamic_AUC",
                "Integrated_Brier_Score",
            ]:
                diff_rows.append({
                    "Missing_percent": pct,
                    "Repeat": rep,
                    "Metric": metric,
                    "Delta_Robust_minus_Standard":
                        float(metrics[ROBUST][metric] - metrics[STANDARD][metric]),
                })

    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(horizon_rows),
        pd.DataFrame(diff_rows),
    )


def summarize_simulation(raw):
    rows = []

    zero = raw[raw["Missing_percent"] == 0].set_index("Model")

    for (model, pct), g in raw.groupby(["Model", "Missing_percent"]):
        row = {
            "Model": model,
            "Missing_percent": int(pct),
            "Repeats": int(len(g)),
            "Interval_type": (
                "deterministic endpoint"
                if int(pct) in (0, 100)
                else "missingness-simulation percentile range"
            ),
        }

        for metric in [
            "C_index",
            "Mean_dynamic_AUC",
            "Integrated_Brier_Score",
        ]:
            v = g[metric].to_numpy(float)
            base = float(zero.loc[model, metric])

            row[f"{metric}_mean"] = float(np.mean(v))
            row[f"{metric}_sd"] = (
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            )
            row[f"{metric}_p2_5"] = float(np.percentile(v, 2.5))
            row[f"{metric}_p97_5"] = float(np.percentile(v, 97.5))
            row[f"{metric}_delta_vs_0pct"] = float(np.mean(v) - base)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["Model", "Missing_percent"]
    ).reset_index(drop=True)


def summarize_horizons(raw):
    rows = []

    for (model, pct, t), g in raw.groupby(
        ["Model", "Missing_percent", "time_days"]
    ):
        for metric in ["AUC_dynamic", "Brier_score"]:
            v = g[metric].to_numpy(float)
            rows.append({
                "Model": model,
                "Missing_percent": int(pct),
                "time_days": int(t),
                "Metric": metric,
                "Repeats": int(len(v)),
                "Mean": float(np.mean(v)),
                "SD": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "P2_5": float(np.percentile(v, 2.5)),
                "P97_5": float(np.percentile(v, 97.5)),
                "Interval_type": (
                    "deterministic endpoint"
                    if int(pct) in (0, 100)
                    else "missingness-simulation percentile range"
                ),
            })

    return pd.DataFrame(rows)


def summarize_simulation_differences(diff_raw):
    rows = []

    for (pct, metric), g in diff_raw.groupby(["Missing_percent", "Metric"]):
        v = g["Delta_Robust_minus_Standard"].to_numpy(float)
        rows.append({
            "Missing_percent": int(pct),
            "Metric": metric,
            "Repeats": int(len(v)),
            "Delta_Robust_minus_Standard_mean": float(np.mean(v)),
            "Delta_SD": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "P2_5": float(np.percentile(v, 2.5)),
            "P97_5": float(np.percentile(v, 97.5)),
            "Interval_type": (
                "deterministic endpoint"
                if int(pct) in (0, 100)
                else "paired missingness-simulation percentile range"
            ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# Paired patient bootstrap
# =====================================================================

def two_sided_sign_p(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan

    left = (np.sum(x <= 0) + 1) / (n + 1)
    right = (np.sum(x >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(left, right)))


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)

    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        q[order[i]] = min(1.0, running)

    return q


def paired_bootstrap(
    train_modes,
    test_modes,
    times,
    n_boot,
    seed,
):
    rng = np.random.default_rng(seed)

    model_rows = []
    diff_rows = []
    horizon_diff_rows = []

    n_test = len(test_modes)

    for frac in FRACTIONS:
        pct = int(round(frac * 100))
        print(f"Paired bootstrap missingness {pct}% | requested={n_boot}")

        model_store = {
            m: {
                "C_index": [],
                "Mean_dynamic_AUC": [],
                "Integrated_Brier_Score": [],
            }
            for m in MODELS
        }

        diff_store = {
            "C_index": [],
            "Mean_dynamic_AUC": [],
            "Integrated_Brier_Score": [],
        }

        horizon_store = {
            int(t): {
                "AUC_dynamic": [],
                "Brier_score": [],
            }
            for t in times
        }

        valid = 0

        for _ in range(int(n_boot)):
            # Same missingness realization for both models.
            if frac == 0.0:
                train_mask = np.zeros(len(train_modes), dtype=bool)
                test_mask = np.zeros(len(test_modes), dtype=bool)
            elif frac == 1.0:
                train_mask = np.ones(len(train_modes), dtype=bool)
                test_mask = np.ones(len(test_modes), dtype=bool)
            else:
                train_mask = exact_missing_mask(len(train_modes), frac, rng)
                test_mask = exact_missing_mask(len(test_modes), frac, rng)

            # Same patient-level bootstrap resample for both models.
            idx = rng.integers(0, n_test, size=n_test)
            test_boot = test_modes.iloc[idx].reset_index(drop=True)

            if test_boot[EVENT].nunique() < 2:
                continue

            rep_metrics = {}
            rep_horizon = {}
            failed = False

            for model in MODELS:
                train_risk = compose_risk(
                    train_modes,
                    model,
                    train_mask,
                )

                test_risk_all = compose_risk(
                    test_modes,
                    model,
                    test_mask,
                )
                test_risk = test_risk_all[idx]

                try:
                    gm, hm = compute_metrics(
                        train_modes,
                        test_boot,
                        train_risk,
                        test_risk,
                        times,
                    )
                except Exception:
                    failed = True
                    break

                rep_metrics[model] = gm
                rep_horizon[model] = hm.set_index("time_days")

            if failed:
                continue

            valid += 1

            for model in MODELS:
                for metric in model_store[model]:
                    model_store[model][metric].append(
                        rep_metrics[model][metric]
                    )

            for metric in diff_store:
                diff_store[metric].append(
                    rep_metrics[ROBUST][metric]
                    - rep_metrics[STANDARD][metric]
                )

            for t in times:
                tt = int(t)
                for metric in ["AUC_dynamic", "Brier_score"]:
                    horizon_store[tt][metric].append(
                        float(
                            rep_horizon[ROBUST].loc[tt, metric]
                            - rep_horizon[STANDARD].loc[tt, metric]
                        )
                    )

        if valid < max(500, int(0.75 * n_boot)):
            raise RuntimeError(
                f"Only {valid}/{n_boot} valid bootstrap replicates "
                f"for {pct}% missingness."
            )

        interval_method = (
            "paired patient bootstrap"
            if pct in (0, 100)
            else "paired patient bootstrap + missingness re-simulation"
        )

        for model in MODELS:
            for metric, values in model_store[model].items():
                v = np.asarray(values, float)
                model_rows.append({
                    "Model": model,
                    "Missing_percent": pct,
                    "Metric": metric,
                    "Bootstrap_valid": int(len(v)),
                    "Bootstrap_mean": float(np.mean(v)),
                    "CI95_lower": float(np.percentile(v, 2.5)),
                    "CI95_upper": float(np.percentile(v, 97.5)),
                    "Interval_method": interval_method,
                })

        for metric, values in diff_store.items():
            v = np.asarray(values, float)
            diff_rows.append({
                "Missing_percent": pct,
                "Metric": metric,
                "Delta_definition": "Proposed - Standard",
                "Bootstrap_valid": int(len(v)),
                "Delta_mean": float(np.mean(v)),
                "CI95_lower": float(np.percentile(v, 2.5)),
                "CI95_upper": float(np.percentile(v, 97.5)),
                "Bootstrap_sign_p_two_sided": two_sided_sign_p(v),
                "Interval_method": interval_method,
            })

        for tt, d in horizon_store.items():
            for metric, values in d.items():
                v = np.asarray(values, float)
                horizon_diff_rows.append({
                    "Missing_percent": pct,
                    "time_days": int(tt),
                    "Metric": metric,
                    "Delta_definition": "Proposed - Standard",
                    "Bootstrap_valid": int(len(v)),
                    "Delta_mean": float(np.mean(v)),
                    "CI95_lower": float(np.percentile(v, 2.5)),
                    "CI95_upper": float(np.percentile(v, 97.5)),
                    "Bootstrap_sign_p_two_sided": two_sided_sign_p(v),
                    "Interval_method": interval_method,
                })

    model_df = pd.DataFrame(model_rows)
    diff_df = pd.DataFrame(diff_rows)
    horizon_df = pd.DataFrame(horizon_diff_rows)

    # 5 scenarios × 3 global metrics = 15 comparisons.
    diff_df["BH_q_across_15_global_tests"] = bh_adjust(
        diff_df["Bootstrap_sign_p_two_sided"].to_numpy(float)
    )

    return model_df, diff_df, horizon_df


# =====================================================================
# Figures
# =====================================================================

def plot_metric(summary, metric, ylabel, filename_stub, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for model in MODELS:
        m = summary[summary["Model"] == model].sort_values("Missing_percent")

        x = m["Missing_percent"].to_numpy(float)
        y = m[f"{metric}_mean"].to_numpy(float)
        lo = m[f"{metric}_p2_5"].to_numpy(float)
        hi = m[f"{metric}_p97_5"].to_numpy(float)

        yerr = np.vstack([
            np.maximum(0.0, y - lo),
            np.maximum(0.0, hi - y),
        ])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=model,
        )

    ax.set_xlabel("Patients with missing clinical modality (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 10, 30, 50, 100])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(outdir / f"{filename_stub}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{filename_stub}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Human-readable summary
# =====================================================================

def build_summary(qc, sim_summary, boot_diff, repeats, bootstrap, times):
    lines = [
        "LUNG1 — FINAL COMPARATIVE MISSING-CLINICAL-MODALITY ROBUSTNESS",
        "",
        "Models",
        f"  Standard: zero transformed clinical vector when the whole clinical modality is absent.",
        f"  Proposed: explicit availability=0 no-clinical path from the gated residual model.",
        "",
        "No model was retrained, retuned, or selected in this analysis.",
        "The same patient-level missingness masks were applied to both models within every repeat.",
        "",
        "QC",
    ]

    for _, r in qc.iterrows():
        lines.append(
            f"  {r['Subset']}: N={int(r['N'])}, events={int(r['Events'])}, "
            f"standard full-risk recomputation max |delta|="
            f"{r['Standard_full_recompute_max_abs_diff']:.3e}, "
            f"pass={bool(r['QC_pass'])}"
        )

    lines += [
        "",
        f"Fixed-cohort missingness repeats at 10/30/50%: {repeats}",
        f"Paired patient bootstrap requested per scenario: {bootstrap}",
        f"Horizons: {' | '.join(str(int(t)) for t in times)} days",
        "",
        "FIXED-COHORT / MISSINGNESS-SIMULATION SUMMARY",
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
                f"C={r['C_index_mean']:.4f} "
                f"(delta0={r['C_index_delta_vs_0pct']:+.4f}), "
                f"mean AUC={r['Mean_dynamic_AUC_mean']:.4f} "
                f"(delta0={r['Mean_dynamic_AUC_delta_vs_0pct']:+.4f}), "
                f"IBS={r['Integrated_Brier_Score_mean']:.4f} "
                f"(delta0={r['Integrated_Brier_Score_delta_vs_0pct']:+.4f})"
            )

    lines += [
        "",
        "PAIRED BOOTSTRAP DIFFERENCES",
        "  Delta = Proposed - Standard.",
        "  For C-index/AUC, positive favors Proposed.",
        "  For IBS, negative favors Proposed.",
        "  q-values use Benjamini-Hochberg across the 15 global scenario x metric tests.",
    ]

    for pct in [0, 10, 30, 50, 100]:
        lines.append(f"  Missing {pct}%:")
        x = boot_diff[boot_diff["Missing_percent"] == pct]
        for metric in [
            "C_index",
            "Mean_dynamic_AUC",
            "Integrated_Brier_Score",
        ]:
            r = x[x["Metric"] == metric].iloc[0]
            lines.append(
                f"    {metric}: delta={r['Delta_mean']:+.4f}, "
                f"95% CI [{r['CI95_lower']:+.4f}, {r['CI95_upper']:+.4f}], "
                f"p={r['Bootstrap_sign_p_two_sided']:.4f}, "
                f"BH q={r['BH_q_across_15_global_tests']:.4f}"
            )

    lines += [
        "",
        "INTERPRETATION GUARDRAILS",
        "  - Fixed-cohort 2.5–97.5% ranges at 10/30/50% quantify missingness-assignment variability; they are not confidence intervals.",
        "  - Bootstrap CIs quantify uncertainty of the paired comparison on the locked test cohort.",
        "  - Do not claim superiority when a paired CI includes zero.",
        "  - Standard zero-vector fallback is a stress-test comparator, not a missing-modality-trained baseline.",
        "  - The held-out test set is never used for training, tuning, or model selection.",
    ]

    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--clinical-dir", required=True)
    ap.add_argument("--standard-dir", required=True)
    ap.add_argument("--robust-dir", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--scenario-repeats", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--times", nargs="+", type=int, default=[365, 730, 1095])
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument(
        "--qc-tol",
        type=float,
        default=1e-5,
        help="Maximum allowed |recomputed standard full risk - saved risk|.",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    vol_dir = Path(args.vol_dir)
    clinical_dir = Path(args.clinical_dir)
    standard_dir = Path(args.standard_dir)
    robust_dir = Path(args.robust_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------------

    required_paths = [
        vol_dir,
        clinical_dir / "train_features.csv",
        clinical_dir / "test_features.csv",
        standard_dir / "best_model.pt",
        standard_dir / "train_risk_scores.csv",
        standard_dir / "test_risk_scores.csv",
        robust_dir / "train_risk_modes.csv",
        robust_dir / "test_risk_modes.csv",
    ]

    missing_paths = [p for p in required_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required paths:\n" + "\n".join(str(p) for p in missing_paths)
        )

    train_features, feature_cols_train = load_feature_table(clinical_dir, "train")
    test_features, feature_cols_test = load_feature_table(clinical_dir, "test")

    if feature_cols_train != feature_cols_test:
        raise RuntimeError("Train and test clinical feature columns differ.")

    feature_cols = feature_cols_train
    print("Clinical transformed feature count:", len(feature_cols))

    # ---------------------------------------------------------------
    # Standard model inference: full vs zero-clinical
    # ---------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type != "cuda":
        print(
            "WARNING: no GPU detected. The one-time 3D CT inference can run on CPU "
            "but will be much slower. Bootstrap itself does not need GPU."
        )

    standard_model, standard_ckpt = load_standard_checkpoint(
        standard_dir / "best_model.pt",
        clinical_dim=len(feature_cols),
        feature_cols=feature_cols,
    )
    standard_model = standard_model.to(device)

    print("Recomputing STANDARD full and zero-clinical risks on TRAIN...")
    std_train_recomputed = infer_standard_full_and_zero(
        standard_model,
        train_features,
        feature_cols,
        vol_dir,
        device,
        args.batch_size,
        args.num_workers,
    )

    print("Recomputing STANDARD full and zero-clinical risks on TEST...")
    std_test_recomputed = infer_standard_full_and_zero(
        standard_model,
        test_features,
        feature_cols,
        vol_dir,
        device,
        args.batch_size,
        args.num_workers,
    )

    # GPU no longer needed after this point.
    standard_model = standard_model.cpu()
    del standard_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Load saved full standard and robust full/no-clinical risks
    # ---------------------------------------------------------------

    std_train_saved = load_standard_saved(
        standard_dir / "train_risk_scores.csv"
    )
    std_test_saved = load_standard_saved(
        standard_dir / "test_risk_scores.csv"
    )

    robust_train = load_robust_modes(
        robust_dir / "train_risk_modes.csv"
    )
    robust_test = load_robust_modes(
        robust_dir / "test_risk_modes.csv"
    )

    train_modes, qc_train = build_mode_table(
        std_train_saved,
        std_train_recomputed,
        robust_train,
        "TRAIN",
        args.qc_tol,
    )
    test_modes, qc_test = build_mode_table(
        std_test_saved,
        std_test_recomputed,
        robust_test,
        "TEST",
        args.qc_tol,
    )

    qc_df = pd.DataFrame([qc_train, qc_test])
    qc_df.to_csv(
        outdir / "Risk_Mode_QC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save zero-clinical risk audit tables.
    std_train_recomputed[
        [ID, TIME, EVENT, "risk_no_clinical"]
    ].to_csv(
        outdir / "Standard_ZeroClinical_Train_Risks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    std_test_recomputed[
        [ID, TIME, EVENT, "risk_no_clinical"]
    ].to_csv(
        outdir / "Standard_ZeroClinical_Test_Risks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nQC:")
    print(qc_df.to_string(index=False))

    # Ensure train/test patient outcomes remain sensible.
    print(
        f"\nAligned TRAIN: n={len(train_modes)}, events={int(train_modes[EVENT].sum())}"
    )
    print(
        f"Aligned TEST:  n={len(test_modes)}, events={int(test_modes[EVENT].sum())}"
    )

    times = valid_times(
        train_modes[TIME].to_numpy(float),
        test_modes[TIME].to_numpy(float),
        args.times,
    )
    print("Evaluation horizons:", [int(t) for t in times])

    # ---------------------------------------------------------------
    # Fixed-cohort missingness simulation
    # ---------------------------------------------------------------

    raw, raw_horizon, raw_diff = run_fixed_cohort_simulation(
        train_modes,
        test_modes,
        times,
        repeats=args.scenario_repeats,
        seed=args.seed,
    )

    sim_summary = summarize_simulation(raw)
    horizon_summary = summarize_horizons(raw_horizon)
    sim_diff_summary = summarize_simulation_differences(raw_diff)

    raw.to_csv(
        outdir / "Missingness_Raw_Repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sim_summary.to_csv(
        outdir / "Missingness_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_diff.to_csv(
        outdir / "Missingness_Paired_Differences_Simulation_Raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sim_diff_summary.to_csv(
        outdir / "Missingness_Paired_Differences_Simulation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_horizon.to_csv(
        outdir / "Missingness_Per_Horizon_Raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    horizon_summary.to_csv(
        outdir / "Missingness_Per_Horizon_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # Paired bootstrap
    # ---------------------------------------------------------------

    boot_model, boot_diff, boot_horizon_diff = paired_bootstrap(
        train_modes,
        test_modes,
        times,
        n_boot=args.bootstrap,
        seed=args.seed + 7_777_777,
    )

    boot_model.to_csv(
        outdir / "Paired_Bootstrap_Model_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot_diff.to_csv(
        outdir / "Paired_Bootstrap_Model_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot_horizon_diff.to_csv(
        outdir / "Paired_Bootstrap_Per_Horizon_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------

    plot_metric(
        sim_summary,
        "C_index",
        "Harrell C-index",
        "Figure_Missingness_Cindex",
        outdir,
    )
    plot_metric(
        sim_summary,
        "Mean_dynamic_AUC",
        "Mean cumulative/dynamic AUC",
        "Figure_Missingness_MeanAUC",
        outdir,
    )
    plot_metric(
        sim_summary,
        "Integrated_Brier_Score",
        "Integrated Brier Score",
        "Figure_Missingness_IBS",
        outdir,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = build_summary(
        qc_df,
        sim_summary,
        boot_diff,
        args.scenario_repeats,
        args.bootstrap,
        times,
    )

    (outdir / "Missingness_Robustness_Summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(summary)
    print("=" * 88)
    print("\nFiles written to:", outdir)


if __name__ == "__main__":
    main()
