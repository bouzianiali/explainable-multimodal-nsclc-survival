#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multimodal LUNG1 survival model: CT + clinical intermediate fusion + FULL Cox-Efron objective.

Inputs
------
1) Standardized CT volumes:
   standardized_lung1_192x160x224/volumes/
     LUNG1-001_standardized_192x160x224.npz
     ...

2) Locked split:
   lung1_split/LUNG1_Split_Assignment.csv

3) Train-only clinical preprocessing already produced by Model A:
   clinical_baselines/model_A/train_features.csv
   clinical_baselines/model_A/validation_features.csv
   clinical_baselines/model_A/test_features.csv

Optional:
4) CT-only best checkpoint:
   ct_only_fullcox_efron/best_model.pt

Architecture
------------
CT branch:
  compact 3D CNN encoder -> 96-D vector

Clinical branch:
  transformed clinical vector -> MLP -> 32-D vector

Intermediate fusion:
  concatenate [CT 96-D | clinical 32-D] -> fusion MLP -> Cox risk score

Important methodological choices
--------------------------------
- Same patient-level split as previous baselines.
- Clinical features are EXACTLY those already preprocessed from TRAIN only.
- Survival.time and deadstatus.event are not model inputs.
- Validation C-index only is used for early stopping/model selection.
- Test is evaluated only after the best validation checkpoint is loaded.
- Tumor mask is not used as an input channel.
- CT encoder can be initialized from the corrected CT-only Full Cox-Efron checkpoint.
- The survival objective is the FULL training-cohort Cox partial likelihood with Efron tie handling.
- Micro-batches are used only for 3D-CNN memory; they do NOT define Cox risk sets.

Outputs
-------
<out>/
  best_model.pt
  training_history.csv
  train_risk_scores.csv
  validation_risk_scores.csv
  test_risk_scores.csv
  Multimodal_Metrics.csv
  Multimodal_Time_Dependent_Metrics.csv
  Multimodal_Summary.txt

Install
-------
python -m pip install torch pandas numpy lifelines scikit-survival tqdm

Recommended Colab command
-------------------------
python train_multimodal_ct_clinical_survival.py \
  --vol-dir "standardized_lung1_192x160x224/volumes" \
  --split "lung1_split/LUNG1_Split_Assignment.csv" \
  --clinical-dir "clinical_baselines/model_A" \
  --ct-checkpoint "ct_only_fullcox_efron/best_model.pt" \
  --out "multimodal_fullcox_efron" \
  --epochs 80 \
  --batch-size 2

Micro-batch size controls GPU memory only. The Cox risk set is always the
full training cohort in the corrected two-pass optimization.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from lifelines.utils import concordance_index
from sksurv.metrics import (
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------
# Clinical feature loading
# ---------------------------------------------------------------------

def load_clinical_feature_tables(clinical_dir: Path):
    """
    Loads the already train-fitted/transformed Model A features.

    Each file contains:
      PatientID
      transformed feature columns
      Survival.time
      deadstatus.event
    """
    paths = {
        "train": clinical_dir / "train_features.csv",
        "validation": clinical_dir / "validation_features.csv",
        "test": clinical_dir / "test_features.csv",
    }

    tables = {}
    feature_cols = None

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        exclude = {
            "PatientID",
            "Survival.time",
            "deadstatus.event",
        }
        cols = [c for c in df.columns if c not in exclude]

        if feature_cols is None:
            feature_cols = cols
        elif cols != feature_cols:
            raise RuntimeError(
                f"Clinical transformed feature columns differ in {name}."
            )

        tables[name] = df

    return tables, feature_cols


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class MultimodalLung1Dataset(Dataset):
    def __init__(
        self,
        split_df: pd.DataFrame,
        clinical_df: pd.DataFrame,
        feature_cols,
        vol_dir: Path,
    ):
        self.vol_dir = Path(vol_dir)
        self.feature_cols = list(feature_cols)

        # IMPORTANT:
        # split_df also contains the original raw clinical columns (e.g. age),
        # while clinical_df contains the transformed Model-A columns, one of
        # which is also named "age". Merging the full split table would create
        # age_x/age_y and similar collisions. Keep only identifiers/outcomes
        # from the locked split, then merge the transformed clinical features.
        base_cols = [
            "PatientID",
            "Survival.time",
            "deadstatus.event",
        ]
        if "Split" in split_df.columns:
            base_cols.append("Split")

        split_base = split_df[base_cols].copy()

        merged = split_base.merge(
            clinical_df[["PatientID"] + self.feature_cols],
            on="PatientID",
            how="left",
            validate="one_to_one",
        )

        if merged[self.feature_cols].isna().any().any():
            bad = merged.loc[
                merged[self.feature_cols].isna().any(axis=1),
                "PatientID",
            ].tolist()
            raise RuntimeError(
                f"Missing transformed clinical features for: {bad[:10]}"
            )

        self.df = merged.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["PatientID"])

        f = self.vol_dir / f"{pid}_standardized_192x160x224.npz"
        if not f.exists():
            raise FileNotFoundError(f)

        z = np.load(f, allow_pickle=False)
        ct = np.asarray(
            z["ct_hu_clipped"],
            dtype=np.float32,
        )

        # Same fixed HU normalization as CT-only baseline:
        # [-1000, 400] -> [0,1]
        ct = np.clip(
            (ct + 1000.0) / 1400.0,
            0.0,
            1.0,
        )

        clinical = row[self.feature_cols].to_numpy(
            dtype=np.float32
        )

        return {
            "ct": torch.from_numpy(ct[None]).float(),
            "clinical": torch.from_numpy(clinical).float(),
            "time": torch.tensor(
                float(row["Survival.time"]),
                dtype=torch.float32,
            ),
            "event": torch.tensor(
                float(row["deadstatus.event"]),
                dtype=torch.float32,
            ),
            "patient_id": pid,
        }


# ---------------------------------------------------------------------
# CT branch - same backbone family as CT-only baseline
# ---------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                cin, cout,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                cout, cout,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
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
        return torch.flatten(x, 1)  # [B,96]


class MultimodalSurvivalModel(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        dropout: float = 0.30,
    ):
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

    def forward(self, ct, clinical):
        z_ct = self.ct_encoder(ct)
        z_clin = self.clinical_encoder(clinical)

        z = torch.cat([z_ct, z_clin], dim=1)
        risk = self.fusion(z).squeeze(1)
        return risk


# ---------------------------------------------------------------------
# Optional CT-only checkpoint transfer
# ---------------------------------------------------------------------

def load_ct_only_encoder(
    model: MultimodalSurvivalModel,
    checkpoint_path: Path,
):
    """
    CT-only model used:
      model.encoder = Sequential(...)
      model.head = ...

    Its checkpoint state keys therefore start with:
      encoder.0...
    Our multimodal encoder keys are:
      ct_encoder.encoder.0...

    We map only encoder weights.
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt["model_state_dict"]
    mapped = {}

    for k, v in state.items():
        if k.startswith("encoder."):
            new_key = "ct_encoder." + k
            mapped[new_key] = v

    missing, unexpected = model.load_state_dict(
        mapped,
        strict=False,
    )

    loaded = len(mapped)

    print(
        f"Loaded {loaded} CT encoder tensors "
        f"from {checkpoint_path}"
    )
    print(
        f"Transfer load: missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )


# ---------------------------------------------------------------------
# FULL training-cohort Cox-Efron objective
# ---------------------------------------------------------------------

def cox_efron_loss_full(risk, time, event):
    """
    Negative Cox partial log-likelihood with Efron handling of tied events.

    IMPORTANT:
    risk/time/event must contain the FULL training cohort for the epoch.
    Micro-batches are only a memory strategy for the 3D network.
    """
    risk = risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1)

    observed = event > 0.5
    n_events = int(observed.sum().item())

    if n_events == 0:
        return risk.sum() * 0.0

    event_times = torch.unique(time[observed])
    loglik = risk.new_tensor(0.0)

    for t in event_times:
        deaths = observed & (time == t)
        d = int(deaths.sum().item())
        riskset = time >= t

        r_risk = risk[riskset]
        r_death = risk[deaths]

        # Common shift for numerical stability.
        m = torch.max(r_risk)
        sum_risk = torch.exp(r_risk - m).sum()
        sum_death = torch.exp(r_death - m).sum()

        loglik = loglik + r_death.sum()

        for ell in range(d):
            frac = float(ell) / float(d)
            denom_shifted = sum_risk - frac * sum_death
            denom_shifted = torch.clamp(
                denom_shifted,
                min=torch.finfo(risk.dtype).tiny,
            )
            loglik = loglik - (
                m + torch.log(denom_shifted)
            )

    return -loglik / float(n_events)


def snapshot_bn_state(model):
    state = {}

    for name, module in model.named_modules():
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):
            state[name] = {
                "running_mean": (
                    None
                    if module.running_mean is None
                    else module.running_mean.detach().clone()
                ),
                "running_var": (
                    None
                    if module.running_var is None
                    else module.running_var.detach().clone()
                ),
                "num_batches_tracked": (
                    None
                    if module.num_batches_tracked is None
                    else module.num_batches_tracked.detach().clone()
                ),
            }

    return state


def restore_bn_state(model, state):
    modules = dict(model.named_modules())

    for name, values in state.items():
        module = modules[name]

        if values["running_mean"] is not None:
            module.running_mean.copy_(
                values["running_mean"]
            )

        if values["running_var"] is not None:
            module.running_var.copy_(
                values["running_var"]
            )

        if values["num_batches_tracked"] is not None:
            module.num_batches_tracked.copy_(
                values["num_batches_tracked"]
            )


def capture_rng_state(device):
    state = {
        "cpu": torch.get_rng_state().clone(),
        "cuda": None,
    }

    if device.type == "cuda":
        state["cuda"] = (
            torch.cuda.get_rng_state(device).clone()
        )

    return state


def restore_rng_state(state, device):
    torch.set_rng_state(
        state["cpu"]
    )

    if (
        device.type == "cuda"
        and state["cuda"] is not None
    ):
        torch.cuda.set_rng_state(
            state["cuda"],
            device,
        )


# ---------------------------------------------------------------------
# Prediction / C-index
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_risk(model, loader, device):
    model.eval()

    rows = []

    for batch in loader:
        ct = batch["ct"].to(
            device,
            non_blocking=True,
        )
        clinical = batch["clinical"].to(
            device,
            non_blocking=True,
        )

        risk = model(
            ct,
            clinical,
        ).detach().cpu().numpy()

        for pid, t, e, r in zip(
            batch["patient_id"],
            batch["time"].numpy(),
            batch["event"].numpy(),
            risk,
        ):
            rows.append(
                (
                    pid,
                    float(t),
                    int(e),
                    float(r),
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "PatientID",
            "Survival.time",
            "deadstatus.event",
            "risk_score",
        ],
    )


def harrell_cindex(df):
    return float(
        concordance_index(
            df["Survival.time"].to_numpy(float),
            -df["risk_score"].to_numpy(float),
            event_observed=df[
                "deadstatus.event"
            ].to_numpy(int),
        )
    )


def bootstrap_cindex(
    df,
    n_boot=1000,
    seed=20260926,
):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(df)

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        b = df.iloc[idx]

        if b["deadstatus.event"].nunique() < 2:
            continue

        try:
            c = harrell_cindex(b)
            if np.isfinite(c):
                vals.append(c)
        except Exception:
            continue

    vals = np.asarray(
        vals,
        dtype=float,
    )

    if len(vals) < 100:
        raise RuntimeError(
            f"Too few valid bootstrap replicates: {len(vals)}"
        )

    return {
        "lower": float(
            np.percentile(vals, 2.5)
        ),
        "upper": float(
            np.percentile(vals, 97.5)
        ),
        "n": int(len(vals)),
    }


# ---------------------------------------------------------------------
# Time-dependent survival metrics
# ---------------------------------------------------------------------

def to_surv_array(df):
    y = np.empty(
        len(df),
        dtype=[
            ("event", "?"),
            ("time", "<f8"),
        ],
    )

    y["event"] = df[
        "deadstatus.event"
    ].astype(bool).to_numpy()

    y["time"] = df[
        "Survival.time"
    ].astype(float).to_numpy()

    return y


def fit_breslow_baseline(train_pred):
    time = train_pred[
        "Survival.time"
    ].to_numpy(float)

    event = train_pred[
        "deadstatus.event"
    ].to_numpy(int)

    risk_raw = train_pred[
        "risk_score"
    ].to_numpy(float)

    shift = float(np.max(risk_raw))
    risk = risk_raw - shift
    exp_risk = np.exp(risk)

    event_times = np.sort(
        np.unique(time[event == 1])
    )

    increments = []

    for et in event_times:
        d = np.sum(
            (time == et)
            & (event == 1)
        )

        risk_sum = np.sum(
            exp_risk[time >= et]
        )

        increments.append(
            d / risk_sum
            if risk_sum > 0
            else 0.0
        )

    return {
        "event_times": event_times,
        "cum_hazard": np.cumsum(
            np.asarray(
                increments,
                dtype=float,
            )
        ),
        "shift": shift,
    }


def survival_probability_matrix(
    pred_df,
    times,
    baseline,
):
    h0 = []

    for t in times:
        pos = np.searchsorted(
            baseline["event_times"],
            t,
            side="right",
        ) - 1

        if pos >= 0:
            h0.append(
                baseline["cum_hazard"][pos]
            )
        else:
            h0.append(0.0)

    h0 = np.asarray(
        h0,
        dtype=float,
    )

    risk = pred_df[
        "risk_score"
    ].to_numpy(float)

    relative_hazard = np.exp(
        risk - baseline["shift"]
    )

    return np.exp(
        -relative_hazard[:, None]
        * h0[None, :]
    )


def valid_eval_times(
    requested,
    train_df,
    eval_df,
):
    requested = np.asarray(
        requested,
        dtype=float,
    )

    upper = min(
        train_df["Survival.time"].max(),
        eval_df["Survival.time"].max(),
    )

    lower = max(
        train_df["Survival.time"].min(),
        eval_df["Survival.time"].min(),
    )

    valid = requested[
        (requested > lower)
        & (requested < upper)
    ]

    if len(valid) == 0:
        raise RuntimeError(
            "No requested evaluation time is usable."
        )

    return valid


def time_dependent_metrics(
    train_pred,
    eval_pred,
    baseline,
    requested_times,
):
    y_train = to_surv_array(
        train_pred
    )
    y_eval = to_surv_array(
        eval_pred
    )

    times = valid_eval_times(
        requested_times,
        train_pred,
        eval_pred,
    )

    risk = eval_pred[
        "risk_score"
    ].to_numpy(float)

    auc_values, mean_auc = (
        cumulative_dynamic_auc(
            y_train,
            y_eval,
            risk,
            times,
        )
    )

    survival_probs = (
        survival_probability_matrix(
            eval_pred,
            times,
            baseline,
        )
    )

    _, brier_values = brier_score(
        y_train,
        y_eval,
        survival_probs,
        times,
    )

    if len(times) >= 2:
        ibs = float(
            integrated_brier_score(
                y_train,
                y_eval,
                survival_probs,
                times,
            )
        )
    else:
        ibs = np.nan

    table = pd.DataFrame({
        "time_days": times,
        "time_years_approx": times / 365.0,
        "AUC_dynamic": auc_values,
        "Brier_score": brier_values,
    })

    return (
        table,
        float(mean_auc),
        ibs,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_full_cox_epoch(
    model,
    dataset,
    optimizer,
    scaler,
    device,
    amp_enabled,
    batch_size,
    num_workers,
    epoch_seed,
):
    """
    Exact full-cohort Cox-Efron gradient with 3D CT micro-batches.

    PASS 1:
      evaluate all training patients, store risks and dLoss/dRisk.

    PASS 2:
      replay the exact same dropout RNG states, recompute risks,
      and propagate the stored dLoss/dRisk through the multimodal network.

    BatchNorm state is restored between passes so its running statistics are
    updated only once per epoch.
    """
    model.train()

    gen = torch.Generator()
    gen.manual_seed(
        epoch_seed
    )

    perm = torch.randperm(
        len(dataset),
        generator=gen,
    ).tolist()

    epoch_subset = Subset(
        dataset,
        perm,
    )

    def make_loader():
        return DataLoader(
            epoch_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            drop_last=False,
        )

    bn_before = snapshot_bn_state(
        model
    )

    first_risks = []
    first_times = []
    first_events = []
    rng_states = []
    batch_sizes = []

    # -------------------------------------------------------------
    # PASS 1: risks for the entire 295-patient training cohort.
    # -------------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(
            make_loader(),
            leave=False,
        ):
            ct = batch["ct"].to(
                device,
                non_blocking=True,
            )
            clinical = batch["clinical"].to(
                device,
                non_blocking=True,
            )

            rng_states.append(
                capture_rng_state(device)
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                risk = model(
                    ct,
                    clinical,
                )

            first_risks.append(
                risk.detach().float().cpu()
            )
            first_times.append(
                batch["time"].detach().double().cpu()
            )
            first_events.append(
                batch["event"].detach().double().cpu()
            )
            batch_sizes.append(
                len(ct)
            )

    risks_cpu = torch.cat(
        first_risks
    ).double()

    times_cpu = torch.cat(
        first_times
    ).double()

    events_cpu = torch.cat(
        first_events
    ).double()

    risk_leaf = (
        risks_cpu.clone()
        .requires_grad_(True)
    )

    full_loss = cox_efron_loss_full(
        risk_leaf,
        times_cpu,
        events_cpu,
    )

    full_loss.backward()

    grad_risk = (
        risk_leaf.grad.detach().float()
    )

    # PASS 1 updated BatchNorm statistics despite no_grad.
    # Restore them so PASS 2 is the one effective update.
    restore_bn_state(
        model,
        bn_before,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    offset = 0
    replay_errors = []

    # -------------------------------------------------------------
    # PASS 2: exact chain-rule replay through the multimodal network.
    # -------------------------------------------------------------
    for bidx, batch in enumerate(
        tqdm(
            make_loader(),
            leave=False,
        )
    ):
        ct = batch["ct"].to(
            device,
            non_blocking=True,
        )
        clinical = batch["clinical"].to(
            device,
            non_blocking=True,
        )

        restore_rng_state(
            rng_states[bidx],
            device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            risk = model(
                ct,
                clinical,
            )

        bs = batch_sizes[bidx]

        reference = risks_cpu[
            offset:offset + bs
        ].float().to(device)

        err = torch.max(
            torch.abs(
                risk.detach().float()
                - reference
            )
        )

        replay_errors.append(
            float(err.cpu())
        )

        grad_chunk = grad_risk[
            offset:offset + bs
        ].to(
            device=device,
            dtype=risk.dtype,
        )

        if scaler.is_enabled():
            scaler.scale(
                risk
            ).backward(
                gradient=grad_chunk
            )
        else:
            risk.backward(
                gradient=grad_chunk
            )

        offset += bs

    max_replay_error = (
        max(replay_errors)
        if replay_errors
        else 0.0
    )

    if max_replay_error > 2e-4:
        raise RuntimeError(
            "Two-pass risk replay mismatch is too large: "
            f"{max_replay_error:.8g}"
        )

    if scaler.is_enabled():
        scaler.step(
            optimizer
        )
        scaler.update()
    else:
        optimizer.step()

    return {
        "full_cohort_cox_efron_loss":
            float(full_loss.detach().cpu()),
        "max_two_pass_risk_replay_error":
            max_replay_error,
        "events":
            int(events_cpu.sum().item()),
    }



# ---------------------------------------------------------------------
# Missingness-aware ModDrop comparator
# ---------------------------------------------------------------------

def validate_tnm_corrected_features(feature_cols):
    cols = list(feature_cols)
    forbidden = {
        "clinical.T.Stage_5", "clinical.T.Stage_5.0",
        "Clinical.N.Stage_4", "Clinical.N.Stage_4.0",
        "Clinical.M.Stage_3", "Clinical.M.Stage_3.0",
    }
    bad = sorted(forbidden.intersection(cols))
    if bad:
        raise RuntimeError(
            "Pre-correction TNM sentinel categories detected: " + ", ".join(bad)
        )

    expected = {
        "clinical.T.Stage_MISSING",
        "Clinical.N.Stage_MISSING",
        "Clinical.M.Stage_MISSING",
    }
    missing = sorted(expected.difference(cols))
    if missing:
        raise RuntimeError(
            "Expected TNM-corrected missing indicators are absent: "
            + ", ".join(missing)
        )

    if len(cols) != 21:
        raise RuntimeError(
            f"Expected 21 TNM-corrected transformed clinical features, got {len(cols)}"
        )


def load_clinical_subset(clinical_dir: Path, subset: str, feature_cols=None):
    path = Path(clinical_dir) / f"{subset}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"PatientID", "Survival.time", "deadstatus.event"}
    miss = sorted(required.difference(df.columns))
    if miss:
        raise RuntimeError(f"{path}: missing required columns {miss}")

    cols = [
        c for c in df.columns
        if c not in ["PatientID", "Survival.time", "deadstatus.event"]
    ]

    if feature_cols is None:
        feature_cols = cols
    elif cols != list(feature_cols):
        raise RuntimeError(
            f"Transformed clinical feature columns differ in {subset}."
        )

    return df, list(feature_cols)


def exact_missing_mask(n: int, fraction: float, seed: int):
    n = int(n)
    fraction = float(fraction)
    if fraction <= 0:
        return np.zeros(n, dtype=bool)
    if fraction >= 1:
        return np.ones(n, dtype=bool)

    k = int(round(fraction * n))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def train_full_cox_epoch_moddrop(
    model,
    dataset,
    optimizer,
    scaler,
    device,
    amp_enabled,
    batch_size,
    num_workers,
    epoch_seed,
    moddrop_prob,
):
    """
    Exact full-training-cohort Cox-Efron gradient with whole-clinical-modality
    dropout (ModDrop). The missingness mask is generated once per epoch and is
    replayed identically in the two Cox passes. Network dropout RNG states are
    also replayed exactly, preserving the original full-cohort gradient method.
    """
    model.train()

    gen = torch.Generator()
    gen.manual_seed(int(epoch_seed))
    perm = torch.randperm(len(dataset), generator=gen).tolist()
    epoch_subset = Subset(dataset, perm)

    # Exact per-epoch missing fraction for stable training while preserving
    # whole-modality dropout semantics.
    missing_epoch = exact_missing_mask(
        len(dataset),
        float(moddrop_prob),
        int(epoch_seed) + 7919,
    )

    def make_loader():
        return DataLoader(
            epoch_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    bn_before = snapshot_bn_state(model)

    first_risks = []
    first_times = []
    first_events = []
    rng_states = []
    batch_sizes = []

    offset = 0

    # PASS 1
    with torch.no_grad():
        for batch in tqdm(make_loader(), leave=False):
            ct = batch["ct"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            bs = len(ct)

            miss = torch.from_numpy(
                missing_epoch[offset:offset + bs]
            ).to(device=device)
            clinical = clinical.clone()
            clinical[miss] = 0.0

            rng_states.append(capture_rng_state(device))

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                risk = model(ct, clinical)

            first_risks.append(risk.detach().float().cpu())
            first_times.append(batch["time"].detach().double().cpu())
            first_events.append(batch["event"].detach().double().cpu())
            batch_sizes.append(bs)
            offset += bs

    risks_cpu = torch.cat(first_risks).double()
    times_cpu = torch.cat(first_times).double()
    events_cpu = torch.cat(first_events).double()

    risk_leaf = risks_cpu.clone().requires_grad_(True)
    full_loss = cox_efron_loss_full(risk_leaf, times_cpu, events_cpu)
    full_loss.backward()
    grad_risk = risk_leaf.grad.detach().float()

    restore_bn_state(model, bn_before)
    optimizer.zero_grad(set_to_none=True)

    offset = 0
    replay_errors = []

    # PASS 2
    for bidx, batch in enumerate(tqdm(make_loader(), leave=False)):
        ct = batch["ct"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        bs = batch_sizes[bidx]

        miss = torch.from_numpy(
            missing_epoch[offset:offset + bs]
        ).to(device=device)
        clinical = clinical.clone()
        clinical[miss] = 0.0

        restore_rng_state(rng_states[bidx], device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            risk = model(ct, clinical)

        reference = risks_cpu[offset:offset + bs].float().to(device)
        err = torch.max(torch.abs(risk.detach().float() - reference))
        replay_errors.append(float(err.cpu()))

        grad_chunk = grad_risk[offset:offset + bs].to(
            device=device,
            dtype=risk.dtype,
        )

        if scaler.is_enabled():
            scaler.scale(risk).backward(gradient=grad_chunk)
        else:
            risk.backward(gradient=grad_chunk)

        offset += bs

    max_replay_error = max(replay_errors) if replay_errors else 0.0
    if max_replay_error > 2e-4:
        raise RuntimeError(
            "Two-pass risk replay mismatch is too large: "
            f"{max_replay_error:.8g}"
        )

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return {
        "full_cohort_cox_efron_loss": float(full_loss.detach().cpu()),
        "max_two_pass_risk_replay_error": max_replay_error,
        "events": int(events_cpu.sum().item()),
        "moddrop_missing_fraction": float(missing_epoch.mean()),
        "moddrop_missing_n": int(missing_epoch.sum()),
    }


@torch.no_grad()
def predict_risk_with_mask(model, loader, device, missing_mask=None):
    model.eval()
    rows = []
    offset = 0

    if missing_mask is not None:
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if len(missing_mask) != len(loader.dataset):
            raise RuntimeError(
                f"Missing-mask length {len(missing_mask)} != dataset length {len(loader.dataset)}"
            )

    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        bs = len(ct)

        if missing_mask is not None:
            miss = torch.from_numpy(
                missing_mask[offset:offset + bs]
            ).to(device=device)
            clinical = clinical.clone()
            clinical[miss] = 0.0

        risk = model(ct, clinical).detach().cpu().numpy()

        for pid, t, e, r in zip(
            batch["patient_id"],
            batch["time"].numpy(),
            batch["event"].numpy(),
            risk,
        ):
            rows.append((pid, float(t), int(e), float(r)))

        offset += bs

    return pd.DataFrame(
        rows,
        columns=["PatientID", "Survival.time", "deadstatus.event", "risk_score"],
    )


@torch.no_grad()
def extract_fusion_embeddings(model, loader, device):
    """Cache CT embeddings and original clinical vectors after model locking."""
    model.eval()
    rows = []
    z_ct_all = []
    clin_all = []

    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        z_ct = model.ct_encoder(ct).detach().float().cpu().numpy()

        z_ct_all.append(z_ct)
        clin_all.append(clinical.detach().float().cpu().numpy())

        for pid, t, e in zip(
            batch["patient_id"],
            batch["time"].numpy(),
            batch["event"].numpy(),
        ):
            rows.append((pid, float(t), int(e)))

    meta = pd.DataFrame(
        rows,
        columns=["PatientID", "Survival.time", "deadstatus.event"],
    )
    return meta, np.concatenate(z_ct_all), np.concatenate(clin_all)


@torch.no_grad()
def risks_from_cached_embeddings(model, z_ct_np, clinical_np, missing_mask, device, batch_size=256):
    model.eval()
    z_ct_np = np.asarray(z_ct_np, dtype=np.float32)
    clinical_np = np.asarray(clinical_np, dtype=np.float32)
    missing_mask = np.asarray(missing_mask, dtype=bool)

    out = []
    for start in range(0, len(z_ct_np), int(batch_size)):
        stop = min(len(z_ct_np), start + int(batch_size))
        z_ct = torch.from_numpy(z_ct_np[start:stop]).to(device)
        clinical = torch.from_numpy(clinical_np[start:stop]).to(device)
        miss = torch.from_numpy(missing_mask[start:stop]).to(device=device)
        clinical = clinical.clone()
        clinical[miss] = 0.0

        z_cl = model.clinical_encoder(clinical)
        risk = model.fusion(torch.cat([z_ct, z_cl], dim=1)).squeeze(1)
        out.append(risk.detach().float().cpu().numpy())

    return np.concatenate(out)


def dataframe_from_cached(meta, risk):
    df = meta.copy()
    df["risk_score"] = np.asarray(risk, dtype=float)
    return df


def summarize_scenario_metrics(raw_df):
    rows = []
    for pct, g in raw_df.groupby("Missing_percent"):
        row = {
            "Missing_percent": int(pct),
            "Repeats": int(len(g)),
            "Interval_type": (
                "deterministic endpoint"
                if int(pct) in (0, 100)
                else "missingness-simulation percentile range"
            ),
        }
        for metric in ["C_index", "Mean_dynamic_AUC", "Integrated_Brier_Score"]:
            v = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(v))
            row[f"{metric}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            row[f"{metric}_p2_5"] = float(np.percentile(v, 2.5))
            row[f"{metric}_p97_5"] = float(np.percentile(v, 97.5))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Missing_percent").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Missingness-aware standard concatenation comparator trained with whole-clinical "
            "ModDrop and exact full-training-cohort Cox-Efron optimization."
        )
    )

    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clinical-dir", required=True)
    ap.add_argument("--ct-checkpoint", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--moddrop", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--scenario-repeats", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--times", type=int, nargs="+", default=[365, 730, 1095])

    args = ap.parse_args()

    if not (0.0 <= args.moddrop <= 1.0):
        raise ValueError("--moddrop must be between 0 and 1")

    set_seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    vol_dir = Path(args.vol_dir)
    npz_count = len(list(vol_dir.glob("*_standardized_192x160x224.npz")))
    if npz_count != 422:
        raise RuntimeError(
            f"Expected 422 standardized NPZ volumes in {vol_dir}, found {npz_count}."
        )

    split = pd.read_csv(args.split)
    train_split = split[split["Split"].str.lower() == "train"].copy()
    val_split = split[split["Split"].str.lower() == "validation"].copy()

    # IMPORTANT: test clinical features and test dataset are intentionally not
    # loaded before model selection.
    train_clin, feature_cols = load_clinical_subset(
        Path(args.clinical_dir), "train", feature_cols=None
    )
    val_clin, _ = load_clinical_subset(
        Path(args.clinical_dir), "validation", feature_cols=feature_cols
    )
    validate_tnm_corrected_features(feature_cols)

    train_ds = MultimodalLung1Dataset(
        train_split, train_clin, feature_cols, vol_dir
    )
    val_ds = MultimodalLung1Dataset(
        val_split, val_clin, feature_cols, vol_dir
    )

    eval_bs = min(args.batch_size, 4)
    train_eval_loader = DataLoader(
        train_ds, batch_size=eval_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("TNM-corrected clinical feature audit: PASS")
    print("Development split only:", {"train": len(train_ds), "validation": len(val_ds)})
    print("TEST LOCK: held-out test data are NOT loaded during training/model selection.")
    print("ModDrop whole-clinical missing probability:", args.moddrop)
    print("Selection score: mean validation C-index at 0%, 30%, and 100% clinical absence.")
    print("Objective: exact FULL training-cohort Cox partial likelihood with Efron ties.")

    if device.type != "cuda":
        print("WARNING: CUDA GPU strongly recommended.")

    model = MultimodalSurvivalModel(
        clinical_dim=len(feature_cols),
        dropout=args.dropout,
    )

    ckpt_path = Path(args.ct_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    load_ct_only_encoder(model, ckpt_path)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # Fixed validation masks. The 30% mask is fixed across epochs so checkpoint
    # selection is deterministic and does not chase missingness-sampling noise.
    val_masks = {
        0: exact_missing_mask(len(val_ds), 0.0, args.seed + 30000),
        30: exact_missing_mask(len(val_ds), 0.30, args.seed + 30030),
        100: exact_missing_mask(len(val_ds), 1.0, args.seed + 30100),
    }

    mask_audit = pd.DataFrame({
        "PatientID": val_ds.df["PatientID"].astype(str),
        "missing_0pct": val_masks[0].astype(int),
        "missing_30pct": val_masks[30].astype(int),
        "missing_100pct": val_masks[100].astype(int),
    })
    mask_audit.to_csv(outdir / "validation_selection_masks.csv", index=False)

    best_score = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []
    best_model_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_info = train_full_cox_epoch_moddrop(
            model=model,
            dataset=train_ds,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epoch_seed=args.seed + 1000 * epoch,
            moddrop_prob=args.moddrop,
        )

        train_pred_full = predict_risk_with_mask(
            model, train_eval_loader, device,
            missing_mask=np.zeros(len(train_ds), dtype=bool),
        )
        train_c = harrell_cindex(train_pred_full)

        val_c = {}
        for pct in [0, 30, 100]:
            pred = predict_risk_with_mask(
                model, val_loader, device, missing_mask=val_masks[pct]
            )
            val_c[pct] = harrell_cindex(pred)

        robust_score = float(np.mean([val_c[0], val_c[30], val_c[100]]))
        scheduler.step(robust_score)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "full_cohort_cox_efron_loss": train_info["full_cohort_cox_efron_loss"],
            "max_two_pass_risk_replay_error": train_info["max_two_pass_risk_replay_error"],
            "train_c_index_full_clinical": train_c,
            "validation_c_index_0pct": val_c[0],
            "validation_c_index_30pct": val_c[30],
            "validation_c_index_100pct": val_c[100],
            "validation_robust_score": robust_score,
            "moddrop_missing_fraction": train_info["moddrop_missing_fraction"],
            "learning_rate": lr,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Cox-Efron={train_info['full_cohort_cox_efron_loss']:.5f} | "
            f"replay={train_info['max_two_pass_risk_replay_error']:.2e} | "
            f"ModDrop={train_info['moddrop_missing_fraction']:.3f} | "
            f"train C={train_c:.4f} | "
            f"val C [0/30/100%]={val_c[0]:.4f}/{val_c[30]:.4f}/{val_c[100]:.4f} | "
            f"robust={robust_score:.4f} | lr={lr:.2e}"
        )

        if robust_score > best_score + 1e-4:
            best_score = robust_score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_robust_score": robust_score,
                    "validation_c_index_0pct": val_c[0],
                    "validation_c_index_30pct": val_c[30],
                    "validation_c_index_100pct": val_c[100],
                    "clinical_feature_cols": feature_cols,
                    "training_objective": "full_training_cohort_cox_partial_likelihood",
                    "tie_method": "Efron",
                    "gradient_method": "two_pass_exact_full_cohort_risk_gradient",
                    "whole_clinical_moddrop": float(args.moddrop),
                    "selection_rule": "mean validation C-index at 0/30/100% clinical absence",
                    "test_used_for_selection": False,
                    "args": vars(args),
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                f"Early stopping. Best epoch={best_epoch}, "
                f"validation robust score={best_score:.4f}"
            )
            break

    pd.DataFrame(history).to_csv(
        outdir / "training_history.csv", index=False, encoding="utf-8-sig"
    )

    # ===============================================================
    # TEST UNLOCK — only after best validation checkpoint is fixed.
    # ===============================================================
    print("\n=== MODEL SELECTION COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Best validation robust score:", f"{best_score:.4f}")
    print("TEST UNLOCK: loading held-out test set for final evaluation only.")

    checkpoint = torch.load(
        best_model_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_split = split[split["Split"].str.lower() == "test"].copy()
    test_clin, _ = load_clinical_subset(
        Path(args.clinical_dir), "test", feature_cols=feature_cols
    )
    test_ds = MultimodalLung1Dataset(
        test_split, test_clin, feature_cols, vol_dir
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Cache model-locked embeddings once; all repeated missingness simulations
    # thereafter vary only the clinical availability mask.
    train_meta, train_zct, train_clin_np = extract_fusion_embeddings(
        model, train_eval_loader, device
    )
    val_meta, val_zct, val_clin_np = extract_fusion_embeddings(
        model, val_loader, device
    )
    test_meta, test_zct, test_clin_np = extract_fusion_embeddings(
        model, test_loader, device
    )

    fractions = [0.0, 0.10, 0.30, 0.50, 1.0]
    raw_rows = []

    for frac in fractions:
        pct = int(round(100 * frac))
        nrep = 1 if pct in (0, 100) else int(args.scenario_repeats)
        print(f"Final held-out test missingness {pct}% | repeats={nrep}")

        for rep in range(nrep):
            train_mask = exact_missing_mask(
                len(train_meta), frac,
                args.seed + 100000 + 1000 * pct + rep,
            )
            test_mask = exact_missing_mask(
                len(test_meta), frac,
                args.seed + 200000 + 1000 * pct + rep,
            )

            train_risk = risks_from_cached_embeddings(
                model, train_zct, train_clin_np, train_mask, device
            )
            test_risk = risks_from_cached_embeddings(
                model, test_zct, test_clin_np, test_mask, device
            )

            train_df = dataframe_from_cached(train_meta, train_risk)
            test_df = dataframe_from_cached(test_meta, test_risk)

            c = harrell_cindex(test_df)
            baseline = fit_breslow_baseline(train_df)
            _, mean_auc, ibs = time_dependent_metrics(
                train_df, test_df, baseline, args.times
            )

            raw_rows.append({
                "Missing_percent": pct,
                "Repeat": rep,
                "N_train_missing": int(train_mask.sum()),
                "N_test_missing": int(test_mask.sum()),
                "C_index": c,
                "Mean_dynamic_AUC": mean_auc,
                "Integrated_Brier_Score": ibs,
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(
        outdir / "MissingnessAware_ModDrop_Raw_Repeats.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df = summarize_scenario_metrics(raw_df)
    summary_df.to_csv(
        outdir / "MissingnessAware_ModDrop_Summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # Deterministic full/no-clinical risks for future paired comparison scripts.
    train_full = risks_from_cached_embeddings(
        model, train_zct, train_clin_np,
        np.zeros(len(train_meta), dtype=bool), device
    )
    train_none = risks_from_cached_embeddings(
        model, train_zct, train_clin_np,
        np.ones(len(train_meta), dtype=bool), device
    )
    test_full = risks_from_cached_embeddings(
        model, test_zct, test_clin_np,
        np.zeros(len(test_meta), dtype=bool), device
    )
    test_none = risks_from_cached_embeddings(
        model, test_zct, test_clin_np,
        np.ones(len(test_meta), dtype=bool), device
    )

    train_modes = train_meta.copy()
    train_modes["risk_full_clinical"] = train_full
    train_modes["risk_no_clinical"] = train_none
    test_modes = test_meta.copy()
    test_modes["risk_full_clinical"] = test_full
    test_modes["risk_no_clinical"] = test_none

    train_modes.to_csv(
        outdir / "train_risk_modes.csv", index=False, encoding="utf-8-sig"
    )
    test_modes.to_csv(
        outdir / "test_risk_modes.csv", index=False, encoding="utf-8-sig"
    )

    # Bootstrap C-index CI for the complete-data test point estimate.
    full_test_df = dataframe_from_cached(test_meta, test_full)
    full_ci = bootstrap_cindex(
        full_test_df, n_boot=args.bootstrap, seed=args.seed + 500000
    )

    lines = [
        "LUNG1 — missingness-aware standard concatenation comparator with ModDrop",
        "",
        "TRAINING / SELECTION",
        f"Train n={len(train_ds)} | validation n={len(val_ds)} | held-out test n={len(test_ds)}",
        "TNM-corrected feature audit: PASS",
        f"Clinical transformed features: {len(feature_cols)}",
        f"Whole-clinical ModDrop probability: {args.moddrop}",
        "Training objective: exact full training-cohort Cox partial likelihood with Efron ties",
        "Selection: mean validation C-index at 0%, 30%, and 100% clinical absence",
        f"Best epoch: {best_epoch}",
        f"Best validation robust score: {best_score:.4f}",
        "Held-out test used during training/model selection: NO",
        "",
        "FINAL HELD-OUT TEST MISSINGNESS SUMMARY",
    ]

    for _, r in summary_df.iterrows():
        pct = int(r["Missing_percent"])
        lines.append(
            f"  Missing {pct}%: C={r['C_index_mean']:.4f}, "
            f"mean AUC={r['Mean_dynamic_AUC_mean']:.4f}, "
            f"IBS={r['Integrated_Brier_Score_mean']:.4f}"
        )

    lines += [
        "",
        "COMPLETE-DATA TEST C-INDEX BOOTSTRAP",
        f"  C-index={harrell_cindex(full_test_df):.4f}, "
        f"95% CI [{full_ci['lower']:.4f}, {full_ci['upper']:.4f}]",
        "",
        "INTERPRETATION",
        "  This is a missingness-aware baseline because whole-clinical modality dropout was applied during training.",
        "  It remains a standard concatenation architecture: there is no explicit availability gate and no CT-teacher anchor.",
        "  Test results are final evaluation only and must not be used to alter hyperparameters, checkpoint selection, or training.",
        "  Intermediate 10/30/50% percentile ranges quantify missingness-assignment variability, not confidence intervals.",
    ]

    summary_text = "\n".join(lines)
    (outdir / "MissingnessAware_ModDrop_Summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print(summary_text)
    print("=" * 88)
    print("Files written to:", outdir)


if __name__ == "__main__":
    main()
