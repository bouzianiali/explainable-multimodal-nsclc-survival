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
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--vol-dir",
        required=True,
    )
    ap.add_argument(
        "--split",
        required=True,
    )
    ap.add_argument(
        "--clinical-dir",
        required=True,
    )
    ap.add_argument(
        "--ct-checkpoint",
        default="",
        help="Optional CT-only best_model.pt",
    )
    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
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
        type=int,
        nargs="+",
        default=[365, 730, 1095],
    )

    args = ap.parse_args()

    set_seed(
        args.seed
    )

    outdir = Path(
        args.out
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = pd.read_csv(
        args.split
    )

    train_split = split[
        split["Split"].str.lower()
        == "train"
    ].copy()

    val_split = split[
        split["Split"].str.lower()
        == "validation"
    ].copy()

    test_split = split[
        split["Split"].str.lower()
        == "test"
    ].copy()

    clinical_tables, feature_cols = (
        load_clinical_feature_tables(
            Path(args.clinical_dir)
        )
    )

    print(
        "Clinical transformed feature count:",
        len(feature_cols),
    )

    train_ds = MultimodalLung1Dataset(
        train_split,
        clinical_tables["train"],
        feature_cols,
        Path(args.vol_dir),
    )

    val_ds = MultimodalLung1Dataset(
        val_split,
        clinical_tables["validation"],
        feature_cols,
        Path(args.vol_dir),
    )

    test_ds = MultimodalLung1Dataset(
        test_split,
        clinical_tables["test"],
        feature_cols,
        Path(args.vol_dir),
    )

    eval_bs = min(
        args.batch_size,
        4,
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Split:",
        {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
    )
    print(
        "Objective: FULL training-cohort Cox partial likelihood, "
        "Efron ties; micro-batch only for memory."
    )

    if device.type != "cuda":
        print(
            "WARNING: CUDA GPU strongly recommended."
        )

    model = MultimodalSurvivalModel(
        clinical_dim=len(feature_cols),
        dropout=args.dropout,
    )

    if args.ct_checkpoint:
        ckpt_path = Path(
            args.ct_checkpoint
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(
                ckpt_path
            )

        load_ct_only_encoder(
            model,
            ckpt_path,
        )

    model = model.to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_val = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []

    best_model_path = (
        outdir / "best_model.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_info = train_full_cox_epoch(
            model=model,
            dataset=train_ds,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epoch_seed=args.seed + 1000 * epoch,
        )

        train_pred = predict_risk(
            model,
            train_eval_loader,
            device,
        )

        val_pred = predict_risk(
            model,
            val_loader,
            device,
        )

        train_c = harrell_cindex(
            train_pred
        )

        val_c = harrell_cindex(
            val_pred
        )

        scheduler.step(
            val_c
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        history.append({
            "epoch": epoch,
            "full_cohort_cox_efron_loss":
                train_info["full_cohort_cox_efron_loss"],
            "max_two_pass_risk_replay_error":
                train_info["max_two_pass_risk_replay_error"],
            "train_c_index": train_c,
            "validation_c_index": val_c,
            "learning_rate": lr,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Cox-Efron="
            f"{train_info['full_cohort_cox_efron_loss']:.5f} | "
            f"replay="
            f"{train_info['max_two_pass_risk_replay_error']:.2e} | "
            f"train C={train_c:.4f} | "
            f"val C={val_c:.4f} | "
            f"lr={lr:.2e}"
        )

        if val_c > best_val + 1e-4:
            best_val = val_c
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "validation_c_index":
                        val_c,
                    "clinical_feature_cols":
                        feature_cols,
                    "training_objective":
                        "full_training_cohort_cox_partial_likelihood",
                    "tie_method":
                        "Efron",
                    "gradient_method":
                        "two_pass_exact_full_cohort_risk_gradient",
                    "micro_batch_size":
                        args.batch_size,
                    "args":
                        vars(args),
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                "Early stopping. "
                f"Best epoch={best_epoch}, "
                f"val C={best_val:.4f}"
            )
            break

    pd.DataFrame(
        history
    ).to_csv(
        outdir
        / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Final evaluation after loading best validation checkpoint
    # --------------------------------------------------------------
    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    train_pred = predict_risk(
        model,
        train_eval_loader,
        device,
    )

    val_pred = predict_risk(
        model,
        val_loader,
        device,
    )

    test_pred = predict_risk(
        model,
        test_loader,
        device,
    )

    train_pred.to_csv(
        outdir
        / "train_risk_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_pred.to_csv(
        outdir
        / "validation_risk_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_pred.to_csv(
        outdir
        / "test_risk_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_c = harrell_cindex(
        train_pred
    )

    val_c = harrell_cindex(
        val_pred
    )

    test_c = harrell_cindex(
        test_pred
    )

    val_ci = bootstrap_cindex(
        val_pred,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    test_ci = bootstrap_cindex(
        test_pred,
        n_boot=args.bootstrap,
        seed=args.seed + 1,
    )

    baseline = fit_breslow_baseline(
        train_pred
    )

    val_td, val_auc, val_ibs = (
        time_dependent_metrics(
            train_pred,
            val_pred,
            baseline,
            args.times,
        )
    )

    test_td, test_auc, test_ibs = (
        time_dependent_metrics(
            train_pred,
            test_pred,
            baseline,
            args.times,
        )
    )

    val_td["Subset"] = (
        "validation"
    )
    test_td["Subset"] = (
        "test"
    )

    pd.concat(
        [val_td, test_td],
        ignore_index=True,
    ).to_csv(
        outdir
        / "Multimodal_Time_Dependent_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = pd.DataFrame([
        {
            "Subset": "train",
            "N": len(train_pred),
            "C_index": train_c,
            "C_index_CI95_lower": np.nan,
            "C_index_CI95_upper": np.nan,
            "Mean_dynamic_AUC": np.nan,
            "Integrated_Brier_Score": np.nan,
        },
        {
            "Subset": "validation",
            "N": len(val_pred),
            "C_index": val_c,
            "C_index_CI95_lower":
                val_ci["lower"],
            "C_index_CI95_upper":
                val_ci["upper"],
            "Mean_dynamic_AUC":
                val_auc,
            "Integrated_Brier_Score":
                val_ibs,
        },
        {
            "Subset": "test",
            "N": len(test_pred),
            "C_index": test_c,
            "C_index_CI95_lower":
                test_ci["lower"],
            "C_index_CI95_upper":
                test_ci["upper"],
            "Mean_dynamic_AUC":
                test_auc,
            "Integrated_Brier_Score":
                test_ibs,
        },
    ])

    metrics.to_csv(
        outdir
        / "Multimodal_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "LUNG1 naive CT + clinical survival model - FULL COX-EFRON CORRECTION",
        "",
        f"Device: {device}",
        f"Best epoch: {best_epoch}",
        (
            "Best validation C-index during training: "
            f"{best_val:.4f}"
        ),
        "Input CT volume: 192x160x224",
        (
            "Clinical transformed feature count: "
            f"{len(feature_cols)}"
        ),
        (
            "CT checkpoint initialization: "
            + (
                args.ct_checkpoint
                if args.ct_checkpoint
                else "none"
            )
        ),
        "",
        "Training objective: full training-cohort Cox partial likelihood",
        "Tied event times: Efron",
        "Memory strategy: two-pass exact dLoss/dRisk replay",
        (
            "Micro-batch size: "
            f"{args.batch_size} (memory only; does not define Cox risk sets)"
        ),
        "",
        f"TRAIN C-index: {train_c:.4f}",
        "",
        "VALIDATION:",
        f"  C-index = {val_c:.4f}",
        (
            "  95% bootstrap CI = "
            f"[{val_ci['lower']:.4f}, "
            f"{val_ci['upper']:.4f}]"
        ),
        (
            "  Mean dynamic AUC = "
            f"{val_auc:.4f}"
        ),
        (
            "  Integrated Brier Score = "
            f"{val_ibs:.4f}"
        ),
    ]

    for _, row in val_td.iterrows():
        lines.append(
            f"  {int(row['time_days'])} days: "
            f"AUC={row['AUC_dynamic']:.4f}, "
            f"Brier={row['Brier_score']:.4f}"
        )

    lines += [
        "",
        "TEST:",
        f"  C-index = {test_c:.4f}",
        (
            "  95% bootstrap CI = "
            f"[{test_ci['lower']:.4f}, "
            f"{test_ci['upper']:.4f}]"
        ),
        (
            "  Mean dynamic AUC = "
            f"{test_auc:.4f}"
        ),
        (
            "  Integrated Brier Score = "
            f"{test_ibs:.4f}"
        ),
    ]

    for _, row in test_td.iterrows():
        lines.append(
            f"  {int(row['time_days'])} days: "
            f"AUC={row['AUC_dynamic']:.4f}, "
            f"Brier={row['Brier_score']:.4f}"
        )

    lines += [
        "",
        "Methodological note:",
        (
            "Clinical preprocessing was inherited from "
            "the train-only Model A pipeline."
        ),
        (
            "Validation alone was used for early stopping "
            "and model selection."
        ),
        (
            "Test was evaluated only after loading the "
            "best validation checkpoint."
        ),
        (
            "The corrected Cox objective uses the full training cohort "
            "with Efron tie handling; micro-batches are only for GPU memory."
        ),
        "",
        "NEXT STEP AFTER REVIEW:",
        (
            "Use the corrected CT-only and corrected naive-fusion checkpoints "
            "to build the Robust V2 Full Cox-Efron model."
        ),
    ]

    summary = "\n".join(
        lines
    )

    (
        outdir
        / "Multimodal_FullCox_Efron_Summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        summary
    )
    print(
        "=" * 72
    )
    print(
        "Files written to:",
        outdir,
    )


if __name__ == "__main__":
    main()
