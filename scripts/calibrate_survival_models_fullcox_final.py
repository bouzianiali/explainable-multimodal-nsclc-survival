#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final calibration evaluation for the four corrected LUNG1 survival models.

Outputs:
- Calibration_Slope.csv
- Calibration_Horizon_Summary.csv
- Calibration_Bins.csv
- Calibration_365d.png/.pdf
- Calibration_730d.png/.pdf
- Calibration_1095d.png/.pdf
- Calibration_Summary.txt

No model is retrained or tuned.
"""

from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar

ID = "PatientID"
TIME = "Survival.time"
EVENT = "deadstatus.event"
MODEL_NAMES = ["Clinical-only", "CT-only", "Naive CT+Clinical", "Robust V2"]


def import_compare_module(path: Path):
    spec = importlib.util.spec_from_file_location("paired_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def km_event_probability(time, event, horizon):
    km = KaplanMeierFitter(alpha=0.05)
    km.fit(np.asarray(time, float), event_observed=np.asarray(event, int))
    s = float(km.predict(horizon))

    ci = km.confidence_interval_survival_function_
    idx = ci.index.to_numpy(dtype=float)
    pos = np.where(idx <= horizon)[0]
    if len(pos) == 0:
        slo = shi = 1.0
    else:
        vals = np.asarray(ci.iloc[pos[-1]], dtype=float)
        slo, shi = float(np.min(vals)), float(np.max(vals))

    return 1.0 - s, 1.0 - shi, 1.0 - slo


def cloglog_survival(s):
    eps = 1e-8
    s = float(np.clip(s, eps, 1.0 - eps))
    return float(np.log(-np.log(s)))


def citl_cloglog(obs_event, pred_event):
    return cloglog_survival(1.0 - obs_event) - cloglog_survival(1.0 - pred_event)


def baseline_hazard_at(t, baseline):
    pos = np.searchsorted(baseline["event_times"], t, side="right") - 1
    return float(baseline["cum_hazard"][pos]) if pos >= 0 else 0.0


def predict_event_probability(risk, horizon, baseline):
    risk = np.asarray(risk, float)
    h0 = baseline_hazard_at(horizon, baseline)
    rel = np.exp(risk - baseline["shift"])
    surv = np.exp(-rel * h0)
    return 1.0 - surv


def efron_negloglik(beta, x, time, event):
    x = np.asarray(x, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    eta = beta * x
    observed = event == 1
    ll = 0.0

    for t in np.unique(time[observed]):
        deaths = observed & (time == t)
        riskset = time >= t
        d = int(deaths.sum())
        er = eta[riskset]
        ed = eta[deaths]
        m = float(np.max(er))
        sr = float(np.exp(er - m).sum())
        sd = float(np.exp(ed - m).sum())
        ll += float(ed.sum())

        for l in range(d):
            denom = max(sr - (l / d) * sd, np.finfo(float).tiny)
            ll -= m + np.log(denom)

    return -ll


def calibration_slope(time, event, risk):
    risk = np.asarray(risk, float)
    x = risk - risk.mean()
    if np.std(x) < 1e-12:
        return np.nan

    res = minimize_scalar(
        efron_negloglik,
        args=(x, np.asarray(time, float), np.asarray(event, int)),
        method="bounded",
        bounds=(-10.0, 10.0),
        options={"xatol": 1e-7},
    )
    return float(res.x) if res.success else np.nan


def make_bins(test, model, horizon, pred_event, n_bins):
    x = pd.DataFrame({
        ID: test[ID].astype(str),
        TIME: test[TIME].to_numpy(float),
        EVENT: test[EVENT].to_numpy(int),
        "pred": np.asarray(pred_event, float),
    })

    x["bin"] = pd.qcut(
        x["pred"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    rows = []
    for b in sorted(x["bin"].dropna().unique()):
        s = x[x["bin"] == b]
        obs, lo, hi = km_event_probability(s[TIME], s[EVENT], horizon)
        rows.append({
            "Model": model,
            "Horizon_days": int(horizon),
            "Bin": int(b) + 1,
            "N": int(len(s)),
            "Events_total": int(s[EVENT].sum()),
            "Mean_predicted_event_probability": float(s["pred"].mean()),
            "Observed_KM_event_probability": float(obs),
            "Observed_KM_CI95_lower": float(lo),
            "Observed_KM_CI95_upper": float(hi),
        })
    return pd.DataFrame(rows)


def save_plot(bins, horizon, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], "--", linewidth=1.2, label="Ideal")

    h = bins[bins["Horizon_days"] == int(horizon)]
    for model in MODEL_NAMES:
        m = h[h["Model"] == model].sort_values("Bin")
        ax.plot(
            m["Mean_predicted_event_probability"],
            m["Observed_KM_event_probability"],
            marker="o",
            linewidth=1.5,
            label=model,
        )

    ax.set_xlabel("Mean predicted event probability")
    ax.set_ylabel("Kaplan-Meier observed event probability")
    ax.set_title(f"Calibration at {int(horizon)} days")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"Calibration_{int(horizon)}d.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"Calibration_{int(horizon)}d.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--compare-script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", nargs="+", type=int, default=[365, 730, 1095])
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cmp = import_compare_module(Path(args.compare_script))

    clinical_dir = root / "clinical_baselines" / "model_A"
    ct_dir = root / "ct_only_fullcox_efron"
    naive_dir = root / "multimodal_fullcox_efron"

    robust_dir = root / "RobustV2_FINAL"
    if not robust_dir.exists():
        robust_dir = root / "gated_moddrop_v2_fullcox_efron"

    train_preds = {
        "Clinical-only": cmp.clinical_risk_from_model(clinical_dir, "train"),
        "CT-only": cmp.standard_risk_file(ct_dir / "train_risk_scores.csv"),
        "Naive CT+Clinical": cmp.standard_risk_file(naive_dir / "train_risk_scores.csv"),
        "Robust V2": cmp.robust_v2_risk_file(robust_dir / "train_risk_modes.csv", "risk_full"),
    }

    test_preds = {
        "Clinical-only": cmp.clinical_risk_from_model(clinical_dir, "test"),
        "CT-only": cmp.standard_risk_file(ct_dir / "test_risk_scores.csv"),
        "Naive CT+Clinical": cmp.standard_risk_file(naive_dir / "test_risk_scores.csv"),
        "Robust V2": cmp.robust_v2_risk_file(robust_dir / "test_risk_modes.csv", "risk_full"),
    }

    train = cmp.align_predictions(train_preds)
    test = cmp.align_predictions(test_preds)

    print(f"Aligned train: {len(train)} | events={int(train[EVENT].sum())}")
    print(f"Aligned test:  {len(test)} | events={int(test[EVENT].sum())}")

    horizons = [int(x) for x in args.times]
    baselines = {}
    pred_event = {m: {} for m in MODEL_NAMES}

    for model in MODEL_NAMES:
        baselines[model] = cmp.fit_breslow(
            train[TIME].to_numpy(float),
            train[EVENT].to_numpy(int),
            train[model].to_numpy(float),
        )
        for t in horizons:
            pred_event[model][t] = predict_event_probability(
                test[model].to_numpy(float), t, baselines[model]
            )

    # Point calibration slope.
    slope_point = []
    for model in MODEL_NAMES:
        slope_point.append({
            "Model": model,
            "Calibration_slope": calibration_slope(test[TIME], test[EVENT], test[model]),
            "Ideal_value": 1.0,
        })
    slope_point = pd.DataFrame(slope_point)

    # Point horizon calibration and grouped curves.
    horizon_rows = []
    bin_parts = []

    for model in MODEL_NAMES:
        for t in horizons:
            p = pred_event[model][t]
            pm = float(np.mean(p))
            obs, lo, hi = km_event_probability(test[TIME], test[EVENT], t)

            horizon_rows.append({
                "Model": model,
                "Horizon_days": int(t),
                "N_test": int(len(test)),
                "Events_total_test": int(test[EVENT].sum()),
                "Mean_predicted_event_probability": pm,
                "Observed_KM_event_probability": obs,
                "Observed_KM_CI95_lower": lo,
                "Observed_KM_CI95_upper": hi,
                "Observed_minus_predicted": obs - pm,
                "CITL_cloglog": citl_cloglog(obs, pm),
            })

            bin_parts.append(make_bins(test, model, t, p, args.bins))

    horizon_point = pd.DataFrame(horizon_rows)
    bins = pd.concat(bin_parts, ignore_index=True)

    # Patient-level bootstrap uncertainty.
    print(f"Bootstrap calibration uncertainty: {args.bootstrap} replicates...")
    rng = np.random.default_rng(args.seed)
    n = len(test)

    slope_dist = {m: [] for m in MODEL_NAMES}
    op_dist = {m: {t: [] for t in horizons} for m in MODEL_NAMES}
    citl_dist = {m: {t: [] for t in horizons} for m in MODEL_NAMES}
    valid = 0

    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, size=n)
        tb = test.iloc[idx].reset_index(drop=True)

        if tb[EVENT].nunique() < 2:
            continue

        valid += 1

        for model in MODEL_NAMES:
            s = calibration_slope(tb[TIME], tb[EVENT], tb[model])
            if np.isfinite(s):
                slope_dist[model].append(s)

            for t in horizons:
                p = pred_event[model][t][idx]
                pm = float(np.mean(p))
                obs, _, _ = km_event_probability(tb[TIME], tb[EVENT], t)
                op = obs - pm
                ci = citl_cloglog(obs, pm)

                if np.isfinite(op):
                    op_dist[model][t].append(op)
                if np.isfinite(ci):
                    citl_dist[model][t].append(ci)

    slope_boot = []
    for model in MODEL_NAMES:
        v = np.asarray(slope_dist[model], float)
        slope_boot.append({
            "Model": model,
            "Bootstrap_valid_slope_replicates": len(v),
            "Slope_bootstrap_mean": float(np.mean(v)),
            "Slope_CI95_lower": float(np.percentile(v, 2.5)),
            "Slope_CI95_upper": float(np.percentile(v, 97.5)),
        })

    slope_final = slope_point.merge(pd.DataFrame(slope_boot), on="Model", how="left")

    hboot = []
    for model in MODEL_NAMES:
        for t in horizons:
            op = np.asarray(op_dist[model][t], float)
            ci = np.asarray(citl_dist[model][t], float)
            hboot.append({
                "Model": model,
                "Horizon_days": t,
                "Bootstrap_valid_replicates": min(len(op), len(ci)),
                "O_minus_P_bootstrap_mean": float(np.mean(op)),
                "O_minus_P_CI95_lower": float(np.percentile(op, 2.5)),
                "O_minus_P_CI95_upper": float(np.percentile(op, 97.5)),
                "CITL_cloglog_bootstrap_mean": float(np.mean(ci)),
                "CITL_cloglog_CI95_lower": float(np.percentile(ci, 2.5)),
                "CITL_cloglog_CI95_upper": float(np.percentile(ci, 97.5)),
            })

    horizon_final = horizon_point.merge(
        pd.DataFrame(hboot),
        on=["Model", "Horizon_days"],
        how="left",
    )

    slope_final.to_csv(outdir / "Calibration_Slope.csv", index=False, encoding="utf-8-sig")
    horizon_final.to_csv(outdir / "Calibration_Horizon_Summary.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(outdir / "Calibration_Bins.csv", index=False, encoding="utf-8-sig")

    for t in horizons:
        save_plot(bins, t, outdir)

    lines = [
        "LUNG1 - FINAL CALIBRATION OF FOUR CORRECTED SURVIVAL MODELS",
        "",
        f"Train n={len(train)}, events={int(train[EVENT].sum())}",
        f"Test n={len(test)}, events={int(test[EVENT].sum())}",
        f"Horizons: {' | '.join(str(t) for t in horizons)} days",
        f"Calibration bins: {args.bins}",
        f"Requested bootstrap replicates: {args.bootstrap}",
        f"Valid bootstrap replicates: {valid}",
        "",
        "GLOBAL COX CALIBRATION SLOPE (ideal = 1)",
    ]

    for _, r in slope_final.iterrows():
        lines.append(
            f"{r['Model']}: slope={r['Calibration_slope']:.4f}, "
            f"95% CI [{r['Slope_CI95_lower']:.4f}, {r['Slope_CI95_upper']:.4f}]"
        )

    lines += [
        "",
        "HORIZON-SPECIFIC CALIBRATION",
        "Observed-minus-predicted ideal = 0; CITL_cloglog ideal = 0.",
        "",
    ]

    for _, r in horizon_final.iterrows():
        lines.append(
            f"{r['Model']} @ {int(r['Horizon_days'])}d: "
            f"pred={r['Mean_predicted_event_probability']:.4f}, "
            f"KM obs={r['Observed_KM_event_probability']:.4f}, "
            f"O-P={r['Observed_minus_predicted']:+.4f}, "
            f"CITL={r['CITL_cloglog']:+.4f}"
        )

    lines += [
        "",
        "Methodological note:",
        "No model was retrained or tuned.",
        "Baseline cumulative hazards were estimated from training predictions only.",
        "Calibration was evaluated on the locked test set.",
        "Quartile calibration curves use Kaplan-Meier observed event risks.",
        "Uncertainty intervals are patient-level bootstrap percentile intervals.",
    ]

    (outdir / "Calibration_Summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("\n".join(lines))
    print("=" * 80)
    print(f"\nFiles written to: {outdir}")


if __name__ == "__main__":
    main()
