#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LUNG1 repeated-seed A/B/C ablation.

Freshly retrains all three variants for each seed. Primary readout: validation/tuning
set at 0%, 30%, and 100% clinical-modality absence. Held-out test results are
retained descriptively only and never used for selection.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS = [
    dict(key="A", label="Gate only", folder="A_gate_only_no_moddrop_no_anchor", clinical_moddrop=0.00, anchor_lambda=0.00),
    dict(key="B", label="Gate + ModDrop", folder="B_gate_moddrop_no_anchor", clinical_moddrop=0.25, anchor_lambda=0.00),
    dict(key="C", label="Gate + ModDrop + anchor", folder="C_gate_moddrop_anchor", clinical_moddrop=0.25, anchor_lambda=1.00),
]
PRIMARY_MISSING = [0, 30, 100]
METRICS = ["C_index_mean", "Mean_dynamic_AUC_mean", "Integrated_Brier_Score_mean"]
LOCKED_SEED0 = {
    "A": {0:[0.5688,0.6298,0.2369],30:[0.5626,0.6208,0.2369],100:[0.5588,0.6091,0.2368]},
    "B": {0:[0.5698,0.6282,0.2371],30:[0.5620,0.6177,0.2369],100:[0.5588,0.6109,0.2366]},
    "C": {0:[0.5635,0.6219,0.2376],30:[0.5593,0.6158,0.2372],100:[0.5593,0.6102,0.2369]},
}

def require(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")

def run_cmd(cmd, log_path: Path):
    print("\n" + "="*96 + "\nRUNNING\n" + " ".join(map(str,cmd)) + "\n" + "="*96)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end=""); log.write(line); log.flush()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed (return code {rc}). See {log_path}")

def trainer_command(a, outdir, seed, v):
    return [sys.executable, str(a.trainer),
        "--vol-dir",str(a.vol_dir), "--split",str(a.split), "--clinical-dir",str(a.clinical_dir),
        "--ct-checkpoint",str(a.ct_checkpoint), "--out",str(outdir),
        "--epochs",str(a.epochs), "--freeze-epochs",str(a.freeze_epochs), "--batch-size",str(a.batch_size),
        "--fusion-lr",str(a.fusion_lr), "--ct-lr",str(a.ct_lr), "--weight-decay",str(a.weight_decay),
        "--dropout",str(a.dropout), "--clinical-moddrop",str(v['clinical_moddrop']),
        "--anchor-lambda",str(v['anchor_lambda']), "--patience",str(a.patience),
        "--scenario-repeats",str(a.scenario_repeats), "--seed",str(seed)]

def load_summary(path, seed, v):
    require(path, f"summary seed={seed} variant={v['key']}")
    d = pd.read_csv(path)
    req = ["Subset","Missing_percent",*METRICS]
    missing = [c for c in req if c not in d.columns]
    if missing: raise RuntimeError(f"{path} missing columns {missing}")
    d = d.copy(); d["Subset"] = d["Subset"].astype(str).str.lower(); d["Seed"] = int(seed)
    d["Variant"] = v["key"]; d["Variant_label"] = v["label"]
    return d

def descriptive_summary(d, subset):
    x = d[(d.Subset==subset) & d.Missing_percent.isin(PRIMARY_MISSING)].copy()
    rows=[]
    for (variant,label,pct),g in x.groupby(["Variant","Variant_label","Missing_percent"]):
        r=dict(Subset=subset, Variant=variant, Variant_label=label, Missing_percent=int(pct), N_seeds=int(g.Seed.nunique()))
        for m in METRICS:
            vals=g[m].to_numpy(float); short=m.replace("_mean","")
            r[f"{short}_mean_across_seeds"]=float(vals.mean())
            r[f"{short}_sd_across_seeds"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
            r[f"{short}_median"]=float(np.median(vals)); r[f"{short}_min"]=float(vals.min()); r[f"{short}_max"]=float(vals.max())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["Missing_percent","Variant"]).reset_index(drop=True)

def paired_contrasts(d, subset):
    x=d[(d.Subset==subset) & d.Missing_percent.isin(PRIMARY_MISSING)].copy(); rows=[]
    for seed in sorted(x.Seed.unique()):
        for pct in PRIMARY_MISSING:
            g=x[(x.Seed==seed)&(x.Missing_percent==pct)].set_index("Variant")
            if not {"A","B","C"}.issubset(g.index): raise RuntimeError(f"Missing A/B/C: seed={seed}, {subset}, {pct}%")
            for m in METRICS:
                A=float(g.loc['A',m]); B=float(g.loc['B',m]); C=float(g.loc['C',m])
                rows.append(dict(Subset=subset,Seed=int(seed),Missing_percent=int(pct),Metric=m.replace("_mean",""),
                                 A=A,B=B,C=C,Delta_B_minus_A=B-A,Delta_C_minus_B=C-B,Delta_C_minus_A=C-A))
    return pd.DataFrame(rows)

def summarize_contrasts(d):
    rows=[]
    for (subset,pct,metric),g in d.groupby(["Subset","Missing_percent","Metric"]):
        r=dict(Subset=subset,Missing_percent=int(pct),Metric=metric,N_seeds=int(g.Seed.nunique()))
        for col in ["Delta_B_minus_A","Delta_C_minus_B","Delta_C_minus_A"]:
            v=g[col].to_numpy(float); r[f"{col}_mean"]=float(v.mean()); r[f"{col}_sd"]=float(v.std(ddof=1)) if len(v)>1 else 0.0
            r[f"{col}_min"]=float(v.min()); r[f"{col}_max"]=float(v.max())
        rows.append(r)
    return pd.DataFrame(rows)

def seed0_qc(d, seed0, tol=0.0015):
    x=d[(d.Seed==seed0)&(d.Subset=="validation")&d.Missing_percent.isin(PRIMARY_MISSING)]; rows=[]
    for V in "ABC":
        for pct in PRIMARY_MISSING:
            g=x[(x.Variant==V)&(x.Missing_percent==pct)]
            if len(g)!=1:
                rows.append(dict(Variant=V,Missing_percent=pct,QC_pass=False,Message="row missing")); continue
            obs=g.iloc[0][METRICS].to_numpy(float); exp=np.asarray(LOCKED_SEED0[V][pct],float); diff=np.abs(obs-exp)
            rows.append(dict(Variant=V,Missing_percent=pct,C_abs_diff=float(diff[0]),AUC_abs_diff=float(diff[1]),IBS_abs_diff=float(diff[2]),
                             QC_pass=bool(np.all(diff<=tol)),Message="PASS vs rounded locked seed0" if np.all(diff<=tol) else "Inspect determinism/code/input state"))
    return pd.DataFrame(rows)

def write_latex_table(s, out):
    label={"A":"A: gate only","B":"B: gate + ModDrop","C":"C: gate + ModDrop + anchor"}
    L=[r"\begin{table}[t]",r"\centering",r"\caption{Repeated-seed A/B/C ablation on the tuning set. Values are mean $\pm$ SD across five training seeds.}",r"\label{tab:repeated_seed_ablation}",r"\small",r"\begin{tabular}{llccc}",r"\toprule",r"Variant & Clinical missing & C-index & Mean AUC & IBS \\",r"\midrule"]
    for V in "ABC":
        first=True
        for _,r in s[s.Variant==V].sort_values("Missing_percent").iterrows():
            vl=label[V] if first else ""; first=False
            c=f"{r.C_index_mean_across_seeds:.4f} $\\pm$ {r.C_index_sd_across_seeds:.4f}"
            a=f"{r.Mean_dynamic_AUC_mean_across_seeds:.4f} $\\pm$ {r.Mean_dynamic_AUC_sd_across_seeds:.4f}"
            b=f"{r.Integrated_Brier_Score_mean_across_seeds:.4f} $\\pm$ {r.Integrated_Brier_Score_sd_across_seeds:.4f}"
            L.append(f"{vl} & {int(r.Missing_percent)}\\% & {c} & {a} & {b} \\\\")
        if V!="C": L.append(r"\addlinespace")
    L += [r"\bottomrule",r"\end{tabular}",r"\vspace{1mm}",r"\parbox{0.94\textwidth}{\footnotesize \textit{Note:} Split, corrected clinical representation, CT-only initialization, Cox--Efron objective, and all non-seed hyperparameters were fixed. The tuning set is the primary repeated-seed readout; held-out test results are descriptive only.}",r"\end{table}"]
    out.write_text("\n".join(L),encoding="utf-8")

def make_plots(s,outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs=[("C_index","C-index","RepeatedSeed_Cindex"),("Mean_dynamic_AUC","Mean dynamic AUC","RepeatedSeed_MeanAUC"),("Integrated_Brier_Score","Integrated Brier score","RepeatedSeed_IBS")]
    for pref,ylabel,stem in specs:
        fig,ax=plt.subplots(figsize=(6.2,4.2))
        for V in "ABC":
            g=s[s.Variant==V].sort_values("Missing_percent")
            ax.errorbar(g.Missing_percent,g[f"{pref}_mean_across_seeds"],yerr=g[f"{pref}_sd_across_seeds"],marker="o",capsize=3,label=g.Variant_label.iloc[0])
        ax.set_xlabel("Clinical modality missing (%)"); ax.set_ylabel(ylabel); ax.set_xticks(PRIMARY_MISSING); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(outdir/f"{stem}.pdf",bbox_inches="tight"); fig.savefig(outdir/f"{stem}.png",dpi=300,bbox_inches="tight"); plt.close(fig)

def write_summary(val,cs,test,args,out):
    L=["LUNG1 — REPEATED-SEED A/B/C ABLATION","",f"Seeds: {', '.join(map(str,args.seeds))}",f"N seeds: {len(args.seeds)}",f"Scenario repeats/model: {args.scenario_repeats}","","PRIMARY — TUNING/VALIDATION"]
    for pct in PRIMARY_MISSING:
        L.append(f"\nClinical missing {pct}%:")
        for _,r in val[val.Missing_percent==pct].iterrows():
            L.append(f"  {r.Variant} {r.Variant_label}: C={r.C_index_mean_across_seeds:.4f}±{r.C_index_sd_across_seeds:.4f}; AUC={r.Mean_dynamic_AUC_mean_across_seeds:.4f}±{r.Mean_dynamic_AUC_sd_across_seeds:.4f}; IBS={r.Integrated_Brier_Score_mean_across_seeds:.4f}±{r.Integrated_Brier_Score_sd_across_seeds:.4f}")
    L += ["","PAIRED COMPONENT CONTRASTS — TUNING/VALIDATION","B-A: adding ModDrop with anchor off. C-B: adding anchor with ModDrop on."]
    for _,r in cs[cs.Subset=="validation"].iterrows():
        L.append(f"  missing={int(r.Missing_percent)}% | {r.Metric}: B-A={r.Delta_B_minus_A_mean:+.4f}±{r.Delta_B_minus_A_sd:.4f}; C-B={r.Delta_C_minus_B_mean:+.4f}±{r.Delta_C_minus_B_sd:.4f}")
    L += ["","SECONDARY — HELD-OUT TEST (DESCRIPTIVE ONLY)"]
    for pct in PRIMARY_MISSING:
        L.append(f"\nClinical missing {pct}%:")
        for _,r in test[test.Missing_percent==pct].iterrows():
            L.append(f"  {r.Variant} {r.Variant_label}: C={r.C_index_mean_across_seeds:.4f}±{r.C_index_sd_across_seeds:.4f}; AUC={r.Mean_dynamic_AUC_mean_across_seeds:.4f}±{r.Mean_dynamic_AUC_sd_across_seeds:.4f}; IBS={r.Integrated_Brier_Score_mean_across_seeds:.4f}±{r.Integrated_Brier_Score_sd_across_seeds:.4f}")
    L += ["","GUARDRAILS","- Do not select A/B/C from held-out test values.","- Emphasize magnitude/direction/SD rather than seed-level significance tests.","- This assesses optimization stability conditional on the same locked CT-only initialization and split.","- External validation remains a separate requirement."]
    out.write_text("\n".join(L),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trainer",type=Path,required=True); ap.add_argument("--vol-dir",type=Path,required=True); ap.add_argument("--split",type=Path,required=True)
    ap.add_argument("--clinical-dir",type=Path,required=True); ap.add_argument("--ct-checkpoint",type=Path,required=True); ap.add_argument("--out-root",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=[20260926,20260927,20260928,20260929,20260930])
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--freeze-epochs",type=int,default=5); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--fusion-lr",type=float,default=1e-4); ap.add_argument("--ct-lr",type=float,default=2e-6); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--dropout",type=float,default=0.30); ap.add_argument("--patience",type=int,default=12); ap.add_argument("--scenario-repeats",type=int,default=500)
    ap.add_argument("--resume",action="store_true"); ap.add_argument("--only-analyze",action="store_true")
    a=ap.parse_args()
    require(a.trainer,"trainer"); require(a.vol_dir,"CT volumes"); require(a.split,"locked corrected split"); require(a.ct_checkpoint,"CT checkpoint")
    for f in ["train_features.csv","validation_features.csv","test_features.csv"]: require(a.clinical_dir/f,f"clinical {f}")
    a.out_root.mkdir(parents=True,exist_ok=True)
    manifest=dict(study="LUNG1_repeated_seed_ABC_ablation",primary_readout="tuning 0/30/100%",secondary_readout="held-out test descriptive only",seeds=a.seeds,scenario_repeats=a.scenario_repeats,fixed_split="295/63/64",variants=VARIANTS,common_hyperparameters=dict(epochs=a.epochs,freeze_epochs=a.freeze_epochs,batch_size=a.batch_size,fusion_lr=a.fusion_lr,ct_lr=a.ct_lr,weight_decay=a.weight_decay,dropout=a.dropout,patience=a.patience))
    (a.out_root/"RepeatedSeed_Manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if not a.only_analyze:
        for seed in a.seeds:
            for v in VARIANTS:
                outdir=a.out_root/f"seed_{seed}"/v["folder"]; expected=outdir/"Robustness_Scenario_Summary.csv"
                if a.resume and expected.exists(): print(f"[RESUME] seed={seed} variant={v['key']} already complete"); continue
                outdir.mkdir(parents=True,exist_ok=True); run_cmd(trainer_command(a,outdir,seed,v),a.out_root/"logs"/f"seed_{seed}_variant_{v['key']}.log"); require(expected,"completed summary")
    parts=[]
    for seed in a.seeds:
        for v in VARIANTS: parts.append(load_summary(a.out_root/f"seed_{seed}"/v["folder"]/"Robustness_Scenario_Summary.csv",seed,v))
    all_df=pd.concat(parts,ignore_index=True); all_df.to_csv(a.out_root/"RepeatedSeed_AllScenarios_PerSeed.csv",index=False,encoding="utf-8-sig")
    all_df[all_df.Missing_percent.isin(PRIMARY_MISSING)].to_csv(a.out_root/"RepeatedSeed_Primary_0_30_100_PerSeed.csv",index=False,encoding="utf-8-sig")
    val=descriptive_summary(all_df,"validation"); test=descriptive_summary(all_df,"test")
    val.to_csv(a.out_root/"RepeatedSeed_Validation_Summary.csv",index=False,encoding="utf-8-sig"); test.to_csv(a.out_root/"RepeatedSeed_Test_Descriptive_Summary.csv",index=False,encoding="utf-8-sig")
    con=pd.concat([paired_contrasts(all_df,"validation"),paired_contrasts(all_df,"test")],ignore_index=True); con.to_csv(a.out_root/"RepeatedSeed_ComponentContrasts_PerSeed.csv",index=False,encoding="utf-8-sig")
    cs=summarize_contrasts(con); cs.to_csv(a.out_root/"RepeatedSeed_ComponentContrasts_Summary.csv",index=False,encoding="utf-8-sig")
    qc=seed0_qc(all_df,a.seeds[0]); qc.to_csv(a.out_root/"RepeatedSeed_Seed0_Reproducibility_QC.csv",index=False,encoding="utf-8-sig")
    write_latex_table(val,a.out_root/"RepeatedSeed_Manuscript_Table.tex"); make_plots(val,a.out_root); write_summary(val,cs,test,a,a.out_root/"RepeatedSeed_Summary.txt")
    print("\n"+(a.out_root/"RepeatedSeed_Summary.txt").read_text(encoding="utf-8")); print("\nCompleted:",a.out_root)
if __name__=="__main__": main()
