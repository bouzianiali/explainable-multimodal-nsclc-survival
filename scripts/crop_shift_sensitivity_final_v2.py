#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc crop-shift sensitivity for locked LUNG1 gated residual model.
No training/tuning/recalibration. Integer translations of the stored 1-mm isotropic
CT are padded with -1000 HU and used as a controlled proxy for crop-centering error.
"""
import argparse, importlib.util, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from scipy.stats import spearmanr


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_module(path):
    spec = importlib.util.spec_from_file_location("lung1_trainer", str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import trainer: {path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def normalize_ct(x):
    return np.clip((x.astype(np.float32)+1000.0)/1400.0, 0.0, 1.0).astype(np.float32, copy=False)


def shift3d_airpad(a, dz, dy, dx, fill=-1000.0):
    out = np.full_like(a, fill, dtype=np.float32)
    src=[]; dst=[]
    for n,s in zip(a.shape,(int(dz),int(dy),int(dx))):
        if abs(s)>=n: return out
        if s>=0: src.append(slice(0,n-s)); dst.append(slice(s,n))
        else: src.append(slice(-s,n)); dst.append(slice(0,n+s))
    out[tuple(dst)] = a[tuple(src)]
    return out


def cindex(t,e,r):
    return float(concordance_index(np.asarray(t,float), -np.asarray(r,float), event_observed=np.asarray(e,int)))


def boot_delta_c(t,e,r0,r1,n_boot,seed):
    rng=np.random.default_rng(seed); vals=[]; n=len(t)
    t=np.asarray(t,float); e=np.asarray(e,int); r0=np.asarray(r0,float); r1=np.asarray(r1,float)
    for _ in range(int(n_boot)):
        idx=rng.integers(0,n,size=n)
        try:
            d=cindex(t[idx],e[idx],r1[idx])-cindex(t[idx],e[idx],r0[idx])
            if np.isfinite(d): vals.append(d)
        except Exception: pass
    if not vals: return 0,np.nan,np.nan,np.nan
    v=np.asarray(vals,float)
    p=min(1.0,2.0*min((np.sum(v<=0)+1)/(len(v)+1),(np.sum(v>=0)+1)/(len(v)+1)))
    return len(v),float(np.percentile(v,2.5)),float(np.percentile(v,97.5)),float(p)


def conditions(mags):
    rows=[("baseline",0,0,0,0,"baseline")]
    for m in mags:
        for d,v in [("z+",(m,0,0)),("z-",(-m,0,0)),("y+",(0,m,0)),("y-",(0,-m,0)),("x+",(0,0,m)),("x-",(0,0,-m))]:
            rows.append((f"{d}_{m}mm",m,*v,d))
    return pd.DataFrame(rows,columns=["condition","magnitude_mm","dz","dy","dx","direction"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trainer",required=True); ap.add_argument("--vol-dir",required=True)
    ap.add_argument("--clinical-dir",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--model-dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--device",choices=["cuda","cpu","auto"],default="cuda")
    ap.add_argument("--magnitudes-mm",nargs="+",type=int,default=[5,10,20])
    ap.add_argument("--bootstrap",type=int,default=2000); ap.add_argument("--seed",type=int,default=20260926)
    ap.add_argument("--risk-qc-tol",type=float,default=1e-5)
    ap.add_argument("--max-patients",type=int,default=None,
                    help="Optional smoke-test limit. Omit for the final 64-patient run.")
    a=ap.parse_args(); seed_all(a.seed)

    trainer=Path(a.trainer); vol_dir=Path(a.vol_dir); clinical_dir=Path(a.clinical_dir)
    ckpt_path=Path(a.checkpoint); model_dir=Path(a.model_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for p in [trainer,vol_dir,clinical_dir,ckpt_path,model_dir]:
        if not p.exists(): raise FileNotFoundError(p)
    vols=sorted(vol_dir.glob("*_standardized_192x160x224.npz"))
    if len(vols)!=422: raise RuntimeError(f"Expected 422 volumes, found {len(vols)} in {vol_dir}")

    if a.device=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        device=torch.device("cuda")
    elif a.device=="cpu": device=torch.device("cpu")
    else: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod=load_module(trainer)
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    feats=list(ckpt["clinical_feature_cols"])
    if len(feats)!=21: raise RuntimeError(f"Expected 21 clinical features, got {len(feats)}")
    forbidden=["T.Stage_5","T.Stage_5.0","N.Stage_4","N.Stage_4.0","M.Stage_3","M.Stage_3.0"]
    bad=[c for c in feats if any(x in c for x in forbidden)]
    if bad: raise RuntimeError(f"Pre-correction TNM features detected: {bad}")
    for c in ["clinical.T.Stage_MISSING","Clinical.N.Stage_MISSING","Clinical.M.Stage_MISSING"]:
        if c not in feats: raise RuntimeError(f"Missing expected feature {c}")

    test=pd.read_csv(clinical_dir/"test_features.csv"); test["PatientID"]=test["PatientID"].astype(str)
    if len(test)!=64: raise RuntimeError(f"Expected 64 test patients, got {len(test)}")
    if a.max_patients is not None:
        if a.max_patients < 1 or a.max_patients > 64:
            raise ValueError("--max-patients must be between 1 and 64.")
        test=test.iloc[:a.max_patients].copy()
    train_modes=pd.read_csv(model_dir/"train_risk_modes.csv"); saved_test=pd.read_csv(model_dir/"test_risk_modes.csv")
    train_modes["PatientID"]=train_modes["PatientID"].astype(str); saved_test["PatientID"]=saved_test["PatientID"].astype(str)

    dropout=float(ckpt.get("args",{}).get("dropout",0.30))
    model=mod.RobustFusionV2(clinical_dim=len(feats),dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"],strict=True); model=model.to(device).eval()
    cond=conditions(a.magnitudes_mm)

    raw_path=out/"CropShift_RawPredictions.csv"
    requested_conditions=set(cond["condition"].astype(str))
    requested_patients=set(test["PatientID"].astype(str))

    cached=pd.DataFrame()
    if raw_path.exists():
        try:
            cached=pd.read_csv(raw_path)
            cached["PatientID"]=cached["PatientID"].astype(str)
            cached=cached[
                cached["PatientID"].isin(requested_patients)
                & cached["condition"].astype(str).isin(requested_conditions)
            ].copy()
            cached=cached.drop_duplicates(["PatientID","condition"],keep="last")
        except Exception as exc:
            print("WARNING: existing raw cache could not be read; recomputing.", exc)
            cached=pd.DataFrame()

    rows=cached.to_dict("records") if len(cached) else []
    done={(str(r["PatientID"]),str(r["condition"])) for r in rows}

    print("="*80); print("LUNG1 CROP-SHIFT SENSITIVITY")
    print("Patients:",len(test)); print("Conditions:",len(cond)); print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    print("TNM-corrected audit: PASS"); print("="*80)

    with torch.inference_mode():
        for i,r in test.reset_index(drop=True).iterrows():
            pid=str(r["PatientID"]); p=vol_dir/f"{pid}_standardized_192x160x224.npz"
            z=np.load(p,allow_pickle=False); ct0=np.asarray(z["ct_hu_clipped"],dtype=np.float32)
            if tuple(ct0.shape)!=(192,160,224): raise RuntimeError(f"{pid}: shape={ct0.shape}")
            cl=torch.from_numpy(r[feats].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
            avail=torch.ones((1,1),dtype=cl.dtype,device=device)
            new_for_patient=0
            for _,c in cond.iterrows():
                key=(pid,str(c["condition"]))
                if key in done:
                    continue
                ct=ct0 if int(c["magnitude_mm"])==0 else shift3d_airpad(
                    ct0,int(c["dz"]),int(c["dy"]),int(c["dx"])
                )
                x=torch.from_numpy(normalize_ct(ct)).unsqueeze(0).unsqueeze(0).to(device)
                rf,base,g,res=model(x,cl,avail)
                rows.append(dict(
                    PatientID=pid,
                    Survival_time=float(r["Survival.time"]),
                    event=int(r["deadstatus.event"]),
                    condition=str(c["condition"]),
                    magnitude_mm=int(c["magnitude_mm"]),
                    direction=str(c["direction"]),
                    dz=int(c["dz"]),dy=int(c["dy"]),dx=int(c["dx"]),
                    risk_full=float(rf.item()),
                    risk_no_clinical=float(base.item()),
                    gate_value=float(g.item()),
                    clinical_residual=float(res.item())
                ))
                done.add(key)
                new_for_patient += 1

            pd.DataFrame(rows).drop_duplicates(
                ["PatientID","condition"],keep="last"
            ).to_csv(raw_path,index=False,encoding="utf-8-sig")
            print(
                f"[{i+1:02d}/{len(test):02d}] {pid} | new={new_for_patient} | cached={len(cond)-new_for_patient}",
                flush=True
            )

    pred=pd.DataFrame(rows).drop_duplicates(["PatientID","condition"],keep="last")
    pred=pred[
        pred["PatientID"].astype(str).isin(requested_patients)
        & pred["condition"].astype(str).isin(requested_conditions)
    ].copy()
    pred.to_csv(raw_path,index=False,encoding="utf-8-sig")

    expected=len(test)*len(cond)
    if len(pred)!=expected:
        raise RuntimeError(
            f"Incomplete raw predictions: expected {expected} patient-condition rows, found {len(pred)}."
        )
    base=pred[pred.condition=="baseline"].copy()
    qc=base.merge(saved_test[["PatientID","risk_full","risk_no_clinical"]],on="PatientID",suffixes=("_recomputed","_saved"),validate="one_to_one")
    qc["abs_err_full"]=np.abs(qc.risk_full_recomputed-qc.risk_full_saved)
    qc["abs_err_no_clinical"]=np.abs(qc.risk_no_clinical_recomputed-qc.risk_no_clinical_saved)
    qc.to_csv(out/"CropShift_RiskQC.csv",index=False,encoding="utf-8-sig")
    mf=float(qc.abs_err_full.max()); mn=float(qc.abs_err_no_clinical.max())
    if max(mf,mn)>a.risk_qc_tol: raise RuntimeError(f"Risk QC failed: {mf:.3e}, {mn:.3e}")

    metrics=[]; horizons=[365,730,1095]; baseidx=base.set_index("PatientID")
    for mode,rcol in [("full_clinical","risk_full"),("no_clinical","risk_no_clinical")]:
        tr=train_modes[["PatientID","Survival.time","deadstatus.event",rcol]].copy().rename(columns={rcol:"risk_score"})
        r0=baseidx[rcol]
        for _,c in cond.iterrows():
            g=pred[pred.condition==c.condition].copy()
            ev=g[["PatientID","Survival_time","event",rcol]].rename(columns={"Survival_time":"Survival.time","event":"deadstatus.event",rcol:"risk_score"})
            m,_=mod.full_metrics(tr,ev,horizons)
            rr=g.set_index("PatientID").loc[r0.index,rcol]
            rho=float(spearmanr(r0.to_numpy(float),rr.to_numpy(float)).statistic)
            metrics.append(dict(mode=mode,condition=c.condition,magnitude_mm=int(c.magnitude_mm),direction=c.direction,
                                C_index=m["C_index"],Mean_dynamic_AUC=m["Mean_dynamic_AUC"],Integrated_Brier_Score=m["Integrated_Brier_Score"],
                                Risk_Spearman_vs_baseline=rho,Risk_MAE_vs_baseline=float(np.mean(np.abs(rr-r0))),
                                Risk_MaxAbs_vs_baseline=float(np.max(np.abs(rr-r0)))))
    pc=pd.DataFrame(metrics)
    for mode in pc["mode"].unique():
        b=pc[(pc["mode"]==mode)&(pc.condition=="baseline")].iloc[0]; idx=pc["mode"]==mode
        pc.loc[idx,"Delta_C_vs_baseline"]=pc.loc[idx,"C_index"]-b.C_index
        pc.loc[idx,"Delta_AUC_vs_baseline"]=pc.loc[idx,"Mean_dynamic_AUC"]-b.Mean_dynamic_AUC
        pc.loc[idx,"Delta_IBS_vs_baseline"]=pc.loc[idx,"Integrated_Brier_Score"]-b.Integrated_Brier_Score
    pc.to_csv(out/"CropShift_PerCondition.csv",index=False,encoding="utf-8-sig")

    mags=[]
    for mode in pc["mode"].unique():
        b=pc[(pc["mode"]==mode)&(pc.condition=="baseline")].iloc[0]
        mags.append(dict(mode=mode,magnitude_mm=0,n_directions=1,C_index_mean=b.C_index,C_index_min=b.C_index,C_index_max=b.C_index,
                         Mean_dynamic_AUC_mean=b.Mean_dynamic_AUC,Mean_dynamic_AUC_min=b.Mean_dynamic_AUC,Mean_dynamic_AUC_max=b.Mean_dynamic_AUC,
                         IBS_mean=b.Integrated_Brier_Score,IBS_min=b.Integrated_Brier_Score,IBS_max=b.Integrated_Brier_Score,
                         Risk_Spearman_mean=1.0,Risk_Spearman_min=1.0,Risk_MAE_mean=0.0,Worst_Delta_C=0.0,Worst_Delta_AUC=0.0,Worst_Delta_IBS=0.0))
        for m in a.magnitudes_mm:
            g=pc[(pc["mode"]==mode)&(pc.magnitude_mm==m)]
            mags.append(dict(mode=mode,magnitude_mm=m,n_directions=len(g),C_index_mean=float(g.C_index.mean()),C_index_min=float(g.C_index.min()),C_index_max=float(g.C_index.max()),
                             Mean_dynamic_AUC_mean=float(g.Mean_dynamic_AUC.mean()),Mean_dynamic_AUC_min=float(g.Mean_dynamic_AUC.min()),Mean_dynamic_AUC_max=float(g.Mean_dynamic_AUC.max()),
                             IBS_mean=float(g.Integrated_Brier_Score.mean()),IBS_min=float(g.Integrated_Brier_Score.min()),IBS_max=float(g.Integrated_Brier_Score.max()),
                             Risk_Spearman_mean=float(g.Risk_Spearman_vs_baseline.mean()),Risk_Spearman_min=float(g.Risk_Spearman_vs_baseline.min()),Risk_MAE_mean=float(g.Risk_MAE_vs_baseline.mean()),
                             Worst_Delta_C=float(g.Delta_C_vs_baseline.min()),Worst_Delta_AUC=float(g.Delta_AUC_vs_baseline.min()),Worst_Delta_IBS=float(g.Delta_IBS_vs_baseline.max())))
    bm=pd.DataFrame(mags); bm.to_csv(out/"CropShift_ByMagnitude.csv",index=False,encoding="utf-8-sig")

    boots=[]
    for mode,rcol in [("full_clinical","risk_full"),("no_clinical","risk_no_clinical")]:
        b=base.sort_values("PatientID"); t=b.Survival_time.to_numpy(float); e=b.event.to_numpy(int); r0=b[rcol].to_numpy(float)
        shifted=cond[cond.condition!="baseline"].reset_index(drop=True)
        for j,c in shifted.iterrows():
            g=pred[pred.condition==c.condition].sort_values("PatientID"); r1=g[rcol].to_numpy(float)
            nboot,lo,hi,p=boot_delta_c(t,e,r0,r1,a.bootstrap,a.seed+10000+j+(0 if mode=="full_clinical" else 1000))
            boots.append(dict(mode=mode,condition=c.condition,magnitude_mm=int(c.magnitude_mm),direction=c.direction,
                              Delta_C_point=cindex(t,e,r1)-cindex(t,e,r0),valid_boot=nboot,ci_low=lo,ci_high=hi,sign_p=p))
    pd.DataFrame(boots).to_csv(out/"CropShift_PairedBootstrap_CIndex.csv",index=False,encoding="utf-8-sig")

    lines=["LUNG1 — FINAL CROP-SHIFT SENSITIVITY OF LOCKED TNM-CORRECTED MODEL","",f"Patients evaluated: {len(test)}",f"Device: {device}",
           "Shift magnitudes: "+", ".join(map(str,a.magnitudes_mm))+" mm","Directions: z+/z-/y+/y-/x+/x-","Voxel spacing: 1 mm isotropic.",
           "Implementation: integer translation of standardized CT with -1000 HU padding.","No model was retrained, tuned, recalibrated, or selected.","",
           "LOCKED-RISK QC",f"  max |recomputed-saved| full={mf:.3e}",f"  max |recomputed-saved| no-clinical={mn:.3e}","  PASS","","BY-MAGNITUDE SUMMARY"]
    for mode in ["full_clinical","no_clinical"]:
        lines += ["",mode.upper()]
        for _,r in bm[bm["mode"]==mode].sort_values("magnitude_mm").iterrows():
            lines.append(f"  {int(r.magnitude_mm):>2d} mm | C mean/min={r.C_index_mean:.4f}/{r.C_index_min:.4f} | AUC mean/min={r.Mean_dynamic_AUC_mean:.4f}/{r.Mean_dynamic_AUC_min:.4f} | IBS mean/max={r.IBS_mean:.4f}/{r.IBS_max:.4f} | rho mean/min={r.Risk_Spearman_mean:.4f}/{r.Risk_Spearman_min:.4f}")
    lines += ["","INTERPRETATION GUARDRAILS","1) Controlled post-hoc sensitivity analysis only.","2) Test set not used for training/tuning/checkpoint selection.",
              "3) Stored-volume translation is a proxy for crop-centering error, not validation of an automatic detector/segmenter.",
              "4) Report degradation honestly; do not claim invariance if performance changes.","5) Paired bootstrap is for C-index delta; AUC/IBS are descriptive sensitivity metrics."]
    s="\n".join(lines); (out/"CropShift_Summary.txt").write_text(s,encoding="utf-8"); print("\n"+s); print("\nFiles written to:",out)

if __name__=="__main__": main()
