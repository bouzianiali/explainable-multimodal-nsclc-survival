#!/usr/bin/env python
"""Prepare the locked 21-feature clinical representation and a clinical Cox baseline.

Input is the locally reconstructed TNM-corrected LUNG1 split table. Source clinical
records are not distributed by this repository; obtain them from TCIA and generate the
corrected split using `prepare_lung1_tnm_corrected.py`.

Transforms are fit on training only:
- age: median imputation + standardization;
- categorical predictors: explicit MISSING category + fixed one-hot schema;
- predictors: age, sex, clinical T/N/M, histology;
- overall stage is not used as a predictor.

Outputs train_features.csv, validation_features.csv, test_features.csv and a clinical
Cox summary. The explicit schema matches `data/clinical_feature_schema_21.txt`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

PRED = ["age", "gender", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage", "Histology"]
CAT_LEVELS = {
    "gender": ["female", "male"],
    "clinical.T.Stage": ["1.0", "2.0", "3.0", "4.0", "MISSING"],
    "Clinical.N.Stage": ["0.0", "1.0", "2.0", "3.0", "MISSING"],
    "Clinical.M.Stage": ["0.0", "1.0", "MISSING"],
    "Histology": ["MISSING", "adenocarcinoma", "large cell", "nos", "squamous cell carcinoma"],
}
EXPECTED = [
    "age","gender_female","gender_male",
    "clinical.T.Stage_1.0","clinical.T.Stage_2.0","clinical.T.Stage_3.0","clinical.T.Stage_4.0","clinical.T.Stage_MISSING",
    "Clinical.N.Stage_0.0","Clinical.N.Stage_1.0","Clinical.N.Stage_2.0","Clinical.N.Stage_3.0","Clinical.N.Stage_MISSING",
    "Clinical.M.Stage_0.0","Clinical.M.Stage_1.0","Clinical.M.Stage_MISSING",
    "Histology_MISSING","Histology_adenocarcinoma","Histology_large cell","Histology_nos","Histology_squamous cell carcinoma",
]

def norm_cat(x):
    if pd.isna(x): return "MISSING"
    return str(x).strip().lower() if not isinstance(x,(int,float,np.number)) else str(float(x))

def fit_stats(train: pd.DataFrame):
    age = pd.to_numeric(train["age"], errors="coerce")
    med = float(age.median())
    filled = age.fillna(med)
    mean = float(filled.mean())
    sd = float(filled.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0: sd = 1.0
    return {"age_median": med, "age_mean_after_imputation": mean, "age_sd_after_imputation": sd}

def transform(df: pd.DataFrame, stats):
    out = pd.DataFrame(index=df.index)
    age = pd.to_numeric(df["age"], errors="coerce").fillna(stats["age_median"])
    out["age"] = (age - stats["age_mean_after_imputation"]) / stats["age_sd_after_imputation"]
    for c, levels in CAT_LEVELS.items():
        s = df[c].map(norm_cat)
        # Preserve known histology labels as written after lower-casing; unknown categories map to MISSING.
        s = s.where(s.isin(levels), "MISSING")
        for level in levels:
            out[f"{c}_{level}"] = (s == level).astype(float)
    out = out[EXPECTED].astype(float)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="TNM-corrected split table generated locally from TCIA data")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.split)
    required={"PatientID","Split","Survival.time","deadstatus.event",*PRED}
    miss=sorted(required.difference(df.columns))
    if miss: raise RuntimeError(f"Missing required columns: {miss}")
    role=df["Split"].astype(str).str.lower().replace({"val":"validation","tune":"validation","tuning":"validation"})
    df=df.copy(); df["_role"]=role
    train=df[df._role.eq("train")]
    if len(train)!=295: print(f"WARNING: expected 295 training rows, got {len(train)}")
    stats=fit_stats(train)
    outdir=Path(args.out); model_dir=outdir/"model_A"; model_dir.mkdir(parents=True,exist_ok=True)
    subsets={"train":"train","validation":"validation","test":"test"}
    transformed={}
    for name,r in subsets.items():
        d=df[df._role.eq(r)].copy()
        X=transform(d,stats)
        z=pd.concat([d[["PatientID","Survival.time","deadstatus.event"]].reset_index(drop=True), X.reset_index(drop=True)],axis=1)
        z.to_csv(model_dir/f"{name}_features.csv",index=False)
        transformed[name]=z
    # Clinical Cox baseline uses the same transformed representation. Penalizer is small only for numerical stability.
    fit=transformed["train"].drop(columns=["PatientID"]).copy()
    cph=CoxPHFitter(penalizer=1e-6)
    cph.fit(fit,duration_col="Survival.time",event_col="deadstatus.event",show_progress=False)
    rows=[]
    for name,z in transformed.items():
        risk=np.asarray(cph.predict_partial_hazard(z.drop(columns=["PatientID","Survival.time","deadstatus.event"]))).reshape(-1)
        c=float(concordance_index(z["Survival.time"], -risk, z["deadstatus.event"]))
        rows.append({"Subset":name,"N":len(z),"Events":int(z["deadstatus.event"].sum()),"C_index":c})
    pd.DataFrame(rows).to_csv(outdir/"LUNG1_Clinical_Cox_Results.csv",index=False)
    (outdir/"clinical_preprocessing.json").write_text(json.dumps({**stats,"features":EXPECTED},indent=2),encoding="utf-8")
    print("Wrote",model_dir,"with",len(EXPECTED),"features")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
