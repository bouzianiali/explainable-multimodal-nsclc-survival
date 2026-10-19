#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correct non-standard LUNG1 TNM sentinel codes before clinical modelling.

Recode:
    clinical.T.Stage == 5  -> missing
    Clinical.N.Stage == 4  -> missing
    Clinical.M.Stage == 3  -> missing

The patient split is NOT regenerated.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(args.clinical)
    split = pd.read_csv(args.split)

    raw = clinical.copy()

    clinical.loc[clinical["clinical.T.Stage"].eq(5), "clinical.T.Stage"] = np.nan
    clinical.loc[clinical["Clinical.N.Stage"].eq(4), "Clinical.N.Stage"] = np.nan
    clinical.loc[clinical["Clinical.M.Stage"].eq(3), "Clinical.M.Stage"] = np.nan

    # Locked split labels are preserved.
    clinical_cols = [
        "age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage", "Histology", "gender",
        "Survival.time", "deadstatus.event",
    ]
    fixed_split = split.drop(columns=[c for c in clinical_cols if c in split.columns]).merge(
        clinical[["PatientID"] + clinical_cols],
        on="PatientID",
        how="left",
        validate="one_to_one",
    )

    fixed_split.to_csv(out / "LUNG1_Split_Assignment_TNM_CORRECTED.csv", index=False)
    clinical.to_csv(out / "NSCLC-Radiomics-Lung1.clinical-TNM-corrected.csv", index=False)

    print("Corrected TNM counts:")
    print("T:", clinical["clinical.T.Stage"].value_counts(dropna=False).sort_index())
    print("N:", clinical["Clinical.N.Stage"].value_counts(dropna=False).sort_index())
    print("M:", clinical["Clinical.M.Stage"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
