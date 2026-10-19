# Explainable multimodal NSCLC survival prediction under clinical-modality absence

Companion reproducibility repository for the manuscript **“Explainable Multimodal Survival Prediction in NSCLC under Clinical-Modality Absence: Development and Internal Evaluation on LUNG1.”**

## What this repository contains

- locked patient **identifiers + split role** (295 train / 63 tuning / 64 held-out internal test);
- TNM sentinel recoding rule (`T=5`, `N=4`, `M=3` -> `MISSING`);
- CT preprocessing code and protocol metadata;
- clinical feature schema (21 transformed features; transformations fit on training only);
- CT-only, standard concatenation, missingness-aware concatenation, and gated-residual A/B/C model code;
- exact full-training-cohort Cox partial likelihood with **Efron ties** using two-pass risk replay;
- controlled clinical-modality missingness analyses;
- paired bootstrap model comparison and calibration;
- Grad-CAM / Grad-CAM++, perturbation faithfulness, clinical SHAP, and multimodal XAI scripts;
- crop-shift sensitivity analysis;
- repeated-seed A/B/C orchestration and the corrected five-seed aggregate outputs.

## Source data are not redistributed

The LUNG1 CT images, tumor segmentations, survival outcomes, and clinical tables are **not included**. Download the source data directly from TCIA:

> Aerts HJWL, Wee L, Rios Velazquez E, et al. *Data From NSCLC-Radiomics (version 4).* The Cancer Imaging Archive; 2014. DOI: **10.7937/K9/TCIA.2015.PF0M9REI**.

Follow the current TCIA license and required data citation. See `THIRD_PARTY_DATA_NOTICE.md`.

## Locked analysis design

| Role | n | events |
|---|---:|---:|
| Training | 295 | 261 |
| Tuning | 63 | 56 |
| Held-out internal test | 64 | 56 |

The test set was not used for fitting, early stopping, hyperparameter selection, or A/B/C selection. Repeated-seed component analysis used the tuning set as the primary readout; test results were descriptive.

## CT preprocessing

1. clip HU to `[-1000, 400]`;
2. tumor-centered crop with a **20 mm** margin;
3. resample to **1 mm isotropic** spacing;
4. standardize to `192 x 160 x 224` in `[z,y,x]` order;
5. pad outside the source field with `-1000 HU`;
6. map clipped HU linearly to `[0,1]` before network input.

The tumor mask is used for crop definition, QC, and XAI localization only; it is not a predictive input channel. This dependence on tumor localization is an explicit deployment limitation.

## Clinical preprocessing

Predictors: age, sex, clinical T/N/M, and histology. Overall stage is descriptive only. Source sentinel-like values `T=5`, `N=4`, and `M=3` are recoded to missing before modeling. Age is median-imputed and standardized from the training set. Categorical variables use explicit missing categories and train-fitted one-hot encoding. The final representation has 21 features (`data/clinical_feature_schema_21.txt`).

## A/B/C variants

- **A — gate only:** ModDrop `0.00`, CT anchor `0.00`.
- **B — gate + ModDrop:** ModDrop `0.25`, CT anchor `0.00`.
- **C — full proposed:** ModDrop `0.25`, CT anchor `1.00`.

All other hyperparameters and the locked CT-only initialization are fixed. The five predefined training seeds are `20260926`–`20260930`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for your driver/runtime. The reported GPU runs were performed in Google Colab on NVIDIA Tesla T4 accelerators when available. Final survival evaluation used `lifelines==0.30.3` and `scikit-survival==0.28.0`.

## Expected local data layout

```text
DATA_ROOT/
  volumes/
    LUNG1-001_standardized_192x160x224.npz
    ...
  clinical_baselines/model_A/
    train_features.csv
    validation_features.csv
    test_features.csv
  LUNG1_Split_Assignment_TNM_CORRECTED.csv
  ct_only_fullcox_efron/best_model.pt
```

The clinical feature CSVs and CT checkpoint are generated locally from TCIA data; they are not committed here.

## Main commands

CT preprocessing:

```bash
python scripts/preprocess_lung1_3d.py --help
```

TNM correction:

```bash
python scripts/prepare_lung1_tnm_corrected.py --help
```

Repeated-seed A/B/C study:

```bash
python scripts/run_ablation_repeated_seeds_final.py \\
  --trainer scripts/train_gated_moddrop_v2_fullcox_efron.py \\
  --vol-dir DATA_ROOT/volumes \\
  --split DATA_ROOT/LUNG1_Split_Assignment_TNM_CORRECTED.csv \\
  --clinical-dir DATA_ROOT/clinical_baselines/model_A \\
  --ct-checkpoint DATA_ROOT/ct_only_fullcox_efron/best_model.pt \\
  --out-root outputs/repeated_seed_ABC \\
  --epochs 60 --freeze-epochs 5 --batch-size 2 \\
  --fusion-lr 1e-4 --ct-lr 2e-6 --weight-decay 1e-4 --dropout 0.30 \\
  --patience 12 --scenario-repeats 500 \\
  --seeds 20260926 20260927 20260928 20260929 20260930
```

Clinical train-only preprocessing and Cox baseline:

```bash
python scripts/prepare_clinical_features_and_cox.py \
  --split DATA_ROOT/LUNG1_Split_Assignment_TNM_CORRECTED.csv \
  --out DATA_ROOT/clinical_baselines
```

Other analysis scripts are in `scripts/` and expose `--help` where applicable.

## Public release and manuscript links

- GitHub: **https://github.com/bouzianiali/explainable-multimodal-nsclc-survival**
- Zenodo archive DOI: **https://doi.org/10.5281/zenodo.23483737**

See `ZENODO_RELEASE.md` for the final archive metadata and publication checklist.


## Creators and citation

- **Ali Bouziani** — ORCID: https://orcid.org/0009-0002-2940-6488
- **Naji Abdelwahab**
- Affiliation: Mohammed First University, National School of Applied Sciences of Oujda (ENSAO), Mathematics, Computer Science and Mechanics Team, Oujda, Morocco.

Please cite the archived software release using DOI **10.5281/zenodo.23483737**. Machine-readable citation metadata are provided in `CITATION.cff`.

## Software provenance

Most bundled analysis scripts are archived final study scripts. The gated full-cohort Cox trainer is a public-release implementation reconstructed from the locked protocol and saved run metadata rather than a byte-for-byte copy of the original trainer. See `PUBLIC_RELEASE_NOTES.md` for the precise provenance boundary.

## Reproducibility scope

The five-seed experiment quantifies optimization variability conditional on a fixed patient split and a fixed CT-only initialization. It does **not** replace repeated-split validation or independent external validation. The manuscript therefore treats external validation as the principal next step before clinical translation.

## License

Code in this repository is released under the MIT License. Third-party LUNG1 data retain their original TCIA terms and are not redistributed.
