# Zenodo archive metadata and final publication checklist

## Fixed release identifiers

- **GitHub repository:** https://github.com/bouzianiali/explainable-multimodal-nsclc-survival
- **Reserved Zenodo DOI:** 10.5281/zenodo.23483737
- **DOI URL:** https://doi.org/10.5281/zenodo.23483737
- **Version:** 1.0.0
- **Resource type:** Software
- **License:** MIT

## Creators

1. **Ali Bouziani** — ORCID: https://orcid.org/0009-0002-2940-6488
2. **Naji Abdelwahab**

Affiliation for both creators:

> Mohammed First University, National School of Applied Sciences of Oujda (ENSAO), Mathematics, Computer Science and Mechanics Team, Oujda, Morocco

## Recommended Zenodo title

**Explainable Multimodal Survival Prediction in NSCLC under Clinical-Modality Absence: Development and Internal Evaluation on LUNG1**

## Recommended description

Software and reproducibility materials supporting a retrospective methodological study of multimodal overall-survival prediction in non-small cell lung cancer (NSCLC) using pretreatment CT and clinical data from the public LUNG1 cohort. The archive includes clinical-data preparation rules, locked patient-split identifiers, CT preprocessing, survival-model training and evaluation scripts, controlled clinical-modality missingness experiments, calibration analyses, explainability analyses, crop-shift sensitivity analysis, and five-seed ablation aggregation. Original LUNG1 imaging and clinical data are not redistributed and should be obtained directly from The Cancer Imaging Archive (TCIA).

## Recommended keywords

NSCLC; non-small cell lung cancer; survival analysis; multimodal learning; deep learning; computed tomography; explainable AI; missing modality; Cox proportional hazards; LUNG1; medical imaging.

## Publication checklist

1. Set **Publication date** to the actual day the Zenodo record is published.
2. Confirm **Software**, version `1.0.0`, **MIT License**, language **English**, programming language **Python**, and development status **Active**.
3. Confirm repository URL: https://github.com/bouzianiali/explainable-multimodal-nsclc-survival
4. Upload the finalized `v1.0.0` repository ZIP.
5. Run `python scripts/check_public_release.py` and require PASS before upload.
6. Preview the Zenodo record and verify author order, affiliation, DOI, title, description, keywords, and file list.
7. Publish only after these checks are complete. The reserved DOI becomes active when the record is published.

## Data boundary

LUNG1 CT images, segmentations, raw clinical tables, patient-level outcomes, and trained checkpoints are intentionally excluded. Source data must be obtained from TCIA and cited according to TCIA requirements.
