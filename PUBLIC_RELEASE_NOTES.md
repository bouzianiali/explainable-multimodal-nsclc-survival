# Public release notes and provenance

This repository is a public reproducibility release for the associated LUNG1 manuscript. It intentionally excludes TCIA source images, segmentations, raw clinical tables, patient-level outcomes, and trained checkpoints.

## Script provenance

Most analysis scripts in `scripts/` are archived study scripts used in the reported workflow or their TNM-corrected final versions. The five-seed aggregate tables in `results/repeated_seed_5seed/` were reconstructed directly from the 15 completed A/B/C run outputs after an orchestration-side root-summary overwrite was identified; the included audit file documents this correction.

The file `scripts/train_gated_moddrop_v2_fullcox_efron.py` is a **public-release implementation reconstructed from the locked protocol and saved run metadata**, rather than a byte-for-byte copy of the original training file. It implements the reported architecture and protocol: the explicit CT-only fallback, whole-clinical ModDrop, CT-teacher anchor, five-epoch CT freeze, final-block fine-tuning, exact two-pass full-training-cohort Cox partial likelihood with Efron ties, and tuning-only checkpoint selection. Parameter counts and A/B/C definitions are checked in `scripts/check_public_release.py` and `src/nsclc_survival/models.py`.

This distinction is disclosed to avoid implying byte-identical software provenance where the original trainer file was not retained in the final script archive. The saved aggregate result files are included for numerical audit; reproducing training from scratch may exhibit ordinary hardware/software-dependent floating-point variation.

## Data boundary

`data/locked_split_identifiers.csv` contains only public LUNG1 identifiers and the fixed analysis role (train/tuning/test). It contains no survival outcome or clinical covariate. All source data must be acquired directly from TCIA and remain governed by TCIA's terms and required citation.
