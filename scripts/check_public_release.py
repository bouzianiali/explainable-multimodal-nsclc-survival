from pathlib import Path
import sys, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
required=[
 'README.md','PUBLIC_RELEASE_NOTES.md','THIRD_PARTY_DATA_NOTICE.md','ZENODO_RELEASE.md','CITATION.cff','requirements.txt','LICENSE',
 'data/locked_split_identifiers.csv','data/TNM_SENTINEL_RECODING_RULE.csv','data/clinical_feature_schema_21.txt',
 'scripts/preprocess_lung1_3d.py','scripts/prepare_lung1_tnm_corrected.py','scripts/prepare_clinical_features_and_cox.py',
 'scripts/train_ct_only_survival_fullcox_efron.py','scripts/train_multimodal_ct_clinical_survival_fullcox_efron.py',
 'scripts/train_standard_moddrop_missingaware_fullcox_efron.py','scripts/train_gated_moddrop_v2_fullcox_efron.py',
 'scripts/compare_survival_models_paired_fullcox_final.py','scripts/calibrate_survival_models_fullcox_final.py',
 'scripts/compare_missingness_robustness_final.py','scripts/compare_missingnessaware_vs_proposed_final.py',
 'scripts/xai_gradcam_gradcampp_tnm_corrected_final.py','scripts/xai_faithfulness_tnm_corrected_final.py',
 'scripts/xai_shap_clinical_tnm_corrected_cpu.py','scripts/xai_multimodal_clinical_synthesis_tnm_corrected_cpu.py',
 'scripts/crop_shift_sensitivity_final_v2.py','scripts/run_ablation_repeated_seeds_final.py',
 'src/nsclc_survival/models.py','config/protocol.yaml'
]
missing=[x for x in required if not (ROOT/x).exists()]
if missing:
    print('MISSING:'); [print(' -',x) for x in missing]; sys.exit(1)
split=pd.read_csv(ROOT/'data/locked_split_identifiers.csv')
assert list(split.columns)==['PatientID','Split'], f'unexpected split columns: {split.columns.tolist()}'
assert len(split)==422 and split.PatientID.nunique()==422
print('split rows',len(split),'unique',split.PatientID.nunique(),split.Split.value_counts().to_dict())
# Model size checks are part of the public protocol audit.
sys.path.insert(0,str(ROOT/'src'))
from nsclc_survival.models import CTOnlyTeacher,GatedResidualSurvival
ct=sum(p.numel() for p in CTOnlyTeacher(0.30).parameters())
g=sum(p.numel() for p in GatedResidualSurvival(21,0.30).parameters())
print('parameter counts CT-only',ct,'gated',g)
assert ct==638393 and g==667131
# Prevent obvious accidental redistribution of common patient-level source file types.
for p in ROOT.rglob('*'):
    if not p.is_file(): continue
    low=p.name.lower()
    if low.endswith(('.dcm','.nii','.nii.gz','.nrrd','.mha','.mhd','.npz','.pt','.pth')):
        raise RuntimeError(f'Potential source data/checkpoint included: {p.relative_to(ROOT)}')
# Final release metadata must be resolved before public deposition.
expected = {
    'github': 'https://github.com/bouzianiali/explainable-multimodal-nsclc-survival',
    'doi': '10.5281/zenodo.23483737',
    'orcid': '0009-0002-2940-6488',
}
for f in ['README.md','CITATION.cff']:
    txt=(ROOT/f).read_text(errors='ignore')
    if 'REPLACE_WITH_' in txt:
        raise RuntimeError(f'Unresolved release placeholder remains in {f}')
for value in expected.values():
    if not any(value in (ROOT/f).read_text(errors='ignore') for f in ['README.md','CITATION.cff']):
        raise RuntimeError(f'Expected release metadata not found: {value}')
print('PUBLIC RELEASE CHECK: PASS')
