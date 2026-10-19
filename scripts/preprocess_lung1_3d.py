#!/usr/bin/env python
# -*- coding: utf-8 -*-
r'''
Preprocess TCIA NSCLC-Radiomics (LUNG1) CT volumes after final mask audit.

For each eligible patient:
1) Load CT slices in the exact SOPInstanceUID order stored in the tumor-mask NPZ.
2) Convert DICOM pixels to Hounsfield Units (HU).
3) Verify CT/mask geometry and PatientID consistency.
4) Clip CT intensities to a configurable HU window.
5) Crop around the tumor using a physical margin in millimeters.
6) Resample CT and mask to common voxel spacing.
7) Save one compressed NPZ per patient.
8) Produce QC tables and a summary.

This script deliberately DOES NOT force a fixed neural-network tensor size yet.
Review the resulting shape distribution first, then choose a fixed input size.

Dependencies:
  python -m pip install pydicom pandas numpy scipy tqdm

Example (Windows CMD, one line):
  python preprocess_lung1_3d.py --root "NSCLC-Radiomics" --cohort "final_mask_audit\\LUNG1_Final_Cohort.csv" --mask-dir "final_mask_audit\\masks" --out "preprocessed_lung1"
'''

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom
from tqdm import tqdm


def resolve_metadata_path(root: Path, metadata_path: str) -> Optional[Path]:
    if metadata_path is None or str(metadata_path).strip() == "":
        return None
    raw = str(metadata_path).strip()
    parts = list(PureWindowsPath(raw).parts)
    cleaned = [p for p in parts if p not in (".", "\\", "/")]
    if cleaned and cleaned[0].lower() == "nsclc-radiomics":
        cleaned = cleaned[1:]
    return root.joinpath(*cleaned)


def list_files(path: Optional[Path]) -> List[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.is_file()])


def load_ct_by_sop_order(ct_dir: Path, expected_sops: List[str], patient_id: str):
    by_sop = {}
    duplicates = []

    for p in list_files(ct_dir):
        try:
            ds = pydicom.dcmread(str(p), force=True)
        except Exception:
            continue

        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue

        sop = str(getattr(ds, "SOPInstanceUID", "")).strip()
        if not sop:
            continue

        if sop in by_sop:
            duplicates.append(sop)
        by_sop[sop] = (p, ds)

    missing = [uid for uid in expected_sops if uid not in by_sop]
    if missing:
        raise RuntimeError(
            f"{len(missing)} CT SOPInstanceUID(s) referenced by mask are missing"
        )

    slices = []
    pids = set()
    series_uids = set()
    study_uids = set()

    for uid in expected_sops:
        p, ds = by_sop[uid]

        pid = str(getattr(ds, "PatientID", "")).strip()
        if pid:
            pids.add(pid)

        series = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        if series:
            series_uids.add(series)

        study = str(getattr(ds, "StudyInstanceUID", "")).strip()
        if study:
            study_uids.add(study)

        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        slices.append(hu)

    if pids and pids != {patient_id}:
        raise RuntimeError(f"Unexpected CT PatientID set: {sorted(pids)}")

    volume = np.stack(slices, axis=0)

    return volume, {
        "duplicate_sop_count": len(duplicates),
        "patient_ids": sorted(pids),
        "series_uids": sorted(series_uids),
        "study_uids": sorted(study_uids),
    }


def tumor_bbox(mask: np.ndarray):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise RuntimeError("Tumor mask is empty")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return lo.astype(int), hi.astype(int)


def expand_bbox_mm(
    lo: np.ndarray,
    hi: np.ndarray,
    shape: Tuple[int, int, int],
    spacing_zyx: np.ndarray,
    margin_mm: float,
):
    spacing = np.asarray(spacing_zyx, dtype=float)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise RuntimeError(f"Invalid spacing: {spacing}")

    margin_vox = np.ceil(float(margin_mm) / spacing).astype(int)
    lo2 = np.maximum(0, lo - margin_vox)
    hi2 = np.minimum(np.asarray(shape), hi + margin_vox)
    return lo2.astype(int), hi2.astype(int), margin_vox.astype(int)


def crop3d(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def compute_zoom_factors(native_spacing_zyx, target_spacing_zyx):
    native = np.asarray(native_spacing_zyx, dtype=float)
    target = np.asarray(target_spacing_zyx, dtype=float)
    if np.any(native <= 0) or np.any(target <= 0):
        raise RuntimeError(
            f"Spacing must be positive. native={native}, target={target}"
        )
    return native / target


def resample_pair(
    ct_crop: np.ndarray,
    mask_crop: np.ndarray,
    native_spacing_zyx: np.ndarray,
    target_spacing_zyx: np.ndarray,
):
    factors = compute_zoom_factors(native_spacing_zyx, target_spacing_zyx)

    ct_res = zoom(
        ct_crop,
        zoom=factors,
        order=1,
        mode="nearest",
        prefilter=False,
    )

    mask_res = zoom(
        mask_crop.astype(np.uint8),
        zoom=factors,
        order=0,
        mode="nearest",
        prefilter=False,
    )
    mask_res = (mask_res > 0).astype(np.uint8)

    if ct_res.shape != mask_res.shape:
        raise RuntimeError(
            f"Resampled CT/mask shapes differ: {ct_res.shape} vs {mask_res.shape}"
        )

    return ct_res, mask_res, factors


def tumor_volume_ml(mask: np.ndarray, spacing_zyx: np.ndarray) -> float:
    voxel_mm3 = float(np.prod(np.asarray(spacing_zyx, dtype=float)))
    return float(mask.sum()) * voxel_mm3 / 1000.0


def shape_text(shape):
    return "x".join(str(int(x)) for x in shape)


def spacing_text(spacing):
    return "x".join(f"{float(x):.6g}" for x in spacing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to NSCLC-Radiomics DICOM folder")
    ap.add_argument("--cohort", required=True, help="Path to LUNG1_Final_Cohort.csv")
    ap.add_argument("--mask-dir", required=True, help="Validated tumor-mask NPZ folder")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--hu-min", type=float, default=-1000.0)
    ap.add_argument("--hu-max", type=float, default=400.0)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    ap.add_argument("--target-z", type=float, default=1.0)
    ap.add_argument("--target-y", type=float, default=1.0)
    ap.add_argument("--target-x", type=float, default=1.0)
    args = ap.parse_args()

    if args.hu_min >= args.hu_max:
        raise ValueError("--hu-min must be smaller than --hu-max")

    root = Path(args.root)
    cohort_path = Path(args.cohort)
    maskdir = Path(args.mask_dir)
    outdir = Path(args.out)
    voldir = outdir / "volumes"
    outdir.mkdir(parents=True, exist_ok=True)
    voldir.mkdir(parents=True, exist_ok=True)

    target_spacing = np.asarray(
        [args.target_z, args.target_y, args.target_x], dtype=np.float32
    )

    cohort = pd.read_csv(cohort_path, dtype=str).fillna("")
    results = []

    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Preprocessing LUNG1"):
        pid = str(row["PatientID"]).strip()

        result = {
            "PatientID": pid,
            "Success": False,
            "Mask_source": "",
            "ROI_name": "",
            "Native_CT_shape_zyx": "",
            "Native_spacing_zyx_mm": "",
            "Native_tumor_voxels": "",
            "Native_tumor_volume_ml": "",
            "Crop_margin_mm": args.margin_mm,
            "Native_crop_shape_zyx": "",
            "Resampled_shape_zyx": "",
            "Target_spacing_zyx_mm": spacing_text(target_spacing),
            "Resampled_tumor_voxels": "",
            "Resampled_tumor_volume_ml": "",
            "Tumor_volume_change_percent": "",
            "Fraction_voxels_below_HUmin_before_clip": "",
            "Fraction_voxels_above_HUmax_before_clip": "",
            "Output_file": "",
            "Error": "",
        }

        try:
            mask_file = maskdir / f"{pid}_tumor_mask.npz"
            if not mask_file.exists():
                raise RuntimeError(f"Validated mask missing: {mask_file}")

            npz = np.load(mask_file, allow_pickle=False)
            mask = np.asarray(npz["mask"], dtype=np.uint8)
            expected_sops = [str(x) for x in npz["ct_sop_uids"].tolist()]
            native_spacing = np.asarray(npz["spacing_zyx"], dtype=np.float32)
            mask_source = str(npz["source"].item())
            roi_name = str(npz["roi_name"].item())

            result["Mask_source"] = mask_source
            result["ROI_name"] = roi_name

            if mask.ndim != 3:
                raise RuntimeError(f"Mask is not 3D: shape={mask.shape}")
            if int(mask.sum()) == 0:
                raise RuntimeError("Validated tumor mask is empty")

            ct_dir = resolve_metadata_path(root, row.get("CT_Path", ""))
            if ct_dir is None or not ct_dir.exists():
                raise RuntimeError(f"CT path not found: {ct_dir}")

            ct, _ = load_ct_by_sop_order(ct_dir, expected_sops, pid)

            if ct.shape != mask.shape:
                raise RuntimeError(
                    f"Native CT/mask shape mismatch: {ct.shape} vs {mask.shape}"
                )

            result["Native_CT_shape_zyx"] = shape_text(ct.shape)
            result["Native_spacing_zyx_mm"] = spacing_text(native_spacing)
            result["Native_tumor_voxels"] = int(mask.sum())

            native_vol = tumor_volume_ml(mask, native_spacing)
            result["Native_tumor_volume_ml"] = native_vol

            below_frac = float(np.mean(ct < args.hu_min))
            above_frac = float(np.mean(ct > args.hu_max))
            result["Fraction_voxels_below_HUmin_before_clip"] = below_frac
            result["Fraction_voxels_above_HUmax_before_clip"] = above_frac

            lo, hi = tumor_bbox(mask)
            lo2, hi2, _ = expand_bbox_mm(
                lo, hi, ct.shape, native_spacing, args.margin_mm
            )

            ct_crop = crop3d(ct, lo2, hi2)
            mask_crop = crop3d(mask, lo2, hi2)
            result["Native_crop_shape_zyx"] = shape_text(ct_crop.shape)

            if int(mask_crop.sum()) != int(mask.sum()):
                raise RuntimeError("Tumor voxels were lost during native-space crop")

            ct_crop = np.clip(ct_crop, args.hu_min, args.hu_max)

            ct_res, mask_res, zoom_factors = resample_pair(
                ct_crop, mask_crop, native_spacing, target_spacing
            )

            if int(mask_res.sum()) == 0:
                raise RuntimeError("Tumor mask became empty after resampling")

            result["Resampled_shape_zyx"] = shape_text(ct_res.shape)
            result["Resampled_tumor_voxels"] = int(mask_res.sum())

            resampled_vol = tumor_volume_ml(mask_res, target_spacing)
            result["Resampled_tumor_volume_ml"] = resampled_vol

            if native_vol > 0:
                volume_change_pct = abs(resampled_vol - native_vol) / native_vol * 100.0
            else:
                volume_change_pct = float("nan")
            result["Tumor_volume_change_percent"] = volume_change_pct

            ct_int16 = np.rint(ct_res).astype(np.int16)
            mask_uint8 = (mask_res > 0).astype(np.uint8)
            output_file = voldir / f"{pid}_preprocessed.npz"

            crop_bounds = np.asarray(
                [
                    [int(lo2[0]), int(hi2[0])],
                    [int(lo2[1]), int(hi2[1])],
                    [int(lo2[2]), int(hi2[2])],
                ],
                dtype=np.int32,
            )

            np.savez_compressed(
                output_file,
                ct_hu_clipped=ct_int16,
                tumor_mask=mask_uint8,
                spacing_zyx=target_spacing.astype(np.float32),
                patient_id=np.asarray(pid),
                source_mask=np.asarray(mask_source),
                roi_name=np.asarray(roi_name),
                hu_clip=np.asarray([args.hu_min, args.hu_max], dtype=np.float32),
                native_spacing_zyx=native_spacing.astype(np.float32),
                native_ct_shape_zyx=np.asarray(ct.shape, dtype=np.int32),
                native_crop_shape_zyx=np.asarray(ct_crop.shape, dtype=np.int32),
                crop_bounds_native_zyx=crop_bounds,
                zoom_factors_zyx=np.asarray(zoom_factors, dtype=np.float32),
            )

            result["Output_file"] = str(output_file)
            result["Success"] = True

        except Exception as e:
            result["Error"] = f"{type(e).__name__}: {e}"

        results.append(result)

    qc = pd.DataFrame(results)
    qc_csv = outdir / "LUNG1_Preprocessing_QC.csv"
    qc.to_csv(qc_csv, index=False, encoding="utf-8-sig")

    ok = qc[qc["Success"] == True].copy()
    failed = qc[qc["Success"] != True].copy()

    merge_cols = [
        "PatientID",
        "Success",
        "Output_file",
        "Resampled_shape_zyx",
        "Target_spacing_zyx_mm",
        "Tumor_volume_change_percent",
    ]
    preprocessed_cohort = cohort.merge(qc[merge_cols], on="PatientID", how="left")
    cohort_csv = outdir / "LUNG1_Preprocessed_Cohort.csv"
    preprocessed_cohort.to_csv(cohort_csv, index=False, encoding="utf-8-sig")

    summary = [
        "LUNG1 3D preprocessing summary",
        "",
        f"Patients requested: {len(qc)}",
        f"Successful: {len(ok)}",
        f"Failed: {len(failed)}",
        f"HU clip: [{args.hu_min}, {args.hu_max}]",
        f"Crop margin: {args.margin_mm} mm",
        f"Target spacing z,y,x: {spacing_text(target_spacing)} mm",
        "",
    ]

    if len(ok):
        shape_arr = np.asarray([
            [int(v) for v in s.split("x")] for s in ok["Resampled_shape_zyx"]
        ])
        summary += [
            "Resampled shape distribution [z,y,x]:",
            f"  minimum: {shape_arr.min(axis=0).tolist()}",
            f"  median:  {np.median(shape_arr, axis=0).tolist()}",
            f"  90th pct:{np.percentile(shape_arr, 90, axis=0).tolist()}",
            f"  95th pct:{np.percentile(shape_arr, 95, axis=0).tolist()}",
            f"  maximum: {shape_arr.max(axis=0).tolist()}",
            "",
        ]

        vol_change = pd.to_numeric(
            ok["Tumor_volume_change_percent"], errors="coerce"
        ).dropna()
        if len(vol_change):
            summary += [
                "Tumor-volume change after mask resampling:",
                f"  median: {vol_change.median():.3f} %",
                f"  95th pct: {vol_change.quantile(0.95):.3f} %",
                f"  maximum: {vol_change.max():.3f} %",
                "",
            ]

    if len(failed):
        summary.append("Failed patients:")
        for _, r in failed.iterrows():
            summary.append(f"  {r['PatientID']}: {r['Error']}")
    else:
        summary += ["Failed patients:", "  None"]

    summary += [
        "",
        "IMPORTANT:",
        "Do not choose a fixed neural-network input shape until the",
        "resampled shape distribution above has been reviewed.",
    ]

    summary_txt = outdir / "LUNG1_Preprocessing_Summary.txt"
    summary_txt.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 70)
    print("\n".join(summary))
    print("=" * 70)
    print(f"QC CSV: {qc_csv}")
    print(f"Preprocessed cohort: {cohort_csv}")
    print(f"Volumes: {voldir}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
