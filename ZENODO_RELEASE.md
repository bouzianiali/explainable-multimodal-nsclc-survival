# GitHub + Zenodo public release workflow

The repository is designed to be public on GitHub and archived with a DOI on Zenodo. Two valid workflows are possible.

## Recommended for the first manuscript submission: manual Zenodo deposit with a reserved DOI

This route avoids a circular problem where the DOI is minted only after a GitHub release, but you want the DOI already written in the manuscript and repository metadata.

1. Create a **public GitHub repository** named `explainable-multimodal-nsclc-survival`.
2. Upload the **contents** of this repository folder (not the enclosing ZIP).
3. On Zenodo, create a new upload with resource type **Software**.
4. Before publishing, use **Get a DOI now!** to reserve a Zenodo DOI.
5. Run:

   ```bash
   python release_tools/finalize_release_links.py \
     --github https://github.com/USERNAME/explainable-multimodal-nsclc-survival \
     --zenodo-doi 10.5281/zenodo.XXXXXXXX
   ```

6. Commit/push the finalized README and `CITATION.cff` to GitHub.
7. Create GitHub tag/release `v1.0.0`.
8. Download or create the `v1.0.0` source ZIP and upload that exact software snapshot to the Zenodo draft.
9. Complete Zenodo metadata (creators, affiliation, title, keywords, MIT license, related manuscript if available) and publish the record.
10. Insert the same GitHub URL and Zenodo DOI into the manuscript Code Availability statement and recompile before journal submission.

Zenodo registers the reserved DOI when the draft is published. Do not delete the draft after reserving the DOI or the reservation is lost.

## Alternative: Zenodo GitHub integration

1. Link your Zenodo account to GitHub.
2. In Zenodo, open the GitHub integration page, sync repositories, and enable this repository.
3. Create a GitHub release/tag such as `v1.0.0`.
4. Zenodo automatically ingests the enabled GitHub release and creates the software record/DOI.
5. Add the resulting DOI to the live GitHub README/CITATION metadata and to the manuscript. If you want the DOI embedded in an archived software snapshot itself, make a subsequent finalized release.

## Suggested metadata

- **Title:** Explainable multimodal NSCLC survival prediction under clinical-modality absence
- **Resource type:** Software
- **Version:** 1.0.0
- **Creators:** Ali Bouziani; Naji Abdelwahab
- **License:** MIT (code only; TCIA data are not redistributed)
- **Keywords:** NSCLC; survival analysis; multimodal learning; explainable AI; CT; Cox model; missing modality; medical imaging
- **Description:** Reproducibility code and aggregate outputs for the associated LUNG1 internal-evaluation manuscript. Source TCIA images, segmentations, clinical tables, patient-level outcomes, and trained checkpoints are excluded.

## Final checks

```bash
python scripts/check_public_release.py
```

The checker should pass. Before public release, no `REPLACE_WITH_...` placeholders should remain in `README.md` or `CITATION.cff`.
