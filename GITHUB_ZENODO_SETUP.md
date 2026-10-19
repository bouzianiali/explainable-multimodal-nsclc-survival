# GitHub + Zenodo setup before journal submission

## 1. Create the GitHub repository

Create a **public** repository named:

`explainable-multimodal-nsclc-survival`

Do not initialize it with a new README, .gitignore, or license because those files are already included here.

Upload the **contents of this folder** (not the outer ZIP as a single file) to the root of the repository.

After upload, the repository address will have the form:

`https://github.com/YOUR_USERNAME/explainable-multimodal-nsclc-survival`

## 2. Reserve a Zenodo DOI

Create a new Zenodo upload and select **Software** as the resource type. Reserve a DOI in the draft before publishing it.

Recommended title:

`Explainable multimodal NSCLC survival prediction under clinical-modality absence`

Use version `1.0.0`, public access, and the MIT license for the code. The TCIA/LUNG1 data are not redistributed.

## 3. Finalize repository metadata

Once both values are known, replace:

- `REPLACE_WITH_GITHUB_URL`
- `REPLACE_WITH_ZENODO_DOI`

in `README.md` and `CITATION.cff`.

You can use:

```bash
python release_tools/finalize_release_links.py \
  --github-url "https://github.com/YOUR_USERNAME/explainable-multimodal-nsclc-survival" \
  --zenodo-doi "10.5281/zenodo.XXXXXXXX"
```

Then commit/push those final metadata changes to GitHub.

## 4. Publish the Zenodo software record

Create a ZIP of the finalized repository and upload that **single ZIP** to the Zenodo software record. Preview the record, verify metadata, then publish it. The reserved DOI becomes active at publication.

## 5. Update the manuscript

In the manuscript Code Availability section, insert the final GitHub URL and Zenodo DOI. The Zenodo software release can also be cited formally in the references.
