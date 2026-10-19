#!/usr/bin/env python
"""Idempotently verify/finalize the public GitHub and Zenodo identifiers.

The defaults correspond to the v1.0.0 public release. An optional manuscript
back-matter file can be updated with the same identifiers.
"""
from pathlib import Path
import argparse
import re

DEFAULT_GITHUB = "https://github.com/bouzianiali/explainable-multimodal-nsclc-survival"
DEFAULT_DOI = "10.5281/zenodo.23483737"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", default=DEFAULT_GITHUB)
    ap.add_argument("--zenodo-doi", default=DEFAULT_DOI)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--backmatter", default="", help="optional manuscript back-matter .tex file")
    a = ap.parse_args()
    root = Path(a.repo_root)

    readme = root / "README.md"
    txt = readme.read_text(encoding="utf-8")
    txt = re.sub(r"- GitHub: \*\*.*?\*\*", f"- GitHub: **{a.github}**", txt)
    txt = re.sub(r"- Zenodo archive DOI: \*\*.*?\*\*", f"- Zenodo archive DOI: **https://doi.org/{a.zenodo_doi}**", txt)
    readme.write_text(txt, encoding="utf-8")

    cff = root / "CITATION.cff"
    txt = cff.read_text(encoding="utf-8")
    txt = re.sub(r'^repository-code: ".*"$', f'repository-code: "{a.github}"', txt, flags=re.M)
    txt = re.sub(r'^doi: ".*"$', f'doi: "{a.zenodo_doi}"', txt, flags=re.M)
    txt = re.sub(r'^url: ".*"$', f'url: "https://doi.org/{a.zenodo_doi}"', txt, flags=re.M)
    cff.write_text(txt, encoding="utf-8")

    if a.backmatter:
        p = Path(a.backmatter)
        s = p.read_text(encoding="utf-8")
        old = re.compile(r"\\section\*\{Code Availability\}.*?(?=\n\\section\*)", re.S)
        new = (
            "\\section*{Code Availability}\n"
            "The reproducibility code, locked split identifiers, protocol metadata, model configurations, "
            "and aggregate analysis outputs are publicly available at "
            f"\\href{{{a.github}}}{{GitHub}} and archived on Zenodo "
            f"(DOI: \\href{{https://doi.org/{a.zenodo_doi}}}{{{a.zenodo_doi}}}). "
            "Raw TCIA imaging, segmentations, clinical tables, patient-level outcomes, and trained checkpoints "
            "are not redistributed.\n"
        )
        if not old.search(s):
            raise RuntimeError("Code Availability section not found")
        p.write_text(old.sub(new, s), encoding="utf-8")

    print("Release identifiers verified/finalized.")


if __name__ == "__main__":
    main()
