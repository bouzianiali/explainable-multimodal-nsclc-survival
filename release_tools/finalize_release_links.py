#!/usr/bin/env python
"""Replace GitHub/Zenodo placeholders in the public release and optionally manuscript back matter."""
from pathlib import Path
import argparse, re

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--github',required=True,help='e.g. https://github.com/USERNAME/explainable-multimodal-nsclc-survival')
    ap.add_argument('--zenodo-doi',required=True,help='e.g. 10.5281/zenodo.12345678')
    ap.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--backmatter',default='',help='optional path to manuscript 06_backmatter_v9.tex')
    a=ap.parse_args(); root=Path(a.repo_root)
    for f in ['README.md','CITATION.cff']:
        p=root/f; s=p.read_text(); s=s.replace('REPLACE_WITH_GITHUB_URL',a.github).replace('REPLACE_WITH_ZENODO_DOI',a.zenodo_doi); p.write_text(s)
    if a.backmatter:
        p=Path(a.backmatter); s=p.read_text()
        old=re.compile(r"\\section\*\{Code Availability\}.*?(?=\n\\section\*)",re.S)
        new=("\\section*{Code Availability}\n"
             "The reproducibility code, locked split identifiers, protocol metadata, model configurations, and aggregate analysis outputs are publicly available at "
             f"\\href{{{a.github}}}{{GitHub}} and archived on Zenodo (DOI: \\href{{https://doi.org/{a.zenodo_doi}}}{{{a.zenodo_doi}}}). "
             "Raw TCIA imaging, segmentations, clinical tables, patient-level outcomes, and trained checkpoints are not redistributed.\n")
        if not old.search(s): raise RuntimeError('Code Availability section not found')
        p.write_text(old.sub(new,s))
    print('Release links finalized.')
if __name__=='__main__': main()
