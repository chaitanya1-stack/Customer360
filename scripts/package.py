"""Package portable source, original data, fitted model and reviewed outputs only."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
top_files={"README.md","START_HERE.md","LICENSE","DATA_LICENSE.md","requirements.txt",
           ".gitignore","run.py","score.py","RetailLens_Walkthrough.ipynb"}
allowed_dirs={"src","sql","tests","docs","scripts","outputs","models",".github"}
files=[]
for p in sorted(root.rglob("*")):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if "__pycache__" in rel.parts or p.suffix==".pyc":continue
    if len(rel.parts)==1:
        include=p.name in top_files
    elif rel.parts[0]=="data":
        include=str(rel) in {"data/raw/online_retail_ii.zip","data/processed/audit.json"}
    else:
        include=rel.parts[0] in allowed_dirs and not any(x.startswith(".") for x in rel.parts[1:])
    if include:files.append(p)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/"SHA256SUMS.json").write_text(json.dumps(manifest,indent=2))
files.append(root/"SHA256SUMS.json")
archive=root.parent/"RetailLens_Customer_Segmentation.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:
        z.write(p,arcname=str(Path("customer_segmentation")/p.relative_to(root)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))
print(json.dumps({"archive":str(archive),"files":len(files),"bytes":archive.stat().st_size,
                  "size_mib":round(archive.stat().st_size/1024**2,2)},indent=2))
