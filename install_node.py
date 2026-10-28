import zipfile
from pathlib import Path

zip_path = Path(r"C:\Users\akash\OneDrive\Desktop\MINI PROJECT\node-v24.20.0-win-x64.zip")
out_dir = Path(r"C:\Users\akash\OneDrive\Desktop\MINI PROJECT\node")
out_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print(f"Extracted to: {out_dir}")
print([p.name for p in sorted(out_dir.iterdir())[:10]])
