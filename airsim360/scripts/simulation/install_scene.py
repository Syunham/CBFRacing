"""Verify and extract only the selected official scene into the workspace."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import hashlib
import json
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT/'downloads/scene_manifest.json').read_text())
scene = sys.argv[1] if len(sys.argv) > 1 else "FactoryExteriorInteriorBundle"
entry = next(x for x in manifest if Path(x['path']).name == f"AirSim360_Pro_{scene}.zip")
archive = ROOT/'downloads'/Path(entry['path']).name
hasher = hashlib.sha256()
with archive.open('rb') as source:
    while block := source.read(1024*1024):
        hasher.update(block)
digest = hasher.hexdigest()
if digest != entry['lfs']['oid'] or archive.stat().st_size != entry['size']:
    raise RuntimeError('Official scene SHA256 or file size mismatch')
destination = ROOT/'simulator'
destination.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as package:
    info = package.infolist()
    for member in info:
        target = (destination/member.filename).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise RuntimeError('Unsafe archive path')
    inventory = {
        'archive':str(archive),'sha256':digest,'file_count':len(info),
        'expanded_bytes':sum(x.file_size for x in info),
        'uproject_files':[x.filename for x in info if x.filename.endswith('.uproject')],
        'executables':[x.filename for x in info if x.filename.endswith('.exe')],
        'files':[{'path':x.filename,'bytes':x.file_size} for x in info],
    }
    report = "package_inventory.json" if scene == "FactoryExteriorInteriorBundle" else f"{scene}_inventory.json"
    (ROOT/'reports'/report).write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    package.extractall(destination)
print(json.dumps({k:v for k,v in inventory.items() if k!='files'},indent=2))
