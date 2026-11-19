"""Download the official Factory package with verified HTTP ranges and SHA256."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / 'downloads/scene_manifest.json').read_text())
SCENE = sys.argv[1] if len(sys.argv) > 1 else "FactoryExteriorInteriorBundle"
ENTRY = next(x for x in MANIFEST if Path(x['path']).name == f"AirSim360_Pro_{SCENE}.zip")
DEST = ROOT / 'downloads' / Path(ENTRY['path']).name
PARTS = ROOT / 'downloads' / ('factory_parts' if SCENE == "FactoryExteriorInteriorBundle" else SCENE + "_parts")
URL = 'https://huggingface.co/datasets/Insta360-Research/AirSim360/resolve/main/' + ENTRY['path']
CHUNK = 8 * 1024 * 1024

def download(index):
    start = index * CHUNK
    end = min(start + CHUNK, ENTRY['size']) - 1
    part = PARTS / f'{index:04d}.part'
    if part.exists() and part.stat().st_size == end - start + 1:
        return index
    for attempt in range(5):
        try:
            request = urllib.request.Request(URL + f'?download=true&part={index}', headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                expected = f"bytes {start}-{end}/{ENTRY['size']}"
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('Server did not honor requested byte range')
                with part.open('wb') as output:
                    while data := response.read(256 * 1024):
                        output.write(data)
            if part.stat().st_size != end - start + 1:
                raise RuntimeError('Incomplete chunk')
            print(f'part {index} complete', flush=True)
            return index
        except Exception as exc:
            print(f'part {index} retry {attempt + 1}: {type(exc).__name__}', flush=True)
            if attempt == 4:
                raise RuntimeError(f'Part {index} failed') from None
            time.sleep(2)

if __name__ == '__main__':
    PARTS.mkdir(parents=True, exist_ok=True)
    count = (ENTRY['size'] + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, range(count)))
    digest = hashlib.sha256()
    with DEST.open('wb') as output:
        for index in range(count):
            with (PARTS / f'{index:04d}.part').open('rb') as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    output.write(data)
    actual = digest.hexdigest()
    if actual != ENTRY['lfs']['oid']:
        raise RuntimeError('SHA256 mismatch')
    report = "download_verification.json" if SCENE == "FactoryExteriorInteriorBundle" else f"{SCENE}_download.json"
    (ROOT / 'reports' / report).write_text(json.dumps({
        'file': str(DEST), 'size': DEST.stat().st_size, 'sha256': actual,
        'expected_sha256': ENTRY['lfs']['oid'], 'verified': True,
        'source': URL,
    }, indent=2))
    print(f'Verified {DEST.name}: {actual}', flush=True)
