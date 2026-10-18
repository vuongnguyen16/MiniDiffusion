"""Download and verify the pretrained 200k EMA checkpoint (standard library only)."""

import hashlib
import json
from pathlib import Path
import urllib.request


FOLDER = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(destination=None):
    metadata = json.loads((FOLDER / 'metadata.json').read_text(encoding='utf-8'))
    destination = Path(destination) if destination is not None else FOLDER / 'ckpt.pt'
    if destination.exists() and destination.stat().st_size == metadata['bytes']:
        if sha256(destination) == metadata['sha256']:
            print(f'Checkpoint already verified: {destination}')
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pt.tmp')
    request = urllib.request.Request(metadata['download_url'],
                                     headers={'User-Agent': 'MiniDiffusion-checkpoint-downloader'})
    digest = hashlib.sha256()
    written = 0
    last_report = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(4 * 1024 * 1024):
                written += output.write(block)
                digest.update(block)
                if written - last_report >= 16 * 1024 * 1024:
                    print(f'Downloaded {written / 1024**2:.1f}/{metadata["bytes"] / 1024**2:.1f} MiB', flush=True)
                    last_report = written
        if written != metadata['bytes'] or digest.hexdigest() != metadata['sha256']:
            raise RuntimeError('Checkpoint size or SHA-256 mismatch; download was not installed')
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'200k EMA checkpoint downloaded and verified: {destination}')
    return destination


if __name__ == '__main__':
    download_checkpoint()
