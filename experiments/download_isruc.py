"""Download and extract official ISRUC-Sleep Cohort I subject archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import requests

from MetaSleepGuard.experiments.common import repo_root


BASE_URL = "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root() / "data/public_sleep/isruc_raw"))
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--extractor", default=None)
    args = parser.parse_args()
    if not 1 <= args.subjects <= 100:
        raise SystemExit("--subjects must be between 1 and 100")
    root = Path(args.data_root).resolve()
    archives = root / "subgroupI_archives"
    extracted = root / "subgroupI"
    archives.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    extractor = find_extractor(args.extractor)
    rows = []
    for subject in range(1, args.subjects + 1):
        url = f"{BASE_URL}/{subject}.rar"
        archive = archives / f"{subject}.rar"
        download_resumable(url, archive)
        target = extracted / f"subject_{subject:03d}"
        target.mkdir(parents=True, exist_ok=True)
        extract_archive(extractor, archive, target)
        rows.append(
            {
                "subject": subject,
                "official_url": url,
                "archive": str(archive),
                "archive_size": archive.stat().st_size,
                "archive_sha256": sha256(archive),
                "extract_dir": str(target),
                "files": sorted(str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()),
            }
        )
        print(f"isruc_subject={subject} archive_size={archive.stat().st_size}")
    manifest = root / "official_download_manifest.json"
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"isruc_manifest={manifest}")


def find_extractor(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("7z"),
        shutil.which("7zz"),
        shutil.which("unrar"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError(
        "ISRUC official files are RAR archives. Install 7-Zip or pass -Extractor/--extractor "
        "to a trusted 7z-compatible executable; dataset files are not replaced with unofficial mirrors."
    )


def download_resumable(url: str, path: Path) -> None:
    existing = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code == 200:
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with path.open(mode) as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_archive(extractor: Path, archive: Path, target: Path) -> None:
    name = extractor.name.lower()
    if "unrar" in name:
        command = [str(extractor), "x", "-o+", str(archive), str(target)]
    else:
        command = [str(extractor), "x", "-y", f"-o{target}", str(archive)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
