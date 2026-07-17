"""Download and extract official ISRUC-Sleep Cohort I subject archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re
import time

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
    failures = []
    manifest = root / "official_download_manifest.json"
    for subject in range(1, args.subjects + 1):
        url = f"{BASE_URL}/{subject}.rar"
        archive = archives / f"{subject}.rar"
        row = {
                "subject": subject,
                "official_url": url,
                "archive": str(archive),
                "archive_size": archive.stat().st_size if archive.exists() else 0,
                "status": "incomplete",
            }
        try:
            expected_size = remote_size(url)
            download_resumable(url, archive, expected_size=expected_size)
            if archive.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Incomplete ISRUC archive {archive.name}: "
                    f"expected {expected_size} bytes, found {archive.stat().st_size}"
                )
            test_archive(extractor, archive)
            target = extracted / f"subject_{subject:03d}"
            target.mkdir(parents=True, exist_ok=True)
            extract_archive(extractor, archive, target)
            row.update(
                {
                    "archive_size": archive.stat().st_size,
                    "expected_size": expected_size,
                    "size_verified": True,
                    "archive_test_passed": True,
                    "archive_sha256": sha256(archive),
                    "extract_dir": str(target),
                    "files": sorted(str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()),
                    "status": "complete",
                }
            )
            print(f"isruc_subject={subject} archive_size={archive.stat().st_size} status=complete")
        except Exception as exc:
            row.update(
                {
                    "archive_size": archive.stat().st_size if archive.exists() else 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(subject)
            print(f"isruc_subject={subject} status=incomplete error={exc}")
        rows.append(row)
        manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"isruc_manifest={manifest}")
    if failures:
        raise SystemExit(f"ISRUC download incomplete for subjects: {failures}. Rerun the same command to resume safely.")


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


def remote_size(url: str, attempts: int = 3) -> int:
    """Return the authoritative object length using a one-byte range request."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes\s+0-0/(\d+)", response.headers.get("Content-Range", ""))
                    if not match:
                        raise RuntimeError(
                            f"Invalid Content-Range for {url}: {response.headers.get('Content-Range')!r}"
                        )
                    return int(match.group(1))
                length = response.headers.get("Content-Length")
                if response.status_code == 200 and length and int(length) > 1:
                    return int(length)
                raise RuntimeError(f"Official ISRUC server did not expose a reliable size for {url}")
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Official ISRUC server was unavailable after {attempts} attempts: {last_error}")


def download_resumable(url: str, path: Path, expected_size: int | None = None) -> None:
    existing = path.stat().st_size if path.exists() else 0
    if expected_size is None:
        expected_size = remote_size(url)
    if existing > expected_size:
        raise RuntimeError(f"Local archive is larger than the official object: {path}")
    if existing == expected_size:
        return
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if existing and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match.group(1)) != existing or int(match.group(3)) != expected_size:
                raise RuntimeError(f"Unsafe resume response for {path.name}: {content_range!r}")
            mode = "ab"
        elif response.status_code == 200:
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP {response.status_code} while downloading {url}")
        with path.open(mode) as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Incomplete download {path.name}: expected {expected_size}, found {actual_size}")


def test_archive(extractor: Path, archive: Path) -> None:
    """Ask the installed RAR extractor to verify CRCs without extracting."""

    name = extractor.name.lower()
    command = [str(extractor), "t", str(archive)]
    if "unrar" not in name:
        command.insert(2, "-y")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ISRUC archive integrity test failed for {archive.name}: {detail}")


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
