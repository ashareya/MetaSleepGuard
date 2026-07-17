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
import base64
import os
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from MetaSleepGuard.experiments.common import repo_root


BASE_URL = "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI"
NEMAR_DATASET_ID = "nm000111"
NEMAR_VERSION = "v1.0.1"
NEMAR_MANIFEST_URL = f"https://data.nemar.org/{NEMAR_DATASET_ID}/{NEMAR_VERSION}/manifest.json"
NEMAR_DOI = "10.82901/nemar.nm000111"
ISRUC_PAPER_DOI = "10.1016/j.cmpb.2015.10.013"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root() / "data/public_sleep/isruc_raw"))
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--source", choices=["nemar", "original"], default="nemar")
    args = parser.parse_args()
    if not 1 <= args.subjects <= 100:
        raise SystemExit("--subjects must be between 1 and 100")
    root = Path(args.data_root).resolve()
    if args.source == "nemar":
        download_nemar(root, args.subjects)
        return
    download_original(root, args.subjects, args.extractor)


def download_nemar(root: Path, subjects: int) -> None:
    """Download a pinned, checksummed ISRUC Subgroup-I subset from NEMAR."""

    root = root / f"nemar_{NEMAR_VERSION.replace('.', '_')}"
    root.mkdir(parents=True, exist_ok=True)
    manifest_response = requests.get(NEMAR_MANIFEST_URL, timeout=(30, 120))
    manifest_response.raise_for_status()
    remote_rows = manifest_response.json()
    selected = select_nemar_manifest_rows(remote_rows, subjects)
    if not selected:
        raise RuntimeError("NEMAR manifest did not contain the requested ISRUC Subgroup-I files")
    rows_by_path: dict[str, dict] = {}

    def fetch(item: dict) -> dict:
        relative = Path(item["path"])
        destination = root / relative
        # ``bytes_url`` is version-pinned and can issue a fresh storage redirect;
        # the manifest's optional signed ``url`` expires during long subset runs.
        stable_url = item["bytes_url"]
        row = {
            "path": relative.as_posix(),
            "size": int(item["size"]),
            "checksum_algorithm": item["checksum_algorithm"],
            "checksum": item["checksum"],
            "bytes_url": item["bytes_url"],
            "download_url": stable_url,
            "status": "incomplete",
        }
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == int(item["size"]):
                verify_nemar_checksum(destination, item["checksum_algorithm"], item["checksum"])
            elif str(item["checksum_algorithm"]).lower() == "git":
                download_nemar_git_file(
                    relative.as_posix(), destination, expected_size=int(item["size"]), bytes_url=item["bytes_url"]
                )
            else:
                download_nemar_segmented(stable_url, destination, expected_size=int(item["size"]), chunks=32)
            verify_nemar_checksum(destination, item["checksum_algorithm"], item["checksum"])
            row.update({"status": "complete", "local_path": str(destination)})
        except Exception as exc:
            # A full-size file with a bad checksum cannot be resumed safely.
            # Remove it so the next invocation starts from independent ranges.
            if destination.exists() and "Checksum mismatch" in str(exc):
                destination.unlink()
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(fetch, item): item["path"] for item in selected}
        for future in as_completed(futures):
            row = future.result()
            rows_by_path[row["path"]] = row
            rows = [rows_by_path[path] for path in sorted(rows_by_path)]
            write_nemar_manifest(root, subjects, rows)
            print(f"nemar_path={row['path']} status={row['status']}", flush=True)
    rows = [rows_by_path[path] for path in sorted(rows_by_path)]
    failures = [row["path"] for row in rows if row["status"] != "complete"]
    manifest = write_nemar_manifest(root, subjects, rows)
    print(f"isruc_manifest={manifest}")
    if failures:
        raise SystemExit(f"NEMAR ISRUC download incomplete for {len(failures)} files; rerun to resume safely")


def select_nemar_manifest_rows(rows: list[dict], subjects: int) -> list[dict]:
    prefixes = {f"sub-I{index:03d}/" for index in range(1, subjects + 1)}
    root_files = {"README.md", "dataset_description.json", "participants.tsv", "participants.json", "scans.json"}
    suffixes = (
        "_eeg.edf",
        "_eeg.json",
        "_events.tsv",
        "_events.json",
        "_channels.tsv",
        "_scans.tsv",
    )
    selected = []
    for row in rows:
        path = str(row.get("path", ""))
        if path in root_files or (any(path.startswith(prefix) for prefix in prefixes) and path.endswith(suffixes)):
            required = {"path", "size", "checksum_algorithm", "checksum", "bytes_url"}
            if not required.issubset(row):
                raise RuntimeError(f"Incomplete NEMAR manifest row: {path}")
            selected.append(row)
    return sorted(selected, key=lambda item: item["path"])


def verify_nemar_checksum(path: Path, algorithm: str, expected: str) -> None:
    if algorithm.lower() == "sha256":
        actual = sha256(path)
    elif algorithm.lower() == "git":
        content = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
    else:
        raise RuntimeError(f"Unsupported NEMAR checksum algorithm: {algorithm}")
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected}, found {actual}")


def download_nemar_git_file(
    relative_path: str, destination: Path, expected_size: int, bytes_url: str | None = None
) -> None:
    """Fetch small versioned metadata through GitHub's contents API."""

    api_url = (
        f"https://api.github.com/repos/nemarDatasets/{NEMAR_DATASET_ID}/contents/"
        f"{quote(relative_path)}?ref={NEMAR_VERSION}"
    )
    try:
        if not bytes_url:
            raise RuntimeError("no NEMAR bytes_url")
        response = requests.get(bytes_url, timeout=(15, 60))
        response.raise_for_status()
        content = response.content
        if len(content) != expected_size:
            raise RuntimeError("NEMAR metadata response size mismatch")
    except Exception:
        try:
            response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"}, timeout=(10, 20))
            response.raise_for_status()
            payload = response.json()
            if payload.get("encoding") != "base64" or not payload.get("content"):
                raise RuntimeError(f"GitHub contents API returned no inline content for {relative_path}")
            content = base64.b64decode(payload["content"])
        except Exception:
            raw_url = f"https://raw.githubusercontent.com/nemarDatasets/{NEMAR_DATASET_ID}/{NEMAR_VERSION}/{relative_path}"
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                raise RuntimeError("curl is required for the NEMAR metadata DNS fallback")
            completed = subprocess.run(
                [
                    curl,
                    "--ssl-no-revoke",
                    "--resolve",
                    "raw.githubusercontent.com:443:185.199.108.133",
                    "--location",
                    "--fail",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--retry-delay",
                    "2",
                    "--max-time",
                    "180",
                    raw_url,
                ],
                capture_output=True,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
            content = completed.stdout
    if len(content) != expected_size:
        raise RuntimeError(f"Metadata size mismatch for {relative_path}: expected {expected_size}, found {len(content)}")
    destination.write_bytes(content)


def download_nemar_segmented(url: str, destination: Path, expected_size: int, chunks: int = 32) -> None:
    """Download one checksummed EDF into independent resumable byte ranges."""

    from MetaSleepGuard.experiments.download_sleep_edf_subset import _byte_ranges, _download_range

    if destination.exists() and destination.stat().st_size == expected_size:
        return
    if destination.exists():
        destination.unlink()
    part_root = destination.parent / f".{destination.name}.parts"
    part_root.mkdir(parents=True, exist_ok=True)
    existing_parts = list(part_root.glob("[0-9][0-9][0-9].part")) + list(
        part_root.glob("[0-9][0-9][0-9].partial")
    )
    if existing_parts:
        # Preserve the partition geometry created by an earlier release/run.
        chunks = max(int(path.stem.split(".")[0]) for path in existing_parts) + 1
    ranges = _byte_ranges(expected_size, chunks)
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(_download_range, url, part_root / f"{index:03d}.part", start, end)
            for index, (start, end) in enumerate(ranges)
        ]
        for future in as_completed(futures):
            future.result()
    temporary = destination.with_suffix(destination.suffix + ".download")
    with temporary.open("wb") as output:
        for index in range(len(ranges)):
            with (part_root / f"{index:03d}.part").open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"Segmented size mismatch for {destination.name}")
    os.replace(temporary, destination)
    for part in part_root.glob("*"):
        if part.is_file():
            part.unlink()
    part_root.rmdir()


def write_nemar_manifest(root: Path, subjects: int, rows: list[dict]) -> Path:
    manifest = root / "download_manifest.json"
    payload = {
        "dataset": "ISRUC-Sleep",
        "distribution": f"NEMAR {NEMAR_DATASET_ID} {NEMAR_VERSION}",
        "dataset_id": NEMAR_DATASET_ID,
        "version": NEMAR_VERSION,
        "nemar_doi": NEMAR_DOI,
        "isruc_paper_doi": ISRUC_PAPER_DOI,
        "manifest_url": NEMAR_MANIFEST_URL,
        "subgroup": "I",
        "requested_subjects": subjects,
        "files": rows,
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download_original(root: Path, subjects: int, extractor_arg: str | None = None) -> None:
    """Legacy downloader for the original University of Coimbra RAR files."""

    archives = root / "subgroupI_archives"
    extracted = root / "subgroupI"
    archives.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    extractor = find_extractor(extractor_arg)
    rows = []
    failures = []
    manifest = root / "official_download_manifest.json"
    for subject in range(1, subjects + 1):
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
    with requests.get(url, headers=headers, stream=True, timeout=(30, 30)) as response:
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


def download_with_retries(url: str, path: Path, expected_size: int, attempts: int = 8) -> None:
    """Resume transiently interrupted NEMAR transfers within one command."""

    existing = path.stat().st_size if path.exists() else 0
    if existing > expected_size:
        path.unlink()
        existing = 0
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl and existing < expected_size:
        result = subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--retry",
                str(attempts),
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(path),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and path.stat().st_size == expected_size:
            return
        raise RuntimeError(f"curl resumable download failed for {path.name}: {result.stderr[-1000:]}")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            download_resumable(url, path, expected_size=expected_size)
            return
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(attempt, 5))
    raise RuntimeError(f"Download failed after {attempts} resumable attempts: {last_error}")


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
