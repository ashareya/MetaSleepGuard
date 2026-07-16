"""Download and verify a reproducible Sleep-EDF subject subset."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import requests


DEFAULT_BASE_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subjects", type=int, default=15)
    parser.add_argument("--recording", type=int, default=1, choices=(1, 2))
    parser.add_argument("--chunks-per-file", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    if args.subjects < 1:
        raise SystemExit("--subjects must be positive")
    if args.chunks_per_file < 1 or args.chunks_per_file > 16:
        raise SystemExit("--chunks-per-file must be between 1 and 16")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    records = _official_records(list(range(args.subjects)), args.recording)
    destination = Path(args.data_root).resolve() / "physionet-sleep-data"
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = download_records(
        records,
        destination,
        args.base_url.rstrip("/"),
        chunks_per_file=args.chunks_per_file,
        workers=args.workers,
    )
    print(f"verified_files={len(downloaded)}")
    print(f"verified_subjects={len({record['subject'] for record in records})}")
    for path in downloaded:
        print(f"verified={path}")


def _official_records(subjects: list[int], recording: int) -> list[dict[str, str]]:
    from mne.datasets.sleep_physionet.age import AGE_SLEEP_RECORDS

    selected = []
    with Path(AGE_SLEEP_RECORDS).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = int(row["subject"])
            if subject in subjects and int(row["night"]) == recording:
                selected.append(row)
    found_subjects = {int(row["subject"]) for row in selected}
    missing = sorted(set(subjects) - found_subjects)
    if missing:
        raise ValueError(f"Official Sleep-EDF records are unavailable for subjects: {missing}")
    return selected


def download_records(
    records: list[dict[str, str]],
    destination: Path,
    base_url: str,
    *,
    chunks_per_file: int,
    workers: int,
) -> list[Path]:
    verified = []
    pending = []
    for record in records:
        target = destination / record["fname"]
        if target.exists() and _sha1(target) == record["sha"]:
            verified.append(target)
        else:
            pending.append((record, target))
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(
                    _download_one,
                    record,
                    target,
                    base_url,
                    chunks_per_file,
                ): target
                for record, target in pending
            }
            for future in as_completed(futures):
                verified.append(future.result())
    return sorted(verified)


def _download_one(
    record: dict[str, str],
    target: Path,
    base_url: str,
    chunks_per_file: int,
) -> Path:
    url = f"{base_url}/{record['fname']}"
    size = _remote_content_length(url)
    part_root = (
        Path(tempfile.gettempdir())
        / "metasleepguard_sleep_edf_parts"
        / record["fname"]
        / f"chunks_{chunks_per_file}_size_{size}"
    )
    part_root.mkdir(parents=True, exist_ok=True)
    ranges = _byte_ranges(size, chunks_per_file)
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [
            pool.submit(_download_range, url, part_root / f"{index:03d}.part", start, end)
            for index, (start, end) in enumerate(ranges)
        ]
        for future in as_completed(futures):
            future.result()
    temporary_target = target.with_suffix(target.suffix + ".download")
    with temporary_target.open("wb") as output:
        for index in range(len(ranges)):
            with (part_root / f"{index:03d}.part").open("rb") as part:
                while block := part.read(1024 * 1024):
                    output.write(block)
    if temporary_target.stat().st_size != size:
        raise RuntimeError(f"Size mismatch for {record['fname']}")
    actual_sha1 = _sha1(temporary_target)
    if actual_sha1 != record["sha"]:
        temporary_target.unlink(missing_ok=True)
        for part in part_root.glob("*"):
            if part.is_file():
                part.unlink()
        raise RuntimeError(
            f"SHA1 mismatch for {record['fname']}: expected={record['sha']} actual={actual_sha1}"
        )
    os.replace(temporary_target, target)
    for part in part_root.glob("*"):
        if part.is_file():
            part.unlink()
    part_root.rmdir()
    print(f"download_verified={record['fname']} bytes={size}", flush=True)
    return target


def _download_range(url: str, path: Path, start: int, end: int) -> None:
    expected_size = end - start + 1
    if path.exists() and path.stat().st_size == expected_size:
        return
    for attempt in range(30):
        try:
            temporary = path.with_suffix(".partial")
            current_size = temporary.stat().st_size if temporary.exists() else 0
            if current_size > expected_size:
                temporary.unlink()
                current_size = 0
            if current_size == expected_size:
                os.replace(temporary, path)
                return
            resume_start = start + current_size
            with temporary.open("ab") as output:
                curl = shutil.which("curl")
                if curl is None:
                    raise RuntimeError("curl is required for resumable Sleep-EDF downloads")
                completed = subprocess.run(
                    [
                        curl,
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "30",
                        "--max-time",
                        "60",
                        "--range",
                        f"{resume_start}-{end}",
                        url,
                    ],
                    check=False,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    text=False,
                )
            if completed.returncode:
                error = completed.stderr.decode(errors="replace").strip()
                raise RuntimeError(error or f"curl exit={completed.returncode}")
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Range size mismatch: expected={expected_size} actual={temporary.stat().st_size}"
                )
            os.replace(temporary, path)
            return
        except Exception:
            if attempt == 29:
                raise
            time.sleep(min(2**attempt, 30))


def _remote_content_length(url: str) -> int:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.head(url, timeout=(30, 60), allow_redirects=True)
            response.raise_for_status()
            size = int(response.headers["Content-Length"])
            if size <= 0:
                raise ValueError("Content-Length must be positive")
            return size
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unable to read remote file size for {url}: {last_error}") from last_error


def _byte_ranges(size: int, chunks: int) -> list[tuple[int, int]]:
    chunk_size = max(1, (size + chunks - 1) // chunks)
    return [
        (start, min(size - 1, start + chunk_size - 1))
        for start in range(0, size, chunk_size)
    ]


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
