#!/usr/bin/env python3
"""Download selected MIMIC-IV-ECG waveform files from a manifest.

This is a small alternative to GNU wget for Windows machines. It reads
download_manifest.csv produced by build_mimic_waveform_manifest.py and saves
.hea/.dat files under the same relative files/pXXXX/... layout.
"""

from __future__ import annotations

import argparse
import getpass
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


def relative_path_from_url(url: str, dataset_version: str = "/mimic-iv-ecg/1.0/") -> Path:
    path = urlparse(url).path
    if dataset_version in path:
        path = path.split(dataset_version, 1)[1]
    else:
        path = Path(path).name
    return Path(path)


def download_one(
    session: requests.Session,
    url: str,
    destination: Path,
    auth: tuple[str, str] | None,
    force: bool,
    timeout: int,
    retries: int,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0

    if force and destination.exists():
        destination.unlink()
        existing_size = 0

    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(
                url,
                stream=True,
                auth=auth,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            ) as response:
                if response.status_code == 416:
                    return "already_complete"
                if response.status_code == 401:
                    raise RuntimeError("401 Unauthorized: check PhysioNet credentials")
                if existing_size > 0 and response.status_code == 200:
                    mode = "wb"
                response.raise_for_status()

                with destination.open(mode + "") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            return "downloaded" if existing_size == 0 else "resumed"
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Failed after {retries} attempts: {url}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset_manifest\download_manifest.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"C:\Users\ISY\Downloads\mimic_ecg_subset"),
    )
    parser.add_argument("--username", default=None, help="PhysioNet username")
    parser.add_argument("--password", default=None, help="PhysioNet password")
    parser.add_argument("--limit", type=int, default=None, help="Download first N ECG records only")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    auth = None
    if args.username:
        password = args.password
        if password is None:
            password = getpass.getpass("PhysioNet password: ")
        auth = (args.username, password)

    urls = []
    for row in manifest.itertuples(index=False):
        urls.append(str(row.hea_url))
        urls.append(str(row.dat_url))

    session = requests.Session()
    counts = {"downloaded": 0, "resumed": 0, "already_complete": 0}

    for index, url in enumerate(urls, start=1):
        rel_path = relative_path_from_url(url)
        destination = args.output_dir / rel_path
        status = download_one(
            session=session,
            url=url,
            destination=destination,
            auth=auth,
            force=args.force,
            timeout=args.timeout,
            retries=args.retries,
        )
        counts[status] = counts.get(status, 0) + 1
        if index == 1 or index % 100 == 0 or index == len(urls):
            print(f"[{index}/{len(urls)}] {status}: {rel_path}")

    print("Done:", counts)
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
