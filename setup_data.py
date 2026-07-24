#!/usr/bin/env python3
"""
Utility script to decompress the project archives and place their content
under the expected data layout:

    data/raw/          <- corpus archive (e.g. the vLLM 0.10.1 source tree)
    data/datasets/      <- datasets archive
        (UnansweredQuestions/, AnsweredQuestions/)

Usage:
    python setup_data.py <corpus_archive> <datasets_archive>

Supported archive formats: .zip, .tar, .tar.gz, .tgz, .tar.bz2
"""

import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

from tqdm import tqdm

RAW_DIR = Path("data/raw")
DATASETS_DIR = Path("data/datasets")


def _is_zip(archive_path: Path) -> bool:
    """Return True if the archive is a valid ZIP file."""
    return zipfile.is_zipfile(archive_path)


def _is_tar(archive_path: Path) -> bool:
    """Return True if the archive is a valid TAR file (any compression)."""
    return tarfile.is_tarfile(archive_path)


def _extract_zip(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip archive to dest_dir, showing a tqdm progress bar."""
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        desc = f"Extracting {archive_path.name}"
        for member in tqdm(members, desc=desc, unit="file"):
            zf.extract(member, dest_dir)


def _extract_tar(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar/.tar.gz/.tar.bz2 archive to dest_dir with progress."""
    with tarfile.open(archive_path) as tf:
        members = tf.getmembers()
        desc = f"Extracting {archive_path.name}"
        for member in tqdm(members, desc=desc, unit="file"):
            tf.extract(member, dest_dir)


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """
    Extract a single archive into dest_dir, creating dest_dir if needed.

    Args:
        archive_path: Path to the .zip/.tar/.tar.gz/.tar.bz2 archive.
        dest_dir: Destination directory the content is extracted into.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If the archive format is not recognised.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    if _is_zip(archive_path):
        _extract_zip(archive_path, dest_dir)
    elif _is_tar(archive_path):
        _extract_tar(archive_path, dest_dir)
    else:
        raise ValueError(
            f"Unsupported or corrupted archive format: {archive_path}"
        )


def _flatten_single_subdir(dest_dir: Path) -> None:
    """
    If dest_dir contains exactly one subdirectory and nothing else (a common
    artifact of archives created from a single top-level folder), move its
    content up one level and remove the now-empty wrapper directory.
    """
    entries: List[Path] = list(dest_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        for item in wrapper.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        wrapper.rmdir()


def main() -> None:
    """
    Parse CLI arguments and extract both archives to their destinations.
    """
    if len(sys.argv) != 3:
        print(
            "Usage: python setup_data.py <vllm_archive> <datasets_archive>"
        )
        sys.exit(1)

    corpus_archive = Path(sys.argv[1])
    datasets_archive = Path(sys.argv[2])

    try:
        print(f"Extracting corpus archive into {RAW_DIR}/ ...")
        extract_archive(corpus_archive, RAW_DIR)
        _flatten_single_subdir(RAW_DIR)
        print(f"Corpus ready under {RAW_DIR}/")

        print(f"Extracting datasets archive into {DATASETS_DIR}/ ...")
        extract_archive(datasets_archive, DATASETS_DIR)
        _flatten_single_subdir(DATASETS_DIR)
        print(f"Datasets ready under {DATASETS_DIR}/")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
