"""Tests for nap.preprocessing.scanner — file scanning and classification."""

import os
import tempfile
from pathlib import Path

from nap.preprocessing.scanner import scan_folder, list_organized_datasets


def test_scan_folder_finds_eeg_files():
    with tempfile.TemporaryDirectory() as tmp:
        # Create mock EEG files
        Path(tmp, "sub-001.vhdr").touch()
        Path(tmp, "sub-001.eeg").touch()
        Path(tmp, "sub-001.vmrk").touch()

        result = scan_folder(tmp)
        extensions = [f["ext"] for f in result]
        assert ".vhdr" in extensions


def test_scan_folder_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        result = scan_folder(tmp)
        assert result == []


def test_list_organized_datasets_empty():
    with tempfile.TemporaryDirectory() as tmp:
        result = list_organized_datasets(tmp)
        assert result == []


def test_list_organized_datasets_with_structure():
    with tempfile.TemporaryDirectory() as tmp:
        # Create organized structure
        ds_dir = Path(tmp, "eeg", "sub-001")
        ds_dir.mkdir(parents=True)
        data_file = ds_dir / "sub-001.vhdr"
        data_file.touch()

        result = list_organized_datasets(tmp)
        assert len(result) == 1
        assert result[0]["name"] == "sub-001.vhdr"
