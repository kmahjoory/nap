"""Tests for nap.cli — CLI helper functions."""

from pathlib import Path

from nap.cli import _extract_subject_id, _get_cpu_name, _get_ram_gb, build_data_tree_lines


def test_extract_subject_id_from_vhdr():
    path = Path("/data/eeg/sub-032301/sub-032301.vhdr")
    assert _extract_subject_id(path) == "sub-032301"


def test_extract_subject_id_from_edf():
    path = Path("/data/sub-001.edf")
    assert _extract_subject_id(path) == "sub-001"


def test_extract_subject_id_strips_extension():
    path = Path("/data/recording.vhdr")
    assert _extract_subject_id(path) == "recording"


def test_extract_subject_id_no_extension():
    path = Path("/data/sub-001")
    assert _extract_subject_id(path) == "sub-001"


def test_get_cpu_name_returns_string():
    name = _get_cpu_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_ram_gb_returns_string():
    ram = _get_ram_gb()
    assert isinstance(ram, str)


def test_build_data_tree_lines_empty(tmp_path):
    lines = build_data_tree_lines(tmp_path)
    assert lines == []


def test_build_data_tree_lines_with_files(tmp_path):
    (tmp_path / "sub-001.vhdr").touch()
    lines = build_data_tree_lines(tmp_path)
    assert len(lines) == 1
    assert "sub-001.vhdr" in lines[0]
