"""Tests for nap.report — HTML report generation."""

import os
import tempfile
from pathlib import Path

from nap.report import Report


def test_set_replaces_placeholder():
    with tempfile.TemporaryDirectory() as tmp:
        report = Report(figures_dir=os.path.join(tmp, "figures"))
        report.set("SUBJECT_ID", "sub-001")
        assert "sub-001" in report._content
        assert "{{SUBJECT_ID}}" not in report._content


def test_append_keeps_placeholder():
    with tempfile.TemporaryDirectory() as tmp:
        report = Report(figures_dir=os.path.join(tmp, "figures"))
        report.append("ARTIFACT_REJECTION_ITERATIONS", "<p>Iteration 1</p>")
        assert "<p>Iteration 1</p>" in report._content
        assert "{{ARTIFACT_REJECTION_ITERATIONS}}" in report._content


def test_append_multiple():
    with tempfile.TemporaryDirectory() as tmp:
        report = Report(figures_dir=os.path.join(tmp, "figures"))
        report.append("ARTIFACT_REJECTION_ITERATIONS", "<p>First</p>")
        report.append("ARTIFACT_REJECTION_ITERATIONS", "<p>Second</p>")
        assert "<p>First</p>" in report._content
        assert "<p>Second</p>" in report._content


def test_save_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        figures_dir = os.path.join(tmp, "figures")
        report = Report(figures_dir=figures_dir)
        report.set("SUBJECT_ID", "sub-001")
        path = report.save(tmp)
        assert os.path.exists(path)
        assert path.endswith("report.html")


def test_save_cleans_unused_placeholders():
    with tempfile.TemporaryDirectory() as tmp:
        report = Report(figures_dir=os.path.join(tmp, "figures"))
        path = report.save(tmp)
        content = Path(path).read_text()
        assert "{{" not in content


def test_save_sets_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        report = Report(figures_dir=os.path.join(tmp, "figures"))
        path = report.save(tmp)
        content = Path(path).read_text()
        assert "TIMESTAMP" not in content
        # Should contain a date-like string
        assert "202" in content
