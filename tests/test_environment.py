"""Tests for nap.environment — dependency and version checking."""

import sys
from unittest.mock import patch

from nap.environment import check_environment


def test_check_environment_mock_mode():
    versions = check_environment(mock=True)
    assert "mne" in versions
    assert "matplotlib" in versions


def test_check_environment_mock_has_anthropic_entry():
    versions = check_environment(mock=True)
    assert "anthropic" in versions


def test_python_version_is_sufficient():
    assert sys.version_info >= (3, 10)
