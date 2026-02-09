"""Environment detection and dependency checking."""

import importlib
import os
import subprocess
import sys

MIN_PYTHON = (3, 10)


def check_environment(mock: bool = False) -> dict:
    """Check Python version, required packages, and API key.

    Returns a dict with package versions.
    Exits with a clear error if something is missing.
    """
    # Python version
    if sys.version_info < MIN_PYTHON:
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        req = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        print(f"\n  Error: Python {req}+ required, got {v}")
        print(f"  Install from python.org or use: pyenv install {req}")
        sys.exit(1)

    missing = []
    versions = {}

    # MNE-Python
    try:
        import mne
        versions["mne"] = mne.__version__
    except ImportError:
        missing.append("mne")

    # matplotlib
    try:
        import matplotlib
        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        missing.append("matplotlib")

    # anthropic (only required in live mode)
    try:
        import anthropic
        versions["anthropic"] = anthropic.__version__
    except ImportError:
        if not mock:
            missing.append("anthropic")
        else:
            versions["anthropic"] = "skipped (mock mode)"

    if missing:
        print(f"\n  Error: Missing required packages: {', '.join(missing)}")
        print(f"  Install them with: pip install {' '.join(missing)}")
        sys.exit(1)

    # API key check (only in live mode)
    if not mock:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n  Error: ANTHROPIC_API_KEY environment variable not set.")
            print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            print("  Or run with --mock to use simulated LLM responses.")
            sys.exit(1)

    return versions


def ensure_package(import_name: str, pip_name: str | None = None):
    """Import a package, auto-installing it if missing.

    Use for optional dependencies that specific skills need.
    Core dependencies should stay in pyproject.toml.
    """
    try:
        importlib.import_module(import_name)
    except ImportError:
        package = pip_name or import_name
        print(f"  Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"],
        )
        importlib.import_module(import_name)
