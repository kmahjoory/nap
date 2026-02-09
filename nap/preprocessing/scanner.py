"""Scan a data folder and catalog all neuroimaging files."""

import os
from pathlib import Path


# Extension -> modality hints (definitive matches)
EEG_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".set"}
MEG_EXTENSIONS = {".sqd", ".con"}
MRI_EXTENSIONS = {".mgz", ".dcm"}

# Extensions that need context to classify
AMBIGUOUS_EXTENSIONS = {".fif", ".nii", ".nii.gz"}

# Companion files (not listed as datasets, but grouped with their header)
COMPANION_EXTENSIONS = {".eeg", ".vmrk", ".fdt"}


def scan_folder(data_dir: str) -> list[dict]:
    """Recursively scan a folder and return a list of detected neuroimaging files.

    Each entry: {"path": str, "name": str, "size_mb": float, "ext": str,
                 "rel_path": str, "hint": str}
    hint is one of: "eeg", "meg", "mri", "fmri", "ambiguous", "companion"
    """
    data_dir = Path(data_dir)
    files = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        # Handle .nii.gz double extension
        if path.name.lower().endswith(".nii.gz"):
            ext = ".nii.gz"

        # Skip non-neuroimaging files
        all_known = EEG_EXTENSIONS | MEG_EXTENSIONS | MRI_EXTENSIONS | AMBIGUOUS_EXTENSIONS | COMPANION_EXTENSIONS
        if ext not in all_known:
            continue

        size_mb = path.stat().st_size / (1024 * 1024)
        rel_path = str(path.relative_to(data_dir))

        # Classify
        if ext in COMPANION_EXTENSIONS:
            hint = "companion"
        elif ext in EEG_EXTENSIONS:
            hint = "eeg"
        elif ext in MEG_EXTENSIONS:
            hint = "meg"
        elif ext in MRI_EXTENSIONS:
            hint = "mri"
        elif ext in AMBIGUOUS_EXTENSIONS:
            hint = _guess_from_name(path.name, rel_path)
        else:
            hint = "unknown"

        files.append({
            "path": str(path),
            "name": path.name,
            "size_mb": size_mb,
            "ext": ext,
            "rel_path": rel_path,
            "hint": hint,
        })

    return files


def _guess_from_name(name: str, rel_path: str) -> str:
    """Guess modality from filename/path patterns."""
    lower = (name + "/" + rel_path).lower()

    if any(k in lower for k in ("bold", "func", "task-")):
        return "fmri"
    if any(k in lower for k in ("t1w", "t2w", "anat", "struct", "mprage", "flair")):
        return "mri"
    if "meg" in lower:
        return "meg"
    if "eeg" in lower:
        return "eeg"

    # .fif default to ambiguous, .nii/.nii.gz default to ambiguous
    return "ambiguous"


def list_organized_datasets(data_dir: str) -> list[dict]:
    """List datasets available in the organized project/data/ folder.

    Returns list of {"path": str, "modality": str, "subject": str, "name": str}.
    Only returns header files (not companions).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    datasets = []
    header_exts = EEG_EXTENSIONS | MEG_EXTENSIONS | MRI_EXTENSIONS | AMBIGUOUS_EXTENSIONS

    for path in sorted(data_dir.rglob("*")):
        if not path.is_symlink() and not path.is_file():
            continue
        if path.suffix.lower() not in header_exts:
            continue
        # Handle .nii.gz
        if path.name.lower().endswith(".nii.gz") and ".nii.gz" not in header_exts:
            continue

        # Extract modality and subject from path: data/<modality>/<subject>/file
        rel = path.relative_to(data_dir)
        parts = rel.parts
        modality = parts[0] if len(parts) > 1 else "unknown"
        subject = parts[1] if len(parts) > 2 else "unknown"

        # Verify symlink target exists
        target_ok = True
        if path.is_symlink():
            target_ok = path.resolve().exists()

        datasets.append({
            "path": str(path),
            "modality": modality,
            "subject": subject,
            "name": path.name,
            "symlink_ok": target_ok,
        })

    return datasets


def format_scan_summary(files: list[dict]) -> str:
    """Format the scan results as a text summary for the LLM."""
    if not files:
        return "No neuroimaging files found."

    lines = []
    for f in files:
        if f["hint"] == "companion":
            continue
        lines.append(f"  {f['rel_path']:<60} {f['size_mb']:>8.1f} MB  [{f['hint']}]")

    return f"Found {len(lines)} neuroimaging files:\n" + "\n".join(lines)
