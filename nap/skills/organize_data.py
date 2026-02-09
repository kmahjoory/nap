"""Skill: Organize Data — scan a folder, identify modalities, create symlinks."""

import os
from pathlib import Path

from nap.skills.base import Skill
from nap.memory import Memory
from nap.preprocessing.scanner import scan_folder, format_scan_summary
from nap.llm import call_text


class OrganizeData(Skill):
    name = "Organize Data"
    status_label = "organizing data"
    needs_confirm = True

    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def act(self, memory: Memory) -> dict[str, str]:
        """Scan the source folder and store results in memory."""
        self.scan_results = scan_folder(self.source_dir)
        self.scan_summary = format_scan_summary(self.scan_results)
        # No plots for this skill
        return {}

    def build_prompt(self, memory: Memory) -> str:
        """Not used — this skill uses call_text instead of vision."""
        return ""

    def see(self, plots: dict[str, str], memory: Memory) -> str:
        """Send file listing to LLM to classify and propose organization."""
        prompt = (
            "You are a neuroimaging data organization expert.\n\n"
            f"I scanned this folder: {self.source_dir}\n\n"
            f"{self.scan_summary}\n\n"
            "For each file, confirm or correct the modality classification "
            "(eeg, meg, mri, fmri). For ambiguous files, decide based on "
            "filename, path, and file size.\n\n"
            "Then propose how to organize them into:\n"
            "  data/eeg/\n  data/meg/\n  data/mri/\n  data/fmri/\n\n"
            "Group files by subject. Keep companion files together "
            "(e.g., .vhdr + .eeg + .vmrk). Be concise."
        )
        return call_text(prompt, mock=memory.mock)

    def confirm(self, findings: str, memory: Memory) -> str | None:
        """Show proposed organization, get approval, then create symlinks."""
        user_decision = super().confirm(findings, memory)

        if user_decision is None or "reject" not in (user_decision or "").lower():
            self._create_symlinks(memory)

        return user_decision

    def _create_symlinks(self, memory: Memory):
        """Create organized symlink structure in project/data/."""
        data_dir = Path(memory.figures_dir).parent / "data"

        # Group non-companion files by hint
        datasets = [f for f in self.scan_results if f["hint"] != "companion"]
        companions = [f for f in self.scan_results if f["hint"] == "companion"]

        created = 0
        for f in datasets:
            modality = f["hint"]
            if modality in ("ambiguous", "unknown"):
                modality = "unclassified"

            # Find subject from path (use parent folder name or filename stem)
            src = Path(f["path"])
            subject = src.parent.name if src.parent.name != Path(self.source_dir).name else "unsorted"

            dest_dir = data_dir / modality / subject
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest = dest_dir / src.name
            if not dest.exists():
                dest.symlink_to(src.resolve())
                created += 1

            # Link companion files (same stem, same source folder)
            for c in companions:
                csrc = Path(c["path"])
                if csrc.parent == src.parent and csrc.stem == src.stem:
                    cdest = dest_dir / csrc.name
                    if not cdest.exists():
                        cdest.symlink_to(csrc.resolve())
                        created += 1

        print(f"\n  {created} symlinks created in {data_dir}")
