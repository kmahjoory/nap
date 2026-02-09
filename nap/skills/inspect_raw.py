"""Skill: Inspect Raw — display metadata, channel layout, and check for noise."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nap.skills.base import Skill, status
from nap.memory import Memory
from nap.preprocessing.eeg import plot_psd


def _print_channel_layout(raw):
    """Print a 2D ASCII topographic layout of EEG channels in the terminal."""
    montage = raw.get_montage()
    if montage is None:
        print("    (No montage set — cannot display channel layout)")
        return

    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]

    # Only include channels present in raw
    ch_names = [ch for ch in raw.ch_names if ch in ch_pos]
    if not ch_names:
        print("    (No channels with known positions)")
        return

    coords = np.array([ch_pos[ch][:2] for ch in ch_names])  # x, y only

    # Filter out channels with NaN positions
    valid = ~np.isnan(coords).any(axis=1)
    ch_names = [ch for ch, v in zip(ch_names, valid) if v]
    coords = coords[valid]
    if len(ch_names) == 0:
        print("    (No channels with valid positions)")
        return

    # Normalize to grid
    width, height = 72, 32
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Add padding to avoid edge clipping
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0

    grid = [[" "] * width for _ in range(height)]

    for name, (x, y) in zip(ch_names, coords):
        col = int((x - x_min) / x_range * (width - len(name) - 1))
        row = int((1 - (y - y_min) / y_range) * (height - 1))  # flip y: nose up
        col = max(0, min(col, width - len(name)))
        row = max(0, min(row, height - 1))

        # Place label if space is free
        fits = all(grid[row][col + k] == " " for k in range(len(name)) if col + k < width)
        if fits:
            for k, ch in enumerate(name):
                if col + k < width:
                    grid[row][col + k] = ch

    G, R = "\033[38;2;100;148;113m", "\033[0m"
    print(f"\n    {G}Channel Layout (2D topographic view){R}")
    print(f"    {'—' * width}")
    for row in grid:
        line = "".join(row).rstrip()
        if line:
            print(f"    {line}")
    print(f"    {'—' * width}")
    print(f"    {G}Nose ↑    Left ←  → Right{R}\n")


def _plot_channel_layout(raw, out_path: str) -> str:
    """Render channel layout as an image using MNE's plot_sensors."""
    fig = raw.plot_sensors(show=False, show_names=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _print_metadata(memory):
    """Print recording metadata to the terminal."""
    G, R = "\033[38;2;100;148;113m", "\033[0m"
    print(f"\n    {G}Recording Metadata{R}")
    print(f"    {'—' * 40}")
    for key, val in memory.data_summary.items():
        print(f"    {key}: {val}")
    print(f"    {'—' * 40}")


class InspectRaw(Skill):
    name = "Inspect Raw"
    status_label = "inspecting raw data"
    needs_confirm = True

    def act(self, memory: Memory) -> dict[str, str]:
        """Print metadata and channel layout, generate PSD plot."""
        if memory.verbose:
            _print_metadata(memory)
        _print_channel_layout(memory.raw)

        status(memory, self.status_label, "loading data")
        fig_dir = Path(memory.figures_dir)
        raw = memory.raw.copy().load_data(verbose=False)

        status(memory, self.status_label, "plotting PSD")
        plots = {}
        plots["raw_psd"] = plot_psd(raw, str(fig_dir / "raw_psd.png"), "Raw PSD (unfiltered)")

        # Channel layout image for report
        status(memory, self.status_label, "plotting channel layout")
        layout_path = str(fig_dir / "channel_layout.png")
        _plot_channel_layout(memory.raw, layout_path)
        if memory.report:
            memory.report.add_image("CHANNEL_LAYOUT", layout_path, "Channel Layout")

        del raw
        return plots

    def see(self, plots: dict[str, str], memory: Memory) -> str:
        """Send plots to LLM, then write results to report."""
        findings = super().see(plots, memory)

        # Write to report
        if memory.report:
            memory.report.add_image("INSPECT_RAW_PLOTS", plots["raw_psd"], "Raw PSD (unfiltered)")
            memory.report.set("INSPECT_RAW_FINDINGS", f'<div class="findings">{findings}</div>')

        return findings

    def build_prompt(self, memory: Memory) -> str:
        """Prompt for noise inspection only."""
        context = memory.get_context()
        return (
            "You are an expert neuroimaging researcher inspecting a raw EEG recording.\n\n"
            f"Recording metadata:\n{context}\n\n"
            "Attached is a PSD plot of the raw unfiltered data.\n\n"
            "Report:\n"
            "1. **Line noise**: Look at the PSD carefully. If there is a sharp narrow peak, "
            "read its exact frequency from the x-axis (50 Hz in Europe/Asia/Africa, "
            "60 Hz in Americas). Report the exact frequency you observe, do not guess.\n"
            "2. **General noise observations**: Overall spectral quality, "
            "broadband noise level, any unusual spectral features.\n\n"
            "Be specific about frequencies and observations."
        )
