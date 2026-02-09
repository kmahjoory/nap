"""Low-level EEG operations — loading, summary, plotting.

No decisions, no LLM calls. Pure MNE operations that skills delegate to.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*Channel locations not available.*")


def _fix_brainvision_refs(path: Path):
    """Fix BrainVision headers that reference renamed companion files.

    Some datasets (e.g., LEMON) have .vhdr files that internally reference
    old filenames. If the referenced files don't exist but files with the
    same stem as the .vhdr do, create symlinks.
    """
    with open(path) as f:
        header = f.read()

    for key in ("DataFile=", "MarkerFile="):
        for line in header.splitlines():
            if line.startswith(key):
                ref_name = line.split("=", 1)[1].strip()
                ref_path = path.parent / ref_name
                if not ref_path.exists():
                    # Try to find the actual file with the .vhdr stem
                    actual = path.parent / (path.stem + ref_path.suffix)
                    if actual.exists():
                        ref_path.symlink_to(actual.name)


def load_raw(data_path: str) -> mne.io.Raw:
    """Load raw EEG file in read-only mode. Never modifies the original file."""
    path = Path(data_path).resolve()  # resolve symlinks so MNE finds companion files
    suffix = path.suffix.lower()

    loaders = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".set": mne.io.read_raw_eeglab,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported: {', '.join(loaders.keys())}"
        )

    # Fix mismatched companion file references in BrainVision headers
    if suffix == ".vhdr":
        _fix_brainvision_refs(path)

    return loader(str(path), preload=False, verbose=False)


def set_standard_montage(raw: mne.io.Raw) -> int:
    """Set standard 10-05 montage on EEG channels. Returns number matched."""
    montage = mne.channels.make_standard_montage("standard_1005")
    matched = set(raw.ch_names) & set(montage.ch_names)
    if matched:
        raw.set_montage(montage, on_missing="ignore", verbose=False)
    return len(matched)


def get_data_summary(raw: mne.io.Raw, data_path: str) -> dict:
    """Extract metadata from raw EEG into a summary dict."""
    path = Path(data_path)
    ch_types = raw.get_channel_types()
    type_counts = {}
    for t in ch_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    channels_str = ", ".join(f"{count} {t.upper()}" for t, count in type_counts.items())

    duration_s = raw.times[-1]
    minutes = int(duration_s // 60)
    seconds = int(duration_s % 60)

    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2.0

    # Events
    try:
        events = mne.find_events(raw, verbose=False)
        n_events = len(events)
        n_types = len(set(events[:, 2]))
        events_str = f"{n_events} ({n_types} types)"
    except Exception:
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            n_events = len(events)
            n_types = len(event_id)
            events_str = f"{n_events} ({n_types} types)"
        except Exception:
            events_str = "None found"

    # Reference
    ref = raw.info.get("custom_ref_applied")
    ref_str = "Custom reference applied" if ref else "Not set / unknown"

    # Bad channels
    bads = raw.info["bads"]
    bads_str = ", ".join(bads) if bads else "None"

    # File size
    total_size = sum(f.stat().st_size for f in path.parent.glob(f"{path.stem}.*"))
    file_size_mb = total_size / (1024 * 1024)

    return {
        "File": path.name,
        "Format": path.suffix.upper().lstrip("."),
        "Channels": channels_str,
        "Sampling rate": f"{sfreq:.1f} Hz",
        "Nyquist frequency": f"{nyquist:.1f} Hz",
        "Duration": f"{minutes} min {seconds} s",
        "Events": events_str,
        "Reference": ref_str,
        "Bad channels (marked)": bads_str,
        "File size": f"{file_size_mb:.1f} MB",
    }


PSD_TARGET_SFREQ = 250.0  # resample to this before PSD if sfreq is higher


def plot_psd(raw: mne.io.Raw, out_path: str, title: str = "PSD") -> str:
    """Compute and save a PSD plot. Resamples to 250 Hz if needed.

    - Resamples copy to 250 Hz for consistent frequency range (0-125 Hz)
    - Log scale (dB) for power
    - X-axis ticks at every 10 Hz
    - Wide figure for better visibility
    """
    raw_psd = raw.copy()
    if raw_psd.info["sfreq"] > PSD_TARGET_SFREQ:
        raw_psd.resample(PSD_TARGET_SFREQ, verbose=False)

    psd = raw_psd.compute_psd(verbose=False)
    data, freqs = psd.get_data(return_freqs=True)  # data: (n_channels, n_freqs) in V^2/Hz
    data_db = 10 * np.log10(data + 1e-30)  # convert to dB

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(freqs, data_db.T, linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xticks(np.arange(0, freqs[-1] + 1, 10))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    del raw_psd
    return out_path


def plot_traces(raw: mne.io.Raw, out_path: str, title: str = "Traces",
                duration: float = 30, n_channels: int = 30) -> str:
    """Save a time-series trace plot. Returns the file path."""
    fig = raw.plot(
        duration=duration, n_channels=n_channels, scalings="auto",
        title=title, show=False,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ica_topomaps(ica, raw, out_path: str) -> str:
    """Plot grid of ICA component topomaps. Requires montage to be set first."""
    fig = ica.plot_components(show=False, inst=raw)
    if isinstance(fig, list):
        for f in fig[1:]:
            plt.close(f)
        fig = fig[0]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ica_spectra(ica, raw, out_path: str) -> str:
    """Plot power spectrum of each ICA component in a grid."""
    sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
    sfreq = raw.info["sfreq"]
    n = ica.n_components_
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    # Resample sources if sfreq > 250 for consistent frequency range
    from scipy.signal import welch
    target_sfreq = min(sfreq, PSD_TARGET_SFREQ)
    if sfreq > PSD_TARGET_SFREQ:
        from scipy.signal import resample_poly
        factor = int(sfreq / PSD_TARGET_SFREQ)
        sources = resample_poly(sources, 1, factor, axis=1)
        target_sfreq = sfreq / factor

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n):
        freqs, pxx = welch(sources[i], fs=target_sfreq, nperseg=min(1024, sources.shape[1]))
        pxx_db = 10 * np.log10(pxx + 1e-30)
        axes[i].plot(freqs, pxx_db, linewidth=0.8, color="steelblue")
        axes[i].set_title(f"IC{i}", fontsize=8)
        axes[i].set_xlim(0, target_sfreq / 2)
        axes[i].set_xticks(np.arange(0, target_sfreq / 2 + 1, 20))
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("ICA Component Power Spectra (dB)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


EPOCH_DURATION = 4.0  # seconds per epoch for heatmap time courses


def plot_ica_timecourses(ica, raw, out_path: str) -> str:
    """Plot ICA time courses as epoch x time heatmaps (one per component).

    Continuous data is split into fixed-length epochs. Each epoch is one row.
    X-axis = time within epoch, Y-axis = epoch number, color = amplitude.
    """
    sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
    sfreq = raw.info["sfreq"]
    n = ica.n_components_

    # Resample to 250 Hz for manageable size
    if sfreq > PSD_TARGET_SFREQ:
        from scipy.signal import resample_poly
        factor = int(sfreq / PSD_TARGET_SFREQ)
        sources = resample_poly(sources, 1, factor, axis=1)
        sfreq = sfreq / factor

    epoch_len = int(EPOCH_DURATION * sfreq)
    n_epochs = sources.shape[1] // epoch_len

    # Trim and reshape: (n_components, n_epochs, epoch_len)
    sources = sources[:, :n_epochs * epoch_len].reshape(n, n_epochs, epoch_len)

    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n):
        data = sources[i]
        vmax = np.percentile(np.abs(data), 99)
        axes[i].imshow(
            data, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            extent=[0, EPOCH_DURATION, n_epochs, 0],
        )
        axes[i].set_title(f"IC{i}", fontsize=8)
        axes[i].set_xlabel("Time (s)", fontsize=6)
        axes[i].set_ylabel("Epoch", fontsize=6)
        axes[i].tick_params(labelsize=5)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"ICA Time Courses ({EPOCH_DURATION:.0f}s epochs, {n_epochs} epochs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def remove_channels(raw: mne.io.Raw, channels: list[str]) -> list[str]:
    """Mark channels as bad. Returns list of channels actually marked."""
    valid = [ch for ch in channels if ch in raw.ch_names]
    raw.info["bads"] = list(set(raw.info["bads"] + valid))
    if valid:
        raw.interpolate_bads(verbose=False)
    return valid


def annotate_segments(raw: mne.io.Raw, segments: list[tuple[float, float]]) -> int:
    """Add bad annotations for time segments. Returns count added."""
    for start, end in segments:
        raw.annotations.append(onset=start, duration=end - start, description="BAD_NAP")
    return len(segments)


def run_ica(raw: mne.io.Raw, n_components: int = 20):
    """Fit ICA on raw data. Returns the ICA object."""
    from mne.preprocessing import ICA
    ica = ICA(n_components=n_components, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)
    return ica


def remove_ica_components(ica, raw: mne.io.Raw, exclude: list[int]) -> mne.io.Raw:
    """Remove ICA components from raw data. Returns modified raw."""
    ica.exclude = exclude
    ica.apply(raw, verbose=False)
    return raw


def apply_notch(raw: mne.io.Raw, freq: float = 50.0) -> None:
    """Apply notch filter at specified frequency + harmonics."""
    nyquist = raw.info["sfreq"] / 2.0
    freqs = [freq * i for i in range(1, int(nyquist // freq) + 1)]
    raw.notch_filter(freqs, verbose=False)


def make_temp_filtered_copy(raw: mne.io.Raw) -> mne.io.Raw:
    """Create a temporary filtered copy for inspection. Caller discards it after use."""
    sfreq = raw.info["sfreq"]
    lp_freq = (sfreq / 2.0) - 10.0
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=1.0, h_freq=lp_freq, verbose=False)
    return raw_filtered
