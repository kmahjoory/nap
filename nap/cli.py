"""NAP CLI — conversation loop and skill orchestration."""

import argparse
import os
import platform
import random
import sys
from pathlib import Path

# NAP green (#649471)
G = "\033[38;2;100;148;113m"
B = "\033[1m"
R = "\033[0m"


def ask(prompt: str) -> str:
    """Get user input. Exits cleanly on quit/exit/q or Ctrl+C."""
    try:
        response = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {G}Goodbye.{R}\n")
        sys.exit(0)
    if response.lower() in ("quit", "exit", "q"):
        print(f"\n  {G}Goodbye.{R}\n")
        sys.exit(0)
    return response


def status(label: str, step: str):
    """Print a single updating status line."""
    sys.stdout.write(f"\r  {G}NAP is {label}...{R} {step}    ")
    sys.stdout.flush()


def print_banner():
    print(f"""{G}{B}
  _   _    _    ____
 | \\ | |  / \\  |  _ \\
 |  \\| | / _ \\ | |_) |
 | |\\  |/ ___ \\|  __/
 |_| \\_/_/   \\_\\_|
{R}\033[38;2;149;160;152m  Neuroimaging Agentic Analysis Platform
  Prototype v0.1.0{R}
""")


def print_read_only_notice():
    print(f"{G}{'=' * 60}")
    print(f"  NOTE: Your raw data will NOT be modified.")
    print(f"  NAP reads your data in read-only mode.")
    print(f"  All outputs (plots, reports) are saved to")
    print(f"  the project folder only.")
    print(f"{'=' * 60}{R}\n")


def print_summary_table(summary: dict):
    print(f"\n  {G}Data Summary{R}")
    print("  " + "-" * 44)
    for key, value in summary.items():
        print(f"  {key:<24} {value}")
    print("  " + "-" * 44 + "\n")


def _get_cpu_name() -> str:
    """Get CPU model name."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _get_ram_gb() -> str:
    """Get total RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / (1024 ** 2):.0f} GB"
    except OSError:
        pass
    return "unknown"


def get_system_info_html() -> str:
    """Return system info as an HTML table."""
    rows = [
        ("CPU", _get_cpu_name()),
        ("RAM", _get_ram_gb()),
        ("Platform", platform.platform()),
    ]
    html = "<table>"
    for key, val in rows:
        html += f"<tr><th>{key}</th><td>{val}</td></tr>"
    html += "</table>"
    return html


def print_system_info():
    print(f"  {G}System:{R} {_get_cpu_name()} | RAM: {_get_ram_gb()}")


def build_data_tree_lines(data_dir: Path) -> list[str]:
    """Build tree view lines of the organized data directory."""
    lines = []

    def _walk(directory: Path, prefix: str = ""):
        entries = sorted(directory.iterdir())
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if not e.is_dir()]

        file_groups = {}
        for f in files:
            file_groups.setdefault(f.stem, []).append(f)

        items = dirs + list(file_groups.values())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            extension = "    " if is_last else "\u2502   "

            if isinstance(item, Path):
                lines.append(f"{prefix}{connector}{item.name}/")
                _walk(item, prefix + extension)
            else:
                main_file = item[0]
                if len(item) > 1:
                    companions = [f.suffix for f in item[1:]]
                    lines.append(f"{prefix}{connector}{main_file.name} (+ companion {'/'.join(companions)} files)")
                else:
                    lines.append(f"{prefix}{connector}{main_file.name}")

    _walk(data_dir)
    return lines


def print_data_tree(data_dir: Path):
    """Print a tree view of the organized data directory."""
    if not data_dir.exists():
        return
    print(f"\n  {G}Data Organization{R}")
    for line in build_data_tree_lines(data_dir):
        print(f"  {line}")
    print()


def print_data_info(summary: dict):
    """Print minimal one-liner of data info."""
    print(f"  {G}Data:{R} {summary['File']} | {summary['Channels']} | {summary['Sampling rate']} | {summary['Duration']}")


def _extract_subject_id(data_path: Path) -> str:
    """Extract subject ID from data path. Falls back to filename stem."""
    EEG_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".fif", ".set", ".eeg", ".vmrk", ".fdt"}

    def _strip_extensions(name: str) -> str:
        while Path(name).suffix.lower() in EEG_EXTENSIONS:
            name = Path(name).stem
        return name

    # Check parent directories for sub-XXXX pattern
    for part in data_path.parts:
        if part.startswith("sub-"):
            return _strip_extensions(part)
    return _strip_extensions(data_path.name)


def main():
    parser = argparse.ArgumentParser(description="NAP -- Neuroimaging Agentic Analysis Platform")
    parser.add_argument("--mock", action="store_true", help="Use simulated LLM responses (no API key needed)")
    parser.add_argument("--project", type=str, help="Project folder for outputs")
    parser.add_argument("--data", type=str, help="Path to raw data file or folder")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (prompts for input)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    verbose = args.verbose
    interactive = args.interactive

    print_banner()

    # Environment check
    from nap.environment import check_environment
    versions = check_environment(mock=args.mock)
    print_system_info()
    if verbose:
        mode_str = "mock (simulated LLM)" if args.mock else "live (Claude API)"
        print(f"  Environment OK -- mode: {mode_str}")
        for pkg, ver in versions.items():
            print(f"    {pkg}: {ver}")
        print()

    # Get paths
    project_dir = args.project
    if not project_dir:
        if interactive:
            project_dir = ask("  Enter project folder path: ")
        else:
            print(f"  Error: --project is required in non-interactive mode.")
            sys.exit(1)

    data_path = args.data
    if not data_path:
        if interactive:
            data_path = ask("  Enter raw data file or folder path: ")
        else:
            print(f"  Error: --data is required in non-interactive mode.")
            sys.exit(1)

    project_dir = Path(project_dir)
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"\n  Error: Path not found: {data_path}")
        sys.exit(1)

    if verbose:
        print(f"\n  Project folder: {project_dir}")
        print_read_only_notice()

    from nap.memory import Memory

    # If data_path is a folder, organize then pick a dataset
    data_tree_html = ""
    if data_path.is_dir():
        org_figures = project_dir / "figures"
        org_figures.mkdir(parents=True, exist_ok=True)
        org_memory = Memory(
            raw=None, data_path=str(data_path), data_summary={},
            figures_dir=str(org_figures), mock=args.mock,
            verbose=verbose, interactive=interactive,
        )

        from nap.skills.organize_data import OrganizeData
        OrganizeData(source_dir=str(data_path)).run(org_memory)

        # Show organized data tree
        organized_dir = project_dir / "data"
        print_data_tree(organized_dir)
        tree_lines = build_data_tree_lines(organized_dir)
        data_tree_html = '<div class="tree">' + "\n".join(tree_lines) + "</div>"

        # List available datasets from organized folder
        from nap.preprocessing.scanner import list_organized_datasets
        datasets = list_organized_datasets(str(organized_dir))

        if not datasets:
            print("\n  Error: No datasets found after organizing.")
            sys.exit(1)

        if len(datasets) == 1:
            data_path = Path(datasets[0]["path"])
            if verbose:
                print(f"\n  Auto-selected: {datasets[0]['name']}")
        elif interactive:
            print(f"\n  {G}Available datasets:{R}")
            for i, ds in enumerate(datasets, 1):
                s = "ok" if ds["symlink_ok"] else "BROKEN"
                print(f"    {i}. [{ds['modality']}] {ds['subject']}/{ds['name']}  ({s})")

            choice = ask(f"\n  Select dataset [1-{len(datasets)}]: ")
            try:
                idx = int(choice) - 1
                data_path = Path(datasets[idx]["path"])
            except (ValueError, IndexError):
                print("  Invalid selection.")
                sys.exit(1)
        else:
            # Non-interactive: pick randomly
            ds = random.choice(datasets)
            data_path = Path(ds["path"])
            if verbose:
                print(f"\n  Randomly selected: {ds['name']}")

        # Verify symlink works
        if data_path.is_symlink() and not data_path.resolve().exists():
            print(f"\n  Error: Symlink broken — target not found: {data_path.resolve()}")
            sys.exit(1)

    # Analysis plan
    analysis_plan = ""
    if interactive:
        print("\n  Describe your analysis (or press Enter to skip):")
        analysis_plan = ask("  > ")
        if analysis_plan:
            print(f"  Plan noted.\n")

    # Load data + build memory
    if not verbose:
        status("loading data", "reading file")
    else:
        print("  Loading data...")

    from nap.preprocessing.eeg import load_raw, get_data_summary, set_standard_montage
    from nap.llm import call_text

    raw = load_raw(str(data_path))
    n_matched = set_standard_montage(raw)
    if verbose:
        if n_matched:
            print(f"  Montage: standard_1005 ({n_matched} channels matched)")
    summary = get_data_summary(raw, str(data_path))
    if verbose:
        print_summary_table(summary)
    else:
        sys.stdout.write("\n")
    print_data_info(summary)

    # Set up per-subject output folder
    subject_id = _extract_subject_id(data_path)
    subject_dir = project_dir / subject_id
    figures_dir = subject_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Create report
    from nap.report import Report
    report = Report(figures_dir=str(figures_dir))
    report.set("SUBJECT_ID", subject_id)
    report.set("SYSTEM_INFO", get_system_info_html())
    report.set("DATA_ORGANIZATION", data_tree_html)

    # Data summary table for report
    summary_html = "<table>"
    for key, val in summary.items():
        summary_html += f"<tr><th>{key}</th><td>{val}</td></tr>"
    summary_html += "</table>"
    report.set("DATA_SUMMARY", summary_html)

    memory = Memory(
        raw=raw,
        data_path=str(data_path),
        data_summary=summary,
        figures_dir=str(figures_dir),
        mock=args.mock,
        analysis_plan=analysis_plan,
        verbose=verbose,
        interactive=interactive,
        report=report,
    )

    # Run Inspect Raw
    from nap.skills.inspect_raw import InspectRaw
    InspectRaw().run(memory)

    # Propose preprocessing plan
    last_findings = memory.log[-1].findings
    plan_prompt = (
        "You are an EEG preprocessing expert. Based on the inspection findings below "
        "and the researcher's analysis plan, propose a concrete preprocessing plan.\n\n"
        f"Researcher's plan: {analysis_plan or 'Not specified'}\n\n"
        f"Inspection findings:\n{last_findings}\n\n"
        "Write a numbered preprocessing plan (5-10 steps). For each step, include:\n"
        "- The action (e.g., 'Mark bad channels')\n"
        "- Specific parameters (e.g., which channels, which frequency)\n"
        "Be concise and specific to this dataset."
    )

    if verbose:
        print(f"\n  {G}Generating preprocessing plan...{R}")
    else:
        status("generating plan", "sending to LLM")

    proposed_plan = call_text(plan_prompt, mock=args.mock)

    if verbose:
        print(f"\n  {G}{'=' * 56}")
        print(f"  Proposed Preprocessing Plan")
        print(f"  {'=' * 56}{R}\n")
        for line in proposed_plan.split("\n"):
            print(f"  {line}")
        print(f"\n  {G}{'=' * 56}{R}")
    else:
        sys.stdout.write("\n")

    # Write plan to report
    plan_html = f'<div class="findings">{proposed_plan}</div>'
    report.set("PREPROCESSING_PLAN", plan_html)

    # User feedback on plan
    if interactive:
        print(f"\n  Your feedback on the plan (or press Enter to approve):")
        plan_feedback = ask("  > ")
        if plan_feedback:
            interpreted = call_text(
                f"The agent proposed this preprocessing plan:\n{proposed_plan}\n\n"
                f"The researcher responded: \"{plan_feedback}\"\n\n"
                "Summarize what the researcher wants changed, in one sentence.",
                mock=args.mock,
            )
            print(f"  Understood: {interpreted}")
        else:
            print("  Plan approved.")

    # Run artifact rejection
    from nap.skills.eeg_artifact_rejection import EEGArtifactRejection
    EEGArtifactRejection().run(memory)

    # Save report
    report_path = report.save(str(subject_dir))

    # Session summary
    if verbose:
        print(f"\n  {G}{'=' * 56}")
        print(f"  Done.")
        print(f"  Report: {report_path}")
        print(f"  Figures: {figures_dir}")
        print(f"  {'=' * 56}{R}\n")
    else:
        print(f"\n  {G}Done.{R} Report: {report_path}")


if __name__ == "__main__":
    main()
