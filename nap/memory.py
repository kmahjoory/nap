"""Memory — shared state between skills + decision log for notebook export."""

from dataclasses import dataclass, field


@dataclass
class SkillRecord:
    """One entry in the decision log — the result of running a skill."""
    skill_name: str
    findings: str
    plots: dict[str, str] = field(default_factory=dict)
    user_decision: str | None = None


class Memory:
    """Shared state that skills read from and write to.

    Data State: current data objects and decisions made so far.
    Decision Log: ordered record of skill executions (becomes the notebook).
    """

    def __init__(self, raw, data_path: str, data_summary: dict, figures_dir: str,
                 mock: bool = False, analysis_plan: str = "",
                 verbose: bool = False, interactive: bool = False,
                 report=None):
        # Data state
        self.raw = raw
        self.data_path = data_path
        self.data_summary = data_summary
        self.figures_dir = figures_dir
        self.mock = mock
        self.analysis_plan = analysis_plan
        self.verbose = verbose
        self.interactive = interactive
        self.report = report
        self.working_raw = None  # mutable copy for preprocessing
        self.bad_channels: list[str] = []
        self.bad_segments: list[tuple[float, float]] = []
        self.filters_applied: list[str] = []
        self.ica = None  # ICA object after fitting

        # Decision log
        self.log: list[SkillRecord] = []

    def add_record(self, record: SkillRecord):
        """Append a skill result to the decision log."""
        self.log.append(record)

    def get_context(self) -> str:
        """Build a text summary of everything done so far, for the next skill's LLM prompt."""
        lines = []
        if self.analysis_plan:
            lines.append(f"Researcher's analysis plan: {self.analysis_plan}\n")
        for k, v in self.data_summary.items():
            lines.append(f"- {k}: {v}")
        if self.bad_channels:
            lines.append(f"- Bad channels identified: {', '.join(self.bad_channels)}")
        if self.filters_applied:
            lines.append(f"- Filters applied: {', '.join(self.filters_applied)}")
        if self.bad_segments:
            segs = [f"{s:.1f}-{e:.1f} s" for s, e in self.bad_segments]
            lines.append(f"- Bad segments: {', '.join(segs)}")
        return "\n".join(lines)

    def print_log(self):
        """Print the decision log to stdout."""
        for i, record in enumerate(self.log, 1):
            print(f"\n  [{i}] {record.skill_name}")
            for line in record.findings.split("\n"):
                print(f"      {line}")
            if record.user_decision:
                print(f"      User: {record.user_decision}")
