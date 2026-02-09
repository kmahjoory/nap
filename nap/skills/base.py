"""Base skill — defines the act/see/evaluate/confirm cycle."""

import sys
from abc import ABC, abstractmethod

from nap.llm import call_with_images, call_text
from nap.memory import Memory, SkillRecord

# NAP green
G = "\033[38;2;100;148;113m"
R = "\033[0m"


def status(memory: Memory, skill_label: str, step: str):
    """Print a status update. Single updating line if not verbose, full line if verbose."""
    if memory.verbose:
        print(f"    {step}")
    else:
        sys.stdout.write(f"\r  {G}NAP is {skill_label}...{R} {step}    ")
        sys.stdout.flush()


class Skill(ABC):
    """Base class for all NAP skills."""

    name: str = "Unnamed Skill"
    status_label: str = "working"
    needs_confirm: bool = False

    @abstractmethod
    def act(self, memory: Memory) -> dict[str, str]:
        """Execute the skill's action. Returns dict of plot name -> file path."""

    @abstractmethod
    def build_prompt(self, memory: Memory) -> str:
        """Build the LLM prompt with domain knowledge + context."""

    def see(self, plots: dict[str, str], memory: Memory) -> str:
        """Send plots to LLM with context. Returns findings string."""
        prompt = self.build_prompt(memory)
        image_paths = list(plots.values())
        return call_with_images(prompt, image_paths, mock=memory.mock)

    def confirm(self, findings: str, memory: Memory) -> str | None:
        """Display findings and get researcher feedback."""
        if memory.verbose:
            print(f"\n  {G}{'=' * 56}")
            print(f"  {self.name} -- Findings")
            print(f"  {'=' * 56}{R}\n")
            for line in findings.split("\n"):
                print(f"  {line}")
            print(f"\n  {G}{'=' * 56}{R}")

        if not self.needs_confirm or not memory.interactive:
            return None

        from nap.cli import ask
        print(f"\n  Your feedback (or press Enter to approve as-is):")
        user_input = ask("  > ")

        if not user_input:
            print("  Approved.")
            return "Approved as-is"

        interpret_prompt = (
            f"The agent ran the skill '{self.name}' and found:\n"
            f"{findings}\n\n"
            f"The researcher responded: \"{user_input}\"\n\n"
            "Summarize the researcher's decision in one concise sentence. "
            "What did they approve, reject, or modify?"
        )
        interpreted = call_text(interpret_prompt, mock=memory.mock)
        print(f"  Understood: {interpreted}")
        return user_input

    def run(self, memory: Memory) -> SkillRecord:
        """Execute the full act/see/evaluate/confirm cycle."""
        if memory.verbose:
            print(f"\n  Running skill: {self.name}...")

        # Act
        plots = self.act(memory)
        if memory.verbose:
            for name, path in plots.items():
                print(f"    Saved: {path}")

        # See + Evaluate
        status(memory, self.status_label, "sending to LLM")
        findings = self.see(plots, memory)

        # Confirm
        status(memory, self.status_label, "done")
        if not memory.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()
        user_decision = self.confirm(findings, memory)

        # Write to memory
        record = SkillRecord(
            skill_name=self.name,
            findings=findings,
            plots=plots,
            user_decision=user_decision,
        )
        memory.add_record(record)
        return record
