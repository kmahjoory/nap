"""Skill: EEG Artifact Rejection — iterative, autonomous, report at the end."""

import json
from pathlib import Path

from nap.skills.base import Skill, status
from nap.memory import Memory, SkillRecord
from nap.llm import call_with_images
from nap.preprocessing.eeg import (
    plot_psd, plot_traces,
    plot_ica_topomaps, plot_ica_spectra, plot_ica_timecourses,
    remove_channels, annotate_segments, run_ica,
    remove_ica_components, apply_notch,
)

DOMAIN_KNOWLEDGE = """
Artifact rejection priorities (conservative -- avoid over-cleaning):

1. REMOVE_CHANNELS: Only flat (disconnected) or zero-correlation channels.
   Do NOT remove noisy channels that ICA can fix.
2. REMOVE_SEGMENTS: Only large movement artifacts affecting most channels.
   Do NOT remove short single-channel artifacts -- ICA handles those.
3. RUN_ICA: Run with 20 components. This prepares for component rejection.
4. REMOVE_ICA_COMPONENTS -- classify each component using ALL three views:

   EYE BLINK (REMOVE): Topomap shows frontal bilateral pattern (Fp1/Fp2).
   Power spectrum has 1/f shape with most power below 5 Hz.
   Time course shows sharp transients every few seconds.
   Usually 1-2 components.

   HORIZONTAL EYE (REMOVE): Topomap shows lateral frontal (F7/F8).
   Time course shows step-like shifts. Usually 1 component.

   MUSCLE (REMOVE only if clearly muscle): Topomap at temporal edges.
   Power spectrum is FLAT or RISING continuously above 15 Hz with NO
   distinct peak -- just broadband high-frequency noise.
   Time course shows irregular high-frequency bursts.

   BETA BRAIN ACTIVITY (KEEP -- this is brain signal, NOT artifact):
   Power spectrum shows a clear PEAK at 15-30 Hz, then power drops
   back down. This is different from muscle which rises continuously.
   Topomap is focal over central/frontal cortex (C3/C4/Cz/Fz area).
   Key difference from muscle: beta has a PEAK then drops; muscle
   has NO peak, just a flat or rising slope into high frequencies.

   ALPHA BRAIN ACTIVITY (ALWAYS KEEP): Power spectrum shows peak at
   8-13 Hz. Topomap focal over posterior cortex (O1/O2/Pz).

   HEARTBEAT (REMOVE only if clearly present): Broad topomap, often
   left-lateralized. Time course shows regular QRS-like peaks ~1/sec.

   If in doubt between muscle and beta: KEEP IT. Removing beta
   activity destroys real brain signal.

5. APPLY_NOTCH: when a clear line noise peak (50 or 60 Hz) is visible.

STOP WHEN data is clean enough. Some noise is normal. Over-cleaning
removes real brain signal. If in doubt, keep it. Fewer removals is better.
"""

MAX_ITERATIONS = 8


class EEGArtifactRejection(Skill):
    name = "EEG Artifact Rejection"
    status_label = "rejecting artifacts"
    needs_confirm = True

    def act(self, _memory: Memory) -> dict[str, str]:
        return {}  # not used — this skill overrides run()

    def build_prompt(self, _memory: Memory) -> str:
        return ""  # not used — this skill builds prompts per iteration

    def run(self, memory: Memory) -> SkillRecord:
        import sys
        G, R = "\033[38;2;100;148;113m", "\033[0m"
        if memory.verbose:
            print(f"\n  {G}Running skill: {self.name} (autonomous)...{R}")

        # Create working copy
        status(memory, self.status_label, "loading data")
        memory.working_raw = memory.raw.copy().load_data(verbose=False)

        fig_dir = Path(memory.figures_dir) / "artifact_rejection"
        fig_dir.mkdir(parents=True, exist_ok=True)

        actions_taken = []

        for iteration in range(MAX_ITERATIONS):
            if memory.verbose:
                print(f"\n  --- Iteration {iteration + 1} ---")

            # Generate plots of current state
            status(memory, self.status_label, f"plotting (iteration {iteration + 1})")
            plots = {}
            plots["psd"] = plot_psd(
                memory.working_raw, str(fig_dir / f"iter{iteration}_psd.png"),
                f"PSD (iteration {iteration})",
            )
            plots["traces"] = plot_traces(
                memory.working_raw, str(fig_dir / f"iter{iteration}_traces.png"),
                f"Traces (iteration {iteration})",
            )

            # Add 3 ICA plots if ICA has been fitted
            if memory.ica is not None:
                plots["ica_topomaps"] = plot_ica_topomaps(
                    memory.ica, memory.working_raw,
                    str(fig_dir / f"iter{iteration}_ica_topomaps.png"),
                )
                plots["ica_spectra"] = plot_ica_spectra(
                    memory.ica, memory.working_raw,
                    str(fig_dir / f"iter{iteration}_ica_spectra.png"),
                )
                plots["ica_timecourses"] = plot_ica_timecourses(
                    memory.ica, memory.working_raw,
                    str(fig_dir / f"iter{iteration}_ica_timecourses.png"),
                )

            # Build prompt with history
            history = "\n".join(
                f"  {i+1}. {a['action']}: {a['params']} — {a['reason']}"
                for i, a in enumerate(actions_taken)
            ) or "  None yet."

            # Describe which plots are attached
            plot_desc = "Attached plots:\n1. PSD of all channels\n2. Time-series traces"
            if memory.ica is not None:
                plot_desc += (
                    "\n3. ICA topomaps (spatial pattern of each component)"
                    "\n4. ICA power spectra (frequency content of each component — "
                    "look for peaks vs flat/rising slopes)"
                    "\n5. ICA time courses (full recording, all components — "
                    "look for transient patterns vs continuous activity)"
                )

            prompt = (
                "You are an EEG artifact rejection expert.\n\n"
                f"{DOMAIN_KNOWLEDGE}\n\n"
                f"Recording metadata:\n{memory.get_context()}\n\n"
                f"Actions taken so far:\n{history}\n\n"
                f"{plot_desc}\n\n"
                "Look at the attached plots. Propose exactly ONE next action.\n"
                "Respond with a JSON object on the FIRST line, then your reasoning.\n\n"
                "Format: {\"action\": \"ACTION_TYPE\", \"params\": \"...\"}\n\n"
                "ACTION_TYPE is one of:\n"
                "- REMOVE_CHANNELS: params = comma-separated channel names\n"
                "- REMOVE_SEGMENTS: params = start-end in seconds (e.g. \"120.0-125.0\")\n"
                "- RUN_ICA: params = number of components (e.g. \"20\")\n"
                "- REMOVE_ICA_COMPONENTS: params = comma-separated component indices\n"
                "- APPLY_NOTCH: params = frequency in Hz (e.g. \"50\")\n"
                "- DONE: params = \"\" (data is clean enough)\n"
            )

            # See
            status(memory, self.status_label, f"sending to LLM (iteration {iteration + 1})")
            image_paths = list(plots.values())
            response = call_with_images(prompt, image_paths, mock=memory.mock)
            if memory.verbose:
                print(f"  LLM: {response[:200]}...")

            # Parse action
            action = self._parse_action(response, memory.mock)
            if action["action"] == "DONE":
                if memory.verbose:
                    print(f"  LLM says: DONE — data is clean enough.")
                break

            # Execute
            reason = response.split("\n", 1)[-1].strip()[:200]
            action["reason"] = reason
            status(memory, self.status_label, f"{action['action'].lower().replace('_', ' ')}")
            self._execute(action, memory)
            actions_taken.append(action)
            if memory.verbose:
                print(f"  Executed: {action['action']} — {action['params']}")

            # Write iteration to HTML report
            if memory.report:
                iter_html = f'<div class="iteration">'
                iter_html += f'<div class="iteration-header">Iteration {iteration + 1}</div>'
                iter_html += f'<p><span class="action-badge">{action["action"]}</span> {action["params"]}</p>'
                iter_html += f'<p>{reason}</p>'
                for plot_name, plot_path in plots.items():
                    memory.report.append_image(
                        "ARTIFACT_REJECTION_ITERATIONS", plot_path, plot_name,
                    )
                iter_html += '</div>'
                memory.report.append("ARTIFACT_REJECTION_ITERATIONS", iter_html)

        # Save report
        status(memory, self.status_label, "saving report")
        report = self._save_report(actions_taken, memory, fig_dir)
        if not memory.verbose:
            sys.stdout.write(f"\r  {G}NAP is {self.status_label}...{R} done\n")
            sys.stdout.flush()

        # Show to user for review
        user_decision = self.confirm(report, memory)

        record = SkillRecord(
            skill_name=self.name,
            findings=report,
            plots={f"artifact_rejection_dir": str(fig_dir)},
            user_decision=user_decision,
        )
        memory.add_record(record)
        return record

    def _parse_action(self, response: str, mock: bool) -> dict:
        """Extract action JSON from LLM response."""
        if mock:
            return {"action": "DONE", "params": ""}

        first_line = response.strip().split("\n")[0]
        # Try to find JSON in the first line
        try:
            # Find JSON object in the line
            start = first_line.index("{")
            end = first_line.rindex("}") + 1
            return json.loads(first_line[start:end])
        except (ValueError, json.JSONDecodeError):
            # If parsing fails, assume DONE
            return {"action": "DONE", "params": ""}

    def _execute(self, action: dict, memory: Memory):
        """Execute one artifact rejection action on working_raw."""
        raw = memory.working_raw
        act = action["action"]
        params = action["params"]

        if act == "REMOVE_CHANNELS":
            channels = [ch.strip() for ch in params.split(",")]
            removed = remove_channels(raw, channels)
            memory.bad_channels.extend(removed)

        elif act == "REMOVE_SEGMENTS":
            parts = params.split("-")
            start, end = float(parts[0]), float(parts[1])
            annotate_segments(raw, [(start, end)])
            memory.bad_segments.append((start, end))

        elif act == "RUN_ICA":
            n = int(params) if params else 20
            # Filter copy for ICA fitting (1 Hz highpass recommended)
            raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
            memory.ica = run_ica(raw_for_ica, n_components=n)
            del raw_for_ica

        elif act == "REMOVE_ICA_COMPONENTS":
            if memory.ica is not None:
                indices = [int(i.strip()) for i in params.split(",")]
                remove_ica_components(memory.ica, raw, indices)

        elif act == "APPLY_NOTCH":
            freq = float(params)
            apply_notch(raw, freq)
            memory.filters_applied.append(f"notch {freq} Hz")

    def _save_report(self, actions: list[dict], memory: Memory, fig_dir: Path) -> str:
        """Generate and save the artifact rejection report."""
        lines = [
            "NAP Artifact Rejection Report",
            f"Data: {Path(memory.data_path).name}",
            "-" * 40,
        ]

        if not actions:
            lines.append("No actions taken — data was clean enough.")
        else:
            for i, a in enumerate(actions, 1):
                lines.append(f"Step {i}: {a['action']} — {a['params']}")
                lines.append(f"  Reason: {a.get('reason', 'N/A')}")

        report_text = "\n".join(lines)

        if memory.verbose:
            print(f"\n  Artifact rejection complete ({len(actions)} actions taken)")

        # Write summary to HTML report
        if memory.report:
            summary_html = '<div class="findings">'
            if not actions:
                summary_html += "<p>No actions taken — data was clean enough.</p>"
            else:
                summary_html += f"<p><strong>{len(actions)} actions taken:</strong></p><ol>"
                for a in actions:
                    summary_html += (
                        f'<li><span class="action-badge">{a["action"]}</span> '
                        f'{a["params"]}<br><em>{a.get("reason", "N/A")}</em></li>'
                    )
                summary_html += "</ol>"
            summary_html += "</div>"
            memory.report.set("ARTIFACT_REJECTION_SUMMARY", summary_html)

        return report_text
