"""JSON + Markdown report writing for Phase 1 integration tests."""
import json
import os
import shutil
from dataclasses import asdict, dataclass, field


@dataclass
class RunReport:
    run_name: str = ""
    config: dict = field(default_factory=dict)
    dataset_manifest: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    seed: int = 42
    initial_lora_checksum: str = ""
    final_lora_checksum: str = ""
    saved_lora_path: str = ""
    hook_counts: dict = field(default_factory=dict)
    per_step_losses: list = field(default_factory=list)
    per_step_metrics: list = field(default_factory=dict)
    parameter_deltas: dict = field(default_factory=dict)
    peak_allocated_vram_gb: float | None = None
    peak_reserved_vram_gb: float | None = None
    runtime_seconds: float | None = None
    assertions: dict = field(default_factory=dict)
    failure_detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def write_json_report(report: RunReport, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    return path


def write_markdown_summary(report: RunReport, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"# Run Report: {report.run_name}",
        "",
        f"- **Runtime:** {report.runtime_seconds:.1f}s" if report.runtime_seconds else "- **Runtime:** N/A",
        f"- **Peak VRAM (allocated):** {report.peak_allocated_vram_gb:.2f} GB"
        if report.peak_allocated_vram_gb
        else "- **Peak VRAM:** N/A",
        f"- **Initial LoRA checksum:** `{report.initial_lora_checksum}`",
        f"- **Final LoRA checksum:** `{report.final_lora_checksum}`",
        f"- **Saved LoRA:** `{report.saved_lora_path}`",
        "",
        "## Hook Counts",
        "",
        "| Metric | Count |",
        "|---|---|",
    ]
    for k, v in report.hook_counts.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per-Step Losses", "", "| Step | Loss |", "|---|---|"]
    for i, loss in enumerate(report.per_step_losses):
        lines.append(f"| {i} | {loss} |")
    if report.failure_detail:
        lines += ["", f"## Failure Detail\n\n{report.failure_detail}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def cleanup_artifacts(save_dir: str, keep: bool = False) -> None:
    """Remove large artifacts unless AI_TOOLKIT_KEEP_TEST_OUTPUTS=1."""
    if keep:
        return
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
