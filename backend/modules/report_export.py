from __future__ import annotations

import json
from datetime import datetime, timezone


def build_report(dataset: dict, analysis: dict | None = None) -> dict:
    report = {
        "app_name": "NeuroBridge",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "analysis": analysis or {},
        "disclaimer": "Educational and research prototype. Not intended for clinical diagnosis or treatment.",
    }
    return report


def report_to_markdown(report: dict) -> str:
    lines = [
        "# NeuroBridge Analysis Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        f"Dataset name: {report['dataset'].get('name', 'Unknown')}",
        f"Dataset type: {report['dataset'].get('dataset_type', 'unknown')}",
        f"Status: {report['dataset'].get('validation_status', 'unknown')}",
        "",
        "## Summary",
        json.dumps(report, indent=2),
    ]
    return "\n".join(lines)
