#!/usr/bin/env python3
"""
Script to compare test coverage between base and current branch.

Reads coverage.json files and generates a comparison report showing
coverage changes at the file and overall level.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_coverage_data(filepath: Path) -> Dict:
    """Load coverage data from a coverage.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_coverage_percent(summary: Dict) -> float:
    """Calculate coverage percentage from summary data."""
    if summary['num_statements'] == 0:
        return 100.0
    return (summary['covered_lines'] / summary['num_statements']) * 100


def compare_coverage(base_file: Path, current_file: Path) -> Tuple[str, int]:
    """
    Compare coverage between base and current branch.

    Returns:
        Tuple of (report string, exit code)
        Exit code: 0 if coverage improved/same, 1 if coverage decreased
    """
    try:
        base_data = load_coverage_data(base_file)
        current_data = load_coverage_data(current_file)
    except FileNotFoundError as e:
        return f"Error: Coverage file not found: {e}", 1
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in coverage file: {e}", 1

    base_total = base_data['totals']
    current_total = current_data['totals']

    base_coverage = calculate_coverage_percent(base_total)
    current_coverage = calculate_coverage_percent(current_total)
    coverage_diff = current_coverage - base_coverage

    # Build report
    report_lines = ["# Coverage Comparison Report\n"]
    report_lines.append("## Overall Coverage\n")
    report_lines.append(f"- **Base Branch**: {base_coverage:.2f}%")
    report_lines.append(f"- **Current Branch**: {current_coverage:.2f}%")

    if coverage_diff > 0:
        report_lines.append(f"- **Change**: +{coverage_diff:.2f}% :green_circle:")
        status_icon = ":white_check_mark:"
    elif coverage_diff < 0:
        report_lines.append(f"- **Change**: {coverage_diff:.2f}% :red_circle:")
        status_icon = ":x:"
    else:
        report_lines.append("- **Change**: No change")
        status_icon = ":white_check_mark:"

    report_lines.append(f"\n**Status**: {status_icon}\n")

    # File-level comparison
    report_lines.append("## File-Level Changes\n")
    report_lines.append("| File | Base Coverage | Current Coverage | Change |")
    report_lines.append("|------|---------------|------------------|--------|")

    base_files = base_data.get('files', {})
    current_files = current_data.get('files', {})

    all_files = set(base_files.keys()) | set(current_files.keys())

    for filepath in sorted(all_files):
        base_file_data = base_files.get(filepath, {})
        current_file_data = current_files.get(filepath, {})

        if base_file_data:
            base_summary = base_file_data['summary']
            base_file_cov = calculate_coverage_percent(base_summary)
        else:
            base_file_cov = 0.0

        if current_file_data:
            current_summary = current_file_data['summary']
            current_file_cov = calculate_coverage_percent(current_summary)
        else:
            current_file_cov = 0.0

        file_diff = current_file_cov - base_file_cov

        # Simplify filepath for display
        display_path = filepath.split('sparsamp_app/')[-1] if 'sparsamp_app/' in filepath else filepath

        if file_diff > 0:
            change_str = f"+{file_diff:.2f}% :green_circle:"
        elif file_diff < 0:
            change_str = f"{file_diff:.2f}% :red_circle:"
        else:
            change_str = "—"

        report_lines.append(
            f"| {display_path} | {base_file_cov:.2f}% | {current_file_cov:.2f}% | {change_str} |"
        )

    report_lines.append("\n---\n")
    report_lines.append(f"**Total Statements**: {current_total['num_statements']} "
                       f"(Base: {base_total['num_statements']})\n")
    report_lines.append(f"**Covered Lines**: {current_total['covered_lines']} "
                       f"(Base: {base_total['covered_lines']})\n")
    report_lines.append(f"**Missing Lines**: {current_total['missing_lines']} "
                       f"(Base: {base_total['missing_lines']})\n")

    report = "\n".join(report_lines)

    # Return exit code 1 if coverage decreased
    exit_code = 1 if coverage_diff < 0 else 0

    return report, exit_code


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: compare_coverage.py <base_coverage.json> <current_coverage.json>")
        sys.exit(1)

    base_file = Path(sys.argv[1])
    current_file = Path(sys.argv[2])

    report, exit_code = compare_coverage(base_file, current_file)
    print(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
