import argparse
import glob
import json
import os
from typing import Any, Dict, List, Sequence


ACCEPT_COLOR = "#2ECC71"
REJECT_COLOR = "#FF5A5F"


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise SystemExit(
            "This script requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    return plt, ListedColormap


def find_input_paths(prefix: str, search_dir: str) -> List[str]:
    patterns = [
        os.path.join(search_dir, f"{prefix}"),
        os.path.join(search_dir, f"{prefix}.*"),
        os.path.join(search_dir, f"{prefix}*"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(
            path
            for path in glob.glob(pattern)
            if is_candidate_input(path)
        )

    matches = sorted(dict.fromkeys(matches))
    if not matches:
        raise SystemExit(
            f"No input file or result directory found in {search_dir!r} "
            f"for prefix {prefix!r}."
        )
    return matches


def is_candidate_input(path: str) -> bool:
    name = os.path.basename(path)
    if name.startswith("."):
        return False
    if os.path.isdir(path):
        return True
    if not os.path.isfile(path):
        return False
    return os.path.splitext(name)[1].lower() in {".json", ".jsonl"}


def short_input_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def output_path_for(input_path: str, output_dir: str) -> str:
    basename = short_input_name(input_path)
    stem = basename if os.path.isdir(input_path) else os.path.splitext(basename)[0]
    return os.path.join(output_dir, f"{stem}_accept_heatmap.png")


def progress_output_path_for(heatmap_output_path: str) -> str:
    directory = os.path.dirname(heatmap_output_path)
    basename = os.path.basename(heatmap_output_path)
    stem, ext = os.path.splitext(basename)
    if stem.endswith("_accept_heatmap"):
        stem = stem[: -len("_accept_heatmap")]
    return os.path.join(directory, f"{stem}_accept_rate_by_progress{ext or '.png'}")


def parse_json_line(line: str, line_no: int) -> Dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc

    if not isinstance(row, dict):
        raise ValueError(
            f"line {line_no}: expected a JSON object like "
            '{"problem_id": 12, "accepts": [1, 0, 1]}.'
        )
    return row


def normalize_accept(value: Any, source: str, index: int) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in {0, 1}:
        return value
    raise ValueError(
        f"{source}: accepts[{index}] must be 0 or 1, got {value!r}."
    )


def validate_row(row: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    if "problem_id" not in row:
        raise ValueError(f"line {line_no}: missing required key 'problem_id'.")
    if "accepts" not in row:
        raise ValueError(f"line {line_no}: missing required key 'accepts'.")

    accepts = row["accepts"]
    if not isinstance(accepts, list):
        raise ValueError(f"line {line_no}: 'accepts' must be a list.")
    if not accepts:
        raise ValueError(f"line {line_no}: 'accepts' must not be empty.")

    return {
        "problem_id": row["problem_id"],
        "accepts": [
            normalize_accept(value, line_no, index)
            for index, value in enumerate(accepts)
        ],
    }


def result_json_files(directory: str) -> List[str]:
    paths = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(".json"):
            continue
        if name.lower() == "accuracy.json":
            continue
        paths.append(path)
    return sorted(paths)


def validate_result_json(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a JSON object.")
    if "accepts" not in data:
        raise ValueError(f"{source}: missing required key 'accepts'.")

    problem_id = data.get("problem_id", data.get("question_id"))
    if problem_id is None:
        raise ValueError(
            f"{source}: missing 'problem_id' or 'question_id'."
        )

    accepts = data["accepts"]
    if not isinstance(accepts, list):
        raise ValueError(f"{source}: 'accepts' must be a list.")
    if not accepts:
        raise ValueError(f"{source}: 'accepts' must not be empty.")

    return {
        "problem_id": problem_id,
        "accepts": [
            normalize_accept(value, 0, index)
            for index, value in enumerate(accepts)
        ],
    }


def preview_jsonl_file(path: str, preview_lines: int) -> None:
    print(f"Input file: {path}")
    print(f"Previewing first {preview_lines} non-empty line(s):")
    shown = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = parse_json_line(line, line_no)
            validated = validate_row(row, line_no)
            print(f"  line {line_no} raw: {line[:300]}")
            print(
                "  parsed: "
                f"problem_id={validated['problem_id']!r}, "
                f"steps={len(validated['accepts'])}, "
                f"accepts_head={validated['accepts'][:20]}"
            )
            shown += 1
            if shown >= preview_lines:
                break

    if shown == 0:
        raise SystemExit("No non-empty JSONL rows found.")


def preview_result_directory(path: str, preview_lines: int) -> None:
    files = result_json_files(path)
    if not files:
        raise SystemExit(f"No per-problem JSON files found in directory: {path}")

    print(f"Input directory: {path}")
    print(f"Previewing first {min(preview_lines, len(files))} JSON file(s):")
    for result_path in files[:preview_lines]:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        row = validate_result_json(data, result_path)
        print(
            f"  file {os.path.basename(result_path)}: "
            f"problem_id={row['problem_id']!r}, "
            f"steps={len(row['accepts'])}, "
            f"accepts_head={row['accepts'][:20]}"
        )


def preview_input(path: str, preview_lines: int) -> None:
    if os.path.isdir(path):
        preview_result_directory(path, preview_lines)
    else:
        preview_jsonl_file(path, preview_lines)


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rows.append(validate_row(parse_json_line(line, line_no), line_no))

    if not rows:
        raise SystemExit("No valid rows found.")
    return rows


def load_result_directory_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    files = result_json_files(path)
    if not files:
        raise SystemExit(f"No per-problem JSON files found in directory: {path}")

    for result_path in files:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(validate_result_json(data, result_path))

    return rows


def load_rows(path: str) -> List[Dict[str, Any]]:
    if os.path.isdir(path):
        return load_result_directory_rows(path)
    return load_jsonl_rows(path)


def sorted_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: row["problem_id"])


def accept_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_accepts = sum(sum(row["accepts"]) for row in rows)
    total_decisions = sum(len(row["accepts"]) for row in rows)
    accept_rate = total_accepts / total_decisions if total_decisions else 0.0
    return {
        "rows": len(rows),
        "total_accepts": total_accepts,
        "total_decisions": total_decisions,
        "accept_rate": accept_rate,
    }


def plot_heatmap(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    plt, ListedColormap = require_matplotlib()

    rows = sorted_rows(rows)
    row_count = len(rows)
    max_steps = max(len(row["accepts"]) for row in rows)

    fig_width = max(10.0, min(28.0, max_steps * 0.18))
    fig_height = max(4.0, min(40.0, row_count * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = ListedColormap([REJECT_COLOR, ACCEPT_COLOR])

    for y, row in enumerate(rows):
        accepts = row["accepts"]
        step_count = len(accepts)
        cell_width = max_steps / step_count
        for step_index, accepted in enumerate(accepts):
            ax.imshow(
                [[accepted]],
                cmap=cmap,
                vmin=0,
                vmax=1,
                extent=(
                    step_index * cell_width,
                    (step_index + 1) * cell_width,
                    y,
                    y + 1,
                ),
                aspect="auto",
                interpolation="nearest",
            )

    ax.set_xlim(0, max_steps)
    ax.set_ylim(row_count, 0)
    ax.set_xlabel("Normalized reasoning progress")
    ax.set_ylabel("Problem ID")
    ax.set_title("Reasoning Step Accept / Reject Heatmap")

    if row_count <= 80:
        ax.set_yticks([i + 0.5 for i in range(row_count)])
        ax.set_yticklabels(
            [
                f"{row['problem_id']} ({len(row['accepts'])})"
                for row in rows
            ],
            fontsize=8,
        )
    else:
        ax.set_yticks([])

    tick_count = 6
    xticks = [max_steps * i / (tick_count - 1) for i in range(tick_count)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{i / (tick_count - 1):.0%}" for i in range(tick_count)])

    red_patch = plt.Rectangle((0, 0), 1, 1, color=REJECT_COLOR, label="Rejected")
    green_patch = plt.Rectangle((0, 0), 1, 1, color=ACCEPT_COLOR, label="Accepted")
    ax.legend(handles=[green_patch, red_patch], loc="upper right", frameon=True)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def accept_rate_by_progress(
    rows: Sequence[Dict[str, Any]], bins: int = 10
) -> List[Dict[str, float]]:
    accepted_weight = [0.0] * bins
    total_weight = [0.0] * bins

    for row in rows:
        accepts = row["accepts"]
        step_count = len(accepts)
        for step_index, accepted in enumerate(accepts):
            step_start = step_index / step_count
            step_end = (step_index + 1) / step_count
            for bin_index in range(bins):
                bin_start = bin_index / bins
                bin_end = (bin_index + 1) / bins
                overlap = max(0.0, min(step_end, bin_end) - max(step_start, bin_start))
                if overlap <= 0:
                    continue
                accepted_weight[bin_index] += accepted * overlap
                total_weight[bin_index] += overlap

    stats = []
    for bin_index in range(bins):
        rate = (
            accepted_weight[bin_index] / total_weight[bin_index]
            if total_weight[bin_index]
            else 0.0
        )
        stats.append(
            {
                "bin_start": bin_index / bins,
                "bin_end": (bin_index + 1) / bins,
                "rate": rate,
                "accepted_weight": accepted_weight[bin_index],
                "total_weight": total_weight[bin_index],
            }
        )
    return stats


def plot_progress_accept_rate(
    rows: Sequence[Dict[str, Any]], output_path: str, bins: int = 10
) -> None:
    plt, _ = require_matplotlib()
    stats = accept_rate_by_progress(rows, bins=bins)
    x_values = [(item["bin_start"] + item["bin_end"]) * 50 for item in stats]
    y_values = [item["rate"] * 100 for item in stats]
    labels = [
        f"{int(item['bin_start'] * 100)}-{int(item['bin_end'] * 100)}%"
        for item in stats
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(
        x_values,
        y_values,
        color=ACCEPT_COLOR,
        marker="o",
        linewidth=2.5,
        markersize=6,
    )
    ax.fill_between(x_values, y_values, color=ACCEPT_COLOR, alpha=0.12)

    for x_value, y_value in zip(x_values, y_values):
        ax.text(
            x_value,
            min(100, y_value + 2),
            f"{y_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("Normalized reasoning progress")
    ax.set_ylabel("Accept Rate (%)")
    ax.set_title("Accept Rate by 10% Reasoning Progress Interval")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def combined_output_path(prefix: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"{prefix}_overall_accept_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an accept/reject heatmap from a JSONL file or a result "
            "directory containing per-problem JSON files. JSONL lines must be "
            '{"problem_id": 12, "accepts": [1, 1, 0]}. Result JSON files may '
            'use either "problem_id" or "question_id".'
        )
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        help="Prefix to match input files/directories, like report_prefix.py.",
    )
    parser.add_argument("--input", help="Exact JSONL file or result directory path.")
    parser.add_argument(
        "--prefix",
        dest="prefix_option",
        help="Find one data file or result directory in the repo root by prefix.",
    )
    parser.add_argument(
        "--search-dir",
        "--search_dir",
        default=".",
        dest="search_dir",
        help="Directory used with --prefix. Default: current working directory.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output PNG path. Only valid with one input. "
            "Default: <input_stem>_accept_heatmap.png"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=".",
        dest="output_dir",
        help="Directory for prefix mode plots. Default: current working directory.",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=5,
        help="Number of non-empty lines to preview before plotting.",
    )
    args = parser.parse_args()
    prefix = args.prefix_option or args.prefix

    if args.input and prefix:
        parser.error("use either --input or a prefix, not both")
    if not args.input and not prefix:
        parser.error("provide --input or a prefix")

    if args.input:
        input_paths = [
            args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
        ]
    else:
        input_paths = find_input_paths(prefix, os.path.abspath(args.search_dir))

    if args.output and len(input_paths) > 1:
        raise SystemExit("--output can only be used when exactly one input is matched.")

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Output directory does not exist: {output_dir}")

    print(f"Searching in: {os.path.abspath(args.search_dir)}")
    if prefix:
        print(f"Found inputs with prefix '{prefix}':")
        for path in input_paths:
            print(f"  {path}")
        print()

    print("Individual Input Results:")
    print("-" * 50)
    all_rows = []
    outputs = []
    for input_path in input_paths:
        try:
            preview_input(input_path, args.preview_lines)
            rows = load_rows(input_path)
        except ValueError as exc:
            raise SystemExit(
                f"Input format mismatch: {exc}\n"
                "Expected either JSONL with one object per line, for example "
                '{"problem_id": 12, "accepts": [1, 1, 0, 1]}, or a result '
                "directory with per-problem JSON files containing 'accepts'. "
                "Please check the file format before plotting."
            ) from exc

        output_path = args.output or output_path_for(input_path, output_dir)
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        progress_output_path = progress_output_path_for(output_path)

        plot_heatmap(rows, output_path)
        plot_progress_accept_rate(rows, progress_output_path)
        outputs.append(output_path)
        outputs.append(progress_output_path)
        all_rows.extend(rows)
        summary = accept_summary(rows)
        print(f"Input: {input_path}")
        print(f"  Rows plotted: {summary['rows']}")
        print(
            "  Accept rate: "
            f"{summary['total_accepts']}/{summary['total_decisions']} "
            f"({summary['accept_rate']:.4f})"
        )
        print(f"Heatmap output: {output_path}")
        print(f"Progress line output: {progress_output_path}")
        print()

    if prefix and all_rows:
        summary = accept_summary(all_rows)
        print("Overall Results:")
        print("-" * 20)
        print(f"Total rows: {summary['rows']}")
        print(
            "Overall accept rate: "
            f"{summary['total_accepts']}/{summary['total_decisions']} "
            f"({summary['accept_rate']:.4f})"
        )

    if prefix and len(input_paths) > 1 and all_rows:
        output_path = combined_output_path(prefix, output_dir)
        progress_output_path = progress_output_path_for(output_path)
        plot_heatmap(all_rows, output_path)
        plot_progress_accept_rate(all_rows, progress_output_path)
        outputs.append(output_path)
        outputs.append(progress_output_path)
        print(f"Overall heatmap: {output_path}")
        print(f"Overall progress line: {progress_output_path}")


if __name__ == "__main__":
    main()
