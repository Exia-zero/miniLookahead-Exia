import argparse
import csv
import json
import os
from statistics import mean, median
from typing import Any, Dict, List, Optional


def list_result_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if name.lower().endswith(".json") and name.lower() != "accuracy.json":
            files.append(path)
    return sorted(files)


def accept_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    accepts = data.get("accepts", [])
    if not isinstance(accepts, list):
        accepts = []

    numeric_accepts = []
    for value in accepts:
        if isinstance(value, bool):
            numeric_accepts.append(1 if value else 0)
        elif isinstance(value, (int, float)):
            numeric_accepts.append(1 if float(value) >= 0.5 else 0)

    num_accepts = data.get("num_accepts", sum(numeric_accepts))
    num_decisions = data.get("num_accept_decisions", len(numeric_accepts))
    accept_rate = data.get(
        "accept_rate", num_accepts / num_decisions if num_decisions else 0.0
    )

    return {
        "num_accepts": num_accepts,
        "num_accept_decisions": num_decisions,
        "accept_rate": accept_rate,
    }


def read_result(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None

    stats = accept_stats(data)
    generation_tokens = data.get("generation_tokens", [])
    token_count = len(generation_tokens) if isinstance(generation_tokens, list) else 0

    return {
        "file": os.path.basename(path),
        "question_id": data.get("question_id"),
        "method": data.get("method"),
        "speed": float(data.get("speed", 0.0) or 0.0),
        "time_taken": float(data.get("time_taken", 0.0) or 0.0),
        "generation_tokens": token_count,
        **stats,
    }


def analyze_directory(directory: str) -> List[Dict[str, Any]]:
    rows = []
    for path in list_result_files(directory):
        row = read_result(path)
        if row is not None:
            row["directory"] = directory
            rows.append(row)
    return rows


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def rows_by_question_id(directory: str) -> Dict[Any, Dict[str, Any]]:
    rows = {}
    for row in analyze_directory(directory):
        question_id = row.get("question_id")
        if question_id is not None:
            rows[question_id] = row
    return rows


def sort_rows(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    reverse = key in {"speed", "time_taken", "generation_tokens", "accept_rate"}
    return sorted(rows, key=lambda row: (row.get(key) is None, row.get(key)), reverse=reverse)


def shorten(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def print_rows(rows: List[Dict[str, Any]]) -> None:
    columns = [
        ("directory", "dir", 30, "<"),
        ("file", "file", 12, "<"),
        ("question_id", "qid", 5, ">"),
        ("method", "method", 8, "<"),
        ("speed", "speed", 9, ">"),
        ("time_taken", "time_s", 10, ">"),
        ("generation_tokens", "tokens", 8, ">"),
        ("num_accepts", "accepts", 8, ">"),
        ("num_accept_decisions", "checks", 8, ">"),
        ("accept_rate", "acc_rate", 8, ">"),
    ]
    header = "  ".join(
        f"{label:{align}{width}}" for _, label, width, align in columns
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        values = {
            "directory": shorten(row.get("directory", ""), 30),
            "file": shorten(row.get("file", ""), 12),
            "question_id": row.get("question_id", ""),
            "method": row.get("method", ""),
            "speed": f"{row.get('speed', 0.0):.4f}",
            "time_taken": f"{row.get('time_taken', 0.0):.2f}",
            "generation_tokens": row.get("generation_tokens", 0),
            "num_accepts": row.get("num_accepts", 0),
            "num_accept_decisions": row.get("num_accept_decisions", 0),
            "accept_rate": f"{row.get('accept_rate', 0.0):.4f}",
        }
        print(
            "  ".join(
                f"{str(values[key]):{align}{width}}"
                for key, _, width, align in columns
            )
        )


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("\nNo result files found.")
        return

    total_accepts = sum(row["num_accepts"] for row in rows)
    total_decisions = sum(row["num_accept_decisions"] for row in rows)
    overall_accept_rate = total_accepts / total_decisions if total_decisions else 0.0

    print("\nSummary")
    print(f"  Files: {len(rows)}")
    print(f"  Avg speed: {mean(row['speed'] for row in rows):.4f}")
    print(f"  Avg time_taken: {mean(row['time_taken'] for row in rows):.2f}")
    print(f"  Avg generation_tokens: {mean(row['generation_tokens'] for row in rows):.2f}")
    print(f"  Total accept decisions: {total_decisions}")
    print(f"  Total accepts: {total_accepts}")
    print(f"  Overall accept rate: {overall_accept_rate:.4f}")


def compare_directories(
    baseline_dir: str, experiment_dir: str
) -> List[Dict[str, Any]]:
    baseline = rows_by_question_id(baseline_dir)
    experiment = rows_by_question_id(experiment_dir)
    common_ids = sorted(set(baseline) & set(experiment))

    rows = []
    for question_id in common_ids:
        base = baseline[question_id]
        exp = experiment[question_id]
        base_time = base["time_taken"]
        exp_time = exp["time_taken"]
        base_speed = base["speed"]
        exp_speed = exp["speed"]

        rows.append(
            {
                "question_id": question_id,
                "baseline_time": base_time,
                "experiment_time": exp_time,
                "time_speedup": safe_div(base_time, exp_time),
                "baseline_speed": base_speed,
                "experiment_speed": exp_speed,
                "speed_ratio": safe_div(exp_speed, base_speed),
                "baseline_tokens": base["generation_tokens"],
                "experiment_tokens": exp["generation_tokens"],
                "experiment_accepts": exp["num_accepts"],
                "experiment_checks": exp["num_accept_decisions"],
                "experiment_accept_rate": exp["accept_rate"],
            }
        )
    return rows


def print_compare_rows(rows: List[Dict[str, Any]]) -> None:
    columns = [
        ("question_id", "qid", 5, ">"),
        ("baseline_time", "base_t", 10, ">"),
        ("experiment_time", "exp_t", 10, ">"),
        ("time_speedup", "time_x", 8, ">"),
        ("baseline_speed", "base_spd", 9, ">"),
        ("experiment_speed", "exp_spd", 9, ">"),
        ("speed_ratio", "speed_x", 8, ">"),
        ("baseline_tokens", "base_tok", 9, ">"),
        ("experiment_tokens", "exp_tok", 9, ">"),
        ("experiment_accept_rate", "acc_rate", 8, ">"),
    ]
    header = "  ".join(
        f"{label:{align}{width}}" for _, label, width, align in columns
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        values = {
            "question_id": row.get("question_id", ""),
            "baseline_time": f"{row.get('baseline_time', 0.0):.2f}",
            "experiment_time": f"{row.get('experiment_time', 0.0):.2f}",
            "time_speedup": f"{row.get('time_speedup', 0.0):.4f}",
            "baseline_speed": f"{row.get('baseline_speed', 0.0):.4f}",
            "experiment_speed": f"{row.get('experiment_speed', 0.0):.4f}",
            "speed_ratio": f"{row.get('speed_ratio', 0.0):.4f}",
            "baseline_tokens": row.get("baseline_tokens", 0),
            "experiment_tokens": row.get("experiment_tokens", 0),
            "experiment_accept_rate": f"{row.get('experiment_accept_rate', 0.0):.4f}",
        }
        print(
            "  ".join(
                f"{str(values[key]):{align}{width}}"
                for key, _, width, align in columns
            )
        )


def print_compare_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("\nNo common question_id found.")
        return

    baseline_time = sum(row["baseline_time"] for row in rows)
    experiment_time = sum(row["experiment_time"] for row in rows)
    baseline_tokens = sum(row["baseline_tokens"] for row in rows)
    experiment_tokens = sum(row["experiment_tokens"] for row in rows)
    baseline_effective_speed = safe_div(baseline_tokens, baseline_time)
    experiment_effective_speed = safe_div(experiment_tokens, experiment_time)
    time_speedups = [row["time_speedup"] for row in rows]
    faster_count = sum(1 for value in time_speedups if value > 1.0)
    total_accepts = sum(row["experiment_accepts"] for row in rows)
    total_checks = sum(row["experiment_checks"] for row in rows)

    print("\nCompare summary")
    print(f"  Common questions: {len(rows)}")
    print(f"  Baseline total time: {baseline_time:.2f}")
    print(f"  Experiment total time: {experiment_time:.2f}")
    print(f"  Time speedup: {safe_div(baseline_time, experiment_time):.4f}x")
    print(f"  Baseline effective speed: {baseline_effective_speed:.4f}")
    print(f"  Experiment effective speed: {experiment_effective_speed:.4f}")
    print(
        "  Effective speed ratio: "
        f"{safe_div(experiment_effective_speed, baseline_effective_speed):.4f}x"
    )
    print(f"  Mean per-question time speedup: {mean(time_speedups):.4f}x")
    print(f"  Median per-question time speedup: {median(time_speedups):.4f}x")
    print(f"  Faster questions: {faster_count}/{len(rows)}")
    print(f"  Experiment accept rate: {safe_div(total_accepts, total_checks):.4f}")


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Show per-question speed and accept rate from run_dataset.py outputs."
    )
    parser.add_argument("dirs", nargs="*", help="One or more result directories")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_DIR", "EXPERIMENT_DIR"),
        help="Compare baseline and experiment directories by common question_id",
    )
    parser.add_argument(
        "--sort",
        choices=[
            "question_id",
            "speed",
            "time_taken",
            "generation_tokens",
            "accept_rate",
        ],
        default="question_id",
        help="Sort output rows",
    )
    parser.add_argument("--csv", help="Optional CSV output path")
    args = parser.parse_args()

    if args.compare:
        rows = compare_directories(args.compare[0], args.compare[1])
        print(f"Baseline:   {args.compare[0]}")
        print(f"Experiment: {args.compare[1]}\n")
        print_compare_rows(rows)
        print_compare_summary(rows)

        if args.csv:
            write_csv(rows, args.csv)
            print(f"\nWrote CSV: {args.csv}")
        return

    if not args.dirs:
        parser.error("dirs are required unless --compare is used")

    rows = []
    for directory in args.dirs:
        rows.extend(analyze_directory(directory))

    rows = sort_rows(rows, args.sort)
    print_rows(rows)
    print_summary(rows)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
