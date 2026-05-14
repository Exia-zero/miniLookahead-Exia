import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple


def extract_boxed_answer(text: str) -> str:
    """Extract the last boxed answer from text."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1] if matches else ""


def load_jsonl_files(directory: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load all per-question JSON result files from a directory."""
    data = []
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename != "accuracy.json":
            file_count += 1
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
    return data, file_count


def calculate_metrics(data: List[Dict[str, Any]]) -> Tuple:
    """Calculate accuracy and speed for given data."""
    # Calculate average speed
    speeds = [item.get("speed", 0) for item in data if "speed" in item]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0

    # Calculate accept rate
    total_accepts = 0
    total_accept_decisions = 0
    for item in data:
        accepts = item.get("accepts", [])
        if isinstance(accepts, list):
            total_accepts += sum(1 for v in accepts if v)
            total_accept_decisions += len(accepts)
    accept_rate = (
        total_accepts / total_accept_decisions if total_accept_decisions else 0
    )

    # Calculate accuracy
    correct = 0
    total = 0
    total_tokens = 0
    total_thinks = 0

    for item in data:
        if "gold" in item and "full_text" in item:
            gold_answer = str(item["gold"]).strip()
            predicted_answer = extract_boxed_answer(item["full_text"]).strip()

            total_thinks += item["full_text"].count("</think>")
            total_tokens += len(item["generation_tokens"])
            # print(len(item['generation_tokens']) / item['time_taken'], item['speed'])
            if predicted_answer.isdigit() and gold_answer.isdigit():
                if int(gold_answer) == int(predicted_answer):
                    correct += 1
            total += 1
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_thinks = total_thinks / total if total > 0 else 0
    accuracy = (correct / total * 100) if total > 0 else 0
    return (
        accuracy,
        avg_speed,
        avg_tokens,
        avg_thinks,
        correct,
        total,
        accept_rate,
        total_accepts,
        total_accept_decisions,
    )


def find_directories_with_prefix(prefix: str, search_dir: str) -> List[str]:
    """Find all directories under search_dir that start with the given prefix."""
    matching_dirs = []

    if not os.path.isdir(search_dir):
        return matching_dirs

    for item in os.listdir(search_dir):
        path = os.path.join(search_dir, item)
        if os.path.isdir(path) and item.startswith(prefix):
            matching_dirs.append(path)

    return sorted(matching_dirs)


def analyze_results(prefix: str, search_dir: str) -> None:
    """Analyze results from directories matching the prefix."""
    directories = find_directories_with_prefix(prefix, search_dir)

    if not directories:
        print(f"No directories found with prefix '{prefix}' under '{search_dir}'")
        return

    print(f"Searching in: {search_dir}")
    print(f"Found directories with prefix '{prefix}': {directories}")
    print()

    all_data = []

    print("Individual Directory Results:")
    print("-" * 50)

    for directory in directories:
        data, file_count = load_jsonl_files(directory)
        all_data.extend(data)

        (
            accuracy,
            avg_speed,
            avg_tokens,
            avg_thinks,
            correct,
            total,
            accept_rate,
            total_accepts,
            total_accept_decisions,
        ) = calculate_metrics(data)

        print(f"Directory: {directory}")
        print(f"  Entries: {len(data)} (from {file_count} files)")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        print(f"  Average tokens: {avg_tokens}")
        print(f"  Average thinks: {avg_thinks}")
        print(f"  Average speed: {avg_speed:.2f}")
        print(
            f"  Accept rate: {total_accepts}/{total_accept_decisions} ({accept_rate:.4f})"
        )
        print()

    if not all_data:
        print("No data found")
        return

    # Calculate overall metrics
    (
        overall_accuracy,
        overall_speed,
        avg_tokens,
        avg_thinks,
        overall_correct,
        overall_total,
        overall_accept_rate,
        overall_total_accepts,
        overall_total_accept_decisions,
    ) = calculate_metrics(all_data)

    print("Overall Results:")
    print("-" * 20)
    print(f"Total entries: {len(all_data)}")
    print(
        f"Overall accuracy: {overall_correct}/{overall_total} ({overall_accuracy:.2f}%)"
    )
    print(f"Overall average speed: {overall_speed:.2f}")
    print(f"Overall average tokens: {avg_tokens}")
    print(f"Overall average thinks: {avg_thinks}")
    print(
        f"Overall accept rate: {overall_total_accepts}/{overall_total_accept_decisions} "
        f"({overall_accept_rate:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSON results from directories with matching prefix"
    )
    parser.add_argument("prefix", help="Prefix to match directory names")
    parser.add_argument(
        "--search-dir",
        "--search_dir",
        default="results",
        dest="search_dir",
        help="Directory containing result run folders",
    )

    args = parser.parse_args()
    analyze_results(args.prefix, args.search_dir)


if __name__ == "__main__":
    main()
