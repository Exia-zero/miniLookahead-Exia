import argparse
import glob
import json
import os
from typing import Any, Dict, List, Sequence


def require_viz_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit(
            "This script requires pandas, seaborn, and matplotlib. "
            "Install them with: pip install pandas seaborn matplotlib"
        ) from exc

    sns.set_theme(style="whitegrid", context="talk")
    return pd, sns, plt


def short_run_name(directory: str) -> str:
    name = os.path.basename(os.path.normpath(directory))
    marker = "_2026"
    if marker in name:
        return name.split(marker, 1)[0]
    return name


def expand_inputs(inputs: Sequence[str], search_dir: str) -> List[str]:
    directories = []
    for item in inputs:
        path = item if os.path.isabs(item) else os.path.join(search_dir, item)
        matches = sorted(glob.glob(path))
        if matches:
            directories.extend(match for match in matches if os.path.isdir(match))
        elif os.path.isdir(path):
            directories.append(path)
    return sorted(dict.fromkeys(directories))


def list_result_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.endswith(".json") and name != "accuracy.json":
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

    num_accepts = int(data.get("num_accepts", sum(numeric_accepts)) or 0)
    num_decisions = int(data.get("num_accept_decisions", len(numeric_accepts)) or 0)
    accept_rate = data.get(
        "accept_rate", num_accepts / num_decisions if num_decisions else 0.0
    )

    return {
        "num_accepts": num_accepts,
        "num_accept_decisions": num_decisions,
        "accept_rate": float(accept_rate or 0.0),
    }


def read_result(path: str, directory: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generation_tokens = data.get("generation_tokens", [])
    token_count = len(generation_tokens) if isinstance(generation_tokens, list) else 0

    return {
        "directory": directory,
        "run": short_run_name(directory),
        "file": os.path.basename(path),
        "question_id": data.get("question_id"),
        "speed": float(data.get("speed", 0.0) or 0.0),
        "time_taken": float(data.get("time_taken", 0.0) or 0.0),
        "tokens": token_count,
        **accept_stats(data),
    }


def load_dataframe(directories: Sequence[str], drop_no_accepts: bool):
    pd, _, _ = require_viz_libs()
    rows = []
    for directory in directories:
        for path in list_result_files(directory):
            try:
                row = read_result(path, directory)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
                continue
            if drop_no_accepts and row["num_accept_decisions"] == 0:
                continue
            rows.append(row)

    if not rows:
        raise SystemExit("No result JSON rows found.")

    df = pd.DataFrame(rows)
    df["question_id"] = df["question_id"].astype("Int64")
    df["accept_rate_pct"] = df["accept_rate"] * 100.0
    df["tokens_k"] = df["tokens"] / 1000.0
    return df


def summarize_by_run(df):
    grouped = (
        df.groupby("run", as_index=False)
        .agg(
            files=("file", "count"),
            avg_speed=("speed", "mean"),
            median_speed=("speed", "median"),
            avg_tokens=("tokens", "mean"),
            median_tokens=("tokens", "median"),
            avg_time=("time_taken", "mean"),
            total_time=("time_taken", "sum"),
            avg_accept_rate=("accept_rate", "mean"),
            total_accepts=("num_accepts", "sum"),
            total_checks=("num_accept_decisions", "sum"),
        )
        .assign(
            pooled_accept_rate=lambda x: x["total_accepts"] / x["total_checks"]
        )
    )

    pooled = {
        "run": "Pooled",
        "files": int(df["file"].count()),
        "avg_speed": df["speed"].mean(),
        "median_speed": df["speed"].median(),
        "avg_tokens": df["tokens"].mean(),
        "median_tokens": df["tokens"].median(),
        "avg_time": df["time_taken"].mean(),
        "total_time": df["time_taken"].sum(),
        "avg_accept_rate": df["accept_rate"].mean(),
        "total_accepts": int(df["num_accepts"].sum()),
        "total_checks": int(df["num_accept_decisions"].sum()),
    }
    pooled["pooled_accept_rate"] = (
        pooled["total_accepts"] / pooled["total_checks"]
        if pooled["total_checks"]
        else 0.0
    )
    return grouped, pooled


def write_summary_markdown(df, out_dir: str, figure_names: Sequence[str]) -> str:
    grouped, pooled = summarize_by_run(df)
    summary_path = os.path.join(out_dir, "summary.md")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Speed, Tokens, and Accept Rate Analysis\n\n")

        f.write("## Per-Run Summary\n\n")
        f.write(
            "| Run | Files | Avg Speed (tok/s) | Median Speed | Avg Tokens | "
            "Median Tokens | Avg Time (s) | Total Time (s) | Avg Accept Rate | "
            "Pooled Accept Rate | Accepts | Checks |\n"
        )
        f.write(
            "| --- | ----: | ----------------: | -----------: | ---------: | "
            "------------: | -----------: | -------------: | --------------: | "
            "-----------------: | ------: | -----: |\n"
        )
        for _, row in grouped.iterrows():
            f.write(
                f"| {row['run']} | {int(row['files'])} | {row['avg_speed']:.4f} | "
                f"{row['median_speed']:.4f} | {row['avg_tokens']:.2f} | "
                f"{row['median_tokens']:.2f} | {row['avg_time']:.2f} | "
                f"{row['total_time']:.2f} | {row['avg_accept_rate']:.4f} | "
                f"{row['pooled_accept_rate']:.4f} | {int(row['total_accepts'])} | "
                f"{int(row['total_checks'])} |\n"
            )
        f.write(
            f"| **Pooled** | **{pooled['files']}** | **{pooled['avg_speed']:.4f}** | "
            f"**{pooled['median_speed']:.4f}** | **{pooled['avg_tokens']:.2f}** | "
            f"**{pooled['median_tokens']:.2f}** | **{pooled['avg_time']:.2f}** | "
            f"**{pooled['total_time']:.2f}** | **{pooled['avg_accept_rate']:.4f}** | "
            f"**{pooled['pooled_accept_rate']:.4f}** | **{pooled['total_accepts']}** | "
            f"**{pooled['total_checks']}** |\n\n"
        )

        f.write("## Correlations\n\n")
        f.write("| Scope | Variables | Pearson | Spearman |\n")
        f.write("| --- | --- | ---: | ---: |\n")
        variables = ["speed", "tokens", "accept_rate"]
        pairs = [("speed", "tokens"), ("speed", "accept_rate"), ("tokens", "accept_rate")]
        for run, run_df in df.groupby("run"):
            pearson_corr = run_df[variables].corr(method="pearson")
            spearman_corr = run_df[variables].corr(method="spearman")
            for left, right in pairs:
                f.write(
                    f"| {run} | {left} vs {right} | "
                    f"{pearson_corr.loc[left, right]:.4f} | "
                    f"{spearman_corr.loc[left, right]:.4f} |\n"
                )
        pearson_corr = df[variables].corr(method="pearson")
        spearman_corr = df[variables].corr(method="spearman")
        for left, right in pairs:
            f.write(
                f"| **Pooled** | **{left} vs {right}** | "
                f"**{pearson_corr.loc[left, right]:.4f}** | "
                f"**{spearman_corr.loc[left, right]:.4f}** |\n"
            )

        f.write("\n## Figures\n\n")
        for name in figure_names:
            f.write(f"- `{name}`\n")

    return summary_path


def save_pairplot(df, out_path: str) -> None:
    _, sns, plt = require_viz_libs()
    plot = sns.pairplot(
        df,
        vars=["speed", "tokens_k", "accept_rate_pct"],
        hue="run",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.76, "edgecolor": "white", "linewidth": 0.5, "s": 65},
        height=3.0,
    )
    plot.fig.suptitle("Pairwise Relationships: Speed, Tokens, Accept Rate", y=1.02)
    plot.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(plot.fig)


def save_scatter(df, out_path: str, x: str, y: str, xlabel: str, ylabel: str, title: str, size: str = None) -> None:
    _, sns, plt = require_viz_libs()
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    kwargs = {
        "data": df,
        "x": x,
        "y": y,
        "hue": "run",
        "alpha": 0.78,
        "edgecolor": "white",
        "linewidth": 0.7,
        "ax": ax,
    }
    if size is not None:
        kwargs.update({"size": size, "sizes": (45, 360)})
    sns.scatterplot(**kwargs)
    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        color="black",
        line_kws={"linestyle": "--", "linewidth": 1.4, "alpha": 0.75},
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_correlation_heatmap(df, out_path: str) -> None:
    _, sns, plt = require_viz_libs()
    corr = df[["speed", "tokens", "accept_rate"]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        xticklabels=["Speed", "Tokens", "Accept Rate"],
        yticklabels=["Speed", "Tokens", "Accept Rate"],
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_run_summary_bars(df, out_path: str) -> None:
    _, sns, plt = require_viz_libs()
    grouped, _ = summarize_by_run(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specs = [
        ("avg_speed", "Avg Speed (tok/s)"),
        ("avg_tokens", "Avg Tokens"),
        ("pooled_accept_rate", "Pooled Accept Rate"),
    ]
    for ax, (key, title) in zip(axes, specs):
        sns.barplot(data=grouped, x="run", y=key, hue="run", dodge=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_accept_rate_distribution(df, out_path: str) -> None:
    _, sns, plt = require_viz_libs()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    sns.boxplot(data=df, x="run", y="accept_rate_pct", hue="run", dodge=False, ax=ax)
    sns.stripplot(
        data=df,
        x="run",
        y="accept_rate_pct",
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.22,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("Accept Rate (%)")
    ax.set_title("Accept Rate Distribution by Run")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_plots(df, out_dir: str) -> List[str]:
    specs = [
        ("pairplot_speed_tokens_accept.png", save_pairplot),
        (
            "scatter_accept_rate_speed_tokens.png",
            lambda data, path: save_scatter(
                data,
                path,
                "accept_rate_pct",
                "speed",
                "Accept Rate (%)",
                "Speed (tok/s)",
                "Speed vs Accept Rate (point size = tokens)",
                "tokens",
            ),
        ),
        (
            "scatter_speed_tokens.png",
            lambda data, path: save_scatter(
                data, path, "tokens", "speed", "Tokens", "Speed (tok/s)", "Speed vs Tokens"
            ),
        ),
        (
            "scatter_accept_rate_speed.png",
            lambda data, path: save_scatter(
                data,
                path,
                "accept_rate_pct",
                "speed",
                "Accept Rate (%)",
                "Speed (tok/s)",
                "Speed vs Accept Rate",
            ),
        ),
        (
            "scatter_accept_rate_tokens.png",
            lambda data, path: save_scatter(
                data,
                path,
                "accept_rate_pct",
                "tokens",
                "Accept Rate (%)",
                "Tokens",
                "Tokens vs Accept Rate",
            ),
        ),
        ("correlation_heatmap.png", save_correlation_heatmap),
        ("run_summary_bars.png", save_run_summary_bars),
        ("accept_rate_distribution.png", save_accept_rate_distribution),
    ]

    plot_paths = []
    for filename, writer in specs:
        path = os.path.join(out_dir, filename)
        writer(df, path)
        plot_paths.append(path)
    return plot_paths


def print_summary(df) -> None:
    grouped, pooled = summarize_by_run(df)
    print("Per-run summary")
    for _, row in grouped.iterrows():
        print(
            f"  {row['run']}: files={int(row['files'])} "
            f"avg_speed={row['avg_speed']:.4f} "
            f"avg_tokens={row['avg_tokens']:.2f} "
            f"pooled_accept_rate={row['pooled_accept_rate']:.4f}"
        )
    print(
        "Pooled: "
        f"files={pooled['files']} avg_speed={pooled['avg_speed']:.4f} "
        f"avg_tokens={pooled['avg_tokens']:.2f} "
        f"pooled_accept_rate={pooled['pooled_accept_rate']:.4f}"
    )

    corr = df[["speed", "tokens", "accept_rate"]].corr(method="pearson")
    print("\nPooled Pearson correlations")
    print(f"  speed vs tokens: {corr.loc['speed', 'tokens']:.4f}")
    print(f"  speed vs accept_rate: {corr.loc['speed', 'accept_rate']:.4f}")
    print(f"  tokens vs accept_rate: {corr.loc['tokens', 'accept_rate']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze speed, generation tokens, and accept rate from result JSON files, "
            "then generate CSV, Markdown, and seaborn/matplotlib visualizations."
        )
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Result directories or glob patterns. Example: 'lookahead_d4_run*'",
    )
    parser.add_argument(
        "--search-dir",
        default=".",
        help="Base directory for relative result dirs or glob patterns.",
    )
    parser.add_argument(
        "--out-dir",
        default="speed_tokens_accept_analysis",
        help="Output directory for CSV, Markdown, and plots.",
    )
    parser.add_argument(
        "--drop-no-accepts",
        action="store_true",
        help="Drop rows with zero accept decisions. Useful when baseline dirs are mixed in.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV and Markdown summary, without PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.dirs or ["lookahead_d4_run*"]
    directories = expand_inputs(inputs, args.search_dir)
    if not directories:
        raise SystemExit("No result directories found.")

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_dataframe(directories, drop_no_accepts=args.drop_no_accepts)

    csv_path = os.path.join(args.out_dir, "rows.csv")
    df.to_csv(csv_path, index=False)

    plot_paths = []
    if not args.no_plots:
        plot_paths = make_plots(df, args.out_dir)

    summary_path = write_summary_markdown(
        df, args.out_dir, [os.path.basename(path) for path in plot_paths]
    )

    print_summary(df)
    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote Markdown summary: {summary_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
