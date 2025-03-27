import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    # Load summary.csv
    current_directory = Path(__file__).resolve().parent
    summary_csv = current_directory / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"File not found: {summary_csv}")
    df = pd.read_csv(summary_csv)

    # Settings
    metrics = ["mae", "mse", "grad", "sad", "conn"]
    group_column = "model"

    # Aggregate
    agg_df = df.groupby(group_column)[metrics].mean().reset_index()

    # Grid layout (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(data=agg_df, x=group_column, y=metric, hue=group_column, palette="Set2", ax=ax, legend=False)
        ax.set_title(metric.upper(), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='x', rotation=15)

        # Value labels
        for i, v in enumerate(agg_df[metric]):
            ax.text(i, v + 0.01 * v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    # Remove last plot if not used
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle("Model Evaluation Metrics by Model", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = summary_csv.parent / "comparison.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"Grid comparison saved to: {plot_path}")


if __name__ == "__main__":
    main()
