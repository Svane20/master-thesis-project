import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(df: pd.DataFrame, title: str, save_path: Path) -> None:
    metrics = ["mae", "mse", "grad", "sad", "conn"]
    group_column = "model"
    # Aggregate metrics by model
    agg_df = df.groupby(group_column)[metrics].mean().reset_index()

    # Create a grid of subplots (2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(data=agg_df, x=group_column, y=metric, hue=group_column,
                    palette="Set2", ax=ax, legend=False)
        ax.set_title(metric.upper(), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='x', rotation=15)

        # Value labels on bars
        for i, v in enumerate(agg_df[metric]):
            ax.text(i, v + 0.01 * v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    # Remove any extra subplot if exists
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.show()
    print(f"Saved plot to: {save_path}")

def main() -> None:
    # Load summary.csv
    current_directory = Path(__file__).resolve().parent
    summary_csv = current_directory / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"File not found: {summary_csv}")
    df = pd.read_csv(summary_csv)

    # Split dataframe into sliding window and non-sliding window
    df_sw = df[df["used_sliding_window"] == True]
    df_non_sw = df[df["used_sliding_window"] == False]

    # Plot for non-sliding window inference
    plot_metrics(
        df_non_sw,
        title="Evaluation Metrics without Sliding Window Inference",
        save_path=summary_csv.parent / "comparison_non_sw.png"
    )

    # Plot for sliding window inference
    plot_metrics(
        df_sw,
        title="Evaluation Metrics with Sliding Window Inference",
        save_path=summary_csv.parent / "comparison_sw.png"
    )

if __name__ == "__main__":
    main()