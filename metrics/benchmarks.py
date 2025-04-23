import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import os

output_dir = './charts/benchmarks'
os.makedirs(output_dir, exist_ok=True)

files = {
    'inference': './csvs/benchmarks/inference_times.csv',
    'batch_inference': './csvs/benchmarks/batch_inference_times.csv',
    'sky_replacement': './csvs/benchmarks/sky_replacement_times.csv',
    'batch_sky_replacement': './csvs/benchmarks/batch_sky_replacement_times.csv',
    'model_load': './csvs/benchmarks/model_load_times.csv',
}
name_map = {'dpt': 'DPT', 'resnet': 'ResNet', 'swin': 'Swin'}
type_map = {'onnx': 'ONNX', 'pytorch': 'PyTorch', 'torchscript': 'TorchScript'}
hardware_list = ['CPU', 'GPU']


def plot_file(key, path):
    df = pd.read_csv(path)
    pivot = df.pivot_table(
        index='model_name',
        columns=['model_type', 'hardware'],
        values='avg_time_sec', aggfunc='mean'
    )
    model_names = pivot.index.tolist()
    fig, axes = plt.subplots(ncols=len(model_names), figsize=(5 * len(model_names), 6))
    for ax, mn in zip(axes, model_names):
        df_plot = pivot.loc[mn].unstack(level='hardware').reindex(hardware_list, axis=1)
        formats = df_plot.index.tolist()
        cpu_vals = df_plot['CPU'].values
        gpu_vals = df_plot['GPU'].values

        x = np.arange(len(formats))
        w = 0.35
        ax.bar(x - w / 2, cpu_vals, w, label='CPU')
        ax.bar(x + w / 2, gpu_vals, w, label='GPU')

        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=None))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())

        max_val = max(cpu_vals.max(), gpu_vals.max())
        bottom, _ = ax.get_ylim()
        ax.set_ylim(bottom, max_val * 1.15)

        ax.set_title(name_map.get(mn, mn), fontsize=14)
        if mn == model_names[0]:
            ax.set_ylabel("Avg Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([type_map.get(fmt, fmt) for fmt in formats])
        ax.grid(True, which='both', linestyle=':')

        for rect in ax.patches:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h * 1.02,
                f"{h:.3f}",
                ha='center', va='bottom', fontsize=8
            )

    fig.suptitle(f"{key.replace('_', ' ').title()} Time by Model and Format", fontsize=16, y=0.98)
    fig.supxlabel("Format", y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])

    output_path = os.path.join(output_dir, f"{key}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


for k, p in files.items():
    plot_file(k, p)
