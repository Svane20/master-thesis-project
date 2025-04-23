import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
format_order = ["onnx", "torchscript", "pytorch"]
format_titles = {"onnx": "ONNX", "torchscript": "TorchScript", "pytorch": "PyTorch"}
model_titles = {"dpt": "DPT", "resnet": "ResNet", "swin": "Swin"}

# Gather median latency per run
stats_files = glob.glob('./csvs/performance/*_stats_history.csv')
file_info = []
for fp in stats_files:
    fname = os.path.basename(fp)
    model, fmt, device, workers = fname.split('_')[0], fname.split('_')[1], fname.split('_')[2], int(fname.split('_')[3])
    file_info.append((model, fmt, device, workers, fp))

# Create output directory
out_dir = './charts/performance'
os.makedirs(out_dir, exist_ok=True)
saved_files = []

# Plot per model
for model in sorted({m for m, *_ in file_info}):
    fig, axes = plt.subplots(2, len(format_order), figsize=(14, 8), sharex=True, sharey=False)
    fig.suptitle(f"{model_titles.get(model, model)}: Median Latency vs Concurrent Users", fontsize=16)

    # compute GPU y-limit
    gpu_latencies = []
    for m, fmt, dev, w, fp in file_info:
        if m == model and dev == 'gpu':
            df = pd.read_csv(fp).dropna(subset=['50%'])
            gpu_latencies.extend(df.groupby('User Count')['50%'].median().values)
    if gpu_latencies:
        ylim_gpu = (0, max(gpu_latencies) * 1.1)
    else:
        ylim_gpu = None

    for row, device in enumerate(['cpu','gpu']):
        for col, fmt in enumerate(format_order):
            ax = axes[row, col]
            ax.set_title(format_titles[fmt])
            # plot each worker count
            for m, f, dev, w, fp in file_info:
                if m == model and dev == device and f == fmt:
                    df = pd.read_csv(fp).dropna(subset=['50%'])
                    grp = df.groupby('User Count')['50%'].median().reset_index()
                    ax.plot(grp['User Count'], grp['50%'], marker='o',
                            label=f"{w} Worker" + ("s" if w != 1 else ""))
            if col == 0:
                ax.set_ylabel(f"{device.upper()} Median Latency (ms)")
            if row == 1:
                ax.set_xlabel("Concurrent Users")
            ax.grid(True)
            if device == 'gpu' and ylim_gpu is not None:
                ax.set_ylim(ylim_gpu)
            if ax.get_lines():
                ax.legend(title="Workers")

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    # Save figure
    filename = f"{model}_median_latency.png"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=300)
    saved_files.append(filepath)
    plt.close(fig)
