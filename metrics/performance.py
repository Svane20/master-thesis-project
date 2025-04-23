import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# Configuration
format_order = ["onnx", "torchscript", "pytorch"]
format_titles = {"onnx": "ONNX", "torchscript": "TorchScript", "pytorch": "PyTorch"}
model_titles = {"dpt": "DPT", "resnet": "ResNet", "swin": "Swin"}

# Paths
stats_glob = './csvs/performance/*_stats_history.csv'
sys_glob   = './csvs/performance/*_sys.csv'
out_dir = './charts/performance'
os.makedirs(out_dir, exist_ok=True)

# Gather stats and sys file info
stat_info = []
for fp in glob.glob(stats_glob):
    fname = os.path.basename(fp)
    model, fmt, device, workers = fname.split('_')[0], fname.split('_')[1], fname.split('_')[2], int(fname.split('_')[3])
    stat_info.append((model, fmt, device, workers, fp))

sys_map = {}
for fp in glob.glob(sys_glob):
    fname = os.path.basename(fp)
    model, fmt, device, workers = fname.split('_')[0], fname.split('_')[1], fname.split('_')[2], int(fname.split('_')[3])
    sys_map[(model, fmt, device, workers)] = fp

# Median Latency (existing)
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(2, len(format_order), figsize=(14, 8), sharex=True)
    fig.suptitle(f"{model_titles.get(model, model)}: Median Latency vs Concurrent Users", fontsize=16)
    # y-limit for GPU
    gpu_lat = []
    for m, f, dev, w, fp in stat_info:
        if m==model and dev=='gpu':
            df = pd.read_csv(fp).dropna(subset=['50%'])
            gpu_lat.extend(df.groupby('User Count')['50%'].median().values)
    ylim_gpu = (0, max(gpu_lat)*1.1) if gpu_lat else None

    for i, device in enumerate(['cpu','gpu']):
        for j, fmt in enumerate(format_order):
            ax = axes[i,j]
            ax.set_title(format_titles[fmt])
            for m,f,dev,w,fp in stat_info:
                if m==model and dev==device and f==fmt:
                    df = pd.read_csv(fp).dropna(subset=['50%'])
                    grp = df.groupby('User Count')['50%'].median().reset_index()
                    ax.plot(grp['User Count'], grp['50%'], 'o-', label=f"{w} Worker"+("s" if w!=1 else ""))
            if j==0: ax.set_ylabel(f"{device.upper()} Median Latency (ms)")
            if i==1: ax.set_xlabel("Concurrent Users")
            ax.grid(True)
            if device=='gpu' and ylim_gpu is not None:
                ax.set_ylim(ylim_gpu)
            if ax.get_lines(): ax.legend(title="Workers")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
    fig.savefig(os.path.join(out_dir, f"{model}_median_latency.png"), dpi=300)
    plt.close(fig)

# Throughput vs Concurrent Users
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(
        2, len(format_order),
        figsize=(14, 8),
        sharex=True,      # share the x-axis only
        sharey=False      # do NOT share y—each row can scale independently
    )
    fig.suptitle(f"{model_titles[model]}: Throughput vs Concurrent Users", fontsize=16)

    for i, device in enumerate(['cpu','gpu']):
        # compute a tight y-limit per device
        max_thru = max(
            pd.read_csv(fp)['Requests/s'].max()
            for m,f,dev,w,fp in stat_info
            if m == model and dev == device
        )
        y_lim = (0, max_thru * 0.6) if max_thru > 0 else None

        for j, fmt in enumerate(format_order):
            ax = axes[i, j]
            ax.set_title(format_titles[fmt])

            # plot each worker count
            for m,f,dev,w,fp in stat_info:
                if m==model and dev==device and f==fmt:
                    df = pd.read_csv(fp)
                    thru = (
                        df
                        .groupby('User Count')['Requests/s']
                        .mean()
                        .reset_index()
                    )
                    ax.plot(
                        thru['User Count'],
                        thru['Requests/s'],
                        'o-',
                        label=f"{w} Worker" + ("s" if w!=1 else "")
                    )

            # only the leftmost col gets a y-label
            if j == 0:
                ax.set_ylabel(f"{device.upper()} Req/s")
            # bottom row gets the x-label
            if i == 1:
                ax.set_xlabel("Concurrent Users")

            ax.grid(True)
            if y_lim:
                ax.set_ylim(y_lim)
            if ax.get_lines():
                ax.legend(title="Workers")

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    fig.savefig(os.path.join(out_dir, f"{model}_throughput.png"), dpi=300)
    plt.close(fig)

# Latency Percentiles over Load
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(
        2, len(format_order),
        figsize=(14, 8),
        sharex=True,
        sharey=False
    )
    fig.suptitle(f"{model_titles[model]}: Latency Percentiles vs Users", fontsize=16)

    for i, device in enumerate(['cpu','gpu']):
        # collect median-percentile series for all formats/workers
        all_percentiles = []
        for m,f,dev,w,fp in stat_info:
            if m!=model or dev!=device:
                continue
            df = pd.read_csv(fp).dropna(subset=['50%','90%','95%','99%'])
            # choose which percentile to base our y-limit on:
            # CPU → ’90%’, GPU → ’95%’
            target = '90%' if device=='cpu' else '95%'
            series = df.groupby('User Count')[target].median()
            all_percentiles.append(series)

        if all_percentiles:
            # concatenate, take the max across all series
            max_base = pd.concat(all_percentiles).max()
            y_lim = (0, max_base * 1.1)
        else:
            y_lim = None

        for j, fmt in enumerate(format_order):
            ax = axes[i, j]
            ax.set_title(format_titles[fmt])

            # plot each percentile
            for perc in ['50%','90%','99%']:
                # we want one line PER worker-count
                for m,f,dev,w,fp in stat_info:
                    if m==model and dev==device and f==fmt:
                        df = pd.read_csv(fp).dropna(subset=['50%','90%','99%'])
                        grp = df.groupby('User Count')[[perc]].median()
                        ax.plot(
                            grp.index,
                            grp[perc],
                            'o-',
                            label=f"{perc} ({w}W)"
                        )

            if j == 0:
                ax.set_ylabel(f"{device.upper()} Latency (ms)")
            if i == 1:
                ax.set_xlabel("Concurrent Users")

            ax.grid(True)
            if y_lim:
                ax.set_ylim(y_lim)
            # only one legend per subplot
            if ax.get_lines():
                ax.legend(title="Percentile / Workers", fontsize='small', loc='upper left')

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    fig.savefig(os.path.join(out_dir, f"{model}_latency_percentiles.png"), dpi=300)
    plt.close(fig)

# Speedup Ratios vs PyTorch
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{model_titles[model]}: Speedup vs PyTorch", fontsize=16)

    for i, device in enumerate(['cpu', 'gpu']):
        # gather median-latency series
        med = {}
        for m, f, dev, w, fp in stat_info:
            if m == model and dev == device:
                df = pd.read_csv(fp).dropna(subset=['50%'])
                med[(f, w)] = df.groupby('User Count')['50%'].median()

        for j, fmt in enumerate(['onnx', 'torchscript']):
            ax = axes[i, j]
            ax.set_title(f"{format_titles[fmt]}/PyTorch")

            # track the finite max ratio
            max_ratio = 0.0

            for (f, w), series in med.items():
                if f == fmt and ('pytorch', w) in med:
                    base = med[('pytorch', w)]
                    ratio = base.div(series).replace([np.inf, -np.inf], np.nan).dropna()
                    if not ratio.empty:
                        series_max = ratio.max()
                        if math.isfinite(series_max):
                            max_ratio = max(max_ratio, series_max)
                    ax.plot(ratio.index, ratio.values, 'o-', label=f"{w} Workers")

            # only set y-limits if we found a valid max_ratio
            if max_ratio > 0 and math.isfinite(max_ratio):
                ax.set_ylim(0, max_ratio * 1.1)

            if i == 1:
                ax.set_xlabel("Concurrent Users")
            if j == 0:
                ax.set_ylabel(f"{device.upper()} Speedup")

            ax.grid(True)
            if ax.get_lines():
                ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    fig.savefig(os.path.join(out_dir, f"{model}_speedup.png"), dpi=300)
    plt.close(fig)
