import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

sys_info = []
for fp in glob.glob(sys_glob):
    fname = os.path.basename(fp)
    model, fmt, device, workers = fname.split('_')[0], fname.split('_')[1], fname.split('_')[2], int(fname.split('_')[3])
    sys_info.append((model, fmt, device, workers, fp))

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

# Throughput vs Concurrent Users
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(2, len(format_order), figsize=(14,8), sharex=True, sharey=True)
    fig.suptitle(f"{model_titles[model]}: Throughput vs Concurrent Users", fontsize=16)
    for i, device in enumerate(['cpu','gpu']):
        for j, fmt in enumerate(format_order):
            ax = axes[i,j]
            ax.set_title(format_titles[fmt])
            for m,f,dev,w,fp in stat_info:
                if m==model and dev==device and f==fmt:
                    df = pd.read_csv(fp)
                    thru = df.groupby('User Count')['Requests/s'].mean().reset_index()
                    ax.plot(thru['User Count'], thru['Requests/s'],'o-', label=f"{w} Workers")
            if i==1: ax.set_xlabel("Concurrent Users")
            if j==0: ax.set_ylabel("Req/s")
            ax.grid(True)
            if ax.get_lines(): ax.legend()
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# Latency Percentiles over Load
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(2, len(format_order), figsize=(14,8), sharex=True, sharey=True)
    fig.suptitle(f"{model_titles[model]}: Latency Percentiles vs Users", fontsize=16)
    for i, device in enumerate(['cpu','gpu']):
        for j, fmt in enumerate(format_order):
            ax = axes[i,j]
            ax.set_title(format_titles[fmt])
            for m,f,dev,w,fp in stat_info:
                if m==model and dev==device and f==fmt:
                    df = pd.read_csv(fp).dropna(subset=['50%','90%','99%'])
                    grp = df.groupby('User Count')[['50%','90%','99%']].median()
                    for perc in grp.columns:
                        ax.plot(grp.index, grp[perc],'o-', label=perc)
            if i==1: ax.set_xlabel("Concurrent Users")
            if j==0: ax.set_ylabel("Latency (ms)")
            ax.grid(True)
            if ax.get_lines(): ax.legend(title="Percentile")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# Speedup Ratios vs PyTorch
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)
    fig.suptitle(f"{model_titles[model]}: Speedup vs PyTorch", fontsize=16)
    for i, device in enumerate(['cpu','gpu']):
        # gather medians per format/worker
        med = {}
        for m,f,dev,w,fp in stat_info:
            if m==model and dev==device:
                df = pd.read_csv(fp).dropna(subset=['50%'])
                med[(f,w)] = df.groupby('User Count')['50%'].median()
        for j, fmt in enumerate(['onnx','torchscript']):
            ax = axes[i,j]
            ax.set_title(f"{format_titles[fmt]}/PyTorch")
            for (f,w), series in med.items():
                if f==fmt and ('pytorch',w) in med:
                    ratio = med[('pytorch',w)] / series
                    ax.plot(ratio.index, ratio.values,'o-', label=f"{w} Workers")
            if i==1: ax.set_xlabel("Concurrent Users")
            if j==0: ax.set_ylabel("Speedup")
            ax.grid(True)
            if ax.get_lines(): ax.legend()
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# Scaling Efficiency at 50 users
fixed_users = 50
for model in sorted({m for m, *_ in stat_info}):
    fig, axes = plt.subplots(1,2, figsize=(12,5), sharey=True)
    fig.suptitle(f"{model_titles[model]}: Scaling Efficiency @ {fixed_users} Users", fontsize=16)

    for i, device in enumerate(['cpu','gpu']):
        ax = axes[i]
        ax.set_title(device.upper()); ax.set_xlabel("Workers")
        if i == 0: ax.set_ylabel("Speedup")

        # gather latencies
        lat = {}
        for m, f, dev, w, fp in stat_info:
            if m == model and dev == device:
                df = pd.read_csv(fp)
                # median at fixed_users
                med_val = df[df['User Count'] == fixed_users]['50%'].dropna().median()
                if np.isfinite(med_val) and med_val > 0:
                    lat.setdefault(f, {})[w] = med_val

        # compute speedup ratios avoiding zero
        for fmt in format_order:
            workers_list = sorted(lat.get(fmt, {}).keys())
            if 1 in workers_list:
                base = lat[fmt][1]
                speedups = []
                for w in workers_list:
                    val = lat[fmt].get(w, np.nan)
                    sp = base / val if val and val > 0 else np.nan
                    speedups.append(sp)
                ax.plot(workers_list, speedups, 'o-', label=format_titles[fmt])

        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0,0,1,0.9])
    plt.show()

# Resource Utilization vs Load
for model, fmt, device, workers, fp in sys_info:
    stats_fp = fp.replace('_sys.csv','_stats_history.csv')
    if not os.path.exists(stats_fp): continue
    df_s = pd.read_csv(stats_fp)
    ucs = df_s['User Count'].unique()
    if len(ucs)!=1: continue
    uc = ucs[0]
    df_u = pd.read_csv(fp)
    col = 'gpu_util_%' if device=='gpu' else 'cpu_%'
    if col not in df_u: continue
    avg = df_u[col].mean()
    plt.figure(figsize=(4,3))
    plt.bar([uc],[avg])
    plt.title(f"{model_titles[model]} {format_titles[fmt]} {device.upper()} {workers}W")
    plt.xlabel("Users"); plt.ylabel("Avg Util %")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Violin Plots of Latency Distributions (1W)
for model in sorted({m for m, *_ in stat_info}):
    for device in ['cpu','gpu']:
        fig, axes = plt.subplots(1,len(format_order), figsize=(14,4), sharey=True)
        fig.suptitle(f"{model_titles[model]} {device.upper()} Latency (1W)", fontsize=14)
        for j, fmt in enumerate(format_order):
            ax = axes[j]
            data, ucs = [], []
            for m,f,dev,w,fp in stat_info:
                if m==model and dev==device and f==fmt and w==1:
                    df = pd.read_csv(fp).dropna(subset=['50%'])
                    grp = df.groupby('User Count')['50%'].apply(list)
                    for uc, vals in grp.items():
                        data.append(vals); ucs.append(uc)
            if data:
                ax.violinplot(data, positions=ucs, showmeans=True)
            ax.set_title(format_titles[fmt])
            ax.set_xlabel("Users")
            if j==0: ax.set_ylabel("Latency (ms)")
            ax.grid(True)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()
