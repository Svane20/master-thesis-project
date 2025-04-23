import re, pathlib, pandas as pd, matplotlib.pyplot as plt

root = pathlib.Path.cwd()
reports = next(p for p in root.rglob('performance') if any(p.rglob('*_stats_history.csv')))
outdir = root / 'charts' / 'performance'
outdir.mkdir(exist_ok=True)

tag_re = re.compile(r'^(?P<model>\w+)_(?P<fmt>\w+)_(?P<hw>cpu|gpu)_(?P<workers>\d+)$')
records = []


def first_present(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found, columns={list(cols)}")


for hist in reports.rglob('*_stats_history.csv'):
    tag = hist.stem.replace('_stats_history', '')
    m = tag_re.match(tag)
    if not m:
        continue
    meta = m.groupdict()

    df = pd.read_csv(hist)

    df = df.apply(pd.to_numeric, errors='ignore')

    users_col = first_present(df.columns, 'User Count', 'Users')
    rps_col = first_present(df.columns,
                            'Total RPS', 'Total Requests/s',
                            'Requests/s', 'RPS')
    med_col = first_present(df.columns, '50%', 'Median Response Time')
    p95_col = first_present(df.columns, '95%', '95%ile', 'P95', 'Total 95%')

    plat = (df[df[users_col] > 0]
            .dropna(subset=[med_col, p95_col], how='any')
            .groupby(users_col, as_index=False)
            .last())

    sys_path = hist.with_name(tag + '_sys.csv')
    if sys_path.exists():
        sys_df = pd.read_csv(sys_path)
        cpu_mean = sys_df['cpu_%'].tail(30).mean()
        gpu_mean = sys_df['gpu_util_%'].tail(30).mean()
    else:
        cpu_mean = gpu_mean = None

    for _, row in plat.iterrows():
        total_req = row.get('Total Request Count', 0)
        total_fail = row.get('Total Failure Count', 0)
        records.append({
            **meta,
            'users': int(row[users_col]),
            'rps': row[rps_col],
            'median': row[med_col],
            'p95': row[p95_col],
            'failrate': (total_fail / total_req * 100) if total_req else 0,
            'cpu': cpu_mean,
            'gpu': gpu_mean,
        })

df = pd.DataFrame(records)


def scatter(col, ylabel, fname, logy=False):
    plt.figure()
    for (model, fmt, hw), grp in df.groupby(['model', 'fmt', 'hw']):
        if grp[col].isna().all():
            continue
        plt.plot(grp['users'], grp[col], marker='o',
                 label=f'{model}-{fmt}-{hw}')
    plt.xlabel('Concurrent users (plateau)')
    plt.ylabel(ylabel)
    if logy:
        plt.yscale('log')
    if plt.gca().get_legend_handles_labels()[1]:
        plt.legend(fontsize='small')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=150)
    plt.close()


scatter('rps', 'Throughput (req/s)', 'throughput_vs_users.png')
scatter('median', 'Median latency (ms)', 'median_lat_vs_users.png')
scatter('p95', 'P95 latency (ms)', 'p95_lat_vs_users.png')
scatter('failrate', 'Failure rate (%)', 'failrate_vs_users.png')

print("All plots saved to", outdir)
