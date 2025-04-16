def write_metrics_to_file(metrics, file_path):
    """
    Write the metrics to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)