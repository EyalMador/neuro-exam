import os, json
import os
import json


def load_data_with_label(path):
    X, y = [], []

    # Collect all .json files
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in folder: {path}")

    for filename in json_files:
        file_path = os.path.join(path, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        biomarkers = data.get("biomarkers", {})
        datapoint = []

        for stats in biomarkers.values():
            if isinstance(stats, dict):
                datapoint.extend([stats.get("mean", 0), stats.get("std", 0)])
            elif isinstance(stats, (int, float)):
                datapoint.append(stats)
            # ignore unsupported types

        text_label = data.get("label", "").lower()
        label = 1 if text_label == "normal" else 0

        X.append(datapoint)
        y.append(label)

    if not X:
        raise ValueError(f"No valid biomarker data found in folder: {path}")

    print(f"Loaded {len(X)} samples with {len(X[0]) if X else 0} features each.")
    return X, y


def load_data_no_label(path, filename):
    X = []

    file_path = os.path.join(path, filename)

    with open(file_path, "r") as f:
        data = json.load(f)

    biomarkers = data.get("biomarkers", {})
    datapoint = []

    for stats in biomarkers.values():
        if isinstance(stats, dict):
            datapoint.extend([stats.get("mean", 0), stats.get("std", 0)])
        elif isinstance(stats, (int, float)):
            datapoint.append(stats)
        # ignore other types

    X.append(datapoint)
    print(f"Loaded '{filename}' with {len(datapoint)} features.")
    return X


def create_temp_folder(paths):
  for path in paths:
    os.makedirs(path, exist_ok=True)
  print("Temporary folders created.")

def cleanup_folder(path):
  os.system(f'rm -rf {path}')
  print("Temporary folder deleted.")
