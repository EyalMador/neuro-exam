import os, json

def load_data_with_label(path):
  X, y = [], []

  for filename in os.listdir(path):
      if not filename.endswith(".json"):
          continue

      file_path = os.path.join(path, filename)
      with open(file_path, "r") as f:
          data = json.load(f)

      # Flatten biomarkers into a simple numeric feature list [mean1, std1, mean2, std2, ...]
      biomarkers = data.get("biomarkers", {})
      datapoint = []
      for name, stats in biomarkers.items():
          datapoint.extend([stats.get("mean", 0), stats.get("std", 0)])

      # Label: 1 for normal, 0 for abnormal
      text_label = data.get("label", "").lower()
      label = 1 if text_label == "normal" else 0

      X.append(datapoint)
      y.append(label)
  print(f"Loaded data and labels: {X}, {y}")
  return X, y

def load_data_no_label(path):
    X = []
    # Find the only file in the folder
    filename = next(f for f in os.listdir(path) if f.endswith(".json"))
    file_path = os.path.join(path, filename)

    with open(file_path, "r") as f:
        data = json.load(f)

    biomarkers = data.get("biomarkers", {})
    datapoint = [v.get(k, 0) for v in biomarkers.values() for k in ("mean", "std")]

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
