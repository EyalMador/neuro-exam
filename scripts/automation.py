import os

DATA_PATH = '/content/drive/MyDrive/neuro-exam/Data/RawVideos/train'
WORKING_FOLDER_PATH = '/content/drive/MyDrive/neuro-exam/temp_script_folder'
LANDMARKS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Landmarks'
BIOMARKERS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Biomarkers'

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

    with open(path, "r") as f:
        data = json.load(f)

    biomarkers = data.get("biomarkers", {})
    datapoint = []

    for name, stats in biomarkers.items():
        datapoint.extend([stats.get("mean", 0), stats.get("std", 0)])

    X.append(datapoint)

    print(f"Loaded 1 sample with {len(datapoint)} features.")
    return X

def train_model(chosen_model):
  print(f"Starting to train model: {chosen_model}")
  data, labels = load_data_with_label(f"{DATA_PATH}/{chosen_model}/training")
  model = SVC(kernel='rbf', probability=True)
  model.fit(data, labels)
  dump(model, f"{MODELS_PATH}/{chosen_model}")
  print(f"Successfully trained and saved model: {chosen_model}")

def predict_result(chosen_model, filename):
  model = load(f"{MODELS_PATH}/{chosen_model}")
  data = load_data_no_label(f"{DATA_PATH}/{chosen_model}/evaluation/{filename}.json")
  if model.predict(data) == 1:
    print("Test is normal!")
  else:
    print("Test abnormal! Call a doctor.")
    
def classify_video(test_type, video_name):
  print("Starting classification process.")
  create_temp_folder()
  extract_landmarks(test_type, video_name)
  calculate_biomarkers(test_type)
  predict_result(test_type)
  cleanup_folder()
  print("Classification process finished.")

def train_model(test_type):
  print("Starting training process.")
  create_temp_folder()
  extract_landmarks(test_type)
  calculate_biomarkers(test_type)
  train_svm(test_type)
  cleanup_folder()
  print("Training process finished.")
