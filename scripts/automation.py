#insert initialization scripts - drive mount, git clone, pip installs
#insert clear command to remove prints

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
