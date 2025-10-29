from scripts.utils import *
import os
from scripts.run_biomarkers import run_biomarkers_with_args
from scripts.run_landmarks import run_extraction_with_args
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np
import json

MODELS_PATH = "/content/drive/MyDrive/neuro-exam/Models"
DATA_PATH = '/content/drive/MyDrive/neuro-exam/Data'
CLASSIFY_PATH = '/content/drive/MyDrive/neuro-exam/Data/Classify'
WORKING_FOLDER_PATH = '/content/drive/MyDrive/neuro-exam/temp_script_folder'
LANDMARKS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Landmarks'
BIOMARKERS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Biomarkers'

def extract_landmarks(test_type, is_test, video_name=None):
  from scripts.run_landmarks import run_landmarks_batch
  print("Extracting landmarks...")
  
  #Classify:
  if video_name is not None and not is_test:
    video_path = CLASSIFY_PATH + '/' + video_name
    run_extraction_with_args(video_path, LANDMARKS_FOLDER_PATH, 'rtmlib', 'body26', video_name, WORKING_FOLDER_PATH)

  #Test:
  if video_name is not None and is_test:
    video_path = DATA_PATH + f'/{test_type}' + f'/{video_name}'
    run_extraction_with_args(video_path, LANDMARKS_FOLDER_PATH, 'rtmlib', 'body26', video_name, WORKING_FOLDER_PATH)
    
  #Train:
  if video_name is None:
    input_dir = f"{DATA_PATH}/{test_type}/test" if is_test else f"{DATA_PATH}/{test_type}/train"
    run_landmarks_batch(
        input_dir=input_dir,
        output_dir=LANDMARKS_FOLDER_PATH,
        lib="rtmlib",
        model_type="body26",
        export_video=False,
        export_json=True,
        export_csv=False
    )

def calculate_biomarkers(test_type):
  print("Calculating biomarkers...")
  for filename in os.listdir(LANDMARKS_FOLDER_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(LANDMARKS_FOLDER_PATH, filename)
            print(f"Loading landmarks from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            run_biomarkers_with_args(test_type, data, BIOMARKERS_FOLDER_PATH, filename)

def train_model(chosen_model):
  print(f"Starting to train model: {chosen_model}")
  data, labels = load_data_with_label(f"{BIOMARKERS_FOLDER_PATH}")
  model = SVC(kernel='rbf', probability=True)
  model.fit(data, labels)
  dump(model, f"{MODELS_PATH}/{chosen_model}")
  print(f"Successfully trained and saved model: {chosen_model}")

def predict_result(chosen_model, filename):
  model = load(f"{MODELS_PATH}/{chosen_model}")
  data = load_data_no_label(BIOMARKERS_FOLDER_PATH)
  return model.predict(data)
    
def classify_video(test_type, video_name):
  print("Starting classification process...")
  try:
    create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
    extract_landmarks(test_type, False, video_name)
    calculate_biomarkers(test_type)
    result = predict_result(test_type, video_name)
    if result == 1:
      print("Test is normal!")
    else:
      print("Test abnormal! Call a doctor.")
    print("Classification process finished successfully.")
  except Exception as e:
    print(e)
  cleanup_folder(WORKING_FOLDER_PATH)

def train(test_type):
  print("Starting training process...")
  try:
    create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
    extract_landmarks(test_type, is_test=False)
    calculate_biomarkers(test_type)
    train_model(test_type)
    print("Training process finished successfully.")
  except Exception as e:
    print(e)
  cleanup_folder(WORKING_FOLDER_PATH)

def test(test_type):
  test_count = 0
  correct_test_count = 0
  print("Starting testing process...")
  try:
    for filename in os.listdir(f"{DATA_PATH}/{test_type}/test"):
      create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
      extract_landmarks(test_type, is_test=True, filename)
      calculate_biomarkers(test_type)
      result = predict_result(test_type, filename)
      true_label = 0 if "abnormal" in filename else 1
      if result == true_label:
        correct_test_count += 1
      test_count += 1
      cleanup_folder(WORKING_FOLDER_PATH)
    print("Testing process finished successfully.")
    accuracy = round((correct_test_count / test_count) * 100, 2)
    print(f"Successful prediction in {correct_test_count}/{test_count} tests. Accuracy: {accuracy}%")
  except Exception as e:
    print(e)
  cleanup_folder(WORKING_FOLDER_PATH)
