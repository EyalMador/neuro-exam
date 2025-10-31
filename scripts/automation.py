from scripts.utils import *
import os
from scripts.run_biomarkers import run_biomarkers_with_args
from scripts.run_landmarks import run_extraction_with_args
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np
import json
import datetime
import shutil

MODELS_PATH = "/content/drive/MyDrive/neuro-exam/Models"
DATA_PATH = '/content/drive/MyDrive/neuro-exam/Data'
CLASSIFY_PATH = '/content/drive/MyDrive/neuro-exam/Data/Classify'
WORKING_FOLDER_PATH = '/content/drive/MyDrive/neuro-exam/Run_Files'
LANDMARKS_FOLDER_PATH = ''
BIOMARKERS_FOLDER_PATH = ''

def copy_file(source_dir, dest_dir, filename):
  source_path = source_dir + f'/{filename}'
  dest_path = dest_dir + f'/{filename}'
  shutil.copy2(source_path, dest_path)

def get_id():
  now = datetime.datetime.now()
  return now.strftime("%Y_%m_%d_%H_%M_%S")
  
def set_paths(id):
  global LANDMARKS_FOLDER_PATH
  global BIOMARKERS_FOLDER_PATH 
  LANDMARKS_FOLDER_PATH = WORKING_FOLDER_PATH + f'/{id}' + '/Landmarks'
  BIOMARKERS_FOLDER_PATH = WORKING_FOLDER_PATH + f'/{id}' + '/Biomarkers'
  
def extract_landmarks(test_type, is_test, video_name=None):
  from scripts.run_landmarks import run_landmarks_batch
  print("Extracting landmarks...")
  
  #Classify:
  if video_name is not None and not is_test:
    video_path = CLASSIFY_PATH + '/' + video_name
    run_extraction_with_args(video_path, LANDMARKS_FOLDER_PATH, 'rtmlib', 'body26', video_name, LANDMARKS_FOLDER_PATH)
    return

  video_dir_path = DATA_PATH + f'/{test_type}'
  if is_test:
    video_dir_path += '/test'
  else:
    video_dir_path += '/train'

  file_list = os.listdir(video_dir_path)
  for filename in file_list:
    filename_json = filename.split('.')[0] + '.json'
    if filename_json not in file_list:
      video_path = video_dir_path + f'/{filename}'
      run_extraction_with_args(video_path, video_dir_path, 'rtmlib', 'body26', filename, LANDMARKS_FOLDER_PATH)
    else:
      print(f"{filename} landmarks already extracted, skipping...")
    copy_file(video_dir_path, LANDMARKS_FOLDER_PATH, filename_json)      

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

def predict_results(chosen_model):
  results = {}
  model = load(f"{MODELS_PATH}/{chosen_model}")
  for filename in os.listdir(BIOMARKERS_FOLDER_PATH):
    data = load_data_no_label(BIOMARKERS_FOLDER_PATH, filename)
    results[filename] = model.predict(data)[0]
  return results
    
def classify_video(test_type, video_name):
  print("Starting classification process...")
  id = get_id()
  set_paths(id)
  try:
    create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
    extract_landmarks(test_type, False, video_name)
    calculate_biomarkers(test_type)
    results = predict_results(test_type)
    result = list(results.values())[0]
    if result == 1:
      print("Test is normal!")
    else:
      print("Test abnormal! Call a doctor.")
  print("Classification process finished successfully.")
  except Exception as e:
    print(e)
  #cleanup_folder(WORKING_FOLDER_PATH)

def train(test_type):
  print("Starting training process...")
  id = get_id()
  set_paths(id)
  try:
    create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
    extract_landmarks(test_type, is_test=False)
    calculate_biomarkers(test_type)
    train_model(test_type)
    print("Training process finished successfully.")
  except Exception as e:
    print(e)
  #cleanup_folder(WORKING_FOLDER_PATH)

def test(test_type):
  test_count = 0
  correct_test_count = 0
  print("Starting testing process...")
  id = get_id()
  set_paths(id)
  try:
    create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
    extract_landmarks(test_type, True)
    calculate_biomarkers(test_type)
    results = predict_results(test_type)
    for filename in os.listdir(f"{DATA_PATH}/{test_type}/test"):
      true_label = 0 if "abnormal" in filename else 1
      if results[filename] == true_label:
        correct_test_count += 1
        print(f"Model predicted correctly! For file {filename} model: {results[filename]} truth: {true_label}")
      else:
        print(f"Model predicted falsely! For file {filename} model: {results[filename]} truth: {true_label}")
      test_count += 1
      #cleanup_folder(WORKING_FOLDER_PATH)
    print("Testing process finished successfully.")
    accuracy = round((correct_test_count / test_count) * 100, 2)
    print(f"Successful prediction in {correct_test_count}/{test_count} tests. Accuracy: {accuracy}%")
  except Exception as e:
    print(e)
  #cleanup_folder(WORKING_FOLDER_PATH)
