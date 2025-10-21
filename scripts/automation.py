from scripts.utils import *
import os

DATA_PATH = '/content/drive/MyDrive/neuro-exam/Data/RawVideos/train'
WORKING_FOLDER_PATH = '/content/drive/MyDrive/neuro-exam/temp_script_folder'
LANDMARKS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Landmarks'
BIOMARKERS_FOLDER_PATH = WORKING_FOLDER_PATH + '/Biomarkers'

def extract_landmarks(test_type, video_name=None):
  from scripts.run_landmarks import run_landmarks_batch
  print("Extracting landmarks...")
  
  #Extract from single video:
  if video_name is not None:
    #insert run_landmarks for single video
  
  #Extract from all videos in folder:
  else:
    run_landmarks_batch(
        input_dir=f"{DATA_PATH}/{test_type}",
        output_dir=LANDMARKS_FOLDER_PATH,
        lib="rtmlib",
        model_type="body26",
        export_video=False,
        export_json=True,
        export_csv=False
    )

def calculate_biomarkers(test_type):
  print("Calculating biomarkers...")
  if video_name is None:
    #insert run_biomarkers for all landmark files in landmark folder

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
  create_temp_folder([LANDMARKS_FOLDER_PATH,BIOMARKERS_FOLDER_PATH])
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
  cleanup_folder(WORKING_FOLDER_PATH)
  print("Training process finished.")
