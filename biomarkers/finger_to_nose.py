models = ["pose","hands"]

def calculate_accuracy(landmarks):
  # find frame of local minimums in z values
  # for each minimum, extract distance between nose and tip of index finger
    # take into account distance from camera
  # calculate overall biomarker for distance

def calculate_smoothness(landmarks):
  pass

def calculate_consistency(landmarks):
  pass

def extract_finger_to_nose_biomarkers(landmarks):
  biomarkers = {}
  biomarkers["accuracy"] = calculate_accuracy(landmarks)
  biomarkers["smoothness"] = calculate_smoothness(landmarks)
  biomarkers["consistency"] = calculate_consistency(landmarks)
  return biomarkers


