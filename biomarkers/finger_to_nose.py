models = ["pose","hands"]

def calculate_accuracy(datapoints):
  # find frame of local minimums in z values
  # for each minimum, extract distance between nose and tip of index finger
    # take into account distance from camera
  # calculate overall biomarker for distance

def calculate_smoothness(datapoints):
  pass

def calculate_consistency(datapoints):
  pass

def extract_finger_to_nose_biomarkers(datapoints):
  biomarkers = {}
  biomarkers["accuracy"] = calculate_accuracy(datapoints)
  biomarkers["smoothness"] = calculate_smoothness(datapoints)
  biomarkers["consistency"] = calculate_consistency(datapoints)
  return biomarkers

