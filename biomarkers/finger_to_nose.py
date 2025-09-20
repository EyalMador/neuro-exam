models = ["pose","hands"]

def calculate_accuracy(datapoints):
  pass

def calculate_smoothness(datapoints):
  pass

def calculate_consistency(datapoints):
  pass

def produce_finger_to_nose_biomarkers(datapoints):
  biomarkers = {}
  biomarkers["accuracy"] = calculate_accuracy(datapoints)
  biomarkers["smoothness"] = calculate_smoothness(datapoints)
  biomarkers["consistency"] = calculate_consistency(datapoints)
  return biomarkers
