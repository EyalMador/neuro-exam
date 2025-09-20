import sys, os
print("CWD:", os.getcwd())
print("sys.path:", sys.path)
print("Dir in CWD:", os.listdir(os.getcwd()))
from biomarkers.finger_to_nose import deviation_biomarker, extract_traj
import json


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
    
if __name__ == "__main__":
    coords = load_json("data/output/Video_1/Landmarks_output/Video_1_out_FTNIDO2.json")
    finger, nose = extract_traj(coords, "LEFT_HAND.INDEX_FINGER_TIP"), extract_traj(coords, "NOSE")
    print(deviation_biomarker(finger_traj=finger, nose_traj=nose))