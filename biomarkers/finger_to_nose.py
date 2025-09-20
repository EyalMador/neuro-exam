import numpy as np
from scipy.signal import find_peaks

models = ["pose","hands"]

def calculate_accuracy(landmarks):
  # find frame of local minimums in z values
  # for each minimum, extract distance between nose and tip of index finger
    # take into account distance from camera
  # calculate overall biomarker for distance

def calculate_smoothness(data, fps=30,
                              finger_key=("hands", "index_finger_tip"),
                              nose_key=("pose", "nose")):
    """
    Calculate overall smoothness of finger-to-nose test with possibly multiple attempts.
    
    data: nested dict model/bodypart/frame -> {"x":..,"y":..,"z":..}
    fps: frames per second
    finger_key: tuple (model, landmark) for finger tip
    nose_key: tuple (model, landmark) for nose
    
    Returns: overall smoothness score (0..1, higher = smoother),
             plus list of per-attempt smoothness values.
    """
    def get_traj(model, landmark):
        lm = data[model][landmark]
        frames = sorted(lm.keys(), key=int)
        coords = np.array([[lm[f]["x"], lm[f]["y"], lm[f]["z"]] for f in frames], dtype=float)
        return coords

    # 1. Extract trajectories
    finger = get_traj(*finger_key)
    nose = get_traj(*nose_key)
    T = min(len(finger), len(nose))
    finger, nose = finger[:T], nose[:T]

    # 2. Distance signal
    dist = np.linalg.norm(finger - nose, axis=1)

    # 3. Find local minima = touches (attempt endpoints)
    inv = -dist
    peaks, _ = find_peaks(inv, prominence=np.std(dist)/2, distance=fps//2)  
    # distance=fps//2 enforces min ~0.5s between touches

    # 4. Segment attempts around each minimum
    attempt_scores = []
    dt = 1.0 / fps

    for p in peaks:
        # define a small window around the attempt (0.5s before, 0.5s after)
        start = max(0, p - int(0.5*fps))
        end   = min(T, p + int(0.5*fps))
        seg = dist[start:end]
        if len(seg) < 4:
            continue

        # jerk of distance
        jerk = (seg[3:] - 3*seg[2:-1] + 3*seg[1:-2] - seg[:-3]) / (dt**3)
        msj = np.mean(jerk**2)
        smoothness = 1.0 / (1.0 + msj)
        attempt_scores.append(smoothness)

    # 5. Aggregate: overall = median of attempts
    if len(attempt_scores) == 0:
        return None, []

    overall = float(np.median(attempt_scores))
    return overall, attempt_scores

def calculate_consistency(landmarks):
  pass

def extract_finger_to_nose_biomarkers(landmarks):
  biomarkers = {}
  biomarkers["accuracy"] = calculate_accuracy(landmarks)
  biomarkers["smoothness"] = calculate_smoothness(landmarks)
  biomarkers["consistency"] = calculate_consistency(landmarks)
  return biomarkers



