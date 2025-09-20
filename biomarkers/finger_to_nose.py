import numpy as np
from scipy.signal import find_peaks

models = ["pose","hands"]

#TODO: --------------------check this functions ------------------

def extract_traj(coords_dict, landmark_name):
    """
    Extract trajectory for a single landmark across frames.
    Works with nested JSON that has 'pose', 'hands', etc.

    Parameters
    ----------
    coords_dict : dict
        Root dictionary from JSON. Expected structure:
            {
              "pose": {
                  "NOSE": {0: {"x":..,"y":..,"z":..,"v":..}, ...},
                  ...
              },
              "hands": {
                  "RIGHT_HAND.INDEX_FINGER_TIP": {0: {"x":..,"y":..,"z":..}, ...},
                  ...
              }
            }

    landmark_name : str
        Name of the landmark to extract (e.g., "NOSE" or "RIGHT_HAND.INDEX_FINGER_TIP").

    Returns
    -------
    np.ndarray of shape (T, 3)
        Trajectory (x,y,z) sorted by frame.

    Raises
    ------
    KeyError
        If the landmark is not found in any section.
    """
    # Search through all top-level sections (pose, hands, etc.)
    for _, section_dict in coords_dict.items():
        if landmark_name in section_dict:
            frames_dict = section_dict[landmark_name]
            frame_keys = sorted(frames_dict.keys(), key=lambda k: int(k))

            traj = np.array([
                [frames_dict[f]["x"], frames_dict[f]["y"], frames_dict[f]["z"]]
                for f in frame_keys
            ], dtype=float)

            return traj

    # If not found in any section
    raise KeyError(
        f"Landmark '{landmark_name}' not found. "
        f"Available sections: {list(coords_dict.keys())}. "
        f"Examples from 'pose': {list(coords_dict.get('pose', {}).keys())[:5]} "
        f"Examples from 'hands': {list(coords_dict.get('hands', {}).keys())[:5]}"
    )



def detect_movement_start(finger_traj, nose_point, window=5, threshold=-0.002, sustain=5):
    dists = np.linalg.norm(finger_traj - nose_point, axis=1)
    vel = np.diff(dists)
    smoothed = np.convolve(vel, np.ones(window)/window, mode="valid")

    count = 0
    for i, v in enumerate(smoothed):
        if v < threshold:
            count += 1
            if count >= sustain:
                return i
        else:
            count = 0
    return 0
  
  
#aims to calculate deviation between movement line and linear line nose-finger
def compute_deviation(finger_traj, nose_point):
    """
    Compute deviation of fingertip trajectory from straight line start→nose.
    Uses detected movement start as anchor.
    
    Returns
    -------
    deviation_mean, deviation_rms, path_ratio
    """
    start_idx = detect_movement_start(finger_traj, nose_point)
    traj = finger_traj[start_idx:]
    start_point = traj[0]

    line_vec = nose_point - start_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    sp_vecs = traj - start_point
    proj_lens = sp_vecs @ line_vec_norm
    proj_points = start_point + np.outer(proj_lens, line_vec_norm)
    deviations = np.linalg.norm(traj - proj_points, axis=1)

    deviation_mean = deviations.mean()
    deviation_rms = np.sqrt((deviations**2).mean())

    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    straight_len = np.linalg.norm(nose_point - start_point)
    path_ratio = path_len / straight_len if straight_len > 0 else np.nan

    return deviation_mean, deviation_rms, path_ratio
  

def deviation_biomarker(finger_traj, nose_traj):
  nose_point = nose_traj.mean(axis=0)
  deviation_mean, deviation_rms, path_ratio = compute_deviation(finger_traj=finger_traj, nose_point=nose_point)
  return {
        "deviation_mean": deviation_mean,
        "deviation_rms": deviation_rms,
        "path_ratio": path_ratio
    }
  

#-----------------------------------------------------------------------------

def calculate_accuracy(landmarks):
  # find frame of local minimums in z values
  # for each minimum, extract distance between nose and tip of index finger
    # take into account distance from camera
  # calculate overall biomarker for distance
  pass

#TODO: extract traj calculations
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



