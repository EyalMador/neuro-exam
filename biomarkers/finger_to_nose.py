import numpy as np
from biomarkers.raise_hands import detect_raise_events
from scipy.signal import detrend, welch


# ---------------helpers--------------------
def calculate_distance_dict(finger_coords, nose_coords):

    frames = sorted(set(finger_coords.keys()) & set(nose_coords.keys()), key=int)
    distances = {}

    for f in frames:
        fx, fy = finger_coords[f]["x"], finger_coords[f]["y"]
        nx, ny = nose_coords[f]["x"], nose_coords[f]["y"]
        distances[f] = np.sqrt((fx - nx)**2 + (fy - ny)**2)

    return distances


def negate_dict(d):
    """
    Negate the values of a numeric dictionary.
    """
    return {k: -v for k, v in d.items()}


def calculate_angle_dict(shoulder, elbow, wrist):
    
    frames = sorted(set(shoulder.keys()) & set(elbow.keys()) & set(wrist.keys()), key=int)
    angles = {}

    for f in frames:
        s = np.array([shoulder[f]["x"], shoulder[f]["y"]])
        e = np.array([elbow[f]["x"], elbow[f]["y"]])
        w = np.array([wrist[f]["x"], wrist[f]["y"]])

        # Compute vectors
        se = s - e
        we = w - e

        # Compute cosine of angle using dot product formula
        dot = np.dot(se, we)
        norm_prod = np.linalg.norm(se) * np.linalg.norm(we)
        if norm_prod == 0:
            continue  # skip invalid frame

        cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        angles[f] = angle

    return angles


def filter_by_angle_threshold(dist_dict, angle_dict, angle_thresh=140):

    common_frames = sorted(set(dist_dict.keys()) & set(angle_dict.keys()), key=int)
    filtered = {
        f: dist_dict[f]
        for f in common_frames
        if angle_dict[f] < angle_thresh
    }
    return filtered


def compute_tremor(tip_coords, fps, tremor_band=(4, 12)):

    frames = sorted(tip_coords.keys(), key=int)
    if len(frames) < fps:  # need at least 1 second of data
        return np.nan

    x = np.array([tip_coords[f]['x'] for f in frames])
    y = np.array([tip_coords[f]['y'] for f in frames])
    signal = np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)
    signal = detrend(signal)

    # Compute power spectral density
    freqs, psd = welch(signal, fs=fps, nperseg=min(256, len(signal)))

    # Integrate PSD over the tremor frequency range (4–12 Hz)
    mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])
    tremor_power = np.trapz(psd[mask], freqs[mask])
    return float(tremor_power)


def compute_finger_to_nose_biomarkers(left_events, right_events,
                                      left_dist, right_dist,
                                      left_tip, right_tip, fps):
    """
    Compute key biomarkers for the Finger-to-Nose test:
    1. Mean event distance per hand
    2. Left-right symmetry
    3. Tremor per hand
    4. Tremor symmetry
    """

    def event_mean_distance(events, dist_dict):
        if not events:
            return np.nan
        event_means = []
        for e in events.values():
            frames = e['frames']
            d = [dist_dict[f] for f in frames if f in dist_dict]
            if len(d) > 0:
                event_means.append(np.mean(d))
        return np.mean(event_means) if event_means else np.nan

    # Mean distance per hand
    left_score = event_mean_distance(left_events, left_dist)
    right_score = event_mean_distance(right_events, right_dist)

    # Symmetry (0 = perfect, 1 = asymmetrical)
    symmetry = abs(left_score - right_score) / max(left_score, right_score) if (left_score and right_score) else np.nan

    # Tremor (per-hand)
    left_tremor = compute_tremor(left_tip, fps)
    right_tremor = compute_tremor(right_tip, fps)

    tremor_symmetry = abs(left_tremor - right_tremor) / max(left_tremor, right_tremor) if (left_tremor and right_tremor) else np.nan

    return {
        "left_mean_dist": left_score,
        "right_mean_dist": right_score,
        "symmetry": symmetry,
        "left_tremor": left_tremor,
        "right_tremor": right_tremor,
        "tremor_symmetry": tremor_symmetry
    }



def extract_finger_to_nose_biomarkers(coords, output_dir, filename, fps=60):
    """
    Detect finger-to-nose events and compute biomarkers.
    """

    #landmarks
    pose = coords["pose"]
    hands = coords["hands"]

    left_shoulder = pose["LEFT_SHOULDER"]
    left_elbow = pose["LEFT_ELBOW"]
    left_wrist = pose["LEFT_WRIST"]

    right_shoulder = pose["RIGHT_SHOULDER"]
    right_elbow = pose["RIGHT_ELBOW"]
    right_wrist = pose["RIGHT_WRIST"]

    nose = pose["NOSE"]
    left_finger = hands["LEFT_TIP"]
    right_finger = hands["RIGHT_TIP"]

    
    left_dist = calculate_distance_dict(left_finger, nose)
    right_dist = calculate_distance_dict(right_finger, nose)

    left_angle = calculate_angle_dict(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle_dict(right_shoulder, right_elbow, right_wrist)

    filtered_left = filter_by_angle_threshold(left_dist, left_angle, angle_thresh=100)
    filtered_right = filter_by_angle_threshold(right_dist, right_angle, angle_thresh=100)

    #  Negate
    neg_left = negate_dict(filtered_left)
    neg_right = negate_dict(filtered_right)

    #Event detection per hand
    left_events = detect_raise_events(neg_left, neg_left)
    right_events = detect_raise_events(neg_right, neg_right)

    #Compute biomarkers
    res = compute_finger_to_nose_biomarkers(
        left_events, right_events,
        left_dist, right_dist,
        left_finger, right_finger, fps
    )

    print("Finger-to-Nose Biomarkers:")
    print(res)
    return res
