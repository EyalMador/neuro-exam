import numpy as np
from scipy.signal import detrend, welch
import numpy as np
from scipy.signal import find_peaks, peak_widths


# ---------------helpers--------------------

def calculate_distance_dict(finger_coords, nose_coords):

    frames = sorted(set(finger_coords.keys()) & set(nose_coords.keys()), key=int)
    distances = {}

    for f in frames:
        fx, fy = finger_coords[f]["x"], finger_coords[f]["y"]
        nx, ny = nose_coords[f]["x"], nose_coords[f]["y"]
        distances[f] = np.sqrt((fx - nx)**2 + (fy - ny)**2)

    return distances


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

def tremor_asymmetry_orders(left_tremor, right_tremor, eps=1e-12):
    """
    כמה 'סדרי-גודל' מפרידים בין הידיים.
    0 = סימטרי, 1 = פי 10, 2 = פי 100, וכו'.
    """
    L = float(left_tremor) + eps
    R = float(right_tremor) + eps
    return abs(np.log10(L / R))

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



def detect_finger_to_nose_events(dist_dict,
                                 prominence=0.15,
                                 width_rel=0.6,
                                 smooth_win=7,
                                 merge_gap=15):


    def smooth_signal(sig, window):
        if len(sig) < window:
            return sig
        kernel = np.ones(window) / window
        return np.convolve(sig, kernel, mode="same")

    def find_valleys(signal, prominence=0.05, width_rel=0.4):
        """Find distinct valleys (minima) in 1D signal."""
        amplitude = signal.max() - signal.min()
        if amplitude == 0:
            return []

        prom = amplitude * prominence
        inverted = -signal  # valleys → peaks in inverted signal
        peaks, props = find_peaks(inverted, prominence=prom, width=1)

        # Measure valley width directly on inverted signal
        widths, _, left_ips, right_ips = peak_widths(
            inverted, peaks, rel_height=width_rel
        )

        valleys = []
        for i in range(len(peaks)):
            start = max(0, int(np.floor(left_ips[i])))
            end = min(len(signal) - 1, int(np.ceil(right_ips[i])))
            valleys.append((start, end))
        return valleys

    # ---- process single signal ----
    frames = sorted(dist_dict.keys(), key=int)
    if not frames:
        return {}

    y = np.array([dist_dict[f] for f in frames])
    y = smooth_signal(y, smooth_win)

    # Detect valleys
    events = find_valleys(y, prominence, width_rel)
    if not events:
        return {}

    # Merge overlapping / close valleys
    merged = []
    current_start, current_end = events[0]

    for start, end in events[1:]:
        if start <= current_end + merge_gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    result = {}
    for i, (start, end) in enumerate(merged):
        result[i] = [int(f) for f in frames[start:end + 1]]

    return result


import matplotlib.pyplot as plt

def plot_finger_to_nose_distances(left_dist, right_dist,
                                  left_events=None,
                                  right_events=None,
                                  title="Finger-to-Nose Distance Over Time"):
    """
    Plot left and right hand distances (to nose) over time, 
    with detected valleys (events) marked.

    Parameters
    ----------
    left_dist, right_dist : dict
        frame → distance
    left_events, right_events : dict or None
        detected events {i: [frames]} (optional)
    title : str
        plot title
    """

    # Sort frames for alignment
    left_frames = np.array(sorted(left_dist.keys(), key=int))
    right_frames = np.array(sorted(right_dist.keys(), key=int))
    left_vals = np.array([left_dist[f] for f in left_frames])
    right_vals = np.array([right_dist[f] for f in right_frames])

    plt.figure(figsize=(12, 5))
    plt.plot(left_frames, left_vals, label="Left Distance", color="blue", lw=2)
    plt.plot(right_frames, right_vals, label="Right Distance", color="red", lw=2, alpha=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Finger–Nose Distance")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Mark detected valleys
    def mark_events(events, color):
        if not events:
            return
        for frames in events.values():
            start, end = frames[0], frames[-1]
            plt.axvspan(start, end, color=color, alpha=0.2)

    mark_events(left_events, "blue")
    mark_events(right_events, "red")

    plt.legend()
    plt.tight_layout()
    plt.show()

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
        for frames in events.values():  # now frames is a list directly
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

    tremor_symmetry = tremor_asymmetry_orders(left_tremor, right_tremor)

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
    

    for name, lm in coords["pose"].items():
        coords["pose"][name] = {int(k): v for k, v in lm.items()}
        
    for name, lm in coords["hands"].items():
        coords["hands"][name] = {int(k): v for k, v in lm.items()}
        
    pose = coords["pose"]
    hands = coords["hands"]

    left_shoulder = pose["LEFT_SHOULDER"]
    left_elbow = pose["LEFT_ELBOW"]
    left_wrist = pose["LEFT_WRIST"]

    right_shoulder = pose["RIGHT_SHOULDER"]
    right_elbow = pose["RIGHT_ELBOW"]
    right_wrist = pose["RIGHT_WRIST"]

    nose = pose["NOSE"]
    left_finger = hands["LEFT_HAND.INDEX_FINGER_TIP"]
    right_finger = hands["RIGHT_HAND.INDEX_FINGER_TIP"]


    left_dist = calculate_distance_dict(left_finger, nose)
    right_dist = calculate_distance_dict(right_finger, nose)
    

    left_angle = calculate_angle_dict(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle_dict(right_shoulder, right_elbow, right_wrist)

    filtered_left = filter_by_angle_threshold(left_dist, left_angle, angle_thresh=0)
    filtered_right = filter_by_angle_threshold(right_dist, right_angle, angle_thresh=0)

    left_events = detect_finger_to_nose_events(filtered_left,
                                            prominence=0.06,
                                            width_rel=0.20,
                                            smooth_win=5,
                                            merge_gap=4)
    right_events = detect_finger_to_nose_events(filtered_right,
                                                prominence=0.06,
                                                width_rel=0.20,
                                                smooth_win=5,
                                                merge_gap=4)

    print(f"right len: {len(right_events)}, right:{right_events}")
    print(f"left len: {len(left_events)}, left:{left_events}")

    
    
    plot_finger_to_nose_distances(filtered_left, filtered_right, left_events, right_events)

    #Compute biomarkers
    res = compute_finger_to_nose_biomarkers(
        left_events, right_events,
        left_dist, right_dist,
        left_finger, right_finger, fps
    )

    print("Finger-to-Nose Biomarkers:")
    print(res)
    return res
