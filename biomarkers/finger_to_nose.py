import numpy as np
from scipy.signal import detrend, welch, butter, filtfilt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from biomarkers.helper import save_biomarkers_json


# ---------------helpers--------------------
def get_shoulder_width(left_shoulder, right_shoulder):
    """Calculate median shoulder width for normalization"""
    frames = sorted(set(left_shoulder.keys()) & set(right_shoulder.keys()))
    widths = []
    for f in frames:
        if f in left_shoulder and f in right_shoulder:
            ls = np.array([left_shoulder[f]["x"], left_shoulder[f]["y"]])
            rs = np.array([right_shoulder[f]["x"], right_shoulder[f]["y"]])
            widths.append(np.linalg.norm(ls - rs))
    return np.median(widths) if widths else 1.0


def normalize_distances(dist_dict, reference_length):
    """Normalize distances by reference length"""
    return {f: d / reference_length for f, d in dist_dict.items()}



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

def compute_tremor(tip_coords, fps, tremor_band=(4, 12), high_pass_cutoff=3.0):
    """
    Compute tremor on the ENTIRE motion sequence.
    Returns: tremor_ratio, tremor_amplitude, peak_frequency
    """
    frames = sorted(tip_coords.keys(), key=int)
    if len(frames) < fps:
        return np.nan, np.nan, np.nan

    x = np.array([tip_coords[f]['x'] for f in frames])
    y = np.array([tip_coords[f]['y'] for f in frames])

    vx = np.diff(x)
    vy = np.diff(y)
    speed = np.sqrt(vx**2 + vy**2)

    # High-pass filter to remove main movement and isolate tremor
    if len(speed) > 20:
        nyquist = fps / 2
        cutoff_norm = high_pass_cutoff / nyquist
        b, a = butter(2, cutoff_norm, btype='high')
        speed_filtered = filtfilt(b, a, speed)
    else:
        speed_filtered = detrend(speed)

    freqs, psd = welch(speed_filtered, fs=fps, nperseg=min(256, len(speed_filtered)))

    mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])
    tremor_power = np.trapz(psd[mask], freqs[mask])
    
    total_mask = (freqs >= 0.5) & (freqs <= 20)
    total_power = np.trapz(psd[total_mask], freqs[total_mask])

    tremor_ratio = tremor_power / total_power if total_power > 0 else 0
    tremor_amplitude = np.std(speed_filtered)
    peak_tremor_freq = freqs[mask][np.argmax(psd[mask])] if tremor_power > 0 else np.nan
    
    return float(tremor_ratio), float(tremor_amplitude), float(peak_tremor_freq)




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



def per_event_minima(events, dist_dict):
    minima = []
    for frames in events.values():
        vals = [dist_dict[f] for f in frames if f in dist_dict]
        if vals:
            minima.append(min(vals))
    return minima


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
    5. smoothness per hand
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
    left_min_mean = np.mean(per_event_minima(left_events, left_dist)) 
    right_score = event_mean_distance(right_events, right_dist)
    right_min_mean = np.mean(per_event_minima(right_events, right_dist)) 


    # Symmetry (0 = perfect, 1 = asymmetrical)
    symmetry = abs(left_score - right_score) / max(left_score, right_score) if (left_score and right_score) else np.nan

    # Tremor (per-hand)
    left_tremor, left_tremor_amp, left_tremor_freq = compute_tremor(left_tip, fps)
    right_tremor, right_tremor_amp, right_tremor_freq = compute_tremor(right_tip, fps)

    tremor_symmetry = tremor_asymmetry_orders(left_tremor, right_tremor)
    

    return {
            #"left_mean_dist": left_score,
            #"right_mean_dist": right_score,
            "left_min_mean": left_min_mean,
            "right_min_mean": right_min_mean,
            "symmetry": symmetry,
            #"left_tremor": left_tremor,
            #"right_tremor": right_tremor,
            #"left_tremor_amplitude": left_tremor_amp,
            #"right_tremor_amplitude": right_tremor_amp,
            #"left_tremor_freq": left_tremor_freq,
            #"right_tremor_freq": right_tremor_freq,
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
    
    shoulder_width = get_shoulder_width(left_shoulder, right_shoulder)

    left_dist_raw = calculate_distance_dict(left_finger, nose)
    right_dist_raw = calculate_distance_dict(right_finger, nose)
    
    left_dist = normalize_distances(left_dist_raw, shoulder_width)
    right_dist = normalize_distances(right_dist_raw, shoulder_width)
    

    left_angle = calculate_angle_dict(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle_dict(right_shoulder, right_elbow, right_wrist)

    filtered_left = filter_by_angle_threshold(left_dist, left_angle, angle_thresh=60)
    filtered_right = filter_by_angle_threshold(right_dist, right_angle, angle_thresh=60)

    left_events = detect_finger_to_nose_events(filtered_left,
                                            prominence=0.03,
                                            width_rel=0.5,
                                            smooth_win=5,
                                            merge_gap=6)
    right_events = detect_finger_to_nose_events(filtered_right,
                                                prominence=0.03,
                                                width_rel=0.5,
                                                smooth_win=5,
                                                merge_gap=6)

    print(f"right len: {len(right_events)}, right:{right_events}")
    print(f"left len: {len(left_events)}, left:{left_events}")
    
    
    
    #check - angle dict plot--------
    frames_L = sorted(left_angle.keys())
    frames_R = sorted(right_angle.keys())

    plt.figure(figsize=(10, 4))
    plt.plot(frames_L, [left_angle[f] for f in frames_L], label="Left Elbow Angle", color='blue')
    plt.plot(frames_R, [right_angle[f] for f in frames_R], label="Right Elbow Angle", color='red', alpha=0.7)
    plt.axhline(140, color='gray', ls='--', lw=1)  # example threshold
    plt.title("Elbow Angle Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    #end of check

    
    
    plot_finger_to_nose_distances(filtered_left, filtered_right, left_events, right_events)

    res = compute_finger_to_nose_biomarkers(
        left_events, right_events,
        left_dist, right_dist,
        left_finger, right_finger, fps
    )

    print("Finger-to-Nose Biomarkers:")
    print(res)
    save_biomarkers_json(res, output_dir, filename)
    return res




