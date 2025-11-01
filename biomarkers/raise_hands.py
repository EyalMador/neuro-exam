
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from biomarkers.helper import save_biomarkers_json




def normalize_pose_coords(pose):
    """
    Normalize RTM pose coordinates:
    - Shoulder midpoint per frame becomes x = 0 (since x is 'height')
    - Divide all x distances by torso height (shoulder - hip)
    Returns a new normalized pose dict (does not modify input).
    """

    left_shoulder, right_shoulder = pose["BODY_5"], pose["BODY_6"]
    left_hip, right_hip = pose.get("BODY_11"), pose.get("BODY_12")

    # --- torso height (along y axis) ---
    if left_hip and right_hip:
        shoulder_y = np.mean([d["y"] for d in left_shoulder.values()] +
                             [d["y"] for d in right_shoulder.values()])
        hip_y = np.mean([d["y"] for d in left_hip.values()] +
                        [d["y"] for d in right_hip.values()])
        S = abs(shoulder_y - hip_y) or 1.0
    else:
        # fallback to shoulder width if hips missing
        shoulder_width = abs(np.mean([d["x"] for d in left_shoulder.values()]) -
                             np.mean([d["x"] for d in right_shoulder.values()]))
        S = shoulder_width or 1.0

    # --- per-frame mid-shoulder y (height baseline) ---
    all_frames = sorted(set(left_shoulder.keys()) & set(right_shoulder.keys()), key=lambda k: int(k))
    mid_y = {f: 0.5 * (left_shoulder[f]["y"] + right_shoulder[f]["y"]) for f in all_frames}

    # --- normalize using y as vertical axis ---
    norm_pose = {}
    for name, lm in pose.items():
        norm_pose[name] = {}
        for f, v in lm.items():
            norm_pose[name][f] = dict(v)
            if f in mid_y:
                # normalized "height" → (shoulder midpoint - current y) / torso height
                norm_pose[name][f]["y"] = (mid_y[f] - v["y"]) / S
    return norm_pose


#TODO: dont know if i should add angle check in each event
def elbow_angle(shoulder, elbow, wrist):
    """Compute elbow angle in degrees.
    θ = arccos( (a · b) / (||a|| * ||b||) )"""
    shoulder_elbow = np.array([shoulder["x"] - elbow["x"], shoulder["y"] - elbow["y"], shoulder.get("z", 0) - elbow.get("z", 0)])
    wrist_elbow = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"], wrist.get("z", 0) - elbow.get("z", 0)])
    cosv = np.dot(shoulder_elbow, wrist_elbow) / (np.linalg.norm(shoulder_elbow) * np.linalg.norm(wrist_elbow) + 1e-9)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def smooth_signal(sig, window=5):
    if len(sig) < window:
        return sig
    kernel = np.ones(window) / window
    return np.convolve(sig, kernel, mode="same")


def detect_raise_events(left_wrist, right_wrist,
                        prominence=0.15,
                        width_rel=0.5,
                        smooth_win=7,
                        merge_gap=15):
    """
    Detect raise-hand events using valley detection.
    Logic:
    - Find valleys (minima) in each hand separately (low x = raised)
    - Get valley boundaries (start/end of each raise)
    - Merge overlapping or close valleys from both hands
    """
    frames = sorted(set(left_wrist.keys()) & set(right_wrist.keys()), key=int)
    if not frames:
        return {}
    
    # Extract and smooth y positions
    ly = smooth_signal(np.array([left_wrist[f]["y"] for f in frames]), smooth_win)
    ry = smooth_signal(np.array([right_wrist[f]["y"] for f in frames]), smooth_win)
    
    # Find valleys (minima) for each hand by inverting the signal
    left_events = find_hand_valleys(ly, prominence, width_rel)
    right_events = find_hand_valleys(ry, prominence, width_rel)
    
    print(f"right events: {right_events}")
    print(f"left events: {left_events}")
    
    # Combine all events from both hands
    all_events = []
    for start, end in left_events:
        all_events.append((start, end, 'left'))
    for start, end in right_events:
        all_events.append((start, end, 'right'))
    
    if not all_events:
        return {}
    
    # Sort by start time
    all_events.sort(key=lambda x: x[0])
    
    # Merge overlapping or close events
    merged_events = []
    current_start, current_end, _ = all_events[0]
    
    for start, end, hand in all_events[1:]:
        if start <= current_end + merge_gap:
            # Overlapping or close - merge by extending end
            current_end = max(current_end, end)
        else:
            # No overlap - save current and start new
            merged_events.append((current_start, current_end))
            current_start = start
            current_end = end
    
    # Add final event
    merged_events.append((current_start, current_end))
    
    # Convert to your dictionary format
    events = {}
    for i, (start, end) in enumerate(merged_events):
        events[i] = [int(f) for f in frames[start:end+1]]
    
    return events


def find_hand_valleys(signal, prominence=0.15, width_rel=0.03):
    """
    Find valleys (minima) of `signal` and return inclusive (start, end) indices.
    - `prominence`: fraction of signal range used as min prominence
    - `width_rel`: minimum valley width as a fraction of series length
    """
    y = np.asarray(signal, dtype=float)
    n = len(y)
    if n == 0:
        return []

    inv = -y  # valleys -> peaks
    prom = (y.max() - y.min()) * float(prominence)
    min_width = max(1, int(n * float(width_rel)))  # width in samples

    # detect valleys as peaks in the inverted signal
    peaks, props = find_peaks(inv, prominence=prom, width=min_width)
    if peaks.size == 0:
        return []

    # boundaries where the inverted peak meets its local baseline
    _, _, left_ips, right_ips = peak_widths(inv, peaks, rel_height=1.0)

    # build inclusive intervals
    intervals = []
    for l, r in zip(left_ips, right_ips):
        s = max(0, int(np.floor(l)))
        e = min(n - 1, int(np.ceil(r)))
        if s <= e:
            intervals.append((s, e))

    # merge overlaps/adjacent
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def symmetry_biomarker(events, left_wrist, right_wrist):
    """
    Compute symmetry for normalized pose data (no scaling needed).

    For each event:
        mean absolute difference between left & right wrist heights.

    Returns
    -------
    dict[event_id] = symmetry_score (float)
    """

    results = {}
    for eid, ev in events.items():
        frames = ev
        if not frames:
            continue

        diffs = []
        for f in frames:
            if f in left_wrist and f in right_wrist:
                lh = left_wrist[f]["y"]   
                rh = right_wrist[f]["y"]
                diffs.append(abs(lh - rh))

        if diffs:
            results[eid] = float(np.mean(diffs))

    return results


def speed_biomarker(events, left_wrist, right_wrist, fps=60):
    """
    Compute speed of hand raise for normalized RTM pose data.

    For each event:
        speed = (max_height - start_height) / duration

    Since:
        - `events` is dict[event_id] = [list of frames]
        - Pose is normalized (x = vertical)
        - No 'hand' info is needed
    """

    results = {}
    for eid, frames in events.items():
        if len(frames) < 2:
            continue

        # Combine left and right wrist heights (average)
        heights = []
        for f in frames:
            if f in left_wrist and f in right_wrist:
                lh = left_wrist[f]["y"]
                rh = right_wrist[f]["y"]
                heights.append((lh + rh) / 2)

        if not heights:
            continue

        start_h = heights[0]
        end_h = min(heights)
        duration = len(frames) / fps  # seconds

        results[eid] = abs(end_h - start_h) / duration if duration > 0 else 0

    return results


def consistency_biomarker(symmetry_scores):
    """1 - normalized std of symmetry scores → 1 = perfect consistency."""
    vals = list(symmetry_scores.values())
    if len(vals) < 2:
        return 1.0
    mean_sym = np.mean(vals)
    std_sym = np.std(vals)
    return float(1.0 - std_sym / (abs(mean_sym) + 1e-7))


def plot_wrist_time_series(frames, left_wrist, right_wrist,
                           title="Wrist trajectories over time", fps=60):
    """
    Plot left/right wrist X and Y coordinates over time (x-axis = seconds).

    Parameters
    ----------
    frames : list[int]
        Sorted frame indices.
    left_wrist, right_wrist : dict[int] = {"x": float, "y": float}
        Wrist coordinates per frame.
    title : str
        Title of the figure.
    fps : int, optional
        Frames per second of the video (default=60).
    """

    # Convert frames → time in seconds
    time = [int(f) / fps for f in frames]

    left_x = [left_wrist[f]["x"] for f in frames]
    left_y = [left_wrist[f]["y"] for f in frames]
    right_x = [right_wrist[f]["x"] for f in frames]
    right_y = [right_wrist[f]["y"] for f in frames]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ---- X positions (height for RTM model) ----
    axes[0].plot(time, left_x, label="Left Wrist X", linewidth=2)
    axes[0].plot(time, right_x, label="Right Wrist X", linewidth=2)
    axes[0].set_ylabel("X position (height)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # ---- Y positions (horizontal motion) ----
    axes[1].plot(time, left_y, label="Left Wrist Y", linewidth=2)
    axes[1].plot(time, right_y, label="Right Wrist Y", linewidth=2)
    axes[1].invert_yaxis()  # optional for normalized coordinates
    axes[1].set_ylabel("Y position")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def all_round_symmetry(left_wrist, right_wrist):
    """
    Global symmetry across the entire test (normalized RTM pose).

    Calculates the mean absolute difference between left and right wrist heights (x axis).
    Assumes input coordinates are already normalized, so no scaling or mid-shoulder adjustment is needed.
    """

    frames = sorted(set(left_wrist.keys()) & set(right_wrist.keys()), key=lambda k: int(k))
    if not frames:
        return None

    diffs = []
    for f in frames:
        if f in left_wrist and f in right_wrist:
            lh = left_wrist[f]["y"]
            rh = right_wrist[f]["y"]
            diffs.append(abs(lh - rh))

    return float(np.mean(diffs)) if diffs else None


def extract_raise_hands_biomarkers(coords, output_dir, filename, fps=60):
    pose_norm = normalize_pose_coords(coords["pose"])
    coords = {"pose": pose_norm}
    
    for name, lm in coords["pose"].items():
        coords["pose"][name] = {int(k): v for k, v in lm.items()}

    
    left_wrist, right_wrist       = coords["pose"]["BODY_9"], coords["pose"]["BODY_10"]
    #left_elbow, right_elbow       = coords["pose"]["BODY_7"], coords["pose"]["BODY_8"]
    #left_shoulder, right_shoulder = coords["pose"]["BODY_5"], coords["pose"]["BODY_6"]
    
    
    frames = sorted(set(left_wrist.keys()) & set(right_wrist.keys()), key=int)
    plot_wrist_time_series(frames, left_wrist, right_wrist, title="Wrist X/Y trajectories (for peak detection)")

    res = {}
    events = detect_raise_events(left_wrist=left_wrist, right_wrist=right_wrist)
    print(len(events))
    print(events)
    
    combined_symmetry = all_round_symmetry(left_wrist, right_wrist)
    res["all_round_symmetry"] = combined_symmetry
    
    symmetry = symmetry_biomarker(events, left_wrist, right_wrist)
    res["symmetry_STD"] = np.std(list(symmetry.values()))
    res["symmetry_mean"] = np.mean(list(symmetry.values()))
    
    speed = speed_biomarker(events, left_wrist, right_wrist, fps=fps)
    res["speed_STD"] = np.std(list(speed.values()))
    res["speed_mean"] = np.mean(list(speed.values()))
    
    consistency = consistency_biomarker(symmetry)
    res["symmetry_consistency"] = consistency

    save_biomarkers_json(res, output_dir, filename)

    print(res)
    return res
