import cv2
import os
from media_pipe_wrapper import MPModel
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_video(path):
    if os.path.exists(path):
        return cv2.VideoCapture(path)
    else:
        raise FileNotFoundError(f"Video not found at {path}")
    
def save_json(video_coords, output_dir=".", output_name="landmarks.json"):
    """Save landmarks dictionary to JSON file."""
    filepath = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(video_coords, f, indent=2)
    print(f"Landmarks saved to {filepath}")
    return filepath

def save_csv(video_coords, output_dir=".", output_name="landmarks.csv"):
    """Save landmarks dictionary to CSV file."""
    rows = []
    for landmark, frames in video_coords.items():
        for frame, values in frames.items():
            rows.append({
                "frame": frame,
                "landmark": landmark,
                "x": values.get("x"),
                "y": values.get("y"),
                "z": values.get("z"),
                "v": values.get("v")
            })

    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Landmarks saved to {filepath}")
    return filepath


def plot_xyz(video_coords, landmark, fps, show):
    """
    Plot x, y, z over time for a single landmark from your coords dict.
    Assumes: video_coords[landmark][frame] = {'x','y','z','v'}.
    """
    if landmark not in video_coords:
        raise KeyError(f"Landmark '{landmark}' not found. Available: {list(video_coords.keys())[:10]}...")

    frames_dict = video_coords[landmark]

    # normalize & sort frame keys numerically when possible
    def _as_int(k):
        try:
            return int(k)
        except Exception:
            return k

    frame_keys = sorted(frames_dict.keys(), key=_as_int)
    frames = np.array([_as_int(k) for k in frame_keys], dtype=float)

    x = np.array([frames_dict[k].get("x", np.nan) for k in frame_keys], dtype=float)
    y = np.array([frames_dict[k].get("y", np.nan) for k in frame_keys], dtype=float)
    z = np.array([frames_dict[k].get("z", np.nan) for k in frame_keys], dtype=float)

    t = frames / fps if fps and fps > 0 else frames
    xlabel = "time (s)" if fps and fps > 0 else "frame"

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.plot(t, z, label="z")
    plt.title(f"{landmark} – x/y/z")
    plt.xlabel(xlabel)
    plt.ylabel("coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if show:
        plt.show()
    else:
        plt.close()



def get_landmarks(lib, m_type, source_video_path, output_video_dir, output_video_name):
    if lib == 'mediapipe':
        cap = get_video(source_video_path)
        model = MPModel(m_type)
        coords = model.processing(cap, save_video_output=True, output_video_dir=output_video_dir, output_video_name=output_video_name )
        return coords
    elif lib == 'openpose':
        raise NotImplementedError("OpenPose support not yet implemented")
    else:
        raise ValueError("Unsupported library")

if __name__ == "__main__":
    test_type = "FTNIDO2"
    model_type = "hands"
    video_num = 1
    source_video_path = f"../data/Raw_Videos/Video_{video_num}/{test_type}.mov"
    
    output_video_path = f"../data/output/Video_{video_num}/Video_output"
    output_video_name = f"Video_{video_num}_out_{test_type}.mp4"
    
    output_json_path = f"../data/output/Video_{video_num}/Landmarks_output"
    output_json_name = f"Video_{video_num}_out_{test_type}.json"
    
    output_csv_path = f"../data/output/Video_{video_num}/Landmarks_output"
    output_csv_name = f"Video_{video_num}_out_{test_type}.csv"
    

    
    coords = get_landmarks('mediapipe', model_type, source_video_path, output_video_path, output_video_name)
    
    save_json(coords, output_json_path, output_json_name)
    save_csv(coords, output_csv_path, output_csv_name)
    plot_xyz(coords, landmark="RIGHT_HAND.INDEX_FINGER_TIP", fps=30, show=True)
    print(f"Extracted {len(coords)} frames of landmarks")
    print(coords.keys())
