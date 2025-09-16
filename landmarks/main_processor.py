import cv2
import os
from media_pipe_wrapper import MPModel
import json
import pandas as pd


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
    test_type = "Raise Hands"
    video_num = 1
    source_video_path = f"../data/Raw_Videos/Video_{video_num}/{test_type}.mp4"
    
    output_video_path = f"../data/output/Video_{video_num}/Video_output"
    output_video_name = f"Video_{video_num}_out_{test_type}.mp4"
    
    output_json_path = f"../data/output/Video_{video_num}/Landmarks_output"
    output_json_name = f"Video_{video_num}_out_{test_type}.json"
    
    output_csv_path = f"../data/output/Video_{video_num}/Landmarks_output"
    output_csv_name = f"Video_{video_num}_out_{test_type}.csv"
    
    
    
    coords = get_landmarks('mediapipe', 'pose', source_video_path, output_video_path, output_video_name)
    
    save_json(coords, output_json_path, output_json_name)
    save_csv(coords, output_csv_path, output_csv_name)
    print(f"Extracted {len(coords)} frames of landmarks")
    print(coords.keys())
