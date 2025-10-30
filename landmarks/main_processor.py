import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import av

from .media_pipe_wrapper import MPModel
from .rtm_wrapper import RTMModel


def get_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found at {path}")

    # --- Open container with PyAV ---
    container = av.open(path)
    stream = container.streams.video[0]

    # Rotation can appear in either container or stream metadata
    rotation = (
        container.metadata.get("rotate")
        or stream.metadata.get("rotate")
        or 0
    )

    try:
        rotation = int(rotation)
    except Exception:
        rotation = 0

    print("Container metadata:", container.metadata)
    print("Stream metadata:", stream.metadata)
    container.close()
    print(f"Detected rotation: {rotation}°")

    # --- Create capture object ---
    cap = cv2.VideoCapture(path)
    return cap, rotation
    
def save_json(video_coords, output_dir=".", output_name="landmarks.json", frame_width=1080, frame_height=1920):
    """Save landmarks dictionary to JSON file with metadata."""
    filepath = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output structure with metadata
    output_data = video_coords
    output_data["metadata"] = {
            "frame_width": frame_width,
            "frame_height": frame_height,
    }
    
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)
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
        # Pose only
        if m_type == "pose":
            cap, rotation = get_video(source_video_path)
            model = MPModel("pose", complexity=2, t_confidence=0.5, d_confidence=0.5, use_world_landmarks=True)
            coords = model.processing(
                cap, save_video_output=True,
                output_video_dir=output_video_dir,
                output_video_name=f"pose_{output_video_name}", rotation=rotation
            )
            return {"pose": coords, "hands": {}}
        
        # Hands only
        elif m_type == "hands":
            cap, rotation = get_video(source_video_path)
            model = MPModel("hands", t_confidence=0.4, d_confidence=0.4, use_world_landmarks=True)
            coords = model.processing(
                cap, save_video_output=True,
                output_video_dir=output_video_dir,
                output_video_name=f"hands_{output_video_name}", rotation=rotation
            )
            return {"pose": {}, "hands": coords}
        
        
        # Holistic - pose + hands
        elif m_type == "holistic":
            cap, rotation = get_video(source_video_path)
            model = MPModel("holistic", use_world_landmarks=True)
            coords = model.processing(
                cap, save_video_output=True,
                output_video_dir=output_video_dir,
                output_video_name=f"holistic_{output_video_name}", rotation=rotation
            )

            # Split holistic coords into pose and hands
            pose_coords = {}
            hands_coords = {}
            for landmark, frames in coords.items():
                if "HAND" in landmark:
                    hands_coords[landmark] = frames
                else:
                    pose_coords[landmark] = frames

            return {"pose": pose_coords, "hands": hands_coords}
        
        else:
            raise ValueError("Unsupported model type (use: 'pose', 'hands', 'holistic')")
        
                        
    elif lib == 'openpose':
        raise NotImplementedError("OpenPose support not yet implemented")
    
    
    elif lib == 'rtmlib':
        if m_type != "body26":
            raise ValueError("RTMLib only supports model_type='body26' for now.")

        cap, rotation = get_video(source_video_path)
        model = RTMModel(model_type="body26", device="cuda", backend="onnxruntime")
        coords = model.processing(
            cap,
            save_video_output=True,
            output_video_dir=output_video_dir,
            output_video_name=f"rtm_{output_video_name}", rotation=rotation
        )
        return {"pose": coords, "hands": {}}
    
    
    else:
        raise ValueError("Unsupported library")

if __name__ == "__main__":
    lib="rtmlib"
    model_type = "body26"

    # All 4 test videos are inside Video_1 folder
    test_videos = [
        #"SAW_Ido",
        "SAW_SIDE"
        #"FTN_far_3_valid",
        #"FTN_close_3_valid",
        #"FTN_1_valid",
        #"FTN_2_invalid_1_valid",
        #"FTN_3_valid",
        #"FTN_3_invalid",
    ]

    for test_type in test_videos:
        source_video_path = f"./data/Raw_Videos/Video_1/{test_type}.mp4"

        # Output dirs
        output_video_path = "./data/output/Video_1/Video_output"
        output_video_name = f"Video_1_out_{test_type}.mp4"

        output_json_path = "../data/output/Video_1/Landmarks_output"
        output_json_name = f"Video_1_out_{test_type}.json"

        output_csv_path = "../data/output/Video_1/Landmarks_output"
        output_csv_name = f"Video_1_out_{test_type}.csv"

        print(f"\n--- Processing {test_type} ---")

        # Run Mediapipe
        coords = get_landmarks(lib, model_type, source_video_path, output_video_path, output_video_name)

        # Save landmarks
        save_json(coords, output_json_path, output_json_name)
        save_csv(coords, output_csv_path, output_csv_name)

        print(f"Finished {test_type}")
