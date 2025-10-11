import numpy as np

def extract_traj(coords_dict, landmarks_names):
    """
    Extract trajectory arrays for multiple landmarks across frames.
    Works with nested JSON that has 'pose', 'hands', etc.

    Parameters
    ----------
    coords_dict : dict
        Root dictionary from JSON.
    landmarks_names : list of str
        Landmark names to extract, e.g. ["NOSE", "RIGHT_HAND.INDEX_FINGER_TIP"].

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping each landmark name -> np.ndarray of shape (T, 3),
        where T is the number of frames sorted by index.
    """
    trajectories = {}

    for landmark_name in landmarks_names:
        found = False
        for section_name, section_dict in coords_dict.items():
            if landmark_name in section_dict:
                found = True
                frames_dict = section_dict[landmark_name]
                frame_keys = sorted(frames_dict.keys(), key=lambda k: int(k))

                traj = np.array([
                    [frames_dict[f]["x"], frames_dict[f]["y"], frames_dict[f]["z"]]
                    for f in frame_keys
                    if f in frames_dict  # ensure frame exists
                ], dtype=float)

                trajectories[landmark_name] = traj
                break  # found it, move to next landmark

        if not found:
            raise KeyError(
                f"Landmark '{landmark_name}' not found.\n"
                f"Available sections: {list(coords_dict.keys())}\n"
                f"Examples from 'pose': {list(coords_dict.get('pose', {}).keys())[:5]}\n"
                f"Examples from 'hands': {list(coords_dict.get('hands', {}).keys())[:5]}"
            )

    return trajectories
