import numpy as np

def extract_traj(coords_dict, landmarks_names):
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
    traj = np.array()
    for _, section_dict in coords_dict.items():
        for landmark_name in landmarks_names:
            if landmark_name in section_dict:
                frames_dict = section_dict[landmark_name]
                frame_keys = sorted(frames_dict.keys(), key=lambda k: int(k))

                traj.append( np.array([
                    [frames_dict[f]["x"], frames_dict[f]["y"], frames_dict[f]["z"]]
                    for f in frame_keys
                ], dtype=float))

            else:
                 # If not found in any section
                raise KeyError(
                    f"Landmark '{landmark_name}' not found. "
                    f"Available sections: {list(coords_dict.keys())}. "
                    f"Examples from 'pose': {list(coords_dict.get('pose', {}).keys())[:5]} "
                    f"Examples from 'hands': {list(coords_dict.get('hands', {}).keys())[:5]}"
                )

        return traj

   