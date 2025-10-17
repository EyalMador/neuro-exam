import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline

def extract_traj(coords_dict, landmarks_names):
    """
    Extract trajectory dictionaries for multiple landmarks.
    
    Parameters
    ----------
    coords_dict : dict
        Root dictionary from JSON (with 'pose', 'hands', etc.)
    landmarks_names : list of str
        Landmark names to extract, e.g. ["LHeel", "RHeel", "LBigToe"]

    Returns
    -------
    list of dict
        List of dictionaries, one per landmark.
        Each dict has frame numbers as keys: {"0": {"x": ..., "y": ..., "z": ...}, "1": {...}, ...}
    """
    trajectories = []

    for landmark_name in landmarks_names:
        found = False
        
        # Check in 'pose' section
        if 'pose' in coords_dict and landmark_name in coords_dict['pose']:
            frames_dict = coords_dict['pose'][landmark_name]
            trajectories.append(frames_dict)
            found = True
        
        # Check in 'hands' section if not found in pose
        elif 'hands' in coords_dict and landmark_name in coords_dict['hands']:
            frames_dict = coords_dict['hands'][landmark_name]
            trajectories.append(frames_dict)
            found = True

        if not found:
            available_pose = list(coords_dict.get('pose', {}).keys())
            available_hands = list(coords_dict.get('hands', {}).keys())
            raise KeyError(
                f"Landmark '{landmark_name}' not found.\n"
                f"Available in 'pose': {available_pose}\n"
                f"Available in 'hands': {available_hands}"
            )

    return trajectories


def rtm_indices_to_names(coords_dict, rtm_mapping):
    new_coords_dict = {"pose": {}, "hands": coords_dict.get("hands", {})}

    # coords_dict["pose"] contains body parts directly (BODY_0, BODY_1, etc.)
    for body_part_key, frames_dict in coords_dict.get("pose", {}).items():
        joint_name = rtm_mapping.get(body_part_key, body_part_key)
        
        # frames_dict contains frame numbers -> coordinates
        new_coords_dict["pose"][joint_name] = frames_dict

    return new_coords_dict




def smooth_datapoints(datapoints, method='savgol', window_length=11, polyorder=3, s=None):
    """
    Smooth landmarks using various methods.
    
    Parameters:
    -----------
    angles : array-like
        Raw angle data
    method : str
        'savgol' for Savitzky-Golay filter
        'spline' for cubic spline smoothing
        'moving_avg' for simple moving average
    window_length : int
        Window size for Savitzky-Golay (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay
    s : float
        Smoothing factor for spline (None = automatic)
    
    Returns:
    --------
    smoothed : array
        Smoothed angle data
    """
    datapoints = np.array(datapoints)
    
    if method == 'savgol':
        # Ensure window_length is odd and valid
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(datapoints))
        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        smoothed = savgol_filter(datapoints, window_length, polyorder)
        
    elif method == 'spline':
        x = np.arange(len(datapoints))
        if s is None:
            s = len(datapoints)  # Automatic smoothing
        spline = UnivariateSpline(x, datapoints, s=s)
        smoothed = spline(x)
        
    elif method == 'moving_avg':
        smoothed = np.convolve(datapoints, np.ones(window_length)/window_length, mode='same')
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return smoothed


def datapoints_local_minimums(data, prominence=0.01, distance=10, 
                              smooth=True, window_length=5):
    """
    Detect local minimums in heel-to-toe distance data.
    This represents moments when heel and toe are closest (foot flat on ground).
    """
    
    minimums = {
        'left': {},
        'right': {}
    }
    
    for side in ['left', 'right']:
        frames = np.array(data[side].keys())
        distances = np.array(data[side][frame] for frame in frames)
        
        if len(distances) == 0:
            minimums[side] = {
                'min_frames': np.array([]),
                'min_values': np.array([]),
                'min_indices': np.array([]),
                'smoothed_distances': np.array([]),
                'all_frames': frames,
                'all_distances': distances
            }
            continue
        
        # Smooth data if requested
        if smooth and len(distances) > window_length:
            smoothed = smooth_datapoints(distances)
        else:
            smoothed = distances.copy()
        
        # Find minimums by inverting signal and finding peaks
        inverted = -smoothed
        min_indices, properties = find_peaks(
            inverted,
            prominence=prominence,
            distance=distance
        )
        
        # Get frames and values at minimums
        min_frames = frames[min_indices]
        min_values = smoothed[min_indices]
        
        minimums[side] = {
            'min_frames': min_frames,
            'min_values': min_values,
            'min_indices': min_indices,
            'smoothed_distances': smoothed,
            'properties': properties
        }
    
    return minimums



def plot_biomarkers(biomarkers):
    """
    Simple plot of all biomarkers.
    
    Args:
        biomarkers: dict from extract_straight_walk_biomarkers()
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Step sizes
    step_sizes = biomarkers['steps']['step_size']['all']
    axes[0, 0].plot(step_sizes, marker='o', linewidth=2)
    axes[0, 0].set_title('Step Sizes')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Size (pixels)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Step times
    step_times = biomarkers['steps']['step_time']['all']
    axes[0, 1].plot(step_times, marker='s', linewidth=2, color='purple')
    axes[0, 1].set_title('Step Times')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Time (s)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Knee angles - get the 'all' lists from statistics
    left_angles = biomarkers['knee_angles']['left']['all']
    right_angles = biomarkers['knee_angles']['right']['all']
    axes[1, 0].plot(left_angles, label='Left', linewidth=2)
    axes[1, 0].plot(right_angles, label='Right', linewidth=2)
    axes[1, 0].set_title('Knee Angles')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics text
    axes[1, 1].axis('off')
    stats = f"""
Step Size:
  Mean: {biomarkers['steps']['step_size']['mean']:.2f}px
  Std: {biomarkers['steps']['step_size']['std']:.2f}px

Step Time:
  Mean: {biomarkers['steps']['step_time']['mean']:.3f}s
  Std: {biomarkers['steps']['step_time']['std']:.3f}s

Knee Angles (all):
  Mean: {biomarkers['knee_angles']['all']['mean']:.1f}°
  Std: {biomarkers['knee_angles']['all']['std']:.1f}°
    """
    axes[1, 1].text(0.1, 0.5, stats, fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Step sizes
    step_sizes = biomarkers['steps']['step_size']['all']
    axes[0, 0].plot(step_sizes, marker='o', linewidth=2)
    axes[0, 0].set_title('Step Sizes')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Size (pixels)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Step times
    step_times = biomarkers['steps']['step_time']['all']
    axes[0, 1].plot(step_times, marker='s', linewidth=2, color='purple')
    axes[0, 1].set_title('Step Times')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Time (s)')
    axes[0, 1].grid(True, alpha=0.3)


    # Knee angles
    left_angles = biomarkers['knee_angles']['left']
    right_angles = biomarkers['knee_angles']['right']
    axes[1, 0].plot(left_angles, label='Left', linewidth=2)
    axes[1, 0].plot(right_angles, label='Right', linewidth=2)
    axes[1, 0].set_title('Knee Angles')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

        
    plt.tight_layout()
    plt.show()



