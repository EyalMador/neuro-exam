import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline
import json, os



import numpy as np

import os
import json

def biomarkers_to_json_format(biomarkers, result="normal"):
    formatted = {
        "biomarkers": {},
        "label": result
    }

    biomarker_counter = 1

    for category_name, category_data in biomarkers.items():
        # Keep dicts with mean/std only
        if isinstance(category_data, dict) and ('mean' in category_data or 'std' in category_data):
            formatted["biomarkers"][category_name] = {
                "mean": round(category_data.get("mean", 0), 2),
                "std": round(category_data.get("std", 0), 2)
            }
            biomarker_counter += 1
        # Keep single numeric values
        elif isinstance(category_data, (int, float)):
            formatted["biomarkers"][category_name] = round(category_data, 2)
            biomarker_counter += 1
        # ignore everything else

    return formatted





def save_biomarkers_json(biomarkers, output_dir, filename, result="normal"):
    """
    Save biomarkers to JSON file in standardized format.
    
    Parameters:
    -----------
    biomarkers : dict
        Biomarkers dictionary from extract_*_biomarkers functions
    output_dir : str
        Directory to save the JSON file
    filename : str
        Name of the output file (with or without .json extension)
    result : str
        Classification result
    
    Returns:
    --------
    str : Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    output_path = os.path.join(output_dir, filename)

    result = 'abnormal' if 'abnormal' in filename else 'normal'
    formatted_data = biomarkers_to_json_format(biomarkers, result)
    
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Biomarkers saved to: {output_path}")
    return output_path


def extract_traj(coords_dict, landmarks_names):

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



def indices_to_names(coords_dict, rtm_mapping):
    new_coords_dict = {"pose": {}, "hands": coords_dict.get("hands", {})}

    # coords_dict["pose"] contains body parts directly (BODY_0, BODY_1, etc.)
    for body_part_key, frames_dict in coords_dict.get("pose", {}).items():
        joint_name = rtm_mapping.get(body_part_key, body_part_key)
        
        # frames_dict contains frame numbers -> coordinates
        new_coords_dict["pose"][joint_name] = frames_dict

    return new_coords_dict


def smooth_datapoints(datapoints, method='savgol', window_length=13, polyorder=3, s=None):

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
                              smooth=True, window_length=11):
    
    print("start minimum function")
    
    minimums = {
        'left': {},
        'right': {}
    }
    smoothed =  {}
    
    for side in ['left', 'right']:
        frames = np.array(list(data[side].keys()))
        distances = np.array([data[side][frame] for frame in frames])
        
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
        print("before smoothe function")
        # Smooth data if requested
        if smooth and len(distances) > window_length:
            smoothed[side] = smooth_datapoints(distances)
        else:
            smoothed[side] = distances.copy()

        print("after smooth function")
        
        # Find minimums by inverting signal and finding peaks
        inverted = -smoothed[side]
        min_indices, properties = find_peaks(
            inverted,
            prominence=prominence,
            distance=distance
        )
        print("after inverting")
        
        # Get frames and values at minimums
        min_frames = frames[min_indices]
        min_values = smoothed[side][min_indices]
        
        minimums[side] = {
            'min_frames': min_frames,
            'min_values': min_values,
            'min_indices': min_indices,
            'smoothed_distances': smoothed[side],
            'properties': properties
        }
    
    return minimums, smoothed


def calc_symmetry(left_stats, right_stats, max_asymmetry_percent=50):

    if left_stats['count'] == 0 or right_stats['count'] == 0:
        return None

    left_mean = left_stats['mean']
    right_mean = right_stats['mean']

    if left_mean is None or right_mean is None:
        return None

    # Calculate Symmetry Index
    avg = (left_mean + right_mean) / 2
    if avg == 0:
        return None

    symmetry_index = (abs(left_mean - right_mean) / avg) * 100

    # Convert to 0-1 score
    # SI = 0% -> score = 1.0
    # SI = max_asymmetry_percent -> score = 0.0
    symmetry_score = max(0.0, min(1.0, 1 - (symmetry_index / max_asymmetry_percent)))

    return float(symmetry_score)



def plot_biomarkers(biomarkers, test_name='straight_walk'):
    """
    Generic plot for biomarkers based on test type.
    
    Args:
        biomarkers: dict from extract_*_biomarkers()
        test_name: str, one of 'straight_walk', 'heel_to_toe', etc.
    """
    
    # Helper function to safely get nested data
    def safe_get(data, *keys, default=None):
        """Safely navigate nested dict/list structures"""
        for key in keys:
            try:
                if isinstance(data, dict):
                    data = data[key]
                elif isinstance(data, list) and isinstance(key, int):
                    data = data[key]
                else:
                    return default
            except (KeyError, IndexError, TypeError):
                return default
        return data if data is not None else default
    
    # ADD THIS NEW HELPER FUNCTION HERE
    def _extract_continuous_distances(biomarkers, key):
        """Extract continuous distance data organized by frame"""
        result = {'Left': [], 'Right': []}
        
        # Try to get the data - it might be nested differently
        distances_data = safe_get(biomarkers, key, default={})
        
        if not distances_data:
            return result
        
        # Check if 'all_distances' exists at the top level
        if 'all_distances' in distances_data:
            all_dist = distances_data['all_distances']
            # Check if it's organized by side
            for side in ['left', 'right']:
                if side in all_dist:
                    side_data = all_dist[side]
                    if isinstance(side_data, dict):
                        frames = sorted(side_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                        values = [side_data[frame] for frame in frames]
                        result[side.capitalize()] = values
                    elif isinstance(side_data, list):
                        result[side.capitalize()] = side_data
        else:
            # Data is organized as distances_data[side][frame_or_all_distances]
            for side in ['left', 'right']:
                side_data = safe_get(distances_data, side, default={})
                
                # Try 'all_distances' key first
                all_distances = side_data.get('all_distances', None)
                
                if all_distances and isinstance(all_distances, dict):
                    # Filter out non-numeric keys like 'left', 'right'
                    frames = [k for k in all_distances.keys() if str(k).isdigit()]
                    frames = sorted(frames, key=lambda x: int(x))
                    values = [all_distances[frame] for frame in frames]
                    result[side.capitalize()] = values
                elif all_distances and isinstance(all_distances, list):
                    result[side.capitalize()] = all_distances
        
        return result
    
    # Define plot configurations for each test type
    plot_configs = {
        'straight_walk': [
            {
                'data': lambda b: safe_get(b, 'steps', 'step_size', 'all', default=[]),
                'title': 'Step Sizes',
                'xlabel': 'Step',
                'ylabel': 'Size (pixels)',
                'marker': 'o',
                'color': None
            },
            {
                'data': lambda b: safe_get(b, 'steps', 'step_time', 'all', default=[]),
                'title': 'Step Times',
                'xlabel': 'Step',
                'ylabel': 'Time (s)',
                'marker': 's',
                'color': 'purple'
            },
            {
                'data': lambda b: {
                    'Left': safe_get(b, 'knee_angles', 'left', 'all', default=[]),
                    'Right': safe_get(b, 'knee_angles', 'right', 'all', default=[])
                },
                'title': 'Knee Angles',
                'xlabel': 'Frame',
                'ylabel': 'Angle (degrees)',
                'type': 'multi_line'
            },
            {
                'type': 'stats',
                'data': lambda b: f"""
Step Size:
  Mean: {safe_get(b, 'steps', 'step_size', 'mean', default=0):.2f}px
  Std: {safe_get(b, 'steps', 'step_size', 'std', default=0):.2f}px

Step Time:
  Mean: {safe_get(b, 'steps', 'step_time', 'mean', default=0):.3f}s
  Std: {safe_get(b, 'steps', 'step_time', 'std', default=0):.3f}s

Knee Angles (all):
  Mean: {safe_get(b, 'knee_angles', 'all', 'mean', default=0):.1f}°
  Std: {safe_get(b, 'knee_angles', 'all', 'std', default=0):.1f}°
                """
            }
        ],
        
        'heel_to_toe': [
            {
                'data': lambda b: {
                    'Left': safe_get(b, 'heel_toe_distances', 'left', 'all', default=[]),
                    'Right': safe_get(b, 'heel_toe_distances', 'right', 'all', default=[])
                },
                'title': 'Heel-to-Toe Minimum Distances',
                'xlabel': 'Measurement',
                'ylabel': 'Distance (pixels)',
                'type': 'multi_line'
            },
            {
                'data': lambda b: _extract_continuous_distances(b, 'heel_toe_distances'),
                'title': 'Heel-to-Toe Distance Over Time',
                'xlabel': 'Frame',
                'ylabel': 'Distance (pixels)',
                'type': 'multi_line'
            },
            {
                'data': lambda b: {
                    'Left': safe_get(b, 'knee_angles', 'left', 'all', default=[]),
                    'Right': safe_get(b, 'knee_angles', 'right', 'all', default=[])
                },
                'title': 'Knee Angles',
                'xlabel': 'Frame',
                'ylabel': 'Angle (degrees)',
                'type': 'multi_line'
            },
            {
                'type': 'stats',
                'data': lambda b: f"""
Heel-to-Toe (Left):
  Mean: {safe_get(b, 'heel_toe_distances', 'left', 'mean', default=0):.2f}px
  Std: {safe_get(b, 'heel_toe_distances', 'left', 'std', default=0):.2f}px
  Count: {safe_get(b, 'heel_toe_distances', 'left', 'count', default=0)}

Heel-to-Toe (Right):
  Mean: {safe_get(b, 'heel_toe_distances', 'right', 'mean', default=0):.2f}px
  Std: {safe_get(b, 'heel_toe_distances', 'right', 'std', default=0):.2f}px
  Count: {safe_get(b, 'heel_toe_distances', 'right', 'count', default=0)}

Knee Angles (all):
  Mean: {safe_get(b, 'knee_angles', 'all', 'mean', default=0):.1f}°
  Std: {safe_get(b, 'knee_angles', 'all', 'std', default=0):.1f}°
                """
            }
        ]
    }
    
    if test_name not in plot_configs:
        raise ValueError(f"Unknown test name: {test_name}. Available: {list(plot_configs.keys())}")
    
    config = plot_configs[test_name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, plot_spec in enumerate(config):
        ax = axes[i]
        
        if plot_spec.get('type') == 'stats':
            # Stats text box
            ax.axis('off')
            stats_text = plot_spec['data'](biomarkers)
            ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        elif plot_spec.get('type') == 'multi_line':
            # Multiple lines on same plot
            data_dict = plot_spec['data'](biomarkers)
            for label, data in data_dict.items():
                if data is not None and len(data) > 0:
                    ax.plot(data, label=label, linewidth=2)
            if any(len(data) > 0 for data in data_dict.values() if data is not None):
                ax.set_title(plot_spec['title'])
                ax.set_xlabel(plot_spec['xlabel'])
                ax.set_ylabel(plot_spec['ylabel'])
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')  # Hide empty plot
        
        else:
            # Single line plot
            data = plot_spec['data'](biomarkers)
            if data is not None:
                ax.plot(data, marker=plot_spec.get('marker'), 
                       linewidth=2, color=plot_spec.get('color'))
                ax.set_title(plot_spec['title'])
                ax.set_xlabel(plot_spec['xlabel'])
                ax.set_ylabel(plot_spec['ylabel'])
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')  # Hide empty plot
    
    plt.tight_layout()
    plt.show()
