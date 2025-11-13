import numpy as np
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import matplotlib.pyplot as plt
from biomarkers.helper import save_biomarkers_json
from scipy.signal import savgol_filter


def plot_knee_angles(knee_angles):
    """
    Plot knee angles over time for left, right, and all knees.
    
    Args:
        knee_angles: Dictionary with 'left', 'right', and 'all' keys,
                     each containing frame:angle pairs
    """
    plt.figure(figsize=(12, 6))
    
    # Plot left knee
    if 'left' in knee_angles and knee_angles['left']:
        frames_left = sorted(knee_angles['left'].keys())
        angles_left = [knee_angles['left'][f] for f in frames_left]
        plt.plot(frames_left, angles_left, label='Left Knee', marker='o', markersize=3)
    
    # Plot right knee
    if 'right' in knee_angles and knee_angles['right']:
        frames_right = sorted(knee_angles['right'].keys())
        angles_right = [knee_angles['right'][f] for f in frames_right]
        plt.plot(frames_right, angles_right, label='Right Knee', marker='o', markersize=3)
    
    # Plot all knees (if different from left/right)
    if 'all' in knee_angles and knee_angles['all']:
        frames_all = sorted(knee_angles['all'].keys())
        angles_all = [knee_angles['all'][f] for f in frames_all]
        plt.plot(frames_all, angles_all, label='All Knees', marker='o', markersize=3, alpha=0.5)
    
    plt.xlabel('Frame')
    plt.ylabel('Knee Angle (degrees)')
    plt.title('Knee Angles Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_foot_distances(distances):
    """
    Plot the distance between left and right toe over time.
    
    Args:
        distances: dict returned from foot_distances()
                   Example: {0: 150.5, 1: 148.3, 2: 145.1, ...}
    """
    
    if not distances:
        print("No distances data")
        return
    
    # Extract frames and distances
    frames = sorted(distances.keys())
    dist_values = [distances[f] for f in frames]
    
    # Create plot
    plt.figure(figsize=(12, 5))
    plt.plot(frames, dist_values, 'b-', linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Frame')
    plt.ylabel('Distance (pixels)')
    plt.title('Distance Between Left and Right Toe Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nFoot Distance Statistics:")
    print(f"  Min: {min(dist_values):.2f} pixels")
    print(f"  Max: {max(dist_values):.2f} pixels")
    print(f"  Mean: {np.mean(dist_values):.2f} pixels")
    print(f"  Std: {np.std(dist_values):.2f} pixels")

    
def is_foot_flat(heel, toe, threshold = 0.5):

    foot_size = foot_size_pixels(heel, toe)
    return ((abs(heel['y'] - toe['y']) /foot_size) < threshold )

def foot_size_pixels(heel, toe):

    heel_np = np.array([heel['x'], heel['y']])
    toe_np = np.array([toe['x'], toe['y']])
    return np.linalg.norm(heel_np - toe_np)

def detect_steps(left_heel, left_toe, right_heel, right_toe):
    frames = range(len(left_heel))
    
    steps = {
        'left': [],
        'right': []
    }
    
    for foot, heel, toe in [('left', left_heel, left_toe), 
                                      ('right', right_heel, right_toe)]:
        in_step = False
        step_start = None

        for frame in frames:
            frame_str = str(frame)

            heel_coords = heel[frame_str]
            toe_coords = toe[frame_str]
            

            foot_is_flat = is_foot_flat(heel_coords, toe_coords, 0.25)
            
            #step start
            if not foot_is_flat and not in_step:
                in_step = True
                step_start = frame

            #step end
            elif foot_is_flat and in_step:
                in_step = False
                if step_start is not None:
                    steps[foot].append((step_start, frame))
                step_start = None

    return steps


def filter_realistic_strides(strides_raw):

    if not strides_raw:
        return [], {'min': 0, 'max': 0, 'removed': 0}
    
    strides_array = np.array(strides_raw)
    
    # Use IQR method to identify outliers
    q1 = np.percentile(strides_array, 25)
    q3 = np.percentile(strides_array, 75)
    iqr = q3 - q1
    
    # Define bounds with safety margins
    lower_bound = max(20, q1 - 1.5 * iqr)  # Min 20 pixels
    upper_bound = min(600, q3 + 1.5 * iqr)  # Max 600 pixels
    
    # Filter
    filtered = [s for s in strides_raw if lower_bound <= s <= upper_bound]
    
    return filtered, {
        'original_count': len(strides_raw),
        'filtered_count': len(filtered),
        'removed': len(strides_raw) - len(filtered),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'original_mean': float(np.mean(strides_array)),
        'filtered_mean': float(np.mean(filtered)) if filtered else 0
    }



def stride_lengths(heel, toe, steps, foot):

    stride_lengths_raw = []
    stride_lengths_normalized = []
    
    foot_steps = steps[foot]
    if len(foot_steps) < 2:
        return [], []
    
    # Calculate all strides first
    for i in range(len(foot_steps) - 1):
        frame1_start = str(foot_steps[i][0])
        frame2_start = str(foot_steps[i + 1][0])
        
        if frame1_start in heel and frame2_start in heel:
            heel1 = heel[frame1_start]
            heel2 = heel[frame2_start]
            
            h1 = np.array([heel1['x'], 0, 0])
            h2 = np.array([heel2['x'], 0, 0])
            stride_length_raw = np.linalg.norm(h2 - h1)
            stride_lengths_raw.append(stride_length_raw)
            
            if frame1_start in toe and frame2_start in toe:
                toe1 = toe[frame1_start]
                toe2 = toe[frame2_start]
                
                foot_size_frame1 = foot_size_pixels(heel1, toe1)
                foot_size_frame2 = foot_size_pixels(heel2, toe2)
                avg_foot_size = (foot_size_frame1 + foot_size_frame2) / 2
                
                stride_length_normalized = stride_length_raw / avg_foot_size if avg_foot_size > 0 else 0
                stride_lengths_normalized.append(stride_length_normalized)
    
    # Filter outliers
    filtered_raw, stats = filter_realistic_strides(stride_lengths_raw)
    
    # Filter normalized to match
    if filtered_raw and stride_lengths_raw:
        mask = [s in filtered_raw for s in stride_lengths_raw]
        filtered_normalized = [n for n, keep in zip(stride_lengths_normalized, mask) if keep]
    else:
        filtered_normalized = []
    
    return filtered_raw, filtered_normalized


def step_statistics(left_heel, left_toe, right_heel, right_toe, fps):
    steps = detect_steps(left_heel, left_toe, right_heel, right_toe)
    
    # stride lengths (both raw and normalized)
    left_strides_raw, left_strides_norm = stride_lengths(left_heel, left_toe, steps, 'left')
    right_strides_raw, right_strides_norm = stride_lengths(right_heel, right_toe, steps, 'right')
    
    strides_norm = left_strides_norm + right_strides_norm
    
    # step times
    step_times_data = step_times(steps, fps)
    step_times_all = step_times_data['all']
    
    # empty cases
    if not strides_norm or not step_times_all:
        return {
            'error': 'No steps detected',
            'step_size': {},
            'step_time': {}
        }
    
    # Calculate regularity for normalized step sizes
    step_size_regularity = 0
    if len(strides_norm) > 2:
        strides_array = np.array(strides_norm)
        step_size_std = np.std(strides_array)
        step_size_mean = np.mean(strides_array)
        step_size_regularity = step_size_std / step_size_mean if step_size_mean > 0 else 999
    
    # Calculate regularity for step times
    step_time_regularity = 0
    if len(step_times_all) > 2:
        times_array = np.array(step_times_all)
        step_time_std = np.std(times_array)
        step_time_mean = np.mean(times_array)
        step_time_regularity = step_time_std / step_time_mean if step_time_mean > 0 else 999
    
    # Calculate left-right asymmetry in normalized step sizes
    left_strides_array = np.array(left_strides_norm) if left_strides_norm else np.array([0])
    right_strides_array = np.array(right_strides_norm) if right_strides_norm else np.array([0])
    
    left_mean = np.mean(left_strides_array) if len(left_strides_array) > 0 else 0
    right_mean = np.mean(right_strides_array) if len(right_strides_array) > 0 else 0
    
    # Asymmetry index
    step_asymmetry = 0
    if (left_mean + right_mean) > 0:
        step_asymmetry = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2) if ((left_mean + right_mean) / 2) > 0 else 999
    
    return {
        'step_size': {
            'mean': float(np.mean(strides_norm)),
            'median': float(np.median(strides_norm)),
            'min': float(np.min(strides_norm)),
            'max': float(np.max(strides_norm)),
            'std': float(np.std(strides_norm)),
            'count': len(strides_norm),
            'all': strides_norm,
            'regularity': float(step_size_regularity),
            'asymmetry': float(step_asymmetry)
        },
        'step_time': {
            'mean': float(np.mean(step_times_all)),
            'median': float(np.median(step_times_all)),
            'min': float(np.min(step_times_all)),
            'max': float(np.max(step_times_all)),
            'std': float(np.std(step_times_all)),
            'count': len(step_times_all),
            'all': step_times_all,
            'regularity': float(step_time_regularity)
        }
    }


def step_times(steps, fps):
    step_times = {
        'left': [],
        'right': [],
        'all': []
    }
    
    for foot in ['left', 'right']:
        for start_frame, end_frame in steps[foot]:
            step_duration = (end_frame - start_frame) / fps
            step_times[foot].append(step_duration)
            step_times['all'].append(step_duration)
    
    return step_times

def foot_distances(left_toe, right_toe):
    distances = {}
    frames = range(len(left_toe))
    for frame in frames:
        ltoe = np.array([left_toe['x'], left_toe['y']])
        rtoe = np.array([right_toe['x'], right_toe['y']])
        distances[frame] = abs(np.linalg.norm(ltoe - rtoe))
    plot_foot_distances(distances)
    return distances





def calc_knee_angles(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle):
    frames = range(len(left_knee))
    
    knee_angles = {
        'left':{},
        'right': {},
        'all': {}
    }
    
    for side, hip, knee, ankle in [
        ('left', left_hip, left_knee, left_ankle),
        ('right', right_hip, right_knee, right_ankle)
    ]:
        for frame in frames:
            frame_str = str(frame)
            
            # Check if all required landmarks exist in this frame
            if (frame_str in hip and 
                frame_str in knee and 
                frame_str in ankle):
                
                hip_coords = hip[frame_str]
                knee_coords = knee[frame_str]
                ankle_coords = ankle[frame_str]
                
                v1 = np.array([hip_coords['x'] - knee_coords['x'], 
                              hip_coords['y'] - knee_coords['y'], 
                              hip_coords['z'] - knee_coords['z']])
                v2 = np.array([ankle_coords['x'] - knee_coords['x'], 
                              ankle_coords['y'] - knee_coords['y'], 
                              ankle_coords['z'] - knee_coords['z']])
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                knee_angles[side][frame] = angle
                knee_angles['all'][frame] = angle
    
    return knee_angles

def knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle):

    knee_angles = calc_knee_angles(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    
    # Plot knee angles
    plot_knee_angles(knee_angles)
    
    # Filter out unrealistic angles (< 90° or > 180° indicate errors/abnormality)
    for side in ['left', 'right', 'all']:
        knee_angles[side] = {frame: angle for frame, angle in knee_angles[side].items() 
                             if 90 <= angle <= 180}
    
    minimum_angles = helper.datapoints_local_minimums(knee_angles)

    all_angles = knee_angles['all']
    
    if not all_angles:
        return {'error': 'No knee angles detected'}

    # Calculate regularity for knee angles
    regularity_scores = {}
    min_angle_values = {}
    
    for side in ['left', 'right']:
        min_frames = minimum_angles[side]['min_frames'].astype(int)
        min_values = minimum_angles[side]['min_values']
        
        if len(min_frames) > 2:
            intervals = np.diff(min_frames)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            regularity_scores[side] = interval_std / interval_mean if interval_mean > 0 else 999
        else:
            regularity_scores[side] = 0
        
        # Track minimum angle values
        if len(min_values) > 0:
            min_angle_values[side] = float(np.min(min_values))
        else:
            min_angle_values[side] = 180

    statistics = {}
    for side in ['left', 'right']:
        side_minimums = minimum_angles[side]['min_values']
        
        # Calculate amplitude (range of motion) - how much knee bends
        all_angles_side = knee_angles[side].values()
        if len(all_angles_side) > 0:
            amplitude = float(max(all_angles_side) - min(all_angles_side))
        else:
            amplitude = 0

        statistics[side] = {
                'mean': float(np.mean(side_minimums)) if len(side_minimums) > 0 else None,
                'median': float(np.median(side_minimums)) if len(side_minimums) > 0 else None,
                'min': float(np.min(side_minimums)) if len(side_minimums) > 0 else None,
                'max': float(np.max(side_minimums)) if len(side_minimums) > 0 else None,
                'std': float(np.std(side_minimums)) if len(side_minimums) > 0 else None,
                'count': len(side_minimums),
                'all': list(side_minimums),
                'regularity': float(regularity_scores[side]),
                'min_angle': min_angle_values[side],
                'amplitude': amplitude
            }
    
    statistics['symmetry_score'] = helper.calc_symmetry(statistics['left'], statistics['right'])
    statistics['regularity_mean'] = float(np.mean([regularity_scores['left'], regularity_scores['right']]))
    
    # Amplitude symmetry - abnormal if one knee bends much more than the other
    left_amp = statistics['left']['amplitude']
    right_amp = statistics['right']['amplitude']
    if (left_amp + right_amp) > 0:
        amplitude_asymmetry = abs(left_amp - right_amp) / ((left_amp + right_amp) / 2)
    else:
        amplitude_asymmetry = 0
    statistics['amplitude_asymmetry'] = float(amplitude_asymmetry)
        
    return statistics


def plot_toe_trajectories(left_toe, right_toe):
    """
    Plot left and right toe trajectories in 2D (X vs Y space).
    
    Args:
        left_toe: dict with frame -> {'x': x, 'y': y, 'z': z}
        right_toe: dict with frame -> {'x': x, 'y': y, 'z': z}
    """
    
    if not left_toe or not right_toe:
        print("No toe data")
        return
    
    # Extract X, Y positions for left toe
    left_frames = sorted(left_toe.keys())
    left_x = [left_toe[f]['x'] for f in left_frames]
    left_y = [left_toe[f]['y'] for f in left_frames]
    
    # Extract X, Y positions for right toe
    right_frames = sorted(right_toe.keys())
    right_x = [right_toe[f]['x'] for f in right_frames]
    right_y = [right_toe[f]['y'] for f in right_frames]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Both trajectories on same plot
    ax = axes[0]
    ax.plot(left_x, left_y, 'b-', linewidth=2, marker='o', markersize=3, label='Left toe')
    ax.plot(right_x, right_y, 'r-', linewidth=2, marker='s', markersize=3, label='Right toe')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Left vs Right Toe Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')  # Equal aspect ratio
    
    # Plot 2: Over time (frame number)
    ax = axes[1]
    ax.plot(left_frames, left_x, 'b-', linewidth=2, label='Left toe X', marker='o', markersize=3)
    ax.plot(left_frames, left_y, 'b--', linewidth=2, label='Left toe Y', marker='o', markersize=3, alpha=0.7)
    ax.plot(right_frames, right_x, 'r-', linewidth=2, label='Right toe X', marker='s', markersize=3)
    ax.plot(right_frames, right_y, 'r--', linewidth=2, label='Right toe Y', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position (pixels)')
    ax.set_title('Toe Positions Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nLeft Toe Statistics:")
    print(f"  X range: {min(left_x):.2f} - {max(left_x):.2f}")
    print(f"  Y range: {min(left_y):.2f} - {max(left_y):.2f}")
    print(f"  X mean: {np.mean(left_x):.2f}, std: {np.std(left_x):.2f}")
    print(f"  Y mean: {np.mean(left_y):.2f}, std: {np.std(left_y):.2f}")
    
    print(f"\nRight Toe Statistics:")
    print(f"  X range: {min(right_x):.2f} - {max(right_x):.2f}")
    print(f"  Y range: {min(right_y):.2f} - {max(right_y):.2f}")
    print(f"  X mean: {np.mean(right_x):.2f}, std: {np.std(right_x):.2f}")
    print(f"  Y mean: {np.mean(right_y):.2f}, std: {np.std(right_y):.2f}")

def extract_straight_walk_biomarkers(landmarks, output_dir, filename, fps=30):
    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle, head] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle", "Head"])
    

    biomarkers = {}
    steps_biomarkers = step_statistics(left_heel, left_toe, right_heel, right_toe, fps)
    knee_biomarkers = knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    plot_toe_trajectories(left_toe, right_toe)


    # Step size biomarkers (now normalized by foot size)
    #biomarkers["step_size"] = steps_biomarkers["step_size"]
    #biomarkers["step_size_regularity"] = steps_biomarkers["step_size"]["regularity"]
    #biomarkers["step_size_asymmetry"] = steps_biomarkers["step_size"].get("asymmetry", 999)
    #biomarkers["step_time"] = steps_biomarkers["step_time"]
    #biomarkers["step_time_regularity"] = steps_biomarkers["step_time"]["regularity"]

    # Knee angle biomarkers
    #biomarkers['knee_angles_left'] = knee_biomarkers['left']
    #biomarkers['knee_angles_right'] = knee_biomarkers['right']
    #biomarkers['knee_symmetry'] = knee_biomarkers['symmetry_score']
    #biomarkers['knee_regularity_mean'] = knee_biomarkers['regularity_mean']
    #biomarkers['knee_regularity_left'] = knee_biomarkers['left']['regularity']
    #biomarkers['knee_regularity_right'] = knee_biomarkers['right']['regularity']

    #biomarkers['knee_amplitude_asymmetry'] = knee_biomarkers['amplitude_asymmetry']

    #helper.plot_biomarkers(biomarkers, "straight_walk")
    save_biomarkers_json(biomarkers, output_dir, filename)

    return biomarkers