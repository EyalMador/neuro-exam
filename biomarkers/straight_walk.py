import numpy as np
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import matplotlib.pyplot as plt
from biomarkers.helper import save_biomarkers_json
from scipy.signal import find_peaks



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


import matplotlib.pyplot as plt
import numpy as np


def plot_step_analysis(left_heel, right_heel, fps=30):
    """
    Simple plot of heel heights and step statistics.
    """
    frames = range(len(left_heel))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left heel height
    left_ys = []
    left_frames = []
    for frame in frames:
        frame_str = str(frame)
        if frame_str in left_heel:
            left_ys.append(left_heel[frame_str]['y'])
            left_frames.append(frame)
    
    axes[0].plot(left_frames, left_ys, 'b-', linewidth=2, label='Left')
    
    # Right heel height
    right_ys = []
    right_frames = []
    for frame in frames:
        frame_str = str(frame)
        if frame_str in right_heel:
            right_ys.append(right_heel[frame_str]['y'])
            right_frames.append(frame)
    
    axes[0].plot(right_frames, right_ys, 'r-', linewidth=2, label='Right')
    axes[0].set_title('Heel Heights Over Time')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Height (pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Statistics
    axes[1].axis('off')
    
    stats = f"Left Heel:\n"
    stats += f"  Mean: {np.mean(left_ys):.2f}\n"
    stats += f"  Std:  {np.std(left_ys):.2f}\n\n"
    stats += f"Right Heel:\n"
    stats += f"  Mean: {np.mean(right_ys):.2f}\n"
    stats += f"  Std:  {np.std(right_ys):.2f}\n"
    
    axes[1].text(0.1, 0.5, stats, fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def gait_parameters(left_hip, right_hip, left_shoulder, right_shoulder, 
                               left_elbow, right_elbow):

    frames = range(len(left_hip))
    
    biomarkers = {
        'hip_vertical_stability': None,
        'hip_depth_motion': None,
        'shoulder_vertical_stability': None,
        'arm_swing_height_left': None,
        'arm_swing_height_right': None,
        'arm_swing_asymmetry': None,
        'torso_lean': None
    }
    
    # Hip vertical stability (lower = better, less bobbing = more efficient)
    left_hip_ys = []
    right_hip_ys = []
    left_hip_zs = []
    right_hip_zs = []
    
    for frame in frames:
        frame_str = str(frame)
        if frame_str in left_hip:
            left_hip_ys.append(left_hip[frame_str]['y'])
            left_hip_zs.append(left_hip[frame_str].get('z', 0))
        if frame_str in right_hip:
            right_hip_ys.append(right_hip[frame_str]['y'])
            right_hip_zs.append(right_hip[frame_str].get('z', 0))
    
    if left_hip_ys and right_hip_ys:
        # Vertical stability: std of y position (lower = less bobbing = efficient)
        left_hip_stability = np.std(left_hip_ys)
        right_hip_stability = np.std(right_hip_ys)
        avg_hip_stability = (left_hip_stability + right_hip_stability) / 2
        biomarkers['hip_vertical_stability'] = float(avg_hip_stability)
        
        # Hip depth motion (z-axis): forward/backward sway
        if left_hip_zs and right_hip_zs:
            hip_depths = []
            for f in frames:
                frame_str = str(f)
                if frame_str in left_hip and frame_str in right_hip:
                    left_z = left_hip[frame_str].get('z', 0)
                    right_z = right_hip[frame_str].get('z', 0)
                    avg_z = (left_z + right_z) / 2
                    hip_depths.append(avg_z)
            
            if hip_depths:
                biomarkers['hip_depth_motion'] = float(np.std(hip_depths))
    
    # Shoulder vertical stability
    left_shoulder_ys = []
    right_shoulder_ys = []
    
    for frame in frames:
        frame_str = str(frame)
        if frame_str in left_shoulder:
            left_shoulder_ys.append(left_shoulder[frame_str]['y'])
        if frame_str in right_shoulder:
            right_shoulder_ys.append(right_shoulder[frame_str]['y'])
    
    if left_shoulder_ys and right_shoulder_ys:
        shoulder_stability = (np.std(left_shoulder_ys) + np.std(right_shoulder_ys)) / 2
        biomarkers['shoulder_vertical_stability'] = float(shoulder_stability)
    
    # Torso lean (forward lean = negative z)
    if left_shoulder_ys and right_shoulder_ys:
        left_shoulder_zs = []
        right_shoulder_zs = []
        for frame in frames:
            frame_str = str(frame)
            if frame_str in left_shoulder:
                left_shoulder_zs.append(left_shoulder[frame_str].get('z', 0))
            if frame_str in right_shoulder:
                right_shoulder_zs.append(right_shoulder[frame_str].get('z', 0))
        
        if left_shoulder_zs and right_shoulder_zs:
            avg_shoulder_z = (np.mean(left_shoulder_zs) + np.mean(right_shoulder_zs)) / 2
            biomarkers['torso_lean'] = float(avg_shoulder_z)
    
    # Arm swing amplitude (vertical motion of elbows)
    left_elbow_ys = []
    right_elbow_ys = []
    
    for frame in frames:
        frame_str = str(frame)
        if frame_str in left_elbow:
            left_elbow_ys.append(left_elbow[frame_str]['y'])
        if frame_str in right_elbow:
            right_elbow_ys.append(right_elbow[frame_str]['y'])
    
    if left_elbow_ys and len(left_elbow_ys) > 2:
        left_arm_swing = max(left_elbow_ys) - min(left_elbow_ys)
        biomarkers['arm_swing_height_left'] = float(left_arm_swing)
    
    if right_elbow_ys and len(right_elbow_ys) > 2:
        right_arm_swing = max(right_elbow_ys) - min(right_elbow_ys)
        biomarkers['arm_swing_height_right'] = float(right_arm_swing)
    
    # Arm swing asymmetry (reduced swing on one side = stroke, Parkinson's)
    if biomarkers['arm_swing_height_left'] and biomarkers['arm_swing_height_right']:
        left_amp = biomarkers['arm_swing_height_left']
        right_amp = biomarkers['arm_swing_height_right']
        if (left_amp + right_amp) > 0:
            arm_asymmetry = abs(left_amp - right_amp) / ((left_amp + right_amp) / 2)
            biomarkers['arm_swing_asymmetry'] = float(arm_asymmetry)
    
    return biomarkers


    
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
        # Extract heel heights (y-coordinates)
        heel_heights = {}
        for frame in frames:
            frame_str = str(frame)
            if frame_str in heel:
                heel_heights[frame] = heel[frame_str]['y']
        
        if not heel_heights:
            continue
        
        # Sort frames and extract heights array
        frames_sorted = sorted(heel_heights.keys())
        heights_array = np.array([heel_heights[f] for f in frames_sorted])
        
        # Smooth to reduce noise
        if len(heights_array) > 7:
            from biomarkers.helper import smooth_datapoints
            smoothed_heights = smooth_datapoints(heights_array, method='savgol', 
                                                window_length=7, polyorder=3)
        else:
            smoothed_heights = heights_array
        
        # Find local minima (heel strikes = lowest points in y-coordinate)
        inverted = -smoothed_heights
        min_indices, properties = find_peaks(inverted, prominence=0.5, distance=8)
        
        # Convert indices back to frame numbers
        heel_strike_frames = [frames_sorted[idx] for idx in min_indices]
        
        # Extract step intervals from consecutive heel strikes
        if len(heel_strike_frames) >= 2:
            for i in range(len(heel_strike_frames) - 1):
                step_start = heel_strike_frames[i]
                step_end = heel_strike_frames[i + 1]
                steps[foot].append((step_start, step_end))
    
    return steps


def stride_lengths(heel, toe, steps, foot):

    stride_lengths_raw = []
    stride_lengths_normalized = []
    step_heights = []  # NEW: vertical lift of foot
    
    foot_steps = steps[foot]
    if len(foot_steps) < 2:
        return stride_lengths_raw, stride_lengths_normalized, step_heights
    
    for i in range(len(foot_steps) - 1):
        frame1_start = str(foot_steps[i][0])
        frame1_end = str(foot_steps[i][1])
        frame2_start = str(foot_steps[i + 1][0])
        frame2_end = str(foot_steps[i + 1][1])
        
        if (frame1_start in heel and frame1_end in heel and 
            frame2_start in heel and frame2_end in heel):
            
            heel1_start = heel[frame1_start]
            heel1_end = heel[frame1_end]
            heel2_start = heel[frame2_start]
            heel2_end = heel[frame2_end]
            
            # Raw stride = horizontal distance (may be small in parallel view)
            h1 = np.array([heel1_start['x'], 0, 0])
            h2 = np.array([heel2_start['x'], 0, 0])
            stride_length_raw = np.linalg.norm(h2 - h1)
            stride_lengths_raw.append(stride_length_raw)
            
            # Step height = how much heel lifts during swing phase
            # Min y during first step's swing phase
            min_y_frame1 = heel1_start['y']
            max_y_frame1 = heel1_end['y']
            
            # Find peak height (min y, assuming y increases downward)
            step_height = abs(max_y_frame1 - min_y_frame1)
            
            # Get foot size for normalization
            if frame1_end in toe and frame2_end in toe:
                toe1 = toe[frame1_end]
                toe2 = toe[frame2_end]
                foot_size_frame1 = foot_size_pixels(heel1_end, toe1)
                foot_size_frame2 = foot_size_pixels(heel2_end, toe2)
                avg_foot_size = (foot_size_frame1 + foot_size_frame2) / 2
                
                stride_length_normalized = stride_length_raw / avg_foot_size if avg_foot_size > 0 else 0
                stride_lengths_normalized.append(stride_length_normalized)
                
                # Normalized step height
                step_height_normalized = step_height / avg_foot_size if avg_foot_size > 0 else step_height
                step_heights.append(step_height_normalized)
    
    return stride_lengths_raw, stride_lengths_normalized, step_heights


def step_statistics(left_heel, left_toe, right_heel, right_toe, fps):

    steps = detect_steps(left_heel, left_toe, right_heel, right_toe)
    
    # stride lengths and step heights
    left_strides_raw, left_strides_norm, left_heights = stride_lengths(left_heel, left_toe, steps, 'left')
    right_strides_raw, right_strides_norm, right_heights = stride_lengths(right_heel, right_toe, steps, 'right')
    
    strides_norm = left_strides_norm + right_strides_norm
    all_step_heights = left_heights + right_heights
    
    # step times
    step_times_data = step_times(steps, fps)
    step_times_all = step_times_data['all']
    
    # Stance stability: how stable foot position during stance phase
    stance_stability_left = []
    stance_stability_right = []
    
    for foot, heel, foot_steps in [('left', left_heel, steps['left']), 
                                     ('right', right_heel, steps['right'])]:
        for step_start, step_end in foot_steps:
            heel_ys = []
            for frame in range(step_start, step_end + 1):
                frame_str = str(frame)
                if frame_str in heel:
                    heel_ys.append(heel[frame_str]['y'])
            
            if heel_ys:
                # Stance stability = std of heel position (lower = more stable)
                stability = np.std(heel_ys)
                if foot == 'left':
                    stance_stability_left.append(stability)
                else:
                    stance_stability_right.append(stability)
    
    # empty cases
    if not strides_norm or not step_times_all:
        return {
            'error': 'No steps detected',
            'step_size': {},
            'step_time': {},
            'step_height': {},
            'stance_stability': {}
        }
    
    # Convert to float lists
    strides_norm = [float(x) for x in strides_norm]
    step_times_all = [float(x) for x in step_times_all]
    all_step_heights = [float(x) for x in all_step_heights]
    
    # Calculate regularity for normalized step sizes
    step_size_regularity = 0
    if len(strides_norm) > 2:
        strides_array = np.array(strides_norm)
        step_size_std = np.std(strides_array)
        step_size_mean = np.mean(strides_array)
        step_size_regularity = step_size_std / step_size_mean if step_size_mean > 0 else 999
    
    # Calculate regularity for step times (MOST IMPORTANT for parallel gait)
    step_time_regularity = 0
    if len(step_times_all) > 2:
        times_array = np.array(step_times_all)
        step_time_std = np.std(times_array)
        step_time_mean = np.mean(times_array)
        step_time_regularity = step_time_std / step_time_mean if step_time_mean > 0 else 999
    
    # Calculate regularity for step heights
    step_height_regularity = 0
    if len(all_step_heights) > 2:
        heights_array = np.array(all_step_heights)
        step_height_std = np.std(heights_array)
        step_height_mean = np.mean(heights_array)
        step_height_regularity = step_height_std / step_height_mean if step_height_mean > 0 else 999
    
    # Left-right asymmetry in step heights (indicates weakness on one side)
    left_strides_array = np.array(left_heights) if left_heights else np.array([0.0])
    right_strides_array = np.array(right_heights) if right_heights else np.array([0.0])
    
    left_mean = np.mean(left_strides_array) if len(left_strides_array) > 0 else 0
    right_mean = np.mean(right_strides_array) if len(right_strides_array) > 0 else 0
    
    step_height_asymmetry = 0
    if (left_mean + right_mean) > 0:
        step_height_asymmetry = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)
    
    # Stance stability asymmetry
    stance_stability_asymmetry = 0
    if stance_stability_left and stance_stability_right:
        left_stability_mean = np.mean(stance_stability_left)
        right_stability_mean = np.mean(stance_stability_right)
        if (left_stability_mean + right_stability_mean) > 0:
            stance_stability_asymmetry = abs(left_stability_mean - right_stability_mean) / ((left_stability_mean + right_stability_mean) / 2)
    
    # Cadence (steps per minute)
    cadence = 0
    if step_times_all:
        avg_step_time = np.mean(step_times_all)
        cadence = 60 / avg_step_time if avg_step_time > 0 else 0
    
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
            'asymmetry': float(0)  # Not reliable in parallel view
        },
        'step_time': {
            'mean': float(np.mean(step_times_all)),
            'median': float(np.median(step_times_all)),
            'min': float(np.min(step_times_all)),
            'max': float(np.max(step_times_all)),
            'std': float(np.std(step_times_all)),
            'count': len(step_times_all),
            'all': step_times_all,
            'regularity': float(step_time_regularity)  # CRITICAL METRIC
        },
        'step_height': {  # NEW: vertical foot clearance
            'mean': float(np.mean(all_step_heights)),
            'median': float(np.median(all_step_heights)),
            'min': float(np.min(all_step_heights)),
            'max': float(np.max(all_step_heights)),
            'std': float(np.std(all_step_heights)),
            'count': len(all_step_heights),
            'all': all_step_heights,
            'regularity': float(step_height_regularity),
            'asymmetry': float(step_height_asymmetry),
            'left_mean': float(left_mean),
            'right_mean': float(right_mean)
        },
        'stance_stability': {  # NEW: stability during weight bearing
            'mean': float(np.mean(stance_stability_left + stance_stability_right)) if (stance_stability_left + stance_stability_right) else 0,
            'left_mean': float(np.mean(stance_stability_left)) if stance_stability_left else 0,
            'right_mean': float(np.mean(stance_stability_right)) if stance_stability_right else 0,
            'asymmetry': float(stance_stability_asymmetry),
            'all': [float(x) for x in (stance_stability_left + stance_stability_right)]
        },
        'cadence': float(cadence)  # NEW: steps per minute
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
    
    #plot_knee_angles(knee_angles)
    
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



def extract_straight_walk_biomarkers(landmarks, output_dir, filename, fps=30):
    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle"])
    
    try:
        [left_shoulder, right_shoulder, left_elbow, right_elbow] = helper.extract_traj(
            rtm_names_landmarks,
            ["LShoulder", "RShoulder", "LElbow", "RElbow"])
        has_upper_body = True
    except KeyError:
        has_upper_body = False
        left_shoulder = right_shoulder = left_elbow = right_elbow = {}

    biomarkers = {}
    steps_biomarkers = step_statistics(left_heel, left_toe, right_heel, right_toe, fps)
    knee_biomarkers = knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)

    plot_step_analysis(left_heel, right_heel, fps=30)

    biomarkers["step_time_regularity"] = steps_biomarkers["step_time"]["regularity"]
    biomarkers["step_height_regularity"] = steps_biomarkers["step_height"]["regularity"]
    biomarkers["step_height_asymmetry"] = steps_biomarkers["step_height"]["asymmetry"]
    biomarkers["stance_stability_asymmetry"] = steps_biomarkers["stance_stability"]["asymmetry"]
    biomarkers["cadence"] = steps_biomarkers["cadence"]
    
    biomarkers['knee_symmetry'] = knee_biomarkers['symmetry_score']
    biomarkers['knee_regularity_mean'] = knee_biomarkers['regularity_mean']
    biomarkers['knee_angles_left'] = knee_biomarkers['left']['mean']
    biomarkers['knee_angles_right'] = knee_biomarkers['right']['mean']
    biomarkers['knee_amplitude_asymmetry'] = knee_biomarkers['amplitude_asymmetry']
    
    if has_upper_body:
        upper_body_biomarkers = gait_parameters(
                left_hip, right_hip, left_shoulder, right_shoulder, 
                left_elbow, right_elbow)
        
        biomarkers['arm_swing_asymmetry'] = upper_body_biomarkers['arm_swing_asymmetry']
        biomarkers['hip_vertical_stability'] = upper_body_biomarkers['hip_vertical_stability']

    save_biomarkers_json(biomarkers, output_dir, filename)

    return biomarkers