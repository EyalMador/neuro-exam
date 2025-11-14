import numpy as np
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import matplotlib.pyplot as plt
from biomarkers.helper import save_biomarkers_json
from scipy.signal import savgol_filter, find_peaks

    
def is_foot_flat(heel, toe, threshold = 0.5):

    foot_size = foot_size_pixels(heel, toe)
    return ((abs(heel['y'] - toe['y']) /foot_size) < threshold )

def foot_size_pixels(heel, toe):

    heel_np = np.array([heel['x'], heel['y']])
    toe_np = np.array([toe['x'], toe['y']])
    return np.linalg.norm(heel_np - toe_np)

def detect_steps(left_toe, right_toe, smooth_window, polyorder):
    distances = foot_distances(left_toe, right_toe)

    distance_values = np.array(list(distances.values()))
    smoothed_distances = savgol_filter(distance_values, smooth_window, polyorder)
    peaks, _ = find_peaks(smoothed_distances, distance=10)
    minimums, _ = find_peaks(-smoothed_distances, distance=10)
    return peaks, minimums, smoothed_distances


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


def step_statistics(left_heel, left_toe, right_heel, right_toe):
    peaks, minimums, smoothed_distances = detect_steps(left_heel, left_toe, smooth_window=11, polyorder=3)
    print("111")
    step_lengths = np.diff(smoothed_distances[peaks])
    mean_step_length = np.mean(step_lengths)
    std_step_length = np.std(step_lengths)
    statistics = {}
    statistics['mean'] = mean_step_length
    statistics['std'] = std_step_length
    return statistics


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
        ltoe = np.array([left_toe[frame]['x'], left_toe[frame]['y']])
        rtoe = np.array([right_toe[frame]['x'], right_toe[frame]['y']])
        distances[frame] = abs(np.linalg.norm(ltoe - rtoe))
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

def max_cc(datapoints):
    left = np.array([v for _, v in sorted(datapoints["left"].items())])
    right = np.array([v for _, v in sorted(datapoints["right"].items())])

    n = min(len(left), len(right))
    left, right = left[:n], right[:n]

    #subtract mean to center the signals
    left = left - np.mean(left)
    right = right - np.mean(right)

    cc_full = np.correlate(left, right, mode='full')
    cc_full = cc_full / (np.std(left) * np.std(right) * n) #normalize cc

    # Find maximum correlation and corresponding lag
    max_idx = np.argmax(cc_full)
    lags = np.arange(-n + 1, n)
    best_lag = lags[max_idx]
    max_cc = cc_full[max_idx]

    return max_cc, best_lag


def knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle):

    knee_angles = calc_knee_angles(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    
    # Plot knee angles
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

def horizontal_foot_place_max_cc(left_toe, right_toe):
    left_x = {frame: left_toe[frame]['x'] for frame in left_toe.keys()}
    right_x = {frame: right_toe[frame]['x'] for frame in right_toe.keys()}
    toes = {
        'left': left_x,
        'right': right_x
    }
    cc, lag = max_cc(toes)
    return cc


def extract_straight_walk_biomarkers(landmarks, output_dir, filename, fps=30):
    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle, head] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle", "Head"])
    

    biomarkers = {}
    #steps_biomarkers = step_statistics(left_heel, left_toe, right_heel, right_toe, fps)
    knee_biomarkers = knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    knee_angles = calc_knee_angles(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    
    angles_max_cc, lag  = max_cc(knee_angles)
    biomarkers["max_angles_cc"] = float(angles_max_cc)
    biomarkers["steps_cc"] = float(horizontal_foot_place_max_cc(left_toe, right_toe))
    biomarkers["step_length"] = step_statistics(left_heel, left_toe, right_heel, right_toe)


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