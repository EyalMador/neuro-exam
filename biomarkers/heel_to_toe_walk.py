import numpy as np
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import biomarkers.straight_walk as sw
from biomarkers.helper import save_biomarkers_json


import matplotlib.pyplot as plt

def plot_heel_toe_distance(distances):

    frames = list(map(int, distances['left'].keys()))
    left_vals = list(distances['left'].values())
    right_vals = list(distances['right'].values())

    plt.figure(figsize=(10, 5))
    plt.plot(frames, left_vals, label='Left heel → Right toe', marker='o')
    plt.plot(frames, right_vals, label='Right heel → Left toe', marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Distance')
    plt.title('Heel-Toe Distance Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()



def heel_toe_distance(left_heel, right_heel, left_toe, right_toe):

    distances = {
        'left':  {},
        'right': {}     
    }
    
    # Get all frames
    frames = range(len(left_heel))
    
    # Calculate distances for each frame
    for frame in frames:
        frame = str(frame)  
        left_heel_coords = np.array([left_heel[frame]['x'], left_heel[frame]['y'], left_heel[frame]['z']])
        left_toe_coords = np.array([left_toe[frame]['x'], left_toe[frame]['y'], left_toe[frame]['z']])
        right_heel_coords = np.array([right_heel[frame]['x'], right_heel[frame]['y'], right_heel[frame]['z']])
        right_toe_coords = np.array([right_toe[frame]['x'], right_toe[frame]['y'], right_toe[frame]['z']])

        left_distance = np.linalg.norm(left_heel_coords - right_toe_coords)
        right_distance = np.linalg.norm(right_heel_coords - left_toe_coords)

        
        distances['left'][frame] = float(left_distance)
        distances['right'][frame] = float(right_distance)
    plot_heel_toe_distance(distances)

  
    return distances


def local_minimum_distances_statistics(left_heel, right_heel, left_toe, right_toe):

    distance_data = heel_toe_distance(left_heel, right_heel, left_toe, right_toe)
    distances_minimums = helper.datapoints_local_minimums(distance_data)
    
    if len(distances_minimums['left']['min_frames']) == 0 or len(distances_minimums['right']['min_frames']) == 0:
        return {'error': 'No data detected'}

    # Get smoothed data for cross-validation
    left_smoothed = distances_minimums['left']['smoothed_distances']
    right_smoothed = distances_minimums['right']['smoothed_distances']

    # Filter minimums: only keep if there's a clear crossing pattern AND no pathological crossing
    for side in ['left', 'right']:
        other_side = 'right' if side == 'left' else 'left'
        
        min_indices = distances_minimums[side]['min_indices']
        min_values = distances_minimums[side]['min_values']
        
        if side == 'left':
            current_smoothed = left_smoothed
            other_smoothed = right_smoothed
        else:
            current_smoothed = right_smoothed
            other_smoothed = left_smoothed
        
        valid_indices = []
        
        for i, min_idx in enumerate(min_indices):
            current_min = current_smoothed[min_idx]
            other_val = other_smoothed[min_idx]
            
            # Check 1: Other line must be significantly higher (crossing pattern)
            if (other_val - current_min) <= 30:
                continue
            
            # Check 2: Ensure this is a TRUE minimum, not a pathological crossing
            window = 3
            start_idx = max(0, min_idx - window)
            end_idx = min(len(current_smoothed), min_idx + window)
            
            local_segment = current_smoothed[start_idx:end_idx+1]
            actual_min_in_segment = np.min(local_segment)
            
            if abs(current_min - actual_min_in_segment) > 5:
                continue
            
            # Check 3: Rate of change should be smooth (not too steep)
            if min_idx > 0 and min_idx < len(current_smoothed) - 1:
                slope_before = abs(current_smoothed[min_idx] - current_smoothed[min_idx - 1])
                slope_after = abs(current_smoothed[min_idx + 1] - current_smoothed[min_idx])
                
                max_slope = max(slope_before, slope_after)
                if max_slope > 50:
                    continue
            
            valid_indices.append(i)
        
        # Keep only valid minimums
        valid_indices = np.array(valid_indices)
        distances_minimums[side]['min_indices'] = min_indices[valid_indices]
        distances_minimums[side]['min_values'] = min_values[valid_indices]
        distances_minimums[side]['min_frames'] = distances_minimums[side]['min_frames'][valid_indices]

    if len(distances_minimums['left']['min_values']) == 0 or len(distances_minimums['right']['min_values']) == 0:
        return {'error': 'No data detected'}

    # Check for regularity: normal gait has consistent spacing between minima
    regularity_scores = {}
    for side in ['left', 'right']:
        min_frames = distances_minimums[side]['min_frames'].astype(int)
        if len(min_frames) > 2:
            intervals = np.diff(min_frames)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            regularity_scores[side] = interval_std / interval_mean if interval_mean > 0 else 999
        else:
            regularity_scores[side] = 0

    # NEW: Calculate per-cycle amplitude consistency
    amplitude_consistency_scores = {}
    
    for side in ['left', 'right']:
        min_frames = distances_minimums[side]['min_frames'].astype(int)
        min_indices = distances_minimums[side]['min_indices']
        
        if side == 'left':
            current_smoothed = left_smoothed
            other_smoothed = right_smoothed
        else:
            current_smoothed = right_smoothed
            other_smoothed = left_smoothed
        
        # For each minimum, calculate the "cycle amplitude"
        # This is the difference between the other line and current line at that point
        cycle_amplitudes = []
        
        for min_idx in min_indices:
            current_min = current_smoothed[min_idx]
            other_val = other_smoothed[min_idx]
            
            # Amplitude = how much the other line is above at this meeting point
            amplitude = other_val - current_min
            cycle_amplitudes.append(amplitude)
        
        cycle_amplitudes = np.array(cycle_amplitudes)
        
        if len(cycle_amplitudes) > 1:
            # Calculate consistency: low std relative to mean = consistent cycles = normal
            amplitude_mean = np.mean(cycle_amplitudes)
            amplitude_std = np.std(cycle_amplitudes)
            
            # Coefficient of variation for amplitudes
            amplitude_cv = amplitude_std / amplitude_mean if amplitude_mean > 0 else 999
            amplitude_consistency_scores[side] = amplitude_cv
        else:
            amplitude_consistency_scores[side] = 0
    
    # Calculate overall amplitude consistency (should be low for normal gait)
    overall_amplitude_cv = np.mean([amplitude_consistency_scores['left'], 
                                     amplitude_consistency_scores['right']])

    statistics = {}
    for side in ['left', 'right']:
        side_distances_minimums = distances_minimums[side]['min_values']

        statistics[side] = {
                'mean': float(np.mean(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
                'median': float(np.median(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
                'min': float(np.min(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
                'max': float(np.max(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
                'std': float(np.std(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
                'count': len(side_distances_minimums),
                'all': list(side_distances_minimums),
                'all_distances': distance_data,
                'regularity': float(regularity_scores[side]),
                'amplitude_consistency': float(amplitude_consistency_scores[side])

            }
    
    statistics['symmetry_score'] = helper.calc_symmetry(statistics['left'], statistics['right'])
    statistics['regularity_mean'] = float(np.mean([regularity_scores['left'], regularity_scores['right']]))
    statistics['amplitude_consistency_mean'] = float(overall_amplitude_cv)
        
    return statistics

def extract_heel_to_toe_biomarkers(landmarks, output_dir, filename):

    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle"])

    
    biomarkers = {}
    htt_distances = local_minimum_distances_statistics(left_heel, right_heel, left_toe, right_toe)
    knee_biomarkers = sw.knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)

    knee_angles = sw.calc_knee_angles(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)
    
    angles_max_cc, lag  = sw.max_cc(knee_angles)
    biomarkers["max_angles_cc"] = float(angles_max_cc)
    biomarkers["htt_distances_left"] = htt_distances['left']
    biomarkers["htt_distances_right"] = htt_distances['right']
    biomarkers["htt_distances_symmetry"] = htt_distances['symmetry_score']
    biomarkers["htt_distances_regularity"] = htt_distances['regularity_mean']

    biomarkers["htt_amplitude"] = htt_distances["amplitude_consistency_mean"]

    biomarkers['knee_angles_left'] = knee_biomarkers['left']
    biomarkers['knee_angles_right'] = knee_biomarkers['right']
    biomarkers['knee_symmetry'] = knee_biomarkers['symmetry_score']

    #helper.plot_biomarkers(biomarkers, "heel_to_toe")
    save_biomarkers_json(biomarkers, output_dir, filename)
    return biomarkers