import numpy as np
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import matplotlib.pyplot as plt
from biomarkers.helper import save_biomarkers_json



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

        """ 
        gaps = [abs(heel[str(frame)]['y'] - toe[str(frame)]['y']) for frame in frames]
        non_zero_gaps = [gap for gap in gaps if gap > 0]
        #min_gap = min(non_zero_gaps) if non_zero_gaps else 0
        min_gap = 0
        print(f'min gap {min_gap}')

        """

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


def stride_lengths(heel, steps, foot):
    stride_lengths = []
    
    foot_steps = steps[foot]
    if len(foot_steps) < 2:
        return stride_lengths
    
    for i in range(len(foot_steps) - 1):
        frame1 = str(foot_steps[i][0])
        #frame2 = str(foot_steps[i + 1][0]) #todo: i or i++ ?
        frame2 = str(foot_steps[i][1])
        
        if frame1 in heel and frame2 in heel:
            heel1 = heel[frame1]
            heel2 = heel[frame2]
            
            h1 = np.array([heel1['x'], 0, 0])
            h2 = np.array([heel2['x'], 0, 0])

            stride_length = np.linalg.norm(h2 - h1)
            stride_lengths.append(stride_length)
    
    return stride_lengths


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


def step_statistics(left_heel, left_toe, right_heel, right_toe, fps):
    steps = detect_steps(left_heel, left_toe, right_heel, right_toe)
    
    # stride lengths
    left_strides = stride_lengths(left_heel, steps, 'left')
    right_strides = stride_lengths(right_heel, steps, 'right')
    strides = left_strides + right_strides
    
    # step times
    step_times_data = step_times(steps, fps)
    step_times_all = step_times_data['all']
    
    # empty cases
    if not strides or not step_times_all:
        return {
            'error': 'No steps detected',
            'step_size': {},
            'step_time': {}
        }
    
    # Calculate regularity for step sizes
    step_size_regularity = 0
    if len(strides) > 2:
        strides_array = np.array(strides)
        step_size_std = np.std(strides_array)
        step_size_mean = np.mean(strides_array)
        # Coefficient of variation: low = regular, high = irregular
        step_size_regularity = step_size_std / step_size_mean if step_size_mean > 0 else 999
    
    # Calculate regularity for step times
    step_time_regularity = 0
    if len(step_times_all) > 2:
        times_array = np.array(step_times_all)
        step_time_std = np.std(times_array)
        step_time_mean = np.mean(times_array)
        step_time_regularity = step_time_std / step_time_mean if step_time_mean > 0 else 999
    
    return {
        'step_size': {
            'mean': float(np.mean(strides)),
            'median': float(np.median(strides)),
            'min': float(np.min(strides)),
            'max': float(np.max(strides)),
            'std': float(np.std(strides)),
            'count': len(strides),
            'all': strides,
            'regularity': float(step_size_regularity)
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
    
    minimum_angles = helper.datapoints_local_minimums(knee_angles)

    all_angles = knee_angles['all']
    
    if not all_angles:
        return {'error': 'No knee angles detected'}

    # Calculate regularity for knee angles (like we did for heel-toe)
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
        
        # Track minimum angle values (abnormal gait has unrealistically low angles)
        if len(min_values) > 0:
            min_angle_values[side] = float(np.min(min_values))
        else:
            min_angle_values[side] = 180

    statistics = {}
    for side in ['left', 'right']:
        side_minimums = minimum_angles[side]['min_values']

        statistics[side] = {
                'mean': float(np.mean(side_minimums)) if len(side_minimums) > 0 else None,
                'median': float(np.median(side_minimums)) if len(side_minimums) > 0 else None,
                'min': float(np.min(side_minimums)) if len(side_minimums) > 0 else None,
                'max': float(np.max(side_minimums)) if len(side_minimums) > 0 else None,
                'std': float(np.std(side_minimums)) if len(side_minimums) > 0 else None,
                'count': len(side_minimums),
                'all': list(side_minimums),
                'regularity': float(regularity_scores[side]),
                'min_angle': min_angle_values[side]
            }
    
    statistics['symmetry_score'] = helper.calc_symmetry(statistics['left'], statistics['right'])
    statistics['regularity_mean'] = float(np.mean([regularity_scores['left'], regularity_scores['right']]))
        
    return statistics


def extract_straight_walk_biomarkers(landmarks, output_dir, filename, fps=30):
    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle"])
    

    biomarkers = {}
    steps_biomarkers = step_statistics(left_heel, left_toe, right_heel, right_toe, fps)
    knee_biomarkers = knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)

    biomarkers["step_size"] = steps_biomarkers["step_size"]
    biomarkers["step_time"] = steps_biomarkers["step_time"]
    biomarkers["step_size_regularity"] = steps_biomarkers["step_size"]["regularity"]
    biomarkers["step_time_regularity"] = steps_biomarkers["step_time"]["regularity"]

    biomarkers['knee_angles_left'] = knee_biomarkers['left']
    biomarkers['knee_angles_right'] = knee_biomarkers['right']
    biomarkers['knee_symmetry'] = knee_biomarkers['symmetry_score']
    biomarkers['knee_regularity'] = knee_biomarkers['regularity_mean']
    biomarkers['knee_regularity_left'] = knee_biomarkers['left']['regularity']
    biomarkers['knee_regularity_right'] = knee_biomarkers['right']['regularity']
    biomarkers['knee_min_angle_left'] = knee_biomarkers['left'].get('min_angle', 180)
    biomarkers['knee_min_angle_right'] = knee_biomarkers['right'].get('min_angle', 180)

    helper.plot_biomarkers(biomarkers, "straight_walk")
    save_biomarkers_json(biomarkers, output_dir, filename)

    return biomarkers