import numpy as np
    
def is_foot_flat(heel, toe, threshold = 0.02):
    return abs(heel['y'] - toe['y']) < threshold

def detect_steps(left_heel, left_toe, right_heel, right_toe):
    frames = sorted([int(f) for f in left_heel.keys()])
    
    steps = {
        'left': [],
        'right': []
    }
    
    for foot, heel, toe in [('left', left_heel, left_toe), 
                                      ('right', right_heel, right_toe)]:
        in_step = False
        step_start = None
        
        for _, frame in enumerate(frames):
            frame_str = str(frame)
            
            if frame_str not in heel or frame_str not in toe:
                continue
                
            heel_coords = heel[frame_str]
            toe_coords = toe[frame_str]
            
            foot_is_flat = is_foot_flat(heel_coords, toe_coords, flatness_threshold = 0.02)
            
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
        frame2 = str(foot_steps[i + 1][0])
        
        if frame1 in heel and frame2 in heel:
            heel1 = heel[frame1]
            heel2 = heel[frame2]
            
            h1 = [heel1['x'], 0, heel1['z']]
            h2 = [heel2['x'], 0, heel2['z']]

            stride_length = abs(np.linalg.norm(h2 - h1))
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
    
    #stride lengths
    left_strides = stride_lengths(left_heel, steps, 'left')
    right_strides = stride_lengths(right_heel, steps, 'right')
    strides = left_strides + right_strides
    
    # step times
    step_times_data = step_times(steps, fps)
    step_times = step_times_data['all']
    
    #empty cases
    if not strides or not step_times:
        return {
            'error': 'No steps detected',
            'step_size': {},
            'step_time': {}
        }
    
    return {
        'step_size': {
            'mean': float(np.mean(strides)),
            'median': float(np.median(strides)),
            'min': float(np.min(strides)),
            'max': float(np.max(strides)),
            'std': float(np.std(strides)),
            'count': len(strides),
            'all': strides
        },
        'step_time': {
            'mean': float(np.mean(step_times)),
            'median': float(np.median(step_times)),
            'min': float(np.min(step_times)),
            'max': float(np.max(step_times)),
            'std': float(np.std(step_times)),
            'count': len(step_times),
            'all': step_times
        }
    }






def knee_angles(left_knee, left_hip, left_ankle, right_knee, right_heep, right_ankle):
    frames = sorted([int(f) for f in left_knee.keys()])
    
    knee_angles = {
        'left': [],
        'right': [],
        'all': []
    }
    
    for side, hip, knee, ankle in [
        ('left', left_hip, left_knee, left_ankle),
        ('right', right_heep, right_knee, right_ankle)
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
                
                knee_angles[side].append(angle)
                knee_angles['all'].append(angle)
    
    return knee_angles


def knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_heep, right_ankle):

    knee_angles = knee_angles(left_knee, left_hip, left_ankle, right_knee, right_heep, right_ankle)
    
    all_angles = knee_angles['all']
    
    if not all_angles:
        return {'error': 'No knee angles detected'}
    
    return {
        'left': {
            'mean': float(np.mean(knee_angles['left'])) if knee_angles['left'] else None,
            'median': float(np.median(knee_angles['left'])) if knee_angles['left'] else None,
            'min': float(np.min(knee_angles['left'])) if knee_angles['left'] else None,
            'max': float(np.max(knee_angles['left'])) if knee_angles['left'] else None,
            'std': float(np.std(knee_angles['left'])) if knee_angles['left'] else None,
            'count': len(knee_angles['left']),
            'all': knee_angles['left']
        },
        'right': {
            'mean': float(np.mean(knee_angles['right'])) if knee_angles['right'] else None,
            'median': float(np.median(knee_angles['right'])) if knee_angles['right'] else None,
            'min': float(np.min(knee_angles['right'])) if knee_angles['right'] else None,
            'max': float(np.max(knee_angles['right'])) if knee_angles['right'] else None,
            'std': float(np.std(knee_angles['right'])) if knee_angles['right'] else None,
            'count': len(knee_angles['right']),
            'all': knee_angles['right']
        },
        'all': {
            'mean': float(np.mean(all_angles)),
            'median': float(np.median(all_angles)),
            'min': float(np.min(all_angles)),
            'max': float(np.max(all_angles)),
            'std': float(np.std(all_angles)),
            'count': len(all_angles),
            'all': all_angles
        }
    }

