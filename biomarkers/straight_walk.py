import numpy as np

    
def is_foot_flat(heel, toe, threshold = 0.02):
    return abs(heel['y'] - toe['y']) < threshold

def detect_steps(coords):
    frames = sorted([int(f) for f in coords['pose']['LEFT_HEEL'].keys()])
    
    steps = {
        'left': [],
        'right': []
    }
    
    for foot, heel, toe in [('left', 'LEFT_HEEL', 'LEFT_TOE'), 
                                      ('right', 'RIGHT_HEEL', 'RIGHT_TOE')]:
        in_step = False
        step_start = None
        
        for _, frame in enumerate(frames):
            frame_str = str(frame)
            
            if frame_str not in coords['pose'][heel] or frame_str not in coords['pose'][toe]:
                continue
                
            heel_coords = coords['pose'][heel][frame_str]
            toe_coords = coords['pose'][toe][frame_str]
            
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


def stride_lengths(coords, steps, foot):
    heel = 'LEFT_HEEL' if foot == 'left' else 'RIGHT_HEEL'
    stride_lengths = []
    
    foot_steps = steps[foot]
    if len(foot_steps) < 2:
        return stride_lengths
    
    for i in range(len(foot_steps) - 1):
        frame1 = str(foot_steps[i][0])
        frame2 = str(foot_steps[i + 1][0])
        
        if frame1 in coords['pose'][heel] and frame2 in coords['pose'][heel]:
            heel1 = coords['pose'][heel][frame1]
            heel2 = coords['pose'][heel][frame2]
            
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


def step_statistics(video):
    coords = video.coords
    steps = detect_steps(coords)
    
    #stride lengths
    left_strides = stride_lengths(coords, steps, 'left')
    right_strides = stride_lengths(coords, steps, 'right')
    strides = left_strides + right_strides
    
    # step times
    step_times_data = step_times(steps, video.fps)
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






def knee_angles(video):

    coords = video.coords
    frames = sorted([int(f) for f in video.coords['pose']['LEFT_KNEE'].keys()])
    
    knee_angles = {
        'left': [],
        'right': [],
        'all': []
    }
    
    for side, hip, knee, ankle in [
        ('left', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
        ('right', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')
    ]:
        for frame in frames:
            frame_str = str(frame)
            
            # Check if all required landmarks exist in this frame
            if (frame_str in coords['pose'][hip] and 
                frame_str in coords['pose'][knee] and 
                frame_str in coords['pose'][ankle]):
                
                hip_coords = coords['pose'][hip][frame_str]
                knee_coords = coords['pose'][knee][frame_str]
                ankle_coords = coords['pose'][ankle][frame_str]
                
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


def knee_angles_statistics(video):

    knee_angles = knee_angles(video)
    
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

