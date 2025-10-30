import numpy as np
import biomarkers.helper as helper
from biomarkers.helper import save_biomarkers_json

models = ["pose","hands"]

def finger_to_nose_distance(right_finger, left_finger, nose):

    distances = {
        'left':  {},
        'right': {}     
    }
    
    # Get all frames (assuming all landmarks have same frames)
    frames = range(len(nose))
    
    # Calculate distances for each frame
    for frame in frames:
        frame = str(frame)  
        nose_coords = np.array([nose[frame]['x'], nose[frame]['y'], nose[frame]['z']])

        if frame in right_finger.keys():
            right_finger_coords = np.array([right_finger[frame]['x'], right_finger[frame]['y'], right_finger[frame]['z']])
            right_distance = np.linalg.norm(right_finger_coords - nose_coords)
            distances['right'][frame] = float(right_distance)

        if frame in left_finger.keys():
            left_finger_coords = np.array([left_finger[frame]['x'], left_finger[frame]['y'], left_finger[frame]['z']])   
            left_distance = np.linalg.norm(left_finger_coords - nose_coords)
            distances['left'][frame] = float(left_distance)
  
    return distances

def touching(distance, threshold=0.5):
    return distance < threshold

def moving_finger(distance_data, frames_number):
    # find the moving finger

    #unmoving finger is resting outside the frame
    if len(distance_data['left'].keys()) < 0.5*frames_number:
        return 'right'
    if len(distance_data['left'].keys()) < 0.5*frames_number:
        return 'left'
    
    for frame in range(frames_number):
        for side in ['left', 'right']:
            if frame in distance_data[side].keys() and touching(distance_data[side][frame]):
                return side
            
    return 'none'



def local_minimum_distances_statistics(right_finger, left_finger, nose):

    distance_data = finger_to_nose_distance(right_finger, left_finger, nose)
    frames = range(len(nose))
    touching_side = moving_finger(distance_data, frames)

    statistics = {}

    if moving_finger=='none':
        return {
                'mean': 1000,
                'std': 100
            }
    distances_minimums = helper.datapoints_local_minimums(distance_data)


    side_distances_minimums = distances_minimums[touching_side]['min_values']

    statistics = {
            'mean': float(np.mean(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
            'median': float(np.median(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
            'min': float(np.min(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
            'max': float(np.max(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
            'std': float(np.std(side_distances_minimums)) if len(side_distances_minimums) > 0 else None,
            'count': len(side_distances_minimums),
            'all': side_distances_minimums,
            'all_distances': distance_data

        }
    
    return statistics
    
def extract_finger_to_nose_biomarkers(landmarks, output_dir, filename):

    [left_finger, right_finger, nose] = helper.extract_traj(landmarks,["LEFT_FINGER", "RIGHT_FINGER", "NOSE"])
    biomarkers = {}

    biomarkers["ftn_distances"] = local_minimum_distances_statistics(right_finger, left_finger, nose)
    save_biomarkers_json(biomarkers, output_dir, filename)


    return biomarkers





