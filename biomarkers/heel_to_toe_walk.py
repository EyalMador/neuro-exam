import numpy as np
from scipy.signal import find_peaks
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
import biomarkers.straight_walk as sw


def heel_toe_distance(left_heel, right_heel, left_toe, right_toe):

    distances = {
        'left': {
            'distances': {}
        },
        'right': {
            'distances': {}
        }
    }
    
    # Get all frames (assuming all landmarks have same frames)
    frames = range(len(left_heel))
    
    # Calculate distances for each frame
    for frame in frames:
            
        left_heel_coords = np.array([left_heel['x'], left_heel['y'], left_heel['z']])
        left_toe_coords = np.array([left_toe['x'], left_toe['y'], left_toe['z']])
        right_heel_coords = np.array([right_heel['x'], right_heel['y'], right_heel['z']])
        right_toe_coords = np.array([right_toe['x'], right_toe['y'], right_toe['z']])

        left_distance = np.linalg.norm(left_heel_coords - right_toe_coords)
        right_distance = np.linalg.norm(right_heel_coords - left_toe_coords)

        
        distances['left'][frame] = float(left_distance)
        distances['right'][frame] = float(right_distance)
  
    return distances



def local_minimum_distances_statistics(left_heel, right_heel, left_toe, right_toe):

    distance_data = heel_toe_distance(left_heel, right_heel, left_toe, right_toe)
    distances_minimums = helper.data_local_minimums(distance_data)

    
    if len(distances_minimums['min_frames']) == 0:
        return {'error': 'No knee angles detected'}

    statistics = {}
    for side in ['left', 'right']:
        statistics[side] = {
                'mean': float(np.mean(distances_minimums[side])) if len(distances_minimums[side]) > 0 else None,
                'median': float(np.median(distances_minimums[side])) if len(distances_minimums[side]) > 0 else None,
                'min': float(np.min(distances_minimums[side])) if len(distances_minimums[side]) > 0 else None,
                'max': float(np.max(distances_minimums[side])) if len(distances_minimums[side]) > 0 else None,
                'std': float(np.std(distances_minimums[side])) if len(distances_minimums[side]) > 0 else None,
                'count': len(distances_minimums[side]),
                'all': distances_minimums[side]
            }
        
    return statistics


def extract_heel_to_toe_biomarkers(landmarks, fps):
    rtm_names_landmarks = helper.rtm_indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe"])


    biomarkers = {}
    biomarkers["heel_toe_distances"] = local_minimum_statistics(left_heel, right_heel, left_toe, right_toe)
    biomarkers["step_statistics"] = sw.step_statistics(left_heel, left_toe, right_heel, right_toe, fps)

    helper.plot_biomarkers(biomarkers)
    return biomarkers
