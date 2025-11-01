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
    
    if len(distances_minimums['left']) == 0 or len(distances_minimums['right']['min_frames']) == 0:
        return {'error': 'No data detected'}


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
                'all': side_distances_minimums,
                'all_distances': distance_data

            }
    statistics['symmetry_score'] = helper.calc_symmetry(statistics['left'], statistics['right'])
        
    return statistics


def extract_heel_to_toe_biomarkers(landmarks, output_dir, filename):

    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle"])

    
    biomarkers = {}
    htt_distances = local_minimum_distances_statistics(left_heel, right_heel, left_toe, right_toe)
    knee_biomarkers = sw.knee_angles_statistics(left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle)

    biomarkers["htt_distances_left"] = htt_distances['left']
    biomarkers["htt_distances_right"] = htt_distances['right']
    biomarkers["htt_distances_symmetry"] = htt_distances['symmetry_score']


    biomarkers['knee_angles_left'] = knee_biomarkers['left']
    biomarkers['knee_angles_right'] = knee_biomarkers['right']
    biomarkers['knee_symmetry'] = knee_biomarkers['symmetry_score']

    #helper.plot_biomarkers(biomarkers, "heel_to_toe")
    save_biomarkers_json(biomarkers, output_dir, filename)
    return biomarkers
