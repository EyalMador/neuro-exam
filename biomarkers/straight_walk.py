import numpy as np
from scipy.signal import savgol_filter, find_peaks
import biomarkers.helper as helper
import landmarks.name_conventions as lnc
from biomarkers.helper import save_biomarkers_json
import matplotlib.pyplot as plt


def stride_length(heel):

    if not heel or len(heel) < 10:
        return []
    
    frames = sorted(heel.keys())
    x_positions = np.array([heel[f]['x'] for f in frames])
    y_positions = np.array([heel[f]['y'] for f in frames])
    y_smoothed = smooth_signal(y_positions)
    
    heel_strike_indices = find_heel_strikes(y_smoothed) #datapoints when heel touch the ground
    
    if len(heel_strike_indices) < 2:
        return []
    
    heel_strike_x_positions = [x_positions[i] for i in heel_strike_indices] #patient walks over x axis
    
    # calc stride:
    strides = []
    for i in range(len(heel_strike_x_positions) - 1):
        x1 = heel_strike_x_positions[i]
        x2 = heel_strike_x_positions[i + 1]
        stride = abs(x2 - x1)
        strides.append(stride)
    
    return strides


def smooth_signal(y_values, window_length=11, polyorder=3):

    if len(y_values) < window_length:
        window_length = len(y_values) if len(y_values) % 2 == 1 else len(y_values) - 1
    
    if window_length % 2 == 0:
        window_length += 1
    
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    
    return savgol_filter(y_values, window_length, polyorder)


def stride_regularity(strides):
    if not strides or len(strides) < 2:
        return 0.0
    
    strides_array = np.array(strides)
    cv = np.std(strides_array) / np.mean(strides_array)
    
    # Convert CV to 0-1 score (lower CV = higher regularity)
    regularity_score = max(0.0, 1.0 - cv)
    return regularity_score


def stride_symmetry(left_strides, right_strides):
    """Calculate symmetry between left and right leg strides."""
    
    if not left_strides or not right_strides:
        return 0.0
    
    left_mean = np.mean(left_strides)
    right_mean = np.mean(right_strides)
    
    average = (left_mean + right_mean) / 2
    
    if average == 0:
        return 0.0
    
    symmetry_index = abs(left_mean - right_mean) / average * 100
    symmetry_score = max(0.0, 1.0 - (symmetry_index / 10.0))
    
    return symmetry_score

def stride_statistics(left_heel, right_heel):
    strides = {}
    statistics = {}
    for side, heel in [('left', left_heel), ('right', right_heel)]:
        strides[side] = stride_length(heel)

    statistics['stride_symetry'] = stride_symmetry(strides["left"], strides["right"])
    statistics['left_regularity'] = stride_regularity(strides["left"])
    statistics['right_regularity'] = stride_regularity(strides["right"])
    return statistics
    



def find_heel_strikes(y_smoothed, prominence=10, distance=5):

    y_inverted = -y_smoothed #to find local minimums
    
    # Find peaks in inverted signal = find minima in original signal
    peaks, properties = find_peaks(y_inverted, prominence=prominence, distance=distance)
    
    return peaks



def plot_heel_strikes(left_result, right_result, title="Gait Analysis"):
    """Plot heel positions and detected heel strikes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # LEFT HEEL - Raw data with smoothed overlay
    ax = axes[0, 0]
    if left_result:
        strides, y_smooth, indices, frames, x_pos, y_pos = left_result
        frames_arr = np.array(frames)
        
        ax.plot(frames_arr, y_pos, 'o-', alpha=0.3, label='Raw Y (noisy)', markersize=3)
        ax.plot(frames_arr, y_smooth, 'b-', linewidth=2, label='Smoothed Y')
        ax.plot(frames_arr[indices], y_smooth[indices], 'r*', markersize=15, label='Heel strikes')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position (height)')
        ax.set_title(f'LEFT HEEL - Found {len(strides)} strides')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # RIGHT HEEL - Raw data with smoothed overlay
    ax = axes[0, 1]
    if right_result:
        strides, y_smooth, indices, frames, x_pos, y_pos = right_result
        frames_arr = np.array(frames)
        
        ax.plot(frames_arr, y_pos, 'o-', alpha=0.3, label='Raw Y (noisy)', markersize=3)
        ax.plot(frames_arr, y_smooth, 'g-', linewidth=2, label='Smoothed Y')
        ax.plot(frames_arr[indices], y_smooth[indices], 'r*', markersize=15, label='Heel strikes')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position (height)')
        ax.set_title(f'RIGHT HEEL - Found {len(strides)} strides')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # STRIDES comparison
    ax = axes[1, 0]
    if left_result and right_result:
        left_strides = left_result[0]
        right_strides = right_result[0]
        
        x_pos_plot = np.arange(max(len(left_strides), len(right_strides)))
        
        if left_strides:
            ax.plot(range(len(left_strides)), left_strides, 'bo-', label='Left strides', linewidth=2, markersize=8)
        if right_strides:
            ax.plot(range(len(right_strides)), right_strides, 'go-', label='Right strides', linewidth=2, markersize=8)
        
        ax.set_xlabel('Stride Number')
        ax.set_ylabel('Stride Length (pixels)')
        ax.set_title('Stride Lengths Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # BIOMARKERS summary
    ax = axes[1, 1]
    ax.axis('off')
    
    if left_result and right_result:
        left_strides = left_result[0]
        right_strides = right_result[0]
        
        left_reg = stride_regularity(left_strides)
        right_reg = stride_regularity(right_strides)
        symmetry = stride_symmetry(left_strides, right_strides)
        
        text = f"""
BIOMARKERS:

LEFT STRIDE:
  Count: {len(left_strides)}
  Mean: {np.mean(left_strides):.1f} px
  Std: {np.std(left_strides):.1f} px
  Regularity: {left_reg:.3f}

RIGHT STRIDE:
  Count: {len(right_strides)}
  Mean: {np.mean(right_strides):.1f} px
  Std: {np.std(right_strides):.1f} px
  Regularity: {right_reg:.3f}

SYMMETRY:
  Score: {symmetry:.3f}
  (1.0 = perfect, 0.0 = asymmetric)
        """
        
        ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='center')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()




def extract_straight_walk_biomarkers(landmarks, output_dir, filename, fps=30):
    rtm_names_landmarks = helper.indices_to_names(landmarks, lnc.rtm_mapping())
    [left_heel, right_heel, left_toe, right_toe, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle, head] = helper.extract_traj(rtm_names_landmarks,["LHeel", "RHeel", "LBigToe", "RBigToe", "LKnee", "Rknee", "LHip", "RHip", "LAnkle", "RAnkle", "Head"])
    biomarkers = {}
    stride_stat = stride_statistics(left_heel, right_heel)
    biomarkers['stride_symetry'] = stride_stat['stride_symetry']
    biomarkers['left_regularity'] = stride_stat["left_regularity"]
    biomarkers['right_regularity'] = stride_stat["right_regularity"]
    print(f"symetry: {biomarkers['stride_symetry']}, left stride: {biomarkers['left_regularity']}, right stride:{biomarkers['right_regularity']}")
    plot_heel_strikes(biomarkers['left_regularity'], biomarkers['right_regularity'])
    save_biomarkers_json(biomarkers, output_dir, filename)
    return biomarkers