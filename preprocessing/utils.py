import cmath
import pickle
import os

import sofar as sf
import numpy as np
import torch
import pandas as pd
import scipy
from scipy.signal import hilbert
import shutil
from pathlib import Path
import re

from preprocessing.barycentric_calcs import calc_barycentric_coordinates, get_triangle_vertices
from preprocessing.KalmanFilter import KalmanFilter

def add_itd(az, el, hrir, side, fs=48000, r=0.0875, c=343):

    az = np.radians(az)
    el = np.radians(el)
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))
    delay_in_sec = (r / c) * (interaural_azimuth + np.sin(interaural_azimuth))
    fractional_delay = delay_in_sec * fs

    sample_delay = int(abs(fractional_delay))

    if (delay_in_sec > 0 and side == 'right') or (delay_in_sec < 0 and side == 'left'):
        N = len(hrir)
        delayed_hrir = np.zeros(N)
        delayed_hrir[sample_delay:] = hrir[0:N - sample_delay]
        sofa_delay = sample_delay
    else:
        sofa_delay = 0
        delayed_hrir = hrir

    return delayed_hrir, sofa_delay

def gen_sofa_file(config, left_hrtf, right_hrtf, az, el):
    source_position = [az + 360 if az < 0 else az, el, 1.5]

    left_hrtf[left_hrtf == 0.0] = 1.0e-08
    left_phase = np.imag(-hilbert(np.log(np.abs(left_hrtf))))
    right_hrtf[right_hrtf == 0.0] = 1.0e-08
    right_phase = np.imag(-hilbert(np.log(np.abs(right_hrtf))))

    left_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(left_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * left_phase))[:config.nbins_hrtf]
    right_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(right_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * right_phase))[:config.nbins_hrtf]

    left_hrir, left_sample_delay = add_itd(az, el, left_hrir, side='left')
    right_hrir, right_sample_delay = add_itd(az, el, right_hrir, side='right')

    full_hrir = [left_hrir, right_hrir]
    delay = [left_sample_delay, right_sample_delay]

    return source_position, full_hrir, delay

def save_sofa(clean_hrtf, config, row_angles, column_angles, sofa_path_output):
    full_hrirs = []
    source_positions = []
    delays = []
    left_full_hrtf = clean_hrtf[:, :, :, :config.nbins_hrtf]   # r x w x h x nbins
    right_full_hrtf = clean_hrtf[:, :, :, config.nbins_hrtf:]

    for i in range(clean_hrtf.size(1)):  # loop through azimuth
        for j in range(clean_hrtf.size(2)): # loop through elevation
            left_hrtf = np.array(left_full_hrtf[0, i, j]) # only one radius
            right_hrtf = np.array(right_full_hrtf[0, i, j])
            az = row_angles[i]
            el = column_angles[j]
            source_position, full_hrir, delay = gen_sofa_file(config, left_hrtf, right_hrtf, az, el)

            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)
    
    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = full_hrirs
    sofa.Data_SamplingRate = config.hrir_samplerate
    sofa.Data_Delay = delays
    sofa.SourcePosition = source_positions
    sf.write_sofa(sofa_path_output, sofa)

def trim_hrir(hrir, start, stop):
    if start < 0:
        hrir_padded = np.pad(hrir, (abs(start), 0), mode='constant')
        trimmed_hrir = hrir_padded[:stop]
    else:
        trimmed_hrir = hrir[start:stop]
    return trimmed_hrir

def remove_itd(hrir, pre_window, length):
    """Remove ITD from HRIR using kalman filter"""
    # normalize such that max(abs(hrir)) == 1
    rescaling_factor = 1 / max(np.abs(hrir))
    normalized_hrir = rescaling_factor * hrir

    # initialise Kalman filter
    x = np.array([[0]])  # estimated initial state
    p = np.array([[0]])  # estimated initial variance

    h = np.array([[1]])  # observation model (observation represents internal state directly)

    # r and q may require tuning
    r = np.array([[np.sqrt(400)]])  # variance of the observation noise
    q = np.array([[0.01]])  # variance of the process noise

    hrir_filter = KalmanFilter(x, p, h, q, r)
    f = np.array([[1]])  # F is state transition model
    for i, z in enumerate(normalized_hrir):
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        # find first time post fit residual exceeds some threshold
        if np.abs(hrir_filter.get_post_fit_residual()) > 0.005:
            over_threshold_index = i
            break
    else:
        print("RuntimeWarning: ITD not removed (Kalman filter did not find a time where post fit residual exceeded threshold).")
        return hrir

    # create fade window in order to taper off HRIR towards the beginning and end
    fadeout_len = 50
    fadeout_interval = -1. / fadeout_len
    fadeout = np.arange(1 + fadeout_interval, fadeout_interval, fadeout_interval).tolist()

    fadein_len = 10
    fadein_interval = 1. / fadein_len
    fadein = np.arange(0.0, 1.0, fadein_interval).tolist()

    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    if start < 0:
        stop = length
    else:
        stop = start + length

    if len(hrir) >= stop:
        trimmed_hrir = trim_hrir(hrir, start, stop)
        fade_window = fadein + [1] * (length - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
    else:
        trimmed_hrir = trim_hrir(hrir, start, -1)
        fade_window = fadein + [1] * (len(trimmed_hrir) - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
        zero_pad = [0] * (length - len(trimmed_hrir))
        faded_hrir = np.ma.append(faded_hrir, zero_pad)
    return faded_hrir

def convert_to_sofa(hrtf_dir, config, row_angles, column_angles, phase_ext='_phase', mag_ext='_mag'):
    sofa_path_output = hrtf_dir + '/sofa_min_phase/'

    hrtf_file_names = [hrir_file_name for hrir_file_name in os.listdir(hrtf_dir)
                       if os.path.isfile(os.path.join(hrtf_dir, hrir_file_name)) and phase_ext not in hrir_file_name]
    
    # Clear/Create directories
    shutil.rmtree(Path(sofa_path_output), ignore_errors=True)
    Path(sofa_path_output).mkdir(parents=True, exist_ok=True)
    # sofa_dir = []
    for f in hrtf_file_names:
        with open(os.path.join(hrtf_dir, f), "rb") as hrtf_file:
            hrtf = pickle.load(hrtf_file) # r x w x h x nbins
            if hrtf.shape == (72, 12, 1, 256):
                hrtf = hrtf.permute(2, 0, 1, 3)
            expected_shape = (1, 72, 12, 256) # 256
            assert hrtf.shape == expected_shape, f"Expected shape {expected_shape}, but got shape {hrtf.shape}"
            sofa_filename_output = os.path.basename(hrtf_file.name).replace('.pickle', '.sofa').replace(mag_ext, '')
            sofa_output = sofa_path_output + sofa_filename_output
            # sofa_dir.append(sofa_output)
            save_sofa(hrtf, config, row_angles, column_angles, sofa_output)

    # return sofa_dir

def get_feature_for_point_tensor(elevation, azimuth, all_coords, subject_features):
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(subject_features[0][int(all_coords_row.azimuth_index)][int(all_coords_row.elevation_index)]))))

def calc_interpolated_feature(triangle_vertices, coeffs, all_coords, subject_features):
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point_tensor(p[0], p[1], all_coords, subject_features)
        features.append(features_p)
    
    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    if len(features) == 3:
        interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]
    else:
        interpolated_feature = features[0]

    return interpolated_feature

def calc_all_interpolated_features(hrtf_sphere, features,  euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs):
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=hrtf_sphere.get_df(),
                                                   subject_features=features)
            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)
    return selected_feature_interpolated

def calc_hrtf(config, hrirs):
    """FFT to obtain HRTF from HRIR"""
    magnitudes = []
    phases = []

    for hrir in hrirs:
        # remove value that corresponds to 0 Hz
        hrtf = scipy.fft.rfft(hrir, config.nbins_hrtf*2)[1:]
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)
    return magnitudes, phases

def interpolate_fft(config, hrtf_sphere, features, full_size, sphere_coords, sphere_triangles, sphere_coeffs):
    """
    hrtf_sphere: HRTF_Sphere object associated with dataset
    features: features for a given subject, 
    sphere_coords: A list of locations of the gridded cubed sphere points to be interpolated,
                     given as (elevation, azimuth)
    sphere_triangles: A list of lists of triangle vertices for barycentric interpolation, where each list of
                             vertices defines the triangle for the corresponding point in sphere
    sphere_coeffs: A list of barycentric coordinates for each location in sphere, corresponding to the triangles
                          described by sphere_triangles
    """
    interpolated_hrirs = calc_all_interpolated_features(hrtf_sphere, features, sphere_coords, sphere_triangles, sphere_coeffs)
    magnitudes, phases = calc_hrtf(config, interpolated_hrirs)
    magnitudes_raw = [[[[] for _ in range(full_size[1])] for _ in range(full_size[0])] for _ in range(1)]
    count = 0
    for i in range(full_size[0]):
        for j in range(full_size[1]):
            magnitudes_raw[0][i][j] = magnitudes[count]
            count += 1
    
    return torch.tensor(np.array(magnitudes_raw))