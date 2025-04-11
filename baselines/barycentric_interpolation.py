import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path
import importlib

from model.dataset import get_sample_coords
from preprocessing.hrtf_sphere import HRTF_Sphere
from preprocessing.utils import interpolate_fft
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates

def run_barycentric_interpolation(config, barycentric_output_path):
    valid_data_path = glob.glob('%s/%s_*' % (config.valid_target_path, config.dataset))
    valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_path]

    # Clear/Create directory
    shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
    Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    data_dir = config.raw_hrtf_dir / config.dataset
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                         'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    row_angles = ds.row_angles
    column_angles = ds.column_angles
    full_size = (len(row_angles), len(column_angles))
    mask = ds[0]['features'].mask
    whole_sphere = HRTF_Sphere(mask=mask, row_angles=row_angles, column_angles=column_angles)
    sphere_coords = whole_sphere.get_sphere_coords()
    with open("bary_log.txt", "a") as f:
        f.write(f"num coords: {len(sphere_coords)}\n")

    selected_coords = get_sample_coords(config.upscale_factor)
    selected_rows = sorted(list(set(coord[0] for coord in selected_coords)))
    selected_cols = sorted(list(set(coord[1] for coord in selected_coords)))
    # num_selected_rows = len(set(coord[0] for coord in selected_coords))
    # num_selected_cols = len(set(coord[1] for coord in selected_coords))

    nbins = config.nbins_hrtf * 2
    num_file = 0
    for file_name in valid_data_file_names:
        with open(config.valid_target_path + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        sphere_coords_lr = []
        sphere_coords_lr_index = []
        num_file += 1
        print("file opened: ", num_file)

        # initialize an empty lr_hrtf
        lr_hrtf = torch.zeros(1, len(selected_rows), len(selected_cols), nbins)
        for coord in selected_coords:
            elevation = column_angles[coord[1]] * np.pi / 180
            azimuth = row_angles[coord[0]] * np.pi / 180
            sphere_coords_lr.append((elevation, azimuth))
            row_idx = selected_rows.index(coord[0])
            col_idx = selected_cols.index(coord[1])
            sphere_coords_lr_index.append((col_idx, row_idx))
            lr_hrtf[:, row_idx, col_idx] = hr_hrtf[:, coord[0], coord[1]]

        euclidean_sphere_triangles = []
        euclidean_sphere_coeffs = []
        for sphere_coord in sphere_coords:
            triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                      sphere_coords=sphere_coords_lr)
            coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                  closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)

        lr_sphere = HRTF_Sphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

        lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]  
        lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]
        barycentric_hr_left = interpolate_fft(config, lr_sphere, lr_hrtf_left, full_size, sphere_coords,
                                              euclidean_sphere_triangles,euclidean_sphere_coeffs)
        barycentric_hr_right = interpolate_fft(config, lr_sphere, lr_hrtf_right, full_size, sphere_coords,
                                               euclidean_sphere_triangles, euclidean_sphere_coeffs)
        barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3)).permute(1, 2, 0, 3) # w x h x r x nbins
        with open(barycentric_output_path + file_name, "wb") as file:
            pickle.dump(barycentric_hr_merged, file)
        print('Created barycentric baseline %s' % file_name.replace('/', ''))
    return sphere_coords