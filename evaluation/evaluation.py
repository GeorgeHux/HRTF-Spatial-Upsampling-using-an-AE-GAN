from SHT_HRTF.SHT_HRTF.model.util import spectral_distortion_metric, spectral_distortion_metric_horizontal, spectral_distortion_metric_vertical
from preprocessing.utils import convert_to_sofa

import shutil
from pathlib import Path
import importlib

import glob
import torch
import pickle
import os
import re
import numpy as np
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import load_data as ld
import spatialaudiometrics
from spatialaudiometrics import visualisation as vis
from spatialaudiometrics import hrtf_metrics as hf
from model.dataset import get_sample_coords
import matlab.engine

def replace_nodes(config, sr_dir, file_name):
    with open(config.valid_target_path + file_name, "rb") as f:
        hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)   # w x h x r x nbins

    selected_coords = get_sample_coords(config.upscale_factor)
    for coord in selected_coords:
        sr_hrtf[coord[0], coord[1], :] = hr_hrtf[coord[0], coord[1], :]

    generated = torch.permute(sr_hrtf[None, :], (0, 4, 3, 1, 2)) # 1 x nbins x r x w x h
    target = torch.permute(hr_hrtf[None, :], (0, 4, 3, 1, 2))

    return target, generated

def run_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_target_path, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
        # Overwrite the generated points that exist in the original data
            with open(config.valid_target_path + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{sr_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3)) 
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))  # 1 x nbins x r x w x h

            error = spectral_distortion_metric(generated, target)

            if not error.ndim == 0:
                print(f"Error is not a scalar value for file {file_name}")
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
        with open(f'{sr_dir}/{file_ext}', "wb") as file:
            pickle.dump(lsd_errors, file)
    else:
        val_data_paths = glob.glob(f"{sr_dir}/{config.dataset}_*")
        val_data_file_names = ['/' + os.path.basename(x) for x in val_data_paths]
        "lsd for horizontal"
        lsd_errors = []
        for file_name in val_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)
            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id, float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))

        with open(f"{sr_dir}/{file_ext}", "wb") as file:
            pickle.dump(lsd_errors, file)

        print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
        with open('log.txt', 'a') as f:
            f.write('Mean LSD Error: %0.3f \n' % np.mean([error[1] for error in lsd_errors]))

    #     lsd_errors = []
    #     for file_name in val_data_file_names:
    #         target, generated = replace_nodes(config, sr_dir, file_name)
    #         error = spectral_distortion_metric(generated, target)
    #         subject_id = ''.join(re.findall(r'\d+', file_name))
    #         lsd_errors.append([subject_id,  float(error.detach())])
    #         print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    #
    #     # with open(f'{config.valid_recon_path}/{config.upscale_factor}/mag/{file_ext}', "wb") as file:
    #     # with open(f'{config.path}/{config.upscale_factor}/{file_ext}', "wb") as file:
    #     with open(f"{sr_dir}/{file_ext}", "wb") as file:
    #         pickle.dump(lsd_errors, file)
    # print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    # with open('log.txt', 'a') as f:
    #     f.write('Mean LSD Error: %0.3f \n' % np.mean([error[1] for error in lsd_errors]))
    # return np.mean([error[1] for error in lsd_errors])


    

# def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
#     imp = importlib.import_module('hrtfdata.full')
#     load_function = getattr(imp, config.dataset)
#     data_dir = config.raw_hrtf_dir / config.dataset
#     ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
#                                                          'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
#     row_angles = ds.row_angles
#     column_angles = ds.column_angles
#
#     file_ext = 'loc_errors.pickle' if file_ext is None else file_ext
#
#     if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
#         nodes_replaced_path = sr_dir
#         hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_target_path + '/sofa_min_phase')]
#     else:
#         sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
#         sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]
#
#         # Clear/Create directories
#         nodes_replaced_path = sr_dir + '/nodes_replaced'
#         shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
#         Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)
#
#         for file_name in sr_data_file_names:
#             target, generated = replace_nodes(config, sr_dir, file_name)
#
#             with open(nodes_replaced_path + file_name, "wb") as file:
#                 pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file) # r x w x h x nbins
#
#         convert_to_sofa(nodes_replaced_path, config, row_angles, column_angles)
#         print('Created valid sofa files')
#
#         directory_path = r'D:\PycharmProjects\Upsample_GAN\SHT_HRTF_localisation\SHT_HRTF\data\SONICOM\valid_target\sofa_min_phase'
#         sofa_dirs = r'D:\PycharmProjects\Upsample_GAN\SHT_HRTF_localisation\SHT_HRTF\results\valid_recon\108\mag\nodes_replaced\sofa_min_phase'
#
#         # List all files in the target directory
#         target_files = os.listdir(directory_path)
#         sofa_files = os.listdir(sofa_dirs)
#         itd_diff_total = 0
#         ild_diff_total = 0
#
#         # Iterate over files and compute differences
#         for sofa_file, target_file in zip(sofa_files, target_files):
#             sofa_file_path = os.path.join(sofa_dirs, sofa_file)
#             target_file_path = os.path.join(directory_path, target_file)
#             print(f'sofa_file: {sofa_file}, target_file: {target_file}')
#
#             if not os.path.isfile(sofa_file_path) or not os.path.isfile(target_file_path):
#                 print(f"One of the files {sofa_file_path} or {target_file_path} does not exist. Skipping.")
#                 continue
#
#             try:
#                 target = ld.HRTF(target_file_path)
#                 generated = ld.HRTF(sofa_file_path)
#                 target, generated = ld.match_hrtf_locations(target, generated)
#                 itd_diff = hf.calculate_itd_difference(target, generated)
#                 ild_diff = hf.calculate_ild_difference(target, generated)
#                 lsd, lsd_mat = hf.calculate_lsd_across_locations(target.hrir, generated.hrir, target.fs)
#                 itd_diff_total += itd_diff
#                 ild_diff_total += ild_diff
#             except Exception as e:
#                 print(f"Error processing files {sofa_file_path} and {target_file_path}: {e}")
#                 continue
#
#     # Optionally, print or save the results
#     print('Mean ITD Differences:', itd_diff_total / len(sofa_files))
#     print('Mean ILD Differences:', ild_diff_total / len(sofa_files))
#
#     sofa_file_path = os.path.join(sofa_dirs, "SONICOM_148.sofa")
#     generated = ld.HRTF(sofa_file_path)
#     target_file_path = os.path.join(directory_path, "SONICOM_10.sofa")
#     target1 = ld.HRTF(target_file_path)
#     target1, generated = ld.match_hrtf_locations(target1, generated)
#     lsd = hf.calculate_lsd_across_locations(target1.hrir, generated.hrir, target1.fs)
#     print(lsd)
#     target_file_path = os.path.join(directory_path, "SONICOM_148.sofa")
#     target2 = ld.HRTF(target_file_path)
#     target2, generated = ld.match_hrtf_locations(target2, generated)
#     lsd = hf.calculate_lsd_across_locations(target2.hrir, generated.hrir, target2.fs)
#     print(lsd)
#     target_file_path = os.path.join(directory_path, "SONICOM_179.sofa")
#     target3 = ld.HRTF(target_file_path)
#     target3, generated = ld.match_hrtf_locations(target3, generated)
#     lsd = hf.calculate_lsd_across_locations(target3.hrir, generated.hrir, target3.fs)
#     print(lsd)
#     target1, target2 = ld.match_hrtf_locations(target1, target2)
#     lsd = hf.calculate_lsd_across_locations(target1.hrir, target2.hrir, target1.fs)
#     print(lsd)
#     target3, target2 = ld.match_hrtf_locations(target3, target2)
#     lsd = hf.calculate_lsd_across_locations(target3.hrir, target2.hrir, target3.fs)
#     print(lsd)

def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    data_dir = config.raw_hrtf_dir / config.dataset
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    row_angles = ds.row_angles
    column_angles = ds.column_angles

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        nodes_replaced_path = sr_dir
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_target_path + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file) # r x w x h x nbins

        convert_to_sofa(nodes_replaced_path, config, row_angles, column_angles)
        print('Created valid sofa files')
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path + '/sofa_min_phase')]

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dir_path)
    eng.addpath(s, nargout=0)

    file_path = config.path
    if not os.path.exists(file_path):
        raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

    loc_errors = []
    for file in hrtf_file_names:
        target_sofa_file = config.valid_target_path + '/sofa_min_phase/' + file
        if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
            generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
        else:
            generated_sofa_file = nodes_replaced_path + '/sofa_min_phase/' + file

        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{file_path}/{file_ext}', "wb") as file:
        pickle.dump(loc_errors, file)

def run_target_localisation_evaluation(config):

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    file_path = f'{config.data_dirs_path}{config.data_dir}'
    if not os.path.exists(file_path):
        raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

    loc_target_errors = []
    target_sofa_path = config.valid_hrtf_merge_dir + '/sofa_min_phase'
    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(target_sofa_path)]
    for file in hrtf_file_names:
        target_sofa_file = target_sofa_path + '/' + file
        generated_sofa_file = target_sofa_file
        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_target_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_target_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_target_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_target_errors]))
    with open(f'{file_path}/{config.dataset}_loc_target_valid_errors.pickle', "wb") as file:
        pickle.dump(loc_target_errors, file)

def replace_single_nodes(config, sr_dir, file_name):

    with open(config.valid_target_path + file_name, "rb") as f:
        hr_hrtf1 = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

    with open(sr_dir + file_name, "rb") as f:
        hr_hrtf = pickle.load(f)   # w x h x r x nbins

    with open(sr_dir + "/SONICOM_175.pickle", "rb") as f:
        sr_hrtf = pickle.load(f)   # w x h x r x nbins

    selected_coords = get_sample_coords(config.upscale_factor)
    # common
    # for coord in selected_coords:
    #     sr_hrtf[coord[0], coord[1], :] = hr_hrtf[coord[0], coord[1], :]
    for coord in selected_coords:
        sr_hrtf[coord[0], coord[1], :] = hr_hrtf1[coord[0], coord[1], :]
        hr_hrtf[coord[0], coord[1], :] = hr_hrtf1[coord[0], coord[1], :]


    generated = torch.permute(sr_hrtf[None, :], (0, 4, 3, 1, 2)) # 1 x nbins x r x w x h
    target = torch.permute(hr_hrtf[None, :], (0, 4, 3, 1, 2))

    return target, generated

def run_single_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):
    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_target_path, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
        # Overwrite the generated points that exist in the original data
            with open(config.valid_target_path + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{sr_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))  # 1 x nbins x r x w x h

            error = spectral_distortion_metric(generated, target)

            if not error.ndim == 0:
                print(f"Error is not a scalar value for file {file_name}")
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
        with open(f'{sr_dir}/{file_ext}', "wb") as file:
            pickle.dump(lsd_errors, file)
    else:
        val_data_paths = glob.glob(f"{sr_dir}/{config.dataset}_*")
        val_data_file_names = ['/' + os.path.basename(x) for x in val_data_paths]
        lsd_errors = []
        for file_name in val_data_file_names:
            target, generated = replace_single_nodes(config, sr_dir, file_name)
            print(generated)
            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))

        # with open(f'{config.valid_recon_path}/{config.upscale_factor}/mag/{file_ext}', "wb") as file:
        # with open(f'{config.path}/{config.upscale_factor}/{file_ext}', "wb") as file:
        with open(f"{sr_dir}/{file_ext}", "wb") as file:
            pickle.dump(lsd_errors, file)
    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open('log.txt', 'a') as f:
        f.write('Mean LSD Error: %0.3f \n' % np.mean([error[1] for error in lsd_errors]))
    return np.mean([error[1] for error in lsd_errors])
