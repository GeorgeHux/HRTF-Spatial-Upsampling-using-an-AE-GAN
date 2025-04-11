import torch
import os
import shutil
from pathlib import Path
import numpy as np
import pickle

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import random_split

from model.dataset import CUDAPrefetcher, CPUPrefetcher, MergeHRTFDataset
from SHT_HRTF.SHT_HRTF.model.dataset import get_sample_coords

import importlib

def compute_sh_degree(config):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    degree = max(1, int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)) 
    return degree

def load_hrtf(config, mean=None, std=None):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)

    id_file_dir = config.train_val_id_dir
    id_filename = id_file_dir + '/train_val_id.pickle'
    with open(id_filename, "rb") as file:
        train_ids, val_ids = pickle.load(file)

    # define transforms
    if mean is None or std is None:
        transform = None
    else:
        transform = (mean, std)

    domain = config.domain
    max_degree = config.max_degree

    left_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': domain}},
                                   subject_ids=train_ids)
    right_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': domain}},
                                subject_ids=train_ids)
    left_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': domain}},
                                subject_ids=val_ids)
    right_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': domain}},
                                subject_ids=val_ids)
    train_dataset = MergeHRTFDataset(left_train, right_train, config.upscale_factor, max_degree=max_degree, transform=transform)
    val_dataset = MergeHRTFDataset(left_val, right_val, config.upscale_factor, max_degree=max_degree, transform=transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(val_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    
    # Place all data on the preprocessing data loader
    if torch.cuda.is_available() and config.ngpu > 0:
        device = torch.device(config.device_name)
        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)
    return train_prefetcher, test_prefetcher


def progress(i, batches, n, num_epochs, timed):
    """Prints progress to console

    :param i: Batch index
    :param batches: total number of batches
    :param n: Epoch number
    :param num_epochs: Total number of epochs
    :param timed: Time per batch
    """
    message = 'batch {} of {}, epoch {} of {}'.format(i, batches, n, num_epochs)
    print(f"Progress: {message}, Time per iter: {timed}")


def spectral_distortion_inner(input_spectrum, target_spectrum, domain="magnitude"):
    numerator = target_spectrum
    denominator = input_spectrum
    if domain == "magnitude":
        return torch.mean((20 * torch.log10(numerator / denominator)) ** 2)
    else:
        return torch.mean((numerator - denominator) ** 2)

# def spectral_distortion_inner(input_spectrum, target_spectrum, domain="magnitude"):
#     epsilon = 1e-8  # Small number to prevent division by zero
#     numerator = target_spectrum
#     denominator = input_spectrum + epsilon  # Add epsilon to avoid division by zero
#
#     if domain == "magnitude":
#         # Calculating the logarithmic difference, safely
#         log_difference = 20 * torch.log10(torch.abs(numerator / denominator))
#         return torch.mean(log_difference ** 2)
#     else:
#         return torch.mean((numerator - denominator) ** 2)

    

def spectral_distortion_metric(generated, target, reduction='mean', domain="magnitude"):
    """Computes the mean spectral distortion metric for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    width = generated.size(3)
    height = generated.size(4)
    total_positions = num_panels * height * width
    total_sd_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for j in range(width):
                for k in range(height):
                    average_over_frequencies = spectral_distortion_inner(generated[b, :, i, j, k],
                                                                         target[b, :, i, j, k], domain)
                    total_all_positions += torch.sqrt(average_over_frequencies)
        sd_metric = total_all_positions / total_positions
        total_sd_metric += sd_metric

    if reduction == 'mean':
        output_loss = total_sd_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_sd_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss

def spectral_distortion_metric_horizontal(generated, target, reduction='mean', domain="magnitude"):
    """Computes the mean spectral distortion metric for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    width = generated.size(3)
    height = generated.size(4)
    total_positions = num_panels * height * width
    total_sd_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for j in range(width):
                average_over_frequencies = spectral_distortion_inner(generated[b, :, i, j, 4],
                                                                     target[b, :, i, j, 4], domain)
                total_all_positions += torch.sqrt(average_over_frequencies)
        sd_metric = total_all_positions / total_positions
        total_sd_metric += sd_metric

    if reduction == 'mean':
        output_loss = total_sd_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_sd_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss

def spectral_distortion_metric_vertical(generated, target, reduction='mean', domain="magnitude"):
    """Computes the mean spectral distortion metric for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    width = generated.size(3)
    height = generated.size(4)
    total_positions = num_panels * height * width
    total_sd_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for k in range(height):
                average_over_frequencies = spectral_distortion_inner(generated[b, :, i, 36, k],
                                                                     target[b, :, i, 36, k], domain)
                total_all_positions += torch.sqrt(average_over_frequencies)
        sd_metric = total_all_positions / total_positions
        total_sd_metric += sd_metric

    if reduction == 'mean':
        output_loss = total_sd_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_sd_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss

def ILD_metric_inner(config, input_spectrum, target_spectrum, domain="magnitude"):
    input_left = input_spectrum[:config.nbins_hrtf]
    input_right = input_spectrum[config.nbins_hrtf:]
    target_left = target_spectrum[:config.nbins_hrtf]
    target_right = target_spectrum[config.nbins_hrtf:]
    if domain == "magnitude":
        input_ILD = torch.mean((20 * torch.log10(input_left / input_right)))
        target_ILD = torch.mean((20 * torch.log10(target_left / target_right)))
    else:
        input_ILD = torch.mean(input_left - input_right)
        target_ILD = torch.mean(target_left - target_right)
    return torch.abs(input_ILD - target_ILD)


def ILD_metric(config, generated, target, reduction="mean"):
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    height = generated.size(3)
    width = generated.size(4)
    total_positions = num_panels * height * width
    domain = config.domain

    total_ILD_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for j in range(height):
                for k in range(width):
                    average_over_frequencies = ILD_metric_inner(config, generated[b, :, i, j, k], target[b, :, i, j, k], domain)
                    total_all_positions += average_over_frequencies
        ILD_metric_batch = total_all_positions / total_positions
        total_ILD_metric += ILD_metric_batch

    if reduction == 'mean':
        output_loss = total_ILD_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_ILD_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss

def sd_ild_loss(config, generated, target, sd_mean, sd_std, ild_mean, ild_std):
    """Computes the mean sd/ild loss for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""

    # calculate SD and ILD metrics
    sd_metric = spectral_distortion_metric(generated, target, domain=config.domain)
    ild_metric = ILD_metric(config, generated, target)
    # with open("log.txt", "a") as f:
    #     f.write(f"sd nan? {torch.isnan(sd_metric).any()}")
    #     f.write(f"ild nan? {torch.isnan(ild_metric).any()}")

    # normalize SD and ILD based on means/standard deviations passed to the function
    sd_norm = torch.div(torch.sub(sd_metric, sd_mean), sd_std)
    ild_norm = torch.div(torch.sub(ild_metric, ild_mean), ild_std)

    # add normalized metrics together
    sum_norms = torch.add(sd_norm, ild_norm)

    # un-normalize
    sum_std = (sd_std ** 2 + ild_std ** 2) ** 0.5
    sum_mean = sd_mean + ild_mean

    output = torch.add(torch.mul(sum_norms, sum_std), sum_mean)

    return output


def cos_similarity_loss(generated, target):
    # print(f"debug---lj generated: {generated.shape}, target: {target.shape}")
    # exit()
    cos_similarity_criterion = nn.CosineSimilarity(dim=2)
    avg_cos_loss_over_frequency = ((1-cos_similarity_criterion(generated, target))**2).mean(1)
    # take square root and average over batch size
    cos_loss = torch.sqrt(avg_cos_loss_over_frequency).mean()
    return cos_loss