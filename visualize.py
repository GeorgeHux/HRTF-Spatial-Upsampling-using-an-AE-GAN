import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from model.model import AutoEncoder
from model.util import load_hrtf
from model.dataset import get_sample_coords
from config import Config

from pathlib import Path
import importlib
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
import matplotlib.pyplot as plt

def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * np.log10(numerator / denominator)) ** 2)

def calc_lsd(ori_hrtf, recon_hrtf, domain):
    total_all_positions = 0
    total_positions = len(recon_hrtf)
    lsd_list = []
    for ori, gen in zip(ori_hrtf, recon_hrtf):
        if domain == 'magnitude_db':
            ori = 10 ** (ori/20)
            gen = 10 ** (gen/20)
        average_over_frequencies = spectral_distortion_inner(abs(gen), abs(ori))
        total_all_positions += np.sqrt(average_over_frequencies)
        lsd_list.append(np.sqrt(average_over_frequencies))
    sd_metric = total_all_positions / total_positions
    print('Log SD (across all positions): %s' % float(sd_metric))
    return np.array(lsd_list)

def replace_lsd(lsd_arr, upscale_factor):
    lsd_2d = lsd_arr.reshape(72, 12)
    selected_coords = get_sample_coords(upscale_factor)
    for coord in selected_coords:
        lsd_2d[coord[0], coord[1], :] = 0
    return lsd_2d


def plot_3d_lsd_combined(lsd_2d_1, lsd_2d_2, row_angles, column_angles, titles, filename):
    row_indices, col_indices = np.meshgrid(row_angles, column_angles)

    fig = plt.figure(figsize=(7, 12))

    max_z = max(max(lsd_2d_1), max(lsd_2d_2))

    # Plotting the first surface
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.plot_surface(row_indices, col_indices, lsd_2d_1.T, cmap='OrRd', edgecolor='none', antialiased=True)
    ax1.set_xlabel('Azimuth (degree)')
    ax1.set_ylabel('Elevation (degree)')
    ax1.set_zlabel('Average LSD Error')
    ax1.set_title(titles[0])
    ax1.set_zlim(0, max_z)

    # Plotting the second surface
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(row_indices, col_indices, lsd_2d_2.T, cmap='OrRd', edgecolor='none', antialiased=True)
    ax2.set_xlabel('Azimuth (degree)')
    ax2.set_ylabel('Elevation (degree)')
    ax2.set_zlabel('Average LSD Error')
    ax2.set_title(titles[1])
    ax2.set_zlim(0, max_z)

    plt.tight_layout()
    plt.savefig(filename)

print("start visualize")
config = Config(False)
config.upscale_factor = 32
data_dir = config.raw_hrtf_dir / config.dataset
imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)
domain = config.domain
ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                        'side': 'left', 'domain': domain}}, subject_ids='first')
row_angles = list(ds.row_angles)
column_angles = list(ds.column_angles)
num_row_angles = len(ds.row_angles)
num_col_angles = len(ds.column_angles)
num_radii = len(ds.radii)
max_degree = config.max_degree
upscale_factor = config.upscale_factor
degree = max(1, int(np.sqrt(num_row_angles*num_col_angles*num_radii/upscale_factor) - 1))
print("domain: ", domain, "upscale factor: ", upscale_factor)

ngpu = config.ngpu

nbins = config.nbins_hrtf
if config.merge_flag:
    nbins = config.nbins_hrtf * 2

device = torch.device(config.device_name if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")
model = AutoEncoder(nbins=nbins, in_degree=degree, latent_dim=config.latent_dim, base_channels=256, out_degree=max_degree)
print("Build model successfully")
model.load_state_dict(torch.load(f"{config.model_path}/{upscale_factor}/Gen.pt", map_location=torch.device("cpu")))
print(f"Load model weight '{os.path.abspath(config.model_path)}' successfully.")

_, test_prefetcher = load_hrtf(config)
test_prefetcher.reset()
batch_data = test_prefetcher.next()
lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                 non_blocking=True, dtype=torch.float)
hrtf = batch_data["hrtf"]
masks = batch_data["mask"]
sample_id = batch_data["id"].item()
print("subject: ", sample_id)

model.eval()
with torch.no_grad():
    recon = model(lr_coefficient)
original_mask = masks[0].numpy().astype(bool)
SHT = SphericalHarmonicsTransform(max_degree, ds.row_angles, ds.column_angles, ds.radii, original_mask)
harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
recon_hrtf = harmonics @ recon[0].T
ori_hrtf = hrtf[0].reshape(nbins, -1).T
lsd_arr = calc_lsd(ori_hrtf, recon_hrtf, domain='magnitude_db')
lsd_2d = replace_lsd(lsd_arr, upscale_factor)

# for barycentric interpolation
file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
with open(config.valid_target_path + file_name, "rb") as f:
    hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
with open(barycentric_output_path + file_name, "rb") as f:
    bary_hrtf = pickle.load(f)   # w x h x r x nbins
hr_hrtf = hr_hrtf.reshape(-1, nbins)
bary_hrtf = bary_hrtf.reshape(-1, nbins)
lsd_arr_bary = calc_lsd(hr_hrtf, bary_hrtf, domain="magnitude")
lsd_2d_bary = replace_lsd(lsd_arr_bary, upscale_factor)
filename = f"/Users/lijian/Downloads/icl/IndividualP/DAF/lsd_{upscale_factor}.png"
titles = ["LSD for AE-GAN", "LSD for barycentric interpolation"]
plot_3d_lsd_combined(lsd_2d, lsd_2d_bary, row_angles, column_angles, titles, filename)