import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import norm
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

# row = [-180., -175., -170., -165., -160., -155., -150., -145., -140., -135., -130., -125.,
#  -120., -115., -110., -105., -100.,  -95.,  -90.,  -85.,  -80.,  -75.,  -70.,  -65.,
#   -60.,  -55.,  -50.,  -45.,  -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,
#     0.  ,  5.  , 10. ,  15.  , 20. ,  25.  , 30.  , 35.  , 40. ,  45.,   50.,   55.,
#    60. ,  65.  , 70. ,  75. ,  80. ,  85.  , 90.  , 95. , 100. , 105.,  110. , 115.,
#   120. , 125. , 130. , 135. , 140. , 145. , 150. , 155. , 160.,  165.,  170.,  175.,]
# col = [-45., -30., -20., -10.,   0.,  10.,  20.,  30.,  45.,  60.,  75.,  90.]

def get_sample_coords(upscale_factor):
    if upscale_factor == 4: # 216 points
        row_idx = [0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 66, 68, 70, 71]
        col_idx = [2, 5, 8, 11]    #[-45.0, 0.0, 45.0]`
        return [(i, j) for i in row_idx for j in col_idx]

    if upscale_factor == 8: # 108 points
        row_idx = [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28, 32, 34, 36, 40, 42, 44, 48, 50, 52, 56, 58, 60, 64, 66, 68]
        col_idx = [2, 5, 8, 11]    #[-45.0, 0.0, 45.0]`
        return [(i, j) for i in row_idx for j in col_idx]

    if upscale_factor == 16:  # 54 points
        row_idx = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68]
        col_idx = [0, 4, 8]    #[-45.0, 0.0, 45.0]
        return [(i, j) for i in row_idx for j in col_idx]

    if upscale_factor == 32:  # 27 points
        row_idx = [0, 8, 16, 24, 32, 40, 48, 56, 64] #[-180.0, -140.0, -100.0, -60.0, -20.0, 20.0, 60.0, 100.0, 140.0]
        col_idx = [0, 4, 8]    #[-45.0, 0.0, 45.0]
        return [(i, j) for i in row_idx for j in col_idx]
    
    if upscale_factor == 48: # 18 points
        row_idx = [0, 12, 24, 36, 48, 60] #[-180, -120, -60, 0, 60, 120]
        col_idx = [1, 4, 8]  # [-30, 0, 45]
        return [(i, j) for i in row_idx for j in col_idx]

    if upscale_factor == 72:  # 12 points
        row_idx = [0, 12, 24, 36, 48, 60] # [-180.0, -120.0, -60.0, 0.0, 60.0, 120.0]
        col_idx = [2, 8]   # [-20, 45]
        return [(i, j) for i in row_idx for j in col_idx]
    
    if upscale_factor == 108:  # 8 points
        row_idx = [0, 18, 36, 54]   # [-180.0, -90.0, 0.0, 90.0]
        col_idx = [0, 8]   # [-20, 45]
        return [(i, j) for i in row_idx for j in col_idx]

    if upscale_factor == 216:  # 4 points
        # row_idx = [18, 54] # -90， 90
        # col_idx = [2, 8]   # [-20, 45]
        # row_idx = [0, 36]    # (-180, 0)
        # col_idx = [0, 6]
        row_idx = [0, 36]    # (-180, 0)
        col_idx = [2, 9]     # (-20, 60)
        return [(i, j) for i in row_idx for j in col_idx]
    
    if upscale_factor == 288:  # 3 points
        return [(18,4), (18,5), (18,3)]   #(-120, 10), (0, 10), (120, 10)

class MergeHRTFDataset(Dataset):
    def __init__(self, left_hrtf, right_hrtf, upscale_factor, max_degree=28, transform=None, desired_snr_db=0) -> None:
        super(MergeHRTFDataset, self).__init__()
        self.left_hrtf = left_hrtf
        self.right_hrtf = right_hrtf
        self.upscale_factor = upscale_factor
        self.num_row_angles, self.num_col_angles = len(self.left_hrtf.row_angles), len(self.left_hrtf.column_angles)
        self.num_radii = len(self.left_hrtf.radii)
        self.degree = max(1, int(np.sqrt(self.num_row_angles*self.num_col_angles*self.num_radii/upscale_factor) - 1)) 
        self.max_degree = max_degree
        self.transform = transform
        self.selected_coords = get_sample_coords(self.upscale_factor)
        self.desired_snr_db = desired_snr_db

    def __getitem__(self, index: int):
        left = self.left_hrtf[index]['features'][:, :, :, 1:]
        right = self.right_hrtf[index]['features'][:, :, :, 1:]
        sample_id = self.left_hrtf.subject_ids[index]
        merge_clean = np.ma.concatenate([left, right], axis=3)
        original_mask = np.all(np.ma.getmaskarray(left), axis=3)
        mask = np.ones((self.num_row_angles, self.num_col_angles, self.num_radii), dtype=bool)
        for coord in self.selected_coords:
            mask[coord[0], coord[1], :] = original_mask[coord[0], coord[1], :]
        lr_SHT = SphericalHarmonicsTransform(self.degree, self.left_hrtf.row_angles,
                                             self.left_hrtf.column_angles,
                                             self.left_hrtf.radii,
                                             mask)
        lr_coefficient = torch.from_numpy(lr_SHT(merge_clean).T)
        hr_SHT = SphericalHarmonicsTransform(self.max_degree, self.left_hrtf.row_angles,
                                             self.left_hrtf.column_angles,
                                             self.left_hrtf.radii,
                                             original_mask)
        hr_coefficient = torch.from_numpy(hr_SHT(merge_clean).T)

        if self.transform is not None:
            mean_lr, mean_full = self.transform[0]
            std_lr, std_full = self.transform[1]
            lr_coefficient = (lr_coefficient - mean_lr) / std_lr
            hr_coefficient = (hr_coefficient - mean_full) / std_full

        merge = torch.from_numpy(merge_clean.data).permute(3, 2, 0, 1)  # nbins x r x w x h
        # noise = torch.from_numpy(merge_noise.data).permute(3, 2, 0, 1)  # nbins x r x w x h
        return {"lr_coefficient": lr_coefficient, "hr_coefficient": hr_coefficient,
                "hrtf": merge, "mask": original_mask, "id": sample_id}
    
    def __len__(self):
        return len(self.left_hrtf)
    

class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)
    

class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v) and k != 'mask' and k != 'id':
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)