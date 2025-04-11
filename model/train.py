import pickle
import scipy
import importlib

from model.util import *
from model.model import *

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import time
import torch.autograd as autograd
from torch.nn.utils import spectral_norm
from plot import plot_losses, plot_hrtf
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform


def train(config, train_prefetcher):
    """ Train the generator and discriminator models

    :param config: Config object containing model hyperparameters
    :param train_prefetcher: prefetcher for training data
    """
    # load the dataset to get the row, column angles info
    domain = config.domain
    with open(f"{config.path}/{config.upscale_factor}/log.txt", 'a') as f:
        f.write(f"domain: {domain}\n\n")
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': domain}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Assign torch device
    ngpu = config.ngpu
    path = config.path

    nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")

    print(f'Using {ngpu} GPUs')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    bs, lr_G, lr_D, latent_dim, critic_iters, max_degree = config.get_train_params()

    # Define network and transfer to CUDA
    in_degree = compute_sh_degree(config)
    netG = AutoEncoder(nbins=nbins, in_degree=in_degree, latent_dim=latent_dim,
                       base_channels=256, out_degree=max_degree).to(device)
    netD = Discriminator(nbins=nbins).to(device)

    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)

    # Define optimizers
    optD = optim.RMSprop(netD.parameters(), lr=lr_D)  # 0.00003
    optG = optim.RMSprop(netG.parameters(), lr=lr_G)

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    cos_similarity_criterion = cos_similarity_loss
    content_criterion = sd_ild_loss

    # mean and std for ILD and SD, which are used for normalization
    # computed based on average ILD and SD for training data, when comparing each individual
    # to every other individual in the training data
    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    margin = 1.8670232e-08

    real_label = 1.
    fake_label = 0.

    if config.transform_flag:
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean, std = pickle.load(f)
        mean = mean.float().to(device)
        std = std.float().to(device)

    if config.start_with_existing_model:
        print(f'Initialized weights using an existing model - {config.existing_model_path}')
        netG.load_state_dict(torch.load(f'{config.existing_model_path}/Gen.pt'))
        netD.load_state_dict(torch.load(f'{config.existing_model_path}/Disc.pt'))

    train_loss_G_list = []
    train_loss_G_adversarial_list = []
    train_loss_G_content_list = []
    train_loss_G_sh_mse_list = []
    train_loss_G_sh_cos_list = []
    train_loss_D_list = []
    train_loss_D_hr_list = []
    train_loss_D_sr_list = []

    train_SD_metric = []

    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        with open(f"{config.path}/{config.upscale_factor}/log.txt", "a") as f:
            f.write(f"\nEpoch: {epoch}\n")
        times = []
        train_loss_G = 0.
        train_loss_G_adversarial = 0.
        train_loss_G_content = 0.
        train_loss_G_sh_mse = 0.
        train_loss_G_sh_cos = 0.
        train_loss_D = 0.
        train_loss_D_hr = 0.
        train_loss_D_sr = 0.

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0

        # Initialize the data loader and load the first batch of data
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            # Transfer in-memory data to CUDA devices to speed up training
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"]

            bs = lr_coefficient.size(0)

            # Generate fake samples using autoencoder
            sr = netG(lr_coefficient)

            # Discriminator Training
            netD.zero_grad()
            # train on real coefficient
            pred_real = netD(hr_coefficient.detach().clone()).view(-1)
            label = torch.full((bs,), real_label, dtype=hr_coefficient.dtype, device=device)
            loss_D_hr = adversarial_criterion(pred_real, label)
            loss_D_hr.backward()
            # train on reconstructed coefficient
            pred_fake = netD(sr.detach().clone()).view(-1)
            label.fill_(fake_label)
            loss_D_sr = adversarial_criterion(pred_fake, label)
            loss_D_sr.backward()

            loss_D = loss_D_hr + loss_D_sr
            train_loss_D += loss_D.item()
            train_loss_D_hr += loss_D_hr.item()
            train_loss_D_sr += loss_D_sr.item()
            # Update D
            optD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            # training Autoencoder
            if batch_index % int(critic_iters) == 0:
                # train decoder
                netG.zero_grad()
                pred_fake = netD(sr).view(-1)
                label.fill_(real_label)
                adversarial_loss_G = config.adversarial_weight * adversarial_criterion(pred_fake, label)
                sh_cos_loss = cos_similarity_criterion(sr, hr_coefficient)
                sh_mse_loss = ((sr - hr_coefficient) ** 2).mean()  # sh coefficient loss
                with open(f"{config.path}/{config.upscale_factor}/log.txt", "a") as f:
                    lr0 = lr_coefficient[0].T  # num coef x nbins
                    sr0 = sr[0].T
                    hr0 = hr_coefficient[0].T
                    f.write(f"lr: {lr0.shape}, {lr0[0, :30]}\n")
                    f.write(f"sr: {sr0.shape}, {sr0[0, :30]}\n")
                    f.write(f"hr: {hr0.shape}, {hr0[0, :30]}\n")
                # convert reconstructed coefficient back to hrtf
                harmonics_list = []
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(max_degree, ds.row_angles, ds.column_angles, ds.radii,
                                                      masks[i].numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float()
                    harmonics_list.append(harmonics)
                harmonics_tensor = torch.stack(harmonics_list).to(device)
                if config.transform_flag:  # unormalize the coefficient
                    recon = recon * std + mean
                recons = (harmonics_tensor @ sr.permute(0, 2, 1)).reshape(bs, num_row_angles, num_col_angles, num_radii,
                                                                          nbins)
                recons = recons.permute(0, 4, 3, 1, 2)  # bs x nbins x r x w x h
                if domain == "magnitude":
                    recons = F.relu(recons) + margin  # filter out negative values and make it non-zero

                # during every 25th epoch and last epoch, save filename for mag spectrum plot
                if epoch % 25 == 0 or epoch == (num_epochs - 1):
                    generated = recons[0].permute(2, 3, 1, 0)  # w x h x r x nbins
                    target = hrtf[0].permute(2, 3, 1, 0)
                    id = batch_data['id'][0].item()
                    filename = f"magnitude_{id}_{epoch}"
                    plot_hrtf(generated.detach().cpu(), target.detach().cpu(), f'{path}/{config.upscale_factor}',
                              filename)

                unweighted_content_loss_G = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
                content_loss_G = config.content_weight * unweighted_content_loss_G
                # Generator total loss
                loss_G = content_loss_G + adversarial_loss_G + sh_cos_loss
                loss_G.backward()

                train_loss_G += loss_G.item()
                train_loss_G_adversarial += adversarial_loss_G.item()
                train_loss_G_content += content_loss_G.item()
                train_loss_G_sh_mse += sh_mse_loss.item()
                train_loss_G_sh_cos += sh_cos_loss.item()
                train_SD_metric.append(unweighted_content_loss_G.item())

                optG.step()

                with open(f"{config.path}/{config.upscale_factor}/log.txt", "a") as f:
                    f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                    f.write(f"dis: {loss_D.item()}\t generator: {loss_G.item()}\n")
                    f.write(f"D_real: {loss_D_hr.item()}, D_fake: {loss_D_sr.item()}\n")
                    f.write(f"content loss: {content_loss_G.item()}, adversarial: {adversarial_loss_G.item()}\n")
                    f.write(f"sh mse: {sh_mse_loss.item()}, sh cos: {sh_cos_loss.item()}\n\n")

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/{config.upscale_factor}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/{config.upscale_factor}/Disc.pt')

                    progress(batch_index, batches, epoch, num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        train_loss_D_list.append(train_loss_D / len(train_prefetcher))
        train_loss_D_hr_list.append(train_loss_D_hr / len(train_prefetcher))
        train_loss_D_sr_list.append(train_loss_D_sr / len(train_prefetcher))
        train_loss_G_list.append(train_loss_G / len(train_prefetcher))
        train_loss_G_content_list.append(train_loss_G_content / len(train_prefetcher))
        train_loss_G_adversarial_list.append(train_loss_G_adversarial / len(train_prefetcher))
        train_loss_G_sh_mse_list.append(train_loss_G_sh_mse / len(train_prefetcher))
        train_loss_G_sh_cos_list.append(train_loss_G_sh_cos / len(train_prefetcher))
        print(f"Avearge epoch loss, discriminator: {train_loss_D_list[-1]}, generator: {train_loss_G_list[-1]}")
        print(f"Avearge epoch loss, D_real: {train_loss_D_hr_list[-1]}, D_fake: {train_loss_D_sr_list[-1]}")
        print(
            f"Avearge content loss: {train_loss_G_content_list[-1]}, adversarial loss: {train_loss_G_adversarial_list[-1]}")
        print(f"Average sh mse loss: {train_loss_G_sh_mse_list[-1]}, sh cos loss: {train_loss_G_sh_cos_list[-1]}")

    # create plot path
    plot_path = path + f'/{config.upscale_factor}/loss_plot'
    shutil.rmtree(Path(plot_path), ignore_errors=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_losses([train_loss_D_list, train_loss_G_list],
                ['Discriminator loss', 'Generator loss'],
                ['red', 'green'],
                path=plot_path, filename='loss_curves', title="Loss curves")
    plot_losses([train_loss_D_list], ['Discriminator loss'], ['red'], path=plot_path, filename='Discriminator_loss',
                title="Dis loss")
    plot_losses([train_loss_G_list], ['Generator loss'], ['green'], path=plot_path, filename='Generator_loss',
                title="Gen loss")
    plot_losses([train_loss_D_hr_list, train_loss_D_sr_list],
                ['Discriminator loss real', 'Discriminator loss fake'],
                ["#5ec962", "#440154"],
                path=plot_path, filename='loss_curves_Dis', title="Discriminator loss curves")
    plot_losses([train_loss_G_sh_mse_list], ['SH mse loss'], ['blue'], path=plot_path, filename='SH_mse_loss',
                title="SH mse loss")
    plot_losses([train_loss_G_sh_cos_list], ['SH cos loss'], ['blue'], path=plot_path, filename='SH_cos_loss',
                title="SH cos loss")
    plot_losses([train_loss_G_adversarial_list, train_loss_G_content_list, train_loss_G_sh_cos_list],
                ['Generator adv loss', 'Generator content loss', 'Coefficient sim loss'],
                ['green', 'purple', 'red'],
                path=plot_path, filename='loss_curves_G', title="Generator loss curves")

    with open(f'{path}/{config.upscale_factor}/train_losses.pickle', "wb") as file:
        pickle.dump((train_loss_D_list, train_loss_D_hr_list, train_loss_D_sr_list,
                     train_loss_G_list, train_loss_G_content_list, train_loss_G_adversarial_list,
                     train_loss_G_sh_cos_list,
                     train_loss_G_sh_mse_list), file)
    print("TRAINING FINISHED")