import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, remote, dataset=None, existing_model_tag=None, data_dir=None):

        # overwrite settings with arguments provided
        self.dataset = dataset if dataset is not None else 'SONICOM'
        self.data_dir = data_dir if data_dir is not None else '/data/' + self.dataset

        if existing_model_tag is not None:
            self.start_with_existing_model = True
        else:
            self.start_with_existing_model = False

        self.existing_model_tag = existing_model_tag if existing_model_tag is not None else None

        # Data processing parameters
        self.gen_sofa_flag = True
        self.nbins_hrtf = 128  # make this a power of 2 256
        self.hrtf_size = 16
        self.upscale_factor = 4  # can only take values: 4, 8, 16, 32, 48, 72, 108, 216, 288
        self.train_samples_ratio = 0.8
        self.hrir_samplerate = 48000.0
        self.transform_flag = False
        self.domain = 'magnitude_db' # magnitude_db
        self.max_degree = 21

        # Data dirs
        if remote:
            # self.data_dir_path = '/HRTF-results'
            # self.raw_hrtf_dir = 'raw data'
            self.data_dir_path = 'D:/PycharmProjects/Upsample_GAN/SHT_HRTF_localisation/SHT_HRTF'
            self.raw_hrtf_dir = Path('D:/PycharmProjects/Upsample_GAN/HRTF Datasets')
            self.amt_dir = 'D:/PycharmProjects/Upsample_GAN/amtoolbox-full-1.4.0'
        else:
            self.data_dir_path = '/Users/lijian/Downloads/shtHRTF'
            self.raw_hrtf_dir = Path('/Users/lijian/Downloads')


        self.result_folder = '/results'
        self.path = f'{self.data_dir_path}{self.result_folder}'
        self.existing_model_path = f'{self.data_dir_path}{self.result_folder}/{self.existing_model_tag}'

        self.valid_recon_path = f'{self.data_dir_path}{self.result_folder}/valid_recon'
        self.valid_target_path = f'{self.data_dir_path}{self.data_dir}/valid_target'
        self.model_path = f'{self.data_dir_path}{self.result_folder}'

        self.baseline_dir = '/baseline_results/' + self.dataset
        self.barycentric_hrtf_dir = self.data_dir_path + self.baseline_dir + '/barycentric/valid'
        self.hrtf_selection_dir = self.data_dir_path + self.baseline_dir + '/hrtf_selection/valid'

        self.train_val_id_dir = self.data_dir_path + self.data_dir + '/train_val_id'

        self.mean_std_filename = self.data_dir_path + self.data_dir + '/mean_std_' + self.dataset
        self.mean_std_coef_dir = self.data_dir_path + self.data_dir + '/coef_mean_std'

        # Training hyperparams
        self.batch_size = 8
        self.num_workers = 1
        self.optimizer = 'adam'
        self.num_epochs = 100  # was originally 250
        self.lr_G = 0.0002
        self.lr_D = 0.00003
        self.latent_dim = 128
        # how often to train the generator
        self.critic_iters = 1

        # Loss function weight
        self.content_weight = 0.01
        self.adversarial_weight = 1

        # betas for Adam optimizer
        # self.beta1 = 0.9
        # self.beta2 = 0.999

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self, n):
        self.raw_hrtf_dir = str(self.raw_hrtf_dir)
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config_files/config_{n}.json', 'w') as f:
            json.dump(j, f)

    def load(self, n):
        with open(f'{self.path}/config_files/config_{n}.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)
        self.raw_hrtf_dir = Path(self.raw_hrtf_dir)

    def get_train_params(self):
        return self.batch_size, self.lr_G, self.lr_D, self.latent_dim, self.critic_iters, self.max_degree
