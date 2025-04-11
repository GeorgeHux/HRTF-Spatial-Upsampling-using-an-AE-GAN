import argparse
from config import Config
from model.train import train
from model.test import test
from model.util import load_hrtf
from preprocessing.hrtf_sphere import HRTF_Sphere
from preprocessing.utils import convert_to_sofa

from baselines.barycentric_interpolation import run_barycentric_interpolation
from baselines.hrtf_selection import run_hrtf_selection
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("-r", "--remote")
args = parser.parse_args()

if args.remote == "True":
    remote = True
elif args.remote == "False":
    remote = False
else:
    raise RuntimeError("Please enter 'True' or 'False' for the remote tag (-r/--remote)")

config = Config(remote)


def objective(lr):
    config.lr_G = lr  # Assuming you want to optimize the generator's learning rate
    # modify train function to return a performance metric like LSD error
    train_prefetcher, _ = load_hrtf(config)
    path = f'{config.path}/{config.upscale_factor}'
    print(f"current lr:{config.lr_G}")
    shutil.rmtree(Path(path), ignore_errors=True)
    Path(path).mkdir(parents=True, exist_ok=True)
    train(config, train_prefetcher)
    _, test_prefetcher = load_hrtf(config)
    test(config, test_prefetcher)
    sr_dir = config.valid_recon_path + f'/{config.upscale_factor}/mag'
    performance_metric = run_lsd_evaluation(config, sr_dir)
    # run_localisation_evaluation(config, sr_dir)
    return {'loss': performance_metric, 'status': STATUS_OK}