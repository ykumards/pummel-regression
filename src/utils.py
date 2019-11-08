import os
import time
from time import gmtime, strftime
import random
import yaml
import pprint
import numpy as np
import scipy
from contextlib import contextmanager
from pathlib import Path
from sklearn import metrics
import torch

@contextmanager
def timer(name: str) -> None:
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

 
def seed_everything(seed:int) -> None:
    "seeding function for reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_stats(
        heldout_true: list, 
        heldout_preds: list, 
        prefix='main_run'
) -> None:
    print(prefix)
    heldout_true = np.array(heldout_true)
    heldout_preds = np.array(heldout_preds)
    print(f"MSE for ho: {metrics.mean_squared_error(heldout_true, heldout_preds)}")
    print(f"MAE for ho: {metrics.mean_absolute_error(heldout_true, heldout_preds)}")
    print(f"Median Abs Error for ho: {metrics.median_absolute_error(heldout_true, heldout_preds)}")
    print(f"R2 for ho: {metrics.r2_score(heldout_true, heldout_preds)}")
    print(f"Max residual error for ho: {np.max(np.abs(np.array(heldout_true) - np.array(heldout_preds)))}")
    print(f"Explained variance for ho: {metrics.explained_variance_score(heldout_true, heldout_preds)}")
    print(f"Corr. for ho: {scipy.stats.stats.pearsonr(heldout_true, heldout_preds)}")
    print(f"Spearman Corr. for ho: {scipy.stats.stats.spearmanr(heldout_true, heldout_preds)}")
    print(f"Means of y_pred and y_true: {np.mean(heldout_preds), np.mean(heldout_true)}")
    print(f"Stddev of y_pred and y_true: {np.std(heldout_preds), np.std(heldout_true)}")
    print(f"Max of y_pred and y_true: {np.max(heldout_preds), np.max(heldout_true)}")
    
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')



class Dict2Object:
    """
    Object that basically converts a dictionary of args 
    to object of args. Purpose is to simplify calling the args
    (from args["lr"] to args.lr)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_config(config_path: str, curr_time: str =None) -> Dict2Object:
    if curr_time is None:
        curr_time = strftime("%y-%m-%d-%H-%M", gmtime())

    with open(config_path, 'r') as stream:
        cfg = yaml.load(stream)
    print("loaded config")
    print("="*90)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    print("="*90)

    args = Dict2Object(**cfg)
    args.logdir += curr_time
    VOLUME_PATH = Path('/home/ykuv/pummel_data/')
    
    args.data_filepath = VOLUME_PATH/Path(args.data_filename)
    args.model_state_file = VOLUME_PATH/Path(args.exp_name + "_model.pth")
    args.optim_state_file = VOLUME_PATH/Path(args.exp_name + "_optim.pth")
    args.exp_name += f"_{args.num_samples}_{args.num_epochs}ep_{args.batch_size}bs_{args.rnn_hidden_size}rhid" 
    args.exp_dir = Path(args.exp_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if not args.disable_cuda and torch.cuda.is_available():
        print("using cuda")
        args.device = torch.device('cuda')
    else:
        print("not using cuda")
        args.device = torch.device('cpu')

    if args.debug:
        args.num_workers = 0
    else:
        args.num_workers = 8

    
    return args
