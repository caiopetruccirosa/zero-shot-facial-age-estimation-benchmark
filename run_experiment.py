import torch
import numpy as np
import random
import argparse

from argparse import Namespace
from benchmark.lib.train import train


def set_fixed_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments() -> Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('benchmark_config', help='', type=str, required=True)
    arg_parser.add_argument('experiment_config', help='', type=str, required=True)
    arg_parser.add_argument('fold_index', help='', type=int, required=True)
    arg_parser.add_argument('wandb_project', help='', default='zero-shot-facial-age-estimation', type=str, required=False)
    arg_parser.add_argument('wandb_mode', help='', default='online', type=str, required=False)
    arg_parser.add_argument('timezone', help='Timezone in which the experiment is running.', default='America/Sao_Paulo', type=str, required=False)
    arg_parser.add_argument('data_dir', help='', type=str, required=True)
    
    return arg_parser.parse_args()


def get_device(device_argument: str|None) -> torch.device:
    if device_argument is not None:
        return torch.device(device_argument)
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')



def main() -> None:
    set_fixed_seed(seed=123)
    args = get_arguments()

    # TODO
    # run experiment


if __name__ == '__main__':
    main()