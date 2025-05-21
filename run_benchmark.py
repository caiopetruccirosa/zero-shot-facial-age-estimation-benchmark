import argparse
import yaml
import os

from lightning.pytorch import seed_everything
from argparse import Namespace
from dotenv import load_dotenv
from datetime import datetime
from pytz import timezone

from lib.trainer import train
from lib.model.base import AgeEstimationModel
from lib.loss import get_head_loss

from dataclasses import dataclass

@dataclass
class WandbConfig:
    project_name: str
    experiment_name: str
    mode: str

@dataclass
class TrainingConfig:
    n_epochs: int
    improve_epochs_patience: int
    batch_size: int
    optimizer: str
    learning_rate: float
    eps: float|None
    betas: tuple[float, float]|None

@dataclass
class ModelConfig:
    head_type: str
    backbone: str
    input_size: tuple[int, int, int]
    pretraining_scheme: str
    pretraining_checkpoint_path: str|None

@dataclass
class BenchmarkDataConfig:
    training_set: str
    training_set_folds: list[tuple[list[int], list[int], list[int]]]
    evaluation_sets: list[str]
    data_root_dir: str

def get_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--experiment_config', help='', type=str, required=True)
    arg_parser.add_argument('--fold_index', help='', type=int, required=True)
    arg_parser.add_argument('--data_root_dir', help='', type=str, required=True)
    arg_parser.add_argument('--device', help='', type=str, choices=['cpu', 'gpu', 'mps'], default='cpu', required=False)
    arg_parser.add_argument('--wandb_project', help='', default='zero-shot-facial-age-estimation-benchmark', type=str, required=False)
    arg_parser.add_argument('--wandb_mode', help='', choices=['online', 'offline', 'disabled'], default='online', type=str, required=False)
    arg_parser.add_argument('--timezone', help='Timezone in which the experiment is running.', default='America/Sao_Paulo', type=str, required=False)
    
    return arg_parser.parse_args()


def get_experiment_configs_from_args(args: Namespace) -> tuple[str, BenchmarkDataConfig, ModelConfig, TrainingConfig, WandbConfig]:
    assert not os.path.exists(args.experiment_config) or os.path.splitext(args.experiment_config)[1] not in ['yaml', 'yml'], 'Invalid experiment config file!'
    assert not os.path.exists(args.data_root_dir), 'Invalid data root directory!'
    
    with open(args.experiment_config, 'r') as f:
        config = yaml.safe_load(f)

    with config['datasets'] as dc:
        with open(f'benchmark/databases/{dc['training_set']}/{dc['training_set_folds']}', 'r') as f:
            folds_content = yaml.safe_load(f)
        
        benchmark_data_config = BenchmarkDataConfig(
            training_set=dc['training_set'],
            training_set_folds=[ (fold['train'], fold['valid'], fold['test']) for fold in folds_content['folds'] ],
            evaluation_sets=dc['evaluation_sets'],
            data_root_dir=args.data_root_dir,
        )

    with config['model'] as mc:
        model_config = ModelConfig(
            head_type=mc['head_type'],
            backbone=mc['backbone'],
            input_size=mc['input_size'],
            pretraining_scheme=mc['pretraining_scheme'],
            pretraining_checkpoint_path=mc['pretrained_checkpoint_path'] if mc['pretraining_scheme'] is 'checkpoint' else None,
        )

    with config['training'] as tc:
        training_config = TrainingConfig(
            n_epochs=tc['n_epochs'],
            improve_epochs_patience=tc['improve_epochs_patience'],
            batch_size=tc['batch_size'],
            learning_rate=tc['learning_rate'],
            optimizer=tc['optimizer'],
            betas=tc['betas'] if tc['optimizer'] is 'adam' else None,
            eps=tc['eps'] if tc['optimizer'] is 'adam' else None,
        )

    wandb_config = WandbConfig(
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        mode=args.mode,
    )

    starting_instant = datetime.now(tz=timezone(args.timezone)).strftime('%Y_%m_%d_%H_%M_%S')
    experiment_id    = f'{benchmark_data_config.training_set}_{model_config.head_type}_{model_config.backbone}_{starting_instant}'

    info = {
        'training_set': benchmark_data_config.training_set,
        'model_head_type': model_config.head_type,
        'model_backbone': model_config.backbone,
        'learning_rate': training_config.learning_rate,
        'n_epochs': training_config.n_epochs,
        'optimizer': training_config.optimizer,
        'batch_size': training_config.batch_size,
    }

    return experiment_id, benchmark_data_config, model_config, training_config, wandb_config


def main():
    load_dotenv()

    seed_everything(seed=123, workers=True)

    args = get_arguments()

    # load experiment info and config
    experiment_id, benchmark_data_config, model_config, training_config, wandb_config = get_experiment_configs_from_args(args)
    experiment_run_dir = f'benchmark/runs/{experiment_id}/{args.fold_index}'

    # assert all datasets in config are supported
    # load datasets

    # assert model config in experiment config are supported
    # build model and loss function
    model = AgeEstimationModel() # type: ignore
    loss_fn = get_head_loss() # type: ignore

    # assert training parameters in experiment config are supported
    # train model
    model = train(
        model=model, 
        loss_fn=loss_fn,
        training_data=None,
        validation_data=None,
        training_config=None,
        experiment_folder=None,
        wandb_project=None,
        device=None,
    )

    # eval on others datasets and save results on benchmark/runs/experiment/foldX/eval_inference


if __name__ == '__main__':
    main()