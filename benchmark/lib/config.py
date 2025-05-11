import yaml

from typing import Any
from dataclasses import dataclass


# ---------------
# Database Config
# ---------------

@dataclass
class DatabaseFolderBasedFoldConfig:
    train_folders: list[int]
    valid_folders: list[int]
    test_folders: list[int]

@dataclass
class DatabaseConfig:
    name: str
    annotations_path: str
    training_folder_based_folds: list[DatabaseFolderBasedFoldConfig]

def get_database_config(config_file: str) -> DatabaseConfig:
    with open(config_file, 'r') as f:
        config_content = yaml.safe_load(f)
    
    return DatabaseConfig(
        name=config_content['name'], 
        annotations_path=config_content['annotations_path'], 
        training_folder_based_folds=[
            DatabaseFolderBasedFoldConfig(
                train_folders=fold['train'], 
                valid_folders=fold['valid'], 
                test_folders=fold['test'],
            )
            for fold in config_content['folds']
        ],
    )

# ----------------


# -----------------
# Experiment Config
# -----------------

@dataclass
class ExperimentModelConfig:
    head_type: str
    backbone: str
    num_backbone_features: int
    input_size: tuple[int, int, int]
    pretraining_scheme: str
    pretrained_checkpoint_path: str|None

@dataclass
class ExperimentTrainingConfig:
    num_epochs: int
    batch_size: int
    lr: float
    optimizer: str
    adam_betas: tuple[float, float]|None
    adam_eps: float|None

def get_experiment_config(config_file: str) -> tuple[ExperimentModelConfig, ExperimentTrainingConfig]:
    with open(config_file, 'r') as f:
        config_content = yaml.safe_load(f)
    
    model_config = ExperimentModelConfig(
        head_type=config_content['model_config']['head_type'],
        backbone=config_content['model_config']['backbone'],
        num_backbone_features=config_content['model_config']['num_backbone_features'],
        input_size=config_content['model_config']['input_size'],
        pretraining_scheme=config_content['model_config']['pretraining_scheme'],
        pretrained_checkpoint_path=config_content['model_config']['pretrained_checkpoint_path'],
    )

    training_config = ExperimentTrainingConfig(
        num_epochs=config_content['training_config']['num_epochs'],
        batch_size=config_content['training_config']['batch_size'],
        lr=config_content['training_config']['lr'],
        optimizer=config_content['training_config']['optimizer'],
        adam_betas=config_content['training_config']['adam_betas'],
        adam_eps=config_content['training_config']['adam_eps'],
    )

    return model_config, training_config


# ----------------