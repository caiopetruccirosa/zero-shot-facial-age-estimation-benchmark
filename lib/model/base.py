import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
from config import ExperimentModelConfig
from lib.model.heads import MegaAgeLayer, CoralLayer


__SUPPORTED_HEAD_TYPES = [ 'classification', 'dldl', 'dldl_v2', 'unimodal_concentrated', 'soft_labels', 'mean_variance', 'regression', 'megaage', 'orcnn',  'extended_binary_classification', 'coral' ]
__SUPPORTED_BACKBONES  = [ 'resnet50' ]


class AgeEstimationModel(nn.Module):
    def __init__(self, config: ExperimentModelConfig, verbose: bool = False):
        assert config.head_type in __SUPPORTED_HEAD_TYPES, f'Unsupported head type "{config.head_type}".'
        assert config.backbone in __SUPPORTED_BACKBONES, f'Unsupported backbone "{config.backbone}".'

        self.age_range = config.age_range
        self.n_classes = self.age_range[1] - self.age_range[0] + 1 # max_age - min_age + 1

        self.head_type = config.head_type
        self.backbone_type = config.backbone

        if self.backbone_type in [ 'resnet50' ]:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone.fc = nn.Identity() # type: ignore
            self.n_extracted_features = self.backbone.fc.in_features
        else:
            raise NotImplementedError()
            
        if self.head_type in [ 'classification', 'dldl', 'dldl_v2', 'unimodal_concentrated', 'soft_labels', 'mean_variance' ]:
            self.head = nn.Linear(self.n_extracted_features, self.n_classes)
        elif self.head_type in [ 'regression' ]:
            self.head = nn.Linear(self.n_extracted_features, 1)
            self.n_classes = -1
        elif self.head_type in [ 'megaage' ]:
            self.head = MegaAgeLayer(self.n_extracted_features, self.n_classes)
        elif self.head_type in [ 'orcnn', 'extended_binary_classification' ]:
            self.head = nn.Linear(self.n_extracted_features, self.n_classes)
        elif self.head_type in [ 'coral' ]:
            self.head = CoralLayer(self.n_extracted_features, self.n_classes)
        else:
            raise NotImplementedError()
        
        self.input_shape = config.input_size
        if verbose:
            summary(self, self.input_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        o = self.head(x)
        return o