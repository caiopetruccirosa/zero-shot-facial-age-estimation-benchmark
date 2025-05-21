import numpy as np
import numpy.typing as npt

from typing import Callable
from torch.utils.data import Dataset
from PIL import Image


class NormalizedFaceImageDataset(Dataset):
    def __init__(self, data_path, image_annotations_path, image_transformation) -> None:
        super().__init__()

        self.image_transformation = image_transformation
        self.image_paths = []
        
    def __getitem__(self, idx: int) -> npt.NDArray[np.uint8]:
        image = read_image(self.image_paths[idx])
        transformed_image = self.image_transformation(image)
        return transformed_image


def read_image(image_path: str) -> npt.NDArray[np.uint8]:
    pil_image = Image.open(image_path).convert('RGB')
    array_image = np.array(pil_image)
    return array_image


def get_image_transformation(albumentations_config_path: str) -> Callable:
    pass