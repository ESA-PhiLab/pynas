from .core.generic_lightning_module import (
    GenericLightningNetwork,
    GenericLightningNetwork_Custom,
    GenericLightningSegmentationNetwork,
)

from .train.losses import CategoricalCrossEntropyLoss
from .train.custom_iou import calculate_iou
from .train.mean_squared_error import MeanSquaredError
from .train.my_early_stopping import EarlyStopping, TrainEarlyStopping

from .core.individual import Individual

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "blocks",
    "core",
    "opt",
    "train",
    "utils",
    "GenericLightningNetwork",
    "GenericLightningNetwork_Custom",
    "GenericLightningSegmentationNetwork",
    "CategoricalCrossEntropyLoss",
    "calculate_iou",
    "MeanSquaredError",
    "EarlyStopping",
    "TrainEarlyStopping",
    "Individual",
]
