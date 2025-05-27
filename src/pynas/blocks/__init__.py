import math
from typing import Callable, Tuple, Optional
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

__all__ = ["activations", "convolutions", "pooling", "heads"]


class Lambda(nn.Module):
    """An utility Module, it allows custom function to be passed

    Args:
        lambd (Callable[Tensor]): A function that does something on a tensor

    Examples:
        >>> add_two = Lambda(lambd x: x + 2)
        >>> add_two(Tensor([0])) // 2
    """

    def __init__(self, lambd: Callable[[Tensor], Tensor]):

        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)
