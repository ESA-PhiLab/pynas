import torch.nn as nn
from functools import partial


class GELU(nn.Module):
    """
        Gaussian Error Linear Unit (GELU) activation function.
        This module implements the GELU activation function, which is used to introduce non-linearity in the network.
        GELU is a smooth, non-monotonic function that models the Gaussian cumulative distribution function.
        It is commonly used in transformer architectures and other advanced models.

        Args:
            x (Tensor): Input tensor to which the GELU activation function is applied.

        Returns:
            Tensor: Output tensor after applying the GELU activation function.
    """
    def forward(self, x):
        return nn.GELU()(x)


class ReLU(nn.Module):
    """
        Rectified Linear Unit (ReLU) activation function.
        This module implements the ReLU activation function, which is widely used in neural networks for introducing
        non-linearity. ReLU is defined as the positive part of its argument, where each element of the input tensor `x`
        that is less than zero is replaced with zero. This function increases the non-linear properties of the decision
        function and the overall network without affecting the receptive fields of the convolution layer.

        Args:
            x (Tensor): Input tensor to which the ReLU activation function is applied.

        Returns:
            Tensor: Output tensor after applying the ReLU activation function.
    """
    def forward(self, x):
        return nn.ReLU()(x)


ReLUInPlace = partial(nn.ReLU, inplace=True)


class Sigmoid(nn.Module):
    """
        Sigmoid activation function.
    """
    def forward(self, x):
        return nn.Sigmoid()(x)

    
class Softmax(nn.Module):
    """
        Softmax activation function.
    """
    def forward(self, x):
        return nn.Softmax(dim=1)(x)