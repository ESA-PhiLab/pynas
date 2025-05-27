import torch.nn as nn


class AvgPool(nn.Sequential):
    """
    Implements an average pooling layer.
    This layer applies a 2D average pooling over an input signal (usually an image) represented as a batch of
    multichannel data. It reduces the spatial dimensions (width and height) of the input by taking the average of
    elements in a kernel-sized window, which slides over the input data with a specified stride.

    Parameters:
    - kernel_size (int, optional): The size of the window for each dimension of the input tensor. Default is 2.
    - stride (int, optional): The stride of the window. Default is 2.

    Example Usage:
        avg_pool_layer = AvgPool(kernel_size=2, stride=2)
    """
    def __init__(self, kernel_size=2, stride=2):
        super(AvgPool, self).__init__(
            nn.AvgPool2d(kernel_size, stride)
        )


class MaxPool(nn.Sequential):
    """
    Implements a max pooling layer.
    This layer applies a 2D max pooling over an input signal (usually an image) represented as a batch of multichannel
    data. It reduces the spatial dimensions (width and height) of the input by taking the maximum value of elements in
    a kernel-sized window, which slides over the input data with a specified stride and padding.

    Parameters:
    - kernel_size (int): The size of the window for each dimension of the input tensor.
    - stride (int, optional): The stride of the window. Defaults to kernel_size if not specified.
    - padding (int, optional): The amount of padding added to all sides of the input. Default is 0.

    Example Usage:
        max_pool_layer = MaxPool(kernel_size=2, stride=2, padding=0)
    """
    def __init__(self, kernel_size=2, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(MaxPool, self).__init__(
            nn.MaxPool2d(kernel_size, stride, padding)
        )
