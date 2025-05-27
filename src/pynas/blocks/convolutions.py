import torch
import torch.nn as nn
from .activations import ReLU


# Classic Conv
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(),
        )


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            activation(),
        )


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction, 1)  # Ensure at least 1 output feature
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvSE(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            ConvBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation),
            SEBlock(reduction=16, in_channels=out_channels),
        )


# MBConv Inverted
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
        expanded_channels = in_channels * expansion_factor
        super().__init__()
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=1, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to narrow
            ConvBnAct(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        res = x
        x = self.steps(x)
        x = torch.add(x, res)
        return x
    

class MBConvNoRes(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
        expanded_channels = in_channels * expansion_factor
        super().__init__()
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=1, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to narrow
            ConvBnAct(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        x = self.steps(x)
        return x


# CSP Conv
class CSPConvBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, activation=ReLU):
        super().__init__()

        # Use the same value for hidden_channels to avoid the issue with odd in_channels
        self.main_channels = in_channels // 2
        self.shortcut_channels = in_channels-self.main_channels

        # Main path (processed part)
        self.main_path = nn.Sequential(
            *[ConvBnAct(
                in_channels=self.main_channels,
                out_channels=self.main_channels,
                activation=activation,
            ) for _ in range(num_blocks)],
        )

        # Shortcut path is just a passthrough
        self.shortcut_path = nn.Identity()

        # Final 1x1 convolution after merging
        self.final_transition = nn.Sequential(
            nn.Conv2d(
                in_channels=self.main_channels+self.shortcut_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels),
            activation(),
        )


    def forward(self, x):
        # Apply first transition which is just a passthrough here
        #shortcut = nn.Identity(x)

        # Splitting the input tensor into two paths
        main_data = x[:, :self.main_channels, :, :]
        shortcut_data = x[:, self.main_channels:, :, :]

        main_data = self.main_path(main_data)
        shortcut_data = self.shortcut_path(shortcut_data)

        # Concatenating the main and shortcut paths
        combined = torch.cat(tensors=(main_data, shortcut_data), dim=1)
        out = self.final_transition(combined)
        return out


class CSPMBConvBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
        super().__init__()

        # Use the same value for hidden_channels to avoid the issue with odd in_channels
        self.main_channels = in_channels // 2
        self.shortcut_channels = in_channels-self.main_channels

        # Main path (processed part)
        self.main_path = nn.Sequential(
            *[MBConv(
                in_channels=self.main_channels,
                out_channels=self.main_channels,
                expansion_factor=expansion_factor,
                activation=activation,
                dw_kernel_size=dw_kernel_size,
            ) for _ in range(num_blocks)],
        )

        # Shortcut path is just a passthrough
        self.shortcut_path = nn.Identity()

        # Final 1x1 convolution after merging
        self.final_transition = nn.Sequential(
            nn.Conv2d(
                in_channels=self.main_channels+self.shortcut_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels),
            activation(),
        )

    def forward(self, x):
        # Apply first transition which is just a passthrough here
        #shortcut = nn.Identity(x)

        # Splitting the input tensor into two paths
        main_data = x[:, :self.main_channels, :, :]
        shortcut_data = x[:, self.main_channels:, :, :]

        main_data = self.main_path(main_data)
        shortcut_data = self.shortcut_path(shortcut_data)

        # Concatenating the main and shortcut paths
        combined = torch.cat(tensors=(main_data, shortcut_data), dim=1)
        out = self.final_transition(combined)
        return out


# DenseNetBlock
class DenseNetBlock(nn.Module):
    """
    Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.
    """
    def __init__(self, in_channels, out_channels, activation=ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        res = x
        x = self.block(x)
        return torch.cat([res, x], dim=1)


# ResNetBlock and derivatives (https://paperswithcode.com/method/resnext)
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, activation=ReLU):
        super().__init__()
        assert out_channels == in_channels
        reduced_channels = in_channels // reduction_factor
        
        if reduced_channels == 0:             
            reduced_channels = in_channels    
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            ConvBnAct(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, activation=activation),
            # Wide to narrow
            ConvBnAct(reduced_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        x = self.steps(x)
        return x
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, activation=ReLU):
        super().__init__()
        assert out_channels == in_channels
        self.main_path = ResNetBasicBlock(
            in_channels=in_channels, 
            out_channels=out_channels,
            reduction_factor=reduction_factor,
            activation=activation,
            )

    def forward(self, x):
        res = x
        x = self.main_path(x)
        x = torch.add(x, res)
        return x
    
 
class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(x)



"""
class ResNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, cardinality=1, activation=ReLU):
        super().__init__()
        assert out_channels == in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cardinality = cardinality

        self.is_divisible = False
        if in_channels % cardinality == 0:
            self.is_divisible = True

        # Works only with combinations with module 0
        self.parallel_channels = self.in_channels // self.cardinality
        self.parallel_paths = []
        for path in range(self.cardinality):
            path = ResNetBasicBlock(
                in_channels=self.parallel_channels, 
                out_channels=self.out_channels,
                reduction_factor=reduction_factor,
                activation=activation(),
            )
            self.parallel_paths.append(path)
        

    def forward(self, x):
        res = x
        paths_sum = torch.zeros_like(x)
        if self.is_divisible:
            i = 0
            while i<self.in_channels:
                path_channels = x[:][i:self.parallel_channels][:][:] # check se funziona.
                path_channels = self.parallel_paths(i//self.cardinality)(path_channels)
                paths_sum = torch.add(paths_sum, path_channels)
                i += self.cardinality
        else:
            print("Cardinality was not divisible, using Identity instead of ResNextBlock.")
        x = torch.add(paths_sum, res)
        return x
"""
