# Description: Contains custom head layers for neural networks.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Dropout(nn.Module):
    """
    Dropout layer for regularization in neural networks.
    
    Args:
        p (float, optional): Probability of an element to be zeroed. Default: 0.5
        inplace (bool, optional): If set to True, will do this operation in-place. Default: False
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        return F.dropout(x, self.p, self.training, self.inplace)


class MultiInputClassifier(nn.Module):
    """
    A PyTorch module for a multi-input classifier that processes multiple input tensors
    with different shapes and combines their features for classification.
    Args:
        input_shapes (List[Tuple[int, ...]]): A list of shapes for each input tensor,
            excluding the batch dimension. Each shape can be either (C, H, W) for spatial
            inputs or (D,) for flat vector inputs.
        common_dim (int, optional): The dimension to which all inputs are projected.
            Defaults to 256.
        mlp_depth (int, optional): The number of layers in the final MLP classifier.
            Defaults to 2.
        mlp_hidden_dim (int, optional): The number of hidden units in each MLP layer.
            Defaults to 512.
        num_classes (int, optional): The number of output classes for classification.
            Defaults to 10.
        use_adaptive_pool (bool, optional): Whether to apply adaptive average pooling
            for spatial inputs. Defaults to True.
        pool_size (Tuple[int, int], optional): The target size for adaptive pooling
            if it is used. Defaults to (4, 4).
    Attributes:
        projections (nn.ModuleList): A list of projection modules for each input tensor.
            These modules transform the inputs to the common dimension.
        flatten (nn.Flatten): A module to flatten the projected tensors.
        total_input_dim (int): The total input dimension after concatenating all
            projected tensors.
        classifier (nn.Sequential): The MLP classifier that processes the concatenated
            features and outputs class probabilities.
    Methods:
        forward(inputs: List[torch.Tensor]) -> torch.Tensor:
            Processes the input tensors, projects them to a common dimension, concatenates
            their features, and passes them through the MLP classifier to produce the
            output logits.
    Raises:
        ValueError: If an input shape is not supported (e.g., not (C, H, W) or (D,)).
    """
    def __init__(
        self,
        input_shapes: List[Tuple[int, ...]],   # shapes of each input tensor (excluding batch dim)
        common_dim: int = 256,                 # project all inputs to this dim
        mlp_depth: int = 2,                    # depth of final MLP
        mlp_hidden_dim: int = 512,             # hidden units in MLP
        num_classes: int = 10,                 # number of output classes
        use_adaptive_pool: bool = True,        # apply adaptive pooling for spatial inputs
        pool_size: Tuple[int, int] = (4, 4)    # target size if pooling is used
    ):
        super().__init__()
        self.projections = nn.ModuleList()
        self.flatten = nn.Flatten(start_dim=1)
        
        for shape in input_shapes:
            in_dim = shape[0]
            if len(shape) == 3:  # (C, H, W)
                proj = nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size) if use_adaptive_pool else nn.Identity(),
                    nn.Conv2d(in_dim, common_dim, kernel_size=1),
                )
                out_dim = common_dim * pool_size[0] * pool_size[1]
            elif len(shape) == 1:  # flat vector
                proj = nn.Sequential(
                    nn.Linear(in_dim, common_dim),
                )
                out_dim = common_dim
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.projections.append(proj)
        
        self.total_input_dim = sum([
            common_dim * pool_size[0] * pool_size[1] if len(shape) == 3 else common_dim
            for shape in input_shapes
        ])

        # Build MLP
        layers = []
        in_dim = self.total_input_dim
        for i in range(mlp_depth - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        projected = []
        for x, proj in zip(inputs, self.projections):
            x = proj(x)
            x = self.flatten(x)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        return self.classifier(x)




class Classifier:
    def __init__(self, encoder, dm, verbose=False):
        # Validate that dm has the necessary attributes.
        if not hasattr(dm, "num_classes") or not hasattr(dm, "input_shape"):
            raise ValueError("dm must have 'num_classes' and 'input_shape' attributes.")
        
        self.encoder = encoder
        self.num_classes = dm.num_classes
        self.input_shape = dm.input_shape
        self.verbose = verbose
        if self.verbose:
            print(f"Input shape: {self.input_shape}")
        
        # Verify that encoder has parameters.
        try:
            next(self.encoder.parameters())
        except StopIteration:
            raise ValueError("Encoder appears to have no parameters.")
        except Exception as e:
            raise ValueError("Provided encoder does not follow expected API.") from e

        # Validate input_shape is a tuple and properly dimensioned.
        if not isinstance(self.input_shape, tuple):
            raise TypeError("input_shape must be a tuple.")
        if len(self.input_shape) == 3:
            if self.verbose:
                print("Adding channel dimension to input shape.")
                print(f"Original input shape: {self.input_shape}")
            self.input_shape = (1,) + self.input_shape
            if self.verbose:
                print(f"Updated input shape: {self.input_shape}")
        elif len(self.input_shape) != 4:
            raise ValueError("input_shape must be of length 3 or 4.")

        self.head_layer = self.build_head(input_shape=self.input_shape)
        
        self.model = nn.Sequential(
            self.encoder,
            self.head_layer
        )
        
        self.valid_model = self.dummy_test()

    def build_head(self, input_shape=(1, 2, 256, 256)):
        # Get the device from the encoder's parameters.
        try:
            device = next(self.encoder.parameters()).device
        except Exception as e:
            raise ValueError("Unable to determine device from encoder parameters.") from e
        
        # Run a dummy input through the encoder to get the feature shape.
        dummy = torch.randn(*input_shape).float().to(device)
        try:
            features = self.encoder(dummy)
        except Exception as e:
            raise RuntimeError("Error when running dummy input through encoder.") from e
        
        if not isinstance(features, torch.Tensor):
            raise TypeError("Encoder output should be a torch.Tensor.")

        if self.verbose:
            print("Feature map shape from the feature extractor:", features.shape)

        # Check that the features tensor has at least 2 dimensions.
        if features.dim() < 2:
            raise ValueError("Encoded features should have at least 2 dimensions.")
        
        # Determine the number of channels from the dummy output.
        feature_channels = features.shape[1]

        # Build the head layer.
        head_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_channels, self.num_classes)
        )
        if self.verbose:
            print("Constructed head layer:", head_layer)
        return head_layer
    
    
    def dummy_test(self):
        try:
            device = next(self.encoder.parameters()).device
            dummy = torch.randn(*self.input_shape).float().to(device)
            output = self.model(dummy)
            if self.verbose:
                print("Network test passed. Output shape from the model:", output.shape)
            
            if not isinstance(output, torch.Tensor):
                raise TypeError("Output of the model should be a torch.Tensor.")
            
            if output.shape[0] != dummy.shape[0]:
                raise ValueError("Batch size mismatch between input and output.")
            
            return True
        except Exception as e:
            if self.verbose:
                print("An error occurred during dummy_test:", e)
            return False
    
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        return self.model(x)