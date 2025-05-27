import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import inspect

from ..blocks import convolutions, pooling, activations


class UNetDecoder(nn.Module):
    """
    A PyTorch implementation of a U-Net decoder module.
    This class implements the decoder part of the U-Net architecture, which
    reconstructs the output from the bottleneck features by progressively
    upsampling and combining them with skip connections from the encoder.
    Attributes:
        num_stages (int): The number of decoding stages, equal to the number of skip connections.
        up_convs (nn.ModuleList): A list of transposed convolution layers for upsampling.
        conv_blocks (nn.ModuleList): A list of convolutional blocks for processing concatenated
                                     upsampled and skip connection features.
        out_conv (nn.Conv2d): The final convolutional layer that produces the output.
    Args:
        encoder_shapes (list of torch.Size): A list of shapes of the encoder features in the order:
                                             [skip0, skip1, ..., skip_(N-1), bottleneck].
                                             Each shape is expected to be a `torch.Size` object.
        num_classes (int, optional): The number of output classes. Default is 2.
    Methods:
        forward(encoder_features):
            Performs the forward pass of the decoder.
            Args:
                encoder_features (list of torch.Tensor): A list of encoder feature maps in the order:
                                                         [skip0, skip1, ..., skip_(N-1), bottleneck].
                                                         The number of feature maps must match the
                                                         number of stages + 1.
            Returns:
                torch.Tensor: The output tensor after decoding.
    """
    def __init__(self, encoder_shapes, num_classes=2, output_shape=None):
        super(UNetDecoder, self).__init__()
        # Expecting encoder_shapes as a list of torch.Size objects in order:
        # [skip0, skip1, ..., skip_(N-1), bottleneck]
        # Number of decoding stages equals the number of skip connections.
        self.num_stages = len(encoder_shapes) - 1
        
        # Build upsampling and convolution blocks dynamically.
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Initial number of channels from the bottleneck.
        in_channels = encoder_shapes[-1][1]
        
        # Iterate over skip connections in reverse order.
        for i in range(self.num_stages - 1, -1, -1):
            skip_channels = encoder_shapes[i][1]
            # Store the expected output size for each upsampling operation
            out_h, out_w = encoder_shapes[i][2], encoder_shapes[i][3]
            
            # Replace transposed convolution with upsampling + conv
            self.up_convs.append(
            nn.Sequential(
                nn.Upsample(size=(out_h, out_w), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, skip_channels, kernel_size=1)
            )
            )
            
            # The conv block takes the concatenated tensor (upsampled + skip) as input.
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_channels),
                    nn.ReLU(inplace=True)
                )
            )
            # Update in_channels to be the skip_channels for the next stage.
            in_channels = skip_channels

        # Final output convolution followed by upsampling to match the input height and width.
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Upsample(size=output_shape, mode='bilinear', align_corners=False)
        )

    def forward(self, encoder_features, verbose=False):
        # encoder_features: list in order [skip0, skip1, ..., skip_(N-1), bottleneck]
        if verbose:
            print("=" * 100)
            print(f"{'Encoder Feature Shapes':^100}")
            print(encoder_features.shape)
            print("=" * 100)

        if len(encoder_features) != self.num_stages + 1:
            raise ValueError(f"Expected {self.num_stages + 1} encoder features, but got {len(encoder_features)}.")

        x = encoder_features[-1]  # start with bottleneck

        # For each decoding stage, use the corresponding skip connection in reverse order.
        for i in range(self.num_stages):
            # Skip connection index: from last skip to the first.
            skip = encoder_features[self.num_stages - 1 - i]
            x = self.up_convs[i](x)
            # Match spatial dimensions with skip connection (just in case upsampling mismatch occurs)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.conv_blocks[i](x)

        output = self.out_conv(x)
        return output


class GenericUNetNetwork(nn.Module):
    """GenericUNetNetwork is a PyTorch-based implementation of a generic U-Net architecture. 
    This class allows for flexible construction of U-Net models by parsing layer configurations 
    and dynamically building the encoder and decoder components.
        parsed_layers (list): A list of layer configurations for building the encoder.
        input_channels (int): Number of input channels for the input tensor. Default is 3.
        input_height (int): Height of the input tensor. Default is 256.
        input_width (int): Width of the input tensor. Default is 256.
        num_classes (int): Number of output classes for the segmentation task. Default is 2.
        max_params (int): Maximum allowed number of parameters for the model. Default is 200,000,000.
        encoder (nn.ModuleList): A list of layers forming the encoder part of the U-Net.
        decoder (nn.ModuleList): A list of layers forming the decoder part of the U-Net.
        encoder_shapes (list): A list of shapes of the encoder outputs for use in the decoder.
        total_params (int): Total number of parameters in the model.
        config (ConfigParser): Configuration parser for reading additional settings from 'config.ini'.
    Methods:
        __init__(self, parsed_layers, input_channels=3, input_height=256, input_width=256, num_classes=2, MaxParams=200_000_000):
            Initializes the GenericUNetNetwork with the given parameters and builds the encoder and decoder.
        encoder_forward(self, x, features_only=True):
            Performs a forward pass through the encoder and optionally returns only the encoder features.
        _encoder_shapes_tracing(self):
            Creates a dummy forward pass through the encoder to determine the shapes of the encoder outputs.
        _build_encoder(self, parsed_layers):
        _build_decoder(self):
            Builds the decoder component of the U-Net model using the encoder shapes and number of output classes.
        forward(self, x):
            Defines the forward pass of the model, passing the input through the encoder and decoder.
        get_activation_fn(activation):
            Retrieves the specified activation function from the `activations` module.
        """
    def __init__(self, parsed_layers, input_channels=3, input_height=256, input_width=256, num_classes=2, MaxParams=200_000_00, encoder_only=False):
        super(GenericUNetNetwork, self).__init__()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.parsed_layers = parsed_layers
        self.num_classes = num_classes
        self.total_params = 0
        self.max_params = MaxParams
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder_shapes = []      # Lists to record all encoder shapes. (batch, channels, height, width) per encoder layer.

        # Encder Building:
        self._build_encoder(parsed_layers)
        if encoder_only:
            self.decoder = nn.Identity() # Set decoder to identity function.
        else:
            # Decoder Building:
            if self.encoder is not None:
                self._build_decoder()
            else:
                raise ValueError("Error building encoder. Decoder not built.")
        



    def encoder_forward(self, x, features_only=True):
        encoder_outputs = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            conv_layer_types = tuple(list_convolution_layers())
            if isinstance(layer, conv_layer_types):
                encoder_outputs.append(x.clone())
        if features_only:
            return encoder_outputs
        else:
            return x


    def _encoder_shapes_tracing(self):
        """
        Creates a dummy forward pass through the encoder to determine output shapes.
        
        This method generates a random tensor with shape (1, 3, 256, 256) and passes it
        through the encoder with features_only=True to obtain the output tensors.
        It then collects the shapes of all output tensors for network architecture analysis.
        
        Returns:
            list: A list containing the shapes of the encoder outputs. For tensor outputs,
                 their shapes are directly included. For list outputs, a list of their
                 individual tensor shapes is included.
        """
        dummy_input = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        output = self.encoder_forward(dummy_input, features_only=True)
        output_shapes = []
        for o in output:
            if isinstance(o, torch.Tensor):
                output_shapes.append(o.shape)
            elif isinstance(o, list):
                output_shapes.append([item.shape for item in o])
        return output_shapes




    def _build_encoder(self, parsed_layers, verbose=False):
        """
        Builds the encoder part of the U-Net model based on the parsed layer configurations.
        Args:
            parsed_layers (list): A list of layer configurations to be used for building the encoder.
                                  Each configuration specifies the type and parameters of the layer.
        Raises:
            AssertionError: If the output dimensions (height or width) of any layer are zero or negative.
            AssertionError: If the total number of parameters exceeds the maximum allowed (`self.max_params`).
            Exception: If any other error occurs during the encoder construction, it is caught, and the encoder is set to None.
        Notes:
            - The method iterates through the parsed layers, constructs each layer using the `build_layer` function,
              and appends it to the encoder (`self.encoder`).
            - The method updates the current dimensions (`self.current_channels`, `self.current_height`, `self.current_width`)
              after each layer is built.
            - Skip connections are recorded for layers that produce activation features intended for use in the decoder.
            - The total number of parameters is tracked and validated against the maximum allowed limit.
            - The shapes of the encoder layers are traced and stored in `self.encoder_shapes` for use in the decoder.
        """
        # Use local variables for current dims.
        if verbose:
            print("=" * 100)
            print(f"{'Building U-Net':^100}")
            print("=" * 100)
        
        self.encoder = nn.ModuleList()
        self.current_channels = self.input_channels
        self.current_height = self.input_height
        self.current_width = self.input_width
        if verbose:
            print(f"Initial Input Dimensions: Channels={self.current_channels}, Height={self.current_height}, Width={self.current_width}")
        
        try:
            for idx, layer in enumerate(parsed_layers):
                result = build_layer(layer, self.config, self.current_channels, self.current_height, self.current_width, idx, self.get_activation_fn)
                layer_inst, self.current_channels, self.current_height, self.current_width = result
                # Assert that the output dimensions are valid.
                assert self.current_height > 0 and self.current_width > 0, f"Invalid output dimensions: height ({self.current_height}) width ({self.current_width}) is zero."         
                self.encoder.append(layer_inst)
                self.total_params += sum(p.numel() for p in layer_inst.parameters())
                assert self.total_params <= self.max_params, f"Exceeded parameter limit. P: {self.total_params:,} > M: {self.max_params:,}"

            # Tracing shapes to be used in decoder.
            self.encoder_shapes = self._encoder_shapes_tracing()
            if verbose:
                print("_" * 100)
                print(f"{'U-Net Encoder Built Successfully!':^100}")
                print("_" * 100)
                print(f"{'- Total Encoder Parameters:':<25} {self.total_params:,}")
                print("=" * 100)
        except Exception as e:
            # self.encoder = None
            if verbose:
                print(f"Error building encoder: {e}")
            raise e


    def _build_decoder(self, verbose=False):
        """
        Builds the decoder component of the U-Net model.

        This method initializes the decoder using the `UNetDecoder` class, 
        passing the encoder shapes and the number of output classes as parameters. 
        It also updates the total parameter count by summing the number of 
        parameters in the decoder.

        Attributes:
            self.decoder (UNetDecoder): The decoder instance for the U-Net model.
            self.total_params (int): The total number of parameters in the model, 
                                     updated to include the decoder's parameters.
        """
        output_shape = (self.input_height, self.input_width)
        self.decoder = UNetDecoder(self.encoder_shapes, self.num_classes, output_shape)
        #self.decoder = UNetDecoder(self.encoder_shapes, self.num_classes)
        self.total_params += sum(p.numel() for p in self.decoder.parameters())
        if verbose:    
            print(f"{'U-Net Decoder Built Successfully!':^100}")
            print("_" * 100)
            print(f"{'- Total Parameters:':<25}{self.total_params:,}")
            print("=" * 100)
        
        

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor after passing through the encoder and decoder.
        """
        x = self.encoder_forward(x)
        # Here is x is a list of encoder outputs.
        x = self.decoder(x)
        return x


    @staticmethod
    def get_activation_fn(activation):
        """
        Retrieves the specified activation function from the `activations` module.

        Args:
            activation (str): The name of the activation function to retrieve.

        Returns:
            Callable: The activation function corresponding to the given name. 
                    If the specified activation function is not found, defaults to `activations.ReLU`.
        """
        return getattr(activations, activation, activations.ReLU)



def list_convolution_layers():
    """
    Retrieves a list of all classes defined in the `convolutions` module.

    This function uses the `inspect` module to dynamically inspect the 
    `convolutions` module and collect all objects that are classes.

    Returns:
        list: A list of class objects defined in the `convolutions` module.
    """
    # List all classes defined in the convolutions module
    return [obj for name, obj in inspect.getmembers(convolutions, inspect.isclass)]


def build_layer(layer, config, current_channels, current_height, current_width, idx, get_activation_fn):
    """
    Builds a neural network layer based on the provided configuration.
    Args:
        layer (dict): A dictionary containing the layer configuration. Must include the key 'layer_type' 
                      which specifies the type of layer to build.
        config (dict): A dictionary containing default configurations for various layer types.
        current_channels (int): The number of input channels to the layer.
        current_height (int): The height of the input tensor to the layer.
        current_width (int): The width of the input tensor to the layer.
        idx (int): The index of the layer in the model (used for debugging or logging purposes).
        get_activation_fn (callable): A function that takes an activation name (str) and returns the 
                                      corresponding activation function.
    Returns:
        tuple: A tuple containing:
            - layer_inst (nn.Module): The instantiated layer object.
            - current_channels (int): The number of output channels after the layer.
            - current_height (int): The height of the output tensor after the layer.
            - current_width (int): The width of the output tensor after the layer.
    Raises:
        ValueError: If the 'layer_type' in the layer dictionary is unknown or unsupported.
    Supported Layer Types:
        - 'ConvAct', 'ConvBnAct', 'ConvSE': Convolutional layers with optional batch normalization 
          and activation.
        - 'MBConv', 'MBConvNoRes': MobileNetV2-style inverted residual blocks.
        - 'CSPConvBlock', 'CSPMBConvBlock': Cross Stage Partial blocks for convolution or MBConv.
        - 'DenseNetBlock': DenseNet-style block with concatenated outputs.
        - 'ResNetBlock': ResNet-style residual block.
        - 'AvgPool', 'MaxPool': Pooling layers (average or max pooling).
        - 'Dropout': Dropout layer for regularization.
    """
    lt = layer['layer_type']
    if lt in ['ConvAct', 'ConvBnAct', 'ConvSE']:
        kernel_size, stride, padding, out_channels, new_height, new_width = parse_conv_params(
            layer, config, lt, current_channels, current_height, current_width
        )
        conv_cls = {
            'ConvAct': convolutions.ConvAct,
            'ConvBnAct': convolutions.ConvBnAct,
            'ConvSE': convolutions.ConvSE,
        }[lt]
        layer_inst = conv_cls(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels, current_height, current_width = out_channels, new_height, new_width
        assert new_height > 0 and new_width > 0, f"Invalid output dimensions: height ({new_height}) width ({new_width}) is zero."
        
    elif lt in ['MBConv', 'MBConvNoRes']:
        expansion_factor = int(layer.get('expansion_factor', config[lt]['default_expansion_factor']))
        dw_kernel_size = int(layer.get('dw_kernel_size', config[lt]['default_dw_kernel_size']))
        conv_cls = {'MBConv': convolutions.MBConv, 'MBConvNoRes': convolutions.MBConvNoRes}[lt]
        layer_inst = conv_cls(
            in_channels=current_channels,
            out_channels=current_channels,
            expansion_factor=expansion_factor,
            dw_kernel_size=dw_kernel_size,
            activation=get_activation_fn(layer['activation']),
        )
        
    elif lt in ['CSPConvBlock', 'CSPMBConvBlock']:
        num_blocks = int(layer.get('num_blocks', config[lt]['default_num_blocks']))
        # Get out_channels from conv params helper even if we don't use all values.
        _, _, _, out_channels, _, _ = parse_conv_params(layer, config, lt, current_channels, current_height, current_width)
        if lt == 'CSPConvBlock':
            layer_inst = convolutions.CSPConvBlock(
                in_channels=current_channels,
                num_blocks=num_blocks,
                activation=get_activation_fn(layer['activation']),
            )
        else:
            expansion_factor = int(layer.get('expansion_factor', config[lt]['default_expansion_factor']))
            dw_kernel_size = int(layer.get('dw_kernel_size', config[lt]['default_dw_kernel_size']))
            layer_inst = convolutions.CSPMBConvBlock(
                in_channels=current_channels,
                expansion_factor=expansion_factor,
                dw_kernel_size=dw_kernel_size,
                num_blocks=num_blocks,
                activation=get_activation_fn(layer['activation']),
            )
        current_channels = out_channels

    elif lt == 'DenseNetBlock':
        out_channels_coeff = float(layer.get('out_channels_coefficient', config['DenseNetBlock']['default_out_channels_coefficient']))
        out_channels = int(current_channels * out_channels_coeff)
        layer_inst = convolutions.DenseNetBlock(
            in_channels=current_channels,
            out_channels=out_channels,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels += out_channels

    elif lt == 'ResNetBlock':
        _ = float(layer.get('out_channels_coefficient', config['ResNetBlock']['default_out_channels_coefficient']))
        layer_inst = convolutions.ResNetBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            activation=get_activation_fn(layer['activation']),
        )

    elif lt in ['AvgPool', 'MaxPool']:
        pool_cls = pooling.AvgPool if lt == 'AvgPool' else pooling.MaxPool
        kernel_size = int(layer.get('kernel_size', config[lt]['default_kernel_size']))
        stride = int(layer.get('tride', config[lt]['default_stride']))
        layer_inst = pool_cls(kernel_size=kernel_size, stride=stride)
        current_height = ((current_height - kernel_size) // stride) + 1
        current_width = ((current_width - kernel_size) // stride) + 1

    elif lt == 'Dropout':
        dropout_rate = float(layer.get('dropout_rate', config['Dropout']['default_dropout_rate']))
        layer_inst = nn.Dropout(p=dropout_rate)

    else:
        raise ValueError(f"Unknown layer type: {lt}")

    return layer_inst, current_channels, current_height, current_width


def parse_conv_params(layer, config, key, current_channels, current_height, current_width):
    """
    Parse convolutional layer parameters and calculate output dimensions.
    
    This function extracts parameters for a convolutional layer from the provided configuration,
    calculates the output dimensions, and returns all necessary values for setting up
    a convolutional layer.
    
    Args:
        layer (dict): Dictionary containing layer-specific configuration parameters.
        config (dict): Dictionary containing default configuration parameters.
        key (str): Key to access specific configurations within the config dictionary.
        current_channels (int): Number of input channels for the current layer.
        current_height (int): Height of the input feature map.
        current_width (int): Width of the input feature map.
    
    Returns:
        tuple: A tuple containing:
            - kernel_size (int): Size of the convolutional kernel.
            - stride (int): Stride of the convolution.
            - padding (int): Padding added to input feature map.
            - out_channels (int): Number of output channels.
            - new_height (int): Height of the output feature map after convolution.
            - new_width (int): Width of the output feature map after convolution.
    """
    kernel_size = int(layer.get('kernel_size', config[key]['default_kernel_size']))
    stride = int(layer.get('stride', config[key]['default_stride']))
    padding = int(layer.get('padding', config[key]['default_padding']))
    out_channels_coeff = float(layer.get('out_channels_coefficient', config[key]['default_out_channels_coefficient']))
    out_channels = int(current_channels * out_channels_coeff)
    new_height = ((current_height - kernel_size + 2 * padding) // stride) + 1
    new_width = ((current_width - kernel_size + 2 * padding) // stride) + 1
    return kernel_size, stride, padding, out_channels, new_height, new_width