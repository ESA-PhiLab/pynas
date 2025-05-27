convolution_layer_vocabulary = {
    'b': 'ConvAct',
    'c': 'ConvBnAct',
    'e': 'ConvSE',
    'd': 'DenseNetBlock',
    'm': 'MBConv',
    'n': 'MBConvNoRes',
    #'o': 'CSPConvBlock',
    #'p': 'CSPMBConvBlock',
    'R': 'ResNetBlock',
    'D': 'Dropout',  # Add Dropout to the convolution layer vocabulary
}

activation_functions_vocabulary = {
    'r': 'ReLU',
    'g': 'GELU',
    'sg' : 'Sigmoid',
    's': 'Softmax',  # Added Softmax
}

pooling_layer_vocabulary = {
    'a': 'AvgPool',
    'M': 'MaxPool',
}



upsampling_layer_vocabulary = {
    'U': 'Upsample',
}

skip_connection_layer_vocabulary = {
    'S': 'SkipConnection',
}

layer_parameters = {
    'ConvAct': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'ConvBnAct': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'ConvSE': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'DenseNetBlock': ['out_channels_coefficient', 'activation'],
    'MBConv': ['expansion_factor', 'activation'],
    'MBConvNoRes': ['expansion_factor', 'activation'],
    'CSPConvBlock': ['num_blocks', 'activation'],
    'CSPMBConvBlock': ['num_blocks', 'expansion_factor', 'activation'],
    'ResNetBlock': ['reduction_factor', 'activation'],
    'AvgPool': [],
    'MaxPool': [],
    'ClassificationHead': [],
    'Dropout': ['dropout_rate'],  # Define the parameter for Dropout
    'Upsample': ['scale_factor', 'mode'], 
    'SkipConnection': [],  # Skip connection does not need additional parameters
}

parameter_vocabulary = {
    'kernel_size': 'k',
    'stride': 's',
    'padding': 'p',
    'out_channels_coefficient': 'o',
    'expansion_factor': 'e',
    'num_blocks': 'n',
    'reduction_factor': 'r',
    'activation': 'a',
    'dropout_rate': 'd',  # Add parameter for dropout probability
    'scale_factor': 'f',  # Add parameter for scale factor
    'mode': 'm',  # Add parameter for mode
}
