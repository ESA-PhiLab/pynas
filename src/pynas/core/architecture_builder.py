
# architecture_builder.py
import random
import configparser
from .vocabulary import convolution_layer_vocabulary, layer_parameters, parameter_vocabulary
from .vocabulary import activation_functions_vocabulary, upsampling_layer_vocabulary, pooling_layer_vocabulary 



def generate_random_architecture_code(min_layers: int = 3, max_layers: int = 5):
    """
    Generates a random architecture code string consisting of layers and pooling layers.
    The function creates a sequence of encoder layers and pooling layers, appending them
    to form a string representation of an architecture. Each layer and pooling layer is 
    separated by an "E". The architecture ends with an additional "E".
    Args:
        min_layers (int): The minimum number of layers to include in the architecture.
        max_layers (int): The maximum number of layers to include in the architecture.
    Returns:
        str: A string representing the randomly generated architecture code.
    """
    architecture_code = ""
    encoder_layers = []

    for _ in range(random.randint(min_layers, max_layers)):
        layer_code = generate_layer_code()
        encoder_layers.append(layer_code)
        architecture_code += layer_code + "E"
        
        pooling_layer_code = generate_pooling_layer_code()
        #pooling_factors.append(pooling_factor)
        architecture_code += pooling_layer_code + "E"
    
    # Insert ender
    architecture_code += "E"
    return architecture_code


def generate_layer_code():
    """
    Generates a string representation of a neural network layer configuration.
    This function randomly selects a layer type from a predefined vocabulary and 
    generates a corresponding layer code based on its parameters. The parameters 
    are configured using values from a `config.ini` file. The generated code 
    includes details such as activation type, kernel size, padding, stride, 
    dropout rate, and other layer-specific attributes.
    Returns:
        str: A string representing the configuration of the generated layer.
    """
    def get_random_value(param, section, is_float=False, is_odd=False, format_str=None):
        """Helper function to get a random value for a parameter."""
        min_val = config.getfloat(section, f'min_{param}') if is_float else config.getint(section, f'min_{param}')
        max_val = config.getfloat(section, f'max_{param}') if is_float else config.getint(section, f'max_{param}')
        value = random.uniform(min_val, max_val) if is_float else random.randint(min_val, max_val)
        
        if is_odd and value % 2 == 0:
            value += 1
            if value > max_val:
                value -= 2
        
        return f"{value:.2f}" if is_float else f"{value:02d}" if format_str == "02d" else str(value)

    layer_type = random.choice(list(convolution_layer_vocabulary.keys()))
    parameters = layer_parameters[convolution_layer_vocabulary[layer_type]]
    layer_code = f"L{layer_type}"

    config = configparser.ConfigParser()
    config.read('config.ini')
    section = convolution_layer_vocabulary[layer_type]

    for param in parameters:
        code = parameter_vocabulary.get(param, "")
        if param == 'activation':
            activation_code = random.choice(['r', 'g'])  # Use only ReLU and GELU for intermediate layers
            layer_code += f"a{activation_code}"
        elif param == 'dropout_rate':
            layer_code += f"{code}{get_random_value(param, section, is_float=True)}"
        elif param == 'kernel_size':
            layer_code += f"{code}{get_random_value(param, section, is_odd=True)}"
        elif param in ['padding', 'stride', 'out_channels_coefficient']:
            format_str = "02d" if param == 'out_channels_coefficient' else None
            layer_code += f"{code}{get_random_value(param, section, format_str=format_str)}"
        elif param != 'num_blocks':  # Skip 'num_blocks' as it's not used
            layer_code += f"{code}{get_random_value(param, section)}"

    layer_code += "n1"
    return layer_code

def generate_pooling_layer_code():
    """
    Generates a string representing a pooling layer configuration.

    The function randomly selects a pooling type from the `pooling_layer_vocabulary` 
    and combines it with a predefined pooling factor to create a pooling layer code.

    Returns:
        str: A string representing the pooling layer configuration in the format 
             "P<pooling_type><pooling_factor>".
    """
    pooling_type = random.choice(list(pooling_layer_vocabulary.keys()))
    pooling_factor = 2  # Example factor, you can randomize or configure this as needed
    pooling_code = f"P{pooling_type}{pooling_factor}"
    return pooling_code #, pooling_factor


def generate_upsampling_layer_code(scale_factor=2):
    """
    Generates a string representing the configuration of an upsampling layer.

    This function reads the available upsampling modes from a configuration file
    ('config.ini') under the 'Upsample' section and randomly selects one of the modes.
    It then combines the scale factor and the selected mode into a formatted string.

    Args:
        scale_factor (int, optional): The scaling factor for the upsampling layer. 
            Defaults to 2.

    Returns:
        str: A formatted string representing the upsampling layer configuration 
        in the format "Uf{scale_factor}m{mode}".
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    modes = config.get('Upsample', 'modes').split(',')
    mode = random.choice(modes)

    upsampling_code = f"Uf{scale_factor}m{mode}"
    return upsampling_code


def generate_skip_connection_code(layer_index):
    """
    Generates a string representing a skip connection identifier for a given layer index.

    Args:
        layer_index (int): The index of the layer for which the skip connection identifier is generated.

    Returns:
        str: A string in the format "S{layer_index}" representing the skip connection identifier.
    """
    return f"S{layer_index}"



def parse_architecture_code(architecture_code):
    """
    Parses a given architecture code string into a list of layer configurations.
    The function interprets the architecture code by splitting it into segments,
    identifying the type of each segment (e.g., convolution, pooling, upsampling, etc.),
    and extracting the associated parameters based on predefined vocabularies and rules.
    Args:
        architecture_code (str): A string representing the architecture code. Each segment
            of the code corresponds to a layer or operation, with specific characters
            denoting the type and parameters of the layer.
    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a parsed layer
            or operation. Each dictionary contains:
            - 'layer_type' (str): The type of the layer (e.g., "Convolution", "Pooling").
            - Additional keys for parameters specific to the layer type, such as:
                - 'scale_factor' (int): The scale factor for upsampling layers.
                - 'mode' (str): The mode for certain operations.
                - 'dropout_rate' (float): The dropout rate for layers with dropout.
                - 'activation' (str): The activation function for layers with activation.
                - 'out_channels_coefficient' (int): Coefficient for output channels.
                - Other parameters as defined in the layer's parameter vocabulary.
    Notes:
        - The function relies on several predefined vocabularies and mappings:
            - `convolution_layer_vocabulary`: Maps codes to convolution layer types.
            - `pooling_layer_vocabulary`: Maps codes to pooling layer types.
            - `head_vocabulary`: Maps codes to head layer types.
            - `upsampling_layer_vocabulary`: Maps codes to upsampling layer types.
            - `layer_parameters`: Defines expected parameters for each layer type.
            - `parameter_vocabulary`: Maps parameter codes to parameter names.
            - `activation_functions_vocabulary`: Maps activation codes to function names.
        - Segments with unknown or unsupported codes are assigned "Unknown" as the layer type.
        - Skip connections are explicitly identified with the type "SkipConnection".
    Example:
        architecture_code = "L1f2mRPE2H3"
        parsed_layers = parse_architecture_code(architecture_code)
        # parsed_layers will be a list of dictionaries representing the parsed layers.
    """
    segments = architecture_code.split('E')[:-1]
    parsed_layers = []

    for segment in segments:
        if not segment:  # Skip empty segments
            continue
        
        segment_type_code = segment[0]
        layer_type_code = segment[1]
        
        # Determine the segment's layer type and corresponding parameters
        if segment_type_code == 'L':
            layer_type = convolution_layer_vocabulary.get(layer_type_code, "Unknown")
        elif segment_type_code == 'P':
            layer_type = pooling_layer_vocabulary.get(layer_type_code, "Unknown")
        elif segment_type_code == 'U':
            layer_type = upsampling_layer_vocabulary.get(segment_type_code, "Unknown")
        elif segment_type_code == 'S':  # For skip connections
            layer_type = "SkipConnection"
        
        # Initialize the dictionary for this segment with its type
        segment_info = {'layer_type': layer_type}
        
        # Get parameter definitions for this layer type
        param_definitions = layer_parameters.get(layer_type, [])

        # Process remaining characters based on the expected parameters for this type
        params = segment[2:]  # All after layer type code
        
        i = 0
        while i < len(params):
            param_code = params[i]
            i += 1
            if param_code == 'f' and 'scale_factor' in param_definitions:
                scale_factor = ''
                while i < len(params) and params[i].isdigit():
                    scale_factor += params[i]
                    i += 1
                segment_info['scale_factor'] = int(scale_factor)
            elif param_code == 'm' and 'mode' in param_definitions:
                segment_info['mode'] = params[i]
                i += 1
            else:
                for param_name, code in parameter_vocabulary.items():
                    if code == param_code and param_name in param_definitions:
                        if param_name == 'dropout_rate':
                            value = params[i:i + 4]
                            i += 4
                            segment_info[param_name] = float(value)
                        elif param_name == 'activation':
                            segment_info[param_name] = activation_functions_vocabulary.get(params[i], "Unknown")
                            i += 1
                        elif param_name == 'out_channels_coefficient':
                            value = params[i:i + 2]
                            i += 2
                            segment_info[param_name] = int(value)
                        else:
                            segment_info[param_name] = params[i]
                            i += 1
                        break
        
        parsed_layers.append(segment_info)

    return parsed_layers

def generate_code_from_parsed_architecture(parsed_layers):
    """
    Generates a compact string representation of a neural network architecture 
    based on a list of parsed layer configurations.
    The function converts each layer's type and parameters into a coded segment 
    using predefined vocabularies and appends them together to form the final 
    architecture code. Each layer segment ends with "E", and the entire 
    architecture code ends with "EE".
    Args:
        parsed_layers (list of dict): A list of dictionaries where each dictionary 
            represents a layer configuration. Each dictionary must contain a 
            'layer_type' key and may include additional parameters specific to 
            the layer type.
    Returns:
        str: A string representing the encoded architecture.
    Notes:
        - The function uses reverse mappings of predefined vocabularies to encode 
          layer types and parameters.
        - Special handling is applied for certain parameters like 'activation', 
          'dropout_rate', and 'out_channels_coefficient'.
        - Layer types such as "Dropout", "Upsample", and "SkipConnection" have 
          specific encoding rules.
    """
    architecture_code = ""
    
    # Utilize the provided configuration directly
    reverse_convolution_layer_vocabulary = {v: k for k, v in convolution_layer_vocabulary.items()}
    reverse_pooling_layer_vocabulary = {v: k for k, v in pooling_layer_vocabulary.items()}
    reverse_activation_functions_vocabulary = {v: k for k, v in activation_functions_vocabulary.items()}
    reverse_upsampling_functions_vocabulary = {v: k for k, v in upsampling_layer_vocabulary.items()}

    for layer in parsed_layers:
        layer_type = layer['layer_type']
        segment_code = ""
        
        # Prepend the type code with "L", "P", or "H" based on the layer type
        if layer_type in reverse_convolution_layer_vocabulary:
            segment_code += "L" + reverse_convolution_layer_vocabulary[layer_type]
        elif layer_type in reverse_pooling_layer_vocabulary:
            segment_code += "P" + reverse_pooling_layer_vocabulary[layer_type]
        elif layer_type == 'Dropout':
            segment_code += "LD"
        elif layer_type == "Upsample": 
            segment_code += "U" + reverse_upsampling_functions_vocabulary[layer_type]
        elif layer_type == "SkipConnection":
            segment_code += "S"

        # Append each parameter and its value
        for param_name, param_value in layer.items():
            if param_name == 'layer_type':  # Skip 'layer_type' as it's already processed
                continue
            
            if param_name in parameter_vocabulary:
                param_code = parameter_vocabulary[param_name]
                
                # Special handling for activation parameters
                if param_name == 'activation':
                    param_value = reverse_activation_functions_vocabulary.get(param_value, param_value)
                
                segment_code += param_code + str(param_value)
            elif layer_type == 'Dropout' and param_name == 'dropout_rate':
                segment_code += f"d{param_value:.2f}"
            elif param_name == 'out_channels_coefficient':
                segment_code += f"o{param_value:02d}"
        
        # Finalize the segment and add it to the architecture code
        architecture_code += segment_code + "E"
    
    # Ensure the architecture code properly ends with "EE"
    return architecture_code + "E"
