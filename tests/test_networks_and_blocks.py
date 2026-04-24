import torch

from pynas.blocks.activations import ReLU
from pynas.blocks.convolutions import ConvAct
from pynas.blocks.heads import MultiInputClassifier
from pynas.core.generic_unet import GenericUNetNetwork


def test_conv_act_preserves_expected_spatial_shape():
    block = ConvAct(
        in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, activation=ReLU
    )
    output = block(torch.randn(2, 3, 8, 8))

    assert output.shape == (2, 4, 8, 8)


def test_multi_input_classifier_combines_feature_maps():
    classifier = MultiInputClassifier(
        input_shapes=[(4, 4, 4), (8,)],
        common_dim=6,
        mlp_depth=2,
        mlp_hidden_dim=5,
        num_classes=3,
        pool_size=(1, 1),
    )

    output = classifier([torch.randn(2, 4, 4, 4), torch.randn(2, 8)])

    assert output.shape == (2, 3)


def test_generic_unet_forward_runs_on_cpu():
    parsed_layers = [
        {
            "layer_type": "ConvBnAct",
            "out_channels_coefficient": 2,
            "kernel_size": "3",
            "stride": "1",
            "padding": "1",
            "activation": "ReLU",
        }
    ]
    model = GenericUNetNetwork(
        parsed_layers=parsed_layers,
        input_channels=3,
        input_height=16,
        input_width=16,
        num_classes=2,
    )

    output = model(torch.randn(1, 3, 16, 16))

    assert output.shape == (1, 2, 16, 16)
    assert model.total_params > 0
