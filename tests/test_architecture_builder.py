import random

from pynas.core import architecture_builder as builder
from pynas.core.config import default_config_path, load_default_config


def test_default_config_is_packaged():
    config = load_default_config()

    assert default_config_path().name == "config.ini"
    assert config.has_section("ConvAct")
    assert config.has_section("Upsample")
    assert config.getint("GA", "max_parameters") > 0


def test_parse_architecture_code_extracts_layer_parameters():
    parsed = builder.parse_architecture_code("Lco02k3s1p1arEE")

    assert parsed == [
        {
            "layer_type": "ConvBnAct",
            "out_channels_coefficient": 2,
            "kernel_size": "3",
            "stride": "1",
            "padding": "1",
            "activation": "ReLU",
        }
    ]


def test_generate_code_from_parsed_architecture_round_trips():
    parsed = [
        {
            "layer_type": "ConvBnAct",
            "out_channels_coefficient": 2,
            "kernel_size": "3",
            "stride": "1",
            "padding": "1",
            "activation": "ReLU",
        }
    ]

    code = builder.generate_code_from_parsed_architecture(parsed)

    assert code == "Lco02k3s1p1arEE"
    assert builder.parse_architecture_code(code) == parsed


def test_generate_random_architecture_uses_bundled_config():
    random.seed(7)

    code = builder.generate_random_architecture_code(min_layers=1, max_layers=1)
    parsed = builder.parse_architecture_code(code)

    assert code.endswith("EE")
    assert len(parsed) >= 2
    assert parsed[0]["layer_type"] != "Unknown"
