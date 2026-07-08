import torch
from arrakis.src.core_arrakis.activation_cache import HookedAutoConfig, HookedAutoModel


def make_model():
    config = HookedAutoConfig(
        name="llama",
        vocab_size=50256,
        hidden_size=8,
        intermediate_size=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return HookedAutoModel(config)


def test_register_hooks_returns_output_and_activations():
    model = make_model()
    input_ids = torch.randint(0, 50256, (1, 10))

    output, activations = model(input_ids)

    assert output is not None
    assert isinstance(activations, dict)
    assert "root" in activations
    assert "layers.0.hook_resid_post" in activations
