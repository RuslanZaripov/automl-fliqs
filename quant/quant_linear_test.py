import torch
import pytest
from torch.nn import Linear
from quant.quant_linear import QuantLinear

torch.manual_seed(15)


@pytest.mark.parametrize(
    "input_tensor, weight_bit, activation_bit, per_channel, act_percentile",
    [
        (torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), 8, 8, False, 0),
        (torch.tensor([[0.5, -0.8, 1.2]], dtype=torch.float32), 4, 8, False, 0),
        (torch.tensor([[-1.2, 0.8, -0.5]], dtype=torch.float32), 4, 4, False, 0),
        (torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), 4, 4, True, 0),
    ],
)
def test_quant_linear(
    input_tensor, weight_bit, activation_bit, per_channel, act_percentile
):
    linear = Linear(in_features=3, out_features=2)
    quant_linear = QuantLinear(
        linear,
        weight_bit=weight_bit,
        activation_bit=activation_bit,
        per_channel=per_channel,
        act_percentile=act_percentile,
    )

    # Store the original weights and bias
    original_weight = linear.weight.clone()
    original_bias = linear.bias.clone()

    # Check if the weights and bias are the same
    assert torch.allclose(quant_linear.weight, original_weight), "Weights mismatch"
    assert torch.allclose(quant_linear.bias, original_bias), "Bias mismatch"

    output = quant_linear(input_tensor)
    assert output.shape == (
        input_tensor.shape[0],
        quant_linear.weight.shape[0],
    ), f"Output shape mismatch"

    # Check the quantization error
    float_output = linear(input_tensor)
    quantization_error = torch.mean(torch.abs(float_output - output))

    # If per_channel is used, compare the quantization error with the error without per_channel
    if per_channel:
        quant_linear_no_per_channel = QuantLinear(
            linear,
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            per_channel=False,
            act_percentile=act_percentile,
        )
        output_no_per_channel = quant_linear_no_per_channel(input_tensor)
        quantization_error_no_per_channel = torch.mean(
            torch.abs(float_output - output_no_per_channel)
        )

        assert (
            quantization_error < quantization_error_no_per_channel
        ), f"Quantization error with per_channel should be less than without per_channel"


if __name__ == "__main__":
    pytest.main([__file__])
