import pytest
from quant.quant_utils import *


@pytest.mark.parametrize(
    "num_bits, x_abs_max, expected",
    [
        (8, torch.tensor(127.0), torch.tensor(1.0)),
        (16, torch.tensor(32767.0), torch.tensor(1.0)),
        (8, torch.tensor([64.0, 32.0]), torch.tensor([0.5, 0.25])),
    ],
)
def test_symmetric_linear_scale(num_bits, x_abs_max, expected):
    result = symmetric_linear_scale(num_bits, x_abs_max)
    assert torch.allclose(
        result, expected, atol=1e-2
    ), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "input_tensor, k, specified_scale",
    [
        (
            torch.tensor([1.2, 3.5, 4.7], dtype=torch.float64, requires_grad=True),
            8,
            torch.tensor(1.0),
        ),
        (
            torch.tensor([-2.5, 1.8, 0.6], dtype=torch.float64, requires_grad=True),
            8,
            torch.tensor(2.0),
        ),
        (
            torch.tensor([0.1, -0.1, 0.8], dtype=torch.float64, requires_grad=True),
            4,
            torch.tensor(0.5),
        ),
    ],
)
def test_symmetric_quant_function(input_tensor, k, specified_scale):
    symmetric_quant = SymmetricQuantFunction.apply

    # Test forward pass
    result = symmetric_quant(input_tensor, k, specified_scale)
    n = 2 ** (k - 1) - 1
    quantized_values = torch.round(input_tensor / specified_scale.view(-1, 1))
    expected = torch.clamp(quantized_values, -n - 1, n)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    # Test backward pass (gradient computation)
    input_tensor.grad = None
    result.sum().backward()
    assert input_tensor.grad is not None, "No gradient was computed"

    # Check that the gradient is not zero
    assert not torch.all(input_tensor.grad == 0), "All gradient values are zero"


if __name__ == "__main__":
    pytest.main([__file__])
