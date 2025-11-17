import torch
import pytest
from simple_policy import SimplePolicy


@pytest.fixture
def input_data():
    return torch.randn(5, 10)


@pytest.fixture
def num_layers():
    return 3


@pytest.fixture
def num_options():
    return 5


@pytest.mark.parametrize("num_layers,num_options", [(3, 5), (5, 7), (7, 10)])
def test_simplepolicy_shape(num_layers, num_options):
    policy = SimplePolicy(num_layers, num_options)
    output = policy()
    # Test output shape
    assert output.shape == (num_layers, num_options)
    # Test output is a probability distribution
    assert torch.allclose(output.sum(dim=1), torch.ones(num_layers))


if __name__ == "__main__":
    pytest.main([__file__])
