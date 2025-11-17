import pytest
from torch.utils.tensorboard import SummaryWriter
from cost import tunas_abs, one_sided_tunas_abs, soft_exponential, hard_exponential


@pytest.fixture
def writer():
    return SummaryWriter()


@pytest.mark.parametrize("cost, cost_beta, cost_target, acc, expected_reward", [
    (15.0, 1.0, 5.0, 0.8, -1.2),
    (5.0, 1.0, 5.0, 0.9, 0.9),
    (3.0, 2.0, 5.0, 0.95, 0.15)
])
def test_tunas_abs(cost, cost_beta, cost_target, acc, expected_reward, writer):
    reward = tunas_abs(cost, cost_beta, cost_target, acc, writer, 0)
    assert pytest.approx(reward, rel=1e-3) == expected_reward


@pytest.mark.parametrize("cost, cost_beta, cost_target, acc, expected_reward", [
    (15.0, 1.0, 5.0, 0.8, -1.2),
    (5.0, 1.0, 5.0, 0.9, 0.9),
    (3.0, 2.0, 5.0, 0.95, 0.95)
])
def test_one_sided_tunas_abs(cost, cost_beta, cost_target, acc, expected_reward, writer):
    reward = one_sided_tunas_abs(cost, cost_beta, cost_target, acc, writer, 0)
    assert pytest.approx(reward, rel=1e-3) == expected_reward


@pytest.mark.parametrize("cost, cost_beta, cost_target, acc, expected_reward", [
    (10.0, 1.0, 5.0, 0.8, 1.6),
    (5.0, 1.0, 5.0, 0.9, 0.9),
    (3.0, 2.0, 6.0, 0.9, 0.225)
])
def test_soft_exponential(cost, cost_beta, cost_target, acc, expected_reward, writer):
    reward = soft_exponential(cost, cost_beta, cost_target, acc, writer, 0)
    assert pytest.approx(reward, rel=1e-3) == expected_reward


@pytest.mark.parametrize("cost, cost_beta, cost_target, acc, expected_reward", [
    (10.0, 1.0, 5.0, 0.8, 1.6),
    (5.0, 1.0, 5.0, 0.9, 0.9),
    (3.0, 2.0, 5.0, 0.95, 0.95)
])
def test_hard_exponential(cost, cost_beta, cost_target, acc, expected_reward, writer):
    reward = hard_exponential(cost, cost_beta, cost_target, acc, writer, 0)
    assert pytest.approx(reward, rel=1e-3) == expected_reward


if __name__ == "__main__":
    pytest.main([__file__])
