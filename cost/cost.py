from torch.utils.tensorboard import SummaryWriter


def one_sided_tunas_abs(
    cost: float,
    cost_beta: float,
    cost_target: float,
    acc: float,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Return the one-sided tunas_abs reward"""
    cost_ratio = max(1, cost / cost_target)
    cost_advantage = abs(1 - cost_ratio)
    cost_penalty = cost_beta * cost_advantage
    reward = acc - cost_penalty

    writer.add_scalar("reward/reward", reward, step)
    writer.add_scalar("reward/cost_ratio", cost_ratio, step)
    writer.add_scalar("reward/cost_penalty", cost_penalty, step)
    writer.add_scalar("reward/penalty_acc_ratio", abs(cost_penalty / acc), step)

    return reward


def tunas_abs(
    cost: float,
    cost_beta: float,
    cost_target: float,
    acc: float,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Return the tunas_abs reward"""
    cost_ratio = cost / cost_target
    cost_advantage = abs(1 - cost_ratio)
    cost_penalty = cost_beta * cost_advantage
    reward = acc - cost_penalty

    writer.add_scalar("reward/reward", reward, step)
    writer.add_scalar("reward/cost_ratio", cost_ratio, step)
    writer.add_scalar("reward/cost_penalty", cost_penalty, step)
    writer.add_scalar("reward/penalty_acc_ratio", abs(cost_penalty / acc), step)

    return reward


def soft_exponential(
    cost: float,
    cost_beta: float,
    cost_target: float,
    acc: float,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Return the soft exponential reward"""
    cost_ratio = cost / cost_target
    reward = acc * cost_ratio**cost_beta

    writer.add_scalar("reward/reward", reward, step)
    writer.add_scalar("reward/cost_ratio", cost_ratio, step)

    return reward


def hard_exponential(
    cost: float,
    cost_beta: float,
    cost_target: float,
    acc: float,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Return the hard exponential reward"""
    cost_ratio = cost / cost_target
    cost_scale = max(1, cost_ratio) ** cost_beta
    reward = acc * cost_scale

    writer.add_scalar("reward/reward", reward, step)
    writer.add_scalar("reward/cost_ratio", cost_ratio, step)
    writer.add_scalar("reward/cost_scale", cost_scale, step)

    return reward
