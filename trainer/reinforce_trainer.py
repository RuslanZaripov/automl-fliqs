from torch import nn
import torch
from transformers import Trainer
from typing import Dict, Union, Any, List
from policy.simple_policy import SimplePolicy
import re
from quant.quant_linear import QuantLinear
from cost.cost import tunas_abs
import math
from torch.utils.tensorboard import SummaryWriter

HEAVISIDE_PERCENTAGE = 0.50


class ReinforceConfig:
    """Configuration for the ReinforceTrainer"""

    def __init__(
        self,
        delay_steps: int = 500,
        cost_beta: float = 1.0,
        ema_alpha: float = 0.99,
        cost_target: int = 50,
        temperature: float = 1.0,
        learning_rate: float = 5e-2,
        cost_type: str = "memory",
        cost_function=tunas_abs,
        max_entropy_beta: float = 0.1,
        entropy_schedule_type: str = "linear",
        weight_only: bool = False,
        single_decision: bool = False,
        block_partitions=None,
    ) -> None:
        self.delay_steps = delay_steps
        self.cost_beta = cost_beta
        self.ema_alpha = ema_alpha
        self.cost_target = cost_target
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.cost_type = cost_type
        self.cost_function = cost_function
        self.max_entropy_beta = max_entropy_beta
        self.entropy_schedule_type = entropy_schedule_type
        self.weight_only = weight_only
        self.single_decision = single_decision
        self.block_partitions = block_partitions


class ReinforceTrainer(Trainer):
    def __init__(
        self,
        quantized_modules: Dict[str, QuantLinear],
        search_options: List[int],
        model: nn.Module,
        training_args: Any,
        reinforce_config: ReinforceConfig,
        log_dir: str = "output/log",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, training_args, **kwargs)
        self.seed = seed
        self.device = training_args.device
        # Unclear why the seed needs to be set again this code, but it does for the torch.randint call
        torch.manual_seed(self.seed)

        # Assign ReinforceConfig fields to the ReinforceTrainer instance
        config_fields = [
            field for field in dir(reinforce_config) if not field.startswith("__")
        ]
        for field in config_fields:
            setattr(self, field, getattr(reinforce_config, field))

        # Only support QLinear for now
        for module in quantized_modules.values():
            assert isinstance(module, QuantLinear)
        self.quantized_modules = quantized_modules
        self.last_bits = {
            name: module.weight_bit for name, module in self.quantized_modules.items()
        }
        self.search_options = search_options
        if self.single_decision:
            self.num_decisions = 1
        elif self.block_partitions is not None:
            # Create mapping from the partition to name of the quantized module
            self.partition_to_name = {}
            # Check if all quantized_modules belong to one and only one partition
            for module_name in quantized_modules.keys():
                matching_partitions = [
                    pattern
                    for pattern in self.block_partitions
                    if re.match(pattern, module_name)
                ]
                num_partitions = len(matching_partitions)
                if num_partitions != 1:
                    raise ValueError(
                        f"The quantized module '{module_name}' should belong to one and only one partition."
                    )
                # Map list of module names to the partition name
                matching_partition = matching_partitions[0]
                if matching_partition in self.partition_to_name.keys():
                    self.partition_to_name[matching_partition].append(module_name)
                else:
                    self.partition_to_name[matching_partition] = [module_name]
            self.num_decisions = len(self.block_partitions)
        else:
            self.num_decisions = len(quantized_modules)
        num_options = len(search_options)
        self.policy_network = SimplePolicy(
            self.num_decisions, num_options, temperature=reinforce_config.temperature
        )
        self.policy_network.to(training_args.device)

        self.rl_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=reinforce_config.learning_rate
        )
        self.current_step = 0
        self.ema_reward = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = self.total_steps(training_args)

    def total_steps(self, trainer_args):
        """Calculate the total number of steps during training to use with entropy schedule"""
        if trainer_args.max_steps > 0:
            return trainer_args.max_steps
        else:
            train_dataloader = self.get_train_dataloader()
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // trainer_args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

            return math.ceil(trainer_args.num_train_epochs * num_update_steps_per_epoch)

    def entropy_beta_schedule(self, current_step: int) -> float:
        """Return the current entropy beta value based on the current step and the entropy schedule type"""
        if self.entropy_schedule_type == "linear":
            return self.linear_schedule(current_step)
        elif self.entropy_schedule_type == "cosine":
            return self.cosine_schedule(current_step)
        elif self.entropy_schedule_type == "constant":
            return self.constant_schedule()
        elif self.entropy_schedule_type == "heaviside":
            return self.heaviside_schedule(current_step)
        else:
            raise ValueError(
                f"Invalid entropy schedule type: {self.entropy_schedule_type}"
            )

    def linear_schedule(self, current_step: int) -> float:
        """Linearly increase the entropy beta value from 0 to max_entropy_beta over the total number of steps"""
        if current_step <= self.delay_steps:
            return 0.0
        if current_step >= self.total_steps:
            return self.max_entropy_beta
        return (
            self.max_entropy_beta
            * (current_step - self.delay_steps)
            / (self.total_steps - self.delay_steps)
        )

    def heaviside_schedule(self, current_step: int) -> float:
        """Jump to max_entropy_beta after a delay"""
        switch_step = HEAVISIDE_PERCENTAGE * self.total_steps
        if current_step <= switch_step:
            return 0.0
        else:
            return self.max_entropy_beta

    def cosine_schedule(self, current_step: int) -> float:
        """Cosine increase the entropy beta value from 0 to max_entropy_beta over the total number of steps"""
        if current_step <= self.delay_steps:
            return 0.0
        if current_step >= self.total_steps:
            return self.max_entropy_beta
        adjusted_step = current_step - self.delay_steps
        adjusted_total_steps = self.total_steps - self.delay_steps
        return (
            self.max_entropy_beta
            * (1 - math.cos(math.pi * adjusted_step / adjusted_total_steps))
            / 2
        )

    def constant_schedule(self) -> float:
        """Return the max_entropy_beta value"""
        return self.max_entropy_beta

    def update_quantized_modules(self, indices: torch.Tensor) -> None:
        """Update the bit widths of the quantized modules based on the indices"""
        if self.block_partitions is not None:
            for i, (partition, module_names) in enumerate(
                self.partition_to_name.items()
            ):
                index = indices[i]
                for module_name in module_names:
                    self.last_bits[module_name] = self.search_options[index]
                    self.quantized_modules[module_name].weight_bit = (
                        self.search_options[index]
                    )
                    if not self.weight_only:
                        self.quantized_modules[module_name].activation_bit = (
                            self.search_options[index]
                        )
        else:
            for i, (name, module) in enumerate(self.quantized_modules.items()):
                index = indices[i] if not self.single_decision else indices[0]
                self.last_bits[name] = self.search_options[index]
                module.weight_bit = self.search_options[index]
                if not self.weight_only:
                    module.activation_bit = self.search_options[index]

    def get_model_cost(self) -> float:
        """Return the cost of the current bit widths of the quantized modules"""
        cost = 0
        for name, module in self.quantized_modules.items():
            if self.cost_type == "memory":
                cost += module.get_memory_cost()
            elif self.cost_type == "compute":
                cost += module.get_compute_cost()
            else:
                raise ValueError(f"Invalid cost type: {self.cost_type}")
        return cost

    def _get_quant_prob_and_indices(self):
        if self.current_step < self.delay_steps:
            num_options = len(self.search_options)
            quant_probs = torch.ones(self.num_decisions, num_options) / num_options
            # quant_probs shape: (num_decisions, num_options)
            quant_sample_indices = torch.randint(num_options, (self.num_decisions,))
            # quant_sample_indices shape: (num_decisions,)
        else:
            # Get the quant_sample probabilities from the policy network
            quant_probs = self.policy_network()

            # Clamp probabilities to avoid numerical issues
            quant_probs = quant_probs.clamp(min=1e-8, max=1.0)

            # Renormalize to ensure probabilities sum to 1
            quant_probs = quant_probs / quant_probs.sum(dim=-1, keepdim=True)

            # Sample a quant_sample from the quant_sample probabilities
            quant_sample_indices = torch.multinomial(quant_probs, 1)

        return quant_probs, quant_sample_indices

    def _update_reward_ema(self, reward: float):
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = (
                self.ema_alpha * self.ema_reward + (1 - self.ema_alpha) * reward
            )

        return self.ema_reward

    def _update_policy(
        self,
        reward_advantage: float,
        quant_probs: torch.Tensor,
        quant_sample_indices: torch.Tensor,
    ):
        if not self.current_step < self.delay_steps:
            # Calculate the entropy regularization term
            log_probs = torch.log(quant_probs)
            entropy = -torch.sum(quant_probs * log_probs, dim=-1)
            mean_entropy = torch.mean(entropy)
            current_entropy_beta = self.entropy_beta_schedule(self.current_step)

            # Use torch.gather to pass gradients through the quant_sample_indices
            selected_log_probs = torch.gather(log_probs, -1, quant_sample_indices)

            # Standard REINFORCE loss with entropy regularization to discourage exploration
            policy_loss = (
                -reward_advantage * torch.mean(selected_log_probs)
                + current_entropy_beta * mean_entropy
            )
            self.rl_optimizer.zero_grad()
            policy_loss.backward()
            self.rl_optimizer.step()

            self.writer.add_scalar(
                "controller/policy_loss", policy_loss, self.current_step
            )
            self.writer.add_scalar(
                "controller/current_entropy_beta",
                current_entropy_beta,
                self.current_step,
            )
            self.writer.add_scalar(
                "controller/mean_entropy", mean_entropy, self.current_step
            )
            if self.block_partitions is not None:
                for i, (partition, module_names) in enumerate(
                    self.partition_to_name.items()
                ):
                    for name in module_names:
                        self.writer.add_scalar(
                            f"samples/{name}",
                            quant_sample_indices[i],
                            self.current_step,
                        )
                        for j, bit in enumerate(self.search_options):
                            self.writer.add_scalar(
                                f"format_probabilities/{name}/{bit}",
                                quant_probs[i, j],
                                self.current_step,
                            )

            else:
                for i, (name, _) in enumerate(self.quantized_modules.items()):
                    index = 0 if self.single_decision else i
                    self.writer.add_scalar(
                        f"samples/{name}",
                        quant_sample_indices[index],
                        self.current_step,
                    )
                    for j, bit in enumerate(self.search_options):
                        self.writer.add_scalar(
                            f"samples/{name}",
                            quant_sample_indices[index],
                            self.current_step,
                        )
                        self.writer.add_scalar(
                            f"format_probabilities/{name}/{bit}",
                            quant_probs[index, j],
                            self.current_step,
                        )

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: int = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs and update the quantized module bit widths.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Get the quant_sample probabilities and indices
        quant_probs, quant_sample_indices = self._get_quant_prob_and_indices()

        # print quant_probs values range
        print(
            f"Quant_probs range: min={quant_probs.min().item()}, max={quant_probs.max().item()}"
        )

        # Update the bit widths of the quantized modules
        self.update_quantized_modules(quant_sample_indices)

        # Combine the loss and cost to get the reward
        loss = super().training_step(model, inputs)
        cost = self.get_model_cost()
        # NOTE: reward function expects "higher is better",
        # but loss is "lower is better", so negate the loss
        acc = -loss
        reward = self.cost_function(
            cost, self.cost_beta, self.cost_target, acc, self.writer, self.current_step
        )
        # Update the EMA
        ema_reward = self._update_reward_ema(reward)
        reward_advantage = reward - self.ema_reward
        # Update the policy network
        self._update_policy(reward_advantage, quant_probs, quant_sample_indices)

        # Track reward metrics with TensorBoard
        self.writer.add_scalar("reward/cost", cost, self.current_step)
        self.writer.add_scalar("reward/acc", acc, self.current_step)
        self.writer.add_scalar("reward/ema_reward", ema_reward, self.current_step)
        self.writer.add_scalar(
            "reward/reward_advantage", reward_advantage, self.current_step
        )

        # Update the current step
        self.current_step += 1

        return loss
