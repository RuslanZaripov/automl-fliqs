import torch
from torch.nn import Module, Parameter, Linear
from quant.quant_utils import *
from typing import Dict, Any, Tuple

FULL_PRECISION_BITS = 16
NUM_PERCENTILE_STEPS = 10
MEGA = 10**6


class QuantModule(Module):
    def __init__(
        self,
        layer,
        weight_bit=4,
        activation_bit=4,
        per_channel=False,
        act_percentile=0,
        weight_percentile=0,
        per_token=False,
    ):
        super(QuantModule, self).__init__()
        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        self.per_channel = per_channel
        self.quant_function = SymmetricQuantFunction.apply
        self.act_percentile = act_percentile
        self.per_token = per_token
        self.weight_percentile = (
            weight_percentile
            if weight_percentile >= 0
            else self._get_optimal_percentile(layer.weight, self._quantize_weight)
        )
        self.register_buffer("x_abs_max", None)

        try:
            self.bias = Parameter(layer.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _get_optimal_percentile(self, x, quant_function):
        """Get the optimal percentile for quantization."""
        num_steps = NUM_PERCENTILE_STEPS
        step_size = 1 / num_steps
        min_error = float("inf")
        optimal_percentile = 0

        for i in range(1, num_steps):
            current_percentile = i * step_size
            quantized_x = quant_function(x, current_percentile)

            # Calculate mean-squared error
            quantization_error = torch.mean((x - quantized_x) ** 2)
            if quantization_error < min_error:
                min_error = quantization_error
                optimal_percentile = current_percentile

        return optimal_percentile

    def __repr__(self):
        s = super(QuantModule, self).__repr__()
        s = (
            "(" + s + " weight_bit={}, activation_bit={}, per_channel={},"
            ", act_percentile={}, weight_percentile={}, per_token={})"
        ).format(
            self.weight_bit,
            self.activation_bit,
            self.per_channel,
            self.act_percentile,
            self.weight_percentile,
            self.per_token,
        )
        return s

    def _quantize_weight(self, weight, weight_percentile):
        if self.per_channel:
            w_abs_max = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        else:
            w_abs_max = torch.abs(weight).max().expand(1)

        if weight_percentile > 0:
            w_abs_max = weight_percentile * w_abs_max

        weight_scale = symmetric_linear_scale(self.weight_bit, w_abs_max)
        quantized_w = (
            self.quant_function(self.weight, self.weight_bit, weight_scale)
            * weight_scale
        )

        return quantized_w

    def _quantize_activation(self, x, act_percentile):
        if self.per_token:
            x_abs_max = x.data.abs().max(dim=-1, keepdim=True).values
        else:
            x_abs_max = x.data.abs().max().view(1)

        if act_percentile > 0:
            x_abs_max = act_percentile * x_abs_max

        if self.x_abs_max is None:
            self.x_abs_max = x_abs_max.clone().detach()
        else:
            self.x_abs_max = x_abs_max

        act_scale = symmetric_linear_scale(self.activation_bit, self.x_abs_max)
        quantized_x = self.quant_function(x, self.activation_bit, act_scale) * act_scale

        return quantized_x

    def _quantize_input_and_weights(self, x):
        if self.activation_bit > 0:
            quantized_x = self._quantize_activation(x, self.act_percentile)
        else:
            quantized_x = x

        if self.weight_bit > 0:
            weight = self.weight.data.detach()
            quantized_w = self._quantize_weight(weight, self.weight_percentile)
        else:
            quantized_w = self.weight

        return quantized_x, quantized_w

    def forward(self, x):
        raise NotImplementedError

    def get_memory_cost(self, include_bias: bool = False) -> float:
        weight_bits = self.weight.numel() * self.weight_bit
        # weight_bits = number of elements in weight tensor * bits per weight
        bias_bits = (self.bias.numel() if include_bias else 0) * FULL_PRECISION_BITS
        # bias_bits = number of elements in bias tensor * bits per bias

        total_bits = weight_bits + bias_bits
        total_bytes = total_bits / 8
        total_mib = total_bytes / (1024 * 1024)

        return total_mib

    @torch.no_grad()
    def get_compute_cost(
        self, include_bias: bool = False, bops_exponent: int = 2
    ) -> int:
        average_bitwidth = (self.weight_bit + self.activation_bit) / 2
        linear_compute = self.weight.numel() * (average_bitwidth**bops_exponent)
        bias_compute = (self.bias.numel() if include_bias else 0) * (
            FULL_PRECISION_BITS**bops_exponent
        )
        bops = linear_compute + bias_compute
        gbops = bops / MEGA

        return gbops


def model_cost(model, include_bias: bool = False, cost_type: str = "memory") -> float:
    """Compute the  cost of a BERT model only considering the QLinear layers"""
    cost = 0
    for name, m in model.named_modules():
        if isinstance(m, QuantModule):
            if cost_type == "memory":
                cost += m.get_memory_cost(include_bias=include_bias)
            elif cost_type == "compute":
                cost += m.get_compute_cost(include_bias=include_bias)
            else:
                raise ValueError(f"Unsupported cost type: {cost_type}")

    return cost


def model_weight_percentiles(model) -> Dict[str, Any]:
    """Compute the percentile percentiles of a BERT model"""
    percentiles = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantModule):
            percentiles[name] = m.weight_percentile

    return percentiles
