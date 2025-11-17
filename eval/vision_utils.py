from typing import Dict, Tuple
import torch
from quant.quant_conv2d import QuantConv2D
from quant.quant_linear import QuantLinear
from transformers.models.resnet.modeling_resnet import ResNetConvLayer, ResNetModel
from transformers.models.efficientnet.modeling_efficientnet import (
    EfficientNetBlock,
    EfficientNetModel,
)
from transformers import AutoModelForImageClassification


def get_vision_model(model_type: str):
    """Get the vision model class types for a given model type"""
    model = AutoModelForImageClassification.from_pretrained(model_type)

    return model


def quantize_vision_model(
    model: torch.nn.Module,
    weight_bits: int = 8,
    activation_bits: int = 8,
    per_channel: bool = False,
    act_range_percentile: float = 0,
    weight_range_percentile: float = 0,
) -> Tuple[torch.nn.Module, Dict[str, QuantConv2D]]:
    """Quantize all nn.Conv2d layers within a given resnet model"""
    quantized_modules = {}

    def replace_layer_with_quantized_conv(layer, layer_name):
        new_layer = QuantConv2D(
            layer,
            weight_bit=weight_bits,
            activation_bit=activation_bits,
            per_channel=per_channel,
            act_percentile=act_range_percentile,
            weight_percentile=weight_range_percentile,
        )
        quantized_modules[layer_name] = new_layer
        return new_layer

    def replace_layer_with_quantized_linear(layer, layer_name):
        new_layer = QuantLinear(
            layer,
            weight_bit=weight_bits,
            activation_bit=activation_bits,
            per_channel=per_channel,
            act_percentile=act_range_percentile,
            weight_percentile=weight_range_percentile,
        )
        quantized_modules[layer_name] = new_layer
        return new_layer

    for name, m in model.named_modules():
        if isinstance(m, (ResNetModel, EfficientNetModel)):
            if hasattr(m, "classifier"):
                m.classifier = replace_layer_with_quantized_linear(
                    m.classifier, name + ".classifier"
                )
        elif isinstance(m, ResNetConvLayer):
            m.convolution = replace_layer_with_quantized_conv(
                m.convolution, name + ".convolution"
            )
        elif isinstance(m, EfficientNetBlock):
            if hasattr(m, "expansion"):
                # expansion
                m.expansion.expand_conv = replace_layer_with_quantized_conv(
                    m.expansion.expand_conv, name + ".expand_conv"
                )

            # depthwise convolution
            m.depthwise_conv.depthwise_conv = replace_layer_with_quantized_conv(
                m.depthwise_conv.depthwise_conv, name + ".depthwise_conv"
            )
            # squeeze and excite
            m.squeeze_excite.reduce = replace_layer_with_quantized_conv(
                m.squeeze_excite.reduce, name + ".squeeze_excite_reduce"
            )
            m.squeeze_excite.expand = replace_layer_with_quantized_conv(
                m.squeeze_excite.expand, name + ".squeeze_excite_expand"
            )
            # projection
            m.projection.project_conv = replace_layer_with_quantized_conv(
                m.projection.project_conv, name + ".project_conv"
            )

    return model, quantized_modules
