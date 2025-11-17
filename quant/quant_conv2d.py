import torch.nn.functional as F
from torch.nn import Parameter
from quant.quant_module import QuantModule


class QuantConv2D(QuantModule):
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
        super(QuantConv2D, self).__init__(
            layer,
            weight_bit,
            activation_bit,
            per_channel,
            act_percentile,
            weight_percentile,
            per_token,
        )

        self.weight = Parameter(layer.weight.data.clone())
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.groups = layer.groups

    def forward(self, x):
        quantized_x, quantized_w = self._quantize_input_and_weights(x)
        out = F.conv2d(
            quantized_x,
            quantized_w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        return out
