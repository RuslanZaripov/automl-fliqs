import torch.nn.functional as F
from torch.nn import Parameter
from quant.quant_module import QuantModule


class QuantLinear(QuantModule):
    def __init__(
        self,
        layer,
        weight_bit=4,
        activation_bit=4,
        per_channel=False,
        act_percentile=0,
        weight_percentile=0,
        per_token=False,
        transpose=False,
    ):
        super(QuantLinear, self).__init__(
            layer,
            weight_bit,
            activation_bit,
            per_channel,
            act_percentile,
            weight_percentile,
            per_token,
        )

        if transpose:
            self.weight = Parameter(layer.weight.data.clone().t())
        else:
            self.weight = Parameter(layer.weight.data.clone())

    def forward(self, x):
        quantized_x, quantized_w = self._quantize_input_and_weights(x)
        linear_out = F.linear(quantized_x, weight=quantized_w, bias=self.bias)

        return linear_out
