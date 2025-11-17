from torch import nn
import torch


class SimplePolicy(nn.Module):
    """Simple policy that directly learns logits."""
    def __init__(self, num_layers, num_options, temperature=1.0):
        super(SimplePolicy, self).__init__()
        self.logits = nn.Parameter(torch.empty(num_layers, num_options))
        nn.init.constant_(self.logits, 0.5)
        self.temperature = temperature

    def forward(self):
        return torch.softmax(self.logits / self.temperature, dim=-1)
