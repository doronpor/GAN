import torch.nn as nn


def basic_weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
