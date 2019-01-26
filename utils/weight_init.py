import torch.nn as nn
from torch.nn.modules.conv import _ConvNd


def basic_weight_init(slop=0, non_linearity='relu', type='kaiming_uniform'):
    """
    return a weight_init method for convolution and batchnorm initialization.
    convolutional layer are initialized using kaimin_uniform initialization
    :param type: type of convolution normalization 'normal', 'kaiming_uniform', 'xavier_normal'
    :param slop: slop of the non linearity
    :param non_linearity: 'relu' or 'leaky_relu'
    :return: weight initialization method
    """
    def weight_init(module):
        if isinstance(module, _ConvNd):
            # both conv and conv transposed inherit from _ConvNd
            if type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight.data, a=slop, nonlinearity=non_linearity)
            elif type == 'xavier':
                nn.init.xavier_normal_(module.weight.data, gain=nn.init.calculate_gain(non_linearity))
            elif type == 'noraml':
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            else:
                raise TypeError('the type of convolution normalization is not supported')

            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    return weight_init
