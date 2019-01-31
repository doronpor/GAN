import torch
import torch.nn as nn
from torch.nn.modules import Module

from GAN.utils.weight_init import basic_weight_init


class Discriminator(Module):
    """
    wgan_gp discriminator. This is actually a critic (see article)
    """
    def __init__(self, cfg: dict):
        super(Discriminator, self).__init__()

        net_params = cfg['network']
        num_features = net_params['discriminator']['num_features']

        self.net = nn.Sequential(
            nn.Conv2d(net_params['num_image_channels'], num_features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1),
            nn.InstanceNorm2d(num_features * 2, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1),
            nn.InstanceNorm2d(num_features * 4, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1),
            nn.InstanceNorm2d(num_features * 8, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 8, 1, 4, 1, 0),
        )

        # init model weights
        self.apply(basic_weight_init(type='noraml'))

    def forward(self, input):
        return self.net(input)


class Generator(Module):
    """
    wgan_gp generator
    """
    def __init__(self, cfg: dict):
        super(Generator, self).__init__()

        net_params = cfg['network']
        self.latent_vector = net_params['generator']['latent_vector']
        num_features = net_params['generator']['num_features']

        # decoder net init
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(net_params['generator']['latent_vector'], num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features, net_params['num_image_channels'], 4, 2, 1),
            nn.Tanh()
        )

        # init model weights
        self.apply(basic_weight_init(type='noraml'))

    def forward(self, input):
        return self.net(input)

    def generate_image(self, batch_size, device):
        # randomize
        noise = torch.randn(batch_size, self.latent_vector, 1, 1, device=device)

        # run forward
        return self.forward(noise)
