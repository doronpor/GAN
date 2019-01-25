import torch
import torch.nn as nn
from torch.nn.modules import Module

from GAN.utils.weight_init import basic_weight_init


class Discriminator(Module):
    """
    dcgan discriminator
    """
    def __init__(self, cfg: dict):
        super(Discriminator, self).__init__()

        num_features = cfg['discriminator']['num_features']

        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(cfg['num_image_channels'], num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # init model weights
        self.apply(basic_weight_init)

    def forward(self, input):
        return self.net(input)


class Generator(Module):
    """
    dcgan generator
    """
    def __init__(self, cfg: dict):
        super(Generator, self).__init__()

        self.latent_vector = cfg['generator']['latent_vector']
        num_features = cfg['generator']['num_features']

        # decoder net init
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(cfg['generator']['latent_vector'], num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features, cfg['num_image_channels'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # init model weights
        self.apply(basic_weight_init)

    def forward(self, input):
        return self.net(input)

    def generate_image(self, batch_size, device):
        # randomize
        noise = torch.randn(batch_size, self.latent_vector, 1, 1, device=device)

        # run forward
        return self.forward(noise)
