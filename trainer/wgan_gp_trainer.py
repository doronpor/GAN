from typing import Type

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import optim

from GAN.trainer.trainer_interface import GanTrainerInterface
from GAN.network.wgan_gp import Discriminator, Generator


class WGanGpTrainer(GanTrainerInterface):
    """
    Trainer for the WGAN-GP architecture
    """

    def __init__(self, cfg: dict):
        # Initialization of decoder and encoder
        self._generator = Generator(cfg)
        self._discriminator = Discriminator(cfg)

        self.lambda_weight = cfg['train']['reg_weight']

    def discriminator(self) -> Module:
        return self._discriminator

    def generator(self) -> Module:
        return self._generator

    def gen_loss(self, input: torch.Tensor) -> torch.Tensor:
        return -input.mean()

    def dis_loss_real(self, input: torch.Tensor) -> torch.Tensor:
        return -input.mean()

    def dis_loss_fake(self, input: torch.Tensor) -> torch.Tensor:
        return input.mean()

    def reg_loss(self, image_real: torch.Tensor, image_fake: torch.Tensor):
        """
        gradient penalty loss. see "Improved Training of Wasserstein GANs article"
        :param image_real: real image
        :param image_fake: generated fake image
        :return:
        """
        # interpolate between images
        batch_size = image_real.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=image_real.device)
        image_interpolated = (alpha * image_real + (1 - alpha) * image_fake).detach()

        # set requires grad to differentiate with image respect
        image_interpolated.requires_grad_()

        dis_interpolated = self._discriminator(image_interpolated)

        # calculate grad_x(D(x))
        gradients = torch.autograd.grad(outputs=dis_interpolated, inputs=image_interpolated,
                                        grad_outputs=torch.ones_like(dis_interpolated),
                                        create_graph=True)[0]

        gradients = gradients.view(gradients.shape[0], -1)

        # calculate regularization loss
        gradient_penalty = self.lambda_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def optimizer(self) -> Type[Optimizer]:
        return optim.Adam

    def to(self, device: torch.device):
        # move generator and discriminator to device
        self._generator.to(device)
        self._discriminator.to(device)
