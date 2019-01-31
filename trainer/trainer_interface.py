from abc import ABC
from typing import Type

from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer
import torch


class GanTrainerInterface(ABC):
    """
    Interface for GAN trainer. Main training loop expects instance of This class.
    """
    def discriminator(self) -> Module:
        raise NotImplementedError('decoder method not implemented')

    def generator(self) -> Module:
        raise NotImplementedError('encoder method not implemented')

    def gen_loss(self, input: torch.Tensor) -> torch.Tensor:
        """
        compute the generator loss
        :param input: generator output G(x)
        :return:
        """
        raise NotImplementedError('loss method not implemented')

    def dis_loss_real(self, input: torch.Tensor) -> torch.Tensor:
        """
        compute discriminator loss for real images
        :param input: discriminator output of real image D(x)
        :return:
        """
        raise NotImplementedError('loss method not implemented')

    def dis_loss_fake(self, input: torch.Tensor) -> torch.Tensor:
        """
        compute discriminator loss for fake image
        :param input: discriminator output for fake image D(G(z))
        :return:
        """
        raise NotImplementedError('loss method not implemented')

    def optimizer(self) -> Type[Optimizer]:
        raise NotImplementedError('optimizer method not implemented')

    def to(self, device: torch.device):
        raise NotImplementedError('to method not implemented')

