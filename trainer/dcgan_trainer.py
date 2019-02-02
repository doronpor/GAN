from GAN.trainer.trainer_interface import GanTrainerInterface
from GAN.network.dcgan_net import Discriminator, Generator
from torch import optim

import torch


class DcGanTrainer(GanTrainerInterface):
    """
    trainer for the dcgan architecture.
    (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    (2) Update G network: maximize log(D(G(z)))
    """
    def __init__(self, cfg: dict):
        # Initialization of decoder and encoder
        self._generator = Generator(cfg)
        self._discriminator = Discriminator(cfg)

        # Initialization of loss
        self.criterion = torch.nn.BCELoss()

        # Initialization of labels
        batch_size = cfg['train']['batch_size']
        smooth_label = cfg['train']['smooth_alpha']
        self.label_real_dis = torch.full((batch_size,), 1 - smooth_label)
        self.label_fake_dis = torch.full((batch_size,), 0)
        self.label_gen = torch.full((batch_size,), 1)

    def to(self, device):
        # move generator and discriminator to device
        self._generator.to(device)
        self._discriminator.to(device)

        # move labels to device
        self.label_real_dis = self.label_real_dis.to(device)
        self.label_fake_dis = self.label_fake_dis.to(device)
        self.label_gen = self.label_gen.to(device)

    def discriminator(self):
        return self._discriminator

    def generator(self):
        return self._generator

    def gen_loss(self, input):
        return self.criterion(input, self.label_gen)

    def dis_loss_real(self, input):
        return self.criterion(input, self.label_real_dis)

    def dis_loss_fake(self, input):
        return self.criterion(input, self.label_fake_dis)

    def optimizer(self):
        return optim.Adam
