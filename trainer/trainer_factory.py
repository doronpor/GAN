from GAN.trainer.trainer_interface import GanTrainerInterface
from GAN.trainer.dcgan_trainer import DcGanTrainer
from GAN.trainer.wgan_gp_trainer import WGanGpTrainer


def factory(cfg: dict) -> GanTrainerInterface:
    if cfg['type'] == 'DCGan':
        return DcGanTrainer(cfg)
    elif cfg['type'] == 'Wgan_GP':
        return WGanGpTrainer(cfg)
    else:
        raise TypeError('Gan of type %s is not supported' % cfg['type'])
