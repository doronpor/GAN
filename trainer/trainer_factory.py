from GAN.trainer.trainer_interface import GanTrainerInterface
from GAN.trainer.dcgan_trainer import DcGanTrainer


def factory(cfg: dict) -> GanTrainerInterface:
    if cfg['type'] == 'DCGan':
        return DcGanTrainer(cfg)
    else:
        raise TypeError('Gan of type %s is not supported' % cfg['type'])
