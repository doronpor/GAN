"""
main training loop for generative adversarial networks
"""

import argparse
import logging
from GAN.utils.base_logger import set_base_logger
from GAN.utils.configuration import load_config

set_base_logger()
logger = logging.getLogger('train')


def train(cfg):
    pass


if __name__ == '__main__':
    # load configuration path
    parser = argparse.ArgumentParser(description='training configuration path')
    parser.add_argument('--config', action='store', default='./cfgs/dcgan.yaml', help='configuration path')

    args = parser.parse_args()
    logger.info('using config path %s' % args.config)

    # load config
    cfg = load_config(args.config)

    # run train
    train(cfg)
