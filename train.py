"""
main training loop for generative adversarial networks
"""

import argparse
import logging
import torch
import os.path as path
import os

from torch import optim
import torchvision.utils as vutils

from GAN.network.dcgan_net import Generator, Discriminator
from GAN.datasets.folder_loader import get_folder_data_loader
from GAN.utils.base_logger import set_base_logger
from GAN.utils.configuration import load_config
from GAN.utils.gif_animation import animate_gif

set_base_logger()
logger = logging.getLogger('train')


def train(cfg: dict, model_path=None, debug=False) -> list:
    """
    main training file for GAN networks
    :param cfg: network configuration file
    :param model_path:  model saving path
    :param debug:   true for debug output
    :return: discriminator generated image list
    """
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")

    # net initialization
    d_net = Discriminator(cfg)
    g_net = Generator(cfg)

    d_net.to(device)
    g_net.to(device)

    # optimizer initialization
    d_optimizer = optim.Adam(d_net.parameters(),
                             lr=cfg['train']['learning_rate'],
                             betas=(cfg['train']['beta1'], 0.999))
    g_optimizer = optim.Adam(g_net.parameters(),
                             lr=cfg['train']['learning_rate'],
                             betas=(cfg['train']['beta1'], 0.999))

    # load snapshot
    # todo load snapshot from config

    # define loss
    criterion = torch.nn.BCELoss()

    # define labels
    batch_size = cfg['train']['batch_size']
    label_real = torch.full((batch_size,), 1, device=device)
    label_gen = torch.full((batch_size,), 0, device=device)

    # load dataset
    data_loader = get_folder_data_loader(cfg['train']['dataset_path'],
                                         cfg['train']['batch_size'],
                                         cfg['train']['workers'])

    if debug:
        # fix noise to create images for debug mode
        z_noise = torch.randn(64, cfg['generator']['latent_vector'], 1, 1, device=device)
        gen_image_list = []
    else:
        gen_image_list = None

    # main train_loop
    logger.info('starting training loop')
    for epoch in range(cfg['train']['epochs']):
        for i, data in enumerate(data_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            d_optimizer.zero_grad()

            # train real image batch
            real_images = data[0].to(device)
            d_output = d_net(real_images).view(-1)
            loss_d_real = criterion(d_output, label_real)
            loss_d_real.backward()

            # train gen image batch
            gen_image = g_net.generate_image(batch_size, device=device)
            d_output = d_net(gen_image.detach()).view(-1)
            loss_d_gen = criterion(d_output, label_gen)
            loss_d_gen.backward()

            loss_d = loss_d_real + loss_d_gen
            # discriminator param update
            d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_optimizer.zero_grad()

            # train generator
            d_output = d_net(gen_image).view(-1)
            loss_g = criterion(d_output, label_real)
            loss_g.backward()
            g_optimizer.step()

            # plot training loss
            if i % 50 == 0:
                logger.info('epoch: [%d/%d] batch: [%d/%d],  Loss_D: %.4f,  Loss_G: %.4f'
                            % (epoch, cfg['train']['epochs'], i, len(data_loader),
                               loss_d.cpu().item(), loss_g.cpu().item()))

            # create generator images for debug
            if debug:
                if (i % 500 == 0) or ((epoch == cfg['train']['epochs'] - 1) and (i == len(data_loader) - 1)):
                    with torch.no_grad():
                        gen_fake = g_net(z_noise).detach().cpu()
                    gen_image_list.append(vutils.make_grid(gen_fake.detach().cpu(), padding=2, normalize=True))

        # save models
        if not path.isdir(path.dirname(model_path)):
            os.mkdir(path.dirname(model_path))

        save_dict = {'generator': g_net.state_dict(),
                     'discriminator': d_net.state_dict()}

        torch.save(save_dict, model_path)

    return gen_image_list


if __name__ == '__main__':
    # load configuration path
    parser = argparse.ArgumentParser(description='training configuration path')
    parser.add_argument('--config', action='store', default='./cfgs/dcgan.yaml', help='configuration path')
    parser.add_argument('--debug', action='store_true', help='generate images from generator')
    parser.add_argument('--gif_path', action='store', default='./generator.gif', help='gif path')
    parser.add_argument('--model_path', action='store', default='./models/dcgan.pth', help='module save path')

    args = parser.parse_args()
    logger.info('using config path: %s' % path.abspath(args.config))

    # load config
    cfg = load_config(args.config)

    # run train
    model_path = path.abspath(path.join(path.dirname(__file__), args.model_path))
    debug_images = train(cfg, model_path=model_path, debug=args.debug)

    # create and save gif
    if args.debug:
        gif_path = path.abspath(path.join(path.dirname(__file__), args.gif_path))
        animate_gif(debug_images, gif_path)
