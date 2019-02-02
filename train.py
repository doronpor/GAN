"""
main training loop for generative adversarial networks
"""

import argparse
import logging

import tensorboardX
import torch
import os.path as path
import os

import torchvision.utils as vutils

from GAN.trainer.trainer_factory import factory
from GAN.datasets.folder_loader import get_folder_data_loader
from GAN.utils.base_logger import set_base_logger
from GAN.utils.configuration import load_config
from GAN.utils.gif_animation import animate_gif

set_base_logger()
logger = logging.getLogger('train')


def train(cfg: dict, model_path: str = None, debug=False) -> list:
    """
    main training file for GAN networks
    :param cfg: network configuration file
    :param model_path:  model saving path
    :param debug:   true for debug output
    :return: discriminator generated image list
    """
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")

    # initialize trainer and move to device
    trainer = factory(cfg)
    trainer.to(device)

    # net initialization
    d_net = trainer.discriminator()
    g_net = trainer.generator()

    # switch to train mode
    d_net.train()
    g_net.train()

    # optimizer initialization
    d_optimizer = trainer.optimizer()(d_net.parameters(), **cfg['train']['optimizer'])
    g_optimizer = trainer.optimizer()(g_net.parameters(), **cfg['train']['optimizer'])

    # load snapshot
    # todo load snapshot from config

    # load dataset
    data_loader = get_folder_data_loader(cfg['train']['dataset_path'],
                                         cfg['train']['batch_size'],
                                         cfg['train']['workers'])

    if debug:
        # fix noise to create images for debug mode
        z_noise = torch.randn(64, cfg['network']['generator']['latent_vector'], 1, 1, device=device)
        gen_image_list = []
    else:
        z_noise = None
        gen_image_list = None

    # Initialize tensorboard
    tb_path = path.abspath(path.join(path.dirname(__file__), cfg['train']['tensorboard_path']))
    writer = tensorboardX.SummaryWriter(tb_path)

    # main train_loop
    batch_size = cfg['train']['batch_size']
    logger.info('starting training loop')
    for epoch in range(cfg['train']['epochs']):
        for i, data in enumerate(data_loader):
            step = epoch * len(data_loader) + i

            ############################
            # (1) Update Discriminator network
            ###########################

            d_optimizer.zero_grad()

            # train real image batch
            real_images = data[0].to(device)
            d_output_real = d_net(real_images).view(-1)
            loss_d_real = trainer.dis_loss_real(d_output_real)
            loss_d_real.backward()

            # train gen image batch
            gen_image = g_net.generate_image(batch_size, device=device)
            d_output_fake = d_net(gen_image.detach()).view(-1)
            loss_d_gen = trainer.dis_loss_fake(d_output_fake)
            loss_d_gen.backward()

            # add regularization
            reg_loss = trainer.reg_loss(real_images.detach(), gen_image.detach())
            if reg_loss is not None:
                reg_loss.backward()

            # discriminator param update
            d_optimizer.step(None)

            loss_d = loss_d_real + loss_d_gen

            ############################
            # (2) Update Generator network
            ###########################
            # update the G network every dis_iter steps
            if step % cfg['train']['dis_iter'] == 0:
                g_optimizer.zero_grad()

                # train generator
                gen_image = g_net.generate_image(batch_size, device=device)
                d_output = d_net(gen_image).view(-1)
                loss_g = trainer.gen_loss(d_output)
                loss_g.backward()
                g_optimizer.step(None)

            # plot training loss
            if step % 50 == 0 and step != 0:
                # plot log to screen
                if reg_loss is None:
                    logger.info('epoch: [%d/%d] batch: [%d/%d],  Loss_D: %.4f,  Loss_G: %.4f'
                                % (epoch, cfg['train']['epochs'], i, len(data_loader),
                                   loss_d.cpu().item(), loss_g.cpu().item()))

                    # plot log to tensorboard
                    writer.add_scalars('Loss', {
                        'Loss_D': loss_d.cpu().item(),
                        'Loss_G': loss_g.cpu().item()
                    }, global_step=step)

                else:
                    logger.info('epoch: [%d/%d] batch: [%d/%d],  Loss_D: %.4f,  Loss_G: %.4f,  Reg: %.4f'
                                % (epoch, cfg['train']['epochs'], i, len(data_loader),
                                   loss_d.cpu().item(), loss_g.cpu().item(), reg_loss.cpu().item()))

                    # plot log to tensorboard
                    writer.add_scalars('Loss', {
                        'Loss_D': loss_d.cpu().item(),
                        'Loss_G': loss_g.cpu().item(),
                        'Loss_Reg': reg_loss.cpu().item()
                    }, global_step=step)

            # create generator images for debug
            if debug:
                if (i % 500 == 0) or ((epoch == cfg['train']['epochs'] - 1) and (i == len(data_loader) - 1)):
                    with torch.no_grad():
                        gen_fake = g_net(z_noise).detach().cpu()
                    gen_image_list.append(vutils.make_grid(gen_fake.detach().cpu(), padding=2, normalize=True))

            # save models at the end of the epoch
            if not path.isdir(path.dirname(model_path)):
                os.mkdir(path.dirname(model_path))

        save_dict = {'epoch': epoch,
                     'generator': g_net.state_dict(),
                     'discriminator': d_net.state_dict()}

        torch.save(save_dict, model_path)

    return gen_image_list


if __name__ == '__main__':
    # load configuration path
    parser = argparse.ArgumentParser(description='training configuration path')
    parser.add_argument('--config', action='store', default='./cfgs/dcgan.yaml', help='configuration path')
    parser.add_argument('--debug', action='store_true', help='generate images from generator')
    parser.add_argument('--gif_path', action='store', default='./generator_sample.gif', help='gif path')
    parser.add_argument('--model_path', action='store', default='./models/dcgan.pth', help='module save path')

    args = parser.parse_args()
    logger.info('using config path: %s' % path.abspath(args.config))

    # load config
    config = load_config(args.config)

    # run train
    model_path = path.abspath(path.join(path.dirname(__file__), args.model_path))
    debug_images = train(config, model_path=model_path, debug=args.debug)

    # create and save gif
    if args.debug:
        gif_path = path.abspath(path.join(path.dirname(__file__), args.gif_path))
        animate_gif(debug_images, gif_path)
