import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import logging

logger = logging.getLogger('gif_animation')


def animate_gif(image_list: list, path: str = None, fps=1.5, writer='imagemagick'):
    """
    create a gid animation from list of images
    :param path: path for saving the animation (None for showing instead of saving)
    :param image_list: list of images for the animation
    :param fps: frames per second
    :param writer: writer type (imagemagick default requires installation)
    :return:
    """
    logging.info('Creating gif animation')
    # create gif animation
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in image_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    if path is not None:
        # save gif animation
        try:
            anim.save(path, writer=writer, fps=fps)
        except Exception as err:
            logger.error('error while trying to create the gif')
            raise err
    else:
        plt.show()
