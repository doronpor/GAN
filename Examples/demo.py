import torch
import matplotlib.pyplot as plt
from typing import Union
from GAN.trainer.trainer_factory import factory


def demo(cfg: dict, model_path: str = None, device='cpu', num_images: int = 10, fps: Union[int, float] = 1):
    """
    demo generate images from noise of a trained generator
    :param cfg: GAN architecture
    :param model_path: generator model path
    :param num_images: number of images to generate
    :param fps: frames per second
    :return:
    """

    # Initiate generator
    generator = factory(cfg).generator()

    # move generator to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator.to(device)

    # load model
    model = torch.load(model_path, map_location=device)
    generator.load_state_dict(model['generator'])

    # run
    generator.eval()
    plt.figure(figsize=(1.5, 1.5))
    plt.axis("off")

    for _ in range(num_images):
        # generate image
        with torch.no_grad():
            image = generator.generate_image(batch_size=1, device=device).cpu().numpy()[0]
        image = image.transpose([1, 2, 0])
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)  # normalize image
        plt.imshow(image)
        plt.show(block=False)
        plt.pause(1 / float(fps))
