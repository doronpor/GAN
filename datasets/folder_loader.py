import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os.path as path
import logging

logger = logging.getLogger('dataset')


def get_folder_data_loader(dataset_path: str, batch_size: int, workers: int, transform=None,
                           shuffle=True) -> torch.utils.data.DataLoader:
    """
    retrieves dataloader based on ImageFolder dataset
    :param dataset_path: dataset path (ImageFolder requires to have a subfloder)
    :param batch_size: image batch size
    :param workers: number of workers for dataloader
    :param transform: transform for the images (see torchvision.transforms)
    :param shuffle: shuffle dataloader images
    :return:
    """
    if transform is None:
        image_size = 64
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset_path = path.abspath(path.join(path.dirname(__file__), dataset_path))
    logger.info('dataset path: %s' % dataset_path)
    dataset = dset.ImageFolder(root=dataset_path,
                               transform=transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=workers, drop_last=True)

    return dataloader
