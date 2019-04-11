import os
import random

import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#import torchvision.transforms.functional as TF


# functional transform loader for segmentation task:
def seg_transformer(image, label, split):
    hflip = random.random() < 0.5
    if hflip and split != 'test':
        image, label = image[..., ::-1].copy(), label[..., ::-1].copy()
    image, label = torch.Tensor(image), torch.Tensor(label)

    return image, label.long()


class LANESDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, split, transform):
        """
        Store the hdf5 files of data to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
            split: the type of this dataset (e.g. train, valid or test)
        """
        self.X = h5py.File(os.path.join(data_dir, 'X_{}.hdf5'.format(split)), 'r')['image']
        self.y = h5py.File(os.path.join(data_dir, 'y_{}.hdf5'.format(split)), 'r')['image']
        self.split = split
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.X)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        """
        return self.transform(self.X[idx], self.y[idx], self.split)


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'valid', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'valid', 'test']:
        if split in types:
            dl = DataLoader(LANESDataset(data_dir, split, seg_transformer), batch_size=params.batch_size, shuffle=False,
                            num_workers=params.num_workers)

            dataloaders[split] = dl

    return dataloaders