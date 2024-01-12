import numpy as np
import tifffile
from torch.utils.data import Dataset as BaseDataset
import config as config


class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        class_values_list (list): values of classes
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = config.CLASSES

    def __init__(
        self,
        images_dir_list,
        class_values_list,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = range(len(images_dir_list))
        self.images_dir_list = images_dir_list
        self.class_values_list = class_values_list

        # functions
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = tifffile.imread(self.images_dir_list[i])

        # one hot labels
        label_one_hot = np.eye(self.CLASSES)[self.class_values_list[i]]

        # label name
        name = self.images_dir_list[i]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, label_one_hot, name

    def __len__(self):
        return len(self.ids)
