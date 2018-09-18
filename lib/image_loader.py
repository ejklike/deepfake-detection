import os
from scandir import scandir
from random import shuffle
import numpy as np
import cv2

from torch import from_numpy
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from lib.utils import get_image_paths_and_labels


class ToTensor(object):
    """
    Convert ndarray image to Tensor.
    (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    """
    def __call__(self, image):
        def _to_tensor(image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1)).astype(np.float32)
            return from_numpy(image)

        image = _to_tensor(image)
        return image


class ImageLoader(Dataset):
    def __init__(self, paths_and_labels):
        self.image_paths_and_labels = paths_and_labels
        self.transform = transforms.Compose([ToTensor()])

    def __getitem__(self, index):
        path, label = self.image_paths_and_labels[index]
        image = self.read_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths_and_labels)

    def _normalize(self, image):
        """
        scale from (0, 255) to (0, 1)
        """
        return image / 255.0

    def read_image(self, image_path):
        try:
            image = self._normalize(cv2.imread(image_path))
        except TypeError:
            raise Exception("Error while reading image", image_path)
        return image


# MEMORY OVERFLOW
# class ImageLoader(Dataset):
#     def __init__(self, paths_and_labels):
#         self.image_paths_and_labels = paths_and_labels
#         self.transform = transforms.Compose([ToTensor()])
#         self.images_and_labels = None

#     def __getitem__(self, index):
#         return self.images_and_labels[index]

#     def __len__(self):
#         return len(self.image_paths_and_labels)

#     def normalize(self, image):
#         """
#         scale from (0, 255) to (0, 1)
#         """
#         return image / 255.0

#     def read_image(self, image_path):
#         try:
#             image = self.normalize(cv2.imread(image_path))
#         except TypeError:
#             raise Exception("Error while reading image", image_path)
#         return image

#     def load_data(self, augment=True):
#         self.images_and_labels = []
#         for path, label in self.image_paths_and_labels:
#             image = self.read_image(path)
#             if self.transform is not None:
#                 image = self.transform(image)
#             self.images_and_labels.append([image, label])
#         return self.images_and_labels

#     def clear_data(self):
#         self.images_and_labels = None