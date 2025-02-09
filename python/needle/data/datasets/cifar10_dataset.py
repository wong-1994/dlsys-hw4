import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        import pickle
        if train:
            batch_size = 10000
            total_size = 5 * batch_size
            self.X = np.empty((total_size, 3072), dtype=np.float32)
            self.y = np.empty((total_size,), dtype=np.int8)
            for i in range(1, 6):
                fp = os.path.join(base_folder, f'data_batch_{i}')
                with open(fp, 'rb') as fo:
                    cifar_dict = pickle.load(fo, encoding='bytes')
                    self.X[(i-1)*batch_size : i*batch_size] = cifar_dict[b'data']
                    self.y[(i-1)*batch_size : i*batch_size] = cifar_dict[b'labels']
        else:
            fp = os.path.join(base_folder, 'test_batch')
            with open(fp, 'rb') as fo:
                cifar_dict = pickle.load(fo, encoding='bytes')
                self.X = cifar_dict[b'data']
                self.y = np.array(cifar_dict[b'labels'])
        self.X = self.X.reshape(-1, 3, 32, 32) / 255.0
        self.p = p
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # X = self.X[index] / 255.0
        # X = X.reshape(3, 32, 32)
        img = self.X[index]
        label = self.y[index]
        if self.transforms:
            img = self.apply_transforms(img)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
