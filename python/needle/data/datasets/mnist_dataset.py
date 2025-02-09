from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import struct
import gzip

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # Read image file
    with gzip.open(image_filesname, "rb") as img_file:
        img_magic, = struct.unpack(">i", img_file.read(4))
        if img_magic != 2051:
            raise ValueError(f"MSB format parse fail.\
                 Expect 2051, but got {img_magic}")

        n_imgs, = struct.unpack(">i", img_file.read(4))

        n_rows, = struct.unpack(">i", img_file.read(4))
        n_cols, = struct.unpack(">i", img_file.read(4))
        if n_rows != 28 or n_cols != 28:
            raise ValueError(f"Data format parse fail.\
                Expect 28*28, but got {n_rows}*{n_cols}")

        X = np.empty((n_imgs, n_rows, n_cols, 1), dtype = np.float32)
        for i in range(n_imgs):
            for j in range(n_rows):
                for k in range(n_cols):
                    X[i][j][k][0], = struct.unpack("B", img_file.read(1))
        X = X / 255.0

    # Read label file
    with gzip.open(label_filename) as label_file:
        label_magic, = struct.unpack(">i", label_file.read(4))
        if label_magic != 2049:
            raise ValueError(f"MSB format parse fail.\
                Expect 2049, but got {label_magic}")

        n_labels, = struct.unpack(">i", label_file.read(4))
        if n_labels != n_imgs:
            raise ValueError(f"Wrong number of labels.\
                Expect {n_imgs}, but got {n_labels}")

        y = np.empty(n_labels, dtype = np.uint8)
        for i in range(n_labels):
            y[i], = struct.unpack("B", label_file.read(1))
        
    return X, y

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            return self.apply_transforms(self.images[index]), self.labels[index]
        return self.images[index], self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION