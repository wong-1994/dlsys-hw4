import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype
        self.model = ndl.nn.Sequential(
            self._convBN(3, 16, 7, 4),
            self._convBN(16, 32, 3, 2),
            self._resBlock(32),
            self._convBN(32, 64, 3, 2),
            self._convBN(64, 128, 3, 2),
            self._resBlock(128),
            ndl.nn.Flatten(),
            ndl.nn.Linear(128, 128, device=device, dtype=dtype),
            ndl.nn.ReLU(),
            ndl.nn.Linear(128, 10, device=device, dtype=dtype),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION

    def _convBN(self, in_channels, out_channels, kernel_size, stride):
        return ndl.nn.Sequential(
            ndl.nn.Conv(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                device=self.device, 
                dtype=self.dtype
            ),
            ndl.nn.BatchNorm2d(dim=out_channels, device=self.device, dtype=self.dtype),
            ndl.nn.ReLU(),
        )

    def _resBlock(self, channels):
        return ndl.nn.Residual(
            ndl.nn.Sequential(
                self._convBN(channels, channels, 3, 1),
                self._convBN(channels, channels, 3, 1),
            )
        )


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
