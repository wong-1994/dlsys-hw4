"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                self.in_channels, 
                self.out_channels, 
                shape=(
                    self.kernel_size, 
                    self.kernel_size, 
                    self.in_channels, 
                    self.out_channels
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.bias = Parameter(
            init.rand(
                self.out_channels,
                low=(self.in_channels*(self.kernel_size**2))**(-0.5),
                high=-(self.in_channels*(self.kernel_size**2))**(-0.5),
                device=device,
                dtype=dtype
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # nchw->nhcw->nhwc
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, self.stride, self.kernel_size//2)
        bias = self.bias.broadcast_to(out.shape)
        out = out + bias
        # nhwc->nhcw->nchw
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION