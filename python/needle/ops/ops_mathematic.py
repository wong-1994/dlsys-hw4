"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * rhs * power(lhs, rhs-1), out_grad * power(lhs, rhs) * log(lhs)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -1 * out_grad * lhs / power_scalar(rhs, 2)
        ### END YOUR SOLUTION

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.ndim < 2:
            return a
        axes = [i for i in range(a.ndim)]
        if self.axes is None:
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            dim1, dim2 = self.axes
            axes[dim1], axes[dim2] = axes[dim2], axes[dim1]
        return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

# ################################################################################## #
# @ Helper Function to find which axes have been changed during broadcast.           #
#                                                                                    #
# Broadcast operation mathes size from last dimension.                               #
# eg: (3, ) cannot be broadcasted to (3, 1, 1), but can be broadcasted to (1, 1, 3). #
# So compare dims from input_node's last dim, looking for dims that don't match      #
# (where those dims' in original array should only be 1, eg: (3, 1) to (2, 3, 3),    #
# though we don't check this implicit conition here).                                #
# ################################################################################## #
def findBroadcastedAxes(original_shape, broadcast_shape):
    num_expend_dims = len(broadcast_shape) - len(original_shape)
    broadcasted_axes = \
        tuple(
            num_expend_dims + i for i, (broadcast_dim, original_dim) in enumerate(
                zip(broadcast_shape[-len(original_shape):], original_shape)    
            ) if broadcast_dim != original_dim
        )
    return tuple(range(num_expend_dims)) + broadcasted_axes

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # ###################################################################### #
        # Summation will erase both expend_dims and simply broadcasted_axes.     #
        # However, those simply broadcasted_axes should be kept to match         #
        # input node's size.                                                     #
        # So 'Reshape' is needed here.                                           #
        # ###################################################################### #
        broadcasted_axes = findBroadcastedAxes(node.inputs[0].shape, self.shape)
        return reshape(summation(out_grad, broadcasted_axes), node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if (axes is not None) and (not isinstance(axes, tuple)):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        out = a
        for axis in self.axes:
            out = out.sum(axis=axis, keepdims=True)
        new_shape = [dim for i, dim in enumerate(out.shape) if i not in self.axes]
        return out.reshape(new_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Restore the axes being erased by summation
        extend_shape = list(node.shape)
        if self.axes is not None:
            for axis in self.axes:
                extend_shape.insert(axis, 1)
        return broadcast_to(reshape(out_grad, extend_shape), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        grad_lhs = matmul(out_grad, transpose(rhs, (-1, -2)))
        broadcasted_axes_lhs = findBroadcastedAxes(lhs.shape, grad_lhs.shape)
        grad_lhs = reshape(summation(grad_lhs, axes = broadcasted_axes_lhs), lhs.shape)

        grad_rhs = matmul(transpose(lhs, (-1, -2)), out_grad)
        broadcasted_axes_rhs = findBroadcastedAxes(rhs.shape, grad_rhs.shape)
        grad_rhs = reshape(summation(grad_rhs, axes = broadcasted_axes_rhs), rhs.shape)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        gate = Tensor(node.inputs[0].cached_data > 0, device=out_grad.device)
        return out_grad * gate
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-tanh(node.inputs[0]) ** 2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        if len(args) == 0:
            raise ValueError()
        shape = list(args[0].shape)
        shape.insert(self.axis, len(args))
        slice_shape = list(args[0].shape)
        slice_shape.insert(self.axis, 1)
        out = NDArray.make(shape, device=args[0].device)
        for idx in range(len(args)):
            sl = [slice(None)] * len(shape)
            sl[self.axis] = slice(idx, idx+1)
            out[tuple(sl)] = (args[idx].reshape(slice_shape))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out = []
        slice_shape = list(A.shape)
        slice_shape.pop(self.axis)
        for idx in range(A.shape[self.axis]):
            sl = [slice(None)] * len(A.shape)
            sl[self.axis] = slice(idx, idx+1)
            out.append(A[tuple(sl)].reshape(slice_shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(axes=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not isinstance(self.axes, tuple):
            self.axes = (self.axes,)
        shape = list(a.shape)
        sl = [slice(None)] * a.ndim 
        for axis in self.axes:
            shape[axis] *= (self.dilation + 1)
            sl[axis] = slice(None, None, self.dilation + 1)
        out = NDArray.make(shape, device=a.device)
        out.fill(0)
        out[tuple(sl)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not isinstance(self.axes, tuple):
            self.axes = (self.axes,)
        shape = list(a.shape)
        sl = [slice(None)] * a.ndim 
        for axis in self.axes:
            shape[axis] //= (self.dilation + 1)
            sl[axis] = slice(None, None, self.dilation + 1)
        out = NDArray.make(shape, device=a.device)
        out = a[tuple(sl)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        assert A.ndim == 4
        assert B.ndim == 4
        assert A.shape[3] == B.shape[2]

        # for padding
        A = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )

        N, H, W, C_in = A.shape
        K_h, K_w, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K_h * K_w * C_in

        im2col_H = H - K_h + 1
        im2col_W = W - K_w + 1
        im2col_shape = (N, im2col_H, im2col_W, K_h, K_w, C_in)
        im2col_strides = (Ns, Hs, Ws, Hs, Ws, Cs)
        
        im2col = NDArray.make(
            shape=im2col_shape, strides=im2col_strides, device=A.device,
            handle=A._handle
        ).reshape((N * im2col_H * im2col_W, inner_dim))

        return (im2col @ B.reshape((inner_dim, C_out))).reshape((N, im2col_H, im2col_W, C_out))[:, ::self.stride, ::self.stride, :]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, kernel = node.inputs
        N, H, W, C_in = X.shape
        K, _, _, C_out = kernel.shape

        out_grad = dilate(out_grad, (1, 2), self.stride-1)

        kernel = flip(kernel, axes=(0, 1))
        kernel = transpose(kernel)
        X_grad = conv(out_grad, kernel, padding=(K-1-self.padding))

        out_grad_t = transpose(out_grad, (0, 1))
        out_grad_t = transpose(out_grad_t, (1, 2))
        X = transpose(X, (0, 3))
        kernel_grad_stride = 1
        kernel_grad = conv(X, out_grad_t, kernel_grad_stride, self.padding)
        kernel_grad = transpose(kernel_grad, (0, 1))
        kernel_grad = transpose(kernel_grad, (1, 2))

        return X_grad, kernel_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


