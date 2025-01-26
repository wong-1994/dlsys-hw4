from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=(1,), keepdims=True)
        exp = array_api.exp(Z - array_api.broadcast_to(max_Z, Z.shape))
        sum_exp = array_api.sum(exp, axis=(1,), keepdims=True)
        log_sum_exp = array_api.log(sum_exp) + max_Z
        return Z - array_api.broadcast_to(log_sum_exp, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node) # shape of node (shape of Z)
        grad_sum = summation(out_grad, axes=(1,)).reshape((softmax.shape[0], 1))
        return out_grad - softmax * grad_sum.broadcast_to(softmax.shape)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp = array_api.exp(Z - array_api.broadcast_to(max_Z, Z.shape))
        sum_exp = array_api.sum(exp, axis=self.axes)
        return array_api.log(sum_exp) + max_Z.reshape(sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].numpy()
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True) # dim of Z
        exp = array_api.exp(Z - array_api.broadcast_to(max_Z, Z.shape)) # shape of Z
        sum_exp = array_api.sum(exp, axis=self.axes, keepdims=True) # dim of Z
        softmax = Tensor(exp / array_api.broadcast_to(sum_exp, exp.shape)) # shape of Z

        if self.axes is not None:
            shape = list(Z.shape)
            for axis in self.axes:
                shape[axis] = 1
            out_grad = out_grad.reshape(shape)
        return out_grad.broadcast_to(Z.shape) * softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

