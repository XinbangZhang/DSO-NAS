import sys
import numpy as np
import mxnet as mx
from math import sqrt
from mxnet.optimizer import Optimizer, SGD, clip
from mxnet.ndarray import NDArray, zeros
from mxnet.ndarray import sgd_update, sgd_mom_update
import config

@mx.optimizer.register
class APGNAG(SGD):
    """APG and NAG.
    """

    def __init__(self, lambda_name=None, gamma=None, lambda_learning_rate=None, lambda_lr_scheduler=None, **kwargs):
        super(APGNAG, self).__init__(**kwargs)
        self.lambda_name = lambda_name
        self.gamma = gamma

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if index.endswith('gamma'):
            wd = 0
        if index.startswith(self.lambda_name):

            if state is not None:
                mom = state
                mom[:] *= self.momentum
                z = weight - lr * grad  # equ 10

                gamma = self.gamma
                z = self.soft_thresholding(z, lr * gamma)
                mom[:] = z - weight + mom  # equ 11
                weight[:] = z + self.momentum * mom  # equ 12

            else:
                assert self.momentum == 0.0
                # TODO add PGD
            # no-negative
            weight[:] = mx.ndarray.maximum(0.0, weight[:])
        else:
            if state is not None:
                mom = state
                mom[:] *= self.momentum
                grad += wd * weight
                mom[:] += grad
                grad[:] += self.momentum * mom
                weight[:] += -lr * grad
            else:
                assert self.momentum == 0.0
                weight[:] += -lr * (grad + wd * weight)

    def update_multi_precision(self, index, weight, grad, state):
        self.update(index, weight, grad, state)
