import sys
import numpy as np
import mxnet as mx
from mxnet import nd
from math import sqrt
from mxnet.optimizer import Optimizer, SGD, clip
from mxnet.ndarray import NDArray, zeros
from mxnet.ndarray import sgd_update, sgd_mom_update
import config

lambda_list = [0]
for i in range(1, config.num_layers):
    lambda_list.append(lambda_list[-1] + (i * 4 + 1) * 4)

@mx.optimizer.register
class APGNAG(SGD):
    """APG and NAG.
    """
    def __init__(self, lambda_name=None, gamma=None, **kwargs):
        super(APGNAG, self).__init__(**kwargs)
        self.lambda_name = lambda_name
        self.gamma = gamma
        self.opt_lambda_flag = True
        self.opt_weight_flag = True
        #adaptive flops
        num_operation = 4
        self.op_start = []
        self.op_end = []
        for i in range(1, config.num_layers):
            lambda_len = i * num_operation + 1
            for j in range(2):
                self.op_start.append(lambda_list[i-1] + lambda_len * j)
                self.op_end.append(lambda_list[i-1]+lambda_len * (j + 1))
        self.flops_all = config.num_layers * num_operation + config.num_layers * num_operation / 2


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
            if self.opt_lambda_flag:
                # APG
                if state is not None:
                    if weight.shape[1] <= 2:
                        gamma = self.gamma
                    else:
                        gamma_count = 2
                        for i in range(len(self.op_start)):
                            gamma_count += mx.nd.sum(weight[:, self.op_start[i]:self.op_end[i]]) > 0
                        gamma_count += mx.nd.sum(weight[:, lambda_list[-1]:] > 0)
                        gamma = self.gamma * (gamma_count + 1) / (self.flops_all + 1)

                    mom = state
                    mom[:] *= self.momentum
                    z = weight - lr * grad  # equ 10
                    z = self.soft_thresholding(z, lr * gamma)
                    mom[:] = z - weight + mom  # equ 11
                    weight[:] = z + self.momentum * mom  # equ 12
                    weight[:] = mx.ndarray.maximum(0.0, weight[:])

                else:
                    assert self.momentum == 0.0
                # TODO add PGD
            # no-negative
            weight[:] = mx.ndarray.maximum(0.0, weight[:])
        elif self.opt_weight_flag:
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

    @staticmethod
    def soft_thresholding(input, alpha):
        return mx.ndarray.sign(input) * mx.ndarray.maximum(0.0, mx.ndarray.abs(input) - alpha)



