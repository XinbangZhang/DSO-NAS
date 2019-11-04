import sys
import numpy as np
import mxnet as mx
from mxnet import nd
from math import sqrt
from mxnet.optimizer import Optimizer, SGD, clip
from mxnet.ndarray import NDArray, zeros
from mxnet.ndarray import sgd_update, sgd_mom_update
import config

num_operation = 4
lambda_list = [0]
for i in range(1, config.num_layers):
    lambda_list.append(lambda_list[-1] + (i * num_operation + 1) * num_operation)

latency_gamma_list = {}
for i in range(2, 5):
    for j in range(1, 5):
        latency_gamma_list['lambda_layer_{}_{}'.format(i, j)] = nd.ones((1, 1 + (i - 1) * 4))
latency_gamma_list['lambda_after'] = nd.ones((1, 16))

@mx.optimizer.register
class APGNAG(SGD):
    """APG and NAG.
    """

    def __init__(self, lambda_name=None, gamma=None, lambda_learning_rate=None, lambda_lr_scheduler=None, **kwargs):
        super(APGNAG, self).__init__(**kwargs)
        self.lambda_name = lambda_name
        self.gamma = gamma
        self.opt_lambda_flag = True
        self.opt_weight_flag = True
        #adaptive flops
        self.op_start = []
        self.op_end = []
        for i in range(1, config.num_layers):
            lambda_len = i * num_operation + 1
            for j in range(2):
                self.op_start.append(lambda_list[i-1] + lambda_len * j)
                self.op_end.append(lambda_list[i-1]+lambda_len * (j + 1))
        self.flops_all = config.num_layers * num_operation + config.num_layers * num_operation / 2

        self.latency_gamma_list = {}
        for i in range(2, 5):
            for j in range(1, 5):
                self.latency_gamma_list['lambda_layer_{}_{}'.format(i, j)] = nd.ones((1, 1 + (i - 1) * 4), ctx=mx.gpu())
        self.latency_gamma_list['lambda_after'] = nd.ones((1, 16), ctx=mx.gpu())

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

                    #adp latency
                    if 'layer' in index or 'after' in index:
                        gamma = self.gamma * self.latency_gamma_list[index]
                    else:
                        gamma = self.gamma

                    mom = state
                    mom[:] *= self.momentum
                    z = weight - lr * grad  # equ 10

                    # sss
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
        alpha_ctx = nd.zeros_like(input, ctx=input.context)
        alpha_ctx[:] = alpha
        return mx.ndarray.sign(input) * mx.ndarray.maximum(0.0, mx.ndarray.abs(input) - alpha_ctx)

    @staticmethod
    def non_convex_thresholding(input, alpha, gamma=1.01):
        shape = input.shape
        context = input.context
        input_np = input.asnumpy().flatten()
        input_abs = np.abs(input_np)
        mid_value = np.sign(input_np) * (input_abs - alpha) / (1.0 - 1.0 / gamma)
        first_index = input_abs <= alpha
        last_index = input_abs >= alpha * gamma
        mid_index = np.logical_not(first_index | last_index)

        input_np[first_index] = 0
        input_np[mid_index] = mid_value[mid_index]
        input_np = mx.nd.array(input_np.reshape(shape))
        return input_np.as_in_context(context)
