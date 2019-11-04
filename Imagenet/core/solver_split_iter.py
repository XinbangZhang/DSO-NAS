import time
import logging
import sys

import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import numpy as np
from mxnet.module import Module
from mxnet import metric
from mxnet.model import BatchEndParam
from collections import namedtuple

def _as_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


class Solver(object):
    def __init__(self, symbol, data_names, label_names,
                 data_shapes, label_shapes, logger=logging,
                 context=mx.cpu(), work_load_list=None, fixed_param_names=None):
        self.symbol = symbol
        self.data_names = data_names
        self.label_names = label_names
        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.context = context
        self.work_load_list = work_load_list
        self.fixed_param_names = fixed_param_names

        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
        self.logger = logger
        self.module = Module(symbol=self.symbol, data_names=self.data_names,
                             label_names=self.label_names, logger=self.logger,
                             context=self.context, work_load_list=self.work_load_list,
                             fixed_param_names=self.fixed_param_names)

        self.logger.info('lr_step: %s', str(config.lr_step))

    def fit(self, train_data, eval_data=None,
            eval_metric='acc', validate_metric=None,
            work_load_list=None, epoch_end_callback=None,
            batch_end_callback=None, fixed_param_prefix=None,
            initializer=None, arg_params=None,
            aux_params=None, allow_missing=False,
            optimizer=None, optimizer_params=None,
            begin_epoch=0, num_epoch=None,share_module = None, kvstore='device'):

        self.module.bind(data_shapes=self.data_shapes, label_shapes=self.label_shapes, for_training=True,
                         shared_module = share_module)
        if share_module is None:

            self.module.init_params(initializer=initializer,
                                    arg_params=arg_params,
                                    aux_params=aux_params,
                                    allow_missing=allow_missing)


            self.module.init_optimizer(kvstore=kvstore,
                                       optimizer=optimizer,
                                       optimizer_params=optimizer_params)

        train_weight_data = train_data[0]
        train_lambda_data = train_data[1]

        if validate_metric is None:
            validate_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
        transform_num = 5
        # training loop
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            end_of_batch = False
            data_iter_lambda = iter(train_lambda_data)
            data_iter_weight = iter(train_weight_data)
            self.module._optimizer.opt_lambda_flag = False
            self.module._optimizer.opt_weight_flag = True
            #first optimzer weight
            next_data_batch = next(data_iter_weight)

            while not end_of_batch:
                data_batch = next_data_batch
                self.module.forward(data_batch, is_train=True)
                self.module.backward()
                self.module.update()
                try:
                    if (nbatch + 2) % transform_num == 0:
                        next_data_batch = next(data_iter_lambda)
                        self.module._optimizer.opt_lambda_flag = True
                        self.module._optimizer.opt_weight_flag = False
                    else:
                        next_data_batch = next(data_iter_weight)
                        self.module._optimizer.opt_lambda_flag = False
                        self.module._optimizer.opt_weight_flag = True

                except StopIteration:
                    end_of_batch = True
                self.module.update_metric(eval_metric, data_batch.label)

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1

            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            arg_params, aux_params = self.module.get_params()

            dele_num = 0
            keep_num = 0
            for key in arg_params:
                if key.startswith('lambda'):
                    lambdas = arg_params[key].asnumpy().reshape(-1)
                    index = np.where(lambdas > 0)
                    if index[0].shape[0] != 0:
                        none_zero_min = np.min(lambdas[index])
                    else:
                        none_zero_min = 0
                    dele_num += np.sum(lambdas == 0)
                    keep_num += np.sum(lambdas != 0)
            self.logger.info('delet_num:{}/{}'.format(dele_num, dele_num + keep_num))
            self.module.set_params(arg_params, aux_params, allow_extra=True)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)
            if eval_data:
                res = self.module.score(eval_data, validate_metric,
                                        score_end_callback=None,
                                        batch_end_callback=None,
                                        reset=True,
                                        epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            train_weight_data.reset()
            train_lambda_data.reset()
