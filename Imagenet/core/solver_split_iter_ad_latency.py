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
from symbol import get_symbol_dsonas_evaluation_symbol
from collections import namedtuple
import cPickle as pickle
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
        self.latency_table = {}
        self.logger.info('lr_step: %s', str(config.lr_step))

    def fit(self, train_data, eval_data=None,
            eval_metric='acc', validate_metric=None,
            work_load_list=None, epoch_end_callback=None,
            batch_end_callback=None, fixed_param_prefix=None,
            initializer=None, arg_params=None,
            aux_params=None, allow_missing=False,
            optimizer=None, optimizer_params=None,
            begin_epoch=0, num_epoch=None,share_module = None,
            kvstore='device'):

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
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params, allow_extra=True)
        train_weight_data = train_data[0]
        train_lambda_data = train_data[1]

        if validate_metric is None:
            validate_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
        if config.dataset == 'cifar10' or 'train1' in config.dataset:
            transform_num = 5
        else:
            transform_num = 2
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
            print epoch
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
                    # arg_params.update(del_arg)
                    # aux_params.update(del_aux)
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
            if epoch >= 0:
                lambda_params = {}
                for key in arg_params.keys():
                    if 'lambda' in key:
                        lambda_params[key] = arg_params[key].copy()
                structure_code = get_lambda_modify_structure_code(lambda_params, num_stages=3, units = [9] * 3, num_ops = 4)
                # print structure_code, structure_code.shape

                latency_list = np.zeros(len(structure_code))
                gamma_list = np.ones(len(structure_code))
                latency_test_count = 0
                for i in range(len(structure_code)):
                    if structure_code[i] == 1:
                        new_stucture = structure_code.copy()
                        new_stucture[i] = 0
                        if structure(new_stucture) in self.latency_table.keys():
                            latency_list[i] = self.latency_table[structure(new_stucture)]
                            # print 'structure exits:', structure(new_stucture)
                        else:
                            latency_test_count = latency_test_count + 1
                            lambda_strucutre = code2lambda(new_stucture)
                            symbol = get_symbol_dsonas_evaluation_symbol(num_classes=1000, num_layers=110,
                                                                       image_shape=[('data', (1, 3, 224, 224))], num_ops=4,
                                                                       load_param=lambda_strucutre)
                            latency = test_speed(symbol)
                            self.latency_table[structure(new_stucture)] = latency
                            latency_list[i] = latency
                print 'latency_test: {}, latency_test_this_epoch: {}'.format(len(self.latency_table.keys()), latency_test_count)
                str_index = np.where(structure_code > 0)
                #edge with small latecy will be punished with large gamma
                latency_all = -latency_list[str_index]
                latency_all -= np.min(latency_all)
                latency_all /= np.max(latency_all) + 1e-9
                latency_all = latency_all * 0.5 + 0.5
                gamma_list[str_index] = latency_all

                point = 0
                for i in range(2, 5):
                    for j in range(1, 5):
                        self.module._optimizer.latency_gamma_list['lambda_layer_{}_{}'.format(i, j)][0, :] = gamma_list[point:point-3+i*4]
                        point = point-3+i*4
                self.module._optimizer.latency_gamma_list['lambda_after'][0, :] = gamma_list[point:]
                print self.module._optimizer.latency_gamma_list

def structure(code):
    num = ''
    for i in range(len(code)):
        if code[i] == 0:
            num += '0'
        else:
            num += '1'
    return num


def get_lambda_modify_structure_code(load_param = None, num_stages = 3, units = [], num_ops = 2):
    lambda_need = {}
    structure_code = np.zeros(0, dtype = int)
    prune_thresold = 0.00
    num_operation = 4

    for i in range(1, num_stages + 1):
        for j in range(1, units[i - 1]):
            if 'lambda_after_stage%d_unit%d_layer_1' % (i, j) not in load_param.keys():
                if 'lambda_after_stage%d_unit%d' % (i, j) in load_param.keys():
                    lambda_after = load_param['lambda_after_stage%d_unit%d' % (i, j)]
                else:
                    lambda_after = load_param['lambda_after']
                for m in range(1, num_ops + 1):
                    load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m)] = \
                        lambda_after[:, (m - 1) * num_operation: m * num_operation]

    for i in range(1, num_stages + 1):
        for j in range(1, units[i - 1]):
            for m in range(1, num_ops + 1):
                if 'lambda_after_stage%d_unit%d_layer_%d' % (i, j, m) in load_param.keys():
                    lambda_after = load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m)].asnumpy().reshape(-1)
                else:
                    lambda_after = load_param['lambda_after_layer_%d' % (m)].asnumpy().reshape(-1)
                for n in range(1, num_operation + 1):
                    # prune small lambda
                    if lambda_after[n - 1] / 1. < prune_thresold:
                        lambda_after[n - 1] = 0
                    lambda_need['stage%d_unit%d_layer%d_op%d' % (i, j, m, n)] = lambda_after[n - 1]

            for m in range(num_ops, 1, -1):
                for mm in range(1, num_operation + 1):
                    if 'lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, mm) in load_param.keys():
                        lambdas = load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, mm)].asnumpy().reshape(-1)
                    else:
                        lambdas = load_param['lambda_layer_%d_%d' % (m, mm)].asnumpy().reshape(-1)
                    if np.sum(lambdas) == 0:
                        lambda_need['stage%d_unit%d_layer%d_op%d' % (i, j, m, mm)] = 0
                    if lambda_need['stage%d_unit%d_layer%d_op%d' % (i, j, m, mm)] > 0:
                        for n in range(1, lambdas.shape[0]):
                            # prune small lambda
                            if lambdas[n] / max(max(lambdas), 1e-9) < prune_thresold:
                                lambdas[n] = 0
                            lambda_need['stage%d_unit%d_layer%d_op%d' %
                                        (i, j, int((n - 1) / num_operation) + 1, (n - 1) % num_operation + 1)] += \
                            lambdas[n]

            for m in range(1, num_ops + 1):
                for n in range(1, num_operation + 1):
                    if m > 1:
                        if 'lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n) in load_param.keys():
                            lambdas = load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n)].asnumpy().reshape(
                                -1)
                        else:
                            lambdas = load_param['lambda_layer_%d_%d' % (m, n)].asnumpy().reshape(-1)
                        if np.sum(lambdas) == 0:
                            lambda_need['stage%d_unit%d_layer%d_op%d' % (i, j, m, n)] = 0

                    if lambda_need['stage%d_unit%d_layer%d_op%d' % (i, j, m, n)] == 0:
                        if 'lambda_after_stage%d_unit%d_layer_%d' % (i, j, m) in load_param.keys():
                            load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m)][0, n - 1] = 0
                        else:
                            load_param['lambda_after'][0, m * num_operation + n - 1] = 0
                        for mm in range(m + 1, num_ops + 1):
                            for nn in range(1, num_operation + 1):
                                if 'lambda_stage%d_unit%d_layer_%d_%d' % (i, j, mm, nn) in load_param.keys():
                                    load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, mm, nn)][
                                        0, (m - 1) * num_operation + n] = 0
                                else:
                                    load_param['lambda_layer_%d_%d' % (mm, nn)][0, (m - 1) * num_operation + n] = 0

                        if m > 1:
                            if 'lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n) in load_param.keys():
                                load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n)][:] = 0
                            else:
                                load_param['lambda_layer_%d_%d' % (m, n)][:] = 0

    if 'lambda_after' in load_param.keys():
        for m in range(1, num_ops + 1):
            load_param['lambda_after'][:, (m - 1) * num_operation : m * num_operation] = \
                load_param['lambda_after_stage%d_unit%d_layer_%d' % (1, 1, m)]

    for i in range(1, 4):
        for j in range(4):
            lambda_np = load_param['lambda_layer_%d_%d' % (i + 1, j + 1)].asnumpy().reshape(-1) > 0
            structure_code = np.concatenate([structure_code, lambda_np], axis=0)
    for i in range(4):
        lambda_after_np = load_param['lambda_after_stage1_unit1_layer_%d' % (i + 1)].asnumpy().reshape(-1) > 0
        structure_code = np.concatenate([structure_code, lambda_after_np], axis=0)

    return structure_code


def code2lambda(code):
    arg_params = {}
    point = 0
    for i in range(1, 4):
        for j in range(4):
            arg_params["lambda_layer_%d_%d" % (i + 1, j + 1)] = mx.nd.zeros((1, 1 + 4 *i))
            arg_params["lambda_layer_%d_%d" % (i + 1, j + 1)][0, :] = code[point:point + 1 + 4 * i]
            point += 1 + 4 * i
    arg_params['lambda_after'] = mx.nd.zeros((1, 4 * 4))
    arg_params['lambda_after'][0, :] = code[point:]
    return arg_params


def test_speed(sym, name=None):
    dev = mx.gpu(config.gpu_list[-1])
    # dev = mx.cpu()
    batch_size = 1
    data_shapes = [('data', (batch_size, 3, 224, 224))]
    module = mx.module.Module(sym, context=dev)
    module.bind(data_shapes=data_shapes, for_training=False, grad_req='null')
    module.init_params()
    # module.save_checkpoint(name, 0)
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    data = [mx.nd.random.normal(loc=0, scale=1, shape=(batch_size, 3, 224, 224))]
    data_batch = Batch(data)
    import time
    all_time = 0
    all_count = 0
    for i in range(100):
        tic = time.time()
        module.forward(data_batch)
        module.get_outputs()[0].wait_to_read()
        # a = module.get_outputs()[0]
        toc = time.time() - tic
        if i >= 10:
            all_count += 1
            all_time += toc
    return all_time / all_count