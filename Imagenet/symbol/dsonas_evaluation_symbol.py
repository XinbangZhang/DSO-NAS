from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import mxnet as mx
import numpy as np
import config
from symbol.operation import *

lambda_list = [0]
for i in range(1, config.num_layers):
    lambda_list.append(lambda_list[-1] + (i * num_operation + 1) * num_operation)


def pruning_useless_lambda_layer(load_param, stage, unit, num_layers=2):
    lambda_need = {}
    for i in range(1, num_layers + 1):
        lambda_after = load_param['lambda_after_stage%d_unit%d_layer_%d' % (stage, unit, i)].asnumpy().reshape(-1)
        for j in range(1, num_operation + 1):
            lambda_need['stage%d_unit%d_layer%d_op%d' % (stage, unit, i, j)] = lambda_after[j - 1]

    for i in range(num_layers, 1, -1):
        for j in range(1, num_operation + 1):
            lambdas = load_param['lambda_stage%d_unit%d_layer_%d_%d' % (stage, unit, i, j)].asnumpy().reshape(-1)
            if np.sum(lambdas) == 0:
                lambda_need['stage%d_unit%d_layer%d_op%d' % (stage, unit, i, j)] = 0
            if lambda_need['stage%d_unit%d_layer%d_op%d' % (stage, unit, i, j)] > 0:
                for m in range(1, lambdas.shape[0]):
                    lambda_need['stage%d_unit%d_layer%d_op%d' %
                                (stage, unit, int((m - 1) / num_operation) + 1, (m - 1) % num_operation + 1)] += \
                        lambdas[m]

    for i in range(1, num_layers + 1):
        for j in range(1, num_operation + 1):
            if i > 1:
                lambdas = load_param['lambda_stage%d_unit%d_layer_%d_%d' % (stage, unit, i, j)].asnumpy().reshape(-1)
                if np.sum(lambdas) == 0:
                    lambda_need['stage%d_unit%d_layer%d_op%d' % (stage, unit, i, j)] = 0

            if lambda_need['stage%d_unit%d_layer%d_op%d' % (stage, unit, i, j)] == 0:
                load_param['lambda_after_stage%d_unit%d_layer_%d' % (stage, unit, i)][0, j - 1] = 0
                for m in range(i + 1, num_layers + 1):
                    for n in range(1, num_operation + 1):
                        load_param['lambda_stage%d_unit%d_layer_%d_%d' % (stage, unit, m, n)][
                            0, (i - 1) * num_operation + j] = 0
                if i > 1:
                    load_param['lambda_stage%d_unit%d_layer_%d_%d' % (stage, unit, i, j)][:] = 0


def pruning_useless_lambda_evaluation(load_param=None, num_stages=3, units=[], num_layers=2):
    for i in range(1, num_stages + 1):
        for j in range(1, units[i - 1]):
            # ad flops situation
            if 'lambda_stage%d_unit%d' % (i, j) in load_param.keys():
                lambdas = load_param['lambda_stage%d_unit%d' % (i, j)]
                for m in range(1, num_layers + 1):
                    for n in range(1, num_operation + 1):
                        if m > 1:
                            lambda_len = (m - 1) * num_operation + 1
                            lambda_start = lambda_list[m - 2] + (n - 1) * lambda_len
                            load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n)] = \
                                lambdas[:, lambda_start:lambda_start + lambda_len]
                    lambda_start = lambda_list[-1] + (m - 1) * num_operation
                    load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m)] = \
                        lambdas[:, lambda_start:lambda_start + num_operation]

            # share search space situation
            if 'lambda_after' in load_param.keys():
                lambda_after = load_param['lambda_after']
                for m in range(1, num_layers + 1):
                    load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m)] = \
                        lambda_after[:, (m - 1) * num_operation: m * num_operation]
                    if m > 1:
                        for n in range(1, num_operation + 1):
                            load_param['lambda_stage%d_unit%d_layer_%d_%d' % (i, j, m, n)] = \
                                load_param['lambda_layer_%d_%d' % (m, n)]

            # full search space situation
            if 'lambda_after_stage%d_unit%d' % (i, j) in load_param.keys():
                for m in range(num_layers):
                    load_param['lambda_after_stage%d_unit%d_layer_%d' % (i, j, m + 1)] = \
                        load_param['lambda_after_stage%d_unit%d' % (i, j)][:,
                        m * num_operation:(m + 1) * num_operation:]

            pruning_useless_lambda_layer(load_param, i, j, num_layers=num_layers)

    return load_param


def dsonas_layer(data, num_filter, name, layer_num, load_param=None, bn_mom=0.9, workspace=256):
    scale_data = [[] for i in range(num_operation)]
    if layer_num == 1:
        lambdas = load_param["lambda_after_" + name + "_layer_1"].asnumpy().reshape(-1)
        for i in range(2, config.num_layers + 1):
            for j in range(1, num_operation + 1):
                lambdas_high = load_param['lambda_' + name + '_layer_%d_%d' % (i, j)].asnumpy().reshape(-1)
                for m in range(num_operation):
                    lambdas[m] += lambdas_high[m + 1]
        for i in range(num_operation):
            if lambdas[i] > 0:
                scale_data[i].append(data[0])
    else:
        for i in range(num_operation):
            lambdas = load_param["lambda_" + name + '_layer_%d_%d' % (layer_num, i + 1)].asnumpy().reshape(-1)
            for j in range(len(lambdas)):
                if lambdas[j] > 0:
                    scale_data[i].append(data[j])

    if len(scale_data[0]) > 0:
        conv1_data = mx.sym.add_n(*scale_data[0])
        conv1 = Conv_Sep(conv1_data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                         name=name + '_%d_conv3x3' % (layer_num), bn_mom=bn_mom, workspace=workspace)
    else:
        conv1 = None

    if len(scale_data[1]) > 0:
        conv2_data = mx.sym.add_n(*scale_data[1])
        conv2 = Conv_Sep(conv2_data, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                         name=name + '_%d_conv5x5' % (layer_num), bn_mom=bn_mom, workspace=workspace)
    else:
        conv2 = None

    if len(scale_data[2]) > 0:
        pool1_data = mx.sym.add_n(*scale_data[2])
        pool1 = Pool(pool1_data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max',
                     name=name + '_%d_max_pool' % (layer_num), bn_mom=bn_mom)
    else:
        pool1 = None

    if len(scale_data[3]) > 0:
        pool2_data = mx.sym.add_n(*scale_data[3])
        pool2 = Pool(pool2_data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg',
                     name=name + '_%d_avg_pool' % (layer_num), bn_mom=bn_mom)
    else:
        pool2 = None

    return [conv1, conv2, pool1, pool2]


def dsonas_cell(data, num_filter, name, num_layers, id_conv=False, load_param=None, bn_mom=0.9, workspace=256):
    output_temp = []
    input = [data]
    for i in range(num_layers):
        layer = dsonas_layer(input, num_filter, name, layer_num=i + 1, load_param=load_param, bn_mom=bn_mom,
                             workspace=workspace)
        num_outputs = len(layer)
        input += layer
        lambdas_after = load_param["lambda_after_" + name + "_layer_%d" % (i + 1)].asnumpy().reshape(-1)
        lambdas_after = lambdas_after.tolist()
        for j in range(num_outputs):
            if lambdas_after[j] > 0:
                assert layer[j] is not None
                output_temp.append(layer[j])
            else:
                output_temp.append(None)

    output = []
    for i in range(len(output_temp)):
        if output_temp[i] is not None:
            output_temp[i] = mx.sym.Convolution(data=output_temp[i], num_filter=num_filter, kernel=(1, 1),
                                                stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace,
                                                name=name + '_output_conv_%d' % (i + 1))
            output.append(output_temp[i])

    if id_conv:
        short_cut = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name + '_id_conv')
    else:
        short_cut = data

    if len(output) == 0:
        return short_cut
    output = mx.sym.add_n(*output)
    output = mx.sym.BatchNorm(data=output, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
    # menmonger
    if config.menmonger:
        output._set_attr(mirror_stage='True')
    output = mx.sym.Activation(data=output, act_type='relu', name=name + '_relu')

    output_sum = output + short_cut
    return output_sum


def dsonas_reduction_cell(data, num_filter, name, load_param=None, bn_mom=0.9, workspace=256):
    output = []
    lambdas = load_param["lambda_" + name].asnumpy().reshape(-1)
    if lambdas[0] > 0:
        conv1 = Conv(data, num_filter=num_filter, kernel=(1, 1), pad=(0, 0), stride=(2, 2), name=name + '_conv1x1',
                     bn_mom=bn_mom, workspace=workspace)
        output.append(conv1)
    if lambdas[1] > 0:
        conv2 = Conv(data, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=name + '_conv3x3',
                     bn_mom=bn_mom, workspace=workspace)
        output.append(conv2)

    output = mx.sym.add_n(*output)
    return output


def dsonas_evaluation(units, num_stages, filter_list, num_classes, image_shape, num_layers=2, bn_mom=0.9, workspace=256,
                      load_param=None, dtype='float32'):
    num_unit = len(units)
    assert (num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    # data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (batch_size, nchannel, height, width) = image_shape
    if height <= 32:  # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[1], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    else:  # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0] // 2, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Convolution(data=body, num_filter=filter_list[0], kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                  no_bias=True, name="conv1", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu1')

    for i in range(num_stages):
        for j in range(units[i] - 1):
            body = dsonas_cell(body, filter_list[i + 1], name='stage%d_unit%d' % (i + 1, j + 1), num_layers=num_layers,
                               id_conv=(j == 0), load_param=load_param, bn_mom=bn_mom, workspace=workspace)
        if i != num_stages - 1:
            body = dsonas_reduction_cell(body, filter_list[i + 2], load_param=load_param,
                                         name='stage%d_unit%d' % (i + 1, units[i]), bn_mom=bn_mom, workspace=workspace)
            if i == 2 and config.use_aux_loss:
                aux_loss = aux_head(body, num_classes=num_classes, bn_mom=bn_mom, dtype=dtype)

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='final_bn')
    body = mx.sym.Activation(data=body, act_type='relu', name='final_relu')
    pool1 = mx.sym.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    if config.smooth_alpha:
        output = mx.sym.SoftmaxOutput(data=fc1, name='softmax', smooth_alpha=0.1)
    else:
        output = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    if config.use_aux_loss:
        # return mx.sym.Group([output, aux_loss])
        return mx.sym.Group([output, aux_loss])
    else:
        return output
    # return body


def get_symbol_dsonas_evaluation_symbol(num_classes, image_shape, num_layers=2, load_param=None, conv_workspace=256,
                                        dtype='float32', **kwargs):
    """Return dsonas symbol for evaluation model
    ----------
    num_classes : int
        Ouput size of symbol
    depth: int
        depth of network
    image_shape : list
        shape of input image
    num_layers : int
        Number of layers for normal cell
    load_param: dict
        input structure parameter
    conv_workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """
    image_shape = [int(l) for l in image_shape[0][1]]
    num_stages = 4
    filter_init = config.retrain_channel
    filter_list = [filter_init] * 2
    for i in range(num_stages - 1):
        filter_list.append(filter_list[-1] * 2)
    units = [3, 3, 14, 7]
    load_param = pruning_useless_lambda_evaluation(load_param, num_stages=num_stages, units=units,
                                                   num_layers=num_layers)

    return dsonas_evaluation(units=units,
                             num_stages=num_stages,
                             filter_list=filter_list,
                             num_classes=num_classes,
                             image_shape=image_shape,
                             num_layers=num_layers,
                             load_param=load_param,
                             workspace=conv_workspace,
                             dtype=dtype)
