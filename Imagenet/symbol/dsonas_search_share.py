import mxnet as mx
import numpy as np
import config
from symbol.operation import *


def dsonas_layer(data, num_filter, name, layer_num, block_lambda=None, bn_mom=0.9, workspace=256):
    input_num = len(data)
    scale_data = [[] for i in range(num_operation)]
    for i in range(num_operation):
        if layer_num > 1:
            lambdas = block_lambda[i]
            lambdas_split = mx.sym.split(lambdas, axis=1, num_outputs=input_num)
            for j in range(input_num):
                scale_data[i].append(mx.sym.broadcast_mul(data[j], lambdas_split[j]))
        else:
            scale_data[i] = data

    conv1_data = mx.sym.add_n(*scale_data[0])
    conv1 = Conv_Sep(conv1_data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                     name=name + '_%d_conv3x3' % (layer_num), bn_mom=bn_mom, workspace=workspace)

    conv2_data = mx.sym.add_n(*scale_data[1])
    conv2 = Conv_Sep(conv2_data, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                     name=name + '_%d_conv5x5' % (layer_num), bn_mom=bn_mom, workspace=workspace)

    pool1_data = mx.sym.add_n(*scale_data[2])
    pool1 = Pool(pool1_data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max',
                 name=name + '_%d_max_pool' % (layer_num), bn_mom=bn_mom)

    pool2_data = mx.sym.add_n(*scale_data[3])
    pool2 = Pool(pool2_data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg',
                 name=name + '_%d_avg_pool' % (layer_num), bn_mom=bn_mom)

    return [conv1, conv2, pool1, pool2]


def dsonas_cell(data, num_filter, name, num_layers, block_lambdas=None, block_lambda_after=None,
                id_conv=False, bn_mom=0.9, workspace=256):
    '''
    data_output = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                  name=name + '_output_bn')
    data_output = mx.sym.Activation(data=data_output, act_type='relu', name=name + '_output_relu')
    '''
    # data_input = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom,
    #                               name=name + '_input_bn')
    # data_input = mx.sym.Activation(data=data_input, act_type='relu', name=name + '_input_relu')

    output = []
    input = [data]
    for i in range(num_layers):
        layer = dsonas_layer(input, num_filter, name, layer_num=i + 1, block_lambda=block_lambdas[i],
                             bn_mom=bn_mom, workspace=workspace)
        num_outputs = len(layer)
        input += layer
        for j in range(num_outputs):
            output.append(layer[j])
    lambdas_after_split = mx.sym.split(block_lambda_after[0], axis=1, num_outputs=len(output))
    for i in range(len(output)):
        output[i] = mx.sym.broadcast_mul(output[i], lambdas_after_split[i])

    for i in range(len(output)):
        output[i] = mx.sym.Convolution(data=output[i], num_filter=num_filter, kernel=(1, 1), stride=(1, 1),
                                       pad=(0, 0), no_bias=True, workspace=workspace,
                                       name=name + '_output_conv_%d' % (i + 1))
    output = mx.sym.add_n(*output)
    output = mx.sym.BatchNorm(data=output, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
    if config.menmonger:
        output._set_attr(mirror_stage='True')
    output = mx.sym.Activation(data=output, act_type='relu', name=name + '_relu')

    if id_conv:
        short_cut = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name + '_id_conv',
                         bn_mom=bn_mom, workspace=workspace)
    else:
        short_cut = data

    return output + short_cut


def dsonas_reduction_cell(data, num_filter, name, bn_mom=0.9, workspace=256):
    lambdas = mx.sym.Variable(shape=(1, 2), dtype='float32', lr_mult=1., wd_mult=0, init=mx.init.One(),
                              name="lambda_" + name)
    lambda_split = mx.sym.split(lambdas, axis=1, num_outputs=2)
    conv1 = Conv(data, num_filter=num_filter, kernel=(1, 1), pad=(0, 0), stride=(2, 2), name=name + '_conv1x1',
                 bn_mom=bn_mom, workspace=workspace)
    conv1 = mx.sym.broadcast_mul(conv1, lambda_split[0])

    conv2 = Conv(data, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=name + '_conv3x3',
                 bn_mom=bn_mom, workspace=workspace)
    conv2 = mx.sym.broadcast_mul(conv2, lambda_split[1])
    output_list = [conv1, conv2]

    output = mx.sym.add_n(*output_list)
    # output = conv1
    return output


def dsonas_share(units, num_stages, filter_list, num_classes, image_shape, num_layers=2, bn_mom=0.9, workspace=256,
                 dtype='float32'):
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

    # lambda
    block_lambdas = [[]]
    block_lambdas_after = []
    for i in range(1, num_layers):
        lambda_layer = []
        for j in range(num_operation):
            lambda_layer.append(
                mx.sym.Variable(shape=(1, 1 + i * num_operation), dtype='float32', lr_mult=1.0, wd_mult=0,
                                init=mx.init.One(), name="lambda_layer_%d_%d" % (i + 1, j + 1)))
        block_lambdas.append(lambda_layer)

    lambda_layer_after = mx.sym.Variable(shape=(1, num_layers * num_operation), dtype='float32', lr_mult=1.0, wd_mult=0,
                                         init=mx.init.One(), name="lambda_after")
    block_lambdas_after.append(lambda_layer_after)

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
                               block_lambdas=block_lambdas, block_lambda_after=block_lambdas_after, id_conv=(j == 0),
                               bn_mom=bn_mom, workspace=workspace)
        if i != num_stages - 1:
            body = dsonas_reduction_cell(body, filter_list[i + 2], name='stage%d_unit%d' % (i + 1, units[i]))

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='final_bn')
    body = mx.sym.Activation(data=body, act_type='relu', name='final_relu')
    pool1 = mx.sym.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    # return body


def get_symbol_dsonas_search_share(num_classes, image_shape, num_layers=2, conv_workspace=256, dtype='float32',
                                   **kwargs):
    """Return dsonas symbol for share search space
    ----------
    num_classes : int
        Ouput size of symbol
    depth: int
        depth of network
    image_shape : list
        shape of input image
    num_layers : int
        Number of layers for normal cell
    conv_workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """
    image_shape = [int(l) for l in image_shape[0][1]]
    num_stages = 4
    filter_init = 28
    filter_list = [filter_init] * 2
    for i in range(num_stages - 1):
        filter_list.append(filter_list[-1] * 2)
    units = [3, 3, 14, 7]

    return dsonas_share(units=units,
                        num_stages=num_stages,
                        filter_list=filter_list,
                        num_classes=num_classes,
                        image_shape=image_shape,
                        num_layers=num_layers,
                        workspace=conv_workspace,
                        dtype=dtype)
