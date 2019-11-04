import config
import sys
sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import numpy as np

def test_speed(sym, name=None):
    dev = mx.gpu()
    batch_size = 1
    data_shapes = [('data', (batch_size, 3, 224, 224))]
    module = mx.module.Module(sym, context=dev)
    module.bind(data_shapes=data_shapes, for_training=False, grad_req='null')
    module.init_params()
    module.save_checkpoint(name, 0)
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    data = [mx.nd.random.normal(loc=0, scale=1, shape=(batch_size, 3, 224, 224))]
    data_batch = Batch(data)
    import time
    all_time = 0
    all_count = 0
    for i in range(200):
        tic = time.time()
        module.forward(data_batch)
        module.get_outputs()[0].wait_to_read()
        toc = time.time() - tic
        if i >= 10:
            all_count += 1
            all_time += toc
    return all_time / all_count


def get_symbol_flops_param(symbol, data_shape_dict):
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    interals = symbol.get_internals()
    _, out_shapes, _ = interals.infer_shape(**data_shape_dict)
    if out_shapes is None:
        raise ValueError("Input shape is incomplete")
    out_shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    params = 0
    flops = 0
    for index in arg_shape_dict.keys():
        if ('weight' in index) and ('aux' not in index):
            para_shape = arg_shape_dict[index]
            feat_shape = out_shape_dict[index[:-6] + 'output']
            weight_size = 1
            for j in range(len(para_shape)):
                weight_size *= para_shape[j]
            if 'fc' in index:
                weight_flops = weight_size
            else:
                weight_flops = weight_size * feat_shape[-1] * feat_shape[-2]
            params += weight_size
            flops += weight_flops
    return params, flops
