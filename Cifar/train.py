import os

os.environ["MXNET_ENABLE_GPU_P2P"] = '0'
import logging, os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.scheduler import cosine_factor_scheduler

from core.momonger import search_plan
from core.solver import Solver
from core.optimizer import *
from data import *
from symbol import *

pic_size = 32
def init_param(train, val, symbol, arg_params={}, aux_params={}):

    if config.use_aux_loss:
        data_shape_dict = {'data': (config.batch_size, 3, pic_size, pic_size),
                           'softmax_label': (config.batch_size,), 'aux_softmax_label': (config.batch_size,)}
    else:
        data_shape_dict = {'data': (config.batch_size, 3, pic_size, pic_size), 'softmax_label': (config.batch_size,)}
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    out_shape_dict = zip(symbol.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    attrs = symbol.attr_dict()
    init_conv_weight = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
    init_fc_weight = mx.init.Normal(0.01)
    init_one = mx.init.One()
    init_zero = mx.init.Zero()
    for k in symbol.list_arguments():
        if k in data_shape_dict:
            continue
        if k not in arg_params:
            print 'init ', k
            arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
            desc = mx.init.InitDesc(k, attrs.get(k, None))
            if k.endswith('weight'):
                if 'conv' in k:
                    init_conv_weight(desc, arg_params[k])
                elif k.startswith('fc'):
                    init_fc_weight(desc, arg_params[k])
            if k.endswith('gamma'):
                init_one(desc, arg_params[k])
            if k.startswith('lambda'):
                init_one(desc, arg_params[k])
    for k in symbol.list_auxiliary_states():
        if k not in aux_params:
            print 'init', k
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
            desc = mx.init.InitDesc(k, attrs.get(k, None))
            if k.endswith('moving_var'):
                init_one(desc, aux_params[k])

    return arg_params, aux_params


def main(config):
    # log file
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(log_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # model folder
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    train, val, num_examples = cifar_iterator(data_dir=config.data_dir,
                                              batch_size=config.batch_size,
                                              kv=kv)
    data_names = ('data',)
    if config.use_aux_loss:
        label_names = ('softmax_label', 'aux_softmax_label',)
    else:
        label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, 3, pic_size, pic_size))]
    if config.use_aux_loss:
        label_shapes = [('softmax_label', (config.batch_size,)), ('aux_softmax_label', (config.batch_size,))]
    else:
        label_shapes = [('softmax_label', (config.batch_size,))]

    arg_params = None
    aux_params = None

    if config.retrain or config.load_model:
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)

    symbol_used = get_symbol_dsonas_evaluation_symbol if config.retrain else get_symbol_dsonas_baseline
    symbol = symbol_used(num_classes=config.num_classes,
                         depth=config.depth,
                         image_shape=data_shapes,
                         num_layers=config.num_layers,
                         load_param=arg_params,
                         )

    if config.menmonger:
        if config.use_aux_loss:
            data_shape_dict = {'data': (config.batch_size, 3, pic_size, pic_size),
                               'softmax_label': (config.batch_size,), 'aux_softmax_label': (config.batch_size,)}
        else:
            data_shape_dict = {'data': (config.batch_size, 3, pic_size, pic_size),
                               'softmax_label': (config.batch_size,)}
        symbol = search_plan(symbol, data=data_shape_dict)

    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if config.cos_lr_scheduler:
        lr_scheduler = cosine_factor_scheduler(0.0001, config.lr, 10, 2, epoch_size)
    elif config.lr_step is not None:
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
    else:
        lr_scheduler = None

    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum}
    optimizer = "nag"
    # optimizer = 'sgd'
    if config.model_search and not config.retrain_model:
        # if (config.sss) or config.stage_train:
        sss_optimizer_params = {'lambda_name': 'lambda',
                                'clip_gradient': 2.0,
                                'gamma': config.gamma}
        optimizer_params.update(sss_optimizer_params)
        optimizer = "apgnag"

    eval_metric = ['acc']
    epoch_end_callback = mx.callback.do_checkpoint("./model/" + config.model_prefix)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs)

    if not config.load_model:
        arg_params = {}
        aux_params = {}
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    arg_params, aux_params = init_param(train, val, symbol, arg_params, aux_params)
    print optimizer

    solver.fit(train_data=train,
               eval_data=val,
               eval_metric=eval_metric,
               epoch_end_callback=epoch_end_callback,
               batch_end_callback=batch_end_callback,
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.begin_epoch,
               num_epoch=config.num_epoch,
               kvstore=kv)


if __name__ == '__main__':
    main(config)
