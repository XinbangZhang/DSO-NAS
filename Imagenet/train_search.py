import logging, os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.momonger import search_plan

if config.split_iter:
    if config.ad_latency:
        from core.solver_split_iter_ad_latency import Solver
        from core.optimizer_split_ad_latency import *
    else:
        from core.solver_split_iter import Solver
        from core.optimizer_split import *
else:
    from core.solver import Solver
    from core.optimizer import *
from data import *
from symbol import *
import os,shutil

pic_size = 224
    
def init_param(train, val, symbol, arg_params = {}, aux_params = {}):
    if type(train) != list:
        data_shape_dict = dict(train.provide_data + train.provide_label)
    else:
        data_shape_dict = dict(train[0].provide_data + train[0].provide_label)
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
    # for k in arg_params.keys():
    #     if k not in symbol.list_arguments():
    #         del arg_params[k]
    #         print k
    # for k in aux_params.keys():
    #     if k not in symbol.list_auxiliary_states():
    #         del aux_params[k]
    #         print k

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

    # set up iterator and symbol
    # iterator
    train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                 batch_size=config.batch_size,
                                                 kv=kv)
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, 3, pic_size, pic_size))]
    label_shapes = [('softmax_label', (config.batch_size,))]
    
    arg_params = None
    aux_params = None
    if config.retrain or config.load_model:
        for model_read in config.model_load_prefix:
            if arg_params is None:
                _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(model_read),
                                                                     config.model_load_epoch)
            else:
                _, arg_params_append, aug_params_append = mx.model.load_checkpoint("model/{}".format(model_read),
                                                                                   config.model_load_epoch)
                for key in arg_params_append.keys():
                    if 'lambda' in key:
                        arg_params[key] = arg_params_append[key]

    if config.share_search_space:
        symbol_search = get_symbol_dsonas_search_share
        symbol_search_prune = get_symbol_dsonas_search_share_prune
    else:
        if config.ad_flops:
            symbol_search = get_symbol_dsonas_search_full_adflops
            symbol_search_prune = get_symbol_dsonas_search_full_adflops_prune
        else:
            symbol_search = get_symbol_dsonas_search_full
            symbol_search_prune = get_symbol_dsonas_search_full

    symbol = symbol_search(num_classes=config.num_classes,
                           image_shape=data_shapes,
                           num_layers=config.num_layers,
                           load_param=arg_params,
                           )

    if config.menmonger:
        data_shape_dict = {'data': (64, 3, pic_size, pic_size), 'softmax_label': (64,)}
        symbol = search_plan(symbol, data=data_shape_dict)
    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if config.lr_step is not None:
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
    else:
        lr_scheduler = None

    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum}
    optimizer = "nag"
    if config.sss and not config.retrain:
        sss_optimizer_params = {'lambda_name': 'lambda',
                                'clip_gradient':2.0,
                                'gamma': config.gamma}
        optimizer_params.update(sss_optimizer_params)
        optimizer = "apgnag"

    eval_metric = ['acc']
    if "imagenet" in config.dataset:
        eval_metric.append(mx.metric.create('top_k_accuracy', top_k=5))

    epoch_end_callback = mx.callback.do_checkpoint("./model/" + config.model_prefix)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)

    if not config.load_model:
        arg_params = {}
        aux_params = {}
    initializer = None
    arg_params, aux_params = init_param(train, val, symbol, arg_params, aux_params)
    #copy config file
    shutil.copy('config/cfgs.py', 'model/' + config.model_prefix + '.cfg')


    latency_table = None
    for i in range(config.begin_epoch, config.num_epoch, 1):
        if i != config.begin_epoch:
            old_module = solver.module
            latency_table = solver.latency_table
            symbol = symbol_search_prune(num_classes=config.num_classes,
                                         image_shape=data_shapes,
                                         num_layers=config.num_layers,
                                         load_param=arg_params,
                                         )
            if config.menmonger:
                data_shape_dict = {'data': (64, 3, pic_size, pic_size), 'softmax_label': (64,)}
                symbol = search_plan(symbol, data=data_shape_dict)
            arg_params, aux_params = init_param(train, val, symbol, arg_params, aux_params)

        else:
            old_module = None

        solver = Solver(symbol=symbol,
                        data_names=data_names,
                        label_names=label_names,
                        data_shapes=data_shapes,
                        label_shapes=label_shapes,
                        logger=logging,
                        context=devs)
        print optimizer
        if latency_table is not None:
            solver.latency_table = latency_table

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
                   begin_epoch=i,
                   num_epoch=i+1,
                   share_module = old_module,
                   kvstore=kv)
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_prefix), i + 1)

if __name__ == '__main__':
    main(config)