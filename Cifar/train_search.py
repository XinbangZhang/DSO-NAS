import logging, os
import config
import sys

sys.path.insert(0, config.mxnet_path)

from data import *

from core.scheduler import multi_factor_scheduler
from core.momonger import search_plan
from symbol import *

if config.split_iter:
    if config.ad_flops:
        from core.solver_split_iter import Solver
        from core.optimizer_split_flops import *
    elif config.ad_mac:
        from core.solver_split_iter import Solver
        from core.optimizer_split_mac import *
    else:
        from core.solver_split_iter import Solver
        from core.optimizer_split import *
else:
    from core.solver import Solver
    from core.optimizer import *

pic_size = 32


def init_param(train, val, symbol, arg_params={}, aux_params={}):
    if type(train) != list:
        data_shape_dict = dict(train.provide_data + train.provide_label)
    else:
        data_shape_dict = dict(train[0].provide_data + train[0].provide_label)
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
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
    model_dir = os.path.join("./model", config.model_prefix)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    train, val, num_examples = cifar_iterator(data_dir=config.data_dir,
                                              batch_size=config.batch_size,
                                              kv=kv)
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, 3, 32, 32))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    arg_params = None
    aux_params = None

    if config.retrain or config.load_model:
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)

    if config.share_search_space:
        symbol_search = get_symbol_dsonas_search_share
        symbol_search_prune = get_symbol_dsonas_search_share_prune
    else:
        if config.ad_flops:
            symbol_search = get_symbol_dsonas_search_full_adflops
            symbol_search_prune = get_symbol_dsonas_search_full_adflops_prune
        else:
            symbol_search = get_symbol_dsonas_search_full
            symbol_search_prune = get_symbol_dsonas_search_full_prune

    # _, arg_params, aux_params = mx.model.load_checkpoint("model/dsonas-20191028-151503/dsonas-20191028-151503", 1)

    # for key in arg_params.keys():
    #     if 'lambda_stage1_unit1' in key:
    #         print key
    #
    # symbol = symbol_search_prune(num_classes=config.num_classes,
    #                              depth=config.depth,
    #                              image_shape=data_shapes,
    #                              num_layers=config.num_layers,
    #                              load_param=arg_params,
    #                              )

    symbol = symbol_search(num_classes=config.num_classes,
                           depth=config.depth,
                           image_shape=data_shapes,
                           num_layers=config.num_layers,
                           load_param=arg_params,
                           )

    if config.menmonger:
        data_shape_dict = {'data': (config.batch_size, 3, pic_size, pic_size), 'softmax_label': (config.batch_size,)}
        symbol = search_plan(symbol, data=data_shape_dict)
    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                          factor=config.lr_factor)

    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum}
    optimizer = "nag"

    if (config.model_search and not config.retrain) or config.stage_train:
        sss_optimizer_params = {'lambda_name': 'lambda',
                                'clip_gradient': 2.0,
                                'gamma': config.gamma}
        optimizer_params.update(sss_optimizer_params)
        optimizer = "apgnag"

    eval_metric = ['acc']
    epoch_end_callback = mx.callback.do_checkpoint(os.path.join(model_dir, config.model_prefix))
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)

    if not config.load_model:
        arg_params = {}
        aux_params = {}
    initializer = None
    arg_params, aux_params = init_param(train, val, symbol, arg_params, aux_params)
    for i in range(config.begin_epoch, config.num_epoch, 5):
        if i != config.begin_epoch:
            old_module = solver.module
            symbol = symbol_search_prune(num_classes=config.num_classes,
                                         depth=config.depth,
                                         image_shape=data_shapes,
                                         num_layers=config.num_layers,
                                         load_param=arg_params,
                                         )
            if config.menmonger:
                data_shape_dict = {'data': (64, 3, 32, 32), 'softmax_label': (64,)}
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
                   num_epoch=i + 5,
                   share_module=old_module,
                   kvstore=kv)
        _, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, config.model_prefix), i + 5)


if __name__ == '__main__':
    main(config)
