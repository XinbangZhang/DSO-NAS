import os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from data import cifar_iterator


def main(config):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./model/' + config.model_load_prefix,
                                                              config.model_load_epoch)

    model = mx.model.FeedForward(symbol, mx.gpu(0), arg_params=arg_params, aux_params=aux_params)
    kv = mx.kvstore.create(config.kv_store)
    _, val, _ = cifar_iterator(data_dir=config.data_dir,
                               batch_size=config.batch_size,
                               kv=kv)
    print model.score(val)


if __name__ == '__main__':
    main(config)
