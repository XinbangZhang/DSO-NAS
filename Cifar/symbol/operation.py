import mxnet as mx
import config

eps = 2e-5
num_operation = 4
fix_gamma = config.fix_gamma and config.retrain
def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), num_group=1, name='',
         bn_mom=0.9, workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, dilate=dilate, no_bias=True, workspace=workspace, name='%s' % (name))
    bn = mx.sym.BatchNorm(data=conv, name='%s_bn' % (name), fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s_relu' % (name))
    return act

def Conv_Sep(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), name='', bn_mom=0.9,
             workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_filter, stride=stride,
                              pad=pad, dilate=dilate, no_bias=True, workspace=workspace, name='%s' % (name) + '_sep')
    conv = mx.sym.Convolution(data=conv, num_filter=num_filter, kernel=(1, 1), num_group=1, stride=(1, 1),
                              pad=(0, 0), dilate=dilate, no_bias=True, workspace=workspace, name='%s' % (name))
    bn = mx.sym.BatchNorm(data=conv, name='%s_bn' % (name), fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s_relu' % (name))
    return act

def Pool(data, kernel=(1, 1), stride=(1, 1), pad=(1, 1), pool_type='max', name='', bn_mom=0.9):
    pool = mx.sym.Pooling(data, kernel=kernel, stride=stride, pad=pad, pool_type=pool_type)
    bn = mx.sym.BatchNorm(data=pool, name='%s_bn' % (name), fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s_relu' % (name))
    return act

def aux_head(data, num_classes, bn_mom, dtype, grad_scale=0.4):
    data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='aux_bn')
    data = mx.sym.Activation(data=data, act_type='relu', name='aux_relu')
    data = mx.sym.Pooling(data=data, kernel=(5, 5), stride=(3, 3), pool_type='avg', name='aux_pool')
    data = Conv(data, num_filter=128, kernel=(1, 1), stride=(1, 1), name='aux_conv1')
    data = Conv(data, num_filter=768, kernel=(2, 2), stride=(1, 1), name='aux_conv2')
    pool1 = mx.sym.Pooling(data=data, global_pool=True, kernel=(1, 1), pool_type='avg', name='aux_avg_pool')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='aux_fc')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return mx.sym.SoftmaxOutput(data=fc1, grad_scale=grad_scale, name='aux_softmax')
