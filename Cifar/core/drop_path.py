import sys
import mxnet as mx
import config

class DropPath(mx.operator.CustomOp):

    def __init__(self, num_layer, drop_keep_prob, **kwargs):
        self.iter = 0
        self.num_layer = num_layer
        self.layer_ratio = (num_layer + 1.0) / (config.depth + 2.0)
        self.drop_keep_prob = 1.0 - self.layer_ratio * (1 - drop_keep_prob)
        self.num_epoch = config.num_epoch
        # self.num_epoch = 1. / 390
        self.batch_mask = 0
        self.generate_make = True


    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = x.copy()
        if is_train:
            step_ratio = min(1, (self.iter + 1.0) / (390 * self.num_epoch))
            self.drop_path_keep_prob = 1.0 - step_ratio * (1 - self.drop_keep_prob)
            y = mx.nd.divide(y, self.drop_path_keep_prob)
            if self.generate_make:
                random_uni = mx.nd.random.uniform(shape=(x.shape[0], 1, 1, 1), dtype = 'float32') + self.drop_path_keep_prob
                self.batch_mask = mx.nd.floor(random_uni)
                self.generate_make = False
                self.iter += 1
            y = mx.nd.broadcast_mul(y, self.batch_mask)
        self.assign(out_data[0], req[0], y)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        grad = out_grad[0].copy()
        grad = grad / self.drop_path_keep_prob
        grad = mx.nd.broadcast_mul(grad, self.batch_mask)
        self.assign(in_grad[0], req[0], grad)
        self.generate_make = True


@mx.operator.register("DropPath")
class DropPathProp(mx.operator.CustomOpProp):
    def __init__(self, num_layer = 0, drop_keep_prob = 0.5, **kwargs):
        super(DropPathProp, self).__init__(need_top_grad=True)
        self.iter = 0
        self.num_layer = float(num_layer)
        self.drop_keep_prob = float(drop_keep_prob)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DropPath(self.num_layer, self.drop_keep_prob)
