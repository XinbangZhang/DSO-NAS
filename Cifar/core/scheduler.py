import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import math
import logging

class CosineFactorScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, lr_min, lr_max, lr_t, lr_multi, epoch_size):
        super(CosineFactorScheduler, self).__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_t = lr_t
        self.lr_multi = lr_multi
        self.epoch_size = epoch_size
        self.base_lr = lr_max
        self.last_epoch = 0
        self.update_num = None

    def __call__(self, num_update):
        if num_update % self.epoch_size == 0:
            #update
            if num_update != self.update_num:
                self.update_num = num_update
                epoch = num_update // self.epoch_size
                cur_epoch = epoch - self.last_epoch
                if cur_epoch >= self.lr_t:
                    self.last_epoch = epoch
                    self.lr_t *= self.lr_multi
                    cur_epoch = epoch - self.last_epoch
                    logging.info("Update[%d]: Change lr T to %d", num_update, self.lr_t)
                rate = float(cur_epoch) / self.lr_t * 3.1415926
                self.base_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(rate))
                logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.base_lr)
        return self.base_lr

def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def cosine_factor_scheduler(lr_min, lr_max, lr_t, lr_multi, epoch_size):
    return CosineFactorScheduler(lr_min, lr_max, lr_t, lr_multi, epoch_size)

