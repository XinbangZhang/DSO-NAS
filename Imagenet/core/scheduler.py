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

class LinerFactorScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, base_lr, lr_max, warm_epoch, total_epoch, epoch_size):
        super(LinerFactorScheduler, self).__init__()
        self.lr = base_lr
        self.base_lr = base_lr
        self.lr_max = lr_max
        self.warm_epoch = warm_epoch
        self.total_epoch = total_epoch
        self.epoch_size = epoch_size
        self.update_num = None
        self.lr_warm_add = (self.lr_max - self.base_lr) / (self.epoch_size * self.warm_epoch + 1e-9)
        self.lr_sub = self.lr_max / (self.epoch_size * (self.total_epoch - self.warm_epoch) + 1e-9)
        if config.begin_epoch != 0:
            self.lr = self.lr_max - self.lr_sub * (config.begin_epoch - self.warm_epoch) * self.epoch_size
            logging.info("Update[%d]: Change learning rate to %0.5e", 0, self.lr)

    def __call__(self, num_update):
        num_update = num_update + config.begin_epoch * self.epoch_size
        if num_update != self.update_num:
            self.update_num = num_update
            if num_update < self.epoch_size * self.warm_epoch:
                self.lr = self.lr + self.lr_warm_add
            else:
                self.lr = max(0, self.lr - self.lr_sub)
            if num_update % self.epoch_size == 0:
                logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.lr)
        return self.lr

class PolyFactorScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, max_update, base_lr, epoch_size, warm_epoch = 5, lr_max = 0.5, pwr = 2):
        super(PolyFactorScheduler, self).__init__()
        self.lr = base_lr
        self.base_lr = base_lr
        self.lr_max = lr_max
        self.warm_epoch = warm_epoch
        self.epoch_size = epoch_size
        self.update_num = None
        self.lr_warm_add = (self.lr_max - self.base_lr) / (self.epoch_size * self.warm_epoch)
        self.lr_down_num = max_update - self.epoch_size * self.warm_epoch
        self.pwr = pwr

    def __call__(self, num_update):
        if config.begin_epoch != 0:
            num_update += config.begin_epoch * self.epoch_size
        if num_update != self.update_num:
            self.update_num = num_update
            if num_update < self.epoch_size * self.warm_epoch:
                self.lr = self.lr + self.lr_warm_add
            else:
                self.lr = max(0, self.lr_max * pow((1-(num_update - self.epoch_size * self.warm_epoch) * 1.0 / self.lr_down_num), self.pwr))
            if num_update % self.epoch_size == 0:
                logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.lr)
        return self.lr
        
def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def liner_factor_scheduler(base_lr, lr_max, warm_epoch, total_epoch, epoch_size):
    return LinerFactorScheduler(base_lr, lr_max, warm_epoch, total_epoch, epoch_size)

def cosine_factor_scheduler(lr_min, lr_max, lr_t, lr_multi, epoch_size):
    return CosineFactorScheduler(lr_min, lr_max, lr_t, lr_multi, epoch_size)

def poly_factor_scheduler(base_lr, num_epoch, epoch_size, lr_max = 0.5, pwr = 2):
    return PolyFactorScheduler(max_update = num_epoch * epoch_size, base_lr = base_lr, lr_max = lr_max, epoch_size = epoch_size, pwr = pwr)

