import os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import numpy as np
from numpy import random

class CutoutIter(mx.io.DataIter):
    def __init__(self, data_iter, size, reset_internal=True):
        super(CutoutIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.current_batch = None

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size
        if hasattr(data_iter, 'default_bucket_key'):
            self.default_bucket_key = data_iter.default_bucket_key

    def reset(self):
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        try:
            self.current_batch = self.data_iter.next()
            index = range(self.batch_size)
            cut_x = random.choice(range(32), self.batch_size)
            cut_x_start = np.clip(cut_x - self.size / 2, 0, 31)
            cut_x_end = np.clip(cut_x + self.size / 2, 0, 32)
            cut_y = random.choice(range(32), self.batch_size)
            cut_y_start = np.clip(cut_y - self.size / 2, 0, 31)
            cut_y_end = np.clip(cut_y + self.size / 2, 0, 32)
            for i in range(self.batch_size):
                self.current_batch.data[0][i, :, cut_x_start[i]:cut_x_end[i], cut_y_start[i]:cut_y_end[i]] = 0
            return True
        except StopIteration:
            return False

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

class AuxlossIter(mx.io.DataIter):
    def __init__(self, data_iter, reset_internal=True):
        super(AuxlossIter, self).__init__()
        self.data_iter = data_iter
        self.reset_internal = reset_internal
        self.current_batch = None

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size
        if hasattr(data_iter, 'default_bucket_key'):
            self.default_bucket_key = data_iter.default_bucket_key

    def reset(self):
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        try:
            self.current_batch = self.data_iter.next()
            self.current_batch.label = [self.current_batch.label[0], self.current_batch.label[0]]
            return True
        except StopIteration:
            return False

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

def cifar_iterator(data_dir, batch_size, kv):

    if data_dir.find('cifar10') >= 0:
        mean=[0.5071, 0.4867, 0.4408]
        std=[0.2675, 0.2565, 0.2761]
    elif data_dir.find('cifar100') >=0:
        mean=[0.4914, 0.4824, 0.4467]
        std=[0.2471, 0.2435, 0.2616]

    mean = [x * 255.0 for x in mean]
    std = [x * 255.0 for x in std]

    if config.split_iter:
        train1 = mx.io.ImageRecordIter(
            path_imgrec=os.path.join(data_dir, "train1.rec"),
            label_width=1,
            data_name='data',
            label_name='softmax_label',
            data_shape=(3, 32, 32),
            batch_size=batch_size,
            pad=4,
            fill_value=127,
            rand_crop=True,
            mean_r=mean[0],
            mean_g=mean[1],
            mean_b=mean[2],
            std_r=std[0],
            std_g=std[1],
            std_b=std[2],
            scale=1,
            rand_mirror=True,
            shuffle=True,
            shuffle_chunk_size=4096,
            preprocess_threads=16,
            prefetch_buffer=16,
            num_parts=kv.num_workers,
            part_index=kv.rank)
        train2 = mx.io.ImageRecordIter(
            path_imgrec=os.path.join(data_dir, "train2.rec"),
            label_width=1,
            data_name='data',
            label_name='softmax_label',
            data_shape=(3, 32, 32),
            batch_size=batch_size,
            pad=4,
            fill_value=127,
            rand_crop=True,
            mean_r=mean[0],
            mean_g=mean[1],
            mean_b=mean[2],
            std_r=std[0],
            std_g=std[1],
            std_b=std[2],
            scale=1,
            rand_mirror=True,
            shuffle=True,
            shuffle_chunk_size=4096,
            preprocess_threads=16,
            prefetch_buffer=16,
            num_parts=kv.num_workers,
            part_index=kv.rank)
        train = [train1, train2]
        val = mx.io.ImageRecordIter(
            path_imgrec=os.path.join(data_dir, "train2.rec"),
            label_width=1,
            data_name='data',
            label_name='softmax_label',
            batch_size=batch_size,
            data_shape=(3, 32, 32),
            mean_r=mean[0],
            mean_g=mean[1],
            mean_b=mean[2],
            std_r=std[0],
            std_g=std[1],
            std_b=std[2],
            scale=1,
            rand_crop=False,
            rand_mirror=False,
            num_parts=kv.num_workers,
            part_index=kv.rank)
    else:
        train_data_name = "train1.rec" if config.train_baseline else "train.rec"
        train = mx.io.ImageRecordIter(
                path_imgrec         = os.path.join(data_dir, train_data_name),
                label_width         = 1,
                data_name           = 'data',
                label_name          = 'softmax_label',
                data_shape          = (3, 32, 32),
                batch_size          = batch_size,
                pad                 = 4,
                fill_value          = 127,
                rand_crop           = True,
                mean_r              = mean[0],
                mean_g              = mean[1],
                mean_b              = mean[2],
                std_r               = std[0],
                std_g               = std[1],
                std_b               = std[2],
                scale               = 1,
                rand_mirror         = True,
                shuffle             = True,
                shuffle_chunk_size  = 4096,
                preprocess_threads  = 16,
                prefetch_buffer     = 16,
                num_parts           = kv.num_workers,
                part_index          = kv.rank)

        val = mx.io.ImageRecordIter(
                path_imgrec         = os.path.join(data_dir, "val.rec"),
                label_width         = 1,
                data_name           = 'data',
                label_name          = 'softmax_label',
                batch_size          = batch_size,
                data_shape          = (3, 32, 32),
                mean_r              = mean[0],
                mean_g              = mean[1],
                mean_b              = mean[2],
                std_r               = std[0],
                std_g               = std[1],
                std_b               = std[2],
                scale               = 1,
                rand_crop           = False,
                rand_mirror         = False,
                num_parts           = kv.num_workers,
                part_index          = kv.rank)

    if config.cutout:
        cutout_size = 16
        print 'using cutout with size:', cutout_size
        if config.split_iter:
            for i in range(len(train)):
                train[i] = CutoutIter(train[i], size = cutout_size)
        else:
            train = CutoutIter(train, size = cutout_size)

    if config.use_aux_loss:
        if config.split_iter:
            for i in range(len(train)):
                train[i] = AuxlossIter(train[i])
        else:
            train = AuxlossIter(train)

    num_examples = 50000 * 0.5 if config.train_baseline else 50000
    return train, val, num_examples
