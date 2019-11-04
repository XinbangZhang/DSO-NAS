import os
import sys
import config
sys.path.insert(0, config.mxnet_path)
import mxnet as mx

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

def imagenet_iterator(data_dir, batch_size, kv):
    if config.split_iter:
        train1 = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train1.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            facebook_aug        = True,
            max_random_area     = 1.0,
            min_random_area     = 0.08,
            max_aspect_ratio    = 4.0 / 3.0,
            min_aspect_ratio    = 3.0 / 4.0,
            brightness          = 0.4,
            contrast            = 0.4,
            saturation          = 0.4,
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            scale               = 1,
            inter_method        = 2,
            rand_mirror         = True,
            shuffle             = True,
            shuffle_chunk_size  = 4096,
            preprocess_threads  = 16,
            prefetch_buffer     = 2,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
        train2 = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train2.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            facebook_aug        = True,
            max_random_area     = 1.0,
            min_random_area     = 0.08,
            max_aspect_ratio    = 4.0 / 3.0,
            min_aspect_ratio    = 3.0 / 4.0,
            brightness          = 0.4,
            contrast            = 0.4,
            saturation          = 0.4,
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            scale               = 1,
            inter_method        = 2,
            rand_mirror         = True,
            shuffle             = True,
            shuffle_chunk_size  = 4096,
            preprocess_threads  = 16,
            prefetch_buffer     = 2,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
        train = [train1, train2]
    else:
        train_data_name = "train1.rec" if config.train_baseline else "train.rec"
        train = mx.io.ImageRecordIter(
                path_imgrec         = os.path.join(data_dir, train_data_name),
                label_width         = 1,
                data_name           = 'data',
                label_name          = 'softmax_label',
                data_shape          = (3, 224, 224),
                batch_size          = batch_size,
                pad                 = 0,
                fill_value          = 127,
                facebook_aug        = True,
                max_random_area     = 1.0,
                min_random_area     = 0.08,
                max_aspect_ratio    = 4.0 / 3.0,
                min_aspect_ratio    = 3.0 / 4.0,
                brightness          = 0.4,
                contrast            = 0.4,
                saturation          = 0.4,
                mean_r              = 123.68,
                mean_g              = 116.28,
                mean_b              = 103.53,
                std_r               = 58.395,
                std_g               = 57.12,
                std_b               = 57.375,
                scale               = 1,
                inter_method        = 2,
                rand_mirror         = True,
                shuffle             = True,
                shuffle_chunk_size  = 4096,
                preprocess_threads  = 16,
                prefetch_buffer     = 2,
                num_parts           = kv.num_workers,
                part_index          = kv.rank
        )
    
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            resize              = 256,
            batch_size          = batch_size,
            data_shape          = (3, 224, 224),
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            scale               = 1,
            inter_method        = 2,
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)
    if config.use_aux_loss:
        if config.split_iter:
            for i in range(len(train)):
                train[i] = AuxlossIter(train[i])
        else:
            train = AuxlossIter(train)
    num_examples = 1281167 * 0.8 if config.train_baseline else 1281167
    return train, val, num_examples
