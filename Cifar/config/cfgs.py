import time

mxnet_path = '/home/zhangxinbang/mxnet/mxnet/python'
gpu_list = [0,1]
dataset = "cifar10"
depth = 110
num_layers = 4
num_operation = 4
model_load_epoch = 100

train_baseline = False
retrain_model = True
model_search = False

share_search_space = False
ad_flops = False
ad_mac = False

load_model = False
split_iter = model_search
cutout = retrain_model
fix_gamma = train_baseline or model_search
menmonger = model_search or train_baseline
cos_lr_scheduler = retrain_model
use_aux_loss = retrain_model

retrain_channel = 8
drop_path_ratio = 0.5
drop_out_keep_ratio = 0.6
gamma = 0.001

# data
data_dir = 'data/' + dataset
batch_size = 64
batch_size *= len(gpu_list)
kv_store = 'device'
# optimizer
lr = 0.1 if model_search else 0.05
wd = 0.0003 if cos_lr_scheduler else 0.0001
momentum = 0.9

model_prefix = "dsonas-{}".format(time.strftime("%Y%m%d-%H%M%S"))
model_load_prefix = ''

lr_step = [120, 180, 220]
num_epoch = 630 if cos_lr_scheduler else 120
lr_factor = 0.1
if retrain_model:
    model_prefix = model_load_prefix + '-retrain'

frequent = 50
begin_epoch = 0
num_classes = 10
