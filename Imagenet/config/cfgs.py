import time
mxnet_path = '/root/mxnet/mxnet-1.2.1/python'
gpu_list = [0,1,2,3,4,5,6,7]
dataset = "imagenet_search"
num_layers = 4
model_load_epoch = 0

train_baseline = False
retrain_model = False
model_search = True

share_search_space = True
ad_latency = False
load_model = False

split_iter = model_search
fix_gamma = train_baseline or model_search
menmonger = model_search or train_baseline
step_lr_scheduler = False
liner_lr_scheduler = retrain_model
cos_lr_scheduler = False
poly_lr_scheduler = False
use_aux_loss = retrain_model

retrain_bn_wd = 0
retrain_channel = 36

smooth_alpha = retrain_model
gamma = 0.01
data_dir = 'data/' + dataset

batch_size = 128 if model_search or train_baseline else 128
batch_size *= len(gpu_list)
kv_store = 'device'
# optimizer
lr = 0.1 #if model_search or train_baseline else 0.1
lr_max = 0.5
wd = 4e-5
momentum = 0.9

model_prefix = "dsonas-{}".format(time.strftime("%Y%m%d-%H%M%S"))
model_load_prefix = []

lr_step = [30, 60, 90]
num_epoch = 120
if liner_lr_scheduler:
    num_epoch = 240
lr_factor = 0.1

frequent = 50
begin_epoch = 0
num_classes = 1000
