import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import models
import ops.My_tests as tests
import ops.datasets as datasets
import ops.loss_landscapes as lls

root = 'C:/Users/pc/Desktop/how-do-vits-work-transformer'
# config_path = "%s/configs/cifar10_vit.yaml" % root
# config_path = "%s/configs/cifar100_vit.yaml" % root
# config_path = "%s/configs/imagenet_vit.yaml" % root

############### 获取一些参数 ###############
import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))  # 将路径添加到环境变量中
print(dir_name)

import argparse
import Uformer_Info.options as options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='remove the haze')).parse_args()  # 解析参数
print(opt)

import Uformer_Info.utils as utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 在import torch之前
import torch

# 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
# 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from Uformer_Info.losses import CharbonnierLoss

from tqdm import tqdm
from Uformer_Info.warmup_scheduler.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from Uformer_Info.utils.loader import get_training_data, get_validation_data

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)  # Uformer + 16_0701_1
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # dir: path/log/Uformer16_0701_1/

logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)  # dir: path/log/Uformer16_0701_1/results/
utils.mkdir(model_dir)

########## Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model框架 ###########
model = utils.get_arch(opt)  # 创建模型 返回模型对象

dataset_name = 'NH'
model_name = 'Uformer_Informer'

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model) + '\n')

######### Optimizer优化器 ###########
start_epoch = 1
# optimizer默认：adamw     weight_decay默认：0.02     lr默认：0.0002
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model = torch.nn.DataParallel(model)  # 仅针对单服务器多gpu 数据并行
model.cuda()

######### Resume ###########
# 默认：False --- 不使用load_checkpoint
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1  # 加载模型后计算开始的epoch
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('\n------------------------------------------------------------------------------')
    print("==> Resuming Training with start epoch:", start_epoch)
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------\n')
    # CosineAnnealingLR 余弦退火调整学习率  nepoch: 250(training epochs)
    # 第二个参数T_max：2*T_max时间后，学习率经过了一个周期变化后还是原来的。该代码中是说从头到尾递减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - start_epoch + 1, eta_min=1e-6)

######### Scheduler调度器 ###########
# warmup默认：False
if opt.warmup and not opt.resume:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    # scheduler.step()
elif not opt.resume:
    step = 50
    print("Using StepLR,step={}!".format(step))
    # 学习率按照epoch进行衰减：每多少个epoch下降多少
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    # scheduler.step()

######### SSIM/PSNR ###########
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

######### Loss ###########
# 使用的损失函数
from Uformer_Info.My_CR import *

criterion = []
criterion.append(CharbonnierLoss().cuda())
# is_ab：False
criterion.append(ContrastLoss(ablation=opt.is_ab))

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}  # train_patchsize 128
train_dataset = get_training_data(opt.train_dir, img_options_train)  # train_dir: dir of train data
# train_workers: train_dataloader workers(12)
# pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
dataset_train = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                           num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
dataset_test = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                          num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset, "\n")

# with open(config_path) as f:
#     args = yaml.load(f)
#     print(args)

############### Investigate the Loss Landscape ###############
scale = 1e-0
n = 21
gpu = torch.cuda.is_available()

# transform = mixup_function(train_args) MixUp
# dataset_train：经过各种transforms转化后的训练数据
transform = utils.MixUp_AUG().aug

### metrics_grid[tuple(ratio)] = (l1, l2, *metrics) ###
metrics_grid = lls.get_loss_landscape(
    model, 1, dataset_train, transform=transform,
    kws=["pos_embed", "relative_position"],
    x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=gpu,
)

leaderboard_path = os.path.join("./checkpoints", "logs", dataset_name, model_name)
Path(leaderboard_path).mkdir(parents=True, exist_ok=True)

metrics_dir = os.path.join(leaderboard_path,
                           "%s_%s_x%s_losslandscape.csv" % (dataset_name, model_name, int(1 / scale)))

# metrics_grid = {tuple(ratio):(l1, l2, *metrics)}  {[-1. -1.]: (l1, l2, *metrics)}
#                [[*grid, *metrics] for grid, metrics in metrics_grid.items()]
#                其中metrics = nll_value, \
#                              cutoffs, cms, accs, uncs, ious, freqs, \
#                              topk_value, brier_value, \
#                              count_bin, acc_bin, conf_bin, ece_value, ecse_value

metrics_list = [[*grid, *metrics] for grid, metrics in metrics_grid.items()]
tests.my_save_metrics(metrics_dir, metrics_list)
print('*'*20 + '\n' + 'Succeed save' + '*'*20 + '\n')

############### Plot the Loss Landscape ###############
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# load losslandscape raw data of ResNet-50 or ViT-Ti
names = ["x", "y", "l1", "l2", "loss_value"]
# path = "%s/resources/results/cifar100_resnet_dnn_50_losslandscape.csv" % root  # for ResNet-50
# path = "%s/resources/results/cifar100_vit_ti_losslandscape.csv" % root  # for ViT-Ti
data = pd.read_csv(metrics_dir, names=names)
data["loss"] = data["loss_value"]

# prepare data
p = int(math.sqrt(len(data)))
shape = [p, p]
print(shape)
xs = data["x"].to_numpy().reshape(shape)
ys = data["y"].to_numpy().reshape(shape)
zs = data["loss"].to_numpy().reshape(shape)

zs = zs - zs[np.isfinite(zs)].min()
zs[zs > 42] = np.nan

# Normalize(vmin=None, vmax=None) 是用来把数据标准化(归一化)到[0,1]这个期间内,
# vmin是设置最小值, vmax是设置最大值，小于最小值就取最小值，大于最大值就取最大值。
norm = plt.Normalize(zs[np.isfinite(zs)].min(), zs[np.isfinite(zs)].max())  # normalize to [0,1]
colors = cm.plasma(norm(zs))
rcount, ccount, _ = colors.shape

fig = plt.figure(figsize=(4.2, 4), dpi=120)
ax = fig.gca(projection="3d")
ax.view_init(elev=15, azim=15)  # angle

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

surf = ax.plot_surface(
    xs, ys, zs,
    rcount=rcount, ccount=ccount,
    facecolors=colors, shade=False,
)
surf.set_facecolor((0, 0, 0, 0))

# remove white spaces
adjust_lim = 0.8
ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_zlim(10, 32)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

plt.savefig('save.jpg')
# plt.show()
