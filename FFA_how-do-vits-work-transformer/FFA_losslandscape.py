import warnings
warnings.filterwarnings("ignore")

import os
from FFA_model.option import opt, model_name, log_dir
from FFA_model.data_utils import *
from torchvision.models import vgg16
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device  # 在import torch之前

import yaml
import copy
from pathlib import Path

from torch.utils.data import DataLoader

import models
import ops.My_tests as tests
import ops.datasets as datasets
import ops.loss_landscapes as lls

from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx


import os

import time
from torch.backends import cudnn
from torch import optim
import warnings
from torch import nn

from FFA_model.option import model_name, log_dir
from FFA_model.data_utils import *
from torchvision.models import vgg16

from FFA_model.metrics import psnr, ssim
from FFA_model.models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim

print('log_dir :', log_dir)
print('model_name:', model_name)


dataset_name = 'FFA_NH'
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

######### parser ###########
print(opt)


import torch

# 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
# 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
torch.backends.cudnn.benchmark = True



models_ = {
    'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
    # 'ots_train': OTS_train_loader,
    # 'ots_test': OTS_test_loader
}
start_time = time.time()
T = opt.steps

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


loader_train = loaders_[opt.trainset]
loader_test = loaders_[opt.testset]
model = models_[opt.net]
model = model.to(opt.device)
if 'cuda' in opt.device:
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = []
criterion.append(nn.L1Loss().to(opt.device))
if opt.perloss:
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(opt.device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(opt.device))
optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                       eps=1e-08)
optimizer.zero_grad()

print('os.path.exists(opt.model_dir): ', str(os.path.exists(opt.model_dir)))

if opt.resume and os.path.exists(opt.model_dir):
    print(f'resume from {opt.model_dir}')
    ckp = torch.load(opt.model_dir)
    model.load_state_dict(ckp['model'], strict=False)



######### DataLoader ###########
print('===> Loading datasets')
dataset_train = loaders_[opt.trainset]
dataset_test = loaders_[opt.testset]
print('===> Loading over')



############### Investigate the Loss Landscape ###############
scale = 1e-0
n = 21
gpu = torch.cuda.is_available()

# transform = mixup_function(train_args) MixUp
# dataset_train：经过各种transforms转化后的训练数据
import Uformer_Info.utils as utils
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
# ax.set_zlim(10, 32)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

plt.savefig('save.jpg')
# plt.show()
