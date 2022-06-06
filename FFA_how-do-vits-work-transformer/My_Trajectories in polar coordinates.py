import warnings

warnings.filterwarnings("ignore")

import os
import yaml
import copy
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import models
import ops.My_tests as tests
import ops.datasets as datasets
import ops.loss_landscapes as lls
import matplotlib as mpl
import matplotlib.pyplot as plt

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


################### 画图 #####################
def draw(theta, r):
    # 防止乱码
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    plt.polar(theta, r,
              color="chartreuse",
              linewidth=1,
              marker="*",
              mfc="b",
              ms=10)

    plt.savefig("Trajectories in polar coordinates.jpg")
    plt.show()


def caluate(ws_best, ws_epoch):  # 已有每个模型参数模的均值，然后用该均值计算
    """
    :param ws_best: {k1:a, k2:aa}
    :param ws_epoch: [{k1:a1, k2:aa1}, {k1:a2, k2:aa2}, ...]
    :return: theta, r
    """
    theta, r = [], []

    """
    因为有好多参数，每个参数都可以计算一个theta, r，因此我们选用平均的
    """
    # ∆w 保存对应epoch的∆wt： wt − woptim 中的 每个参数的相减值 [{k1:a1-a, k2:aa1-aa}, {k1:a2-a, k2:aa2-aa}, ...]
    der_w = []
    for epoch in ws_epoch:
        der_w_epoch = {}
        for k in ws_best:
            der_w_epoch[k] = epoch[k] - ws_best[k]
        der_w.append(der_w_epoch)

    # ∆winit {k1:a1-a, k2:aa1-aa}
    der_w_init = der_w[0]
    # der_w_init_as ||∆winit|| 分别对每个参数都计算||param||
    der_w_init_as = {}
    for der_w_init_k in der_w_init:
        der_w_init_as[der_w_init_k] = torch.norm(der_w_init[der_w_init_k], p=2)

    # der_w [{k1:a1-a, k2:aa1-aa}, {k1:a2-a, k2:aa2-aa}, ...]
    for epoch in der_w:  # epoch {k1:a1-a, k2:aa1-aa}
        theta_para, r_para = [], []
        for der_wt_k in epoch:  # der_wt_k k1
            # der_wt_k参数对应的||∆wt||
            der_wt_a = torch.norm(epoch[der_wt_k], p=2)
            # der_wt_k参数对应的||∆winit||
            der_w_init_a = der_w_init_as[der_wt_k]

            # 保存der_wt_k参数计算的r_para, theta_para
            r_para.append(der_wt_a / der_w_init_a)
            theta_para.append(math.acos(torch.dot(epoch[der_wt_k], der_w_init[der_wt_k]) / (der_wt_a * der_w_init_a)))

        # 计算该epoch下的所有参数对应r和theta的均值
        r.append(torch.mean(r_para))
        theta.append(torch.mean(theta_para))

    return theta, r


if __name__ == '__main__':
    """
    w_t是模型的好多参数
    ∆wt = w_t − w_optim意味着：仅仅对应参数向量进行相减
    ||∆wt||意味着：所有参数向量去模后再求均值
    ∆wt·∆winit：对应向量相乘 --- ∆wt · ∆winit=|∆wt| |∆winit| cosθ 
    
    论文中没详细说图二如何绘出，因此对于 Wt-Woptm：
        1. 求每个模型参数模的均值，然后用该均值计算, 但对每个模型均值仅为一个值，在计算theta时与公式分子不一样，因此不对
        2. 让模型每个参数先互相计算，最好计算均值
    """

    """
    得到每个epoch路径权重的代码
    """
    path = r'C:\Users\pc\Desktop\how-do-vits-work-transformer\pretrain_weight'
    model_epoch_path = []
    if os.path.isdir(path):
        fileList = os.listdir(path)
        for f in fileList:
            model_epoch_path.append(path + '/' + f)
    print(model_epoch_path)


    ################### 对最好Weight的模型 #####################
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model, path_chk_rest)

    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    # 加载最好的权重
    ws_best = copy.deepcopy(model.state_dict())
    ws_best = {k: v.float() for k, v in ws_best.items()}

    ################### 对每个epoch模型 #####################
    # 记录每个epoch所以向量模的均值
    ws_epoch = []

    for path in model_epoch_path:
        # 加载每个epoch权重
        utils.load_checkpoint(model, path)

        model = model.cuda() if torch.cuda.is_available() else model.cpu()
        # 加载最好的权重
        ws_best = copy.deepcopy(model.state_dict())
        ws_best = {k: v.float() for k, v in ws_best.items()}
        ws_epoch.append(ws_best)

    theta, r = caluate(ws_best, ws_epoch)
    draw(theta, r)
