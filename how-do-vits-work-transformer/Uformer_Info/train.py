import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))  # 将路径添加到环境变量中
print(dir_name)

import argparse
import options

# 重新训练
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0' --train_ps 128 --train_dir /root/Datasets/NH-HAZE/train_patches --env 32_0705_1 --val_dir /root/Datasets/NH-HAZE/test_patches --embed_dim 32 --warmup

# 继续训练
# python3 ./train.py --arch Uformer --resume --batch_size 32 --gpu '0' --train_ps 128 --train_dir /root/Datasets/NH-HAZE/train_patches --env 32_resume --val_dir /root/Datasets/NH-HAZE/test_patches --embed_dim 32 --warmup

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()  # 解析参数
print(opt)

import utils

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

from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

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

######### Model ###########
model_restoration = utils.get_arch(opt)  # 创建模型 返回模型对象

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer优化器 ###########
start_epoch = 1
# optimizer默认：adamw     weight_decay默认：0.02     lr默认：0.0002
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)  # 仅针对单服务器多gpu 数据并行
model_restoration.cuda()

######### Resume ###########
# 默认：False --- 不使用load_checkpoint
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1  # 加载模型后计算开始的epoch, 开始的epoch为之前的+1 （但感觉可以直接为之前的epoch）
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

######### Loss ###########
# 使用的损失函数
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}  # train_patchsize 128
train_dataset = get_training_data(opt.train_dir, img_options_train)  # train_dir: dir of train data
# train_workers: train_dataloader workers(12)
# pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

######### validation ###########
# 训练未启动，查看此时val数据的PSNR
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate((val_loader), 0):
        # val_loader的__getitem__返回的是：clean, noisy, clean_filename, noisy_filename
        target = data_val[0].cuda()  # [B, C, W, H]
        input_ = data_val[1].cuda()
        filenames = data_val[2]
        psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
    psnr_val_rgb = sum(psnr_val_rgb) / len_valset
    print('\nInput & GT (PSNR) -->%.4f dB\n' % (psnr_val_rgb))

######### train ###########
print('\n===> Start Epoch {} End Epoch {}\n'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4  # eval四次
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(train_loader, 0):
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()  # [B, C, W, H]
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)

        # autocast上下文应该只包含网络的前向过程（包括loss的计算）
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            # torch.clamp 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
            restored = torch.clamp(restored, 0, 1)
            loss = criterion(restored, target)
        """
        loss_scaler 继承 NativeScaler 这个类，它的作用本质上是 loss.backward(create_graph=create_graph) 和 optimizer.step()。
        在 __call__ () 函数的内部实现了 loss.backward(create_graph=create_graph) 功能和 optimizer.step() 功能。
        
        loss.backward()
        optimizer.step()
        等价于下面的代码：
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
        """
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())  # 对优化器进行更新
        epoch_loss += loss.item()

        #### Evaluation评估 ####
        if (i + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()  # [B, C, W, H]
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    # torch.clamp 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),  # state_dict变量存放训练过程中需要学习的权重和偏执系数
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD: %d, best_it_SIDD: %d, Best_PSNR_SIDD: %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD: %d, best_it_SIDD: %d, Best_PSNR_SIDD: %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')

                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()  # 每个epoch后调用

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_last_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_last_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    # checkpoint默认50
    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())
