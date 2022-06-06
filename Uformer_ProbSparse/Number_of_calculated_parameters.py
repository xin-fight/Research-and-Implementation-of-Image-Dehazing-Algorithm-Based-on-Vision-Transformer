from torchstat import stat
import torchvision.models as models
import argparse
import math
import os

import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/root/Uformer/')

from utils.loader import get_validation_data
import utils

from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
# parser.add_argument('--input_dir', default='/root/Datasets/NH-HAZE/test_patches/',
#                     type=str, help='Directory of validation images')
parser.add_argument('--input_dir', default='/root/Datasets/NH-HAZE/test/',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/root/results/NH-HAZE/',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/Uformer32/models/uformer32_denoising_sidd.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='-1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
opt = parser.parse_args()

# python3 ./My_train.py --arch Uformer --batch_size 32 --env My_32_0705_1 --gpu '0' --train_ps 128 --train_dir /root/Datasets/NH-HAZE/train_patches --val_dir /root/Datasets/NH-HAZE/test_patches --embed_dim 32 --warmup

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 在import torch之前

import torch

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    from My_model_1 import Uformer

    model = Uformer(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                    token_projection=opt.token_projection, token_mlp=opt.token_mlp)


    # model = torch.nn.DataParallel(model)  # 仅针对单服务器多gpu 数据并行
    print(get_parameter_number(model))

    input_ = (3, opt.train_ps, opt.train_ps)

    # from torchsummary import summary
    # print(summary(model, input_size=input_))

    # print(stat(model.to(torch.device('cuda:0')), input_))
    print(stat(model, input_))

    print('*' * 10)
    print()
    print()
    print('*' * 10)

    """My_model内容和model一样，但是为了便于torchstat的输入，所以改进了一些维度"""
    from My_model import UNet, Uformer

    model = Uformer(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                    token_projection=opt.token_projection, token_mlp=opt.token_mlp)

    print(get_parameter_number(model))

    print(stat(model, input_))

    print('*' * 10)
    print()
    print()
    print('*' * 10)
