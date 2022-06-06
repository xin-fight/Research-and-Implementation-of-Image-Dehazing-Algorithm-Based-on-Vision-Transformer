import os, sys
import argparse
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/wangzd/uformer/')

from utils.loader import get_validation_data
import utils

from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/home/dell/桌面/TPAMI2022/Dehazing/#dataset/NH_haze/test/',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/long_NH/',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./My_best_model/S_Best_PSNR: 21.1591 | the_SIMM: 0.7765.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='-1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', default='True', help='Save denoised images in result directory')
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
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint_CPU(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

# model_restoration.cuda()

model_restoration.eval()
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1]
        filenames = data_test[2]

        B, C, H, W = rgb_noisy.shape

        L_H = H - H // args.train_ps * args.train_ps
        L_W = W - W // args.train_ps * args.train_ps

        L = max(H, W)
        L = (L // args.train_ps + 1) * args.train_ps
        L = 1664  # 输入是1200 * 1600时 ps = 128
        L_H = L - H
        L_W = L - W

        big_matrix = torch.zeros((B, C, L, L)).type_as(data_test[1])

        big_matrix[:, :, :H, :W] = rgb_noisy[:, :, :H, :W]
        big_matrix[:, :, :H, W:W + L_W] = rgb_noisy[:, :, :, :L_W]
        big_matrix[:, :, H:H + L_H, :] = big_matrix[:, :, :L_H, :]

        rgb_restored = model_restoration(big_matrix)
        rgb_restored = rgb_restored[:, :, :H, :W]
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

        if args.save_images:
            utils.save_img(os.path.join(args.result_dir, filenames[0]), img_as_ubyte(rgb_restored))

psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)
print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
