import cv2
import numpy as np
import torch
import pickle
from math import exp

import torch.nn.functional as F
from torch.autograd import Variable


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


def load_img(filepath):
    # cv2.imread()接口读图像，读进来直接是BGR 格式数据格式在 0~255  通道格式为(W,H,C)
    # cv2.cvtColor(p1,p2) 是颜色空间转换函数 BGR2RGB
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


########## PSNR ##########
def myPSNR(tar_img, prd_img):
    # torch.clamp 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
    # 变到[0, 1]后，此时公式中的MAXI为1
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    """
    对于彩色图像来说，一般由三通道组成，我们以RGB图像为例:分别计算 RGB 各个通道上的 PSNR\SSIM均值，然后取平均值（除以3）。
    """
    rmse = (imdff ** 2).mean().sqrt()  # 根号MSE  Python库中的mean()方法一般都会直接帮我们计算好最终的均值结果，不必我们手动再除以3：
    ps = 20 * torch.log10(1 / rmse)  # 计算得到PSNR --- 注意：此时公式中的MAXI为1
    return ps


# validation时 average=False
# Evaluation时 False
# img: [B, C, W, H]
def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR) / len(PSNR) if average else sum(PSNR)


########## SSIM ##########
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
