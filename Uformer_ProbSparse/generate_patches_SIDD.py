from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

# SIDD：/Uformer
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/NH_haze/train --tar_dir /home/dell/桌面/2022毕业设计/Datasets/NH-HAZE/train_patches
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/NH_haze/test --tar_dir /home/dell/桌面/2022毕业设计/Datasets/NH-HAZE/test_patches
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/Dense_Haze_Ntire19/train --tar_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Datasets/Dense_Haze_Ntire19/train_patches
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/Dense_Haze_Ntire19/test --tar_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Datasets/Dense_Haze_Ntire19/test_patches
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/0O-HAZY_NTIRE_2018/train --tar_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Datasets/O-HAZY_500/train_patches
# python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/0O-HAZY_NTIRE_2018/test --tar_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Datasets/O-HAZY_500/test_patches

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../SIDD_Medium_Srgb/Data', type=str,
                    help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/denoising/sidd/train', type=str,
                    help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
#######################################################
parser.add_argument('--num_patches', default=500, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches  # 默认300
NUM_CORES = args.num_cores  # cpu核数，默认10

noisy_patchDir = os.path.join(tar, 'hazy')
clean_patchDir = os.path.join(tar, 'gt')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

# get sorted folders
files_GT = natsorted(glob(os.path.join(src, 'gt', '*.png')))
files_HAZY = natsorted(glob(os.path.join(src, 'hazy', '*.png')))

noisy_files, clean_files = [], []
for file_ in files_GT:
    clean_files.append(file_)
for file_ in files_HAZY:
    noisy_files.append(file_)


def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    # 生成patches
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i + 1, j + 1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i + 1, j + 1)), clean_patch)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))
