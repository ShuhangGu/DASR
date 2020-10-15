import argparse
import sys
from os import listdir
from os.path import join, basename
from tqdm import tqdm
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import torch
import utils
sys.path.append('../PerceptualSimilarity')
sys.path.append('../')
import PerceptualSimilarity as ps


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--dir', default=None, type=str, help='directory of the images that are being tested')
parser.add_argument('--gt_dir', default=None, type=str, help='directory of the gt images')
parser.add_argument('--border_crop', default=None, type=int, help='number of pixels to remove at the border')
opt = parser.parse_args()

lpips_network = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())
# lpips_network = ps.PerceptualLoss(use_gpu=torch.cuda.is_available())

files = [join(opt.dir, x) for x in listdir(opt.dir) if utils.is_image_file(x)]
files.sort()
gt_files = [join(opt.gt_dir, x) for x in listdir(opt.gt_dir) if utils.is_image_file(x)]
gt_files.sort()

psnr = psnr_col = ssim = lpips = 0
files_bar = tqdm(files)
for file, gt_file in zip(files_bar, gt_files):
    if basename(file) != basename(gt_file):
        print("WARNING: file and gt_file do not have the same name")
    img = np.array(Image.open(file))[..., :3]
    gt_img = np.array(Image.open(gt_file))
    if img.size != gt_img.size:
        w, h, _ = np.shape(img)
        w_gt, h_gt, _ = np.shape(gt_img)
        gt_img = gt_img[:w, :h, :]
    if opt.border_crop is not None and opt.border_crop > 0:
        img = img[opt.border_crop:-opt.border_crop, opt.border_crop:-opt.border_crop, :]
        gt_img = gt_img[opt.border_crop:-opt.border_crop, opt.border_crop:-opt.border_crop, :]
    psnr += compare_psnr(img, gt_img)
    psnr_col += compare_psnr(img.mean(1).mean(0) / 255., gt_img.mean(1).mean(0) / 255.)
    ssim += compare_ssim(img, gt_img, multichannel=True)
    img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.
    gt_img = torch.Tensor(gt_img).permute(2, 0, 1).unsqueeze(0) / 255.
    if torch.cuda.is_available():
        img = img.cuda()
        gt_img = gt_img.cuda()
    lpips += lpips_network.forward(img, gt_img, normalize=True).mean().data.item()

psnr /= len(files)
psnr_col /= len(files)
ssim /= len(files)
lpips /= len(files)

print("PSNR: " + str(psnr))
print("PSNR_col: " + str(psnr_col))
print("SSIM: " + str(ssim))
print("LPIPS: " + str(lpips))
