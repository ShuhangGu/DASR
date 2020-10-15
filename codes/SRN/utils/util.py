import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging
import torch.nn.parallel as P
import math
import torch.nn as nn
####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def setup_old_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


####################
# image convert
####################

def forward_chop(img, scale, model, shave=20, min_size=160000):
    # scale = 1 if self.input_large else self.scale[self.idx_scale]
    n_GPUs = min(1, 4)
    # height, width
    h, w = img.size()[-2:]

    top = slice(0, h // 2 + shave)
    bottom = slice(h - h // 2 - shave, h)
    left = slice(0, w // 2 + shave)
    right = slice(w - w // 2 - shave, w)
    x_chops = [torch.cat([
        img[..., top, left],
        img[..., top, right],
        img[..., bottom, left],
        img[..., bottom, right]])]

    y_chops = []
    if h * w < 4 * min_size:
        for i in range(0, 4, n_GPUs):
            x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
            y = P.data_parallel(model, *x, range(n_GPUs))
            if not isinstance(y, list): y = [y]
            if not y_chops:
                y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
            else:
                for y_chop, _y in zip(y_chops, y):
                    y_chop.extend(_y.chunk(n_GPUs, dim=0))
    else:
        for p in zip(*x_chops):
            p = map(lambda x:x.reshape([1]+list(x.size())), p)
            y = forward_chop(*p, scale, model, shave=shave, min_size=min_size)
            if not isinstance(y, list): y = [y]
            if not y_chops:
                y_chops = [[_y] for _y in y]
            else:
                for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

    h *= scale
    w *= scale
    h, w = round(h), round(w)
    if h % 2 != 0: h += 1
    if w % 2 != 0: w += 1
    top = slice(0, h // 2)
    bottom = slice(h - h // 2, h)
    bottom_r = slice(h // 2 - h, None)
    left = slice(0, w // 2)
    right = slice(w - w // 2, w)
    right_r = slice(w // 2 - w, None)

    # batch size, number of color channels
    b, c = y_chops[0][0].size()[:-2]
    y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    for y_chop, _y in zip(y_chops, y):
        _y[..., top, left] = y_chop[0][..., top, left]
        _y[..., top, right] = y_chop[1][..., top, right_r]
        _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
        _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

    if len(y) == 1: y = y[0]

    return y


def b_split(batch, mask):
    real_data, fake_data = [], []
    for i in range(len(mask)):
        j = int(mask[i])
        if j == 0:
            fake_data.append(torch.unsqueeze(batch[i], dim=0))
        elif j == 1:
            real_data.append(torch.unsqueeze(batch[i], dim=0))
    if real_data:
        real_data = torch.cat(real_data)
    if fake_data:
        fake_data = torch.cat(fake_data)

    return fake_data, real_data

def b_merge(real_data, fake_data, mask):
    res = []
    for i in range(len(mask)):
        j = int(mask[i])
        # m, n = 0, 0
        if j == 0:
            res.append(torch.unsqueeze(fake_data[i], dim=0))
            # m += 1
        elif j == 1:
            res.append(torch.unsqueeze(real_data[i], dim=0))
            # n += 1
    return torch.cat(res)



def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import os
    from data.util import bgr2ycbcr
    sr_path = '/media/4T/Dizzy/BasicSR-master/results/Test/DeviceVal20_2xd/'
    hr_path = '/media/4T/Dizzy/SR_classical_DataSet/RealWorldDataSet/Device_degration_Data/City100_iPhoneX/HR_val/'
    psnrtotal, ssimtotal = 0, 0
    psnr_ytotal, ssim_ytotoal = 0, 0
    idx = 0
    crop_border = 4
    for name in os.listdir(hr_path):
        name = name.split('.')[0]
        sr_img_np = np.array(Image.open(sr_path+name+'.png'))/255
        hr_img_np = np.array(Image.open(hr_path+name+'.PNG'))/255
        sr_img_np = sr_img_np[crop_border:-crop_border, crop_border:-crop_border, :]
        hr_img_np = hr_img_np[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = calculate_psnr(hr_img_np*255, sr_img_np*255)
        ssim_ = calculate_ssim(hr_img_np*255, sr_img_np*255)
        psnrtotal += psnr
        ssimtotal += ssim_
        sr_img_np_y = bgr2ycbcr(sr_img_np, only_y=True)
        hr_img_np_y = bgr2ycbcr(hr_img_np, only_y=True)
        psnr = calculate_psnr(sr_img_np_y*255, hr_img_np_y*255)
        ssim_ = calculate_ssim(sr_img_np_y*255, hr_img_np_y*255)
        psnr_ytotal += psnr
        ssim_ytotoal += ssim_
        idx += 1

    print('PSNR: ',psnrtotal/idx,'SSIM: ', ssimtotal/idx)
    print('PSNR_y: ',psnr_ytotal/idx,'SSIM_y: ', ssim_ytotoal/idx)


