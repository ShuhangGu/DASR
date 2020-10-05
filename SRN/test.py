import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
from utils.receptive_cal import *
from PIL import Image

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name, 'imgs')
    util.mkdir(dataset_dir)
    if opt['save_RealorFake']:
        dataset_dir_patchrealfake = os.path.join(opt['path']['results_root'], test_set_name, 'RealOrFake')
        util.mkdir(dataset_dir_patchrealfake)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    for data in test_loader:
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        model.feed_data(data, False)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        visuals = model.get_current_visuals(need_HR=need_HR)

        sr_img = util.tensor2img(visuals['SR'])  # uint8
        if opt['save_RealorFake']:
            patch_RealFake = visuals['realorfake']
            need_HR = True

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        if opt['save_RealorFake']:
            save_patch_path = os.path.join(dataset_dir_patchrealfake, img_name + '.npy')
            np.save(save_patch_path, patch_RealFake)
            # print(patch_RealFake.shape)
            # weights_matrix = util

        # calculate PSNR and SSIM
        if need_HR:
            gt_img = util.tensor2img(visuals['HR'])
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            if opt['val_lpips']:
                lpips = visuals['LPIPS'].numpy()

            crop_border = test_loader.dataset.opt['scale']
            cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if opt['val_lpips']:
                test_results['lpips'].append(lpips)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                if opt['val_lpips']:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; LPIPS: {:.3f}.'\
                        .format(img_name, psnr, ssim, psnr_y, ssim_y, lpips))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f};.'\
                        .format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            # logger.info(img_name)
            logger.info('{} - mean_rf: {}'.format(img_name, np.mean(visuals['realorfake'])))

    if need_HR:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        if opt['val_lpips']:
            avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('----Average PSNR/SSIM/LPIPS results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.3f}\n'\
                    .format(test_set_name, ave_psnr, ave_ssim, avg_lpips))
        else:
            logger.info('----Average PSNR/SSIM/LPIPS results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n' \
                    .format(test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
                .format(ave_psnr_y, ave_ssim_y))
