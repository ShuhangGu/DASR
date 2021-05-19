import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work

    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        # resume_state[]
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(logdir='../../SRN_tb_logger/' + opt['name'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state, opt['train'])  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            model.feed_data(train_data, True)
            model.optimize_parameters(current_step)

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # training samples
            if opt['train']['save_tsamples'] and current_step % opt['train']['save_tsamples'] == 0:
                fake_LRs = os.listdir(opt['datasets']['train']['dataroot_fake_LR'])
                real_LRs = os.listdir(opt['datasets']['train']['dataroot_real_LR'])
                HRs = os.listdir(opt['datasets']['train']['dataroot_HR'])

                for i in range(5):
                    random_index = np.random.choice(range(len(fake_LRs)))
                    fake_LR_path = os.path.join(opt['datasets']['train']['dataroot_fake_LR'], fake_LRs[random_index])
                    real_LR_path = os.path.join(opt['datasets']['train']['dataroot_real_LR'], real_LRs[random_index])
                    HR_path = os.path.join(opt['datasets']['train']['dataroot_HR'], HRs[random_index])
                    fake_LR = np.array(Image.open(fake_LR_path))
                    real_LR = np.array(Image.open(real_LR_path))
                    HR = np.array(Image.open(HR_path))

                    h, w, _ = fake_LR.shape
                    fake_LR = fake_LR[h // 2 - 64:h // 2 + 64, w//2 - 64:w//2+64, :]
                    h, w, _ = HR.shape
                    HR = HR[h // 2 - 64*4:h // 2 + 64*4, w//2 - 64*4:w//2+64*4, :]

                    h, w, _ = real_LR.shape
                    real_LR = real_LR[h // 2 - 64:h // 2 + 64, w//2 - 64:w//2+64, :]


                    fake_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(fake_LR, (2, 0, 1)))).float().unsqueeze(0) / 255
                    real_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(real_LR, (2, 0, 1)))).float().unsqueeze(0) / 255
                    HR = torch.from_numpy(np.ascontiguousarray(np.transpose(HR, (2, 0, 1)))).float().unsqueeze(0) / 255
                    LR = torch.cat([fake_LR, real_LR], dim=0)

                    data = {'LR': LR, 'HR': HR}
                    model.feed_data(data, False)
                    model.test(tsamples=True)
                    visuals = model.get_current_visuals(tsamples=True)
                    fake_SR = visuals['SR'][0]
                    real_SR = visuals['SR'][1]
                    fake_hf = visuals['hf'][0]
                    real_hf = visuals['hf'][1]
                    HR = visuals['HR']
                    HR_hf = visuals['HR_hf'][0]


                    # image_1 = torch.cat([fake_LR[0], fake_SR[0]], dim=2)
                    # image_2 = torch.cat([real_LR[0], real_SR[0]], dim=2)
                    image_1 = np.clip(torch.cat([fake_SR, HR, real_SR], dim=2), 0, 1)
                    image_2 = np.clip(torch.cat([fake_hf, HR_hf, real_hf], dim=2), 0, 1)
                    image = torch.cat([image_1, image_2], dim=1)
                    tb_logger.add_image('train/train_samples_{}'.format(str(i)), image, current_step)
                logger.info('Saved training Samples')


            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                avg_lpips = 0.0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data, False)
                    model.test()


                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    if 'HR' in opt['datasets']['val']['mode']:
                        gt_img = util.tensor2img(visuals['HR'])  # uint8
                    log_info = '{}'.format(val_data['HR_path'][0].split('/')[-1])

                    if opt['val_lpips']:
                        lpips = visuals['LPIPS']
                        avg_lpips += lpips
                        log_info += '         LPIPS:{:.3f}'.format(lpips.numpy())
                    if opt['use_domain_distance_map']:
                        ada_w = visuals['adaptive_weights']
                        log_info += '         Adaptive weights:{:.2f}'.format(ada_w.numpy())
                        # logger.info('{} LPIPS: {:.3f}'.format(val_data['HR_path'][0].split('/')[-1], lpips.numpy()))
                        # print('img:', val_data['HR_path'][0].split('/')[-1], 'LPIPS: %.3f' % lpips.numpy())
                    # else:
                    #     print('img:', val_data['LR_path'][0].split('/')[-1])
                    logger.info(log_info)
                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    if 'HR' in opt['datasets']['val']['mode']:
                        crop_size = opt['scale']
                        gt_img = gt_img / 255.
                        sr_img = sr_img / 255.
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                avg_psnr = avg_psnr / idx
                if opt['val_lpips']:
                    avg_lpips = avg_lpips / idx
                    print('Mean LPIPS:', avg_lpips.numpy())
                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                if opt['val_lpips']:
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, LPIPS: {:.4f}'.format(
                        epoch, current_step, avg_psnr, avg_lpips))
                else:
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('LPIPS', avg_lpips, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
