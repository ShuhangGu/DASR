import sys
import os
import cv2
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from utils.util import forward_chop
import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss, PerceptualLoss, discriminator_loss
logger = logging.getLogger('base')
from pytorch_wavelets import DWTForward, DWTInverse
from utils.util import b_split, b_merge
from models.modules.architecture import FilterHigh, FilterLow
from PerceptualSimilarity.models.util import PerceptualLoss as val_lpips
import utils.util as util
from PerceptualSimilarity.util import util as util_LPIPS


class DASR_Adaptive_Model(BaseModel):
    def __init__(self, opt):
        super(DASR_Adaptive_Model, self).__init__(opt)
        train_opt = opt['train']
        self.chop = opt['chop']
        self.scale = opt['scale']
        self.val_lpips = opt['val_lpips']
        self.use_domain_distance_map = opt['use_domain_distance_map']
        if self.is_train:
            self.use_patchD_opt = opt['network_patchD']['use_patchD_opt']

            # GD gan loss
            self.ragan = train_opt['ragan']
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_H_target_w = train_opt['gan_H_target']
            self.l_gan_H_source_w = train_opt['gan_H_source']

            # patchD gan loss
            self.cri_patchD_gan = discriminator_loss

        # define networks and load pretrained models

        self.netG = networks.define_G(opt).to(self.device)  # G
        self.net_patchD = networks.define_patchD(opt).to(self.device)
        if self.is_train:
            if self.l_gan_H_target_w > 0:
                self.netD_target = networks.define_D(opt).to(self.device)  # D
                self.netD_target.train()
            if self.l_gan_H_source_w > 0:
                self.netD_source = networks.define_pairD(opt).to(self.device)  # D
                self.netD_source.train()
            self.netG.train()

        self.load()  # load G and D if needed


        # Frequency Separation
        self.norm = opt['FS_norm']
        if opt['FS']['fs'] == 'wavelet':
            # Wavelet
            self.DWT2 = DWTForward(J=1, mode='reflect', wave='haar').to(self.device)
            self.fs = self.wavelet_s
            self.filter_high = FilterHigh(kernel_size=opt['FS']['fs_kernel_size'], gaussian=True).to(self.device)
        elif opt['FS']['fs'] == 'gau':
            # Gaussian
            self.filter_low, self.filter_high = FilterLow(kernel_size=opt['FS']['fs_kernel_size'], gaussian=True).to(self.device), \
                                            FilterHigh(kernel_size=opt['FS']['fs_kernel_size'], gaussian=True).to(self.device)
            self.fs = self.filter_func
        elif opt['FS']['fs'] == 'avgpool':
            # avgpool
            self.filter_low, self.filter_high = FilterLow(kernel_size=opt['FS']['fs_kernel_size']).to(self.device), \
                                            FilterHigh(kernel_size=opt['FS']['fs_kernel_size']).to(self.device)
            self.fs = self.filter_func
        else:
            raise NotImplementedError('FS type [{:s}] not recognized.'.format(opt['FS']['fs']))

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
                self.l_pix_LL_w = train_opt['pixel_LL_weight']
                self.sup_LL = train_opt['sup_LL']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            self.l_fea_type = train_opt['feature_criterion']
            # G feature loss
            if train_opt['feature_weight'] > 0:
                if self.l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif self.l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif self.l_fea_type == 'LPIPS':
                    self.cri_fea = PerceptualLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(self.l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea and self.l_fea_type in ['l1', 'l2']:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # D_update_ratio and D_init_iters are for WGAN
            self.G_update_inter = train_opt['G_update_inter']
            self.D_update_inter = train_opt['D_update_inter']
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers

            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            if self.l_gan_H_target_w > 0:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D_target = torch.optim.Adam(self.netD_target.parameters(), lr=train_opt['lr_D'], \
                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D_target)

            if self.l_gan_H_source_w > 0:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D_source = torch.optim.Adam(self.netD_source.parameters(), lr=train_opt['lr_D'], \
                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D_source)

            # Patch Discriminator
            if self.use_patchD_opt:
                self.optimizer_patchD = torch.optim.Adam(self.net_patchD.parameters(),
                                                         lr=opt['network_patchD']['lr'],
                                                         betas=[opt['network_patchD']['beta1_G'], 0.999])
                self.optimizers.append(self.optimizer_patchD)
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()
        self.fake_H = None

        # # Debug
        if self.val_lpips:
            self.cri_fea_lpips = val_lpips(model='net-lin', net='alex').to(self.device)

    def feed_data(self, data, istrain):
        # LR
        if istrain and 'HR' in data:  # train or val
            HR_pair = data['HR'].to(self.device)
            HR_unpair = data['HR_unpair'].to(self.device)
            # fake_w = data['fake_w'].to(self.device)
            real_LR = data['LR_real'].to(self.device)
            fake_LR = data['LR_fake'].to(self.device)

            with torch.no_grad():
                self.var_L = torch.cat([fake_LR, real_LR], dim=0)
                self.var_H = torch.cat([HR_pair, HR_unpair], dim=0)

            self.mask = []
            B = self.var_L.shape[0]
            self.mask += [0] * (B // 2)
            self.mask += [1] * (B - B//2)

        else:
            self.var_L = data['LR'].to(self.device)
            if 'HR' in data:
                self.var_H = data['HR'].to(self.device)
                self.needHR = True
            else:
                self.needHR = False




    def optimize_parameters(self, step):
        del self.fake_H
        torch.cuda.empty_cache()
        self.adaptive_weights = self.net_patchD(self.var_L)
        B = self.var_L.shape[0]

        # self.weights = adaptive_weights
        if self.use_domain_distance_map:
            self.domain_distance_map = self.adaptive_weights[:B//2, ...]
            self.domain_distance_map = F.interpolate(self.domain_distance_map,
                                                     size=(self.var_H.shape[2], self.var_H.shape[3]),
                                                     mode='bilinear', align_corners=False)
        if self.use_patchD_opt:
            fake_weights, real_weights = self.adaptive_weights[:B//2], self.adaptive_weights[B//2:]
            patch_D_gan_loss = self.cri_patchD_gan(real_weights, fake_weights)
            self.optimizer_patchD.zero_grad()
            patch_D_gan_loss.backward(retain_graph=True)
            self.optimizer_patchD.step()


        # G
        if step % self.G_update_inter == 0:
            self.fake_H = self.netG(self.var_L, self.adaptive_weights)
        else:
            with torch.no_grad():
                self.fake_H = self.netG(self.var_L, self.adaptive_weights)
        self.fake_LL, self.fake_Hc = self.fs(self.fake_H, norm=self.norm)
        self.real_LL, self.real_Hc = self.fs(self.var_H, norm=self.norm)

        # Splitting data
        # Fake data
        self.fake_SR_source, _ = b_split(self.fake_H, self.mask)
        self.fake_SR_LL_source, _ = b_split(self.fake_LL, self.mask)
        self.fake_SR_Hf_source, self.fake_SR_Hf_target = b_split(self.fake_Hc, self.mask)

        # Real data
        self.real_HR_source, _ = b_split(self.var_H, self.mask)
        self.real_HR_LL_source, _ = b_split(self.real_LL, self.mask)
        self.real_HR_Hf_source, self.real_HR_Hf_target = b_split(self.real_Hc, self.mask)


        if step % self.G_update_inter == 0:
            l_g_total = 0
            if self.cri_pix:  # pixel loss
                if self.use_domain_distance_map:
                    l_g_pix = self.l_pix_w * \
                              torch.mean(self.domain_distance_map * torch.abs(self.fake_SR_source - self.real_HR_source))
                else:
                    l_g_pix = self.cri_pix(self.fake_SR_source, self.real_HR_source)
                l_g_total += self.l_pix_w * l_g_pix

                if self.sup_LL:
                    l_g_LL_pix = self.cri_pix(self.fake_SR_LL_source, self.real_HR_LL_source)
                    l_g_total += self.l_pix_LL_w * l_g_LL_pix

            if self.l_fea_type in ['l1', 'l2'] and self.cri_fea:  # feature loss
                real_fea = self.netF(self.real_HR_source).detach()
                fake_fea = self.netF(self.fake_SR_source)
                l_g_fea = self.cri_fea(fake_fea, real_fea)

                l_g_total += self.l_fea_w * l_g_fea

            elif self.l_fea_type == 'LPIPS' and self.cri_fea:
                l_g_fea = self.cri_fea(self.fake_SR_source, self.real_HR_source)
                l_g_total += self.l_fea_w * l_g_fea


            # G gan target loss
            if self.l_gan_H_target_w > 0:
                pred_g_Hf_target_fake = self.netD_target(self.fake_SR_Hf_target)

                if self.ragan:
                    pred_g_Hf_target_real = self.netD_target(self.real_HR_Hf_target).detach()
                    l_g_gan_target_Hf = self.l_gan_H_target_w * \
                        (self.cri_gan(pred_g_Hf_target_fake - pred_g_Hf_target_real.mean(0, keepdim=True), True) +
                        self.cri_gan(pred_g_Hf_target_real - pred_g_Hf_target_fake.mean(0, keepdim=True), False)) / 2
                else:
                    l_g_gan_target_Hf = self.cri_gan(pred_g_Hf_target_fake, True)
                l_g_total += self.l_gan_H_target_w * l_g_gan_target_Hf

            # G_gan_source_loss
            if self.l_gan_H_source_w > 0:
                pred_g_Hf_source_fake = self.netD_source(self.fake_SR_Hf_source)
                if self.ragan:
                    pred_g_Hf_source_real = self.netD_source(self.real_HR_Hf_source).detach()
                    l_g_gan_source_Hf = self.l_gan_H_source_w * \
                               (self.cri_gan(pred_g_Hf_source_fake - pred_g_Hf_source_real.mean(0, keepdim=True), True) +
                                self.cri_gan(pred_g_Hf_source_real - pred_g_Hf_source_fake.mean(0, keepdim=True), False)) / 2
                else:
                    l_g_gan_source_Hf = self.l_gan_H_source_w * self.cri_gan(pred_g_Hf_source_fake, True)
                l_g_total += l_g_gan_source_Hf

            self.optimizer_G.zero_grad()
            l_g_total.backward()
            self.optimizer_G.step()
        else:
            self.optimizer_G.zero_grad()


        # D
        if step % self.D_update_inter == 0:

            # target domain
            if self.l_gan_H_target_w > 0:
                pred_d_target_real = self.netD_target(self.real_HR_Hf_target.detach())
                pred_d_target_fake = self.netD_target(self.fake_SR_Hf_target.detach())  # detach to avoid BP to G
                if self.ragan:
                    l_d_target_real = self.cri_gan(pred_d_target_real - pred_d_target_fake.mean(0, keepdim=True), True)
                    l_d_target_fake = self.cri_gan(pred_d_target_fake - pred_d_target_real.mean(0, keepdim=True), False)
                else:
                    l_d_target_real = self.cri_gan(pred_d_target_real, True)
                    l_d_target_fake = self.cri_gan(pred_d_target_fake, False)

                l_d_target_total = (l_d_target_real + l_d_target_fake) / 2

                self.optimizer_D_target.zero_grad()
                l_d_target_total.backward()
                self.optimizer_D_target.step()
        else:
            self.optimizer_D_target.zero_grad()

        if step % self.D_update_inter == 0:
            # source domain
            if self.l_gan_H_source_w > 0:
                pred_d_source_real = self.netD_source(self.real_HR_Hf_source.detach())
                pred_d_source_fake = self.netD_source(self.fake_SR_Hf_source.detach())  # detach to avoid BP to G

                if self.ragan:
                    l_d_source_real = self.cri_gan(pred_d_source_real - pred_d_source_fake.mean(0, keepdim=True), True)
                    l_d_source_fake = self.cri_gan(pred_d_source_fake - pred_d_source_real.mean(0, keepdim=True), False)
                else:
                    l_d_source_real = self.cri_gan(pred_d_source_real, True)
                    l_d_source_fake = self.cri_gan(pred_d_source_fake, False)

                l_d_source_total = (l_d_source_fake + l_d_source_real) / 2

                self.optimizer_D_source.zero_grad()
                l_d_source_total.backward()
                self.optimizer_D_source.step()
        else:
            self.optimizer_D_source.zero_grad()

        # set log
        if step % self.G_update_inter == 0:
            # G
            if self.cri_pix:
                self.log_dict['loss/l_g_pix'] = l_g_pix.item()
                if self.sup_LL:
                    self.log_dict['loss/l_g_LL_pix'] = l_g_LL_pix.item()
            if self.cri_fea:
                self.log_dict['loss/l_g_fea'] = l_g_fea.item()
            if self.l_gan_H_target_w > 0:
                self.log_dict['loss/l_g_gan_target_Hf'] = l_g_gan_target_Hf.item()
            if self.l_gan_H_source_w > 0:
                self.log_dict['loss/l_g_gan_source_H'] = l_g_gan_source_Hf.item()

        # D outputs
        if step % self.D_update_inter == 0:
            if self.l_gan_H_target_w > 0:
                self.log_dict['loss/l_d_target_total'] = l_d_target_total.item()
                self.log_dict['disc_Score/D_real_target_H'] = 1 / (1 + torch.exp(-torch.mean(pred_d_target_real.detach())).item())
                self.log_dict['disc_Score/D_fake_target_H'] = 1 / (1 + torch.exp(-torch.mean(pred_d_target_fake.detach())).item())
            if self.l_gan_H_source_w > 0:
                self.log_dict['loss/l_d_total'] = l_d_source_total.item()
                self.log_dict['disc_Score/D_real_source_H'] = 1 / (1 + torch.exp(-torch.mean(pred_d_source_real.detach())).item())
                self.log_dict['disc_Score/D_fake_source_H'] = 1 / (1 + torch.exp(-torch.mean(pred_d_source_fake.detach())).item())
            if self.use_patchD_opt:
                self.log_dict['patchD_Score/real_weights'] = 1 / (1 + torch.exp(-torch.mean(real_weights.detach())).item())
                self.log_dict['patchD_Score/fake_weights'] = 1 / (1 + torch.exp(-torch.mean(fake_weights.detach())).item())

    def test(self, tsamples=False):
        torch.cuda.empty_cache()
        self.netG.eval()
        with torch.no_grad():
            self.adaptive_weights = self.net_patchD(self.var_L)
            if self.chop:
                self.fake_H = forward_chop(self.var_L, self.scale, self.netG, min_size=320000)
            else:
                self.fake_H = self.netG(self.var_L, self.adaptive_weights)
            if not tsamples and self.val_lpips:
                fake_H, real_H = util.tensor2img(self.fake_H), util.tensor2img(self.var_H)
                fake_H, real_H = fake_H[:, :, [2, 1, 0]], real_H[:, :, [2, 1, 0]]
                fake_H, real_H = util_LPIPS.im2tensor(fake_H), util_LPIPS.im2tensor(real_H)
                self.LPIPS = self.cri_fea_lpips(fake_H, real_H)[0][0][0][0]
            self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True, tsamples=False):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        if tsamples:
            out_dict['hf'] = self.filter_high(self.fake_H).float().cpu()
            out_dict['gt_hf'] = self.filter_high(self.var_H).float().cpu()
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
            out_dict['HR_hf'] = self.filter_high(self.var_H).detach().float().cpu()
        if not tsamples:
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        else:
            out_dict['SR'] = self.fake_H.detach().float().cpu()
        if not tsamples and self.val_lpips:
            out_dict['LPIPS'] = self.LPIPS.detach().float().cpu()
        if not tsamples and self.needHR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        if self.use_domain_distance_map:
            out_dict['adaptive_weights'] = torch.mean(self.adaptive_weights.float().cpu())
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:

            # Discriminator_wlt
            if self.l_gan_H_target_w > 0:
                s, n = self.get_network_description(self.netD_target)
                if isinstance(self.netD_target, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD_target.__class__.__name__,
                                                     self.netD_target.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD_target.__class__.__name__)

                logger.info('Network D_target structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)


            if self.l_gan_H_source_w > 0:

                # Discriminator_pair
                s, n = self.get_network_description(self.netD_source)
                if isinstance(self.netD_source, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD_source.__class__.__name__,
                                                     self.netD_source.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD_source.__class__.__name__)

                logger.info('Network D_source structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            if self.cri_fea and self.l_fea_type in ['l1', 'l2']:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

        load_path_D_target = self.opt['path']['pretrain_model_D_target']
        if self.opt['is_train'] and load_path_D_target is not None:
            logger.info('Loading pretrained model for D_target [{:s}] ...'.format(load_path_D_target))
            self.load_network(load_path_D_target, self.netD_target)

        load_path_D_source = self.opt['path']['pretrain_model_D_source']
        if self.opt['is_train'] and load_path_D_source is not None:
            logger.info('Loading pretrained model for D_source [{:s}] ...'.format(load_path_D_source))
            self.load_network(load_path_D_source, self.netD_source)

        load_path_patch_Discriminator = self.opt['path']['Patch_Discriminator']
        if load_path_patch_Discriminator is not None:
            checkpoint = torch.load(load_path_patch_Discriminator)
            patch_D_state_dict = checkpoint['models_d_state_dict']
            logger.info('Loading pretrained model for Patch_Discriminator [{:s}] ...'.format(load_path_patch_Discriminator))
            if isinstance(self.net_patchD, nn.DataParallel):
                self.net_patchD = self.net_patchD.module
            self.net_patchD.load_state_dict(patch_D_state_dict)
            # self.load_network(load_path_patch_Discriminator, self.net_patchD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.l_gan_H_target_w > 0:
            self.save_network(self.netD_target, 'D_target', iter_step)
        if self.l_gan_H_source_w > 0:
            self.save_network(self.netD_source, 'D_source', iter_step)

    def wavelet_s(self, x, norm=False):
        LL, Hc = self.DWT2(x)
        Hc = Hc[0]
        if norm:
            LL, Hc = LL * 0.5, Hc * 0.5 + 0.5  # norm [0, 1]

        LH, HL, HH = Hc[:, :, 0, :, :], \
                     Hc[:, :, 1, :, :], \
                     Hc[:, :, 2, :, :]
        Hc = torch.cat((LH, HL, HH), dim=1)
        return LL, Hc

    def filter_func(self, x, norm=False):
        low_f, high_f = self.filter_low(x), self.filter_high(x)
        if norm:
            high_f = high_f * 0.5 + 0.5
        return low_f, high_f


