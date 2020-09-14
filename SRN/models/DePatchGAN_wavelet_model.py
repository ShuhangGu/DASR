import os
import logging
from collections import OrderedDict
from utils.util import forward_chop
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from pytorch_wavelets import DWTForward, DWTInverse
from utils.receptive_cal import *
import models.networks as networks
from models.modules.loss import GANLoss, GradientPenaltyLoss, discriminator_loss, generator_loss, PerceptualLoss
from PerceptualSimilarity.models.util import PerceptualLoss as val_lpips
from .base_model import BaseModel
import utils.util as util
from PerceptualSimilarity.util import util as util_LPIPS
logger = logging.getLogger('base')
from utils.util import DWT

class DePatch_wavelet_GANModel(BaseModel):
    def __init__(self, opt):
        super(DePatch_wavelet_GANModel, self).__init__(opt)
        train_opt = opt['train']
        self.chop = opt['chop']
        self.scale = opt['scale']
        self.is_test = opt['is_test']
        self.val_lpips = opt['val_lpips']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        if self.is_test:
            self.netD = networks.define_D(opt).to(self.device)
            self.netD.train()
        self.load()  # load G and D if needed
        # Wavelet

        # self.DWT2 = DWTForward(J=1, mode='symmetric', wave='haar').to(self.device)
        self.DWT2 = DWT().to(self.device)
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
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None


            # G feature loss
            if train_opt['feature_weight'] > 0:

                self.l_fea_type = train_opt['feature_criterion']
                if self.l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif self.l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif self.l_fea_type == 'LPIPS':
                    self.cri_fea = PerceptualLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
                self.l_fea_type = None

            if self.cri_fea and self.l_fea_type in ['l1', 'l2']:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.ragan = train_opt['ragan']
            self.cri_gan_G = generator_loss
            self.cri_gan_D = discriminator_loss
            # D_update_ratio and D_init_iters are for WGAN
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
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

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

        self.cri_fea_lpips = val_lpips(model='net-lin', net='alex').to(self.device)

    def feed_data(self, data, istrain=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if istrain:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)
        else:
            self.var_H = data['HR'].to(self.device)

    def optimize_parameters(self, step):
        # G
        self.optimizer_G.zero_grad()
        self.fake_L = self.netG(self.var_H)

        # Wavelets
        self.lf_fake, self.hf_fake = self.DWT2(self.fake_L)
        _, self.hf_real = self.DWT2(self.var_ref)
        self.lf_bicubic, _ = self.DWT2(self.var_L)

        if norm:
            self.lf_fake, self.lf_bicubic = self.lf_fake / 2., self.lf_bicubic / 2.
            self.hf_fake, self.hf_real = self.hf_fake * 0.5 + 0.5, self.hf_real * 0.5 + 0.5



        # from PIL import Image
        # import numpy as np
        # a, b = Image.fromarray(np.uint8(self.lf_bicubic[0].detach().cpu().numpy().transpose(1, 2, 0)*255)), \
        #        Image.fromarray(np.uint8(self.hf_real[0][:3].detach().cpu().numpy().transpose(1, 2, 0)*255))
        # # c = Image.fromarray(np.uint8(self.var_L[0].cpu().numpy().transpose(1, 2, 0)*255))
        # # d = Image.fromarray(np.uint8(self.fake_L[0].detach().cpu().numpy().transpose(1, 2, 0)*255))
        # a.show()
        # b.show()
        # c.show()
        # d.show()

        # self.hf_fake = self.hf_fake[0].to(self.device) * 0.5 + 0.5
        # # self.fake_wlt_Ht = self.fake_wlt_Ht[0].to(self.device)
        # LH, HL, HH = self.hf_fake[:, 0, :, :, :], \
        #              self.hf_fake[:, 1, :, :, :], \
        #              self.hf_fake[:, 2, :, :, :]
        # self.hf_fake = torch.cat((LH, HL, HH), dim=1)  # cat
        # # self.fake_wlt_Ht = (LH + HL + HH) / 3.
        #
        #
        # self.hf_real = self.hf_real[0].to(self.device) * 0.5 + 0.5
        # # self.wave_H = self.wave_H[0].to(self.device)
        # LH, HL, HH = self.hf_real[:, 0, :, :, :], \
        #              self.hf_real[:, 1, :, :, :], \
        #              self.hf_real[:, 2, :, :, :]
        # self.hf_real = torch.cat((LH, HL, HH), dim=1)  # cat
        # self.wave_H = (LH + HL + HH) / 3.

        # rand = torch.rand(1).item()
        # sample = rand * self.var_ref + (1 - rand) * self.fake_L
        # _, sample = self.DWT2(sample)
        # # sample = sample[0].to(self.device) * 0.5 + 0.5
        # # LH, HL, HH = sample[:, 0, :, :, :], \
        # #              sample[:, 1, :, :, :], \
        # #              sample[:, 2, :, :, :]
        # # sample = torch.cat((LH, HL, HH), dim=1)  # cat
        # gp_tex = self.netD(sample)
        # gradient = torch.autograd.grad(gp_tex.mean(), sample, create_graph=True)[0]
        # grad_pen = 10 * (gradient.norm() - 1) ** 2

        if self.ragan:
            # real_tex = model_d(disc_img, fake_img)
            # fake_tex = model_d(fake_img, disc_img)
            pred_d_real_H = self.netD(self.hf_real)
            pred_d_fake_H = self.netD(self.hf_fake)

            real_tex = pred_d_real_H - (pred_d_fake_H).mean(0, keepdim=True)

            pred_d_real_H = self.netD(self.hf_real)
            pred_d_fake_H = self.netD(self.hf_fake)
            fake_tex = pred_d_fake_H - (pred_d_real_H).mean(0, keepdim=True)

        else:
            real_tex, fake_tex = self.netD(self.hf_real), self.netD(self.hf_fake)

        self.optimizer_D.zero_grad()
        l_d_total = self.cri_gan_D(real_tex, fake_tex, True, grad_pen)
        l_d_total.backward(retain_graph=True)
        self.optimizer_D.step()

        # G

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.lf_fake, self.lf_bicubic)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                if self.l_fea_type == 'LPIPS':
                    l_g_fea = self.l_fea_w * self.cri_fea(self.fake_L, self.var_L)
                    l_g_total += l_g_fea
                elif self.l_fea_type in ['l1', 'l2']:
                    real_fea = self.netF(self.var_L).detach()
                    fake_fea = self.netF(self.fake_L)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_total += l_g_fea

            # pred_g_fake_H = self.netD(self.hf_fake)
            l_g_gan = self.l_gan_w * self.cri_gan_G(fake_tex, True, None)

            l_g_total += l_g_gan
            self.optimizer_G.zero_grad()
            l_g_total.backward()
            self.optimizer_G.step()

        # if self.opt['train']['gan_type'] == 'wgan-gp':
        #     batch_size = self.var_ref.size(0)
        #     if self.random_pt.size(0) != batch_size:
        #         self.random_pt.resize_(batch_size, 1, 1, 1)
        #     self.random_pt.uniform_()  # Draw random interpolation points
        #     interp = self.random_pt * self.fake_L.detach() + (1 - self.random_pt) * self.var_ref
        #     interp.requires_grad = True
        #     interp_crit = self.netD(interp)
        #     l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)
        #     l_d_total += l_d_gp

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
                # self.log_dict['l_g_LL_pix'] = l_g_LL_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # D
        self.log_dict['l_d_total'] = l_d_total.item()
        # self.log_dict['l_d_fake'] = l_d_fake.item()

        # if self.opt['train']['gan_type'] == 'wgan-gp':
        #     self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real_H'] = real_tex.mean().data.item()
        self.log_dict['D_fake_H'] = fake_tex.mean().data.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.chop:
                self.fake_L = forward_chop(self.var_H, 1/self.scale, self.netG, min_size=160000)
            else:
                self.fake_L = self.netG(self.var_H)

            if self.is_test:  # Save Domain Distance Map
                sig = torch.nn.Sigmoid()
                __, hfc = self.DWT2(self.fake_L)
                # hfc = hfc[0] * 0.5 + 0.5
                hfc = hfc[0]
                LH, HL, HH = hfc[:, 0, :, :, :], \
                             hfc[:, 1, :, :, :], \
                             hfc[:, 2, :, :, :]
                hfc = torch.cat((LH, HL, HH), dim=1)
                realorfake = sig(self.netD(hfc)).cpu().detach().numpy()
                currentLayer_h, currentLayer_w = receptive_cal(hfc.shape[2]), receptive_cal(hfc.shape[3])
                self.realorfake = getWeights(realorfake, hfc, currentLayer_h, currentLayer_w)

            if self.val_lpips:
                fake_L, real_L = util.tensor2img(self.fake_L), util.tensor2img(self.var_L)
                fake_L, real_L = fake_L[:, :, [2, 1, 0]], real_L[:, :, [2, 1, 0]]
                fake_L, real_L = util_LPIPS.im2tensor(fake_L), util_LPIPS.im2tensor(real_L)
                self.LPIPS = self.cri_fea_lpips(fake_L, real_L)[0][0][0][0]
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()

        out_dict['SR'] = self.fake_L.detach()[0].float().cpu()
        if self.val_lpips:
            out_dict['LPIPS'] = self.LPIPS.detach().float().cpu()

        if self.is_test:
            out_dict['realorfake'] = self.realorfake
        if need_HR:
            out_dict['HR'] = self.var_L.detach()[0].float().cpu()
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
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
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
        load_path_D = self.opt['path']['pretrain_model_D']
        if load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
