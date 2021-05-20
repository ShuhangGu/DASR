import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
# import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'sr_resnet':  # SRResNet
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'sft_arch':  # SFT-GAN
        netG = sft_arch.SFT_Net()

    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    elif which_model == 'RRDB_mask':
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'],
                            norm_type=opt_net['norm_type'],
                            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    elif which_model == 'De_Resnet':
        netG = arch.De_Resnet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], downscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=opt_net['act_type'], mode=opt_net['mode'])

    elif which_model == 'DSGAN':
        netG = arch.DSGAN_Generator()

    elif which_model == 'De_Resnet_bilinear':
        netG = arch.De_Resnet_bilinear(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], downscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'])

    elif which_model == 'De_Resnet2xd':
        netG = arch.De_Resnetdx2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], downscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'])

    elif which_model == 'De_RRDB':
        netG = arch.De_Resnet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], downscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'])
    elif which_model == 'RRDB_SEAN':
        netG = arch.RRDBNet_SEAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv', nb_ada=opt_net['ada_nb'])
    elif which_model == 'RRDB_Residual_conv':  # RRDB
        netG = arch.RRDBNet_Residual_conv(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    elif which_model == 'RRDB_Residual_conv_concat':  # RRDB
        netG = arch.RRDBNet_Residual_conv_concat(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv', nb_ada=opt_net['ada_nb'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        # netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
        #     norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'],  nf=opt_net['nf'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_96_patch':
        netD = arch.Discriminator_VGG_96_patch(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192_wavelet':
        netD = arch.Discriminator_VGG_192_wavelet(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_patch':
        netD = arch.Discriminator_VGG_patch(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_48':
        netD = arch.Discriminator_VGG_48(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    elif which_model == 'discriminator_patch':
        netD = arch.NLayerDiscriminator(opt_net['in_nc'], n_layers=opt_net['n_layers'])
    elif which_model == 'DSGAN':
        netD = arch.DiscriminatorBasic(n_input_channels=opt_net['in_nc'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD

def define_pairD(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_pairD']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_48':
        netD = arch.Discriminator_VGG_48(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    elif which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128()
    elif which_model == 'discriminator_patch':
        netD = arch.NLayerDiscriminator(opt_net['in_nc'], opt_net['nf'], opt_net['n_layers'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD

def define_patchD(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_patchD']
    which_model = opt_net['which_patchD']
    norm_layer = opt_net['norm_layer']
    FS_type = opt_net['FS_type']
    kernel_size = opt_net['kernel_size']

    if which_model == 'FSD':
        net_patchD = arch.FS_Discriminator(kernel_size=kernel_size, D_arch='FSD',
                                           filter_type=FS_type, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Patch Discriminator model [{:s}] not recognized'.format(opt_net))
    init_weights(net_patchD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(net_patchD)
    return netD

def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF