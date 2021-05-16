import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'De_Resnet':
        from .Degradation_Resnet import  DegrationModel as M
    elif model == 'De_patch_wavelet_GAN':
        from .DePatchGAN_wavelet_model import DePatch_wavelet_GANModel as M
    elif model == 'DASR':
        from .DASR_model import DASR_Model as M
    elif model == 'DASR_Adaptive_Model':
        from .DASR_Adaptive_model import DASR_Adaptive_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
