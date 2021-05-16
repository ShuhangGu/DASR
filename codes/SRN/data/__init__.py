'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHR_Trans_Wavelet_GAN':
        from data.LRHR_Trans_Wavelet_GAN import LRHRTransWaveletGAN as D
    elif mode == 'LRHR_wavelet_unpair':
        from data.LRHR_wavelet_unpairMix_dataset import LRHR_wavelet_Mixunpair_Dataset as D
    elif mode == 'LRHR_wavelet_unpair_fake_real_w_EQ':
        from data.LRHR_wavelet_unpairEq_dataset import LRHR_wavelet_Equnpair_Dataset as D
    elif mode == 'LRHR_wavelet_unpair_fake_weights_EQ':
        from data.LRHR_wavelet_unpairEq_fake_w_dataset import LRHR_wavelet_Equnpair_Dataset as D
    elif mode == 'LRHR_unpair':
        from data.LRHR_unpair_dataset import LRHR_unpair_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
