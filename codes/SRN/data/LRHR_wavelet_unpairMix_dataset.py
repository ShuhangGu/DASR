import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image

class LRHR_wavelet_Mixunpair_Dataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHR_wavelet_Mixunpair_Dataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.paths_RealLR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
            self.LR_env, self.paths_RealLR = util.get_image_paths(opt['data_type'], opt['dataroot_RealLR'])
            self.LR_env, self.paths_weights = util.get_image_paths(opt['data_type'], opt['dataroot_weights'])

        assert self.paths_HR, 'Error: HR path is empty.'
        # if self.paths_LR and self.paths_HR:
        #     assert len(self.paths_LR) == len(self.paths_HR), \
        #         'HR and LR datasets have different number of images - {}, {}.'.format(\
        #         len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]
        self.DWT2 = DWTForward(J=1, wave='haar', mode='zero')

    def __getitem__(self, index):
        # HR_path, LR_path = None, None fake real
        isReal = False
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        # print(self.paths_LR[-10:])
        # print(self.paths_HR[-10:])
        # get LR image
        LR_path = self.paths_LR[index]    # len(LR) > len(HR)

        img_LR = util.read_img(self.LR_env, LR_path)


        # get HR image
        if self.opt['prefix'] not in os.path.basename(LR_path):
            HR_path = self.paths_HR[index]
            img_HR = util.read_img(self.HR_env, HR_path)

            weights_path = self.paths_weights[index]   # load domain distance weights .npy
            img_weights = np.load(weights_path)[0].transpose((1, 2, 0))
            img_weights = cv2.resize(img_weights, (img_HR.shape[1], img_HR.shape[0]), interpolation=cv2.INTER_LINEAR)
             
        else:
            isReal = True
            numofHR = len(self.paths_HR)
            index_HR = np.random.randint(0, numofHR)
            HR_path = self.paths_HR[index_HR]
            img_HR = util.read_img(self.HR_env, HR_path)
            img_weights = np.ones_like(img_HR)


        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]


        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            # Href, Wref, Cref = img_TransRefer.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

            if isReal:
                H_real, W_real, C_real = img_HR.shape
                rnd_h_real = random.randint(0, max(0, H_real - HR_size))
                rnd_w_real = random.randint(0, max(0, W_real - HR_size))
                img_HR = img_HR[rnd_h_real:rnd_h_real + HR_size, rnd_w_real:rnd_w_real + HR_size, :]
                img_weights = img_weights[rnd_h_real:rnd_h_real + HR_size, rnd_w_real:rnd_w_real + HR_size, :]
            else:
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
                img_weights = img_weights[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]



            # rnd_href = random.randint(0, max(0, Href - HR_size))
            # rnd_wref = random.randint(0, max(0, Wref - HR_size))
            # img_TransRefer = img_TransRefer[rnd_href:rnd_href + HR_size, rnd_wref:rnd_wref + HR_size, :]
            # augmentation - flip, rotate
            img_LR, img_HR, img_weights = util.augment([img_LR, img_HR, img_weights], self.opt['use_flip'], \
                self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0] # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_weights = torch.from_numpy(np.ascontiguousarray(np.transpose(img_weights, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        # img_TransRefer = torch.from_numpy(np.ascontiguousarray(np.transpose(img_TransRefer, (2, 0, 1)))).float()

        # Wavelet Trans
        if self.opt['phase'] == 'train':
            img_HR = img_HR.reshape([1, img_HR.shape[0], img_HR.shape[1], img_HR.shape[2]])
            LL, H_cat = self.DWT2(img_HR)
            LL = LL[0]*0.5

            H_cat = H_cat[0][0]*0.5+0.5
            w_c, C, H, W = H_cat.shape
            H_cat = H_cat.reshape([w_c*C, H, W])

            img_HR = img_HR[0].detach()

            if LR_path is None:
                LR_path = HR_path


        return {'LR': img_LR, 'HR': img_HR, 'weights': img_weights,'LR_path': LR_path, 'HR_path': HR_path,
                "isReal": isReal, 'wt_LL': LL.detach(), 'Ht_cat': H_cat.detach()}

    def __len__(self):
        return len(self.paths_LR)
