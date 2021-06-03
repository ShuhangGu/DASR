import shutil
import argparse
import os
import torch.utils.data
import yaml
import model
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from receptive_cal import *
import numpy as np

def domain_distance_map_handler(fake_img, D_out, convnet, fs_type):
    if fs_type.lower() == 'gau' or fs_type == 'avg_pool':
        ddm_shape = (fake_img.shape[0], 1, fake_img.shape[2], fake_img.shape[3])
    elif fs_type.lower() == 'wavelet':
        ddm_shape = (fake_img.shape[0], 1, fake_img.shape[2] // 2, fake_img.shape[3] // 2)
    else:
        raise NotImplementedError('Frequency Separation [{:s}] not recognized'.format(fs_type))
    ddm = torch.zeros(ddm_shape)
    currentLayer_h, currentLayer_w = receptive_cal(ddm.shape[2], convnet), receptive_cal(ddm.shape[3], convnet)
    ddm = getWeights(D_out, ddm, currentLayer_h, currentLayer_w)
    return ddm


parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--generator', default='DeResnet', type=str, help='Generator model to use')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--discriminator', default='FSD', type=str, help='Discriminator model to use')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--filter', default='gau', type=str, help='set filter')
parser.add_argument('--cat_or_sum', default='cat', type=str, help='set wavelet bands type')
parser.add_argument('--norm_layer', default='Instance', type=str, help='set type of discriminator norm layer')
parser.add_argument('--artifacts', default='tdsr', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='0603_DSN_LRs', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='aim2019', type=str, help='selecting different datasets')
parser.add_argument('--including_source_ddm', dest='including_source_ddm', action='store_true', help='generate ddm from '
                                                                                                     'source images')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4, 1, 2], help='super resolution upscale factor')

opt = parser.parse_args()

# define input and target directories
with open('../paths.yml', 'r') as stream:
    PATHS = yaml.load(stream, Loader=yaml.FullLoader)

if opt.dataset == 'aim2019':
    input_source_dir = PATHS['aim2019']['tdsr']['source']
    input_target_dir = PATHS['aim2019']['tdsr']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'ntire2020':
    input_source_dir = PATHS['ntire2020']['tdsr']['source']
    input_target_dir = PATHS['ntire2020']['tdsr']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'realsr_tddiv2k':
    input_source_dir = PATHS['realsr']['tddiv2k']['source']
    input_target_dir = PATHS['realsr']['tddiv2k']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'realsr_tdrealsr':
    input_source_dir = PATHS['realsr']['tdrealsr']['source']
    input_target_dir = PATHS['realsr']['tdrealsr']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'realsr_tdrealsr_2x':
    input_source_dir = PATHS['realsr']['tdrealsr_x2']['source']
    input_target_dir = PATHS['realsr']['tdrealsr_x2']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'camerasr':
    input_source_dir = PATHS['camerasr']['tdsr']['source']
    input_target_dir = PATHS['camerasr']['tdsr']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]



tdsr_lr_dir = '../../DSN_results/' + opt.name
tdsr_lr_img_dir = os.path.join(tdsr_lr_dir, 'imgs_from_target')
tdsr_lr_ddm_t_dir = os.path.join(tdsr_lr_dir, 'ddm_target')
tdsr_lr_ddm_s_dir = os.path.join(tdsr_lr_dir, 'ddm_source')

if not os.path.exists(tdsr_lr_img_dir):
    os.makedirs(tdsr_lr_img_dir)
if not os.path.exists(tdsr_lr_ddm_s_dir):
    os.makedirs(tdsr_lr_ddm_s_dir)
if not os.path.exists(tdsr_lr_ddm_t_dir):
    os.makedirs(tdsr_lr_ddm_t_dir)

# prepare neural networks
if opt.generator == 'DSGAN':
    model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
elif opt.generator == 'DeResnet':
    model_g = model.De_resnet(n_res_blocks=opt.num_res_blocks, scale=opt.upscale_factor)
else:
    raise NotImplementedError('Generator model [{:s}] not recognized'.format(opt.generator))


model_d = model.Discriminator(kernel_size=opt.kernel_size, wgan=opt.wgan, highpass=opt.highpass,
                              D_arch=opt.discriminator, norm_layer=opt.norm_layer, filter_type=opt.filter,
                              cs=opt.cat_or_sum)


# convnet = [[conv_layer], [conv_layer], [conv_layer], ...]
# conv_layer = [kernel_size, stride, padding]
if opt.discriminator == 'FSD':
    convnet = [[5, 1, 2], [5, 1, 2], [5, 1, 2], [5, 1, 2]]
elif opt.discriminator == 'nld_s1':
    convnet = [[4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1]]
elif opt.discriminator == 'nld_s2':
    convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
else:
    raise NotImplementedError('Please specified conv_net of discriminator.')

model_g = model_g.eval()
model_d = model_d.eval()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))
if torch.cuda.is_available():
    model_g = model_g.cuda()
    model_d = model_d.cuda()

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    epoch = checkpoint['epoch']
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    model_d.load_state_dict(checkpoint['models_d_state_dict'])
    print('Using model at epoch %d' % epoch)
else:
    print('Use --checkpoint to define the model parameters used')
    exit()

if opt.checkpoint is not None:
    shutil.copyfile(opt.checkpoint, os.path.join(tdsr_lr_dir, opt.name+'.tar'))
    print('Copying {} to {}'.format(opt.checkpoint, os.path.join(tdsr_lr_dir, opt.name+'.tar')))

# generate the noisy images
smallest_size = 1000000000
with torch.no_grad():
    for file in tqdm(target_files, desc='Generating images from target'):
        # load HR image
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        # Apply model to input_img
        if torch.cuda.is_available():
            input_img = input_img.unsqueeze(0).cuda()
        fake_img = model_g(input_img)

        D_out = model_d(fake_img).cpu().detach().numpy()
        ddm = domain_distance_map_handler(fake_img, D_out, convnet, opt.filter)
        # Save input_noisy_img as HR image for TDSR
        fake_img = fake_img.squeeze(0).cpu()
        path = os.path.join(tdsr_lr_img_dir, os.path.basename(file))
        TF.to_pil_image(fake_img).save(path, 'PNG')
        np.save(os.path.join(tdsr_lr_ddm_t_dir, os.path.basename(file).split('.')[0]), ddm)

if opt.including_source_ddm:
    with torch.no_grad():
        for file in tqdm(source_files, desc='Generating images from source files'):
            # load HR image
            input_img = Image.open(file)
            input_img = TF.to_tensor(input_img)

            if torch.cuda.is_available():
                input_img = input_img.unsqueeze(0).cuda()

            D_out = model_d(input_img).cpu().detach().numpy()
            ddm = domain_distance_map_handler(input_img, D_out, convnet, opt.filter)
            np.save(os.path.join(tdsr_lr_ddm_s_dir, os.path.basename(file).split('.')[0]), ddm)