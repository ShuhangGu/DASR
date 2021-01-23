import argparse
import os
import torch.utils.data
import yaml
import model
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--track', default='train', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[4,2], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

if opt.dataset == 'aim2019':
    path_sdsr = PATHS['datasets']['aim2019'] + '/generated/sdsr/'
    path_tdsr = PATHS['datasets']['aim2019'] + '/generated/tdsr/'
    input_source_dir = PATHS['aim2019']['tdsr']['source']
    # input_target_dir = PATHS['aim2019']['tdsr']['target']
    input_target_dir = "/media/4T/Dizzy/AIM/AIM_datasets/DIV2K_trainval_weights/HR"
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
else:
    path_sdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_sdsr/'
    path_tdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_tdsr/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = []

sdsr_hr_dir = path_sdsr + 'HR'
sdsr_lr_dir = path_sdsr + 'LR'
tdsr_hr_dir = path_tdsr + 'HR'
tdsr_lr_dir = path_tdsr + 'LR'

if not os.path.exists(sdsr_hr_dir):
    os.makedirs(sdsr_hr_dir)
if not os.path.exists(sdsr_lr_dir):
    os.makedirs(sdsr_lr_dir)
if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

# prepare neural networks
model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
model_g = model_g.eval()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))
if torch.cuda.is_available():
    model_g = model_g.cuda()

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    epoch = checkpoint['epoch']
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    print('Using model at epoch %d' % epoch)
else:
    print('Use --checkpoint to define the model parameters used')
    exit()

# generate the noisy images
smallest_size = 1000000000
with torch.no_grad():
    # for file in tqdm(source_files, desc='Generating images from source'):
    #     # load HR image
    #     input_img = Image.open(file)
    #     input_img = TF.to_tensor(input_img)
    #
    #     # Save HR image as HR image for SDSR
    #     path = os.path.join(sdsr_hr_dir, os.path.basename(file))
    #     TF.to_pil_image(input_img).save(path, 'PNG')
    #
    #     # Resize HR image, run model and save it as LR image for SDSR
    #     resize1_img = utils.imresize(input_img, 1.0 / opt.upscale_factor, True)
    #     if torch.cuda.is_available():
    #         resize1_img = resize1_img.unsqueeze(0).cuda()
    #     resize1_noisy_img = model_g(resize1_img).squeeze(0).cpu()
    #     path = os.path.join(sdsr_lr_dir, os.path.basename(file))
    #     TF.to_pil_image(resize1_noisy_img).save(path, 'PNG')
    #
    #     # Resize HR image to clean it up and make sure it can be resized again
    #     resize2_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)
    #     _, w, h = resize2_img.size()
    #     w = w - w % opt.upscale_factor
    #     h = h - h % opt.upscale_factor
    #     resize2_cut_img = resize2_img[:, :w, :h]
    #
    #     # Save resize2_cut_img as HR image for TDSR
    #     path = os.path.join(tdsr_hr_dir, os.path.basename(file))
    #     TF.to_pil_image(resize2_cut_img).save(path, 'PNG')
    #
    #     # Generate resize3_cut_img and apply model
    #     resize3_cut_img = utils.imresize(resize2_cut_img, 1.0 / opt.upscale_factor, True)
    #     if torch.cuda.is_available():
    #         resize3_cut_img = resize3_cut_img.unsqueeze(0).cuda()
    #     resize3_cut_noisy_img = model_g(resize3_cut_img).squeeze(0).cpu()
    #
    #     # Save resize3_cut_noisy_img as LR image for TDSR
    #     path = os.path.join(tdsr_lr_dir, os.path.basename(file))
    #     TF.to_pil_image(resize3_cut_noisy_img).save(path, 'PNG')

    for file in tqdm(target_files, desc='Generating images from target'):
        # load HR image
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        # Save input_img as HR image for TDSR
        path = os.path.join(tdsr_hr_dir, os.path.basename(file))
        TF.to_pil_image(input_img).save(path, 'PNG')

        # generate resized version of input_img
        resize_img = utils.imresize(input_img, 1.0 / opt.upscale_factor, True)

        # Apply model to input_img and resize_img
        if torch.cuda.is_available():
            input_img = input_img.unsqueeze(0).cuda()
            resize_img = resize_img.unsqueeze(0).cuda()
        input_noisy_img = model_g(input_img).squeeze(0).cpu()
        resize_noisy_img = model_g(resize_img).squeeze(0).cpu()

        # Save input_noisy_img as HR image for SDSR
        path = os.path.join(sdsr_hr_dir, os.path.basename(file))
        TF.to_pil_image(input_noisy_img).save(path, 'PNG')

        # Save resize_noisy_img as HR image for SDSR and TDSR
        path = os.path.join(sdsr_lr_dir, os.path.basename(file))
        TF.to_pil_image(resize_noisy_img).save(path, 'PNG')
        path = os.path.join(tdsr_lr_dir, os.path.basename(file))
        TF.to_pil_image(resize_noisy_img).save(path, 'PNG')
