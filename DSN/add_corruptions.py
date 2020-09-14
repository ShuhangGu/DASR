import os
import argparse
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
import yaml
import utils


# Standard parameters:
#   gaussian noise: 8.0
#   gaussian blur: 1.6
#   jpg: 30

parser = argparse.ArgumentParser(description='Generate noisy images')
parser.add_argument('--noise_std_dev', default=8.0, type=float, help='standard deviation of the added gaussian noise')
parser.add_argument('--blur_std_dev', default=None, type=float, help='standard deviation of the added gaussian blur')
parser.add_argument('--jpg_quality', default=30, type=int, help='quality used for jpeg compression')
parser.add_argument('--artifacts', default='gaussian', type=str, help='selecting different artifacts type')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--resolution', default='hr', type=str, choices=['hr', 'lr'], help='choose whether to use HR or LR')
opt = parser.parse_args()

# define input and target directories
with open('paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

input_dirs = [PATHS[opt.dataset]['clean'][opt.resolution]['train'],
              PATHS[opt.dataset]['clean'][opt.resolution]['valid']]
target_dirs = [PATHS[opt.dataset][opt.artifacts][opt.resolution]['train'],
               PATHS[opt.dataset][opt.artifacts][opt.resolution]['valid']]

for index in range(len(target_dirs)):
    if not os.path.exists(target_dirs[index]):
        os.makedirs(target_dirs[index])

# loop over all input directories to generate the noisy images
for index, input_dir in enumerate(input_dirs):
    print('\nComputing images from directory ' + input_dir)
    files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if utils.is_image_file(x)]
    for file in tqdm(files):
        # create noisy HR image
        image = Image.open(file)
        path = os.path.join(target_dirs[index], os.path.basename(file))
        if opt.noise_std_dev is not None:
            image = utils.gaussian_noise(image, opt.noise_std_dev)
        if opt.blur_std_dev is not None:
            image = image.filter(ImageFilter.GaussianBlur(opt.blur_std_dev))
        if opt.jpg_quality is not None:
            path = os.path.join(target_dirs[index], os.path.basename(file))
            jpeg_path = os.path.splitext(path)[0] + '.jpg'
            image.save(jpeg_path, 'JPEG', quality=opt.jpg_quality)
            image = Image.open(jpeg_path)
            os.remove(jpeg_path)

        image.save(path, 'PNG')
