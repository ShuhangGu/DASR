from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import utils
import numpy as np


class Train_Deresnet_Dataset(Dataset):
    def __init__(self, noisy_dir, cleandir, crop_size, upscale_factor=4, cropped=False, flips=False, rotations=False, **kwargs):
        super(Train_Deresnet_Dataset, self).__init__()
        # get all directories used for training
        if isinstance(noisy_dir, str):
            noisy_dir = [noisy_dir]
        if isinstance(cleandir, str):
            cleandir = [cleandir]
        self.noisy_dir_files = []
        self.cleandir_files = []
        for n_dir in noisy_dir:
            self.noisy_dir_files += [join(n_dir, x) for x in listdir(n_dir) if utils.is_image_file(x)]
        for n_dir in cleandir:
            self.cleandir_files += [join(n_dir, x) for x in listdir(n_dir) if utils.is_image_file(x)]
        # intitialize image transformations and variables
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size)
        ])
        self.crop_transform = T.RandomCrop(crop_size // upscale_factor)
        self.upscale_factor = upscale_factor
        self.cropped = cropped
        self.rotations = rotations

    def __getitem__(self, index):
        # get downscaled and cropped image (if necessary)
        index_noisy, index_clean = index, np.random.randint(0, len(self.cleandir_files))
        noisy_image = self.input_transform(Image.open(self.noisy_dir_files[index_noisy]))

        clean_image = self.input_transform(Image.open(self.cleandir_files[index_clean]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            noisy_image = TF.rotate(noisy_image, angle)
            angle = random.choice([0, 90, 180, 270])
            clean_image = TF.rotate(clean_image, angle)
        if self.cropped:
            cropped_image_noisy = self.crop_transform(noisy_image)
        clean_image = TF.to_tensor(clean_image)
        resized_image = utils.imresize(clean_image, 1.0 / self.upscale_factor, True)
        # resized_image = clean_image
        if self.cropped:
            return clean_image, resized_image, TF.to_tensor(cropped_image_noisy)
        else:
            return resized_image

    def __len__(self):
        return len(self.noisy_dir_files)


class TrainDataset(Dataset):
    def __init__(self, noisy_dir, crop_size, upscale_factor=4, cropped=False, flips=False, rotations=False, **kwargs):
        super(TrainDataset, self).__init__()
        # get all directories used for training
        if isinstance(noisy_dir, str):
            noisy_dir = [noisy_dir]
        self.files = []
        for n_dir in noisy_dir:
            self.files += [join(n_dir, x) for x in listdir(n_dir) if utils.is_image_file(x)]
        # intitialize image transformations and variables
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size)
        ])
        self.crop_transform = T.RandomCrop(crop_size // upscale_factor)
        self.upscale_factor = upscale_factor
        self.cropped = cropped
        self.rotations = rotations

    def __getitem__(self, index):
        # get downscaled and cropped image (if necessary)
        noisy_image = self.input_transform(Image.open(self.files[index]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            noisy_image = TF.rotate(noisy_image, angle)
        if self.cropped:
            cropped_image = self.crop_transform(noisy_image)
        noisy_image = TF.to_tensor(noisy_image)
        resized_image = utils.imresize(noisy_image, 1.0 / self.upscale_factor, True)
        if self.cropped:
            return resized_image, TF.to_tensor(cropped_image)
        else:
            return resized_image

    def __len__(self):
        return len(self.files)


class DiscDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor=4, flips=False, rotations=False, **kwargs):
        super(DiscDataset, self).__init__()
        self.files = [join(dataset_dir, x) for x in listdir(dataset_dir) if utils.is_image_file(x)]
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size // upscale_factor)
        ])
        self.rotations = rotations

    def __getitem__(self, index):
        # get real image for discriminator (same as cropped in TrainDataset)
        image = self.input_transform(Image.open(self.files[index]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
        return TF.to_tensor(image)

    def __len__(self):
        return len(self.files)


class ValDataset(Dataset):
    def __init__(self, hr_dir, upscale_factor, lr_dir=None, crop_size_val=None, **kwargs):
        super(ValDataset, self).__init__()
        self.hr_files = [join(hr_dir, x) for x in listdir(hr_dir) if utils.is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size_val
        if lr_dir is None:
            self.lr_files = None
        else:
            self.lr_files = [join(lr_dir, x) for x in listdir(lr_dir) if utils.is_image_file(x)]

    def __getitem__(self, index):
        # get downscaled, cropped and gt (if available) image
        hr_image = Image.open(self.hr_files[index])
        w, h = hr_image.size
        cs = utils.calculate_valid_crop_size(min(w, h), self.upscale_factor)
        if self.crop_size is not None:
            cs = min(cs, self.crop_size)
        cropped_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(hr_image))
        hr_image = T.CenterCrop(cs)(hr_image)
        hr_image = TF.to_tensor(hr_image)
        resized_image = utils.imresize(hr_image, 1.0 / self.upscale_factor, True)
        if self.lr_files is None:
            return resized_image, cropped_image, resized_image
        else:
            lr_image = Image.open(self.lr_files[index])
            lr_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(lr_image))
            return resized_image, cropped_image, lr_image

    def __len__(self):
        return len(self.hr_files)


class Val_Deresnet_Dataset(Dataset):
    def __init__(self, hr_dir, upscale_factor, lr_dir=None, crop_size_val=None, **kwargs):
        super(Val_Deresnet_Dataset, self).__init__()
        self.hr_files = [join(hr_dir, x) for x in listdir(hr_dir) if utils.is_image_file(x)]
        self.hr_files.sort()
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size_val
        if lr_dir is None:
            self.lr_files = None
        else:
            self.lr_files = [join(lr_dir, x) for x in listdir(lr_dir) if utils.is_image_file(x)]
            self.lr_files.sort()
    def __getitem__(self, index):
        # get downscaled, cropped and gt (if available) image
        hr_image = Image.open(self.hr_files[index])
        w, h = hr_image.size
        cs = utils.calculate_valid_crop_size(min(w, h), self.upscale_factor)
        if self.crop_size is not None:
            cs = min(cs, self.crop_size)

        hr_image = T.CenterCrop(cs)(hr_image)
        hr_image = TF.to_tensor(hr_image)
        resized_image = utils.imresize(hr_image, 1.0 / self.upscale_factor, True)
        if self.lr_files is None:
            return resized_image, cropped_image, resized_image
        else:
            lr_image = Image.open(self.lr_files[index])

            cropped_image = TF.to_tensor(T.RandomCrop(cs // self.upscale_factor)(lr_image))
            lr_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(lr_image))
            return hr_image, resized_image, cropped_image, lr_image

    def __len__(self):
        return len(self.hr_files)
