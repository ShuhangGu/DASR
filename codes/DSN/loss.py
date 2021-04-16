import torch
import random
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from model import FilterLow
import sys
sys.path.insert(0, '../')
import PerceptualSimilarity.models.util as ps
from pytorch_wavelets import DWTForward

def generator_loss(labels, wasserstein=False, weights=None):
    if not isinstance(labels, list):
        labels = (labels,)
    if weights is None:
        weights = [1.0 / len(labels)] * len(labels)
    loss = 0.0
    for label, weight in zip(labels, weights):
        if wasserstein:
            loss += weight * torch.mean(-label)
        else:
            loss += weight * torch.mean(-torch.log(label + 1e-8))
    return loss


def discriminator_loss(reals, fakes, wasserstein=False, grad_penalties=None, weights=None):
    if not isinstance(reals, list):
        reals = (reals,)
    if not isinstance(fakes, list):
        fakes = (fakes,)
    if weights is None:
        weights = [1.0 / len(fakes)] * len(fakes)
    loss = 0.0
    if wasserstein:
        if not isinstance(grad_penalties, list):
            grad_penalties = (grad_penalties,)
        for real, fake, weight, grad_penalty in zip(reals, fakes, weights, grad_penalties):
            loss += weight * (-real.mean() + fake.mean() + grad_penalty)
    else:
        for real, fake, weight in zip(reals, fakes, weights):
            loss += weight * (-torch.log(real + 1e-8).mean() - torch.log(1 - fake + 1e-8).mean())
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, use_perceptual_loss=True, wgan=False, w_col=1,
                 w_tex=0.001, w_per=0.1, gaussian=False, lpips_rot_flip=False, **kwargs):
        super(GeneratorLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.per_type = kwargs['per_type']
        if kwargs['filter'].lower() == 'gau':
            self.color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,
                                      gaussian=True)
        elif kwargs['filter'].lower() == 'avg_pool':
            self.color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,
                                          gaussian=False)
        elif kwargs['filter'].lower() == 'wavelet':
            self.color_filter = self.filter_wavelet_LL
        else:
            raise NotImplementedError('Frequency Separation type [{:s}] not recognized'.format(kwargs['filter']))

        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()
        if isinstance(self.color_filter, FilterLow):
            self.color_filter = self.color_filter.cuda()
        if self.per_type == 'LPIPS':
            self.perceptual_loss = PerceptualLoss(rotations=lpips_rot_flip, flips=lpips_rot_flip)
        elif self.per_type == 'VGG':
            self.perceptual_loss = PerceptualLossVGG16()
        else:
            raise NotImplemented('{} is not recognized'.format(self.per_type))
        self.use_perceptual_loss = use_perceptual_loss
        self.wasserstein = wgan
        self.w_col = w_col
        self.w_tex = w_tex
        self.w_per = w_per
        self.last_tex_loss = 0
        self.last_per_loss = 0
        self.last_col_loss = 0
        self.gaussian = gaussian
        self.last_mean_loss = 0

    def forward(self, tex_labels, out_images, target_images):
        # Adversarial Texture Loss
        self.last_tex_loss = generator_loss(tex_labels, wasserstein=self.wasserstein)
        # Perception Loss
        self.last_per_loss = self.perceptual_loss(out_images, target_images)
        # Color Loss
        self.last_col_loss = self.color_loss(out_images, target_images)
        loss = self.w_col * self.last_col_loss + self.w_tex * self.last_tex_loss
        if self.use_perceptual_loss:
            loss += self.w_per * self.last_per_loss
        return loss

    def color_loss(self, x, y):
        return self.pixel_loss(self.color_filter(x), self.color_filter(y))

    def rgb_loss(self, x, y):
        return self.pixel_loss(x.mean(3).mean(2), y.mean(3).mean(2))

    def mean_loss(self, x, y):
        return self.pixel_loss(x.view(x.size(0), -1).mean(1), y.view(y.size(0), -1).mean(1))

    def filter_wavelet_LL(self, x, norm=True):
        DWT2 = DWTForward(J=1, wave='haar', mode='reflect').cuda()
        LL, Hc = DWT2(x)

        return LL * 0.5 if norm else LL


class PerceptualLossLPIPS(nn.Module):
    def __init__(self):
        super(PerceptualLossLPIPS, self).__init__()
        self.loss_network = ps.PerceptualLoss(use_gpu=torch.cuda.is_available())

    def forward(self, x, y):
        return self.loss_network.forward(x, y, normalize=True).mean()


class PerceptualLossVGG16(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG16, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))


class PerceptualLossVGG19(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        if torch.cuda.is_available():
            loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))


class PerceptualLoss(nn.Module):
    def __init__(self, rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()
        self.loss = PerceptualLossLPIPS()
        self.rotations = rotations
        self.flips = flips

    def forward(self, x, y):
        if self.rotations:
            k_rot = random.choice([-1, 0, 1])
            x = torch.rot90(x, k_rot, [2, 3])
            y = torch.rot90(y, k_rot, [2, 3])
        if self.flips:
            if random.choice([True, False]):
                x = torch.flip(x, (2,))
                y = torch.flip(y, (2,))
            if random.choice([True, False]):
                x = torch.flip(x, (3,))
                y = torch.flip(y, (3,))
        return self.loss(x, y)
