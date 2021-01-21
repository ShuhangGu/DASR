import random
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')
from PerceptualSimilarity.models import util as ps
# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss



class PerceptualLossLPIPS(nn.Module):
    def __init__(self):
        super(PerceptualLossLPIPS, self).__init__()
        self.loss_network = ps.PerceptualLoss(use_gpu=torch.cuda.is_available())

    def forward(self, x, y):
        return self.loss_network.forward(x, y, normalize=True).mean()


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


if __name__ == '__main__':
    a = PerceptualLossLPIPS()