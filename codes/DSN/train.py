import argparse
import os
from PIL import Image
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tvutils
import data_loader as loader
import yaml
import loss
import model
import utils
import json
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

def saveimgs(img_list, img_name, savepath):
    img = img_list[0][0].cpu().numpy().transpose((1,2,0))
    img = Image.fromarray(np.uint8(img*255))
    img.save(savepath+img_name.split('.')[0]+'.png')
    print(savepath+img_name.split('.')[0]+'.png')

parser = argparse.ArgumentParser(description='Train Downscaling Models')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[1, 2, 4], help='super resolution upscale factor')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--crop_size_val', default=256, type=int, help='validation images crop size')
parser.add_argument('--batch_size', default=4, type=int, help='batch size used')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers used')
parser.add_argument('--num_epochs', default=400, type=int, help='total train epoch number')
parser.add_argument('--num_decay_epochs', default=150, type=int, help='number of epochs during which lr is decayed')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
parser.add_argument('--adam_beta_1', default=0.5, type=float, help='beta_1 for adam optimizer of gen and disc')
parser.add_argument('--val_interval', default=5, type=int, help='validation interval')
parser.add_argument('--val_img_interval', default=5, type=int, help='interval for saving validation images')
parser.add_argument('--save_model_interval', default=5, type=int, help='interval for saving the model')
parser.add_argument('--artifacts', default='tdsr', type=str, help='selecting different artifacts type')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--flips', dest='flips', action='store_true', help='if activated train images are randomly flipped')
parser.add_argument('--rotations', dest='rotations', action='store_true',
                    help='if activated train images are rotated by a random angle from {0, 90, 180, 270}')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--ragan', dest='ragan', action='store_true',
                    help='if activated then RaGAN is used instead of normal GAN')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--no_per_loss', dest='use_per_loss', action='store_false',
                    help='if activated no perceptual loss is used')
parser.add_argument('--lpips_rot_flip', dest='lpips_rot_flip', action='store_true',
                    help='if activated images are randomly flipped and rotated before being fed to lpips')
parser.add_argument('--per_type', default='LPIPS', type=str, help='selecting different Perceptual loss type')
parser.add_argument('--disc_freq', default=1, type=int, help='number of steps until a discriminator updated is made')
parser.add_argument('--gen_freq', default=1, type=int, help='number of steps until a generator updated is made')
parser.add_argument('--w_col', default=1, type=float, help='weight of color loss')
parser.add_argument('--w_tex', default=0.005, type=float, help='weight of texture loss')
parser.add_argument('--w_per', default=0.01, type=float, help='weight of perceptual loss')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to start from')
parser.add_argument('--save_path', default=None, type=str, help='additional folder for saving the data')
parser.add_argument('--generator', default='DeResnet', type=str, help='set generator architecture. (DSGAN, DeResnet)')
parser.add_argument('--discriminator', default='FSD', type=str,
                    help='set discriminator architecture. (FSD, nld_s1, nld_s2)')
parser.add_argument('--filter', default='gau', type=str, help='set filter in (gau, avg_pool, wavelet)')
parser.add_argument('--cat_or_sum', default='cat', type=str, help='set wavelet bands type in (cat, sum)')
parser.add_argument('--norm_layer', default='Instance', type=str,
                    help='set type of discriminator norm layer in (Instance, Batch)')
parser.add_argument('--no_saving', dest='saving', action='store_false',
                    help='if activated the model and results are not saved')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Activate to ensure whole pipeline works well')
opt = parser.parse_args()

# fix random seeds
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# prepare data and DataLoaders
with open('../paths.yml', 'r') as stream:
    PATHS = yaml.load(stream, Loader=yaml.FullLoader)
if opt.dataset == 'aim2019':
    train_set = loader.Train_Deresnet_Dataset(PATHS['aim2019'][opt.artifacts]['source'], PATHS['aim2019'][opt.artifacts]['target'],
                                    cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.Val_Deresnet_Dataset(PATHS['aim2019'][opt.artifacts]['valid_hr'],
                                lr_dir=PATHS['aim2019'][opt.artifacts]['valid_lr'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

elif opt.dataset == 'ntire2020':
    train_set = loader.Train_Deresnet_Dataset(PATHS['ntire2020'][opt.artifacts]['source'], PATHS['ntire2020'][opt.artifacts]['target'],
                                    cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.Val_Deresnet_Dataset(PATHS['ntire2020'][opt.artifacts]['valid_hr'],
                                lr_dir=PATHS['ntire2020'][opt.artifacts]['valid_lr'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

elif opt.dataset == 'realsr':
    train_set = loader.Train_Deresnet_Dataset(PATHS['realsr'][opt.artifacts]['source'], PATHS['realsr'][opt.artifacts]['target'],
                                    cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.Val_Deresnet_Dataset(PATHS['realsr'][opt.artifacts]['valid_hr'],
                                lr_dir=PATHS['realsr'][opt.artifacts]['valid_lr'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)


elif opt.dataset == 'camerasr':
    train_set = loader.Train_Deresnet_Dataset(PATHS['camerasr'][opt.artifacts]['source'], PATHS['aim2019'][opt.artifacts]['target'],
                                    cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.Val_Deresnet_Dataset(PATHS['camerasr'][opt.artifacts]['valid_hr'],
                                lr_dir=PATHS['camerasr'][opt.artifacts]['valid_lr'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
else:
    train_set = loader.TrainDataset(PATHS[opt.dataset][opt.artifacts]['hr']['train'], cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.ValDataset(PATHS[opt.dataset][opt.artifacts]['hr']['valid'],
                                lr_dir=PATHS[opt.dataset][opt.artifacts]['lr']['valid'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

# prepare neural networks
if opt.generator.lower() == 'dsgan':
    model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
elif opt.generator.lower() == 'deresnet':
    model_g = model.De_resnet(n_res_blocks=opt.num_res_blocks, scale=opt.upscale_factor)
else:
    raise NotImplementedError('Generator model [{:s}] not recognized'.format(opt.generator))
print('# Initializing {}'.format(opt.generator))
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()), '\n')

model_d = model.Discriminator(kernel_size=opt.kernel_size, wgan=opt.wgan, highpass=opt.highpass,
                              D_arch=opt.discriminator, norm_layer=opt.norm_layer, filter_type=opt.filter,
                              cs=opt.cat_or_sum)
print('# discriminator parameters:', sum(param.numel() for param in model_d.parameters()))


g_loss_module = loss.GeneratorLoss(**vars(opt))

# filters are used for generating validation images
filter_low_module = model.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.filter == 'gau', include_pad=False)
filter_high_module = model.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.filter == 'gau', include_pad=False)
print('# FS type: {}, kernel size={}'.format(opt.filter, opt.kernel_size))
if torch.cuda.is_available():
    model_g = model_g.cuda()
    model_d = model_d.cuda()
    filter_low_module = filter_low_module.cuda()
    filter_high_module = filter_high_module.cuda()

# define optimizers
optimizer_g = optim.Adam(model_g.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
optimizer_d = optim.Adam(model_d.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
start_decay = opt.num_epochs - opt.num_decay_epochs
scheduler_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / opt.num_decay_epochs)
scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=scheduler_rule)
scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=scheduler_rule)

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    model_d.load_state_dict(checkpoint['models_d_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    print('Continuing training at epoch %d' % start_epoch)
else:
    start_epoch = 1
    iteration = 1

# prepare tensorboard summary
summary_path = ''
if opt.saving:
    if opt.save_path is None:
        save_path = ''
    else:
        save_path = os.path.join('../../DSN_experiments', opt.save_path)
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    dir_index = 0

    summary_path = os.path.join('../../DSN_tb_logger', opt.save_path)
    while os.path.isdir(os.path.join(summary_path, str(dir_index))):
        dir_index += 1
    summary_path = os.path.join(summary_path, str(dir_index))
    writer = SummaryWriter(summary_path)
    print('Saving summary into directory ' + summary_path + '/')

    ## save args
    args_path = os.path.join(save_path, 'commandline_args.txt')
    with open(args_path, 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

# training iteration
for epoch in range(start_epoch, opt.num_epochs + 1):
    train_bar = tqdm(train_loader, desc='[%d/%d]' % (epoch, opt.num_epochs))
    model_g.train()
    model_d.train()

    for input_img, bicubic_img, disc_img in train_bar:

        iteration += 1
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            disc_img = disc_img.cuda()
            bicubic_img = bicubic_img.cuda()

        # Estimate scores of fake and real images
        if opt.generator == 'DSGAN':
            input_img = bicubic_img
            fake_img = model_g(bicubic_img)
        elif opt.generator == 'DeResnet':
            fake_img = model_g(input_img)
        else:
            raise NotImplementedError('Generator model [{:s}] not recognized'.format(opt.generator))

        if opt.ragan:
            real_tex = model_d(disc_img, fake_img)
            fake_tex = model_d(fake_img, disc_img)
        else:
            real_tex = model_d(disc_img)
            fake_tex = model_d(fake_img)

        # Update Discriminator network
        if iteration % opt.disc_freq == 0:
            # calculate gradient penalty
            if opt.wgan:
                rand = torch.rand(1).item()
                sample = rand * disc_img + (1 - rand) * fake_img
                gp_tex = model_d(sample)
                gradient = torch.autograd.grad(gp_tex.mean(), sample, create_graph=True)[0]
                grad_pen = 10 * (gradient.norm() - 1) ** 2
            else:
                grad_pen = None
            # update discriminator
            model_d.zero_grad()
            d_tex_loss = loss.discriminator_loss(real_tex, fake_tex, wasserstein=opt.wgan, grad_penalties=grad_pen)
            d_tex_loss.backward(retain_graph=True)
            optimizer_d.step()
            # save data to tensorboard
            if opt.saving:
                writer.add_scalar('loss/d_tex_loss', d_tex_loss, iteration)
                if opt.wgan:
                    writer.add_scalar('disc_score/gradient_penalty', grad_pen.mean().data.item(), iteration)

        # Update Generator network
        if iteration % opt.gen_freq == 0:
            # update discriminator
            model_g.zero_grad()
            if opt.generator == 'DSGAN':
                g_loss = g_loss_module(fake_tex, fake_img, input_img)
            elif opt.generator == 'DeResnet':
                g_loss = g_loss_module(fake_tex, fake_img, bicubic_img)
            else:
                raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(opt.D))


            assert not torch.isnan(g_loss), 'Generator loss returns NaN values'
            g_loss.backward()
            optimizer_g.step()
            # save data to tensorboard
            if opt.saving:
                writer.add_scalar('loss/perceptual_loss', g_loss_module.last_per_loss, iteration)
                writer.add_scalar('loss/color_loss', g_loss_module.last_col_loss, iteration)
                writer.add_scalar('loss/g_tex_loss', g_loss_module.last_tex_loss, iteration)
                writer.add_scalar('loss/g_overall_loss', g_loss, iteration)

        # save data to tensorboard
        rgb_loss = g_loss_module.rgb_loss(fake_img, input_img)
        mean_loss = g_loss_module.mean_loss(fake_img, input_img)
        if opt.saving:
            writer.add_scalar('loss/rgb_loss', rgb_loss, iteration)
            writer.add_scalar('loss/mean_loss', mean_loss, iteration)
            writer.add_scalar('disc_score/real', real_tex.mean().data.item(), iteration)
            writer.add_scalar('disc_score/fake', fake_tex.mean().data.item(), iteration)
        train_bar.set_description(desc='[%d/%d]' % (epoch, opt.num_epochs))
        if opt.debug:
            opt.val_interval = 1
            opt.save_model_interval = 1
            opt.val_img_interval = 1
            break

    scheduler_d.step()
    scheduler_g.step()
    if opt.saving:
        writer.add_scalar('param/learning_rate', torch.Tensor(scheduler_g.get_lr()), epoch)



    # validation step
    if epoch % opt.val_interval == 0 or epoch % opt.val_img_interval == 0:
        val_bar = tqdm(val_loader, desc='[Validation]')
        model_g.eval()
        val_images = []
        with torch.no_grad():
            # initialize variables to estimate averages
            mse_sum = psnr_sum = rgb_loss_sum = mean_loss_sum = 0
            per_loss_sum = col_loss_sum = tex_loss_sum = 0

            # validate on each image in the val dataset
            for index, (hr_img, bicubic_img, disc_img, target_img) in enumerate(val_bar):
                if opt.generator == 'DSGAN':
                    input_img = bicubic_img
                elif opt.generator == 'DeResnet':
                    input_img = hr_img
                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    bicubic_img = bicubic_img.cuda()
                    target_img = target_img.cuda()
                fake_img = torch.clamp(model_g(input_img), min=0, max=1)

                mse = ((fake_img - target_img) ** 2).mean().data
                mse_sum += mse
                psnr_sum += -10 * torch.log10(mse)
                rgb_loss_sum += g_loss_module.rgb_loss(fake_img, target_img)
                mean_loss_sum += g_loss_module.mean_loss(fake_img, target_img)
                per_loss_sum += g_loss_module.perceptual_loss(fake_img, target_img)
                col_loss_sum += g_loss_module.color_loss(fake_img, target_img)

                # generate images
                if epoch % opt.val_img_interval == 0 and epoch != 0:
                    blur = filter_low_module(fake_img)
                    hf = filter_high_module(fake_img)
                    val_image_list = [
                        utils.display_transform()(target_img.data.cpu().squeeze(0)),
                        utils.display_transform()(fake_img.data.cpu().squeeze(0)),
                        utils.display_transform()(disc_img.squeeze(0)),
                        utils.display_transform()(blur.data.cpu().squeeze(0)),
                        utils.display_transform()(hf.data.cpu().squeeze(0))]
                    n_val_images = len(val_image_list)
                    val_images.extend(val_image_list)

            if opt.saving and len(val_loader) > 0:
                # save validation values
                writer.add_scalar('val/mse', mse_sum/len(val_set), iteration)
                writer.add_scalar('val/psnr', psnr_sum / len(val_set), iteration)
                writer.add_scalar('val/rgb_error', rgb_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/mean_error', mean_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/perceptual_error', per_loss_sum / len(val_set), iteration)
                writer.add_scalar('val/color_error', col_loss_sum / len(val_set), iteration)

                # save image results
                if epoch % opt.val_img_interval == 0 and epoch != 0:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // (n_val_images * 5))
                    val_save_bar = tqdm(val_images, desc='[Saving results]')
                    for index, image in enumerate(val_save_bar):
                        image = tvutils.make_grid(image, nrow=n_val_images, padding=5)
                        out_path = 'val/target_fake_tex_disc_f-wav_t-wav_' + str(index)
                        writer.add_image('val/target_fake_crop_low_high_' + str(index), image, iteration)

    # save model parameters
    if opt.saving and epoch % opt.save_model_interval == 0 and epoch != 0:
        path = os.path.join(save_path, 'checkpoints', 'iteration_{}.tar'.format(iteration))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        state_dict = {
            'epoch': epoch,
            'iteration': iteration,
            'fs_type': opt.filter,
            'fs_kernel_size': opt.kernel_size,
            'D_type': opt.discriminator,
            'model_g_state_dict': model_g.state_dict(),
            'models_d_state_dict': model_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
        }
        torch.save(state_dict, path)
        path = os.path.join(save_path, 'checkpoints', 'last_iteration.tar')
        torch.save(state_dict, path)
