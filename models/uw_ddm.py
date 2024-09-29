import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet

import torch.utils.tensorboard as tb 

from torchview import draw_graph


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm




# class ANet(nn.Module):
#     def __init__(self):
#         super(ANet,self).__init__()

#         block = [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.AdaptiveAvgPool2d((1,1))]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()] #[nn.Sigmoid()] #[nn.PReLU()]
#         self.block = nn.Sequential(*block)

#     def forward(self,x):
#         return self.block(x)

class ANet(nn.Module):
    def __init__(self):
        super(ANet,self).__init__()

        self.conv1 = nn.Conv2d(3,3,3,padding = 1)
        self.activation1 = nn.PReLU()

        self.conv2 = nn.Conv2d(3,3,3,padding = 1)
        self.activation2 = nn.PReLU()

        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_1 = nn.Conv2d(3,3,3,padding = 1)
        self.activation2_1 = nn.PReLU()

        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_2 = nn.Conv2d(3,3,3,padding = 1)
        self.activation2_2 = nn.PReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(3,3,1)
        self.activation3 = nn.PReLU()
        self.conv4 = nn.Conv2d(3,3,1)
        self.activation4 = nn.PReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.activation2(out)

        out = self.max_pool1(out)
        out = self.conv2_1(out)
        out = self.activation2_1(out)

        out = self.max_pool2(out)
        out = self.conv2_2(out)
        out = self.activation2_2(out)

        out = self.avgpool(out)
        out = self.conv3(out)
        out = self.activation3(out)
        out = self.conv4(out)
        out = self.activation4(out)
        # breakpoint()
        return out


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()

        # self.p = 0.25

        # in-branch
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding='same')
        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, padding='same')
        self.normalize1 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation1 = nn.PReLU() #nn.Sigmoid()
        # self.dropout1 = nn.Dropout(p=self.p)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
        self.normalize2 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation2 = nn.PReLU() #nn.Sigmoid()
        # self.dropout2 = nn.Dropout(p=self.p)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.normalize3 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation3 = nn.PReLU() #nn.Sigmoid()
        # self.dropout3 = nn.Dropout(p=self.p)
        self.conv3_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.activation3_ = nn.PReLU() #nn.Sigmoid()

        # sub-branch 1
        self.max_pool2 = nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
        self.normalize4 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation4 = nn.PReLU() #nn.Sigmoid()
        # self.dropout4 = nn.Dropout(p=self.p)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.normalize5 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation5 = nn.PReLU() #nn.Sigmoid()
        self.conv5_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.activation5_ = nn.PReLU() #nn.Sigmoid()
        # self.dropout5 = nn.Dropout(p=self.p)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        self.normalize6 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
        self.activation6 = nn.PReLU() #nn.Sigmoid()
        # self.dropout6 = nn.Dropout(p=self.p)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)

        # sub-branch 2
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv7 = nn.Conv2d(in_channels=17, out_channels=16, kernel_size=5, padding='same')
        self.normalize7 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation7 = nn.PReLU() #nn.Sigmoid()
        # self.dropout7 = nn.Dropout(p=self.p)
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.normalize8 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation8 = nn.PReLU() #nn.Sigmoid()
        self.conv8_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.activation8_ = nn.PReLU() #nn.Sigmoid()
        # self.dropout8 = nn.Dropout(p=self.p)
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        self.normalize9 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
        self.activation9 = nn.PReLU() #nn.Sigmoid()
        # self.dropout9 = nn.Dropout(p=self.p)
        self.deconv2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)

        # out-branch
        self.conv10 = nn.Conv2d(in_channels=17, out_channels=16, kernel_size=3, padding='same')
        self.normalize10 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation10 = nn.PReLU() #nn.Sigmoid()
        # self.dropout10 = nn.Dropout(p=self.p)
        self.conv10_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.activation10_ = nn.PReLU() #nn.Sigmoid()
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.normalize11 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
        self.activation11 = nn.PReLU() #nn.Sigmoid()
        # self.dropout11 = nn.Dropout(p=self.p)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding='same')
        self.normalize12 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
        self.activation12 = nn.PReLU() #Sigmoid()
        # self.dropout12 = nn.Dropout(p=self.p)
    
    def forward(self, x):
        # in-branch
        out = self.conv1(x)
        # out = self.normalize1(out)
        out_activation1 = self.activation1(out)
        # out_activation1 = self.dropout1(out_activation1)
        out = self.max_pool1(out_activation1)
        out = self.conv2(out)
        # out = self.normalize2(out)
        out = self.activation2(out)
        # out = self.dropout2(out)
        out = self.conv3(out)
        # out = self.normalize3(out)
        out_conv3 = self.activation3(out)
        # out_conv3 = self.dropout3(out_conv3)
        out_conv3 = self.conv3_(out_conv3)
        out_conv3 = self.activation3_(out_conv3)

        # sub-branch 1
        out = self.max_pool2(out_conv3)
        out = self.conv4(out)
        # out = self.normalize4(out)
        out = self.activation4(out)
        # out = self.dropout4(out)
        out = self.conv5(out)
        # out = self.normalize5(out)
        out = self.activation5(out)
        # out = self.dropout5(out)
        out = self.conv5_(out)
        out = self.activation5_(out)
        out = self.conv6(out)
        # out = self.normalize6(out)
        out = self.activation6(out)
        # out = self.dropout6(out)
        out_deconv1 = self.deconv1(out)

        # sub-branch 2
        out = self.max_pool3(out_conv3)
        out = torch.cat((out_deconv1, out), dim=1)
        out = self.conv7(out)
        # out = self.normalize7(out)
        out = self.activation7(out)
        # out = self.dropout7(out)
        out = self.conv8(out)
        # out = self.normalize8(out)
        out = self.activation8(out)
        # out = self.dropout8(out)
        out = self.conv8_(out)
        out = self.activation8_(out)
        out = self.conv9(out)
        # out = self.normalize9(out)
        out = self.activation9(out)
        # out = self.dropout9(out)
        out_deconv2 = self.deconv2(out)

        # out-branch
        out = torch.cat((out_deconv2, out_conv3), dim=1)
        out = self.conv10(out)
        # out = self.normalize10(out)
        out = self.activation10(out)
        out = self.conv10_(out)
        out = self.activation10_(out)
        # out = self.dropout10(out)
        out = self.deconv3(out)
        out = torch.cat((out, out_activation1), dim=1)
        out = self.conv11(out)
        # out = self.normalize11(out)
        out = self.activation11(out)
        # out = self.dropout11(out)
        out = self.conv12(out)
        # out = self.normalize12(out)
        out = self.activation12(out)
        # out = self.dropout12(out)

        if (torch.unique(torch.isnan(out)) == torch.tensor([True], device='cuda:0')).item():
            breakpoint()

        return out


class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN,self).__init__()

        self.ANet = ANet()
        self.tNet = TNet()

    def forward(self,x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x*0+A,x),1))
        out = ((x-A)*t + A)
        # breakpoint()
        return out #torch.clamp(out,0.,1.)


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(criterion, model_theta, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model_theta(torch.cat([x0, x], dim=1), t.float())
    return criterion(e, output) #(e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# def transformation_loss(model_phi, x0, y0, t, e, b):
#     a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
#     x = x0* a.sqrt() + e * (1.0 - a).sqrt()
#     y = y0* a.sqrt() + e * (1.0 - a).sqrt()
#     y_ = model_phi(x, t.float())
#     return (y - y_).square().sum(dim=(1, 2, 3)).mean(dim=0)

def transformation_loss(criterion, model_theta, model_phi, x0, y0, t, e, b):
    # A-net: (in: x_0 -> out: A)
    # T-net: (in: x_0, A -> out: T)
    # diffusion underwater physical model: (in: x_0, A, T, t, e, b -> out: y_t)
    
    # breakpoint()
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    e_ = model_theta(torch.cat([x0, x], dim=1), t.float())

    # x0_ = (x - (1.0-a).sqrt()*e_) / a.sqrt()
    y0_ = model_phi(x0)
    # y0_ = model_phi(x0_)
    # y0_ = model_phi(x0, a)
    y = y0 * a.sqrt() + e * (1.0 - a).sqrt()
    y_ = y0_ * a.sqrt() + e_ * (1.0 - a).sqrt() ## existing model_theta -> fine-tune model_theta ?
    # breakpoint()
    return criterion(y0, y0_) + criterion(y, y_), criterion(y, y_) #criterion(y0, y0_) + criterion(y, y_) #(y0 - y0_).square().sum(dim=(1, 2, 3)).mean(dim=0) + (y - y_).square().sum(dim=(1, 2, 3)).mean(dim=0)

    # y0_ = model_phi(x0)

    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # y =  a.sqrt() * y0 + e * (1.0 - a).sqrt()
    # y_ = a.sqrt() * y0_ + e * (1.0 - a).sqrt()

    # return (y - y_).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusionUWPhysical(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model_theta = DiffusionUNet(config, conditional=True)

        self.model_theta.to(self.device)
        self.model_theta = torch.nn.DataParallel(self.model_theta)
        self.ema_helper_theta = EMAHelper()
        self.ema_helper_theta.register(self.model_theta)
        self.optimizer_theta = utils.optimize.get_optimizer(self.config, self.model_theta.parameters())

        self.model_phi = PhysicalNN()
        self.model_phi.to(self.device)
        self.model_phi = torch.nn.DataParallel(self.model_phi)
        self.ema_helper_phi = EMAHelper()
        self.ema_helper_phi.register(self.model_phi)
        self.optimizer_phi = torch.optim.Adam(self.model_phi.parameters(), lr=0.0001)
        
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        self.tb_logger = tb.SummaryWriter(log_dir=config.tb_path)


    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']

        self.model_theta.load_state_dict(checkpoint['state_dict_theta'], strict=True)
        self.optimizer_theta.load_state_dict(checkpoint['optimizer_theta'])
        self.ema_helper_theta.load_state_dict(checkpoint['ema_helper_theta'])

        self.model_phi.load_state_dict(checkpoint['state_dict_phi'], strict=True)
        self.optimizer_phi.load_state_dict(checkpoint['optimizer_phi'])
        self.ema_helper_phi.load_state_dict(checkpoint['ema_helper_phi'])

        if ema:
            self.ema_helper_theta.ema(self.model_theta)
            self.ema_helper_phi.ema(self.model_phi)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], checkpoint['step']))
    
    def load_ddm_ckpt_(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        # self.start_epoch = checkpoint['epoch']
        # self.step = checkpoint['step']

        self.model_theta.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer_theta.load_state_dict(checkpoint['optimizer'])
        # self.ema_helper_theta.load_state_dict(checkpoint['ema_helper'])

        # checkpoint2 = utils.logging.load_checkpoint("ckpts/S-UIEB_ddpm-55-20000.pth.tar", None)
        # self.model_phi.load_state_dict(checkpoint2['state_dict_phi'], strict=True)
        # self.optimizer_phi.load_state_dict(checkpoint2['optimizer_phi'])

        # self.model_theta.load_state_dict(checkpoint['state_dict_theta'], strict=True)
        # self.optimizer_theta.load_state_dict(checkpoint['optimizer_theta'])
        # self.ema_helper_theta.load_state_dict(checkpoint['ema_helper_theta'])

        # checkpoint2 = utils.logging.load_checkpoint("ckpts/model_best_val_305.pth.tar", None)
        # self.model_phi.load_state_dict(checkpoint2['state_dict'], strict=True)
        # self.optimizer_phi.load_state_dict(checkpoint2['optimizer'])

        if ema:
            self.ema_helper_theta.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], checkpoint['step']))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        criterion = nn.MSELoss()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
            # self.load_ddm_ckpt_(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_start_ = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                ### NOTE: x = [x_0, y_0] (x_0: raw image, y_0: ref image)
                # breakpoint()
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                
                data_time = time.time() - data_start
                self.model_theta.train() #.eval()
                self.model_phi.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)

                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # breakpoint()
                # loss denoise
                loss_theta = noise_estimation_loss(criterion, self.model_theta, x[:, :3, :, :], t, e, b)

                self.tb_logger.add_scalar("loss_theta", loss_theta, global_step=self.step)

                if self.step % 10 == 0:
                    print(f"[denoise] step: {self.step}, loss: {loss_theta.item()}, data time: {data_time}")

                # self.optimizer_theta.zero_grad()
                # loss_theta.backward()
                # self.optimizer_theta.step()
                # self.ema_helper_theta.update(self.model_theta)


                ## loss distribution transformation

                loss_phi_total, loss_phi = transformation_loss(criterion, self.model_theta, self.model_phi, x[:, :3, :, :], x[:, 3:, :, :], t, e, b)

                self.tb_logger.add_scalars("loss_phi", {"loss_phi_total": loss_phi_total, "loss_phi": loss_phi}, global_step=self.step)

                if self.step % 10 == 0:
                    print(f"[transform] step: {self.step}, loss: {loss_phi.item()}, data time: {data_time}")

                # self.optimizer_phi.zero_grad()
                # loss_phi.backward()
                # self.optimizer_phi.step()
                # self.ema_helper_phi.update(self.model_phi)
                
                self.optimizer_theta.zero_grad()
                self.optimizer_phi.zero_grad()
                loss_theta.backward()
                loss_phi_total.backward()
                self.optimizer_theta.step()
                self.ema_helper_theta.update(self.model_theta)
                self.optimizer_phi.step()
                self.ema_helper_phi.update(self.model_phi)


                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict_theta': self.model_theta.state_dict(),
                        'optimizer_theta': self.optimizer_theta.state_dict(),
                        'ema_helper_theta': self.ema_helper_theta.state_dict(),
                        'state_dict_phi': self.model_phi.state_dict(),
                        'optimizer_phi': self.optimizer_phi.state_dict(),
                        'ema_helper_phi': self.ema_helper_phi.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm-' + str(epoch) + '-' + str(self.step) ))
                
                data_start = time.time()
            
            print(f"[epoch={epoch}] Elapsed: {time.time()-data_start_}")

    def sample_image(self, x_cond, x, eta=0., last=True):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        xs = utils.sampling.generalized_steps_uw_2(x, x_cond, seq, self.model_theta, self.model_phi, self.betas, eta=eta)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_image_(self, x_cond, x, eta=0., last=True):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        xs = utils.sampling.generalized_steps_uw_3(x, x_cond, seq, self.model_theta, self.model_phi, self.betas, eta=eta)
        # breakpoint()
        if last:
            xs = xs[0][-1], xs[1][-1], xs[2][-1], xs[3][-1], xs[4][-1]
        return xs

        # xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model_theta, self.betas, eta=eta)
        # # breakpoint()
        # if last:
        #     xs = xs[0][-1]
        # return xs
    
    def sample_image_unconditional(self, x, eta=1., last=True):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        xs = utils.sampling.generalized_steps_unconditional(x, seq, self.model_theta, self.betas, eta=eta)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
