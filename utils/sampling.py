import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
import math

# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        print(seq)
        print(seq_next)
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # print(i, j, seq[0], seq[-1])
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            # print(x_cond.shape)
            # print(xt.shape)
            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def generalized_steps_unconditional(x, seq, model, b, eta):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        print(seq)
        print(seq_next)
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # print(i, j, seq[0], seq[-1])
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            # print(x_cond.shape)
            # print(xt.shape)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            # print(eta)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            print(c1, c2)
            # breakpoint()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


import time

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn as nn

criterion = nn.MSELoss()

def generalized_steps_uw(x, x_cond, seq, model_theta, model_phi, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt_list = [x]
        yt_list = []

        mse_x0_t_list = []
        psnr_x0_t_list = []
        ssim_x0_t_list = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t_start = time.time()

            # print(i, j, seq[0], seq[-1])
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xt_list[-1].to('cuda')
            
            # print(x_cond.shape)
            # print(xt.shape)
            et = model_theta(torch.cat([x_cond, xt], dim=1), t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
            xt_list.append(xt_next.to('cpu'))

            # compute quality x0 vs x0_t
            np_img1 = inverse_data_transform(x_cond).squeeze().permute(1,2,0).to('cpu').numpy()
            np_img2 = inverse_data_transform(x0_t).squeeze().permute(1,2,0).to('cpu').numpy()
            # breakpoint()
            mse_x0_t_list.append(criterion(x0_t, x_cond).to('cpu').item())
            psnr_x0_t_list.append(psnr(np_img1, np_img2, data_range=np_img1.max() - np_img1.min()))
            ssim_x0_t_list.append(ssim(np_img1, np_img2, data_range=np_img1.max() - np_img1.min(), channel_axis=2))

            if i == seq[-1]:
                yt_next = model_phi(xt_next, t.float())

            elif (seq[0] < i) and (i < seq[-1]):
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)

                yt_next = yt_next + model_phi(xt_next, t.float())

                # distribution shift
                yt_next = (1/math.sqrt(2)) * yt_next + (1 - math.sqrt(2)) * (at_next.sqrt() * y0_t + c2 * et)

            elif i == seq[0]:
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)
            
            yt_list.append(yt_next)

            print(f"Time per step: {time.time()-t_start}")
            # breakpoint()

        # breakpoint()

    return yt_list, xt_list


def generalized_steps_uw_2(x, x_cond, seq, model_theta, model_phi, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt_list = [x]
        yt_list = []
        A_list = []
        T_list = []
        y0_list = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t_start = time.time()

            # print(i, j, seq[0], seq[-1])
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xt_list[-1].to('cuda')
            
            # print(x_cond.shape)
            # print(xt.shape)
            et = model_theta(torch.cat([x_cond, xt], dim=1), t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
            xt_list.append(xt_next.to('cpu'))

            y0_ = model_phi(x_cond)
            A = model_phi.module.ANet(x_cond)
            T = model_phi.module.tNet(torch.cat((x_cond*0+A,x_cond),1))

            A_list.append(A)
            T_list.append(T)
            y0_list.append(y0_)

            if i == seq[-1]:    
                yt_next = y0_* at_next.sqrt() + et * (1.0 - at_next).sqrt()

            elif (seq[0] < i) and (i < seq[-1]):
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)

                yt_next = yt_next + y0_* at_next.sqrt() + et * (1.0 - at_next).sqrt() #model_phi(xt_next, t.float())

                # distribution shift
                yt_next = (1/math.sqrt(2)) * yt_next + (1 - math.sqrt(2)) * (at_next.sqrt() * y0_t + c2 * et)

            elif i == seq[0]:
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)
            
            yt_list.append(yt_next)

            print(f"Time per step: {time.time()-t_start}")

    return yt_list, xt_list, A_list, T_list, y0_list



# import torch.nn as nn

# class ANet(nn.Module):
#     def __init__(self):
#         super(ANet,self).__init__()

#         self.conv1 = nn.Conv2d(3,3,3,padding = 1)
#         self.activation1 = nn.PReLU()

#         self.conv2 = nn.Conv2d(3,3,3,padding = 1)
#         self.activation2 = nn.PReLU()

#         self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv2_1 = nn.Conv2d(3,3,3,padding = 1)
#         self.activation2_1 = nn.PReLU()

#         self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv2_2 = nn.Conv2d(3,3,3,padding = 1)
#         self.activation2_2 = nn.PReLU()

#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.conv3 = nn.Conv2d(3,3,1)
#         self.activation3 = nn.PReLU()
#         self.conv4 = nn.Conv2d(3,3,1)
#         self.activation4 = nn.PReLU()

#     def forward(self,x):
#         out = self.conv1(x)
#         out = self.activation1(out)
#         out = self.conv2(out)
#         out = self.activation2(out)

#         out = self.max_pool1(out)
#         out = self.conv2_1(out)
#         out = self.activation2_1(out)

#         out = self.max_pool2(out)
#         out = self.conv2_2(out)
#         out = self.activation2_2(out)

#         out = self.avgpool(out)
#         out = self.conv3(out)
#         out = self.activation3(out)
#         out = self.conv4(out)
#         out = self.activation4(out)
#         # breakpoint()
#         return out


# class TNet(nn.Module):
#     def __init__(self):
#         super(TNet, self).__init__()

#         # self.p = 0.25

#         # in-branch
#         self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding='same')
#         # self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, padding='same')
#         self.normalize1 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation1 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout1 = nn.Dropout(p=self.p)
#         self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
#         self.normalize2 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation2 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout2 = nn.Dropout(p=self.p)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.normalize3 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation3 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout3 = nn.Dropout(p=self.p)
#         self.conv3_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.activation3_ = nn.PReLU() #nn.Sigmoid()

#         # sub-branch 1
#         self.max_pool2 = nn.MaxPool2d(kernel_size = 4, stride = 4)
#         self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
#         self.normalize4 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation4 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout4 = nn.Dropout(p=self.p)
#         self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.normalize5 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation5 = nn.PReLU() #nn.Sigmoid()
#         self.conv5_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.activation5_ = nn.PReLU() #nn.Sigmoid()
#         # self.dropout5 = nn.Dropout(p=self.p)
#         self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same')
#         self.normalize6 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
#         self.activation6 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout6 = nn.Dropout(p=self.p)
#         self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)

#         # sub-branch 2
#         self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv7 = nn.Conv2d(in_channels=17, out_channels=16, kernel_size=5, padding='same')
#         self.normalize7 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation7 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout7 = nn.Dropout(p=self.p)
#         self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.normalize8 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation8 = nn.PReLU() #nn.Sigmoid()
#         self.conv8_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.activation8_ = nn.PReLU() #nn.Sigmoid()
#         # self.dropout8 = nn.Dropout(p=self.p)
#         self.conv9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same')
#         self.normalize9 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
#         self.activation9 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout9 = nn.Dropout(p=self.p)
#         self.deconv2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)

#         # out-branch
#         self.conv10 = nn.Conv2d(in_channels=17, out_channels=16, kernel_size=3, padding='same')
#         self.normalize10 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation10 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout10 = nn.Dropout(p=self.p)
#         self.conv10_ = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
#         self.activation10_ = nn.PReLU() #nn.Sigmoid()
#         self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0)
#         self.conv11 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
#         self.normalize11 = nn.BatchNorm2d(16) #nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True) #nn.BatchNorm2d(16)
#         self.activation11 = nn.PReLU() #nn.Sigmoid()
#         # self.dropout11 = nn.Dropout(p=self.p)
#         self.conv12 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding='same')
#         self.normalize12 = nn.BatchNorm2d(1) #nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-6, affine=True) #nn.BatchNorm2d(1)
#         self.activation12 = nn.PReLU() #Sigmoid()
#         # self.dropout12 = nn.Dropout(p=self.p)
    
#     def forward(self, x):
#         # in-branch
#         out = self.conv1(x)
#         # out = self.normalize1(out)
#         out_activation1 = self.activation1(out)
#         # out_activation1 = self.dropout1(out_activation1)
#         out = self.max_pool1(out_activation1)
#         out = self.conv2(out)
#         # out = self.normalize2(out)
#         out = self.activation2(out)
#         # out = self.dropout2(out)
#         out = self.conv3(out)
#         # out = self.normalize3(out)
#         out_conv3 = self.activation3(out)
#         # out_conv3 = self.dropout3(out_conv3)
#         out_conv3 = self.conv3_(out_conv3)
#         out_conv3 = self.activation3_(out_conv3)

#         # sub-branch 1
#         out = self.max_pool2(out_conv3)
#         out = self.conv4(out)
#         # out = self.normalize4(out)
#         out = self.activation4(out)
#         # out = self.dropout4(out)
#         out = self.conv5(out)
#         # out = self.normalize5(out)
#         out = self.activation5(out)
#         # out = self.dropout5(out)
#         out = self.conv5_(out)
#         out = self.activation5_(out)
#         out = self.conv6(out)
#         # out = self.normalize6(out)
#         out = self.activation6(out)
#         # out = self.dropout6(out)
#         out_deconv1 = self.deconv1(out)

#         # sub-branch 2
#         out = self.max_pool3(out_conv3)
#         out = torch.cat((out_deconv1, out), dim=1)
#         out = self.conv7(out)
#         # out = self.normalize7(out)
#         out = self.activation7(out)
#         # out = self.dropout7(out)
#         out = self.conv8(out)
#         # out = self.normalize8(out)
#         out = self.activation8(out)
#         # out = self.dropout8(out)
#         out = self.conv8_(out)
#         out = self.activation8_(out)
#         out = self.conv9(out)
#         # out = self.normalize9(out)
#         out = self.activation9(out)
#         # out = self.dropout9(out)
#         out_deconv2 = self.deconv2(out)

#         # out-branch
#         out = torch.cat((out_deconv2, out_conv3), dim=1)
#         out = self.conv10(out)
#         # out = self.normalize10(out)
#         out = self.activation10(out)
#         out = self.conv10_(out)
#         out = self.activation10_(out)
#         # out = self.dropout10(out)
#         out = self.deconv3(out)
#         out = torch.cat((out, out_activation1), dim=1)
#         out = self.conv11(out)
#         # out = self.normalize11(out)
#         out = self.activation11(out)
#         # out = self.dropout11(out)
#         out = self.conv12(out)
#         # out = self.normalize12(out)
#         out = self.activation12(out)
#         # out = self.dropout12(out)

#         if (torch.unique(torch.isnan(out)) == torch.tensor([True], device='cuda:0')).item():
#             breakpoint()

#         return out


# class PhysicalNN(nn.Module):
#     def __init__(self):
#         super(PhysicalNN,self).__init__()

#         self.ANet = ANet()
#         self.tNet = TNet()

#     def forward(self,x):
#         A = self.ANet(x)
#         t = self.tNet(torch.cat((x*0+A,x),1))
#         out = ((x-A)*t + A)
#         # breakpoint()
#         return out #torch.clamp(out,0.,1.)

def generalized_steps_uw_3(x, x_cond, seq, model_theta, model_phi, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt_list = [x]
        x0_t_list = []
        yt_list = []
        A_list = []
        T_list = []
        y0_list = []

        mse_x0_t_list = []
        psnr_x0_t_list = []
        ssim_x0_t_list = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t_start = time.time()

            # print(i, j, seq[0], seq[-1])
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xt_list[-1].to('cuda')
            
            # print(x_cond.shape)
            # print(xt.shape)
            et = model_theta(torch.cat([x_cond, xt], dim=1), t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
            xt_list.append(xt_next.to('cpu'))


            y0_ = model_phi(x_cond)
            A = model_phi.module.ANet(x_cond)
            T = model_phi.module.tNet(torch.cat((x_cond*0+A,x_cond),1))
            x0_t_list.append(x0_t)
            A_list.append(A)
            T_list.append(T)
            y0_list.append(y0_)
            # breakpoint()

            # compute quality x0 vs x0_t
            np_img1 = inverse_data_transform(x_cond).squeeze().permute(1,2,0).to('cpu').numpy()
            np_img2 = inverse_data_transform(x0_t).squeeze().permute(1,2,0).to('cpu').numpy()
            # breakpoint()
            mse_x0_t_list.append(criterion(x0_t, x_cond).to('cpu').item())
            psnr_x0_t_list.append(psnr(np_img1, np_img2, data_range=np_img1.max() - np_img1.min()))
            ssim_x0_t_list.append(ssim(np_img1, np_img2, data_range=np_img1.max() - np_img1.min(), channel_axis=2))


            if i == seq[-1]:    
                yt_next = y0_* at_next.sqrt() + et * (1.0 - at_next).sqrt()

            elif (seq[0] < i) and (i < seq[-1]):
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)

                # yt_next = yt_next + y0_* at_next.sqrt() + et * (1.0 - at_next).sqrt() #model_phi(xt_next, t.float())
                yt_next = yt_next + y0_* at_next.sqrt() + 1.2*et * (1.0 - at_next).sqrt()
                # yt_next = 0.3*yt_next + 0.7*y0_* at_next.sqrt() + 0.7*et * (1.0 - at_next).sqrt()

                # distribution shift
                yt_next = (1/math.sqrt(2)) * yt_next + (1 - math.sqrt(2)) * (at_next.sqrt() * y0_t + c2 * et)

            elif i == seq[0]:
                yt = yt_list[-1].to('cuda')

                y0_t = (yt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                yt_next = at_next.sqrt() * y0_t + c2 * et + c1 * torch.randn_like(x)
            
            yt_list.append(yt_next)

            print(f"Time per step: {time.time()-t_start}")
        
        # breakpoint()

    return yt_list, x0_t_list, A_list, T_list, y0_list




def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                for i in range(0, len(corners), manual_batching_size):
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                               xt_patch[i:i+manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds
