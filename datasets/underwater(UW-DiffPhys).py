import torch
import numpy as np
import torchvision
import glob

from PIL import Image


class Underwater:
    def __init__(self, config):
        self.config = config
        #self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self):

        train_dataset = UnderwaterDataset(self.config, self.config.data.image_size, train=True)
        val_dataset = UnderwaterDataset(self.config, self.config.data.image_size, train=False)
        print(len(train_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        # val_loader = None

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        ) 

        return train_loader, val_loader


# class UnderwaterDataset(torch.utils.data.Dataset):
#     def __init__(self, config, im_size, train=True):
#         #print(data_path)
#         if train:
#             data_path = f"data/{config.data.dataset}/train"
#         else:
#             data_path = f"data/{config.data.dataset}/val"

#         self.im_size = im_size
#         # self.input_fnames = glob.glob(f"{data_path}/raw/*")
#         self.input_fnames = glob.glob(f"{data_path}/raw_{config.data.category}/*")
#         # self.gt_fnames = glob.glob(f"{data_path}/raw_{config.data.category}/*")
#         self.gt_fnames = glob.glob(f"{data_path}/ref_{config.data.category}/*")
#         self.to_tensor = torchvision.transforms.ToTensor()

#     def __len__(self):
#         return len(self.input_fnames)

#     def __getitem__(self, idx):
#         input_im = Image.open(self.input_fnames[idx])
#         input_im = input_im.resize((self.im_size, self.im_size), Image.Resampling.LANCZOS)
#         input_im = self.to_tensor(input_im)
#         gt_im = Image.open(self.gt_fnames[idx])
#         gt_im = gt_im.resize((self.im_size, self.im_size), Image.Resampling.LANCZOS)
#         gt_im = self.to_tensor(gt_im)

#         #print('heyyy 0', input_im.size())
#         im = torch.cat([input_im, gt_im], dim=0)
#         # print('heyyy 1', im.shape)
#         return im, torch.tensor([idx])
    

class UnderwaterDataset(torch.utils.data.Dataset):
    def __init__(self, config, im_size, train=True):
        #print(data_path)
        if train:
            data_path = f"data/{config.data.dataset}/train"
        else:
            data_path = f"data/{config.data.dataset}/val"
            
        self.im_size = im_size
        # self.input_fnames = glob.glob(f"{data_path}/raw/*")
        self.input_fnames = glob.glob(f"{data_path}/raw/*")
        # self.gt_fnames = glob.glob(f"{data_path}/raw_{config.data.category}/*")
        self.gt_fnames = glob.glob(f"{data_path}/ref/*")
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.input_fnames)

    def __getitem__(self, idx):
        input_im = Image.open(self.input_fnames[idx])
        input_im = input_im.resize((self.im_size, self.im_size), Image.Resampling.LANCZOS)
        input_im = self.to_tensor(input_im)
        gt_im = Image.open(self.gt_fnames[idx])
        gt_im = gt_im.resize((self.im_size, self.im_size), Image.Resampling.LANCZOS)
        gt_im = self.to_tensor(gt_im)

        #print('heyyy 0', input_im.size())
        im = torch.cat([input_im, gt_im], dim=0)
        # print('heyyy 1', im.shape)
        return im, torch.tensor([idx])
