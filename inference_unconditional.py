import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils

from models import DenoisingDiffusion, DiffusiveRestoration
from PIL import Image


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--eta", type=float, default=0,
                        help="Number of implicit sampling steps")
    parser.add_argument('--seed', default=1234, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def main():
    args, config = parse_args_and_config()
    to_tensor = torchvision.transforms.ToTensor()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    diffusion = DenoisingDiffusion(args, config)

    diffusion.load_ddm_ckpt(args.resume, ema=True)
    
    diffusion.model.eval()

    with torch.no_grad():
        # x_cond = x[:, :3, :, :].to(self.diffusion.device)
        # x_output = self.diffusive_restoration(x_cond, r=r)
        # x_cond = Image.open(args.condition_image)
        # x_cond = x_cond.resize((config.data.image_size, config.data.image_size), Image.BICUBIC)
        # x_cond = to_tensor(x_cond).to(diffusion.device)
        # utils.logging.save_image(x_cond, f"results/input.png")
        # x_cond = x_cond[None, :, :, :]
        # print(x_cond.size())

        x = torch.randn((1, 3, 256, 256), device=diffusion.device)
        x_output = diffusion.sample_image_unconditional(x, eta=args.eta)
        x_output = inverse_data_transform(x_output)
        utils.logging.save_image(x_output, f"results/output.png")

main()