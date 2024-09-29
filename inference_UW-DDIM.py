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

import time

from models import DenoisingDiffusionUWPhysical
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
    parser.add_argument("--condition_image", required=True, type=str,
                        help="Conditional Image")
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

def data_transform(X):
    return 2 * X - 1.0

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

    diffusion = DenoisingDiffusionUWPhysical(args, config)
    

    diffusion.load_ddm_ckpt(args.resume, ema=True)
    
    diffusion.model_theta.eval()
    diffusion.model_phi.eval()
    # breakpoint()
    with torch.no_grad():
        
        ### eval x_cond folder
        x_cond_fnames = os.listdir(args.condition_image) #Image.open(args.condition_image)
        for fname in x_cond_fnames:
            fname_ = fname.split('.')[0]
            fname = f"{args.condition_image}/{fname}"
            # idx = fname.split('/')[-1].split('.bmp')[0]
            # print(fname)
            # breakpoint()
            x_cond = Image.open(fname)
            x_cond = x_cond.resize((config.data.image_size, config.data.image_size), Image.Resampling.LANCZOS)
            x_cond = to_tensor(x_cond).to(diffusion.device)
            # utils.logging.save_image(x_cond, f"results/input.png")
            x_cond = data_transform(x_cond[None, :, :, :])
            # print(x_cond.size())

            x = torch.randn(x_cond.size(), device=diffusion.device)
            t = time.time()
            y_output, _x0, A, T, y0 = diffusion.sample_image_(x_cond, x, eta=args.eta)
            # y_output = diffusion.sample_image_(x_cond, x, eta=args.eta)
            print(f"Total time taken: {time.time() - t}\n")
            y_output = inverse_data_transform(y_output)
            #_x0_output = inverse_data_transform(_x0)
            # A_output = inverse_data_transform(A)
            # T_output = torch.tensor(np.dot(inverse_data_transform(T).cpu().numpy().transpose(0,2,3,1), [0.2989, 0.5870, 0.1140]))
            # y0_output = inverse_data_transform(y0)
            # breakpoint()

            utils.logging.save_image(y_output, f"results/out/uieb-90/128x128/uw-diffphys/{fname_}.png")
            # utils.logging.save_image(y_output, f"results/out/u45/128x128/uw-diffphys/{fname_}.png")
            # utils.logging.save_image(y_output, f"results/out/uieb-chal60/128x128/uw-diffphys/{fname_}.png")
            
            #utils.logging.save_image(_x0_output, f"results/out/uieb-chal60/128x128/uw-diffphys/{fname_}_x0.png")

            # utils.logging.save_image(_x0_output, f"results/out/suid_uieb/{idx}_x0_.png")
        
        # print(f"Total time taken: {time.time() - t}")

main()