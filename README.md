# Underwater Image Enhancement with Physical-based Denoising Diffusion Implicit Models

This is the code repository of the following [paper](https://arxiv.org/abs/2409.18476).

**Abstract**: Underwater vision is crucial for autonomous underwater vehicles (AUVs), and enhancing degraded underwater images in real-time on a resource-constrained AUV is a key challenge due to factors like light absorption and scattering, or the sufficient model computational complexity to resolve such factors. Traditional image enhancement techniques lack adaptability to varying underwater conditions, while learning-based methods, particularly those using convolutional neural networks (CNNs) and generative adversarial networks (GANs), offer more robust solutions but face limitations such as inadequate enhancement, unstable training, or mode collapse. Denoising diffusion probabilistic models (DDPMs) have emerged as a state-of-the-art approach in image-to-image tasks but require intensive computational complexity to achieve the desired underwater image enhancement (UIE) using the recent UW-DDPM solution. To address these challenges, this paper introduces UW-DiffPhys, a novel physical-based and diffusion-based UIE approach. UW-DiffPhys combines light-computation physical-based UIE network components with a denoising U-Net to replace the computationally intensive distribution transformation U-Net in the existing UW-DDPM framework, reducing complexity while maintaining performance. Additionally, the Denoising Diffusion Implicit Model (DDIM) is employed to accelerate the inference process through non-Markovian sampling. Experimental results demonstrate that UW-DiffPhys achieved a substantial reduction in computational complexity and inference time compared to UW-DDPM, with competitive performance in key metrics such as PSNR, SSIM, UCIQE, and an improvement in the overall underwater image quality UIQM metric.

## Data: LSUI-UIEB
Download from: https://drive.google.com/drive/folders/1vqEKHNmdvWe6rludXSlkTwL-Kh18Q4FO?usp=sharing  
Then put into `./data` folder.

## Checkpoints
Download from: https://drive.google.com/file/d/13dZWgUW7tCgBf4SVvqoCZUAHniKcMiU8/view?usp=sharing  
Then put into `./ckpts` folder.

## Training
python train_UW-DDIM.py --config underwater_lsui_uieb_128.yml --resume ckpts/LSUI_UIEB_ddpm.pth.tar

## Inference
python inference_UW-DDIM.py --config underwater_lsui_uieb_128.yml --resume ckpts/LSUI_UIEB_ddpm.pth.tar --sampling_timesteps 25 --eta 0 --condition_image data/LSUI_UIEB/val/raw --seed 5


## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@misc{2409.18476,
  Author = {Bach Nguyen Gia and Chanh Minh Tran and Kamioka Eiji and Tan Phan Xuan},
  Title = {Underwater Image Enhancement with Physical-based Denoising Diffusion Implicit Models},
  Year = {2024},
  Eprint = {arXiv:2409.18476},
}
```

## Acknowledgments

Parts of this code repository is based on the following works:

* https://github.com/IGITUGraz/WeatherDiffusion/tree/main
