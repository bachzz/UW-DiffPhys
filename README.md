# Underwater Image Enhancement with Physical-based Denoising Diffusion Implicit Models

This is the code repository of the following [paper](https://arxiv.org/abs/2409.18476).

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
