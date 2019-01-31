# GAN (Generative Adverserial Network)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

PyTorch Generative adversarial networks

Requirements: python3.6, pytorch 1.0, torchvision, imagemagick

#### Implemented GAN methods

* **DCGANs** - "deep convolutional generative adversarial networks" [article](https://arxiv.org/abs/1511.06434)

  basic dcgan network is based on pytorch [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  
* **WGAN-GP** - "Improved Training of Wasserstein GANs" [article](https://arxiv.org/pdf/1704.00028.pdf)
  
#### Improving training

- [ ] Minibatch discrimination
- [x] One-sided label smoothing
- [ ] Experience replay
- [ ] Conditional GAN

## Example

#### training dcgan:
1. Download [celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and unpack.
2. Set path for the dataset in dcgan.yaml
3. run training

```
python train.py --cfg ./cfgs/dcgan.yaml
```

#### celeb a output (10 epoch gif):
![](https://github.com/doronpor/GAN/blob/master/models/generator_sample.gif)

#### training WGAN-GP
1. Download [celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and unpack.
2. Set path for the dataset in wgan_gp.yaml
3. run training

```
python train.py --cfg ./cfgs/wgan_gp.yaml
```
