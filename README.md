# GAN (Generative Adverserial Network)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

PyTorch Generative adversarial networks

Requirements: python3.6, pytorch 1.0, torchvision, imagemagick

* deep convolutional generative adversarial networks (DCGANs) article [link](https://arxiv.org/abs/1511.06434)

  basic dcgan network is based on pytorch [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  
#### improving training

- [ ] Minibatch discrimination
- [ ] One-sided label smoothing
- [ ] Experience replay
- [ ] Conditional GAN

## Example

#### training dcgan:
1. download [celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
2. run training

```
python train.py --cfg ./cfgs/dcgan.yaml
```

#### celeb a output:
![](https://github.com/doronpor/GAN/blob/master/models/generator_sample.gif)
