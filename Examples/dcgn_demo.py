# Script to generate images from noise of a trained generator
import os.path as path

from GAN.Examples.demo import demo
from GAN.utils.configuration import load_config

cfg_path = path.abspath(path.join(path.dirname(__file__), '../cfgs/dcgan.yaml'))
cfg = load_config(cfg_path)

model_path = path.abspath(path.join(path.dirname(__file__), '../models/dcgan.pth'))

demo(cfg, model_path)
