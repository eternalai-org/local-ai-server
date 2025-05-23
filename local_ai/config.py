from yaml import load, Loader

CONFIG = load(open("configs/8x4090.yaml", "r"), Loader=Loader)