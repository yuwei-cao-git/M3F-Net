import wandb
import time
from utils.trainer import train
import argparse

# local machine: wandb login --cloud --relogin

wandb.init(project='M3F-Net', group='cln04v81')
config = wandb.config

def main():
    # update: 
    # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
    train(config)