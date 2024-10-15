import wandb
from utils.trainer import train
import argparse

# local machine: wandb login --cloud --relogin

def main():
    wandb.init(project='M3F-Net')
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_resume_version')
    args = parser.parse_args()
    
    # update: 
    # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
    train(wandb.config, args)
    
if __name__ == "__main__":
    main()