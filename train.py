import wandb
import time
from utils.trainer import train

# local machine: wandb login --cloud --relogin

def main():
    wandb.init(project='M3F-Net', group='0w8598wd')
    # update: 
    # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
    train(wandb.config)
    
sweep_id = 'M3F-Net/0w8598wd'
time.sleep(3)
wandb.agent(sweep_id, function=main, count=1)