import wandb
from utils.trainer import train
import traceback
# local machine: wandb login --cloud --relogin

def main():
    try:
        run=wandb.init()
        config = wandb.config
        # update: 
        # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
        train(config)
        run.finish()
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    sweep_id = 'ubc-yuwei-cao/M3F-Net/d53a2kjr'  # Replace with your actual sweep ID
    wandb.agent(sweep_id, function=main)