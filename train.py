import wandb
from utils.trainer import train

# local machine: wandb login --cloud --relogin

def main():
    run=wandb.init()
    config = wandb.config
    # update: 
    # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
    train(config)
    run.finish()

if __name__ == '__main__':
    sweep_id = 'ubc-yuwei-cao/M3F-Net/d53a2kjr'  # Replace with your actual sweep ID
    wandb.agent(sweep_id, function=main)