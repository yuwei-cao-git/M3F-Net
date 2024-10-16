import wandb
from utils.trainer import train
import traceback
import os
# local machine: wandb login --cloud --relogin

def main():
    try:
        run=wandb.init(project='M3F_Net_Sweep')
        config = wandb.config
        # update: 
        # wandb sweep --update ubc-yuwei-cao/M3F-Net/0w8598wd ./conf/config.yaml
        train(config)
        run.finish()
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    #sweep_id = 'ubc-yuwei-cao/M3F_Net/cln04v81'
    #sweep_id = wandb.sweep('ubc-yuwei-cao/M3F_Net_Sweep/qra46txj', project='M3F_Net_Sweep')
    #wandb.agent(sweep_id, function=main)
    main()