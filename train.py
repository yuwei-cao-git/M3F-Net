from utils.trainer import train
import wandb

def main():
    run=wandb.init()
    config = wandb.config
    
    if config.use_residual:
        log_name = "ResUnet_"
    else:
        log_name = "Unet_"
    if config.use_mf:
        log_name += 'MF_'
    log_name += str(config.resolution)
    
    # User specifies which datasets to use
    datasets_to_use = ['rmf_s2/spring/tiles_128','rmf_s2/summer/tiles_128','rmf_s2/fall/tiles_128','rmf_s2/winter/tiles_128']
    
    # Call the train function with parsed arguments
    train(
        config,
        data_dir='./data',
        datasets_to_use=datasets_to_use,
        log_name=log_name,
        )

if __name__ == "__main__":
    main()
