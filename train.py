import argparse
from utils.trainer import train
import wandb

def main():
    wandb.init()
    
    config = wandb.config
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    optimizer_name = config.optimizer
    max_epoch = config.epochs
    scheduler = config.scheduler
    resolution = config.resolution
    use_mf = config.use_mf
    use_residual = config.use_residual
    mode = config.mode
    if use_residual:
        log_name = "ResUnet_"
    else:
        log_name = "Unet_"
    if use_mf:
        log_name += 'MF_'
    log_name += str(resolution)
    
    # User specifies which datasets to use
    datasets_to_use = ['rmf_s2/spring/tiles_128','rmf_s2/summer/tiles_128','rmf_s2/fall/tiles_128','rmf_s2/winter/tiles_128']
    
    # Call the train function with parsed arguments
    train(
        data_dir='./data',
        datasets_to_use=datasets_to_use,
        resolution=resolution,
        log_name=log_name,
        num_epoch=max_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer_name,
        scheduler=scheduler,
        mode=mode,
        use_mf=use_mf,
        use_residual=use_residual)

if __name__ == "__main__":
    main()
