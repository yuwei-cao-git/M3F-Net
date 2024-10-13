from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import wandb
import pandas as pd
from models.model import Model
from dataset.s2 import TreeSpeciesDataModule
from os.path import join

run=wandb.init()
config = wandb.config

def load_tile_names(file_path):
    """
    Load tile names from a .txt file.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        tile_names (list): List of tile names.
    """
    with open(file_path, 'r') as f:
        tile_names = f.read().splitlines()
    return tile_names

def train():
    seed_everything(1)
    data_dir='./data'
    # User specifies which datasets to use
    datasets_to_use = ['rmf_s2/spring/tiles_128','rmf_s2/summer/tiles_128','rmf_s2/fall/tiles_128','rmf_s2/winter/tiles_128']
    
    # Tile names for train, validation, and test
    tile_names = {
        'train': load_tile_names(join(data_dir, f'{config.resolution}m', 'dataset/train_tiles.txt')),
        'val': load_tile_names(join(data_dir, f'{config.resolution}m', 'dataset/val_tiles.txt')),
        'test': load_tile_names(join(data_dir, f'{config.resolution}m', 'dataset/test_tiles.txt'))
    }
    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(
        tile_names=tile_names,
        processed_dir=join(data_dir, f'{config.resolution}m'),  # Base directory where the datasets are stored
        datasets_to_use=datasets_to_use,
        batch_size=config.batch_size,
        num_workers=8
    )
    
    # Call setup explicitly to initialize datasets
    data_module.setup(stage='fit')
    # Access `n_bands` after the dataset has been initialized
    n_bands = data_module.train_dataset.n_bands
    
    # Use the calculated input channels from the DataModule to initialize the model
    model = Model(
        n_bands=n_bands,  # Example channel config
        n_classes=config.n_classes,
        use_mf=config.use_mf,
        use_residual=config.use_residual,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        scheduler=config.scheduler,
        scheduler_params={'patience': 3, 'factor': 0.5}
    )

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Track the validation loss
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Only save the best model
        mode='min'  # We want to minimize the validation loss
    )
    if config.use_residual:
        log_name = "ResUnet_"
    else:
        log_name = "Unet_"
    if config.use_mf:
        log_name += 'MF_'
    log_name += str(config.resolution)
    wandb_logger = WandbLogger(project='M3F-Net', name=log_name)
    
    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        strategy='ddp',
        callbacks=[checkpoint_callback]
    )
    
    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)

    # Save the best model after training
    trainer.save_checkpoint(f"../logs/checkpoints/{log_name}/final_model.pt")
    
    # Load the saved model
    #model = UNetLightning.load_from_checkpoint("final_model.ckpt")
    run.finish()
    
wandb.agent('ubc-yuwei-cao/M3F-Net/qexghn0n', function=train, count=10)