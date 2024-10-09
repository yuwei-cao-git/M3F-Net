from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import wandb
import pandas as pd
from model.ResUnet_MF import ResUNet_MF
from dataset.s2 import TreeSpeciesDataModule
from os.path import join

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

def train(data_dir, datasets_to_use, resolution, log_name, num_epoch=10, mode='img', use_mf=True, use_residual=True):
    wandb.init()
    # Tile names for train, validation, and test
    tile_names = {
        'train': load_tile_names(join(data_dir, f'{resolution}m', 'dataset/train_tiles.txt')),
        'val': load_tile_names(join(data_dir, f'{resolution}m', 'dataset/val_tiles.txt')),
        'test': load_tile_names(join(data_dir, f'{resolution}m', 'dataset/test_tiles.txt'))
    }
    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(
        tile_names=tile_names,
        processed_dir=join(data_dir, f'{resolution}m'),  # Base directory where the datasets are stored
        datasets_to_use=datasets_to_use,
        batch_size=8,
        num_workers=4
    )
    
    # Call setup explicitly to initialize datasets
    data_module.setup(stage='fit')
    # Access `n_bands` after the dataset has been initialized
    n_bands = data_module.train_dataset.n_bands
    
    # Use the calculated input channels from the DataModule to initialize the model
    model = ResUNet_MF(
        n_bands=n_bands,  # Example channel config
        out_channels=9,
        use_mf=use_mf,
        use_residual=use_residual,
        optimizer_type="adam",
        learning_rate=1e-3,
        scheduler_type="plateau",
        scheduler_params={'patience': 3, 'factor': 0.5}
    )

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Track the validation loss
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Only save the best model
        mode='min'  # We want to minimize the validation loss
    )

    csv_logger = CSVLogger(save_dir='../logs/csv_logs', name=log_name)
    wandb_logger = WandbLogger(name=log_name, save_dir='../logs/wandb_logs', offline=True)
    
    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=num_epoch,
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_callback]
    )
    wandb_logger.log_text('parameters.txt', dataframe=pd.DataFrame({'datasets': [datasets_to_use], 'num_epoches': num_epoch, 'resolution': resolution}))

    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)

    # Save the best model after training
    trainer.save_checkpoint(f"../logs/checkpoints/{log_name}/final_model.pt")
    # Load the saved model
    #model = UNetLightning.load_from_checkpoint("final_model.ckpt")
    wandb.finish()