from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models.model import Model
from dataset.s2 import TreeSpeciesDataModule
import os

def train(args):
    seed_everything(1)
    
    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(args)
    
    # Call setup explicitly to initialize datasets
    data_module.setup(stage='fit')
    
    # Use the calculated input channels from the DataModule to initialize the model
    model = Model(args)

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Track the validation loss
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Only save the best model
        mode='min'  # We want to minimize the validation loss
    )
    if args["use_residual"]:
        log_name = "ResUnet_"
    else:
        log_name = "Unet_"
    if args["use_mf"]:
        log_name += 'MF_'
    log_name += str(args["resolution"])
    wandb_logger = WandbLogger(name=log_name)
    
    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=args["epochs"],
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
        devices=args["gpus"],
        num_nodes=1,
        #strategy='ddp',
    )
    
    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)

    # Save the best model after training
    trainer.save_checkpoint(os.path.join(args["save_dir"], log_name, 'final_model.pt'))
    
    # Load the saved model
    #model = UNetLightning.load_from_checkpoint("final_model.ckpt")