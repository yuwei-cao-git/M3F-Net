from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from models.fuse_model import SuperpixelModel
from dataset.superpixel import SuperpixelDataModule
import os


def train(config):
    seed_everything(1)

    # Initialize the DataModule
    data_module = SuperpixelDataModule(config)

    # Call setup explicitly to initialize datasets
    data_module.setup(stage="fit")

    # Use the calculated input channels from the DataModule to initialize the model
    model = SuperpixelModel(config)
    print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Track the validation loss
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  # Only save the best model
        mode="min",  # We want to minimize the validation loss
    )
    if config["use_residual"]:
        log_name = "Fuse_pointnext_ResUnet_"
    else:
        log_name = "Fuse_pointnext_Unet_"
    if config["use_mf"]:
        log_name += "MF_"
    log_name += str(config["resolution"])

    config["save_dir"] = os.path.join(os.getcwd(), "logs", config["log_name"])
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    wandb_logger = WandbLogger(project="M3F-Net-fusion", name=log_name)

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
        devices=config["gpus"],
        num_nodes=1,
        strategy="ddp",
    )
    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)

    # Save the best model after training
    trainer.save_checkpoint(
        os.path.join(config["save_dir"], log_name, "final_model.pt")
    )

    # Load the saved model
    # model = UNetLightning.load_from_checkpoint("final_model.ckpt")
