from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# from pytorch_lightning.utilities.model_summary import ModelSummary
from dataset.superpixel import SuperpixelDataModule
import os
from .common import generate_eva, PointCloudLogger


def train(config):
    seed_everything(1)
    log_name = config["log_name"]
    save_dir = os.path.join(config["save_dir"], log_name)
    log_dir = os.path.join(save_dir, "wandblogs")
    chk_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(chk_dir):
        os.mkdir(chk_dir)

    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(
        project="M3F-Net-fuse-v2",
        group="train_group",
        save_dir=log_dir,
        # log_model=True,
    )
    # point_logger = PointCloudLogger(trainer=Trainer)
    # Define a checkpoint callback to save the best model
    metric = "sys_f1" if config["task"] == "classify" else "ave_val_r2"
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,  # Track the validation loss
        dirpath=chk_dir,
        filename="final_model",
        save_top_k=1,  # Only save the best model
        mode="min",  # We want to minimize the validation loss
    )
    early_stopping = EarlyStopping(
        monitor=metric,  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
        verbose=True,
    )

    # Initialize the DataModule
    data_module = SuperpixelDataModule(config)

    # Call setup explicitly to initialize datasets
    data_module.setup(stage="fit")

    # Use the calculated input channels from the DataModule to initialize the model
    if config["task"] == "classify":
        from models.leading_species_classify import SuperpixelModel
    else:
        from models.fuse_model import SuperpixelModel
    model = SuperpixelModel(config)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=[wandb_logger],
        callbacks=early_stopping,  # [early_stopping, checkpoint_callback],
        devices=config["gpus"],
        num_nodes=1,
        strategy="ddp",
    )
    # Train the model
    trainer.fit(model, data_module)

    # using a pandas DataFrame to recode best results
    if model.best_test_outputs is not None:
        output_dir = os.path.join(
            save_dir,
            "outputs",
        )
        sp_df = generate_eva(model.best_test_outputs, config["classes"], output_dir)
        # wandb_logger.log_text(key="preds", dataframe=sp_df)

    # Test the model after training
    trainer.test(model, data_module)

    # Save the best model after training
    # trainer.save_checkpoint(os.path.join(chk_dir, "final_model.pt"))

    # Load the saved model
    # model = SuperpixelModel.load_from_checkpoint("final_model.pt")
