from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# from pytorch_lightning.utilities.model_summary import ModelSummary
from ray import tune, train

from models.fuse_model import SuperpixelModel
from dataset.superpixel import SuperpixelDataModule

import os
import time
import wandb
from .common import generate_eva, PointCloudLogger


def train_func(config):
    seed_everything(1)
    if config["fuse_feature"]:
        log_name = f"Fuse_ff_pointnext_{config['encoder']}"
        if config["mamba_fuse"]:
            log_name = f"Fuse_ff_mamba_pointnext_{config['encoder']}"
    else:
        log_name = f"Fuse_pointnext_{config['encoder']}"
    if config["use_residual"]:
        log_name += "_ResUnet_"
    else:
        log_name += "_Unet_"
    if config["use_mf"]:
        log_name += "MF_"
        if config["spatial_attention"]:
            log_name += "SES_"
        else:
            log_name += "SE_"
    log_name += str(config["resolution"])
    log_name += f"_trial_{tune.Trainable().trial_id}"
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
        group="tune_group",
        name=f"trial_{tune.Trainable().trial_id}",
        save_dir=log_dir,
        log_model=True,
    )

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="fuse_val_r2",  # Track the validation loss
        dirpath=chk_dir,
        filename="best-model-{epoch:02d}-{fuse_val_r2:.2f}",
        save_top_k=1,  # Only save the best model
        mode="max",  # We want to minimize the validation loss
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="min",  # Set "min" for validation loss
        verbose=True,
    )
    # Initialize the DataModule
    data_module = SuperpixelDataModule(config)

    model = SuperpixelModel(config)
    # 1print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        logger=[wandb_logger],
        callbacks=[early_stopping, checkpoint_callback],
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
        sp_df = generate_eva(model, config["classes"], output_dir)
        wandb_logger.log_text(key="preds", dataframe=sp_df)
    else:
        print("No best test output found!")

    # Report the final metric to Ray Tune
    final_result = trainer.callback_metrics["fuse_val_r2"].item()
    train.report({"fuse_val_r2": final_result})

    # Save the best model after training
    trainer.save_checkpoint(
        os.path.join(
            chk_dir,
            "final_model.pt",
        )
    )

    # Test the model after training
    if config["eval"]:
        trainer.test(model, data_module)

    # Load the saved model
    # model = SuperpixelModel.load_from_checkpoint("final_model.pt")

    time.sleep(5)  # Wait for wandb to finish logging
    # wandb.finish()
