from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# from pytorch_lightning.utilities.model_summary import ModelSummary
from ray import tune, train

from dataset.superpixel import SuperpixelDataModule

import os
import time
import wandb
from .common import generate_eva


def train_func(config):
    seed_everything(1)
    if config["task"] == "regression":
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
    else:
        if config["mamba_fuse"]:
            log_name = f"lead_mamba_unet_pointnext_{config['encoder']}_lr_{config['lr']}_op_{config['optimizer']}"
        else:
            log_name = f"fuse_lead_mlp_unet_pointnext_{config['encoder']}_lr_{config['lr']}_op_{config['optimizer']}"

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
    project_name = "M3F-Net-lead" if config["task"] == "classify" else "M3F-Net-fuse"
    wandb_logger = WandbLogger(
        project=project_name,
        group="tune_group",
        name=f"trial_{tune.Trainable().trial_id}",
        save_dir=log_dir,
        log_model=True,
    )
    metric = "sys_f1" if config["task"] == "classify" else "fuse_val_r2"
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,  # Track the validation loss
        dirpath=chk_dir,
        filename="final_model",
        save_top_k=1,  # Only save the best model
        mode="max",  # We want to minimize the validation loss
    )
    early_stopping = EarlyStopping(
        monitor=metric,  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
        verbose=True,
    )
    # Initialize the DataModule
    data_module = SuperpixelDataModule(config)

    # Use the calculated input channels from the DataModule to initialize the model
    if config["task"] == "classify":
        from models.leading_species_classify import SuperpixelModel
    else:
        from models.fuse_model import SuperpixelModel
    model = SuperpixelModel(config)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

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
        sp_df = generate_eva(model.best_test_outputs, config["classes"], output_dir)
        wandb_logger.log_text(key="preds", dataframe=sp_df)
    else:
        print("No best test output found!")

    # Report the final metric to Ray Tune
    final_result = trainer.callback_metrics[metric].item()
    train.report({f"{metric}": final_result})
    print("Best Hyperparameters:", final_result.config)

    # Save the best model after training
    """
    trainer.save_checkpoint(
        os.path.join(
            chk_dir,
            "final_model.ckpt",
        )
    )
    """
    # Test the model after training
    if config["eval"]:
        trainer.test(model, data_module)

    # Load the saved model
    # model = SuperpixelModel.load_from_checkpoint("final_model.ckpt")

    time.sleep(5)  # Wait for wandb to finish logging
    # wandb.finish()
