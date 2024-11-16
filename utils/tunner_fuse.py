from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import WandbLogger
# from pytorch_lightning.utilities.model_summary import ModelSummary
from ray import tune, train

from models.fuse_model import SuperpixelModel
from dataset.superpixel import SuperpixelDataModule

import os
import time
import wandb
from .common import generate_eva

class PointCloudLogger(Callback):
    def on_validation_batch_end(self, wandb_logger, trainer, pl_module, outputs, batch, batch_idx):
        # Access data and potentially augmented point cloud from current batch
        if batch_idx == 0:
            n = 4
            point_clouds =  [pc for pc in batch["point_cloud"][:n]]
            labels = [label for label in batch["label"][:n]]
            captions_1 = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(labels[:n], outputs[0][:n])]
            wandb.log({"point_cloud": wandb.Object3D(point_clouds)}, captions=captions_1)

            images = [img for img in batch["images"][:n]]
            per_pixel_labels = [pixel_label for pixel_label in batch["per_pixel_labels"][:n]]
            captions = [f'Image Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(per_pixel_labels[:n], outputs[1][:n])]
            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)

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

    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(
        project="M3F-Net-fuse",
        group="tune_v4",
        name=f"{log_name}_trial_{tune.Trainable().trial_id}",
        save_dir=config["save_dir"],
        log_model=True,
    )
    
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="fuse_val_r2",  # Track the validation loss
        filename="best-model-{epoch:02d}-{fuse_val_r2:.2f}",
        save_top_k=1,  # Only save the best model
        mode="max",  # We want to minimize the validation loss
    )
    early_stopping = EarlyStopping(
        monitor="fuse_val_r2",  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
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

    # Report the final metric to Ray Tune
    final_result = trainer.callback_metrics["fuse_val_r2"].item()
    train.report({"fuse_val_r2": final_result})

    # Save the best model after training
    trainer.save_checkpoint(
        os.path.join(
            config["save_dir"],
            f"{log_name}_trial_{tune.Trainable().trial_id}",
            "final_model.pt",
        )
    )
    # using a pandas DataFrame to recode best results
    if hasattr(model, 'best_test_outputs') and model.best_test_outputs is not None:
        output_dir = f"checkpoints/{log_name}_trial_{tune.Trainable().trial_id}/output"
        generate_eva(model, trainer, config["classes"], output_dir)

    # Test the model after training
    if config["eval"]:
        trainer.test(model, data_module)

    # Load the saved model
    # model = SuperpixelModel.load_from_checkpoint("final_model.ckpt")

    time.sleep(5)  # Wait for wandb to finish logging
    # wandb.finish()
